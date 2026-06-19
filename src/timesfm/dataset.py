"""Forecast dataset for the TimesFM classifier.

Yields per window:

- ``past_mid``  : ``(T_past,)`` raw mid-price series
- ``past_ofi``  : ``(T_past,)`` normalised net OFI — **OFI mode only**
- ``label``     : 3-class direction label

OFI normalisation statistics are saved in ``meta["ofi_stats"]`` and the checkpoint
so they can be reproduced at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from . import features as feat
from . import labels as lab


def _fraction_split_bounds(n, train_frac, val_frac):
    return int(n * train_frac), int(n * (train_frac + val_frac))


def _starts(lo, hi, t_total, stride, days):
    return [
        s
        for s in range(lo, hi - t_total + 1, stride)
        if days[s] == days[s + t_total - 1]
    ]


class ForecastDataset(Dataset):
    def __init__(self, mid, ofi_norm, starts, labels, config) -> None:
        self.mid = mid
        self.ofi = ofi_norm  # (N,) float32 or None
        self.starts = starts
        self.t_past = config["T_past"]
        self.labels = labels

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        past_mid = self.mid[s : s + self.t_past].astype(np.float32)
        item = {
            "past_mid": torch.from_numpy(past_mid),
            "label": int(self.labels[idx]),
        }
        if self.ofi is not None:
            item["past_ofi"] = torch.from_numpy(
                self.ofi[s : s + self.t_past].astype(np.float32)
            )
        return item


def _cache_path(config: dict) -> Path:
    tf = int(config["train_frac"] * 100)
    vf = int(config["val_frac"] * 100)
    mode = config.get("feature_mode", "ofi")
    name = (
        f"timesfm_{config['exchange']}_{config['pair']}_{mode}"
        f"_T{config['T_total']}_s{config['stride']}_split{tf}-{vf}_clf.npz"
    )
    return Path(config["cache_dir"]) / name


def _window_labels(starts, mid, config, alpha):
    t_past, k = config["T_past"], config["label_k"]
    n = len(starts)
    l_vals = np.zeros(n, np.float64)
    labels = np.zeros(n, np.int64)
    for j, s in enumerate(starts):
        win_mid = mid[s : s + config["T_total"]]
        lv = lab.compute_l(win_mid, t_past, k)
        l_vals[j] = lv
        if alpha is not None:
            labels[j] = lab.label_from_l(lv, alpha)
    return labels, l_vals


def _class_balance(labels):
    counts = np.bincount(labels, minlength=3)
    frac = counts / max(counts.sum(), 1)
    return {"down": float(frac[0]), "stationary": float(frac[1]), "up": float(frac[2])}


def build_datasets(config: dict):
    """Return ``(train_ds, val_ds, test_ds, alpha, meta)``."""
    cache = _cache_path(config)
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)
    use_ofi = config.get("feature_mode", "ofi") == "ofi"

    if cache.exists():
        logger.info("loading dataset cache {}", cache.name)
        z = np.load(cache, allow_pickle=True)
        mid = z["mid"]
        ofi_norm = z["ofi_norm"] if "ofi_norm" in z.files else None
        ofi_stats = (
            {"mean": float(z["ofi_stats"][0]), "std": float(z["ofi_stats"][1])}
            if "ofi_stats" in z.files
            else None
        )
        starts = {s: z[f"starts_{s}"] for s in ("train", "val", "test")}
        labels = {s: z[f"labels_{s}"] for s in ("train", "val", "test")}
        alpha = float(z["alpha"])
    else:
        exch, pair = config["exchange"], config["pair"]
        base = Path("data") / f"{exch}_data"
        snaps = feat.load_orderbook(
            str(base / f"{pair}_orderbook.csv"), config["n_levels"]
        )
        logger.info("loaded {} snapshots for {}/{}", len(snaps), exch, pair)
        mid = feat.mid_series(snaps)
        days = snaps["time"].dt.normalize().to_numpy()
        n = len(snaps)
        train_end, val_end = _fraction_split_bounds(
            n, config["train_frac"], config["val_frac"]
        )

        if use_ofi:
            raw_ofi = feat.net_ofi_series(snaps)
            ofi_mean = float(raw_ofi[:train_end].mean())
            ofi_std = max(float(raw_ofi[:train_end].std()), 1e-8)
            ofi_norm = ((raw_ofi - ofi_mean) / ofi_std).astype(np.float32)
            ofi_stats = {"mean": ofi_mean, "std": ofi_std}
            logger.info(
                "net OFI stats (train): mean={:.4f} std={:.4f}", ofi_mean, ofi_std
            )
        else:
            ofi_norm = None
            ofi_stats = None

        t_total, stride = config["T_total"], config["stride"]
        starts = {
            "train": np.array(
                _starts(0, train_end, t_total, stride, days), dtype=np.int64
            ),
            "val": np.array(
                _starts(train_end, val_end, t_total, stride, days), dtype=np.int64
            ),
            "test": np.array(
                _starts(val_end, n, t_total, stride, days), dtype=np.int64
            ),
        }
        logger.info("windows: {}", {k: len(v) for k, v in starts.items()})

        _, train_l = _window_labels(starts["train"], mid, config, alpha=None)
        alpha = (
            float(config["label_alpha"])
            if config.get("label_alpha", -1) > 0
            else lab.calibrate_alpha(train_l)
        )
        labels = {s: _window_labels(starts[s], mid, config, alpha)[0] for s in starts}

        save_dict = {
            "mid": mid,
            "alpha": alpha,
            **{f"starts_{s}": starts[s] for s in starts},
            **{f"labels_{s}": labels[s] for s in labels},
        }
        if ofi_norm is not None:
            save_dict["ofi_norm"] = ofi_norm
            save_dict["ofi_stats"] = np.array([ofi_stats["mean"], ofi_stats["std"]])
        np.savez_compressed(cache, **save_dict)
        logger.info("cached dataset -> {}", cache.name)

    datasets = {
        s: ForecastDataset(mid, ofi_norm, starts[s], labels[s], config)
        for s in ("train", "val", "test")
    }
    meta = {
        "counts": {s: len(starts[s]) for s in starts},
        "class_balance": _class_balance(labels["train"]),
        "total_snapshots": int(mid.shape[0]),
        "ofi_stats": ofi_stats,
    }
    return datasets["train"], datasets["val"], datasets["test"], alpha, meta
