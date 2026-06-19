"""Forecast dataset for the CSDI classifier.

Splits the snapshot stream temporally by fraction, slides ``T_past`` windows at
``stride`` (skipping day-straddlers), normalizes the row stream on training data,
and yields per window the past multivariate features ``(2, R, T_past)`` and the
3-class direction label.
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
    def __init__(self, rows_norm, starts, meta, config) -> None:
        self.rows = rows_norm  # (N, R, 2) float32
        self.starts = starts
        self.t_past = config["T_past"]
        self.labels = meta["labels"]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        past = self.rows[s : s + self.t_past]  # (T_past, R, 2)
        past = torch.from_numpy(np.transpose(past, (2, 1, 0)).copy())  # (2, R, T_past)
        return {"past": past, "label": int(self.labels[idx])}


def _cache_path(config: dict) -> Path:
    tf = int(config["train_frac"] * 100)
    vf = int(config["val_frac"] * 100)
    name = (
        f"csdi_{config['exchange']}_{config['pair']}_{config['feature_mode']}"
        f"_T{config['T_total']}_s{config['stride']}_split{tf}-{vf}_clf.npz"
    )
    return Path(config["cache_dir"]) / name


def _window_labels(starts, mid, config, alpha):
    t_past, k = config["T_past"], config["label_k"]
    n = len(starts)
    labels = np.zeros(n, np.int64)
    l_vals = np.zeros(n, np.float64)
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
    """Return ``(train_ds, val_ds, test_ds, normalizer, alpha, meta)``."""
    cache = _cache_path(config)
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)

    if cache.exists():
        logger.info("loading dataset cache {}", cache.name)
        z = np.load(cache, allow_pickle=True)
        rows_norm = z["rows_norm"]
        starts = {s: z[f"starts_{s}"] for s in ("train", "val", "test")}
        labels = {s: z[f"labels_{s}"] for s in ("train", "val", "test")}
        normalizer = feat.RollingNormalizer.from_dict(config, z["norm"].item())
        alpha = float(z["alpha"])
    else:
        exch, pair = config["exchange"], config["pair"]
        base = Path("data") / f"{exch}_data"
        snaps = feat.load_orderbook(
            str(base / f"{pair}_orderbook.csv"), config["n_levels"]
        )
        trades = (
            feat.load_trades(str(base / f"{pair}_trades.csv"))
            if config["feature_mode"] == "ofi"
            else None
        )
        logger.info("loaded {} snapshots for {}/{}", len(snaps), exch, pair)

        rows = feat.build_global_rows(snaps, trades, config)
        mid = feat.mid_series(snaps)
        days = snaps["time"].dt.normalize().to_numpy()
        n = len(snaps)
        train_end, val_end = _fraction_split_bounds(
            n, config["train_frac"], config["val_frac"]
        )
        normalizer = feat.RollingNormalizer(config)
        normalizer.fit(rows[:train_end])
        rows_norm = normalizer.transform(rows)

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
        labels = {
            s: _window_labels(starts[s], mid, config, alpha)[0] for s in starts
        }
        np.savez_compressed(
            cache,
            rows_norm=rows_norm,
            norm=normalizer.to_dict(),
            alpha=alpha,
            **{f"starts_{s}": starts[s] for s in starts},
            **{f"labels_{s}": labels[s] for s in labels},
        )
        logger.info("cached dataset -> {}", cache.name)

    datasets = {
        s: ForecastDataset(rows_norm, starts[s], {"labels": labels[s]}, config)
        for s in ("train", "val", "test")
    }
    meta = {
        "counts": {s: len(starts[s]) for s in starts},
        "class_balance": _class_balance(labels["train"]),
        "total_snapshots": int(rows_norm.shape[0]),
    }
    return datasets["train"], datasets["val"], datasets["test"], normalizer, alpha, meta
