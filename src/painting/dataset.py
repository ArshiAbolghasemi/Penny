"""Dataset construction for the painting approach (spec sections 1, 2.7, 3).

Reads order-book/trade CSVs, builds the row stream, splits temporally by fraction
(train/val/test), slides ``T_total`` windows at ``stride`` while skipping
day-straddlers, fits the normalizer and the OFI->price coefficient ``gamma`` on
training data, calibrates ``alpha``, and caches everything to ``.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from . import features as feat
from . import labels as lab


def _fraction_split_bounds(
    n: int, train_frac: float, val_frac: float
) -> tuple[int, int]:
    return int(n * train_frac), int(n * (train_frac + val_frac))


def _starts(lo, hi, t_total, stride, days) -> list[int]:
    out = []
    for s in range(lo, hi - t_total + 1, stride):
        if days[s] == days[s + t_total - 1]:
            out.append(s)
    return out


def _fit_gamma(rows, mid, config) -> tuple[float, float]:
    """OLS-through-origin slope of ``dmid`` on net best-level OFI + its R^2."""
    ofi = feat.best_level_ofi(rows, config)[1:]
    dmid = np.diff(mid)
    denom = float(np.sum(ofi * ofi))
    if denom < 1e-12:
        return 0.0, 0.0
    gamma = float(np.sum(ofi * dmid) / denom)
    pred = gamma * ofi
    ss_res = float(np.sum((dmid - pred) ** 2))
    ss_tot = float(np.sum((dmid - dmid.mean()) ** 2)) + 1e-12
    return gamma, 1.0 - ss_res / ss_tot


class LOBImageDataset(Dataset):
    """Padded square LOB images over the normalized row stream."""

    def __init__(self, rows_norm, starts, meta, config) -> None:
        self.rows = rows_norm
        self.starts = starts
        self.t_total = config["T_total"]
        self.t_past = config["T_past"]
        self.n2 = 2 * config["n_levels"]
        self.padded = config["padded_size"]
        self.is_ofi = config["feature_mode"] == "ofi"
        self.mask = torch.from_numpy(feat.build_mask(config)).permute(2, 0, 1)
        self.labels = meta["labels"]
        self.l = meta["l"]
        self.mid_ref = meta["mid_ref"]
        self.bwd = meta["bwd_smoothed"]
        self.true_mid = meta["true_mid"]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        s = self.starts[idx]
        window = self.rows[s : s + self.t_total]
        img = np.transpose(window, (1, 0, 2)).copy()  # (R, T, 2)
        if self.is_ofi:
            img[: self.n2, 0, 0] = 0.0
        padded, _ = feat.pad_levels(img, self.padded)
        image = torch.from_numpy(padded).permute(2, 0, 1)
        return {
            "image": image,
            "mask": self.mask,
            "label": int(self.labels[idx]),
            "l": float(self.l[idx]),
            "mid_ref": float(self.mid_ref[idx]),
            "bwd_smoothed": float(self.bwd[idx]),
            "true_mid": torch.from_numpy(self.true_mid[idx].astype(np.float32)),
        }


def _cache_path(config: dict) -> Path:
    tf = int(config["train_frac"] * 100)
    vf = int(config["val_frac"] * 100)
    name = (
        f"painting_{config['exchange']}_{config['pair']}_{config['feature_mode']}"
        f"_T{config['T_total']}_s{config['stride']}_split{tf}-{vf}.npz"
    )
    return Path(config["cache_dir"]) / name


def _window_meta(starts, mid, config, alpha) -> dict:
    t_past, k = config["T_past"], config["label_k"]
    n = len(starts)
    out = {
        "l": np.zeros(n, np.float64),
        "labels": np.zeros(n, np.int64),
        "mid_ref": np.zeros(n, np.float64),
        "bwd_smoothed": np.zeros(n, np.float64),
        "true_mid": np.zeros((n, config["T_total"]), np.float64),
    }
    use_anchor = config["feature_mode"] == "lob"
    for j, s in enumerate(starts):
        win_mid = mid[s : s + config["T_total"]]
        out["true_mid"][j] = win_mid
        out["l"][j] = lab.compute_l(win_mid, t_past, k)
        out["bwd_smoothed"][j] = lab.smoothed_backward_mid(win_mid, t_past, k)
        out["mid_ref"][j] = win_mid[0] if use_anchor else win_mid[t_past - 1]
        if alpha is not None:
            out["labels"][j] = lab.label_from_l(out["l"][j], alpha)
    return out


def _class_balance(labels: np.ndarray) -> dict:
    counts = np.bincount(labels, minlength=3)
    frac = counts / max(counts.sum(), 1)
    return {"down": float(frac[0]), "stationary": float(frac[1]), "up": float(frac[2])}


def build_datasets(config: dict):
    """Return ``(train_ds, val_ds, test_ds, normalizer, gamma, alpha, meta)``."""
    cache = _cache_path(config)
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)

    if cache.exists():
        logger.info("loading dataset cache {}", cache.name)
        z = np.load(cache, allow_pickle=True)
        rows_norm = z["rows_norm"]
        starts = {s: z[f"starts_{s}"] for s in ("train", "val", "test")}
        metas = {s: z[f"meta_{s}"].item() for s in ("train", "val", "test")}
        normalizer = feat.RollingNormalizer.from_dict(config, z["norm"].item())
        gamma, alpha = float(z["gamma"]), float(z["alpha"])
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

        gamma, r2 = _fit_gamma(rows[:train_end], mid[:train_end], config)
        if config["feature_mode"] == "ofi":
            logger.info("gamma={:.6g}  R^2={:.4f}", gamma, r2)
            if r2 < 0.05:
                logger.warning(
                    "OFI->price R^2={:.4f} < 0.05; mid reconstruction weak", r2
                )

        t_total, stride = config["T_total"], config["stride"]
        starts = {
            "train": _starts(0, train_end, t_total, stride, days),
            "val": _starts(train_end, val_end, t_total, stride, days),
            "test": _starts(val_end, n, t_total, stride, days),
        }
        logger.info("windows: {}", {k: len(v) for k, v in starts.items()})

        train_l = _window_meta(starts["train"], mid, config, alpha=None)
        if config["label_alpha"] and config["label_alpha"] > 0:
            alpha = float(config["label_alpha"])
        else:
            alpha = lab.calibrate_alpha(train_l["l"])
        metas = {s: _window_meta(starts[s], mid, config, alpha) for s in starts}
        starts = {s: np.array(v, dtype=np.int64) for s, v in starts.items()}

        np.savez_compressed(
            cache,
            rows_norm=rows_norm,
            norm=normalizer.to_dict(),
            gamma=gamma,
            alpha=alpha,
            **{f"starts_{s}": starts[s] for s in starts},
            **{f"meta_{s}": metas[s] for s in metas},
        )
        logger.info("cached dataset -> {}", cache.name)

    datasets = {
        s: LOBImageDataset(rows_norm, starts[s], metas[s], config)
        for s in ("train", "val", "test")
    }
    meta = {
        "counts": {s: len(starts[s]) for s in starts},
        "class_balance": _class_balance(metas["train"]["labels"]),
        "level_starts": feat.level_starts(config),
        "total_snapshots": int(rows_norm.shape[0]),
    }
    return (
        datasets["train"],
        datasets["val"],
        datasets["test"],
        normalizer,
        gamma,
        alpha,
        meta,
    )
