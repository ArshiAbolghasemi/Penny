"""Windowed LOB dataset shared across DeepLOB, JointDiffusion, and LOBTransformer.

Features are pre-normalised by the loader with a causal trailing rolling
z-score spanning day boundaries (see ``crypto.loader``).  The memmap is
accessed page-by-page so RAM scales with ``batch_size × T_past × F``, not
total dataset size.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from .features import extract_features, n_features
from .labels import build_labels
from .loader import build_cache


class LOBDataset(Dataset):
    def __init__(
        self, feat: np.memmap, starts: np.ndarray, labels: np.ndarray, t_past: int
    ) -> None:
        self.feat = feat
        self.starts = starts
        self.labels = labels
        self.t_past = t_past

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        s = self.starts[idx]
        window = self.feat[s : s + self.t_past].astype(np.float32)
        x = torch.from_numpy(window.copy()).unsqueeze(0)  # (1, T, F)
        return {"x": x, "label": int(self.labels[s + self.t_past - 1])}


def _valid_starts(
    lo: int,
    hi: int,
    t_past: int,
    label_k: int,
    labels: np.ndarray,
    timestamps: np.ndarray,
    stride: int,
) -> np.ndarray:
    starts = []
    for s in range(lo, hi - t_past - label_k + 1, stride):
        centre = s + t_past - 1
        if centre + label_k >= hi:
            break
        ts_win = timestamps[s : s + t_past]
        gaps = np.diff(ts_win)
        if len(gaps) > 0 and gaps.max() > 10 * np.median(gaps):
            continue
        if labels[centre] < 0:
            continue
        starts.append(s)
    return np.array(starts, dtype=np.int64)


def build_datasets(config: dict):
    """Return ``(train_ds, val_ds, test_ds, alpha, meta)``."""
    feat, mid, timestamps = build_cache(
        config,
        extract_features_fn=extract_features,
        n_features_fn=n_features,
        tag="lob",
    )

    N = len(mid)
    train_end = int(N * config["train_frac"])
    val_end = int(N * (config["train_frac"] + config["val_frac"]))

    labels, alpha = build_labels(mid, config, train_end)

    k, t_past, stride = config["label_k"], config["T_past"], config["stride"]
    train_starts = _valid_starts(0, train_end, t_past, k, labels, timestamps, stride)
    val_starts = _valid_starts(
        train_end, val_end, t_past, k, labels, timestamps, stride
    )
    test_starts = _valid_starts(val_end, N, t_past, k, labels, timestamps, stride)

    logger.info(
        "windows — train:{} val:{} test:{}",
        len(train_starts),
        len(val_starts),
        len(test_starts),
    )

    def _balance(starts):
        lbl = labels[[s + t_past - 1 for s in starts]]
        c = np.bincount(lbl, minlength=3) / max(len(lbl), 1)
        return {"down": float(c[0]), "stationary": float(c[1]), "up": float(c[2])}

    meta = {
        "counts": {
            "train": len(train_starts),
            "val": len(val_starts),
            "test": len(test_starts),
        },
        "class_balance": _balance(train_starts),
        "alpha": alpha,
        "n_features": feat.shape[1],
        "n_snapshots": N,
    }

    train_ds = LOBDataset(feat, train_starts, labels, t_past)
    val_ds = LOBDataset(feat, val_starts, labels, t_past)
    test_ds = LOBDataset(feat, test_starts, labels, t_past)
    return train_ds, val_ds, test_ds, alpha, meta
