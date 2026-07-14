"""Shared walk-forward evaluation harness for classical (ARIMA/VAR) baselines.

Unlike the neural models, ARIMA/VAR have no learned weights to checkpoint: each
evaluation point is forecast from a model re-fit on the trailing ``T_past``
history at that point only (no lookahead). ``eval_stride`` subsamples the
evaluation points since re-fitting per point is expensive; it defaults to
``max(stride, label_k)`` so points don't overlap their own label horizon.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score

from .dataset import _valid_starts
from .labels import DOWN, STATIONARY, UP, build_labels


def trend_class(trend_ratio: float, alpha: float) -> int:
    if not np.isfinite(trend_ratio):
        return STATIONARY
    if trend_ratio < -alpha:
        return DOWN
    if trend_ratio > alpha:
        return UP
    return STATIONARY


def eval_points(
    mid: np.ndarray, timestamps: np.ndarray, config: dict, train_end: int, val_end: int
):
    """Return ``(splits, labels, alpha)``; ``splits["val"/"test"]`` are centre
    indices to forecast at, matching ``dataset.py``'s ``centre = s + T_past - 1``."""
    k, t_past, stride = config["label_k"], config["T_past"], config["stride"]
    labels, alpha = build_labels(mid, config, train_end)
    eval_stride = int(config.get("eval_stride", max(stride, k)))

    def centres(lo, hi):
        starts = _valid_starts(lo, hi, t_past, k, labels, timestamps, eval_stride)
        return [int(s + t_past - 1) for s in starts]

    splits = {"val": centres(train_end, val_end), "test": centres(val_end, len(mid))}
    return splits, labels, alpha


def report(y_true, y_pred, name: str = "TEST") -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    logger.info(
        "{}  accuracy={:.4f}  macro_f1={:.4f}  n={}", name, acc, f1, len(y_true)
    )
    logger.info("{}  confusion (rows=true down/stat/up):\n{}", name, cm)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "n": int(len(y_true)),
        "confusion": cm.tolist(),
    }
