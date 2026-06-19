"""Trend labels from the smoothed mid-price (Binance / DeepLOB formulation).

    bwd         = mean(mid[t - k : t])
    fwd         = mean(mid[t : t + k])
    trend_ratio = (fwd - bwd) / bwd

Label assignment:
    0 (down)       if trend_ratio < -alpha
    1 (stationary) if |trend_ratio| <= alpha
    2 (up)         if trend_ratio > alpha

``alpha`` is calibrated on the **training set** to the 33.3rd percentile of
|trend_ratio|, producing roughly balanced class frequencies.

Shared by ``crypto.deeplob``, ``crypto.jointdiff``, and ``crypto.timesfm``.
"""

from __future__ import annotations

import numpy as np


DOWN, STATIONARY, UP = 0, 1, 2


def compute_trend_series(mid: np.ndarray, k: int) -> np.ndarray:
    """Return per-snapshot trend ratio; NaN for the first and last ``k`` positions."""
    N = len(mid)
    trend = np.full(N, np.nan, dtype=np.float64)
    cs = np.cumsum(np.concatenate([[0.0], mid]))
    for t in range(k, N - k):
        bwd = (cs[t] - cs[t - k]) / k
        fwd = (cs[t + k] - cs[t]) / k
        if bwd > 1e-12:
            trend[t] = (fwd - bwd) / bwd
    return trend


def calibrate_alpha(trend_train: np.ndarray) -> float:
    """33rd percentile of |trend_ratio| on training data — yields balanced classes."""
    valid = trend_train[np.isfinite(trend_train)]
    return float(np.percentile(np.abs(valid), 100.0 / 3.0))


def assign_labels(trend: np.ndarray, alpha: float) -> np.ndarray:
    """Map trend ratios → {0, 1, 2}; invalid positions get -1."""
    labels = np.full(len(trend), -1, dtype=np.int64)
    valid = np.isfinite(trend)
    labels[valid & (trend < -alpha)] = DOWN
    labels[valid & (np.abs(trend) <= alpha)] = STATIONARY
    labels[valid & (trend > alpha)] = UP
    return labels


def build_labels(
    mid: np.ndarray, config: dict, train_end: int
) -> tuple[np.ndarray, float]:
    """Return ``(labels, alpha)``.  ``labels[t] == -1`` for invalid positions."""
    k = config["label_k"]
    trend = compute_trend_series(mid, k)
    alpha = (
        float(config["label_alpha"])
        if config.get("label_alpha", -1) > 0
        else calibrate_alpha(trend[:train_end])
    )
    return assign_labels(trend, alpha), alpha
