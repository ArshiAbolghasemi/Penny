"""DeepLOB trend labels for the TimesFM forecaster (spec section 3).

Class encoding: ``0 = down``, ``1 = stationary``, ``2 = up``.
"""

from __future__ import annotations

import numpy as np

DOWN, STATIONARY, UP = 0, 1, 2


def smoothed_backward_mid(mid: np.ndarray, t_past: int, k: int) -> float:
    return float(np.mean(mid[t_past - k : t_past]))


def smoothed_forward_mid(mid: np.ndarray, t_past: int, k: int) -> float:
    return float(np.mean(mid[t_past : t_past + k]))


def compute_l(mid: np.ndarray, t_past: int, k: int) -> float:
    bwd = smoothed_backward_mid(mid, t_past, k)
    fwd = smoothed_forward_mid(mid, t_past, k)
    return (fwd - bwd) / (bwd + 1e-12)


def label_from_l(trend_value: float, alpha: float) -> int:
    if trend_value > alpha:
        return UP
    if trend_value < -alpha:
        return DOWN
    return STATIONARY


def calibrate_alpha(l_values: np.ndarray, stationary_frac: float = 1.0 / 3.0) -> float:
    """``alpha`` = the ``stationary_frac`` quantile of ``|l|`` (≈balanced thirds)."""
    return float(np.quantile(np.abs(l_values), stationary_frac))
