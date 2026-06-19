"""DeepLOB trend labels (spec section 3).

Pure functions for: the smoothed-mid trend ratio ``l``, the 3-class label, and
the training-set ``alpha`` calibration for balanced classes.  Class encoding:
``0 = down``, ``1 = stationary``, ``2 = up``.
"""

from __future__ import annotations

import numpy as np

DOWN, STATIONARY, UP = 0, 1, 2


def smoothed_backward_mid(mid: np.ndarray, t_past: int, k: int) -> float:
    """Mean of the ``k`` mids ending at the boundary — cols ``[t_past-k, t_past)``."""
    return float(np.mean(mid[t_past - k : t_past]))


def smoothed_forward_mid(mid: np.ndarray, t_past: int, k: int) -> float:
    """Mean of the first ``k`` future mids — cols ``[t_past, t_past+k)``."""
    return float(np.mean(mid[t_past : t_past + k]))


def trend_ratio(forward_mid: float, backward_mid: float) -> float:
    """``l = (fwd - bwd) / bwd``."""
    return (forward_mid - backward_mid) / (backward_mid + 1e-12)


def compute_l(mid: np.ndarray, t_past: int, k: int) -> float:
    """Trend ratio ``l`` from a full ground-truth mid window."""
    bwd = smoothed_backward_mid(mid, t_past, k)
    fwd = smoothed_forward_mid(mid, t_past, k)
    return trend_ratio(fwd, bwd)


def label_from_l(trend_value: float, alpha: float) -> int:
    """Map a trend ratio to a class using threshold ``alpha``."""
    if trend_value > alpha:
        return UP
    if trend_value < -alpha:
        return DOWN
    return STATIONARY


def calibrate_alpha(l_values: np.ndarray, stationary_frac: float = 1.0 / 3.0) -> float:
    """``alpha`` = the ``stationary_frac`` quantile of ``|l|`` (≈balanced thirds).

    The literal "66.7th percentile" wording in the spec would instead make 2/3 of
    windows stationary; we follow the stated "one third each class" goal.
    """
    return float(np.quantile(np.abs(l_values), stationary_frac))
