"""Forward return labeling for Feishu A-share equity data.

Label definition:
    return_t = (close_{t+1} - vwap_{t}) / |vwap_{t}|

    0 = Down       (return_t < -alpha)
    1 = Stationary (|return_t| <= alpha)
    2 = Up         (return_t >  alpha)

Threshold values used in the literature:
    v2: alpha = 0.015  (±1.5 % — original Shifu paper)
    v3: alpha = 0.002  (±0.2 % — stricter, more balanced classes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2


def compute_forward_returns(daily_df: pd.DataFrame) -> pd.Series:
    """Compute per-day forward returns aligned to the input index.

    return_t = (close_{t+1} - vwap_{t}) / |vwap_{t}|

    The last row always yields NaN (no next day).

    Args:
        daily_df: Rows sorted by date ascending, with columns ``close`` and
                  ``vwap_0930_0935``.

    Returns:
        Series of float forward returns, index aligned to ``daily_df``.
    """
    df = daily_df.sort_values("date").reset_index(drop=True)
    next_close = df["close"].shift(-1)
    vwap = df["vwap_0930_0935"]
    return (next_close - vwap) / vwap.abs()


def assign_labels(returns: pd.Series, alpha: float) -> np.ndarray:
    """Map continuous returns to {0, 1, 2}; NaN → -1 (masked).

    Args:
        returns: Forward return series.
        alpha:   Symmetric threshold in return units (e.g. 0.015 = 1.5 %).

    Returns:
        int64 array of length ``len(returns)``.
        -1 indicates invalid / missing label (to be filtered out by the dataset builder).
    """
    labels = np.full(len(returns), -1, dtype=np.int64)
    r = returns.values
    valid = np.isfinite(r)
    labels[valid & (r < -alpha)] = DOWN
    labels[valid & (np.abs(r) <= alpha)] = FLAT
    labels[valid & (r > alpha)] = UP
    return labels
