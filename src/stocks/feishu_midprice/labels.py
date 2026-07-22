"""Mid-price trend labeling for Feishu A-share equity data.

Unlike the ``stocks.feishu`` task (which labels the ``(close_{t+1} - vwap_t)``
traded-price return that straddles the overnight gap), this variant is a genuine
**mid-price trend prediction**: both endpoints are the quote mid-price

    mid_t = (best_bid_t + best_ask_t) / 2

taken at a consistent daily reference snapshot (the closing snapshot, ~15:00).
The label is the sign of the mid-to-mid return over a configurable horizon of
``h`` trading days:

    return_t = (mid_{t+h} - mid_t) / mid_t

    0 = Down       (return_t <  lo)
    1 = Stationary (lo <= return_t <= hi)
    2 = Up         (return_t >  hi)

Class boundaries — balanced tertiles
------------------------------------
``lo`` and ``hi`` are the **tertile cut points** (default 1/3 and 2/3 quantiles)
of the return distribution, so the three classes come out ~33 / 33 / 33 instead
of the flat-heavy split a fixed symmetric ``alpha`` produces. They are calibrated
on the **training rows only** (see :func:`compute_class_thresholds`, invoked from
:func:`stocks.feishu_midprice.build.build_datasets`) so no val/test information
leaks into the label definition; the same two scalars are then applied to every
split. A fixed symmetric band is still available via ``label_mode="fixed"``
(``lo = -alpha``, ``hi = +alpha``).

Horizon ``h`` is the only axis that changes between the sweep configs
(``configs/stocks/feishu_midprice/<model>_h<h>.json``); ``h = 1`` reproduces the
next-day mid move, larger ``h`` predicts longer trends.

Causality mirrors ``stocks.feishu``: a T-day window ending on day ``t-1`` is
paired with the label anchored at ``mid_t`` (the entry day, one day after the
window ends) and resolved at ``mid_{t+h}``. No *feature* ever reads ``mid_t`` or
``mid_{t+h}`` — the mid series is used only to form labels — so there is no
lookahead, exactly as the entry ``vwap`` is unknown-but-unused in the base task.

Note: mid returns use raw (un-adjusted) intraday prices, matching the base
task's raw ``close``/``vwap`` convention. A-share corporate actions inside a
short ``h`` are rare; adjust upstream if you extend to long horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocks.feishu.labels import DOWN, FLAT, UP, assign_labels

__all__ = [
    "DOWN",
    "FLAT",
    "UP",
    "assign_labels",
    "assign_labels_quantile",
    "compute_class_thresholds",
    "compute_mid_forward_returns",
]


def compute_mid_forward_returns(mid: pd.Series | np.ndarray, horizon: int) -> pd.Series:
    """Compute per-day horizon-``h`` forward mid-price returns.

    return_t = (mid_{t+h} - mid_t) / |mid_t|

    The last ``horizon`` rows always yield NaN (no mid ``h`` days ahead), and any
    row whose anchor mid is missing / non-positive also yields NaN so the dataset
    builder masks it out.

    Args:
        mid:     Per-asset closing mid-price series, **sorted by date ascending**.
        horizon: Forward horizon in trading days (>= 1).

    Returns:
        Float Series of forward mid returns, index aligned to ``mid``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    s = pd.Series(np.asarray(mid, dtype=np.float64))
    anchor = s.where(s > 0)  # non-positive / zero mids are invalid anchors
    future = anchor.shift(-horizon)
    return (future - anchor) / anchor.abs()


def compute_class_thresholds(
    returns: np.ndarray | pd.Series,
    quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
) -> tuple[float, float]:
    """Return the ``(lo, hi)`` return cut points for balanced tertile classes.

    ``lo`` and ``hi`` are the requested quantiles of the *finite* returns, so
    labeling with them yields ~equal Down / Flat / Up masses. Pass the
    **training-split returns only** to keep the label definition leakage-free.

    A near-zero-mass return distribution (e.g. many identical mids at a short
    horizon) can collapse ``lo == hi``; the Flat class then simply absorbs the
    tie mass — the tertiles are still the best equal split the data allows.

    Args:
        returns:   Forward returns (NaNs allowed; ignored).
        quantiles: The two cut quantiles, ascending. Default ``(1/3, 2/3)``.

    Returns:
        ``(lo, hi)`` floats; ``(0.0, 0.0)`` if no finite returns are present.
    """
    q_lo, q_hi = quantiles
    if not (0.0 <= q_lo <= q_hi <= 1.0):
        raise ValueError(f"quantiles must be ascending in [0, 1], got {quantiles}")
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0, 0.0
    lo, hi = np.quantile(r, [q_lo, q_hi])
    return float(lo), float(hi)


def assign_labels_quantile(
    returns: np.ndarray | pd.Series, lo: float, hi: float
) -> np.ndarray:
    """Map returns to {0, 1, 2} by two cut points; NaN → -1 (masked).

    Down = ``r < lo``, Up = ``r > hi``, Stationary = ``lo <= r <= hi`` (tie mass
    at a cut point falls into Flat). With ``lo, hi`` set to the 1/3, 2/3 return
    quantiles this produces balanced classes; with ``lo = -alpha, hi = +alpha``
    it reproduces the fixed symmetric band.

    Args:
        returns: Forward return series.
        lo:      Down/Flat cut point.
        hi:      Flat/Up cut point (``>= lo``).

    Returns:
        int64 array of length ``len(returns)``; -1 marks invalid / missing.
    """
    r = np.asarray(returns, dtype=np.float64)
    labels = np.full(len(r), -1, dtype=np.int64)
    valid = np.isfinite(r)
    labels[valid & (r < lo)] = DOWN
    labels[valid & (r >= lo) & (r <= hi)] = FLAT
    labels[valid & (r > hi)] = UP
    return labels
