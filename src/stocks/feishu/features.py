"""Feature engineering for Feishu A-share LOB + daily data.

Feature mode: 24-slot intraday OFI (240) + 19 OHLCV = 259 features.

OFI features (240)
------------------
24 intraday time slots × 10 LOB levels.  Slot OFI is placed at the exact
10-min mark (09:40, 09:50, …); off-grid 5-min bars are zero-filled.
Normalised with a causal 5-day rolling z-score (no lookahead).

OHLCV features (19)
-------------------
14 engineered daily features + 5 raw (open, close, volume, low, high).
Cross-sectional z-scored across all assets on the same day (done in
build.py, not here).

After all normalisations, values are clipped to [-5, 5].

Chinese A-share trading hours (24 slots, 10-min grid)
------------------------------------------------------
  Morning:    9:40–11:20  (11 slots every 10 min)
  Afternoon: 13:00–15:00  (13 slots every 10 min)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS: list[str] = [
    # Morning session 09:40–11:20 (11 slots, 10-min)
    "09:40",
    "09:50",
    "10:00",
    "10:10",
    "10:20",
    "10:30",
    "10:40",
    "10:50",
    "11:00",
    "11:10",
    "11:20",
    # Afternoon session 13:00–15:00 (13 slots, 10-min)
    "13:00",
    "13:10",
    "13:20",
    "13:30",
    "13:40",
    "13:50",
    "14:00",
    "14:10",
    "14:20",
    "14:30",
    "14:40",
    "14:50",
    "15:00",
]
N_SLOTS: int = len(SLOTS)  # 24
N_LEVELS: int = 10
N_OFI: int = N_SLOTS * N_LEVELS  # 240

# Precomputed for O(1) lookup in snap_to_slots
_SLOT_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SLOTS)}
N_OFI_ROLL: int = 5

OHLCV_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "amihud",
    "volume_zscore",
    "rsi_14",
    "ma_dist_5",
    "ma_dist_20",
    "open_close_ret",
    "high_low_range",
    "close_vwap_dist",
    "open",
    "close",
    "volume",
    "low",
    "high",
]
N_OHLCV: int = len(OHLCV_COLS)  # 19
CLIP_VAL: float = 5.0


def n_features(config: dict) -> int:
    """Total feature count for the given config."""
    return N_OFI + N_OHLCV  # 259


# ── tick-level OFI ────────────────────────────────────────────────────────────


def compute_ofi_tick(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signed Cont-OFI for all 10 levels at each tick.

    Args:
        df: One day's intraday LOB snapshot data with columns
            ``bid_price_{0..9}``, ``bid_volume_{0..9}``,
            ``ask_price_{0..9}``, ``ask_volume_{0..9}``
            (0-indexed, best bid/ask at level 0).

    Returns:
        DataFrame with columns ``ofi_0 … ofi_9``, aligned to ``df``.
    """
    result = pd.DataFrame(index=df.index)
    for i in range(N_LEVELS):
        prev_bp = df[f"bid_price_{i}"].shift(1)
        prev_bv = df[f"bid_volume_{i}"].shift(1)
        prev_ap = df[f"ask_price_{i}"].shift(1)
        prev_av = df[f"ask_volume_{i}"].shift(1)

        dp_b = df[f"bid_price_{i}"] - prev_bp
        bofi = np.where(
            dp_b > 0,
            df[f"bid_volume_{i}"],
            np.where(dp_b < 0, -prev_bv, df[f"bid_volume_{i}"] - prev_bv),
        )
        dp_a = df[f"ask_price_{i}"] - prev_ap
        aofi = np.where(
            dp_a < 0,
            df[f"ask_volume_{i}"],
            np.where(dp_a > 0, -prev_av, df[f"ask_volume_{i}"] - prev_av),
        )
        result[f"ofi_{i}"] = bofi - aofi

    result.iloc[0] = 0.0  # first tick has no prior state
    return result.fillna(0.0)


def snap_to_slots(ofi_df: pd.DataFrame, day_df: pd.DataFrame) -> np.ndarray:
    """Place per-tick OFI into the 24 intraday 10-min slot grid.

    Only ticks whose ``time`` exactly matches a slot string (``HH:MM`` or
    ``HH:MM:SS``) are placed; off-grid ticks (e.g. the 5-min bars between
    10-min marks) contribute zero.  When multiple ticks share a slot the last
    one wins (consistent with the reference notebook behaviour).

    Args:
        ofi_df:  Per-tick OFI DataFrame (output of :func:`compute_ofi_tick`).
        day_df:  Companion rows with a ``time`` column.

    Returns:
        ``(N_SLOTS, N_LEVELS)`` float32 array; un-matched slots are zero.
    """
    ofi_cols = [f"ofi_{i}" for i in range(N_LEVELS)]
    ofi_arr = ofi_df[ofi_cols].values.astype(np.float32)

    # Accept both "HH:MM" and "HH:MM:SS" by stripping to first 5 chars for lookup
    times = day_df["time"].astype(str).str[:5]
    slot_idx = times.map(_SLOT_TO_IDX).to_numpy()

    out = np.zeros((N_SLOTS, N_LEVELS), dtype=np.float32)
    keep = ~pd.isna(slot_idx)
    if keep.any():
        valid_idx = slot_idx[keep].astype(np.int64)
        out[valid_idx] = ofi_arr[keep]  # last one wins for any duplicate
    return out


def compute_ofi_slots_chunk(
    df: pd.DataFrame,
    group_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised (asset, day)-wise OFI → slot grid over a whole streamed chunk.

    Equivalent to running :func:`compute_ofi_tick` + :func:`snap_to_slots` on
    every ``(asset, day)`` group of *df* separately, but computed with a single
    pass of array ops instead of ~1M pandas groupby iterations.  The tick shift
    is masked at group boundaries so no OFI ever crosses an asset or a day.

    Args:
        df:       Chunk rows **already sorted** by ``(asset, day, time)``, with
                  0-indexed ``bid/ask_price/volume_{0..9}`` columns and ``time``.
        group_id: ``(n_rows,)`` int64 group label per row (one per (asset, day)),
                  constant within a group and changing at every group boundary.

    Returns:
        ``(rows, cols, vals)`` scatter triplet where ``vals[k]`` belongs at
        slot-grid position ``cols[k]`` of group ``rows[k]``.  ``cols`` indexes
        the flattened ``(N_SLOTS, N_LEVELS)`` grid, i.e. ``slot * N_LEVELS + level``.
        Off-grid ticks are dropped; duplicates keep last-write-wins semantics
        when scattered in order.
    """
    n = len(df)
    if n == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)

    bid_p = df[[f"bid_price_{i}" for i in range(N_LEVELS)]].to_numpy(np.float32)
    bid_v = df[[f"bid_volume_{i}" for i in range(N_LEVELS)]].to_numpy(np.float32)
    ask_p = df[[f"ask_price_{i}" for i in range(N_LEVELS)]].to_numpy(np.float32)
    ask_v = df[[f"ask_volume_{i}" for i in range(N_LEVELS)]].to_numpy(np.float32)

    def _shift(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[0] = np.nan
        out[1:] = a[:-1]
        return out

    prev_bp, prev_bv = _shift(bid_p), _shift(bid_v)
    prev_ap, prev_av = _shift(ask_p), _shift(ask_v)

    dp_b = bid_p - prev_bp
    bofi = np.where(dp_b > 0, bid_v, np.where(dp_b < 0, -prev_bv, bid_v - prev_bv))
    dp_a = ask_p - prev_ap
    aofi = np.where(dp_a < 0, ask_v, np.where(dp_a > 0, -prev_av, ask_v - prev_av))
    ofi = np.nan_to_num(bofi - aofi, nan=0.0).astype(np.float32)

    # first tick of every (asset, day) group has no prior state → 0
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = group_id[1:] != group_id[:-1]
    ofi[first] = 0.0

    slot_idx = (
        df["time"].astype(str).str[:5].map(_SLOT_TO_IDX).to_numpy(dtype=np.float64)
    )
    keep = ~np.isnan(slot_idx)
    if not keep.any():
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)

    slot = slot_idx[keep].astype(np.int64)
    rows = np.repeat(group_id[keep], N_LEVELS)
    cols = (slot[:, None] * N_LEVELS + np.arange(N_LEVELS)[None, :]).reshape(-1)
    vals = ofi[keep].reshape(-1)
    return rows, cols, vals


def causal_rolling_zscore(
    matrix: np.ndarray,
    window: int = N_OFI_ROLL,
) -> np.ndarray:
    """Causal rolling z-score applied feature-wise (no lookahead).

    For day ``t``, statistics come from days ``[max(0, t-window), t)``
    (past only).

    CRITICAL: only normalise a feature when its rolling std is meaningfully
    non-zero. The OFI grid is sparse — most (slot, level) cells are zero-filled,
    so their look-back std is ~0. Dividing a non-zero numerator by a tiny std
    explodes the value to ~1e14 and poisons every downstream model (NaN losses).
    Such flat features are therefore left mean-centred only (÷1.0). Warm-up days
    with fewer than 2 past days carry no reliable statistics, so they are left at
    zero rather than scaled by an arbitrary constant. This mirrors the validated
    notebook procedure.

    Args:
        matrix: ``(n_days, n_feat)`` float32 array.
        window: Look-back in trading days (default 5).

    Returns:
        Same-shape float32 array; warm-up rows (< 2 past days) are zero.
    """
    n_days = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float32)
    for t in range(n_days):
        past = matrix[max(0, t - window) : t]
        if len(past) < 2:
            continue  # warm-up: no reliable stats → leave row at 0
        mu = past.mean(axis=0)
        sigma = past.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)  # don't blow up flat features
        out[t] = (matrix[t] - mu) / sigma
    return out


# ── closing-snapshot LOB features ─────────────────────────────────────────────


# ── daily OHLCV features ──────────────────────────────────────────────────────


def compute_ohlcv_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 14 engineered + 5 raw daily features = 19 total.

    Args:
        daily_df: Per-asset daily rows sorted by date ascending, with columns
                  ``open``, ``high``, ``low``, ``close``, ``volume``,
                  ``vwap_0930_0935``.

    Returns:
        DataFrame with columns ``OHLCV_COLS``, same row count as input.
        Rows with insufficient history contain NaN.
    """
    df = daily_df.sort_values("date").reset_index(drop=True)

    log_ret = np.log(df["close"] / df["close"].shift(1))

    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)
    df["vol_5d"] = log_ret.rolling(5).std()
    df["vol_20d"] = log_ret.rolling(20).std()

    value = df["close"] * df["volume"]
    df["amihud"] = (df["ret_1d"].abs() / value).replace([np.inf, -np.inf], np.nan)

    vol_ma20 = df["volume"].rolling(20).mean()
    vol_std20 = df["volume"].rolling(20).std().replace(0, 1.0)
    df["volume_zscore"] = (df["volume"] - vol_ma20) / vol_std20

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-8))

    df["ma_dist_5"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["ma_dist_20"] = df["close"] / df["close"].rolling(20).mean() - 1

    df["open_close_ret"] = (df["close"] - df["open"]) / (df["open"].abs() + 1e-8)
    df["high_low_range"] = (df["high"] - df["low"]) / (df["close"].abs() + 1e-8)
    df["close_vwap_dist"] = (df["close"] - df["vwap_0930_0935"]) / (
        df["vwap_0930_0935"].abs() + 1e-8
    )

    return df[OHLCV_COLS].fillna(0.0)
