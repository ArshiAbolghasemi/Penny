"""Feature engineering for Feishu A-share LOB + daily data.

Two modes controlled by ``config["feature_mode"]``:

  ``"ofi"`` — 24-slot intraday OFI (480) + 19 OHLCV  = 499 features
  ``"lob"`` — closing-snapshot price/vol (4n) + 19 OHLCV = 4n+19 features

OFI features (480)
------------------
24 intraday time slots × 10 LOB levels × 2 (bid OFI + ask OFI).
Matches the crypto feature convention: bid and ask OFI are stored
separately with a signed-log transform applied to each, rather than
combined as a net value.  Slot OFI is placed at the exact 10-min mark
(09:40, 09:50, …); off-grid 5-min bars are zero-filled.
Normalised with a causal 5-day rolling z-score (no lookahead).

Feature layout per slot (20 values):
  [0 : N_LEVELS)   signed-log bid OFI for levels 0–9
  [N_LEVELS : 2*N_LEVELS) signed-log ask OFI for levels 0–9

LOB features (4n for n levels)
-------------------------------
Last intraday snapshot of each trading day:
  [0   : n)    bid price offset  = (mid - bid_p[i]) / mid
  [n   : 2n)   ask price offset  = (ask_p[i] - mid) / mid
  [2n  : 3n)   log1p bid volume per level
  [3n  : 4n)   log1p ask volume per level
Normalised with per-asset causal 5-day rolling z-score.

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
N_OFI: int = N_SLOTS * N_LEVELS * 2  # 480  (bid + ask OFI, matching crypto)

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


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def n_features(config: dict) -> int:
    """Total feature count for the given config."""
    mode = config.get("feature_mode", "ofi")
    n = config.get("n_lob_levels", N_LEVELS)
    if mode == "ofi":
        return N_OFI + N_OHLCV  # 499
    else:  # lob
        return 4 * n + N_OHLCV  # 59 for n=10


# ── tick-level OFI ────────────────────────────────────────────────────────────


def compute_ofi_tick(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cont-OFI for all 10 levels at each tick (crypto convention).

    Bid and ask OFI are stored separately with a signed-log transform,
    matching the crypto feature pipeline.

    Args:
        df: One day's intraday LOB snapshot data with columns
            ``bid_price_{0..9}``, ``bid_volume_{0..9}``,
            ``ask_price_{0..9}``, ``ask_volume_{0..9}``
            (0-indexed, best bid/ask at level 0).

    Returns:
        DataFrame with columns ``bofi_0 … bofi_9``, ``aofi_0 … aofi_9``
        (20 columns total), aligned to ``df``.
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
        result[f"bofi_{i}"] = _signed_log(bofi)
        result[f"aofi_{i}"] = _signed_log(aofi)

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
                 Columns: ``bofi_0 … bofi_9``, ``aofi_0 … aofi_9``.
        day_df:  Companion rows with a ``time`` column.

    Returns:
        ``(N_SLOTS, N_LEVELS * 2)`` float32 array; un-matched slots are zero.
        Column order: bid OFI levels 0–9, then ask OFI levels 0–9.
    """
    all_cols = [f"bofi_{i}" for i in range(N_LEVELS)] + [
        f"aofi_{i}" for i in range(N_LEVELS)
    ]
    ofi_arr = ofi_df[all_cols].values.astype(np.float32)  # (T, 2*N_LEVELS)

    # Accept both "HH:MM" and "HH:MM:SS" by stripping to first 5 chars for lookup
    times = day_df["time"].astype(str).str[:5]
    slot_idx = times.map(_SLOT_TO_IDX).to_numpy()

    out = np.zeros((N_SLOTS, N_LEVELS * 2), dtype=np.float32)
    keep = ~pd.isna(slot_idx)
    if keep.any():
        valid_idx = slot_idx[keep].astype(np.int64)
        out[valid_idx] = ofi_arr[keep]  # last one wins for any duplicate
    return out


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


def extract_lob_day(day_df: pd.DataFrame, n_levels: int = N_LEVELS) -> np.ndarray:
    """Extract LOB features from the last intraday snapshot of one trading day.

    Args:
        day_df:   One day's intraday data (any number of ticks); last row used.
        n_levels: Number of LOB levels (default 10).

    Returns:
        ``(4 * n_levels,)`` float32: bid price offsets, ask price offsets,
        log1p bid volumes, log1p ask volumes.
    """
    last = day_df.iloc[-1]
    eps = 1e-12

    bid_p = np.array(
        [last[f"bid_price_{i}"] for i in range(n_levels)], dtype=np.float64
    )
    bid_v = np.array(
        [last[f"bid_volume_{i}"] for i in range(n_levels)], dtype=np.float64
    )
    ask_p = np.array(
        [last[f"ask_price_{i}"] for i in range(n_levels)], dtype=np.float64
    )
    ask_v = np.array(
        [last[f"ask_volume_{i}"] for i in range(n_levels)], dtype=np.float64
    )

    mid = (bid_p[0] + ask_p[0]) / 2.0
    bid_off = (mid - bid_p) / (mid + eps)
    ask_off = (ask_p - mid) / (mid + eps)

    return np.concatenate([bid_off, ask_off, np.log1p(bid_v), np.log1p(ask_v)]).astype(
        np.float32
    )


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
