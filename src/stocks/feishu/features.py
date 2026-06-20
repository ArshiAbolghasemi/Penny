"""Feature engineering for Feishu A-share LOB + daily data.

Two modes controlled by ``config["feature_mode"]``:

  ``"ofi"`` — 24-slot intraday OFI (240) + 19 OHLCV  = 259 features
  ``"lob"`` — closing-snapshot price/vol (4n) + 19 OHLCV = 4n+19 features

OFI features (240)
------------------
24 intraday time slots × 10 LOB levels.  Slot OFI is the sum of per-tick
signed Cont-OFI within each slot.  Normalised with a causal 5-day rolling
z-score (no lookahead).

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
the train script, not here).

After all normalisations, values are clipped to [-5, 5].

Chinese A-share trading hours (24 slots)
-----------------------------------------
  Morning:    9:40–11:20  (11 slots every 10 min)
  Afternoon: 13:00–15:00  (13 slots every 10 min)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS: list[str] = [
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
    mode = config.get("feature_mode", "ofi")
    n = config.get("n_lob_levels", N_LEVELS)
    if mode == "ofi":
        return N_OFI + N_OHLCV  # 259
    else:  # lob
        return 4 * n + N_OHLCV  # 59 for n=10


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
    """Aggregate per-tick OFI into 24 intraday slots.

    Args:
        ofi_df:  Per-tick OFI DataFrame (output of :func:`compute_ofi_tick`).
        day_df:  Companion rows with a ``time`` column (``HH:MM`` prefix).

    Returns:
        ``(N_SLOTS, N_LEVELS)`` float32 array; missing slots are zero.
    """
    combined = ofi_df.copy()
    combined["slot"] = day_df["time"].astype(str).str[:5]

    ofi_cols = [f"ofi_{i}" for i in range(N_LEVELS)]
    out = np.zeros((N_SLOTS, N_LEVELS), dtype=np.float32)
    for j, slot in enumerate(SLOTS):
        mask = combined["slot"] == slot
        if mask.any():
            out[j] = combined.loc[mask, ofi_cols].sum().values.astype(np.float32)
    return out


def causal_rolling_zscore(
    matrix: np.ndarray,
    window: int = N_OFI_ROLL,
) -> np.ndarray:
    """Causal rolling z-score applied feature-wise (no lookahead).

    For day ``t``, statistics come from days ``[max(0, t-window), t)``
    (past only).  Day 0 bootstraps from its own row.

    Args:
        matrix: ``(n_days, n_feat)`` float32 array.
        window: Look-back in trading days (default 5).

    Returns:
        Same-shape float32 array.
    """
    n_days = matrix.shape[0]
    out = np.empty_like(matrix, dtype=np.float32)
    for t in range(n_days):
        start = max(0, t - window)
        past = matrix[start:t] if t > 0 else matrix[:1]
        mu = past.mean(axis=0)
        sigma = past.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
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

    return df[OHLCV_COLS]
