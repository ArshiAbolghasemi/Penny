"""Parquet-based LOB data loader with causal rolling-window normalization.

Reads pre-resampled parquet files produced by ``scripts/resample_binance.py``:

    data/resampled/{SYMBOL}.parquet.gz

Each file contains all available dates for one symbol, already with trades and
quotes joined and resampled to a fixed interval (default 10 s).  Features are
z-scored **causally** with a trailing rolling window of ``norm_window`` candles
(default 2000 ≈ 1–4 days at 10 s).  For candle ``t`` the mean/std come from
candles ``[t - norm_window + 1, t]`` — past and present only, never future —
and the window spans day boundaries (it is not reset per day).  This removes the
intra-day lookahead of the old per-day scheme, where an early-morning candle was
normalized using the whole day's stats (including later candles).

The normalized feature array is written to a numpy memmap (one-time build).
Subsequent calls return the cached memmap immediately.

Usage
-----
    from crypto.loader import build_cache
    feat, mid, ts = build_cache(config, extract_features_fn, n_features_fn, tag)
    # feat : np.memmap (N, F) float32  — pre-normalized
    # mid  : np.ndarray (N,) float64
    # ts   : np.ndarray (N,) int64     — microseconds UTC (bin boundary)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger


def _rolling_mean_std(
    x: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal trailing rolling mean/std (inclusive of the current row).

    For row ``t`` the statistics use rows ``[max(0, t - window + 1), t]`` — past
    and present only, never future.  Computed in O(N·F) via cumulative sums, so
    the window may be large (default 2000) at no extra cost.

    Args:
        x:      ``(N, F)`` float64 raw feature matrix in chronological order.
        window: Trailing window length in candles.

    Returns:
        ``(mean, std, count)`` — ``mean``/``std`` are ``(N, F)``; ``count`` is
        ``(N, 1)`` (rows available in each window).  Population std (ddof=0).
    """
    n, f = x.shape
    zero = np.zeros((1, f), dtype=np.float64)
    cs = np.vstack([zero, np.cumsum(x, axis=0)])  # (N+1, F)
    cs2 = np.vstack([zero, np.cumsum(x * x, axis=0)])  # (N+1, F)
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    hi = idx + 1
    cnt = (hi - lo).reshape(-1, 1).astype(np.float64)  # (N, 1)
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    mean = s / cnt
    var = np.maximum(s2 / cnt - mean**2, 0.0)
    return mean, np.sqrt(var), cnt


def _cache_paths(config: dict, tag: str) -> dict[str, Path]:
    cache = Path(config["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    sym = config["symbol"]
    n = config["n_lob_levels"]
    mode = config.get("feature_mode", "ofi")
    norm_window = int(config.get("norm_window", 2000))
    # `roll{w}` in the key invalidates caches built under the old same-day
    # (intra-day-lookahead) normalization, and separates different windows.
    prefix = cache / f"{sym}_n{n}_{mode}_roll{norm_window}_{tag}"
    return {
        "feat": prefix.with_suffix(".feat.npy"),
        "mid": prefix.with_suffix(".mid.npy"),
        "ts": prefix.with_suffix(".ts.npy"),
    }


def build_cache(
    config: dict,
    extract_features_fn: Callable,
    n_features_fn: Callable,
    tag: str = "default",
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    """Return ``(features_memmap, mid_array, timestamps_array)``.

    Features are z-scored with a causal trailing rolling window of
    ``config["norm_window"]`` candles (default 2000).  Cache is rebuilt only when
    the .npy files are absent.

    Args:
        config:               Requires ``data_dir`` (path to resampled parquets),
                              ``cache_dir``, ``symbol``, ``n_lob_levels``,
                              ``feature_mode`` (``"ofi"``/``"lob"``).
        extract_features_fn:  ``(day_df, config) → (N, F) float32``
        n_features_fn:        ``(config) → int``
        tag:                  Short model-family string (e.g. ``"lob"``).
    """
    paths = _cache_paths(config, tag)
    F = n_features_fn(config)

    if all(p.exists() for p in paths.values()):
        mid = np.load(paths["mid"])
        ts = np.load(paths["ts"])
        N = len(mid)
        feat = np.memmap(paths["feat"], dtype=np.float32, mode="r", shape=(N, F))
        logger.info("loaded cache '{}': {:,} rows, {} features", tag, N, F)
        return feat, mid, ts

    symbol = config["symbol"]
    parquet = Path(config["data_dir"]) / f"{symbol}.parquet.gz"
    if not parquet.exists():
        script = (
            "resample_nobitex.py"
            if config.get("exchange") == "nobitex"
            else "resample_binance.py"
        )
        raise FileNotFoundError(
            f"Resampled parquet not found: {parquet}\n"
            f"Run:  uv run python scripts/{script}"
        )

    logger.info("building '{}' cache from {}", tag, parquet)
    df = pd.read_parquet(parquet)
    # Global chronological order is required for the rolling-window normalization
    # (the window spans day boundaries); also keeps each day's features causal.
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["_date"] = df["timestamp_utc"].dt.date

    N_total = len(df)
    feat_mm = np.memmap(paths["feat"], dtype=np.float32, mode="w+", shape=(N_total, F))
    mid_arr = np.empty(N_total, dtype=np.float64)
    ts_arr = np.empty(N_total, dtype=np.int64)

    # ── Pass 1: extract RAW (un-normalized) features per day ──────────────────
    # Feature extraction stays per-day because OFI/returns reset at day edges
    # (first row of each day has no prior tick); normalization is global (Pass 2).
    ptr = 0
    for date, day_df in df.groupby("_date", sort=True):
        day_df = day_df.reset_index(drop=True)
        N_day = len(day_df)
        logger.info("  {} — {} rows", date, N_day)

        raw = extract_features_fn(day_df, config)  # (N_day, F) float32
        feat_mm[ptr : ptr + N_day] = raw
        mid_arr[ptr : ptr + N_day] = day_df["mid"].values
        ts_arr[ptr : ptr + N_day] = day_df["bin"].values.astype(np.int64)
        ptr += N_day

    # ── Pass 2: causal trailing rolling z-score over the full time series ─────
    norm_window = int(config.get("norm_window", 2000))
    raw_all = np.asarray(feat_mm[:ptr], dtype=np.float64)  # (ptr, F)
    mean, std, cnt = _rolling_mean_std(raw_all, norm_window)
    # don't divide by a degenerate window: <2 candles (row 0) or ~flat feature
    std = np.where((cnt < 2) | (std < 1e-8), 1.0, std)
    feat_mm[:ptr] = ((raw_all - mean) / std).astype(np.float32)
    del raw_all, mean, std, cnt

    feat_mm.flush()
    del feat_mm

    np.save(paths["mid"], mid_arr[:ptr])
    np.save(paths["ts"], ts_arr[:ptr])
    logger.info("cache '{}' built: {:,} rows → {}", tag, ptr, paths["feat"])

    feat = np.memmap(paths["feat"], dtype=np.float32, mode="r", shape=(ptr, F))
    return feat, mid_arr[:ptr], ts_arr[:ptr]
