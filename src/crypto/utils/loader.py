"""Parquet-based LOB data loader with per-day normalization.

Reads pre-resampled parquet files produced by ``scripts/resample_binance.py``:

    data/resampled/{SYMBOL}.parquet.gz

Each file contains all available dates for one symbol, already with trades and
quotes joined and resampled to a fixed interval (default 10 s).  Features are
z-scored using each calendar day's own mean and std (no lookahead).

The normalized feature array is written to a numpy memmap (one-time build).
Subsequent calls return the cached memmap immediately.

Usage
-----
    from crypto.utils.loader import build_cache
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


def _cache_paths(config: dict, tag: str) -> dict[str, Path]:
    cache = Path(config["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    sym = config["symbol"]
    n = config["n_lob_levels"]
    mode = config.get("feature_mode", "ofi")
    prefix = cache / f"{sym}_n{n}_{mode}_{tag}"
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

    Features are z-scored per calendar day.  Cache is rebuilt only when the
    .npy files are absent.

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
    df["_date"] = df["timestamp_utc"].dt.date

    N_total = len(df)
    feat_mm = np.memmap(paths["feat"], dtype=np.float32, mode="w+", shape=(N_total, F))
    mid_arr = np.empty(N_total, dtype=np.float64)
    ts_arr = np.empty(N_total, dtype=np.int64)

    ptr = 0
    for date, day_df in df.groupby("_date"):
        day_df = day_df.reset_index(drop=True)
        N_day = len(day_df)
        logger.info("  {} — {} rows", date, N_day)

        raw = extract_features_fn(day_df, config)  # (N_day, F) float32

        day_mean = raw.mean(axis=0)
        day_std = raw.std(axis=0)
        day_std[day_std < 1e-8] = 1.0
        norm = ((raw - day_mean) / day_std).astype(np.float32)

        feat_mm[ptr : ptr + N_day] = norm
        mid_arr[ptr : ptr + N_day] = day_df["mid"].values
        ts_arr[ptr : ptr + N_day] = day_df["bin"].values.astype(np.int64)
        ptr += N_day

    feat_mm.flush()
    del feat_mm

    np.save(paths["mid"], mid_arr[:ptr])
    np.save(paths["ts"], ts_arr[:ptr])
    logger.info("cache '{}' built: {:,} rows → {}", tag, ptr, paths["feat"])

    feat = np.memmap(paths["feat"], dtype=np.float32, mode="r", shape=(ptr, F))
    return feat, mid_arr[:ptr], ts_arr[:ptr]
