"""Multi-source Binance data loader with per-day normalization.

For each calendar day the loader reads three file types:
  binance_book_snapshot_25_{date}_{symbol}.csv.gz   — full LOB (primary)
  binance_trades_{date}_{symbol}.csv.gz             — trade ticks
  binance_quotes_{date}_{symbol}.csv.gz             — best-bid/ask updates

Trades and quote updates are aggregated to the snapshot time-grid via vectorized
searchsorted (no Python loops over rows).  Raw features are z-scored using that
day's own mean and std so no lookahead crosses day boundaries.

The normalized feature array is written to a numpy memmap (one-time build).
Subsequent calls return the cached memmap immediately.

Usage
-----
    from crypto.utils.loader import build_cache
    feat, mid, ts = build_cache(config, extract_features_fn, n_features_fn, tag)
    # feat : np.memmap (N, F) float32  — pre-normalized
    # mid  : np.ndarray (N,) float64
    # ts   : np.ndarray (N,) int64     — microseconds UTC
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger


def _snap_usecols(n: int) -> list[str]:
    cols = ["timestamp"]
    for i in range(n):
        cols += [f"bids[{i}].price", f"bids[{i}].amount"]
    for i in range(n):
        cols += [f"asks[{i}].price", f"asks[{i}].amount"]
    return cols


def _discover_days(
    data_dir: str, symbol: str
) -> list[tuple[str, Path, Path | None, Path | None]]:
    """Return sorted (date, snap_path, trades_path, quotes_path) tuples."""
    base = Path(data_dir)
    pat = re.compile(
        rf"binance_book_snapshot_25_(\d{{4}}-\d{{2}}-\d{{2}})_{re.escape(symbol)}\.csv\.gz$"
    )
    days = []
    for f in sorted(base.iterdir()):
        m = pat.match(f.name)
        if not m:
            continue
        date = m.group(1)
        trades = base / f"binance_trades_{date}_{symbol}.csv.gz"
        quotes = base / f"binance_quotes_{date}_{symbol}.csv.gz"
        days.append(
            (
                date,
                f,
                trades if trades.exists() else None,
                quotes if quotes.exists() else None,
            )
        )
    if not days:
        raise FileNotFoundError(f"No book_snapshot_25 files for {symbol} in {data_dir}")
    return days


def _aggregate_trades(
    snap_ts: np.ndarray, trades_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Map each trade to the snapshot whose timestamp >= trade timestamp, then reduce."""
    ts = trades_df["timestamp"].values.astype(np.int64)
    amt = trades_df["amount"].values.astype(np.float64)
    px = trades_df["price"].values.astype(np.float64)
    # "buy" = aggressor lifted the offer
    buy = trades_df["side"].values == "buy"

    N = len(snap_ts)
    idx = np.clip(np.searchsorted(snap_ts, ts, side="left"), 0, N - 1)

    buy_vol = np.zeros(N)
    sell_vol = np.zeros(N)
    count = np.zeros(N, dtype=np.int64)
    vwap_num = np.zeros(N)
    vwap_den = np.zeros(N)

    np.add.at(buy_vol, idx[buy], amt[buy])
    np.add.at(sell_vol, idx[~buy], amt[~buy])
    np.add.at(count, idx, 1)
    np.add.at(vwap_num, idx, px * amt)
    np.add.at(vwap_den, idx, amt)

    vwap = np.zeros(N)
    mask = vwap_den > 0
    vwap[mask] = vwap_num[mask] / vwap_den[mask]
    return {"buy_vol": buy_vol, "sell_vol": sell_vol, "count": count, "vwap": vwap}


def _aggregate_quotes(
    snap_ts: np.ndarray, quotes_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Map each quote update to its snapshot interval and compute spread/range stats."""
    ts = quotes_df["timestamp"].values.astype(np.int64)
    ask_p = quotes_df["ask_price"].values.astype(np.float64)
    bid_p = quotes_df["bid_price"].values.astype(np.float64)
    spread = ask_p - bid_p
    mid_q = (ask_p + bid_p) / 2.0

    N = len(snap_ts)
    idx = np.clip(np.searchsorted(snap_ts, ts, side="left"), 0, N - 1)

    n_upd = np.zeros(N, dtype=np.int64)
    spread_sum = np.zeros(N)
    mid_max = np.full(N, -np.inf)
    mid_min = np.full(N, np.inf)

    np.add.at(n_upd, idx, 1)
    np.add.at(spread_sum, idx, spread)
    np.maximum.at(mid_max, idx, mid_q)
    np.minimum.at(mid_min, idx, mid_q)

    spread_mean = np.full(N, np.nan)
    mask = n_upd > 0
    spread_mean[mask] = spread_sum[mask] / n_upd[mask]
    mid_range = np.zeros(N)
    mid_range[mask] = mid_max[mask] - mid_min[mask]
    return {"n_updates": n_upd, "spread_mean": spread_mean, "mid_range": mid_range}


def _empty_trades(N: int) -> dict[str, np.ndarray]:
    return {
        "buy_vol": np.zeros(N),
        "sell_vol": np.zeros(N),
        "count": np.zeros(N, dtype=np.int64),
        "vwap": np.zeros(N),
    }


def _empty_quotes(N: int) -> dict[str, np.ndarray]:
    return {
        "n_updates": np.zeros(N, dtype=np.int64),
        "spread_mean": np.full(N, np.nan),
        "mid_range": np.zeros(N),
    }


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
        config:               Requires ``data_dir``, ``cache_dir``, ``symbol``,
                              ``n_lob_levels``, ``feature_mode`` (``"ofi"``/``"lob"``).
        extract_features_fn:  ``(snap_df, trades_agg, quotes_agg, config) → (N, F) float32``
                              OFI features reset at day boundaries (no prev-row arg).
        n_features_fn:        ``(config) → int``
        tag:                  Short model-family string (e.g. ``"deeplob"``).
    """
    paths = _cache_paths(config, tag)
    F = n_features_fn(config)

    if all(p.exists() for p in paths.values()):
        mid = np.load(paths["mid"])
        ts = np.load(paths["ts"])
        N = len(mid)
        feat = np.memmap(paths["feat"], dtype=np.float32, mode="r", shape=(N, F))
        logger.info("loaded cache '{}': {:,} snapshots, {} features", tag, N, F)
        return feat, mid, ts

    symbol = config["symbol"]
    n = config["n_lob_levels"]
    days = _discover_days(config["data_dir"], symbol)
    logger.info("building '{}' cache: {} days for {}", tag, len(days), symbol)

    N_total = 0
    for _, snap_path, _, _ in days:
        with gzip.open(snap_path, "rb") as fh:
            N_total += (
                sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
                - 1
            )

    feat_mm = np.memmap(paths["feat"], dtype=np.float32, mode="w+", shape=(N_total, F))
    mid_arr = np.empty(N_total, dtype=np.float64)
    ts_arr = np.empty(N_total, dtype=np.int64)

    ptr = 0
    snap_cols = _snap_usecols(n)

    for date, snap_path, trades_path, quotes_path in days:
        logger.info("  {} …", date)

        snap_df = pd.read_csv(snap_path, usecols=snap_cols, dtype=np.float64)
        snap_df.dropna(inplace=True)
        if snap_df.empty:
            continue
        N_day = len(snap_df)
        snap_ts = snap_df["timestamp"].values.astype(np.int64)

        trades_agg = (
            _aggregate_trades(
                snap_ts,
                pd.read_csv(
                    trades_path, usecols=["timestamp", "side", "price", "amount"]
                ),
            )
            if trades_path
            else _empty_trades(N_day)
        )
        quotes_agg = (
            _aggregate_quotes(
                snap_ts,
                pd.read_csv(
                    quotes_path, usecols=["timestamp", "ask_price", "bid_price"]
                ),
            )
            if quotes_path
            else _empty_quotes(N_day)
        )

        raw = extract_features_fn(snap_df, trades_agg, quotes_agg, config)  # (N_day, F)

        # Per-day z-score
        day_mean = raw.mean(axis=0)
        day_std = raw.std(axis=0)
        day_std[day_std < 1e-8] = 1.0
        norm = ((raw - day_mean) / day_std).astype(np.float32)

        mid_day = (
            snap_df["bids[0].price"].values + snap_df["asks[0].price"].values
        ) / 2.0

        feat_mm[ptr : ptr + N_day] = norm
        mid_arr[ptr : ptr + N_day] = mid_day
        ts_arr[ptr : ptr + N_day] = snap_ts
        ptr += N_day

    feat_mm.flush()
    del feat_mm  # release before truncating
    # Trim file to actual written rows (dropna may reduce count vs N_total)
    paths["feat"].open("r+b").truncate(ptr * F * 4)

    np.save(paths["mid"], mid_arr[:ptr])
    np.save(paths["ts"], ts_arr[:ptr])
    logger.info("cache '{}' built: {:,} snapshots → {}", tag, ptr, paths["feat"])

    feat = np.memmap(paths["feat"], dtype=np.float32, mode="r", shape=(ptr, F))
    return feat, mid_arr[:ptr], ts_arr[:ptr]
