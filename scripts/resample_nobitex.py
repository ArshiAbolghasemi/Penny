"""
Normalize Nobitex LOB data to the same parquet schema as resample_binance.py.

Input directory (default: data/nobitex_data/):
  {SYMBOL}_orderbook.csv  — time, bid_price_1..N, bid_volume_1..N, ask_price_1..N, ask_volume_1..N
  {SYMBOL}_trades.csv     — snapshot_time, trade_time, price, volume, direction

Output directory (default: data/resampled/nobitex/):
  {SYMBOL}.parquet.gz     — same column schema as Binance resampled parquet

Column mapping
--------------
  bid_price_{i}  →  bids[i-1].price
  bid_volume_{i} →  bids[i-1].amount
  ask_price_{i}  →  asks[i-1].price
  ask_volume_{i} →  asks[i-1].amount

Usage
-----
    uv run python scripts/resample_nobitex.py
    uv run python scripts/resample_nobitex.py --levels 20
    uv run python scripts/resample_nobitex.py --data-dir data/nobitex_data --out-dir data/resampled/nobitex
    uv run python scripts/resample_nobitex.py --symbols BTCIRT USDTIRT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
_INTERVAL_S = 10


def _bin_col(ts_us: pd.Series, interval_s: int) -> pd.Series:
    interval_us = interval_s * 1_000_000
    return (ts_us // interval_us) * interval_us


def _lob_rename(n: int) -> dict[str, str]:
    m: dict[str, str] = {}
    for i in range(1, n + 1):
        m[f"bid_price_{i}"] = f"bids[{i - 1}].price"
        m[f"bid_volume_{i}"] = f"bids[{i - 1}].amount"
        m[f"ask_price_{i}"] = f"asks[{i - 1}].price"
        m[f"ask_volume_{i}"] = f"asks[{i - 1}].amount"
    return m


def _discover(data_dir: Path) -> list[str]:
    """Return symbol names that have an orderbook file."""
    return [
        f.name.replace("_orderbook.csv", "")
        for f in sorted(data_dir.iterdir())
        if f.name.endswith("_orderbook.csv")
    ]


def _process_symbol(
    data_dir: Path,
    symbol: str,
    n_levels: int,
    interval_s: int,
) -> pd.DataFrame:
    ob_path = data_dir / f"{symbol}_orderbook.csv"
    tr_path = data_dir / f"{symbol}_trades.csv"

    # ── orderbook ─────────────────────────────────────────────────────────────
    # Data is already at 10s intervals — deduplicate on time string, keep last.
    rename = _lob_rename(n_levels)
    lob_cols = list(rename.keys())
    ob = pd.read_csv(ob_path, usecols=["time"] + lob_cols)
    ob.rename(columns=rename, inplace=True)
    ob = ob.drop_duplicates(subset="time", keep="last").reset_index(drop=True)

    ob["_t"] = pd.to_datetime(ob["time"])
    ob["bin"] = _bin_col(ob["_t"].astype(np.int64) // 1000, interval_s)
    ob["mid"] = (ob["bids[0].price"] + ob["asks[0].price"]) / 2.0
    ob["spread"] = ob["asks[0].price"] - ob["bids[0].price"]
    ob.insert(
        0, "timestamp_utc", pd.to_datetime(ob["bin"] // 1000, unit="ms", utc=True)
    )
    ob.drop(columns=["_t"], inplace=True)

    # ── trades ────────────────────────────────────────────────────────────────
    # snapshot_time in trades is the same timestamp as time in orderbook.
    # Aggregate directly on that string — no bin recomputation needed.
    if tr_path.exists():
        tr = pd.read_csv(
            tr_path, usecols=["snapshot_time", "price", "volume", "direction"]
        )
        tr["buy_vol"] = np.where(tr["direction"] == "buy", tr["volume"], 0.0)
        tr["sell_vol"] = np.where(tr["direction"] == "sell", tr["volume"], 0.0)
        tr["vwap_num"] = tr["price"] * tr["volume"]

        agg = (
            tr.groupby("snapshot_time")
            .agg(
                trade_count=("volume", "count"),
                buy_vol=("buy_vol", "sum"),
                sell_vol=("sell_vol", "sum"),
                vwap_num=("vwap_num", "sum"),
                total_vol=("volume", "sum"),
            )
            .reset_index()
            .rename(columns={"snapshot_time": "time"})
        )
        agg["vwap"] = np.where(
            agg["total_vol"] > 0, agg["vwap_num"] / agg["total_vol"], np.nan
        )
        denom = agg["buy_vol"] + agg["sell_vol"]
        agg["trade_imbalance"] = np.where(
            denom > 0, (agg["buy_vol"] - agg["sell_vol"]) / denom, np.nan
        )
        agg.drop(columns=["vwap_num", "total_vol"], inplace=True)

        result = ob.merge(agg, on="time", how="left")
    else:
        logger.warning("no trades file for {} — trade columns will be zero", symbol)
        result = ob.copy()

    # Bins with no trades → zero activity (not unknown)
    result["trade_count"] = result["trade_count"].fillna(0)
    result["buy_vol"] = result["buy_vol"].fillna(0)
    result["sell_vol"] = result["sell_vol"].fillna(0)
    result["trade_imbalance"] = result["trade_imbalance"].fillna(0)
    # vwap left NaN when no trades — features.py falls back to mid in that case

    result.drop(columns=["time"], inplace=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Nobitex LOB data to resampled parquet."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to process (default: all found in data-dir)",
    )
    parser.add_argument(
        "--levels", type=int, default=20, help="LOB depth levels to keep (default: 20)"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Raw CSV directory (default: data/nobitex_data)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/resampled/nobitex)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _ROOT / "data" / "nobitex_data"
    out_dir = (
        Path(args.out_dir) if args.out_dir else _ROOT / "data" / "resampled" / "nobitex"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = args.symbols or _discover(data_dir)
    if not symbols:
        logger.error("No *_orderbook.csv files found in {}", data_dir)
        sys.exit(1)

    logger.info("Source  : {}", data_dir)
    logger.info("Symbols : {}", symbols)
    logger.info("Levels  : {}  |  Interval: {}s", args.levels, _INTERVAL_S)
    logger.info("Output  : {}/", out_dir)

    for sym in symbols:
        out_path = out_dir / f"{sym}.parquet.gz"
        logger.info("Processing {} …", sym)
        try:
            df = _process_symbol(data_dir, sym, args.levels, _INTERVAL_S)
        except Exception as e:
            logger.error("  {} failed: {}", sym, e)
            continue

        dates = df["timestamp_utc"].dt.date.unique()
        logger.info(
            "  {} rows × {} cols  |  {} days ({} → {})",
            len(df),
            len(df.columns),
            len(dates),
            dates.min(),
            dates.max(),
        )

        df.to_parquet(out_path, index=False, compression="gzip")
        size_mb = out_path.stat().st_size / 1e6
        logger.info("  → {}  {:.1f} MB", out_path.name, size_mb)

    files = sorted(out_dir.glob("*.parquet.gz"))
    sep = "─" * 50
    logger.info(sep)
    logger.info("{:<22}  {:>8}  {:>6}", "File", "Rows", "MB")
    logger.info(sep)
    for f in files:
        df = pd.read_parquet(f, columns=["bin"])
        logger.info("{:<22}  {:>8,}  {:>6.1f}", f.name, len(df), f.stat().st_size / 1e6)
    logger.info(sep)


if __name__ == "__main__":
    main()
