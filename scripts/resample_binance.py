"""
Resample Binance LOB data to fixed intervals, one .parquet.gz file per symbol.

For each bin:
  - book_snapshot_25 : last LOB state in the bin  (last-tick resample)
  - trades           : aggregated buy/sell vol, count, VWAP, imbalance
  - quotes           : last best-bid/ask in the bin

Output directory: data/resampled/
  BTCUSDT.parquet.gz
  ETHUSDT.parquet.gz
  ...  (one file per symbol, all dates concatenated)

Usage
-----
    uv run python scripts/resample_binance.py --interval 0.25
    uv run python scripts/resample_binance.py --interval 10 --levels 10
    uv run python scripts/resample_binance.py --date 2026-06-09
    uv run python scripts/resample_binance.py --data-dir data/binance.bak
    uv run python scripts/resample_binance.py --out-dir data/resampled
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent


# ── helpers ────────────────────────────────────────────────────────────────────


def _bin_col(ts_us: pd.Series, interval_s: float) -> pd.Series:
    interval_us = int(interval_s * 1_000_000)
    return (ts_us // interval_us) * interval_us


def _discover(data_dir: Path, date: str | None = None) -> dict[str, list[str]]:
    """Return {symbol: [dates]} for all available snapshot files."""
    pat = re.compile(r"binance_book_snapshot_25_(\d{4}-\d{2}-\d{2})_(\w+)\.csv\.gz")
    result: dict[str, list[str]] = {}
    for f in sorted(data_dir.iterdir()):
        m = pat.match(f.name)
        if not m:
            continue
        d, sym = m.group(1), m.group(2)
        if date and d != date:
            continue
        result.setdefault(sym, []).append(d)
    return result


def _lob_cols(n: int) -> list[str]:
    cols = []
    for i in range(n):
        cols += [
            f"bids[{i}].price",
            f"bids[{i}].amount",
            f"asks[{i}].price",
            f"asks[{i}].amount",
        ]
    return cols


# ── per-file resamplers ────────────────────────────────────────────────────────


def _resample_snapshot(path: Path, interval_s: float, n_levels: int) -> pd.DataFrame:
    header_cols = set(pd.read_csv(path, nrows=0).columns)
    usecols = ["timestamp"] + [c for c in _lob_cols(n_levels) if c in header_cols]
    df = pd.read_csv(path, usecols=usecols, dtype=np.float64)
    df.dropna(inplace=True)
    if df.empty:
        return pd.DataFrame()

    df["bin"] = _bin_col(df["timestamp"].astype(np.int64), interval_s)
    result = df.drop(columns=["timestamp"]).groupby("bin").last().reset_index()

    if "bids[0].price" in result.columns and "asks[0].price" in result.columns:
        result["mid"] = (result["bids[0].price"] + result["asks[0].price"]) / 2.0
        result["spread"] = result["asks[0].price"] - result["bids[0].price"]
    return result


def _resample_trades(path: Path, interval_s: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "bin",
                "trade_count",
                "buy_vol",
                "sell_vol",
                "vwap",
                "trade_imbalance",
            ]
        )
    df = pd.read_csv(path, usecols=["timestamp", "side", "price", "amount"])
    df.dropna(inplace=True)
    if df.empty:
        return pd.DataFrame()

    df["bin"] = _bin_col(df["timestamp"].astype(np.int64), interval_s)
    df["buy_vol"] = np.where(df["side"] == "buy", df["amount"], 0.0)
    df["sell_vol"] = np.where(df["side"] == "sell", df["amount"], 0.0)
    df["vwap_num"] = df["price"] * df["amount"]

    agg = (
        df.groupby("bin")
        .agg(
            trade_count=("amount", "count"),
            buy_vol=("buy_vol", "sum"),
            sell_vol=("sell_vol", "sum"),
            vwap_num=("vwap_num", "sum"),
            total_vol=("amount", "sum"),
        )
        .reset_index()
    )
    agg["vwap"] = np.where(
        agg["total_vol"] > 0, agg["vwap_num"] / agg["total_vol"], np.nan
    )
    denom = agg["buy_vol"] + agg["sell_vol"]
    agg["trade_imbalance"] = np.where(
        denom > 0, (agg["buy_vol"] - agg["sell_vol"]) / denom, np.nan
    )
    return agg.drop(columns=["vwap_num", "total_vol"])


def _resample_quotes(path: Path, interval_s: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "bin",
                "quote_bid_price",
                "quote_ask_price",
                "quote_bid_amount",
                "quote_ask_amount",
            ]
        )
    df = pd.read_csv(
        path,
        usecols=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"],
        dtype=np.float64,
    )
    df.dropna(inplace=True)
    if df.empty:
        return pd.DataFrame()

    df["bin"] = _bin_col(df["timestamp"].astype(np.int64), interval_s)
    return (
        df.drop(columns=["timestamp"])
        .groupby("bin")
        .last()
        .reset_index()
        .rename(
            columns={
                "bid_price": "quote_bid_price",
                "ask_price": "quote_ask_price",
                "bid_amount": "quote_bid_amount",
                "ask_amount": "quote_ask_amount",
            }
        )
    )


# ── one day × one symbol ───────────────────────────────────────────────────────


def _process_day(
    data_dir: Path, date: str, symbol: str, interval_s: float, n_levels: int
) -> pd.DataFrame:
    snap = _resample_snapshot(
        data_dir / f"binance_book_snapshot_25_{date}_{symbol}.csv.gz",
        interval_s,
        n_levels,
    )
    if snap.empty:
        return pd.DataFrame()

    trades = _resample_trades(
        data_dir / f"binance_trades_{date}_{symbol}.csv.gz", interval_s
    )
    quotes = _resample_quotes(
        data_dir / f"binance_quotes_{date}_{symbol}.csv.gz", interval_s
    )

    df = snap.merge(trades, on="bin", how="left").merge(quotes, on="bin", how="left")
    df.insert(
        0, "timestamp_utc", pd.to_datetime(df["bin"] // 1000, unit="ms", utc=True)
    )
    return df


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample Binance data to fixed intervals, one file per symbol."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Bin size in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--levels", type=int, default=10, help="LOB depth levels to keep (default: 10)"
    )
    parser.add_argument(
        "--date", default=None, help="Process only this date YYYY-MM-DD (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Raw CSV source directory (default: data/binance)",
    )
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (default: data/resampled)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _ROOT / "data" / "binance"
    out_dir = (
        Path(args.out_dir) if args.out_dir else _ROOT / "data" / "resampled" / "binance"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = _discover(data_dir, args.date)
    if not schedule:
        logger.error(
            "No snapshot CSV files found in {}{}",
            data_dir,
            f" for {args.date}" if args.date else "",
        )
        sys.exit(1)

    n_sym = len(schedule)
    n_total = sum(len(dates) for dates in schedule.values())
    logger.info("Source  : {}", data_dir)
    logger.info("Symbols : {}  |  day×symbol pairs: {}", n_sym, n_total)
    logger.info("Interval: {}s  |  LOB levels: {}", args.interval, args.levels)
    logger.info("Output  : {}/", out_dir)

    for sym_idx, (symbol, dates) in enumerate(sorted(schedule.items()), 1):
        out_path = out_dir / f"{symbol}.parquet.gz"
        logger.info("[{}/{}] {}  ({} days)", sym_idx, n_sym, symbol, len(dates))

        day_frames: list[pd.DataFrame] = []
        for date in sorted(dates):
            try:
                df = _process_day(data_dir, date, symbol, args.interval, args.levels)
                if df.empty:
                    logger.warning("  {}  empty — skipped", date)
                    continue
                day_frames.append(df)
                logger.debug("  {}  {:>5} bins", date, len(df))
            except Exception as e:
                logger.error("  {}  {}", date, e)

        if not day_frames:
            logger.warning("  no data for {} — skipping", symbol)
            continue

        combined = pd.concat(day_frames, ignore_index=True)
        combined.sort_values("bin", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        combined.to_parquet(out_path, index=False, compression="gzip")
        size_mb = out_path.stat().st_size / 1e6
        logger.info(
            "  → {}  {:,} rows × {} cols  {:.1f} MB",
            out_path.name,
            len(combined),
            len(combined.columns),
            size_mb,
        )

    # summary
    files = sorted(out_dir.glob("*.parquet.gz"))
    sep = "─" * 50
    logger.info(sep)
    logger.info("{:<20}  {:>10}  {:>6}", "File", "Rows", "MB")
    logger.info(sep)
    for f in files:
        df = pd.read_parquet(f, columns=["bin"])
        logger.info(
            "{:<20}  {:>10,}  {:>6.1f}", f.name, len(df), f.stat().st_size / 1e6
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
