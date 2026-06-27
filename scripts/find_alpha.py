"""Find near-optimal alpha for one or more symbols.

Usage:
    uv run python scripts/find_alpha.py --exchange binance --symbols USDCUSDT BTCUSDT
    uv run python scripts/find_alpha.py --exchange nobitex --symbols BTCIRT USDTIRT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from crypto.labels import compute_trend_series


TRAIN_FRAC = 0.70


def find_alpha(mid: np.ndarray, k: int, train_frac: float = TRAIN_FRAC) -> dict:
    trend = compute_trend_series(mid, k)
    train_end = int(len(trend) * train_frac)
    train_trend = trend[:train_end]
    valid = train_trend[np.isfinite(train_trend)]
    abs_valid = np.abs(valid)

    # scan 200 candidate alphas from 0 to 99th pct of |trend_ratio|
    max_alpha = float(np.percentile(abs_valid, 99))
    candidates = np.linspace(0, max_alpha, 200)

    best_alpha, best_imbalance = candidates[0], float("inf")
    rows = []
    for a in candidates:
        n_down = int((valid < -a).sum())
        n_stat = int((np.abs(valid) <= a).sum())
        n_up = int((valid > a).sum())
        total = n_down + n_stat + n_up
        if total == 0:
            continue
        fracs = np.array([n_down, n_stat, n_up]) / total
        imbalance = float(np.max(np.abs(fracs - 1 / 3)))
        rows.append((a, fracs[0], fracs[1], fracs[2], imbalance))
        if imbalance < best_imbalance:
            best_imbalance = imbalance
            best_alpha = a

    # current calibrated alpha (33rd pct of |trend|)
    auto_alpha = float(np.percentile(abs_valid, 100.0 / 3.0))

    return {
        "best_alpha": best_alpha,
        "best_imbalance": best_imbalance,
        "auto_alpha": auto_alpha,
        "n_valid": len(valid),
        "rows": rows,
    }


def load_mid(data_dir: Path, symbol: str) -> np.ndarray:
    for ext in (".parquet.gz", ".parquet"):
        p = data_dir / f"{symbol}{ext}"
        if p.exists():
            df = pd.read_parquet(p, columns=["mid"])
            return df["mid"].to_numpy(dtype=np.float64)
    raise FileNotFoundError(f"No parquet for {symbol} in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print class-balance table for each alpha",
    )
    args = parser.parse_args()

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(f"data/resampled/{args.exchange}")
    )

    for sym in args.symbols:
        print(f"\n{'=' * 60}")
        print(f"Symbol: {sym}  (k={args.k})")
        try:
            mid = load_mid(data_dir, sym)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        res = find_alpha(mid, args.k)
        print(f"  auto alpha (33rd pct):  {res['auto_alpha']:.6f}")
        print(
            f"  best alpha (min imbal): {res['best_alpha']:.6f}  "
            f"(max|frac-1/3|={res['best_imbalance']:.4f})"
        )

        # show balance at a few key alphas
        print(f"\n  {'alpha':>10}  {'down':>7}  {'stat':>7}  {'up':>7}  {'imbal':>7}")
        print(f"  {'-' * 46}")
        # sample 15 evenly spaced rows from the scan
        rows = res["rows"]
        step = max(1, len(rows) // 14)
        shown = rows[::step]
        # always include best
        best_row = min(rows, key=lambda r: r[4])
        if best_row not in shown:
            shown = sorted(shown + [best_row], key=lambda r: r[0])
        for a, fd, fs, fu, imb in shown:
            marker = " ◄ best" if abs(a - res["best_alpha"]) < 1e-10 else ""
            print(
                f"  {a:>10.6f}  {fd:>7.3f}  {fs:>7.3f}  {fu:>7.3f}  {imb:>7.4f}{marker}"
            )


if __name__ == "__main__":
    main()
