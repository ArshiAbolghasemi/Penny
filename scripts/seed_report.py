"""Collect seed-sweep summaries into one mean ± std table.

Every multi-seed training run writes ``<run>_seed_summary.json`` next to its run
directories (see ``utils.training.summarize_seed_runs``).  This script gathers all
of them under a checkpoint root and prints one row per (model, symbol, horizon),
so a paper table can be filled from a single command.

Usage::

    uv run python scripts/seed_report.py                        # checkpoints/
    uv run python scripts/seed_report.py checkpoints/coinbase --csv outputs/seeds.csv

The reported spread is **training (seed) variance on a fixed test set**: it says
how much the metric moves when only the seed changes, and is not a confidence
interval for the metric nor a test of one model against another — for that use
``scripts/stochlob_significance.py``, which handles the overlapping-window
dependence between models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _row(summary_path: Path) -> dict | None:
    """One tidy row from a ``*_seed_summary.json``, or None if it is unreadable."""
    try:
        s = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    runs = s.get("runs") or []
    if not runs:
        return None

    # Identify the run from the first seed's saved config rather than by parsing
    # the directory name — the model tag is whatever precedes `_{symbol}`.
    cfg = {}
    cfg_path = Path(runs[0]["run_dir"]) / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
    symbol = cfg.get("symbol", "")
    name = Path(runs[0]["run_dir"]).name
    model = name.split(f"_{symbol}")[0] if symbol and f"_{symbol}" in name else name

    row = {
        "model": model,
        "symbol": symbol,
        "feature_mode": cfg.get("feature_mode", ""),
        "k": cfg.get("label_k", ""),
        "n_seeds": s.get("n_seeds", len(runs)),
        "seeds": ",".join(str(x) for x in s.get("seeds", [])),
    }
    for metric, short in (("accuracy", "acc"), ("macro_f1", "f1")):
        m = s.get("metrics", {}).get(metric, {})
        row[f"{short}_mean"] = m.get("mean")
        row[f"{short}_std"] = m.get("std")
    row["summary"] = str(summary_path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?", default="checkpoints", help="checkpoint root to scan"
    )
    parser.add_argument("--csv", default=None, help="also write the table to this path")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [r for p in sorted(root.rglob("*_seed_summary.json")) if (r := _row(p))]
    if not rows:
        print(f"no *_seed_summary.json found under {root}")
        return

    df = pd.DataFrame(rows).sort_values(["symbol", "k", "model"], kind="stable")
    show = df.drop(columns=["summary"]).copy()
    for short in ("acc", "f1"):
        show[short] = [
            "n/a"
            if pd.isna(m)
            else (f"{m:.4f}" if pd.isna(s) else f"{m:.4f} ± {s:.4f}")
            for m, s in zip(show.pop(f"{short}_mean"), show.pop(f"{short}_std"))
        ]
    print(show.to_string(index=False))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
