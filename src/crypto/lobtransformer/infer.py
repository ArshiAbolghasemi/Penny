"""Inference for LOBTransformer.

Loads the last T_past rows from a Binance book_snapshot_25 file (plus optional
trades and quotes) and returns a direction prediction.

Usage::

    uv run python -m crypto.lobtransformer.infer --checkpoint <dir> \\
        --snapshot data/binance/binance_book_snapshot_25_2024-01-15_BTCUSDT.csv.gz \\
        [--trades  data/binance/binance_trades_2024-01-15_BTCUSDT.csv.gz] \\
        [--quotes  data/binance/binance_quotes_2024-01-15_BTCUSDT.csv.gz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger

from crypto.utils.features import extract_features
from crypto.utils.loader import (
    _aggregate_quotes,
    _aggregate_trades,
    _empty_quotes,
    _empty_trades,
    _snap_usecols,
)
from crypto.utils.training import resolve_device

from .model import LOBTransformer

CLASS_NAMES = {0: "down", 1: "stationary", 2: "up"}


@torch.no_grad()
def predict(checkpoint_dir, snapshot_path, trades_path=None, quotes_path=None, device="cpu"):
    """Return direction prediction from Binance snapshot file(s).

    Args:
        checkpoint_dir: Path to directory containing ``best.pt``.
        snapshot_path:  Path to ``binance_book_snapshot_25_*.csv.gz``.
        trades_path:    Optional ``binance_trades_*.csv.gz``.
        quotes_path:    Optional ``binance_quotes_*.csv.gz``.
        device:         Device string.

    Returns:
        dict with ``label`` (int), ``label_name`` (str),
        ``probs`` (dict with down/stationary/up floats).
    """
    dev = resolve_device(device)
    ckpt = torch.load(Path(checkpoint_dir) / "best.pt", map_location=dev, weights_only=False)
    config = ckpt["config"]
    t_past = config["T_past"]
    n = config["n_lob_levels"]

    snap_df = pd.read_csv(snapshot_path, usecols=_snap_usecols(n), dtype=np.float64)
    snap_df.dropna(inplace=True)
    snap_df = snap_df.iloc[-t_past:].reset_index(drop=True)
    if len(snap_df) < t_past:
        raise ValueError(f"need at least {t_past} rows, got {len(snap_df)}")

    snap_ts = snap_df["timestamp"].values.astype(np.int64)

    trades_agg = (
        _aggregate_trades(
            snap_ts,
            pd.read_csv(trades_path, usecols=["timestamp", "side", "price", "amount"]),
        )
        if trades_path and Path(trades_path).exists()
        else _empty_trades(t_past)
    )
    quotes_agg = (
        _aggregate_quotes(
            snap_ts,
            pd.read_csv(quotes_path, usecols=["timestamp", "ask_price", "bid_price"]),
        )
        if quotes_path and Path(quotes_path).exists()
        else _empty_quotes(t_past)
    )

    raw = extract_features(snap_df, trades_agg, quotes_agg, config)  # (T_past, F)

    # Normalize by this window's own stats (approximates per-day training normalization)
    win_std = raw.std(axis=0)
    win_std[win_std < 1e-8] = 1.0
    norm = ((raw - raw.mean(axis=0)) / win_std).astype(np.float32)

    x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(dev)  # (1, 1, T, F)

    model = LOBTransformer(config).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs = F.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
    label = int(probs.argmax())
    return {
        "label": label,
        "label_name": CLASS_NAMES[label],
        "probs": {"down": float(probs[0]), "stationary": float(probs[1]), "up": float(probs[2])},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="LOBTransformer inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--trades", default=None)
    p.add_argument("--quotes", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out = predict(args.checkpoint, args.snapshot, args.trades, args.quotes, args.device)
    print("LOBTransformer signal")
    print(f"  label : {out['label']} ({out['label_name']})")
    print(
        f"  probs : down={out['probs']['down']:.3f}"
        f"  stat={out['probs']['stationary']:.3f}"
        f"  up={out['probs']['up']:.3f}"
    )


if __name__ == "__main__":
    main()
