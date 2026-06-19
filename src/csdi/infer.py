"""Inference for the CSDI forecaster (spec section 8).

Usage::

    uv run python -m csdi.infer --checkpoint <dir> \
        --orderbook data/nobitex_data/BTCIRT_orderbook.csv [--trades ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from . import features as feat
from . import labels as lab
from .model import CSDIForecastModel

CLASS_NAMES = {0: "down", 1: "stationary", 2: "up"}


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("cuda", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested not in ("cuda", "mps"):
        return torch.device(requested)
    logger.warning("{} unavailable; using cpu", requested)
    return torch.device("cpu")


@torch.no_grad()
def predict(checkpoint_dir, past_orderbook, trades=None, device="cpu"):
    dev = _resolve_device(device)
    ckpt = torch.load(
        Path(checkpoint_dir) / "best.pt", map_location=dev, weights_only=False
    )
    config = ckpt["config"]
    normalizer = feat.RollingNormalizer.from_dict(config, ckpt["normalizer"])
    model = CSDIForecastModel(config).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    alpha = float(ckpt["alpha"])
    t_past, k = config["T_past"], config["label_k"]

    window = past_orderbook.iloc[-t_past:].reset_index(drop=True)
    rows = feat.build_global_rows(window, trades, config)
    rows_norm = normalizer.transform(rows)  # (T_past, R, 2)
    past = torch.from_numpy(np.transpose(rows_norm, (2, 1, 0)).copy()).unsqueeze(0)

    mid = feat.mid_series(window)
    boundary = float(mid[t_past - 1])
    bwd = float(np.mean(mid[t_past - k : t_past]))
    pred_ret = model.forecast({"past": past}, dev)
    fut_mid = boundary * (1.0 + pred_ret)
    l_val = float((fut_mid[:, :k].mean() - bwd) / (bwd + 1e-12))
    label = lab.label_from_l(l_val, alpha)
    return {
        "label": label,
        "label_name": CLASS_NAMES[label],
        "mean_l": l_val,
        "future_mid_mean": fut_mid[0].cpu().numpy(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Penny CSDI inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--orderbook", required=True)
    p.add_argument("--trades", default=None)
    p.add_argument("--index", type=int, default=-1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    ckpt = torch.load(
        Path(args.checkpoint) / "best.pt", map_location="cpu", weights_only=False
    )
    ob = feat.load_orderbook(args.orderbook, ckpt["config"]["n_levels"])
    end = args.index if args.index >= 0 else len(ob)
    ob = ob.iloc[:end]
    trades = feat.load_trades(args.trades) if args.trades else None
    out = predict(args.checkpoint, ob, trades, args.device)
    print("Penny CSDI signal")
    print(f"  label   : {out['label']} ({out['label_name']})")
    print(f"  mean l  : {out['mean_l']:.6f}")


if __name__ == "__main__":
    main()
