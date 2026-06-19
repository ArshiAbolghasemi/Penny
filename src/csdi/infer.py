"""Inference for the CSDI direction classifier.

Usage::

    uv run python -m csdi.infer --checkpoint <dir> \
        --orderbook data/nobitex_data/BTCIRT_orderbook.csv [--trades ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from . import features as feat
from .model import CSDIClassifier

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
    model = CSDIClassifier(config).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    t_past = config["T_past"]

    window = past_orderbook.iloc[-t_past:].reset_index(drop=True)
    rows = feat.build_global_rows(window, trades, config)
    rows_norm = normalizer.transform(rows)
    past = torch.from_numpy(np.transpose(rows_norm, (2, 1, 0)).copy()).unsqueeze(0)

    logits = model.predict({"past": past}, dev)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label = int(probs.argmax())
    return {
        "label": label,
        "label_name": CLASS_NAMES[label],
        "probs": {"down": float(probs[0]), "stationary": float(probs[1]), "up": float(probs[2])},
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
    print(f"  label  : {out['label']} ({out['label_name']})")
    print(f"  probs  : down={out['probs']['down']:.3f}  stat={out['probs']['stationary']:.3f}  up={out['probs']['up']:.3f}")


if __name__ == "__main__":
    main()
