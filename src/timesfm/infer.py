"""Inference for the TimesFM direction classifier.

Usage::

    uv run python -m timesfm.infer --checkpoint <dir> \
        --orderbook data/nobitex_data/BTCIRT_orderbook.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from . import features as feat
from .model import TimesFMClassifier

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
def predict(checkpoint_dir, orderbook_df, device="cpu"):
    """Return direction prediction from a past orderbook window.

    Args:
        checkpoint_dir: Path to directory containing ``best.pt``.
        orderbook_df: DataFrame with LOB columns (already loaded).
        device: Device string.

    Returns:
        dict with keys ``label`` (int), ``label_name`` (str),
        ``probs`` (dict with down/stationary/up floats).
    """
    dev = _resolve_device(device)
    ckpt = torch.load(
        Path(checkpoint_dir) / "best.pt", map_location=dev, weights_only=False
    )
    config = ckpt["config"]
    ofi_stats = ckpt.get("ofi_stats")
    use_ofi = config.get("feature_mode", "ofi") == "ofi"

    model = TimesFMClassifier(config).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    t_past = config["T_past"]
    window = orderbook_df.iloc[-t_past:].reset_index(drop=True)

    mid = feat.mid_series(window).astype(np.float32)
    past_mid = torch.from_numpy(mid).unsqueeze(0)  # (1, T_past)

    batch = {"past_mid": past_mid}
    if use_ofi:
        if ofi_stats is None:
            raise RuntimeError("OFI mode requires ofi_stats in checkpoint")
        raw_ofi = feat.net_ofi_series(window).astype(np.float32)
        ofi_norm = (raw_ofi - ofi_stats["mean"]) / max(ofi_stats["std"], 1e-8)
        batch["past_ofi"] = torch.from_numpy(ofi_norm).unsqueeze(0)  # (1, T_past)

    logits = model.predict(batch, dev)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label = int(probs.argmax())
    return {
        "label": label,
        "label_name": CLASS_NAMES[label],
        "probs": {
            "down": float(probs[0]),
            "stationary": float(probs[1]),
            "up": float(probs[2]),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Penny TimesFM inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--orderbook", required=True)
    p.add_argument("--index", type=int, default=-1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = torch.load(
        Path(args.checkpoint) / "best.pt", map_location="cpu", weights_only=False
    )
    ob = feat.load_orderbook(args.orderbook, ckpt["config"]["n_levels"])
    end = args.index if args.index >= 0 else len(ob)
    ob = ob.iloc[:end]

    out = predict(args.checkpoint, ob, args.device)
    print("Penny TimesFM signal")
    print(f"  label  : {out['label']} ({out['label_name']})")
    print(
        f"  probs  : down={out['probs']['down']:.3f}"
        f"  stat={out['probs']['stationary']:.3f}"
        f"  up={out['probs']['up']:.3f}"
    )


if __name__ == "__main__":
    main()
