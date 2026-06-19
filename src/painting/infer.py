"""Inference for the painting approach (spec section 8).

Usage::

    uv run python -m painting.infer --checkpoint <dir> \
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
from .diffusion import Diffusion
from .model import build_model

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
def predict(checkpoint_dir, past_orderbook, trades=None, n_samples=None, device="cpu"):
    dev = _resolve_device(device)
    ckpt = torch.load(
        Path(checkpoint_dir) / "best.pt", map_location=dev, weights_only=False
    )
    config = ckpt["config"]
    normalizer = feat.RollingNormalizer.from_dict(config, ckpt["normalizer"])
    level_starts = np.asarray(ckpt["level_starts"])
    model = build_model(config, normalizer, level_starts).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    gamma, alpha = float(ckpt["gamma"]), float(ckpt["alpha"])
    n = config["n_levels"]
    t_past, t_total, k = config["T_past"], config["T_total"], config["label_k"]
    ns = n_samples or config["n_samples"]

    window = past_orderbook.iloc[-t_past:].reset_index(drop=True)
    rows = feat.build_global_rows(window, trades, config)
    rows_norm = normalizer.transform(rows)
    img = np.zeros((feat.n_rows(config), t_total, 2), dtype=np.float32)
    img[:, :t_past, :] = np.transpose(rows_norm, (1, 0, 2))
    if config["feature_mode"] == "ofi":
        img[: 2 * n, 0, 0] = 0.0
    padded, _ = feat.pad_levels(img, config["padded_size"])
    image = torch.from_numpy(padded).permute(2, 0, 1)
    mask = torch.from_numpy(feat.build_mask(config)).permute(2, 0, 1)

    mid = feat.mid_series(window)
    mid_ref = (
        float(mid[0]) if config["feature_mode"] == "lob" else float(mid[t_past - 1])
    )
    bwd = float(np.mean(mid[t_past - k : t_past]))

    diffusion = Diffusion(config, dev)
    x0_known = image.unsqueeze(0).repeat(ns, 1, 1, 1).to(dev)
    m = mask.unsqueeze(0).repeat(ns, 1, 1, 1).to(dev)
    painted = diffusion.sample(model, x0_known, m, config["ddim_steps"], dev)
    fut_mid = model.future_mid(painted, {"mid_ref": torch.full((ns,), mid_ref)}, gamma)
    fwd = fut_mid[:, :k].mean(dim=1).cpu().numpy()
    l_vals = (fwd - bwd) / (bwd + 1e-12)
    votes = np.bincount(
        [lab.label_from_l(float(x), alpha) for x in l_vals], minlength=3
    )
    modal = int(votes.argmax())
    mean_l, std_l = float(l_vals.mean()), float(l_vals.std())
    return {
        "label": modal,
        "label_name": CLASS_NAMES[modal],
        "votes": {
            "down": int(votes[0]),
            "stationary": int(votes[1]),
            "up": int(votes[2]),
        },
        "mean_l": mean_l,
        "std_l": std_l,
        "signal_ratio": mean_l / (std_l + 1e-12),
        "future_mid_mean": fut_mid.mean(dim=0).cpu().numpy(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Penny painting inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--orderbook", required=True)
    p.add_argument("--trades", default=None)
    p.add_argument("--index", type=int, default=-1)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    ckpt = torch.load(
        Path(args.checkpoint) / "best.pt", map_location="cpu", weights_only=False
    )
    ob = feat.load_orderbook(args.orderbook, ckpt["config"]["n_levels"])
    end = args.index if args.index >= 0 else len(ob)
    ob = ob.iloc[:end]
    trades = feat.load_trades(args.trades) if args.trades else None
    out = predict(args.checkpoint, ob, trades, args.n_samples, args.device)
    print("Penny painting signal")
    print(f"  label        : {out['label']} ({out['label_name']})")
    print(f"  votes        : {out['votes']}")
    print(f"  mean l       : {out['mean_l']:.6f}")
    print(f"  signal ratio : {out['signal_ratio']:.4f}")


if __name__ == "__main__":
    main()
