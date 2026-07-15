"""Attention-vs-IG agreement table across the XAI layer's three models.

Usage::

    uv run python -m xai.run_agreement checkpoints/nobitex/BTCIRT \
        --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 \
                 jumpgatelob_levy_BTCIRT_ofi_k10

Runs IG and the attention probes over the **same** window subsample for each
model, rank-correlates them, and writes ``agreement.json`` next to the
checkpoints.  The time axis is comparable across all three; the feature axis
exists only for DLA (see :mod:`xai.agreement`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from loguru import logger

from crypto.dataset import build_datasets
from utils.training import resolve_device, set_seed
from xai.agreement import agreement, collect_attention, format_table
from xai.attribution import attribute_dataset
from xai.run_ig import _resolve_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory holding the checkpoint dirs")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", default="zero", choices=["zero", "mean"])
    ap.add_argument("--n-windows", type=int, default=2048)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    cache: dict[tuple, object] = {}
    for name in args.models:
        ckpt_dir = args.root / name
        config = json.loads((ckpt_dir / "config.json").read_text())
        set_seed(config.get("seed", 42))
        device = resolve_device(config.get("device", "cuda"))

        # Rebuilding per model would open a fresh memmap over the same 184k-row
        # cache for each one and never drop the old ones; the process is killed
        # part-way through the second model. The windows must be identical across
        # models anyway for the comparison to mean anything, so build once per
        # distinct data spec and reuse.
        key = (
            config["symbol"],
            config["cache_dir"],
            config["n_lob_levels"],
            config.get("feature_mode", "ofi"),
            config["T_past"],
            config["label_k"],
            config["stride"],
            config["train_frac"],
            config["val_frac"],
        )
        if key not in cache:
            cache[key] = dict(
                zip(("train", "val", "test"), build_datasets(config)[:3])
            )
        dataset = cache[key][args.split]

        model = _resolve_model(ckpt_dir, config).to(device)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

        # IG and the probes must see the *same* windows or the correlation is
        # comparing two different samples; both draw this index set.
        idx = np.arange(len(dataset))
        if args.n_windows < len(idx):
            idx = np.random.default_rng(args.seed).choice(
                idx, args.n_windows, replace=False
            )
            idx.sort()

        ig = attribute_dataset(
            model,
            dataset,
            config,
            device,
            baseline=args.baseline,
            n_windows=args.n_windows,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        attn = collect_attention(model, dataset, config, device, idx)
        rows.append(agreement(ig, attn, type(model).__name__))

    table = format_table(rows)
    logger.info("attention vs IG rank agreement ({} baseline):\n{}", args.baseline, table)

    out = args.out or args.root / "agreement.json"
    out.write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "split": args.split,
                "n_windows": args.n_windows,
                "n_steps": args.n_steps,
                "seed": args.seed,
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
