"""Deletion / insertion faithfulness check for the XAI layer's three models.

Usage::

    uv run python -m xai.run_faithfulness checkpoints/nobitex/BTCIRT \
        --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 \
                 jumpgatelob_levy_BTCIRT_ofi_k10

Runs IG, then masks features in attribution order and in random order over the
same windows, and reports the AUC gap.  Writes ``faithfulness.json``.

This is the check that decides whether the attribution and agreement numbers mean
anything: if masking the top-attributed features costs no more accuracy than
masking random ones, the attributions carry no information about the model and
nothing downstream of them should be reported.
"""

from __future__ import annotations

import argparse
import gc
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
from xai.attribution import attribute_dataset
from xai.faithfulness import faithfulness
from xai.run_ig import _resolve_model


def _table(rows: list[dict]) -> str:
    head = (
        f"{'model':<12} {'del AUC':>8} {'del rand':>9} {'del d':>7} "
        f"{'ins AUC':>8} {'ins rand':>9} {'ins d':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        d, i = r["deletion"], r["insertion"]
        lines.append(
            f"{r['model']:<12} {d['auc']:8.4f} {d['random_auc']:9.4f} "
            f"{r['delta']['deletion']:+7.4f} {i['auc']:8.4f} {i['random_auc']:9.4f} "
            f"{r['delta']['insertion']:+7.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", default="zero", choices=["zero", "mean"])
    ap.add_argument("--n-windows", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=11)
    ap.add_argument("--n-random", type=int, default=5)
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

        # one dataset per distinct data spec: rebuilding per model leaks a
        # memmap over the same cache each time (see xai.run_agreement)
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
            cache[key] = dict(zip(("train", "val", "test"), build_datasets(config)[:3]))
        dataset = cache[key][args.split]

        model = _resolve_model(ckpt_dir, config).to(device)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

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
        logger.info("{} faithfulness:", type(model).__name__)
        r = faithfulness(
            model,
            dataset,
            config,
            device,
            idx,
            ig["per_feature"],
            baseline=args.baseline,
            n_points=args.n_points,
            n_random=args.n_random,
            seed=args.seed,
        )
        r["model"] = type(model).__name__
        rows.append(r)

        # Release each model's weights, checkpoint and IG buffers before the next
        # one. Growth per model is modest (~50 MB on CPU), but large sweeps on a
        # memory-tight box get killed by the OS with no traceback and exit 0, so
        # keep the floor low rather than debug that twice.
        del model, ckpt, ig
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(
        "deletion/insertion vs random ({} baseline)\n"
        "lower deletion AUC and higher insertion AUC are better\n{}",
        args.baseline,
        _table(rows),
    )
    out = args.out or args.root / "faithfulness.json"
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
