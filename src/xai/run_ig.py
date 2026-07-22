"""Run Integrated Gradients on a trained checkpoint.

Usage::

    uv run python -m xai.run_ig checkpoints/nobitex/BTCIRT/ctabl_BTCIRT_ofi_k10
    uv run python -m xai.run_ig <ckpt_dir> --baseline mean --n-windows 1024

Reads the run's own ``config.json`` so the dataset, feature mode and window
geometry always match what the checkpoint was trained on.  Writes ``ig_<baseline>.npz``
(arrays) and ``ig_<baseline>.json`` (group shares + provenance) into the
checkpoint directory.
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
from xai.attribution import attribute_dataset
from xai.features import feature_groups, feature_names, group_attribution

# checkpoint-dir prefix → (module, class).  The three models in the XAI layer's
# scope; JointDiT is deliberately out of scope.
_MODELS = {
    "ctabl_": ("models.ctabl", "CTABL"),
    "dla_": ("models.dla", "DLA"),
    "jumpgatelob_": ("models.jumpgatelob", "JumpGateLOB"),
}


def _resolve_model(ckpt_dir: Path, config: dict) -> torch.nn.Module:
    name = ckpt_dir.name
    for prefix, (mod, cls) in _MODELS.items():
        if name.startswith(prefix):
            import importlib

            model = getattr(importlib.import_module(mod), cls)(config)
            logger.info("model {} from {}", cls, ckpt_dir.name)
            return model
    raise ValueError(
        f"cannot infer model from {name!r}; expected one of {sorted(_MODELS)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", type=Path)
    ap.add_argument(
        "--baseline",
        default="zero",
        choices=["zero", "mean"],
        help="zero = no order flow (default); mean = training-window mean",
    )
    ap.add_argument("--n-windows", type=int, default=2048)
    # 128 keeps the completeness error ~0.3% of a typical logit; feature ranks are
    # already stable (Spearman 1.00 vs 256 steps), so higher buys nothing.
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--target", default="predicted", choices=["predicted", "label"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    config = json.loads((ckpt_dir / "config.json").read_text())
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "cuda"))

    splits = dict(zip(("train", "val", "test"), build_datasets(config)[:3]))
    dataset = splits[args.split]

    model = _resolve_model(ckpt_dir, config).to(device)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    result = attribute_dataset(
        model,
        dataset,
        config,
        device,
        baseline=args.baseline,
        n_windows=args.n_windows,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        target=args.target,
    )

    groups = feature_groups(config)
    shares = group_attribution(result["per_feature"], groups)
    names = feature_names(config)
    order = np.argsort(result["per_feature"])[::-1]

    logger.info("attribution share by group ({} baseline):", args.baseline)
    for g, v in sorted(shares.items(), key=lambda kv: -kv[1]):
        logger.info("    {:<16} {:6.2%}", g, v)
    logger.info("top-8 features:")
    for i in order[:8]:
        logger.info("    {:<18} {:.5f}", names[i], result["per_feature"][i])

    stem = ckpt_dir / f"ig_{args.baseline}"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        per_feature=result["per_feature"],
        per_time=result["per_time"],
        attr_mean=result["attr_mean"],
        per_window_time=result["per_window_time"],
        targets=result["targets"],
        labels=result["labels"],
    )
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": ckpt_dir.name,
                "split": args.split,
                "baseline": result["baseline"],
                "target": args.target,
                "n_windows": result["n_windows"],
                "n_steps": result["n_steps"],
                "completeness_err": result["completeness_err"],
                "group_shares": shares,
                "feature_names": names,
                "per_feature": result["per_feature"].tolist(),
            },
            indent=2,
        )
    )
    logger.info("wrote {} and {}", stem.with_suffix(".npz"), stem.with_suffix(".json"))


if __name__ == "__main__":
    main()
