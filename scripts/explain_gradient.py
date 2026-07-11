"""Explain a trained model's trend predictions with Integrated Gradients + GradientSHAP.

Loads a checkpoint, rebuilds its test split (same calendar-day split used at
training time — see ``crypto.dataset.build_datasets``), samples matched
windows per trend class, and runs both shared gradient-based XAI methods
(see ``xai/gradient_methods.py`` for why these two and not LIME/KernelSHAP
here). Saves one heatmap PNG per explained window plus a raw ``.pt`` of all
attributions for later cross-model comparison (Task 4).

Usage::

    uv run python scripts/explain_gradient.py \\
        --model ctabl --checkpoint checkpoints/nobitex/BTCIRT/ctabl_BTCIRT_lob_20260101_000000 \\
        --n-per-class 8 --out results/xai/ctabl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import torch
from loguru import logger

from crypto.dataset import build_datasets
from utils.training import resolve_device
from xai.attribution import classifier_fn, target_classes
from xai.baselines import mean_baseline, zero_baseline
from xai.gradient_methods import (
    GradientAttributionConfig,
    gradient_shap,
    integrated_gradients,
)
from xai.registry import load_checkpoint
from xai.sampling import CLASS_NAMES, sample_by_class, stack_windows
from xai.visualize import plot_attribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ctabl", "dla", "jumpgatelob"])
    parser.add_argument("--checkpoint", required=True, help="path to best.pt or its containing dir")
    parser.add_argument("--n-per-class", type=int, default=8)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--gradshap-samples", type=int, default=50)
    parser.add_argument("--out", default=None, help="output dir; defaults next to the checkpoint")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = resolve_device("cuda")

    loaded = load_checkpoint(args.model, ckpt_path, device)
    config = loaded.config
    logger.info(
        "loaded {}  symbol={} mode={}  device={}",
        args.model, config.get("symbol"), config.get("feature_mode"), device,
    )

    _, _, test_ds, _, meta = build_datasets(config)
    logger.info("test windows: {}", meta["counts"]["test"])

    samples = sample_by_class(test_ds, n_per_class=args.n_per_class)
    grad_config = GradientAttributionConfig(
        ig_steps=args.ig_steps, gradient_shap_samples=args.gradshap_samples
    )

    # ckpt_path may be a best.pt file or its containing checkpoint dir (see
    # xai.registry.load_checkpoint) — resolve to the checkpoint dir either way
    # so the default output path is unique per model/run, not per shared parent.
    ckpt_dir = ckpt_path.parent if ckpt_path.suffix == ".pt" else ckpt_path
    out_dir = Path(args.out) if args.out else ckpt_dir / "xai" / "gradient"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_base = mean_baseline(test_ds).to(device)
    all_attrs = {"integrated_gradients": [], "gradient_shap": []}

    for cls, windows in samples.items():
        if not windows:
            continue
        x = stack_windows(windows).to(device).float()
        with torch.no_grad():
            logits = classifier_fn(args.model, loaded.model)(x)
        target = target_classes(logits)

        ig_attrs = integrated_gradients(args.model, loaded.model, x, target, config=grad_config)
        # GradientSHAP needs a *distribution* of baselines: zero window + mean
        # window + light noise around each, so the expectation isn't over a
        # single degenerate point.
        base_pool = torch.cat(
            [zero_baseline(x[:1]), mean_base, mean_base + 0.01 * torch.randn_like(mean_base)],
            dim=0,
        )
        gs_attrs = gradient_shap(args.model, loaded.model, x, target, base_pool, config=grad_config)

        for i, (ig_a, gs_a) in enumerate(zip(ig_attrs, gs_attrs)):
            tag = f"{CLASS_NAMES[cls]}_{windows[i].index}"
            for attr, method in ((ig_a, "ig"), (gs_a, "gradshap")):
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_attribution(
                    attr, config["n_lob_levels"], config.get("feature_mode", "ofi"), ax=ax
                )
                fig.tight_layout()
                fig.savefig(out_dir / f"{method}_{tag}.png", dpi=150)
                plt.close(fig)
        all_attrs["integrated_gradients"].extend(ig_attrs)
        all_attrs["gradient_shap"].extend(gs_attrs)

        logger.info(
            "class={:<11} n={}  predicted-target matches label for {}/{}",
            CLASS_NAMES[cls], len(windows),
            sum(1 for w, t in zip(windows, target.tolist()) if w.label == t),
            len(windows),
        )

    torch.save(all_attrs, out_dir / "attributions.pt")
    logger.info("saved {} heatmaps + attributions.pt -> {}", 2 * sum(len(v) for v in samples.values()), out_dir)


if __name__ == "__main__":
    main()
