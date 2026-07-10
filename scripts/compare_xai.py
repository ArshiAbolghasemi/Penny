"""Cross-model XAI comparison: run all four models on the same windows and compare.

Ties together Task 2 (shared Integrated Gradients) and Task 3 (per-model
native explanations) into the Task 4 deliverable: for a fixed symbol/config,
load all four checkpoints, sample the *same* test windows once, explain every
model on those windows with both its native method and IG, then report:

  1. Pairwise agreement (cosine similarity on comparable maps, time-profile
     correlation, top-k time-step overlap) — see ``xai.compare``.
  2. Attribution-guided deletion faithfulness (does the model's own top-ranked
     cells hurt its own prediction more than random deletion?) — ``xai.faithfulness``.
  3. Aggregate book-level / time-lag importance per model — ``xai.aggregate``.

Requires one trained checkpoint per model on the *same* symbol + feature_mode
+ label_k, since agreement metrics are only meaningful when every model is
predicting the same 3-way task on the same inputs.

Usage::

    uv run python scripts/compare_xai.py \\
        --ctabl checkpoints/nobitex/BTCIRT/ctabl_.../best.pt \\
        --dla checkpoints/nobitex/BTCIRT/dla_.../best.pt \\
        --jointdit checkpoints/nobitex/BTCIRT/jointdit_.../best.pt \\
        --jumpgatelob checkpoints/nobitex/BTCIRT/jumpgatelob_.../best.pt \\
        --n-per-class 6 --out results/xai/compare_btcirt_lob
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from crypto.dataset import build_datasets
from utils.training import resolve_device
from xai.aggregate import aggregate_maps, top_features
from xai.attribution import classifier_fn, target_classes
from xai.compare import (
    HAS_FEATURE_RESOLUTION,
    compare_models,
    gradient_attribution_to_comparable_map,
    time_profile,
    to_comparable_map,
)
from xai.faithfulness import deletion_curve
from xai.gradient_methods import GradientAttributionConfig, integrated_gradients
from xai.native import explain_native
from xai.registry import load_checkpoint
from xai.sampling import CLASS_NAMES, sample_by_class, stack_windows

MODEL_NAMES = ["ctabl", "dla", "jointdit", "jumpgatelob"]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}", default=None, help=f"checkpoint path for {name}")
    parser.add_argument("--n-per-class", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    active_names = [name for name in MODEL_NAMES if getattr(args, name) is not None]
    if len(active_names) < 2:
        raise SystemExit(
            "need at least 2 of --ctabl/--dla/--jointdit/--jumpgatelob to compare"
        )
    if len(active_names) < len(MODEL_NAMES):
        logger.warning(
            "comparing only {} (no checkpoint given for {})",
            active_names, [n for n in MODEL_NAMES if n not in active_names],
        )

    device = resolve_device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {
        name: load_checkpoint(name, getattr(args, name), device) for name in active_names
    }
    ref_config = loaded[active_names[0]].config
    n_levels = ref_config["n_lob_levels"]
    feature_mode = ref_config.get("feature_mode", "ofi")
    for name, lm in loaded.items():
        if lm.config.get("symbol") != ref_config.get("symbol") or lm.config.get(
            "feature_mode"
        ) != feature_mode:
            logger.warning(
                "{} was trained on symbol={} mode={}, expected symbol={} mode={} "
                "(cross-model comparison assumes identical task/inputs)",
                name, lm.config.get("symbol"), lm.config.get("feature_mode"),
                ref_config.get("symbol"), feature_mode,
            )

    # Any one model's config determines the shared test split (same symbol,
    # same T_past/label_k/stride => identical windows across all four).
    _, _, test_ds, _, meta = build_datasets(ref_config)
    logger.info("shared test windows: {}", meta["counts"]["test"])
    samples = sample_by_class(test_ds, n_per_class=args.n_per_class)

    grad_config = GradientAttributionConfig()
    per_model_maps: dict[str, list[np.ndarray]] = {n: [] for n in active_names}
    all_agreements = []
    all_deletion = {n: [] for n in active_names}

    for cls, windows in samples.items():
        if not windows:
            continue
        x_by_model = {
            name: stack_windows(windows).to(device).float() for name in active_names
        }
        for i, w in enumerate(windows):
            comparable_maps: dict[str, torch.Tensor] = {}
            time_profiles: dict[str, torch.Tensor] = {}

            for name in active_names:
                model = loaded[name].model
                x_single = x_by_model[name][i : i + 1]
                fn = classifier_fn(name, model)
                with torch.no_grad():
                    logits = fn(x_single)
                target = target_classes(logits)

                native = explain_native(name, model, x_single)
                T, F = x_single.shape[2], x_single.shape[3]
                comparable_maps[name] = to_comparable_map(name, native, 0, T, F)
                time_profiles[name] = time_profile(name, native, 0)
                per_model_maps[name].append(comparable_maps[name].detach().cpu())

                ig_attr = integrated_gradients(
                    name, model, x_single, target, config=grad_config
                )[0]
                dc = deletion_curve(fn, x_single[0], ig_attr.scores.abs(), int(target[0]))
                all_deletion[name].append(dc)

            agreements = compare_models(comparable_maps, time_profiles)
            all_agreements.extend(agreements)

        logger.info("class={:<11} n={} explained", CLASS_NAMES[cls], len(windows))

    # ---- 1. Aggregate agreement summary ----------------------------------
    pair_summary: dict[str, dict[str, float]] = {}
    for pair in {a.pair for a in all_agreements}:
        vals = [a for a in all_agreements if a.pair == pair]
        pair_summary["-".join(pair)] = {
            "mean_cosine_similarity": float(np.mean([v.cosine_similarity for v in vals])),
            "mean_time_profile_correlation": float(
                np.mean([v.time_profile_correlation for v in vals])
            ),
            "mean_top_k_overlap": float(np.mean([v.top_k_overlap for v in vals])),
            "n_windows": len(vals),
        }
    logger.info("pairwise agreement:")
    for pair, s in sorted(pair_summary.items()):
        logger.info(
            "  {:<25} cos={:.3f}  time_corr={:.3f}  top_k_jaccard={:.3f}",
            pair, s["mean_cosine_similarity"], s["mean_time_profile_correlation"], s["mean_top_k_overlap"],
        )

    # ---- 2. Faithfulness (deletion) summary -------------------------------
    faithfulness_summary: dict[str, dict[str, float]] = {}
    for name in active_names:
        curves = all_deletion[name]
        gaps = [c.faithfulness_gap for c in curves]
        faithfulness_summary[name] = {
            "mean_faithfulness_gap": float(np.mean(gaps)),
            "frac_windows_beats_random": float(np.mean([g > 0 for g in gaps])),
            "n_windows": len(curves),
        }
        logger.info(
            "faithfulness  {:<12} mean_gap={:.4f}  beats_random={:.1%}  (n={})",
            name, faithfulness_summary[name]["mean_faithfulness_gap"],
            faithfulness_summary[name]["frac_windows_beats_random"], len(curves),
        )

    # ---- 3. Aggregate per-level / per-lag importance ----------------------
    agg_summary: dict[str, dict] = {}
    fig, axes = plt.subplots(2, len(active_names), figsize=(5 * len(active_names), 8), squeeze=False)
    for col, name in enumerate(active_names):
        maps = [torch.as_tensor(m) for m in per_model_maps[name]]
        agg = aggregate_maps(name, "native", maps, n_levels, feature_mode)
        axes[0, col].bar(range(len(agg.lag_importance)), agg.lag_importance)
        axes[0, col].set_title(f"{name}\nlag importance (0=oldest)")
        axes[0, col].set_xlabel("time step")

        if HAS_FEATURE_RESOLUTION.get(name, False):
            agg_summary[name] = {"top_features": top_features(agg, k=10)}
            order = np.argsort(agg.feature_importance)[::-1][:15]
            axes[1, col].barh(range(len(order)), agg.feature_importance[order][::-1])
            axes[1, col].set_yticks(range(len(order)))
            axes[1, col].set_yticklabels([agg.feature_labels[i] for i in order][::-1], fontsize=7)
            axes[1, col].set_title("top-15 features")
        else:
            # This model's native map is a (T,) time-only signal broadcast
            # across every feature (see xai.compare.HAS_FEATURE_RESOLUTION) —
            # every feature is tied by construction, so a "top features" bar
            # chart here would show 15 identical bars and mislead a reader.
            agg_summary[name] = {
                "top_features": "n/a (no native per-feature resolution — see time-lag panel above)"
            }
            axes[1, col].axis("off")
            axes[1, col].text(
                0.5, 0.5,
                "no per-feature resolution\n(see time-lag panel above)",
                ha="center", va="center", wrap=True, fontsize=9,
            )
            axes[1, col].set_title("top-15 features")
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_importance.png", dpi=150)
    plt.close(fig)

    summary = {
        "symbol": ref_config.get("symbol"),
        "feature_mode": feature_mode,
        "n_windows_per_class": args.n_per_class,
        "pairwise_agreement": pair_summary,
        "faithfulness": faithfulness_summary,
        "aggregate_importance": agg_summary,
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote comparison_summary.json + aggregate_importance.png -> {}", out_dir)


if __name__ == "__main__":
    main()
