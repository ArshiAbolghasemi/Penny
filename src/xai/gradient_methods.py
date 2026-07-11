"""Shared cross-model baseline: Integrated Gradients + GradientSHAP via captum.

Both methods are model-agnostic (they only need ``classifier_fn``'s
``(B,1,T,F) -> (B,3)`` signature from :mod:`xai.attribution`), so this is the
one XAI track that produces directly comparable attributions across CTABL,
DLA, and JumpGateLOB on identical windows — the common yardstick for Task 4's
cross-model comparison, on top of which each model's native method (Task 3)
is layered. JointDiT is intentionally excluded from the XAI layer (decision
2026-07-11) — see docs/xai/README.md.

Why these two, and why not plain KernelSHAP/LIME here: see the design
discussion in conversation — LOB windows are heavily autocorrelated across
both axes (adjacent levels, adjacent timesteps), so perturbation-based
methods that flip individual cells independently generate off-manifold,
physically impossible order books (crossed spreads, contradictory depth).
IG and GradientSHAP instead interpolate along a straight path from a baseline
window to the real one — every intermediate point is at least a smooth
combination of two real feature vectors, and gradients are cheap (tens of
backward passes, not thousands of forward passes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from captum.attr import GradientShap, IntegratedGradients

from xai.attribution import Attribution, ClassifierFn, classifier_fn
from xai.baselines import zero_baseline


@dataclass
class GradientAttributionConfig:
    ig_steps: int = 50
    gradient_shap_samples: int = 50
    gradient_shap_stdevs: float = 0.09  # noise added around each baseline draw


def _squeeze_channel(scores: torch.Tensor) -> torch.Tensor:
    # captum preserves input shape (B, 1, T, F); drop the singleton channel
    # so callers/visualize.py work with the same (T, F) as Attribution.scores.
    return scores.squeeze(1)


def integrated_gradients(
    model_name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor | None = None,
    config: GradientAttributionConfig | None = None,
) -> list[Attribution]:
    """Run Integrated Gradients on a batch, one :class:`Attribution` per sample.

    Args:
        model_name: Registry key (for labeling the returned Attributions).
        model:      Loaded model (``xai.registry.load_checkpoint(...).model``).
        x:          ``(B, 1, T, F)`` input windows.
        target:     ``(B,)`` int class indices to explain (see
                    ``xai.attribution.target_classes``).
        baseline:   ``(1, T, F)`` or ``(B, 1, T, F)``; defaults to the zero
                    window (appropriate since these are z-scored features).
        config:     Step-count overrides.
    """
    config = config or GradientAttributionConfig()
    fn: ClassifierFn = classifier_fn(model_name, model)
    ig = IntegratedGradients(fn)

    base = zero_baseline(x) if baseline is None else baseline.expand_as(x)
    scores, _ = ig.attribute(
        x,
        baselines=base,
        target=target,
        n_steps=config.ig_steps,
        return_convergence_delta=True,
    )
    scores = _squeeze_channel(scores)

    return [
        Attribution(
            model_name=model_name,
            method="integrated_gradients",
            target_class=int(target[i]),
            scores=scores[i].detach(),
            input=x[i, 0].detach(),
        )
        for i in range(x.shape[0])
    ]


def gradient_shap(
    model_name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    baseline_distribution: torch.Tensor,
    config: GradientAttributionConfig | None = None,
) -> list[Attribution]:
    """Run GradientSHAP on a batch, one :class:`Attribution` per sample.

    Args:
        baseline_distribution: ``(K, 1, T, F)`` pool of reference windows
            GradientSHAP samples from (e.g. a handful of zero/mean baselines,
            or real windows from another class) — unlike IG's single
            baseline, GradientSHAP expects a *distribution* to integrate over,
            which is what gives it its (approximate) Shapley-value grounding.
    """
    config = config or GradientAttributionConfig()
    fn: ClassifierFn = classifier_fn(model_name, model)
    gs = GradientShap(fn)

    scores, _ = gs.attribute(
        x,
        baselines=baseline_distribution,
        target=target,
        n_samples=config.gradient_shap_samples,
        stdevs=config.gradient_shap_stdevs,
        return_convergence_delta=True,
    )
    scores = _squeeze_channel(scores)

    return [
        Attribution(
            model_name=model_name,
            method="gradient_shap",
            target_class=int(target[i]),
            scores=scores[i].detach(),
            input=x[i, 0].detach(),
        )
        for i in range(x.shape[0])
    ]
