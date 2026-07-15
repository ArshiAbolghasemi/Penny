"""Deletion / insertion curves — do the attributions actually predict behaviour?

An attribution map that *looks* plausible is worth nothing on its own: rank
correlations and heatmaps are claims about the model that have to be paid for.
The test is behavioural.  Rank the features by attribution, then

* **deletion** — progressively replace the top-ranked features with the
  baseline.  If the attributions are faithful, accuracy falls *fast*.
* **insertion** — start from an all-baseline window and progressively restore the
  top-ranked features.  If they are faithful, accuracy recovers *fast*.

Each curve is summarised by its area (AUC, mean accuracy across the sweep):
**lower deletion AUC is better, higher insertion AUC is better**.  Both are read
against the *random* ranking run under the identical masking procedure — that
control is the whole point, because a curve on its own says nothing about
whether the ordering carried information.

Masking replaces features with the **same baseline the attribution was computed
against** (zeros = no order flow by default; see :mod:`xai.attribution`).  Using a
different filler would score the attributions against a counterfactual they never
claimed, which is the usual way this test is quietly rigged.

Ranking is **global** (one feature order for the whole subsample, from the mean
|attribution|), matching how the paper reports attributions.  Per-window ranking
would measure a different, more permissive claim.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from .attribution import make_baseline


def _fractions(n_features: int, n_points: int) -> np.ndarray:
    """Evenly spaced fractions of features to mask, always including 0 and 1."""
    return np.unique(np.linspace(0.0, 1.0, n_points))


@torch.no_grad()
def _masked_accuracy(
    model,
    dataset,
    indices: np.ndarray,
    device: torch.device,
    order: np.ndarray,
    k: int,
    mode: str,
    baseline: str | torch.Tensor,
    batch_size: int,
) -> float:
    """Accuracy with ``k`` features masked (deletion) or restored (insertion).

    ``order`` is the feature ranking, most important first.  Labels come from the
    dataset, so this measures real accuracy — not agreement with the unmasked
    prediction.
    """
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False
    )
    correct = 0
    total = 0
    sel = torch.as_tensor(order[:k].copy(), dtype=torch.long, device=device)
    for batch in loader:
        x = batch["x"].to(device).float()
        y = batch["label"].to(device)
        base = make_baseline(baseline, x, dataset)
        if mode == "deletion":
            # start from the real window, blank the top-k features
            xm = x.clone()
            if k > 0:
                xm[..., sel] = base[..., sel]
        elif mode == "insertion":
            # start from an all-baseline window, restore the top-k features
            xm = base.clone()
            if k > 0:
                xm[..., sel] = x[..., sel]
        else:
            raise ValueError(f"mode must be deletion|insertion, got {mode!r}")
        pred = model.predict({"x": xm}, device).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def curve(
    model,
    dataset,
    config: dict,
    device: torch.device,
    indices: np.ndarray,
    order: np.ndarray,
    mode: str = "deletion",
    baseline: str | torch.Tensor = "zero",
    n_points: int = 11,
    batch_size: int | None = None,
) -> dict:
    """Accuracy as a function of how many top-ranked features are masked/restored.

    Returns ``{"fractions", "accuracy", "auc", "mode"}`` where ``auc`` is the
    trapezoidal area under the accuracy-vs-fraction curve.
    """
    F = int(config["n_features"])
    bs = batch_size or config.get("batch_size", 64)
    fracs = _fractions(F, n_points)
    accs = [
        _masked_accuracy(
            model, dataset, indices, device, order, int(round(f * F)), mode,
            baseline, bs,
        )
        for f in fracs
    ]
    accs_arr = np.asarray(accs, dtype=float)
    return {
        "fractions": fracs,
        "accuracy": accs_arr,
        "auc": float(np.trapz(accs_arr, fracs)),
        "mode": mode,
    }


def faithfulness(
    model,
    dataset,
    config: dict,
    device: torch.device,
    indices: np.ndarray,
    per_feature: np.ndarray,
    baseline: str | torch.Tensor = "zero",
    n_points: int = 11,
    n_random: int = 5,
    seed: int = 42,
    batch_size: int | None = None,
) -> dict:
    """Deletion + insertion curves for an attribution ranking vs random controls.

    Args:
        per_feature: ``(F,)`` importance scores — typically IG's ``per_feature``.
        n_random:    Random orderings to average for the control band.  Several,
                     because a single random draw is itself noisy and the whole
                     claim rests on beating this control.

    Returns:
        Curves and AUCs for the attribution ranking and the random control, plus
        ``"delta"`` — the AUC gaps.  Faithful attributions give a *negative*
        deletion delta and a *positive* insertion delta.
    """
    order = np.argsort(per_feature)[::-1].copy()  # most important first
    rng = np.random.default_rng(seed)
    out: dict = {"n_windows": int(len(indices)), "baseline": baseline}

    for mode in ("deletion", "insertion"):
        attr_curve = curve(
            model, dataset, config, device, indices, order, mode, baseline,
            n_points, batch_size,
        )
        rnd_accs = []
        rnd_aucs = []
        for _ in range(n_random):
            ro = rng.permutation(len(per_feature))
            c = curve(
                model, dataset, config, device, indices, ro, mode, baseline,
                n_points, batch_size,
            )
            rnd_accs.append(c["accuracy"])
            rnd_aucs.append(c["auc"])
        rnd_mean = np.mean(rnd_accs, axis=0)
        out[mode] = {
            "fractions": attr_curve["fractions"].tolist(),
            "accuracy": attr_curve["accuracy"].tolist(),
            "auc": attr_curve["auc"],
            "random_accuracy": rnd_mean.tolist(),
            "random_auc": float(np.mean(rnd_aucs)),
            "random_auc_std": float(np.std(rnd_aucs)),
        }
        logger.info(
            "{:<9} AUC attr={:.4f}  random={:.4f}±{:.4f}  delta={:+.4f}",
            mode,
            attr_curve["auc"],
            out[mode]["random_auc"],
            out[mode]["random_auc_std"],
            attr_curve["auc"] - out[mode]["random_auc"],
        )

    out["delta"] = {
        # faithful: deletion below random (negative), insertion above (positive)
        "deletion": out["deletion"]["auc"] - out["deletion"]["random_auc"],
        "insertion": out["insertion"]["auc"] - out["insertion"]["random_auc"],
    }
    return out
