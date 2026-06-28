"""Gradient surgery (PCGrad) for the two JointDiffCFG objectives.

JointDiffCFG optimises a shared U-Net backbone (``time_mlp``, ``asset_emb``,
``stem``, ``downs``) with two losses:

  * a diffusion / data-distribution loss  (MSE on predicted noise), routed
    through the decoder (``ups`` / ``out_conv``);
  * a trend-classification loss           (cross-entropy on the head), routed
    through ``classifier``.

When their gradients point in conflicting directions (negative inner product on
the shared backbone) a naive ``diff + lambda * cls`` sum lets one task degrade
the other.  PCGrad (Yu et al., 2020, *Gradient Surgery for Multi-Task Learning*,
arXiv:2001.06782) removes the conflicting component: each gradient is projected
onto the normal plane of the other before summing.  For two tasks ``a``, ``b``
with ``<g_a, g_b> < 0``::

    g_a' = g_a - (<g_a, g_b> / ||g_b||^2) g_b
    g_b' = g_b - (<g_b, g_a> / ||g_a||^2) g_a
    g    = g_a' + g_b'

Only the *shared* parameters — those that receive a gradient from **both**
losses — are de-conflicted.  Task-specific parameters (the trend head, the
decoder / output conv) receive their natural single-task gradient unchanged.
"""

from __future__ import annotations

from typing import Iterable

import torch


def pcgrad_backward(
    loss_diff: torch.Tensor,
    loss_cls: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
) -> float:
    """Populate ``p.grad`` for ``params`` with PCGrad-projected gradients.

    Args:
        loss_diff: data-distribution (diffusion) loss tensor.
        loss_cls:  classification loss tensor, **already scaled** by
                   ``lambda_trend`` so the projection respects task weighting.
        params:    iterable of model parameters (the optimiser's param set).

    Returns:
        The cosine similarity of the two task gradients on the shared backbone
        (in ``[-1, 1]``; ``nan`` if either side has no shared gradient), so the
        caller can log how often / how strongly the two objectives conflict.
        Replaces ``loss.backward()`` — do **not** also call it.
    """
    params = [p for p in params if p.requires_grad]

    g_diff = torch.autograd.grad(
        loss_diff, params, retain_graph=True, allow_unused=True
    )
    g_cls = torch.autograd.grad(loss_cls, params, retain_graph=False, allow_unused=True)

    # Inner product / squared norms over the SHARED params only (grad from both).
    dev = params[0].device
    dot = torch.zeros((), device=dev)
    n_diff_sq = torch.zeros((), device=dev)
    n_cls_sq = torch.zeros((), device=dev)
    shared: list[bool] = []
    for gd, gc in zip(g_diff, g_cls):
        is_shared = gd is not None and gc is not None
        shared.append(is_shared)
        if is_shared:
            dot += (gd * gc).sum()
            n_diff_sq += (gd * gd).sum()
            n_cls_sq += (gc * gc).sum()

    conflict = bool(dot < 0) and float(n_diff_sq) > 0.0 and float(n_cls_sq) > 0.0
    coeff_diff = dot / n_cls_sq if conflict else None  # remove g_diff ∥ g_cls
    coeff_cls = dot / n_diff_sq if conflict else None  # remove g_cls  ∥ g_diff

    for p, gd, gc, is_shared in zip(params, g_diff, g_cls, shared):
        if gd is None and gc is None:
            p.grad = None
        elif not is_shared:
            p.grad = gd if gd is not None else gc  # task-specific: pass through
        elif conflict:
            p.grad = (gd - coeff_diff * gc) + (gc - coeff_cls * gd)
        else:
            p.grad = gd + gc

    denom = float(n_diff_sq.sqrt() * n_cls_sq.sqrt())
    return float(dot) / denom if denom > 0.0 else float("nan")
