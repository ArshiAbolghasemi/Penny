"""CTABL's own explanation: the TABL layer's temporal soft-attention.

``models.ctabl.TABL`` (the final stage of ``CTABLBody``) computes an explicit
``(B, d2, t1) -> softmax over t1`` attention matrix ``a`` and mixes it back
into the bilinear output with a learned scalar ``lam`` (ctabl.py:70-76):

    x = W1 @ X                     # (B, d2, t1) feature transform
    e = x @ W (structure matrix)   # (B, d2, t1)
    a = softmax(e, dim=-1)         # <- this is the attention we extract
    x = lam * (x * a) + (1-lam) * x

This is a genuine, load-bearing part of the forward pass (not a post-hoc
proxy), so reading it out via a forward hook is the most faithful explanation
available for this model — no gradients or surrogate needed.

Caveat: ``a`` lives in the ``(d2, t1)`` space *after* the first two BL
projections have already reduced the raw ``(D, T)`` input, so it is not
directly a per-input-feature/per-input-timestep map. ``t1 == T`` (bl2 keeps
time at t2 = T//2 as its own axis, but the *TABL* module's own ``t1`` input
dimension equals ``ctabl_t2`` from ``bl2``, i.e. an already-downsampled time
axis) — see the shape note in :func:`extract_tabl_attention`. Present it as
"which downsampled time-bin the final classification stage relied on", not a
raw per-timestep map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.ctabl import CTABL


@dataclass
class TABLExplanation:
    attention: torch.Tensor  # (B, d2, t1) softmax weights, t1 = ctabl_t2 (post bl2 downsample)
    lam: float  # learned mixing weight in [0, 1]
    time_importance: torch.Tensor  # (B, t1) — attention averaged over d2 channels


def extract_tabl_attention(model: CTABL, x: torch.Tensor) -> TABLExplanation:
    """Run the model forward while capturing the TABL layer's internal attention.

    Args:
        model: A loaded ``CTABL`` instance (``xai.registry.load_checkpoint``).
        x:     ``(B, 1, T, F)`` input windows.
    """
    tabl = model.body.tabl
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inputs, output):
        # Re-derive `a` exactly as TABL.forward does, from the same `inputs`
        # the hook receives — cheaper and more transparent than patching
        # TABL.forward to stash an extra attribute.
        (inp,) = inputs
        feat = torch.einsum("od,bdt->bot", module.W1, inp)  # (B, d2, t1)
        e = torch.einsum("bot,tu->bou", feat, module.W)  # (B, d2, t1)
        captured["a"] = torch.softmax(e, dim=-1).detach()

    handle = tabl.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()

    a = captured["a"]
    lam = float(tabl.lam.clamp(0.0, 1.0).item())
    return TABLExplanation(
        attention=a,
        lam=lam,
        time_importance=a.mean(dim=1),  # (B, t1) — average over d2 channels
    )
