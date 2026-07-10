"""JumpGateLOB's own explanation: pooling attention + the Lévy noise-state gate.

Unlike DLA/CTABL, JumpGateLOB's classification attention sits over *latent*
GRU context (``H0``), not raw input features, so a plain attention map is
harder to read as "which input cell mattered". Instead this module surfaces
the two things that are both (a) genuinely load-bearing in the forward pass
and (b) interpretable in domain terms:

  1. ``AttentionPool`` weights (models/modules.py:50-66) — the single learned
     query's attention over the ``T`` trunk timesteps immediately before the
     3-way classifier head (jumpgatelob.py:288-289). This is the one place
     JumpGateLOB genuinely compresses "which timestep mattered" into a
     scalar-per-step map, so it's the closest analogue to DLA's beta.
  2. The intrinsic Lévy noise-state gate ``pi = sigmoid(pi_logit)`` and
     ``logW_hat`` (jumpgatelob.py:233-235, 308) from the shared
     ``NoiseStateEstimator``. These aren't post-hoc explanations at all —
     they are the model's own belief about whether the window sits in a
     jump/heavy-tail regime, which is exactly the kind of "why" a generic
     attribution method cannot recover. Reported as a scalar overlay
     alongside the pooling attention, not a full (T, F) map.

Both ``need_weights=False`` calls (jumpgatelob.py: TemporalAttnBlock.attn via
modules.py's AttentionPool.attn) are re-run here with weights enabled, exactly
mirroring the trained forward pass — no retraining, no approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.jumpgatelob import JumpGateLOB


@dataclass
class JumpGateExplanation:
    pool_attention: torch.Tensor  # (B, T) — AttentionPool's weights over trunk timesteps
    logW_hat: torch.Tensor  # (B,) inferred log Lévy noise-scale
    pi: torch.Tensor  # (B,) inferred jump-regime gate, sigmoid(pi_logit) in [0, 1]


@torch.no_grad()
def extract_jumpgate_readout(model: JumpGateLOB, x: torch.Tensor) -> JumpGateExplanation:
    """Run the clean-window classify path, capturing pool attention + gate state.

    Args:
        model: A loaded ``JumpGateLOB`` instance.
        x:     ``(B, 1, T, F)`` clean windows (matches ``predict()``/``classify()``:
               ``t = 0``, no added noise).
    """
    t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    H, logW_hat, pi_logit, _c = model.trunk(x, t, None)  # (B, T, D)

    pool = model.pool
    q = pool.query.expand(H.shape[0], -1, -1)
    _out, w = pool.attn(q, H, H, need_weights=True, average_attn_weights=True)
    # w: (B, 1, T) — one query attending over T trunk timesteps
    pool_attention = w.squeeze(1)  # (B, T)

    return JumpGateExplanation(
        pool_attention=pool_attention,
        logW_hat=logW_hat,
        pi=torch.sigmoid(pi_logit),
    )
