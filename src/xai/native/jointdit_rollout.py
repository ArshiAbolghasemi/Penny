"""JointDiT's own explanation: Attention Rollout over its DiT blocks.

JointDiT (models/jointdit.py) is a DiT — ``depth`` stacked ``DiTBlock``s, each
a standard ``nn.MultiheadAttention`` self-attention over patch tokens, with
U-ViT additive long skips (encoder block i's output added into decoder block
``N-1-i``'s input, jointdit.py:173-185). Two things make raw attention maps
insufficient here and Attention Rollout (Abnar & Zuada, 2020) the right
method instead of using the *last* block's map alone:

  1. Depth: 6 blocks compose, so what block 6 attends to already reflects
     information mixed through blocks 1-5 — rollout multiplies attention
     matrices across the stack (with an added identity for the residual
     connection) to trace the recursive attention flow back to the input
     patch grid.
  2. ``need_weights=False`` is hardcoded in ``DiTBlock.forward`` (jointdit.py:78)
     so attention weights are never returned by the public forward pass. This
     module re-runs each block's attention call with
     ``need_weights=True, average_attn_weights=False`` instead of patching the
     trained module, so the captured weights are exactly what the checkpoint
     already computed — nothing is approximated or retrained.

Explanations must target the **classifier** path (``jointdit.py``'s
``predict()`` contract: clean window, ``t = 0``) — the U-ViT skips and
adaLN-Zero conditioning are identical whether decoding noise or pooling for
classification, since both share the same token stream through ``_encode``.

Output is unpatchified back to ``(T, F)`` via the model's own
``_unpatchify``-equivalent upsampling (nearest-neighbour repeat per patch,
since rollout scores are per *patch*, not per pixel — there is no learned
inverse to invert, unlike the epsilon head's ``FinalLayer``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.jointdit import JointDiT, _modulate


@dataclass
class RolloutExplanation:
    patch_scores: torch.Tensor  # (B, gt, gf) rollout importance per patch
    scores: torch.Tensor  # (B, T, F) upsampled back to input resolution


def _block_attention(
    block, x: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-run one DiTBlock's forward, but with attention weights captured.

    Mirrors ``DiTBlock.forward`` (jointdit.py:75-82) exactly; the only change
    is ``need_weights=True, average_attn_weights=False`` on the attention call.
    """
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = block.ada(c).chunk(6, dim=1)
    h = _modulate(block.norm1(x), shift_a, scale_a)
    a, w = block.attn(h, h, h, need_weights=True, average_attn_weights=False)
    # w: (B, heads, N, N)
    x = x + gate_a.unsqueeze(1) * a
    h = _modulate(block.norm2(x), shift_m, scale_m)
    x = x + gate_m.unsqueeze(1) * block.mlp(h)
    return x, w


@torch.no_grad()
def extract_attention_rollout(
    model: JointDiT, x: torch.Tensor, discard_ratio: float = 0.0
) -> RolloutExplanation:
    """Compute Attention Rollout for a batch of clean windows.

    Args:
        model: A loaded ``JointDiT`` instance.
        x:     ``(B, 1, T, F)`` clean windows (matches the ``predict()``
               contract: this is *not* a noised ``x_t``).
        discard_ratio: Optionally zero out the lowest ``discard_ratio``
            fraction of attention weights per row before rolling out (Abnar &
            Zuada's noise-suppression variant). ``0.0`` = off.
    """
    b = x.shape[0]
    t = torch.zeros(b, dtype=torch.long, device=x.device)
    dim = model.pos_time.shape[-1]
    c = model._temb(t, dim)
    tok = model._tokenize(x)  # (B, N, D), N = gt*gf

    N = tok.shape[1]
    rollout = torch.eye(N, device=x.device).unsqueeze(0).expand(b, -1, -1).clone()

    skips: list[torch.Tensor] = []
    half = model.depth // 2
    for i, blk in enumerate(model.blocks):
        if i < half:
            tok, w = _block_attention(blk, tok, c)
            skips.append(tok)
        elif i >= model.depth - half:
            tok, w = _block_attention(blk, tok + skips.pop(), c)
        else:
            tok, w = _block_attention(blk, tok, c)

        w_avg = w.mean(dim=1)  # (B, N, N) average over heads
        if discard_ratio > 0.0:
            flat = w_avg.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                thresh = flat.kthvalue(k, dim=1, keepdim=True).values
                w_avg = torch.where(w_avg < thresh.unsqueeze(-1), torch.zeros_like(w_avg), w_avg)
        # add identity for the residual connection, renormalise rows to sum to 1
        w_res = 0.5 * w_avg + 0.5 * torch.eye(N, device=x.device).unsqueeze(0)
        w_res = w_res / w_res.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(w_res, rollout)

    # importance of each source patch = how much attention flows *into* it,
    # summed over all destination tokens (mean-pooled classifier reads all tokens)
    patch_importance = rollout.sum(dim=1)  # (B, N)
    patch_scores = patch_importance.view(b, model.gt, model.gf)

    # upsample each patch's scalar score to its p x p pixel footprint, then
    # crop back from the padded grid to (T, F) — same crop model._unpatchify uses.
    p = model.p
    up = patch_scores.unsqueeze(1)  # (B, 1, gt, gf)
    up = F.interpolate(up, scale_factor=p, mode="nearest")  # (B, 1, gt*p, gf*p)
    up = up[:, 0, : model.T, : model.F]  # (B, T, F)

    return RolloutExplanation(patch_scores=patch_scores, scores=up)
