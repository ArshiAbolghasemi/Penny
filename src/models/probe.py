"""Shared machinery for the decoupled two-phase train procedure.

The procedure applies to any joint generative backbone that exposes the
``denoise(x, sigma) -> (x0_hat, logits)`` contract — currently :class:`JointDiT`
(DiT trunk).  The architecture is not modified: Phase 1 trains the full generative
pathway on a single generative objective (classifier excluded from the loss graph),
Phase 2 freezes that trunk and trains only a probe head + temporal aggregator on
intermediate activations tapped from **one** preconditioned forward pass (no
sampling).

This module provides the parts both phases share:

  * :func:`build_backbone` — config-driven trunk dispatch.
  * :func:`default_tap_blocks` / :func:`tap_modules` — the intermediate
    (mid-decoder / U-ViT mid-skip) blocks whose activations Phase 2 taps.
  * :class:`TrunkFeatureExtractor` — registers forward hooks, runs one frozen
    preconditioned pass per swept ``sigma``, and pools every ``{block × sigma}``
    tap to a common ``(B, time, feature)`` tensor.
  * :class:`TemporalProbe` — attention-pool / GRU aggregator + shallow MLP head
    (the only trainable module in Phase 2).
  * :func:`ordinal_ce` — ordinal-aware, class-weighted 3-class loss.
  * :func:`class_weights_from_labels` — inverse-frequency weights (down-weights
    the flat majority).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jointdit import JointDiT
from models.modules import AttentionPool

# Backbones exposing denoise(x, sigma) -> (x0_hat, logits); add new trunks here.
_BACKBONES = {"jointdit": JointDiT}
# Friendly alias so configs can say "dit".
_ALIASES = {"dit": "jointdit"}


def build_backbone(config: dict) -> nn.Module:
    """Construct the generative trunk named by ``config['backbone']`` (default DiT)."""
    name = config.get("backbone", "jointdit")
    name = _ALIASES.get(name, name)
    if name not in _BACKBONES:
        raise ValueError(
            f"unknown backbone '{config.get('backbone')}'; choose from "
            f"{list(_BACKBONES)} or aliases {list(_ALIASES)}"
        )
    return _BACKBONES[name](config)


def default_tap_blocks(model: nn.Module) -> list[int]:
    """Indices of the intermediate blocks Phase 2 taps by default.

    The decoder-half DiT blocks (mirror of the U-ViT skips) — "mid/late decoder"
    activations that still carry per-time-step structure (not the final denoiser
    output).
    """
    if hasattr(model, "blocks"):  # DiT
        depth = len(model.blocks)
        return list(range(depth // 2, depth))
    raise TypeError(f"no known tap points on {type(model).__name__}")


def tap_modules(model: nn.Module, indices: list[int]) -> list[nn.Module]:
    """Resolve tap-block indices to the actual sub-modules to hook."""
    if hasattr(model, "blocks"):
        return [model.blocks[i] for i in indices]
    raise TypeError(f"no known tap points on {type(model).__name__}")


def _tap_to_time_feat(act: torch.Tensor, grid: tuple[int, int] | None) -> torch.Tensor:
    """Pool a DiT token activation's column patches → per-time-step features.

    ``(B, N, D)`` with patch grid ``(gt, gf)`` → reshape to ``(B, gt, gf, D)`` and
    mean over the column patches ``gf`` → ``(B, gt, D)``.
    """
    assert act.dim() == 3, f"unexpected tap ndim={act.dim()}"
    assert grid is not None, "DiT tap needs the patch grid (gt, gf)"
    gt, gf = grid
    b, n, d = act.shape
    return act.reshape(b, gt, gf, d).mean(dim=2)  # (B, gt, D)


def _resample_time(x: torch.Tensor, length: int) -> torch.Tensor:
    """Linear-resample a ``(B, T, f)`` tensor along time to ``length``."""
    if x.shape[1] == length:
        return x
    y = F.interpolate(
        x.transpose(1, 2), size=length, mode="linear", align_corners=False
    )
    return y.transpose(1, 2)


class TrunkFeatureExtractor:
    """Frozen-trunk feature tap: one preconditioned forward per swept ``sigma``.

    For each ``sigma`` in ``sigmas`` a noised window ``x0 + sigma·n`` is pushed
    through the frozen ``denoise`` once (``torch.no_grad``); forward hooks capture
    the requested intermediate blocks.  Every ``{block × sigma}`` tap is pooled to
    ``(B, time, feat)``, resampled to a common time length, and concatenated along
    the feature axis → ``(B, L, total_feat)``.  No sampling loop, no gradient into
    the trunk.
    """

    def __init__(
        self, model: nn.Module, block_indices: list[int], sigmas: list[float]
    ) -> None:
        self.model = model
        self.sigmas = list(sigmas)
        self.modules = tap_modules(model, block_indices)
        self.grid = (model.gt, model.gf) if hasattr(model, "gt") else None
        self._acts: dict[int, torch.Tensor] = {}
        self._handles = [
            m.register_forward_hook(self._hook(j)) for j, m in enumerate(self.modules)
        ]

    def _hook(self, j: int):
        def fn(_module, _inp, out):
            self._acts[j] = out.detach()

        return fn

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    @property
    def n_taps(self) -> int:
        return len(self.modules) * len(self.sigmas)

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor, time_len: int) -> torch.Tensor:
        self.model.eval()
        taps: list[torch.Tensor] = []
        for sigma in self.sigmas:
            n = torch.randn_like(x0)
            sig = torch.full((x0.shape[0],), float(sigma), device=x0.device)
            self._acts.clear()
            self.model.denoise(x0 + sigma * n, sig)  # triggers hooks
            for j in range(len(self.modules)):
                tf = _tap_to_time_feat(self._acts[j], self.grid)  # (B, T_j, f_j)
                taps.append(_resample_time(tf, time_len))
        return torch.cat(taps, dim=-1)  # (B, time_len, total_feat)

    @torch.no_grad()
    def feature_dim(self, x0: torch.Tensor, time_len: int) -> int:
        return self(x0[:1], time_len).shape[-1]


class TemporalProbe(nn.Module):
    """Phase-2 trainable head: temporal aggregator + shallow MLP over frozen feats.

    ``feats`` is ``(B, L, in_dim)`` from :class:`TrunkFeatureExtractor`.  The time
    axis is collapsed by an attention pool (learned query over time) or a 1-layer
    GRU, then a dropout-heavy MLP maps the summary to 3 trend logits.  This is the
    *only* module with gradients in Phase 2.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        aggregator: str = "attn",
        heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.aggregator = aggregator
        if aggregator == "gru":
            self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
            agg_dim = hidden
        elif aggregator == "attn":  # learned query over time (default for U-Net)
            self.pool = AttentionPool(in_dim, heads=heads, dropout=dropout)
            agg_dim = in_dim
        elif aggregator == "mlp":  # mean-pool over time → plain MLP (default for DiT)
            agg_dim = in_dim
        else:
            raise ValueError(
                f"aggregator must be 'attn' | 'mlp' | 'gru', got {aggregator}"
            )
        self.head = nn.Sequential(
            nn.LayerNorm(agg_dim),
            nn.Dropout(dropout),
            nn.Linear(agg_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.in_norm(feats)
        if self.aggregator == "gru":
            _, h = self.rnn(feats)
            summary = h.squeeze(0)
        elif self.aggregator == "attn":
            summary = self.pool(feats)
        else:  # mlp: simple mean-pool over the time axis
            summary = feats.mean(dim=1)
        return self.head(summary)


def class_weights_from_labels(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency 3-class weights (down-weights the flat majority)."""
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (3.0 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def ordinal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    lam_ordinal: float = 0.5,
) -> torch.Tensor:
    """Ordinal-aware, class-weighted 3-class loss.

    Class-weighted cross-entropy plus an EMD-style penalty on the *distance*
    between the predicted expected class and the true class — a down↔up confusion
    (distance 2) is penalized harder than down↔flat (distance 1), encoding the
    ordering down < flat < up.
    """
    ce = F.cross_entropy(logits, target, weight=class_weights)
    p = F.softmax(logits, dim=1)
    k = torch.arange(3, device=logits.device, dtype=p.dtype)
    expected = (p * k).sum(dim=1)  # expected class index in [0, 2]
    ordinal = ((expected - target.to(p.dtype)) ** 2).mean()
    return ce + lam_ordinal * ordinal
