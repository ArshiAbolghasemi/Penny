"""Shared building blocks used across model files.

Centralised here so each model file imports from one place instead of
duplicating definitions or chaining imports through other model files.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── BiN ───────────────────────────────────────────────────────────────────────


class BiN(nn.Module):
    """Bilinear normalisation applied to (B, T, F).

    Learns a softmax-weighted convex mix of:
      - temporal branch: z-score each feature across T
      - feature  branch: z-score each timestep across F
    """

    def __init__(self, T: int, F: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma_t = nn.Parameter(torch.ones(F))
        self.beta_t = nn.Parameter(torch.zeros(F))
        self.gamma_f = nn.Parameter(torch.ones(T))
        self.beta_f = nn.Parameter(torch.zeros(T))
        self.mix = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mt = x.mean(1, keepdim=True)
        st = x.std(1, keepdim=True) + self.eps
        xt = (x - mt) / st * self.gamma_t + self.beta_t
        mf = x.mean(2, keepdim=True)
        sf = x.std(2, keepdim=True) + self.eps
        xf = (x - mf) / sf * self.gamma_f[None, :, None] + self.beta_f[None, :, None]
        w = torch.softmax(self.mix, 0)
        return w[0] * xt + w[1] * xf


# ── Attention pooling ──────────────────────────────────────────────────────────


class AttentionPool(nn.Module):
    """Single learned query attends over a (B, N, D) sequence -> (B, D).

    Drop-in replacement for ``x.mean(dim=1)`` / ``AdaptiveAvgPool2d(1)`` before
    a classification head — same input/output shape contract, but the summary
    vector is a learned weighted combination of tokens instead of a uniform one.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pooled summary; with ``return_attn`` also the ``(B, N)`` pooling
        weights — how much each token contributed to the summary."""
        q = self.query.expand(x.shape[0], -1, -1)
        if not return_attn:
            out, _ = self.attn(q, x, x, need_weights=False)
            return out.squeeze(1)
        out, w = self.attn(q, x, x, need_weights=True, average_attn_weights=True)
        return out.squeeze(1), w.squeeze(1)  # (B, D), (B, N)


# ── Sinusoidal time embedding ─────────────────────────────────────────────────


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer or float timestep t (B,) to sinusoidal embedding (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ── GroupNorm helper ──────────────────────────────────────────────────────────


def _groups(ch: int) -> int:
    """Largest power-of-two divisor of ch, capped at 8, for GroupNorm."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


# ── U-Net building blocks ─────────────────────────────────────────────────────


class TimeDoubleConv(nn.Module):
    """Two 3×3 convs with an additive scalar time embedding."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = x + self.temb(temb).unsqueeze(-1).unsqueeze(-1)
        return self.act(self.norm2(self.conv2(x)))


class Down(nn.Module):
    """MaxPool2d(2) followed by TimeDoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = TimeDoubleConv(in_ch, out_ch, temb_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x), temb)


class Up(nn.Module):
    """Nearest-neighbour upsample + channel reduction + skip concat + TimeDoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = TimeDoubleConv(out_ch + skip_ch, out_ch, temb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, temb)


# ── Utility ───────────────────────────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Cross-level attention (used by the joint diffusion models) ──────────────


class LevelAttention(nn.Module):
    """Self-attention across the ``F`` book levels, applied per (batch, timestep).

    Input/output ``(B, C, T, F)``; the ``F`` positions are the attention tokens, so
    every level can attend to every other level (cross-level mixing).
    """

    def __init__(self, channels: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * t, f, c)  # (B*T, F, C) tokens = levels
        hn = self.norm(h)
        a, _ = self.attn(hn, hn, hn, need_weights=False)
        h = h + a
        return h.reshape(b, t, f, c).permute(0, 3, 1, 2)  # (B, C, T, F)


class LevelMLP(nn.Module):
    """Token-mixing MLP across the ``F`` book levels (MLP-Mixer style, Tolstikhin
    et al. 2021), applied per (batch, timestep) — same interface/contract as
    :class:`LevelAttention` (same reshape, same internal residual) but mixes
    across ``F`` with a 2-layer MLP instead of self-attention: no O(F^2) cost, no
    learned queries/keys, a cheaper and simpler cross-level mixer.

    Input/output ``(B, C, T, F)``.
    """

    def __init__(
        self, channels: int, n_features: int, mult: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        hidden = max(1, n_features * mult)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * t, f, c)  # (B*T, F, C) tokens = levels
        hn = self.norm(h)
        m = self.mlp(hn.transpose(1, 2)).transpose(1, 2)  # mix across F (tokens)
        h = h + m
        return h.reshape(b, t, f, c).permute(0, 3, 1, 2)  # (B, C, T, F)
