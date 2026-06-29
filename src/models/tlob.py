"""TLOB — Temporal-LOB Transformer (Figure 2, TLOB paper).

Input : ``(B, 1, T_past, n_features)`` — same format as DeepLOB / LOBTransformer.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Architecture (per TLOBBlock, bottom→top = first→last):
  1. BilinearNorm  — diagonal W_L (T,) and W_R (dim,):
                     Z[b,t,d] = w_L[t] * x[b,t,d] * w_R[d]
  2. DualAttentionBlock — pre-norm temporal self-attention (embed=dim, seq=T)
                          then pre-norm spatial self-attention (embed=T, seq=dim)
  3. MLPLOBBlock   — row-wise feature-mixing MLP then column-wise temporal-mixing MLP

The linear projection Linear(F, dim) is applied once before the block stack, so
every TLOBBlock receives (B, T, dim); BilinearNorm inside each block therefore
uses dim (not original F).

Config keys
-----------
T_past          : window length T
n_features      : raw feature dimension F  (set by dataset builder)
tlob_dim        : model dim               (default 64)
tlob_n_blocks   : number of TLOBBlocks    (default 4)
tlob_n_heads    : attention heads         (default 1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── BilinearNorm ──────────────────────────────────────────────────────────────


class BilinearNorm(nn.Module):
    """Diagonal bilinear norm: Z = diag(w_L) @ X @ diag(w_R).

    Diagonal constraint keeps O(T+dim) params and prevents rank collapse.
    Initialised to ones so the layer starts as identity.
    """

    def __init__(self, T: int, dim: int) -> None:
        super().__init__()
        self.w_L = nn.Parameter(torch.ones(T))  # (T,)
        self.w_R = nn.Parameter(torch.ones(dim))  # (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim) — scale rows by w_L, columns by w_R
        return x * self.w_L[None, :, None] * self.w_R[None, None, :]


# ── SinusoidalPositionalEncoding ──────────────────────────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE added once after the input projection."""

    def __init__(self, T: int, dim: int) -> None:
        super().__init__()
        pe = torch.zeros(T, dim)
        pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe  # (B, T, dim)


# ── MLPLOBBlock ───────────────────────────────────────────────────────────────


class MLPLOBBlock(nn.Module):
    """Row-wise feature-mixing MLP then column-wise temporal-mixing MLP.

    Feature-mixing  (per t): LayerNorm → Linear(dim,4*dim) → GeLU → Linear(4*dim,dim) + residual
    Temporal-mixing (per d): LayerNorm → Linear(T,4*T)     → GeLU → Linear(4*T,T)     + residual
    """

    def __init__(self, T: int, dim: int) -> None:
        super().__init__()
        self.feat_norm = nn.LayerNorm(dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.temp_norm = nn.LayerNorm(T)
        self.temp_mlp = nn.Sequential(
            nn.Linear(T, 4 * T), nn.GELU(), nn.Linear(4 * T, T)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        x = x + self.feat_mlp(self.feat_norm(x))  # feature-mixing: (B, T, dim)
        xt = x.transpose(1, 2)  # (B, dim, T)
        xt = xt + self.temp_mlp(self.temp_norm(xt))  # temporal-mixing: (B, dim, T)
        return xt.transpose(1, 2)  # (B, T, dim)


# ── DualAttentionBlock ────────────────────────────────────────────────────────


class DualAttentionBlock(nn.Module):
    """Pre-norm temporal self-attention then pre-norm spatial self-attention.

    Temporal: (B, T, dim) — seq=T,   embed=dim, n_heads divides dim.
    Spatial:  (B, dim, T) — seq=dim, embed=T,   n_heads must divide T.
              (achieved by transposing before attending)
    """

    def __init__(self, T: int, dim: int, n_heads: int) -> None:
        super().__init__()
        # temporal attention — attend across T, embed per token = dim
        self.t_norm = nn.LayerNorm(dim)
        self.t_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # spatial attention — after transpose (B, dim, T): seq=dim, embed per token = T
        self.s_norm = nn.LayerNorm(T)
        self.s_attn = nn.MultiheadAttention(T, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)

        # --- temporal attention (across T) ---
        h = self.t_norm(x)
        h, _ = self.t_attn(h, h, h)
        x = x + h  # (B, T, dim)

        # --- spatial attention (across dim, i.e. feature channels) ---
        xs = x.transpose(1, 2)  # (B, dim, T)
        h = self.s_norm(xs)
        h, _ = self.s_attn(h, h, h)
        xs = xs + h  # (B, dim, T)
        return xs.transpose(1, 2)  # (B, T, dim)


# ── TLOBBlock ─────────────────────────────────────────────────────────────────


class TLOBBlock(nn.Module):
    """One TLOB block: BilinearNorm → DualAttention → MLPLOBBlock."""

    def __init__(self, T: int, dim: int, n_heads: int) -> None:
        super().__init__()
        self.bilinear = BilinearNorm(T, dim)
        self.dual_attn = DualAttentionBlock(T, dim, n_heads)
        self.mlp = MLPLOBBlock(T, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bilinear(x)  # diagonal row/col scaling
        x = self.dual_attn(x)  # temporal then spatial attention
        x = self.mlp(x)  # feature-mixing then temporal-mixing
        return x  # (B, T, dim)


# ── TLOB ─────────────────────────────────────────────────────────────────────


class TLOB(nn.Module):
    """TLOB classifier.

    Accepts ``(B, 1, T, F)`` (squeezed internally) or ``(B, T, F)``.
    Returns ``(B, 3)`` trend logits.
    """

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F = config["n_features"]
        dim = config.get("tlob_dim", 64)
        n_blocks = config.get("tlob_n_blocks", 4)
        n_heads = config.get("tlob_n_heads", 1)

        self.T = T
        self.F = F

        self.proj = nn.Linear(F, dim)
        self.pe = SinusoidalPositionalEncoding(T, dim)
        self.blocks = nn.ModuleList(TLOBBlock(T, dim, n_heads) for _ in range(n_blocks))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(T * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, T, F) → (B, T, F)
        assert x.dim() == 3 and x.shape[1] == self.T and x.shape[2] == self.F, (
            f"Expected (B, {self.T}, {self.F}), got {tuple(x.shape)}"
        )
        x = self.pe(self.proj(x))  # (B, T, dim)
        for block in self.blocks:
            x = block(x)  # (B, T, dim)
        return self.head(x)  # (B, 3)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
