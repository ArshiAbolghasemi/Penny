"""CTABL — Temporal Attention-augmented Bilinear Network for LOB data.

Based on: Tran et al., "Temporal Attention-Augmented Bilinear Network for
Financial Time-Series Data Analysis" (IEEE TNNLS 2019).

Operates directly on the (D features × T time) matrix with bilinear layers that
jointly transform the feature and temporal modes:

  * ``BL``   — bilinear layer  Y = φ(W₁ X W₂ + B).
  * ``TABL`` — bilinear layer with a temporal soft-attention step between the
    feature and temporal transforms, mixed back in by a learnable scalar λ.

CTABL stacks BL → BL → TABL, reducing (D, T) → (d₁, T) → (d₂, T/2) → (3, 1).

Input : ``(B, 1, T_past, n_features)`` — transposed internally to ``(B, D, T)``.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Config keys
-----------
ctabl_d1       features after the first BL layer   (default 60)
ctabl_d2       features after the second BL layer  (default 120)
ctabl_t2       time steps after the second BL      (default T_past // 2)
ctabl_dropout  dropout between layers              (default 0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import count_parameters as count_parameters  # re-export


class BL(nn.Module):
    """Bilinear layer: ``(B, d1, t1) -> phi(W1 X W2 + B) -> (B, d2, t2)``."""

    def __init__(self, d1: int, t1: int, d2: int, t2: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d2, d1))  # feature transform
        self.W2 = nn.Parameter(torch.empty(t1, t2))  # temporal transform
        self.B = nn.Parameter(torch.zeros(d2, t2))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("od,bdt->bot", self.W1, x)  # (B, d2, t1)
        x = torch.einsum("bot,ts->bos", x, self.W2) + self.B  # (B, d2, t2)
        return self.act(x)


class TABL(nn.Module):
    """Temporal Attention-augmented Bilinear Layer.

    ``(B, d1, t1) -> (B, d2, t2)``.  Between the feature transform (W1) and the
    temporal transform (W2) a soft attention over the temporal axis is computed
    via a learnable (t1 × t1) structure matrix and mixed back with weight λ.
    """

    def __init__(self, d1: int, t1: int, d2: int, t2: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d2, d1))  # feature transform
        self.W = nn.Parameter(torch.eye(t1))  # temporal attention structure
        self.W2 = nn.Parameter(torch.empty(t1, t2))  # temporal transform
        self.B = nn.Parameter(torch.zeros(d2, t2))
        self.lam = nn.Parameter(torch.tensor(0.5))  # attention mixing weight
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = torch.einsum("od,bdt->bot", self.W1, x)  # (B, d2, t1)
        e = torch.einsum("bot,tu->bou", x, self.W)  # (B, d2, t1)
        a = torch.softmax(e, dim=-1)  # temporal attention
        lam = self.lam.clamp(0.0, 1.0)
        x = lam * (x * a) + (1.0 - lam) * x
        out = torch.einsum("bot,ts->bos", x, self.W2) + self.B  # (B, d2, t2)
        if return_attn:
            return out, a
        return out


class CTABLBody(nn.Module):
    """Shared BL → BL → TABL trunk mapping ``(B, D, T)`` to ``(B, 3)`` logits."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        D = config["n_features"]
        d1 = config.get("ctabl_d1", 60)
        d2 = config.get("ctabl_d2", 120)
        t2 = config.get("ctabl_t2", max(T // 2, 1))
        drop = config.get("ctabl_dropout", 0.1)

        self.bl1 = BL(D, T, d1, T)
        self.bl2 = BL(d1, T, d2, t2)
        self.tabl = TABL(d2, t2, 3, 1)
        self.dropout = nn.Dropout(drop)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, D, T)
        x = self.dropout(self.bl1(x))
        x = self.dropout(self.bl2(x))
        if not return_attn:
            x = self.tabl(x)  # (B, 3, 1)
            return x.squeeze(-1)  # (B, 3)
        x, a = self.tabl(x, return_attn=True)
        attn = {"temporal": a, "lam": self.tabl.lam.detach().clamp(0.0, 1.0)}
        return x.squeeze(-1), attn


class CTABL(nn.Module):
    """CTABL classifier over a (D × T) LOB matrix."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.body = CTABLBody(config)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trend logits; with ``return_attn`` also the TABL readouts.

        The attention dict holds ``"temporal"`` ``(B, d2, t2)`` — the TABL
        softmax over the (halved) temporal axis per output feature — and
        ``"lam"``, the scalar the layer learned for mixing attention against the
        raw bilinear path (0 = attention unused, 1 = attention only).
        """
        x = x.squeeze(1).transpose(1, 2)  # (B, 1, T, F) -> (B, F=D, T)
        return self.body(x, return_attn=return_attn)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
