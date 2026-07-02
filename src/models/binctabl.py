"""BINCTABL — Bilinear-normalised CTABL for LOB data.

Based on: Tran et al., "Data Normalization for Bilinear Structures in
High-Frequency Financial Time-series" (ICPR 2020) — prepends the Bilinear
Normalization (BiN) layer to CTABL.  BiN adaptively normalises the input along
both the temporal and feature modes (a learnable convex mix of the two z-scores)
before the bilinear trunk.

Input : ``(B, 1, T_past, n_features)``.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Config keys — the BiN layer adds no hyperparameters; the trunk reuses the CTABL
keys (``ctabl_d1``, ``ctabl_d2``, ``ctabl_t2``, ``ctabl_dropout``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.ctabl import CTABLBody
from models.modules import BiN, count_parameters as count_parameters  # re-export


class BINCTABL(nn.Module):
    """BiN → CTABL classifier over a (T × F) LOB window."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.bin = BiN(config["T_past"], config["n_features"])
        self.body = CTABLBody(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)  # (B, T, F)
        x = self.bin(x)  # (B, T, F) bilinear-normalised
        x = x.transpose(1, 2)  # (B, F=D, T)
        return self.body(x)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
