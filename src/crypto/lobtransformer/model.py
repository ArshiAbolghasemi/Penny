"""LOBTransformer: transformer classifier over a pre-normalised LOB feature window.

Input : ``x`` ``(B, 1, T_past, F)`` — same format as DeepLOB / JointDiffusion.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

``F`` is set by ``config["n_features"]`` (written by train.py after building the
dataset) and equals ``2n+11`` in OFI mode or ``4n+11`` in LOB mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LOBTransformer(nn.Module):
    """Transformer classifier over a windowed LOB feature matrix."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.t_past = config["T_past"]
        n_features = config["n_features"]
        d = config.get("lobt_hidden", 256)
        heads = config.get("lobt_heads", 8)
        layers = config.get("lobt_layers", 4)

        self.input_proj = nn.Linear(n_features, d)
        self.pos = nn.Parameter(torch.randn(1, self.t_past, d) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=d * 2,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) → (B, T, F)
        x = x.squeeze(1)
        h = self.input_proj(x) + self.pos  # (B, T, d)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))  # (B, 3)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Return class logits ``(B, 3)`` from a standard LOB batch."""
        return self(batch["x"].to(device).float())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
