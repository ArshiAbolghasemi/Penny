"""LOBTransformer: transformer classifier over a LOB feature window.

Input : ``(B, 1, T_past, n_features)`` — same format as DeepLOB.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import (
    AttentionPool,
    count_parameters as count_parameters,  # re-export
)


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
        pool_heads = config.get("lobt_pool_heads", 4)

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
        self.pool = AttentionPool(d, heads=pool_heads)
        self.head = nn.Linear(d, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)  # (B, T, F)
        h = self.input_proj(x) + self.pos  # (B, T, d)
        h = self.encoder(h)
        return self.head(self.pool(h))  # (B, 3)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
