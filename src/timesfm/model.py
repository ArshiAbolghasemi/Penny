"""TimesFM classifier: transformer over past mid-price returns (+ net OFI in OFI
mode) that directly predicts price direction (down / stationary / up).

Input : ``past_mid`` ``(B, T_past)`` and optionally ``past_ofi`` ``(B, T_past)``.
Output: ``(B, 3)`` class logits.
Loss  : CrossEntropy(logits, label).

The ``timesfm_pretrained`` config flag and TimesFM package are no longer used for
inference; the model is a self-contained transformer classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimesFMClassifier(nn.Module):
    """Transformer classifier over window-relative mid returns (+ optional OFI)."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.t_past = config["T_past"]
        self.use_ofi = config.get("feature_mode", "ofi") == "ofi"
        n_features = 2 if self.use_ofi else 1
        d = config.get("timesfm_hidden", 256)
        heads = config.get("timesfm_heads", 8)
        layers = config.get("timesfm_layers", 4)

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

    def forward(
        self, past_mid: torch.Tensor, past_ofi: torch.Tensor | None = None
    ) -> torch.Tensor:
        boundary = past_mid[:, -1].clamp(min=1e-8)
        past_returns = past_mid / boundary.view(-1, 1) - 1.0  # (B, T_past)
        if self.use_ofi and past_ofi is not None:
            features = torch.stack([past_returns, past_ofi], dim=-1)  # (B, T_past, 2)
        else:
            features = past_returns.unsqueeze(-1)  # (B, T_past, 1)
        h = self.input_proj(features) + self.pos  # (B, T_past, d)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))  # (B, 3) logits

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Return class logits ``(B, 3)``."""
        past_mid = batch["past_mid"].to(device).float()
        past_ofi = batch.get("past_ofi")
        if past_ofi is not None:
            past_ofi = past_ofi.to(device).float()
        return self(past_mid, past_ofi)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
