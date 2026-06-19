"""CSDI classifier: two-axis (feature + time) transformer that directly predicts
price direction (down / stationary / up) from the multivariate past window.

Input : ``(B, 2, R, T_past)`` normalised LOB feature tensor.
Output: ``(B, 3)`` class logits.
Loss  : CrossEntropy(logits, label).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _transformer(channels, heads, ff):
    layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=ff,
        activation="gelu",
        batch_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=1)


class TwoAxisBlock(nn.Module):
    """One residual block: transformer over time, then over features."""

    def __init__(self, channels, heads):
        super().__init__()
        self.time_layer = _transformer(channels, heads, channels * 2)
        self.feature_layer = _transformer(channels, heads, channels * 2)
        self.norm = nn.GroupNorm(1, channels)

    def _time(self, y, k, length):
        b, c, _ = y.shape
        y = y.reshape(b, c, k, length).permute(0, 2, 1, 3).reshape(b * k, c, length)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(b, k, c, length).permute(0, 2, 1, 3).reshape(b, c, k * length)

    def _feature(self, y, k, length):
        b, c, _ = y.shape
        y = y.reshape(b, c, k, length).permute(0, 3, 1, 2).reshape(b * length, c, k)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(b, length, c, k).permute(0, 2, 3, 1).reshape(b, c, k * length)

    def forward(self, x, k, length):
        b, c, _, _ = x.shape
        y = x.reshape(b, c, k * length)
        y = self._time(y, k, length)
        y = self._feature(y, k, length)
        y = self.norm(y)
        out = (x.reshape(b, c, k * length) + y) / np.sqrt(2.0)
        return out.reshape(b, c, k, length)


class CSDIClassifier(nn.Module):
    """Multivariate two-axis transformer → 3-class direction logits."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.r = 2 * config["n_levels"] + config["n_trade_rows"]
        self.k = 2 * self.r
        self.t_past = config["T_past"]
        c = config.get("csdi_channels", 64)
        heads = config.get("csdi_heads", 8)
        layers = config.get("csdi_layers", 4)
        self.input_projection = nn.Conv1d(1, c, 1)
        self.feature_embedding = nn.Embedding(self.k, c)
        self.blocks = nn.ModuleList(TwoAxisBlock(c, heads) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, 3))
        self.channels = c

    def _time_pe(self, length, c, device):
        pos = torch.arange(length, device=device).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, c, 2, device=device).float() * -(np.log(10000.0) / c)
        )
        pe = torch.zeros(length, c, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        b = past.shape[0]
        length = self.t_past
        value = past.reshape(b, self.k, length)
        x = self.input_projection(value.reshape(b, 1, self.k * length))
        x = x.reshape(b, self.channels, self.k, length)
        pe = self._time_pe(length, self.channels, past.device)
        x = x + pe.permute(1, 0).reshape(1, self.channels, 1, length)
        f_emb = self.feature_embedding(torch.arange(self.k, device=past.device))
        x = x + f_emb.permute(1, 0).reshape(1, self.channels, self.k, 1)
        for block in self.blocks:
            x = block(x, self.k, length)
        pooled = x.mean(dim=(2, 3))  # (B, C)
        return self.head(pooled)  # (B, 3) logits

    def predict(self, batch, device) -> torch.Tensor:
        """Return class logits ``(B, 3)``."""
        return self(batch["past"].to(device))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
