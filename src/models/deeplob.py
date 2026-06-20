"""DeepLOB — convolutional + inception + LSTM classifier for LOB data.

Based on: Zhang et al., "DeepLOB: Deep Convolutional Neural Networks for
Limit Order Books", IEEE Transactions on Signal Processing, 2019.

Input : ``(B, 1, T_past, n_features)`` — single-channel image.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Architecture
------------
1. Conv block  — (1×2) + (4×1) + (4×1) convolutions.
2. Inception   — three parallel temporal paths (k=1,3,5) concatenated.
3. AvgPool     — collapses feature axis to 1.
4. LSTM        — models long-range temporal dependencies.
5. Head        — dropout + linear → 3 logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, out_ch, kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_ch, out_ch, kernel_size=(4, 1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_ch, out_ch, kernel_size=(4, 1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01),
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.path1(x), self.path2(x), self.path3(x)], dim=1)


class DeepLOB(nn.Module):
    """DeepLOB: CNN → Inception → LSTM → 3-class logits."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        conv_f = config.get("deeplob_conv_filters", 32)
        inc_f = config.get("deeplob_inception_filters", 64)
        lstm_h = config.get("deeplob_lstm_hidden", 64)
        drop = config.get("deeplob_dropout", 0.1)

        self.conv_block = _ConvBlock(conv_f)
        self.inception = _InceptionBlock(conv_f, inc_f)
        self.feat_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.lstm = nn.LSTM(
            input_size=3 * inc_f, hidden_size=lstm_h, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(lstm_h, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)  # (B, conv_f, T-6, F+1)
        x = self.inception(x)  # (B, 3*inc_f, T-6, F+1)
        x = self.feat_pool(x).squeeze(-1)  # (B, 3*inc_f, T-6)
        x = x.permute(0, 2, 1)  # (B, T-6, 3*inc_f)
        _, (h, _) = self.lstm(x)
        return self.head(self.dropout(h.squeeze(0)))

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
