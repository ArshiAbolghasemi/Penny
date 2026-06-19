"""CSDI classifier: a 2D convolutional U-Net over the multivariate past window
that directly predicts price direction (down / stationary / up).

The past window is treated as a 2-channel image with the LOB feature rows on the
height axis and time on the width axis.  A U-Net contracting/expanding path with
skip connections extracts spatial features; the final full-resolution feature map
is globally pooled and mapped to 3 class logits.

Input : ``(B, 2, R, T_past)`` normalised LOB feature tensor.
Output: ``(B, 3)`` class logits.
Loss  : CrossEntropy(logits, label).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm + GELU (U-Net block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale by 2 (max-pool) then a double convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscale to the skip-connection size, concatenate, then a double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CSDIClassifier(nn.Module):
    """2D conv U-Net over the multivariate past window → 3-class direction logits."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        base = config.get("csdi_channels", 64)
        depth = config.get("csdi_depth", 3)

        chans = [base * (2**i) for i in range(depth + 1)]  # encoder widths
        self.stem = DoubleConv(2, base)
        self.downs = nn.ModuleList(Down(chans[i], chans[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            Up(chans[i + 1], chans[i], chans[i]) for i in reversed(range(depth))
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base, base),
            nn.GELU(),
            nn.Linear(base, 3),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        x = self.stem(past)  # (B, base, R, T_past)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.head(x)  # (B, 3) logits

    def predict(self, batch, device) -> torch.Tensor:
        """Return class logits ``(B, 3)``."""
        return self(batch["past"].to(device))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
