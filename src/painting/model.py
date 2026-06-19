"""Painting model: pretrained 2D-UNet inpainting backbone + trend head.

Loads a pretrained ``UNet2DModel`` from HuggingFace Diffusers and adapts its
input/output channels (3→5 in, 3→2 out) via ``ignore_mismatched_sizes=True``.
Fine-tuned jointly with a DeepLOB trend head using masked diffusion loss and
timestep-weighted cross-entropy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DModel
from loguru import logger


class TrendHead(nn.Module):
    """Linear map from the scalar trend ratio ``l`` to 3 class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 3)

    def forward(self, trend_value: torch.Tensor) -> torch.Tensor:
        return self.fc(trend_value.view(-1, 1))


def reconstruct_future_mid(bb_ch0, ba_ch0, config, mid_ref, gamma) -> torch.Tensor:
    """Future mid ``(B, T_future)`` from *denormalized* best bid/ask channel-0 series."""
    if config["feature_mode"] == "lob":
        return (bb_ch0 + ba_ch0) / 2.0  # raw prices
    net_ofi = ba_ch0 - bb_ch0  # aofi - bofi
    return mid_ref.view(-1, 1) + torch.cumsum(net_ofi, dim=1) * gamma


class UNetInpaintModel(nn.Module):
    """Pretrained 2D-UNet inpainting backbone on the padded square image."""

    family = "inpaint"

    def __init__(self, config, normalizer, level_starts):
        super().__init__()
        self.config = config
        self.norm = normalizer
        self.level_starts = np.asarray(level_starts)
        model_id = config["pretrained_model_id"]
        logger.info("loading pretrained UNet '{}' (adapting 3->5 in, 3->2 out)", model_id)
        self.unet = UNet2DModel.from_pretrained(
            model_id,
            in_channels=5,
            out_channels=2,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )

    def predict_noise(self, x_t, t, history, mask):
        return self.unet(torch.cat([x_t, history, mask], dim=1), t).sample

    def future_mid(self, x0_hat, batch, gamma):
        n = self.config["n_levels"]
        t_past, t_total = self.config["T_past"], self.config["T_total"]
        ls = self.level_starts
        fut = slice(t_past, t_total)
        bb = x0_hat[:, 0, ls[n - 1] : ls[n], fut].mean(dim=1)
        ba = x0_hat[:, 0, ls[n] : ls[n + 1], fut].mean(dim=1)
        bb = bb * float(self.norm.std[n - 1, 0]) + float(self.norm.mean[n - 1, 0])
        ba = ba * float(self.norm.std[n, 0]) + float(self.norm.mean[n, 0])
        mid_ref = batch["mid_ref"].to(x0_hat.device).float()
        return reconstruct_future_mid(bb, ba, self.config, mid_ref, gamma)


def build_model(config, normalizer, level_starts) -> nn.Module:
    return UNetInpaintModel(config, normalizer, level_starts)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
