"""JointDiffusion: time-conditioned U-Net that denoises and classifies jointly.

Input : ``x_t (B, 1, T_past, F)`` noisy window + integer timestep ``t (B,)``.
Output: ``(eps_hat (B,1,T,F), logits (B,3))``.

At inference call ``predict(batch, device)`` which evaluates the clean window
at ``t = 0`` (no noise) → ``logits (B, 3)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class TimeDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = x + self.temb(temb).unsqueeze(-1).unsqueeze(-1)
        return self.act(self.norm2(self.conv2(x)))


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = TimeDoubleConv(in_ch, out_ch, temb_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x), temb)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = TimeDoubleConv(out_ch + skip_ch, out_ch, temb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, temb)


class JointDiffusion(nn.Module):
    """Time-conditioned U-Net trained jointly to denoise and classify trend."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        base = config.get("jd_base_channels", 32)
        depth = config.get("jd_depth", 2)
        temb_dim = config.get("jd_time_emb", 128)
        self.temb_dim = temb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        chans = [base * (2**i) for i in range(depth + 1)]
        self.stem = TimeDoubleConv(1, base, temb_dim)
        self.downs = nn.ModuleList(
            Down(chans[i], chans[i + 1], temb_dim) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(chans[i + 1], chans[i], chans[i], temb_dim)
            for i in reversed(range(depth))
        )
        self.out_conv = nn.Conv2d(base, 1, 1)
        bottleneck = chans[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
            nn.Dropout(config.get("jd_dropout", 0.1)),
            nn.Linear(bottleneck, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        x = self.stem(x_t, temb)
        skips = [x]
        for down in self.downs:
            x = down(x, temb)
            skips.append(x)
        logits = self.classifier(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, temb)
        return self.out_conv(x), logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch["x"].to(device).float()
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        _, logits = self(x, t)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
