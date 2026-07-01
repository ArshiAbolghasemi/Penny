"""JointDiffusionDF: JointDiffusion trained with Diffusion Forcing.

Diffusion Forcing (Chen et al., 2024, "Diffusion Forcing: Next-token Prediction
Meets Full-Sequence Diffusion") assigns each *sequence position its own
independent noise level* rather than one shared scalar timestep for the whole
window.  For an LOB window ``(B, 1, T, F)`` the sequence axis is time, so the
timestep becomes ``t (B, T)`` — every one of the T timesteps is noised
independently and the network denoises them jointly, conditioned on the full
per-timestep noise-level vector.

This is the same U-Net + joint-classification design as JointDiffusion; the only
difference is the conditioning:

  * ``t`` is ``(B, T)`` instead of ``(B,)``.
  * the timestep embedding is a per-timestep sequence ``(B, T, temb_dim)`` that is
    added along the time axis of each feature map, ``F.interpolate``-d down to the
    pooled time resolution at every U-Net level.

Input : ``x_t (B, 1, T, F)`` noisy window + per-timestep ``t (B, T)``.
Output: ``(eps_hat (B, 1, T, F), logits (B, 3))``.

At inference call ``predict(batch, device)`` which evaluates the clean window at
``t = 0`` everywhere → ``logits (B, 3)`` (identical contract to JointDiffusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jointdiff import BiN, _groups, sinusoidal_embedding


class SeqTimeDoubleConv(nn.Module):
    """Two 3×3 convs with a per-timestep additive time embedding.

    ``temb_seq`` is ``(B, T0, temb_dim)`` at the original sequence length T0; it is
    projected to ``out_ch`` and resampled along time to the current feature-map
    resolution before being added (broadcast over the feature axis).
    """

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb_seq: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        emb = self.temb(temb_seq).transpose(1, 2)  # (B, out_ch, T0)
        if emb.shape[-1] != x.shape[2]:
            emb = F.interpolate(
                emb, size=x.shape[2], mode="linear", align_corners=False
            )
        x = x + emb.unsqueeze(-1)  # (B, out_ch, T', 1) broadcast over F'
        return self.act(self.norm2(self.conv2(x)))


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = SeqTimeDoubleConv(in_ch, out_ch, temb_dim)

    def forward(self, x: torch.Tensor, temb_seq: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x), temb_seq)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = SeqTimeDoubleConv(out_ch + skip_ch, out_ch, temb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb_seq: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, temb_seq)


class JointDiffusionDF(nn.Module):
    """JointDiffusion with per-timestep (Diffusion Forcing) noise conditioning."""

    family = "joint_diffusion"  # same predict/forward contract as JointDiffusion

    def __init__(self, config: dict) -> None:
        super().__init__()
        base = config.get("jd_base_channels", 32)
        depth = config.get("jd_depth", 2)
        temb_dim = config.get("jd_time_emb", 128)
        self.temb_dim = temb_dim

        T = config.get("T_past")
        F_dim = config.get("n_features")
        self.bin = BiN(T, F_dim) if (T and F_dim) else None

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        chans = [base * (2**i) for i in range(depth + 1)]
        self.stem = SeqTimeDoubleConv(1, base, temb_dim)
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
            nn.GELU(),
            nn.Dropout(config.get("jd_dropout", 0.1)),
            nn.Linear(bottleneck, 3),
        )

        # DDPM ᾱ schedule for per-timestep forward noising (see add_noise).
        t_max = config.get("T_max", 1000)
        betas = torch.linspace(
            config.get("beta_start", 1e-4),
            config.get("beta_end", 0.02),
            t_max,
            dtype=torch.float64,
        )
        ab = torch.cumprod(1.0 - betas, dim=0).float()
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1mab", (1.0 - ab).sqrt())

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Per-timestep forward diffusion. x0/noise: (B,1,T,F); t: (B,T) longs."""
        sa = self.sqrt_ab[t][:, None, :, None]  # (B,1,T,1)
        sb = self.sqrt_1mab[t][:, None, :, None]
        return sa * x0 + sb * noise

    def _temb_seq(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T) → (B, T, temb_dim)
        b, T = t.shape
        emb = sinusoidal_embedding(t.reshape(-1), self.temb_dim).view(b, T, -1)
        return self.time_mlp(emb)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        if self.bin is not None:
            x_t = self.bin(x_t.squeeze(1)).unsqueeze(1)
        temb_seq = self._temb_seq(t)
        x = self.stem(x_t, temb_seq)
        skips = [x]
        for down in self.downs:
            x = down(x, temb_seq)
            skips.append(x)
        logits = self.classifier(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, temb_seq)
        return self.out_conv(x), logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch["x"].to(device).float()
        t = torch.zeros(x.shape[0], x.shape[2], dtype=torch.long, device=device)
        _, logits = self(x, t)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
