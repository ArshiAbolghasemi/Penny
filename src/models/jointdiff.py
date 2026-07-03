"""JointDiffusion: time-conditioned U-Net that denoises and classifies jointly.

Input : ``x_t (B, 1, T_past, F)`` noisy window + integer timestep ``t (B,)``.
Output: ``(eps_hat (B,1,T,F), logits (B,3))``.

At inference call ``predict(batch, device)`` which evaluates the clean window
at ``t = 0`` (no noise) → ``logits (B, 3)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.consistency import precond
from models.modules import (
    AttentionPool,
    BiN,
    Down,
    TimeDoubleConv,
    Up,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


class JointDiffusion(nn.Module):
    """Time-conditioned U-Net trained jointly to denoise and classify trend."""

    family = "joint_diffusion"

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
        pool_heads = config.get("jd_pool_heads", 4)
        self.pool = AttentionPool(bottleneck, heads=pool_heads)
        self.classifier = nn.Linear(bottleneck, 3)

        # Consistency-model (EDM) parameters — only used when trained as a
        # consistency model (cm_enabled); forward() is left unchanged so DDPM
        # trainers keep working.
        self.sigma_data = float(config.get("cm_sigma_data", 0.5))
        self.sigma_min = float(config.get("cm_sigma_min", 0.002))
        self.consistency = bool(config.get("cm_enabled", False))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        # t carries the timestep (DDPM) or c_noise (consistency); both are handled
        # identically by the sinusoidal embedding, which accepts float inputs.
        if self.bin is not None:
            x_t = self.bin(x_t.squeeze(1)).unsqueeze(1)
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        x = self.stem(x_t, temb)
        skips = [x]
        for down in self.downs:
            x = down(x, temb)
            skips.append(x)
        tokens = skips[-1].flatten(2).transpose(1, 2)  # (B, H*W, C)
        logits = self.classifier(self.pool(tokens))
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, temb)
        return self.out_conv(x), logits

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor):
        """Consistency function f_theta(x, sigma) -> (x0_hat, logits). sigma: (B,)."""
        c_skip, c_out, c_in, c_noise = precond(sigma, self.sigma_data, self.sigma_min)
        v = (-1,) + (1,) * (x.dim() - 1)  # (B,1,1,1)
        raw, logits = self(c_in.view(v) * x, c_noise)
        x0 = c_skip.view(v) * x + c_out.view(v) * raw
        return x0, logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch["x"].to(device).float()
        b = x.shape[0]
        if self.consistency:
            sigma = torch.full((b,), self.sigma_min, device=device)
            _, logits = self.denoise(x, sigma)
        else:
            t = torch.zeros(b, dtype=torch.long, device=device)
            _, logits = self(x, t)
        return logits
