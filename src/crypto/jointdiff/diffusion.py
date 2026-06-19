"""Minimal Gaussian diffusion (linear-beta DDPM) for JointDiffusion.

Only the forward process + helpers needed for joint training are kept:
``q_sample`` (add noise) and ``x0_from_eps`` (Tweedie reconstruction).  An
optional ancestral ``sample`` is provided for generating synthetic windows, but
it is not used by the classification objective.
"""

from __future__ import annotations

import torch
from loguru import logger


class Diffusion:
    """Linear-beta Gaussian diffusion with schedule buffers on ``device``."""

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device = device
        self.T_max = int(config.get("T_max", 1000))
        betas = torch.linspace(
            config.get("beta_start", 1e-4),
            config.get("beta_end", 0.02),
            self.T_max,
            device=device,
        )
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        logger.info(
            "diffusion: T_max={} beta=[{:.5f},{:.5f}]",
            self.T_max,
            float(betas[0]),
            float(betas[-1]),
        )

    def _abar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars[t].view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._abar(t)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def x0_from_eps(
        self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        ab = self._abar(t)
        return (x_t - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)

    @torch.no_grad()
    def sample(self, model, shape, device=None) -> torch.Tensor:
        """Ancestral DDPM sampling of a fresh feature window (optional/generative)."""
        device = device or self.device
        x = torch.randn(shape, device=device)
        for ti in reversed(range(self.T_max)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            eps, _ = model(x, t)
            beta = self.betas[ti]
            alpha = self.alphas[ti]
            ab = self.alpha_bars[ti]
            mean = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if ti > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x
