"""DDPM diffusion with DDIM sampling and RePaint inpainting (painting only).

Linear-beta forward process and deterministic (eta=0) DDIM reverse steps.  For
inpainting (RePaint, Lugmayr et al. 2022) the known past region (mask 0) is
overwritten at every reverse step with a correctly-noised copy of the ground
truth, so only the future region (mask 1) is generated.
"""

from __future__ import annotations

import torch
from loguru import logger


class Diffusion:
    """Gaussian diffusion with a linear beta schedule (buffers on ``device``)."""

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device = device
        self.T_max = int(config["T_max"])
        self.ddim_steps = int(config["ddim_steps"])
        betas = torch.linspace(
            config["beta_start"], config["beta_end"], self.T_max, device=device
        )
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        logger.info(
            "diffusion: T_max={} beta=[{:.5f},{:.5f}] ddim_steps={}",
            self.T_max,
            config["beta_start"],
            config["beta_end"],
            self.ddim_steps,
        )

    def _abar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars[t].view(-1, 1, 1, 1)

    def q_sample(self, x0, t, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._abar(t)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def _x0_from_eps(self, x_t, eps, t) -> torch.Tensor:
        ab = self._abar(t)
        return (x_t - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)

    def ddim_step(self, x_t, eps_hat, t, t_prev) -> torch.Tensor:
        x0 = self._x0_from_eps(x_t, eps_hat, t)
        ab_prev = torch.where(
            t_prev.view(-1, 1, 1, 1) >= 0,
            self.alpha_bars[t_prev.clamp(min=0)].view(-1, 1, 1, 1),
            torch.ones_like(self._abar(t)),
        )
        return torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps_hat

    def repaint_step(self, x_t, eps_hat, t, t_prev, x0_known, mask) -> torch.Tensor:
        """DDIM step on the future (mask 1); re-paste the noised known past (mask 0)."""
        generated = self.ddim_step(x_t, eps_hat, t, t_prev)
        known = self.q_sample(x0_known, t_prev.clamp(min=0))
        return mask * generated + (1.0 - mask) * known

    @torch.no_grad()
    def sample(
        self, model, x0_known, mask, ddim_steps=None, device=None
    ) -> torch.Tensor:
        """RePaint-DDIM reverse diffusion via the wrapper's ``predict_noise``."""
        device = device or self.device
        steps = ddim_steps or self.ddim_steps
        x0_known = x0_known.to(device)
        mask = mask.to(device)
        history = x0_known * (1.0 - mask)
        x = torch.randn_like(x0_known)
        ts = torch.linspace(self.T_max - 1, 0, steps, device=device).long()
        for i in range(steps):
            t = ts[i].repeat(x.shape[0])
            t_prev = (
                ts[i + 1] if i + 1 < steps else torch.tensor(-1, device=device)
            ).repeat(x.shape[0])
            eps = model.predict_noise(x, t, history, mask)
            x = self.repaint_step(x, eps, t, t_prev, x0_known, mask)
        return x
