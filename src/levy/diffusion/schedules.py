"""Noise schedules for the forward process.

Two standard, minimal schedules — no EDM preconditioning:

* **VP** (variance-preserving, DDPM): ``x_t = a_t x_0 + b_t eps`` with
  ``a_t = sqrt(alpha_bar_t)``, ``b_t = sqrt(1 - alpha_bar_t)`` under a linear beta
  schedule.  The *marginal noise scale* (std of the additive perturbation) is
  ``sigma_t = b_t``.
* **VE** (variance-exploding): ``x_t = x_0 + sigma_t eps`` with ``sigma_t`` a
  geometric grid from ``sigma_min`` to ``sigma_max`` (``a_t = 1``).

For the Lévy process the additive perturbation is *not* Gaussian, but the schedule
still supplies the contraction ``a_t`` and the diffusion (Brownian) scale
``sigma_t`` that seeds the jump-diffusion kernel; see :mod:`levy.diffusion.forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    """Per-timestep schedule tensors, all shape ``(T,)`` indexed by integer t.

    ``a`` is the signal contraction (VP < 1, VE == 1); ``sigma`` is the Brownian
    (Gaussian) noise std at each timestep.
    """

    kind: str
    a: torch.Tensor  # signal coefficient a_t
    sigma: torch.Tensor  # Brownian noise std sigma_t
    num_timesteps: int

    def to(self, device: torch.device | str) -> "NoiseSchedule":
        return NoiseSchedule(
            self.kind, self.a.to(device), self.sigma.to(device), self.num_timesteps
        )

    def gather(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a_t, sigma_t)`` for a batch of integer timesteps ``t (B,)``."""
        return self.a.to(t.device)[t], self.sigma.to(t.device)[t]


def make_vp_schedule(
    num_timesteps: int, beta_start: float, beta_end: float
) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    a = alpha_bar.sqrt().float()
    sigma = (1.0 - alpha_bar).sqrt().float()
    return NoiseSchedule("vp", a, sigma, num_timesteps)


def make_ve_schedule(
    num_timesteps: int, sigma_min: float, sigma_max: float
) -> NoiseSchedule:
    # geometric interpolation in log-sigma
    sigma = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(sigma_min)),
            torch.log(torch.tensor(sigma_max)),
            num_timesteps,
        )
    ).float()
    a = torch.ones(num_timesteps, dtype=torch.float32)
    return NoiseSchedule("ve", a, sigma, num_timesteps)


def make_schedule(cfg) -> NoiseSchedule:
    """Build a schedule from a :class:`levy.config.DiffusionConfig`."""
    if cfg.schedule == "vp":
        return make_vp_schedule(cfg.num_timesteps, cfg.beta_start, cfg.beta_end)
    if cfg.schedule == "ve":
        return make_ve_schedule(cfg.num_timesteps, cfg.sigma_min, cfg.sigma_max)
    raise ValueError(f"unknown schedule '{cfg.schedule}' (expected 'vp' or 've')")
