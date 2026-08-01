"""Plain Gaussian (Brownian) forward process with the **closed-form** score.

The Gaussian reference point for the two non-Gaussian corruption kernels in this
repo — the finite-variance jump-diffusion of :mod:`levy.diffusion` (used by
:class:`~models.jumpgatelob.JumpGateLOB`) and the infinite-variance α-stable law of
:mod:`models.alphastable` (used by :class:`~models.alphastablelob.AlphaStableLOB`).
Both of those are **Gaussian scale mixtures** ``u = √W·ξ`` whose score has no closed
form and must be tabulated by Monte-Carlo; here ``W = σ_t²`` is *deterministic*, the
mixture collapses, and the score is exact:

    x_t = a_t·x₀ + σ_t·ε ,  ε ~ N(0, I)      u = x_t − a_t·x₀ = σ_t·ε
    ∇_{x_t} log q(x_t | x₀) = −u / σ_t²

So there is no score table to build, no MC error, and no jump/tail hyperparameters —
which is exactly the point of the ablation: everything else (schedule, sampler shape,
DSM weighting) is identical to the Lévy path, so a Gaussian-vs-Lévy difference in
downstream trend accuracy is attributable to the *corruption law* alone.

The API deliberately mirrors :class:`levy.diffusion.forward.ForwardProcess`
(``add_noise`` / ``score_target``, plus the DSM weight and low-``t`` helpers that
``crypto.train_jumpgatelob`` keeps module-level) so the two trainers read
line-for-line the same.  ``models/ddpm.py`` covers the *other* Gaussian use case —
ε-prediction for JointDiT — and has no score API; this class is not a replacement
for it.

Holds schedule tensors only (no parameters); passed to the trainer alongside the
model, mirroring ``ForwardProcess`` / ``AlphaStableDiffusion``.
"""

from __future__ import annotations

import math

import torch


def make_vp_schedule(
    num_timesteps: int, beta_start: float, beta_end: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance-preserving (DDPM) ``(a_t, σ_t)`` under a linear-β schedule.

    Identical construction to ``levy.diffusion.schedules.make_vp_schedule`` — kept
    local so the Gaussian reference carries no dependency on the Lévy package.
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return alpha_bar.sqrt().float(), (1.0 - alpha_bar).sqrt().float()


def make_ve_schedule(
    num_timesteps: int, sigma_min: float, sigma_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance-exploding ``(a_t = 1, σ_t)`` — geometric grid in ``log σ``."""
    sigma = torch.exp(
        torch.linspace(math.log(sigma_min), math.log(sigma_max), num_timesteps)
    ).float()
    return torch.ones(num_timesteps, dtype=torch.float32), sigma


class GaussianDiffusion:
    """Gaussian forward process ``q(x_t|x₀) = N(a_t x₀, σ_t² I)`` + exact score."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "vp",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        sigma_min: float = 1e-2,
        sigma_max: float = 50.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if schedule == "vp":
            a, sigma = make_vp_schedule(num_timesteps, beta_start, beta_end)
        elif schedule == "ve":
            a, sigma = make_ve_schedule(num_timesteps, sigma_min, sigma_max)
        else:
            raise ValueError(f"unknown schedule '{schedule}' (expected 'vp' or 've')")
        self.kind = schedule
        self.num_timesteps = num_timesteps
        self.device = torch.device(device)
        self.a = a.to(self.device)  # signal coefficient a_t   (T,)
        self.sigma = sigma.to(self.device)  # noise std σ_t     (T,)

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        self.device = torch.device(device)
        self.a = self.a.to(self.device)
        self.sigma = self.sigma.to(self.device)
        return self

    def gather(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(a_t, σ_t)`` for a batch of integer timesteps ``t (B,)``."""
        return self.a.to(t.device)[t], self.sigma.to(t.device)[t]

    @staticmethod
    def _bcast(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return v.reshape((-1,) + (1,) * (x.dim() - 1))

    # ---- forward process ----------------------------------------------------
    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_t, u)`` where ``u = x_t − a_t·x₀ = σ_t·ε`` is the noise."""
        a_t, sigma_t = self.gather(t)
        u = self._bcast(sigma_t, x0) * torch.randn_like(x0)
        return self._bcast(a_t, x0) * x0 + u, u

    def score_target(
        self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Exact DSM target ``∇_{x_t} log q(x_t|x₀) = −u/σ_t²``."""
        a_t, sigma_t = self.gather(t)
        u = x_t - self._bcast(a_t, x0) * x0
        return -u / self._bcast(sigma_t, x0).pow(2).clamp_min(1e-12)

    # ---- training helpers ---------------------------------------------------
    def mean_W(self, t: torch.Tensor) -> torch.Tensor:
        """Total noise variance ``E[W_t] = σ_t²`` — the per-sample DSM weight.

        The score magnitude is ``~1/σ_t²``, so weighting the squared error by ``σ_t²``
        keeps the target ``O(1)`` at every timestep.  Same role (and same formula
        minus the jump term) as ``crypto.train_jumpgatelob._mean_W``; with this weight
        the score loss is algebraically ``‖σ_t·ŝ + ε‖²``, i.e. ε-prediction MSE.
        """
        return self.gather(t)[1] ** 2

    def low_t_indices(self, device: torch.device | None = None) -> torch.Tensor:
        """Timesteps where signal dominates noise (SNR ≥ 1) — the levels the robust
        classification pass draws from.  VP: ``ᾱ_t ≥ 0.5``; VE: ``σ_t < 1`` (features
        are z-scored).  Matches ``crypto.train_jumpgatelob._low_t_indices``."""
        device = device or self.device
        if self.kind == "vp":
            mask = (self.a.to(device) ** 2) >= 0.5
        else:
            mask = self.sigma.to(device) < 1.0
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        return idx if len(idx) > 0 else torch.zeros(1, dtype=torch.long, device=device)
