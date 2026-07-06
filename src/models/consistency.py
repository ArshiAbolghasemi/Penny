"""Consistency-model helpers (Song et al. 2023; improved: Song & Dhariwal 2023).

EDM/Karras parameterisation for consistency *training* — no teacher, no
distillation.  A consistency function ``f_theta(x, sigma)`` maps any noised point
on a diffusion trajectory back to the clean signal, with the boundary condition
``f_theta(x, sigma_min) = x`` enforced structurally by the c_skip/c_out
preconditioning (so it holds for any network weights).

Consistency Training (CT) minimises the self-consistency loss

    L = lambda(sigma_n) * d( f_theta (x + sigma_{n+1} z, sigma_{n+1}),
                             f_theta-(x + sigma_n     z, sigma_n) )

over adjacent Karras noise levels sigma_n < sigma_{n+1} sharing the same noise z.
``theta-`` is a stop-gradient copy of theta (improved CT uses the online weights
directly, i.e. no EMA target).  d is the Pseudo-Huber metric and lambda = 1/Δsigma.
"""

from __future__ import annotations

import math

import torch


def karras_sigmas(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """N ascending Karras noise levels, ``sigmas[0]=sigma_min .. sigmas[-1]=sigma_max``."""
    i = torch.arange(n, device=device, dtype=dtype)
    inv = 1.0 / rho
    lo = sigma_min**inv
    hi = sigma_max**inv
    return (lo + i / (n - 1) * (hi - lo)) ** rho


def precond(
    sigma: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    kappa: torch.Tensor | float | None = None,
):
    """(t-)EDM preconditioning coefficients (c_skip, c_out, c_in, c_noise).

    Gaussian EDM (Karras 2022) by default.  When a per-sample Student-t scale
    ``kappa`` is supplied (t-EDM, Pandey et al. 2025), the noise variance at level
    ``sigma`` is ``kappa·sigma^2`` rather than ``sigma^2``, so the variance-carrying
    coefficients pick up ``kappa`` exactly as if ``sigma -> sigma·sqrt(kappa)``:

        c_in   = 1 / sqrt(kappa·sigma^2 + sigma_data^2)
        c_skip = sigma_data^2 / (kappa·(sigma - sigma_min)^2 + sigma_data^2)
        c_out  = sigma_data·sqrt(kappa)·(sigma - sigma_min)
                 / sqrt(kappa·sigma^2 + sigma_data^2)

    ``c_noise`` still embeds the *schedule* level ``sigma`` (where on the trajectory
    we are), not the latent scale.  At ``sigma = sigma_min`` we get ``c_skip = 1`` and
    ``c_out = 0`` for any ``kappa``, so the consistency boundary ``f(x, sigma_min) = x``
    holds regardless of the tail draw.  ``kappa = None`` (or ``1``) recovers Gaussian
    EDM exactly (the ``nu -> inf`` limit).
    """
    if kappa is None:
        kappa = 1.0
    kroot = kappa**0.5
    den = torch.sqrt(kappa * sigma**2 + sigma_data**2)
    c_skip = sigma_data**2 / (kappa * (sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = sigma_data * kroot * (sigma - sigma_min) / den
    c_in = 1.0 / den
    c_noise = 0.25 * torch.log(sigma.clamp_min(1e-20))
    return c_skip, c_out, c_in, c_noise


def sample_kappa(
    nu: float | None,
    shape: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-sample Student-t scale ``kappa ~ Inverse-Gamma(nu/2, nu/2)`` (t-EDM).

    Drawn via ``kappa = 1 / g`` with ``g ~ Gamma(nu/2, rate=nu/2)`` so that
    ``sqrt(kappa)·eps`` (``eps ~ N(0, I)``) is marginally multivariate Student-t with
    ``nu`` degrees of freedom.  ``E[kappa] = nu / (nu - 2)`` for ``nu > 2``; as
    ``nu -> inf`` the mixing collapses to ``kappa = 1`` (Gaussian).  A very large /
    missing ``nu`` short-circuits to ones so the Gaussian limit is exact and cheap.
    """
    if nu is None or nu >= 1e6:
        return torch.ones(shape, device=device, dtype=dtype)
    half = 0.5 * float(nu)
    g = torch.distributions.Gamma(half, half).sample(torch.Size(shape))
    return (1.0 / g.clamp_min(1e-12)).to(device=device, dtype=dtype)


def interval_weights(
    sigmas: torch.Tensor, p_mean: float = -1.1, p_std: float = 2.0
) -> torch.Tensor:
    """Improved-CT lognormal probabilities over the N-1 adjacent sigma intervals."""
    lo, hi = sigmas[:-1], sigmas[1:]

    def cdf(s: torch.Tensor) -> torch.Tensor:
        return torch.erf((torch.log(s) - p_mean) / (math.sqrt(2.0) * p_std))

    w = (cdf(hi) - cdf(lo)).clamp_min(1e-12)
    return w / w.sum()


def pseudo_huber_const(numel: int, k: float = 0.00054) -> float:
    """Improved-CT Pseudo-Huber constant c = k·sqrt(d) for data dimensionality ``numel``."""
    return k * math.sqrt(numel)


def discretization_n(step: int, total_steps: int, s0: int = 10, s1: int = 1280) -> int:
    """Improved-CT annealing schedule for the number of discretization steps N(k).

    Song & Dhariwal (2023) grow the number of Karras noise levels during training
    via a doubling schedule: ``N`` starts at ``s0+1`` and doubles every ``K'``
    steps up to ``s1+1``, where ``K' = total_steps / (log2(s1/s0) + 1)``.  More
    levels ⇒ smaller adjacent-sigma gaps ⇒ lower consistency-matching bias late in
    training.  Returns the number of sigma *boundaries* (so there are ``N-1``
    adjacent intervals to sample from).
    """
    k_prime = max(1, math.floor(total_steps / (math.log2(s1 / s0) + 1.0)))
    return min(s0 * 2 ** (step // k_prime), s1) + 1
