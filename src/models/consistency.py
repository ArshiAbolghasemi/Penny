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


def precond(sigma: torch.Tensor, sigma_data: float, sigma_min: float):
    """EDM preconditioning coefficients (c_skip, c_out, c_in, c_noise) for ``sigma``."""
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = sigma_data * (sigma - sigma_min) / torch.sqrt(sigma**2 + sigma_data**2)
    c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)
    c_noise = 0.25 * torch.log(sigma.clamp_min(1e-20))
    return c_skip, c_out, c_in, c_noise


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
