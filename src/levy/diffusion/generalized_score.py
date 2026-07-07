"""Generalized (non-Gaussian) score for a finite-activity jump-diffusion kernel.

Following Baule (2025, arXiv:2503.06558): denoising *score* matching still uses a
plain MSE, but the regression target is the score of the **actual** forward kernel,
which here is not Gaussian.  We exploit a subordinated-Gaussian ("generalized
Laplace") jump law so the whole kernel is a **Gaussian scale mixture**, for which
the isotropic score collapses to a cheap 1-D table.

Kernel.  The additive perturbation at timestep ``t`` is

    u = x_t - a_t x_0 = sqrt(W) * xi,   xi ~ N(0, I_d),

    W = sigma_t^2  +  sum_{k=1}^{N} S_k,   N ~ Poisson(Lambda_t),  S_k ~ Gamma(shape, scale)

i.e. a Brownian variance ``sigma_t^2`` plus a compound-Poisson sum of gamma
subordinators (each jump ``z_k = sqrt(S_k) xi_k`` is an isotropic generalized
Laplace vector, finite variance ``shape*scale`` per component, trivially sampled).

Score of a Gaussian scale mixture.  With ``phi(u) = E_W[ N(u; 0, W I_d) ]``,

    grad_u log phi(u) = -u * E[ 1/W | u ] = -u * h(r),   r = |u|,

where the posterior expectation weights each ``W`` by the Gaussian likelihood
``N(u;0,W I_d) ∝ W^{-d/2} exp(-r^2 / 2W)``.  ``h(r)`` depends on ``u`` only through
the radius ``r`` — the promised isotropic 1-D table — and is precomputed offline by
Monte-Carlo over ``W`` (see :func:`build_score_table`).

Sanity limit.  ``Lambda_t = 0`` ⇒ ``N=0`` ⇒ ``W = sigma_t^2`` deterministically ⇒
``h(r) = 1/sigma_t^2`` and the score is the ordinary Gaussian score ``-u/sigma_t^2``.
This is asserted in tests/levy/test_generalized_score.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class JumpParams:
    """Generalized-Laplace jump amplitude law (gamma subordinator)."""

    gamma_shape: float = 1.0
    gamma_scale: float = 1.0

    def mean_jump_var(self) -> float:
        """Per-component variance E[S] = shape*scale of one jump amplitude."""
        return self.gamma_shape * self.gamma_scale


def jump_intensity(
    schedule_len: int, jump_rate: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Expected jump count ``Lambda_t`` per timestep, shape ``(T,)``.

    TODO(user): design choice — here ``Lambda_t = jump_rate * (t+1)/T`` grows
    linearly so late (high-noise) steps see more jumps.  Alternatives: tie to the
    accumulated Brownian variance, or hold it constant.  Confirm the intended
    physical meaning of ``jump_rate``.
    """
    t = torch.arange(1, schedule_len + 1, device=device, dtype=torch.float32)
    return jump_rate * t / schedule_len


def sample_W(
    sigma_t: float,
    lambda_t: float,
    jump: JumpParams,
    n: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw ``n`` samples of the mixing variance ``W = sigma_t^2 + sum_k S_k``.

    ``sum_{k=1}^N Gamma(shape, scale) = Gamma(N*shape, scale)`` for ``N>0`` and ``0``
    for ``N=0``, so we sample ``N ~ Poisson`` then a single gamma per draw.
    """
    dev = torch.device(device)
    var = sigma_t * sigma_t
    W = torch.full((n,), var, device=dev, dtype=torch.float32)
    if lambda_t <= 0.0:
        return W
    rate = torch.full((n,), float(lambda_t), device=dev)
    N = torch.poisson(rate, generator=generator)  # (n,)
    mask = N > 0
    if mask.any():
        conc = N[mask] * jump.gamma_shape  # per-sample gamma shape
        # torch.distributions.Gamma has no generator arg; use _standard_gamma on the
        # concentration then scale by ``gamma_scale`` (mean = shape*scale).
        g = torch._standard_gamma(conc) * jump.gamma_scale
        W[mask] = W[mask] + g.to(torch.float32)
    return W


def sample_W_batched(
    sigma_b: torch.Tensor,  # (B,) Brownian std per sample
    lambda_b: torch.Tensor,  # (B,) expected jump count per sample
    jump: JumpParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Batched mixing variance ``W (B,)`` for a batch whose timesteps differ.

    Vectorized twin of :func:`sample_W` used by the forward process at train time.
    """
    W = sigma_b.to(torch.float32) ** 2
    N = torch.poisson(lambda_b.clamp_min(0.0), generator=generator)  # (B,)
    mask = N > 0
    if mask.any():
        conc = N[mask] * jump.gamma_shape
        g = torch._standard_gamma(conc) * jump.gamma_scale
        W = W.clone()
        W[mask] = W[mask] + g.to(torch.float32)
    return W


@dataclass
class GeneralizedScoreTable:
    """Precomputed isotropic score magnitude ``h(r) = E[1/W | r]``.

    Per-timestep radius grids adapt to where ``|u|`` actually lands (it concentrates
    near ``sqrt(E[W_t] * d)`` in high dimension).  Both tensors are ``(T, num_r)``.
    """

    r_grid: torch.Tensor  # (T, num_r) ascending, r_grid[:,0] == 0
    h: torch.Tensor  # (T, num_r) score magnitude
    d: int  # data dimensionality the table was built for

    def to(self, device: torch.device | str) -> "GeneralizedScoreTable":
        return GeneralizedScoreTable(self.r_grid.to(device), self.h.to(device), self.d)

    def h_at(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate ``h`` at radii ``r (B,)`` for integer timesteps ``t (B,)``."""
        rg = self.r_grid.to(r.device)[t]  # (B, num_r)
        hg = self.h.to(r.device)[t]  # (B, num_r)
        r = r.clamp_min(0.0).unsqueeze(1)  # (B,1)
        idx = torch.searchsorted(rg, r).clamp(1, rg.shape[1] - 1)  # (B,1)
        r0 = torch.gather(rg, 1, idx - 1)
        r1 = torch.gather(rg, 1, idx)
        h0 = torch.gather(hg, 1, idx - 1)
        h1 = torch.gather(hg, 1, idx)
        w = ((r - r0) / (r1 - r0).clamp_min(1e-12)).clamp(0.0, 1.0)
        return (h0 + w * (h1 - h0)).squeeze(1)  # (B,)

    def score(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generalized score ``grad_u log phi(u) = -u * h(|u|)``.

        ``u`` is ``(B, ...)``; the radius is taken over all non-batch dims.
        """
        b = u.shape[0]
        flat = u.reshape(b, -1)
        r = flat.norm(dim=1)  # (B,)
        h = self.h_at(r, t)  # (B,)
        return -u * h.reshape(b, *([1] * (u.dim() - 1)))


def build_score_table(
    d: int,
    sigma: torch.Tensor,  # (T,) Brownian std per timestep
    lambda_t: torch.Tensor,  # (T,) expected jump count per timestep
    jump: JumpParams,
    num_r: int = 512,
    mc_samples: int = 20000,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> GeneralizedScoreTable:
    """Precompute ``h(r) = E[1/W | r]`` on a per-timestep radius grid via MC over W.

    For each timestep we draw ``W_j`` and an independent ``|xi|^2_j ~ ChiSquare(d)``
    (sampled as ``Gamma(d/2, 2)`` — no d-dimensional draw needed), place a radius
    grid out to just past the largest sampled ``r_j = sqrt(W_j |xi|^2_j)``, and
    evaluate the log-weighted posterior mean of ``1/W`` with a stable log-sum-exp.
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    T = sigma.shape[0]
    r_grid = torch.empty(T, num_r, device=dev)
    h_tab = torch.empty(T, num_r, device=dev)
    half_d = 0.5 * d

    for ti in range(T):
        W = sample_W(float(sigma[ti]), float(lambda_t[ti]), jump, mc_samples, gen, dev)
        W = W.clamp_min(1e-12)
        # radius where samples land: r = sqrt(W * chi2_d),  chi2_d ~ Gamma(d/2, 2)
        chi2 = (
            torch._standard_gamma(torch.full((mc_samples,), half_d, device=dev)) * 2.0
        )
        r_samp = torch.sqrt(W * chi2)
        r_max = torch.quantile(r_samp, 0.9999).clamp_min(1e-6) * 1.02
        grid = torch.linspace(0.0, float(r_max), num_r, device=dev)  # (num_r,)

        # log weight g_j(r) ∝ W_j^{-d/2} exp(-r^2 / 2 W_j)   -> (num_r, mc)
        logW = torch.log(W)  # (mc,)
        inv2W = 0.5 / W  # (mc,)
        log_g = -half_d * logW[None, :] - (grid[:, None] ** 2) * inv2W[None, :]
        m = log_g.max(dim=1, keepdim=True).values  # (num_r,1)
        wts = torch.exp(log_g - m)  # (num_r, mc)
        num = (wts * (1.0 / W)[None, :]).sum(dim=1)  # E-weighted 1/W
        den = wts.sum(dim=1)
        h = num / den.clamp_min(1e-30)  # (num_r,)

        r_grid[ti] = grid
        h_tab[ti] = h

    return GeneralizedScoreTable(r_grid=r_grid, h=h_tab, d=d)


def gaussian_score(u: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    """Closed-form Gaussian score ``-u / sigma_t^2`` (the ``Lambda -> 0`` limit)."""
    v = (-1,) + (1,) * (u.dim() - 1)
    return -u / (sigma_t.reshape(v) ** 2)
