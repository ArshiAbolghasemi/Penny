"""α-stable (Lévy-stable) diffusion via subordinated Gaussian + tabulated score.

A symmetric α-stable random vector has **infinite variance** and **power-law tails**
``~|x|^{-α-1}`` for ``α ∈ (0, 2)`` (``α = 2`` is Gaussian), which matches the fat tails
of high-frequency LOB returns far better than a Gaussian or a finite-variance jump
kernel.  Its density/score have **no closed form** — the obstacle that makes α-stable
diffusion hard.

We sidestep that with the classical **subordinated-Gaussian** representation: a
symmetric α-stable is a Gaussian scale mixture

    u = √W · ξ ,   ξ ~ N(0, I_d) ,   W = σ_t² · A ,   A ~ PositiveStable(α/2)

where ``A`` is a positive (maximally-skewed) ``α/2``-stable subordinator, sampled by
Kanter's (1975) algorithm.  Marginalising over ``A`` makes ``u`` genuinely α-stable.
Because the kernel is a Gaussian scale mixture, its isotropic score collapses to

    ∇_u log q(u) = -u · h(|u|) ,   h(r) = E[ 1/W | |u| = r ]

a **1-D table** precomputed by Monte-Carlo over ``A`` (same construction as the
finite-variance Lévy path in :mod:`levy.diffusion.generalized_score`, but with an
α-stable subordinator instead of compound-Poisson gamma jumps).

Sanity limit: ``α → 2`` ⇒ ``A → 1`` (the ``α/2 → 1`` positive stable is degenerate at
1) ⇒ ``W = σ_t²`` ⇒ ``h(r) = 1/σ_t²`` and the score is the ordinary Gaussian score.

Numerical note: the subordinator ``A`` is heavy-tailed, so ``x_t`` has huge outliers.
The forward returns an EDM-style input scale ``c_in = 1/√(1 + W)`` that the trainer
applies to the network input so the trunk always sees an ``O(1)`` window; the score
*target* is unchanged.  ``A`` is clipped at a high quantile (``clip_q``) during both
table-build and sampling to keep the Monte-Carlo estimate and training stable
(a truncated-stable law — still far heavier-tailed than Gaussian).

Holds schedule tensors + the score table only (no parameters); passed to the trainer
alongside the model, mirroring ``ForwardProcess`` / ``ImprovedDiffusion``.
"""

from __future__ import annotations

import math

import torch


def cosine_alpha_bar(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine ``ᾱ_t`` schedule (Nichol & Dhariwal), ``(T,)`` float64, descending."""
    t = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
    f = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    abar = f / f[0]
    return abar[1:].clamp(1e-6, 0.9999)


def sample_pos_stable(
    beta: float, shape: tuple[int, ...], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Positive (one-sided) ``β``-stable draws, ``β ∈ (0, 1)``, via Kanter (1975).

    ``β ≥ 1`` short-circuits to ones (the degenerate limit = Gaussian subordinator).
    """
    if beta >= 1.0:
        return torch.ones(shape, device=device)
    b = float(beta)
    U = torch.rand(shape, device=device).clamp(1e-7, 1 - 1e-7) * math.pi
    E = (-torch.log(torch.rand(shape, device=device).clamp_min(1e-12))).clamp_min(1e-12)
    term1 = torch.sin(b * U) / torch.sin(U).clamp_min(1e-12).pow(1.0 / b)
    term2 = (torch.sin((1.0 - b) * U) / E).clamp_min(1e-30).pow((1.0 - b) / b)
    return (term1 * term2).clamp_min(1e-12)


class AlphaStableDiffusion:
    """Subordinated-Gaussian α-stable forward process with a tabulated score."""

    def __init__(
        self,
        d: int,
        num_timesteps: int = 1000,
        alpha: float = 1.7,
        cosine_s: float = 0.008,
        num_r: int = 256,
        mc_samples: int = 8192,
        clip_q: float = 0.999,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0.0 < alpha <= 2.0:
            raise ValueError(f"alpha must be in (0, 2], got {alpha}")
        self.d = d
        self.num_timesteps = num_timesteps
        self.alpha = alpha
        self.beta = alpha / 2.0  # subordinator stability index
        self.device = torch.device(device)

        abar = cosine_alpha_bar(num_timesteps, cosine_s).to(self.device)  # (T,)
        self.sqrt_abar = abar.sqrt().float()
        self.sigma = (1.0 - abar).sqrt().float()  # Gaussian-conditional noise scale σ_t
        self._clip_q = clip_q

        # Precompute the 1-D score table h_t(r) = E[1/W_t | r], W_t = σ_t² · A.
        self.r_grid, self.h = self._build_table(num_r, mc_samples, seed)

    # ---- subordinator sampling (with quantile clip for stability) -----------
    def _sample_A(self, shape: tuple[int, ...]) -> torch.Tensor:
        A = sample_pos_stable(self.beta, shape, self.device)
        if self._clip_q is not None and self.beta < 1.0:
            cap = torch.quantile(A.flatten().float(), self._clip_q)
            A = A.clamp_max(cap)
        return A

    def _build_table(self, num_r: int, mc: int, seed: int):
        """Per-timestep radius grid + score magnitude ``h(r) = E[1/W|r]`` by MC over A."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        # sample the subordinator on CPU with a fixed seed for a reproducible table
        A_cpu = sample_pos_stable(self.beta, (mc,), "cpu")
        if self._clip_q is not None and self.beta < 1.0:
            A_cpu = A_cpu.clamp_max(torch.quantile(A_cpu.float(), self._clip_q))
        A = A_cpu.to(self.device).double()
        half_d = 0.5 * self.d
        # |ξ|² ~ ChiSquare(d) = Gamma(d/2, 2); reuse one draw across timesteps
        chi2 = (
            torch._standard_gamma(
                torch.full((mc,), half_d, dtype=torch.float64),
                generator=g,
            ).to(self.device)
            * 2.0
        )
        T = self.num_timesteps
        r_grid = torch.empty(T, num_r, device=self.device)
        h_tab = torch.empty(T, num_r, device=self.device)
        for ti in range(T):
            W = (self.sigma[ti].double() ** 2 * A).clamp_min(1e-12)  # (mc,)
            r_samp = torch.sqrt(W * chi2)
            r_max = torch.quantile(r_samp, 0.9999).clamp_min(1e-6) * 1.02
            grid = torch.linspace(0.0, float(r_max), num_r, device=self.device)
            logW = torch.log(W)
            inv2W = 0.5 / W
            log_g = -half_d * logW[None, :] - (grid[:, None] ** 2) * inv2W[None, :]
            m = log_g.max(dim=1, keepdim=True).values
            wts = torch.exp(log_g - m)
            num = (wts * (1.0 / W)[None, :]).sum(dim=1)
            den = wts.sum(dim=1).clamp_min(1e-30)
            r_grid[ti] = grid
            h_tab[ti] = (num / den).float()
        return r_grid, h_tab

    # ---- forward process ----------------------------------------------------
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """Return ``(x_t, u, c_in)``.

        ``x_t = √ᾱ_t x₀ + √W ε`` with ``W = σ_t² A`` (α-stable via subordination);
        ``u = √W ε`` is the additive noise; ``c_in = 1/√(1 + W)`` is the EDM-style input
        scale the trainer applies so the network sees an ``O(1)`` window.
        """
        n = x0.dim()
        v = (-1,) + (1,) * (n - 1)
        A = self._sample_A((x0.shape[0],))  # (B,)
        sigma = self.sigma.to(t.device)[t]  # (B,)
        W = (sigma**2 * A).clamp_min(1e-12)  # (B,) mixing variance
        eps = torch.randn_like(x0)
        u = W.sqrt().view(v) * eps
        x_t = self.sqrt_abar.to(t.device)[t].view(v) * x0 + u
        c_in = (1.0 / (1.0 + W).sqrt()).view(v)
        return x_t, u, c_in

    def _h_at(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate ``h`` at radii ``r (B,)`` for integer timesteps ``t (B,)``."""
        rg = self.r_grid.to(r.device)[t]  # (B, num_r)
        hg = self.h.to(r.device)[t]
        r = r.clamp_min(0.0).unsqueeze(1)
        idx = torch.searchsorted(rg, r).clamp(1, rg.shape[1] - 1)
        r0 = torch.gather(rg, 1, idx - 1)
        r1 = torch.gather(rg, 1, idx)
        h0 = torch.gather(hg, 1, idx - 1)
        h1 = torch.gather(hg, 1, idx)
        w = ((r - r0) / (r1 - r0).clamp_min(1e-12)).clamp(0.0, 1.0)
        return (h0 + w * (h1 - h0)).squeeze(1)  # (B,)

    def score_target(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generalized α-stable score ``∇_u log q(u) = -u · h(|u|)`` (radius over all
        non-batch dims)."""
        b = u.shape[0]
        r = u.reshape(b, -1).norm(dim=1)  # (B,)
        h = self._h_at(r, t)  # (B,)
        return -u * h.reshape(b, *([1] * (u.dim() - 1)))
