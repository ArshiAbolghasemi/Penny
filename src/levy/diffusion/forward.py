"""Forward (noising) process with a Gaussian / Lévy ablation toggle.

Wraps a :class:`NoiseSchedule` and (for the Lévy path) a precomputed
:class:`GeneralizedScoreTable`.  Exposes exactly the two operations training needs:

* :meth:`add_noise` — sample ``x_t`` from ``q(x_t | x_0)``.
* :meth:`score_target` — the denoising-score-matching regression target
  ``grad_{x_t} log q(x_t | x_0)`` (Gaussian score, or the tabulated generalized
  score for the jump-diffusion kernel).

The ablation switch is ``diffusion.process in {"gaussian", "levy"}``; everything
else (schedule, sampling of ``x_t``) is shared, so the two are directly comparable.
"""

from __future__ import annotations

import torch

from levy.diffusion.generalized_score import (
    GeneralizedScoreTable,
    JumpParams,
    build_score_table,
    gaussian_score,
    jump_intensity,
    sample_W_batched,
)
from levy.diffusion.schedules import NoiseSchedule, make_schedule


class ForwardProcess:
    def __init__(self, cfg, d: int, device: torch.device | str = "cpu"):
        """``cfg`` is a :class:`levy.config.DiffusionConfig`; ``d`` is the flattened
        data dimensionality (channels * seq_len * n_features)."""
        self.cfg = cfg
        self.d = d
        self.process = cfg.process
        self.device = torch.device(device)
        self.schedule: NoiseSchedule = make_schedule(cfg).to(device)
        self.jump = JumpParams(cfg.jump_gamma_shape, cfg.jump_gamma_scale)
        self.lambda_t = jump_intensity(cfg.num_timesteps, cfg.jump_rate, device)
        self._gen = torch.Generator(device=self.device).manual_seed(cfg.table_seed)

        self.table: GeneralizedScoreTable | None = None
        if self.process == "levy":
            self.table = build_score_table(
                d=d,
                sigma=self.schedule.sigma,
                lambda_t=self.lambda_t,
                jump=self.jump,
                num_r=cfg.table_num_r,
                mc_samples=cfg.table_mc_samples,
                seed=cfg.table_seed,
                device=device,
            )
        elif self.process != "gaussian":
            raise ValueError(
                f"unknown process '{self.process}' (expected 'gaussian' or 'levy')"
            )

    def to(self, device: torch.device | str) -> "ForwardProcess":
        self.device = torch.device(device)
        self.schedule = self.schedule.to(device)
        self.lambda_t = self.lambda_t.to(device)
        if self.table is not None:
            self.table = self.table.to(device)
        return self

    def _bcast(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return v.reshape((-1,) + (1,) * (x.dim() - 1))

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_t, u)`` where ``u = x_t - a_t x_0`` is the additive noise.

        Gaussian: ``u = sigma_t * eps``.  Lévy: ``u = sqrt(W) * eps`` with ``W`` a
        per-sample gamma-subordinated mixing variance (Gaussian scale mixture).
        """
        a_t, sigma_t = self.schedule.gather(t)
        eps = torch.randn_like(x0)
        if self.process == "gaussian":
            scale = sigma_t
        else:
            lam = self.lambda_t.to(t.device)[t]
            W = sample_W_batched(sigma_t, lam, self.jump, self._gen)
            scale = torch.sqrt(W)
        u = self._bcast(scale, x0) * eps
        x_t = self._bcast(a_t, x0) * x0 + u
        return x_t, u

    def score_target(
        self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """DSM target ``grad_{x_t} log q(x_t|x_0)`` for the chosen process."""
        a_t, sigma_t = self.schedule.gather(t)
        u = x_t - self._bcast(a_t, x0) * x0
        if self.process == "gaussian":
            return gaussian_score(u, sigma_t)
        assert self.table is not None
        return self.table.score(u, t)
