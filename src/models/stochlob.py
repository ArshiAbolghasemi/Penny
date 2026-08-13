"""StochLOB with inference-time stochastic latent forecasting.

The historical LOB window is encoded once by the original AlphaStableLOB trunk.
Only the pooled latent state is rolled forward: the 60 observations are never used
as integration steps.  The default transition softly routes a fixed-alpha stable
increment and a compound-Poisson jump at every state-dependent Euler step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from models.alphastablelob import AlphaStableLOB


@dataclass
class RolloutOutput:
    """Trajectory logits and stepwise quantities used for scientific diagnostics."""

    logits: torch.Tensor  # (M, B, C)
    terminal: torch.Tensor  # (M, B, D)
    routing: torch.Tensor  # (steps, M, B, 2)
    stable_scale: torch.Tensor  # (steps, M, B, D)
    jump_intensity: torch.Tensor  # (steps, M, B, 1)
    latent_norm: torch.Tensor  # (steps + 1, M, B)

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.softmax(dim=-1)

    @property
    def mean_probability(self) -> torch.Tensor:
        return self.probabilities.mean(dim=0)

    @property
    def predictive_entropy(self) -> torch.Tensor:
        p = self.mean_probability.clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-1)

    @property
    def trajectory_disagreement(self) -> torch.Tensor:
        p = self.probabilities
        return ((p - p.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean(dim=0)


class _ConditionedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_symmetric_alpha_stable(
    shape: tuple[int, ...], alpha: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Chambers-Mallows-Stuck draws from a unit symmetric alpha-stable law."""
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    if alpha == 2.0:
        # In this parameterisation S_2 has characteristic function exp(-|t|^2).
        return math.sqrt(2.0) * torch.randn(shape, device=device, dtype=dtype)
    u = (torch.rand(shape, device=device, dtype=dtype) - 0.5) * math.pi
    w = -torch.log(torch.rand(shape, device=device, dtype=dtype).clamp_min(1e-7))
    cos_u = torch.cos(u).clamp_min(1e-7)
    if abs(alpha - 1.0) < 1e-6:
        return torch.tan(u)
    first = torch.sin(alpha * u) / cos_u.pow(1.0 / alpha)
    second = (torch.cos((1.0 - alpha) * u) / w).clamp_min(1e-7)
    return first * second.pow((1.0 - alpha) / alpha)


class LatentStochasticDynamics(nn.Module):
    """Euler rollout with shared drift and configurable stochastic ablations."""

    MODES: ClassVar[set[str]] = {
        "deterministic",
        "gaussian",
        "stable",
        "jump",
        "routed",
    }

    def __init__(self, latent_dim: int, config: dict) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mode = config.get("latent_dynamics", "routed")
        if self.mode not in self.MODES:
            raise ValueError(f"latent_dynamics must be one of {sorted(self.MODES)}")
        self.alpha = float(config.get("latent_alpha", 1.62))
        if not 1.0 < self.alpha <= 2.0:
            raise ValueError("latent_alpha must be in (1, 2] for the initial model")
        self.steps = int(config.get("latent_steps", 15))
        if self.steps < 1:
            raise ValueError("latent_steps must be >= 1")
        self.dt = 1.0 / self.steps

        horizons = tuple(int(k) for k in config.get("horizons", [10, 20, 50, 100]))
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be unique")
        self.horizons = horizons
        self.horizon_to_index = {k: i for i, k in enumerate(horizons)}
        hdim = int(config.get("horizon_emb_dim", 16))
        self.horizon_embedding = nn.Embedding(len(horizons), hdim)
        hidden = int(config.get("latent_hidden", max(64, latent_dim)))
        cond_dim = latent_dim + hdim
        self.drift = _ConditionedMLP(cond_dim, hidden, latent_dim)
        self.stable_scale = _ConditionedMLP(cond_dim, hidden, latent_dim)
        self.jump_intensity = _ConditionedMLP(cond_dim, hidden, 1)
        self.jump_size = _ConditionedMLP(cond_dim, hidden, latent_dim)
        self.router = _ConditionedMLP(cond_dim, hidden, 2)
        self.gaussian_scale = _ConditionedMLP(cond_dim, hidden, latent_dim)
        self.state_norm = nn.LayerNorm(latent_dim)

        self.min_scale = float(config.get("latent_min_scale", 1e-3))
        self.max_scale = float(config.get("latent_max_scale", 1.0))
        self.max_intensity = float(config.get("latent_max_intensity", 20.0))
        self.noise_clip = float(config.get("latent_noise_clip", 10.0))
        self.jump_clip = float(config.get("latent_jump_clip", 5.0))
        self.latent_clip = float(config.get("latent_state_clip", 20.0))

    def _horizon(self, horizon: torch.Tensor) -> torch.Tensor:
        flat = horizon.long().reshape(-1)
        indices = torch.empty_like(flat)
        found = torch.zeros_like(flat, dtype=torch.bool)
        for value, index in self.horizon_to_index.items():
            mask = flat == value
            indices[mask] = index
            found |= mask
        if not bool(found.all()):
            bad = flat[~found].unique().tolist()
            raise ValueError(f"unsupported horizons {bad}; configured {self.horizons}")
        return self.horizon_embedding(indices).reshape(*horizon.shape, -1)

    def forward(
        self, z0: torch.Tensor, horizon: torch.Tensor, trajectories: int
    ) -> dict:
        if trajectories < 1:
            raise ValueError("trajectories must be >= 1")
        b, d = z0.shape
        z = z0.unsqueeze(0).expand(trajectories, b, d).clone()
        h = horizon.reshape(b)
        he = self._horizon(h).unsqueeze(0).expand(trajectories, -1, -1)
        routes, scales, rates = [], [], []
        norms = [z.norm(dim=-1)]

        for _ in range(self.steps):
            zn = self.state_norm(z)
            cond = torch.cat([zn, he], dim=-1)
            mu = self.drift(cond)
            gamma = F.softplus(self.stable_scale(cond)).clamp(
                self.min_scale, self.max_scale
            )
            rate = F.softplus(self.jump_intensity(cond)).clamp_max(self.max_intensity)
            jump_size = self.jump_size(cond).clamp(-self.jump_clip, self.jump_clip)
            route = self.router(cond).softmax(dim=-1)

            stable_noise = sample_symmetric_alpha_stable(
                tuple(z.shape), self.alpha, z.device, z.dtype
            ).clamp(-self.noise_clip, self.noise_clip)
            stable_inc = gamma * (self.dt ** (1.0 / self.alpha)) * stable_noise

            expected_count = rate * self.dt
            sampled_count = torch.poisson(expected_count)
            # Exact Poisson forward sample, identity gradient w.r.t. its mean.
            count = sampled_count.detach() + expected_count - expected_count.detach()
            jump_inc = count * jump_size

            if self.mode == "deterministic":
                stochastic = torch.zeros_like(z)
                effective_route = torch.zeros_like(route)
            elif self.mode == "gaussian":
                sigma = F.softplus(self.gaussian_scale(cond)).clamp(
                    self.min_scale, self.max_scale
                )
                stochastic = sigma * math.sqrt(self.dt) * torch.randn_like(z)
                effective_route = torch.zeros_like(route)
            elif self.mode == "stable":
                stochastic = stable_inc
                effective_route = torch.stack(
                    [torch.ones_like(route[..., 0]), torch.zeros_like(route[..., 1])],
                    -1,
                )
            elif self.mode == "jump":
                stochastic = jump_inc
                effective_route = torch.stack(
                    [torch.zeros_like(route[..., 0]), torch.ones_like(route[..., 1])],
                    -1,
                )
            else:
                stochastic = route[..., :1] * stable_inc + route[..., 1:] * jump_inc
                effective_route = route

            z = z + self.dt * mu + stochastic
            z = self.state_norm(z).clamp(-self.latent_clip, self.latent_clip)
            routes.append(effective_route)
            scales.append(gamma)
            rates.append(rate)
            norms.append(z.norm(dim=-1))

        return {
            "terminal": z,
            "routing": torch.stack(routes),
            "stable_scale": torch.stack(scales),
            "jump_intensity": torch.stack(rates),
            "latent_norm": torch.stack(norms),
        }


class StochLOB(AlphaStableLOB):
    """Existing StochLOB encoder plus stochastic predictive futures."""

    family = "stochastic_forecaster"

    def __init__(self, config: dict) -> None:
        encoder_config = dict(config)
        encoder_config["astable_use_score_head"] = False
        super().__init__(encoder_config)
        self.config = dict(config)
        self.default_horizon = int(config.get("label_k", 10))
        self.train_trajectories = int(config.get("train_trajectories", 4))
        self.test_trajectories = int(config.get("test_trajectories", 20))
        self.dynamics = LatentStochasticDynamics(self.D, config)
        hdim = self.dynamics.horizon_embedding.embedding_dim
        hidden = int(config.get("latent_hidden", max(64, self.D)))
        self.classifier = nn.Sequential(
            nn.Linear(self.D + hdim, hidden),
            nn.SiLU(),
            self.cls_dropout,
            nn.Linear(hidden, 3),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the observed history once into the current market state z0."""
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        H, _ = self.trunk(x, t)
        return self.pool(H)

    def _horizon_tensor(
        self, horizon, batch: int, device: torch.device
    ) -> torch.Tensor:
        if horizon is None:
            return torch.full(
                (batch,), self.default_horizon, device=device, dtype=torch.long
            )
        if not torch.is_tensor(horizon):
            return torch.full((batch,), int(horizon), device=device, dtype=torch.long)
        horizon = horizon.to(device).long().reshape(-1)
        if horizon.numel() == 1:
            horizon = horizon.expand(batch)
        if horizon.numel() != batch:
            raise ValueError(f"expected {batch} horizon values, got {horizon.numel()}")
        return horizon

    def forecast(
        self, z0: torch.Tensor, horizon=None, trajectories: int | None = None
    ) -> RolloutOutput:
        m = self.train_trajectories if trajectories is None else int(trajectories)
        h = self._horizon_tensor(horizon, z0.shape[0], z0.device)
        result = self.dynamics(z0, h, m)
        terminal = result["terminal"]
        he = self.dynamics._horizon(h).unsqueeze(0).expand(m, -1, -1)
        logits = self.classifier(torch.cat([terminal, he], dim=-1))
        return RolloutOutput(logits=logits, **result)

    def classify(
        self, x: torch.Tensor, horizon=None, trajectories: int | None = None
    ) -> torch.Tensor:
        """Return per-trajectory logits, shape (M, B, 3)."""
        return self.forecast(self.encode(x), horizon, trajectories).logits

    def forward(
        self, x: torch.Tensor, horizon=None, trajectories: int | None = None
    ) -> RolloutOutput:
        return self.forecast(self.encode(x), horizon, trajectories)

    @torch.no_grad()
    def predict_distribution(
        self, batch: dict, device: torch.device, trajectories: int | None = None
    ) -> RolloutOutput:
        x = batch["x"].to(device).float()
        horizon = batch.get("horizon", self.default_horizon)
        return self.forward(x, horizon, trajectories or self.test_trajectories)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        # Log probabilities preserve the repository-wide logits contract while making
        # run_test's softmax equal to the Monte-Carlo mean probability.
        return (
            self.predict_distribution(batch, device)
            .mean_probability.clamp_min(1e-8)
            .log()
        )
