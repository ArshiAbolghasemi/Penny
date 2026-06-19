"""TimesFM forecaster: univariate mid-return prediction + trend head.

Forecasts ``T_future`` window-relative mid returns from the ``T_past`` observed
returns.  If the optional ``timesfm`` package and its pretrained weights are
available they form a (frozen) forecast prior that is blended with a trainable
residual transformer; otherwise the residual transformer is trained from scratch
so the pipeline always runs.  A log line states which path is active.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class TrendHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 3)

    def forward(self, trend_value: torch.Tensor) -> torch.Tensor:
        return self.fc(trend_value.view(-1, 1))


class _ResidualForecaster(nn.Module):
    def __init__(self, t_past, t_future, d, heads, layers):
        super().__init__()
        self.input = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, t_past, d) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=d * 2,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d, t_future)

    def forward(self, past_returns):
        h = self.input(past_returns.unsqueeze(-1)) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class TimesFMForecastModel(nn.Module):
    """Univariate mid-return forecaster, optionally warm-started from TimesFM."""

    family = "forecast"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.t_past = config["T_past"]
        self.t_future = config["T_future"]
        self.residual = _ResidualForecaster(
            self.t_past,
            self.t_future,
            config.get("timesfm_hidden", 256),
            config.get("timesfm_heads", 8),
            config.get("timesfm_layers", 4),
        )
        self.blend = nn.Parameter(torch.tensor(0.5))
        self.tfm = None
        if config.get("timesfm_pretrained", True):
            self._try_load(config)

    def _try_load(self, config: dict) -> None:
        repo = config.get("timesfm_repo", "google/timesfm-2.0-500m-pytorch")
        try:
            import timesfm

            self.tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="cpu",
                    per_core_batch_size=config.get("batch_size", 16),
                    horizon_len=self.t_future,
                    context_len=self.t_past,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo),
            )
            logger.info("TimesFM pretrained prior loaded from {}", repo)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "TimesFM unavailable ({}); training the residual forecaster from "
                "scratch. Install with `uv sync --extra timesfm` for the prior.",
                type(exc).__name__,
            )
            self.tfm = None

    @torch.no_grad()
    def _prior(self, past_mid, boundary):
        if self.tfm is None:
            return torch.zeros(past_mid.shape[0], self.t_future, device=past_mid.device)
        series = [row.detach().cpu().numpy() for row in past_mid]
        fc, _ = self.tfm.forecast(series, freq=[0] * len(series))
        fc = torch.as_tensor(fc, dtype=past_mid.dtype, device=past_mid.device)
        return fc / boundary.view(-1, 1) - 1.0

    def forward(self, past_mid, boundary):
        past_returns = past_mid / boundary.view(-1, 1) - 1.0
        residual = self.residual(past_returns)
        prior = self._prior(past_mid, boundary)
        return prior * self.blend + residual

    def forecast(self, batch, device) -> torch.Tensor:
        """Predict window-relative future mid returns ``(B, T_future)``."""
        true_mid = batch["true_mid"].to(device).float()
        boundary = true_mid[:, self.t_past - 1]
        return self(true_mid[:, : self.t_past], boundary)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
