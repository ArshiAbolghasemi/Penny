"""JumpGate-ScoreGrad: the JumpGate Lévy-jump design on a joint window denoiser.

Same forward process, noise-state estimator and gating as :class:`JumpGateUNet`,
but the backbone is a **recurrent-conditioned, jointly-coupled window denoiser**
(an evolution of the ScoreGrad/TimeGrad idea, Yan et al. 2021):

* a **bidirectional GRU** unrolls the LOB window along time, producing a
  per-timestep context ``H (B, T, 2*num_cells)``.  The window ends at the
  prediction point and the label lives strictly outside it, so looking both
  directions inside the window leaks nothing.
* a **flat (constant-resolution) denoiser** operates on the whole ``(T, F)`` grid
  at once.  Each residual block couples **timesteps jointly** with a *dilated
  temporal* ``Conv2d`` over ``T`` (replacing the old per-timestep feature-axis
  conv), then mixes **book levels** with *cross-level attention over ``F``*
  (replacing the feature-axis conv) — no U-Net pooling over ``T``.  Temporal
  convs use ``replicate`` padding: neither time nor book levels are periodic, so
  the old ``circular`` padding is dropped.

Every timestep's noise is thus predicted with information from the whole window
(joint coupling) rather than independently through its GRU state.  The diffusion
runs over the full grid conditioned on the recurrent context.

JumpGate additions carried over from :class:`JumpGateUNet`:

* ``g_phi`` (:class:`NoiseStateEstimator`) infers ``(logW_hat, pi_logit)``;
* the per-block **diffusion-step** vector encodes ``(t, logW)`` via ``w_conditioning``
  (``none`` | ``inferred`` | ``oracle``) — where W-awareness enters the denoiser;
* **gated experts**: two output projections mixed by ``pi = sigmoid(pi_logit)``;
* the **trend head** reads an attention-pool over the GRU context; inference is
  feature-only (GRU + trend head on the clean window, no denoiser, no sampling).

Despite the "ScoreGrad" name this is an **epsilon-prediction** model (see the
trainer); ``recover_score = -eps / W`` converts to the score for sampling utils.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jumpgateunet import NoiseStateEstimator
from models.modules import (
    AttentionPool,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


class DiffusionStepMLP(nn.Module):
    """Build the per-block diffusion-step vector from ``t`` (and optionally ``logW``).

    ScoreGrad's ``diffusion_embedding`` generalized to be W-aware: the only place
    the inferred/true noise state enters the denoiser.
    """

    def __init__(self, temb_dim: int, hidden: int, w_conditioning: str) -> None:
        super().__init__()
        self.temb_dim = temb_dim
        self.w_conditioning = w_conditioning
        cond_in = temb_dim if w_conditioning == "none" else 2 * temb_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self, t: torch.Tensor, logW_hat: torch.Tensor, logW_oracle: torch.Tensor | None
    ) -> torch.Tensor:
        temb = sinusoidal_embedding(t, self.temb_dim)
        if self.w_conditioning == "none":
            return self.mlp(temb)
        if self.w_conditioning == "oracle":
            logw = logW_hat.detach() if logW_oracle is None else logW_oracle
        else:  # inferred
            logw = logW_hat.detach()
        wemb = sinusoidal_embedding(logw, self.temb_dim)
        return self.mlp(torch.cat([temb, wemb], dim=-1))


class LevelAttention(nn.Module):
    """Self-attention across the ``F`` book levels, applied per (batch, timestep).

    Input/output ``(B, C, T, F)``; the ``F`` positions are the attention tokens, so
    every level can attend to every other level (cross-level mixing) — the
    replacement for the old feature-axis convolution.
    """

    def __init__(self, channels: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * t, f, c)  # (B*T, F, C) tokens = levels
        hn = self.norm(h)
        a, _ = self.attn(hn, hn, hn, need_weights=False)
        h = h + a
        return h.reshape(b, t, f, c).permute(0, 3, 1, 2)  # (B, C, T, F)


class ResidualBlock(nn.Module):
    """Dilated *temporal* conv + gated activation + cross-level attention + skip.

    Couples timesteps jointly (dilated conv over ``T``) and levels (attention over
    ``F``).  ``x`` is ``(B, C, T, F)``; ``cond`` is the GRU context ``(B, T, cond_dim)``;
    ``diffusion_step`` is ``(B, residual_hidden)``.  ``replicate`` padding preserves
    ``T`` without assuming periodicity.
    """

    def __init__(
        self,
        residual_hidden: int,
        residual_channels: int,
        cond_dim: int,
        dilation: int,
        level_heads: int,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.dilated_conv = nn.Conv2d(
            residual_channels,
            2 * residual_channels,
            kernel_size=(3, 1),
            padding=(dilation, 0),
            dilation=(dilation, 1),
            padding_mode=pad_mode,
        )
        self.diffusion_projection = nn.Linear(residual_hidden, residual_channels)
        self.cond_projection = nn.Linear(cond_dim, 2 * residual_channels)
        self.level_attn = LevelAttention(residual_channels, level_heads)
        self.output_projection = nn.Conv2d(residual_channels, 2 * residual_channels, 1)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, cond, diffusion_step):
        # inject diffusion step (broadcast over T, F)
        d = self.diffusion_projection(diffusion_step).unsqueeze(-1).unsqueeze(-1)
        y = self.dilated_conv(x + d)  # (B, 2C, T, F)
        # inject per-timestep GRU conditioner (broadcast over F)
        c = self.cond_projection(cond).permute(0, 2, 1).unsqueeze(-1)  # (B, 2C, T, 1)
        y = y + c
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B, C, T, F)
        y = self.level_attn(y)  # cross-level mixing over F
        y = F.leaky_relu(self.output_projection(y), 0.4)  # (B, 2C, T, F)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WindowDenoiser(nn.Module):
    """Flat, joint (T, F) grid denoiser conditioned on the GRU context."""

    def __init__(
        self,
        cond_dim: int,
        residual_hidden: int,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        level_heads: int = 2,
        gated_experts: bool = False,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.input_projection = nn.Conv2d(1, residual_channels, 1)
        self.residual_layers = nn.ModuleList(
            ResidualBlock(
                residual_hidden,
                residual_channels,
                cond_dim,
                dilation=2 ** (i % dilation_cycle_length),
                level_heads=level_heads,
                pad_mode=pad_mode,
            )
            for i in range(residual_layers)
        )
        self.skip_projection = nn.Conv2d(residual_channels, residual_channels, 1)
        self.out0 = nn.Conv2d(residual_channels, 1, 1)
        self.out1 = nn.Conv2d(residual_channels, 1, 1) if gated_experts else None
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.out0.weight)
        if self.out1 is not None:
            nn.init.zeros_(self.out1.weight)

    def forward(self, x_t, cond, dstep, pi=None):
        """``x_t (B,1,T,F)``, ``cond (B,T,cond_dim)``, ``dstep (B,residual_hidden)``.

        Returns predicted noise ``(B, 1, T, F)`` — a 2-expert mix when ``self.out1``
        exists and ``pi (B,)`` is given.
        """
        x = F.leaky_relu(self.input_projection(x_t), 0.4)  # (B, C, T, F)
        skips = []
        for layer in self.residual_layers:
            x, s = layer(x, cond, dstep)
            skips.append(s)
        x = torch.stack(skips).sum(0) / math.sqrt(len(self.residual_layers))
        x = F.leaky_relu(self.skip_projection(x), 0.4)  # (B, C, T, F)
        eps0 = self.out0(x)  # (B, 1, T, F)
        if self.out1 is not None and pi is not None:
            eps1 = self.out1(x)
            v = (-1, 1, 1, 1)
            eps = (1.0 - pi).view(v) * eps0 + pi.view(v) * eps1
        else:
            eps = eps0
        return eps


class JumpGateScoreGrad(nn.Module):
    """biGRU encoder + joint window denoiser + trend head, with JumpGate machinery."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        F_dim = config["n_features"]
        temb_dim = config.get("jdl_time_emb", 128)
        self.temb_dim = temb_dim
        self.F = F_dim

        # JumpGate flags
        self.w_conditioning = config.get("w_conditioning", "none")
        if self.w_conditioning not in ("none", "inferred", "oracle"):
            raise ValueError(
                f"w_conditioning must be none|inferred|oracle, got {self.w_conditioning!r}"
            )
        self.gated_experts = bool(config.get("gated_experts", False))
        self.gate_grad = config.get("gate_grad", "detach")
        if self.gate_grad not in ("detach", "flow"):
            raise ValueError(f"gate_grad must be detach|flow, got {self.gate_grad!r}")

        # Backbone hyperparameters
        num_cells = config.get("sg_num_cells", 64)
        num_layers = config.get("sg_num_layers", 2)
        residual_hidden = config.get("sg_residual_hidden", 64)

        self.gru = nn.GRU(
            input_size=F_dim,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=config.get("sg_rnn_dropout", 0.0) if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        cond_dim = 2 * num_cells  # bidirectional
        self.gphi = NoiseStateEstimator(
            F_dim, temb_dim, hidden=config.get("jg_gphi_hidden", 64)
        )
        self.dstep = DiffusionStepMLP(temb_dim, residual_hidden, self.w_conditioning)
        self.denoiser = WindowDenoiser(
            cond_dim=cond_dim,
            residual_hidden=residual_hidden,
            residual_layers=config.get("sg_residual_layers", 8),
            residual_channels=config.get("sg_residual_channels", 8),
            dilation_cycle_length=config.get("sg_dilation_cycle", 2),
            level_heads=config.get("sg_level_heads", 2),
            gated_experts=self.gated_experts,
            pad_mode=config.get("sg_pad_mode", "replicate"),
        )

        # trend head over the GRU context
        self.pool = AttentionPool(cond_dim, heads=config.get("jdl_pool_heads", 4))
        self.cls_dropout = nn.Dropout(config.get("cls_dropout", 0.0))
        self.classifier = nn.Linear(cond_dim, 3)

    def _encode(self, x_t: torch.Tensor) -> torch.Tensor:
        """GRU context ``H (B, T, 2*num_cells)`` from a window ``(B, 1, T, F)``."""
        H, _ = self.gru(x_t.squeeze(1))  # (B, T, 2*num_cells)
        return H

    def _trend_logits(self, H: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cls_dropout(self.pool(H)))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        logW_oracle: torch.Tensor | None = None,
    ):
        """Return ``(eps_hat (B,1,T,F), logits (B,3), logW_hat (B,), pi_logit (B,))``."""
        temb_t = sinusoidal_embedding(t, self.temb_dim)
        logW_hat, pi_logit = self.gphi(x_t, temb_t)

        H = self._encode(x_t)  # (B, T, cond_dim)
        logits = self._trend_logits(H)

        dstep = self.dstep(t, logW_hat, logW_oracle)  # (B, residual_hidden)
        pi = None
        if self.gated_experts:
            pi = torch.sigmoid(pi_logit)
            if self.gate_grad == "detach":
                pi = pi.detach()
        eps_hat = self.denoiser(x_t, H, dstep, pi)  # (B, 1, T, F)
        return eps_hat, logits, logW_hat, pi_logit

    @staticmethod
    def recover_score(eps_hat: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        v = (-1,) + (1,) * (eps_hat.dim() - 1)
        return -eps_hat / W_hat.reshape(v)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: GRU + trend head on the clean window (no denoiser)."""
        x = batch["x"].to(device).float()
        return self._trend_logits(self._encode(x))
