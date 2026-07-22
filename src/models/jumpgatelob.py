"""JumpGateLOB: jump-diffusion score-matching joint classifier for noisy LOB windows.

A deliberately **simple** joint diffusion-classifier (same shape as
:class:`~models.alphastablelob.AlphaStableLOB`): a shared trunk — optional ``BiN`` →
(bi)GRU local encoder → **one** DiT-style temporal self-attention layer — feeding a
**trend head** (the only path used at inference) and a single-channel **score head**
(a training-time auxiliary).  No noise-state estimator, no gated experts, no
``w_conditioning`` — the earlier gated variant of this architecture is preserved in
git history.

What makes it the *jump-diffusion* model is the training procedure
(``crypto.train_jumpgatelob``), built for LOB windows that are **noisy and contain
jumps**:

  1. **Jump-diffusion forward process** (``src/levy``): the additive noise is
     ``u = √W·ξ`` with ``W = σ_t² + Σ_k S_k`` — Brownian variance plus
     compound-Poisson gamma jumps — so the trunk is trained on perturbations that
     look like market microstructure noise *and* discrete jumps.
  2. **Generalized denoising score matching** (Baule 2025): the score head regresses
     the *true* score of that non-Gaussian kernel, ``∇log q(x_t|x₀) = −u·h(|u|)``
     with ``h`` a precomputed 1-D table — not ε-prediction, not a Gaussian score.
  3. **Noise-consistent classification**: the trend head is additionally trained on
     jump-noised low-``t`` windows (CE + KL-consistency to its own clean prediction),
     so the *inference path itself* is robust to noise and jumps — the denoising
     auxiliary alone only regularises the trunk.

Inference contract matches every other crypto model: ``predict(batch, device) →
logits (B, 3)`` from a single clean-window pass (no sampling loop).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    AttentionPool,
    BiN,
    LevelAttention,
    LevelMLP,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    # x: (B, N, D); shift/scale: (B, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TemporalAttnBlock(nn.Module):
    """One DiT-style temporal self-attention layer over ``T`` (adaLN-Zero)."""

    def __init__(self, dim: int, heads: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sa, ca, ga, sm, cm, gm = self.ada(c).chunk(6, dim=1)
        h = _modulate(self.norm1(x), sa, ca)
        if return_attn:
            a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        else:
            a, _ = self.attn(h, h, h, need_weights=False)
        x = x + ga.unsqueeze(1) * a
        h = _modulate(self.norm2(x), sm, cm)
        x = x + gm.unsqueeze(1) * self.mlp(h)
        if return_attn:
            # the adaLN-Zero gates are t-conditioned: how much this block writes
            # back into the residual stream at the current noise level
            return x, {"self": w, "gate_attn": ga, "gate_mlp": gm}
        return x


class DiffBlock(nn.Module):
    """Grid diffusion block: feature-axis mixing over ``F`` + trunk-context injection,
    adaLN-Zero conditioned on ``t``.  Operates on ``(B, C, T, F)``."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ctx_dim: int,
        feat_mix: str,
        feat_heads: int,
        pad_mode: str,
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.ada = nn.Linear(cond_dim, 3 * channels)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.ctx = nn.Linear(ctx_dim, channels)  # per-timestep trunk context
        if feat_mix == "attn":
            self.mix = LevelAttention(channels, feat_heads)
        elif feat_mix == "conv":
            self.mix = nn.Conv2d(
                channels, channels, (1, 3), padding=(0, 1), padding_mode=pad_mode
            )
        elif feat_mix == "mlp":
            if n_features is None:
                raise ValueError("feat_mix='mlp' requires n_features")
            self.mix = LevelMLP(channels, n_features)
        else:
            raise ValueError(f"feat_mix must be attn|conv|mlp, got {feat_mix!r}")

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=1)  # each (B, C)
        v = (-1, x.shape[1], 1, 1)
        h = self.norm(x) * (1 + scale.view(v)) + shift.view(v)
        h = h + self.ctx(H).permute(0, 2, 1).unsqueeze(-1)  # (B, C, T, 1) over F
        h = F.silu(self.mix(h))
        return x + gate.view(v) * h


class ScoreHead(nn.Module):
    """Flat grid net predicting the jump-diffusion score ``ŝ (B, 1, T, F)``."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ctx_dim: int,
        n_blocks: int,
        feat_mix: str,
        feat_heads: int,
        pad_mode: str,
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(1, channels, 1)
        self.blocks = nn.ModuleList(
            DiffBlock(
                channels, cond_dim, ctx_dim, feat_mix, feat_heads, pad_mode, n_features
            )
            for _ in range(n_blocks)
        )
        self.out = nn.Conv2d(channels, 1, 1)  # single-channel score
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, c, H):
        x = self.input_projection(x_t)  # (B, C, T, F)
        for blk in self.blocks:
            x = blk(x, c, H)
        return self.out(x)  # (B, 1, T, F)


class JumpGateLOB(nn.Module):
    """(bi)GRU + one temporal-attention trunk; trend head + jump-diffusion score head."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        F_dim = config["n_features"]
        temb_dim = config.get("jdl_time_emb", 128)
        self.temb_dim = temb_dim
        self.F = F_dim

        # ---- adaptive input normalization (front-end) -----------------------
        self.bin = (
            BiN(config["T_past"], F_dim) if config.get("use_bin", False) else None
        )

        # ---- local encoder ---------------------------------------------------
        self.local = config.get("jgl_local", "gru")
        hidden = config.get("jgl_gru_hidden", 64)
        bidir = bool(config.get("jgl_bidirectional", True))
        if self.local == "gru":
            self.gru = nn.GRU(
                input_size=F_dim,
                hidden_size=hidden,
                num_layers=config.get("jgl_gru_layers", 2),
                dropout=config.get("jgl_gru_dropout", 0.0)
                if config.get("jgl_gru_layers", 2) > 1
                else 0.0,
                batch_first=True,
                bidirectional=bidir,
            )
            D = hidden * (2 if bidir else 1)
        elif self.local == "conv":
            D = hidden
            self.embed = nn.Linear(F_dim, D)
            self.tconv = nn.Sequential(
                nn.Conv1d(D, D, 3, padding=1, padding_mode="replicate"),
                nn.SiLU(),
                nn.Conv1d(D, D, 3, padding=1, padding_mode="replicate"),
            )
        else:
            raise ValueError(f"jgl_local must be gru|conv, got {self.local!r}")
        self.D = D

        # ---- timestep conditioning c = MLP(emb(t)) --------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )

        # ---- one temporal-attention layer -----------------------------------
        self.temporal = TemporalAttnBlock(
            D,
            heads=config.get("jgl_attn_heads", 4),
            cond_dim=temb_dim,
            dropout=config.get("jgl_attn_dropout", 0.1),
        )

        # ---- trend head ------------------------------------------------------
        self.pool = AttentionPool(D, heads=config.get("jdl_pool_heads", 4))
        self.cls_dropout = nn.Dropout(config.get("cls_dropout", 0.0))
        self.classifier = nn.Linear(D, 3)

        # ---- score head ------------------------------------------------------
        self.score_head = ScoreHead(
            channels=config.get("jgl_diff_channels", 16),
            cond_dim=temb_dim,
            ctx_dim=D,
            n_blocks=config.get("jgl_diff_blocks", 2),
            feat_mix=config.get("jgl_feat_mix", "conv"),
            feat_heads=config.get("jgl_feat_heads", 2),
            pad_mode=config.get("jgl_pad_mode", "reflect"),
            n_features=F_dim,
        )

    # ---- trunk --------------------------------------------------------------
    def _local(self, x: torch.Tensor) -> torch.Tensor:
        s = x.squeeze(1)  # (B, T, F)
        if self.bin is not None:
            s = self.bin(s)
        if self.local == "gru":
            H, _ = self.gru(s)
            return H
        h = self.embed(s).transpose(1, 2)  # (B, D, T)
        return self.tconv(h).transpose(1, 2)  # (B, T, D)

    def _cond(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.temb_dim))

    def trunk(self, x: torch.Tensor, t: torch.Tensor, return_attn: bool = False):
        """Return ``(H (B,T,D), c (B,temb_dim))``, plus attention if requested."""
        c = self._cond(t)
        H0 = self._local(x)  # (B, T, D)
        T = H0.shape[1]
        pos = sinusoidal_embedding(torch.arange(T, device=x.device), self.D).unsqueeze(
            0
        )
        if return_attn:
            H, attn = self.temporal(H0 + pos, c, return_attn=True)
            return H, c, attn
        H = self.temporal(H0 + pos, c)
        return H, c

    def _trend_logits(
        self, H: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not return_attn:
            return self.classifier(self.cls_dropout(self.pool(H)))
        pooled, w = self.pool(H, return_attn=True)
        return self.classifier(self.cls_dropout(pooled)), w

    # ---- task-specific passes (training uses these separately) --------------
    def classify(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trend logits at ``t = 0`` — used for clean windows at inference *and* for
        jump-noised windows in the noise-consistency loss (deployment never knows the
        noise level, so the classifier never conditions on ``t``).

        ``return_attn`` additionally yields the trunk's temporal self-attention
        ``"self"`` ``(B, T, T)``, the trend head's ``"pool"`` weights ``(B, T)``
        over timesteps, and the t-conditioned adaLN-Zero gates. ``t`` overrides
        the default zero timestep purely for analysis — sweeping it shows how the
        trunk re-weights itself as the noise level rises; inference must leave it
        ``None``.
        """
        if t is None:
            t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if not return_attn:
            H, _ = self.trunk(x, t)
            return self._trend_logits(H)
        H, _, attn = self.trunk(x, t, return_attn=True)
        logits, pool_w = self._trend_logits(H, return_attn=True)
        return logits, {**attn, "pool": pool_w}

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the jump-diffusion score on the noised window at timestep ``t``."""
        H, c = self.trunk(x_t, t)
        return self.score_head(x_t, c, H)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """Joint pass: ``(ŝ, logits)``."""
        H, c = self.trunk(x_t, t)
        logits = self._trend_logits(H)
        s_hat = self.score_head(x_t, c, H)
        return s_hat, logits

    def trunk_parameters(self):
        """All params except the trend head (for a frozen-trunk phase-2 probe)."""
        head = set(map(id, self.pool.parameters())) | set(
            map(id, self.classifier.parameters())
        )
        return (p for p in self.parameters() if id(p) not in head)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: trunk + trend head on the clean window."""
        x = batch["x"].to(device).float()
        return self.classify(x)
