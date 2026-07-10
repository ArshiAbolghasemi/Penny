"""JumpGateLOB: a Lévy, W-aware joint diffusion-classifier with ONE temporal-attention
layer, built for feature-only LOB trend inference.

Instead of a deep stack of jointly-coupled residual blocks, the *global* timestep
coupling is a **single** temporal self-attention layer over ``T`` on top of a
**local** recurrent (or conv) encoder — the rest of the capacity goes into a small
grid diffusion head.

Shared trunk (used by both losses; run once per pass):

  1. **local encoder** — a (bi)GRU over the window gives per-timestep, order-aware
     context ``H0 (B, T, D)`` (``D = 2*hidden`` when bidirectional).  A GRU is fine
     because the window ends at the prediction point and the label lives strictly
     outside it, so bidirectionality leaks nothing.  ``jgl_local="conv"`` swaps in a
     causal-free temporal conv stack.
  2. **one temporal self-attention layer** over ``T`` with sinusoidal positional
     encoding — the single global-coupling layer (a DiT block: adaLN-Zero + MHA +
     MLP).  Do **not** stack more unless an ablation justifies it.

  ``(t, logŴ)`` are injected via **adaLN-Zero** (identity at init) from a conditioning
  vector ``c = MLP(emb(t) [, emb(logŴ)])`` selected by ``w_conditioning``.

Two heads share the trunk:

  * **trend head** — attention-pool over ``T`` → 3 logits.  Feature-only inference
    runs *only* the trunk + this head on the clean window (no reverse sampling).
  * **diffusion head** — a small flat, constant-``(T,F)`` grid net (no U-Net pooling):
    each block mixes **book levels** over ``F`` (cross-level attention *or* a conv
    with reflect/replicate padding — never circular), injects the per-timestep trunk
    context ``H``, and is adaLN-Zero conditioned on ``(t, logŴ)``.  ε-prediction with
    **gated experts** ``ε̂ = (1−π)ε₀ + π·ε₁``, ``π = σ(π_logit)``.

Lévy machinery kept from the JumpGate family: ``NoiseStateEstimator`` (``g_phi``) →
``(logŴ, π_logit)`` (normal variance-mean mixture: Gaussian-given-``W`` with jumps
gating ``π``); ``recover_score = −ε/W``; ``w_conditioning ∈ {none, inferred, oracle}``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    AttentionPool,
    BiN,
    DiffusionStepMLP,
    LevelAttention,
    NoiseStateEstimator,
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sa, ca, ga, sm, cm, gm = self.ada(c).chunk(6, dim=1)
        h = _modulate(self.norm1(x), sa, ca)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + ga.unsqueeze(1) * a
        h = _modulate(self.norm2(x), sm, cm)
        x = x + gm.unsqueeze(1) * self.mlp(h)
        return x


class DiffBlock(nn.Module):
    """Grid diffusion block: feature-axis mixing over ``F`` + trunk-context injection,
    adaLN-Zero conditioned on ``(t, logŴ)``.  Operates on ``(B, C, T, F)``."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ctx_dim: int,
        feat_mix: str,
        feat_heads: int,
        pad_mode: str,
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
        else:
            raise ValueError(f"feat_mix must be attn|conv, got {feat_mix!r}")

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=1)  # each (B, C)
        v = (-1, x.shape[1], 1, 1)
        h = self.norm(x) * (1 + scale.view(v)) + shift.view(v)
        h = h + self.ctx(H).permute(0, 2, 1).unsqueeze(-1)  # (B, C, T, 1) over F
        h = F.silu(self.mix(h))
        return x + gate.view(v) * h


class DiffHead(nn.Module):
    """Flat grid ε-net conditioned on the trunk context, with gated experts."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        ctx_dim: int,
        n_blocks: int,
        feat_mix: str,
        feat_heads: int,
        pad_mode: str,
        gated_experts: bool,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(1, channels, 1)
        self.blocks = nn.ModuleList(
            DiffBlock(channels, cond_dim, ctx_dim, feat_mix, feat_heads, pad_mode)
            for _ in range(n_blocks)
        )
        self.out0 = nn.Conv2d(channels, 1, 1)
        self.out1 = nn.Conv2d(channels, 1, 1) if gated_experts else None
        nn.init.zeros_(self.out0.weight)
        if self.out1 is not None:
            nn.init.zeros_(self.out1.weight)

    def forward(self, x_t, c, H, pi=None):
        x = self.input_projection(x_t)  # (B, C, T, F)
        for blk in self.blocks:
            x = blk(x, c, H)
        eps0 = self.out0(x)
        if self.out1 is not None and pi is not None:
            eps1 = self.out1(x)
            v = (-1, 1, 1, 1)
            return (1.0 - pi).view(v) * eps0 + pi.view(v) * eps1
        return eps0


class JumpGateLOB(nn.Module):
    """(bi)GRU + one temporal-attention layer trunk, grid diffusion head + trend head."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        F_dim = config["n_features"]
        temb_dim = config.get("jdl_time_emb", 128)
        self.temb_dim = temb_dim
        self.F = F_dim

        self.w_conditioning = config.get("w_conditioning", "none")
        if self.w_conditioning not in ("none", "inferred", "oracle"):
            raise ValueError(
                f"w_conditioning must be none|inferred|oracle, got {self.w_conditioning!r}"
            )
        self.gated_experts = bool(config.get("gated_experts", False))
        self.gate_grad = config.get("gate_grad", "detach")
        if self.gate_grad not in ("detach", "flow"):
            raise ValueError(f"gate_grad must be detach|flow, got {self.gate_grad!r}")

        # ---- adaptive input normalization (front-end) -----------------------
        # BiN normalizes each window (per-feature over T mixed with per-timestep
        # over F), removing the level/scale non-stationarity between the calendar-day
        # train/val/test regimes.  Matches the JumpGateUNet front-end.  Applied only
        # to the trunk's encoder input — the raw noised window still feeds the
        # diffusion head so the eps target is unchanged.
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

        # ---- conditioning + noise-state -------------------------------------
        self.gphi = NoiseStateEstimator(
            F_dim, temb_dim, hidden=config.get("jg_gphi_hidden", 64)
        )
        # cond vector c = MLP(emb(t) [, emb(logW)]) -> temb_dim, feeds every adaLN.
        self.cond_mlp = DiffusionStepMLP(temb_dim, temb_dim, self.w_conditioning)

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

        # ---- diffusion head --------------------------------------------------
        self.diff_head = DiffHead(
            channels=config.get("jgl_diff_channels", 16),
            cond_dim=temb_dim,
            ctx_dim=D,
            n_blocks=config.get("jgl_diff_blocks", 2),
            feat_mix=config.get("jgl_feat_mix", "conv"),
            feat_heads=config.get("jgl_feat_heads", 2),
            pad_mode=config.get("jgl_pad_mode", "reflect"),
            gated_experts=self.gated_experts,
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

    def trunk(self, x: torch.Tensor, t: torch.Tensor, logW_oracle: torch.Tensor | None):
        """Return ``(H (B,T,D), logW_hat (B,), pi_logit (B,), c (B,temb_dim))``."""
        temb = sinusoidal_embedding(t, self.temb_dim)
        logW_hat, pi_logit = self.gphi(x, temb)
        c = self.cond_mlp(t, logW_hat, logW_oracle)
        H0 = self._local(x)  # (B, T, D)
        T = H0.shape[1]
        pos = sinusoidal_embedding(torch.arange(T, device=x.device), self.D).unsqueeze(
            0
        )
        H = self.temporal(H0 + pos, c)
        return H, logW_hat, pi_logit, c

    def _trend_logits(self, H: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cls_dropout(self.pool(H)))

    # ---- task-specific passes (training uses these separately) --------------
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Trend logits from the *clean* window at ``t = 0`` (matches inference)."""
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        H, _, _, _ = self.trunk(x, t, None)
        return self._trend_logits(H)

    def diffuse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        logW_oracle: torch.Tensor | None = None,
    ):
        """ε-prediction on the *noised* window.  Returns ``(eps_hat, logW_hat, pi_logit)``."""
        H, logW_hat, pi_logit, c = self.trunk(x_t, t, logW_oracle)
        pi = None
        if self.gated_experts:
            pi = torch.sigmoid(pi_logit)
            if self.gate_grad == "detach":
                pi = pi.detach()
        eps_hat = self.diff_head(x_t, c, H, pi)
        return eps_hat, logW_hat, pi_logit

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        logW_oracle: torch.Tensor | None = None,
    ):
        """Convenience joint pass on a single input (eval/compat):
        ``(eps_hat, logits, logW_hat, pi_logit)``."""
        H, logW_hat, pi_logit, c = self.trunk(x_t, t, logW_oracle)
        logits = self._trend_logits(H)
        pi = None
        if self.gated_experts:
            pi = torch.sigmoid(pi_logit)
            if self.gate_grad == "detach":
                pi = pi.detach()
        eps_hat = self.diff_head(x_t, c, H, pi)
        return eps_hat, logits, logW_hat, pi_logit

    @staticmethod
    def recover_score(eps_hat: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        v = (-1,) + (1,) * (eps_hat.dim() - 1)
        return -eps_hat / W_hat.reshape(v)

    def trunk_parameters(self):
        """All params except the trend head — for the Baranchuk phase-2 freeze."""
        head = set(map(id, self.pool.parameters())) | set(
            map(id, self.classifier.parameters())
        )
        return (p for p in self.parameters() if id(p) not in head)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: trunk + trend head on the clean window."""
        x = batch["x"].to(device).float()
        return self.classify(x)
