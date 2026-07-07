"""JumpGate-UNet: a W-aware, jump-gated joint diffusion-classifier.

A variant of :class:`JointDiffusionLevy` that (a) predicts epsilon instead of the
generalized score, (b) carries a small **noise-state estimator** ``g_phi`` that
infers the per-sample realized mixing variance ``W`` and a jump indicator from the
noised window, and (c) uses those inferred quantities to condition the backbone, to
gate two denoising "experts", and to softly gate the trend loss.

Components
----------
* **Backbone** — the same FiLM/adaLN U-Net blocks as ``JointDiffusionLevy`` (reused
  directly).  The only change is *what* is fed to the FiLM conditioning: a vector
  ``MLP(concat(emb(t), emb(logW)))`` rather than ``MLP(emb(t))``.
* **g_phi** (:class:`NoiseStateEstimator`) — strided ``Conv1d`` stack + global
  average pool + MLP on ``(x_t, emb(t))`` → ``(logW_hat, pi_logit)``.  Trained
  *only* by the supervised ``L_W`` (its outputs feed the rest of the net detached).
* **Gated experts** — two decoder output heads ``eps_0, eps_1`` mixed by
  ``pi = sigmoid(pi_logit)``: ``eps_hat = (1-pi) eps_0 + pi eps_1``.

Ablation flags (config) — with all of them at their defaults the model is a plain
epsilon-prediction joint U-Net (t-only conditioning, single eps head), i.e. g_phi
is a passive auxiliary head that does not touch the main path:

* ``w_conditioning``: ``"none"`` (default) | ``"inferred"`` | ``"oracle"``.
  ``inferred`` conditions on ``logW_hat.detach()``; ``oracle`` teacher-forces the
  true ``logW`` (passed to :meth:`forward`).
* ``gated_experts``: ``bool`` (default ``False``) — two-expert mixture vs single head.
* ``gate_grad``: ``"detach"`` (default) | ``"flow"`` — whether the mixture weight
  ``pi`` carries gradient into ``g_phi`` (``detach`` keeps "g_phi trained only by L_W").

Inference is feature-only: :meth:`predict` runs g_phi + encoder + trend head on the
clean window at ``t = 0`` (no decoder, no sampling loop).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.jointdifflevy import DownF, FiLMDoubleConv, UpF
from models.modules import (
    AttentionPool,
    BiN,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


class NoiseStateEstimator(nn.Module):
    """g_phi: infer ``(logW_hat, pi_logit)`` from the noised window and timestep.

    Strided ``Conv1d`` stack over the time axis (features as channels) → global
    average pool → MLP on the pooled vector concatenated with the timestep
    embedding.  Outputs two scalars per sample.
    """

    def __init__(
        self, n_features: int, temb_dim: int, hidden: int = 64, n_conv: int = 3
    ) -> None:
        super().__init__()
        chs = [n_features] + [hidden] * n_conv
        self.convs = nn.ModuleList(
            nn.Conv1d(chs[i], chs[i + 1], kernel_size=3, stride=2, padding=1)
            for i in range(n_conv)
        )
        self.act = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden + temb_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )

    def forward(
        self, x_t: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = x_t.squeeze(1).transpose(1, 2)  # (B, F, T)
        for conv in self.convs:
            h = self.act(conv(h))
        h = h.mean(dim=-1)  # global average pool -> (B, hidden)
        out = self.mlp(torch.cat([h, temb], dim=-1))  # (B, 2)
        return out[:, 0], out[:, 1]  # logW_hat (B,), pi_logit (B,)


class JumpGateUNet(nn.Module):
    """W-aware, jump-gated epsilon-prediction U-Net + trend head."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        base = config.get("jdl_base_channels", 32)
        depth = config.get("jdl_depth", 2)
        temb_dim = config.get("jdl_time_emb", 128)
        mode = config.get("jdl_cond", "film")
        if mode not in ("film", "adaln"):
            raise ValueError(f"jdl_cond must be 'film' or 'adaln', got {mode!r}")
        self.temb_dim = temb_dim

        # JumpGate flags
        self.w_conditioning = config.get("w_conditioning", "none")
        if self.w_conditioning not in ("none", "inferred", "oracle"):
            raise ValueError(
                "w_conditioning must be none|inferred|oracle, got "
                f"{self.w_conditioning!r}"
            )
        self.gated_experts = bool(config.get("gated_experts", False))
        self.gate_grad = config.get("gate_grad", "detach")
        if self.gate_grad not in ("detach", "flow"):
            raise ValueError(f"gate_grad must be detach|flow, got {self.gate_grad!r}")

        T = config.get("T_past")
        F_dim = config.get("n_features")
        self.bin = BiN(T, F_dim) if (T and F_dim) else None

        # Conditioning MLP: input is emb(t) alone (w_conditioning="none") or
        # concat(emb(t), emb(logW)) otherwise.  With "none" it is byte-for-byte the
        # JointDiffusionLevy time_mlp, so the model reduces to the plain backbone.
        cond_in = temb_dim if self.w_conditioning == "none" else 2 * temb_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.gphi = NoiseStateEstimator(
            F_dim, temb_dim, hidden=config.get("jg_gphi_hidden", 64)
        )

        chans = [base * (2**i) for i in range(depth + 1)]
        self.stem = FiLMDoubleConv(1, base, temb_dim, mode)
        self.downs = nn.ModuleList(
            DownF(chans[i], chans[i + 1], temb_dim, mode) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            UpF(chans[i + 1], chans[i], chans[i], temb_dim, mode)
            for i in reversed(range(depth))
        )
        self.out_conv0 = nn.Conv2d(base, 1, 1)
        self.out_conv1 = nn.Conv2d(base, 1, 1) if self.gated_experts else None

        bottleneck = chans[-1]
        self.pool = AttentionPool(bottleneck, heads=config.get("jdl_pool_heads", 4))
        self.cls_dropout = nn.Dropout(config.get("cls_dropout", 0.0))
        self.classifier = nn.Linear(bottleneck, 3)

    # ---- conditioning -------------------------------------------------------
    def _conditioning(
        self, t: torch.Tensor, x_t: torch.Tensor, logW_oracle: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(c, logW_hat, pi_logit)``: the FiLM conditioning vector plus the
        raw g_phi outputs (for the supervised L_W and gating)."""
        temb_t = sinusoidal_embedding(t, self.temb_dim)
        logW_hat, pi_logit = self.gphi(x_t, temb_t)
        if self.w_conditioning == "none":
            c = self.cond_mlp(temb_t)
        else:
            if self.w_conditioning == "oracle":
                if logW_oracle is None:  # e.g. inference — fall back to inferred
                    logw = logW_hat.detach()
                else:
                    logw = logW_oracle
            else:  # inferred
                logw = logW_hat.detach()
            wemb = sinusoidal_embedding(logw, self.temb_dim)
            c = self.cond_mlp(torch.cat([temb_t, wemb], dim=-1))
        return c, logW_hat, pi_logit

    def _trend_logits(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Pool bottleneck tokens -> dropout -> 3-class trend logits."""
        tokens = bottleneck.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return self.classifier(self.cls_dropout(self.pool(tokens)))

    def encode(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.bin is not None:
            x = self.bin(x.squeeze(1)).unsqueeze(1)
        h = self.stem(x, c)
        skips = [h]
        for down in self.downs:
            h = down(h, c)
            skips.append(h)
        return h, skips

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        logW_oracle: torch.Tensor | None = None,
    ):
        """Return ``(eps_hat (B,1,T,F), logits (B,3), logW_hat (B,), pi_logit (B,))``."""
        c, logW_hat, pi_logit = self._conditioning(t, x_t, logW_oracle)
        h, skips = self.encode(x_t, c)
        logits = self._trend_logits(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            h = up(h, skip, c)
        eps0 = self.out_conv0(h)
        if self.gated_experts:
            eps1 = self.out_conv1(h)
            pi = torch.sigmoid(pi_logit)
            if self.gate_grad == "detach":
                pi = pi.detach()
            v = (-1, 1, 1, 1)
            eps_hat = (1.0 - pi).view(v) * eps0 + pi.view(v) * eps1
        else:
            eps_hat = eps0
        return eps_hat, logits, logW_hat, pi_logit

    @staticmethod
    def recover_score(eps_hat: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        """Score from the eps prediction: ``s = -eps_hat / W_hat`` (for sampling utils)."""
        v = (-1,) + (1,) * (eps_hat.dim() - 1)
        return -eps_hat / W_hat.reshape(v)

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: g_phi + encoder + trend head on the clean window
        at ``t = 0``.  Skips the decoder and any generative sampling."""
        x = batch["x"].to(device).float()
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        c, _, _ = self._conditioning(t, x, None)
        _, skips = self.encode(x, c)
        return self._trend_logits(skips[-1])
