"""JointDiT: a Diffusion Transformer (DiT) trained jointly to denoise and classify.

Same joint objective as JointDiffusion (Deja et al., 2023) but the U-Net backbone
is replaced by a DiT (Peebles & Xie, 2023):

  1. Patchify   the (T × F) LOB window into non-overlapping p×p patches → tokens.
  2. DiT blocks self-attention + MLP, each modulated by the timestep embedding
                via adaLN-Zero (per-block shift/scale/gate produced from t).
  3. Denoise    a final adaLN layer + linear un-patchifies the tokens back to
                ε̂ (B, 1, T, F), trained with ε-prediction MSE.
  4. Classify   the token sequence is mean-pooled and an MLP head predicts the
                trend label (down / flat / up).

Input : ``x_t (B, 1, T, F)`` noisy window + integer timestep ``t (B,)``.
Output: ``(eps_hat (B, 1, T, F), logits (B, 3))``.

At inference call ``predict(batch, device)`` which evaluates the clean window at
``t = 0`` → ``logits (B, 3)``.  Identical contract to every other crypto model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    # x: (B, N, D); shift/scale: (B, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero timestep conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        # produces shift/scale/gate for both the attention and MLP sub-blocks
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=1)
        h = _modulate(self.norm1(x), shift_a, scale_a)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * a
        h = _modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """adaLN-Zero final layer mapping tokens back to patch pixels."""

    def __init__(self, dim: int, patch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class JointDiT(nn.Module):
    """DiT backbone trained jointly to denoise (ε-pred) and classify trend."""

    family = "joint_diffusion"  # same predict/forward contract as JointDiffusion

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        p = config.get("jdit_patch", 4)
        dim = config.get("jdit_dim", 192)
        depth = config.get("jdit_depth", 6)
        heads = config.get("jdit_heads", 6)
        mlp_ratio = config.get("jdit_mlp_ratio", 4.0)
        dropout = config.get("jdit_dropout", 0.1)

        self.T, self.F, self.p = T, F_dim, p
        # pad (T, F) up to whole patches; grid is fixed from the config dims
        self.gt = (T + p - 1) // p
        self.gf = (F_dim + p - 1) // p
        self.pad_t = self.gt * p - T
        self.pad_f = self.gf * p - F_dim
        n_tokens = self.gt * self.gf

        self.patch = nn.Conv2d(1, dim, kernel_size=p, stride=p)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            DiTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        )
        self.final = FinalLayer(dim, p, out_ch=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos, std=0.02)
        # adaLN-Zero: zero the modulation outputs so blocks start as identity
        for blk in self.blocks:
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def _temb(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, dim))

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, p*p*1) → (B, 1, T, F) cropped back from the padded grid
        B = x.shape[0]
        p, gt, gf = self.p, self.gt, self.gf
        x = x.view(B, gt, gf, p, p, 1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 1, gt * p, gf * p)
        return x[:, :, : self.T, : self.F]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        dim = self.pos.shape[-1]
        x = F.pad(x_t, (0, self.pad_f, 0, self.pad_t))  # (B, 1, gt*p, gf*p)
        tok = self.patch(x).flatten(2).transpose(1, 2) + self.pos  # (B, N, D)
        c = self._temb(t, dim)
        for blk in self.blocks:
            tok = blk(tok, c)
        eps_hat = self._unpatchify(self.final(tok, c))
        logits = self.classifier(tok.mean(dim=1))
        return eps_hat, logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch["x"].to(device).float()
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        _, logits = self(x, t)
        return logits
