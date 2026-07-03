"""JointDiT: a Diffusion Transformer (DiT) trained jointly to denoise and classify.

Same joint objective as JointDiffusion (Deja et al., 2023) but the U-Net backbone
is replaced by a DiT (Peebles & Xie, 2023) restructured for LOB windows:

  1. Tokenize   the ``(T × F)`` window with **no patchify** — every ``(timestep,
                level)`` cell is its own token, produced by a shared linear
                projection of the per-cell value.  The F axis carries the
                per-level features chosen by the dataset (OFI/depth or LOB price
                offsets), so one column ≈ one price level.
  2. Position   **factored** learned positional embeddings — a time-position table
                ``(T, D)`` and a level-position table ``(F, D)`` are broadcast-added
                so token ``(t, l)`` gets ``pos_time[t] + pos_level[l]``.
  3. DiT blocks self-attention + MLP, each modulated by the timestep embedding via
                adaLN-Zero (per-block shift/scale/gate produced from t), with
                **U-ViT additive long skips**: encoder-half block ``i`` is added
                into the input of decoder-half block ``N-1-i``.
  4. Denoise    a final adaLN layer + linear maps each token back to one scalar,
                reshaped to ``eps_hat (B, 1, T, F)``.
  5. Classify   the token sequence is mean-pooled and an MLP head predicts the
                trend label (down / flat / up).

Two training contracts share this one backbone (as in :class:`JointDiffusion`):

  * ``forward(x_t, t) -> (eps_hat, logits)`` — raw ε-prediction network, used by
    the DDPM DiT trainer (``crypto.train_jointdit``).
  * ``denoise(x, sigma) -> (x0_hat, logits)`` — EDM-preconditioned consistency
    function ``f_theta``, used by the consistency (``train_jointdit_cm``) and
    drift (``train_jointdit_drift``) trainers.

At inference call ``predict(batch, device)`` → ``logits (B, 3)`` (identical
contract to every other crypto model).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.consistency import precond
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
    """adaLN-Zero final layer mapping each token back to one scalar cell."""

    def __init__(self, dim: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class JointDiT(nn.Module):
    """DiT backbone trained jointly to denoise (ε-pred / consistency) and classify."""

    family = "joint_diffusion"  # same predict/forward contract as JointDiffusion

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        dim = config.get("jdit_dim", 192)
        depth = config.get("jdit_depth", 6)
        heads = config.get("jdit_heads", 6)
        mlp_ratio = config.get("jdit_mlp_ratio", 4.0)
        dropout = config.get("jdit_dropout", 0.1)

        self.T, self.F, self.depth = T, F_dim, depth

        # No patchify: one token per (timestep, level) cell, shared 1→D projection.
        self.embed = nn.Linear(1, dim)
        # Factored positional embeddings: separate time / level tables, added.
        self.pos_time = nn.Parameter(torch.zeros(1, T, 1, dim))
        self.pos_level = nn.Parameter(torch.zeros(1, 1, F_dim, dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            DiTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        )
        self.final = FinalLayer(dim, out_ch=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )

        # EDM / consistency preconditioning parameters — only used by denoise();
        # forward() is left as a raw ε-network so DDPM trainers keep working.
        self.sigma_data = float(config.get("cm_sigma_data", 0.5))
        self.sigma_min = float(config.get("cm_sigma_min", 0.002))
        self.consistency = bool(config.get("cm_enabled", False))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_time, std=0.02)
        nn.init.normal_(self.pos_level, std=0.02)
        # adaLN-Zero: zero the modulation outputs so blocks start as identity.
        for blk in self.blocks:
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def _tokenize(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, 1, T, F) -> tokens (B, T*F, D) with factored positions added.
        B = x_t.shape[0]
        cells = x_t.squeeze(1).unsqueeze(-1)  # (B, T, F, 1)
        tok = self.embed(cells) + self.pos_time + self.pos_level  # (B, T, F, D)
        return tok.reshape(B, self.T * self.F, -1)

    def _encode(self, tok: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Run the DiT blocks with U-ViT additive long skips (block i ↔ N-1-i)."""
        skips: list[torch.Tensor] = []
        half = self.depth // 2
        for i, blk in enumerate(self.blocks):
            if i < half:  # encoder half — stash outputs
                tok = blk(tok, c)
                skips.append(tok)
            elif i >= self.depth - half:  # decoder half — add mirror skip
                tok = blk(tok + skips.pop(), c)
            else:  # middle block (odd depth) — no skip
                tok = blk(tok, c)
        return tok

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        # t carries the DDPM timestep or the EDM c_noise; both feed the same
        # sinusoidal embedding (which accepts float inputs).
        dim = self.pos_time.shape[-1]
        c = self.time_mlp(sinusoidal_embedding(t, dim))
        tok = self._encode(self._tokenize(x_t), c)
        eps_hat = self.final(tok, c).reshape(x_t.shape[0], self.T, self.F)
        logits = self.classifier(tok.mean(dim=1))
        return eps_hat.unsqueeze(1), logits

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor):
        """EDM consistency function f_theta(x, sigma) -> (x0_hat, logits). sigma: (B,)."""
        c_skip, c_out, c_in, c_noise = precond(sigma, self.sigma_data, self.sigma_min)
        v = (-1,) + (1,) * (x.dim() - 1)  # (B,1,1,1)
        raw, logits = self(c_in.view(v) * x, c_noise)
        x0 = c_skip.view(v) * x + c_out.view(v) * raw
        return x0, logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch["x"].to(device).float()
        b = x.shape[0]
        if self.consistency:  # read logits from the denoised (sigma_min) pass
            sigma = torch.full((b,), self.sigma_min, device=device)
            _, logits = self.denoise(x, sigma)
        else:  # DDPM path: evaluate the clean window at t = 0
            t = torch.zeros(b, dtype=torch.long, device=device)
            _, logits = self(x, t)
        return logits
