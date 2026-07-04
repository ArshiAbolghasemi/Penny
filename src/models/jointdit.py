"""JointDiT: a Diffusion Transformer (DiT) trained jointly to denoise and classify.

Same joint objective as JointDiffusion (Deja et al., 2023) but the U-Net backbone
is replaced by a DiT (Peebles & Xie, 2023) restructured for LOB windows:

  1. Tokenize   the ``(T × F)`` window **per price level** (no patchify).  Using
                the known ``features.py`` column layout, each price level at each
                timestep becomes one token — a linear projection of that level's
                channel features (LOB: bid/ask price offset + bid/ask log-volume =
                4 ch; OFI: net Cont-OFI = 1 ch).  The 11 non-level microstructure
                features become one extra "global" token per timestep.  So a
                timestep contributes ``L + 1`` tokens and ``N = T·(L+1)`` — far
                fewer than one-token-per-cell, while keeping each level's identity
                intact (a p×p patch would blend price/volume/global rows, which are
                heterogeneous quantities, and blur which level a value came from).
  2. Position   **factored** learned positional embeddings — a time table ``(T,D)``
                and a slot table ``(L+1, D)`` are broadcast-added, so token
                ``(t, slot)`` gets ``pos_time[t] + pos_level[slot]``.
  3. DiT blocks self-attention + MLP, each modulated by the timestep embedding via
                adaLN-Zero, with **U-ViT additive long skips**: encoder-half block
                ``i`` is added into the input of decoder-half block ``N-1-i``.
  4. Denoise    a final adaLN layer + per-slot linear heads map tokens back to the
                level channels / global features, reassembled into ``(B, 1, T, F)``.
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

N_GLOBAL = 11  # non-level microstructure/trade/quote features (see crypto.features)


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


class JointDiT(nn.Module):
    """DiT backbone trained jointly to denoise (ε-pred / consistency) and classify."""

    family = "joint_diffusion"  # same predict/forward contract as JointDiffusion

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        n = config["n_lob_levels"]
        mode = config.get("feature_mode", "ofi")
        dim = config.get("jdit_dim", 192)
        depth = config.get("jdit_depth", 6)
        heads = config.get("jdit_heads", 6)
        mlp_ratio = config.get("jdit_mlp_ratio", 4.0)
        dropout = config.get("jdit_dropout", 0.1)

        # Per-level layout (crypto.features): LOB packs 4 channels/level (bid/ask
        # price offset + bid/ask log-volume), OFI packs 1 (net Cont-OFI); both add
        # N_GLOBAL non-level features carried by a single global token per timestep.
        self.mode = mode
        self.L = n  # price levels (= level tokens per timestep)
        self.C = 4 if mode == "lob" else 1  # channels per level
        self.G = N_GLOBAL
        self.P = self.L + 1  # tokens per timestep (levels + 1 global)
        self.T, self.F = T, F_dim
        assert self.C * self.L + self.G == F_dim, (
            f"feature layout mismatch: C*L+G={self.C * self.L + self.G} != F={F_dim} "
            f"(mode={mode}, n={n})"
        )
        self.depth = depth

        self.level_embed = nn.Linear(self.C, dim)
        self.global_embed = nn.Linear(self.G, dim)
        # Factored positional embeddings: time table + slot table (L levels + global).
        self.pos_time = nn.Parameter(torch.zeros(1, T, 1, dim))
        self.pos_slot = nn.Parameter(torch.zeros(1, 1, self.P, dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            DiTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        )
        # Final adaLN-Zero: shared modulated norm, then per-slot reconstruction heads.
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head_level = nn.Linear(dim, self.C)
        self.head_global = nn.Linear(dim, self.G)
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
        nn.init.normal_(self.pos_slot, std=0.02)
        # adaLN-Zero: zero the modulation outputs so blocks start as identity.
        for blk in self.blocks:
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        for head in (self.head_level, self.head_global):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _split_features(self, x: torch.Tensor):
        """(B, T, F) → (level feats (B,T,L,C), global feats (B,T,G))."""
        lvl_flat = x[..., : self.C * self.L]  # channel-major: [ch0(L), ch1(L), …]
        lvl = lvl_flat.reshape(*x.shape[:2], self.C, self.L).transpose(2, 3)
        glb = x[..., self.C * self.L :]
        return lvl, glb  # (B,T,L,C), (B,T,G)

    def _merge_features(self, lvl: torch.Tensor, glb: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_split_features`: (B,T,L,C),(B,T,G) → (B, T, F)."""
        lvl_flat = lvl.transpose(2, 3).reshape(*lvl.shape[:2], self.C * self.L)
        return torch.cat([lvl_flat, glb], dim=-1)

    def _tokenize(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, 1, T, F) -> tokens (B, T*P, D) with factored positions added.
        lvl, glb = self._split_features(x_t.squeeze(1))
        level_tok = self.level_embed(lvl)  # (B, T, L, D)
        global_tok = self.global_embed(glb).unsqueeze(2)  # (B, T, 1, D)
        tok = torch.cat([level_tok, global_tok], dim=2)  # (B, T, P, D)
        tok = tok + self.pos_time + self.pos_slot
        return tok.reshape(tok.shape[0], self.T * self.P, -1)

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

    def _reconstruct(self, tok: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Final adaLN + per-slot heads: tokens (B, T*P, D) → eps (B, 1, T, F)."""
        shift, scale = self.final_ada(c).chunk(2, dim=1)
        h = _modulate(self.final_norm(tok), shift, scale)
        h = h.reshape(h.shape[0], self.T, self.P, -1)  # (B, T, P, D)
        lvl = self.head_level(h[:, :, : self.L, :])  # (B, T, L, C)
        glb = self.head_global(h[:, :, self.L, :])  # (B, T, G)
        return self._merge_features(lvl, glb).unsqueeze(1)  # (B, 1, T, F)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        # t carries the DDPM timestep or the EDM c_noise; both feed the same
        # sinusoidal embedding (which accepts float inputs).
        dim = self.pos_time.shape[-1]
        c = self.time_mlp(sinusoidal_embedding(t, dim))
        tok = self._encode(self._tokenize(x_t), c)
        eps_hat = self._reconstruct(tok, c)
        logits = self.classifier(tok.mean(dim=1))
        return eps_hat, logits

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
