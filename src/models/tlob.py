"""TLOB — Temporal-LOB Transformer, faithful to the reference implementation.

Architecture
------------
  1. BiN                 — bilinear normalisation on raw (B, T, F) before projection.
  2. Linear(F, dim) + positional encoding (sinusoidal by default, learnable optional).
  3. 2 × n_blocks alternating TransformerLayers, with a transpose after every layer:
       even index → temporal attention on (B, T, dim)
       odd  index → spatial  attention on (B, dim, T)   (after the transpose)
     Each TransformerLayer:
       QKV expanded to dim×heads → MHA(embed=dim×heads) → Linear(dim×heads→dim)
       → residual → LayerNorm → MLP(dim→4dim→final_dim) → (residual if same dim).
     The last pair of layers outputs dim//4 and T//4 respectively.
  4. Flatten → dynamic MLP (halve by ×4 until dim<128) → Linear(→3).

Config keys
-----------
T_past          window length T
n_features      raw feature dimension F   (set by dataset builder)
tlob_dim        model dim                 (default 64)
tlob_n_blocks   number of block pairs     (default 2)
tlob_n_heads    attention heads           (default 1)
tlob_sin_emb    use sinusoidal PE         (default True; False = learnable)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import BiN, count_parameters as count_parameters  # re-export


# ── Positional encoding ───────────────────────────────────────────────────────


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding, added after the input projection."""

    def __init__(self, T: int, dim: int, n: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for sinusoidal PE, got {dim}")
        pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
        den = torch.pow(n, 2 * torch.arange(0, dim // 2, dtype=torch.float) / dim)
        pe = torch.zeros(T, dim)
        pe[:, 0::2] = torch.sin(pos / den)
        pe[:, 1::2] = torch.cos(pos / den)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


# ── TransformerLayer ──────────────────────────────────────────────────────────


class TransformerLayer(nn.Module):
    """One transformer layer with QKV expansion, post-norm, and a 2-layer MLP.

    QKV are each projected from ``hidden_dim`` to ``hidden_dim * num_heads`` so
    that every head sees the full per-token embedding (head_dim = hidden_dim).
    After attention the output is projected back to ``hidden_dim``.  The MLP then
    maps ``hidden_dim → 4·hidden_dim → final_dim``.  A residual is added on the
    MLP output only when ``final_dim == hidden_dim`` (dimensions match).
    """

    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int) -> None:
        super().__init__()
        expanded = hidden_dim * num_heads
        self.q = nn.Linear(hidden_dim, expanded)
        self.k = nn.Linear(hidden_dim, expanded)
        self.v = nn.Linear(hidden_dim, expanded)
        self.attn = nn.MultiheadAttention(expanded, num_heads, batch_first=True)
        self.w0 = nn.Linear(expanded, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, final_dim),
        )
        self.same_dim = final_dim == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        q, k, v = self.q(x), self.k(x), self.v(x)
        a, _ = self.attn(q, k, v)
        x = self.w0(a) + res
        x = self.norm(x)
        out = self.mlp(x)
        if self.same_dim:
            out = out + res
        return out


# ── TLOB ─────────────────────────────────────────────────────────────────────


class TLOB(nn.Module):
    """TLOB classifier.

    Accepts ``(B, 1, T, F)`` (squeezed internally) or ``(B, T, F)``.
    Returns ``(B, 3)`` trend logits.
    """

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F = config["n_features"]
        dim = config.get("tlob_dim", 64)
        n_blocks = config.get("tlob_n_blocks", 2)
        n_heads = config.get("tlob_n_heads", 1)
        sin_emb = config.get("tlob_sin_emb", True)

        self.T = T
        self.F = F

        self.bin = BiN(T, F)
        self.proj = nn.Linear(F, dim)
        if sin_emb:
            self.pe = SinusoidalPE(T, dim)
        else:
            self.pe = nn.Parameter(torch.randn(1, T, dim))

        # Build 2*n_blocks alternating TransformerLayers.
        # Even layers: temporal attention — hidden=dim,   seq=T.
        # Odd  layers: spatial  attention — hidden=T,     seq=dim.
        # Last pair reduces output to dim//4 (temporal) and T//4 (spatial).
        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(n_blocks):
            last = i == n_blocks - 1
            self.layers.append(
                TransformerLayer(dim, n_heads, dim // 4 if last else dim)
            )
            self.layers.append(TransformerLayer(T, n_heads, T // 4 if last else T))

        # Dynamic classifier MLP: flatten → shrink by ×4 until <128 → Linear(3).
        total = (dim // 4) * (T // 4)
        head: list[nn.Module] = []
        while total > 128:
            head += [nn.Linear(total, total // 4), nn.GELU()]
            total //= 4
        head.append(nn.Linear(total, 3))
        self.head = nn.Sequential(*head)

    def _apply_pe(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.pe, SinusoidalPE):
            return self.pe(x)
        return x + self.pe  # learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, T, F) → (B, T, F)
        x = self.bin(x)  # bilinear normalisation  (B, T, F)
        x = self._apply_pe(self.proj(x))  # project + PE        (B, T, dim)
        for layer in self.layers:
            x = layer(x)  # TransformerLayer
            x = x.permute(0, 2, 1)  # swap seq ↔ embed for next layer
        # after 2*n_blocks permutes: (B, T//4, dim//4)
        return self.head(x.reshape(x.shape[0], -1))  # (B, 3)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
