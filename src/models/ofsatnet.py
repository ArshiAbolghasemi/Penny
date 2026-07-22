"""OF-SATNet — single-asset variant of OF-MATNet's multi-axis attention Transformer.

Based on: Bandealinaeini, Sharifkhani & Salavati, "Attention-Based Multi-Asset Order
Flow Networks for Enhanced Mid-Price Prediction" (ICAIF '25), arXiv:2025 / ACM
3768292.3770430. The paper's full model, OF-MATNet, attends over three axes —
temporal, cross-asset, and order-book level — where the cross-asset axis is built
from peer assets pre-selected via rolling-window Granger causality. **OF-SATNet is
the paper's own single-asset ablation**: cross-asset attention is dropped (N=1) and
only the Temporal and Level paths remain, each an independent Transformer encoder
with positional encoding. This file implements that single-asset variant only; the
cross-asset / Granger-causality machinery is intentionally out of scope here.

Per the paper, the Level path attends over the top-``L`` order-book OFI levels while
the Temporal path attends over the ``T``-step history of the full per-timestep
feature vector. Both paths project to a shared embedding dim ``D``, are augmented
with sinusoidal positional encoding, and are each processed by a standard
``nn.TransformerEncoder``. Their pooled representations are concatenated

    h_final = Concat(h_T, h_L) in R^(2D)

and mapped by a final linear layer to the prediction target (Eq. 7-8 in the paper,
minus the cross-asset term ``h_N``). The paper's target is the next-step mid-price
return, trained with MSE; this repo's shared benchmark instead frames every model as
3-class down/flat/up trend classification, so OF-SATNet follows the same
``family = "classifier"`` contract as every other baseline here and is trained with
cross-entropy under the shared protocol.

Input : ``(B, 1, T_past, n_features)`` — squeezed to ``(B, T, F)``.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Level-path input layout
------------------------
The Level path needs the ``L`` per-level OFI series. The first ``L`` feature
columns *are* the per-level OFI (level 0..L-1) — see ``crypto/features.py`` OFI
mode — so the Level path slices ``[:, :, :L]`` directly.

Config keys
-----------
ofsatnet_levels        number of order-book levels L forming the Level path
                        (must be <= n_features)                        (default 10)
ofsatnet_dim            shared projection dim D                        (default 64)
ofsatnet_heads          attention heads per Transformer encoder         (default 4)
ofsatnet_layers         Transformer encoder layers per path             (default 2)
ofsatnet_ff_dim         feed-forward dim inside each encoder layer      (default 4*D)
ofsatnet_dropout        dropout (encoder + head)                       (default 0.1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.modules import count_parameters as count_parameters  # re-export


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding, added after the input projection."""

    def __init__(self, length: int, dim: int, n: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for sinusoidal PE, got {dim}")
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        den = torch.pow(n, 2 * torch.arange(0, dim // 2, dtype=torch.float) / dim)
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos / den)
        pe[:, 1::2] = torch.cos(pos / den)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, length, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class AttentionAxis(nn.Module):
    """One axis of OF-MATNet's multi-axis attention: project -> PE -> Transformer -> pool.

    Shared by the Temporal and Level paths (Section 4.3 of the paper); only the
    input projection dim and sequence length differ between instantiations.
    """

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        dim: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.pe = SinusoidalPE(seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, in_dim)
        x = self.pe(self.proj(x))
        x = self.encoder(x)
        return x.mean(dim=1)  # pooled embedding (B, dim)


class OFSATNet(nn.Module):
    """OF-SATNet: Temporal + Level multi-axis attention (single-asset OF-MATNet)."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        L = config.get("ofsatnet_levels", 10)
        if L > F_dim:
            raise ValueError(
                f"ofsatnet_levels={L} exceeds n_features={F_dim}"
            )
        dim = config.get("ofsatnet_dim", 64)
        heads = config.get("ofsatnet_heads", 4)
        layers = config.get("ofsatnet_layers", 2)
        ff_dim = config.get("ofsatnet_ff_dim", 4 * dim)
        drop = config.get("ofsatnet_dropout", 0.1)

        self.L = L
        self.temporal = AttentionAxis(F_dim, T, dim, heads, layers, ff_dim, drop)
        self.level = AttentionAxis(T, L, dim, heads, layers, ff_dim, drop)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(2 * dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, T, F) -> (B, T, F)

        h_t = self.temporal(x)  # temporal path: attends over T,   tokens = F-dim vectors

        h_l = self.level(self._level_input(x))  # level path: attends over L levels

        h_final = torch.cat([h_t, h_l], dim=-1)  # Eq. 7 (cross-asset term dropped)
        return self.head(self.dropout(h_final))

    def _level_input(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the (B, L, T) per-level OFI series for the Level path.

        The first L columns are the per-level OFI (see crypto/features.py OFI mode).
        """
        levels = x[:, :, : self.L]  # (B, T, L)
        return levels.transpose(1, 2)  # (B, L, T) — tokens = per-level series

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
