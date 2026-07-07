"""JointDiffusionLevy: score-predicting U-Net + trend classifier (Deja-style).

The backbone is a 2-D U-Net over LOB windows ``x (B, 1, T_past, F)`` with
**FiLM / adaLN timestep conditioning** (config ``jdl_cond``): every conv block
modulates its GroupNorm output with a per-channel ``(1 + scale) * h + shift``
computed from the sinusoidal timestep embedding.  ``adaln`` uses an affine-free
norm and zero-initialised modulation (identity at init), ``film`` keeps the norm
affine and learns the modulation on top.

Two heads share the encoder (joint diffusion-classifier, Deja et al. 2023):

* **score head** — the decoder + 1x1 conv predicts the *generalized score*
  ``grad log q(x_t|x_0)`` of the (Gaussian or Lévy) forward kernel; trained with
  the weighted DSM MSE in ``crypto.train_jointdiff_levy``.
* **trend head** — AttentionPool over bottleneck tokens -> 3-class logits
  (down / stationary / up) for the horizon ``label_k`` baked into the dataset.

``predict(batch, device)`` is the **feature-only inference path**: it runs just
the encoder + trend head on the clean window at ``t = 0`` — no decoder, no
generative sampling loop — for low-latency deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import (
    AttentionPool,
    BiN,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class FiLMDoubleConv(nn.Module):
    """Two 3x3 convs, each followed by GroupNorm and FiLM/adaLN modulation.

    ``mode="film"``: affine GroupNorm + learned scale/shift from temb.
    ``mode="adaln"``: affine-free GroupNorm + zero-init scale/shift (identity at
    init, the DiT-style adaptive layer norm).
    """

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, mode: str) -> None:
        super().__init__()
        affine = mode == "film"
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch, affine=affine)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch, affine=affine)
        self.mod = nn.Linear(temb_dim, 4 * out_ch)  # (scale1, shift1, scale2, shift2)
        if mode == "adaln":
            nn.init.zeros_(self.mod.weight)
            nn.init.zeros_(self.mod.bias)
        self.act = nn.SiLU()
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        s1, b1, s2, b2 = self.mod(temb).chunk(4, dim=1)  # each (B, C)
        v = (-1, self.out_ch, 1, 1)
        x = self.norm1(self.conv1(x))
        x = self.act(x * (1.0 + s1.view(v)) + b1.view(v))
        x = self.norm2(self.conv2(x))
        return self.act(x * (1.0 + s2.view(v)) + b2.view(v))


class DownF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, mode: str) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = FiLMDoubleConv(in_ch, out_ch, temb_dim, mode)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x), temb)


class UpF(nn.Module):
    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, temb_dim: int, mode: str
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = FiLMDoubleConv(out_ch + skip_ch, out_ch, temb_dim, mode)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, temb)


class JointDiffusionLevy(nn.Module):
    """Time-conditioned U-Net predicting the generalized score + trend logits."""

    family = "joint_diffusion"

    def __init__(self, config: dict) -> None:
        super().__init__()
        base = config.get("jdl_base_channels", 32)
        depth = config.get("jdl_depth", 2)
        temb_dim = config.get("jdl_time_emb", 128)
        mode = config.get("jdl_cond", "film")  # "film" | "adaln"
        if mode not in ("film", "adaln"):
            raise ValueError(f"jdl_cond must be 'film' or 'adaln', got {mode!r}")
        self.temb_dim = temb_dim

        T = config.get("T_past")
        F_dim = config.get("n_features")
        self.bin = BiN(T, F_dim) if (T and F_dim) else None

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
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
        self.out_conv = nn.Conv2d(base, 1, 1)
        bottleneck = chans[-1]
        pool_heads = config.get("jdl_pool_heads", 4)
        self.pool = AttentionPool(bottleneck, heads=pool_heads)
        self.classifier = nn.Linear(bottleneck, 3)

    def _embed(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.temb_dim))

    def encode(
        self, x: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encoder only: returns (bottleneck feature map, skip list)."""
        if self.bin is not None:
            x = self.bin(x.squeeze(1)).unsqueeze(1)
        h = self.stem(x, temb)
        skips = [h]
        for down in self.downs:
            h = down(h, temb)
            skips.append(h)
        return h, skips

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """Full joint pass: ``(score_hat (B,1,T,F), logits (B,3))``."""
        temb = self._embed(t)
        h, skips = self.encode(x_t, temb)
        tokens = skips[-1].flatten(2).transpose(1, 2)  # (B, H*W, C)
        logits = self.classifier(self.pool(tokens))
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            h = up(h, skip, temb)
        return self.out_conv(h), logits

    @torch.no_grad()
    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Feature-only inference: encoder + trend head on the clean window at
        ``t = 0``. Skips the decoder and any generative sampling entirely."""
        x = batch["x"].to(device).float()
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        temb = self._embed(t)
        _, skips = self.encode(x, temb)
        tokens = skips[-1].flatten(2).transpose(1, 2)
        return self.classifier(self.pool(tokens))
