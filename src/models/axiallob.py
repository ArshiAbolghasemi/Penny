"""Axial-LOB — axial-attention classifier for LOB data.

Based on: Kisiel & Gorse, "Axial-LOB: High-Frequency Trading with Axial Attention"
(arXiv:2212.01807), which applies the position-sensitive axial attention of
Axial-DeepLab (Wang et al., 2020) to the LOB image.

The (T × F) window is treated as an image; each axial block factorises 2D
attention into a height (time) axial attention followed by a width (feature)
axial attention, each with a learnable relative-position embedding. Channels are
expanded by a 1×1 conv, processed by ``axial_blocks`` residual axial blocks, then
global-pooled and mapped to 3 trend logits.

Input : ``(B, 1, T_past, n_features)``.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Config keys
-----------
axial_channels  embedding channels for the axial blocks  (default 32)
axial_groups    attention heads (must divide channels)    (default 8)
axial_blocks    number of residual axial blocks           (default 2)
axial_dropout   dropout before the head                   (default 0.1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import count_parameters as count_parameters  # re-export


class AxialAttention(nn.Module):
    """Position-sensitive axial multi-head self-attention along one axis.

    Faithful to the Axial-DeepLab formulation: queries, keys and values each
    carry a learnable relative-position embedding, and the three similarity terms
    (content, query-position, key-position) are gated by a shared BatchNorm.

    ``kernel_size`` must equal the length of the attended axis.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        groups: int,
        kernel_size: int,
        width: bool = False,
    ) -> None:
        super().__init__()
        if in_planes % groups != 0 or out_planes % groups != 0:
            raise ValueError("planes must be divisible by groups")
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        if self.group_planes % 2 != 0:
            raise ValueError("out_planes // groups must be even")
        self.kernel_size = kernel_size
        self.width = width

        self.qkv_transform = nn.Conv1d(
            in_planes, out_planes * 2, kernel_size=1, bias=False
        )
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Relative-position embedding, indexed by (key - query) offset.
        self.relative = nn.Parameter(
            torch.randn(self.group_planes * 2, kernel_size * 2 - 1)
        )
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer("flatten_index", relative_index.view(-1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1.0 / self.in_planes))
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.group_planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W).  Reorient so the attended axis is last.
        if self.width:
            x = x.permute(0, 2, 1, 3)  # (N, H, C, W)  — attend over W
        else:
            x = x.permute(0, 3, 1, 2)  # (N, W, C, H)  — attend over H
        N, S, C, L = x.shape  # S = folded batch axis, L = attended-axis length
        x = x.contiguous().view(N * S, C, L)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * S, self.groups, self.group_planes * 2, L),
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=2,
        )

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.group_planes * 2, self.kernel_size, self.kernel_size
        )
        q_emb, k_emb, v_emb = torch.split(
            all_embeddings,
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=0,
        )

        qr = torch.einsum("bgci,cij->bgij", q, q_emb)
        kr = torch.einsum("bgci,cij->bgij", k, k_emb).transpose(2, 3)
        qk = torch.einsum("bgci,bgcj->bgij", q, k)

        stacked = torch.cat([qk, qr, kr], dim=1)
        stacked = (
            self.bn_similarity(stacked).view(N * S, 3, self.groups, L, L).sum(dim=1)
        )
        similarity = F.softmax(stacked, dim=3)
        sv = torch.einsum("bgij,bgcj->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_emb)
        out = torch.cat([sv, sve], dim=-1).view(N * S, self.out_planes * 2, L)
        out = self.bn_output(out).view(N, S, self.out_planes, 2, L).sum(dim=-2)

        if self.width:
            out = out.permute(0, 2, 1, 3)  # back to (N, C, H, W)
        else:
            out = out.permute(0, 2, 3, 1)
        return out


class AxialBlock(nn.Module):
    """Residual block: 1×1 down → height axial → width axial → 1×1 up."""

    def __init__(self, planes: int, groups: int, H: int, W: int) -> None:
        super().__init__()
        mid = planes
        self.conv_down = nn.Conv2d(planes, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.height_attn = AxialAttention(mid, mid, groups, kernel_size=H, width=False)
        self.width_attn = AxialAttention(mid, mid, groups, kernel_size=W, width=True)
        self.conv_up = nn.Conv2d(mid, planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv_down(x)))
        out = self.height_attn(out)
        out = self.width_attn(out)
        out = self.bn2(self.conv_up(out))
        return self.relu(out + identity)


class AxialLOB(nn.Module):
    """Axial-attention classifier over a (T × F) LOB window."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        channels = config.get("axial_channels", 32)
        groups = config.get("axial_groups", 8)
        n_blocks = config.get("axial_blocks", 2)
        drop = config.get("axial_dropout", 0.1)

        self.stem = nn.Conv2d(1, channels, 1)
        self.blocks = nn.ModuleList(
            AxialBlock(channels, groups, T, F_dim) for _ in range(n_blocks)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # (B, C, T, F)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)  # (B, C)
        return self.head(self.dropout(x))

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
