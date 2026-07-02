"""DLA — DeepLOB-Attention classifier for LOB data.

Based on: Zhang & Zohren, "Multi-Horizon Forecasting for Limit Order Books"
(arXiv:2105.10430) — the DeepLOB encoder followed by an attention mechanism.

This is the single-horizon (classification) variant: the DeepLOB convolutional +
inception feature extractor feeds an LSTM whose *full* output sequence is pooled
by an additive (Bahdanau-style) temporal attention head instead of taking only
the last hidden state.

Input : ``(B, 1, T_past, n_features)`` — single-channel image.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Config keys
-----------
dla_conv_filters       conv-block channels        (default 32)
dla_inception_filters  per-path inception filters (default 64)
dla_lstm_hidden        LSTM hidden size           (default 64)
dla_dropout            dropout before the head    (default 0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.deeplob import _ConvBlock, _InceptionBlock
from models.modules import count_parameters as count_parameters  # re-export


class DLA(nn.Module):
    """DeepLOB encoder → LSTM → additive temporal attention → 3-class logits."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        conv_f = config.get("dla_conv_filters", 32)
        inc_f = config.get("dla_inception_filters", 64)
        lstm_h = config.get("dla_lstm_hidden", 64)
        drop = config.get("dla_dropout", 0.1)

        self.conv_block = _ConvBlock(conv_f)
        self.inception = _InceptionBlock(conv_f, inc_f)
        self.feat_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.lstm = nn.LSTM(
            input_size=3 * inc_f, hidden_size=lstm_h, num_layers=1, batch_first=True
        )
        # Additive attention over the LSTM output sequence.
        self.attn_w = nn.Linear(lstm_h, lstm_h)
        self.attn_v = nn.Linear(lstm_h, 1, bias=False)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(lstm_h, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)  # (B, conv_f, T-6, F+1)
        x = self.inception(x)  # (B, 3*inc_f, T-6, F+1)
        x = self.feat_pool(x).squeeze(-1)  # (B, 3*inc_f, T-6)
        x = x.permute(0, 2, 1)  # (B, T-6, 3*inc_f)
        seq, _ = self.lstm(x)  # (B, T-6, H)  full sequence
        scores = self.attn_v(torch.tanh(self.attn_w(seq)))  # (B, T-6, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T-6, 1)
        ctx = (weights * seq).sum(dim=1)  # (B, H)
        return self.head(self.dropout(ctx))

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
