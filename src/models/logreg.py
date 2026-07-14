"""LogReg — multinomial logistic regression on the flattened LOB window.

A trainable *linear* classifier: it flattens the ``(T_past, n_features)`` window to
a single vector and applies one affine layer to 3 class logits — no hidden layers, no
nonlinearity. Trained with the same cross-entropy / AdamW / cosine-schedule protocol
and evaluated on the same ``stride=1`` windows as the neural models, so it is a
directly comparable linear floor (the role ridge/linear plays in the FI-2010
benchmark, Ntakaris et al. 2018). Weight decay acts as the usual L2 regulariser.

Input : ``(B, 1, T_past, n_features)`` — single-channel image (shared dataset format).
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import count_parameters as count_parameters  # re-export


class LogReg(nn.Module):
    """Multinomial logistic regression: flatten window → single linear layer → logits."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        t_past = config["T_past"]
        n_features = config["n_features"]
        self.in_dim = t_past * n_features
        self.linear = nn.Linear(self.in_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) -> (B, T*F)
        return self.linear(x.reshape(x.size(0), -1))

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
