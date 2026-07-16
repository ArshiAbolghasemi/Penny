"""LinVAR — linear-recurrent (VAR-style) classifier with a softmax readout.

The "linear VAR baseline" from the deep-LOB literature (e.g. Sirignano & Cont, 2019,
*Universal features of price formation*): a **linear** recurrent state accumulates the
order-book feature stream, and a linear readout maps it to class logits. Trained
end-to-end on the trend labels with cross-entropy — same objective, harness, windows
and metrics as the neural models — so it is a directly comparable *linear* baseline
sitting between static ``LogReg`` and the nonlinear deep models.

    h_t   = A·h_{t-1} + B·X_t                     # linear recurrent state (no activation)
    logit = C·X_T     + D·h_T   → softmax(3)       # readout at the window's last step

The quote is binary ``P(price_t > 0)``; this repo's task is 3-class down/flat/up, so the
logistic link G becomes a softmax over 3 classes. No nonlinearity in the recurrence —
that is the point: it keeps the model linear-in-features.

Input : ``(B, 1, T_past, n_features)`` — single-channel image (shared dataset format).
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import count_parameters as count_parameters  # re-export


class LinVAR(nn.Module):
    """Linear recurrent (VAR-style) state + linear softmax readout."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        f = config["n_features"]
        h = config.get("linvar_hidden", 32)
        self.hidden = h
        self.A = nn.Linear(h, h, bias=False)  # h_{t-1} -> h_t   (state transition)
        self.B = nn.Linear(f, h, bias=True)  # X_t     -> h_t   (feature injection)
        self.C = nn.Linear(f, 3, bias=True)  # X_T     -> logit (direct readout)
        self.D = nn.Linear(h, 3, bias=False)  # h_T     -> logit (state readout)
        # small state-transition init so the 60-step linear unroll stays stable
        nn.init.uniform_(self.A.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) -> (B, T, F)
        x = x.squeeze(1)
        b, t, _ = x.shape
        h = x.new_zeros(b, self.hidden)
        for step in range(t):
            h = self.A(h) + self.B(x[:, step])  # linear recurrence, no activation
        return self.C(x[:, -1]) + self.D(h)  # readout at the last timestep

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
