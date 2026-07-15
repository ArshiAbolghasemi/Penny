"""DLA — Dual-Stage Temporal Attention architecture for LOB data.

Based on: Guo & Chen, "Dual-Stage Temporal Attention" (2022), which applies the
dual-stage attention RNN (DA-RNN; Qin et al., 2017) to limit-order-book windows.

Two attention stages wrap an encoder/decoder LSTM pair:

  * **Stage 1 — input attention** (encoder): at every time step an attention over
    the ``F`` input (driving) series reweights the features before they enter the
    encoder LSTMCell, conditioned on the previous encoder hidden + cell state.
  * **Stage 2 — temporal attention** (decoder): a decoder LSTMCell attends over
    all encoder hidden states, forming a context vector each step; the final
    decoder hidden state feeds a linear head → 3 trend logits.

Input : ``(B, 1, T_past, n_features)`` — squeezed to ``(B, T, F)``.
Output: ``(B, 3)`` class logits  (0=down, 1=stationary, 2=up).

Config keys
-----------
dla_encoder_hidden  encoder LSTM hidden size (m)  (default 64)
dla_decoder_hidden  decoder LSTM hidden size (p)  (default 64)
dla_dropout         dropout before the head       (default 0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import count_parameters as count_parameters  # re-export


class InputAttentionEncoder(nn.Module):
    """Stage 1: LSTM encoder with per-timestep attention over input features."""

    def __init__(self, n_features: int, hidden: int, T: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.T = T
        self.lstm = nn.LSTMCell(n_features, hidden)
        self.W_e = nn.Linear(2 * hidden, T)  # from [h_{t-1}; s_{t-1}]
        self.U_e = nn.Linear(T, T, bias=False)  # applied to each driving series
        self.v_e = nn.Linear(T, 1, bias=False)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        b = x.shape[0]
        driving = x.permute(0, 2, 1)  # (B, F, T) — each feature's full series
        u_e = self.U_e(driving)  # (B, F, T), constant across the loop
        h = x.new_zeros(b, self.hidden)
        s = x.new_zeros(b, self.hidden)
        states = []
        alphas = [] if return_attn else None
        for t in range(self.T):
            hs = torch.cat([h, s], dim=1)  # (B, 2m)
            part1 = self.W_e(hs).unsqueeze(1)  # (B, 1, T)
            e = self.v_e(torch.tanh(part1 + u_e)).squeeze(-1)  # (B, F)
            alpha = torch.softmax(e, dim=1)  # (B, F)
            if return_attn:
                alphas.append(alpha)
            x_tilde = alpha * x[:, t, :]  # (B, F) attention-weighted input
            h, s = self.lstm(x_tilde, (h, s))
            states.append(h)
        enc = torch.stack(states, dim=1)  # (B, T, m)
        if return_attn:
            return enc, torch.stack(alphas, dim=1)  # (B, T, F)
        return enc


class TemporalAttentionDecoder(nn.Module):
    """Stage 2: decoder LSTM attending over encoder hidden states."""

    def __init__(self, enc_hidden: int, dec_hidden: int, T: int) -> None:
        super().__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.T = T
        self.lstm = nn.LSTMCell(enc_hidden, dec_hidden)
        self.W_d = nn.Linear(2 * dec_hidden, enc_hidden)  # from [d_{t-1}; s'_{t-1}]
        self.U_d = nn.Linear(enc_hidden, enc_hidden, bias=False)
        self.v_d = nn.Linear(enc_hidden, 1, bias=False)

    def forward(
        self, enc: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # enc: (B, T, m)
        b = enc.shape[0]
        u_d = self.U_d(enc)  # (B, T, m), constant across the loop
        d = enc.new_zeros(b, self.dec_hidden)
        s = enc.new_zeros(b, self.dec_hidden)
        betas = [] if return_attn else None
        for _ in range(self.T):
            ds = torch.cat([d, s], dim=1)  # (B, 2p)
            part1 = self.W_d(ds).unsqueeze(1)  # (B, 1, m)
            score = self.v_d(torch.tanh(part1 + u_d)).squeeze(-1)  # (B, T)
            beta = torch.softmax(score, dim=1)  # (B, T)
            if return_attn:
                betas.append(beta)
            context = (beta.unsqueeze(-1) * enc).sum(dim=1)  # (B, m)
            d, s = self.lstm(context, (d, s))
        if return_attn:
            # every decoder step's attention; step -1 feeds the head
            return d, torch.stack(betas, dim=1)  # (B, T_dec, T)
        return d  # (B, p) final decoder hidden state


class DLA(nn.Module):
    """Dual-Stage Temporal Attention: input-attention encoder → temporal decoder."""

    family = "classifier"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        enc_h = config.get("dla_encoder_hidden", 64)
        dec_h = config.get("dla_decoder_hidden", 64)
        drop = config.get("dla_dropout", 0.1)

        self.encoder = InputAttentionEncoder(F_dim, enc_h, T)
        self.decoder = TemporalAttentionDecoder(enc_h, dec_h, T)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(dec_h, 3)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trend logits; with ``return_attn`` also both attention stages.

        The attention dict holds ``"input"`` ``(B, T, F)`` — stage-1 weights over
        the input features at each timestep — and ``"temporal"``
        ``(B, T_dec, T)`` — stage-2 weights over encoder timesteps at each
        decoder step, the last of which produces the classified state.
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, T, F) → (B, T, F)
        if not return_attn:
            enc = self.encoder(x)  # (B, T, m)
            d = self.decoder(enc)  # (B, p)
            return self.head(self.dropout(d))
        enc, alpha = self.encoder(x, return_attn=True)
        d, beta = self.decoder(enc, return_attn=True)
        logits = self.head(self.dropout(d))
        return logits, {"input": alpha, "temporal": beta}

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        return self(batch["x"].to(device).float())
