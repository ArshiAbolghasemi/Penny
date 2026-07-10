"""DLA's own explanation: its two built-in attention stages.

``models.dla.DLA`` is *designed* to be interpretable — it is literally the
Dual-Stage Attention RNN (Qin et al., 2017) applied to LOB windows. Both
stages already compute real softmax attention inside their forward loops:

  * Stage 1 (``InputAttentionEncoder``, dla.py:57-58): ``alpha (B, F)`` per
    timestep — which input features/levels the encoder weights before
    feeding its LSTMCell.
  * Stage 2 (``TemporalAttentionDecoder``, dla.py:87-88): ``beta (B, T)`` per
    decoder step — which encoder timesteps the decoder's context vector
    draws from.

Neither is exposed by the public ``forward()`` (the loops only keep the
LSTM states, discarding alpha/beta each iteration), so this module
re-implements the two forward loops verbatim to also collect the attention
tensors, rather than patching the trained layers. This is not an approximation:
it is the exact same arithmetic the model already runs, just also returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.dla import DLA


@dataclass
class DLAExplanation:
    alpha: torch.Tensor  # (B, T, F) stage-1 input attention, per encoder timestep
    beta: torch.Tensor  # (B, T) stage-2 temporal attention, encoder steps attended to


@torch.no_grad()
def extract_dla_attention(model: DLA, x: torch.Tensor) -> DLAExplanation:
    """Run DLA's two attention stages, collecting alpha/beta at every step.

    Args:
        model: A loaded ``DLA`` instance.
        x:     ``(B, 1, T, F)`` or ``(B, T, F)`` input windows.
    """
    if x.dim() == 4:
        x = x.squeeze(1)  # (B, T, F)
    enc = model.encoder
    dec = model.decoder

    b, T, _F = x.shape
    driving = x.permute(0, 2, 1)  # (B, F, T)
    u_e = enc.U_e(driving)  # (B, F, T)
    h = x.new_zeros(b, enc.hidden)
    s = x.new_zeros(b, enc.hidden)
    enc_states = []
    alphas = []
    for t in range(enc.T):
        hs = torch.cat([h, s], dim=1)
        part1 = enc.W_e(hs).unsqueeze(1)
        e = enc.v_e(torch.tanh(part1 + u_e)).squeeze(-1)  # (B, F)
        alpha = torch.softmax(e, dim=1)
        alphas.append(alpha)
        x_tilde = alpha * x[:, t, :]
        h, s = enc.lstm(x_tilde, (h, s))
        enc_states.append(h)
    enc_out = torch.stack(enc_states, dim=1)  # (B, T, m)
    alpha_all = torch.stack(alphas, dim=1)  # (B, T, F)

    u_d = dec.U_d(enc_out)  # (B, T, m)
    d = enc_out.new_zeros(b, dec.dec_hidden)
    s = enc_out.new_zeros(b, dec.dec_hidden)
    betas = []
    for _ in range(dec.T):
        ds = torch.cat([d, s], dim=1)
        part1 = dec.W_d(ds).unsqueeze(1)
        score = dec.v_d(torch.tanh(part1 + u_d)).squeeze(-1)  # (B, T)
        beta = torch.softmax(score, dim=1)
        betas.append(beta)
        context = (beta.unsqueeze(-1) * enc_out).sum(dim=1)
        d, s = dec.lstm(context, (d, s))

    # DLA's decoder runs T steps but only the *final* hidden state feeds the
    # head (dla.py:116-117); the final step's beta is therefore the one that
    # actually influenced the logits — earlier steps' betas describe
    # intermediate decoder states that get discarded.
    beta_final = betas[-1]  # (B, T)

    return DLAExplanation(alpha=alpha_all, beta=beta_final)
