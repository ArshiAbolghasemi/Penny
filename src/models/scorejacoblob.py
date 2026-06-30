"""ScoreJacobLOB — score-Jacobian-attention diffusion classifier for LOB trends.

A four-stage, single-phase model for mid-price trend prediction (down / flat / up):

  Stage 1  BiN              — adaptive bilinear normalisation along the time and
                              feature axes (removes price-vs-volume magnitude
                              disparity and non-stationarity).
  Stage 2  DiffBackbone2D   — a 2-D U-Net diffusion model with Conv2d kernels that
                              slide simultaneously across the time axis and the
                              feature/level axis, trained with the
                              v-parameterisation + min-SNR loss weighting.
  Stage 3  Score-Jacobian   — the input-space Jacobian of the score s_θ(x̂, t*)
                              is marginalised into a temporal saliency A_t ∈ ℝ^T
                              and a feature saliency A_f ∈ ℝ^F at K noise levels
                              (multi-scale).  The marginal column magnitudes of
                              ∂s/∂x are estimated by a Hutchinson random-projection
                              average: n_probes Rademacher cotangents v give VJPs
                              Jᵀv whose mean absolute value is, by the Khintchine
                              inequality, proportional to the per-input-element
                              Jacobian norm.  During training create_graph=True
                              keeps the VJPs inside the computation graph so
                              gradients flow back through them into the UNet and
                              BiN weights.
  Stage 4  Saliency head    — the bottleneck h (B, C_b, T_b, F_b) is average-
                              pooled (JointDiffusion-style) and concatenated with
                              an MLP-extracted feature from the Jacobian temporal
                              + feature saliencies, then a simple MLP classifies.

Training (single phase)
-----------------------
L_total = L_diff + γ · L_class

  L_diff   v-prediction MSE with min-SNR-γ weighting
  L_class  cross-entropy on the trend label
  γ        scalar (sjl_lambda_class, default 1.0)

No PCGrad, no gradient surgery, no stop-gradient anywhere.  Both losses
backpropagate normally through all shared weights every step.

Batch format: ``{"x": (B, 1, T, F), "label": int}`` — identical to every other
crypto model.  ``predict(batch, device) → (B, 3)`` logits.

Note on compute: extract_saliency executes K·n_probes vector-Jacobian products
per step (independent of T and F).  Raise sjl_n_probes for a lower-variance
saliency estimate, or shrink sjl_t_star, to trade accuracy for speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.jointdiff import _groups, sinusoidal_embedding


# ── Stage 1: adaptive Bilinear Normalisation ─────────────────────────────────


class BiN(nn.Module):
    """Bilinear normalisation: normalise across time and across features, then mix.

    Temporal branch z-scores each feature column across time (learnable per-feature
    γ_t, β_t); feature branch z-scores each time row across features (learnable
    per-time γ_f, β_f).  A softmax-mixed convex combination of the two branches is
    returned, so the layer learns how much temporal vs. cross-sectional
    normalisation each dataset needs.
    """

    def __init__(self, T: int, F_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma_t = nn.Parameter(torch.ones(F_dim))
        self.beta_t = nn.Parameter(torch.zeros(F_dim))
        self.gamma_f = nn.Parameter(torch.ones(T))
        self.beta_f = nn.Parameter(torch.zeros(T))
        self.mix = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        mt = x.mean(dim=1, keepdim=True)
        st = x.std(dim=1, keepdim=True) + self.eps
        xt = (x - mt) / st * self.gamma_t + self.beta_t

        mf = x.mean(dim=2, keepdim=True)
        sf = x.std(dim=2, keepdim=True) + self.eps
        xf = (x - mf) / sf * self.gamma_f[None, :, None] + self.beta_f[None, :, None]

        w = torch.softmax(self.mix, dim=0)
        return w[0] * xt + w[1] * xf


# ── Stage 2: 2D U-Net diffusion backbone ─────────────────────────────────────


class ResBlock2D(nn.Module):
    """Two 2-D convs + GroupNorm + SiLU with additive time embedding and skip."""

    def __init__(self, ci: int, co: int, temb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(co), co)
        self.temb = nn.Linear(temb_dim, co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(co), co)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Down2D(nn.Module):
    def __init__(self, ci: int, co: int, temb_dim: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(ci, ci, 2, stride=2)
        self.block = ResBlock2D(ci, co, temb_dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x), temb)


class Up2D(nn.Module):
    def __init__(self, ci: int, skip_ch: int, co: int, temb_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(ci, co, 1)
        self.block = ResBlock2D(co + skip_ch, co, temb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1), temb)


class DiffBackbone2D(nn.Module):
    """2-D U-Net over (time × feature) with a conv-only bottleneck (v-prediction).

    Like JointDiffusion's backbone: no bottleneck attention or channel-space MLP —
    the only learned mixing at the bottleneck is the conv stack in the down path.
    The cheap MLP feature extraction lives in the head, on the Jacobian saliencies.

    Input:  (B, 1, T, F) — single-channel LOB window
    Output: (B, 1, T, F) via forward_v, or (B, C_b, T_b, F_b) via bottleneck
    """

    def __init__(
        self,
        base: int,
        depth: int,
        temb_dim: int,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.temb_dim = temb_dim
        self.use_checkpoint = use_checkpoint
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        chans = [base * (2**i) for i in range(depth + 1)]
        self.bottleneck_ch = chans[-1]
        self.stem = ResBlock2D(1, base, temb_dim)
        self.downs = nn.ModuleList(
            Down2D(chans[i], chans[i + 1], temb_dim) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up2D(chans[i + 1], chans[i], chans[i], temb_dim)
            for i in reversed(range(depth))
        )
        self.out = nn.Conv2d(base, 1, 1)

    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.temb_dim))

    def _run(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        """Run a sub-block, gradient-checkpointed while training to cap memory.

        ``use_reentrant=False`` is mandatory here: the score-Jacobian VJPs call
        backward with ``create_graph=True``, so the checkpointed region must
        support higher-order (double) backward.
        """
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _encode(self, x: torch.Tensor, temb: torch.Tensor):
        h = self._run(self.stem, x, temb)  # (B, base, T, F)
        skips = [h]
        for down in self.downs:
            h = self._run(down, h, temb)
            skips.append(h)
        return h, skips

    def bottleneck(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the down-path bottleneck (B, C_b, T_b, F_b)."""
        return self._encode(x, self._temb(t))[0]

    def forward_v(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict v-parameterised target, shape (B, 1, T, F)."""
        temb = self._temb(t)
        h, skips = self._encode(x, temb)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            h = self._run(up, h, skip, temb)
        return self.out(h)


# ── ScoreJacobLOB ────────────────────────────────────────────────────────────


class ScoreJacobLOB(nn.Module):
    """Score-Jacobian-attention diffusion classifier (see module docstring)."""

    family = "score_jacob"

    def __init__(self, config: dict) -> None:
        super().__init__()
        T = config["T_past"]
        F_dim = config["n_features"]
        self.T = T
        self.F = F_dim

        base = config.get("sjl_base_channels", 64)
        depth = config.get("sjl_depth", 2)
        temb_dim = config.get("sjl_time_emb", 128)
        hidden = config.get("sjl_head_hidden", 128)

        self.bin = BiN(T, F_dim)
        self.backbone = DiffBackbone2D(
            base,
            depth,
            temb_dim,
            use_checkpoint=config.get("sjl_grad_checkpoint", True),
        )

        self.t_star = list(config.get("sjl_t_star", [20, 50, 100]))
        self.repr_t = int(config.get("sjl_repr_t", 0))
        self.n_probes = int(config.get("sjl_n_probes", 8))
        self.min_snr_gamma = float(config.get("sjl_min_snr_gamma", 5.0))
        self.lambda_class = float(config.get("sjl_lambda_class", 1.0))

        T_max = config.get("T_max", 1000)
        betas = torch.linspace(
            config.get("beta_start", 1e-4),
            config.get("beta_end", 0.02),
            T_max,
            dtype=torch.float64,
        )
        ab = torch.cumprod(1.0 - betas, dim=0).float()
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1mab", (1.0 - ab).sqrt())

        C_b = self.backbone.bottleneck_ch
        # MLP feature extractor over the Jacobian temporal+feature saliencies.
        self.sal_mlp = nn.Sequential(
            nn.Linear(T + F_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Simple classifier over [pooled bottleneck ‖ saliency features].
        self.head = nn.Sequential(
            nn.Linear(C_b + hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.get("sjl_dropout", 0.1)),
            nn.Linear(hidden, 3),
        )

    # ── diffusion math ──────────────────────────────────────────────────────

    def _ab(self, t: torch.Tensor, ndim: int = 4):
        a = self.sqrt_ab[t]
        s = self.sqrt_1mab[t]
        view = (-1,) + (1,) * (ndim - 1)
        return a.view(*view), s.view(*view)

    def diffusion_loss(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """v-prediction MSE with min-SNR-γ weighting. x0: (B, 1, T, F)."""
        a, s = self._ab(t)
        x_t = a * x0 + s * noise
        v_pred = self.backbone.forward_v(x_t, t)
        v_tgt = a * noise - s * x0
        snr = (a[:, 0, 0, 0] ** 2) / (s[:, 0, 0, 0] ** 2 + 1e-8)  # (B,)
        w = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)
        mse = ((v_pred - v_tgt) ** 2).mean(dim=(1, 2, 3))  # (B,)
        return (w * mse).mean()

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score s_θ(x, t) = ∇_x log p_t(x) from v-prediction. x: (B, 1, T, F)."""
        a, s = self._ab(t)
        v = self.backbone.forward_v(x, t)
        eps = s * x + a * v
        return -eps / s

    # ── Stage 3: score-Jacobian saliency ────────────────────────────────────

    def extract_saliency(self, x_hat_2d: torch.Tensor):
        """Marginal temporal/feature saliencies from the input-space score Jacobian.

        Args:
            x_hat_2d: BiN-normalised input, shape (B, 1, T, F).

        Returns:
            (wt (B, T), wf (B, F)) normalised to mean 1 per sample.

        During training (self.training=True) the VJPs are computed with
        create_graph=True so gradients flow back through them into the backbone
        and BiN weights.  During inference create_graph=False saves memory.
        """
        B, _, T, Fd = x_hat_2d.shape
        sens = x_hat_2d.new_zeros(B, 1, T, Fd)  # mean |∂s/∂x| over probes/levels

        with torch.enable_grad():
            # If x_hat_2d has no grad tracking (e.g. inference under no_grad),
            # create a detached leaf so VJPs can still be computed.
            if not x_hat_2d.requires_grad:
                x = x_hat_2d.detach().requires_grad_(True)
                create_graph = False
            else:
                x = x_hat_2d
                create_graph = self.training

            for ts in self.t_star:
                t = torch.full((B,), ts, dtype=torch.long, device=x.device)
                s = self.score(x, t)  # (B, 1, T, F)
                for _ in range(self.n_probes):
                    # Rademacher cotangent v ∈ {-1, +1}; Jᵀv has the shape of x.
                    v = torch.randint(0, 2, s.shape, device=s.device).to(s.dtype)
                    v = v * 2.0 - 1.0
                    (g,) = torch.autograd.grad(
                        s,
                        x,
                        grad_outputs=v,
                        retain_graph=True,
                        create_graph=create_graph,
                    )
                    sens = sens + g.abs()

        sens = sens / (len(self.t_star) * self.n_probes)
        wt = sens.sum(dim=(1, 3))  # marginalise over channel + feature → (B, T)
        wf = sens.sum(dim=(1, 2))  # marginalise over channel + time    → (B, F)
        wt = wt / (wt.mean(dim=1, keepdim=True) + 1e-8)
        wf = wf / (wf.mean(dim=1, keepdim=True) + 1e-8)
        return wt, wf

    # ── Stage 4: saliency-MLP classification head ───────────────────────────

    def head_logits(
        self, wt: torch.Tensor, wf: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        # h: (B, C_b, T_b, F_b); wt: (B, T); wf: (B, F)
        pooled = h.mean(dim=(-2, -1))  # (B, C_b) — JointDiffusion-style pooling
        sal = self.sal_mlp(torch.cat([wt, wf], dim=1))  # (B, hidden)
        return self.head(torch.cat([pooled, sal], dim=1))  # (B, 3)

    def predict(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Trend logits from a raw batch dict. Accepts ``{"x": (B,1,T,F), ...}``."""
        x = batch["x"].to(device).float()
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, T, F)
        x_hat = self.bin(x)
        x_hat_2d = x_hat.unsqueeze(1)  # (B, 1, T, F)
        wt, wf = self.extract_saliency(x_hat_2d)
        t = torch.full((x.shape[0],), self.repr_t, dtype=torch.long, device=device)
        h = self.backbone.bottleneck(x_hat_2d, t)
        return self.head_logits(wt, wf, h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
