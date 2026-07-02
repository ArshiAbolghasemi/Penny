"""Classifier-free guidance diffusion model for LOB trend prediction.

Trains a class-conditional 2D-conv UNet with condition dropout (p=0.15 by
default).  At inference, trend class is determined by MC likelihood ratio:

    log_ratio[y] = (1/K) Σ_k [ ‖ε - ε_θ(x_t,t,∅)‖² - ‖ε - ε_θ(x_t,t,y)‖² ]

This is the Bayes-optimal classifier under the DDPM generative model
(classifier-free guidance, Ho & Salimans 2022).  ``argmax(log_ratios)`` is the
predicted class; ``softmax(log_ratios)`` gives calibrated trend probabilities.

Architecture
------------
Identical UNet backbone to JointDiffusion (shared building blocks), plus a
learnable class embedding table (4 entries: down / stationary / up / null-∅)
whose output is *added* to the time embedding before injection into every
residual block.  No separate classifier head — classification is purely
post-hoc via the likelihood ratio.

Config keys
-----------
dc_base_channels  : base channel count for the UNet   (default 32)
dc_depth          : number of down/up stages           (default 2)
dc_time_emb       : time + class embedding dim         (default 128)
dc_p_uncond       : condition dropout prob at train    (default 0.15)
dc_mc_samples     : MC draws at test-time predict      (default 50)
dc_mc_val_samples : MC draws during validation         (default 10)
T_max             : total DDPM timesteps               (default 1000)
beta_start        : β₁                                 (default 1e-4)
beta_end          : βT                                 (default 0.02)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules import (
    Down,
    TimeDoubleConv,
    Up,
    count_parameters as count_parameters,  # re-export
    sinusoidal_embedding,
)


class DiffusionClassifier(nn.Module):
    """Class-conditional denoiser — classify trend via MC likelihood ratio.

    Class indices: 0 = down, 1 = stationary, 2 = up, 3 = null (∅).
    """

    family = "diffusion_classifier"
    N_CLASSES = 3
    NULL_CLASS = 3

    def __init__(self, config: dict) -> None:
        super().__init__()
        base = config.get("dc_base_channels", 32)
        depth = config.get("dc_depth", 2)
        temb_dim = config.get("dc_time_emb", 128)
        self.temb_dim = temb_dim
        self.p_uncond = config.get("dc_p_uncond", 0.15)
        self.mc_samples = config.get("dc_mc_samples", 50)
        self.mc_val_samples = config.get("dc_mc_val_samples", 10)

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.class_emb = nn.Embedding(self.N_CLASSES + 1, temb_dim)

        chans = [base * (2**i) for i in range(depth + 1)]
        self.stem = TimeDoubleConv(1, base, temb_dim)
        self.downs = nn.ModuleList(
            Down(chans[i], chans[i + 1], temb_dim) for i in range(depth)
        )
        self.ups = nn.ModuleList(
            Up(chans[i + 1], chans[i], chans[i], temb_dim)
            for i in reversed(range(depth))
        )
        self.out_conv = nn.Conv2d(base, 1, 1)

        # DDPM noise schedule stored as buffers so predict() needs no external scheduler
        T = config.get("T_max", 1000)
        betas = torch.linspace(
            config.get("beta_start", 1e-4),
            config.get("beta_end", 0.02),
            T,
            dtype=torch.float64,
        )
        alpha_bar = torch.cumprod(1.0 - betas, dim=0).float()
        self.register_buffer("_sqrt_ab", alpha_bar.sqrt())
        self.register_buffer("_sqrt_1mab", (1.0 - alpha_bar).sqrt())
        self._T = T

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise.

        Args:
            x_t: ``(B, 1, T_past, F)`` noised window.
            t:   ``(B,)`` integer timestep.
            y:   ``(B,)`` class index — 0/1/2 for trend labels, 3 for null (∅).

        Returns:
            ``eps_hat`` of shape ``(B, 1, T_past, F)``.
        """
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        temb = temb + self.class_emb(y)
        x = self.stem(x_t, temb)
        skips = [x]
        for down in self.downs:
            x = down(x, temb)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip, temb)
        return self.out_conv(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_noise(
        self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        extra = x0.dim() - 1
        sa = self._sqrt_ab[t].view(-1, *([1] * extra))
        sb = self._sqrt_1mab[t].view(-1, *([1] * extra))
        return sa * x0 + sb * eps

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        batch: dict,
        device: torch.device,
        mc_samples: int | None = None,
    ) -> torch.Tensor:
        """MC likelihood-ratio classification.

        For each of K MC draws: sample (t, ε), form x_t, run one unconditional
        pass and one batched conditional pass (all 3 classes stacked), accumulate
        ``err_uncond - err_cond`` per class.  Returns ``(B, 3)`` log-ratio
        logits (positive = more likely than the marginal).

        Args:
            batch:      Dict with key ``"x"`` of shape ``(B, 1, T, F)``.
            device:     Target device.
            mc_samples: Override ``dc_mc_samples`` from config.
        """
        K = mc_samples if mc_samples is not None else self.mc_samples
        x0 = batch["x"].to(device).float()
        B = x0.shape[0]
        log_ratios = torch.zeros(B, self.N_CLASSES, device=device)
        null_y = torch.full((B,), self.NULL_CLASS, dtype=torch.long, device=device)

        for _ in range(K):
            t = torch.randint(0, self._T, (B,), device=device)
            eps = torch.randn_like(x0)
            x_t = self._add_noise(x0, eps, t)

            eps_uncond = self(x_t, t, null_y)
            err_u = (eps - eps_uncond).pow(2).mean(dim=[1, 2, 3])  # (B,)

            # batch all 3 conditional passes together: (3B, ...)
            x_rep = x_t.repeat(3, 1, 1, 1)
            t_rep = t.repeat(3)
            y_cond = torch.cat(
                [
                    torch.full((B,), c, dtype=torch.long, device=device)
                    for c in range(self.N_CLASSES)
                ]
            )
            eps_rep = eps.repeat(3, 1, 1, 1)
            err_c = (eps_rep - self(x_rep, t_rep, y_cond)).pow(2).mean(dim=[1, 2, 3])
            err_c = err_c.view(self.N_CLASSES, B).T  # (B, 3)

            log_ratios += err_u.unsqueeze(1) - err_c

        return log_ratios / K  # (B, 3)
