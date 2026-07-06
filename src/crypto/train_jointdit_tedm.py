"""Train JointDiT as a **t-EDM** consistency model (heavy-tailed CT, no teacher).

Heavy-tailed variant of ``crypto.train_jointdit_cm``: the Gaussian diffusion core
is replaced by the **t-EDM** formulation (Pandey et al. 2025, arXiv:2410.14171) so
the generative branch models the extreme moves / heavy tails of 10 ms BTC LOB
returns.  Everything else — the DiT backbone, the trend head, low-noise-only
classification, the EMA target, annealing ``N(k)``, and the Kendall-Gal
uncertainty weighting — is identical to the Gaussian consistency trainer.

The single change is the noise model, via the Gaussian scale-mixture identity:
a multivariate Student-t is a Gaussian scaled by ``sqrt(kappa)`` with
``kappa ~ Inverse-Gamma(nu/2, nu/2)`` drawn once per sample.  Concretely:

  * **Forward kernel** ``x_t = x0 + sigma·sqrt(kappa)·z`` — the EDM sigma schedule
    is unchanged; only the additive-noise *distribution* becomes Student-t.
  * **Preconditioning** ``models.consistency.precond`` picks up ``kappa`` (the
    ``sigma^2 -> kappa·sigma^2`` substitution); ``c_noise`` still embeds the
    schedule ``sigma``.  At ``sigma_min`` the boundary ``f(x, sigma_min) = x`` holds
    for any ``kappa``, so the classification-inference path (denoise at ``sigma_min``)
    is unaffected by the tail.
  * **Consistency pairing** the scale ``kappa`` is sampled **once per sample and
    shared** across both points ``(sigma_n, sigma_{n+1})`` of the trajectory (as is
    the noise ``z``).  Resampling ``kappa`` per point would place the two points on
    *different* t-trajectories and collapse training.

The consistency distance metric stays the (already outlier-robust) Pseudo-Huber
loss.  ``nu`` is the single tail-control knob (config ``tedm_nu`` or ``--nu``):
large ``nu`` recovers Gaussian-EDM behaviour exactly, smaller ``nu`` ⇒ heavier
tails.  Inference reads the trend logits from the denoised (``sigma_min``) pass.

Usage::

    uv run python -m crypto.train_jointdit_tedm configs/crypto/nobitex/jointdit/btcirt_ofi.json --nu 5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from models.consistency import (
    discretization_n,
    interval_weights,
    karras_sigmas,
    pseudo_huber_const,
    sample_kappa,
)
from models.jointdit import JointDiT, count_parameters
from utils.evaluate import run_test
from utils.flops import log_gflops
from utils.training import (
    build_cosine_schedule,
    measure_sigma_data,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


class UncertaintyWeighting(nn.Module):
    """Kendall & Gal (2018) homoscedastic multi-task weighting.

    Combines ``losses = [L_consistency, L_ce]`` as ``sum_i 0.5·e^{-s_i}·L_i +
    0.5·s_i`` with learnable log-variances ``s_i``; its parameters are optimised
    jointly with the network.
    """

    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = losses[0].new_zeros(())
        for i, loss in enumerate(losses):
            total = (
                total + 0.5 * torch.exp(-self.log_var[i]) * loss + 0.5 * self.log_var[i]
            )
        return total


@torch.no_grad()
def _ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def _sigma_grid(n_scales: int, config: dict, device: torch.device):
    """Karras sigmas + lognormal interval sampling weights for ``n_scales`` levels."""
    sigmas = karras_sigmas(
        n_scales,
        config.get("cm_sigma_min", 0.002),
        config.get("cm_sigma_max", 80.0),
        config.get("cm_rho", 7.0),
        device=device,
    )
    weights = interval_weights(
        sigmas, config.get("cm_p_mean", -1.1), config.get("cm_p_std", 2.0)
    )
    return sigmas, weights


def _train_epoch(model, ema, mtl, loader, optimizer, lr_sched, config, device, cm):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    ema_decay = config.get("cm_ema_decay", 0.999)
    sigma_data = model.sigma_data
    huber_c = cm["huber_c"]
    nu = cm["nu"]
    s0 = config.get("cm_s0", 10)
    s1 = config.get("cm_s1", 1280)
    tot = con = cls = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]

        # Annealing N(k): recompute the Karras grid whenever the step count crosses
        # a doubling boundary (so adjacent-sigma gaps shrink late in training).
        n_scales = discretization_n(cm["step"], cm["total_steps"], s0, s1)
        if n_scales != cm["n_scales"]:
            cm["sigmas"], cm["weights"] = _sigma_grid(n_scales, config, device)
            cm["n_scales"] = n_scales
        sigmas, weights = cm["sigmas"], cm["weights"]

        # Adjacent Karras levels per sample, shared noise z (the shared-noise trick).
        idx = torch.multinomial(weights, b, replacement=True)  # in [0, N-2]
        sig_lo, sig_hi = sigmas[idx], sigmas[idx + 1]  # (B,)
        z = torch.randn_like(x0)
        v = (-1,) + (1,) * (x0.dim() - 1)

        # t-EDM: one Student-t scale kappa per sample, SHARED across both points of
        # the consistency pair (like z) so they stay on the same t-trajectory.
        kappa = sample_kappa(nu, (b,), device=device)  # (B,)
        kroot = (kappa**0.5).view(v)  # (B,1,1,1)
        x_hi = x0 + sig_hi.view(v) * kroot * z
        x_lo = x0 + sig_lo.view(v) * kroot * z

        x0_hi, logits = model.denoise(x_hi, sig_hi, kappa)  # online (higher noise)
        with torch.no_grad():
            x0_lo, _ = ema.denoise(x_lo, sig_lo, kappa)  # EMA target (lower noise)

        diff = (x0_hi - x0_lo).flatten(1)
        d = torch.sqrt((diff**2).sum(1) + huber_c**2) - huber_c  # Pseudo-Huber (B,)
        lam = 1.0 / (sig_hi - sig_lo)  # (B,)
        con_loss = (lam * d).mean()

        # Classification only at low noise (sigma < sigma_data) to avoid injecting
        # noisy gradients into the trend head.
        low = sig_hi < sigma_data
        if low.any():
            ce = F.cross_entropy(logits, label, reduction="none")
            cls_loss = ce[low].mean()
        else:
            cls_loss = logits.new_zeros(())
        loss = mtl([con_loss, cls_loss])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()
        _ema_update(ema, model, ema_decay)
        cm["step"] += 1
        tot += loss.item()
        con += con_loss.item()
        cls += cls_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "consistency": con / n, "trend": cls / n}


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    ce, correct, n = 0.0, 0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = model.predict(batch, device)
        ce += F.cross_entropy(logits, label).item()
        correct += (logits.argmax(1) == label).sum().item()
        n += len(label)
    return ce / max(len(loader), 1), correct / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/nobitex/jointdit/btcirt_ofi.json",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=None,
        help="Student-t degrees of freedom (override config['tedm_nu']); "
        "large ⇒ Gaussian EDM, smaller ⇒ heavier tails",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    config["cm_enabled"] = True  # consistency predict path (denoise at sigma_min)
    nu = args.nu if args.nu is not None else float(config.get("tedm_nu", 5.0))
    config["tedm_nu"] = nu

    seed = resolve_seed(config)
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jointdit_tedm_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    # EDM sigma_data is calibrated to the true std of the LOB windows.  Measure it
    # from the training set (auto by default) rather than trusting the config guess.
    if config.get("cm_sigma_data_auto", True):
        config["cm_sigma_data"] = measure_sigma_data(train_ds)
    logger.info(
        "  cm_sigma_data={:.4f} (measured from train windows)", config["cm_sigma_data"]
    )

    logger.info(
        "JointDiT (t-EDM consistency)  symbol={}  mode={}  nu={}",
        config["symbol"],
        config.get("feature_mode"),
        nu,
    )
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  alpha={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = JointDiT(config).to(device)
    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    mtl = UncertaintyWeighting(2).to(device)

    nw = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    total_steps = config["epochs"] * len(train_loader)
    s0 = config.get("cm_s0", 10)
    n0 = discretization_n(0, total_steps, s0, config.get("cm_s1", 1280))
    sigmas, weights = _sigma_grid(n0, config, device)
    cm = {
        "sigmas": sigmas,
        "weights": weights,
        "n_scales": n0,
        "huber_c": pseudo_huber_const(config["T_past"] * config["n_features"]),
        "step": 0,
        "total_steps": total_steps,
        "nu": nu,
    }

    gflops = log_gflops(model, train_ds, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  N(0)={}→{}  ema_decay={}  device={}",
        count_parameters(model) / 1e6,
        gflops,
        n0,
        config.get("cm_s1", 1280) + 1,
        config.get("cm_ema_decay", 0.999),
        device,
    )

    optimizer = AdamW(
        list(model.parameters()) + list(mtl.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    lr_sched = build_cosine_schedule(optimizer, config, total_steps)

    best, patience, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        tr = _train_epoch(
            model, ema, mtl, train_loader, optimizer, lr_sched, config, device, cm
        )
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | total={:.4f} consistency={:.4f} trend={:.4f}"
            " | N={} val_ce={:.4f} acc={:.4f}",
            epoch,
            tr["total"],
            tr["consistency"],
            tr["trend"],
            cm["n_scales"],
            val_ce,
            val_acc,
        )
        history.append({"epoch": epoch, **tr, "val_ce": val_ce, "val_acc": val_acc})

        if val_ce < best:
            best, patience = val_ce, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "alpha": alpha,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("early stopping at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    run_test(model, test_ds, config, device)


if __name__ == "__main__":
    main()
