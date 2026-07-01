"""Train JointDiffusionDF as a *consistency model* with Diffusion Forcing.

Combines Consistency Training (no teacher/distillation) with Diffusion Forcing:
every timestep of the window gets its *own independent* pair of adjacent Karras
noise levels, and the network is trained to be self-consistent per timestep while
jointly classifying the trend:

    L = L_consistency + lambda_trend * CE(logits, label)

    L_consistency = mean_over(B,T) [ lambda(sigma_n) * PseudoHuber(
                        f_theta (x0 + sigma_{n+1} z, sigma_{n+1}),
                        f_theta-(x0 + sigma_n     z, sigma_n) ) ]

with per-timestep sigma pairs, shared noise z, and ``theta-`` = online weights with
stop-gradient (improved CT).  Classification logits come from the online pass.

Usage::

    uv run python -m crypto.train_jointdiffdf configs/crypto/nobitex/jointdiffdf/btcirt_ofi_k10.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from models.consistency import (
    interval_weights,
    karras_sigmas,
    pseudo_huber_const,
)
from models.jointdiffdf import JointDiffusionDF, count_parameters
from utils.evaluate import run_test
from utils.flops import log_gflops
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


def _train_epoch(model, loader, optimizer, lr_sched, config, device, cm):
    model.train()
    lam_cls = config.get("lambda_trend", 1.0)
    grad_clip = config.get("grad_clip", 1.0)
    sigmas, weights, huber_c = cm["sigmas"], cm["weights"], cm["huber_c"]
    tot = con = cls = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b, _, T, _ = x0.shape

        # Diffusion Forcing: independent adjacent Karras pair per (sample, timestep).
        idx = torch.multinomial(weights, b * T, replacement=True).view(b, T)
        sig_lo, sig_hi = sigmas[idx], sigmas[idx + 1]  # (B, T)
        z = torch.randn_like(x0)
        x_hi = x0 + sig_hi[:, None, :, None] * z
        x_lo = x0 + sig_lo[:, None, :, None] * z

        x0_hi, logits = model.denoise(x_hi, sig_hi)  # online (higher noise)
        with torch.no_grad():
            x0_lo, _ = model.denoise(x_lo, sig_lo)  # target (stop-grad, same weights)

        # Pseudo-Huber per timestep (reduce over channel + feature axes).
        se = ((x0_hi - x0_lo) ** 2).sum(dim=(1, 3))  # (B, T)
        d = torch.sqrt(se + huber_c**2) - huber_c  # (B, T)
        lam = 1.0 / (sig_hi - sig_lo)  # (B, T)
        con_loss = (lam * d).mean()
        cls_loss = F.cross_entropy(logits, label)
        loss = con_loss + lam_cls * cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()
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
        default="configs/crypto/nobitex/jointdiffdf/btcirt_ofi_k10.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    config["cm_enabled"] = True  # consistency-model predict path

    seed = resolve_seed(config)
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jointdiffdf_cm_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info(
        "JointDiffusionDF (consistency + diffusion forcing)  symbol={}  mode={}",
        config["symbol"],
        config.get("feature_mode"),
    )
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  alpha={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = JointDiffusionDF(config).to(device)

    n_scales = config.get("cm_num_scales", 40)
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
    # Per-timestep Pseudo-Huber: each timestep slice has n_features elements.
    huber_c = pseudo_huber_const(config["n_features"])
    cm = {"sigmas": sigmas, "weights": weights, "huber_c": huber_c}

    gflops = log_gflops(model, train_ds, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  N_scales={}  lambda_trend={}  device={}",
        count_parameters(model) / 1e6,
        gflops,
        n_scales,
        config.get("lambda_trend", 1.0),
        device,
    )

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

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    lr_sched = build_cosine_schedule(
        optimizer, config, config["epochs"] * len(train_loader)
    )

    best, patience, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        tr = _train_epoch(model, train_loader, optimizer, lr_sched, config, device, cm)
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | total={:.4f} consistency={:.4f} trend={:.4f}"
            " | val_ce={:.4f} acc={:.4f}",
            epoch,
            tr["total"],
            tr["consistency"],
            tr["trend"],
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
