"""Train ScoreJacobLOB on a single crypto symbol — single combined phase.

A single training loop optimises BiN, the 2D UNet, the Jacobian extraction path,
and the classification head end-to-end with:

    L_total = L_diff + γ · L_class

where L_diff is the v-prediction diffusion loss (min-SNR weighted) and L_class is
cross-entropy.  γ is set via sjl_lambda_class (default 1.0).  Both losses
backpropagate through all shared weights every step; no stop-gradient anywhere.

The Jacobian saliency runs K·n_probes vector-Jacobian products per step
(K = len(sjl_t_star), n_probes = sjl_n_probes), independent of T and F.  Raise
sjl_n_probes for a lower-variance estimate, or shrink sjl_t_star, to trade
accuracy for speed.

Usage::

    uv run python -m crypto.train_scorejacoblob configs/crypto/binance/scorejacoblob/btcusdt_ofi.json
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
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from models.scorejacoblob import ScoreJacobLOB, count_parameters
from utils.evaluate import run_test
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


# ── combined training epoch ───────────────────────────────────────────────────


def _train_epoch(model, loader, optimizer, scheduler, device, t_max, grad_clip):
    model.train()
    total_loss = total_diff = total_cls = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()
        label = batch["label"].to(device)
        if x0.dim() == 4:
            x0 = x0.squeeze(1)  # (B, T, F)
        B = x0.shape[0]

        x_hat = model.bin(x0)  # (B, T, F)
        x_hat_2d = x_hat.unsqueeze(1)  # (B, 1, T, F)

        # diffusion loss on noisy input
        t = torch.randint(0, t_max, (B,), device=device)
        noise = torch.randn_like(x_hat_2d)
        L_diff = model.diffusion_loss(x_hat_2d, t, noise)

        # classification via score-Jacobian attention
        wt, wf = model.extract_saliency(x_hat_2d)
        t_repr = torch.full((B,), model.repr_t, dtype=torch.long, device=device)
        h = model.backbone.bottleneck(x_hat_2d, t_repr)
        logits = model.head_logits(wt, wf, h)
        L_class = F.cross_entropy(logits, label)

        L_total = L_diff + model.lambda_class * L_class

        optimizer.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += L_total.item()
        total_diff += L_diff.item()
        total_cls += L_class.item()
        n += 1

    denom = max(n, 1)
    return total_loss / denom, total_diff / denom, total_cls / denom


# ── validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _validate(model, loader, device, t_max):
    model.eval()
    diff_total = 0.0
    n = 0
    y_true, y_pred = [], []
    for batch in loader:
        x0 = batch["x"].to(device).float()
        if x0.dim() == 4:
            x0 = x0.squeeze(1)
        B = x0.shape[0]

        x_hat = model.bin(x0)
        x_hat_2d = x_hat.unsqueeze(1)

        # diffusion loss (no gradient needed)
        t = torch.randint(0, t_max, (B,), device=device)
        noise = torch.randn_like(x_hat_2d)
        diff_total += model.diffusion_loss(x_hat_2d, t, noise).item()

        # classification — extract_saliency internally re-enables grad for VJPs
        wt, wf = model.extract_saliency(x_hat_2d)
        t_repr = torch.full((B,), model.repr_t, dtype=torch.long, device=device)
        h = model.backbone.bottleneck(x_hat_2d, t_repr)
        logits = model.head_logits(wt, wf, h)

        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
        n += 1

    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    acc = float((torch.tensor(y_true) == torch.tensor(y_pred)).float().mean())
    return f1, acc, diff_total / max(n, 1)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/binance/scorejacoblob/btcusdt_ofi.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    seed = resolve_seed(config)
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    grad_clip = config.get("grad_clip", 1.0)
    t_max = config.get("T_max", 1000)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"scorejacoblob_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    model = ScoreJacobLOB(config).to(device)
    logger.info(
        "ScoreJacobLOB  symbol={}  mode={}",
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
    logger.info(
        "  features={}  params={:.2f}M  t_star={}  K={}  λ_class={}",
        meta["n_features"],
        count_parameters(model) / 1e6,
        model.t_star,
        len(model.t_star),
        model.lambda_class,
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

    epochs = config.get("epochs", 100)
    patience_limit = config.get("patience", 15)
    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = build_cosine_schedule(optimizer, config, epochs * len(train_loader))

    logger.info("── training ({} epochs, patience={}) ──", epochs, patience_limit)
    history = []
    best_f1, patience = -1.0, 0

    for epoch in range(epochs):
        tr_total, tr_diff, tr_cls = _train_epoch(
            model, train_loader, optimizer, scheduler, device, t_max, grad_clip
        )
        val_f1, val_acc, val_diff = _validate(model, val_loader, device, t_max)
        logger.info(
            "  epoch {:3d} | train={:.4f} (diff={:.4f} cls={:.4f})"
            " | val_f1={:.4f} val_acc={:.4f} val_diff={:.4f}",
            epoch,
            tr_total,
            tr_diff,
            tr_cls,
            val_f1,
            val_acc,
            val_diff,
        )
        history.append(
            {
                "epoch": epoch,
                "train": tr_total,
                "train_diff": tr_diff,
                "train_cls": tr_cls,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_diff": val_diff,
            }
        )

        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
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
            if patience >= patience_limit:
                logger.info("  early stop at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("best val macro-F1 = {:.4f}", best_f1)
    run_test(model, test_ds, config, device)


if __name__ == "__main__":
    main()
