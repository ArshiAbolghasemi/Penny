"""Training entry point for the painting (image inpainting) approach.

Usage::

    uv run python -m painting.train configs/painting/ofi.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import evaluate as ev
from . import labels as lab
from .dataset import build_datasets
from .diffusion import Diffusion
from .model import TrendHead, build_model, count_parameters


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.warning("cuda unavailable; using mps")
            return torch.device("mps")
        logger.warning("cuda unavailable; using cpu")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("mps unavailable; using cpu")
        return torch.device("cpu")
    return torch.device(requested)


def build_optimizer_scheduler(model, trend_head, config, total_steps):
    optimizer = AdamW(
        list(model.parameters()) + list(trend_head.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    warmup = config["warmup_steps"]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, LambdaLR(optimizer, lr_lambda)


def _masked_loss(eps_hat, noise, mask):
    m = mask.expand_as(eps_hat)
    return ((eps_hat - noise) ** 2 * m).sum() / m.sum().clamp(min=1)


def train_one_epoch(
    model, trend_head, diffusion, loader, optimizer, scheduler, config, gamma, device
):
    model.train()
    trend_head.train()
    t_max, k, lam = config["T_max"], config["label_k"], config["lambda_trend"]
    tot = dif = trd = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        bwd = batch["bwd_smoothed"].to(device).float()
        history = image * (1.0 - mask)
        b = image.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(image)
        x_t = diffusion.q_sample(image, t, noise)
        eps_hat = model.predict_noise(x_t, t, history, mask)
        diff_loss = _masked_loss(eps_hat, noise, mask)

        x0_hat = diffusion._x0_from_eps(x_t, eps_hat, t)
        fut_mid = model.future_mid(x0_hat, batch, gamma)
        l_pred = (fut_mid[:, :k].mean(dim=1) - bwd) / (bwd + 1e-12)
        ce = F.cross_entropy(trend_head(l_pred), label, reduction="none")
        w = (1.0 - t.float() / t_max) ** 2
        loss = diff_loss + lam * (w * ce).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(trend_head.parameters()),
            config["grad_clip"],
        )
        optimizer.step()
        scheduler.step()
        tot += loss.item()
        dif += diff_loss.item()
        trd += (w * ce).mean().item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "obj": dif / n, "trend": trd / n}


@torch.no_grad()
def validate_objective(model, diffusion, loader, config, device):
    model.eval()
    t_max = config["T_max"]
    total, n = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        history = image * (1.0 - mask)
        b = image.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(image)
        x_t = diffusion.q_sample(image, t, noise)
        eps_hat = model.predict_noise(x_t, t, history, mask)
        total += _masked_loss(eps_hat, noise, mask).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate_label_accuracy(
    model, diffusion, dataset, config, gamma, alpha, device, n_windows, seed=0
):
    model.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n_windows, len(dataset)), replace=False)
    k, ns = config["label_k"], config["n_samples"]
    correct = 0
    for i in idxs:
        s = dataset[int(i)]
        x0_known = s["image"].unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
        m = s["mask"].unsqueeze(0).repeat(ns, 1, 1, 1).to(device)
        painted = diffusion.sample(model, x0_known, m, config["ddim_steps"], device)
        batch = {"mid_ref": torch.full((ns,), float(s["mid_ref"]))}
        fut_mid = model.future_mid(painted, batch, gamma)
        fwd = fut_mid[:, :k].mean(dim=1).cpu().numpy()
        bwd = float(s["bwd_smoothed"])
        l_vals = (fwd - bwd) / (bwd + 1e-12)
        modal = int(
            np.bincount(
                [lab.label_from_l(float(x), alpha) for x in l_vals], minlength=3
            ).argmax()
        )
        correct += int(modal == s["label"])
    acc = correct / max(len(idxs), 1)
    logger.info("val label acc={:.4f} on {} windows", acc, len(idxs))
    return acc


def print_summary(config, meta, gamma, alpha, n_params, device, ckpt_dir):
    cb = meta["class_balance"]
    iv = config["snapshot_interval_sec"]
    logger.info("Painting — pretrained UNet image inpainting diffusion")
    logger.info("  feature_mode  : {}", config["feature_mode"])
    logger.info(
        "  total         : {:,} snapshots (~{:.1f} days)",
        meta["total_snapshots"],
        meta["total_snapshots"] * iv / 86400,
    )
    logger.info(
        "  splits        : train={} val={} test={}",
        meta["counts"]["train"],
        meta["counts"]["val"],
        meta["counts"]["test"],
    )
    logger.info("  label_alpha   : {:.6f}", alpha)
    logger.info(
        "  class balance : down={:.1%} stat={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
    )
    if config["feature_mode"] == "ofi":
        logger.info("  ofi gamma     : {:.6g}", gamma)
    logger.info("  params        : ~{:.2f}M backbone", n_params / 1e6)
    logger.info("  device        : {}  |  ckpt: {}/", device, ckpt_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Penny painting model.")
    parser.add_argument("config", nargs="?", default="configs/painting/ofi.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_root"])
        / f"painting_pretrained_{config['feature_mode']}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, normalizer, gamma, alpha, meta = build_datasets(config)
    level_starts = meta["level_starts"]
    diffusion = Diffusion(config, device)
    model = build_model(config, normalizer, level_starts).to(device)
    trend_head = TrendHead().to(device)
    print_summary(config, meta, gamma, alpha, count_parameters(model), device, ckpt_dir)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    total_steps = max(config["epochs"] * len(train_loader), 1)
    optimizer, scheduler = build_optimizer_scheduler(
        model, trend_head, config, total_steps
    )

    best_val, patience, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        tr = train_one_epoch(
            model,
            trend_head,
            diffusion,
            train_loader,
            optimizer,
            scheduler,
            config,
            gamma,
            device,
        )
        val_obj = validate_objective(model, diffusion, val_loader, config, device)
        val_acc = validate_label_accuracy(
            model,
            diffusion,
            val_ds,
            config,
            gamma,
            alpha,
            device,
            config["val_eval_windows"],
            seed=epoch,
        )
        logger.info(
            "epoch {} | train total={:.5f} diff={:.5f} trend={:.5f} | val diff={:.5f} acc={:.4f}",
            epoch,
            tr["total"],
            tr["obj"],
            tr["trend"],
            val_obj,
            val_acc,
        )
        history.append({"epoch": epoch, **tr, "val_obj": val_obj, "val_acc": val_acc})
        if val_obj < best_val:
            best_val, patience = val_obj, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "trend_head": trend_head.state_dict(),
                    "config": config,
                    "normalizer": normalizer.to_dict(),
                    "gamma": gamma,
                    "alpha": alpha,
                    "level_starts": level_starts,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            logger.info("saved checkpoint (val diff {:.5f})", best_val)
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("early stopping at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ev.run_test(model, diffusion, test_ds, config, gamma, alpha, device)


if __name__ == "__main__":
    main()
