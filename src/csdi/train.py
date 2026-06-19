"""Training entry point for the CSDI price-direction forecaster.

Custom loss = future mid-return MSE (price prediction) + DeepLOB trend
cross-entropy.  Usage::

    uv run python -m csdi.train configs/csdi/ofi.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import evaluate as ev
from . import labels as lab
from .dataset import build_datasets
from .model import CSDIForecastModel, TrendHead, count_parameters


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


def _step(model, trend_head, batch, config, device):
    t_past, k = config["T_past"], config["label_k"]
    true_mid = batch["true_mid"].to(device).float()
    boundary = true_mid[:, t_past - 1]
    target_ret = true_mid[:, t_past:] / boundary.view(-1, 1) - 1.0
    bwd = batch["bwd_smoothed"].to(device).float()
    label = batch["label"].to(device)

    pred_ret = model.forecast(batch, device)
    price_loss = F.mse_loss(pred_ret, target_ret)
    fut_mid = boundary.view(-1, 1) * (1.0 + pred_ret)
    l_pred = (fut_mid[:, :k].mean(dim=1) - bwd) / (bwd + 1e-12)
    trend_loss = F.cross_entropy(trend_head(l_pred), label)
    return price_loss, trend_loss


def train_one_epoch(model, trend_head, loader, optimizer, scheduler, config, device):
    model.train()
    trend_head.train()
    lam = config["lambda_trend"]
    tot = obj = trd = 0.0
    n = 0
    for batch in loader:
        price_loss, trend_loss = _step(model, trend_head, batch, config, device)
        loss = price_loss + lam * trend_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(trend_head.parameters()),
            config["grad_clip"],
        )
        optimizer.step()
        scheduler.step()
        tot += loss.item()
        obj += price_loss.item()
        trd += trend_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "obj": obj / n, "trend": trd / n}


@torch.no_grad()
def validate_objective(model, loader, config, device):
    model.eval()
    t_past = config["T_past"]
    total, n = 0.0, 0
    for batch in loader:
        true_mid = batch["true_mid"].to(device).float()
        boundary = true_mid[:, t_past - 1]
        target_ret = true_mid[:, t_past:] / boundary.view(-1, 1) - 1.0
        pred_ret = model.forecast(batch, device)
        total += F.mse_loss(pred_ret, target_ret).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate_label_accuracy(model, loader, config, alpha, device):
    model.eval()
    t_past, k = config["T_past"], config["label_k"]
    correct = total = 0
    for batch in loader:
        true_mid = batch["true_mid"].to(device).float()
        boundary = true_mid[:, t_past - 1]
        pred_ret = model.forecast(batch, device)
        fut_mid = boundary.view(-1, 1) * (1.0 + pred_ret)
        bwd = batch["bwd_smoothed"].to(device).float()
        l_pred = ((fut_mid[:, :k].mean(dim=1) - bwd) / (bwd + 1e-12)).cpu().numpy()
        labels = batch["label"].numpy()
        for lv, true in zip(l_pred, labels):
            correct += int(lab.label_from_l(float(lv), alpha) == true)
            total += 1
    acc = correct / max(total, 1)
    logger.info("val label acc={:.4f} on {} windows", acc, total)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Penny CSDI forecaster.")
    parser.add_argument("config", nargs="?", default="configs/csdi/ofi.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_root"]) / f"csdi_{config['feature_mode']}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, normalizer, alpha, meta = build_datasets(config)
    model = CSDIForecastModel(config).to(device)
    trend_head = TrendHead().to(device)
    cb = meta["class_balance"]
    logger.info("CSDI forecaster — feature_mode={}", config["feature_mode"])
    logger.info(
        "  splits      : train={} val={} test={}",
        meta["counts"]["train"],
        meta["counts"]["val"],
        meta["counts"]["test"],
    )
    logger.info("  label_alpha : {:.6f}", alpha)
    logger.info(
        "  class bal   : down={:.1%} stat={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
    )
    logger.info(
        "  params      : ~{:.2f}M  |  device {}", count_parameters(model) / 1e6, device
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    total_steps = max(config["epochs"] * len(train_loader), 1)
    optimizer, scheduler = build_optimizer_scheduler(
        model, trend_head, config, total_steps
    )

    best_val, patience, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        tr = train_one_epoch(
            model, trend_head, train_loader, optimizer, scheduler, config, device
        )
        val_obj = validate_objective(model, val_loader, config, device)
        val_acc = validate_label_accuracy(model, val_loader, config, alpha, device)
        logger.info(
            "epoch {} | train total={:.6f} price={:.6f} trend={:.5f} | val price={:.6f} acc={:.4f}",
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
                    "alpha": alpha,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            logger.info("saved checkpoint (val price {:.6f})", best_val)
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("early stopping at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ev.run_test(model, test_ds, config, alpha, device)


if __name__ == "__main__":
    main()
