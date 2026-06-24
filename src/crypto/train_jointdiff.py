"""Train JointDiffusion on crypto LOB data.

   Joint objective (Deja et al., 2023):
       L = MSE(eps_hat, noise) + lambda_trend * w(t) * CE(logits, label)
        w(t) = (1 - t/T_max)^2   when trend_taper else 1

Usage::

    uv run python -m crypto.train_jointdiff configs/crypto/binance/jointdiff/btcusdt_ofi.json
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
from models.ddpm import DDPMScheduler
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from utils.evaluate import run_test
from utils.training import build_cosine_schedule, resolve_device
from models.jointdiff import JointDiffusion, count_parameters


def _train_epoch(model, sched, loader, optimizer, lr_sched, config, device):
    model.train()
    t_max = sched.config.num_train_timesteps
    lam = config.get("lambda_trend", 1.0)
    trend_taper = config.get("trend_taper", False)
    grad_clip = config.get("grad_clip", 1.0)
    tot = dif = trd = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()
        label = batch["label"].to(device)
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(x0)
        x_t = sched.add_noise(x0, noise, t)

        eps_hat, logits = model(x_t, t)
        diff_loss = F.mse_loss(eps_hat, noise)
        if trend_taper:
            w = (1.0 - t.float() / t_max) ** 2
        else:
            w = 1.0
        cls_loss = (w * F.cross_entropy(logits, label, reduction="none")).mean()
        loss = diff_loss + lam * cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()
        tot += loss.item()
        dif += diff_loss.item()
        trd += cls_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "diff": dif / n, "trend": trd / n}


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
        default="configs/crypto/binance/jointdiff/btcusdt_ofi.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jointdiff_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info(
        "JointDiffusion  symbol={}  mode={}",
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

    noise_sched = DDPMScheduler(
        num_train_timesteps=config.get("T_max", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
        beta_schedule="linear",
        clip_sample=False,
    )
    model = JointDiffusion(config).to(device)
    logger.info(
        "  params={:.2f}M  lambda_trend={}  device={}",
        count_parameters(model) / 1e6,
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
        tr = _train_epoch(
            model, noise_sched, train_loader, optimizer, lr_sched, config, device
        )
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | total={:.4f} diff={:.4f} trend={:.4f} | val_ce={:.4f} acc={:.4f}",
            epoch,
            tr["total"],
            tr["diff"],
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
