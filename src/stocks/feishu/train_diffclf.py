"""Train DiffusionClassifier on Feishu A-share equity data.

Classifier-free guidance diffusion (Ho & Salimans, 2022):
  - Training: DDPM noise-prediction MSE with class condition dropout
  - Validation: fast MC likelihood-ratio CE (K = dc_mc_val_samples) for
    early stopping — ties checkpointing to classification, not just diffusion
  - Test: full MC predict (K = dc_mc_samples) via shared run_test

Usage::

    uv run python -m stocks.feishu.train_diffclf configs/stocks/feishu/diffclf_ofi.json
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

from models.ddpm import DDPMScheduler
from models.diffclf import DiffusionClassifier, count_parameters
from stocks.feishu.build import build_datasets, discover_symbols
from stocks.feishu.features import n_features as feishu_n_features
from utils.evaluate import run_test
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


def _train_epoch(model, sched, loader, optimizer, lr_sched, device, grad_clip):
    model.train()
    T = sched.config.num_train_timesteps
    p_uncond = model.p_uncond
    total, n = 0.0, 0
    for batch in loader:
        x0 = batch["x"].to(device).float()
        label = batch["label"].to(device)
        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = sched.add_noise(x0, eps, t)

        # condition dropout: replace y with NULL_CLASS with probability p_uncond
        y = label.clone()
        drop = torch.rand(B, device=device) < p_uncond
        y[drop] = DiffusionClassifier.NULL_CLASS

        eps_hat = model(x_t, t, y)
        loss = F.mse_loss(eps_hat, eps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _validate(model, loader, device, mc_samples):
    model.eval()
    ce, correct, n = 0.0, 0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = model.predict(batch, device, mc_samples=mc_samples)
        ce += F.cross_entropy(logits, label).item()
        correct += (logits.argmax(1) == label).sum().item()
        n += len(label)
    return ce / max(len(loader), 1), correct / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/stocks/feishu/diffclf_ofi.json"
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"diffclf_{config.get('feature_mode', 'ofi')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    data_dir = Path(config["data_dir"])
    symbols = discover_symbols(data_dir, config)
    config["n_features"] = feishu_n_features(config)
    config["T_past"] = config.get("T_past", 50)

    noise_sched = DDPMScheduler(
        num_train_timesteps=config.get("T_max", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
        beta_schedule="linear",
        clip_sample=False,
    )
    model = DiffusionClassifier(config).to(device)
    logger.info(
        "DiffusionClassifier [Feishu]  mode={}  symbols={}  n_features={}",
        config.get("feature_mode"),
        len(symbols),
        config["n_features"],
    )
    logger.info(
        "  params={:.2f}M  p_uncond={}  mc_val={}  mc_test={}  device={}",
        count_parameters(model) / 1e6,
        model.p_uncond,
        model.mc_val_samples,
        model.mc_samples,
        device,
    )

    train_ds, val_ds, test_ds, meta = build_datasets(config, data_dir, symbols)
    cb = meta["class_balance"]
    logger.info(
        "  windows  train={}  val={}  test={}", len(train_ds), len(val_ds), len(test_ds)
    )
    logger.info(
        "  train balance  down={:.1%} stat={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
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
        tr_loss = _train_epoch(
            model, noise_sched, train_loader, optimizer, lr_sched, device, grad_clip
        )
        val_ce, val_acc = _validate(model, val_loader, device, model.mc_val_samples)
        logger.info(
            "epoch {} | diff={:.4f} | val_ce={:.4f} acc={:.4f}",
            epoch,
            tr_loss,
            val_ce,
            val_acc,
        )
        history.append(
            {"epoch": epoch, "diff_loss": tr_loss, "val_ce": val_ce, "val_acc": val_acc}
        )

        if val_ce < best:
            best, patience = val_ce, 0
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch},
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
