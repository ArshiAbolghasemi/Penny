"""Training entry point for JointDiffusion.

Joint objective per batch (Deja et al., 2023):

    t      ~ Uniform{0, …, T_max-1}                      (per-sample timestep)
    x_t    = sqrt(abar_t)·x0 + sqrt(1-abar_t)·noise        (forward diffusion)
    eps, logits = model(x_t, t)
    L_diff = MSE(eps, noise)                               (denoising)
    L_cls  = w(t) · CE(logits, label),  w(t) = (1 - t/T_max)^2
    L      = L_diff + lambda_trend · mean(L_cls)

The timestep weight ``w(t)`` down-weights classification on heavily-noised
windows (whose label is barely recoverable), keeping the classifier well-posed
while the shared encoder still benefits from the denoising signal at all noise
levels.  At inference the trend is read from the clean window at ``t = 0``.

Usage::

    uv run python -m crypto.jointdiff.train configs/crypto/jointdiff/btcusdt_ofi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.utils.training import build_cosine_schedule, resolve_device

from crypto.utils.dataset import build_datasets
from crypto.utils.evaluate import run_test
from .diffusion import Diffusion
from .model import JointDiffusion, count_parameters


def _train_epoch(model, diffusion, loader, optimizer, scheduler, config, device):
    model.train()
    t_max = diffusion.T_max
    lam = config.get("lambda_trend", 1.0)
    grad_clip = config.get("grad_clip", 1.0)
    tot = dif = trd = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()
        label = batch["label"].to(device)
        b = x0.shape[0]

        t = torch.randint(0, t_max, (b,), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        eps_hat, logits = model(x_t, t)
        diff_loss = F.mse_loss(eps_hat, noise)

        ce = F.cross_entropy(logits, label, reduction="none")
        w = (1.0 - t.float() / t_max) ** 2
        cls_loss = (w * ce).mean()

        loss = diff_loss + lam * cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        tot += loss.item()
        dif += diff_loss.item()
        trd += cls_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "diff": dif / n, "trend": trd / n}


@torch.no_grad()
def _validate(model, loader, device):
    """Validate on the *classification* objective at t=0 (the inference setting)."""
    model.eval()
    ce_total, correct, n = 0.0, 0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = model.predict(batch, device)
        ce_total += F.cross_entropy(logits, label).item()
        correct += (logits.argmax(dim=1) == label).sum().item()
        n += len(label)
    return ce_total / max(len(loader), 1), correct / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train JointDiffusion on Binance LOB data."
    )
    parser.add_argument(
        "config", nargs="?", default="configs/crypto/jointdiff/btcusdt_ofi.json"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jointdiff_{config['symbol']}_{config['feature_mode']}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]

    cb = meta["class_balance"]
    logger.info(
        "JointDiffusion — symbol={} feature_mode={}",
        config["symbol"],
        config["feature_mode"],
    )
    logger.info("  snapshots : {:,}", meta["n_snapshots"])
    logger.info(
        "  windows   : train={} val={} test={}",
        meta["counts"]["train"],
        meta["counts"]["val"],
        meta["counts"]["test"],
    )
    logger.info("  alpha     : {:.6f}", alpha)
    logger.info(
        "  class bal : down={:.1%} stat={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
    )
    logger.info("  features  : {}", meta["n_features"])

    diffusion = Diffusion(config, device)
    model = JointDiffusion(config).to(device)
    logger.info(
        "  params    : ~{:.2f}M  |  lambda_trend={}  |  device {}",
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
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    total_steps = config["epochs"] * len(train_loader)
    scheduler = build_cosine_schedule(optimizer, config, total_steps)

    best_val_ce, patience_count, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        tr = _train_epoch(
            model, diffusion, train_loader, optimizer, scheduler, config, device
        )
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | total={:.4f} diff={:.4f} trend={:.4f} | val_ce={:.4f} val_acc={:.4f}",
            epoch,
            tr["total"],
            tr["diff"],
            tr["trend"],
            val_ce,
            val_acc,
        )
        history.append({"epoch": epoch, **tr, "val_ce": val_ce, "val_acc": val_acc})

        if val_ce < best_val_ce:
            best_val_ce, patience_count = val_ce, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "alpha": alpha,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            logger.info("  -> checkpoint saved (val_ce={:.4f})", best_val_ce)
        else:
            patience_count += 1
            if patience_count >= config["patience"]:
                logger.info("early stopping at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    run_test(model, test_ds, config, device)


if __name__ == "__main__":
    main()
