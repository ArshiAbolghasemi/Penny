"""Training entry point for LOBTransformer.

Loss = CrossEntropy(logits, label).
Usage::

    uv run python -m crypto.lobtransformer.train configs/crypto/lobtransformer/btcusdt_ofi.json
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

from crypto.utils.dataset import build_datasets
from crypto.utils.evaluate import run_test
from crypto.utils.training import build_cosine_schedule, resolve_device

from .model import LOBTransformer, count_parameters


def _train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = model.predict(batch, device)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _validate(model, loader, device):
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
        description="Train LOBTransformer on Binance LOB data."
    )
    parser.add_argument(
        "config", nargs="?", default="configs/crypto/lobtransformer/btcusdt_ofi.json"
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
        / f"lobtransformer_{config['symbol']}_{config['feature_mode']}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]

    cb = meta["class_balance"]
    logger.info(
        "LOBTransformer — symbol={} feature_mode={}",
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

    model = LOBTransformer(config).to(device)
    logger.info(
        "  params    : ~{:.2f}M  |  device {}",
        count_parameters(model) / 1e6,
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
    total_steps = max(config["epochs"] * len(train_loader), 1)
    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = build_cosine_schedule(optimizer, config, total_steps)

    best_val_ce, patience, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        train_ce = _train_epoch(model, train_loader, optimizer, scheduler, device)
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | train ce={:.4f} | val ce={:.4f} acc={:.4f}",
            epoch,
            train_ce,
            val_ce,
            val_acc,
        )
        history.append(
            {"epoch": epoch, "train_ce": train_ce, "val_ce": val_ce, "val_acc": val_acc}
        )
        if val_ce < best_val_ce:
            best_val_ce, patience = val_ce, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "alpha": alpha,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            logger.info("  -> checkpoint saved (val ce {:.4f})", best_val_ce)
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
