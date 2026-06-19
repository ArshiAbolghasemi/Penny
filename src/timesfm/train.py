"""Training entry point for the TimesFM direction classifier.

Loss = CrossEntropy(logits, label).
Usage::

    uv run python -m timesfm.train configs/timesfm/ofi.json
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
from .dataset import build_datasets
from .model import TimesFMClassifier, count_parameters


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


def build_optimizer_scheduler(model, config, total_steps):
    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    warmup = config["warmup_steps"]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device):
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
def validate(model, loader, device):
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
    parser = argparse.ArgumentParser(description="Train Penny TimesFM classifier.")
    parser.add_argument("config", nargs="?", default="configs/timesfm/ofi.json")
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
        / f"timesfm_{config['feature_mode']}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    model = TimesFMClassifier(config).to(device)
    cb = meta["class_balance"]
    use_ofi = config.get("feature_mode", "ofi") == "ofi"
    logger.info(
        "TimesFM classifier — feature_mode={} ofi_features={}",
        config["feature_mode"],
        use_ofi,
    )
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
    if use_ofi and meta.get("ofi_stats"):
        s = meta["ofi_stats"]
        logger.info(
            "  ofi stats   : mean={:.4f} std={:.4f}", s["mean"], s["std"]
        )
    logger.info(
        "  params      : ~{:.2f}M  |  device {}",
        count_parameters(model) / 1e6,
        device,
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    total_steps = max(config["epochs"] * len(train_loader), 1)
    optimizer, scheduler = build_optimizer_scheduler(model, config, total_steps)

    best_val_ce, patience, history = float("inf"), 0, []
    for epoch in range(config["epochs"]):
        train_ce = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_ce, val_acc = validate(model, val_loader, device)
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
                    "ofi_stats": meta.get("ofi_stats"),
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
            logger.info("saved checkpoint (val ce {:.4f})", best_val_ce)
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("early stopping at epoch {}", epoch)
                break

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ev.run_test(model, test_ds, config, device)


if __name__ == "__main__":
    main()
