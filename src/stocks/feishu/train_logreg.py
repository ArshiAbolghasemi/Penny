"""Train LogReg on Feishu A-share equity data.

Usage::

    uv run python -m stocks.feishu.train_logreg configs/stocks/feishu/logreg_ofi.json
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

from utils.evaluate import run_test
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)
from models.logreg import LogReg, count_parameters
from stocks.feishu.build import build_datasets, discover_symbols
from stocks.feishu.features import n_features as feishu_n_features


def _train_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = model.predict(batch, device)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/stocks/feishu/logreg_ofi.json"
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
        / f"logreg_{config.get('feature_mode', 'ofi')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    data_dir = Path(config["data_dir"])
    symbols = discover_symbols(data_dir, config)
    config["n_features"] = feishu_n_features(config)

    logger.info(
        "LogReg [Feishu]  mode={}  symbols={}  n_features={}",
        config.get("feature_mode"),
        len(symbols),
        config["n_features"],
    )
    logger.info(
        "  params={:.2f}M  device={}", count_parameters(LogReg(config)) / 1e6, device
    )

    train_ds, test_ds, meta = build_datasets(config, data_dir, symbols)
    cb = meta["class_balance"]
    logger.info(
        "  windows  in-sample(train)={}  out-of-sample(test)={}",
        len(train_ds),
        len(test_ds),
    )
    logger.info(
        "  train balance  down={:.1%} stat={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = LogReg(config).to(device)
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

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = build_cosine_schedule(
        optimizer, config, config["epochs"] * len(train_loader)
    )

    best, history = float("inf"), []
    for epoch in range(config["epochs"]):
        tr_ce = _train_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip
        )
        logger.info("epoch {} | train_ce={:.4f}", epoch, tr_ce)
        history.append({"epoch": epoch, "train_ce": tr_ce})

        if tr_ce < best:
            best = tr_ce
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch},
                ckpt_dir / "best.pt",
            )

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    metrics = run_test(model, test_ds, config, device)
    (ckpt_dir / "metrics.json").write_text(
        json.dumps({"out_of_sample": metrics}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
