"""Train DLA on Feishu A-share equity data.

Usage::

    uv run python -m stocks.feishu.train_dla configs/stocks/feishu/dla_ofi.json
"""

from __future__ import annotations

import argparse
import copy
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
from utils.flops import log_gflops
from utils.training import (
    add_seed_args,
    build_cosine_schedule,
    resolve_device,
    resolve_seeds,
    seed_worker,
    set_seed,
    summarize_seed_runs,
)
from models.dla import DLA, count_parameters
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


def _run_seed(config, args, seed: int, multi_seed: bool) -> dict:
    """Train and test one seed; returns its test metrics for aggregation."""
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    grad_clip = config.get("grad_clip", 1.0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"dla_{config.get('feature_mode', 'ofi')}_{stamp}"
    )
    if multi_seed:
        ckpt_dir = ckpt_dir.with_name(f"{ckpt_dir.name}_seed{seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_sink = logger.add(ckpt_dir / "train.log", level="DEBUG")

    data_dir = Path(config["data_dir"])
    symbols = discover_symbols(data_dir, config)
    config["n_features"] = feishu_n_features(config)

    logger.info(
        "DLA [Feishu]  mode={}  symbols={}  n_features={}",
        config.get("feature_mode"),
        len(symbols),
        config["n_features"],
    )
    logger.info(
        "  params={:.2f}M  device={}", count_parameters(DLA(config)) / 1e6, device
    )

    train_ds, val_ds, test_ds, meta = build_datasets(config, data_dir, symbols)
    cb = meta["class_balance"]
    logger.info(
        "  windows  train={}  val={}  test(out-of-sample)={}",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    logger.info(
        "  train balance  down={:.1%} stat={:.1%} up={:.1%}",
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = DLA(config).to(device)
    logger.info("  gflops/sample={:.3f}", log_gflops(model, train_ds, device))
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
    scheduler = build_cosine_schedule(
        optimizer, config, config["epochs"] * len(train_loader)
    )

    best, history = float("inf"), []
    for epoch in range(config["epochs"]):
        tr_ce = _train_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip
        )
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | train_ce={:.4f} val_ce={:.4f} val_acc={:.4f}",
            epoch,
            tr_ce,
            val_ce,
            val_acc,
        )
        history.append(
            {"epoch": epoch, "train_ce": tr_ce, "val_ce": val_ce, "val_acc": val_acc}
        )

        # model selection on the held-out in-sample val slice (lowest val CE)
        if val_ce < best:
            best = val_ce
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

    logger.remove(log_sink)
    return {
        "seed": seed,
        "run_dir": str(ckpt_dir),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/stocks/feishu/dla_ofi.json"
    )
    add_seed_args(parser)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())

    seeds = resolve_seeds(config, args.seeds)
    if len(seeds) > 1:
        logger.info("training {} seeds: {}", len(seeds), seeds)
    runs = [_run_seed(copy.deepcopy(config), args, s, len(seeds) > 1) for s in seeds]
    if len(seeds) > 1:
        summarize_seed_runs(runs)


if __name__ == "__main__":
    main()
