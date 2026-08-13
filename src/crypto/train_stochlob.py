"""Train StochLOB with stochastic latent dynamics active at train and test time.

Use ``--dynamics`` for deterministic/Gaussian/stable/jump/routed ablations and
``--classification-objective`` to compare pathwise and marginal Monte Carlo losses.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from crypto.dataset import build_datasets
from models.stochlob import LatentStochasticDynamics, StochLOB
from utils.stochastic_evaluate import evaluate_stochastic
from utils.training import (
    add_seed_args,
    build_cosine_schedule,
    resolve_device,
    resolve_seeds,
    seed_worker,
    set_seed,
    summarize_seed_runs,
)


class HorizonDataset(Dataset):
    """Attach the supervised forecast horizon to every existing LOB sample."""

    def __init__(self, dataset: Dataset, horizon: int) -> None:
        self.dataset = dataset
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        item["horizon"] = self.horizon
        return item


def _config_for_horizon(config: dict, horizon: int) -> dict:
    sub = copy.deepcopy(config)
    sub["label_k"] = int(horizon)
    if "cache_dirs" in config:
        sub["cache_dir"] = config["cache_dirs"][str(horizon)]
    elif "cache_dir" in sub:
        sub["cache_dir"] = re.sub(r"/k\d+(?=/|$)", f"/k{horizon}", sub["cache_dir"])
    return sub


def build_stochlob_datasets(config: dict):
    """Build one horizon or concatenate k={10,20,50,100} with explicit k fields."""
    horizons = (
        [int(k) for k in config.get("horizons", [10, 20, 50, 100])]
        if config.get("train_all_horizons", False)
        else [int(config.get("label_k", 10))]
    )
    splits = [[], [], []]
    thresholds, metas = {}, []
    for horizon in horizons:
        train, val, test, threshold, meta = build_datasets(
            _config_for_horizon(config, horizon)
        )
        for bucket, dataset in zip(splits, (train, val, test)):
            bucket.append(HorizonDataset(dataset, horizon))
        thresholds[str(horizon)] = threshold
        metas.append(meta)

    def combine(items: list[Dataset]) -> Dataset:
        return items[0] if len(items) == 1 else ConcatDataset(items)

    meta = {
        "n_features": metas[0]["n_features"],
        "counts": {
            name: sum(m["counts"][name] for m in metas)
            for name in ("train", "val", "test")
        },
        "thresholds": thresholds,
        "horizons": horizons,
    }
    return combine(splits[0]), combine(splits[1]), combine(splits[2]), meta


def _router_entropy(routing: torch.Tensor) -> torch.Tensor:
    p = routing.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1).mean()


def _classification_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pathwise CE and MC marginal predictive NLL.

    ``logits`` has shape ``(M, B, C)``.  The marginal calculation uses
    log-sum-exp rather than materializing/then logging the mean probabilities, so
    it remains stable even when the correct class has very small probability.
    """
    trajectories, batch, classes = logits.shape
    expanded_labels = labels.unsqueeze(0).expand(trajectories, -1).reshape(-1)
    pathwise = F.cross_entropy(
        logits.reshape(trajectories * batch, classes),
        expanded_labels,
        label_smoothing=label_smoothing,
    )
    trajectory_log_probs = F.log_softmax(logits, dim=-1)
    marginal_log_probs = torch.logsumexp(trajectory_log_probs, dim=0) - math.log(
        trajectories
    )
    target_nll = -marginal_log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    smooth_nll = -marginal_log_probs.mean(dim=1)
    marginal = (
        (1.0 - label_smoothing) * target_nll + label_smoothing * smooth_nll
    ).mean()
    return pathwise, marginal


def train_epoch(model, loader, optimizer, scheduler, config, device, epoch):
    model.train()
    classification_objective = config.get("classification_objective", "pathwise")
    if classification_objective not in {"pathwise", "marginal"}:
        raise ValueError("classification_objective must be pathwise|marginal")
    route_weight = float(config.get("lambda_route", 0.0))
    anneal_epochs = max(1, int(config.get("route_anneal_epochs", 10)))
    route_weight *= max(0.0, 1.0 - epoch / anneal_epochs)
    smoothing = float(config.get("label_smoothing", 0.0))
    totals = {
        key: 0.0
        for key in (
            "loss",
            "classification",
            "pathwise_classification",
            "marginal_classification",
            "route_entropy",
        )
    }
    count = 0

    for batch in loader:
        x = batch["x"].to(device).float()
        label = batch["label"].to(device)
        horizon = batch["horizon"].to(device)
        rollout = model(x, horizon, model.train_trajectories)
        pathwise_loss, marginal_loss = _classification_losses(
            rollout.logits, label, label_smoothing=smoothing
        )
        cls_loss = (
            pathwise_loss if classification_objective == "pathwise" else marginal_loss
        )
        entropy = _router_entropy(rollout.routing)
        # Negative entropy gently encourages early exploration; it is not a 50/50 target.
        loss = cls_loss - route_weight * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()

        totals["loss"] += loss.item()
        totals["classification"] += cls_loss.item()
        totals["pathwise_classification"] += pathwise_loss.item()
        totals["marginal_classification"] += marginal_loss.item()
        totals["route_entropy"] += entropy.item()
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device, trajectories: int) -> tuple[float, float, dict]:
    model.eval()
    labels, predictions = [], []
    route_sum = torch.zeros(2)
    gamma_sum = rate_sum = norm_max = 0.0
    batches = 0
    for batch in loader:
        out = model.predict_distribution(batch, device, trajectories)
        labels.extend(batch["label"].tolist())
        predictions.extend(out.mean_probability.argmax(dim=1).cpu().tolist())
        route_sum += out.routing.mean(dim=(0, 1, 2)).cpu()
        gamma_sum += out.stable_scale.mean().item()
        rate_sum += out.jump_intensity.mean().item()
        norm_max = max(norm_max, out.latent_norm.max().item())
        batches += 1
    f1 = float(
        f1_score(
            labels, predictions, average="macro", labels=[0, 1, 2], zero_division=0
        )
    )
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / max(len(labels), 1)
    diagnostics = {
        "pi_alpha": float(route_sum[0] / max(batches, 1)),
        "pi_jump": float(route_sum[1] / max(batches, 1)),
        "stable_scale": gamma_sum / max(batches, 1),
        "jump_intensity": rate_sum / max(batches, 1),
        "max_latent_norm": norm_max,
    }
    return f1, accuracy, diagnostics


def run_seed(config: dict, args, seed: int, multi_seed: bool) -> dict:
    config["seed"] = seed
    if args.dynamics is not None:
        config["latent_dynamics"] = args.dynamics
    if args.classification_objective is not None:
        config["classification_objective"] = args.classification_objective
    generator = set_seed(seed)
    device = resolve_device(config["device"])
    train_ds, val_ds, test_ds, meta = build_stochlob_datasets(config)
    config["n_features"] = meta["n_features"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode = config.get("latent_dynamics", "routed")
    classification_objective = config.get("classification_objective", "pathwise")
    output = (
        Path(config["checkpoint_dir"])
        / f"stochlob_{mode}_{classification_objective}_{config['symbol']}_{stamp}"
    )
    if multi_seed:
        output = output.with_name(f"{output.name}_seed{seed}")
    output.mkdir(parents=True, exist_ok=True)
    log_sink = logger.add(output / "train.log", level="DEBUG")

    model = StochLOB(config).to(device)
    workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
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
    logger.info(
        "StochLOB mode={} cls={} alpha={} steps={} M_train={} M_test={} horizons={}",
        mode,
        config.get("classification_objective", "pathwise"),
        model.dynamics.alpha,
        model.dynamics.steps,
        model.train_trajectories,
        model.test_trajectories,
        meta["horizons"],
    )

    best, stale, history = float("-inf"), 0, []
    for epoch in range(config["epochs"]):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, config, device, epoch
        )
        val_f1, val_acc, diagnostics = validate(
            model, val_loader, device, int(config.get("val_trajectories", 5))
        )
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
            **diagnostics,
        }
        history.append(row)
        logger.info(
            "ep {} loss={:.4f} cls={:.4f} path={:.4f} marginal={:.4f} f1={:.4f} "
            "pi=({:.3f},{:.3f}) gamma={:.3f} lambda={:.3f} |z|max={:.2f}",
            epoch,
            row["loss"],
            row["classification"],
            row["pathwise_classification"],
            row["marginal_classification"],
            val_f1,
            row["pi_alpha"],
            row["pi_jump"],
            row["stable_scale"],
            row["jump_intensity"],
            row["max_latent_norm"],
        )
        if val_f1 > best:
            best, stale = val_f1, 0
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch},
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= config["patience"]:
                break

    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "training_log.json").write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate_stochastic(model, test_ds, config, device, model.test_trajectories)
    sweep = {}
    for trajectories in config.get("trajectory_sweep", [1, 5, 10, 20, 50]):
        metrics = evaluate_stochastic(
            model,
            test_ds,
            config,
            device,
            int(trajectories),
            include_mc_dropout=False,
        )
        sweep[str(trajectories)] = {
            key: metrics[key]
            for key in (
                "accuracy",
                "macro_f1",
                "ece",
                "brier",
                "mean_trajectory_disagreement",
            )
        }
    (output / "metrics.json").write_text(
        json.dumps({"test": test, "trajectory_sweep": sweep}, indent=2)
    )
    logger.remove(log_sink)
    return {
        "seed": seed,
        "run_dir": str(output),
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/coinbase/stochlob/btcirt_ofi_k10.json",
    )
    parser.add_argument("--dynamics", choices=sorted(LatentStochasticDynamics.MODES))
    parser.add_argument("--classification-objective", choices=["pathwise", "marginal"])
    add_seed_args(parser)
    args = parser.parse_args()
    path = Path(args.config)
    if not path.exists():
        logger.error("config not found: {}", path)
        sys.exit(1)
    config = json.loads(path.read_text())
    seeds = resolve_seeds(config, args.seeds)
    runs = [
        run_seed(copy.deepcopy(config), args, seed, len(seeds) > 1) for seed in seeds
    ]
    if len(seeds) > 1:
        summarize_seed_runs(runs)


if __name__ == "__main__":
    main()
