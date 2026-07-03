"""Train JointDiT with *Drift* (Generative Modeling via Drifting, Lambert et al.).

Prompt-2 procedure on the DiT backbone.  Instead of a noise schedule or
consistency matching, the DiT is trained as a **single-step generator** whose
output distribution *drifts* toward the data distribution.  The generator is the
EDM consistency map applied to pure noise::

    x_gen = f_theta(sigma_max * z, sigma_max)          # noise -> LOB window

Generated windows are flattened to feature particles and pulled toward a goal
computed by the multi-scale kernel drift force (attracted to real windows from a
memory bank, optionally repelled from a bank of past generations).  The trend
head is trained jointly on clean real windows (the ``sigma_min`` pass that
``predict`` uses)::

    L = drift_loss(phi(x_gen), phi(x_pos), phi(x_neg)) + lambda_trend * CE(logits, label)

``phi`` is the identity flatten (each window is one particle in R^{T*F}); the
classification gradient enriches the shared DiT encoder with directionally
predictive features the drift loss alone would not discover.  Inference is a
single forward pass — no ODE integration, no iterative sampling.

Usage::

    uv run python -m crypto.train_jointdit_drift configs/crypto/nobitex/jointdit/btcirt_ofi.json
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

from crypto.dataset import build_datasets
from models.drift import WindowMemoryBank, drift_loss
from models.jointdit import JointDiT, count_parameters
from utils.evaluate import run_test
from utils.flops import log_gflops
from utils.training import (
    build_cosine_schedule,
    measure_sigma_data,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


def _train_epoch(model, loader, optimizer, lr_sched, config, device, drift, banks):
    model.train()
    lam_cls = config.get("lambda_trend", 1.0)
    grad_clip = config.get("grad_clip", 1.0)
    sigma_max = drift["sigma_max"]
    sigma_min = model.sigma_min
    r_list = drift["r_list"]
    n_pos = drift["pos_per_sample"]
    n_neg = drift["neg_per_sample"]
    pos_bank, neg_bank = banks["pos"], banks["neg"]
    tot = dft = cls = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]
        pos_bank.add(x0, label)

        # One-step generation: consistency map from pure noise at sigma_max.
        z = torch.randn_like(x0)
        sig_hi = torch.full((b,), sigma_max, device=device)
        x_gen, _ = model.denoise(sigma_max * z, sig_hi)  # (B, 1, T, F)

        # Drift needs a warm memory bank; skip the drift term until it fills.
        if pos_bank.ready(n_pos):
            pos_x, _ = pos_bank.sample(n_pos)
            phi_gen = x_gen.flatten(1).unsqueeze(0)  # (1, B, S)
            phi_pos = pos_x.to(device).flatten(1).unsqueeze(0)  # (1, P, S)
            phi_neg = None
            if n_neg > 0 and neg_bank.ready(n_neg):
                neg_x, _ = neg_bank.sample(n_neg)
                phi_neg = neg_x.to(device).flatten(1).unsqueeze(0)  # (1, N, S)
            dft_loss, _ = drift_loss(phi_gen, phi_pos, phi_neg, R_list=r_list)
            dft_loss = dft_loss.mean()
        else:
            dft_loss = torch.zeros((), device=device)
        if n_neg > 0:
            neg_bank.add(x_gen, label)

        # Trend head on clean real windows (matches predict()'s sigma_min pass).
        sig_lo = torch.full((b,), sigma_min, device=device)
        _, logits = model.denoise(x0, sig_lo)
        cls_loss = F.cross_entropy(logits, label)
        loss = dft_loss + lam_cls * cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()
        tot += loss.item()
        dft += dft_loss.item()
        cls += cls_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "drift": dft / n, "trend": cls / n}


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
        default="configs/crypto/nobitex/jointdit/btcirt_ofi.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    config["cm_enabled"] = True  # consistency generator / predict path

    seed = resolve_seed(config)
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jointdit_drift_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    # The one-step generator uses EDM preconditioning too, so sigma_data must match
    # the true std of the LOB windows — measure it from the training set.
    if config.get("cm_sigma_data_auto", True):
        config["cm_sigma_data"] = measure_sigma_data(train_ds)
    logger.info(
        "  cm_sigma_data={:.4f} (measured from train windows)", config["cm_sigma_data"]
    )

    logger.info(
        "JointDiT (drift)  symbol={}  mode={}",
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

    model = JointDiT(config).to(device)

    drift = {
        "sigma_max": float(config.get("cm_sigma_max", 80.0)),
        "r_list": tuple(config.get("drift_r_list", (0.02, 0.05, 0.2))),
        "pos_per_sample": int(config.get("drift_pos_per_sample", 32)),
        "neg_per_sample": int(config.get("drift_neg_per_sample", 0)),
    }
    banks = {
        "pos": WindowMemoryBank(int(config.get("drift_pos_bank", 4096))),
        "neg": WindowMemoryBank(int(config.get("drift_neg_bank", 4096))),
    }

    gflops = log_gflops(model, train_ds, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  sigma_max={}  pos={}  neg={}"
        "  lambda_trend={}  device={}",
        count_parameters(model) / 1e6,
        gflops,
        drift["sigma_max"],
        drift["pos_per_sample"],
        drift["neg_per_sample"],
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
        tr = _train_epoch(
            model, train_loader, optimizer, lr_sched, config, device, drift, banks
        )
        val_ce, val_acc = _validate(model, val_loader, device)
        logger.info(
            "epoch {} | total={:.4f} drift={:.4f} trend={:.4f} | val_ce={:.4f} acc={:.4f}",
            epoch,
            tr["total"],
            tr["drift"],
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
