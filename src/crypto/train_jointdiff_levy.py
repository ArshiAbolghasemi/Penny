"""Train JointDiffusionLevy: joint Lévy jump-diffusion + trend classifier.

Forward process (Baule 2025, arXiv:2503.06558): a finite-activity jump-diffusion
``dx = mu dt + sigma dW + z dJ`` with compound-Poisson jumps whose amplitudes are
isotropic generalized-Laplace (subordinated Gaussian).  The denoising loss stays a
standard MSE, but the regression target is the **generalized score** of the
jump-diffusion kernel, precomputed offline as an isotropic 1-D table and
interpolated at train time (``levy.diffusion``).  Plain VP (DDPM-style) or VE
schedule — no EDM preconditioning, no consistency distillation.

Joint objective (Deja et al. 2023) on the shared U-Net encoder:

    L_diff  = wbar_t * || s_hat(x_t, t) - grad log q(x_t|x_0) ||^2   (score MSE,
              wbar_t = E[W_t] normalizes the score scale across timesteps)
    L_trend = CE(logits, label)   only on low-noise samples (SNR >= 1 gate)

combined as ``L_diff + lambda_trend * L_trend`` (fixed weight from the config).

Ablation toggle: ``diffusion_process`` = ``"levy"`` | ``"gaussian"`` switches the
noising kernel + score target; everything else is identical.

Inference is feature-only: ``model.predict`` runs encoder + trend head on the
clean window at t=0 (no decoder, no sampling loop).

Usage::

    uv run python -m crypto.train_jointdiff_levy configs/crypto/nobitex/jointdifflevy/btcirt_ofi_k10.json
    uv run python -m crypto.train_jointdiff_levy ... --process gaussian   # ablation
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
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from levy.config import DiffusionConfig
from levy.diffusion import ForwardProcess
from models.jointdifflevy import JointDiffusionLevy, count_parameters
from utils.evaluate import run_test
from utils.flops import log_gflops
from utils.training import (
    build_cosine_schedule,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


def _diffusion_cfg(config: dict) -> DiffusionConfig:
    """Map the flat repo config onto the levy DiffusionConfig dataclass."""
    return DiffusionConfig(
        process=config.get("diffusion_process", "levy"),
        schedule=config.get("schedule", "vp"),
        num_timesteps=config.get("T_max", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
        sigma_min=config.get("ve_sigma_min", 1e-2),
        sigma_max=config.get("ve_sigma_max", 50.0),
        jump_rate=config.get("levy_jump_rate", 1.0),
        jump_gamma_shape=config.get("levy_gamma_shape", 1.0),
        jump_gamma_scale=config.get("levy_gamma_scale", 1.0),
        table_num_r=config.get("levy_table_num_r", 512),
        table_mc_samples=config.get("levy_table_mc", 20000),
        table_seed=config.get("seed", 42),
    )


def _mean_W(fp: ForwardProcess, t: torch.Tensor) -> torch.Tensor:
    """Analytic mean total variance ``E[W_t] = sigma_t^2 + Lambda_t*shape*scale``.

    Used as the per-sample DSM weight so the weighted score target is O(1) at every
    timestep (raw score magnitude is ~1/W).  TODO(user): standard lambda(t)=sigma^2
    weighting generalized to the jump kernel — confirm this choice.
    """
    _, sigma_t = fp.schedule.gather(t)
    w = sigma_t**2
    if fp.process == "levy":
        w = w + fp.lambda_t.to(t.device)[t] * fp.jump.mean_jump_var()
    return w


def _train_epoch(model, fp, loader, optimizer, lr_sched, config, device):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_cls = config.get("lambda_trend", 1.0)
    t_max = fp.schedule.num_timesteps
    a_sq = (fp.schedule.a**2).to(device)
    # trend CE only where signal dominates noise.  VP: alpha_bar >= 0.5 (SNR >= 1);
    # VE: sigma < 1 (features are causally z-scored, so sigma_data ~ 1).
    if fp.schedule.kind == "vp":
        low_t = a_sq >= 0.5
    else:
        low_t = fp.schedule.sigma.to(device) < 1.0

    tot = dif = cls = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)

        x_t, _ = fp.add_noise(x0, t)
        target = fp.score_target(x_t, x0, t)  # generalized (or Gaussian) score
        s_hat, logits = model(x_t, t)

        # weighted score-matching MSE: wbar_t * ||s_hat - target||^2
        w = _mean_W(fp, t)  # (B,)
        se = ((s_hat - target) ** 2).flatten(1).mean(1)  # (B,)
        diff_loss = (w * se).mean()

        low = low_t[t]
        if low.any():
            ce = F.cross_entropy(logits, label, reduction="none")
            cls_loss = ce[low].mean()
        else:
            cls_loss = logits.new_zeros(())

        loss = diff_loss + lam_cls * cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()

        tot += loss.item()
        dif += diff_loss.item()
        cls += cls_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "diff": dif / n, "trend": cls / n}


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


@torch.no_grad()
def _per_class_report(model, dataset, config, device) -> dict:
    """Precision / recall / F1 per class (down/stationary/up) on *dataset*."""
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    y_true, y_pred = [], []
    for batch in loader:
        logits = model.predict(batch, device)
        y_true.extend(batch["label"].tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    rep = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["down", "stationary", "up"],
        zero_division=0,
        output_dict=True,
    )
    logger.info(
        "TEST per-class P/R/F1:\n{}",
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["down", "stationary", "up"],
            zero_division=0,
        ),
    )
    return rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/nobitex/jointdifflevy/btcirt_ofi_k10.json",
    )
    parser.add_argument(
        "--process",
        choices=["levy", "gaussian"],
        default=None,
        help="ablation override for config['diffusion_process']",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    if args.process is not None:
        config["diffusion_process"] = args.process
    process = config.get("diffusion_process", "levy")

    seed = resolve_seed(config)
    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jointdifflevy_{process}_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info(
        "JointDiffusionLevy  symbol={}  mode={}  process={}  schedule={}",
        config["symbol"],
        config.get("feature_mode"),
        process,
        config.get("schedule", "vp"),
    )
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  alpha={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = JointDiffusionLevy(config).to(device)

    # Forward process; the levy path precomputes the generalized-score table once
    # (offline, MC over the mixing variance W) before training starts.
    d = config["T_past"] * config["n_features"]
    logger.info("  building {} forward process (d={}) ...", process, d)
    fp = ForwardProcess(_diffusion_cfg(config), d=d, device=device)

    gflops = log_gflops(model, train_ds, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  cond={}  jump_rate={}  device={}",
        count_parameters(model) / 1e6,
        gflops,
        config.get("jdl_cond", "film"),
        config.get("levy_jump_rate", 1.0) if process == "levy" else 0.0,
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
            model, fp, train_loader, optimizer, lr_sched, config, device
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
    metrics = run_test(model, test_ds, config, device)
    report = _per_class_report(model, test_ds, config, device)
    (ckpt_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "confusion": metrics["confusion"].tolist(),
                "per_class": report,
                "process": process,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
