"""Train JumpGateLOB: Lévy W-aware joint diffusion-classifier, feature-only inference.

Joint objective on a **shared trunk**, with **separate passes** so the trend head is
always trained on the same clean-window distribution it sees at inference:

    L_cls   = CE(classify(x0), label)                    # clean pass, t = 0
    L_diff  = || eps_hat - eps ||^2                       # noised pass, sampled t
    L_W     = MSE(logW_hat, log W) + BCE(pi_logit, jump_flag)   # trains g_phi
    L_jump  = BCE(pi_logit, data_jump)   # self-supervised market-jump nudge
    L       = L_cls + lambda_diff * L_diff + mu_W * L_W + mu_jump * L_jump

``(x_t, eps, W, jump_flag)`` come from ``fp.add_noise_eps`` (Lévy jump-diffusion, or
the Gaussian bypass).  ``g_phi`` outputs are detached everywhere except ``L_W`` /
``L_jump`` (and the gate mixture when ``gate_grad="flow"``).

Modes:
  * default   — joint (both losses each step).
  * --baseline — plain classifier: ``L_cls`` only, no diffusion / g_phi (ladder's
    no-diffusion reference).
  * --baranchuk — two-phase diagnostic (Baranchuk et al. 2022): phase 1 trains
    diffusion only; phase 2 freezes the trunk and trains only the trend head on the
    frozen features.  Tests how linearly-separable the diffusion features are.

Model selection / early stopping on **trend-head macro-F1** (feature-only), not
denoising MSE.  Train and val F1 are both logged so the **F1 gap** (noise-fitting)
is visible.

Usage::

    uv run python -m crypto.train_jumpgatelob configs/crypto/nobitex/jumpgatelob/btcirt_ofi_k10.json
    uv run python -m crypto.train_jumpgatelob ... --process gaussian   # ablation
    uv run python -m crypto.train_jumpgatelob ... --baseline           # plain-classifier
    uv run python -m crypto.train_jumpgatelob ... --baranchuk          # two-phase diagnostic
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
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from levy.config import DiffusionConfig
from levy.diffusion import ForwardProcess
from models.jumpgatelob import JumpGateLOB, count_parameters
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
    """Map the flat repo config onto the levy DiffusionConfig (eps-prediction, so the
    generalized-score table is bypassed with ``table_num_r=1``)."""
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
        table_num_r=1,
        table_mc_samples=1,
        table_seed=config.get("seed", 42),
    )


def _data_jump_flag(x0: torch.Tensor, k: float) -> torch.Tensor:
    """Self-supervised market-jump target: level-averaged increment > ``k`` realized-
    vol units (distinct from the forward-process ``jump_flag``)."""
    agg = x0.squeeze(1).mean(dim=-1)  # (B, T)
    dif = agg[:, 1:] - agg[:, :-1]
    rv = dif.std(dim=1).clamp_min(1e-8)
    return (dif.abs().max(dim=1).values > k * rv).float()


def _train_epoch(
    model, fp, loader, optimizer, lr_sched, config, device, oracle, do_cls, do_diff
):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_diff = config.get("lambda_diff", 1.0)
    mu_W = config.get("mu_W", 0.1)
    mu_jump = config.get("mu_jump", 0.05)
    jump_rv_k = config.get("jump_rv_k", 4.0)
    label_smoothing = config.get("label_smoothing", 0.0)
    t_max = fp.schedule.num_timesteps

    tot = clsm = difm = lwm = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]

        loss = x0.new_zeros(())
        cls_loss = diff_loss = L_W = x0.new_zeros(())

        if do_cls:
            logits = model.classify(x0)  # clean pass, t = 0
            cls_loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
            loss = loss + cls_loss

        if do_diff:
            t = torch.randint(0, t_max, (b,), device=device)
            x_t, eps, W, jump_flag = fp.add_noise_eps(x0, t)
            logW = torch.log(W.clamp_min(1e-12))
            eps_hat, logW_hat, pi_logit = model.diffuse(
                x_t, t, logW_oracle=logW if oracle else None
            )
            diff_loss = F.mse_loss(eps_hat, eps)
            L_W = F.mse_loss(logW_hat, logW) + F.binary_cross_entropy_with_logits(
                pi_logit, jump_flag
            )
            L_jump = F.binary_cross_entropy_with_logits(
                pi_logit, _data_jump_flag(x0, jump_rv_k)
            )
            loss = loss + lam_diff * diff_loss + mu_W * L_W + mu_jump * L_jump

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip
        )
        optimizer.step()
        lr_sched.step()

        tot += loss.item()
        clsm += cls_loss.item()
        difm += diff_loss.item()
        lwm += L_W.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "cls": clsm / n, "diff": difm / n, "L_W": lwm / n}


@torch.no_grad()
def _f1_ce_acc(model, loader, device, max_batches=None):
    """Feature-only macro-F1 / CE / accuracy (F1 drives selection)."""
    model.eval()
    ce, n = 0.0, 0
    y_true, y_pred = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        label = batch["label"].to(device)
        logits = model.predict(batch, device)
        ce += F.cross_entropy(logits, label).item()
        y_true.extend(label.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        n += len(label)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(n, 1)
    f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    return f1, ce / max(n, 1), acc


@torch.no_grad()
def _validate_noise_state(model, fp, loader, config, device) -> dict:
    """logW RMSE + jump AUROC on the noised forward pass."""
    model.eval()
    t_max = fp.schedule.num_timesteps
    oracle = config.get("w_conditioning", "none") == "oracle"
    logw_t, logw_hat, flags, pis = [], [], [], []
    for batch in loader:
        x0 = batch["x"].to(device).float()
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        x_t, _, W, jump_flag = fp.add_noise_eps(x0, t)
        logW = torch.log(W.clamp_min(1e-12))
        _, logW_hat, pi_logit = model.diffuse(
            x_t, t, logW_oracle=logW if oracle else None
        )
        logw_t.append(logW.cpu())
        logw_hat.append(logW_hat.cpu())
        flags.append(jump_flag.cpu())
        pis.append(torch.sigmoid(pi_logit).cpu())
    logw_t = torch.cat(logw_t)
    logw_hat = torch.cat(logw_hat)
    flags = torch.cat(flags).numpy()
    pis = torch.cat(pis).numpy()
    rmse = float(torch.sqrt(torch.mean((logw_hat - logw_t) ** 2)))
    auroc = (
        float(roc_auc_score(flags, pis))
        if flags.min() == 0 and flags.max() == 1
        else float("nan")
    )
    return {"logW_rmse": rmse, "jump_auroc": auroc}


@torch.no_grad()
def _per_class_report(model, dataset, config, device) -> dict:
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    y_true, y_pred = [], []
    for batch in loader:
        logits = model.predict(batch, device)
        y_true.extend(batch["label"].tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
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
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["down", "stationary", "up"],
        zero_division=0,
        output_dict=True,
    )


def _new_optim_sched(model, config, total_steps):
    opt = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    return opt, build_cosine_schedule(opt, config, total_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/nobitex/jumpgatelob/btcirt_ofi_k10.json",
    )
    parser.add_argument("--process", choices=["levy", "gaussian"], default=None)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="plain classifier: L_cls only, no diffusion / g_phi",
    )
    parser.add_argument(
        "--baranchuk",
        action="store_true",
        help="two-phase diagnostic: diffusion-only, then frozen-trunk head training",
    )
    args = parser.parse_args()
    if args.baseline and args.baranchuk:
        logger.error("--baseline and --baranchuk are mutually exclusive")
        sys.exit(1)

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
    mode = "baseline" if args.baseline else ("baranchuk" if args.baranchuk else process)
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jumpgatelob_{mode}_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info(
        "JumpGateLOB  symbol={} mode={} process={} w_cond={} gated={}",
        config["symbol"],
        mode,
        process,
        config.get("w_conditioning", "none"),
        config.get("gated_experts", False),
    )
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  label_thr={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = JumpGateLOB(config).to(device)
    d = config["T_past"] * config["n_features"]
    fp = ForwardProcess(_diffusion_cfg(config), d=d, device=device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  device={}",
        count_parameters(model) / 1e6,
        log_gflops(model, train_ds, device),
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
    # capped train-F1 pass (same #batches as val) to watch the noise-fitting gap
    train_eval_batches = max(1, len(val_loader))

    epochs = config["epochs"]
    oracle = config.get("w_conditioning", "none") == "oracle"
    p1 = config.get("baranchuk_phase1_epochs", epochs // 2)
    optimizer, lr_sched = _new_optim_sched(model, config, epochs * len(train_loader))

    best, patience, history = float("-inf"), 0, []
    for epoch in range(epochs):
        # decide per-epoch losses & (Baranchuk) trunk freeze
        if args.baseline:
            do_cls, do_diff, phase = True, False, "baseline"
        elif args.baranchuk:
            if epoch < p1:
                do_cls, do_diff, phase = False, True, "diff"
            else:
                do_cls, do_diff, phase = True, False, "head"
                if epoch == p1:  # freeze the trunk, fresh optimizer on the head only
                    for p in model.trunk_parameters():
                        p.requires_grad_(False)
                    optimizer, lr_sched = _new_optim_sched(
                        model, config, (epochs - p1) * len(train_loader)
                    )
                    best, patience = float("-inf"), 0  # phase-2 selection is fresh
        else:
            do_cls, do_diff, phase = True, True, "joint"

        tr = _train_epoch(
            model,
            fp,
            train_loader,
            optimizer,
            lr_sched,
            config,
            device,
            oracle,
            do_cls,
            do_diff,
        )
        val_f1, val_ce, val_acc = _f1_ce_acc(model, val_loader, device)
        train_f1, _, _ = _f1_ce_acc(model, train_loader, device, train_eval_batches)
        row = {
            "epoch": epoch,
            "phase": phase,
            **tr,
            "val_f1": val_f1,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "train_f1": train_f1,
            "f1_gap": train_f1 - val_f1,
        }
        if do_diff:
            row.update(_validate_noise_state(model, fp, val_loader, config, device))
        logger.info(
            "ep {} [{}] | cls={:.4f} diff={:.4f} L_W={:.4f}"
            " | val_f1={:.4f} acc={:.4f} | train_f1={:.4f} gap={:+.4f}{}",
            epoch,
            phase,
            tr["cls"],
            tr["diff"],
            tr["L_W"],
            val_f1,
            val_acc,
            train_f1,
            train_f1 - val_f1,
            f" | logW_rmse={row['logW_rmse']:.3f} auroc={row['jump_auroc']:.3f}"
            if do_diff
            else "",
        )
        history.append(row)

        # only select on F1 when the trend head is being trained (skip diff-only phase)
        selectable = do_cls
        if selectable and val_f1 > best:
            best, patience = val_f1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "alpha": alpha,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )
        elif selectable:
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
                "mode": mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
