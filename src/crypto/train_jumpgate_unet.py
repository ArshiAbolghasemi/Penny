"""Train JumpGate-UNet: a W-aware, jump-gated joint diffusion-classifier.

A variant of ``crypto.train_jointdiff_levy`` that swaps generalized-score matching
for **epsilon prediction** and adds a supervised noise-state estimator ``g_phi``:

    L_diff = || eps_hat - eps ||^2                        (epsilon MSE)
    L_W    = MSE(logW_hat, log W) + BCE(pi_logit, jump_flag)   (trains g_phi ONLY)
    L_cls  = mean( gate * CE(logits, label) )
    L      = L_diff + lambda_trend * L_cls + mu_W * L_W

where ``(x_t, eps, W, jump_flag)`` come from ``fp.add_noise_eps`` (Lévy jump-
diffusion, or the Gaussian bypass where ``W = sigma_t^2`` and ``jump_flag = 0``).

``g_phi``'s outputs feed the backbone detached (conditioning), the two-expert
mixture (detached unless ``gate_grad="flow"``), and the classifier gate (detached),
so ``g_phi`` is trained purely by ``L_W`` — while the rest of the network learns to
*use* the inferred noise state.

Classifier gate (``soft_cls_gate``):
  * off (default): hard ``low = a_t^2 >= 0.5`` (SNR>=1), CE averaged over the kept
    subset — identical to the current levy trainer.
  * on: soft ``gamma = sigmoid(kappa * (log a_t^2 - logW_hat.detach()))`` — a smooth,
    W-aware version of the same "is the signal recoverable?" test.

Ablation flags (``w_conditioning`` none|inferred|oracle, ``gated_experts``,
``soft_cls_gate``, ``gate_grad``): with all off the model is a plain epsilon-
prediction joint U-Net and ``g_phi`` is a passive auxiliary head.  ``--process
gaussian`` keeps the Gaussian bypass and makes ``logW_hat`` regress ``log sigma_t^2``.

Extra val diagnostics (logged, not in metrics.json): logW RMSE, jump AUROC, and
per-gate-bin CE.

Usage::

    uv run python -m crypto.train_jumpgate_unet configs/crypto/nobitex/jumpgateunet/btcirt_ofi_k10.json
    uv run python -m crypto.train_jumpgate_unet ... --process gaussian   # ablation
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
from sklearn.metrics import classification_report, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from levy.config import DiffusionConfig
from levy.diffusion import ForwardProcess
from models.jumpgateunet import JumpGateUNet, count_parameters
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
    """Map the flat repo config onto the levy DiffusionConfig dataclass.

    JumpGate uses eps-prediction, so the generalized-score table is never built;
    ``table_num_r=1, table_mc_samples=1`` keeps the (unused) levy setup cheap.
    """
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


def _hard_gate(a_sq_t: torch.Tensor) -> torch.Tensor:
    return (a_sq_t >= 0.5).float()


def _soft_gate(
    a_sq_t: torch.Tensor, logW_hat: torch.Tensor, kappa: float
) -> torch.Tensor:
    log_a2 = torch.log(a_sq_t.clamp_min(1e-12))
    return torch.sigmoid(kappa * (log_a2 - logW_hat.detach()))


def _train_epoch(model, fp, loader, optimizer, lr_sched, config, device):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_cls = config.get("lambda_trend", 1.0)
    mu_W = config.get("mu_W", 0.1)
    kappa = config.get("cls_gate_kappa", 4.0)
    soft_gate = bool(config.get("soft_cls_gate", False))
    label_smoothing = config.get("label_smoothing", 0.0)
    oracle = config.get("w_conditioning", "none") == "oracle"
    t_max = fp.schedule.num_timesteps
    a_sq = (fp.schedule.a**2).to(device)

    tot = dif = cls = lw = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)

        x_t, eps, W, jump_flag = fp.add_noise_eps(x0, t)
        logW = torch.log(W.clamp_min(1e-12))  # (B,)
        eps_hat, logits, logW_hat, pi_logit = model(
            x_t, t, logW_oracle=logW if oracle else None
        )

        # epsilon MSE
        diff_loss = F.mse_loss(eps_hat, eps)

        # noise-state supervision (trains g_phi only; its outputs are detached elsewhere)
        L_W = F.mse_loss(logW_hat, logW) + F.binary_cross_entropy_with_logits(
            pi_logit, jump_flag
        )

        # trend loss with (soft or hard) noise-aware gate
        ce = F.cross_entropy(
            logits, label, reduction="none", label_smoothing=label_smoothing
        )
        if soft_gate:
            gate = _soft_gate(a_sq[t], logW_hat, kappa)
            cls_loss = (gate * ce).mean()
        else:
            low = a_sq[t] >= 0.5
            cls_loss = ce[low].mean() if low.any() else logits.new_zeros(())

        loss = diff_loss + lam_cls * cls_loss + mu_W * L_W

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_sched.step()

        tot += loss.item()
        dif += diff_loss.item()
        cls += cls_loss.item()
        lw += L_W.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "diff": dif / n, "trend": cls / n, "L_W": lw / n}


@torch.no_grad()
def _validate(model, loader, device):
    """Feature-only trend metrics (drives checkpointing / early stopping)."""
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
def _validate_noise_state(model, fp, loader, config, device) -> dict:
    """Diagnostics on the *noised* forward pass: logW RMSE, jump AUROC, per-gate CE."""
    model.eval()
    t_max = fp.schedule.num_timesteps
    a_sq = (fp.schedule.a**2).to(device)
    kappa = config.get("cls_gate_kappa", 4.0)
    oracle = config.get("w_conditioning", "none") == "oracle"

    logw_t, logw_hat, flags, pis, gates, ces = [], [], [], [], [], []
    for batch in loader:
        x0 = batch["x"].to(device).float()
        label = batch["label"].to(device)
        b = x0.shape[0]
        t = torch.randint(0, t_max, (b,), device=device)
        x_t, _, W, jump_flag = fp.add_noise_eps(x0, t)
        logW = torch.log(W.clamp_min(1e-12))
        _, logits, logW_hat, pi_logit = model(
            x_t, t, logW_oracle=logW if oracle else None
        )
        gate = _soft_gate(a_sq[t], logW_hat, kappa)
        logw_t.append(logW.cpu())
        logw_hat.append(logW_hat.cpu())
        flags.append(jump_flag.cpu())
        pis.append(torch.sigmoid(pi_logit).cpu())
        gates.append(gate.cpu())
        ces.append(F.cross_entropy(logits, label, reduction="none").cpu())

    logw_t = torch.cat(logw_t)
    logw_hat = torch.cat(logw_hat)
    flags = torch.cat(flags).numpy()
    pis = torch.cat(pis).numpy()
    gates = torch.cat(gates)
    ces = torch.cat(ces)

    rmse = float(torch.sqrt(torch.mean((logw_hat - logw_t) ** 2)))
    # AUROC needs both classes present (undefined for the Gaussian path: no jumps)
    auroc = (
        float(roc_auc_score(flags, pis))
        if flags.min() == 0 and flags.max() == 1
        else float("nan")
    )
    # CE within soft-gate bins
    edges = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.01])
    bin_ce = []
    for i in range(len(edges) - 1):
        m = (gates >= edges[i]) & (gates < edges[i + 1])
        bin_ce.append(float(ces[m].mean()) if m.any() else float("nan"))
    return {"logW_rmse": rmse, "jump_auroc": auroc, "gate_bin_ce": bin_ce}


@torch.no_grad()
def _per_class_report(model, dataset, config, device) -> dict:
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
        default="configs/crypto/nobitex/jumpgateunet/btcirt_ofi_k10.json",
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
        / f"jumpgateunet_{process}_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info(
        "JumpGateUNet  symbol={}  mode={}  process={}  schedule={}",
        config["symbol"],
        config.get("feature_mode"),
        process,
        config.get("schedule", "vp"),
    )
    logger.info(
        "  w_cond={}  gated_experts={}  soft_cls_gate={}  gate_grad={}  mu_W={}",
        config.get("w_conditioning", "none"),
        config.get("gated_experts", False),
        config.get("soft_cls_gate", False),
        config.get("gate_grad", "detach"),
        config.get("mu_W", 0.1),
    )
    logger.info("  windows train={} val={} test={}", *meta["counts"].values())
    logger.info(
        "  alpha={:.6f}  down={:.1%} flat={:.1%} up={:.1%}",
        alpha,
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = JumpGateUNet(config).to(device)

    d = config["T_past"] * config["n_features"]
    fp = ForwardProcess(_diffusion_cfg(config), d=d, device=device)

    gflops = log_gflops(model, train_ds, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  jump_rate={}  device={}",
        count_parameters(model) / 1e6,
        gflops,
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
        tr = _train_epoch(model, fp, train_loader, optimizer, lr_sched, config, device)
        val_ce, val_acc = _validate(model, val_loader, device)
        ns = _validate_noise_state(model, fp, val_loader, config, device)
        logger.info(
            "epoch {} | total={:.4f} diff={:.4f} trend={:.4f} L_W={:.4f}"
            " | val_ce={:.4f} acc={:.4f}"
            " | logW_rmse={:.3f} jump_auroc={:.3f} gate_ce={}",
            epoch,
            tr["total"],
            tr["diff"],
            tr["trend"],
            tr["L_W"],
            val_ce,
            val_acc,
            ns["logW_rmse"],
            ns["jump_auroc"],
            "[" + ", ".join(f"{c:.2f}" for c in ns["gate_bin_ce"]) + "]",
        )
        history.append(
            {"epoch": epoch, **tr, "val_ce": val_ce, "val_acc": val_acc, **ns}
        )

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
