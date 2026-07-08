"""Train JumpGate-ScoreGrad: a W-aware, jump-gated joint diffusion-classifier.

A variant of ``crypto.train_jointdiff_levy`` that swaps generalized-score matching
for **epsilon prediction** and adds a supervised noise-state estimator ``g_phi``:

    L_diff = || eps_hat - eps ||^2                        (epsilon MSE)
    L_W    = MSE(logW_hat, log W) + BCE(pi_logit, jump_flag)   (trains g_phi ONLY)
    L_jump = BCE(pi_logit, data_jump)     (self-supervised market-jump nudge)
    L_cls  = sum( w_t * CE(logits, label) ) / sum(w_t)
    L      = L_diff + lambda_trend * L_cls + mu_W * L_W + mu_jump * L_jump

where ``(x_t, eps, W, jump_flag)`` come from ``fp.add_noise_eps`` (Lévy jump-
diffusion, or the Gaussian bypass where ``W = sigma_t^2`` and ``jump_flag = 0``).

``g_phi``'s outputs feed the backbone detached (conditioning), the two-expert
mixture (detached unless ``gate_grad="flow"``), and the classifier gate (detached),
so ``g_phi`` is trained purely by the supervised terms — while the rest of the
network learns to *use* the inferred noise state.

Trend-loss weighting over diffusion ``t`` (``cls_t_anneal``, default on):
  * default: ``w_t = exp(-(t/t_max)/tau)`` with ``tau`` annealed geometrically from
    ``cls_tau_start`` to ``cls_tau_end`` across epochs, so the trend head is trained
    on ever-cleaner GRU passes — matching the feature-only inference at ``t = 0``
    (removes the train/deploy mismatch).
  * ``cls_t_anneal=false`` falls back to the ``soft_cls_gate`` (or hard SNR>=1) gate.

``L_jump`` gives ``pi_logit`` a small, self-supervised *market-jump* signal
(window increments exceeding ``jump_rv_k`` realized-vol units); keep ``mu_jump``
small since ``pi_logit`` also carries the forward-process ``jump_flag`` in ``L_W``.

Model selection / early stopping is on **trend-head macro-F1** (feature-only
inference), not denoising MSE.

Ablation flags (``w_conditioning`` none|inferred|oracle, ``gated_experts``,
``soft_cls_gate``, ``gate_grad``).  ``--process gaussian`` keeps the Gaussian
bypass; ``--baseline`` trains a plain-GRU classifier (GRU+pool+trend head on CE,
no diffusion, no g_phi) — the ladder's no-diffusion reference point.

Extra val diagnostics (logged, not in metrics.json): logW RMSE, jump AUROC, and
per-gate-bin CE.

Usage::

    uv run python -m crypto.train_jumpgate_score configs/crypto/nobitex/jumpgatescore/btcirt_ofi_k10.json
    uv run python -m crypto.train_jumpgate_score ... --process gaussian   # ablation
    uv run python -m crypto.train_jumpgate_score ... --baseline           # plain-GRU control

W-conditioning ladder (decide by trend-head macro-F1)::

    for w in none inferred oracle; do
      uv run python -m crypto.train_jumpgate_score CONFIG   # set w_conditioning=$w in CONFIG
    done
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
from models.jumpgatescore import JumpGateScoreGrad, count_parameters
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


def _cls_t_weight(t: torch.Tensor, t_max: int, tau: float) -> torch.Tensor:
    """Per-sample weight concentrating the trend loss on small diffusion ``t``.

    ``exp(-(t/t_max)/tau)`` — as ``tau`` anneals down over training the trend head
    is trained on ever-cleaner GRU passes, so its input distribution matches the
    feature-only inference path (clean window at ``t = 0``).
    """
    tf = t.float() / max(t_max, 1)
    return torch.exp(-tf / max(tau, 1e-6))


def _data_jump_flag(x0: torch.Tensor, k: float) -> torch.Tensor:
    """Self-supervised jump target from the window itself (distinct from the
    forward-process ``jump_flag``).

    Flags a window whose largest per-timestep increment of the level-averaged
    feature exceeds ``k`` realized-vol units — a data-driven "a jump happened here"
    signal used to nudge ``pi_logit`` toward *market* jumps.
    """
    agg = x0.squeeze(1).mean(dim=-1)  # (B, T) level-averaged feature
    dif = agg[:, 1:] - agg[:, :-1]  # (B, T-1) increments
    rv = dif.std(dim=1).clamp_min(1e-8)  # (B,) realized vol
    return (dif.abs().max(dim=1).values > k * rv).float()  # (B,)


def _train_epoch(
    model, fp, loader, optimizer, lr_sched, config, device, epoch, epochs, baseline
):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_cls = config.get("lambda_trend", 1.0)
    mu_W = config.get("mu_W", 0.1)
    mu_jump = config.get("mu_jump", 0.05)
    jump_rv_k = config.get("jump_rv_k", 4.0)
    kappa = config.get("cls_gate_kappa", 4.0)
    soft_gate = bool(config.get("soft_cls_gate", False))
    anneal = bool(config.get("cls_t_anneal", True))
    tau0 = config.get("cls_tau_start", 0.5)
    tau1 = config.get("cls_tau_end", 0.05)
    label_smoothing = config.get("label_smoothing", 0.0)
    oracle = config.get("w_conditioning", "none") == "oracle"
    t_max = fp.schedule.num_timesteps
    a_sq = (fp.schedule.a**2).to(device)
    # geometric anneal of the trend-loss temperature over training
    frac = epoch / max(epochs - 1, 1)
    tau = tau0 * (tau1 / tau0) ** frac

    tot = dif = cls = lw = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]

        if baseline:
            # plain-GRU-classifier control: no diffusion, no g_phi — CE on the clean
            # pass only (same GRU + pool + trend head as feature-only inference).
            logits = model._trend_logits(model._encode(x0))
            cls_loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
            diff_loss = logits.new_zeros(())
            L_W = logits.new_zeros(())
            loss = cls_loss
        else:
            t = torch.randint(0, t_max, (b,), device=device)
            x_t, eps, W, jump_flag = fp.add_noise_eps(x0, t)
            logW = torch.log(W.clamp_min(1e-12))  # (B,)
            eps_hat, logits, logW_hat, pi_logit = model(
                x_t, t, logW_oracle=logW if oracle else None
            )

            # epsilon MSE
            diff_loss = F.mse_loss(eps_hat, eps)

            # noise-state supervision: forward jump_flag (L_W) + a small self-supervised
            # data-jump nudge on the same pi_logit (keep mu_jump small — two targets).
            L_W = F.mse_loss(logW_hat, logW) + F.binary_cross_entropy_with_logits(
                pi_logit, jump_flag
            )
            L_jump = F.binary_cross_entropy_with_logits(
                pi_logit, _data_jump_flag(x0, jump_rv_k)
            )

            # trend loss: annealed-toward-small-t weighting (default), else the
            # soft/hard noise-aware gate.
            ce = F.cross_entropy(
                logits, label, reduction="none", label_smoothing=label_smoothing
            )
            if anneal:
                w = _cls_t_weight(t, t_max, tau)
                cls_loss = (w * ce).sum() / w.sum().clamp_min(1e-8)
            elif soft_gate:
                gate = _soft_gate(a_sq[t], logW_hat, kappa)
                cls_loss = (gate * ce).mean()
            else:
                low = a_sq[t] >= 0.5
                cls_loss = ce[low].mean() if low.any() else logits.new_zeros(())

            loss = diff_loss + lam_cls * cls_loss + mu_W * L_W + mu_jump * L_jump

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
    return {
        "total": tot / n,
        "diff": dif / n,
        "trend": cls / n,
        "L_W": lw / n,
        "tau": tau,
    }


@torch.no_grad()
def _validate(model, loader, device):
    """Feature-only trend metrics. Macro-F1 drives checkpointing / early stopping
    (inference is feature-only, so we select on trend-head F1, not denoising MSE)."""
    model.eval()
    ce, n = 0.0, 0
    y_true, y_pred = [], []
    for batch in loader:
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
    return ce / max(len(loader), 1), acc, f1


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
        default="configs/crypto/nobitex/jumpgatescore/btcirt_ofi_k10.json",
    )
    parser.add_argument(
        "--process",
        choices=["levy", "gaussian"],
        default=None,
        help="ablation override for config['diffusion_process']",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="plain-GRU-classifier control: train only GRU+pool+trend head on CE "
        "(no diffusion, no g_phi) — the ladder's no-diffusion reference point",
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
    tag = "baseline" if args.baseline else process
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jumpgatescore_{tag}_{config['symbol']}_{config.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    train_ds, val_ds, test_ds, alpha, meta = build_datasets(config)
    config["n_features"] = meta["n_features"]
    cb = meta["class_balance"]

    logger.info(
        "JumpGateScoreGrad  symbol={}  mode={}  process={}  schedule={}",
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

    model = JumpGateScoreGrad(config).to(device)

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

    # select on trend-head macro-F1 (feature-only inference), not denoising MSE
    best, patience, history = float("-inf"), 0, []
    epochs = config["epochs"]
    for epoch in range(epochs):
        tr = _train_epoch(
            model,
            fp,
            train_loader,
            optimizer,
            lr_sched,
            config,
            device,
            epoch,
            epochs,
            args.baseline,
        )
        val_ce, val_acc, val_f1 = _validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            **tr,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        if args.baseline:
            logger.info(
                "epoch {} | trend={:.4f} | val_ce={:.4f} acc={:.4f} f1={:.4f}",
                epoch,
                tr["trend"],
                val_ce,
                val_acc,
                val_f1,
            )
        else:
            ns = _validate_noise_state(model, fp, val_loader, config, device)
            row.update(ns)
            logger.info(
                "epoch {} | total={:.4f} diff={:.4f} trend={:.4f} L_W={:.4f} tau={:.3f}"
                " | val_ce={:.4f} acc={:.4f} f1={:.4f}"
                " | logW_rmse={:.3f} jump_auroc={:.3f} gate_ce={}",
                epoch,
                tr["total"],
                tr["diff"],
                tr["trend"],
                tr["L_W"],
                tr["tau"],
                val_ce,
                val_acc,
                val_f1,
                ns["logW_rmse"],
                ns["jump_auroc"],
                "[" + ", ".join(f"{c:.2f}" for c in ns["gate_bin_ce"]) + "]",
            )
        history.append(row)

        if val_f1 > best:
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
