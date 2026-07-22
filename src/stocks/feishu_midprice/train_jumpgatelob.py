"""Train JumpGateLOB: jump-diffusion score matching + noise-consistent classification.

Joint objective on a shared trunk, with **separate passes**, built for LOB data that
is noisy and contains jumps.  Three terms, all always active:

    L_cls    = CE(classify(x0), label)                       # clean pass, t = 0
    L_score  = w̄_t · || ŝ(x_t, t) − ∇log q(x_t|x0) ||²       # generalized score matching
    L_robust = CE(classify(x̃), label)                        # jump-noised low-t pass
             + robust_kl · KL( p(x̃) ‖ p(x0).detach() )       # clean/noisy consistency
    L        = L_cls + lambda_diff · L_score + mu_robust · L_robust

* **Forward process** — the Lévy jump-diffusion of ``src/levy``: additive noise
  ``u = √W·ξ`` with ``W = σ_t² + Σ_k S_k`` (Brownian variance + compound-Poisson gamma
  jumps).  ``--process gaussian`` ablates the jumps away.
* **L_score** — denoising score matching against the *tabulated generalized score* of
  that non-Gaussian kernel, ``∇log q = −u·h(|u|)`` (Baule 2025), weighted per sample by
  ``w̄_t = E[W_t]`` so the target is O(1) at every timestep.  This is what shapes the
  trunk on jump-diffusion perturbations.
* **L_robust** — the trend head classifies **jump-noised** windows drawn from the same
  forward process at low ``t`` (the SNR ≥ 1 region, so the label is still recoverable),
  always at the classifier's ``t = 0`` conditioning (deployment never knows the noise
  level).  CE keeps it correct under noise; the KL term pulls the noisy prediction
  toward its own clean prediction — this trains the *inference path itself* to be
  robust to noise and jumps.

Model selection / early stopping on **trend-head macro-F1** (feature-only); train and
val F1 are both logged so the noise-fitting gap is visible.

Modes:
  * default    — joint (all three losses each step).
  * --process gaussian — ablation: no jumps in the forward process.
  * --baseline — plain classifier: ``L_cls`` only.

Usage::

    uv run python -m stocks.feishu_midprice.train_jumpgatelob configs/stocks/feishu_midprice/jumpgatelob_h1.json
    uv run python -m stocks.feishu_midprice.train_jumpgatelob ... --process gaussian
    uv run python -m stocks.feishu_midprice.train_jumpgatelob ... --baseline
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
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
from stocks.feishu_midprice.build import build_datasets, discover_symbols
from stocks.feishu.features import n_features as feishu_n_features


def _diffusion_cfg(config: dict) -> DiffusionConfig:
    """Map the flat repo config onto the levy DiffusionConfig (score-matching path, so
    the generalized-score table is built for real)."""
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
    """Analytic mean total variance ``E[W_t] = σ_t² + Λ_t·shape·scale``.

    Per-sample DSM weight so the weighted score target is O(1) at every timestep
    (raw score magnitude is ~1/W)."""
    _, sigma_t = fp.schedule.gather(t)
    w = sigma_t**2
    if fp.process == "levy":
        w = w + fp.lambda_t.to(t.device)[t] * fp.jump.mean_jump_var()
    return w


def _low_t_indices(fp: ForwardProcess, device: torch.device) -> torch.Tensor:
    """Timesteps where signal dominates noise (SNR ≥ 1) — the noise levels the robust
    classification pass draws from.  VP: ᾱ_t ≥ 0.5; VE: σ_t < 1 (features z-scored)."""
    if fp.schedule.kind == "vp":
        mask = (fp.schedule.a.to(device) ** 2) >= 0.5
    else:
        mask = fp.schedule.sigma.to(device) < 1.0
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    return idx if len(idx) > 0 else torch.zeros(1, dtype=torch.long, device=device)


def _train_epoch(
    model, fp, low_t, loader, optimizer, lr_sched, config, device, do_diff
):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_diff = config.get("lambda_diff", 1.0)
    mu_robust = config.get("mu_robust", 0.5)
    robust_kl = config.get("robust_kl", 1.0)
    label_smoothing = config.get("label_smoothing", 0.0)
    t_max = fp.schedule.num_timesteps

    tot = clsm = scm = robm = 0.0
    n = 0
    for batch in loader:
        x0 = batch["x"].to(device).float()  # (B, 1, T, F)
        label = batch["label"].to(device)
        b = x0.shape[0]

        # clean pass — trend head sees exactly what inference sees
        logits = model.classify(x0)
        cls_loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
        loss = cls_loss
        score_loss = rob_loss = x0.new_zeros(())

        if do_diff:
            # generalized score matching on the jump-diffusion kernel
            t = torch.randint(0, t_max, (b,), device=device)
            x_t, _ = fp.add_noise(x0, t)
            s_target = fp.score_target(x_t, x0, t)
            s_hat = model.score(x_t, t)
            w = _mean_W(fp, t)
            score_loss = (w * ((s_hat - s_target) ** 2).flatten(1).mean(1)).mean()

            # noise-consistent classification: jump-noised low-t windows, classified
            # at t=0 conditioning (deployment never knows the noise level)
            t_rob = low_t[torch.randint(0, len(low_t), (b,), device=device)]
            x_rob, _ = fp.add_noise(x0, t_rob)
            logits_rob = model.classify(x_rob)
            rob_ce = F.cross_entropy(logits_rob, label, label_smoothing=label_smoothing)
            rob_con = F.kl_div(
                F.log_softmax(logits_rob, dim=1),
                F.softmax(logits.detach(), dim=1),
                reduction="batchmean",
            )
            rob_loss = rob_ce + robust_kl * rob_con

            loss = loss + lam_diff * score_loss + mu_robust * rob_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip
        )
        optimizer.step()
        lr_sched.step()

        tot += loss.item()
        clsm += cls_loss.item()
        scm += score_loss.item()
        robm += rob_loss.item()
        n += 1
    n = max(n, 1)
    return {"total": tot / n, "cls": clsm / n, "score": scm / n, "robust": robm / n}


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
def _noisy_f1(model, fp, low_t, loader, device, max_batches=None):
    """Macro-F1 on jump-noised low-t windows — the robustness metric the noise-
    consistency loss is trying to move."""
    model.eval()
    y_true, y_pred = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x0 = batch["x"].to(device).float()
        t_rob = low_t[torch.randint(0, len(low_t), (x0.shape[0],), device=device)]
        x_rob, _ = fp.add_noise(x0, t_rob)
        logits = model.classify(x_rob)
        y_true.extend(batch["label"].tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    return float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/stocks/feishu_midprice/jumpgatelob_h1.json",
    )
    parser.add_argument("--process", choices=["levy", "gaussian"], default=None)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="plain classifier: L_cls only, no diffusion / robustness losses",
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
    mode = "baseline" if args.baseline else process
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"jumpgatelob_{mode}_{config.get('feature_mode', 'ofi')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    data_dir = Path(config["data_dir"])
    symbols = discover_symbols(data_dir, config)
    config["n_features"] = feishu_n_features(config)

    logger.info(
        "JumpGateLOB [Feishu]  mode={} process={} symbols={} (score-matching + noise-consistency)",
        mode,
        process,
        len(symbols),
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
        "  label_alpha={:.6f}  down={:.1%} stat={:.1%} up={:.1%}",
        config.get("alpha", 0.015),
        cb["down"],
        cb["stationary"],
        cb["up"],
    )

    model = JumpGateLOB(config).to(device)
    d = config["T_past"] * config["n_features"]
    fp = ForwardProcess(_diffusion_cfg(config), d=d, device=device)
    low_t = _low_t_indices(fp, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  low-t region: {} steps (≤ t={})  device={}",
        count_parameters(model) / 1e6,
        log_gflops(model, train_ds, device),
        len(low_t),
        int(low_t.max()),
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
    train_eval_batches = max(1, len(val_loader))

    epochs = config["epochs"]
    do_diff = not args.baseline
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    lr_sched = build_cosine_schedule(optimizer, config, epochs * len(train_loader))

    best, history = float("-inf"), []
    for epoch in range(epochs):
        tr = _train_epoch(
            model, fp, low_t, train_loader, optimizer, lr_sched, config, device, do_diff
        )
        val_f1, val_ce, val_acc = _f1_ce_acc(model, val_loader, device)
        train_f1, _, _ = _f1_ce_acc(model, train_loader, device, train_eval_batches)
        noisy_f1 = _noisy_f1(model, fp, low_t, val_loader, device)
        row = {
            "epoch": epoch,
            **tr,
            "val_f1": val_f1,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "train_f1": train_f1,
            "f1_gap": train_f1 - val_f1,
            "noisy_val_f1": noisy_f1,
        }
        logger.info(
            "ep {} | cls={:.4f} score={:.4f} robust={:.4f}"
            " | val_f1={:.4f} acc={:.4f} noisy_f1={:.4f} | train_f1={:.4f} gap={:+.4f}",
            epoch,
            tr["cls"],
            tr["score"],
            tr["robust"],
            val_f1,
            val_acc,
            noisy_f1,
            train_f1,
            train_f1 - val_f1,
        )
        history.append(row)

        # model selection on the held-out in-sample val slice (highest macro-F1)
        if val_f1 > best:
            best = val_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "label_alpha": config.get("alpha", 0.015),
                    "epoch": epoch,
                },
                ckpt_dir / "best.pt",
            )

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    metrics = run_test(model, test_ds, config, device)
    report = _per_class_report(model, test_ds, config, device)
    (ckpt_dir / "metrics.json").write_text(
        json.dumps(
            {"out_of_sample": metrics, "per_class": report},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
