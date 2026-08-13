"""Train AlphaStableLOB: α-stable joint diffusion-classifier (feature-only inference).

Same joint structure as ``stocks.feishu_midprice.train_jumpgatelob`` — a shared trunk trained on
**separate passes** so the trend head always sees the clean-window distribution it
sees at inference — but the generative branch uses a genuine **α-stable (Lévy-stable)**
forward process (heavy, power-law tails; :mod:`models.alphastable`) and is trained by
**generalized denoising score matching** against the tabulated α-stable score.  Three
terms — but ``L_robust`` is gated by ``mu_robust``, which every shipped config
sets to ``0``:

    L_cls    = CE(classify(x0), label)                       # clean pass, t = 0
    L_score  = mean || ŝ(c_in·x_t, t) − ∇log q(x_t|x0) ||²   # noised pass, sampled t
    L_robust = CE(classify(c_in·x̃), label)                   # α-stable-noised low-t pass
             + robust_kl · KL( p(x̃) ‖ p(x0).detach() )       # clean/noisy consistency
    L        = L_cls + lambda_diff · L_score + mu_robust · L_robust

``∇log q = −u·h(|u|)`` is the isotropic score of the subordinated-Gaussian α-stable
kernel (``h`` a precomputed 1-D table); ``c_in = 1/√(1+W)`` is an EDM-style input scale
that keeps the heavy-tailed noised window ``O(1)`` for the network.

**L_robust** (the same noise-consistency term ``stocks.feishu_midprice.train_jumpgatelob`` uses, here
driven by heavy-tailed rather than jump-diffusion noise) makes the *inference path
itself* tolerant of corruption: ``x̃`` is the α-stable forward applied at a **low t**
(the SNR ≥ 1 region, ᾱ_t ≥ 0.5, so the label is still recoverable), classified at the
head's ``t = 0`` conditioning — deployment never knows the noise level.  CE keeps the
noisy prediction correct; the KL term pulls it toward the model's own clean
prediction.  Without it the α-stable branch only regularises the trunk *indirectly*
through the score head, and the classifier never sees a heavy-tailed window.

The robust pass reuses the score pass's ``c_in`` scaling: unlike the finite-variance
jump-diffusion, an α-stable draw can be enormous even at low ``t``, and an unscaled
window would blow up the trunk.  ``c_in`` is a single per-sample scalar, so it only
changes the window's contrast, not its pattern.

Model selection and early stopping are on **trend-head macro-F1** (feature-only), not
the score loss; train and val F1 are both logged so the noise-fitting gap is visible,
alongside ``noisy_val_f1`` — macro-F1 on α-stable-noised validation windows, the
robustness metric ``L_robust`` is trying to move.

Modes:
  * default    — joint: ``L_cls`` + ``L_score`` each step, plus ``L_robust``
                 only when ``mu_robust > 0`` (shipped configs set it to 0).
  * --baseline — plain classifier: ``L_cls`` only, no diffusion / robustness losses.

Usage::

    uv run python -m stocks.feishu_midprice.train_alphastablelob configs/stocks/feishu_midprice/alphastablelob_h1.json
    uv run python -m stocks.feishu_midprice.train_alphastablelob ... --alpha 1.5
    uv run python -m stocks.feishu_midprice.train_alphastablelob ... --baseline
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.alphastable import AlphaStableDiffusion
from models.alphastablelob import AlphaStableLOB, count_parameters
from stocks.feishu.features import n_features as feishu_n_features
from stocks.feishu_midprice.build import build_datasets, discover_symbols
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


def _low_t_indices(diff, device: torch.device) -> torch.Tensor:
    """Timesteps where signal dominates noise (ᾱ_t ≥ 0.5) — the noise levels the robust
    classification pass draws from, so the trend label is still recoverable from ``x̃``.

    ``diff.sqrt_abar`` is ``√ᾱ_t`` on the cosine schedule, hence the square.  Same
    SNR ≥ 1 criterion as ``crypto.train_jumpgatelob._low_t_indices`` on its VP schedule.
    """
    mask = (diff.sqrt_abar.to(device) ** 2) >= 0.5
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    return idx if len(idx) > 0 else torch.zeros(1, dtype=torch.long, device=device)


def _train_epoch(
    model, diff, low_t, loader, optimizer, lr_sched, config, device, do_diff
):
    model.train()
    grad_clip = config.get("grad_clip", 1.0)
    lam_diff = config.get("lambda_diff", 1.0)
    mu_robust = config.get("mu_robust", 0.5)
    robust_kl = config.get("robust_kl", 1.0)
    label_smoothing = config.get("label_smoothing", 0.0)
    t_max = diff.num_timesteps

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
        score_loss = rob_loss = torch.zeros((), device=device)

        if do_diff:
            # generalized score matching on the α-stable kernel
            t = torch.randint(0, t_max, (b,), device=device)
            x_t, u, c_in = diff.add_noise(x0, t)
            s_target = diff.score_target(u, t)
            s_hat = model.score(c_in * x_t, t)
            score_loss = F.mse_loss(s_hat, s_target)

            # noise-consistent classification: heavy-tailed low-t windows, classified
            # at t=0 conditioning (deployment never knows the noise level).  c_in is
            # the same EDM input scale the score pass uses — an α-stable draw can be
            # enormous even at low t, and an unscaled window would blow up the trunk.
            t_rob = low_t[torch.randint(0, len(low_t), (b,), device=device)]
            x_rob, _, c_in_rob = diff.add_noise(x0, t_rob)
            logits_rob = model.classify(c_in_rob * x_rob)
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
def _noisy_f1(model, diff, low_t, loader, device, max_batches=None):
    """Macro-F1 on α-stable-noised low-t windows — the robustness metric the
    noise-consistency loss is trying to move."""
    model.eval()
    y_true, y_pred = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x0 = batch["x"].to(device).float()
        t_rob = low_t[torch.randint(0, len(low_t), (x0.shape[0],), device=device)]
        x_rob, _, c_in = diff.add_noise(x0, t_rob)
        logits = model.classify(c_in * x_rob)
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


def _run_seed(config, args, seed: int, multi_seed: bool) -> dict:
    """Train and test one seed; returns its test metrics for aggregation."""
    if args.alpha is not None:
        config["astable_alpha"] = args.alpha
    alpha = float(config.get("astable_alpha", 1.7))

    config["seed"] = seed
    generator = set_seed(seed)

    device = resolve_device(config["device"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode = "baseline" if args.baseline else "joint"
    ckpt_dir = (
        Path(config["checkpoint_dir"])
        / f"alphastablelob_{mode}_a{alpha}_{config.get('feature_mode', 'ofi')}_{stamp}"
    )
    if multi_seed:
        ckpt_dir = ckpt_dir.with_name(f"{ckpt_dir.name}_seed{seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_sink = logger.add(ckpt_dir / "train.log", level="DEBUG")

    data_dir = Path(config["data_dir"])
    symbols = discover_symbols(data_dir, config)
    config["n_features"] = feishu_n_features(config)

    logger.info(
        "AlphaStableLOB [Feishu]  mode={} alpha={}  symbols={}",
        mode,
        alpha,
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

    model = AlphaStableLOB(config).to(device)
    d = config["T_past"] * config["n_features"]
    diff = AlphaStableDiffusion(
        d=d,
        num_timesteps=config.get("T_max", 1000),
        alpha=alpha,
        cosine_s=config.get("cosine_s", 0.008),
        num_r=config.get("astable_num_r", 256),
        mc_samples=config.get("astable_mc", 8192),
        clip_q=config.get("astable_clip_q", 0.999),
        seed=seed,
        device=device,
    )
    low_t = _low_t_indices(diff, device)
    logger.info(
        "  params={:.2f}M  gflops/sample={:.3f}  lambda_diff={} mu_robust={}"
        "  low-t region: {} steps (≤ t={})  device={}",
        count_parameters(model) / 1e6,
        log_gflops(model, train_ds, device),
        config.get("lambda_diff", 1.0),
        config.get("mu_robust", 0.5),
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
            model,
            diff,
            low_t,
            train_loader,
            optimizer,
            lr_sched,
            config,
            device,
            do_diff,
        )
        val_f1, val_ce, val_acc = _f1_ce_acc(model, val_loader, device)
        train_f1, _, _ = _f1_ce_acc(model, train_loader, device, train_eval_batches)
        noisy_f1 = _noisy_f1(model, diff, low_t, val_loader, device)
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
        "config",
        nargs="?",
        default="configs/stocks/feishu_midprice/alphastablelob_h1.json",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="stability index in (0,2] (override config['astable_alpha']); "
        "smaller ⇒ heavier tails, 2.0 = Gaussian",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="plain classifier: L_cls only, no diffusion / robustness losses",
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
