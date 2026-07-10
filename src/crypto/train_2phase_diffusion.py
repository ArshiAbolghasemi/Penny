"""Two-phase diffusion→probe training in ONE command (JointDiT backbone).

Runs both decoupled phases sequentially in a single process — Phase 1 (generative)
then Phase 2 (frozen-trunk probe) — without ever sharing a loss between them:

  PHASE 1  train the generative trunk (A–D) on the PAST window only, one objective
           (``config['objective']`` = ``edm`` denoising score matching, or
           ``drift`` drift-matching).  Classifier head E is excluded from the loss
           graph.  Checkpoints ``trunk.pt``.

  PHASE 2  freeze the trunk, tap intermediate (mid-decoder / U-ViT mid-skip) block
           activations from a SINGLE preconditioned forward per swept ``sigma*``,
           and train ONLY a probe (temporal aggregator + shallow MLP head) with an
           ordinal-aware, class-weighted loss.

The backbone is ``backbone: "dit"`` (:class:`JointDiT`) — the architecture is not
modified; only ``denoise`` is used.  The probe aggregator (``mlp``: mean-pool → MLP,
or ``attn``: attention-pool) is set per config.

Config layout: a shared base plus optional ``phase1`` / ``phase2`` override blocks,
e.g. ``{"epochs": 80, ..., "phase2": {"epochs": 40, "lr": 1e-3}}``.

Usage::

    uv run python -m crypto.train_2phase_diffusion configs/crypto/nobitex/twophase/btcirt_ofi_k10_dit.json
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

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from models.drift import WindowMemoryBank, drift_loss
from models.modules import count_parameters
from models.probe import (
    TemporalProbe,
    TrunkFeatureExtractor,
    build_backbone,
    class_weights_from_labels,
    default_tap_blocks,
    ordinal_ce,
)
from utils.evaluate import run_test
from utils.training import (
    build_cosine_schedule,
    measure_sigma_data,
    resolve_device,
    resolve_seed,
    seed_worker,
    set_seed,
)


# ───────────────────────── Phase 1: generative objectives ──────────────────────


def _sample_sigma(b, config, device):
    """EDM log-normal noise-level sampling, clamped to [sigma_min, sigma_max]."""
    sigma = torch.exp(
        config.get("edm_p_mean", -1.2)
        + config.get("edm_p_std", 1.2) * torch.randn(b, device=device)
    )
    return sigma.clamp(
        config.get("cm_sigma_min", 0.002), config.get("cm_sigma_max", 80.0)
    )


def _edm_precond(sigma, sigma_data):
    """Pure-EDM (Karras 2022) coefficients — no sigma_min boundary, so c_out>0."""
    ss = sigma**2 + sigma_data**2
    c_skip = sigma_data**2 / ss
    c_out = sigma * sigma_data / ss.sqrt()
    c_in = 1.0 / ss.sqrt()
    c_noise = 0.25 * torch.log(sigma.clamp_min(1e-20))
    return c_skip, c_out, c_in, c_noise


def _edm_loss(model, x0, config):
    """EDM denoising score-matching loss (no consistency; classifier discarded).

    Uses pure-EDM preconditioning on the raw network ``model(·)`` rather than the
    model's consistency ``denoise`` (whose ``c_out ∝ (sigma - sigma_min)`` vanishes
    at ``sigma_min`` and blows up the ``1/c_out^2`` weight).  ``c_in``/``c_noise``
    are identical to the consistency path, so the trunk sees the same inputs the
    Phase-2 extractor will feed it.
    """
    sigma = _sample_sigma(x0.shape[0], config, x0.device)
    c_skip, c_out, c_in, c_noise = _edm_precond(sigma, model.sigma_data)
    v = (-1,) + (1,) * (x0.dim() - 1)
    x_sigma = x0 + sigma.view(v) * torch.randn_like(x0)
    raw, _ = model(c_in.view(v) * x_sigma, c_noise)  # E excluded: logits discarded
    x0_hat = c_skip.view(v) * x_sigma + c_out.view(v) * raw
    w = 1.0 / (c_out**2)  # EDM loss weight lambda(sigma) = 1/c_out^2
    return (w * ((x0_hat - x0) ** 2).flatten(1).mean(dim=1)).mean()


def _drift_loss(model, x0, config, device, drift, banks):
    """Drift-matching loss (classifier excluded)."""
    b = x0.shape[0]
    banks["pos"].add(x0, torch.zeros(b, dtype=torch.long))
    z = torch.randn_like(x0)
    sig_hi = torch.full((b,), drift["sigma_max"], device=device)
    x_gen, _ = model.denoise(drift["sigma_max"] * z, sig_hi)
    n_pos, n_neg = drift["pos_per_sample"], drift["neg_per_sample"]
    if banks["pos"].ready(n_pos):
        pos_x, _ = banks["pos"].sample(n_pos)
        phi_gen = x_gen.flatten(1).unsqueeze(0)
        phi_pos = pos_x.to(device).flatten(1).unsqueeze(0)
        phi_neg = None
        if n_neg > 0 and banks["neg"].ready(n_neg):
            neg_x, _ = banks["neg"].sample(n_neg)
            phi_neg = neg_x.to(device).flatten(1).unsqueeze(0)
        loss, _ = drift_loss(phi_gen, phi_pos, phi_neg, R_list=drift["r_list"])
        loss = loss.mean()
    else:
        loss = torch.zeros((), device=device)
    if n_neg > 0:
        banks["neg"].add(x_gen, torch.zeros(b, dtype=torch.long))
    return loss


def run_phase1(config, objective, train_ds, val_ds, device, generator, ckpt_dir):
    """Train the generative trunk (single objective); return the trunk.pt path."""
    config = {**config, "cm_enabled": True}
    if config.get("cm_sigma_data_auto", True):
        config["cm_sigma_data"] = measure_sigma_data(train_ds)
    model = build_backbone(config).to(device)

    drift = banks = None
    if objective == "drift":
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

    def step(x0):
        if objective == "edm":
            return _edm_loss(model, x0, config)
        return _drift_loss(model, x0, config, device, drift, banks)

    logger.info(
        "PHASE 1 [{}]  backbone={}  params={:.2f}M  sigma_data={:.4f}",
        objective,
        config.get("backbone"),
        count_parameters(model) / 1e6,
        config["cm_sigma_data"],
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
    grad_clip = config.get("grad_clip", 1.0)

    best, patience, trunk_path = float("inf"), 0, ckpt_dir / "trunk.pt"
    for epoch in range(config["epochs"]):
        model.train()
        tot = n = 0
        for batch in train_loader:
            loss = step(batch["x"].to(device).float())
            # drift warmup: before the memory bank fills, the loss is a grad-less
            # zero — accumulate it for logging but take no optimizer step.
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_sched.step()
            tot += loss.item()
            n += 1
        model.eval()
        with torch.no_grad():
            vtot = vn = 0
            for batch in val_loader:
                vtot += step(batch["x"].to(device).float()).item()
                vn += 1
        val = vtot / max(vn, 1)
        logger.info(
            "  ph1 epoch {} | train={:.4f} val={:.4f}", epoch, tot / max(n, 1), val
        )
        if val < best:
            best, patience = val, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "backbone": config.get("backbone"),
                    "phase": objective,
                    "epoch": epoch,
                },
                trunk_path,
            )
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("  ph1 early stop at epoch {}", epoch)
                break
    logger.info("PHASE 1 done — best val={:.4f}  trunk → {}", best, trunk_path)
    return trunk_path


# ───────────────────────── Phase 2: frozen-trunk probe ─────────────────────────


class _ProbeInference:
    def __init__(self, extractor, probe, time_len):
        self.extractor, self.probe, self.time_len = extractor, probe, time_len

    def eval(self):
        self.probe.eval()
        return self

    @torch.no_grad()
    def predict(self, batch, device):
        return self.probe(self.extractor(batch["x"].to(device).float(), self.time_len))


@torch.no_grad()
def _accuracy(extractor, probe, loader, time_len, device):
    probe.eval()
    correct = n = 0
    for batch in loader:
        logits = probe(extractor(batch["x"].to(device).float(), time_len))
        correct += (logits.argmax(1) == batch["label"].to(device)).sum().item()
        n += len(batch["label"])
    return correct / max(n, 1)


def run_phase2(
    config, trunk_path, train_ds, val_ds, test_ds, meta, device, generator, ckpt_dir
):
    """Freeze the trunk, train the probe; return the report."""
    ckpt = torch.load(trunk_path, map_location=device, weights_only=False)
    trunk_config = ckpt["config"]
    assert trunk_config["n_features"] == meta["n_features"], (
        "trunk/data feature mismatch"
    )
    trunk = build_backbone(trunk_config).to(device)
    trunk.load_state_dict(ckpt["model"])
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)

    blocks = config.get("probe_blocks") or default_tap_blocks(trunk)
    sigmas = config.get("probe_sigmas", [0.5])
    time_len = int(config.get("probe_time_len", getattr(trunk, "gt", config["T_past"])))
    extractor = TrunkFeatureExtractor(trunk, blocks, sigmas)
    x_probe = torch.stack([train_ds[i]["x"] for i in range(2)]).float().to(device)
    feat_dim = extractor(x_probe, time_len).shape[-1]

    aggregator = config.get("probe_aggregator", "mlp")
    probe = TemporalProbe(
        in_dim=feat_dim,
        hidden=config.get("probe_hidden", 128),
        aggregator=aggregator,
        heads=config.get("probe_heads", 4),
        dropout=config.get("probe_dropout", 0.5),
    ).to(device)
    logger.info(
        "PHASE 2 probe  agg={}  taps: blocks={} sigmas={} → {} taps · feat_dim={} time_len={}",
        aggregator,
        blocks,
        sigmas,
        extractor.n_taps,
        feat_dim,
        time_len,
    )
    logger.info(
        "  trunk(frozen)={:.2f}M  probe(trainable)={:.2f}M",
        count_parameters(trunk) / 1e6,
        count_parameters(probe) / 1e6,
    )

    cw = class_weights_from_labels(
        np.array([train_ds[i]["label"] for i in range(len(train_ds))]), device
    )
    lam_ord = config.get("probe_lambda_ordinal", 0.5)
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
        probe.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("probe_weight_decay", 1e-2),
    )
    lr_sched = build_cosine_schedule(
        optimizer, config, config["epochs"] * len(train_loader)
    )

    best, patience, history, probe_path = 0.0, 0, [], ckpt_dir / "probe.pt"
    for epoch in range(config["epochs"]):
        probe.train()
        tot = n = 0
        for batch in train_loader:
            feats = extractor(batch["x"].to(device).float(), time_len)  # frozen
            loss = ordinal_ce(probe(feats), batch["label"].to(device), cw, lam_ord)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sched.step()
            tot += loss.item()
            n += 1
        val_acc = _accuracy(extractor, probe, val_loader, time_len, device)
        logger.info(
            "  ph2 epoch {} | ord_ce={:.4f} val_acc={:.4f}",
            epoch,
            tot / max(n, 1),
            val_acc,
        )
        history.append({"epoch": epoch, "loss": tot / max(n, 1), "val_acc": val_acc})
        if val_acc > best:
            best, patience = val_acc, 0
            torch.save(
                {
                    "probe": probe.state_dict(),
                    "config": config,
                    "trunk": str(trunk_path),
                    "blocks": blocks,
                    "sigmas": sigmas,
                    "time_len": time_len,
                },
                probe_path,
            )
        else:
            patience += 1
            if patience >= config["patience"]:
                logger.info("  ph2 early stop at epoch {}", epoch)
                break

    probe.load_state_dict(
        torch.load(probe_path, map_location=device, weights_only=False)["probe"]
    )
    test_metrics = run_test(
        _ProbeInference(extractor, probe, time_len).eval(), test_ds, config, device
    )
    extractor.close()
    (ckpt_dir / "phase2_log.json").write_text(json.dumps(history, indent=2))
    return {
        "probe_val_acc": best,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }


# ─────────────────────────────── orchestrator ──────────────────────────────────


def _phase_config(base: dict, phase: str) -> dict:
    """Merge the shared base with the phase-specific override block."""
    cfg = {k: v for k, v in base.items() if k not in ("phase1", "phase2")}
    cfg.update(base.get(phase, {}))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/crypto/nobitex/twophase/btcirt_ofi_k10_dit.json",
    )
    parser.add_argument("--backbone", default=None, help="override config['backbone']")
    parser.add_argument(
        "--objective", default=None, help="edm | drift (override config)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    base = json.loads(config_path.read_text())
    if args.backbone:
        base["backbone"] = args.backbone
    if args.objective:
        base["objective"] = args.objective
    objective = base.get("objective", "edm")

    seed = resolve_seed(base)
    base["seed"] = seed
    generator = set_seed(seed)
    device = resolve_device(base["device"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = (
        Path(base["checkpoint_dir"])
        / f"twophase_{base.get('backbone')}_{objective}_{base['symbol']}"
        f"_{base.get('feature_mode', '')}_{stamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    # One dataset build shared by both phases (same past-window tensor).
    train_ds, val_ds, test_ds, alpha, meta = build_datasets(base)
    base["n_features"] = meta["n_features"]
    logger.info(
        "TWO-PHASE  backbone={}  objective={}  symbol={}  mode={}  k={}  device={}",
        base.get("backbone"),
        objective,
        base["symbol"],
        base.get("feature_mode"),
        base.get("label_k"),
        device,
    )

    p1 = _phase_config(base, "phase1")
    trunk_path = run_phase1(
        p1, objective, train_ds, val_ds, device, generator, ckpt_dir
    )

    p2 = _phase_config(base, "phase2")
    p2["n_features"] = meta["n_features"]
    report = run_phase2(
        p2, trunk_path, train_ds, val_ds, test_ds, meta, device, generator, ckpt_dir
    )

    (ckpt_dir / "config.json").write_text(json.dumps(base, indent=2))
    (ckpt_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info(
        "TWO-PHASE done — probe_val_acc={:.4f} | test_acc={:.4f} macro_f1={:.4f}",
        report["probe_val_acc"],
        report["test_accuracy"],
        report["test_macro_f1"],
    )


if __name__ == "__main__":
    main()
