"""Shared training utilities for all crypto + equity model families.

Provides ``resolve_device`` (cuda→mps→cpu fallback), ``build_cosine_schedule``
(linear warmup + cosine decay), and reproducibility helpers (``resolve_seed`` /
``set_seed`` / ``seed_worker``) so every model family trains under an identical,
seed-controlled protocol — a prerequisite for a fair cross-model comparison.

For seed sweeps, ``add_seed_args`` / ``resolve_seeds`` / ``summarize_seed_runs``
let any training script train the same config once per seed and report test
accuracy and macro-F1 as mean ± std across seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR

#: Test metrics aggregated across seeds by :func:`summarize_seed_runs`.
SEED_METRICS = ("accuracy", "macro_f1")


def _parse_seed_list(raw) -> list[int]:
    """Parse seeds from a string / list of strings; comma *or* space separated.

    Accepts ``"1,2,3"``, ``"1 2 3"`` and ``["1,2", "3"]`` alike, so the same parser
    serves ``$PENNY_SEED`` and ``--seeds`` without the caller normalising first.
    Duplicates are dropped (order preserved) — training the same seed twice would
    contribute a spurious zero-variance sample to the reported std.
    """
    parts = raw if isinstance(raw, (list, tuple)) else [raw]
    tokens = [t for p in parts for t in str(p).replace(",", " ").split()]
    seeds: list[int] = []
    for tok in tokens:
        try:
            s = int(tok)
        except ValueError:
            raise ValueError(f"seed must be an integer, got {tok!r}") from None
        if s not in seeds:
            seeds.append(s)
    return seeds


def resolve_seed(config: dict) -> int:
    """Resolve a single run seed.  Precedence: ``$PENNY_SEED`` > ``config["seed"]`` > 42.

    Kept for callers that train exactly once; multi-seed entry points should use
    :func:`resolve_seeds`.  If ``$PENNY_SEED`` holds a list, the first seed wins.
    """
    env = os.environ.get("PENNY_SEED")
    if env is not None and env.strip():
        seeds = _parse_seed_list(env)
        if len(seeds) > 1:
            logger.warning(
                "PENNY_SEED={} lists {} seeds but this caller trains once; using {}",
                env,
                len(seeds),
                seeds[0],
            )
        if seeds:
            return seeds[0]
    return int(config.get("seed", 42))


def resolve_seeds(config: dict, cli=None) -> list[int]:
    """Resolve the list of seeds to train over.

    Precedence: ``--seeds`` > ``$PENNY_SEED`` > ``config["seeds"]`` > the single
    seed from :func:`resolve_seed`.  Always returns at least one seed, so a caller
    that passes nothing behaves exactly as it did before seeds were sweepable.
    """
    if cli:
        return _parse_seed_list(cli)
    env = os.environ.get("PENNY_SEED")
    if env is not None and env.strip():
        seeds = _parse_seed_list(env)
        if seeds:
            return seeds
    cfg = config.get("seeds")
    if cfg:
        seeds = _parse_seed_list(cfg)
        if seeds:
            return seeds
    return [resolve_seed(config)]


def add_seed_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--seeds`` flag to a training script's parser."""
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help="train once per seed and report mean ± std of the test metrics "
        "(e.g. --seeds 0 1 2 or --seeds 0,1,2). Falls back to $PENNY_SEED, then "
        "config['seeds'], then the single config['seed'].",
    )


def summarize_seed_runs(runs, out_dir=None) -> dict:
    """Aggregate per-seed test metrics into mean ± std and persist the summary.

    Args:
        runs: one dict per seed as returned by a script's per-seed entry point —
            ``{"seed": int, "run_dir": str, "accuracy": float, "macro_f1": float}``.
        out_dir: where to write ``<run>_seed_summary.json``; defaults to the parent
            of the first run directory (i.e. ``config["checkpoint_dir"]``).

    The spread reported here is **training (seed) variance on a fixed test set** —
    it is not a confidence interval for the metric itself, and it is not the
    dependence-aware paired interval that ``scripts/stochlob_significance.py``
    computes for model-vs-model differences.  Quote it as "mean ± std over N seeds".

    Returns the summary dict (also written to JSON).  ``std`` is the sample
    standard deviation (``ddof=1``) and is ``None`` for a single seed, where the
    spread is undefined rather than zero.
    """
    runs = [r for r in runs if r]
    if not runs:
        logger.warning("seed summary skipped: no completed runs")
        return {}

    summary = {
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "runs": runs,
        "metrics": {},
    }
    for key in SEED_METRICS:
        vals = [float(r[key]) for r in runs if r.get(key) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else None
        summary["metrics"][key] = {
            "mean": float(arr.mean()),
            "std": std,
            "sem": (std / math.sqrt(len(arr))) if std is not None else None,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": vals,
        }

    logger.info("SEED SWEEP  n={}  seeds={}", summary["n_seeds"], summary["seeds"])
    for key, m in summary["metrics"].items():
        logger.info(
            "  {:<10} mean={:.4f}  std={}  min={:.4f}  max={:.4f}",
            key,
            m["mean"],
            "n/a" if m["std"] is None else f"{m['std']:.4f}",
            m["min"],
            m["max"],
        )
        logger.info(
            "  {:<10} per-seed: {}",
            key,
            "  ".join(f"{r['seed']}={float(r[key]):.4f}" for r in runs if key in r),
        )

    first = Path(runs[0]["run_dir"])
    out_dir = Path(out_dir) if out_dir is not None else first.parent
    base = re.sub(r"_seed-?\d+$", "", first.name)
    out_path = out_dir / f"{base}_seed_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("  summary → {}", out_path)
    summary["summary_path"] = str(out_path)
    return summary


def set_seed(seed: int) -> torch.Generator:
    """Seed Python / NumPy / Torch (CPU + all CUDA devices) for reproducibility.

    Returns a CPU ``torch.Generator`` to hand to ``DataLoader(generator=...)`` so
    shuffling order is deterministic too.  ``cudnn.benchmark`` is disabled so the
    same seed yields the same run on the same hardware; we deliberately do *not*
    force ``use_deterministic_algorithms`` (some conv kernels lack a deterministic
    backward and would raise), which is fine here — every model still gets the
    identical seed, data order, and init protocol.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` — reseed NumPy/random per worker.

    Without this, multi-worker shuffling reintroduces nondeterminism even after
    ``set_seed``.  Pass alongside ``generator=`` from :func:`set_seed`.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    """Return a ``torch.device``, falling back gracefully when hardware is absent.

    Priority: ``"cuda"`` → MPS (Apple Silicon) → CPU.
    """
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.warning("cuda unavailable; falling back to mps")
            return torch.device("mps")
        logger.warning("cuda unavailable; falling back to cpu")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("mps unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def measure_sigma_data(dataset, max_windows: int = 4096) -> float:
    """Empirical EDM ``sigma_data`` — the std of the (clean) training windows.

    EDM/consistency preconditioning is calibrated to the standard deviation of the
    data distribution the network denoises.  For LOB windows the inputs are the
    causally z-scored feature images, whose std is close to but not exactly 1
    (trailing-window normalization lets current values exceed unit scale), so a
    hardcoded guess miscalibrates ``c_skip``/``c_out``.  This measures it directly
    from up to ``max_windows`` training windows (population std over all elements).

    Returns a positive float (falls back to 1.0 on an empty/degenerate dataset).
    """
    n = len(dataset)
    if n == 0:
        return 1.0
    step = max(1, n // max_windows)
    total = sq = count = 0.0
    for i in range(0, n, step):
        x = dataset[i]["x"]
        x = x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
        total += x.sum().item()
        sq += (x * x).sum().item()
        count += x.numel()
    if count == 0:
        return 1.0
    mean = total / count
    var = max(sq / count - mean * mean, 0.0)
    return math.sqrt(var) or 1.0


def build_cosine_schedule(optimizer, config: dict, total_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay over ``total_steps``.

    Args:
        optimizer:   The optimizer to wrap.
        config:      Must contain ``"warmup_steps"`` (int).
        total_steps: Total number of scheduler ``step()`` calls planned.
    """
    warmup = config.get("warmup_steps", 500)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)
