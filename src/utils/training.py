"""Shared training utilities for all crypto + equity model families.

Provides ``resolve_device`` (cuda→mps→cpu fallback), ``build_cosine_schedule``
(linear warmup + cosine decay), and reproducibility helpers (``resolve_seed`` /
``set_seed`` / ``seed_worker``) so every model family trains under an identical,
seed-controlled protocol — a prerequisite for a fair cross-model comparison.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR


def resolve_seed(config: dict) -> int:
    """Resolve the run seed.  Precedence: ``$PENNY_SEED`` > ``config["seed"]`` > 42.

    The env override lets one config be launched across several seeds (e.g.
    ``PENNY_SEED=1,2,3``) without editing files — use it to report mean ± std
    instead of a single noisy run.
    """
    env = os.environ.get("PENNY_SEED")
    if env is not None and env.strip():
        return int(env)
    return int(config.get("seed", 42))


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
