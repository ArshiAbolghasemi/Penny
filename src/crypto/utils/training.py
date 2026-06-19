"""Shared training utilities for all crypto model families.

Provides ``resolve_device`` (cuda→mps→cpu fallback) and ``build_cosine_schedule``
(linear warmup + cosine decay), which appear identically in every train.py.
"""

from __future__ import annotations

import math

import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR


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
