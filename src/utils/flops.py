"""Uniform inference-GFLOPs measurement for every crypto model.

All models expose ``predict(batch, device) -> (B, 3)`` logits, so a single
``predict`` on a one-sample batch gives a fair per-sample inference FLOP count
(for diffusion classifiers this includes their sampling loop, i.e. the true
inference cost).  Uses ``torch.utils.flop_counter.FlopCounterMode``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def measure_gflops(model, batch: dict, device: torch.device) -> float:
    """Total GFLOPs of one ``model.predict(batch, device)`` call."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            model.predict(batch, device)
        gflops = counter.get_total_flops() / 1e9
    except Exception:
        gflops = float("nan")
    finally:
        if was_training:
            model.train()
    return gflops


def log_gflops(model, dataset: Dataset, device: torch.device) -> float:
    """Measure per-sample inference GFLOPs using the first sample of ``dataset``."""
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    return measure_gflops(model, batch, device)
