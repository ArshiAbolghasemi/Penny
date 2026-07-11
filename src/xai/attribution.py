"""Unified attribution interface: every XAI method returns a ``(T, F)`` map.

All three XAI-registered models (CTABL, DLA, JumpGateLOB — JointDiT is
intentionally excluded from this layer, see docs/xai/README.md) consume the
same ``(B, 1, T, F)`` window (rows = LOB levels / trade features, columns...
actually time is axis 2 and features axis 3 — see ``crypto/features.py``),
and every model exposes ``predict(batch, device) -> (B, 3)`` logits
(``utils/evaluate.py``). Pinning every attribution method to that one shared
``(T, F)`` output lets per-model native methods (attention, gates) and shared
gradient methods (Task 2) be plotted and compared on identical axes.

``ClassifierFn`` wraps whatever forward path a model needs for classification
(joint_diffusion models must go through the clean-window ``t=0`` inference
path, not the noised denoiser) into the single signature every attribution
method below expects: ``(B, 1, T, F) -> (B, 3)`` logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

ClassifierFn = Callable[[torch.Tensor], torch.Tensor]


def classifier_fn(name: str, model: torch.nn.Module) -> ClassifierFn:
    """Return the ``(B,1,T,F) -> (B,3)`` logit function used for attribution.

    ``jumpgatelob`` is a ``joint_diffusion`` model whose ``forward`` also
    returns a denoising output; attribution must target only the
    classification path, evaluated on the clean window (matching what
    ``predict()`` does at inference: ``t = 0``, no added noise).
    """
    if name == "jumpgatelob":
        return lambda x: model.classify(x)
    # classifier-family models (ctabl, dla, ...) take (B,1,T,F) directly.
    return lambda x: model(x)


@dataclass
class Attribution:
    """A single explained sample: attribution map aligned to the input window.

    ``scores``: ``(T, F)`` tensor/array, same axes as the input window
    (``batch["x"][i, 0]``). Sign is preserved where the method supports it
    (e.g. IG); attention-derived maps are non-negative.
    """

    model_name: str
    method: str
    target_class: int
    scores: torch.Tensor  # (T, F)
    input: torch.Tensor  # (T, F) — the explained window, for overlay plots


def target_classes(logits: torch.Tensor, mode: str = "predicted") -> torch.Tensor:
    """Resolve the class index each attribution method should explain.

    ``mode="predicted"`` explains the model's own argmax (what did it actually
    decide); ``mode="true"`` requires passing labels separately and isn't
    handled here — callers explaining against ground truth should index
    ``batch["label"]`` directly instead of using this helper.
    """
    if mode != "predicted":
        raise ValueError(f"unsupported mode: {mode!r}")
    return logits.argmax(dim=1)
