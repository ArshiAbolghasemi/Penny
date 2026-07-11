"""Checkpoint loading + model-family dispatch shared by every XAI method.

Every ``crypto.train_*`` script writes ``best.pt`` as
``{"model": state_dict, "config": dict, ...}`` and every model class takes a
single ``config`` dict in its constructor and exposes
``predict(batch, device) -> (B, 3)`` logits (see ``utils/evaluate.py``). This
module is the one place that maps a model name to its class so XAI code
doesn't hardcode imports per method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from models.ctabl import CTABL
from models.dla import DLA
from models.jumpgatelob import JumpGateLOB

# name -> (class, family). "family" mirrors each class's own `family` attribute
# (classifier | joint_diffusion) and controls which XAI paths are valid later
# (e.g. joint_diffusion models must be explained via their `t=0` clean-window
# inference path, never the noised denoiser path).
#
# JointDiT is intentionally excluded from the XAI layer (decision 2026-07-11) —
# it is not a registered XAI model even though models.jointdit.JointDiT and its
# training scripts remain in the codebase.
MODEL_REGISTRY: dict[str, Callable[[dict], nn.Module]] = {
    "ctabl": CTABL,
    "dla": DLA,
    "jumpgatelob": JumpGateLOB,
}


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    config: dict
    checkpoint_path: Path


def load_checkpoint(name: str, checkpoint_path: str | Path, device: torch.device) -> LoadedModel:
    """Load a trained model + its training config from a ``best.pt`` checkpoint.

    Args:
        name:            Key into ``MODEL_REGISTRY`` (e.g. ``"ctabl"``).
        checkpoint_path:  Path to a ``best.pt`` file (as written by the
                          ``crypto.train_*`` scripts), or its containing dir.
        device:           Target device.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )

    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model_cls = MODEL_REGISTRY[name]
    model = model_cls(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return LoadedModel(name=name, model=model, config=config, checkpoint_path=path)
