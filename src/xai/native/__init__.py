"""Per-model native explanation methods (Task 3).

Each submodule reads out a genuinely load-bearing part of one model's forward
pass (an attention weight, a learned gate) rather than approximating it with
a post-hoc method — see the module docstrings for why each is the most
faithful choice for its architecture:

  - ``ctabl_attention``      — CTABL's TABL soft-attention + mixing scalar.
  - ``dla_attention``        — DLA's dual-stage input (alpha) + temporal (beta) attention.
  - ``jointdit_rollout``     — JointDiT's Attention Rollout across DiT blocks.
  - ``jumpgatelob_readout``  — JumpGateLOB's AttentionPool weights + Levy gate (pi, logW).

``explain_native`` dispatches on the model name so Task 4's cross-model
comparison can call one function regardless of which model it's explaining.
"""

from __future__ import annotations

import torch

from models.ctabl import CTABL
from models.dla import DLA
from models.jointdit import JointDiT
from models.jumpgatelob import JumpGateLOB

from .ctabl_attention import TABLExplanation, extract_tabl_attention
from .dla_attention import DLAExplanation, extract_dla_attention
from .jointdit_rollout import RolloutExplanation, extract_attention_rollout
from .jumpgatelob_readout import JumpGateExplanation, extract_jumpgate_readout

NativeExplanation = (
    TABLExplanation | DLAExplanation | RolloutExplanation | JumpGateExplanation
)


def explain_native(model_name: str, model: torch.nn.Module, x: torch.Tensor) -> NativeExplanation:
    """Dispatch to the right native-explanation function for ``model_name``.

    Args:
        model_name: One of ``ctabl``, ``dla``, ``jointdit``, ``jumpgatelob``
            (matches ``xai.registry.MODEL_REGISTRY`` keys).
        model:      The loaded model instance.
        x:          ``(B, 1, T, F)`` clean input windows.
    """
    if model_name == "ctabl":
        assert isinstance(model, CTABL)
        return extract_tabl_attention(model, x)
    if model_name == "dla":
        assert isinstance(model, DLA)
        return extract_dla_attention(model, x)
    if model_name == "jointdit":
        assert isinstance(model, JointDiT)
        return extract_attention_rollout(model, x)
    if model_name == "jumpgatelob":
        assert isinstance(model, JumpGateLOB)
        return extract_jumpgate_readout(model, x)
    raise ValueError(f"no native explanation for model {model_name!r}")
