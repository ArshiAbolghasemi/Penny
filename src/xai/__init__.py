"""Explainability (XAI) layer for the Penny trend classifiers.

Scope: JumpGateLOB, CTABL and DLA on the nobitex/BTCIRT ``k10`` checkpoints.

Two complementary views, deliberately kept distinct:

* **Attribution** (:mod:`xai.attribution`) — Integrated Gradients over the input
  window via the shared ``predict(batch, device) → (B, 3)`` contract.  This is
  what actually moves the logit, and it is the reference the other view is
  measured against.
* **Mechanism probes** — each architecture's own attention readouts, reached
  with ``return_attn=True``.  These show what the model *routed*, which is not
  the same claim as what *changed the output* (Jain & Wallace, "Attention is not
  Explanation"), so they are never reported as attributions.
"""

from __future__ import annotations

from .features import FeatureGroups, feature_groups, feature_names, group_attribution

__all__ = [
    "FeatureGroups",
    "feature_groups",
    "feature_names",
    "group_attribution",
]
