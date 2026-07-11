"""Cross-model attribution comparison (Task 4).

Turns the per-model outputs of Task 2 (shared IG/GradientSHAP,
``xai.gradient_methods``) and Task 3 (native explanations,
``xai.native.explain_native``) into one comparable ``(T, F)`` representation
per model, then computes agreement/divergence metrics across models on the
*same* windows (sampled once via ``xai.sampling.sample_by_class`` and reused
for every model, so "model A attends here, model B attends there" is a fact
about the models, not an artifact of explaining different inputs).

Normalisation note: the three native methods live at different resolutions —
DLA's alpha is a full ``(T, F)`` map; CTABL's time_importance and
JumpGateLOB's pool_attention are ``(T,)`` only (CTABL has no per-feature axis
because TABL's attention lives in the post-projection ``d2`` channel space,
not the input's ``F``; JumpGateLOB's pooling attention sits over GRU latent
context, not raw features — see the native module docstrings). Comparing
them fairly means either (a) broadcasting the time-only signals across ``F``
so every model has a ``(T, F)`` map, and/or (b) also comparing on the reduced
``(T,)`` axis alone, which is the axis all three *do* share. This module does
both — ``to_comparable_map`` for the (T, F) case, ``time_profile`` for the
axis every method can produce.

JointDiT is intentionally excluded from the XAI layer (decision 2026-07-11) —
see docs/xai/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xai.attribution import Attribution
from xai.native.ctabl_attention import TABLExplanation
from xai.native.dla_attention import DLAExplanation
from xai.native.jumpgatelob_readout import JumpGateExplanation

NativeExplanation = TABLExplanation | DLAExplanation | JumpGateExplanation

# Models whose native explanation has genuine per-feature resolution (a real
# (T, F) map) vs. a (T,) time-only signal broadcast across F for display —
# see the normalisation note above. Any per-feature ranking (e.g. "top
# features") computed from a broadcast model's comparable map is a tie by
# construction and should not be presented as a real ranking.
HAS_FEATURE_RESOLUTION = {"dla": True, "ctabl": False, "jumpgatelob": False}


def _normalize(m: torch.Tensor) -> torch.Tensor:
    """Min-max normalise a non-negative map to [0, 1]; all-zero maps pass through."""
    lo, hi = m.min(), m.max()
    if (hi - lo).abs() < 1e-12:
        return torch.zeros_like(m)
    return (m - lo) / (hi - lo)


def time_profile(model_name: str, expl: NativeExplanation, sample_idx: int) -> torch.Tensor:
    """Return the ``(T,)`` time-importance curve every native method can produce.

    This is the one axis all four models share (CTABL and JumpGateLOB have no
    native per-feature resolution), so it's the fairest single number line for
    "which timesteps did each model rely on".
    """
    if isinstance(expl, TABLExplanation):
        # (B, t1) where t1 = ctabl_t2, an already-downsampled time axis —
        # NOT the same length as T. Caller must resample to T separately if
        # aligning against other models; returned as-is here.
        return expl.time_importance[sample_idx]
    if isinstance(expl, DLAExplanation):
        return expl.beta[sample_idx]
    if isinstance(expl, JumpGateExplanation):
        return expl.pool_attention[sample_idx]
    raise TypeError(f"unrecognised native explanation type: {type(expl)}")


def to_comparable_map(
    model_name: str, expl: NativeExplanation, sample_idx: int, T: int, F: int
) -> torch.Tensor:
    """Return a normalised ``(T, F)`` map for any native explanation, for overlay plots.

    Time-only signals (CTABL, JumpGateLOB) are broadcast equally across
    ``F`` — this makes the plot legible side-by-side with DLA's genuine
    per-feature maps, but the broadcast axis carries no real per-feature
    information and should be captioned as such.
    """
    if isinstance(expl, TABLExplanation):
        t1 = expl.time_importance.shape[-1]
        prof = expl.time_importance[sample_idx]  # (t1,)
        if t1 != T:
            prof = torch.nn.functional.interpolate(
                prof.view(1, 1, -1), size=T, mode="nearest"
            ).view(-1)
        return _normalize(prof.unsqueeze(-1).expand(T, F).clone())
    if isinstance(expl, DLAExplanation):
        return _normalize(expl.alpha[sample_idx])  # (T, F) already
    if isinstance(expl, JumpGateExplanation):
        prof = expl.pool_attention[sample_idx]  # (T,)
        return _normalize(prof.unsqueeze(-1).expand(T, F).clone())
    raise TypeError(f"unrecognised native explanation type: {type(expl)}")


@dataclass
class AgreementResult:
    pair: tuple[str, str]
    cosine_similarity: float  # over the flattened (T, F) comparable maps
    time_profile_correlation: float  # Pearson r over the shared (T,) axis
    top_k_overlap: float  # Jaccard overlap of each model's top-k time steps


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _top_k_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    union = top_a | top_b
    if not union:
        return 0.0
    return len(top_a & top_b) / len(union)


def compare_models(
    comparable_maps: dict[str, torch.Tensor],
    time_profiles: dict[str, torch.Tensor],
    top_k: int = 10,
) -> list[AgreementResult]:
    """Pairwise agreement metrics across models on one explained window.

    Args:
        comparable_maps: ``model_name -> (T, F)`` normalised map (from
            :func:`to_comparable_map`, or an IG/GradientSHAP ``Attribution``
            with ``.abs()`` + :func:`_normalize` applied by the caller).
        time_profiles:   ``model_name -> (T,)`` curve (from :func:`time_profile`).
        top_k: Number of top-attended timesteps used for the Jaccard overlap.
    """
    names = sorted(comparable_maps)
    results = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            map_a = comparable_maps[a].detach().cpu().numpy().ravel()
            map_b = comparable_maps[b].detach().cpu().numpy().ravel()
            prof_a = time_profiles[a].detach().cpu().numpy()
            prof_b = time_profiles[b].detach().cpu().numpy()
            # time profiles may differ in length (CTABL's t1 != T); resample
            # the longer/shorter to match before correlating.
            if len(prof_a) != len(prof_b):
                target = max(len(prof_a), len(prof_b))
                prof_a = np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(prof_a)), prof_a)
                prof_b = np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(prof_b)), prof_b)
            results.append(
                AgreementResult(
                    pair=(a, b),
                    cosine_similarity=_cosine(map_a, map_b),
                    time_profile_correlation=_pearson(prof_a, prof_b),
                    top_k_overlap=_top_k_jaccard(prof_a, prof_b, top_k),
                )
            )
    return results


def gradient_attribution_to_comparable_map(attr: Attribution) -> torch.Tensor:
    """Normalise an IG/GradientSHAP ``Attribution`` (signed) into a comparable map.

    Uses ``|scores|`` since native methods (attention/rollout) are already
    non-negative — comparing signed vs. unsigned maps directly would bias
    every agreement metric toward the unsigned method.
    """
    return _normalize(attr.scores.abs())
