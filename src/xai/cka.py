"""Do three different inductive biases learn the same representation?

Step A (:mod:`xai.representation`) asks where the trend becomes decodable *inside*
one trunk.  This module compares trunks against each other: a bilinear network
(CTABL), an LSTM dual-attention encoder/decoder (DLA) and a diffusion-conditioned
GRU+attention trunk (JumpGateLOB) all reach ~0.67 test accuracy on the same data
— do they get there through similar representations, or different ones?

**Linear CKA** (Kornblith et al. 2019, "Similarity of Neural Network
Representations Revisited") answers that.  It is invariant to orthogonal
transforms and isotropic scaling — so it does not care that the layers have
different widths or arbitrary basis orientations — while remaining sensitive to
genuine representational differences.  That invariance is what makes a bilinear
layer comparable to a GRU at all.  No training is involved, so unlike a probe
there is no capacity confound.

Comparability rests on one precondition, checked by the runner rather than
assumed: every model must see the **same windows in the same order**, since CKA
compares column spaces over a shared sample axis.  The k10 BTCIRT configs share
all ten data-spec keys, so this holds.

The unbiased HSIC estimator (Song et al. 2012) is used rather than the biased
one: the biased form inflates towards 1 as the sample count approaches the
feature count, which would manufacture agreement exactly where these wide layers
(d up to 128) are most at risk of it.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def _center_gram(k: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix."""
    n = k.shape[0]
    unit = np.ones((n, n)) / n
    return k - unit @ k - k @ unit + unit @ k @ unit


def _hsic_unbiased(k: np.ndarray, l: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012).

    The biased estimator drifts towards 1 as ``n`` approaches the feature
    dimension, which would fabricate similarity between wide layers on small
    samples — the exact regime here.
    """
    n = k.shape[0]
    kk = k.copy()
    ll = l.copy()
    np.fill_diagonal(kk, 0.0)
    np.fill_diagonal(ll, 0.0)
    ones = np.ones(n)
    term1 = float(np.sum(kk * ll))
    term2 = float(ones @ kk @ ones) * float(ones @ ll @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 * float(ones @ kk @ ll @ ones) / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def linear_cka(x: np.ndarray, y: np.ndarray, unbiased: bool = True) -> float:
    """Linear CKA between two activation matrices over the same samples.

    Args:
        x: ``(N, d1)`` activations — rows must be the same samples, in the same
           order, as ``y``.
        y: ``(N, d2)``.  ``d1`` and ``d2`` may differ.

    Returns:
        Similarity in ``[0, 1]`` (1 = representations equal up to an orthogonal
        transform and isotropic scale).
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: {x.shape[0]} vs {y.shape[0]}")
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    kx = x @ x.T
    ky = y @ y.T
    if unbiased:
        hsic_xy = _hsic_unbiased(kx, ky)
        hsic_xx = _hsic_unbiased(kx, kx)
        hsic_yy = _hsic_unbiased(ky, ky)
    else:
        kx = _center_gram(kx)
        ky = _center_gram(ky)
        hsic_xy = float(np.sum(kx * ky))
        hsic_xx = float(np.sum(kx * kx))
        hsic_yy = float(np.sum(ky * ky))
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom <= 0:
        return float("nan")
    return float(hsic_xy / denom)


def cka_matrix(
    acts_a: dict[str, np.ndarray],
    acts_b: dict[str, np.ndarray],
    taps_a: list[str],
    taps_b: list[str],
    unbiased: bool = True,
) -> np.ndarray:
    """Layer × layer CKA between two models' tap activations."""
    m = np.zeros((len(taps_a), len(taps_b)))
    for i, ta in enumerate(taps_a):
        for j, tb in enumerate(taps_b):
            m[i, j] = linear_cka(acts_a[ta], acts_b[tb], unbiased=unbiased)
    return m


def cka_stability(
    acts_a: dict[str, np.ndarray],
    acts_b: dict[str, np.ndarray],
    taps_a: list[str],
    taps_b: list[str],
    n_splits: int = 4,
    seed: int = 42,
    unbiased: bool = True,
) -> dict:
    """CKA over disjoint sample subsets, to see whether the matrix is trustworthy.

    CKA depends on the sample count; a value computed once says nothing about how
    much of it is noise.  Splitting the windows into disjoint halves-of-halves and
    recomputing gives a spread — if that spread is wide, the heatmap is not a
    result.

    Returns ``{"mean", "std", "max_std"}`` where the first two are ``(A, B)``
    matrices over the splits.
    """
    n = next(iter(acts_a.values())).shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    chunks = np.array_split(idx, n_splits)
    mats = []
    for c in chunks:
        sa = {k: v[c] for k, v in acts_a.items()}
        sb = {k: v[c] for k, v in acts_b.items()}
        mats.append(cka_matrix(sa, sb, taps_a, taps_b, unbiased=unbiased))
    stack = np.stack(mats)
    return {
        "mean": stack.mean(0),
        "std": stack.std(0),
        "max_std": float(stack.std(0).max()),
    }


def format_matrix(
    m: np.ndarray, taps_a: list[str], taps_b: list[str], name_a: str, name_b: str
) -> str:
    """Render a CKA matrix as a fixed-width table."""
    w = max(9, max((len(t) for t in taps_b), default=0) + 1)
    corner = f"{name_a} vs {name_b}"
    head = f"{corner:<14}" + "".join(f"{t:>{w}}" for t in taps_b)
    lines = [head, "-" * len(head)]
    for i, ta in enumerate(taps_a):
        lines.append(f"{ta:<14}" + "".join(f"{m[i, j]:>{w}.3f}" for j in range(len(taps_b))))
    return "\n".join(lines)


def log_matrix(
    m: np.ndarray, taps_a: list[str], taps_b: list[str], name_a: str, name_b: str
) -> None:
    logger.info("CKA {} vs {}\n{}", name_a, name_b, format_matrix(m, taps_a, taps_b, name_a, name_b))
