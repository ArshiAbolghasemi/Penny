"""Do the architectures' attention mechanisms agree with what actually drives them?

Attention says what a model *routed*; Integrated Gradients says what *changed the
logit*.  They are different claims, and the literature is explicit that the first
does not license the second (Jain & Wallace, "Attention is not Explanation";
Wiegreffe & Pinter's reply).  Rather than assert either way, this module measures
it: reduce both signals to a vector over the same axis and rank-correlate them.

What can honestly be compared
-----------------------------
The three models do not expose attention over the same axis, so there is no single
"agreement number" for all of them — pretending otherwise would be the decorative
version of this analysis.

* **Time** (``T_past`` steps) — available for all three, so this is the
  cross-model comparison.  IG contributes ``per_time``; the probes contribute
  DLA's stage-2 ``temporal``, JumpGateLOB's ``pool``, and CTABL's TABL
  ``temporal`` *projected back* through ``bl2.W2`` (the attention lives on the
  halved ``t2`` axis after the second bilinear layer, so a reshape would be
  wrong).
* **Features** (``n_features`` columns) — **only DLA** attends over the input
  features directly (stage-1 ``input``).  CTABL's TABL attention lives over
  learned ``d2`` channels and JumpGateLOB's over timesteps; neither has an input
  feature axis to compare, and inventing one by projecting through learned
  weights would measure the projection, not the model.

Correlations use Spearman (monotone rank agreement, the question actually being
asked) with Kendall's tau reported alongside as the rank-tie-robust check.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Subset


def _dispersion(w: np.ndarray) -> float:
    """How far a weight vector is from uniform, as a fraction of uniform.

    A rank correlation computed over a nearly flat vector is dominated by noise:
    the ranks are real, but the differences being ranked are negligible.  This
    ratio (std / uniform weight) travels with every correlation so the number is
    never read without knowing whether the mechanism discriminated at all.
    """
    n = w.shape[0]
    if n == 0:
        return 0.0
    return float(w.std() / (1.0 / n))


def _rank_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Spearman + Kendall between two same-length score vectors.

    ``dispersion`` describes ``b`` (the attention side); see :func:`_dispersion`.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    rho = spearmanr(a, b)
    tau = kendalltau(a, b)
    return {
        "spearman": float(rho.correlation),
        "spearman_p": float(rho.pvalue),
        "kendall": float(tau.correlation),
        "kendall_p": float(tau.pvalue),
        "dispersion": _dispersion(b),
        "n": int(a.shape[0]),
    }


@torch.no_grad()
def collect_attention(
    model: torch.nn.Module,
    dataset,
    config: dict,
    device: torch.device,
    indices: np.ndarray,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Mean attention readouts over ``indices``, reduced to per-axis vectors.

    Returns whichever of these the architecture supports:

    * ``"time"``    ``(T_past,)`` — importance over input timesteps.
    * ``"feature"`` ``(n_features,)`` — importance over input features (DLA only).
    * ``"lam"``     scalar — CTABL's learned attention-mixing weight.
    """
    model.eval()
    bs = batch_size or config.get("batch_size", 64)
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=bs, shuffle=False, num_workers=0
    )
    name = type(model).__name__

    time_sum = None
    feat_sum = None
    n = 0
    lam = None

    for batch in loader:
        x = batch["x"].to(device).float()
        if name == "JumpGateLOB":
            _, attn = model.classify(x, return_attn=True)
        else:
            _, attn = model(x, return_attn=True)

        if name == "DLA":
            # stage 1: (B, T, F) weights over features at each timestep.
            # Mean over time = how much each feature is attended overall.
            feat = attn["input"].mean(1)  # (B, F)
            # stage 2: (B, T_dec, T) — the last decoder step produces the state
            # the head classifies, so that row is the one that matters.
            tim = attn["temporal"][:, -1, :]  # (B, T)
        elif name == "JumpGateLOB":
            # the trend head pools over timesteps; those weights are what the
            # classifier actually reads.
            tim = attn["pool"]  # (B, T)
            feat = None
        elif name == "CTABL":
            # TABL attention is (B, d2_out=3, t2) on the HALVED time axis that
            # bl2 produced. Project back to the input's T with bl2's temporal
            # transform W2 (T x t2): column j of W2 says how input step i feeds
            # t2 step j, so |W2| @ attn maps attention onto input timesteps.
            a = attn["temporal"].mean(1)  # (B, t2)
            w2 = model.body.bl2.W2.detach().abs()  # (T, t2)
            tim = a @ w2.t()  # (B, T)
            feat = None
            lam = float(attn["lam"])
        else:
            raise ValueError(f"no attention reduction defined for {name}")

        time_sum = tim.sum(0).cpu().numpy() if time_sum is None else time_sum + tim.sum(0).cpu().numpy()
        if feat is not None:
            f = feat.sum(0).cpu().numpy()
            feat_sum = f if feat_sum is None else feat_sum + f
        n += x.shape[0]

    out: dict[str, np.ndarray] = {"time": time_sum / max(n, 1)}
    if feat_sum is not None:
        out["feature"] = feat_sum / max(n, 1)
    if lam is not None:
        out["lam"] = np.array(lam)
    return out


def agreement(
    ig_result: dict, attn: dict[str, np.ndarray], model_name: str
) -> dict:
    """Rank-correlate IG against the attention probes on every shared axis.

    Args:
        ig_result:  Output of :func:`xai.attribution.attribute_dataset`.
        attn:       Output of :func:`collect_attention` on the *same* windows.
        model_name: Label for the returned row.

    Returns:
        ``{"model", "time": {...}, "feature": {...} | None, "lam": float | None}``
        where each axis entry carries Spearman/Kendall and their p-values.
    """
    row: dict = {"model": model_name, "feature": None, "lam": None}
    row["time"] = _rank_agreement(np.asarray(ig_result["per_time"]), attn["time"])
    if "feature" in attn:
        row["feature"] = _rank_agreement(
            np.asarray(ig_result["per_feature"]), attn["feature"]
        )
    if "lam" in attn:
        row["lam"] = float(attn["lam"])
    logger.info(
        "{:<12} time rho={:+.3f} (p={:.1e}){}",
        model_name,
        row["time"]["spearman"],
        row["time"]["spearman_p"],
        ""
        if row["feature"] is None
        else "  feature rho={:+.3f} (p={:.1e})".format(
            row["feature"]["spearman"], row["feature"]["spearman_p"]
        ),
    )
    return row


def format_table(rows: list[dict]) -> str:
    """Render the agreement rows as a fixed-width table for logs / the paper.

    ``disp`` is the attention vector's spread relative to uniform: near 0 the
    mechanism barely discriminates, so its rho ranks near-identical weights and
    should not be read as a strong claim either way.
    """
    head = (
        f"{'model':<12} {'time rho':>9} {'time tau':>9} {'t disp':>7} "
        f"{'feat rho':>9} {'feat tau':>9} {'f disp':>7} {'lam':>6}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        f_rho = "  n/a" if r["feature"] is None else f"{r['feature']['spearman']:+.3f}"
        f_tau = "  n/a" if r["feature"] is None else f"{r['feature']['kendall']:+.3f}"
        f_dsp = "  n/a" if r["feature"] is None else f"{r['feature']['dispersion']:.3f}"
        lam = "   n/a" if r["lam"] is None else f"{r['lam']:.3f}"
        lines.append(
            f"{r['model']:<12} {r['time']['spearman']:+9.3f} {r['time']['kendall']:+9.3f} "
            f"{r['time']['dispersion']:7.3f} {f_rho:>9} {f_tau:>9} {f_dsp:>7} {lam:>6}"
        )
    return "\n".join(lines)
