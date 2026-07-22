"""Integrated Gradients over the LOB input window.

Model-agnostic by construction: every Penny classifier exposes
``predict(batch, device) → (B, 3)``, so one implementation covers CTABL, DLA and
JumpGateLOB with no per-model branching, and the numbers are directly
comparable across them.

The baseline
------------
Attribution is always *relative to a baseline* — IG explains ``F(x) −
F(baseline)``, so the baseline is what "no contribution" means and it is a
modelling decision, not a default to be taken on faith.

The default here is **zeros**, which in this pipeline is the genuine
no-order-flow state rather than an arbitrary origin:

* features are z-scored by a *causal trailing rolling* window (``crypto.loader``),
  and per-level OFI is signed and near-stationary, so its rolling mean sits at
  ~0.  A true no-flow row (raw OFI = 0) therefore lands at z ≈ 0.04 ± 0.08 at
  level 1, tightening to ≈ −0.003 ± 0.012 by level 20 — normalised zero *is*
  raw zero, to within a rounding error;
* raw per-level OFI is exactly zero in 25–36% of bins (quiet book), so the
  baseline sits on the single most common state in the data rather than
  off-manifold — the usual objection to zero baselines does not bite here;
* "what would this prediction be with no order flow?" is the counterfactual an
  OFI paper is asking, and it is Cont's own null.

``baseline="mean"`` (the training-window mean) is provided as the robustness
check: feature rankings agree with the zero baseline at Spearman ≈ 0.99, so
conclusions do not hinge on the choice — which is worth reporting rather than
assuming.

Note that the rolling z-score makes the ``z → raw`` map *time-varying*:
attributions are comparable in normalised space, but converting them to raw OFI
units requires de-scaling each row by that row's own ``std``, never a single
global constant.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class _PredictWrapper(torch.nn.Module):
    """Adapt ``predict(batch, device)`` to the tensor→tensor callable captum wants.

    Keeping the wrapper this thin is the point: attribution runs against the same
    logits the reported test metrics came from, so it cannot drift from the
    evaluated model.

    The joint-diffusion models (JumpGateLOB, AlphaStableLOB) decorate ``predict``
    with ``@torch.no_grad()`` — right for their inference path, fatal for IG,
    which needs a graph back to the input.  For those we call the undecorated
    ``classify`` the decorated ``predict`` itself delegates to, so the computation
    is identical and only the no-grad guard is dropped.  ``predict`` is left
    exactly as it is.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        # classify(x) is predict()'s own body for the joint-diffusion family
        self._classify = getattr(model, "classify", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._classify is not None:
            return self._classify(x)
        return self.model.predict({"x": x}, self.device)


def make_baseline(
    kind: str | torch.Tensor,
    like: torch.Tensor,
    dataset: Dataset | None = None,
    n_windows: int = 2000,
) -> torch.Tensor:
    """Build the IG baseline tensor, broadcast to ``like``'s shape.

    Args:
        kind:      ``"zero"`` (no order flow — the default and the one with a
                   clean economic reading), ``"mean"`` (mean training window, the
                   robustness check), or an explicit tensor.
        like:      Tensor whose shape/dtype/device the baseline must match.
        dataset:   Required for ``"mean"``; sampled for the average window.
        n_windows: How many windows to average for ``"mean"``.
    """
    if isinstance(kind, torch.Tensor):
        return kind.to(like.device, like.dtype).expand_as(like)
    if kind == "zero":
        return torch.zeros_like(like)
    if kind == "mean":
        if dataset is None:
            raise ValueError("baseline='mean' requires a dataset to average over")
        n = min(n_windows, len(dataset))
        w = torch.stack([dataset[i]["x"] for i in range(n)]).mean(0, keepdim=True)
        return w.to(like.device, like.dtype).expand_as(like)
    raise ValueError(f"baseline must be 'zero'|'mean'|Tensor, got {kind!r}")


# Windows × n_steps the network may see in one IG call.  1024 runs comfortably on
# CPU for all three models, including DLA's per-timestep Python LSTM loop; the
# training batch_size (64) would put 4096 through it and get the process killed.
_IG_BATCH_BUDGET = 1024


def _ig_batch_size(requested: int | None, n_steps: int, config: dict) -> int:
    """Windows per IG call, capped so ``batch_size × n_steps`` stays affordable."""
    if requested is not None:
        return requested
    return max(1, min(config.get("batch_size", 64), _IG_BATCH_BUDGET // max(n_steps, 1)))


@torch.no_grad()
def _predicted_targets(model, batch, device) -> torch.Tensor:
    return model.predict(batch, device).argmax(1)


def attribute_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    config: dict,
    device: torch.device,
    baseline: str | torch.Tensor = "zero",
    n_windows: int | None = 2048,
    n_steps: int = 64,
    batch_size: int | None = None,
    target: str = "predicted",
    seed: int = 42,
) -> dict:
    """Integrated Gradients over ``dataset``, reduced to reusable summaries.

    Args:
        model:      Any model with the shared ``predict`` contract.
        dataset:    Windowed dataset; items give ``"x"`` ``(1, T, F)`` and ``"label"``.
        config:     Model config (``batch_size`` used when not overridden).
        device:     Target device.
        baseline:   See :func:`make_baseline`; defaults to the zero (no-flow) baseline.
        n_windows:  Random subsample size; ``None`` uses the whole dataset.  IG costs
                    ``n_steps`` forward+backward passes per window, so the full test
                    split is rarely worth it.
        n_steps:    Riemann steps along the path.  Completeness error is reported —
                    raise this if it is not small.
        batch_size: Windows per IG call.  **Not** the forward-pass batch: captum
                    expands each window ``n_steps`` times, so the network really
                    sees ``batch_size × n_steps`` samples at once.  Defaults to
                    keeping that product near :data:`_IG_BATCH_BUDGET` rather
                    than reusing the training ``batch_size``, which silently
                    exhausts memory on the recurrent models (DLA's 60-step
                    Python LSTM loop at 64×64 = 4096 windows is killed by the
                    OS with no traceback).
        target:     ``"predicted"`` attributes each window's own argmax class —
                    explaining what the model *did*.  ``"label"`` attributes the
                    ground-truth class instead.
        seed:       Subsample seed.

    Returns:
        ``{"per_feature", "per_time", "attr_mean", "targets", "labels",
        "completeness_err", "n_windows", "n_steps", "baseline"}`` — ``per_feature``
        is ``(F,)`` mean ``|attr|`` over windows and time, ``per_time`` is ``(T,)``,
        and ``attr_mean`` is the signed ``(T, F)`` mean.
    """
    model.eval()
    bs = _ig_batch_size(batch_size, n_steps, config)

    idx = np.arange(len(dataset))
    if n_windows is not None and n_windows < len(idx):
        idx = np.random.default_rng(seed).choice(idx, n_windows, replace=False)
        idx.sort()  # keep chronological order; cheaper memmap paging
    subset = torch.utils.data.Subset(dataset, idx.tolist())
    loader = DataLoader(subset, batch_size=bs, shuffle=False, num_workers=0)

    wrapper = _PredictWrapper(model, device)
    ig = IntegratedGradients(wrapper)

    sum_abs = None  # (T, F) running |attr|
    sum_signed = None
    per_time_chunks: list[np.ndarray] = []
    targets: list[int] = []
    labels: list[int] = []
    delta_max = 0.0
    n_seen = 0

    for batch in loader:
        x = batch["x"].to(device).float()
        tgt = (
            _predicted_targets(model, batch, device)
            if target == "predicted"
            else batch["label"].to(device).long()
        )
        base = make_baseline(baseline, x, dataset)

        # cuDNN refuses to backprop through an RNN that is in eval() mode
        # ("cudnn RNN backward can only be called in training mode"), and IG needs
        # that backward pass — this bites JumpGateLOB/DLA on GPU but never on CPU.
        # Disabling the cuDNN RNN path falls back to the native kernel, which has
        # no train/eval restriction and is numerically equivalent. Scoped to the
        # attribute() call so nothing else changes; a no-op on CPU.
        with torch.backends.cudnn.flags(enabled=False):
            attr, delta = ig.attribute(
                x,
                baselines=base,
                target=tgt,
                n_steps=n_steps,
                return_convergence_delta=True,
            )
        a = attr.squeeze(1).detach()  # (B, T, F)
        abs_a = a.abs()

        chunk_abs = abs_a.sum(0).cpu().numpy()
        chunk_signed = a.sum(0).cpu().numpy()
        sum_abs = chunk_abs if sum_abs is None else sum_abs + chunk_abs
        sum_signed = chunk_signed if sum_signed is None else sum_signed + chunk_signed
        per_time_chunks.append(abs_a.mean(-1).cpu().numpy())  # (B, T)

        targets.extend(tgt.cpu().tolist())
        labels.extend(batch["label"].tolist())
        delta_max = max(delta_max, float(delta.abs().max()))
        n_seen += x.shape[0]

    mean_abs = sum_abs / max(n_seen, 1)  # (T, F)
    out = {
        "per_feature": mean_abs.mean(0),  # (F,)
        "per_time": mean_abs.mean(1),  # (T,)
        "attr_mean": sum_signed / max(n_seen, 1),  # (T, F) signed
        "per_window_time": np.concatenate(per_time_chunks, 0),  # (N, T)
        "targets": np.array(targets),
        "labels": np.array(labels),
        "completeness_err": delta_max,
        "n_windows": n_seen,
        "n_steps": n_steps,
        "baseline": baseline if isinstance(baseline, str) else "tensor",
    }
    logger.info(
        "IG  windows={} steps={} batch={} (x{} = {} fwd) baseline={} | "
        "max completeness err={:.3e}",
        n_seen,
        n_steps,
        bs,
        n_steps,
        bs * n_steps,
        out["baseline"],
        delta_max,
    )
    return out
