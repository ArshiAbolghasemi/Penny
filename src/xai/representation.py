"""Where in a trunk does trend information become linearly decodable?

The attribution layer (:mod:`xai.attribution`) answers *which inputs* move the
logit.  This module answers a different question — *which layers* hold the
signal — by tapping frozen intermediate activations and fitting a **linear**
classifier on each.

Deliberately linear.  A deeper probe (an MLP, or ``models.probe.TemporalProbe``)
can manufacture decodability the layer never had, which is exactly the confound
that makes probe results unfalsifiable: past some capacity you are measuring the
probe, not the representation.  A single ``Linear(d, 3)`` on mean-pooled
activations keeps the claim honest — "a linear readout of this layer recovers the
trend at rate X".

Every probe is reported against a **shuffled-label control** fitted the same way.
Without it a probe number is unreadable: high accuracy could just mean the layer
is wide, and a linear map onto 3 classes over enough dimensions can fit noise.

Tap points are the natural architectural boundaries, verified to hook cleanly:

===========  =====================================  ===================
model        taps                                   activation
===========  =====================================  ===================
CTABL        ``body.bl1`` → ``body.bl2``            ``(B, d, T)``
DLA          ``encoder`` → ``decoder``              ``(B, T, d)`` / ``(B, d)``
JumpGateLOB  ``gru`` → ``temporal`` → ``pool``      ``(B, T, d)`` / ``(B, d)``
===========  =====================================  ===================

CTABL's ``tabl`` is deliberately *not* tapped: it emits ``(B, 3, 1)`` — the
logits themselves — so probing it would measure the classifier, not a
representation.  Note the axis order differs across families (CTABL is
feature-major, the others time-major); each tap declares its own layout rather
than relying on a shape heuristic that would silently pool the wrong axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Subset

from models.probe import class_weights_from_labels


@dataclass(frozen=True)
class Tap:
    """One probe site: a dotted module path plus how to read its output.

    Args:
        name:   Short label used in tables.
        path:   Dotted attribute path from the model root.
        layout: ``"btd"`` (B, T, d) · ``"bdt"`` (B, d, T) · ``"bd"`` (already
                pooled).  Declared, not inferred: a square activation would make
                any shape heuristic silently ambiguous.
        index:  Element to take when the module returns a tuple (e.g. GRU).
    """

    name: str
    path: str
    layout: str
    index: int | None = None


# Ordered shallow → deep. Depth is *within* a trunk; the models have different
# layer counts, so absolute positions are not comparable across them.
TAPS: dict[str, tuple[Tap, ...]] = {
    "CTABL": (
        Tap("bl1", "body.bl1", "bdt"),
        Tap("bl2", "body.bl2", "bdt"),
    ),
    "DLA": (
        Tap("encoder", "encoder", "btd"),
        Tap("decoder", "decoder", "bd"),
    ),
    "JumpGateLOB": (
        Tap("gru", "gru", "btd", index=0),
        Tap("temporal", "temporal", "btd"),
        Tap("pool", "pool", "bd"),
    ),
}


def _resolve(model: nn.Module, path: str) -> nn.Module:
    mod = model
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def _pool(act: torch.Tensor, layout: str) -> torch.Tensor:
    """Collapse a tap activation to ``(B, d)`` by averaging over time."""
    if layout == "bd":
        return act
    if layout == "btd":
        return act.mean(dim=1)
    if layout == "bdt":
        return act.mean(dim=2)
    raise ValueError(f"layout must be btd|bdt|bd, got {layout!r}")


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    dataset,
    config: dict,
    device: torch.device,
    indices: np.ndarray,
    taps: tuple[Tap, ...],
    batch_size: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Frozen, time-pooled activations at every tap, plus the labels.

    Returns ``({tap_name: (N, d)}, labels (N,))`` — run through ``predict`` so the
    activations come from the same path the reported metrics do.
    """
    model.eval()
    bs = batch_size or config.get("batch_size", 64)
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=bs, shuffle=False
    )
    grabbed: dict[str, torch.Tensor] = {}
    handles = []

    def _hook(tap: Tap):
        def fn(_m, _i, out):
            if isinstance(out, tuple):
                out = out[tap.index if tap.index is not None else 0]
            grabbed[tap.name] = out.detach()

        return fn

    for tap in taps:
        handles.append(_resolve(model, tap.path).register_forward_hook(_hook(tap)))

    chunks: dict[str, list[np.ndarray]] = {t.name: [] for t in taps}
    labels: list[int] = []
    try:
        for batch in loader:
            grabbed.clear()
            model.predict({"x": batch["x"]}, device)
            for tap in taps:
                chunks[tap.name].append(
                    _pool(grabbed[tap.name].float(), tap.layout).cpu().numpy()
                )
            labels.extend(batch["label"].tolist())
    finally:
        for h in handles:
            h.remove()

    return (
        {k: np.concatenate(v, axis=0) for k, v in chunks.items()},
        np.asarray(labels),
    )


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Fit ``Linear(d, 3)`` on frozen activations; return test accuracy / macro-F1.

    Activations are standardised with the *training* split's statistics only —
    fitting the scaler on the test split would leak, and the whole point of this
    number is that it is honest.
    """
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    mu = train_x.mean(0, keepdims=True)
    sd = train_x.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    tx = torch.from_numpy(((train_x - mu) / sd).astype(np.float32)).to(device)
    ex = torch.from_numpy(((test_x - mu) / sd).astype(np.float32)).to(device)
    ty = torch.from_numpy(train_y.astype(np.int64)).to(device)

    probe = nn.Linear(tx.shape[1], 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights_from_labels(train_y, device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(tx), ty)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(ex).argmax(1).cpu().numpy()
    return {
        "accuracy": float((pred == test_y).mean()),
        "macro_f1": float(
            f1_score(test_y, pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "dim": int(train_x.shape[1]),
    }


def probe_layers(
    model: nn.Module,
    train_ds,
    test_ds,
    config: dict,
    device: torch.device,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    **probe_kw,
) -> list[dict]:
    """Linear-probe every tap of ``model``, each against a shuffled-label control.

    The control is fitted on the same activations with permuted training labels:
    it measures what this layer's width and the probe's capacity can fit *without*
    any real signal, which is the only thing that makes the real number readable.
    """
    name = type(model).__name__
    if name not in TAPS:
        raise ValueError(f"no taps defined for {name}; known: {sorted(TAPS)}")
    taps = TAPS[name]

    tr_acts, tr_y = collect_activations(
        model, train_ds, config, device, train_idx, taps
    )
    te_acts, te_y = collect_activations(model, test_ds, config, device, test_idx, taps)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(tr_y)

    rows = []
    for i, tap in enumerate(taps):
        real = fit_linear_probe(
            tr_acts[tap.name], tr_y, te_acts[tap.name], te_y, device, seed=seed,
            **probe_kw,
        )
        ctrl = fit_linear_probe(
            tr_acts[tap.name], shuffled, te_acts[tap.name], te_y, device, seed=seed,
            **probe_kw,
        )
        row = {
            "model": name,
            "tap": tap.name,
            "depth": i + 1,
            "n_taps": len(taps),
            # unequal trunk depths (CTABL 2, DLA 2, JumpGateLOB 3) make absolute
            # positions incomparable; normalised depth is what travels
            "depth_frac": (i + 1) / len(taps),
            "dim": real["dim"],
            "accuracy": real["accuracy"],
            "macro_f1": real["macro_f1"],
            "control_accuracy": ctrl["accuracy"],
            "delta": real["accuracy"] - ctrl["accuracy"],
        }
        rows.append(row)
        logger.info(
            "{:<12} {:<9} d={:<4} acc={:.4f} f1={:.4f} | shuffled={:.4f} "
            "delta={:+.4f}",
            name,
            tap.name,
            row["dim"],
            row["accuracy"],
            row["macro_f1"],
            row["control_accuracy"],
            row["delta"],
        )
    return rows


def format_table(rows: list[dict]) -> str:
    """Render probe rows as a fixed-width table."""
    head = (
        f"{'model':<12} {'tap':<9} {'depth':>5} {'dim':>5} {'acc':>7} "
        f"{'f1':>7} {'shuffled':>9} {'delta':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['model']:<12} {r['tap']:<9} {r['depth']}/{r['n_taps']:<3} "
            f"{r['dim']:>5} {r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} "
            f"{r['control_accuracy']:>9.4f} {r['delta']:>+7.4f}"
        )
    return "\n".join(lines)
