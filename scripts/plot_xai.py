"""Turn the XAI-layer artefacts into the paper's figures.

Usage::

    uv run python scripts/plot_xai.py checkpoints/nobitex/BTCIRT/xai_results
    uv run python scripts/plot_xai.py <results_dir> --outdir docs/xai/figures

Reads the JSON / NPZ files written by ``python -m xai.run_all`` (or the
individual runners) and renders one publication figure per analysis, covering all
three layers of the story:

* **feature attribution** — per-group IG share (grouped bars, all models) and the
  signed ``(time × feature)`` attribution heatmap per model;
* **faithfulness** — deletion / insertion curves vs. the random control;
* **layer attribution** — probe accuracy vs. normalised depth with the
  shuffled-label control band;
* **cross-model** — the CKA layer×layer matrices and the attention-vs-IG
  agreement bars;
* **gate sweep** — JumpGateLOB adaLN-Zero gate magnitude and jump-noise
  robustness over the diffusion timestep.

Every figure is written as both ``.pdf`` (vector, for LaTeX ``\includegraphics``)
and ``.png`` (for quick preview / slides). Missing artefacts are skipped with a
warning rather than aborting: a partial ``run_all`` still yields every figure it
has data for. No seaborn, no styles that need network fonts — Kaggle-safe.

The plotter is intentionally read-only over the artefacts: it never recomputes a
number, so a figure can never disagree with the JSON the report quotes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on Kaggle / CI
import matplotlib.pyplot as plt
import numpy as np

# Short, stable display names + a fixed colour per model so a model is the same
# colour in every figure of the paper.
MODEL_LABEL = {
    "CTABL": "CTABL",
    "DLA": "DLA",
    "JumpGateLOB": "JumpGateLOB",
    "ctabl_BTCIRT_ofi_k10": "CTABL",
    "dla_BTCIRT_ofi_k10": "DLA",
    "jumpgatelob_levy_BTCIRT_ofi_k10": "JumpGateLOB",
}
MODEL_COLOR = {
    "CTABL": "#4C72B0",
    "DLA": "#DD8452",
    "JumpGateLOB": "#55A868",
}
MODEL_ORDER = ["CTABL", "DLA", "JumpGateLOB"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _label(key: str) -> str:
    return MODEL_LABEL.get(key, key)


def _color(display_name: str) -> str:
    return MODEL_COLOR.get(display_name, "#777777")


def _save(fig, outdir: Path, stem: str) -> None:
    """Write a figure as both PDF (LaTeX) and PNG (preview), then close it."""
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _find(results: Path, *names: str) -> Path | None:
    """First existing path among ``names`` under ``results`` (flat or nested)."""
    for n in names:
        p = results / n
        if p.exists():
            return p
    # also try a direct glob so ``<model>__ig_zero.json`` style names resolve
    for n in names:
        hits = sorted(results.glob(n))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------- #
# Layer 1 — feature attribution
# --------------------------------------------------------------------------- #
def plot_group_shares(results: Path, models: list[str], outdir: Path) -> None:
    """Grouped bars: each model's IG attribution share per feature group."""
    per_model: dict[str, dict[str, float]] = {}
    for m in models:
        j = _load_json(_find(results, f"{m}__ig_zero.json", f"{m}/ig_zero.json")) or {}
        if "group_shares" in j:
            per_model[_label(m)] = j["group_shares"]
    if not per_model:
        print("  [skip] group shares: no ig_zero.json found")
        return

    # union of group names, in first model's order then any extras
    groups: list[str] = []
    for shares in per_model.values():
        for g in shares:
            if g not in groups:
                groups.append(g)

    names = [n for n in MODEL_ORDER if n in per_model] + [
        n for n in per_model if n not in MODEL_ORDER
    ]
    x = np.arange(len(groups))
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(groups)), 4))
    for i, name in enumerate(names):
        vals = [per_model[name].get(g, 0.0) for g in groups]
        ax.bar(x + i * w, vals, w, label=name, color=_color(name))
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("IG attribution share")
    ax.set_title("Feature-group attribution (zero baseline)")
    ax.legend(frameon=False)
    _save(fig, outdir, "fig_ig_group_shares")


def plot_attr_heatmaps(results: Path, models: list[str], outdir: Path) -> None:
    """Per-model signed (time × feature) IG heatmap, shared symmetric scale."""
    data: dict[str, np.ndarray] = {}
    names_by_model: dict[str, list[str]] = {}
    for m in models:
        npz = _find(results, f"{m}__ig_zero.npz", f"{m}/ig_zero.npz")
        js = _load_json(_find(results, f"{m}__ig_zero.json", f"{m}/ig_zero.json"))
        if npz is None:
            continue
        arr = np.load(npz)
        if "attr_mean" not in arr:
            continue
        data[_label(m)] = arr["attr_mean"]  # (T, F)
        if js and "feature_names" in js:
            names_by_model[_label(m)] = js["feature_names"]
    if not data:
        print("  [skip] attribution heatmaps: no ig_zero.npz found")
        return

    vmax = max(float(np.abs(a).max()) for a in data.values()) or 1.0
    names = [n for n in MODEL_ORDER if n in data] + [
        n for n in data if n not in MODEL_ORDER
    ]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4), squeeze=False)
    im = None
    for ax, name in zip(axes[0], names):
        a = data[name]  # (T, F)
        im = ax.imshow(
            a.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower"
        )
        ax.set_title(name)
        ax.set_xlabel("time step (past window)")
        ax.grid(False)
    axes[0][0].set_ylabel("feature index")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="signed IG (mean over windows)")
    fig.suptitle("Signed attribution over the input window")
    _save(fig, outdir, "fig_ig_heatmaps")


# --------------------------------------------------------------------------- #
# Layer 1 sanity — faithfulness
# --------------------------------------------------------------------------- #
def plot_faithfulness(results: Path, outdir: Path) -> None:
    j = _load_json(_find(results, "faithfulness.json"))
    if not j or "rows" not in j:
        print("  [skip] faithfulness: no faithfulness.json")
        return
    rows = j["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for mode, ax in zip(("deletion", "insertion"), axes):
        for r in rows:
            name = _label(r["model"])
            d = r[mode]
            fr = d["fractions"]
            ax.plot(fr, d["accuracy"], "-o", ms=3, color=_color(name), label=name)
            if "random_accuracy" in d:
                ax.plot(fr, d["random_accuracy"], "--", color=_color(name), alpha=0.5)
        ax.set_title(f"{mode.capitalize()} (solid = IG order, dashed = random)")
        ax.set_xlabel("fraction of features " + ("removed" if mode == "deletion" else "restored"))
        ax.set_ylabel("accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Attribution faithfulness: deletion / insertion vs. random")
    _save(fig, outdir, "fig_faithfulness")


# --------------------------------------------------------------------------- #
# Layer 2 — per-layer probes
# --------------------------------------------------------------------------- #
def plot_probes(results: Path, outdir: Path) -> None:
    j = _load_json(_find(results, "probes.json"))
    if not j or "rows" not in j:
        print("  [skip] probes: no probes.json")
        return
    rows = j["rows"]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(_label(r["model"]), []).append(r)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, rs in by_model.items():
        rs = sorted(rs, key=lambda r: r["depth"])
        depth = [r["depth_frac"] for r in rs]
        acc = [r["accuracy"] for r in rs]
        ctrl = [r["control_accuracy"] for r in rs]
        c = _color(name)
        ax.plot(depth, acc, "-o", color=c, label=name)
        ax.plot(depth, ctrl, "--", color=c, alpha=0.5)
        for r in rs:
            ax.annotate(
                r["tap"],
                (r["depth_frac"], r["accuracy"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=c,
            )
    ax.set_xlabel("normalised trunk depth (shallow → deep)")
    ax.set_ylabel("linear-probe accuracy")
    ax.set_title("Where trend becomes linearly decodable\n(solid = probe, dashed = shuffled-label control)")
    ax.legend(frameon=False)
    _save(fig, outdir, "fig_probes")


# --------------------------------------------------------------------------- #
# Layer 3 — cross-model CKA + agreement
# --------------------------------------------------------------------------- #
def plot_cka(results: Path, outdir: Path) -> None:
    j = _load_json(_find(results, "cka.json"))
    if not j or "pairs" not in j:
        print("  [skip] cka: no cka.json")
        return
    cross = [p for p in j["pairs"] if p["model_a"] != p["model_b"]]
    if not cross:
        print("  [skip] cka: no cross-model pairs")
        return
    fig, axes = plt.subplots(1, len(cross), figsize=(4.0 * len(cross), 3.8), squeeze=False)
    im = None
    for ax, p in zip(axes[0], cross):
        m = np.array(p["cka"])
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(p["taps_b"])))
        ax.set_xticklabels(p["taps_b"], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(p["taps_a"])))
        ax.set_yticklabels(p["taps_a"], fontsize=8)
        ax.set_xlabel(_label(p["model_b"]))
        ax.set_ylabel(_label(p["model_a"]))
        ax.grid(False)
        for i in range(m.shape[0]):
            for k in range(m.shape[1]):
                ax.text(
                    k, i, f"{m[i, k]:.2f}", ha="center", va="center",
                    color="white" if m[i, k] < 0.6 else "black", fontsize=7,
                )
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="linear CKA")
    fig.suptitle("Cross-model representational similarity (CKA)")
    fig.subplots_adjust(wspace=0.35)  # keep each panel's y-label off its neighbour
    _save(fig, outdir, "fig_cka")


def plot_agreement(results: Path, outdir: Path) -> None:
    j = _load_json(_find(results, "agreement.json"))
    if not j or "rows" not in j:
        print("  [skip] agreement: no agreement.json")
        return
    rows = j["rows"]
    names = [_label(r["model"]) for r in rows]
    time_rho = [r["time"]["spearman"] for r in rows]
    time_tau = [r["time"]["kendall"] for r in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4))
    ax.bar(x - w / 2, time_rho, w, label="Spearman ρ", color="#4C72B0")
    ax.bar(x + w / 2, time_tau, w, label="Kendall τ", color="#C44E52")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("rank agreement (time axis)")
    ax.set_title("Attention vs. IG agreement over the time axis")
    ax.legend(frameon=False)
    _save(fig, outdir, "fig_agreement")


# --------------------------------------------------------------------------- #
# JumpGateLOB gate / robustness sweep
# --------------------------------------------------------------------------- #
def plot_gate_sweep(results: Path, models: list[str], outdir: Path) -> None:
    path = None
    for m in models:
        path = _find(results, f"{m}__gate_sweep.json", f"{m}/gate_sweep.json")
        if path:
            break
    if path is None:
        path = _find(results, "gate_sweep.json")
    j = _load_json(path) if path else None
    if not j or "gates" not in j:
        print("  [skip] gate sweep: no gate_sweep.json")
        return
    g, rob = j["gates"], j.get("robustness", {})
    boundary = j.get("low_t_boundary")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(g["t"], g["gate_attn"], "-o", ms=3, label="attn gate", color="#4C72B0")
    axes[0].plot(g["t"], g["gate_mlp"], "-o", ms=3, label="mlp gate", color="#DD8452")
    axes[0].set_xlabel("diffusion timestep t")
    axes[0].set_ylabel("mean |adaLN-Zero gate|")
    axes[0].set_title("Gate magnitude vs. t")
    axes[0].legend(frameon=False)

    if rob:
        axes[1].plot(rob["t"], rob["accuracy_t0"], "-o", ms=3, label="deployed (t=0 cond.)", color="#55A868")
        axes[1].plot(rob["t"], rob["accuracy_oracle"], "--s", ms=3, label="oracle (told t)", color="#8172B3")
        axes[1].set_xlabel("jump-noise level t")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Robustness to jump noise")
        axes[1].legend(frameon=False)
    if boundary is not None:
        for ax in axes:
            ax.axvline(boundary, color="grey", ls=":", lw=1)
    fig.suptitle("JumpGateLOB adaLN-Zero gate / robustness sweep")
    _save(fig, outdir, "fig_gate_sweep")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="dir with the run_all artefacts")
    ap.add_argument("--outdir", type=Path, default=None, help="figure output dir")
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "ctabl_BTCIRT_ofi_k10",
            "dla_BTCIRT_ofi_k10",
            "jumpgatelob_levy_BTCIRT_ofi_k10",
        ],
    )
    args = ap.parse_args()
    outdir = args.outdir or (args.results / "figures")
    print(f"reading artefacts from {args.results}")
    print(f"writing figures to    {outdir}")

    plot_group_shares(args.results, args.models, outdir)
    plot_attr_heatmaps(args.results, args.models, outdir)
    plot_faithfulness(args.results, outdir)
    plot_probes(args.results, outdir)
    plot_cka(args.results, outdir)
    plot_agreement(args.results, outdir)
    plot_gate_sweep(args.results, args.models, outdir)
    print("done.")


if __name__ == "__main__":
    main()
