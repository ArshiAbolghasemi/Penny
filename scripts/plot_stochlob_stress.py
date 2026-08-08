"""Does diffusion beat the CE-only baseline *under stress*? — BTCIRT, plotted.

``scripts/stochlob_significance.py`` answers the pooled question and finds mostly
no difference. This script asks the conditional one: the diffusion objectives are
*motivated* by heavy tails and turbulent books, so if they earn their keep anywhere
it should be in the turbulent slice of the test set — and a pooled average would
hide exactly that.

Each test window is scored three ways, using the same formulas as
``scripts/plot_stress_vs_variance_robustness.py`` so numbers stay comparable with
the existing diagnostic. All three come from the level-averaged feature stream
``x`` with one-step increments ``Δx``:

    stress        = max|Δx| / std(Δx)     one violent move relative to the window's
                                          own volatility — a jump, not a regime
    variance      = var(Δx)               sustained high volatility — a regime,
                                          not a jump
    jump_residual = log max|Δx| − fit(log var(Δx))
                                          how much bigger the largest move is than
                                          this window's variance predicts, i.e. the
                                          jump signal with the variance level
                                          divided out

Windows are ranked into deciles per condition, and macro-F1 is computed per decile
for the baseline and for each diffusion model. Decile 10 is the stressed slice.

What is plotted
---------------
Per horizon, a 2x3 figure:

  * row 1 — macro-F1 by decile, all four models. The baseline is drawn in gray as
    the reference; the three diffusion models carry the categorical hues. Context.
  * row 2 — Δmacro-F1 (model − baseline) by decile with 95% bootstrap bands and a
    zero reference line. This is the panel that answers the question: above zero
    means diffusion wins in that slice.

Plus a summary figure: the extreme-decile Δ with confidence interval for every
(horizon, model), so the "does it help where it should" answer is one glance.

Inference under scattered subsets
---------------------------------
A decile is not contiguous in time — it is a scattered subset selected by score.
Resampling those windows directly would destroy the overlap dependence the block
scheme exists to preserve (``stride=1``, ``T_past=60``: neighbouring windows share
59 of 60 rows). So blocks are always formed over the **full** window index and the
decile mask is applied *inside* each block: a resample draws contiguous blocks and
counts only their in-decile windows. The statistics themselves are imported from
``stochlob_significance`` rather than reimplemented, so the block length rule, the
permutation test and the bootstrap cannot drift between the two scripts.

Usage::

    uv run python scripts/plot_stochlob_stress.py
    uv run python scripts/plot_stochlob_stress.py --horizons 10 100 --n-boot 4000

Writes ``outputs/figures/stochlob_stress/`` (PDFs) and
``outputs/stochlob_stress_deciles.csv`` — the CSV is also the table view that the
aqua series' sub-3:1 contrast obliges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
for _p in (REPO, REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from crypto.dataset import build_datasets  # noqa: E402
from stochlob_significance import (  # noqa: E402
    DATA_CONFIG,
    FEATURE_MODE,
    HORIZONS,
    SYMBOL,
    TARGET_PREFIXES,
    _macro_f1_from_conf,
    discover_runs,
    load_model,
    paired_block_bootstrap,
    paired_block_permutation,
    predict_all,
    _block_length,
)
from torch.utils.data import DataLoader  # noqa: E402

BATCH = 256
STAT_SEED = 20260806
N_DECILES = 10

CONDITIONS = ("stress", "variance", "jump_residual")
CONDITION_LABEL = {
    "stress": "stress   max|Δx| / std(Δx)",
    "variance": "variance   var(Δx)",
    "jump_residual": "jump residual   variance-controlled",
}

# Emphasis encoding: the baseline is the reference the others are measured against,
# so it takes the de-emphasis gray and the three diffusion models take categorical
# slots 1-3. Those three validate on the all-pairs pairlist in light mode (worst
# CVD ΔE 9.2, worst normal-vision ΔE 24.0). Figures are light-surface PDFs for
# print, so only the light steps are used.
BASELINE_COLOR = "#52514e"
MODEL_COLOR = {
    "gaussgatelob": "#2a78d6",  # blue
    "jumpgatelob": "#eb6834",  # orange
    "alphastablelob": "#1baf7a",  # aqua
}
GRID_COLOR = "#d9d8d4"
ZERO_COLOR = "#8a8985"

plt.rcParams.update(
    {
        # Type-42 keeps PDF text selectable and embeddable; matplotlib's default
        # Type-3 is rejected by many venues.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.edgecolor": "#b8b7b2",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ── per-window condition scores (formulas mirror plot_stress_vs_variance_*) ────
def window_scores(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(stress, variance, max_abs)`` for a batch of windows ``(B, 1, T, F)``."""
    agg = x.squeeze(1).mean(-1)  # (B, T) level-averaged stream
    dif = agg[:, 1:] - agg[:, :-1]
    rv = dif.std(dim=1).clamp_min(1e-8)
    max_abs = dif.abs().max(dim=1).values
    return (
        (max_abs / rv).cpu().numpy(),
        dif.var(dim=1).cpu().numpy(),
        max_abs.cpu().numpy(),
    )


def jump_residual(variance: np.ndarray, max_abs: np.ndarray) -> np.ndarray:
    """log max|Δx| minus its linear fit on log var(Δx) — jumpiness net of level.

    Positive residual = the window's largest move is bigger than its own realized
    variance predicts. This separates "a genuine jump" from "just a loud regime",
    which the raw ``stress`` score conflates.
    """
    eps = 1e-12
    log_var, log_max = np.log(variance + eps), np.log(max_abs + eps)
    ok = np.isfinite(log_var) & np.isfinite(log_max)
    if ok.sum() < 3 or np.nanstd(log_var[ok]) < eps:
        return log_max - float(np.nanmean(log_max[ok]))
    slope, intercept = np.polyfit(log_var[ok], log_max[ok], deg=1)
    return log_max - (slope * log_var + intercept)


def decile_of(score: np.ndarray) -> np.ndarray:
    """Rank windows into ``N_DECILES`` equal-count bins (1 = calmest)."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.int64)
    ranks[order] = np.arange(len(score))
    return (ranks * N_DECILES // len(score)) + 1


def _block_conf_masked(
    y_true: np.ndarray, y_pred: np.ndarray, block: int, mask: np.ndarray
) -> np.ndarray:
    """``(n_blocks, 9)`` confusion counts, blocks over the FULL index, masked inside.

    Blocking must follow time, but a decile is a scattered subset. Forming blocks
    over every window and counting only the masked ones inside each keeps the
    resampling unit contiguous while still measuring the subset.
    """
    n = len(y_true)
    cell = np.arange(n) // block
    n_cells = int(cell[-1]) + 1
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros((n_cells, 9), dtype=np.int64)
    return np.bincount(
        cell[idx] * 9 + (3 * y_true[idx] + y_pred[idx]), minlength=n_cells * 9
    ).reshape(n_cells, 9)


def collect(horizons: tuple[int, ...], n_boot: int, seed: int, device) -> pd.DataFrame:
    runs = discover_runs(horizons)
    if not runs["baseline"]:
        raise SystemExit("no stochlob_baseline runs found")

    rows = []
    for k in horizons:
        if k not in runs["baseline"]:
            logger.warning("k={}: no baseline run, skipped", k)
            continue
        targets = [t for t in TARGET_PREFIXES if k in runs[t]]
        if not targets:
            logger.warning("k={}: no target runs, skipped", k)
            continue

        cfg = json.loads((REPO / DATA_CONFIG.format(fm=FEATURE_MODE, k=k)).read_text())
        _, _, test_ds, _alpha, _meta = build_datasets(cfg)
        loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
        t_past = int(cfg["T_past"])

        ys, st, va, mx = [], [], [], []
        for b in loader:
            s, v, m = window_scores(b["x"].float())
            st.append(s)
            va.append(v)
            mx.append(m)
            ys.append(b["label"].numpy())
        y_true = np.concatenate(ys)
        scores = {
            "stress": np.concatenate(st),
            "variance": np.concatenate(va),
        }
        scores["jump_residual"] = jump_residual(scores["variance"], np.concatenate(mx))
        deciles = {c: decile_of(scores[c]) for c in CONDITIONS}
        logger.info("k={:<4} test windows={:,}  T_past={}", k, len(y_true), t_past)

        preds = {}
        for name in ("baseline", *targets):
            run, cls = runs[name][k]
            model = load_model(run, cls, device)
            preds[name] = predict_all(model, loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(
                "  {:<16} {:<14} pooled macro_f1={:.4f}",
                name,
                cls.__name__,
                _macro_f1_from_conf(np.bincount(3 * y_true + preds[name], minlength=9)),
            )

        base_ok = (preds["baseline"] == y_true).astype(float)
        for cond in CONDITIONS:
            dec = deciles[cond]
            for name in targets:
                # One block length per (k, condition, model), from the paired
                # correctness difference over the full series.
                block = _block_length(
                    (preds[name] == y_true).astype(float) - base_ok, t_past
                )
                for d in range(1, N_DECILES + 1):
                    mask = dec == d
                    ca = _block_conf_masked(y_true, preds[name], block, mask)
                    cb = _block_conf_masked(y_true, preds["baseline"], block, mask)
                    sd = seed + 1009 * targets.index(name) + 17 * k + 31 * d
                    boot = paired_block_bootstrap(ca, cb, n_boot, sd)
                    rows.append(
                        {
                            "k": k,
                            "condition": cond,
                            "target": name,
                            "decile": d,
                            "n_windows": int(mask.sum()),
                            "block_windows": block,
                            "f1_target": _macro_f1_from_conf(ca.sum(0)),
                            "f1_baseline": _macro_f1_from_conf(cb.sum(0)),
                            "d_macro_f1": _macro_f1_from_conf(ca.sum(0))
                            - _macro_f1_from_conf(cb.sum(0)),
                            "d_f1_lo": boot["d_macro_f1"]["lo"],
                            "d_f1_hi": boot["d_macro_f1"]["hi"],
                            # The permutation p-value is the calibrated one, but it
                            # is only needed for the headline extreme slice.
                            "d_f1_p": (
                                paired_block_permutation(ca, cb, n_boot, sd)
                                if d == N_DECILES
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────
def _style(ax):
    ax.grid(True, color=GRID_COLOR, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _place_labels(ax, entries, x_at, pad_frac=0.055):
    """Direct-label series at ``x_at``, nudged apart so they never overlap.

    The Δ lines converge near zero in the calm deciles, which stacks the labels on
    top of each other. Sort by value and enforce a minimum vertical gap (a fraction
    of the axis span) so every label stays readable — direct labels are the relief
    the aqua series' sub-3:1 contrast requires, so they cannot be allowed to collide.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * pad_frac
    placed = []
    for y, text, color in sorted(entries, key=lambda e: e[0]):
        if placed and y - placed[-1][0] < gap:
            y = placed[-1][0] + gap
        placed.append((y, text, color))
    for y, text, color in placed:
        ax.annotate(
            text,
            xy=(x_at, y),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=8,
            va="center",
            annotation_clip=False,
        )


def plot_horizon(df: pd.DataFrame, k: int, out_dir: Path) -> Path:
    sub = df[df["k"] == k]
    targets = [t for t in MODEL_COLOR if t in set(sub["target"])]
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(13, 7), sharex=True)
    x = np.arange(1, N_DECILES + 1)

    for j, cond in enumerate(CONDITIONS):
        cs = sub[sub["condition"] == cond]

        # row 1 — absolute macro-F1 by decile (context)
        ax = axes[0, j]
        base = cs[cs["target"] == targets[0]].sort_values("decile")
        ax.plot(
            x,
            base["f1_baseline"],
            color=BASELINE_COLOR,
            lw=2,
            marker="o",
            ms=4,
            label="stochlob_baseline",
            zorder=3,
        )
        for t in targets:
            r = cs[cs["target"] == t].sort_values("decile")
            ax.plot(
                x, r["f1_target"], color=MODEL_COLOR[t], lw=2, marker="o", ms=4, label=t
            )
        ax.set_title(CONDITION_LABEL[cond], fontsize=9, loc="left", color="#52514e")
        if j == 0:
            ax.set_ylabel("macro-F1")
        _style(ax)

        # row 2 — Δ vs baseline with 95% bootstrap bands (the answer panel)
        ax = axes[1, j]
        ax.axhline(0, color=ZERO_COLOR, lw=1.5, ls="--", zorder=2)
        labels = []
        for t in targets:
            r = cs[cs["target"] == t].sort_values("decile")
            ax.fill_between(
                x, r["d_f1_lo"], r["d_f1_hi"], color=MODEL_COLOR[t], alpha=0.15, lw=0
            )
            ax.plot(x, r["d_macro_f1"], color=MODEL_COLOR[t], lw=2, marker="o", ms=4)
            labels.append(
                (
                    float(r["d_macro_f1"].iloc[-1]),
                    t.replace("gatelob", "").replace("lob", ""),
                    MODEL_COLOR[t],
                )
            )
        _place_labels(ax, labels, N_DECILES)
        ax.set_xlabel(f"{cond} decile   (10 = most stressed)")
        if j == 0:
            ax.set_ylabel("Δ macro-F1  (model − baseline)")
        ax.set_xticks(x)
        _style(ax)

    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"{SYMBOL}  k={k} — diffusion vs CE-only baseline, by stress decile",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "above zero (bottom row) = the diffusion objective wins that slice; "
        "bands are 95% block-bootstrap intervals",
        fontsize=8,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = out_dir / f"stochlob_stress_k{k}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_extreme_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    """Extreme-decile Δ with CI for every (horizon, model) — the one-glance answer."""
    ext = df[df["decile"] == N_DECILES]
    ks = sorted(ext["k"].unique())
    targets = [t for t in MODEL_COLOR if t in set(ext["target"])]
    fig, axes = plt.subplots(
        1, len(CONDITIONS), figsize=(13, 3.6), sharey=True, squeeze=False
    )

    for j, cond in enumerate(CONDITIONS):
        ax = axes[0, j]
        ax.axhline(0, color=ZERO_COLOR, lw=1.5, ls="--", zorder=2)
        cs = ext[ext["condition"] == cond]
        width = 0.8 / max(len(targets), 1)
        for i, t in enumerate(targets):
            r = cs[cs["target"] == t].set_index("k").reindex(ks)
            pos = np.arange(len(ks)) + (i - (len(targets) - 1) / 2) * width
            mid = r["d_macro_f1"].to_numpy(dtype=float)
            lo = r["d_f1_lo"].to_numpy(dtype=float)
            hi = r["d_f1_hi"].to_numpy(dtype=float)
            ax.errorbar(
                pos,
                mid,
                yerr=[mid - lo, hi - mid],
                fmt="o",
                ms=7,
                lw=0,
                elinewidth=2,
                capsize=3,
                color=MODEL_COLOR[t],
                label=t if j == 0 else None,
            )
            # Mark the slices that survive the calibrated permutation test. Anchor
            # above the whisker top, not the point, or the glyph lands on the bar.
            for xp, top, pv in zip(pos, hi, r["d_f1_p"].to_numpy(dtype=float)):
                if np.isfinite(pv) and pv < 0.05:
                    ax.annotate(
                        "*",
                        xy=(xp, top),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        color=MODEL_COLOR[t],
                        fontsize=13,
                        fontweight="bold",
                    )
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_title(CONDITION_LABEL[cond], fontsize=9, loc="left", color="#52514e")
        if j == 0:
            ax.set_ylabel("Δ macro-F1  (model − baseline)")
        _style(ax)

    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"{SYMBOL} — most-stressed decile only: does diffusion beat the baseline?",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "* = p < 0.05, block sign-flip permutation; bars are 95% block-bootstrap "
        "intervals",
        fontsize=8,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = out_dir / "stochlob_stress_extreme_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=STAT_SEED)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("symbol={} device={} deciles={}", SYMBOL, device, N_DECILES)

    df = collect(tuple(args.horizons), args.n_boot, args.seed, device)
    if df.empty:
        raise SystemExit("nothing collected — check checkpoints and horizons")

    out_dir = REPO / "outputs" / "figures" / "stochlob_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = REPO / "outputs" / "stochlob_stress_deciles.csv"
    df.to_csv(csv_path, index=False)
    logger.info("saved → {}", csv_path)

    for k in sorted(df["k"].unique()):
        logger.info("saved → {}", plot_horizon(df, k, out_dir))
    logger.info("saved → {}", plot_extreme_summary(df, out_dir))

    # Headline: the extreme slice, which is the whole point of the exercise.
    ext = df[df["decile"] == N_DECILES]
    logger.info("")
    logger.info("=== most-stressed decile (10/10) — Δ macro-F1 vs baseline ===")
    for line in (
        ext[
            [
                "k",
                "condition",
                "target",
                "n_windows",
                "d_macro_f1",
                "d_f1_lo",
                "d_f1_hi",
                "d_f1_p",
            ]
        ]
        .to_string(index=False)
        .splitlines()
    ):
        logger.info("  {}", line)
    wins = int((ext["d_f1_p"] < 0.05).sum())
    logger.info(
        "{} of {} extreme-decile contrasts are significant at p<0.05 (uncorrected)",
        wins,
        len(ext),
    )


if __name__ == "__main__":
    main()
