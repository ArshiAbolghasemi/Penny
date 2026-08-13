"""Paired significance test: the stochastic-gate family vs. the StochLOB baseline (BTCIRT).

Answers one question with a p-value attached: **does the diffusion objective buy
anything over the same trunk trained with plain cross-entropy?**

    stochlob_baseline   trunk + trend head, L = L_cls                  ← baseline
    GaussGateLOB        + Brownian score matching + noise-consistency
    JumpGateLOB         + Lévy jump-diffusion score matching       }  same trunk,
    AlphaStableLOB      + α-stable score matching                  }  same shapes

All four share one architecture, so each contrast isolates the corruption law and
the score objective rather than model capacity. Every model is scored on the
**same** test windows and the same labels, which the script asserts rather than
assumes, at every horizon ``k ∈ {10, 20, 50, 100}``.

Scope: Coinbase **BTCIRT** only, ``checkpoints/coinbase/BTCIRT``. The baseline is
the ``stochlob_baseline*`` run set — ``train_gaussgatelob --baseline``, which
optimises ``L_cls`` alone (the score head is constructed but never called, so its
grads stay ``None`` and AdamW skips it: the training math is exactly the ablation).
The model class is read from each run's config keys (``ggl_*`` / ``jgl_*`` /
``astable_*``), not from its directory name, so the baseline runs resolve correctly
despite their rename artifact.

Why block resampling
--------------------
BTCIRT is a single time series with ``stride=1``, so consecutive test windows share
``T_past - 1`` of their ``T_past`` rows. Per-window correctness is therefore heavily
autocorrelated and every i.i.d.-window test (McNemar, a paired t-test, a naive
bootstrap) overstates significance badly — measured at 87% false positives against
a nominal 5% in simulation. Both resampling schemes here operate on **contiguous
blocks of windows** instead, so the overlap-induced dependence is carried inside a
block rather than broken across draws.

Block length is ``BLOCK_TAU_MULT * tau_hat``, floored at ``T_past`` (windows overlap
by that much by construction) — see ``BLOCK_TAU_MULT`` for the simulated
false-positive rates that fix the multiplier. Note this is deliberately longer than
``scripts/model_comparison.py``'s ``block = max(k, ceil(tau))``.

Permutation, not bootstrap, for the p-value
-------------------------------------------
The headline p-value comes from a block **sign-flip permutation** test: under the
null that two models are exchangeable, swapping which one owns a given block's
confusion counts leaves the joint law unchanged, so the permutation distribution is
the exact null distribution up to Monte-Carlo error. The block bootstrap supplies
the confidence interval, but its sign p-value runs anti-conservative — at this
sample size its variance estimate is only ~0.86x the true sampling SD. Both are
reported; ``d_f1_p`` (permutation) is the one to quote.

Because a confusion matrix is additive over disjoint blocks, each block's 3x3
counts are precomputed once per comparison and a resample is just a sum of block
count vectors — so ``--n-boot`` is cheap regardless of test-set size.

Multiplicity
------------
Three models are tested against one shared baseline at each horizon, so
per-comparison p-values are corrected family-wise with Holm–Bonferroni within each
horizon: uniformly more powerful than plain Bonferroni and valid under arbitrary
dependence between the tests, which matters because the three targets share a trunk
and are highly correlated.

Usage::

    uv run python scripts/stochlob_significance.py
    uv run python scripts/stochlob_significance.py --horizons 10 20 --n-boot 5000

Writes ``outputs/stochlob_significance.{csv,json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

REPO = Path(__file__).resolve().parent.parent
# Run configs store repo-relative paths ("data/resampled/coinbase"), which the
# crypto cache loader resolves against the cwd.
os.chdir(REPO)
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from torch.utils.data import DataLoader

from crypto.dataset import build_datasets
from models.alphastablelob import AlphaStableLOB
from models.gaussgatelob import GaussGateLOB
from models.jumpgatelob import JumpGateLOB

EXCHANGE = "coinbase"
SYMBOL = "BTCIRT"
FEATURE_MODE = "ofi"
HORIZONS = (10, 20, 50, 100)

CHECKPOINT_ROOT = REPO / "checkpoints" / EXCHANGE / SYMBOL
DATA_CONFIG = "configs/crypto/coinbase/jumpgatelob/btcirt_{fm}_k{k}.json"

# Directory-name prefixes. The baseline runs carry a rename artifact
# ("stochlob_baselinegatelob_baseline_..."), so match on the leading token only.
BASELINE_PREFIX = "stochlob_baseline"
TARGET_PREFIXES = {
    "gaussgatelob": "gaussgatelob_",
    "jumpgatelob": "jumpgatelob_",
    "alphastablelob": "alphastablelob_",
}

# Architecture is identified by the config-key namespace each trainer writes, not
# by the run directory name — the baseline runs are GaussGateLOB instances whose
# directory says "stochlob_baseline".
CONFIG_PREFIX_TO_CLASS = {
    "ggl_": GaussGateLOB,
    "jgl_": JumpGateLOB,
    "astable_": AlphaStableLOB,
}

STAT_SEED = 20260806
BATCH = 256


# ── metrics from a flat 3x3 confusion vector ──────────────────────────────────
def _macro_f1_from_conf(c9: np.ndarray) -> float:
    """Macro-F1 from a flat (9,) confusion vector indexed ``3*y_true + y_pred``.

    Matches ``sklearn.f1_score(average="macro", labels=[0,1,2], zero_division=0)``:
    a class with no true and no predicted instances contributes 0, not NaN.
    """
    m = c9.reshape(3, 3).astype(np.float64)
    tp = np.diag(m)
    fn = m.sum(1) - tp
    fp = m.sum(0) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(3), where=denom > 0)
    return float(f1.mean())


def _accuracy_from_conf(c9: np.ndarray) -> float:
    tot = c9.sum()
    return float(np.diag(c9.reshape(3, 3)).sum() / tot) if tot else 0.0


def _integrated_ac_time(x: np.ndarray, max_lag: int = 500) -> float:
    """Initial-positive-sequence estimate of the integrated autocorrelation time.

    Same estimator ``scripts/model_comparison.py`` uses, applied to the paired
    correctness difference.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 1.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-24:
        return 1.0
    tau = 1.0
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        rho = float(np.dot(x[lag:], x[:-lag]) / denom)
        if rho <= 0:
            break
        tau += 2.0 * rho
    return float(max(tau, 1.0))


# Block length = BLOCK_TAU_MULT * tau_hat, floored at T_past.
#
# Setting block = tau_hat (what scripts/model_comparison.py does) is NOT enough:
# tau_hat estimates the integrated autocorrelation time correctly, but a block
# resampler needs blocks several times longer than tau to recover the variance of
# a pooled statistic. Simulated on a true null with a persistent autocorrelated
# contrast, the false-positive rate at nominal 5% was:
#
#     block =  1*tau  ->  11.6%      block =  4*tau  ->   5.6%
#     block =  2*tau  ->   8.0%      block =  5*tau  ->   4.8%
#     block =  3*tau  ->   6.8%      block =  7*tau  ->   3.2%
#
# 4x is the smallest multiplier that calibrates, so it costs the least power
# (calibration is bought with power here: at 1*tau the test "detects" far more,
# but roughly two thirds of those detections are false).
BLOCK_TAU_MULT = 4


def _block_length(paired_diff: np.ndarray, t_past: int) -> int:
    """Block length in windows for a paired per-window correctness difference.

    Floored at ``T_past`` because consecutive windows share ``T_past - 1`` rows by
    construction — a block shorter than that cannot contain the overlap it exists
    to preserve.
    """
    tau = _integrated_ac_time(paired_diff)
    n = len(paired_diff)
    return int(min(n, max(1, np.ceil(BLOCK_TAU_MULT * tau), t_past)))


def _block_conf(y_true: np.ndarray, y_pred: np.ndarray, block: int) -> np.ndarray:
    """``(n_blocks, 9)`` confusion counts over contiguous non-overlapping blocks.

    Confusion counts are additive over disjoint window groups, so every resample
    below reduces to summing rows of this array instead of rescanning windows.
    """
    n = len(y_true)
    cell = np.arange(n) // block
    n_cells = int(cell[-1]) + 1
    return np.bincount(cell * 9 + (3 * y_true + y_pred), minlength=n_cells * 9).reshape(
        n_cells, 9
    )


def paired_block_permutation(
    conf_a: np.ndarray, conf_b: np.ndarray, n_perm: int, seed: int
) -> float:
    """Two-sided p-value by swapping the two models' block-level confusions.

    This is the **primary** p-value: calibrated by construction rather than by an
    asymptotic variance argument. Under the null that A and B are exchangeable,
    swapping which model owns a block's confusion counts leaves the joint law
    unchanged, and swapping whole blocks preserves the overlap-induced dependence
    inside each block.
    """
    rng = np.random.default_rng(seed)
    n_cells = conf_a.shape[0]
    obs = _macro_f1_from_conf(conf_a.sum(0)) - _macro_f1_from_conf(conf_b.sum(0))
    count = 0
    for _ in range(n_perm):
        swap = (rng.random(n_cells) < 0.5)[:, None]
        d = _macro_f1_from_conf(np.where(swap, conf_b, conf_a).sum(0)) - (
            _macro_f1_from_conf(np.where(swap, conf_a, conf_b).sum(0))
        )
        count += abs(d) >= abs(obs) - 1e-15
    # (1 + count) / (1 + n_perm) keeps the p-value valid rather than ever hitting 0
    return (1.0 + count) / (n_perm + 1.0)


def paired_block_bootstrap(
    conf_a: np.ndarray, conf_b: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Bootstrap the paired difference ``metric(a) - metric(b)`` over blocks.

    Both models are evaluated on the *same* resampled blocks each iteration; that
    pairing cancels the shared window-to-window difficulty variation and leaves the
    model contrast, which is why the interval is far tighter than two independent
    per-model intervals would suggest.
    """
    rng = np.random.default_rng(seed)
    n_cells = conf_a.shape[0]
    d_f1 = np.empty(n_boot)
    d_acc = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, n_cells, n_cells)
        ca, cb = conf_a[sel].sum(0), conf_b[sel].sum(0)
        d_f1[i] = _macro_f1_from_conf(ca) - _macro_f1_from_conf(cb)
        d_acc[i] = _accuracy_from_conf(ca) - _accuracy_from_conf(cb)

    out = {}
    for key, arr in (("d_macro_f1", d_f1), ("d_acc", d_acc)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
        out[key] = {
            "boot_mean": float(arr.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "p": float(min(1.0, p)),
        }
    return out


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, returned in input order.

    Valid under arbitrary dependence between the tests, which is required here:
    the three targets share a trunk, so their p-values are far from independent.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])  # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj.tolist()


# ── checkpoint discovery ──────────────────────────────────────────────────────
def _run_config(run: Path) -> dict | None:
    cj = run / "config.json"
    if cj.exists():
        try:
            return json.loads(cj.read_text())
        except json.JSONDecodeError:
            pass
    bp = run / "best.pt"
    if bp.exists():
        return torch.load(bp, map_location="cpu", weights_only=False)["config"]
    return None


def _class_for(cfg: dict, run_name: str):
    """Resolve the model class from the config-key namespace the trainer wrote."""
    for prefix, cls in CONFIG_PREFIX_TO_CLASS.items():
        if any(key.startswith(prefix) for key in cfg):
            return cls
    raise SystemExit(
        f"cannot identify architecture for run {run_name!r}: no "
        f"{sorted(CONFIG_PREFIX_TO_CLASS)} keys in its config"
    )


def discover_runs(horizons: tuple[int, ...]) -> dict:
    """``{name -> {k -> (run_dir, cls)}}`` for the baseline and the three targets."""
    if not CHECKPOINT_ROOT.exists():
        raise SystemExit(f"checkpoint root does not exist: {CHECKPOINT_ROOT}")
    found: dict[str, dict] = {name: {} for name in ("baseline", *TARGET_PREFIXES)}
    for run in sorted(CHECKPOINT_ROOT.iterdir()):  # lexical == chronological
        if not run.is_dir() or not (run / "best.pt").exists():
            continue
        if run.name.startswith(BASELINE_PREFIX):
            name = "baseline"
        else:
            name = next(
                (t for t, p in TARGET_PREFIXES.items() if run.name.startswith(p)), None
            )
        if name is None:
            continue
        cfg = _run_config(run)
        if cfg is None:
            continue
        k = cfg.get("label_k")
        if (
            cfg.get("feature_mode") == FEATURE_MODE
            and cfg.get("symbol") == SYMBOL
            and k in horizons
        ):
            found[name][k] = (run, _class_for(cfg, run.name))  # newest wins
    return found


def load_model(run: Path, cls, device: torch.device):
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = cls(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, loader, device) -> np.ndarray:
    return np.concatenate(
        [model.predict(b, device).argmax(1).cpu().numpy() for b in loader]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=STAT_SEED)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    horizons = tuple(args.horizons)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("symbol={} feature_mode={} device={}", SYMBOL, FEATURE_MODE, device)
    logger.info("checkpoint root: {}", CHECKPOINT_ROOT)

    runs = discover_runs(horizons)
    for name in ("baseline", *TARGET_PREFIXES):
        avail = ", ".join(f"k{k}:{'OK' if k in runs[name] else '--'}" for k in horizons)
        logger.info("  {:<16} {}", name, avail)
    if not runs["baseline"]:
        raise SystemExit(
            f"no baseline runs found: expected directories starting with "
            f"{BASELINE_PREFIX!r} under {CHECKPOINT_ROOT}"
        )

    rows = []
    for k in horizons:
        if k not in runs["baseline"]:
            logger.warning("k={}: no baseline run, horizon skipped", k)
            continue
        targets = [t for t in TARGET_PREFIXES if k in runs[t]]
        if not targets:
            logger.warning("k={}: no target runs, horizon skipped", k)
            continue

        cfg = json.loads((REPO / DATA_CONFIG.format(fm=FEATURE_MODE, k=k)).read_text())
        _, _, test_ds, alpha, _meta = build_datasets(cfg)
        loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
        t_past = int(cfg["T_past"])

        # y_true is read from the loader once and every model is asserted against
        # it, so a window/label misalignment cannot silently pass through.
        y_true = np.concatenate([b["label"].numpy() for b in loader])
        logger.info(
            "k={:<4} test windows={:,}  T_past={}  alpha={:.6g}  class balance={}",
            k,
            len(y_true),
            t_past,
            alpha,
            np.round(np.bincount(y_true, minlength=3) / len(y_true), 3).tolist(),
        )

        preds: dict[str, np.ndarray] = {}
        point: dict[str, dict] = {}
        for name in ("baseline", *targets):
            run, cls = runs[name][k]
            model = load_model(run, cls, device)
            yp = predict_all(model, loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if len(yp) != len(y_true):
                raise AssertionError(
                    f"{name} k={k}: {len(yp)} predictions for {len(y_true)} windows"
                )
            preds[name] = yp
            total = np.bincount(3 * y_true + yp, minlength=9)
            point[name] = {
                "run": run.name,
                "cls": cls.__name__,
                "macro_f1": _macro_f1_from_conf(total),
                "accuracy": _accuracy_from_conf(total),
            }
            logger.info(
                "  {:<16} {:<14} macro_f1={:.4f} accuracy={:.4f}",
                name,
                cls.__name__,
                point[name]["macro_f1"],
                point[name]["accuracy"],
            )

        base_ok = (preds["baseline"] == y_true).astype(float)
        for i, name in enumerate(targets):
            diff = (preds[name] == y_true).astype(float) - base_ok
            block = _block_length(diff, t_past)
            ca = _block_conf(y_true, preds[name], block)
            cb = _block_conf(y_true, preds["baseline"], block)
            seed = args.seed + 1009 * i + 17 * k
            p_perm = paired_block_permutation(ca, cb, args.n_boot, seed)
            boot = paired_block_bootstrap(ca, cb, args.n_boot, seed)
            rows.append(
                {
                    "k": k,
                    "target": name,
                    "block_windows": block,
                    "n_blocks": int(ca.shape[0]),
                    "macro_f1_target": point[name]["macro_f1"],
                    "macro_f1_baseline": point["baseline"]["macro_f1"],
                    "d_macro_f1": point[name]["macro_f1"]
                    - point["baseline"]["macro_f1"],
                    "d_f1_lo": boot["d_macro_f1"]["lo"],
                    "d_f1_hi": boot["d_macro_f1"]["hi"],
                    "d_f1_p": p_perm,
                    "d_f1_p_boot": boot["d_macro_f1"]["p"],
                    "d_acc_pp": 100.0
                    * (point[name]["accuracy"] - point["baseline"]["accuracy"]),
                    "d_acc_lo_pp": 100.0 * boot["d_acc"]["lo"],
                    "d_acc_hi_pp": 100.0 * boot["d_acc"]["hi"],
                    "d_acc_p_boot": boot["d_acc"]["p"],
                    "baseline_run": point["baseline"]["run"],
                    "target_run": point[name]["run"],
                }
            )

    if not rows:
        raise SystemExit("no comparisons ran — check checkpoints and horizons")

    df = pd.DataFrame(rows)
    # Holm within each horizon: the three targets are one family per k.
    df["d_f1_p_holm"] = np.nan
    for k, sub in df.groupby("k"):
        df.loc[sub.index, "d_f1_p_holm"] = holm_bonferroni(sub["d_f1_p"].tolist())
    df["verdict"] = np.where(
        df["d_f1_p_holm"] < 0.05,
        np.where(df["d_macro_f1"] > 0, "diffusion better", "baseline better"),
        "no difference",
    )

    logger.info("")
    logger.info("=== {} — stochastic-gate family vs. stochlob_baseline ===", SYMBOL)
    logger.info(
        "block sign-flip permutation + block bootstrap, n={} resamples", args.n_boot
    )
    show = df[
        [
            "k",
            "target",
            "block_windows",
            "n_blocks",
            "d_macro_f1",
            "d_f1_lo",
            "d_f1_hi",
            "d_f1_p",
            "d_f1_p_holm",
            "d_acc_pp",
            "verdict",
        ]
    ]
    for line in show.to_string(index=False).splitlines():
        logger.info("  {}", line)
    logger.info(
        "d_* are (target − baseline); d_f1_lo/hi is the 95% bootstrap percentile "
        "interval."
    )
    logger.info(
        "d_f1_p is the block permutation p-value (the calibrated one), Holm-corrected "
        "within each k."
    )
    logger.info(
        "The bootstrap sign p-value is kept in the CSV as d_f1_p_boot but runs "
        "anti-conservative — quote the permutation column."
    )

    out_dir = REPO / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "stochlob_significance.csv"
    json_path = out_dir / "stochlob_significance.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "feature_mode": FEATURE_MODE,
                "horizons": list(horizons),
                "n_boot": args.n_boot,
                "seed": args.seed,
                "block_tau_mult": BLOCK_TAU_MULT,
                "comparisons": df.to_dict("records"),
            },
            indent=2,
            default=str,
        )
    )
    logger.info("saved → {}", csv_path)
    logger.info("saved → {}", json_path)


if __name__ == "__main__":
    main()
