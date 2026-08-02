#!/usr/bin/env python3
"""Plot conditional model performance by stress decile and variance decile.

This is the script version of the notebook robustness diagnostic:

* stress   = max one-step absolute move / realized volatility, per window
* variance = realized variance of the same one-step moves, per window

It evaluates each discovered checkpoint on the same test windows, saves two
decile-curve PDFs, and writes a CSV with calm/extreme summary metrics for both
conditioning variables.

Example
-------
rtk uv run python scripts/plot_stress_vs_variance_robustness.py \
  --symbol BNBIRT \
  --feature-mode ofi \
  --horizons 10 20 50 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "src").is_dir():
            return p
    raise RuntimeError(f"could not find repo root above {start}")


REPO = find_repo_root(Path.cwd())
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

from crypto.dataset import build_datasets  # noqa: E402
from models.alphastablelob import AlphaStableLOB  # noqa: E402
from models.binctabl import BINCTABL  # noqa: E402
from models.ctabl import CTABL  # noqa: E402
from models.deeplob import DeepLOB  # noqa: E402
from models.dla import DLA  # noqa: E402
from models.gaussgatelob import GaussGateLOB  # noqa: E402
from models.jumpgatelob import JumpGateLOB  # noqa: E402
from models.linvar import LinVAR  # noqa: E402
from models.logreg import LogReg  # noqa: E402
from models.ofsatnet import OFSATNet  # noqa: E402
from models.tlob import TLOB  # noqa: E402


MODEL_REGISTRY = {
    "dla": (DLA, "dla_"),
    "tlob": (TLOB, "tlob_"),
    "ctabl": (CTABL, "ctabl_"),
    "binctabl": (BINCTABL, "binctabl_"),
    "deeplob": (DeepLOB, "deeplob_"),
    "alphastablelob_1.5": (AlphaStableLOB, "alphastablelob_joint_a1.5_"),
    "linvar": (LinVAR, "linvar_"),
    "logreg": (LogReg, "logreg_"),
    "jumpgatelob": (JumpGateLOB, "jumpgatelob_"),
    "gaussgatelob": (GaussGateLOB, "gaussgatelob_"),
    "ofsatnet": (OFSATNet, "ofsatnet_"),
}

LABELS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BNBIRT")
    p.add_argument("--feature-mode", default="ofi")
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--extreme-q", type=float, default=0.90)
    p.add_argument("--metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Default: checkpoints/coinbase/<SYMBOL>",
    )
    p.add_argument(
        "--data-config-template",
        default="configs/crypto/coinbase/jumpgatelob/{symbol_lower}_{feature_mode}_k{k}.json",
        help="Template used to build the shared test set.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: outputs/figures/crypto/coinbase/<symbol>/stress_variance",
    )
    return p.parse_args()


def run_config(run_dir: Path) -> dict | None:
    config_json = run_dir / "config.json"
    if config_json.exists():
        try:
            return json.loads(config_json.read_text())
        except Exception:
            pass

    best_pt = run_dir / "best.pt"
    if best_pt.exists():
        try:
            return torch.load(best_pt, map_location="cpu", weights_only=False)["config"]
        except Exception:
            pass

    return None


def discover_checkpoints(args: argparse.Namespace) -> dict[str, dict[int, Path]]:
    checkpoint_root = (
        args.checkpoint_root or REPO / "checkpoints" / "coinbase" / args.symbol
    )
    selected = {m: MODEL_REGISTRY[m] for m in args.models}
    found: dict[str, dict[int, Path]] = {m: {} for m in selected}

    if not checkpoint_root.exists():
        print(f"! checkpoint root does not exist: {checkpoint_root}")
        return found

    for run in sorted(checkpoint_root.iterdir()):
        if not run.is_dir():
            continue
        for tag, (_cls, prefix) in selected.items():
            if not run.name.startswith(prefix):
                continue
            cfg = run_config(run)
            if cfg is None:
                continue
            k = cfg.get("label_k")
            if (
                cfg.get("symbol") == args.symbol
                and cfg.get("feature_mode") == args.feature_mode
                and k in args.horizons
            ):
                found[tag][int(k)] = run

    return found


def load_ckpt(path: Path) -> dict:
    p = path / "best.pt" if path.is_dir() else path
    return torch.load(p, map_location="cpu", weights_only=False)


def build_model(tag: str, ckpt: dict, device: torch.device):
    cls, _prefix = MODEL_REGISTRY[tag]
    model = cls(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def data_config_path(args: argparse.Namespace, k: int) -> Path:
    return REPO / args.data_config_template.format(
        symbol=args.symbol,
        symbol_lower=args.symbol.lower(),
        feature_mode=args.feature_mode,
        k=k,
    )


def window_scores(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return stress and realized variance for a batch of windows.

    Both are computed from the same level-averaged stream:

    stress   = max(abs(Δx)) / std(Δx)
    variance = var(Δx)
    """
    agg = x.squeeze(1).mean(-1)
    dif = agg[:, 1:] - agg[:, :-1]
    rv = dif.std(dim=1).clamp_min(1e-8)
    stress = dif.abs().max(dim=1).values / rv
    variance = dif.var(dim=1)
    return stress.cpu().numpy(), variance.cpu().numpy()


def test_set_and_scores(
    args: argparse.Namespace, k: int
) -> tuple[object, np.ndarray, dict[str, np.ndarray]]:
    cfg = json.loads(data_config_path(args, k).read_text())
    _train_ds, _val_ds, test_ds, _alpha, _meta = build_datasets(cfg)

    ys: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []

    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
    for batch in loader:
        x = batch["x"].float()
        stress, variance = window_scores(x)
        stress_parts.append(stress)
        variance_parts.append(variance)
        ys.append(batch["label"].numpy())

    y_true = np.concatenate(ys)
    scores = {
        "stress": np.concatenate(stress_parts),
        "variance": np.concatenate(variance_parts),
    }
    return test_ds, y_true, scores


@torch.no_grad()
def evaluate(model, test_ds, args: argparse.Namespace, device: torch.device):
    yt: list[np.ndarray] = []
    yp: list[np.ndarray] = []
    pr: list[np.ndarray] = []

    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
    for batch in loader:
        logits = model.predict(batch, device)
        pr.append(Fn.softmax(logits, dim=1).cpu().numpy())
        yp.append(logits.argmax(dim=1).cpu().numpy())
        yt.append(batch["label"].numpy())

    return np.concatenate(yt), np.concatenate(yp), np.concatenate(pr)


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if len(y_true) == 0:
        return float("nan")
    if metric == "accuracy":
        return float((y_true == y_pred).mean())
    return float(
        f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    )


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    # Include exact min/max values even when search lands on a boundary.
    eps = max(float(np.nanstd(values)) * 1e-12, 1e-12)
    edges[0] -= eps
    edges[-1] += eps
    return np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)


def decile_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    n_bins: int,
    metric: str,
) -> np.ndarray:
    idx = quantile_bins(score, n_bins)
    out = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        mask = idx == b
        out[b] = metric_value(y_true[mask], y_pred[mask], metric)
    return out


def calm_extreme_rows(
    results: dict[tuple[str, int], dict[str, np.ndarray]],
    score_by_k: dict[int, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for score_name in ["stress", "variance"]:
        for (tag, k), r in sorted(results.items(), key=lambda x: (x[0][1], x[0][0])):
            score = score_by_k[k][score_name]
            thr = np.quantile(score, args.extreme_q)
            calm = score <= thr
            extreme = score > thr

            calm_acc = metric_value(r["y_true"][calm], r["y_pred"][calm], "accuracy")
            extreme_acc = metric_value(
                r["y_true"][extreme], r["y_pred"][extreme], "accuracy"
            )
            calm_f1 = metric_value(r["y_true"][calm], r["y_pred"][calm], "macro_f1")
            extreme_f1 = metric_value(
                r["y_true"][extreme], r["y_pred"][extreme], "macro_f1"
            )

            rows.append(
                {
                    "condition": score_name,
                    "model": tag,
                    "k": k,
                    "threshold": float(thr),
                    "n_calm": int(calm.sum()),
                    "n_extreme": int(extreme.sum()),
                    "acc_calm": calm_acc,
                    "acc_extreme": extreme_acc,
                    "acc_drop": calm_acc - extreme_acc,
                    "f1_calm": calm_f1,
                    "f1_extreme": extreme_f1,
                    "f1_drop": calm_f1 - extreme_f1,
                    "f1_retention": extreme_f1 / max(calm_f1, 1e-12),
                }
            )
    return pd.DataFrame(rows)


def plot_condition(
    results: dict[tuple[str, int], dict[str, np.ndarray]],
    score_by_k: dict[int, dict[str, np.ndarray]],
    score_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Path:
    horizons = [k for k in args.horizons if any(k2 == k for (_tag, k2) in results)]
    if not horizons:
        raise RuntimeError("no evaluated horizons to plot")

    ncols = min(2, len(horizons))
    nrows = int(np.ceil(len(horizons) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 4.0 * nrows),
        sharey=True,
        squeeze=False,
    )

    x = np.arange(1, args.bins + 1)
    for ax, k in zip(axes.ravel(), horizons):
        for tag in args.models:
            r = results.get((tag, k))
            if r is None:
                continue
            y = decile_curve(
                r["y_true"],
                r["y_pred"],
                score_by_k[k][score_name],
                args.bins,
                args.metric,
            )
            ax.plot(x, y, marker="o", linewidth=1.6, markersize=3.5, label=tag)

        ax.set_title(f"k={k}")
        ax.set_xlabel(f"{score_name} quantile bin, low → high")
        ax.grid(True, alpha=0.25)

    for ax in axes[:, 0]:
        ax.set_ylabel(args.metric.replace("_", "-"))
    for ax in axes.ravel()[len(horizons) :]:
        ax.axis("off")

    title = (
        f"{args.metric.replace('_', '-')} by {score_name} decile — "
        f"{args.symbol} / {args.feature_mode}"
    )
    fig.suptitle(title)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))

    out = (
        out_dir
        / f"{args.symbol.lower()}_{args.feature_mode}_{score_name}_deciles_{args.metric}.pdf"
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.models = [m for m in args.models if m in MODEL_REGISTRY]

    device = torch.device(args.device)
    out_dir = args.out_dir or (
        REPO
        / "outputs"
        / "figures"
        / "crypto"
        / "coinbase"
        / args.symbol.lower()
        / "stress_variance"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    print("repo:", REPO)
    print("device:", device)
    print("models:", ", ".join(args.models))
    print("out:", out_dir.relative_to(REPO))

    checkpoints = discover_checkpoints(args)
    score_by_k: dict[int, dict[str, np.ndarray]] = {}
    results: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for k in args.horizons:
        print(f"\nhorizon k={k}")
        try:
            test_ds, y_true, scores = test_set_and_scores(args, k)
        except Exception as exc:
            print(f"  ! could not build test set: {exc}")
            continue

        score_by_k[k] = scores
        print(
            f"  test windows={len(y_true)} "
            f"stress p90={np.quantile(scores['stress'], args.extreme_q):.3f} "
            f"variance p90={np.quantile(scores['variance'], args.extreme_q):.3g}"
        )

        for tag in args.models:
            path = checkpoints.get(tag, {}).get(k)
            if path is None:
                print(f"  {tag:<20} — no checkpoint")
                continue

            ckpt = load_ckpt(path)
            cfg = ckpt["config"]
            if cfg.get("label_k") != k or cfg.get("feature_mode") != args.feature_mode:
                print(f"  {tag:<20} — config mismatch, skipped")
                continue

            model = build_model(tag, ckpt, device)
            yt, yp, _probs = evaluate(model, test_ds, args, device)
            if not np.array_equal(yt, y_true):
                raise RuntimeError(f"label/window misalignment for {tag} k={k}")

            results[(tag, k)] = {"y_true": yt, "y_pred": yp}
            print(
                f"  {tag:<20} "
                f"acc={metric_value(yt, yp, 'accuracy'):.4f} "
                f"macro_f1={metric_value(yt, yp, 'macro_f1'):.4f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("no checkpoints were evaluated")

    summary = calm_extreme_rows(results, score_by_k, args)
    csv_out = (
        out_dir
        / f"{args.symbol.lower()}_{args.feature_mode}_stress_variance_summary.csv"
    )
    summary.to_csv(csv_out, index=False)

    stress_pdf = plot_condition(results, score_by_k, "stress", args, out_dir)
    variance_pdf = plot_condition(results, score_by_k, "variance", args, out_dir)

    print("\nsaved:")
    print(" ", stress_pdf.relative_to(REPO))
    print(" ", variance_pdf.relative_to(REPO))
    print(" ", csv_out.relative_to(REPO))
    print(
        "\nInterpretation: stress deciles test jump-like shocks; variance deciles test "
        "sustained high-volatility windows. Do not collapse the two claims."
    )


if __name__ == "__main__":
    main()
