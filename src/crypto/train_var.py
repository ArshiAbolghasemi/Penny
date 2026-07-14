"""Walk-forward VAR baseline for LOB trend classification.

Fits a bivariate ``VAR(p)`` on trailing ``[mid log-return, best-level OFI]``
history at each evaluation point (no lookahead; lag order chosen by AIC up to
``var_maxlags``), forecasts ``label_k`` steps ahead, integrates the predicted
log-returns to a forward mid, and buckets the trend-ratio with the same
thresholds used by the neural models (``crypto.labels``). A classical
statistical benchmark, not a learned model, so there is no checkpoint to save
beyond the run's config and metrics.

Usage::

    uv run python -m crypto.train_var configs/crypto/nobitex/var/btcirt_ofi_k10.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from statsmodels.tsa.vector_ar.var_model import VAR

from crypto.classical import eval_points, report, trend_class
from crypto.features import extract_features, n_features
from crypto.loader import build_cache


warnings.filterwarnings("ignore")


def _forecast_trend(mid, ofi0, centre: int, t_past: int, k: int, maxlags: int) -> float:
    ret = np.diff(np.log(mid[centre - t_past : centre + 1]))
    ofi = ofi0[centre - t_past + 1 : centre + 1]
    hist = np.column_stack([ret, ofi])
    bwd = mid[centre - k : centre].mean()
    try:
        fit = VAR(hist).fit(maxlags=maxlags, ic="aic")
        fc = fit.forecast(hist[-fit.k_ar :], steps=k)  # (k, 2)
        fwd = float(mid[centre] * np.exp(fc[:, 0].sum()))
    except Exception:
        fwd = float(mid[centre])  # fall back to a random-walk forecast
    return (fwd - bwd) / bwd if bwd > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/crypto/nobitex/var/btcirt_ofi_k10.json"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    maxlags = int(config.get("var_maxlags", 5))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(config["checkpoint_dir"]) / f"var_{config['symbol']}_{stamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    feat, mid, ts = build_cache(config, extract_features, n_features, tag="lob")
    ofi0 = np.asarray(
        feat[:, 0], dtype=np.float64
    )  # best-level net Cont-OFI (z-scored)
    N = len(mid)
    train_end = int(N * config["train_frac"])
    val_end = int(N * (config["train_frac"] + config["val_frac"]))

    splits, labels, alpha = eval_points(mid, ts, config, train_end, val_end)
    t_past, k = config["T_past"], config["label_k"]
    logger.info(
        "VAR  symbol={}  maxlags={}  alpha={:.6f}  eval points: val={} test={}",
        config["symbol"],
        maxlags,
        alpha,
        len(splits["val"]),
        len(splits["test"]),
    )

    results = {}
    for split in ("val", "test"):
        centres = splits[split]
        y_true, y_pred = [], []
        for i, c in enumerate(centres):
            trend_ratio = _forecast_trend(mid, ofi0, c, t_past, k, maxlags)
            y_pred.append(trend_class(trend_ratio, alpha))
            y_true.append(int(labels[c]))
            if (i + 1) % 200 == 0:
                logger.info("  {} {}/{}", split, i + 1, len(centres))
        results[split] = report(y_true, y_pred, name=split.upper())

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
