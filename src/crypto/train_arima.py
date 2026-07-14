"""Walk-forward ARIMA baseline for LOB trend classification.

Fits ``ARIMA(p,d,q)`` on the trailing ``T_past`` mid-price history at each
evaluation point (no lookahead), forecasts ``label_k`` steps ahead, and
buckets the forecast trend-ratio with the same down/stationary/up thresholds
used by the neural models (``crypto.labels``) — a classical statistical
benchmark, not a learned model, so there is no checkpoint to save beyond the
run's config and metrics.

Usage::

    uv run python -m crypto.train_arima configs/crypto/nobitex/arima/btcirt_ofi_k10.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

from loguru import logger
from statsmodels.tsa.arima.model import ARIMA

from crypto.classical import eval_points, report, trend_class
from crypto.features import extract_features, n_features
from crypto.loader import build_cache

warnings.filterwarnings("ignore")


def _forecast_trend(
    mid, centre: int, t_past: int, k: int, order: tuple[int, int, int]
) -> float:
    hist = mid[centre - t_past + 1 : centre + 1]
    bwd = mid[centre - k : centre].mean()
    try:
        fit = ARIMA(hist, order=order).fit()
        fwd = float(fit.forecast(steps=k).mean())
    except Exception:
        fwd = float(hist[-1])  # fall back to a random-walk forecast
    return (fwd - bwd) / bwd if bwd > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/crypto/nobitex/arima/btcirt_ofi_k10.json"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: {}", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    order = tuple(config.get("arima_order", [2, 1, 2]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(config["checkpoint_dir"]) / f"arima_{config['symbol']}_{stamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.add(ckpt_dir / "train.log", level="DEBUG")

    _, mid, ts = build_cache(config, extract_features, n_features, tag="lob")
    N = len(mid)
    train_end = int(N * config["train_frac"])
    val_end = int(N * (config["train_frac"] + config["val_frac"]))

    splits, labels, alpha = eval_points(mid, ts, config, train_end, val_end)
    t_past, k = config["T_past"], config["label_k"]
    logger.info(
        "ARIMA{}  symbol={}  alpha={:.6f}  eval points: val={} test={}",
        order,
        config["symbol"],
        alpha,
        len(splits["val"]),
        len(splits["test"]),
    )

    results = {}
    for split in ("val", "test"):
        centres = splits[split]
        y_true, y_pred = [], []
        for i, c in enumerate(centres):
            trend_ratio = _forecast_trend(mid, c, t_past, k, order)
            y_pred.append(trend_class(trend_ratio, alpha))
            y_true.append(int(labels[c]))
            if (i + 1) % 200 == 0:
                logger.info("  {} {}/{}", split, i + 1, len(centres))
        results[split] = report(y_true, y_pred, name=split.upper())

    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))
    (ckpt_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
