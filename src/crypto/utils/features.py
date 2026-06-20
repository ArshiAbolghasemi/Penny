"""Feature extraction from resampled Binance LOB parquet data.

Two modes controlled by ``config["feature_mode"]``:

  ``"lob"`` — price offsets + log volumes per level (classical DeepLOB input)
  ``"ofi"`` — per-level signed Cont-OFI + signed log transform (default)

Both modes append the same 11 microstructure / trade / quote features.

Feature layout
--------------
OFI mode  (n levels):
  [0 : n)       bid OFI per level  (signed-log)
  [n : 2n)      ask OFI per level  (signed-log)
  [2n : 2n+3)   spread/mid, log-depth-imbalance, log-return
  [2n+3 : 2n+8) log-buy-vol, log-sell-vol, trade-imbalance, log-trade-count, vwap-dev
  [2n+8 : 2n+11) log-trade-count (activity proxy), spread-norm, |log-ret| (range proxy)
  total = 2n + 11

LOB mode  (n levels):
  [0 : n)       bid price offset = (mid - bid_p[i]) / mid
  [n : 2n)      ask price offset = (ask_p[i] - mid) / mid
  [2n : 3n)     log1p bid volume per level
  [3n : 4n)     log1p ask volume per level
  [4n : 4n+3)   spread/mid, log-depth-imbalance, log-return
  [4n+3 : 4n+8) log-buy-vol, log-sell-vol, trade-imbalance, log-trade-count, vwap-dev
  [4n+8 : 4n+11) log-trade-count (activity proxy), spread-norm, |log-ret| (range proxy)
  total = 4n + 11

Input DataFrame columns (from resampled parquet)
-------------------------------------------------
  bids[i].price, bids[i].amount, asks[i].price, asks[i].amount  (i = 0..n-1)
  mid, spread
  trade_count, buy_vol, sell_vol, vwap, trade_imbalance
  quote_bid_price, quote_ask_price, quote_bid_amount, quote_ask_amount
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def n_features(config: dict) -> int:
    n = config["n_lob_levels"]
    mode = config.get("feature_mode", "ofi")
    lob = 2 * n if mode == "ofi" else 4 * n
    return lob + 11


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def extract_features(day_df: pd.DataFrame, config: dict) -> np.ndarray:
    """Compute raw (un-normalised) feature matrix for one calendar day.

    Args:
        day_df:  One day's slice of the resampled parquet, reset_index applied.
        config:  Dict with ``n_lob_levels`` and ``feature_mode``.

    Returns:
        float32 array of shape ``(N, n_features(config))``.
    """
    n = config["n_lob_levels"]
    mode = config.get("feature_mode", "ofi")
    N = len(day_df)
    F = n_features(config)
    out = np.zeros((N, F), dtype=np.float32)
    eps = 1e-12

    bid_p = np.stack([day_df[f"bids[{i}].price"].values for i in range(n)], axis=1)
    bid_v = np.stack([day_df[f"bids[{i}].amount"].values for i in range(n)], axis=1)
    ask_p = np.stack([day_df[f"asks[{i}].price"].values for i in range(n)], axis=1)
    ask_v = np.stack([day_df[f"asks[{i}].amount"].values for i in range(n)], axis=1)
    mid = day_df["mid"].values.astype(np.float64)

    col = 0

    if mode == "ofi":
        for i in range(n):
            prev_bp = np.roll(bid_p[:, i], 1)
            prev_bp[0] = bid_p[0, i]
            prev_bv = np.roll(bid_v[:, i], 1)
            prev_bv[0] = bid_v[0, i]
            dp_b = bid_p[:, i] - prev_bp
            bofi = np.where(
                dp_b > 0,
                bid_v[:, i],
                np.where(dp_b < 0, -prev_bv, bid_v[:, i] - prev_bv),
            )
            bofi[0] = 0.0

            prev_ap = np.roll(ask_p[:, i], 1)
            prev_ap[0] = ask_p[0, i]
            prev_av = np.roll(ask_v[:, i], 1)
            prev_av[0] = ask_v[0, i]
            dp_a = ask_p[:, i] - prev_ap
            aofi = np.where(
                dp_a < 0,
                ask_v[:, i],
                np.where(dp_a > 0, -prev_av, ask_v[:, i] - prev_av),
            )
            aofi[0] = 0.0

            out[:, col] = _signed_log(bofi).astype(np.float32)
            out[:, col + 1] = _signed_log(aofi).astype(np.float32)
            col += 2
    else:  # lob
        for i in range(n):
            out[:, i] = ((mid - bid_p[:, i]) / (mid + eps)).astype(np.float32)
            out[:, n + i] = ((ask_p[:, i] - mid) / (mid + eps)).astype(np.float32)
            out[:, 2 * n + i] = np.log1p(bid_v[:, i]).astype(np.float32)
            out[:, 3 * n + i] = np.log1p(ask_v[:, i]).astype(np.float32)
        col = 4 * n

    # ── Microstructure (3) ────────────────────────────────────────────────────
    spread_norm = day_df["spread"].values / (mid + eps)
    total_bid = bid_v.sum(axis=1)
    total_ask = ask_v.sum(axis=1)
    log_dimbal = np.log((total_bid + 1e-8) / (total_ask + 1e-8))

    prev_mid = np.roll(mid, 1)
    prev_mid[0] = mid[0]
    log_ret = np.log((mid + eps) / (prev_mid + eps))
    log_ret[0] = 0.0

    out[:, col] = spread_norm.astype(np.float32)
    out[:, col + 1] = log_dimbal.astype(np.float32)
    out[:, col + 2] = log_ret.astype(np.float32)
    col += 3

    # ── Trade features (5) ────────────────────────────────────────────────────
    buy_v = day_df["buy_vol"].fillna(0.0).values
    sell_v = day_df["sell_vol"].fillna(0.0).values
    cnt = day_df["trade_count"].fillna(0.0).values
    vwap = day_df["vwap"].values
    total_t = buy_v + sell_v
    t_imbal = (buy_v - sell_v) / (total_t + 1e-8)
    vwap_dev = np.where(
        total_t > 0,
        (np.where(np.isfinite(vwap), vwap, mid) - mid) / (mid + eps),
        0.0,
    )

    out[:, col] = np.log1p(buy_v).astype(np.float32)
    out[:, col + 1] = np.log1p(sell_v).astype(np.float32)
    out[:, col + 2] = t_imbal.astype(np.float32)
    out[:, col + 3] = np.log1p(cnt).astype(np.float32)
    out[:, col + 4] = vwap_dev.astype(np.float32)
    col += 5

    # ── Quote / activity features (3) ────────────────────────────────────────
    # Original raw-data features: log_n_quote_updates, spread_mean_norm, mid_range_norm
    # Resampled parquet doesn't store intra-bin quote count or spread mean.
    # Substitutes that preserve the same semantic intent:
    #   log_n_quote_updates → log1p(trade_count)   (both reflect market activity)
    #   spread_mean_norm    → spread / mid          (already computed above)
    #   mid_range_norm      → |log_ret|             (inter-bin price movement proxy)
    out[:, col] = np.log1p(cnt).astype(np.float32)
    out[:, col + 1] = spread_norm.astype(np.float32)
    out[:, col + 2] = np.abs(log_ret).astype(np.float32)

    return out
