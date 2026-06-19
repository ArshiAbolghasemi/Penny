"""Univariate feature/preprocessing for the TimesFM forecaster.

TimesFM is univariate, so the only feature is the **mid-price** series.  The
``feature_mode`` field is kept for parity with the other approaches and recorded
in the config, but both modes forecast the same mid series (TimesFM cannot ingest
the multivariate book).  No row-feature normalizer is needed — the forecaster
works in window-relative returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_orderbook(path: str, n_levels: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"order-book file {path} has no 'time' column")
    df["time"] = pd.to_datetime(df["time"])
    keep = ["time"]
    for i in range(1, n_levels + 1):
        keep += [
            f"bid_price_{i}",
            f"bid_volume_{i}",
            f"ask_price_{i}",
            f"ask_volume_{i}",
        ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"order-book file {path} missing columns: {missing}")
    return df[keep].sort_values("time").drop_duplicates("time").reset_index(drop=True)


def mid_series(snaps: pd.DataFrame) -> np.ndarray:
    return ((snaps["bid_price_1"] + snaps["ask_price_1"]) / 2.0).to_numpy(np.float64)
