"""Feature extraction for the TimesFM forecaster.

In OFI mode the model receives two input channels per timestep: the window-relative
mid-price return and the normalised best-level net OFI (``aofi_best - bofi_best``).
In LOB mode only the mid-price return is used.  The TimesFM pretrained prior is
always univariate (mid only); the multivariate signal is fed to the residual
transformer only.
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


def _bid_ofi(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Bid-side Cont OFI; positive = bid liquidity added. First element = 0."""
    dp = np.diff(price, prepend=price[0])
    prev = np.roll(vol, 1)
    e = np.where(dp > 0, vol, np.where(dp == 0, vol - prev, -prev))
    e[0] = 0.0
    return e


def _ask_ofi(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Ask-side Cont OFI; positive = ask liquidity added. First element = 0."""
    dp = np.diff(price, prepend=price[0])
    prev = np.roll(vol, 1)
    e = np.where(dp < 0, vol, np.where(dp == 0, vol - prev, -prev))
    e[0] = 0.0
    return e


def net_ofi_series(snaps: pd.DataFrame) -> np.ndarray:
    """Best-level net OFI = aofi_best - bofi_best per snapshot."""
    bp = snaps["bid_price_1"].to_numpy(np.float64)
    bv = snaps["bid_volume_1"].to_numpy(np.float64)
    ap = snaps["ask_price_1"].to_numpy(np.float64)
    av = snaps["ask_volume_1"].to_numpy(np.float64)
    return _ask_ofi(ap, av) - _bid_ofi(bp, bv)
