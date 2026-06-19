"""Multivariate feature/preprocessing for the CSDI forecaster.

Builds the same per-snapshot row stream ``(N, R, 2)`` as the painting approach
(``R = 2n + 3``): channel 0 is per-level Cont order flow (``ofi``: bid row =
``bofi``, ask row = ``aofi``) or the raw level price (``lob``); channel 1 is
signed resting depth + trade features.  The forecaster consumes the past window
of this stream directly as a ``(2R, T_past)`` multivariate series — no square
padding or inpainting mask is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


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


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "trade_time" not in df.columns:
        raise ValueError(f"trades file {path} has no 'trade_time' column")
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    return df.sort_values("trade_time").reset_index(drop=True)


def mid_series(snaps: pd.DataFrame) -> np.ndarray:
    return ((snaps["bid_price_1"] + snaps["ask_price_1"]) / 2.0).to_numpy(np.float64)


def n_rows(config: dict) -> int:
    return 2 * config["n_levels"] + config["n_trade_rows"]


def _bid_row(level_idx: int, n: int) -> int:
    return n - 1 - level_idx


def _ask_row(level_idx: int, n: int) -> int:
    return n + level_idx


def _bid_ofi(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
    dp = np.diff(price, prepend=price[0])
    prev = np.roll(vol, 1)
    e = np.where(dp > 0, vol, np.where(dp == 0, vol - prev, -prev))
    e[0] = 0.0
    return e


def _ask_ofi(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
    dp = np.diff(price, prepend=price[0])
    prev = np.roll(vol, 1)
    e = np.where(dp < 0, vol, np.where(dp == 0, vol - prev, -prev))
    e[0] = 0.0
    return e


def _trade_rows(snaps, trades, interval_sec: int) -> np.ndarray:
    n = len(snaps)
    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = 0.0
    out[:, 1] = 0.5
    out[:, 2] = 0.5
    if trades is None or len(trades) == 0:
        return out
    snap_ns = snaps["time"].to_numpy("datetime64[ns]").astype(np.int64)
    tr_ns = trades["trade_time"].to_numpy("datetime64[ns]").astype(np.int64)
    vol = trades["volume"].to_numpy(np.float64)
    is_buy = (trades["direction"].astype(str).str.lower() == "buy").to_numpy()
    win = np.int64(interval_sec) * 1_000_000_000

    def prefix(a):
        return np.concatenate([[0.0], np.cumsum(a)])

    p_vol = prefix(vol)
    p_buyvol = prefix(np.where(is_buy, vol, 0.0))
    p_cnt = prefix(np.ones_like(vol))
    p_buycnt = prefix(is_buy.astype(np.float64))
    left = np.searchsorted(tr_ns, snap_ns - win, side="left")
    right = np.searchsorted(tr_ns, snap_ns, side="right")
    tot_vol = p_vol[right] - p_vol[left]
    buy_vol = p_buyvol[right] - p_buyvol[left]
    cnt = p_cnt[right] - p_cnt[left]
    buy_cnt = p_buycnt[right] - p_buycnt[left]
    has = cnt > 0
    out[:, 0] = np.log1p(tot_vol)
    out[has, 1] = buy_vol[has] / (tot_vol[has] + 1e-12)
    out[has, 2] = buy_cnt[has] / (cnt[has] + 1e-12)
    return out


def build_global_rows(snaps, trades, config: dict) -> np.ndarray:
    n = config["n_levels"]
    r = n_rows(config)
    nsnap = len(snaps)
    mode = config["feature_mode"]
    rows = np.zeros((nsnap, r, 2), dtype=np.float64)
    for i in range(n):
        bp = snaps[f"bid_price_{i + 1}"].to_numpy(np.float64)
        bv = snaps[f"bid_volume_{i + 1}"].to_numpy(np.float64)
        ap = snaps[f"ask_price_{i + 1}"].to_numpy(np.float64)
        av = snaps[f"ask_volume_{i + 1}"].to_numpy(np.float64)
        rb, ra = _bid_row(i, n), _ask_row(i, n)
        if mode == "ofi":
            rows[:, rb, 0] = _bid_ofi(bp, bv)
            rows[:, ra, 0] = _ask_ofi(ap, av)
        elif mode == "lob":
            rows[:, rb, 0] = bp
            rows[:, ra, 0] = ap
        else:
            raise ValueError(f"unknown feature_mode: {mode!r}")
        rows[:, rb, 1] = bv
        rows[:, ra, 1] = -av
    rows[:, 2 * n : 2 * n + 3, 1] = _trade_rows(
        snaps, trades, config["snapshot_interval_sec"]
    )
    return rows


class RollingNormalizer:
    """Per-row/per-channel z-score + outlier clip, fit on training only and frozen."""

    def __init__(self, config: dict) -> None:
        self.window = int(config["norm_window_snapshots"])
        self.clip_q = float(config["clip_percentile"])
        self.mean = self.std = self.clip = None

    def fit(self, rows_train: np.ndarray) -> None:
        fit_rows = (
            rows_train[-self.window :] if len(rows_train) > self.window else rows_train
        )
        self.mean = fit_rows.mean(axis=0)
        std = fit_rows.std(axis=0)
        std[std == 0] = 1.0
        self.std = std
        z = (rows_train - self.mean) / self.std
        self.clip = np.quantile(np.abs(z), self.clip_q, axis=0)
        self.clip[self.clip == 0] = 1.0
        logger.info("normalizer fit on {} rows", len(fit_rows))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("normalizer used before fit()")
        z = (rows - self.mean) / (self.std + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "clip": self.clip}

    @classmethod
    def from_dict(cls, config: dict, d: dict) -> "RollingNormalizer":
        obj = cls(config)
        obj.mean, obj.std, obj.clip = d["mean"], d["std"], d["clip"]
        return obj
