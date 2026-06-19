"""Image feature/preprocessing for the painting approach (spec section 2).

Builds a per-snapshot row stream ``(N, R, 2)`` with ``R = 2n + 3``:

- rows ``0..n-1``   : bid levels (row ``n-1`` = best bid)
- rows ``n..2n-1``  : ask levels (row ``n`` = best ask)
- rows ``2n..2n+2`` : trade-feature rows (channel 1 only)

Channel 0: ``ofi`` mode = per-level Cont order flow (bid row = ``bofi``, ask row
= ``aofi``; aggregate best signal = ``aofi - bofi``); ``lob`` mode = raw level
price.  Channel 1: signed resting depth + trade features.  A trading window
slices ``T_total`` consecutive snapshots, which are then padded to a square image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def load_orderbook(path: str, n_levels: int) -> pd.DataFrame:
    """Load an order-book CSV: parse ``time``, sort, drop duplicate timestamps."""
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
    """Load a trades CSV with ``trade_time``, ``price``, ``volume``, ``direction``."""
    df = pd.read_csv(path)
    if "trade_time" not in df.columns:
        raise ValueError(f"trades file {path} has no 'trade_time' column")
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    return df.sort_values("trade_time").reset_index(drop=True)


def mid_series(snaps: pd.DataFrame) -> np.ndarray:
    """Mid price per snapshot: ``(best_bid + best_ask) / 2``."""
    return ((snaps["bid_price_1"] + snaps["ask_price_1"]) / 2.0).to_numpy(np.float64)


def n_rows(config: dict) -> int:
    return 2 * config["n_levels"] + config["n_trade_rows"]


def _bid_row(level_idx: int, n: int) -> int:
    return n - 1 - level_idx


def _ask_row(level_idx: int, n: int) -> int:
    return n + level_idx


def _bid_ofi(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Bid-side Cont OFI ``bofi``; positive = bid liquidity added. Col 0 = 0."""
    dp = np.diff(price, prepend=price[0])
    prev = np.roll(vol, 1)
    e = np.where(dp > 0, vol, np.where(dp == 0, vol - prev, -prev))
    e[0] = 0.0
    return e


def _ask_ofi(price: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Ask-side Cont OFI ``aofi``; positive = ask liquidity added. Col 0 = 0."""
    dp = np.diff(price, prepend=price[0])
    prev = np.roll(vol, 1)
    e = np.where(dp < 0, vol, np.where(dp == 0, vol - prev, -prev))
    e[0] = 0.0
    return e


def _trade_rows(snaps, trades, interval_sec: int) -> np.ndarray:
    """``(N, 3)`` trade features: log1p(vol), buy-vol ratio, buy-count ratio."""
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
    """Build the per-snapshot row stream ``(N, R, 2)``."""
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


def best_level_ofi(rows: np.ndarray, config: dict) -> np.ndarray:
    """Net best-level order flow = ``aofi_best - bofi_best``."""
    n = config["n_levels"]
    return rows[:, _ask_row(0, n), 0] - rows[:, _bid_row(0, n), 0]


def pad_levels(img: np.ndarray, padded_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Repeat rows so the level axis reaches ``padded_size``; return (padded, starts)."""
    levels = img.shape[0]
    base, rem = divmod(padded_size, levels)
    repeats = np.full(levels, base, dtype=int)
    repeats[:rem] += 1
    padded = np.repeat(img, repeats, axis=0)
    starts = np.concatenate([[0], np.cumsum(repeats)]).astype(int)
    return padded, starts


def level_starts(config: dict) -> np.ndarray:
    dummy = np.zeros((n_rows(config), 1, 2), dtype=np.float32)
    _, starts = pad_levels(dummy, config["padded_size"])
    return starts


def build_mask(config: dict) -> np.ndarray:
    """Padded inpainting mask ``(padded_size, T_total, 1)``: 0 past, 1 future."""
    mask = np.zeros((config["padded_size"], config["T_total"], 1), dtype=np.float32)
    mask[:, config["T_past"] :, :] = 1.0
    return mask


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

    def denorm_channel0(self, value: np.ndarray, row: int) -> np.ndarray:
        return value * self.std[row, 0] + self.mean[row, 0]

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "clip": self.clip}

    @classmethod
    def from_dict(cls, config: dict, d: dict) -> "RollingNormalizer":
        obj = cls(config)
        obj.mean, obj.std, obj.clip = d["mean"], d["std"], d["clip"]
        return obj
