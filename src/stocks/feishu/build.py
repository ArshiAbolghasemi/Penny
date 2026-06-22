"""In-RAM feature builder + dataset factory for Feishu A-share equity data.

Memory model
------------
The whole feature matrix is built **in RAM** every run — no disk cache. We hold
a single per-(asset, day) feature array of shape ``(N_rows, NF)`` (~1 GB for the
in-sample set: ~1.06M day-rows × 259 float32 features) and slide T-day windows
lazily at training time (see :class:`~stocks.feishu.dataset.LOBDataset`).
Materialising every window would instead duplicate each day ~T times (tens of
GB), so the compact day-matrix + lazy slicing is both small and fast.

Building fresh each run (rather than reusing a memmap cache) guarantees the
features always reflect the current normalisation code — a stale cache built
before a normalisation fix is the classic source of NaN losses.

Expected data_dir contents (two flat multi-asset parquet files)::

    data_dir/
      lob_data_in_sample.parquet    # 5-min LOB snapshots, all symbols
      daily_data_in_sample.parquet  # daily OHLCV, all symbols

LOB file columns: asset_id, trade_day_id, time (HH:MM:SS),
  bid_price_1..10, ask_price_1..10, bid_volume_1..10, ask_volume_1..10.
  Columns are renamed 1-indexed → 0-indexed before passing to features.py.

Daily file columns: asset_id, trade_day_id, open, high, low, close,
  volume, amount, adj_factor, vwap_0930_0935.
  trade_day_id is renamed → "date" before passing to features/labels.

Label (causal pairing)
----------------------
A window of T days ending at day t is paired with the row label at day t+1:
  label_{t+1} = (close_{t+2} - vwap_{t+1}) / vwap_{t+1}
By end-of-day t the trader knows all features through day t but NOT the
morning vwap of day t+1 (the entry price), so there is zero leakage.

Split
-----
Per-asset chronological split of that asset's windows:
  train : first 70 %   val : next 15 %   test : last 15 %

Public API
----------
- discover_symbols(data_dir, config) → sorted list of symbol names (asset_id)
- build_datasets(config, data_dir, cache_dir, symbols)
      → (train_ds, val_ds, test_ds, meta)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stocks.feishu.dataset import LOBDataset
from stocks.feishu.features import (
    CLIP_VAL,
    N_LEVELS,
    N_OFI,
    N_OHLCV,
    causal_rolling_zscore,
    compute_ofi_tick,
    compute_ohlcv_features,
    extract_lob_day,
    n_features,
    snap_to_slots,
)
from stocks.feishu.labels import assign_labels, compute_forward_returns

_LOB_FILE = "lob_data_in_sample.parquet"
_DAILY_FILE = "daily_data_in_sample.parquet"
_SYM_COL = "asset_id"
_DAY_COL = "trade_day_id"
_TIME_COL = "time"

_TRAIN_FRAC = 0.70
_VAL_CUM = 0.85  # cumulative; val = [70%, 85%), test = [85%, 100%)


def _lob_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 1-indexed LOB columns (bid/ask_price/volume_1..10) to 0-indexed."""
    rename = {}
    for i in range(1, 11):
        for prefix in ("bid_price", "ask_price", "bid_volume", "ask_volume"):
            src = f"{prefix}_{i}"
            if src in df.columns:
                rename[src] = f"{prefix}_{i - 1}"
    return df.rename(columns=rename)


def discover_symbols(data_dir: str | Path, config: dict | None = None) -> list[str]:
    """Return sorted list of asset_id values present in the LOB flat file."""
    if config is None:
        config = {}
    root = Path(data_dir).resolve()
    lob_file = config.get("lob_file", _LOB_FILE)
    sym_col = config.get("symbol_col", _SYM_COL)
    p = root / lob_file
    if not p.exists():
        raise FileNotFoundError(
            f"LOB file not found: {p}\n"
            f"Contents of {root}: {sorted(e.name for e in root.iterdir())}"
        )
    lob = pd.read_parquet(p, columns=[sym_col])
    symbols = sorted(lob[sym_col].dropna().unique().tolist())
    if not symbols:
        raise ValueError(f"No symbols found in column '{sym_col}' of {p}")
    logger.info("Discovered {} symbols from {}", len(symbols), lob_file)
    return symbols


# ── feature build (in RAM) ───────────────────────────────────────────────────


def _build_feature_matrix(
    config: dict,
    data_dir: str | Path,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the per-(asset, day) feature matrix and row arrays in RAM.

    Built fresh from the parquet files every call — no disk cache — so the
    features always reflect the current normalisation code.

    Returns:
        ``(feat, row_labels, row_asset, NF)`` where
          feat        : ``(N_rows, NF)`` float32 array (in RAM).
          row_labels  : ``(N_rows,)`` int64 causal labels (-1 = invalid).
          row_asset   : ``(N_rows,)`` int64 asset index per row (contiguous).
          NF          : feature count.
    """
    nf = n_features(config)
    mode = config.get("feature_mode", "ofi")
    alpha = config["alpha"]
    n_levels = config.get("n_lob_levels", N_LEVELS)
    n_lob_feat = N_OFI if mode == "ofi" else 4 * n_levels
    root = Path(data_dir).resolve()
    sym_col = config.get("symbol_col", _SYM_COL)
    day_col = config.get("day_col", _DAY_COL)
    time_col = config.get("time_col", _TIME_COL)
    lob_file = config.get("lob_file", _LOB_FILE)
    daily_file = config.get("daily_file", _DAILY_FILE)

    logger.info("building feishu features (RAM) mode={} nf={}", mode, nf)

    # ── Pass 1 (daily, small): per-asset OHLCV raw feats, labels, day order ──
    daily_all = pd.read_parquet(root / daily_file).rename(columns={day_col: "date"})
    daily_all["date"] = daily_all["date"].astype(str)

    sym_set = set(symbols)
    daily_all = daily_all[daily_all[sym_col].isin(sym_set)]

    dates_map: dict[str, list[str]] = {}
    ohlcv_raw_map: dict[str, np.ndarray] = {}  # sym → (n_days, 19)
    labels_map: dict[str, np.ndarray] = {}  # sym → (n_days,) int64

    for sym, daily_sym in daily_all.groupby(sym_col, sort=True):
        daily_sym = daily_sym.sort_values("date").reset_index(drop=True)
        sym_dates = daily_sym["date"].tolist()
        if len(sym_dates) == 0:
            continue
        dates_map[sym] = sym_dates
        try:
            ohlcv_df = compute_ohlcv_features(daily_sym)
            ohlcv_raw_map[sym] = ohlcv_df.values.astype(np.float32)  # (n_days, 19)
        except Exception as exc:
            logger.warning("OHLCV failed for {}: {}", sym, exc)
            ohlcv_raw_map[sym] = np.zeros((len(sym_dates), N_OHLCV), dtype=np.float32)
        fwd = compute_forward_returns(daily_sym)
        labels_map[sym] = assign_labels(fwd, alpha)
    del daily_all

    # ── Row layout: assets in sorted order, days in date order ──────────────
    ordered_syms = [s for s in symbols if s in dates_map]
    ranges: dict[str, tuple[int, int]] = {}
    ptr = 0
    for i, sym in enumerate(ordered_syms):
        n_days = len(dates_map[sym])
        ranges[sym] = (ptr, ptr + n_days)
        ptr += n_days
    n_rows = ptr
    if n_rows == 0:
        raise ValueError("No (asset, day) rows to build — check data files.")

    sym_to_idx = {s: i for i, s in enumerate(ordered_syms)}
    feat = np.zeros((n_rows, nf), dtype=np.float32)
    row_labels = np.full(n_rows, -1, dtype=np.int64)
    row_asset = np.empty(n_rows, dtype=np.int64)
    ohlcv_block = np.zeros((n_rows, N_OHLCV), dtype=np.float32)

    for sym in ordered_syms:
        lo, hi = ranges[sym]
        row_asset[lo:hi] = sym_to_idx[sym]
        row_labels[lo:hi] = labels_map[sym]
        ohlcv_block[lo:hi] = ohlcv_raw_map[sym]

    # ── Pass 2 (LOB, large): per-asset OFI/LOB features → feat[:, :n_lob] ────
    lob_cols = [sym_col, day_col, time_col]
    for i in range(N_LEVELS):
        j = i + 1
        lob_cols += [
            f"bid_price_{j}",
            f"ask_price_{j}",
            f"bid_volume_{j}",
            f"ask_volume_{j}",
        ]
    lob_all = pd.read_parquet(root / lob_file, columns=lob_cols)
    lob_all = _lob_rename(lob_all).rename(columns={day_col: "date"})
    lob_all["date"] = lob_all["date"].astype(str)
    lob_all = lob_all[lob_all[sym_col].isin(set(ordered_syms))]

    for sym, lob_sym in lob_all.groupby(sym_col, sort=False):
        if sym not in ranges:
            continue
        lo, hi = ranges[sym]
        sym_dates = dates_map[sym]
        date_to_local = {d: k for k, d in enumerate(sym_dates)}
        lob_sym = lob_sym.sort_values(["date", time_col])

        block = np.zeros((len(sym_dates), n_lob_feat), dtype=np.float32)
        for d, day_lob in lob_sym.groupby("date", sort=False):
            k = date_to_local.get(d)
            if k is None or len(day_lob) == 0:
                continue
            day_lob = day_lob.reset_index(drop=True)
            if mode == "ofi":
                ofi_df = compute_ofi_tick(day_lob)
                block[k] = snap_to_slots(ofi_df, day_lob).reshape(-1)  # (240,)
            else:
                block[k] = extract_lob_day(day_lob, n_levels)  # (4n,)

        block = causal_rolling_zscore(block)
        feat[lo:hi, :n_lob_feat] = np.clip(block, -CLIP_VAL, CLIP_VAL)
    del lob_all

    # ── Pass 3: cross-sectional z-score of OHLCV per day → feat[:, n_lob:] ───
    # Group global rows by their calendar day across all assets.
    day_to_rows: dict[str, list[int]] = {}
    for sym in ordered_syms:
        lo, _ = ranges[sym]
        for k, d in enumerate(dates_map[sym]):
            day_to_rows.setdefault(d, []).append(lo + k)

    for d, rows in day_to_rows.items():
        idx = np.asarray(rows, dtype=np.int64)
        mat = ohlcv_block[idx].astype(np.float64)
        mu = np.nanmean(mat, axis=0)
        sigma = np.nanstd(mat, axis=0)
        sigma = np.where(~np.isfinite(sigma) | (sigma < 1e-8), 1.0, sigma)
        normed = ((mat - mu) / sigma).astype(np.float32)
        feat[idx, n_lob_feat:] = np.clip(normed, -CLIP_VAL, CLIP_VAL)

    del ohlcv_block

    # final safety net: no non-finite values ever reach the model
    np.nan_to_num(feat, copy=False, nan=0.0, posinf=CLIP_VAL, neginf=-CLIP_VAL)
    logger.info("feishu features built (in RAM): {:,} rows × {} feat", n_rows, nf)
    return feat, row_labels, row_asset, nf


# ── dataset factory ──────────────────────────────────────────────────────────


def _asset_ranges(row_asset: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous ``[lo, hi)`` row ranges, one per asset index."""
    if len(row_asset) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(row_asset)) + 1
    edges = [0, *boundaries.tolist(), len(row_asset)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def build_datasets(
    config: dict,
    data_dir: str | Path,
    cache_dir: str | Path,
    symbols: list[str],
) -> tuple[LOBDataset, LOBDataset, LOBDataset, dict]:
    """Return ``(train_ds, val_ds, test_ds, meta)`` over the in-RAM feature matrix.

    Windows are computed per asset (causal label, no straddling), then split
    70 / 15 / 15 chronologically within each asset.

    Note: ``cache_dir`` is accepted for API compatibility but unused — features
    are built fresh in RAM each call (no disk cache).
    """
    T = config["T_past"]
    feat, row_labels, row_asset, nf = _build_feature_matrix(config, data_dir, symbols)

    train_starts: list[int] = []
    val_starts: list[int] = []
    test_starts: list[int] = []

    for lo, hi in _asset_ranges(row_asset):
        # valid starts: window [s, s+T) within asset, causal label row_labels[s+T] valid
        asset_starts = [s for s in range(lo, hi - T) if row_labels[s + T] >= 0]
        if not asset_starts:
            continue
        n = len(asset_starts)
        n_train = int(_TRAIN_FRAC * n)
        n_val = int(_VAL_CUM * n)
        train_starts.extend(asset_starts[:n_train])
        val_starts.extend(asset_starts[n_train:n_val])
        test_starts.extend(asset_starts[n_val:])

    train_arr = np.asarray(train_starts, dtype=np.int64)
    val_arr = np.asarray(val_starts, dtype=np.int64)
    test_arr = np.asarray(test_starts, dtype=np.int64)

    def _balance(starts: np.ndarray) -> dict:
        if len(starts) == 0:
            return {"down": 0.0, "stationary": 0.0, "up": 0.0}
        lbl = row_labels[starts + T]
        c = np.bincount(lbl, minlength=3) / len(lbl)
        return {"down": float(c[0]), "stationary": float(c[1]), "up": float(c[2])}

    meta = {
        "counts": {
            "train": len(train_arr),
            "val": len(val_arr),
            "test": len(test_arr),
        },
        "class_balance": _balance(train_arr),
        "n_features": nf,
        "n_rows": len(row_asset),
    }
    logger.info(
        "windows — train:{} val:{} test:{}",
        meta["counts"]["train"],
        meta["counts"]["val"],
        meta["counts"]["test"],
    )

    train_ds = LOBDataset(feat, train_arr, row_labels, T)
    val_ds = LOBDataset(feat, val_arr, row_labels, T)
    test_ds = LOBDataset(feat, test_arr, row_labels, T)
    return train_ds, val_ds, test_ds, meta
