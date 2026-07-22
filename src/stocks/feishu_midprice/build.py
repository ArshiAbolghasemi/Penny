"""In-RAM feature builder + dataset factory for Feishu **mid-price** prediction.

This is the mid-price-trend sibling of :mod:`stocks.feishu.build`. Features are
identical (240 intraday slot-OFI + 19 daily OHLCV = 259, reused verbatim from
:mod:`stocks.feishu.features`); the **only** difference is the label:

  base task  : label from ``(close_{t+1} - vwap_t) / vwap_t``   (traded prices)
  this task  : label from ``(mid_{t+h}  - mid_t)  / mid_t``     (quote mid-price)

where ``mid_t = (best_bid + best_ask) / 2`` is taken from the **closing LOB
snapshot** of day ``t`` (the last intraday snapshot, ~15:00; override the exact
slot with ``config['mid_ref_time']``), and ``h = config['horizon']`` is the
forward horizon in trading days. See :mod:`stocks.feishu_midprice.labels`.

Extraction of the closing mid reuses the same streamed-dask machinery as the OFI
pass (row-group partitions, trailing-day carry so no ``(asset, day)`` is ever
split across a chunk), reading only the best-level price columns.

Train/val/test protocol, causality guarantees, and the in-RAM memory model are
unchanged from :mod:`stocks.feishu.build` — refer to that module's docstring.

Public API
----------
- discover_symbols(data_dir, config) → sorted symbols across both periods
- build_datasets(config, data_dir, symbols) → (train_ds, val_ds, test_ds, meta)
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
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
    compute_ofi_slots_chunk,
    compute_ohlcv_features,
    n_features,
)
from stocks.feishu_midprice.labels import (
    assign_labels_quantile,
    compute_class_thresholds,
    compute_mid_forward_returns,
)

_LOB_FILE = "lob_data_in_sample.parquet"
_DAILY_FILE = "daily_data_in_sample.parquet"
_LOB_FILE_OOS = "lob_data_release_stage_out_of_sample.parquet"
_DAILY_FILE_OOS = "daily_data_release_stage_out_of_sample.parquet"
_SYM_COL = "asset_id"
_DAY_COL = "trade_day_id"
_TIME_COL = "time"

# Raw (1-indexed) best-level price columns used to form the closing mid-price.
_BID1_COL = "bid_price_1"
_ASK1_COL = "ask_price_1"

# Row groups per streamed LOB partition (see stocks.feishu.build).
_ROW_GROUPS_PER_CHUNK = 8

# In-sample chronological train/val cut (override with config["train_frac"]).
_INSAMPLE_TRAIN_FRAC = 0.80


def _paths(data_dir: str | Path, config: dict) -> dict[str, Path]:
    """Resolve the four parquet paths from config (with defaults)."""
    root = Path(data_dir).resolve()
    return {
        "lob": root / config.get("lob_file", _LOB_FILE),
        "daily": root / config.get("daily_file", _DAILY_FILE),
        "lob_oos": root / config.get("lob_file_oos", _LOB_FILE_OOS),
        "daily_oos": root / config.get("daily_file_oos", _DAILY_FILE_OOS),
    }


def _require(paths: dict[str, Path]) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Feishu data file(s):\n  " + "\n  ".join(missing)
        )


def _lob_columns(sym_col: str, day_col: str, time_col: str) -> list[str]:
    cols = [sym_col, day_col, time_col]
    for i in range(N_LEVELS):
        j = i + 1
        cols += [
            f"bid_price_{j}",
            f"ask_price_{j}",
            f"bid_volume_{j}",
            f"ask_volume_{j}",
        ]
    return cols


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
    """Return the sorted union of asset ids across both daily files."""
    if config is None:
        config = {}
    paths = _paths(data_dir, config)
    _require(paths)
    sym_col = config.get("symbol_col", _SYM_COL)

    seen: set[str] = set()
    for key in ("daily", "daily_oos"):
        col = dd.read_parquet(paths[key], columns=[sym_col], engine="pyarrow")[
            sym_col
        ].compute()
        seen |= set(col.dropna().tolist())
    symbols = sorted(seen)
    if not symbols:
        raise ValueError(f"No symbols found in column '{sym_col}'")
    logger.info("Discovered {} symbols (in-sample ∪ out-of-sample)", len(symbols))
    return symbols


# ── feature build (in RAM) ───────────────────────────────────────────────────


def _read_daily(path: Path, sym_col: str, day_col: str, symbols: set[str]):
    """Load one daily file (small) via dask, restricted to *symbols*."""
    df = dd.read_parquet(path, engine="pyarrow").compute()
    df = df.rename(columns={day_col: "date"})
    df["date"] = df["date"].astype(str)
    return df[df[sym_col].isin(symbols)]


def _stream_ofi_into(
    path: Path,
    config: dict,
    ofi_block: np.ndarray,
    sym_to_idx: dict[str, int],
    day_to_idx: dict[str, int],
    row_of: np.ndarray,
) -> None:
    """Stream one LOB parquet file and accumulate raw slot-OFI into *ofi_block*.

    Identical to :func:`stocks.feishu.build._stream_ofi_into`: dask row-group
    partitions reduced one at a time, with the trailing (possibly truncated)
    trade day of each partition carried into the next so no ``(asset, day)``
    group is ever computed from partial data.
    """
    sym_col = config.get("symbol_col", _SYM_COL)
    day_col = config.get("day_col", _DAY_COL)
    time_col = config.get("time_col", _TIME_COL)
    n_rg = int(config.get("lob_row_groups_per_chunk", _ROW_GROUPS_PER_CHUNK))

    ddf = dd.read_parquet(
        path,
        columns=_lob_columns(sym_col, day_col, time_col),
        engine="pyarrow",
        split_row_groups=n_rg,
    )
    logger.info("streaming OFI from {} in {} partitions", path.name, ddf.npartitions)

    carry: pd.DataFrame | None = None
    for i in range(ddf.npartitions):
        part = ddf.partitions[i].compute()
        if carry is not None and len(carry):
            part = pd.concat([carry, part], ignore_index=True)
            carry = None
        if part.empty:
            continue
        if i < ddf.npartitions - 1:
            last_day = part[day_col].iloc[-1]
            tail = part[day_col] == last_day
            carry, part = part[tail].copy(), part[~tail]
            if part.empty:
                continue
        _reduce_ofi_chunk(
            part, sym_col, day_col, time_col, ofi_block, sym_to_idx, day_to_idx, row_of
        )

    if carry is not None and len(carry):
        _reduce_ofi_chunk(
            carry, sym_col, day_col, time_col, ofi_block, sym_to_idx, day_to_idx, row_of
        )


def _reduce_ofi_chunk(
    part: pd.DataFrame,
    sym_col: str,
    day_col: str,
    time_col: str,
    ofi_block: np.ndarray,
    sym_to_idx: dict[str, int],
    day_to_idx: dict[str, int],
    row_of: np.ndarray,
) -> None:
    """Compute slot-OFI for every (asset, day) in *part* and scatter into rows."""
    a_idx = part[sym_col].map(sym_to_idx).to_numpy(dtype=np.float64)
    d_idx = part[day_col].astype(str).map(day_to_idx).to_numpy(dtype=np.float64)
    known = ~(np.isnan(a_idx) | np.isnan(d_idx))
    if not known.any():
        return

    part = part[known]
    a_idx = a_idx[known].astype(np.int64)
    d_idx = d_idx[known].astype(np.int64)
    rows = row_of[a_idx, d_idx]
    have_row = rows >= 0
    if not have_row.any():
        return
    part, rows = part[have_row], rows[have_row]

    times = part[time_col].astype(str).to_numpy()
    order = np.lexsort((times, rows))
    part = _lob_rename(part.iloc[order].reset_index(drop=True))
    rows = rows[order]

    grp_rows, grp_cols, vals = compute_ofi_slots_chunk(part, rows)
    if len(vals):
        ofi_block[grp_rows, grp_cols] = vals


def _stream_mid_into(
    path: Path,
    config: dict,
    mid_block: np.ndarray,
    sym_to_idx: dict[str, int],
    day_to_idx: dict[str, int],
    row_of: np.ndarray,
) -> None:
    """Stream one LOB file and write each (asset, day)'s **closing mid** into
    *mid_block*.

    Reads only ``(asset, day, time, bid_price_1, ask_price_1)``. The closing mid
    is the mid of the day's last intraday snapshot, or of the snapshot matching
    ``config['mid_ref_time']`` (``HH:MM``) when that key is set. The same
    trailing-day carry as the OFI pass guarantees each day is whole in one chunk,
    so "last snapshot of the day" is exact at partition boundaries.
    """
    sym_col = config.get("symbol_col", _SYM_COL)
    day_col = config.get("day_col", _DAY_COL)
    time_col = config.get("time_col", _TIME_COL)
    ref_time = config.get("mid_ref_time")  # e.g. "15:00"; None → last snapshot
    n_rg = int(config.get("lob_row_groups_per_chunk", _ROW_GROUPS_PER_CHUNK))

    ddf = dd.read_parquet(
        path,
        columns=[sym_col, day_col, time_col, _BID1_COL, _ASK1_COL],
        engine="pyarrow",
        split_row_groups=n_rg,
    )
    logger.info("streaming mid from {} in {} partitions", path.name, ddf.npartitions)

    carry: pd.DataFrame | None = None
    for i in range(ddf.npartitions):
        part = ddf.partitions[i].compute()
        if carry is not None and len(carry):
            part = pd.concat([carry, part], ignore_index=True)
            carry = None
        if part.empty:
            continue
        if i < ddf.npartitions - 1:
            last_day = part[day_col].iloc[-1]
            tail = part[day_col] == last_day
            carry, part = part[tail].copy(), part[~tail]
            if part.empty:
                continue
        _reduce_mid_chunk(
            part,
            sym_col,
            day_col,
            time_col,
            ref_time,
            mid_block,
            sym_to_idx,
            day_to_idx,
            row_of,
        )

    if carry is not None and len(carry):
        _reduce_mid_chunk(
            carry,
            sym_col,
            day_col,
            time_col,
            ref_time,
            mid_block,
            sym_to_idx,
            day_to_idx,
            row_of,
        )


def _reduce_mid_chunk(
    part: pd.DataFrame,
    sym_col: str,
    day_col: str,
    time_col: str,
    ref_time: str | None,
    mid_block: np.ndarray,
    sym_to_idx: dict[str, int],
    day_to_idx: dict[str, int],
    row_of: np.ndarray,
) -> None:
    """Write the closing (or ref-time) mid for every (asset, day) in *part*."""
    a_idx = part[sym_col].map(sym_to_idx).to_numpy(dtype=np.float64)
    d_idx = part[day_col].astype(str).map(day_to_idx).to_numpy(dtype=np.float64)
    known = ~(np.isnan(a_idx) | np.isnan(d_idx))
    if not known.any():
        return

    part = part[known]
    a_idx = a_idx[known].astype(np.int64)
    d_idx = d_idx[known].astype(np.int64)
    rows = row_of[a_idx, d_idx]
    have_row = rows >= 0
    if not have_row.any():
        return
    part, rows = part[have_row], rows[have_row]

    times = part[time_col].astype(str).str[:5].to_numpy()
    bid = part[_BID1_COL].to_numpy(dtype=np.float64)
    ask = part[_ASK1_COL].to_numpy(dtype=np.float64)

    if ref_time is not None:
        keep = times == ref_time[:5]
        if not keep.any():
            return
        rows, times, bid, ask = rows[keep], times[keep], bid[keep], ask[keep]

    # sort by (row, time); the last row of each (asset, day) group is its close
    order = np.lexsort((times, rows))
    rows_s, bid_s, ask_s = rows[order], bid[order], ask[order]
    last = np.empty(len(rows_s), dtype=bool)
    last[-1] = True
    last[:-1] = rows_s[1:] != rows_s[:-1]

    r_last = rows_s[last]
    mid = 0.5 * (bid_s[last] + ask_s[last])
    # a crossed / empty book (mid <= 0, or NaN price) stays invalid → NaN
    mid = np.where(np.isfinite(mid) & (mid > 0), mid, np.nan)
    mid_block[r_last] = mid


def _build_feature_matrix(
    config: dict,
    data_dir: str | Path,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """Build the per-(asset, day) feature matrix and row arrays in RAM.

    Returns ``(feat, row_returns, row_asset, row_oos, NF, ordered_syms)`` where
    ``row_returns`` is the horizon-``h`` forward mid return per row (NaN where
    invalid). Thresholding into Down/Flat/Up classes is deferred to
    :func:`build_datasets`, which calibrates the tertile cut points on the
    training rows only.
    """
    nf = n_features(config)
    horizon = int(config["horizon"])
    sym_col = config.get("symbol_col", _SYM_COL)
    day_col = config.get("day_col", _DAY_COL)
    paths = _paths(data_dir, config)
    _require(paths)

    logger.info("building feishu-midprice features (RAM) nf={} horizon={}", nf, horizon)

    # ── Pass 1 (daily, small): per-asset OHLCV raw feats + day order ─────────
    sym_set = set(symbols)
    daily_is = _read_daily(paths["daily"], sym_col, day_col, sym_set)
    daily_oos = _read_daily(paths["daily_oos"], sym_col, day_col, sym_set)
    oos_days = set(daily_oos["date"].unique().tolist())
    daily_all = pd.concat([daily_is, daily_oos], ignore_index=True)
    del daily_is, daily_oos

    dates_map: dict[str, list[str]] = {}
    ohlcv_raw_map: dict[str, np.ndarray] = {}  # sym → (n_days, 19)

    for sym, daily_sym in daily_all.groupby(sym_col, sort=True):
        daily_sym = daily_sym.sort_values("date").reset_index(drop=True)
        sym_dates = daily_sym["date"].tolist()
        if len(sym_dates) == 0:
            continue
        dates_map[sym] = sym_dates
        try:
            ohlcv_df = compute_ohlcv_features(daily_sym)
            ohlcv_raw_map[sym] = ohlcv_df.values.astype(np.float32)
        except Exception as exc:
            logger.warning("OHLCV failed for {}: {}", sym, exc)
            ohlcv_raw_map[sym] = np.zeros((len(sym_dates), N_OHLCV), dtype=np.float32)
    del daily_all

    # ── Row layout: assets in sorted order, days in date order ──────────────
    ordered_syms = [s for s in symbols if s in dates_map]
    ranges: dict[str, tuple[int, int]] = {}
    ptr = 0
    for sym in ordered_syms:
        n_days = len(dates_map[sym])
        ranges[sym] = (ptr, ptr + n_days)
        ptr += n_days
    n_rows = ptr
    if n_rows == 0:
        raise ValueError("No (asset, day) rows to build — check data files.")

    sym_to_idx = {s: i for i, s in enumerate(ordered_syms)}
    all_days = sorted({d for sym in ordered_syms for d in dates_map[sym]})
    day_to_idx = {d: i for i, d in enumerate(all_days)}

    feat = np.zeros((n_rows, nf), dtype=np.float32)
    row_returns = np.full(n_rows, np.nan, dtype=np.float64)
    row_asset = np.empty(n_rows, dtype=np.int64)
    row_oos = np.zeros(n_rows, dtype=bool)
    ohlcv_block = np.zeros((n_rows, N_OHLCV), dtype=np.float32)
    mid_block = np.full(n_rows, np.nan, dtype=np.float64)
    row_of = np.full((len(ordered_syms), len(all_days)), -1, dtype=np.int64)

    for sym in ordered_syms:
        lo, hi = ranges[sym]
        a = sym_to_idx[sym]
        row_asset[lo:hi] = a
        ohlcv_block[lo:hi] = ohlcv_raw_map[sym]
        d_idx = np.fromiter(
            (day_to_idx[d] for d in dates_map[sym]), dtype=np.int64, count=hi - lo
        )
        row_of[a, d_idx] = np.arange(lo, hi, dtype=np.int64)
        row_oos[lo:hi] = [d in oos_days for d in dates_map[sym]]

    logger.info(
        "rows: {:,} total ({:,} in-sample, {:,} out-of-sample) over {} assets",
        n_rows,
        int((~row_oos).sum()),
        int(row_oos.sum()),
        len(ordered_syms),
    )

    # ── Pass 2 (LOB, large): streamed OFI features → feat[:, :N_OFI] ─────────
    ofi_block = feat[:, :N_OFI]
    for key in ("lob", "lob_oos"):
        _stream_ofi_into(paths[key], config, ofi_block, sym_to_idx, day_to_idx, row_of)

    for sym in ordered_syms:
        lo, hi = ranges[sym]
        normed = causal_rolling_zscore(ofi_block[lo:hi])
        ofi_block[lo:hi] = np.clip(normed, -CLIP_VAL, CLIP_VAL)

    # ── Pass 2b (LOB): streamed closing mid → mid_block → horizon returns ───
    for key in ("lob", "lob_oos"):
        _stream_mid_into(paths[key], config, mid_block, sym_to_idx, day_to_idx, row_of)

    n_have_mid = int(np.isfinite(mid_block).sum())
    logger.info(
        "closing mid resolved for {:,}/{:,} (asset, day) rows", n_have_mid, n_rows
    )
    for sym in ordered_syms:
        lo, hi = ranges[sym]
        fwd = compute_mid_forward_returns(mid_block[lo:hi], horizon)
        row_returns[lo:hi] = np.asarray(fwd, dtype=np.float64)

    # ── Pass 3: cross-sectional z-score of OHLCV per day → feat[:, N_OFI:] ──
    for d in range(len(all_days)):
        idx = row_of[:, d]
        idx = idx[idx >= 0]
        if len(idx) == 0:
            continue
        mat = ohlcv_block[idx].astype(np.float64)
        mu = np.nanmean(mat, axis=0)
        sigma = np.nanstd(mat, axis=0)
        sigma = np.where(~np.isfinite(sigma) | (sigma < 1e-8), 1.0, sigma)
        normed = ((mat - mu) / sigma).astype(np.float32)
        feat[idx, N_OFI:] = np.clip(normed, -CLIP_VAL, CLIP_VAL)

    del ohlcv_block, mid_block

    np.nan_to_num(feat, copy=False, nan=0.0, posinf=CLIP_VAL, neginf=-CLIP_VAL)
    logger.info(
        "feishu-midprice features built (in RAM): {:,} rows × {} feat", n_rows, nf
    )
    return feat, row_returns, row_asset, row_oos, nf, ordered_syms


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
    symbols: list[str],
) -> tuple[LOBDataset, LOBDataset, LOBDataset, dict]:
    """Return ``(train_ds, val_ds, test_ds, meta)`` over the in-RAM feature matrix.

    Windows are computed per asset (causal label, no straddling) and assigned by
    the period of their **label day**, with the in-sample train/val cut made
    chronologically per asset (default 80/20, override with
    ``config['train_frac']``). Identical windowing to :mod:`stocks.feishu.build`
    — only the row labels differ (horizon-``h`` mid trend).

    Class boundaries
    ----------------
    With ``label_mode="quantile"`` (default) the Down/Flat/Up cut points are the
    ``config['class_quantiles']`` tertiles (default 1/3, 2/3) of the **training**
    label-row returns, giving ~33/33/33 balanced classes with no val/test
    leakage. With ``label_mode="fixed"`` a symmetric band ``(-alpha, +alpha)`` is
    used instead. The same two scalars are then applied to every split.
    """
    T = config["T_past"]
    train_frac = float(config.get("train_frac", _INSAMPLE_TRAIN_FRAC))
    feat, row_returns, row_asset, row_oos, nf, ordered_syms = _build_feature_matrix(
        config, data_dir, symbols
    )

    valid_row = np.isfinite(row_returns)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for lo, hi in _asset_ranges(row_asset):
        starts = np.arange(lo, max(lo, hi - T), dtype=np.int64)
        if len(starts) == 0:
            continue
        starts = starts[valid_row[starts + T]]
        oos = row_oos[starts + T]
        test_parts.append(starts[oos])

        insample = starts[~oos]
        n_train = int(train_frac * len(insample))
        train_parts.append(insample[:n_train])
        val_parts.append(insample[n_train:])

    empty = np.empty(0, dtype=np.int64)
    train_arr = np.concatenate(train_parts) if train_parts else empty
    val_arr = np.concatenate(val_parts) if val_parts else empty
    test_arr = np.concatenate(test_parts) if test_parts else empty

    # ── calibrate class boundaries on TRAIN label rows only (no leakage) ────
    label_mode = config.get("label_mode", "quantile")
    if label_mode == "fixed":
        alpha = float(config["alpha"])
        lo_thr, hi_thr = -alpha, alpha
    elif label_mode == "quantile":
        quantiles = tuple(config.get("class_quantiles", (1.0 / 3.0, 2.0 / 3.0)))
        train_returns = row_returns[train_arr + T] if len(train_arr) else row_returns
        lo_thr, hi_thr = compute_class_thresholds(train_returns, quantiles)
    else:
        raise ValueError(f"unknown label_mode: {label_mode!r}")

    row_labels = assign_labels_quantile(row_returns, lo_thr, hi_thr)
    logger.info(
        "label_mode={} thresholds lo={:.5f} hi={:.5f}", label_mode, lo_thr, hi_thr
    )

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
        "train_frac": train_frac,
        "horizon": int(config["horizon"]),
        "label_mode": label_mode,
        "class_thresholds": {"lo": lo_thr, "hi": hi_thr},
        "class_balance": _balance(train_arr),
        "val_class_balance": _balance(val_arr),
        "test_class_balance": _balance(test_arr),
        "n_features": nf,
        "n_rows": len(row_asset),
        "n_assets": len(ordered_syms),
        "symbols": ordered_syms,
    }
    logger.info(
        "windows — train:{} val:{} (in-sample {:.0%}/{:.0%}) test:{} (out-of-sample) "
        "n_assets:{} horizon:{}",
        meta["counts"]["train"],
        meta["counts"]["val"],
        train_frac,
        1 - train_frac,
        meta["counts"]["test"],
        meta["n_assets"],
        meta["horizon"],
    )
    logger.info(
        "train balance  down={down:.1%} stat={stationary:.1%} up={up:.1%}",
        **meta["class_balance"],
    )

    train_ds = LOBDataset(feat, train_arr, row_labels, T, row_asset=row_asset)
    val_ds = LOBDataset(feat, val_arr, row_labels, T, row_asset=row_asset)
    test_ds = LOBDataset(feat, test_arr, row_labels, T, row_asset=row_asset)
    return train_ds, val_ds, test_ds, meta
