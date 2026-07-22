"""In-RAM feature builder + dataset factory for Feishu A-share equity data.

Train / eval protocol
---------------------
There is **no internal split**.  The model trains on *every* in-sample window
and is then evaluated once on the held-out out-of-sample period:

    train : in-sample file          (D001 … D484)
    test  : out-of-sample file      (D485 … D726)

Both periods are built into a single contiguous per-asset feature matrix so
that windows and rolling statistics carry across the boundary — the first
``T_past`` out-of-sample days are evaluable because their look-back window
reaches back into the in-sample tail.  A window is assigned to *test* iff its
**label day** falls in the out-of-sample period, so nothing that a training
label depends on is ever an out-of-sample prediction target.

Every normalisation is point-in-time: the OFI rolling z-score is causal
(past days only) and the OHLCV z-score is cross-sectional within a single day.
Computing them over the concatenated series therefore leaks no future
information into the training windows.

Memory model
------------
The whole feature matrix is built **in RAM** every run — no disk cache. We hold
a single per-(asset, day) feature array of shape ``(N_rows, NF)`` (~1.7 GB for
in-sample + out-of-sample: ~1.6M day-rows × 259 float32 features) and slide
T-day windows lazily at training time (see
:class:`~stocks.feishu.dataset.LOBDataset`).  Materialising every window would
instead duplicate each day ~T times (tens of GB).

The LOB parquet files are far too large to load whole (~3 GB on disk, ~10 GB as
a DataFrame), so they are streamed with **dask**: ``dd.read_parquet`` splits the
file into row-group partitions that are read (in parallel) and reduced into the
feature matrix one at a time, bounding peak RAM to a single partition.  Both LOB
files are sorted by trade day, so a chunk boundary can only ever split a day —
the trailing day of each partition is buffered and prepended to the next so no
(asset, day) group is ever computed from partial data.

Building fresh each run (rather than reusing a memmap cache) guarantees the
features always reflect the current normalisation code — a stale cache built
before a normalisation fix is the classic source of NaN losses.

Expected data_dir contents (flat multi-asset parquet files)::

    data_dir/
      lob_data_in_sample.parquet                      # 5-min LOB snapshots
      daily_data_in_sample.parquet                    # daily OHLCV
      lob_data_release_stage_out_of_sample.parquet
      daily_data_release_stage_out_of_sample.parquet

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

Public API
----------
- discover_symbols(data_dir, config) → sorted symbols across both periods
- build_datasets(config, data_dir, symbols) → (train_ds, test_ds, meta)
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
from stocks.feishu.labels import assign_labels, compute_forward_returns

_LOB_FILE = "lob_data_in_sample.parquet"
_DAILY_FILE = "daily_data_in_sample.parquet"
_LOB_FILE_OOS = "lob_data_release_stage_out_of_sample.parquet"
_DAILY_FILE_OOS = "daily_data_release_stage_out_of_sample.parquet"
_SYM_COL = "asset_id"
_DAY_COL = "trade_day_id"
_TIME_COL = "time"

# Row groups per streamed LOB partition. Each row group is ~200k rows, so 8
# keeps a partition around 1.6M rows (~0.5 GB) — small enough to hold twice
# (partition + carry-over buffer) on any training node.
_ROW_GROUPS_PER_CHUNK = 8


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


def _lob_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 1-indexed LOB columns (bid/ask_price/volume_1..10) to 0-indexed."""
    rename = {}
    for i in range(1, 11):
        for prefix in ("bid_price", "ask_price", "bid_volume", "ask_volume"):
            src = f"{prefix}_{i}"
            if src in df.columns:
                rename[src] = f"{prefix}_{i - 1}"
    return df.rename(columns=rename)


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


def discover_symbols(data_dir: str | Path, config: dict | None = None) -> list[str]:
    """Return the sorted union of asset ids across both daily files.

    The universe spans both periods: assets that only trade in-sample still
    contribute training windows, and assets that only appear out-of-sample are
    still evaluated (with an untrained asset embedding, for models that use one).
    """
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

    Reads the file as dask row-group partitions and reduces each into
    ``ofi_block`` in place, so peak memory is one partition rather than the
    whole file.  The file is trade-day sorted, so only the final day of a
    partition can be incomplete; those rows are carried into the next partition.

    Args:
        ofi_block:  ``(N_rows, N_OFI)`` float32 destination, indexed by global row.
        sym_to_idx: symbol → asset index.
        day_to_idx: day string → global day index.
        row_of:     ``(n_assets, n_days)`` int64 lookup, -1 where the
                    (asset, day) pair has no daily row.
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
    logger.info("streaming {} in {} dask partitions", path.name, ddf.npartitions)

    carry: pd.DataFrame | None = None
    for i in range(ddf.npartitions):
        part = ddf.partitions[i].compute()
        if carry is not None and len(carry):
            part = pd.concat([carry, part], ignore_index=True)
            carry = None
        if part.empty:
            continue

        # hold back the last (possibly truncated) trade day for the next chunk
        if i < ddf.npartitions - 1:
            last_day = part[day_col].iloc[-1]
            tail = part[day_col] == last_day
            carry, part = part[tail].copy(), part[~tail]
            if part.empty:
                continue

        _reduce_lob_chunk(
            part, sym_col, day_col, time_col, ofi_block, sym_to_idx, day_to_idx, row_of
        )

    if carry is not None and len(carry):
        _reduce_lob_chunk(
            carry, sym_col, day_col, time_col, ofi_block, sym_to_idx, day_to_idx, row_of
        )


def _reduce_lob_chunk(
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
    rows = row_of[a_idx, d_idx]  # global feature row per tick
    have_row = rows >= 0
    if not have_row.any():
        return
    part, rows = part[have_row], rows[have_row]

    # sort ticks into (asset, day, time) order — stable, so ties keep file order
    times = part[time_col].astype(str).to_numpy()
    order = np.lexsort((times, rows))
    part = _lob_rename(part.iloc[order].reset_index(drop=True))
    rows = rows[order]

    # `rows` is constant within an (asset, day) group and changes at every
    # boundary, so it doubles as the group id for the vectorised OFI pass.
    grp_rows, grp_cols, vals = compute_ofi_slots_chunk(part, rows)
    if len(vals):
        ofi_block[grp_rows, grp_cols] = vals


def _build_feature_matrix(
    config: dict,
    data_dir: str | Path,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """Build the per-(asset, day) feature matrix and row arrays in RAM.

    In-sample and out-of-sample days are concatenated per asset in date order,
    so rolling statistics and T-day windows carry across the period boundary.

    Returns:
        ``(feat, row_labels, row_asset, row_oos, NF, ordered_syms)`` where
          feat         : ``(N_rows, NF)`` float32 array (in RAM).
          row_labels   : ``(N_rows,)`` int64 causal labels (-1 = invalid).
          row_asset    : ``(N_rows,)`` int64 asset index per row (contiguous).
          row_oos      : ``(N_rows,)`` bool, True for out-of-sample days.
          NF           : feature count.
          ordered_syms : list[str] of symbols in row order.
    """
    nf = n_features(config)
    alpha = config["alpha"]
    sym_col = config.get("symbol_col", _SYM_COL)
    day_col = config.get("day_col", _DAY_COL)
    paths = _paths(data_dir, config)
    _require(paths)

    logger.info("building feishu features (RAM) mode=OFI nf={}", nf)

    # ── Pass 1 (daily, small): per-asset OHLCV raw feats, labels, day order ──
    sym_set = set(symbols)
    daily_is = _read_daily(paths["daily"], sym_col, day_col, sym_set)
    daily_oos = _read_daily(paths["daily_oos"], sym_col, day_col, sym_set)
    oos_days = set(daily_oos["date"].unique().tolist())
    daily_all = pd.concat([daily_is, daily_oos], ignore_index=True)
    del daily_is, daily_oos

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
    row_labels = np.full(n_rows, -1, dtype=np.int64)
    row_asset = np.empty(n_rows, dtype=np.int64)
    row_oos = np.zeros(n_rows, dtype=bool)
    ohlcv_block = np.zeros((n_rows, N_OHLCV), dtype=np.float32)
    row_of = np.full((len(ordered_syms), len(all_days)), -1, dtype=np.int64)

    for sym in ordered_syms:
        lo, hi = ranges[sym]
        a = sym_to_idx[sym]
        row_asset[lo:hi] = a
        row_labels[lo:hi] = labels_map[sym]
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

    # ── Pass 2 (LOB, large): streamed OFI features → feat[:, :N_OFI] ────────
    # Raw slot-OFI is accumulated for both periods first, then z-scored per
    # asset over the concatenated series (causal, so out-of-sample days never
    # inform in-sample ones).
    ofi_block = feat[:, :N_OFI]
    for key in ("lob", "lob_oos"):
        _stream_ofi_into(paths[key], config, ofi_block, sym_to_idx, day_to_idx, row_of)

    for sym in ordered_syms:
        lo, hi = ranges[sym]
        normed = causal_rolling_zscore(ofi_block[lo:hi])
        ofi_block[lo:hi] = np.clip(normed, -CLIP_VAL, CLIP_VAL)

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

    del ohlcv_block

    # final safety net: no non-finite values ever reach the model
    np.nan_to_num(feat, copy=False, nan=0.0, posinf=CLIP_VAL, neginf=-CLIP_VAL)
    logger.info("feishu features built (in RAM): {:,} rows × {} feat", n_rows, nf)
    return feat, row_labels, row_asset, row_oos, nf, ordered_syms


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
) -> tuple[LOBDataset, LOBDataset, dict]:
    """Return ``(train_ds, test_ds, meta)`` over the in-RAM feature matrix.

    Windows are computed per asset (causal label, no straddling) and assigned by
    the period of their **label day**: in-sample → train, out-of-sample → test.
    There is no validation split; the model trains on the whole in-sample set.

    Each ``__getitem__`` returns ``{"x":…, "label":…, "asset": int}``, where
    ``asset`` indexes ``meta["symbols"]`` — models that condition on asset
    identity (e.g. JointDiffCFG) can use it, others ignore it.
    """
    T = config["T_past"]
    feat, row_labels, row_asset, row_oos, nf, ordered_syms = _build_feature_matrix(
        config, data_dir, symbols
    )

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for lo, hi in _asset_ranges(row_asset):
        starts = np.arange(lo, max(lo, hi - T), dtype=np.int64)
        if len(starts) == 0:
            continue
        valid = row_labels[starts + T] >= 0
        oos = row_oos[starts + T]
        train_parts.append(starts[valid & ~oos])
        test_parts.append(starts[valid & oos])

    empty = np.empty(0, dtype=np.int64)
    train_arr = np.concatenate(train_parts) if train_parts else empty
    test_arr = np.concatenate(test_parts) if test_parts else empty

    def _balance(starts: np.ndarray) -> dict:
        if len(starts) == 0:
            return {"down": 0.0, "stationary": 0.0, "up": 0.0}
        lbl = row_labels[starts + T]
        c = np.bincount(lbl, minlength=3) / len(lbl)
        return {"down": float(c[0]), "stationary": float(c[1]), "up": float(c[2])}

    meta = {
        "counts": {"train": len(train_arr), "test": len(test_arr)},
        "class_balance": _balance(train_arr),
        "test_class_balance": _balance(test_arr),
        "n_features": nf,
        "n_rows": len(row_asset),
        "n_assets": len(ordered_syms),
        "symbols": ordered_syms,
    }
    logger.info(
        "windows — train(in-sample):{} test(out-of-sample):{} n_assets:{}",
        meta["counts"]["train"],
        meta["counts"]["test"],
        meta["n_assets"],
    )

    train_ds = LOBDataset(feat, train_arr, row_labels, T, row_asset=row_asset)
    test_ds = LOBDataset(feat, test_arr, row_labels, T, row_asset=row_asset)
    return train_ds, test_ds, meta
