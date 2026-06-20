"""HDF5 cache builder for Feishu A-share equity data.

Expected directory layout::

    data_dir/
      {SYMBOL}/
        intraday.parquet   # tick LOB: date, time, bid/ask price_{0..9}, bid/ask volume_{0..9}
        daily.parquet      # daily OHLCV: date, open, high, low, close, volume, vwap_0930_0935

Public API
----------
- discover_symbols(data_dir)  → sorted list of symbol names
- compute_date_splits(data_dir, symbols, train_frac, val_frac) → (train_dates, val_dates, test_dates)
- build_hdf5(config, split, split_dates, data_dir, cache_dir, symbols) → Path
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stocks.feishu.dataset import WindowWriter
from stocks.feishu.features import (
    CLIP_VAL,
    N_LEVELS,
    causal_rolling_zscore,
    compute_ofi_tick,
    compute_ohlcv_features,
    extract_lob_day,
    n_features,
    snap_to_slots,
)
from stocks.feishu.labels import assign_labels, compute_forward_returns


def discover_symbols(data_dir: str | Path) -> list[str]:
    """Return sorted list of symbols that have both intraday.parquet and daily.parquet."""
    root = Path(data_dir)
    symbols = []
    for d in sorted(root.iterdir()):
        if (
            d.is_dir()
            and (d / "intraday.parquet").exists()
            and (d / "daily.parquet").exists()
        ):
            symbols.append(d.name)
    if not symbols:
        raise FileNotFoundError(f"No symbols found under {root}")
    return symbols


def compute_date_splits(
    data_dir: str | Path,
    symbols: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[set[str], set[str], set[str]]:
    """Chronological 70/15/15 split by unique trading date across all symbols.

    Returns:
        (train_dates, val_dates, test_dates) — each a ``set[str]`` of ``YYYY-MM-DD`` strings.
    """
    all_dates: set[str] = set()
    root = Path(data_dir)
    for sym in symbols:
        daily = pd.read_parquet(root / sym / "daily.parquet", columns=["date"])
        all_dates.update(daily["date"].astype(str).unique())
    sorted_dates = sorted(all_dates)
    n = len(sorted_dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_dates = set(sorted_dates[:n_train])
    val_dates = set(sorted_dates[n_train : n_train + n_val])
    test_dates = set(sorted_dates[n_train + n_val :])
    logger.info(
        "Date splits: train={} val={} test={} (total={})",
        len(train_dates),
        len(val_dates),
        len(test_dates),
        n,
    )
    return train_dates, val_dates, test_dates


def _cache_key(config: dict, split: str) -> str:
    """Stable short hash to version the HDF5 cache by config parameters."""
    sig = {
        k: config[k]
        for k in ("feature_mode", "n_lob_levels", "T_past", "alpha")
        if k in config
    }
    sig["split"] = split
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:8]


def build_hdf5(
    config: dict,
    split: str,
    split_dates: set[str],
    data_dir: str | Path,
    cache_dir: str | Path,
    symbols: list[str],
) -> Path:
    """Build (or return cached) HDF5 file for one data split.

    Pipeline
    --------
    1. Per-symbol: load intraday/daily data, compute LOB/OFI features,
       apply causal rolling z-score.
    2. Cross-sectional z-score of OHLCV across all assets per day.
    3. Concatenate, clip to ±5, slide T-past windows, write to HDF5.

    Args:
        config:      Full config dict (must contain feature_mode, T_past, alpha, n_lob_levels).
        split:       ``"train"``, ``"val"``, or ``"test"``.
        split_dates: Set of date strings (YYYY-MM-DD) belonging to this split.
        data_dir:    Root data directory.
        cache_dir:   Directory where HDF5 caches are stored.
        symbols:     Symbol names (subdirectories of data_dir).

    Returns:
        Path to the created/existing HDF5 file.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config, split)
    path = out_dir / f"{split}_{key}.h5"

    if path.exists():
        logger.info("Cache hit: {}", path)
        return path

    mode = config.get("feature_mode", "ofi")
    T = config["T_past"]
    alpha = config["alpha"]
    n_levels = config.get("n_lob_levels", N_LEVELS)
    nf = n_features(config)
    root = Path(data_dir)

    logger.info(
        "Building HDF5 {} split={} mode={} T={} nf={}", path.name, split, mode, T, nf
    )

    # ── Pass 1: per-asset feature extraction ─────────────────────────────────
    lob_feats: dict[str, dict[str, np.ndarray]] = {}
    ohlcv_feats: dict[str, dict[str, np.ndarray]] = {}
    labels_map: dict[str, dict[str, int]] = {}
    sorted_dates_map: dict[str, list[str]] = {}

    for sym in symbols:
        intra = pd.read_parquet(root / sym / "intraday.parquet")
        daily = pd.read_parquet(root / sym / "daily.parquet")
        daily["date"] = daily["date"].astype(str)
        intra["date"] = intra["date"].astype(str)
        daily = daily.sort_values("date").reset_index(drop=True)

        sym_dates = sorted(daily["date"].unique())
        sorted_dates_map[sym] = sym_dates

        # LOB / OFI features per day
        lob_rows: list[np.ndarray] = []
        for d in sym_dates:
            day_df = intra[intra["date"] == d].reset_index(drop=True)
            if len(day_df) == 0:
                lob_rows.append(
                    np.zeros(
                        (N_LEVELS * 24) if mode == "ofi" else 4 * n_levels,
                        dtype=np.float32,
                    )
                )
                continue
            if mode == "ofi":
                ofi_df = compute_ofi_tick(day_df)
                vec = snap_to_slots(ofi_df, day_df).reshape(-1)  # (240,)
            else:
                vec = extract_lob_day(day_df, n_levels)  # (4n,)
            lob_rows.append(vec)

        lob_mat = np.stack(lob_rows, axis=0)  # (n_days, n_lob_feat)
        lob_norm = causal_rolling_zscore(lob_mat)
        lob_feats[sym] = {d: lob_norm[i] for i, d in enumerate(sym_dates)}

        # OHLCV features per day
        try:
            ohlcv_df = compute_ohlcv_features(daily)
            ohlcv_df["date"] = daily["date"].values
            for _, row in ohlcv_df.iterrows():
                ohlcv_feats.setdefault(sym, {})[str(row["date"])] = row.drop(
                    "date"
                ).values.astype(np.float32)
        except Exception as exc:
            logger.warning("OHLCV failed for {}: {}", sym, exc)

        # Labels (forward returns on full timeline)
        fwd = compute_forward_returns(daily)
        lbl = assign_labels(fwd, alpha)
        labels_map[sym] = {d: int(lbl[i]) for i, d in enumerate(sym_dates)}

    # ── Pass 2: cross-sectional z-score of OHLCV per day ─────────────────────
    all_dates_sorted = sorted({d for sym in symbols for d in sorted_dates_map[sym]})
    ohlcv_norm: dict[str, dict[str, np.ndarray]] = {sym: {} for sym in symbols}

    for d in all_dates_sorted:
        vecs = []
        sym_on_day = []
        for sym in symbols:
            if sym in ohlcv_feats and d in ohlcv_feats[sym]:
                vecs.append(ohlcv_feats[sym][d])
                sym_on_day.append(sym)
        if not vecs:
            continue
        mat = np.stack(vecs, axis=0).astype(np.float64)  # (k, 19)
        mu = mat.mean(axis=0)
        sigma = mat.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        normed = ((mat - mu) / sigma).astype(np.float32)
        for i, sym in enumerate(sym_on_day):
            ohlcv_norm[sym][d] = normed[i]

    # ── Pass 3: build windows and write HDF5 ─────────────────────────────────
    writer = WindowWriter(str(path), T=T, NF=nf)
    total_windows = 0

    for asset_idx, sym in enumerate(symbols):
        sym_dates = sorted_dates_map[sym]
        windows_x: list[np.ndarray] = []
        windows_lbl: list[int] = []

        for i in range(T, len(sym_dates)):
            window_dates = sym_dates[i - T : i]
            last_date = window_dates[-1]

            if not all(d in split_dates for d in window_dates):
                continue
            lbl_val = labels_map[sym].get(last_date, -1)
            if lbl_val == -1:
                continue

            # (T, nf) feature matrix
            rows = []
            ok = True
            for d in window_dates:
                lob_v = lob_feats[sym].get(d)
                ohlcv_v = ohlcv_norm[sym].get(d)
                if lob_v is None or ohlcv_v is None:
                    ok = False
                    break
                feat = np.concatenate([lob_v, ohlcv_v], axis=0).astype(np.float32)
                rows.append(feat)
            if not ok:
                continue

            x = np.stack(rows, axis=0)  # (T, nf)
            x = np.clip(x, -CLIP_VAL, CLIP_VAL)
            windows_x.append(x)
            windows_lbl.append(lbl_val)

        if windows_x:
            X = np.stack(windows_x, axis=0)[:, np.newaxis, :, :]  # (N, 1, T, NF)
            Y = np.array(windows_lbl, dtype=np.int64)
            writer.write(X, Y, asset_idx)
            total_windows += len(X)

    writer.close()
    logger.info("Built {} | windows={}", path.name, total_windows)
    return path
