"""Robust LOBSTER multi-asset data engineering pipeline for the Penny workspace.
Dynamically handles level selection (_1, _5, _10) and matches spatial 4D model expectations.
"""

from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger

# Workspace absolute import
from src.stocks.feishu.dataset_lobster import LOBSTERDataset

_TRAIN_FRAC = 0.70
_VAL_CUM = 0.85
_CLIP_VAL = 5.0
_START_SEC = 34200.0
_END_SEC = 57600.0

_FILE_REGEX = re.compile(r"^([A-Za-z0-9\-]+)_(\d{4}-\d{2}-\d{2})_\d+_\d+_(message|orderbook)_(\d+)")


def discover_symbols(data_dir: str | Path, config: dict | None = None) -> list[str]:
    """Scan data directory and isolate unique clean asset tickers."""
    root = Path(data_dir).resolve()
    symbols = set()
    
    if not root.exists():
        logger.error(f"Target data directory does not exist: {root}")
        return []
        
    for f in root.glob("*_message_*"):
        match = _FILE_REGEX.match(f.name)
        if match:
            symbols.add(match.group(1))
            
    if not symbols:
        logger.warning(f"No valid LOBSTER files captured inside: {root}")
    return sorted(list(symbols))


def _compute_multi_level_ofi(ob_matrix: np.ndarray, n_levels: int) -> np.ndarray:
    """Multi-level Order Flow Imbalance calculation logic."""
    n_ticks = ob_matrix.shape[0]
    ofi_matrix = np.zeros((n_ticks, n_levels), dtype=np.float32)
    
    for lvl in range(n_levels):
        idx = lvl * 4
        ask_p, ask_v = ob_matrix[:, idx], ob_matrix[:, idx + 1]
        bid_p, bid_v = ob_matrix[:, idx + 2], ob_matrix[:, idx + 3]
        
        df_bid_p = np.diff(bid_p, prepend=bid_p[0])
        df_bid_v = np.diff(bid_v, prepend=bid_v[0])
        df_ask_p = np.diff(ask_p, prepend=ask_p[0])
        df_ask_v = np.diff(ask_v, prepend=ask_v[0])
        
        ofi_bid = np.where(df_bid_p > 0, bid_v, np.where(df_bid_p == 0, df_bid_v, 0))
        ofi_ask = np.where(df_ask_p < 0, ask_v, np.where(df_ask_p == 0, df_ask_v, 0))
        ofi_matrix[:, lvl] = ofi_bid - ofi_ask
        
    return ofi_matrix


def _load_and_pad_lobster(ob_path: Path, target_levels: int) -> np.ndarray:
    """Adaptive padding layout to extract exact raw level structures."""
    if ob_path.suffix == ".parquet":
        df = pd.read_parquet(ob_path)
    else:
        df = pd.read_csv(ob_path, header=None)
        
    raw_arr = df.values.astype(np.float32)
    available_cols = raw_arr.shape[1]
    target_cols = target_levels * 4
    
    if available_cols >= target_cols:
        return raw_arr[:, :target_cols]
    
    padded = np.zeros((raw_arr.shape[0], target_cols), dtype=np.float32)
    padded[:, :available_cols] = raw_arr
    for i in range(available_cols, target_cols, 4):
        padded[:, i] = raw_arr[:, 0]
        padded[:, i + 2] = raw_arr[:, 2]
    return padded


def _snap_to_time_slots(timestamps: np.ndarray, raw_feats: np.ndarray, n_slots: int) -> np.ndarray:
    """Resample continuous order book events into uniform sequential time grids."""
    slot_edges = np.linspace(_START_SEC, _END_SEC, n_slots + 1)
    feat_dim = raw_feats.shape[1]
    daily_grid = np.zeros((n_slots, feat_dim), dtype=np.float32)
    
    slot_indices = np.searchsorted(slot_edges, timestamps, side='right') - 1
    slot_indices = np.clip(slot_indices, 0, n_slots - 1)
    
    current_feat = raw_feats[0]
    tick_ptr = 0
    n_ticks = len(timestamps)
    
    for s in range(n_slots):
        while tick_ptr < n_ticks and slot_indices[tick_ptr] == s:
            current_feat = raw_feats[tick_ptr]
            tick_ptr += 1
        daily_grid[s] = current_feat
        
    return daily_grid


def _build_feature_matrix(config: dict, data_dir: str | Path, symbols: list[str]):
    """Gather cross-sectional sequences across assets grouped carefully by date."""
    root = Path(data_dir).resolve()
    mode = config.get("feature_mode", "ofi")
    target_levels = config.get("n_lob_levels", 10)
    n_slots = config.get("n_slots", 78)
    alpha = config.get("alpha", 0.015)
    
    dates_map, feat_map, labels_map = {}, {}, {}
    
    for sym in symbols:
        all_msg_files = [f for f in root.glob(f"{sym}_*_message_*") if _FILE_REGEX.match(f.name)]
        date_groups = defaultdict(list)
        
        for f in all_msg_files:
            match = _FILE_REGEX.match(f.name)
            date_str = match.group(2)
            lvl_val = int(match.group(4))
            date_groups[date_str].append((lvl_val, f))
            
        sym_dates, sym_daily_blocks, sym_closes = [], [], []
        
        for date_str in sorted(date_groups.keys()):
            avail_files = date_groups[date_str]
            chosen_msg_f = None
            for lvl, f in avail_files:
                if lvl == target_levels:
                    chosen_msg_f = f
                    break
            if chosen_msg_f is None:
                avail_files.sort(key=lambda x: x[0], reverse=True)
                chosen_msg_f = avail_files[0][1]
                
            match = _FILE_REGEX.match(chosen_msg_f.name)
            current_level = match.group(4)
            ob_name = chosen_msg_f.name.replace(f"_message_{current_level}", f"_orderbook_{current_level}")
            ob_f = root / ob_name
            if not ob_f.exists(): continue
                
            try:
                if chosen_msg_f.suffix == ".parquet":
                    msg_df = pd.read_parquet(chosen_msg_f)
                else:
                    msg_df = pd.read_csv(chosen_msg_f, header=None, usecols=[0])
                    
                timestamps = msg_df.values.flatten().astype(np.float64)
                matrix = _load_and_pad_lobster(ob_f, target_levels)
                
                raw_feats = _compute_multi_level_ofi(matrix, target_levels) if mode == "ofi" else matrix
                day_grid = _snap_to_time_slots(timestamps, raw_feats, n_slots)
                mid_close = (matrix[-1, 0] + matrix[-1, 2]) / 2.0
                
                sym_dates.append(date_str)
                sym_daily_blocks.append(day_grid)
                sym_closes.append(mid_close)
            except Exception as e:
                logger.warning(f"Error skipping day execution for {sym} on {date_str}: {e}")
                
        if len(sym_dates) < 2: continue
            
        closes = np.array(sym_closes, dtype=np.float32)
        fwd_returns = (closes[1:] - closes[:-1]) / closes[:-1]
        
        # Vectorized alignment mapping daily classification labels to every internal slot
        labels = np.ones(len(fwd_returns), dtype=np.int64)
        labels[fwd_returns < -alpha] = 0
        labels[fwd_returns > alpha] = 2
        
        valid_days = len(fwd_returns)
        if valid_days == 0: continue
            
        sym_feats_stacked = np.concatenate(sym_daily_blocks[:valid_days], axis=0)
        row_labels_stacked = np.repeat(labels, n_slots)
        
        dates_map[sym] = sym_dates[:valid_days]
        feat_map[sym] = sym_feats_stacked
        labels_map[sym] = row_labels_stacked

    ordered_syms = [s for s in symbols if s in dates_map]
    if not ordered_syms:
        raise ValueError("No structural samples left. Check parameters or directory integrity.")
        
    total_rows = sum(feat_map[s].shape[0] for s in ordered_syms)
    nf = feat_map[ordered_syms[0]].shape[1]
    
    feat = np.zeros((total_rows, nf), dtype=np.float32)
    row_labels = np.full(total_rows, -1, dtype=np.int64)
    row_asset = np.empty(total_rows, dtype=np.int64)
    
    ptr = 0
    sym_to_idx = {s: i for i, s in enumerate(ordered_syms)}
    for sym in ordered_syms:
        n_elements = feat_map[sym].shape[0]
        hi = ptr + n_elements
        feat[ptr:hi] = feat_map[sym]
        row_labels[ptr:hi] = labels_map[sym]
        row_asset[ptr:hi] = sym_to_idx[sym]
        ptr = hi
        
    mu, sigma = np.mean(feat, axis=0), np.std(feat, axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    feat = np.clip((feat - mu) / sigma, -_CLIP_VAL, _CLIP_VAL)
    
    return feat, row_labels, row_asset, nf, ordered_syms


def build_datasets_multi(config: dict, data_dir: str | Path, symbols: list[str]):
    """Primary pipeline compiler returning structured train/val/test data splits."""
    T = config["T_past"]
    n_slots = config.get("n_slots", 78)
    feat, row_labels, row_asset, nf, ordered_syms = _build_feature_matrix(config, data_dir, symbols)
    
    train_starts, val_starts, test_starts = [], [], []
    
    # Track boundaries cleanly to separate sequences based on corporate assets
    boundaries = np.flatnonzero(np.diff(row_asset)) + 1
    edges = [0, *boundaries.tolist(), len(row_asset)]
    asset_ranges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    
    for lo, hi in asset_ranges:
        # Prevent rolling window spillover into adjacent assets
        asset_starts = [s for s in range(lo, hi - T) if row_labels[s + T - 1] >= 0]
        if not asset_starts: continue
        
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
        if len(starts) == 0: return {"down": 0.0, "stationary": 0.0, "up": 0.0}
        lbl = row_labels[starts + T - 1]
        c = np.bincount(lbl, minlength=3) / len(lbl)
        return {"down": float(c[0]), "stationary": float(c[1]), "up": float(c[2])}

    meta = {
        "counts": {"train": len(train_arr), "val": len(val_arr), "test": len(test_arr)},
        "class_balance": _balance(train_arr),
        "n_features": nf,
        "n_rows": len(row_asset),
        "n_assets": len(ordered_syms),
        "symbols": ordered_syms,
    }
    
    logger.info("LOBSTER Dataset blocks mapped to continuous 4D space dimensions successfully.")
    
    train_ds = LOBSTERDataset(feat, train_arr, row_labels, T, row_asset=row_asset)
    val_ds = LOBSTERDataset(feat, val_arr, row_labels, T, row_asset=row_asset)
    test_ds = LOBSTERDataset(feat, test_arr, row_labels, T, row_asset=row_asset)
    
    return train_ds, val_ds, test_ds, meta