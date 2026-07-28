# Windows, splits & normalisation

How the normalised feature series (`crypto/loader.py`) becomes train/val/test
windows (`crypto/dataset.py`).

## Causal rolling normalisation — `crypto/loader.py`

Features are z-scored with a **causal trailing rolling window** of `norm_window`
bins (default 2000):

```
mean_t, std_t computed over rows [max(0, t - norm_window + 1), t]     # past + present only
feat_t := (raw_t - mean_t) / std_t
```

- **No lookahead.** Bin `t`'s statistics never use any future bin. This fixes the
  intra-day leakage of naive per-day scaling (where an early-morning bin would be
  normalised using the whole day, including later bins).
- The window **spans day boundaries** — normalisation is global over the full
  chronological series, even though *feature extraction* is per-day.
- Degenerate windows (fewer than 2 bins, or a near-constant feature with
  `std < 1e-8`) fall back to `std = 1` to avoid divide-by-zero blow-ups.
- Computed in `O(N·F)` via cumulative sums, so a large window is free.

The normalised matrix is written once to a numpy **memmap** and reused. The cache key
encodes `symbol`, level count, feature mode and `norm_window`
(`{SYMBOL}_n{n}_{mode}_roll{window}_{tag}`), so changing any of them rebuilds the
cache rather than silently reusing a stale one. The memmap is paged lazily, so RAM
scales with `batch_size × T_past × F`, not dataset size.

## Chronological split — `crypto/dataset.py`

The split is a **contiguous fraction of the time series**, not a random shuffle:

```
train : [0,            train_end)          train_frac              (default 0.70)
val   : [train_end,    val_end)            val_frac                (default 0.15)
test  : [val_end,      N)                  remainder               (default 0.15)
```

Both `alpha` (label calibration) and the rolling normaliser are frozen from the
training region, so val/test never inform preprocessing.

## Sliding windows

A window is `T_past` consecutive bins with stride `stride`:

- **Input** `x` — shape `(1, T_past, F)` (a single-channel image; the leading `1` is
  the channel axis every model expects).
- **Label** — the trend label at the window's **last** bin.
- **Gap skipping** — if the largest timestamp gap inside a window exceeds
  `10 × median gap`, the window is dropped. This throws out windows that straddle a
  day boundary or a data outage, so no window mixes discontinuous market regimes.
- **Invalid labels** — windows whose label bin is `-1` (the first/last `k` bins) are
  dropped.

The per-split window counts and the realised `down / flat / up` balance are logged at
the start of every run and stored in the run metadata.

## Key config knobs

| Key | Meaning | Typical |
|-----|---------|---------|
| `T_past`      | window length (bins)             | 60 |
| `stride`      | sliding-window stride            | 1 |
| `label_k`     | trend horizon (bins)             | 10 / 20 / 50 / 100 |
| `norm_window` | causal rolling z-score window    | 2000 |
| `train_frac` / `val_frac` | chronological split | 0.70 / 0.15 |
| `n_lob_levels`| LOB levels kept                  | 10 (Binance) / 20 (Coinbase) |
| `feature_mode`| `ofi` or `lob`                   | `ofi` |
| `cache_dir`   | where the `.npy` memmap lives     | `data/cache/…` |
