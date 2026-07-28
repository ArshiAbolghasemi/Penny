# Feishu data

Chinese A-share equity LOB + daily data, used for the equity leg of the project
(`src/stocks/feishu`). Unlike the crypto pipeline, there is no resample step or
disk cache — the whole feature matrix is built **in RAM** fresh every run,
streamed off the raw parquet files with dask (see
[`build.py`](../../src/stocks/feishu/build.py) for the full walkthrough).

## Raw layout

Raw parquet files live flat under `data/stocks/feishu/` (DVC-tracked):

```
lob_data_in_sample.parquet                     # 5-min intraday LOB snapshots, in-sample
daily_data_in_sample.parquet                   # daily OHLCV, in-sample
lob_data_release_stage_out_of_sample.parquet   # LOB snapshots, out-of-sample
daily_data_release_stage_out_of_sample.parquet # daily OHLCV, out-of-sample
```

- **LOB file** — `asset_id`, `trade_day_id`, `time` (`HH:MM:SS`), plus
  1-indexed `bid_price_1..10`, `ask_price_1..10`, `bid_volume_1..10`,
  `ask_volume_1..10`. Renamed to 0-indexed before feature extraction.
- **Daily file** — `asset_id`, `trade_day_id`, `open`, `high`, `low`, `close`,
  `volume`, `amount`, `adj_factor`, `vwap_0930_0935`. `trade_day_id` is renamed
  to `date`.

The LOB files are large (~2 GB / ~1.1 GB on disk); they are streamed as dask
row-group partitions rather than loaded whole, so peak RAM stays bounded to one
partition (~0.5 GB). Both files are trade-day sorted, and the trailing
(possibly-truncated) day of each partition is carried into the next so no
`(asset, day)` group is ever computed from a split chunk.

## Symbols

The asset universe is the **union** of symbols across both `daily` files
(in-sample ∪ out-of-sample) — an asset that only trades in one period still
contributes windows for that period. Discovered at runtime via
`discover_symbols()`; there is no fixed symbol list in config.

## Splits

Three splits, assigned by each window's **label day**, built as one contiguous
per-asset series so rolling stats and windows carry across the boundary:

| Split | Source | Definition |
|-------|--------|------------|
| `train` | in-sample | first `train_frac` (default 0.80) chronologically, per asset |
| `val`   | in-sample | remaining `1 - train_frac`, per asset — model selection only |
| `test`  | out-of-sample file | untouched, final metrics |

The in-sample cut is chronological per asset (not global), so every val
window's label day comes strictly after every train window's *for that asset*.
A val window's look-back may reach into the train period — that mirrors the
test condition (out-of-sample windows look back into in-sample) rather than
leaking, since every normalisation below is point-in-time.

## Features — `src/stocks/feishu/features.py`

259 features per `(asset, day)` row: **240 intraday OFI** + **19 daily OHLCV**.

**OFI (240)** — 24 intraday time slots × 10 LOB levels. Cont-OFI is computed
tick-by-tick, then snapped onto a fixed 10-minute grid (09:40, 09:50, … 11:20,
13:00, … 15:00 — 11 morning + 13 afternoon slots); off-grid ticks contribute
nothing, and an unmatched slot is left zero. Normalised with a **causal 5-day
rolling z-score** per asset (`causal_rolling_zscore`, day `t` uses stats from
`[t-5, t)` only); flat (near-zero-variance) slots are left mean-centred rather
than divided by a ~0 std, and warm-up days (< 2 prior days) are left at zero.

**OHLCV (19)** — 14 engineered daily features (`ret_1d/5d/10d/20d`,
`vol_5d/20d`, `amihud`, `volume_zscore`, `rsi_14`, `ma_dist_5/20`,
`open_close_ret`, `high_low_range`, `close_vwap_dist`) plus 5 raw
(`open`, `close`, `volume`, `low`, `high`). Normalised **cross-sectionally**:
z-scored across all assets on the same day (not causal — there is no
look-ahead risk since this is a same-day, not forward-looking, feature).

All features are clipped to `[-5, 5]` after normalisation, then any remaining
non-finite value is zeroed as a final safety net.

## Labels

Two labeling schemes share the same raw data and features; they differ only in
`src/stocks/{feishu,feishu_midprice}/labels.py` and the resulting checkpoint
tree / config directory.

### `feishu` — traded-price forward return

```
return_t = (close_{t+1} - vwap_t) / |vwap_t|

0 = Down       (return_t < -alpha)
1 = Stationary (|return_t| <= alpha)
2 = Up         (return_t >  alpha)
```

`vwap_t` is the day-`t` opening VWAP (`vwap_0930_0935`) — the price a trader
entering at the open of day `t` actually gets — compared against day `t+1`'s
close. `alpha` is a fixed config value (`0.015` in the original Shifu-paper
setting, `0.002` for a stricter/more-balanced split), not calibrated from data.

### `feishu_midprice` — mid-to-mid trend over a horizon

```
mid_t = (best_bid_t + best_ask_t) / 2   (closing snapshot, ~15:00)
return_t = (mid_{t+h} - mid_t) / mid_t

0 = Down       (return_t <  lo)
1 = Stationary (lo <= return_t <= hi)
2 = Up         (return_t >  hi)
```

A genuine mid-price trend label (both endpoints are quote mids, avoiding the
traded-price/overnight-gap asymmetry of the base task). `h` (trading days) is
the swept axis across `configs/stocks/feishu_midprice/<model>_h{1,2,3,5,10}.json`.
`lo`/`hi` default to the **1/3, 2/3 quantiles of the training-split returns**
(`label_mode: "quantile"`, `compute_class_thresholds`) for ~balanced classes;
`label_mode: "fixed"` reproduces the fixed `±alpha` band instead.

### Causal pairing (both schemes)

A `T_past`-day window ending on day `t-1` is paired with the label anchored at
day `t` (the day after the window ends) and resolved using day `t+1`
(`feishu`) or day `t+h` (`feishu_midprice`). No feature ever reads the label's
anchor or resolution day — only prior-day features are in the window — so
there is no lookahead in either scheme.

## Config

Feishu configs set:

```json
"data_dir": "data/stocks/feishu",
"lob_file": "lob_data_in_sample.parquet",
"daily_file": "daily_data_in_sample.parquet",
"lob_file_oos": "lob_data_release_stage_out_of_sample.parquet",
"daily_file_oos": "daily_data_release_stage_out_of_sample.parquet",
"n_lob_levels": 10,
"feature_mode": "ofi",
"T_past": 50,
"alpha": 0.015,
"train_frac": 0.8
```

`feishu_midprice` configs additionally set `"horizon"`, `"label_mode"` and
`"class_quantiles"`, and drop `"alpha"` unless `label_mode: "fixed"`. Unlike
the crypto configs, `feature_mode` only supports `"ofi"` here — there is no
`"lob"` mode for this task, and `n_features()` always returns **259**
regardless of config.
