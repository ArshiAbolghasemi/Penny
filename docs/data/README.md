# Data

Penny trains on **limit-order-book (LOB) snapshots** from two crypto exchanges, plus
a separate Chinese A-share equity dataset (Feishu), and turns them into fixed-length
windows with a 3-class trend label (`down / flat / up`). Everything downstream of the
raw dumps is deterministic and config-driven, so a run is fully reproducible from a
single JSON file plus the DVC-tracked data.

Raw market data is **not** in git — it is tracked by [DVC](https://dvc.org) on an
S3-compatible remote. See [Getting the data](#getting-the-data) below.

> The crypto pipeline (below) is shared across both exchanges. The equity dataset
> has its own pipeline — in-RAM feature build, no resample/cache step — documented
> separately in [feishu.md](feishu.md).

## The two exchanges

| Exchange | Pairs | Levels kept | Bin size | Raw format doc |
|----------|-------|-------------|----------|----------------|
| Binance  | BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ADAUSDT, AVAXUSDT, BNBUSDT, DOGEUSDT, USDCUSDT | 10 | configurable (0.25 s default) | [binance.md](binance.md) |
| Coinbase (Iranian) | BTCIRT, USDTIRT | 20 | 10 s (already binned at source) | [coinbase.md](coinbase.md) |

Both exchanges are normalised to the **same intermediate parquet schema**, so all of
the feature/label/windowing logic below is shared across them. IRT = Iranian Toman.

## Pipeline overview

```
raw CSVs (per exchange)                      docs/data/binance.md · coinbase.md
        │  scripts/resample_binance.py
        │  scripts/resample_coinbase.py
        ▼
data/resampled/{exchange}/{SYMBOL}.parquet.gz     one row per time-bin, unified schema
        │  crypto/features.py   → raw feature matrix (per calendar day)
        │  crypto/loader.py     → causal rolling z-score → memmap cache
        ▼
data/cache/…/{SYMBOL}_…npy                         normalised features + mid + timestamps
        │  crypto/labels.py     → smoothed-mid trend label + alpha calibration
        │  crypto/dataset.py    → sliding windows, calendar-aware split
        ▼
(train / val / test) windows  x:(1, T_past, F)  label:{0,1,2}
```

The stages, in order:

### 1. Resample raw → unified parquet
Per-exchange scripts collapse irregular tick data into fixed time-bins and join the
book snapshot, trades and quotes into one row per bin. Output columns are identical
across exchanges:

```
bids[i].price, bids[i].amount, asks[i].price, asks[i].amount   (i = 0 … n-1)
mid, spread
trade_count, buy_vol, sell_vol, vwap, trade_imbalance
quote_bid_price, quote_ask_price, quote_bid_amount, quote_ask_amount
timestamp_utc, bin
```

Exchange-specific details (raw filenames, column mapping, quirks) live in
[binance.md](binance.md) and [coinbase.md](coinbase.md).

### 2. Feature extraction — `crypto/features.py`
Each bin row becomes a feature vector. Two mutually-exclusive **feature modes**
select what channel-0 looks like, both followed by the same 11 microstructure /
trade / quote features:

- **`ofi`** (default) — per-level [Cont Order-Flow Imbalance](https://arxiv.org/abs/1011.6402);
  `n + 11` features.
- **`lob`** — classical DeepLOB price-offset + log-volume representation;
  `4n + 11` features.

The full column layout for both modes is documented in
**[features.md](features.md)**.

### 3. Labels — `crypto/labels.py`
The DeepLOB smoothed-mid trend label: compare the mean mid over the next `k` bins to
the mean mid over the previous `k` bins, threshold the ratio at `±alpha` into
`down / flat / up`. `alpha` is **calibrated on the training split only** to the
33.3rd percentile of `|trend_ratio|`, giving roughly balanced classes. Full
definition in **[labels.md](labels.md)**.

### 4. Windowing, splits & normalisation
- **Windows** — a length-`T_past` sliding window (stride `stride`); its label is the
  label at the window's last bin. Windows that straddle a large timestamp gap
  (e.g. a day boundary or a data outage) are skipped.
- **Split** — a chronological `train_frac / val_frac / rest` cut over the full
  time series (defaults 0.70 / 0.15 / 0.15).
- **Normalisation** — features are z-scored with a **causal trailing rolling
  window** (`norm_window`, default 2000 bins): bin `t` uses stats from
  `[t − norm_window + 1, t]` only — never the future. This removes the intra-day
  lookahead of naive per-day scaling. `alpha` and the normaliser are both frozen
  from the training region.
- **Cache** — the normalised feature matrix is written once to a numpy memmap and
  reused on subsequent runs (keyed by symbol, level count, feature mode and
  `norm_window`).

The full windowing/split/normalisation reference is in
**[windows-and-normalization.md](windows-and-normalization.md)**.

## Getting the data

Raw and resampled data are DVC-tracked, not in git.

```bash
uv run dvc pull        # download data/ from the remote
uv run dvc push        # upload new/changed data
```

The remote is an **S3-compatible Cloudflare R2 bucket** (`penny`). Credentials live
in `.dvc/config.local`, which is **not** committed.

> **Getting credentials:** the R2 access keys are not in the repo. **Contact the
> team** to obtain them, then place them in `.dvc/config.local` (git-ignored). Once
> configured, `uv run dvc pull` will fetch everything under `data/`.

If you have raw exchange dumps but no resampled parquet yet, regenerate them:

```bash
uv run python scripts/resample_binance.py --interval 10 --levels 10
uv run python scripts/resample_coinbase.py --levels 20
```

## Reference

| File | Contents |
|------|----------|
| [binance.md](binance.md) | Binance raw CSV schema, resampling, symbols |
| [coinbase.md](coinbase.md) | Coinbase raw CSV schema, resampling, symbols |
| [features.md](features.md) | `ofi` / `lob` feature modes, full column layout |
| [labels.md](labels.md) | Trend-label definition, `alpha` calibration |
| [windows-and-normalization.md](windows-and-normalization.md) | Windows, splits, causal rolling z-score, caching |
| [feishu.md](feishu.md) | Feishu A-share equity data — raw schema, in-RAM feature build, both label schemes |
