# Coinbase data

LOB data from [Coinbase](https://coinbase.com), a US-based crypto exchange. Prices are
quoted in Iranian Toman (IRT). The raw feed is already sampled at a fixed 10 s
cadence, so resampling is really a **schema-normalisation** step that maps Coinbase's
column names onto the shared parquet schema.

## Symbols

`BTCIRT`, `USDTIRT`.

## Raw layout

Raw CSVs live under `data/coinbase_data/` (DVC-tracked), one pair of files per symbol:

```
{SYMBOL}_orderbook.csv   # time, bid_price_1..N, bid_volume_1..N, ask_price_1..N, ask_volume_1..N
{SYMBOL}_trades.csv      # snapshot_time, trade_time, price, volume, direction
```

- **orderbook** — `time` (ISO timestamp) plus 1-indexed `bid_price_{i}` /
  `bid_volume_{i}` / `ask_price_{i}` / `ask_volume_{i}` for `i = 1 … N`.
- **trades** — `snapshot_time` (matches the orderbook `time`), `trade_time`,
  `price`, `volume`, `direction` (`buy`/`sell`).

## Resampling — `scripts/resample_coinbase.py`

Because the book is already at 10 s intervals, the script deduplicates on the
timestamp string (keep last) and renames columns to the shared 0-indexed schema:

```
bid_price_{i}  → bids[i-1].price      ask_price_{i}  → asks[i-1].price
bid_volume_{i} → bids[i-1].amount     ask_volume_{i} → asks[i-1].amount
```

`mid` and `spread` are derived from level 0. Trades are aggregated on
`snapshot_time` (the same timestamp as the book) into `trade_count`, `buy_vol`,
`sell_vol`, `vwap`, `trade_imbalance`. Bins with no trades get zero activity (not
NaN); `vwap` is left NaN and the feature extractor falls back to `mid`.

```bash
# defaults: --levels 20, interval fixed at 10 s
uv run python scripts/resample_coinbase.py
uv run python scripts/resample_coinbase.py --symbols BTCIRT USDTIRT
```

Output: `data/resampled/coinbase/{SYMBOL}.parquet.gz`.

## Config

Coinbase configs set:

```json
"exchange": "coinbase",
"symbol": "BTCIRT",
"data_dir": "data/resampled/coinbase",
"n_lob_levels": 20
```

With `n_lob_levels = 20`: `ofi` mode → **31** features, `lob` mode → **91**
features (see [features.md](features.md)). Setting `"exchange": "coinbase"` also makes
the loader point you at `resample_coinbase.py` if the parquet is missing.
