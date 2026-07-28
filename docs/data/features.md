# Feature extraction

`crypto/features.py` turns each resampled parquet row into a feature vector. The
`feature_mode` config key selects one of two representations for the LOB block; both
are followed by the **same 11** microstructure / trade / quote features.

Feature extraction runs **per calendar day** (OFI and returns need a previous tick,
which resets at each day's first row). Normalisation is applied globally afterwards
by the loader — see [windows-and-normalization.md](windows-and-normalization.md).

Let `n = n_lob_levels`.

## `ofi` mode (default) — `n + 11` features

Channel 0 is the per-level **Cont Order-Flow Imbalance**, computed from raw
per-level prices and volumes (no signed-log transform). The first row of each day
has no previous snapshot and is set to 0.

| Index range | Feature |
|-------------|---------|
| `[0, n)`        | net Cont-OFI per level (`bid-OF − ask-OF`) |
| `[n, n+3)`      | spread/mid, log depth-imbalance, log-return |
| `[n+3, n+8)`    | log buy-vol, log sell-vol, trade-imbalance, log trade-count, vwap-dev |
| `[n+8, n+11)`   | log trade-count (activity), spread-norm, `|log-ret|` (range proxy) |

Per-level OFI is the standard Cont definition:

```
bid-OF_i = 1{p_b↑}·v_b + 1{p_b=}·(v_b − v_b_prev) + 1{p_b↓}·(−v_b_prev)
ask-OF_i = 1{p_a↑}·(−v_a_prev) + 1{p_a=}·(v_a − v_a_prev) + 1{p_a↓}·v_a
OFI_i    = bid-OF_i − ask-OF_i
```

## `lob` mode — `4n + 11` features

Channel 0 is the classical DeepLOB price-offset + log-volume representation.

| Index range | Feature |
|-------------|---------|
| `[0, n)`        | bid price offset `(mid − bid_price_i) / mid` |
| `[n, 2n)`       | ask price offset `(ask_price_i − mid) / mid` |
| `[2n, 3n)`      | `log1p(bid_volume_i)` |
| `[3n, 4n)`      | `log1p(ask_volume_i)` |
| `[4n, 4n+3)`    | spread/mid, log depth-imbalance, log-return |
| `[4n+3, 4n+8)`  | log buy-vol, log sell-vol, trade-imbalance, log trade-count, vwap-dev |
| `[4n+8, 4n+11)` | log trade-count (activity), spread-norm, `|log-ret|` (range proxy) |

## The shared 11-feature tail

Both modes append the same block:

**Microstructure (3)**
- `spread / mid`
- `log( total_bid_vol / total_ask_vol )` — depth imbalance
- `log-return` of the mid vs the previous bin (0 at the first row)

**Trade (5)**
- `log1p(buy_vol)`, `log1p(sell_vol)`
- `trade_imbalance = (buy − sell) / (buy + sell)`
- `log1p(trade_count)`
- `vwap_dev = (vwap − mid) / mid` (0 when no trades; `vwap`→`mid` fallback)

**Quote / activity (3)** — the resampled parquet lacks intra-bin quote counts, so
these substitute with same-intent proxies:
- `log1p(trade_count)` (activity proxy for quote-update count)
- `spread / mid`
- `|log-return|` (inter-bin range proxy)

## Feature counts

| `n_lob_levels` | `ofi` (`n+11`) | `lob` (`4n+11`) |
|----------------|----------------|-----------------|
| 10 (Binance)   | 21             | 51              |
| 20 (Coinbase)   | 31             | 91              |

The exact count is computed by `features.n_features(config)` and written back into
the config as `n_features` at dataset-build time, so every model reads its true input
width from there.

## Why OFI is the default

The **generative diffusion models** (JointDiT, JumpGateLOB) denoise the feature
image. OFI is a signed, roughly stationary flow quantity — a more natural target for
a score/ε-prediction network than raw price offsets, and it keeps the input compact
(`n + 11` vs `4n + 11`). The `lob` mode is retained for parity with the classical
DeepLOB-family baselines.
