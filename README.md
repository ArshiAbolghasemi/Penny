# Penny

LOB direction forecasting on Binance crypto data. Three model families — **DeepLOB**, **JointDiffusion**, and **LOBTransformer** — share a common data pipeline and are trained to predict short-term price direction (down / stationary / up) from limit order book snapshots.

---

## Table of Contents

- [Data](#data)
- [Feature Modes](#feature-modes)
- [Labels](#labels)
- [Shared Pipeline](#shared-pipeline)
- [Models](#models)
  - [DeepLOB](#deeplob)
  - [JointDiffusion](#jointdiffusion)
  - [LOBTransformer](#lobtransformer)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Slurm](#slurm)
- [Configuration Reference](#configuration-reference)

---

## Data

Raw data lives under `data/binance/` and is tracked by DVC. Each calendar day produces three gzipped CSV files per symbol:

| File | Rows/day | Description |
|---|---|---|
| `binance_book_snapshot_25_{date}_{symbol}.csv.gz` | ~737k | Full LOB snapshots, 25 bid + 25 ask levels |
| `binance_trades_{date}_{symbol}.csv.gz` | ~4.3M | Executed trade ticks |
| `binance_quotes_{date}_{symbol}.csv.gz` | ~642k | Best bid/ask updates |

**Snapshot columns:** `timestamp` (µs UTC), `bids[i].price`, `bids[i].amount`, `asks[i].price`, `asks[i].amount` for i in 0…24.

**Trades columns:** `timestamp`, `price`, `amount`, `side` (`"buy"` = aggressor lifted the offer).

**Quotes columns:** `timestamp`, `bid_price`, `ask_price`.

Pull data with DVC:

```bash
uv run dvc pull
```

---

## Feature Modes

All models accept a `feature_mode` config key that selects the input representation. With `n_lob_levels = 10`:

### OFI mode — `"feature_mode": "ofi"` — 31 features

Per-level signed Cont order-flow imbalance (signed-log transform), plus microstructure, trade, and quote aggregates.

| Slice | Size | Description |
|---|---|---|
| `[0 : n)` | 10 | Bid OFI per level — `sign(Δv) · log1p(|Δv|)` |
| `[n : 2n)` | 10 | Ask OFI per level |
| `[2n : 2n+3)` | 3 | Spread/mid, log depth-imbalance, log-return |
| `[2n+3 : 2n+8)` | 5 | log-buy-vol, log-sell-vol, trade-imbalance, log-trade-count, VWAP deviation |
| `[2n+8 : 2n+11)` | 3 | log-n-quote-updates, mean-spread-norm, mid-range-norm |

### LOB mode — `"feature_mode": "lob"` — 51 features

Classical DeepLOB price offsets and log volumes, with the same 11 microstructure/trade/quote features appended.

| Slice | Size | Description |
|---|---|---|
| `[0 : n)` | 10 | Bid price offset = `(mid − bid_p[i]) / mid` |
| `[n : 2n)` | 10 | Ask price offset = `(ask_p[i] − mid) / mid` |
| `[2n : 3n)` | 10 | `log1p(bid_volume[i])` per level |
| `[3n : 4n)` | 10 | `log1p(ask_volume[i])` per level |
| `[4n : 4n+11)` | 11 | Same microstructure + trade + quote features as OFI |

Trades and quotes are aggregated to snapshot timestamps via vectorized `searchsorted` (no Python loops). Features are **z-scored per calendar day** — each day's mean and std are computed on that day's own rows so no lookahead crosses day boundaries. The normalized array is written to a numpy memmap cache; subsequent runs load it immediately.

---

## Labels

Labels follow the smoothed-mid trend formulation from DeepLOB (Zhang et al., 2019):

```
bwd(t)         = mean(mid[t−k : t])
fwd(t)         = mean(mid[t : t+k])
trend_ratio(t) = (fwd − bwd) / bwd
```

Three classes:

| Label | Value | Condition |
|---|---|---|
| Down | 0 | `trend_ratio < −α` |
| Stationary | 1 | `|trend_ratio| ≤ α` |
| Up | 2 | `trend_ratio > α` |

`α` is calibrated on the **training split** to the 33rd percentile of `|trend_ratio|`, producing roughly balanced class frequencies. `label_k = 5` means a 5-snapshot forward/backward horizon.

---

## Shared Pipeline

All three models share `src/crypto/utils/`:

| Module | Role |
|---|---|
| `loader.py` | Per-day file discovery, trade/quote aggregation, memmap cache builder |
| `features.py` | OFI and LOB feature extraction (`extract_features`, `n_features`) |
| `dataset.py` | `LOBDataset` (memmap-backed) + `build_datasets` → train/val/test split |
| `labels.py` | `compute_trend_series`, `calibrate_alpha`, `assign_labels`, `build_labels` |
| `evaluate.py` | `run_test` → accuracy, macro-F1, confusion matrix |
| `training.py` | `resolve_device` (cuda→mps→cpu fallback), `build_cosine_schedule` (warmup + cosine) |

Every model exposes the same interface: `model.predict(batch, device) → (B, 3) logits`, so `run_test` works identically across all three.

**Split:** 70% train / 15% val / 15% test by row index. Windows that straddle a day boundary or contain irregular timestamp gaps are excluded. `stride = 1` means every possible starting position is a window.

---

## Models

### DeepLOB

Based on Zhang et al., *"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"*, IEEE TSP 2019.

**Input:** `(B, 1, T_past, F)` — single-channel image, time on height axis, features on width axis.
**Output:** `(B, 3)` logits.

**Architecture:**

```
Input (B, 1, T, F)
    │
    ▼
Conv Block         (1×2) → (4×1) → (4×1) convolutions
                   BatchNorm2d + LeakyReLU(0.01) after each
                   output: (B, 32, T−6, F+1)
    │
    ▼
Inception Block    3 parallel paths with temporal kernel heights 1, 3, 5
                   each outputs 64 channels → concatenated: (B, 192, T−6, F+1)
    │
    ▼
AdaptiveAvgPool2d  collapses feature axis → (B, 192, T−6)
    │
    ▼
LSTM               input=192, hidden=64, 1 layer → last hidden state (B, 64)
    │
    ▼
Dropout(0.1)
    │
    ▼
Linear(64 → 3)     logits
```

**Training:** cross-entropy on the 3-class label, AdamW optimizer, linear warmup then cosine LR decay, early stopping on validation CE.

**Default hyperparameters:**

| Key | OFI | LOB |
|---|---|---|
| `T_past` | 100 | 100 |
| `label_k` | 5 | 5 |
| `deeplob_conv_filters` | 32 | 32 |
| `deeplob_inception_filters` | 64 | 64 |
| `deeplob_lstm_hidden` | 64 | 64 |
| `deeplob_dropout` | 0.1 | 0.1 |
| `lr` | 1e-4 | 1e-4 |
| `batch_size` | 64 | 64 |
| `epochs / patience` | 50 / 10 | 50 / 10 |

---

### JointDiffusion

Based on Deja et al., *"Joint Generative Modeling and Discriminative Classification"*, 2023.

**Core idea:** a single time-conditioned U-Net is trained jointly to denoise LOB windows (diffusion objective) and classify price trend from the bottleneck. The shared encoder means the denoising task regularizes the classifier — it can't overfit to surface patterns without also learning to reconstruct the full feature window.

**Input:** noisy window `x_t (B, 1, T_past, F)` + integer timestep `t (B,)`.
**Output:** `(eps_hat (B, 1, T, F), logits (B, 3))`.

**Forward diffusion** uses `DDPMScheduler` from HuggingFace `diffusers` (linear beta schedule):

```
x_t = sqrt(ᾱ_t) · x₀  +  sqrt(1 − ᾱ_t) · ε,    ε ~ N(0, I)
ᾱ_t = product(1 − βₛ, s=1..t),    β linearly spaced in [1e-4, 0.02] over T=1000 steps
```

**Architecture:**

```
Timestep t
    │
Sinusoidal embedding(t, dim=128) → MLP → temb (B, 128)

Input x_t (B, 1, T, F)
    │
    ▼
Stem              TimeDoubleConv(1 → 32)
                  [Conv2d(3×3) → GroupNorm → SiLU → +temb bias → Conv2d(3×3) → GroupNorm → SiLU]
    │ skip₀
    ▼
Down 1            MaxPool2d(2) + TimeDoubleConv(32 → 64)
    │ skip₁
    ▼
Down 2            MaxPool2d(2) + TimeDoubleConv(64 → 128)
    │
    ├──────────────────────────────────┐
    │                                  ▼
    │                    Bottleneck (B, 128, T/4, F/4)
    │                         │
    │                    AdaptiveAvgPool2d(1) → Flatten
    │                         │
    │                    Linear(128, 128) → SiLU → Dropout(0.1) → Linear(128, 3)
    │                         │
    │                      logits (B, 3)
    ▼
Up 1              Upsample(nearest) + skip₁ concat + TimeDoubleConv(128+64 → 64)
    │
    ▼
Up 2              Upsample(nearest) + skip₀ concat + TimeDoubleConv(64+32 → 32)
    │
    ▼
Conv2d(1×1)       → eps_hat (B, 1, T, F)
```

GroupNorm (not BatchNorm) is used throughout so that single-sample inference at `t = 0` has consistent statistics.

**Joint loss:**

```
L_diff  = MSE(eps_hat, ε)
w(t)    = (1 − t / T)²                       ← down-weights heavily-noised timesteps
L_cls   = mean( w(t) · CE(logits, label) )
L       = L_diff + λ · L_cls
```

`w(t)` equals 1 at `t = 0` (clean window, label fully recoverable) and 0 at `t = T` (pure noise). This keeps the classifier signal meaningful at every noise level without requiring the model to classify from noise.

**Inference:** `model.predict(batch, device)` runs the clean window through the U-Net at `t = 0` and returns the bottleneck logits. No sampling or denoising is performed.

**Default hyperparameters:**

| Key | OFI | LOB |
|---|---|---|
| `T_past` | 100 | 100 |
| `label_k` | 5 | 5 |
| `T_max` | 1000 | 1000 |
| `beta_start / beta_end` | 1e-4 / 0.02 | 1e-4 / 0.02 |
| `jd_base_channels` | 32 | 32 |
| `jd_depth` | 2 | 2 |
| `jd_time_emb` | 128 | 128 |
| `jd_dropout` | 0.1 | 0.1 |
| `lambda_trend` | 1.0 | 1.0 |
| `lr` | 1e-4 | 1e-4 |
| `batch_size` | 64 | 64 |
| `epochs / patience` | 50 / 10 | 50 / 10 |

---

### LOBTransformer

A pure transformer classifier over the full LOB feature window. No diffusion, no recurrence — self-attention across the time axis with the full F-dimensional feature vector at each step.

**Input:** `(B, 1, T_past, F)` — same format as the other models.
**Output:** `(B, 3)` logits.

**Architecture:**

```
Input (B, 1, T, F)
    │
    squeeze(1)
    │
    ▼  (B, T, F)
Linear(F → d)         project each timestep's feature vector into d-dimensional space
    +
Learned positional    nn.Parameter(1, T, d) — added to every sample in the batch
    │
    ▼  (B, T, d)
Transformer Encoder   L layers, each:
                        MultiHeadAttention (H heads)
                        FFN: Linear(d, 2d) → GELU → Linear(2d, d)
                        LayerNorm + residual connections
    │
    ▼
mean over T           (B, d)
    │
    ▼
Head                  Linear(d, d) → GELU → Linear(d, 3)
    │
    ▼  (B, 3) logits
```

**Training:** cross-entropy loss, AdamW, linear warmup + cosine LR decay, early stopping on val CE. `stride = 1` is used to maximize label coverage — the transformer processes large batches efficiently so the increased window count is manageable.

**Default hyperparameters:**

| Key | OFI | LOB |
|---|---|---|
| `T_past` | 100 | 100 |
| `label_k` | 10 | 10 |
| `lobt_hidden` (d) | 256 | 256 |
| `lobt_heads` (H) | 8 | 8 |
| `lobt_layers` (L) | 4 | 4 |
| `lr` | 3e-4 | 3e-4 |
| `batch_size` | 64 | 64 |
| `epochs / patience` | 80 / 15 | 80 / 15 |

---

## Project Structure

```
Penny/
├── src/crypto/
│   ├── utils/
│   │   ├── loader.py           Binance file discovery, trade/quote aggregation, memmap cache
│   │   ├── features.py         OFI and LOB feature extraction
│   │   ├── dataset.py          LOBDataset + build_datasets (shared by all models)
│   │   ├── labels.py           smoothed-mid trend labels + alpha calibration
│   │   ├── evaluate.py         run_test: accuracy / macro-F1 / confusion
│   │   └── training.py         resolve_device, build_cosine_schedule
│   ├── deeplob/
│   │   ├── model.py            Conv Block → Inception Block → LSTM → head
│   │   └── train.py
│   ├── jointdiff/
│   │   ├── model.py            time-conditioned U-Net (denoising + bottleneck classifier)
│   │   └── train.py            DDPMScheduler from diffusers
│   └── lobtransformer/
│       ├── model.py            Linear projection → Transformer Encoder → mean-pool → head
│       ├── train.py
│       └── infer.py            single-window inference from Binance snapshot files
├── configs/crypto/
│   ├── deeplob/                btcusdt_ofi.json  btcusdt_lob.json
│   ├── jointdiff/              btcusdt_ofi.json  btcusdt_lob.json
│   └── lobtransformer/         btcusdt_ofi.json  btcusdt_lob.json
├── slurm/
│   ├── deeplob_{ofi,lob}.slurm
│   ├── jointdiff_{ofi,lob}.slurm
│   └── lobtransformer_{ofi,lob}.slurm
├── data/                       tracked by DVC, not committed to git
│   └── binance/                binance_book_snapshot_25_*, binance_trades_*, binance_quotes_*
└── pyproject.toml
```

---

## Setup

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
# CPU — local dev / non-GPU nodes
uv sync --extra cpu

# CUDA 12.8 — A100, H100, RTX 30xx/40xx
uv sync --extra cu128

# CUDA 12.6 — V100, older driver
uv sync --extra cu126

# Apple Silicon
uv sync --extra mps

# Pull data from the DVC remote (Cloudflare R2)
uv run dvc pull
```

---

## Training

```bash
# DeepLOB
uv run python -m crypto.deeplob.train configs/crypto/deeplob/btcusdt_ofi.json
uv run python -m crypto.deeplob.train configs/crypto/deeplob/btcusdt_lob.json

# JointDiffusion
uv run python -m crypto.jointdiff.train configs/crypto/jointdiff/btcusdt_ofi.json
uv run python -m crypto.jointdiff.train configs/crypto/jointdiff/btcusdt_lob.json

# LOBTransformer
uv run python -m crypto.lobtransformer.train configs/crypto/lobtransformer/btcusdt_ofi.json
uv run python -m crypto.lobtransformer.train configs/crypto/lobtransformer/btcusdt_lob.json
```

Or via the installed CLI entry points after `uv sync`:

```bash
penny-deeplob        configs/crypto/deeplob/btcusdt_ofi.json
penny-jointdiff      configs/crypto/jointdiff/btcusdt_ofi.json
penny-lobtransformer configs/crypto/lobtransformer/btcusdt_ofi.json
```

Each run creates a timestamped checkpoint directory:

```
{checkpoint_dir}/{model}_{symbol}_{feature_mode}_{timestamp}/
├── best.pt            model weights + config snapshot + calibrated alpha
├── config.json        full config with n_features filled in
└── train.log          per-epoch loss and metric log
```

The feature cache is built on the first run and reused automatically:

```
{cache_dir}/{symbol}_n{n}_{mode}_lob.{feat,mid,ts}.npy
```

---

## Inference

LOBTransformer includes a single-window inference script:

```bash
uv run python -m crypto.lobtransformer.infer \
    --checkpoint checkpoints/lobtransformer_BTCUSDT_ofi_20240115_120000 \
    --snapshot   data/binance/binance_book_snapshot_25_2024-01-15_BTCUSDT.csv.gz \
    --trades     data/binance/binance_trades_2024-01-15_BTCUSDT.csv.gz \
    --quotes     data/binance/binance_quotes_2024-01-15_BTCUSDT.csv.gz \
    --device     cpu
```

Output:

```
LOBTransformer signal
  label : 2 (up)
  probs : down=0.082  stat=0.251  up=0.667
```

`--trades` and `--quotes` are optional; missing files fall back to zero-filled aggregates. The last `T_past` rows of the snapshot file are used. Features are normalized by the window's own mean/std at inference time.

---

## Slurm

```bash
sbatch slurm/deeplob_ofi.slurm
sbatch slurm/jointdiff_lob.slurm
sbatch slurm/lobtransformer_ofi.slurm
```

Override the config without editing the script:

```bash
sbatch --export=ALL,CONFIG=configs/crypto/deeplob/btcusdt_lob.json slurm/deeplob_ofi.slurm
```

Resource allocations:

| Script | Memory | Wall time | Notes |
|---|---|---|---|
| `deeplob_{ofi,lob}` | 24 GB | 8 h | CNN+LSTM, lightest model |
| `jointdiff_{ofi,lob}` | 32 GB | 16 h | U-Net + diffusion, slowest |
| `lobtransformer_{ofi,lob}` | 32 GB | 12 h | Transformer, stride=1 |

---

## Configuration Reference

Keys shared by all three models:

| Key | Description |
|---|---|
| `symbol` | Binance trading pair, e.g. `"BTCUSDT"` |
| `data_dir` | Directory containing the gzipped Binance CSV files |
| `cache_dir` | Directory for the numpy memmap feature cache |
| `checkpoint_dir` | Root directory for checkpoint subdirectories |
| `feature_mode` | `"ofi"` or `"lob"` |
| `n_lob_levels` | LOB depth levels to use (1–25) |
| `T_past` | Window length in snapshots |
| `label_k` | Smoothed-mid horizon (snapshots forward and backward) |
| `label_alpha` | Manual alpha; set to `-1` to auto-calibrate from training data |
| `train_frac` / `val_frac` | Fraction of snapshots for train / val; remainder is test |
| `stride` | Step between window start positions |
| `lr` | Peak learning rate |
| `weight_decay` | AdamW weight decay |
| `warmup_steps` | Linear LR warmup steps before cosine decay |
| `grad_clip` | Gradient norm clip |
| `batch_size` | Mini-batch size |
| `epochs` | Maximum training epochs |
| `patience` | Early-stopping patience in epochs |
| `device` | `"cuda"`, `"mps"`, or `"cpu"` |

Model-specific keys:

| Key | Model | Description |
|---|---|---|
| `deeplob_conv_filters` | DeepLOB | Conv block output channels |
| `deeplob_inception_filters` | DeepLOB | Inception path channels per parallel stream |
| `deeplob_lstm_hidden` | DeepLOB | LSTM hidden size |
| `deeplob_dropout` | DeepLOB | Dropout before the linear head |
| `T_max` | JointDiffusion | Total diffusion timesteps |
| `beta_start` / `beta_end` | JointDiffusion | Linear beta schedule endpoints |
| `jd_base_channels` | JointDiffusion | Stem output channels; doubled at each Down block |
| `jd_depth` | JointDiffusion | Number of Down/Up block pairs |
| `jd_time_emb` | JointDiffusion | Timestep embedding dimension |
| `jd_dropout` | JointDiffusion | Dropout in the bottleneck classifier MLP |
| `lambda_trend` | JointDiffusion | Classification loss weight λ |
| `lobt_hidden` | LOBTransformer | Transformer model dimension d |
| `lobt_heads` | LOBTransformer | Number of self-attention heads |
| `lobt_layers` | LOBTransformer | Number of Transformer Encoder layers |
