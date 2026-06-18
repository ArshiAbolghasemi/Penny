# Penny

Penny is an **inpainting diffusion model for limit order book (LOB) forecasting**, based on
*Painting the Market* (Backhouse et al., arXiv:2509.05107v1).  It treats a past + future LOB
window as a 2D image, corrupts it with Gaussian noise via a forward diffusion process, and
trains a 2D UNet to inpaint the future region conditioned on the observed past.  An auxiliary
DeepLOB trend-classification loss steers the denoiser toward directionally consistent futures.
The model is trained on 10-second LOB snapshots of BTC/IRT from Nobitex (an Iranian crypto
exchange), covering approximately 9 days of data.

---

## Table of Contents

1. [Data](#1-data)
2. [Feature Extraction](#2-feature-extraction)
3. [Image Construction](#3-image-construction)
4. [Normalization](#4-normalization)
5. [Dataset Splitting and Windowing](#5-dataset-splitting-and-windowing)
6. [DeepLOB Trend Labels](#6-deeplob-trend-labels)
7. [OFI Mid-Price Reconstruction](#7-ofi-mid-price-reconstruction)
8. [Model Architecture](#8-model-architecture)
9. [Diffusion Process](#9-diffusion-process)
10. [Training](#10-training)
11. [Validation](#11-validation)
12. [Test Evaluation](#12-test-evaluation)
13. [Inference](#13-inference)
14. [Setup and Usage](#14-setup-and-usage)
15. [Configuration](#15-configuration)
16. [Project Structure](#16-project-structure)

---

## 1. Data

Raw market data for each exchange is stored under `data/` and tracked by DVC (not Git).

| Exchange | Pairs |
|---|---|
| Nobitex | BTCIRT, USDTIRT |
| Bitpin | BTC_IRT, USDT_IRT |
| Wallex | BTCTMN, USDTTMN |
| Tabdeal | BTCIRT, USDTIRT |
| Ramzinex | BTC_IRT, USDT_IRT |

IRT and TMN both denote Iranian Toman (different naming conventions per exchange).

**Order book snapshots** (`*_orderbook.csv`) contain a timestamp and up to 20 bid/ask price-volume
levels per snapshot, captured every 10 seconds.  **Trade ticks** (`*_trades.csv`) contain each
individual trade's timestamp, price, volume, and direction (buy/sell).

The default training pair is **BTCIRT on Nobitex**.  If the raw data has a large gap (e.g. a
16-hour outage), `scripts/trim_gap.py` drops all rows before the largest detected gap so that
the model only sees continuous data.

---

## 2. Feature Extraction

Each snapshot contributes **R = 2n + 3** rows of features, where n is the number of LOB levels
per side (default 10), giving R = 23 rows.  Features are organized into two channels per row.

### Row layout

The rows are indexed from 0 to R−1 in a specific order that places the spread at the center:

- Rows **0 to n−1**: bid levels, ordered from the deepest (worst) bid at row 0 up to the
  best bid at row **n−1**.
- Rows **n to 2n−1**: ask levels, ordered from the best ask at row **n** down to the deepest
  ask at row **2n−1**.
- Rows **2n, 2n+1, 2n+2**: three trade-feature rows (channel 1 only; channel 0 is zero).

This arrangement places the tightest spread—best bid at row n−1 and best ask at row n—at the
center of the image.

### Channel 0 — flow (`feature_mode` switch)

**OFI mode (default):** Channel 0 carries per-level Order Flow Imbalance (Cont et al. 2013).
OFI measures the net volume pressure arriving at each price level between consecutive snapshots:

- **Bid level i**: if the bid price improved (moved up), the full new volume counts as buy
  pressure (+vol); if it stayed the same, only the volume change counts (+Δvol); if it
  worsened (moved down), the previously posted volume is counted as cancelled (−prev_vol).
- **Ask level i**: the mirror logic, stored buy-positive (a tightening ask = buying pressure).

The first snapshot in any series has no predecessor, so its OFI is set to zero.  Positive values
at both the best bid and best ask rows indicate aggressive buy-side activity.

**LOB mode:** Channel 0 carries signed price offsets from the instantaneous mid-price
(`bid_price_i − mid`, `ask_price_i − mid`).  This preserves absolute price structure at the
cost of losing flow information.

### Channel 1 — state

Channel 1 carries resting signed depth for LOB rows: positive for bids (+volume), negative for
asks (−volume).  This encodes the passive liquidity profile at each level.

### Trade rows (rows 2n to 2n+2, channel 1)

Three aggregate trade features are computed per snapshot over the preceding `snapshot_interval_sec`
seconds using prefix sums over the sorted trade tape:

1. **Log total volume** — `log(1 + Σ volume)` of all trades in the window.
2. **Buy-volume ratio** — fraction of total volume that was initiated by buyers.
3. **Buy-count ratio** — fraction of individual trades that were buyer-initiated.

When no trades occurred in the interval, log volume is 0 and both ratios default to 0.5.
Channel 0 of these three rows is unused (set to zero) since OFI is not meaningful for the
aggregate trade stream.

---

## 3. Image Construction

### From row stream to image

For each sliding window of T_total consecutive snapshots, the row stream slice of shape
`(T_total, R, 2)` is transposed to `(R, T_total, 2)`, making rows the height axis and time
the width axis.  The first T_past columns are the observed past; the last T_future columns are
the region to be inpainted (the target).

Default dimensions: **T_past = 156** (≈ 26 min), **T_future = 100** (≈ 17 min),
**T_total = 256** (≈ 43 min).

In OFI mode, the channel-0 value at column 0 of the LOB rows is forced to zero because OFI
requires a preceding snapshot, which does not exist at the window boundary.

### Square padding

The UNet requires a square input.  Since R = 23 is much smaller than T_total = 256, each row
is repeated an approximately equal number of times along the height axis so that the padded
height reaches `padded_size = 256`.  Rows are padded in a round-robin manner so some original
rows get one extra repeat.  The mapping from original row index to padded slice is stored as
`level_starts`, allowing features to be read back from padded coordinates later.

The final image tensor is `(2, 256, 256)` — two channels, height 256, width 256.

### Inpainting mask

A binary mask of shape `(1, 256, 256)` marks the future region: all columns from T_past onward
are set to 1 (to be generated), and all past columns are set to 0 (known, to be re-pasted).
The mask is the same for every window.

---

## 4. Normalization

A `RollingNormalizer` computes per-row, per-channel z-score statistics from the **last
`norm_window_snapshots` rows of the training split only** (default 8 640 rows ≈ 1 day), then
freezes them for the entire run.  No training statistics are recomputed on validation or test
data — this prevents any look-ahead leakage.

The normalization procedure:

1. Compute the mean and standard deviation per (row, channel) from the fitting window.
2. Z-score the entire row stream: `z = (x − mean) / std`.
3. Clip by the 95th percentile of |z| observed over the full training set (outlier robustness).
4. Cast to float32.

The normalizer's mean, std, and clip tensors are saved in every checkpoint alongside the model
weights, so inference can restore them without reprocessing the training data.

---

## 5. Dataset Splitting and Windowing

### Calendar-day split

Data is divided strictly by calendar day — no random shuffling.  The ordered unique days are
enumerated, and the first 6 days go to training, the next 2 to validation, the last 1 to test.
Split boundaries fall exactly at midnight, so a row belongs to exactly one split.

### Sliding windows with stride

Within each split, windows of length T_total are extracted starting every `stride` snapshots
(default 30, i.e. every 5 minutes).  Consecutive windows therefore overlap heavily (88% overlap
at the default settings), which multiplies the number of training examples from ≈ 34 per day
(non-overlapping) to ≈ 280 per day.

**Day-boundary skipping:** any window whose first snapshot and last snapshot fall on different
calendar days is silently discarded.  This prevents the model from ever seeing a past-region
that spans midnight while the future region spans the next day — a temporal boundary that
carries no real market structure.

### Approximate window counts

With 9 days of ~8 640 snapshots/day and a stride of 30:
- **Training (6 days)**: ~1 650 windows
- **Validation (2 days)**: ~560 windows
- **Test (1 day)**: ~280 windows

Exact counts vary with the actual number of snapshots per day and depend on how many windows
straddle midnight.

---

## 6. DeepLOB Trend Labels

Following the DeepLOB formulation (Ntakaris et al. 2018), the trend label is derived from the
**smoothed mid-price** rather than a single-point comparison, which is more robust to
microstructure noise.

### Trend ratio

For a window with T_past observed steps and the boundary at position T_past:

```
bwd = mean(mid[T_past − k : T_past])       # mean of the last k past mids
fwd = mean(mid[T_past : T_past + k])       # mean of the first k future mids
l   = (fwd − bwd) / bwd                   # fractional change
```

`label_k = 10` by default, giving 100-second smoothing on each side of the boundary.

### Class thresholding

A scalar threshold `alpha` converts `l` to a 3-class label:
- **Up (2)**: `l > alpha` — future mid rises meaningfully
- **Down (0)**: `l < −alpha` — future mid falls meaningfully
- **Stationary (1)**: `|l| ≤ alpha` — no significant movement

### Alpha calibration

`alpha` is calibrated on the training set to achieve approximately balanced classes (one-third
each).  It is set to the **33.3rd percentile of |l|** across all training windows, so by
definition one-third of training windows have `|l| < alpha` (stationary) and the remaining
two-thirds split evenly between up and down.

Balanced classes prevent the model from defaulting to "stationary" and make accuracy a
meaningful metric with a 33% random baseline.  Alpha is frozen after training and applied
unchanged during validation, test, and inference.

**Note on `label_alpha` in config:** setting `label_alpha = -1` triggers auto-calibration
(recommended).  Any positive value overrides calibration and uses that value directly.

---

## 7. OFI Mid-Price Reconstruction

In OFI mode, channel 0 carries flow quantities (OFI), not prices.  The future mid-price series
needed for the trend label and evaluation metrics must therefore be **reconstructed** from the
generated OFI values.

**Gamma (γ) fitting:** an OLS regression (through the origin) is fit on the training split
between the first-difference of the real mid-price (Δmid) and the best-level OFI (sum of best
bid + best ask OFI at each snapshot).  The slope `γ` captures how many IRT of mid-price change
correspond to one unit of net order-flow at the best quote.

**Reconstruction:** given a generated OFI sequence in the future region, the future mid-price
is reconstructed as:

```
mid[T_past + t] = mid[T_past] + γ · cumsum(ofi_best[T_past : T_past + t + 1])
```

The anchor `mid[T_past]` is the real mid-price at the boundary (last known snapshot), so the
reconstruction is tied to observed reality before drifting into the generated future.

`γ` is frozen after training and saved in the checkpoint.  If R² < 0.05 at fit time, a warning
is logged (the OFI→price relationship is too weak for reliable reconstruction).  In LOB mode
this step is unnecessary — the price channel is read directly.

---

## 8. Model Architecture

Penny uses two jointly trained components.

### 2D UNet (inpainting backbone)

The backbone is a `UNet2DModel` from HuggingFace Diffusers — a standard 2D convolutional UNet
with skip connections, group normalization, and optional self-attention blocks.

**Input:** 5 channels — the noisy image (2 channels), a history tensor (the ground-truth
past with the future zeroed out, 2 channels), and the inpainting mask (1 channel).  Concatenating
these along the channel axis lets the UNet see exactly which columns are known and which must
be generated, without any architecture changes.

**Output:** 2 channels — the predicted noise `ε̂` at the same spatial resolution as the input.

**Block structure (4 encoder / 4 decoder blocks):**

| Block | Width | Type |
|---|---|---|
| 1 | 64 | Plain convolution |
| 2 | 128 | Plain convolution |
| 3 | 256 | Plain convolution |
| 4 (bottleneck) | 256 | Self-attention + convolution |

The decoder mirrors the encoder with skip connections.  Self-attention only at the bottleneck
(block 4) keeps computation tractable on a 256×256 image while still allowing global temporal
context.  Each block has 2 convolutional layers with dropout 0.1.  Total trainable parameters
≈ 12 million.

The image is processed at full 256×256 resolution throughout; no explicit temporal downsampling
is applied inside the UNet (the time axis is treated as the width dimension).

### Trend Head

A minimal linear layer with 6 parameters maps the scalar predicted trend ratio `l_pred` to
3-class logits:

```
logits = W · l_pred + b,   W ∈ ℝ^{3×1},  b ∈ ℝ^3
```

This head is trained jointly with the UNet via the trend loss (see Training), forcing the
denoiser to produce futures consistent with a directional signal.

---

## 9. Diffusion Process

Penny uses **Denoising Diffusion Probabilistic Models (DDPM)** for training and
**Denoising Diffusion Implicit Models (DDIM) + RePaint** for inference.

### Forward process (DDPM)

A linear noise schedule interpolates `T_max = 1000` noise levels from
`β₁ = 0.0001` to `β₁₀₀₀ = 0.02`.  The cumulative product `ᾱₜ = ∏ᵢ₌₁ᵗ (1 − βᵢ)`
determines how much signal remains at each timestep.  The closed-form forward diffusion
equation adds Gaussian noise to a clean image `x₀` in a single step:

```
xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε,    ε ~ N(0, I)
```

At t = 0, xₜ ≈ x₀ (almost no noise).  At t = T_max, xₜ ≈ pure noise.

### Reverse process — DDIM sampling (inference)

Instead of running all 1000 reverse steps, DDIM allows a deterministic sub-sequence of
`ddim_steps = 50` timesteps linearly spaced from T_max−1 down to 0.  At each step from
timestep t to t_prev, the UNet predicts the noise `ε̂`, which is used to recover a clean
estimate `x̂₀` via the **Tweedie formula**:

```
x̂₀ = (xₜ − √(1 − ᾱₜ) · ε̂) / √ᾱₜ
```

The DDIM update then steps to t_prev deterministically (η = 0):

```
x_{t_prev} = √ᾱ_{t_prev} · x̂₀ + √(1 − ᾱ_{t_prev}) · ε̂
```

50 DDIM steps produce samples comparable in quality to 1000 DDPM steps at roughly 20× lower
cost.

### RePaint inpainting

At every reverse step, RePaint enforces the known-region constraint by **re-pasting** the
real past columns with the correct noise level for timestep t_prev:

```
known_noised  = √ᾱ_{t_prev} · x₀_past + √(1 − ᾱ_{t_prev}) · ε_new
x_{t_prev}    = mask · ddim_step + (1 − mask) · known_noised
```

Where mask = 1 marks the future (generated) region and mask = 0 marks the past (re-pasted
from real data).  This prevents the generated future from leaking information into the past
columns and ensures the boundary between known and unknown is always sharp and correctly
conditioned on real market data.

The result is that the UNet only needs to generate the future columns; the past is always real.
This is the core of the *Painting the Market* approach.

---

## 10. Training

Training minimizes a combination of two losses on each batch.

### Masked diffusion loss

A random timestep `t ~ Uniform{0, …, T_max−1}` is sampled per image.  The image is noised
via the forward process (`q_sample`), and the UNet predicts the added noise.  The loss is the
mean squared error between the predicted noise and the true noise, **computed only over the
future region** (where mask = 1):

```
L_diff = (1 / |future pixels|) · Σ_{future} (ε̂ − ε)²
```

Restricting the loss to the future region means the UNet is rewarded only for correctly
denoising the inpainted area — it need not recover noise in the known past, which would be
inconsistent with the RePaint re-pasting during inference.

### Trend loss

During training, after the UNet predicts noise at timestep t, the Tweedie formula recovers a
denoised estimate `x̂₀`.  The future mid-price series is reconstructed from `x̂₀` using the
same `painted_future_mid` function used at inference (either reading the LOB price channel or
integrating OFI × γ).  From this reconstructed mid, the predicted trend ratio is computed:

```
fwd_pred = mean(reconstructed mid over first k future steps)
l_pred   = (fwd_pred − bwd_real) / bwd_real
```

where `bwd_real` is the true backward-smoothed mid from the dataset (a constant, not a
gradient-bearing node).  The trend head maps `l_pred` to 3-class logits, and a cross-entropy
loss is computed against the ground-truth label.

**Timestep weighting:** the Tweedie estimate is only a good reconstruction of the clean image
at low noise levels (small t).  At high noise levels (large t), the UNet has almost no signal
and the trend estimate is meaningless.  The trend loss is therefore weighted by:

```
w(t) = (1 − t / T_max)²
```

This weight approaches 1 near t = 0 (clean-ish input, meaningful prediction) and approaches 0
near t = T_max (pure noise, trend gradient suppressed).

### Combined loss

```
L = L_diff + λ · L_trend
```

With `λ = lambda_trend = 0.5`.  The two terms are complementary: L_diff teaches the UNet to
produce realistic LOB structure (the generative quality), while L_trend steers the generated
futures toward directionally consistent market outcomes.

### Optimizer and schedule

**AdamW** over all UNet and trend-head parameters jointly, with learning rate 3×10⁻⁴ and
weight decay 10⁻⁴.  Gradients are clipped to L2 norm 1.0 per step.

The learning-rate schedule is **linear warmup followed by cosine decay**: for the first
`warmup_steps = 200` gradient steps the learning rate rises linearly from 0 to the peak; then
it follows a cosine curve down toward 0 over the remaining steps.  This avoids large early
gradient updates when the model is far from any solution.

---

## 11. Validation

Validation runs at the end of every epoch and computes two separate signals.

### Diffusion loss (early-stopping metric)

The same masked diffusion MSE is computed on the full validation set with the model in eval
mode (dropout disabled).  Gradients are not computed.  This is the **only metric used for
early stopping** — the model checkpoint with the lowest validation diffusion loss is saved.

Early stopping triggers if the validation diffusion loss does not improve for `patience = 20`
consecutive epochs.

### Label accuracy (monitoring)

Running the full DDIM + RePaint sampler over the entire validation set every epoch is expensive.
Instead, a random subset of `val_eval_windows = 50` validation windows is sampled, each
inpainted `n_samples = 20` times.  The modal label across the 20 samples is compared to the
ground-truth label.  The resulting accuracy is logged each epoch as a monitoring signal but
does **not** influence checkpointing or early stopping — it is purely informational.

---

## 12. Test Evaluation

After training completes, the best checkpoint is reloaded and evaluated on the held-out test
set (day 9).  Every test window is inpainted `n_samples = 20` times.  Six metrics are reported:

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of windows where the modal predicted label matches the ground-truth label |
| **Macro F1** | Unweighted mean of per-class F1 scores (equal weight to all three classes) |
| **Confusion matrix** | 3×3 matrix of true-vs-predicted label counts |
| **Trend-ratio Pearson r** | Pearson correlation between mean predicted `l` and ground-truth `l` across all test windows — measures whether the model tracks directional intensity |
| **Mid-price MAE** | Mean absolute error (in IRT) between the mean reconstructed future mid-price and the real mid-price, averaged over the first k steps and all test windows |
| **Spread Wasserstein** | Wasserstein-1 distance between the distribution of painted best-bid/ask spreads and the real spread distribution — **LOB mode only** (OFI mode has no price channel) |

A random baseline achieves ~33.3% accuracy and ~0.333 macro F1.  A model that only learns
directionality but generates unrealistic structure will show high accuracy but poor Wasserstein
distance.

---

## 13. Inference

At inference time, only T_past snapshots of real order-book data are required.  The procedure:

1. Build the `(R, T_total, 2)` image with the past filled in and the future zeroed out.
2. Normalize with the frozen training statistics from the checkpoint.
3. Run the DDIM + RePaint sampler `n_samples = 20` times to generate 20 candidate futures.
4. Reconstruct the future mid-price from each sample (LOB price channel or γ-integrated OFI).
5. Compute the trend ratio `l` for each sample and classify with the frozen `alpha`.
6. Take the **modal label** across the 20 samples as the final prediction.

The inference output includes: the modal label and its name (down/stationary/up), the vote
distribution (how many samples voted for each class), the mean and standard deviation of `l`
across samples, the signal ratio (mean_l / std_l, a conviction measure), and the mean
reconstructed future mid-price trajectory.

---

## 14. Setup and Usage

```bash
# Install dependencies
uv sync

# Fetch data from the Cloudflare R2 DVC remote
uv run dvc pull

# Remove pre-gap rows from nobitex data (check first with --dry-run)
uv run python scripts/trim_gap.py --dry-run
uv run python scripts/trim_gap.py

# Train (OFI mode by default)
uv run python scripts/train_penny.py

# Train in LOB mode
uv run python scripts/train_penny.py configs/config_lob.json

# Inference from a saved checkpoint
uv run python scripts/infer_penny.py \
    --checkpoint checkpoints/ofi_nobitex_BTCIRT_<stamp> \
    --orderbook data/nobitex_data/BTCIRT_orderbook.csv \
    --trades data/nobitex_data/BTCIRT_trades.csv
```

---

## 15. Configuration

All settings live in `configs/config.json` (OFI mode, default) and `configs/config_lob.json`
(LOB mode).

| Field | Default | Description |
|---|---|---|
| `feature_mode` | `"ofi"` | `"ofi"` (per-level OFI) or `"lob"` (price offsets) |
| `exchange` | `"nobitex"` | Exchange name (used for data path and checkpoint naming) |
| `pair` | `"BTCIRT"` | Trading pair |
| `n_levels` | `10` | LOB depth levels per side |
| `snapshot_interval_sec` | `10` | Seconds between order-book snapshots |
| `T_past` | `156` | Observed past window length (~26 min at 10s) |
| `T_future` | `100` | Future window to inpaint (~17 min at 10s) |
| `T_total` | `256` | Total window length = T_past + T_future |
| `padded_size` | `256` | Square size for UNet input (256×256) |
| `train_days` | `6` | Calendar days in training split |
| `val_days` | `2` | Calendar days in validation split |
| `test_days` | `1` | Calendar days in test split |
| `stride` | `30` | Snapshots between window starts (5 min at 10s) |
| `norm_window_snapshots` | `8640` | Rolling window for normalizer fit (~1 day) |
| `clip_percentile` | `0.95` | Outlier clip at this quantile of \|z\| |
| `n_trade_rows` | `3` | Number of trade-feature rows |
| `label_k` | `10` | Smoothing steps each side of the boundary (100s each) |
| `label_alpha` | `-1` | Trend threshold; -1 = auto-calibrate for balanced thirds |
| `beta_start` | `0.0001` | First beta in the DDPM linear schedule |
| `beta_end` | `0.02` | Last beta in the DDPM linear schedule |
| `T_max` | `1000` | Total diffusion timesteps |
| `ddim_steps` | `50` | DDIM reverse steps at inference/evaluation |
| `unet_filters` | `[64,128,256,256]` | Channel widths for the 4 UNet blocks |
| `self_attn_at_block` | `4` | Block index (1-based) with self-attention (bottleneck) |
| `dropout` | `0.1` | Dropout rate inside UNet blocks |
| `lr` | `3e-4` | Peak learning rate (AdamW) |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `warmup_steps` | `200` | Gradient steps for linear warmup |
| `grad_clip` | `1.0` | Gradient L2 norm clip |
| `batch_size` | `8` | Training batch size |
| `epochs` | `200` | Maximum training epochs |
| `patience` | `20` | Early stopping patience (val diffusion loss) |
| `lambda_trend` | `0.5` | Weight of the trend loss relative to the diffusion loss |
| `n_samples` | `20` | Diffusion samples per window at inference/evaluation |
| `val_eval_windows` | `50` | Validation windows sampled for label accuracy each epoch |
| `device` | `"cuda"` | Training device (`"cuda"` or `"cpu"`) |
| `cache_dir` | `"data/cache"` | Directory for the `.npz` dataset cache |
| `checkpoint_root` | `"checkpoints"` | Root directory for checkpoint folders |

---

## 16. Project Structure

```
configs/
  config.json            OFI mode (default training config)
  config_lob.json        LOB mode
data/                    Raw CSV data (DVC-tracked, not in Git)
  nobitex_data/
    BTCIRT_orderbook.csv
    BTCIRT_trades.csv
  ...
scripts/
  train_penny.py         Training entry point
  infer_penny.py         Inference entry point
  trim_gap.py            Drops pre-gap rows from nobitex CSVs
src/penny/
  labels.py              DeepLOB trend ratio, class thresholding, alpha calibration
  features.py            OFI/depth/trade row builder, padding, mask, RollingNormalizer
  dataset.py             Calendar-day split, sliding windows, gamma + alpha fit, cache
  diffusion.py           DDPM schedule, q_sample, DDIM step, RePaint step + sampler
  model.py               build_unet, TrendHead, painted_future_mid
  train.py               Training epoch, validation diffusion loss, validation label accuracy
  evaluate.py            Test metrics (6 metrics)
```

---

## References

- Backhouse et al., *Painting the Market: A Generative Diffusion Model for LOB Simulation*,
  arXiv:2509.05107v1, 2025.
- Ntakaris et al., *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with
  Machine Learning Methods*, Journal of Forecasting, 2018.
- Cont, Kukanov, Stoikov, *The Price Impact of Order Book Events*, Journal of Financial
  Econometrics, 2014.
- Lugmayr et al., *RePaint: Inpainting using Denoising Diffusion Probabilistic Models*,
  CVPR 2022.
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
