# Penny

**Penny trains diffusion models that are, at the same time, discriminative and
generative** — a single backbone learns to *denoise* a limit-order-book (LOB) window
(the generative objective) **and** to *classify* its short-term price direction (the
discriminative objective), sharing one representation. The generative diffusion task
regularises the encoder so the features the classifier reuses are richer than plain
supervised training would produce; at inference only the cheap classification path
runs.

The project forecasts a 3-class LOB **trend** (`down / flat / up`) on crypto data from
**Binance** (USDT pairs) and **Nobitex** (Iranian Toman pairs), and ships a suite of
established **discriminative baselines** alongside the joint models for comparison.

- **Joint generative–discriminative models:** JointDiT (Diffusion Transformer, with
  five training objectives — DDPM, consistency, t-EDM, drift, Lévy — plus a two-phase
  probe), JumpGateLOB (jump-diffusion score matching + noise-consistent classification,
  feature-only inference), and AlphaStableLOB (the same trunk with a genuine α-stable,
  power-law-tailed forward process).
- **Discriminative baselines:** DeepLOB, CTABL, BiN-CTABL, TLOB, DLA, Axial-LOB.

See **[docs/models](docs/models/README.md)** for every model and **[docs/data](docs/data/README.md)**
for the data pipeline.

## Setup

Requirements: **Python 3.10** and [`uv`](https://docs.astral.sh/uv/).

Dependencies (notably PyTorch) are installed via a hardware **extra** — pick the one
matching your machine:

```bash
uv sync --extra cpu      # CPU-only (local dev / non-GPU nodes)
uv sync --extra mps      # Apple Silicon (Metal)
uv sync --extra cu118    # CUDA 11.8  (Pascal/Volta/Turing)
uv sync --extra cu126    # CUDA 12.6  (Turing/Ampere/Ada)
uv sync --extra cu128    # CUDA 12.8
uv sync --extra cu130    # CUDA 13.0  (Blackwell; PyTorch 2.11 default)
```

On SLURM the extra is auto-detected from the GPU compute capability (see
[Running on SLURM](#running-on-slurm)).

## Get the data (DVC)

Raw and resampled market data are **not** in git — they are tracked with
[DVC](https://dvc.org) on an S3-compatible **Cloudflare R2** bucket (`penny`).

```bash
uv run dvc pull      # download data/ from the remote
uv run dvc push      # upload new/changed data
```

> **DVC credentials:** the R2 access keys are **not** stored in the repo — they belong
> in `.dvc/config.local` (git-ignored). **Contact the team** to obtain the
> credentials, drop them into `.dvc/config.local`, then run `uv run dvc pull`.

If you have raw exchange dumps but no resampled parquet, regenerate them:

```bash
uv run python scripts/resample_binance.py --interval 10 --levels 10
uv run python scripts/resample_nobitex.py --levels 20
```

Details: [docs/data](docs/data/README.md).

## Train a model

Every run is driven by a single JSON config under `configs/`. The invocation is
`python -m crypto.train_<model> <config.json>`:

```bash
# discriminative baseline
uv run python -m crypto.train_deeplob   configs/crypto/binance/deeplob/btcusdt_ofi.json

# joint generative–discriminative (base DDPM objective)
uv run python -m crypto.train_jointdit  configs/crypto/nobitex/jointdit/btcirt_ofi_k10.json

# JointDiT alternative training objectives (same backbone)
uv run python -m crypto.train_jointdit_cm    configs/crypto/nobitex/jointdit_cm/btcirt_ofi_k10.json
uv run python -m crypto.train_jointdit_tedm  configs/crypto/nobitex/jointdit_tedm/btcirt_ofi_k10.json --nu 5
uv run python -m crypto.train_jointdit_drift configs/crypto/nobitex/jointdit_drift/btcirt_ofi_k10.json
uv run python -m crypto.train_jointdit_levy  configs/crypto/nobitex/jointditlevy/btcirt_ofi_k10.json

# Lévy jump-aware joint model (feature-only inference)
uv run python -m crypto.train_jumpgatelob    configs/crypto/nobitex/jumpgatelob/btcirt_ofi_k10.json

# α-stable joint model (heavy, power-law-tailed Lévy noise + generalized score matching)
uv run python -m crypto.train_alphastablelob configs/crypto/nobitex/alphastablelob/btcirt_ofi_k10.json

# classical statistical baselines (no learned weights, walk-forward re-fit)
uv run python -m crypto.train_arima configs/crypto/nobitex/arima/btcirt_ofi_k10.json
uv run python -m crypto.train_var   configs/crypto/nobitex/var/btcirt_ofi_k10.json
```

Each run builds/loads the feature cache, trains with early stopping, restores the best
checkpoint, and logs test metrics. Outputs (`best.pt`, `config.json`,
`training_log.json`, `train.log`) go to a timestamped folder under the config's
`checkpoint_dir`.

Configs are organised as `configs/crypto/{exchange}/{model}/{symbol}_{mode}_{k}.json`,
where `mode ∈ {ofi, lob}` is the feature representation and `k` is the label horizon.
Multi-seed runs: set `PENNY_SEED=…` to override the config seed without editing files.

## Running on SLURM

Batch scripts under `slurm/` submit any config to a GPU node and auto-select the
CUDA/CPU extra from the detected GPU:

```bash
sbatch slurm/nobitex/btcirt/k10/jointdit_ofi.slurm
CONFIG=configs/crypto/binance/deeplob/btcusdt_lob.json sbatch slurm/binance/btcusdt/k10/deeplob_lob.slurm
```

Layout mirrors the configs: `slurm/{exchange}/{symbol}/{k}/{model}_{mode}.slurm`
(per-symbol horizon sweeps) plus flat `slurm/{exchange}/{symbol}/` folders where a
symbol has no horizon sweep.

## Documentation

| Area | Start here |
|------|-----------|
| **Data** — exchanges, resampling, features, labels, windowing, normalisation | [docs/data/README.md](docs/data/README.md) |
| **Models** — every architecture + training procedure (with diagrams) | [docs/models/README.md](docs/models/README.md) |

Per-topic files:

- Data: [binance](docs/data/binance.md) · [nobitex](docs/data/nobitex.md) ·
  [features](docs/data/features.md) · [labels](docs/data/labels.md) ·
  [windows & normalisation](docs/data/windows-and-normalization.md)
- Models: [DeepLOB](docs/models/deeplob.md) · [CTABL](docs/models/ctabl.md) ·
  [BiN-CTABL](docs/models/binctabl.md) · [TLOB](docs/models/tlob.md) ·
  [DLA](docs/models/dla.md) · [Axial-LOB](docs/models/axiallob.md) ·
  [JointDiT](docs/models/jointdit.md) · [JumpGateLOB](docs/models/jumpgatelob.md) ·
  [AlphaStableLOB](docs/models/alphastablelob.md) · [ARIMA](docs/models/arima.md) ·
  [VAR](docs/models/var.md)

## Repository layout

```
configs/          per-model / per-symbol / per-horizon JSON configs
scripts/          resample_binance.py, resample_nobitex.py, find_alpha.py
slurm/            SLURM batch scripts mirroring configs/
src/
  crypto/         data pipeline (features, labels, dataset, loader) + train_*.py entry points
  models/         model architectures + shared diffusion machinery (ddpm, consistency, drift, probe)
  levy/           Lévy jump-diffusion forward process + tabulated generalized score
  stocks/feishu/  equity (A-share) pipeline — DeepLOB only
  utils/          training loop helpers, evaluation, FLOPs, PCGrad
docs/             this documentation
data/             DVC-tracked (not in git)
```
