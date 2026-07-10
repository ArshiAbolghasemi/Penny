# Models

Penny frames LOB **trend forecasting** (`down / flat / up`) two ways:

1. **Discriminative baselines** — supervised classifiers trained with plain
   cross-entropy. Established LOB architectures, reproduced for comparison.
2. **Joint generative–discriminative models** — the core of the project. A single
   backbone is trained to be **both** a generative diffusion model (denoising /
   score-matching / consistency over the LOB window) **and** a trend classifier,
   sharing one representation. The generative objective acts as a powerful auxiliary
   that shapes features the discriminative head reuses; at inference only the cheap
   feature-only classification path runs.

Every model exposes the **same inference contract** so they are directly comparable:

```python
model.predict(batch, device) -> logits  # (B, 3)  →  0=down, 1=flat, 2=up
```

and every model is scored by the same `utils/evaluate.run_test` (accuracy, macro-F1,
confusion matrix, mean class probabilities).

## Shared training protocol

All families train under one seed-controlled protocol (`utils/training.py`):

- **Optimiser** AdamW, **LR schedule** linear warmup → cosine decay.
- **Reproducibility** seeded Python/NumPy/Torch + seeded DataLoader workers;
  `PENNY_SEED` env override for multi-seed runs.
- **Selection** early stopping on validation cross-entropy (macro-F1 for
  JumpGateLOB), best checkpoint restored before the test pass.
- **Device** `cuda → mps → cpu` graceful fallback.
- **Data** the shared pipeline in [../data](../data/README.md).

Each run writes `best.pt`, `config.json` and `training_log.json` to a timestamped
`checkpoint_dir`, then logs test metrics.

## Model catalogue

### Discriminative baselines

| Model | One-liner | Doc |
|-------|-----------|-----|
| **DeepLOB**  | CNN + Inception + LSTM | [deeplob.md](deeplob.md) |
| **CTABL**    | Temporal Attention-augmented Bilinear Network | [ctabl.md](ctabl.md) |
| **BiN-CTABL**| CTABL with an adaptive Bilinear-Normalisation front-end | [binctabl.md](binctabl.md) |
| **TLOB**     | Temporal-LOB transformer (alternating temporal/spatial attention) | [tlob.md](tlob.md) |
| **DLA**      | Dual-Stage Temporal Attention (DA-RNN) | [dla.md](dla.md) |
| **Axial-LOB**| Axial-attention over the LOB image | [axiallob.md](axiallob.md) |

### Joint generative–discriminative

| Model | One-liner | Doc |
|-------|-----------|-----|
| **JointDiT** | Diffusion Transformer trained to jointly denoise + classify; ships with five training objectives (DDPM, consistency, t-EDM, drift, Lévy) plus a two-phase probe variant | [jointdit.md](jointdit.md) |
| **JumpGateLOB** | Lévy, jump-aware joint diffusion-classifier built for feature-only trend inference | [jumpgatelob.md](jumpgatelob.md) |
| **AlphaStableLOB** | JumpGateLOB's GRU+attention trunk with a genuine **α-stable** (Lévy-stable, power-law-tailed) forward process, trained by generalized score matching | [alphastablelob.md](alphastablelob.md) |

## Shared building blocks

`models/modules.py` centralises pieces reused across families: `BiN` (bilinear
normalisation), `AttentionPool`, sinusoidal timestep embeddings, U-Net conv blocks,
and the JumpGate `NoiseStateEstimator` / `DiffusionStepMLP` / `LevelAttention`.

The diffusion machinery behind the joint models:

- `models/ddpm.py` — minimal linear-β DDPM scheduler.
- `models/alphastable.py` — α-stable (Lévy-stable) forward process + tabulated generalized score.
- `models/consistency.py` — EDM/Karras preconditioning + Consistency-Training helpers.
- `models/drift.py` — one-step "Generative Modeling via Drifting" loss + memory bank.
- `models/probe.py` — backbone-agnostic two-phase (generative → frozen-trunk probe) machinery.
- `src/levy/` — Lévy jump-diffusion forward process + tabulated generalized score.

These are documented in context within [jointdit.md](jointdit.md) and
[jumpgatelob.md](jumpgatelob.md).
