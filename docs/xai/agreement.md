# Attention vs IG — rank agreement

*Do the architectures' attention mechanisms agree with what actually drives
them?* This is the layer's **headline**: where a model's routing and its
attributions disagree, the routing was not the explanation.

- **Reference:** Jain & Wallace, *Attention is not Explanation*, NAACL 2019;
  Wiegreffe & Pinter, *Attention is not not Explanation*, EMNLP 2019.
- **Type:** mechanism-vs-attribution agreement (Layer 3 of the [three-layer
  story](README.md)).
- **Source:** `src/xai/agreement.py`
- **Runner:** `xai.run_agreement` → `agreement.json`

## Idea

Attention says what a model *routed*; [Integrated Gradients](attribution.md) says
what *changed the logit*. They are different claims, and the literature is
explicit that the first does not license the second. Rather than assert either
way, this module **measures** it: reduce both signals to a vector over the same
axis and rank-correlate them.

Correlations use **Spearman** (monotone rank agreement — the question actually
being asked) with **Kendall's tau** alongside as the tie-robust check.

## What can honestly be compared

The three models do not expose attention over the same axis, so there is **no
single agreement number** for all of them — pretending otherwise would be the
decorative version of this analysis.

| Axis | Available for | How the attention side is built |
|------|---------------|---------------------------------|
| **time** (`T_past`) | all three → the cross-model comparison | DLA stage-2 `temporal` (last decoder step); JumpGateLOB `pool`; CTABL TABL `temporal` **projected back** through `bl2.W2` |
| **feature** (`n_features`) | **DLA only** | DLA stage-1 `input`, mean over time |

CTABL's TABL attention lives on the halved `t2` axis *after* the second bilinear
layer, so it is mapped onto input timesteps through `|bl2.W2|` — a reshape would
be wrong. CTABL's and JumpGateLOB's attention have no input-feature axis to
compare, and **inventing one** by projecting through learned weights would
measure the projection, not the model — so the feature row is DLA-only.

## Dispersion travels with every number

A rank correlation over a nearly flat weight vector is dominated by noise: the
ranks are real, but the differences being ranked are negligible. Every
correlation carries a **dispersion** ratio (std ÷ uniform weight): near 0 the
mechanism barely discriminated, so its `rho` should not be read as a strong claim
either way.

## I/O

- **Input** an IG result (from [attribution](attribution.md)) and the attention
  readouts collected on the *same* windows.
- **Output** one row per model: the `time` agreement (Spearman/Kendall + p +
  dispersion), the `feature` agreement (DLA only, else `None`), and CTABL's
  learned `lam` mixing weight.

## Running

```bash
uv run python -m xai.run_agreement checkpoints/nobitex/BTCIRT \
    --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 jumpgatelob_levy_BTCIRT_ofi_k10 \
    --baseline zero --n-windows 2048 --n-steps 128
```

Collects each model's attention readouts and IG on the same windows,
rank-correlates them and writes `agreement.json`. The attention readouts
themselves are documented in [attention-readouts.md](attention-readouts.md).
