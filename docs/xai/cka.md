# Cross-model comparison — linear CKA

*Do three different inductive biases learn the same representation?* Where
[probes](probes.md) ask where the trend is decodable *inside* one trunk, CKA
compares trunks **against each other**.

- **Reference:** Kornblith, Norouzi, Lee & Hinton, *Similarity of Neural Network
  Representations Revisited*, ICML 2019; unbiased HSIC from Song et al. 2012.
- **Type:** cross-model representation similarity (Layer 3 of the [three-layer
  story](README.md)).
- **Source:** `src/xai/cka.py`
- **Runner:** `xai.run_cka` → `cka.json`

## Idea

A bilinear network (CTABL), an LSTM dual-attention encoder/decoder (DLA) and a
diffusion-conditioned GRU+attention trunk (JumpGateLOB) all reach ~0.67 test
accuracy on the same data. Do they get there through **similar representations,
or different ones?**

**Linear CKA** answers that. It is invariant to orthogonal transforms and
isotropic scaling — so it does not care that the layers have different widths or
arbitrary basis orientations — while staying sensitive to genuine
representational differences. That invariance is what makes a bilinear layer
comparable to a GRU at all, and because no training is involved there is **no
capacity confound** (unlike a probe).

The runner builds a **layer × layer** CKA matrix between each pair of trunks,
reusing the same tap points as the [probes](probes.md).

## Preconditions, checked not assumed

- **Same windows, same order.** CKA compares column spaces over a shared sample
  axis, so every model must see identical windows in identical order. The runner
  verifies the k10 BTCIRT configs share all ten data-spec keys rather than
  trusting it.
- **Unbiased HSIC.** The biased estimator inflates towards 1 as the sample count
  approaches the feature count — which would manufacture agreement exactly where
  these wide layers (`d` up to 128) are most at risk of it. The unbiased
  estimator is the default.

## Is the matrix trustworthy?

CKA depends on the sample count, so a value computed once says nothing about how
much of it is noise. `cka_stability` recomputes the matrix over disjoint sample
subsets and returns the **spread** (`mean`, `std`, `max_std`): a wide spread
means the heatmap is not a result. This travels with every reported matrix.

## I/O

- **Input** two models' tap activations `{tap: (N, d)}` over the same `N` windows.
- **Output** an `(A, B)` similarity matrix in `[0, 1]` per model pair (1 = equal
  up to an orthogonal transform and isotropic scale), plus the stability spread.

## Running

```bash
uv run python -m xai.run_cka checkpoints/nobitex/BTCIRT \
    --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 jumpgatelob_levy_BTCIRT_ofi_k10 \
    --n-windows 2048 --n-splits 4
```

Writes the pairwise matrices and their stability spread to `cka.json`. The other
Layer-3 view — whether a model's *attention* agrees with its attributions — is
[agreement](agreement.md).
