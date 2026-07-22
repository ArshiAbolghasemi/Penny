# Layer attribution — linear probes

*Which layers hold the signal?* Where [attribution](attribution.md) answers
*which inputs* move the logit, this answers *where in a trunk* the trend becomes
linearly decodable, by fitting a linear classifier on frozen intermediate
activations.

- **Reference:** Alain & Bengio, *Understanding intermediate layers using linear
  classifier probes*, 2016.
- **Type:** representation analysis (Layer 2 of the [three-layer story](README.md)).
- **Source:** `src/xai/representation.py`
- **Runner:** `xai.run_probes` → `probes.json`

## Idea

Tap a trunk's activations at its natural architectural boundaries, mean-pool over
time to `(B, d)`, and fit a single `Linear(d, 3)` on each. The probe accuracy is
the claim: *"a linear readout of this layer recovers the trend at rate X."*

**Deliberately linear.** A deeper probe (an MLP, or `models.probe.TemporalProbe`)
can manufacture decodability the layer never had — past some capacity you measure
the probe, not the representation. A single linear map keeps the claim
falsifiable.

## Tap points

Taps are declared per model (not inferred from shape — a square activation would
make any heuristic silently ambiguous). Each declares its own axis layout because
the families differ (CTABL is feature-major, the others time-major):

| Model | Taps (shallow → deep) | Activation |
|-------|-----------------------|------------|
| **CTABL** | `body.bl1` → `body.bl2` | `(B, d, T)` |
| **DLA** | `encoder` → `decoder` | `(B, T, d)` / `(B, d)` |
| **JumpGateLOB** | `gru` → `temporal` → `pool` | `(B, T, d)` / `(B, d)` |

CTABL's `tabl` is deliberately **not** tapped: it emits `(B, 3, 1)` — the logits
themselves — so probing it would measure the classifier, not a representation.
Trunk depths are unequal (CTABL 2, DLA 2, JumpGateLOB 3), so absolute positions
are incomparable across models; the runner reports a **normalised `depth_frac`**
that travels.

## The control is the whole point

Every probe is reported against a **shuffled-label control** fitted the same way
(same activations, permuted training labels). Without it a probe number is
unreadable: high accuracy could just mean the layer is wide, and a linear map
onto 3 classes over enough dimensions can fit noise. The reported `delta`
(real − shuffled) is what carries signal.

Two more anti-leak details:

- activations are standardised with the **training split's** statistics only;
- everything runs through `predict`, so probed activations come from the same
  path the reported test metrics do.

## I/O

- **Input** train + test datasets, their window indices, and the model.
- **Output** one row per tap: `dim`, `accuracy`, `macro_f1`,
  `control_accuracy`, `delta`, plus `depth` / `depth_frac`.

## Running

```bash
uv run python -m xai.run_probes checkpoints/nobitex/BTCIRT \
    --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 jumpgatelob_levy_BTCIRT_ofi_k10 \
    --n-train 8192 --n-test 4096 --epochs 200
```

Probes every tap of each model against its shuffled control and writes
`probes.json`. The cross-model version of this question — do different trunks
reach the same representation — is [CKA](cka.md).
