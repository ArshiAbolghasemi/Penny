# Attention readouts (`return_attn=True`)

The mechanism probes need each architecture's own attention weights. Rather than
re-derive them externally, the three in-scope models grow an **opt-in
`return_attn=True` path** that returns their internal attention alongside the
logits. This is the source-side change the XAI layer rests on.

- **Type:** model instrumentation feeding [agreement](agreement.md) and
  [gate-sweep](gate-sweep.md).
- **Source:** `src/models/ctabl.py`, `src/models/dla.py`,
  `src/models/jumpgatelob.py`, `src/models/modules.py`

## Contract

The flag is **purely additive and off by default**. Every existing caller
(training loops, inference, `AlphaStableLOB`, the two-phase probe) uses the
default path and is untouched; only `src/xai/` passes `return_attn=True`. With
the flag off the forward paths are unchanged and the `predict` contract is
identical, so the readouts cannot drift from the evaluated model.

```python
logits, attn = model(x, return_attn=True)              # CTABL, DLA
logits, attn = model.classify(x, return_attn=True)     # JumpGateLOB
```

## What each model exposes

| Model | Key | Shape | Meaning |
|-------|-----|-------|---------|
| **CTABL** | `temporal` | `(B, d2, t2)` | TABL softmax over the (halved) temporal axis, per output feature |
| | `lam` | scalar | learned mix of attention vs the raw bilinear path (0 = attention unused, 1 = attention only) |
| **DLA** | `input` | `(B, T, F)` | stage-1 weights over input features per timestep |
| | `temporal` | `(B, T_dec, T)` | stage-2 weights over encoder timesteps per decoder step (last row feeds the head) |
| **JumpGateLOB** | `self` | `(B, T, T)` | trunk temporal self-attention |
| | `pool` | `(B, T)` | trend-head pooling weights over timesteps |
| | `gate_attn` / `gate_mlp` | | the `t`-conditioned adaLN-Zero gates |

`AttentionPool` (`modules.py`) grows the same flag, returning its `(B, N)`
pooling weights; `JumpGateLOB.classify` additionally takes an optional `t` to
sweep the noise level for analysis (inference leaves it `None`).

## How the readouts are consumed

- **[agreement](agreement.md)** reduces each to a per-axis vector: DLA's `input`
  (mean over time) gives the feature axis; DLA `temporal` (last step),
  JumpGateLOB `pool`, and CTABL `temporal` (projected through `bl2.W2`) give the
  time axis. CTABL's `lam` is reported directly.
- **[gate-sweep](gate-sweep.md)** reads `gate_attn` / `gate_mlp` across `t`.

## Numerical note

With the flag **off**, CTABL and DLA logits are bit-identical to the pre-change
model. JumpGateLOB differs by ~3e-07 (~3× float32 eps, argmax agreement 1.0)
*only when the flag is on*, because `nn.MultiheadAttention` takes a fused kernel
only with `need_weights=False` — an upstream torch path difference, not a change
in behaviour. Since training and inference never pass the flag, deployed and
trained behaviour is unchanged and no checkpoint is invalidated.
