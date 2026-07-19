# Explainability (XAI)

The XAI layer asks **why** three of Penny's trend classifiers predict what they
predict, and whether their internal mechanisms actually explain their outputs.
Every analysis is deterministic, config-driven and reads a trained checkpoint's
own `config.json`, so a run reproduces from the checkpoint plus the DVC-tracked
data alone.

Scope is deliberately narrow: **JumpGateLOB, CTABL and DLA** on the
nobitex/BTCIRT `k10` checkpoints. JointDiT and AlphaStableLOB are out of scope,
and only the `k=10` horizon is covered — keeping the comparison honest matters
more than breadth here. See [Scope](#scope) for why.

The layer keeps **two views deliberately distinct** and reports their agreement
rather than either on its own:

1. **Attribution** — Integrated Gradients over the input window via the shared
   `predict(batch, device) → (B, 3)` contract. This is *what moved the logit*,
   and it is the reference every other view is measured against.
2. **Mechanism probes** — each architecture's own attention readouts, reached
   with `return_attn=True`. These show *what the model routed*, which is not the
   same claim as what changed the output (Jain & Wallace, *Attention is not
   Explanation*), so they are never reported as attributions.

The headline is their **agreement**: where a model's routing and its
attributions disagree, the routing was not the explanation.

## The three-layer story

The analyses stack into one narrative, and the one-command runner
([run-all.md](run-all.md)) executes them in this order:

```
Layer 1  Feature attribution  — which inputs move the logit?
         · Integrated Gradients (zero + mean baselines)        attribution.md
         · deletion / insertion faithfulness (the sanity check) faithfulness.md

Layer 2  Layer attribution     — which layers hold the signal?
         · per-trunk linear probes on frozen activations        probes.md

Layer 3  Cross-model comparison — do different biases learn the same thing?
         · pairwise linear CKA between the three trunks          cka.md
         · attention-vs-IG rank agreement                        agreement.md
         · JumpGateLOB-only adaLN-Zero gate / robustness sweep   gate-sweep.md
```

Each layer earns the next: attribution numbers are only readable once
faithfulness shows the ranking carries behaviour; the probe answers a different
question (*where*, not *which input*) that attribution cannot; and the
cross-model layer only makes sense once each model has been characterised on its
own.

## Shared design principles

Every analysis in the layer follows the same rules, and they are the point:

- **Ride the inference contract.** Attribution, probes and CKA all run through
  the same `predict`/`classify` path the reported test metrics came from, so the
  numbers cannot drift from the evaluated model. The thin `_PredictWrapper` only
  drops the joint models' `@torch.no_grad()` guard so IG can backprop —
  the computation is otherwise identical.
- **Always report a control.** A probe is read against a shuffled-label fit; a
  faithfulness curve against a random-ranking band; an attention correlation
  against its own dispersion. A number without its control is unreadable.
- **Name the baseline.** IG explains `F(x) − F(baseline)`, so the baseline is a
  modelling decision. The default is **zero = no order flow** (raw OFI is exactly
  zero in 25–36% of bins, so it sits on the most common real state, not
  off-manifold); `mean` is provided as the robustness check and agrees at
  Spearman ≈ 0.99.
- **Only compare what is comparable.** The three models do not expose attention
  over the same axis, so there is no single agreement number for all of them;
  the time axis is shared by all three, the feature axis only by DLA. Inventing a
  shared axis by projecting through learned weights would measure the projection,
  not the model.

## Running the layer

The whole layer runs from one command (Kaggle-ready), or each analysis
standalone. Full details in [run-all.md](run-all.md).

```bash
# everything, over the three in-scope k10 checkpoints
uv run python -m xai.run_all checkpoints/nobitex/BTCIRT

# prove the wiring in a couple of minutes on CPU (numbers are not reportable)
uv run python -m xai.run_all checkpoints/nobitex/BTCIRT --smoke

# turn the collected artefacts into the paper's figures
uv run python scripts/plot_xai.py checkpoints/nobitex/BTCIRT/xai_results \
    --outdir docs/xai/figures
```

`run_all` collects every artefact into one `xai_results/` directory
(`probes.json`, `cka.json`, `agreement.json`, `faithfulness.json`, per-model
`ig_*.json/.npz` and `gate_sweep.json`, plus a `manifest.json`), and a failing
stage never aborts the rest of the run.

## Attribution engine

Integrated Gradients comes from [captum](https://captum.ai), pinned
`>=0.8.0,<0.9.0`: captum 0.9 requires `torch>=2.3`, which conflicts with the
`cu118` extra (torch 2.1.2). The IG implementation is otherwise model-agnostic —
one code path covers all three models because they share the `predict` contract.

## Scope

The layer covers exactly the models where the three-way comparison is
meaningful and the checkpoints are settled:

| Model | In scope | Why |
|-------|----------|-----|
| **CTABL** | ✅ | bilinear + TABL attention — a distinct inductive bias to compare |
| **DLA** | ✅ | LSTM dual-stage attention — the only model attending over input features |
| **JumpGateLOB** | ✅ | diffusion-conditioned GRU+attention — carries the adaLN-Zero gates |
| AlphaStableLOB | ❌ | shares JumpGateLOB's trunk; adds nothing to the comparison |
| JointDiT | ❌ | excluded by team decision — not part of the explainability story |

Only the `k=10` horizon is analysed in this phase.

## Reference

| File | Contents |
|------|----------|
| [attribution.md](attribution.md) | Integrated Gradients, the zero/mean baselines, feature groups |
| [faithfulness.md](faithfulness.md) | Deletion / insertion curves, the random control |
| [probes.md](probes.md) | Per-layer linear probes on frozen activations, tap points |
| [cka.md](cka.md) | Pairwise linear CKA between the three trunks, HSIC, stability |
| [agreement.md](agreement.md) | Attention vs IG rank agreement, what can honestly be compared |
| [gate-sweep.md](gate-sweep.md) | JumpGateLOB adaLN-Zero gate + robustness sweep over `t` |
| [attention-readouts.md](attention-readouts.md) | The `return_attn=True` mechanism probes each model exposes |
| [run-all.md](run-all.md) | The one-command runner, standalone runners, figures |
