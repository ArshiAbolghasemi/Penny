# Running the XAI layer

The whole layer runs from **one command**, designed for an unattended Kaggle
run, or each analysis standalone.

- **Source:** `src/xai/run_all.py` (orchestrator), `scripts/plot_xai.py` (figures)
- **Runners:** `xai.run_ig`, `xai.run_faithfulness`, `xai.run_probes`,
  `xai.run_cka`, `xai.run_agreement`, `xai.run_gate_sweep`

## One command

```bash
# everything, over the three in-scope k10 checkpoints
uv run python -m xai.run_all checkpoints/nobitex/BTCIRT

# choose the checkpoints / output dir explicitly
uv run python -m xai.run_all checkpoints/nobitex/BTCIRT \
    --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 jumpgatelob_levy_BTCIRT_ofi_k10 \
    --out results/xai

# prove the wiring in a couple of minutes on CPU
uv run python -m xai.run_all checkpoints/nobitex/BTCIRT --smoke
```

`run_all` runs every analysis in the [three-layer story](README.md), in order,
over the three in-scope `k10` checkpoints:

1. **Feature attribution** — [IG](attribution.md) per model (zero + mean
   baselines) plus the [deletion/insertion faithfulness](faithfulness.md) check.
2. **Layer attribution** — per-trunk [linear probes](probes.md).
3. **Cross-model comparison** — [linear CKA](cka.md), the [attention-vs-IG
   agreement](agreement.md) table, and the JumpGateLOB-only [gate/robustness
   sweep](gate-sweep.md).

### Design choices that matter for an unattended run

- **Per-model IG artefacts stay next to their checkpoint** (`run_ig` writes
  there), because the probe / CKA / agreement / faithfulness stages all re-read
  the *same* window subsample and it must be identical across them. Everything
  aggregate is *also* copied into `--out` so the whole deliverable is one
  downloadable directory.
- **A failing stage never aborts the run.** Each stage is isolated; its exception
  is logged and recorded in `manifest.json`, and the remaining stages still run.
  A three-hour sweep that dies on the last plot must not throw away the first five
  results.
- **`--smoke`** shrinks every sample size so the full pipeline runs end to end in
  a couple of minutes on CPU. It proves the wiring, **not** the science — smoke
  numbers are not reportable.
- The orchestrator **shells out to the standalone runners' `main()`** rather than
  re-implementing them, so there is exactly one code path per analysis and the
  standalone commands stay first-class.

### Flags

| Flag | Meaning |
|------|---------|
| `--models` | checkpoint dir names (default: the three in-scope k10 models) |
| `--out` | collected-artefacts dir (default `<root>/xai_results`) |
| `--baselines` | IG baselines to run (default `zero mean`) |
| `--split` | `train` / `val` / `test` (default `test`) |
| `--skip` | stages to skip (`ig probes cka agreement faithfulness gate`) |
| `--smoke` | tiny sample sizes; wiring only |
| `--seed` | subsample seed (default 42) |

## Outputs

`run_all` collects one flat `xai_results/` directory:

| Artefact | From |
|----------|------|
| `<model>__ig_zero.{json,npz}`, `__ig_mean.{json,npz}` | [IG](attribution.md) |
| `faithfulness.json` | [faithfulness](faithfulness.md) |
| `probes.json` | [probes](probes.md) |
| `cka.json` | [CKA](cka.md) |
| `agreement.json` | [agreement](agreement.md) |
| `<model>__gate_sweep.json` | [gate sweep](gate-sweep.md) |
| `manifest.json` | per-stage status, timings, sample sizes, provenance |

## Figures

`scripts/plot_xai.py` turns the collected artefacts into the paper's figures,
each written as both `.pdf` (vector, for LaTeX) and `.png` (preview / slides);
missing artefacts are skipped with a warning.

```bash
uv run python scripts/plot_xai.py checkpoints/nobitex/BTCIRT/xai_results \
    --outdir docs/xai/figures
```

| Figure | Content |
|--------|---------|
| `fig_ig_group_shares` | per-model IG feature-group shares |
| `fig_ig_heatmaps` | signed IG `(T, F)` attribution heatmaps |
| `fig_faithfulness` | deletion / insertion curves vs random |
| `fig_probes` | per-layer probe accuracy vs shuffled control |
| `fig_cka` | pairwise CKA matrices |
| `fig_agreement` | attention-vs-IG rank agreement |
| `fig_gate_sweep` | JumpGateLOB gates + robustness vs `t` |

## Environment note

The layer needs [captum](https://captum.ai) (`>=0.8.0,<0.9.0` — 0.9 requires
`torch>=2.3`, which conflicts with the `cu118` extra). On Kaggle the standard
`uv sync --extra <gpu>` install covers it. See the main
[README](../../README.md#setup) for the hardware extras.
