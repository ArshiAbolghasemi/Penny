# OF-SATNet — Order-Flow Single-Asset Transformer

Multi-axis attention Transformer over temporal and order-book-level dimensions of
per-level order flow imbalance (OFI). The single-asset ablation of OF-MATNet.

- **Reference:** Bandealinaeini, Sharifkhani & Salavati, *Attention-Based Multi-Asset
  Order Flow Networks for Enhanced Mid-Price Prediction*, ICAIF '25
  (ACM 3768292.3770430).
- **Type:** discriminative classifier.
- **Source:** `src/models/ofsatnet.py`
- **Trainers:** `crypto.train_ofsatnet`, `stocks.feishu.train_ofsatnet`

## Idea

The paper's full model, **OF-MATNet**, attends over three axes of a multi-asset OFI
tensor — **temporal**, **cross-asset**, and **order-book level** — where the
cross-asset axis is built from peer assets pre-selected via rolling-window Granger
causality. **OF-SATNet is the paper's own single-asset baseline** (Section 5.3):
with only one asset (`N = 1`), cross-asset attention carries no information, so it
drops out and only the **Temporal** and **Level** paths remain. This repo implements
that single-asset variant only — the cross-asset / Granger-causality machinery is
out of scope by design, not an omission.

Each remaining path is a standard `nn.TransformerEncoder` with sinusoidal positional
encoding, applied along a different reshaping of the input window:

- **Temporal path** — sequence of `T` per-timestep feature vectors; attention finds
  temporal dependencies in how the full feature vector evolves.
- **Level path** — sequence of `L` per-level OFI series (the first `L` feature
  columns, each transposed to a length-`T` series); attention finds dependencies
  across order-book depth.

The two pooled path embeddings are concatenated and mapped to the output —
`h_final = Concat(h_T, h_L)` (Eq. 7 in the paper, minus the `h_N` cross-asset term).

## Architecture

```mermaid
flowchart TD
    X["input (B, 1, T, F)"] --> S["squeeze -> (B, T, F)"]

    S --> TP["Temporal path: (B, T, F)"]
    S --> LS["slice first L feature cols, transpose -> (B, L, T)"]

    subgraph TEMP["Temporal attention axis"]
        direction TB
        T1["Linear(F -> D) + sinusoidal PE"] --> T2["TransformerEncoder (T tokens)"]
        T2 --> T3["mean-pool -> h_T in R^D"]
    end
    TP --> TEMP

    subgraph LEVEL["Level attention axis"]
        direction TB
        L1["Linear(T -> D) + sinusoidal PE"] --> L2["TransformerEncoder (L tokens)"]
        L2 --> L3["mean-pool -> h_L in R^D"]
    end
    LS --> LEVEL

    TEMP --> CAT["Concat(h_T, h_L) in R^2D"]
    LEVEL --> CAT
    CAT --> HEAD["Dropout -> Linear(-> 3)"]
    HEAD --> O["logits (B, 3)"]
```

## I/O

- **Input** `(B, 1, T_past, n_features)`. The first `ofsatnet_levels` feature
  columns must be the per-level OFI slice (true for `feature_mode: "ofi"` in both
  the crypto and Feishu feature pipelines — see `crypto/features.py` /
  `stocks/feishu/features.py`).
- **Output** `(B, 3)` trend logits.

## Config keys

| Key | Meaning | Default |
|-----|---------|---------|
| `ofsatnet_levels`  | order-book levels `L` forming the Level path (paper: `L=10`); must be `<= n_features` | 10 |
| `ofsatnet_dim`     | shared projection dim `D`                        | 64 |
| `ofsatnet_heads`   | attention heads per Transformer encoder           | 4 |
| `ofsatnet_layers`  | Transformer encoder layers per path               | 2 |
| `ofsatnet_ff_dim`  | feed-forward dim inside each encoder layer        | `4 * ofsatnet_dim` |
| `ofsatnet_dropout` | dropout (encoder + head)                          | 0.1 |

## Training

Supervised cross-entropy under the shared protocol (this repo's benchmark task is
3-class trend classification; the paper's own objective is MSE on the next-step
return — see [README](README.md#shared-training-protocol)).

```bash
uv run python -m crypto.train_ofsatnet configs/crypto/nobitex/ofsatnet/usdtirt_ofi_k10.json
uv run python -m stocks.feishu.train_ofsatnet configs/stocks/feishu/ofsatnet_ofi.json
```
