"""Pairwise CKA between the XAI layer's three trunks.

Usage::

    uv run python -m xai.run_cka checkpoints/nobitex/BTCIRT \
        --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 \
                 jumpgatelob_levy_BTCIRT_ofi_k10

Collects every model's frozen tap activations over the **same** windows, then
reports a layer × layer linear-CKA matrix for each model pair, with a stability
spread over disjoint subsamples.  Writes ``cka.json``.

Three architectures with different inductive biases reach ~0.67 on this task; the
matrices say whether they converged on similar representations to do it.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from loguru import logger

from crypto.dataset import build_datasets
from utils.training import resolve_device, set_seed
from xai.cka import cka_matrix, cka_stability, format_matrix
from xai.representation import TAPS, collect_activations
from xai.run_ig import _resolve_model

# Data-spec keys that must match for activations to be comparable across models.
_SPEC_KEYS = (
    "symbol",
    "cache_dir",
    "n_lob_levels",
    "feature_mode",
    "T_past",
    "label_k",
    "stride",
    "train_frac",
    "val_frac",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-windows", type=int, default=2048)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    configs = {
        n: json.loads((args.root / n / "config.json").read_text()) for n in args.models
    }

    # CKA compares column spaces over a shared sample axis: if the models were fed
    # different windows the matrix would be meaningless, so refuse rather than
    # silently produce numbers.
    spec = {k: configs[args.models[0]].get(k) for k in _SPEC_KEYS}
    for n, c in configs.items():
        diff = {k: (spec[k], c.get(k)) for k in _SPEC_KEYS if c.get(k) != spec[k]}
        if diff:
            raise ValueError(
                f"{n} has a different data spec, so its activations are not "
                f"comparable: {diff}"
            )

    dataset = None
    acts: dict[str, dict[str, np.ndarray]] = {}
    labels = None
    tap_names: dict[str, list[str]] = {}

    for name in args.models:
        config = configs[name]
        set_seed(config.get("seed", 42))
        device = resolve_device(config.get("device", "cuda"))
        if dataset is None:
            splits = dict(zip(("train", "val", "test"), build_datasets(config)[:3]))
            dataset = splits[args.split]
            idx = np.arange(len(dataset))
            if args.n_windows < len(idx):
                idx = np.random.default_rng(args.seed).choice(
                    idx, args.n_windows, replace=False
                )
                idx.sort()

        model = _resolve_model(args.root / name, config).to(device)
        ckpt = torch.load(
            args.root / name / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        key = type(model).__name__
        taps = TAPS[key]
        a, y = collect_activations(model, dataset, config, device, idx, taps)
        acts[key] = a
        tap_names[key] = [t.name for t in taps]
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise RuntimeError(f"{key} saw different labels; windows are not aligned")
        logger.info("{}: tapped {}", key, {k: v.shape for k, v in a.items()})

        del model, ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_rows = []
    for ka, kb in combinations(acts, 2):
        m = cka_matrix(acts[ka], acts[kb], tap_names[ka], tap_names[kb])
        stab = cka_stability(
            acts[ka],
            acts[kb],
            tap_names[ka],
            tap_names[kb],
            n_splits=args.n_splits,
            seed=args.seed,
        )
        logger.info(
            "CKA {} vs {} (max std over {} disjoint splits: {:.3f})\n{}",
            ka,
            kb,
            args.n_splits,
            stab["max_std"],
            format_matrix(m, tap_names[ka], tap_names[kb], ka, kb),
        )
        out_rows.append(
            {
                "model_a": ka,
                "model_b": kb,
                "taps_a": tap_names[ka],
                "taps_b": tap_names[kb],
                "cka": m.tolist(),
                "split_mean": stab["mean"].tolist(),
                "split_std": stab["std"].tolist(),
                "max_split_std": stab["max_std"],
            }
        )

    # within-model self-similarity: a layer vs itself must be 1.0, and the
    # off-diagonals say how much each trunk changes across its own depth
    for k in acts:
        m = cka_matrix(acts[k], acts[k], tap_names[k], tap_names[k])
        logger.info("CKA {} vs itself\n{}", k, format_matrix(m, tap_names[k], tap_names[k], k, k))
        out_rows.append(
            {
                "model_a": k,
                "model_b": k,
                "taps_a": tap_names[k],
                "taps_b": tap_names[k],
                "cka": m.tolist(),
            }
        )

    out = args.out or args.root / "cka.json"
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_windows": int(len(idx)),
                "seed": args.seed,
                "estimator": "unbiased_hsic",
                "pairs": out_rows,
            },
            indent=2,
        )
    )
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
