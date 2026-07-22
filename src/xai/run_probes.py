"""Per-layer linear probes for the XAI layer's three models.

Usage::

    uv run python -m xai.run_probes checkpoints/nobitex/BTCIRT \
        --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 \
                 jumpgatelob_levy_BTCIRT_ofi_k10

Taps each trunk's frozen intermediate activations and fits a linear classifier on
each one, against a shuffled-label control.  Writes ``probes.json``.

Answers where trend information becomes linearly decodable inside each trunk —
a claim about layers, not inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from loguru import logger

from crypto.dataset import build_datasets
from utils.training import resolve_device, set_seed
from xai.representation import format_table, probe_layers
from xai.run_ig import _resolve_model


def _subsample(n: int, k: int, seed: int) -> np.ndarray:
    idx = np.arange(n)
    if k < n:
        idx = np.random.default_rng(seed).choice(idx, k, replace=False)
        idx.sort()
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict] = []
    cache: dict[tuple, object] = {}
    for name in args.models:
        ckpt_dir = args.root / name
        config = json.loads((ckpt_dir / "config.json").read_text())
        set_seed(config.get("seed", 42))
        device = resolve_device(config.get("device", "cuda"))

        key = (
            config["symbol"],
            config["cache_dir"],
            config["n_lob_levels"],
            config.get("feature_mode", "ofi"),
            config["T_past"],
            config["label_k"],
            config["stride"],
            config["train_frac"],
            config["val_frac"],
        )
        if key not in cache:
            cache[key] = dict(zip(("train", "val", "test"), build_datasets(config)[:3]))
        splits = cache[key]

        model = _resolve_model(ckpt_dir, config).to(device)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

        tr_idx = _subsample(len(splits["train"]), args.n_train, args.seed)
        te_idx = _subsample(len(splits["test"]), args.n_test, args.seed)

        rows.extend(
            probe_layers(
                model,
                splits["train"],
                splits["test"],
                config,
                device,
                tr_idx,
                te_idx,
                seed=args.seed,
                epochs=args.epochs,
            )
        )

        del model, ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(
        "linear probes on frozen activations "
        "(delta = accuracy above a shuffled-label control)\n{}",
        format_table(rows),
    )
    out = args.out or args.root / "probes.json"
    out.write_text(
        json.dumps(
            {
                "n_train": args.n_train,
                "n_test": args.n_test,
                "epochs": args.epochs,
                "seed": args.seed,
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
