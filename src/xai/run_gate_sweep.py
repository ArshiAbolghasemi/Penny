"""JumpGateLOB adaLN-Zero gate / robustness sweep over the diffusion timestep.

Usage::

    uv run python -m xai.run_gate_sweep \
        checkpoints/nobitex/BTCIRT/jumpgatelob_levy_BTCIRT_ofi_k10

Sweeps ``t`` and reports (a) how hard the trunk's gates write into the residual
stream on clean windows, and (b) accuracy on jump-noised windows at both the
deployed ``t=0`` conditioning and an oracle that is told the noise level.  Writes
``gate_sweep.json``.

JumpGateLOB only — the probe reads adaLN-Zero gates, which CTABL and DLA do not
have.
"""

from __future__ import annotations

import argparse
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
from xai.gate_sweep import (
    build_forward_process,
    format_table,
    gate_sweep,
    low_t_boundary,
    robustness_sweep,
    sweep_timesteps,
)
from xai.run_ig import _resolve_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", type=Path)
    ap.add_argument("--n-windows", type=int, default=1024)
    ap.add_argument("--n-points", type=int, default=11)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    config = json.loads((ckpt_dir / "config.json").read_text())
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "cuda"))

    model = _resolve_model(ckpt_dir, config).to(device)
    if type(model).__name__ != "JumpGateLOB":
        raise ValueError(
            f"the gate sweep reads adaLN-Zero gates, which only JumpGateLOB has; "
            f"got {type(model).__name__}"
        )
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    splits = dict(zip(("train", "val", "test"), build_datasets(config)[:3]))
    dataset = splits[args.split]
    idx = np.arange(len(dataset))
    if args.n_windows < len(idx):
        idx = np.random.default_rng(args.seed).choice(idx, args.n_windows, replace=False)
        idx.sort()

    fp = build_forward_process(config, device)
    t_max = config.get("T_max", 1000)
    ts = sweep_timesteps(t_max, args.n_points)
    boundary = low_t_boundary(fp)
    logger.info(
        "sweeping t over {} points in [0, {}); robust loss trained on t <= {}",
        len(ts),
        t_max,
        boundary,
    )

    gates = gate_sweep(model, dataset, config, device, idx, ts)
    robust = robustness_sweep(model, dataset, config, device, idx, ts, fp, seed=args.seed)
    logger.info("JumpGateLOB t-sweep\n{}", format_table(gates, robust, boundary))

    out = args.out or ckpt_dir / "gate_sweep.json"
    out.write_text(
        json.dumps(
            {
                "model": ckpt_dir.name,
                "split": args.split,
                "n_windows": int(len(idx)),
                "t_max": t_max,
                "low_t_boundary": boundary,
                "gates": gates,
                "robustness": robust,
            },
            indent=2,
        )
    )
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
