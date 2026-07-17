"""One-command driver for the whole XAI layer — the entry point meant for Kaggle.

Usage::

    uv run python -m xai.run_all checkpoints/nobitex/BTCIRT
    uv run python -m xai.run_all checkpoints/nobitex/BTCIRT --smoke
    uv run python -m xai.run_all checkpoints/nobitex/BTCIRT \
        --models ctabl_BTCIRT_ofi_k10 dla_BTCIRT_ofi_k10 \
                 jumpgatelob_levy_BTCIRT_ofi_k10 --out results/xai

Runs every analysis in the three-layer XAI story, in order, over the three
in-scope k10 checkpoints, and collects the outputs so a single Kaggle run
produces everything the paper needs:

1. **Feature attribution** (saliency) — Integrated Gradients per model, zero and
   mean baselines (:mod:`xai.run_ig`), plus the deletion/insertion faithfulness
   check that licenses reading those numbers (:mod:`xai.run_faithfulness`).
2. **Layer attribution** (representation) — per-trunk linear probes on frozen
   activations against a shuffled control (:mod:`xai.run_probes`).
3. **Cross-model comparison** — linear CKA between the three trunks
   (:mod:`xai.run_cka`) and the attention-vs-IG agreement table
   (:mod:`xai.run_agreement`), plus the JumpGateLOB-only adaLN-Zero gate /
   robustness sweep (:mod:`xai.run_gate_sweep`).

Design choices that matter for an unattended Kaggle run:

* **Per-model IG artefacts stay next to their checkpoint** (``run_ig`` writes
  there), because the probe / CKA / agreement / faithfulness stages all re-read
  the same window subsample and it must be identical across them. Everything
  *aggregate* (probes.json, cka.json, agreement.json, faithfulness.json,
  gate_sweep.json, manifest.json) is also copied into ``--out`` so the whole
  deliverable is one downloadable directory.
* **A failing stage does not abort the run.** Each stage is isolated; its
  exception is logged and recorded in the manifest, and the remaining stages
  still run. A three-hour sweep that dies on the last plot should not throw away
  the first five results.
* **``--smoke``** shrinks every sample size so the full pipeline runs end to end
  in a couple of minutes on CPU. It proves the wiring, not the science: the
  numbers from a smoke run are not meant to be reported.

This module deliberately shells out to the existing single-analysis runners via
their ``main()`` rather than re-implementing them, so there is exactly one code
path per analysis and the standalone commands stay first-class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from loguru import logger

# The three models in the XAI layer's scope, in the order they appear in tables.
# JointDiT / AlphaStableLOB and the k!=10 checkpoints are deliberately excluded.
DEFAULT_MODELS = (
    "ctabl_BTCIRT_ofi_k10",
    "dla_BTCIRT_ofi_k10",
    "jumpgatelob_levy_BTCIRT_ofi_k10",
)

# The one model that carries adaLN-Zero gates; the gate sweep runs on it alone.
GATE_MODEL_HINT = "jumpgatelob"

# Sample sizes for a real run vs. a smoke run. The smoke numbers are chosen to
# exercise every branch (batching, multi-baseline, controls) while finishing in
# minutes on a 4-core CPU box.
FULL = {
    "ig_windows": 2048,
    "ig_steps": 128,
    "probe_train": 8192,
    "probe_test": 4096,
    "probe_epochs": 200,
    "cka_windows": 2048,
    "cka_splits": 4,
    "agree_windows": 2048,
    "agree_steps": 128,
    "faith_windows": 1024,
    "faith_steps": 64,
    "faith_points": 11,
    "faith_random": 5,
    "gate_windows": 1024,
    "gate_points": 11,
}
SMOKE = {
    "ig_windows": 64,
    "ig_steps": 16,
    "probe_train": 256,
    "probe_test": 128,
    "probe_epochs": 30,
    "cka_windows": 128,
    "cka_splits": 2,
    "agree_windows": 64,
    "agree_steps": 16,
    "faith_windows": 64,
    "faith_steps": 16,
    "faith_points": 5,
    "faith_random": 2,
    "gate_windows": 64,
    "gate_points": 5,
}


def _run_argv(module_main, argv: list[str]) -> None:
    """Invoke a runner's ``main()`` with a temporarily swapped ``sys.argv``.

    The single-analysis runners parse ``sys.argv`` with argparse, so we set it,
    call, and always restore — even on exception — rather than refactor six
    ``main()`` signatures just for the orchestrator.
    """
    saved = sys.argv
    sys.argv = argv
    try:
        module_main()
    finally:
        sys.argv = saved


def _stage(manifest: list[dict], name: str, fn) -> None:
    """Run one stage, timing it and recording success/failure without aborting.

    A stage failure is logged with its traceback and written to the manifest, but
    the sweep continues: on an unattended Kaggle run, losing the last analysis
    must not discard the ones that already succeeded.
    """
    logger.info("=" * 72)
    logger.info("stage: {}", name)
    logger.info("=" * 72)
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        logger.success("stage {} done in {:.1f}s", name, dt)
        manifest.append({"stage": name, "status": "ok", "seconds": round(dt, 1)})
    except Exception as exc:  # noqa: BLE001 — isolation is the whole point here
        dt = time.time() - t0
        logger.error("stage {} FAILED after {:.1f}s: {}", name, dt, exc)
        logger.error("\n{}", traceback.format_exc())
        manifest.append(
            {
                "stage": name,
                "status": "failed",
                "seconds": round(dt, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _copy_outputs(root: Path, models: list[str], out: Path) -> list[str]:
    """Gather every artefact the stages wrote into a single ``out`` directory.

    Per-model IG files live beside their checkpoint (the downstream stages need
    them there); this pulls a namespaced copy of each into ``out`` so the paper's
    figure/report step reads one flat directory. Returns the list of basenames
    copied, for the manifest.
    """
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    # aggregate artefacts written at the root
    for fname in ("probes.json", "cka.json", "agreement.json", "faithfulness.json"):
        src = root / fname
        if src.exists():
            shutil.copy2(src, out / fname)
            copied.append(fname)

    # per-model artefacts: IG (both baselines) and the gate sweep
    for m in models:
        ckpt = root / m
        for fname in (
            "ig_zero.json",
            "ig_zero.npz",
            "ig_mean.json",
            "ig_mean.npz",
            "gate_sweep.json",
            "metrics.json",
        ):
            src = ckpt / fname
            if src.exists():
                dst = out / f"{m}__{fname}"
                shutil.copy2(src, dst)
                copied.append(dst.name)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory holding the checkpoint dirs")
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="checkpoint dir names (default: the three in-scope k10 models)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="collected-artefacts dir (default: <root>/xai_results)",
    )
    ap.add_argument(
        "--baselines",
        nargs="+",
        default=["zero", "mean"],
        choices=["zero", "mean"],
        help="IG baselines to run (default: both; the paper reports zero, mean is "
        "the robustness check)",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sample sizes so the whole pipeline runs in minutes on CPU; "
        "proves the wiring, not the science",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["ig", "probes", "cka", "agreement", "faithfulness", "gate"],
        help="stages to skip",
    )
    args = ap.parse_args()

    # Imported here (not at module top) so that --help works even if a heavy
    # dependency of one runner is missing.
    from xai import run_agreement, run_cka, run_faithfulness, run_gate_sweep, run_ig
    from xai import run_probes

    cfg = SMOKE if args.smoke else FULL
    root: Path = args.root
    out = args.out or root / "xai_results"
    models: list[str] = args.models
    gate_models = [m for m in models if GATE_MODEL_HINT in m]

    missing = [m for m in models if not (root / m / "best.pt").exists()]
    if missing:
        raise SystemExit(
            f"missing checkpoint(s) under {root}: {missing}. "
            f"Have: {sorted(p.name for p in root.glob('*') if p.is_dir())}"
        )

    logger.info(
        "XAI run_all | mode={} | split={} | models={} | out={}",
        "SMOKE" if args.smoke else "full",
        args.split,
        models,
        out,
    )
    started = datetime.now(timezone.utc).isoformat()
    manifest: list[dict] = []

    # --- Layer 1: feature attribution (IG), one file per model per baseline -----
    if "ig" not in args.skip:
        for baseline in args.baselines:
            for m in models:
                _stage(
                    manifest,
                    f"ig[{baseline}]:{m}",
                    lambda m=m, baseline=baseline: _run_argv(
                        run_ig.main,
                        [
                            "run_ig",
                            str(root / m),
                            "--baseline",
                            baseline,
                            "--n-windows",
                            str(cfg["ig_windows"]),
                            "--n-steps",
                            str(cfg["ig_steps"]),
                            "--split",
                            args.split,
                        ],
                    ),
                )

    # --- Layer 1 sanity: deletion/insertion faithfulness (all models, one file) -
    if "faithfulness" not in args.skip:
        _stage(
            manifest,
            "faithfulness",
            lambda: _run_argv(
                run_faithfulness.main,
                [
                    "run_faithfulness",
                    str(root),
                    "--models",
                    *models,
                    "--baseline",
                    args.baselines[0],
                    "--n-windows",
                    str(cfg["faith_windows"]),
                    "--n-steps",
                    str(cfg["faith_steps"]),
                    "--n-points",
                    str(cfg["faith_points"]),
                    "--n-random",
                    str(cfg["faith_random"]),
                    "--split",
                    args.split,
                    "--seed",
                    str(args.seed),
                ],
            ),
        )

    # --- Layer 2: per-trunk linear probes (all models, one file) ----------------
    if "probes" not in args.skip:
        _stage(
            manifest,
            "probes",
            lambda: _run_argv(
                run_probes.main,
                [
                    "run_probes",
                    str(root),
                    "--models",
                    *models,
                    "--n-train",
                    str(cfg["probe_train"]),
                    "--n-test",
                    str(cfg["probe_test"]),
                    "--epochs",
                    str(cfg["probe_epochs"]),
                    "--seed",
                    str(args.seed),
                ],
            ),
        )

    # --- Layer 3: cross-model CKA (all models, one file) ------------------------
    if "cka" not in args.skip:
        _stage(
            manifest,
            "cka",
            lambda: _run_argv(
                run_cka.main,
                [
                    "run_cka",
                    str(root),
                    "--models",
                    *models,
                    "--n-windows",
                    str(cfg["cka_windows"]),
                    "--n-splits",
                    str(cfg["cka_splits"]),
                    "--split",
                    args.split,
                    "--seed",
                    str(args.seed),
                ],
            ),
        )

    # --- Layer 3: attention-vs-IG agreement (all models, one file) --------------
    if "agreement" not in args.skip:
        _stage(
            manifest,
            "agreement",
            lambda: _run_argv(
                run_agreement.main,
                [
                    "run_agreement",
                    str(root),
                    "--models",
                    *models,
                    "--baseline",
                    args.baselines[0],
                    "--n-windows",
                    str(cfg["agree_windows"]),
                    "--n-steps",
                    str(cfg["agree_steps"]),
                    "--split",
                    args.split,
                    "--seed",
                    str(args.seed),
                ],
            ),
        )

    # --- JumpGateLOB-only adaLN-Zero gate / robustness sweep --------------------
    if "gate" not in args.skip:
        if not gate_models:
            logger.warning(
                "no JumpGateLOB checkpoint among {}; skipping the gate sweep",
                models,
            )
        for m in gate_models:
            _stage(
                manifest,
                f"gate:{m}",
                lambda m=m: _run_argv(
                    run_gate_sweep.main,
                    [
                        "run_gate_sweep",
                        str(root / m),
                        "--n-windows",
                        str(cfg["gate_windows"]),
                        "--n-points",
                        str(cfg["gate_points"]),
                        "--split",
                        args.split,
                        "--seed",
                        str(args.seed),
                    ],
                ),
            )

    # --- collect everything into one directory + write the manifest -------------
    copied = _copy_outputs(root, models, out)
    n_ok = sum(1 for s in manifest if s["status"] == "ok")
    n_fail = sum(1 for s in manifest if s["status"] == "failed")
    manifest_doc = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "root": str(root),
        "out": str(out),
        "models": models,
        "gate_models": gate_models,
        "baselines": args.baselines,
        "split": args.split,
        "seed": args.seed,
        "sample_sizes": cfg,
        "stages": manifest,
        "n_ok": n_ok,
        "n_failed": n_fail,
        "artefacts": sorted(copied),
    }
    (out / "manifest.json").write_text(json.dumps(manifest_doc, indent=2))

    logger.info("=" * 72)
    logger.info("run_all done: {} ok, {} failed", n_ok, n_fail)
    logger.info("artefacts collected in {}", out)
    logger.info("manifest: {}", out / "manifest.json")
    if n_fail:
        logger.warning(
            "failed stages: {}",
            [s["stage"] for s in manifest if s["status"] == "failed"],
        )


if __name__ == "__main__":
    main()
