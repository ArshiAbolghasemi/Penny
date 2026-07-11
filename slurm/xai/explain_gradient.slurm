#!/bin/bash
#SBATCH --job-name=penny-xai-explain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-all

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

# Required env vars (checkpoint paths are timestamped at train time, so they
# can't be hardcoded the way training configs are):
#   MODEL       one of ctabl | dla | jumpgatelob
#   CHECKPOINT  path to a best.pt or its containing checkpoint dir
# Optional:
#   N_PER_CLASS default 8
#   OUT_DIR     defaults to <checkpoint_dir>/xai/gradient
#
# JointDiT is intentionally excluded from the XAI layer (decision
# 2026-07-11) — see docs/xai/README.md.
#
# Example submission:
#   MODEL=ctabl CHECKPOINT=checkpoints/nobitex/BTCIRT/ctabl_BTCIRT_lob_20260101_000000 \
#     sbatch slurm/xai/explain_gradient.slurm

: "${MODEL:?set MODEL=ctabl|dla|jumpgatelob}"
: "${CHECKPOINT:?set CHECKPOINT=path/to/best.pt or its dir}"
N_PER_CLASS="${N_PER_CLASS:-8}"

echo "Penny XAI [gradient] model=$MODEL checkpoint=$CHECKPOINT | host=$(hostname) | start=$(date)"

if command -v module >/dev/null 2>&1; then module purge || true; fi
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi

if [[ -z "${UV_EXTRA:-}" ]]; then
    if ! command -v nvidia-smi &>/dev/null; then
        UV_EXTRA="cpu"
    else
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
             | head -1 | tr -d '. ')
        if   [[ -z "$CC" ]];      then UV_EXTRA="cpu"
        elif [[ "$CC" -lt 75 ]];  then UV_EXTRA="cu118"
        elif [[ "$CC" -lt 100 ]]; then UV_EXTRA="cu126"
        else                           UV_EXTRA="cu130"
        fi
    fi
    echo "GPU compute cap: ${CC:-none}  →  auto UV_EXTRA=$UV_EXTRA"
else
    echo "UV_EXTRA=$UV_EXTRA (user override)"
fi

export UV_PROJECT_ENVIRONMENT=".venv-${UV_EXTRA}"
uv sync --extra "$UV_EXTRA"

ARGS=(--model "$MODEL" --checkpoint "$CHECKPOINT" --n-per-class "$N_PER_CLASS")
if [[ -n "${OUT_DIR:-}" ]]; then ARGS+=(--out "$OUT_DIR"); fi

uv run python scripts/explain_gradient.py "${ARGS[@]}"
echo "done: $(date)"
