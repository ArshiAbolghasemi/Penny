#!/bin/bash
#SBATCH --job-name=penny-xai-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-all

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

# Required env vars — one checkpoint per model, all trained on the same
# symbol + feature_mode + label_k (cross-model agreement/faithfulness
# metrics only mean something when every model solves the same task on the
# same inputs; scripts/compare_xai.py warns if configs mismatch):
#   CTABL_CKPT, DLA_CKPT, JUMPGATELOB_CKPT
#   OUT_DIR       output directory for comparison_summary.json + figures
# Optional:
#   N_PER_CLASS   default 6
#
# JointDiT is intentionally excluded from the XAI layer (decision
# 2026-07-11) — see docs/xai/README.md.
#
# Example submission:
#   CTABL_CKPT=checkpoints/nobitex/BTCIRT/ctabl_BTCIRT_lob_.../best.pt \
#   DLA_CKPT=checkpoints/nobitex/BTCIRT/dla_BTCIRT_lob_.../best.pt \
#   JUMPGATELOB_CKPT=checkpoints/nobitex/BTCIRT/jumpgatelob_BTCIRT_lob_.../best.pt \
#   OUT_DIR=results/xai/compare_btcirt_lob \
#     sbatch slurm/xai/compare_xai.slurm

: "${CTABL_CKPT:?set CTABL_CKPT}"
: "${DLA_CKPT:?set DLA_CKPT}"
: "${JUMPGATELOB_CKPT:?set JUMPGATELOB_CKPT}"
: "${OUT_DIR:?set OUT_DIR}"
N_PER_CLASS="${N_PER_CLASS:-6}"

echo "Penny XAI [compare] out=$OUT_DIR | host=$(hostname) | start=$(date)"

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

uv run python scripts/compare_xai.py \
    --ctabl "$CTABL_CKPT" \
    --dla "$DLA_CKPT" \
    --jumpgatelob "$JUMPGATELOB_CKPT" \
    --n-per-class "$N_PER_CLASS" \
    --out "$OUT_DIR"
echo "done: $(date)"
