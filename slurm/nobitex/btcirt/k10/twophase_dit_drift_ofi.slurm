#!/bin/bash
#SBATCH --job-name=penny-btcirt-2ph-dit-drft-ofi-k10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-all

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

echo "Penny [Nobitex BTCIRT] twophase dit drift OFI k=10 | host=$(hostname) | start=$(date)"

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

CONFIG="${CONFIG:-configs/crypto/nobitex/twophase/btcirt_ofi_k10_dit_drift.json}"
export UV_PROJECT_ENVIRONMENT=".venv-${UV_EXTRA}"
uv sync --extra "$UV_EXTRA"
uv run python -m crypto.train_2phase_diffusion "$CONFIG"
echo "done: $(date)"
