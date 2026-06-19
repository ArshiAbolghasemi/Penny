#!/bin/bash
#SBATCH --job-name=penny-timesfm-lob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-all

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

echo "Penny TimesFM LOB | host=$(hostname) | start=$(date)"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

CONFIG="${CONFIG:-configs/crypto/timesfm/btcusdt_lob.json}"
uv run python -m crypto.timesfm.train "$CONFIG"

echo "done: $(date)"
