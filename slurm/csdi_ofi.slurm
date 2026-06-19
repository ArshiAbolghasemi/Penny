#!/bin/bash
#SBATCH --job-name=penny-csdi-ofi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH -p gpu

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

echo "Penny CSDI OFI | host=$(hostname) | start=$(date)"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

CONFIG="${CONFIG:-configs/csdi/ofi.json}"
uv run python -m csdi.train "$CONFIG"

echo "done: $(date)"
