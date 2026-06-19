#!/bin/bash
#SBATCH --job-name=penny-painting-ofi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p gpu

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

echo "Penny painting OFI | host=$(hostname) | start=$(date)"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

CONFIG="${CONFIG:-configs/painting/ofi.json}"
uv run python -m painting.train "$CONFIG"

echo "done: $(date)"
