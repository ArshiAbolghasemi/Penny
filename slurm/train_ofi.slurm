#!/bin/bash

#SBATCH --job-name=penny-ofi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-short

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running Penny OFI training from $ROOT_DIR"
echo "Host: $(hostname)"
echo "Start: $(date)"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/bitcoin/config_ofi.json}"

uv run python scripts/train_penny.py "$CONFIG_PATH"

echo "End: $(date)"
