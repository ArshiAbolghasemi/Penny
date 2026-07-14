#!/bin/bash
#SBATCH --job-name=penny-btcirt-arima-ofi-k20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p gpu-all

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

echo "Penny [Nobitex BTCIRT] arima OFI k=20 | host=$(hostname) | start=$(date)"

if command -v module >/dev/null 2>&1; then module purge || true; fi
if [[ -n "${VENV_PATH:-}" ]]; then source "$VENV_PATH/bin/activate"; fi

UV_EXTRA="${UV_EXTRA:-cpu}"
CONFIG="${CONFIG:-configs/crypto/nobitex/arima/btcirt_ofi_k20.json}"
export UV_PROJECT_ENVIRONMENT=".venv-${UV_EXTRA}"
uv sync --extra "$UV_EXTRA"
uv run python -m crypto.train_arima "$CONFIG"
echo "done: $(date)"
