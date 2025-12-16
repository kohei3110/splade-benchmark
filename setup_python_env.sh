#!/usr/bin/env bash
set -euo pipefail

# Python environment setup for SPLADE benchmarking
# Usage: ./setup_python_env.sh [--device cpu|cuda] [ENV_DIR]
# Default ENV_DIR: ~/splade_benchmark_env

DEVICE="cuda"
ENV_DIR="${HOME}/splade_benchmark_env"

usage() {
  cat <<'USAGE'
Usage: ./setup_python_env.sh [--device cpu|cuda] [ENV_DIR]

--device cpu   Install CPU-only PyTorch wheels.
--device cuda  Install CUDA-enabled PyTorch wheels (default).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # Backward-compatible positional ENV_DIR
      ENV_DIR="$1"
      shift
      ;;
  esac
done

if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be 'cpu' or 'cuda' (got: ${DEVICE})" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

log() { echo "[setup_python_env] $*"; }

log "Checking python3 ensurepip availability..."
if ! python3 -m ensurepip --version >/dev/null 2>&1; then
  log "ensurepip is missing. Install venv package: sudo apt install -y python3-venv python3.12-venv || sudo apt install -y python3-venv"
  exit 1
fi

ensure_venv() {
  if python3 -m ensurepip --version >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    log "python3-venv not available; installing via apt..."
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3.12-venv || sudo apt-get install -y python3-venv
  else
    log "python3-venv not available and apt-get not found. Please install python3-venv manually." >&2
    exit 1
  fi
}

ensure_venv

log "Creating virtual environment at: ${ENV_DIR}"
python3 -m venv "${ENV_DIR}"

log "Activating virtual environment..."
# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

log "Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

if [[ "${DEVICE}" == "cuda" ]]; then
  log "Installing PyTorch (CUDA 12.1 wheels)..."
  pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
else
  log "Installing PyTorch (CPU-only wheels)..."
  pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1
fi

log "Installing Python dependencies from requirements.txt..."
pip install -r "${REQ_FILE}"

log "Verifying torch CUDA availability..."
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
PY

log "Environment ready. Activate with: source ${ENV_DIR}/bin/activate"
