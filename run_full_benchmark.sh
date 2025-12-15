#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR=${ENV_DIR:-"${HOME}/splade_benchmark_env"}

log() { echo "[run_full_benchmark] $*"; }

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  log "Python environment not found at ${ENV_DIR}. Run setup_python_env.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

log "GPU info:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "nvidia-smi not found; ensure NVIDIA drivers are installed." >&2
fi

log "Preparing data..."
python "${SCRIPT_DIR}/scripts/prepare_data.py" --seq-lengths 128 256 512 1024 --samples 500

log "Running benchmark..."
MODEL_NAME=${MODEL:-"bizreach-inc/light-splade-japanese-14M"}
log "Using model: ${MODEL_NAME}"
python "${SCRIPT_DIR}/benchmark_splade.py" \
  --model "${MODEL_NAME}" \
  --batch-sizes 1 8 32 64 128 256 \
  --seq-lengths 128 256 512 1024 \
  --warmup 10 \
  --iterations 100 \
  --log-level DEBUG

LATEST_JSON=$(ls -t "${SCRIPT_DIR}/results/metrics"/*.json 2>/dev/null | head -n 1)
if [[ -z "${LATEST_JSON}" ]]; then
  log "No result JSON found; skipping visualization." >&2
  exit 0
fi

log "Generating plots from ${LATEST_JSON}..."
python "${SCRIPT_DIR}/scripts/visualize_results.py" --input "${LATEST_JSON}" --outdir "${SCRIPT_DIR}/results/plots"

log "Done. Results in results/metrics and plots in results/plots."
