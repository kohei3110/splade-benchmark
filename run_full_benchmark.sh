#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR=${ENV_DIR:-"${HOME}/splade_benchmark_env"}

DEVICE=${DEVICE:-"cuda"}
PROFILE=${PROFILE:-"full"}
VM_SKU=${VM_SKU:-""}

log() { echo "[run_full_benchmark] $*"; }

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  log "Python environment not found at ${ENV_DIR}. Run setup_python_env.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

if [[ "${DEVICE}" == "cuda" ]]; then
  log "GPU info:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    log "nvidia-smi not found; ensure NVIDIA drivers are installed." >&2
  fi
else
  log "DEVICE=cpu: skipping nvidia-smi"
fi

log "Preparing data..."

# Stable E2E env defaults (mainly for GPU runs; keep optional for CPU).
if [[ "${DEVICE}" == "cuda" ]]; then
  : "${TOKENIZERS_PARALLELISM:=false}"
  : "${OMP_NUM_THREADS:=1}"
  : "${MKL_NUM_THREADS:=1}"
  : "${OPENBLAS_NUM_THREADS:=1}"
  : "${NUMEXPR_NUM_THREADS:=1}"
fi

case "${PROFILE}" in
  full)
    SEQ_LENGTHS=(128 256 512 1024)
    BATCH_SIZES=(1 8 32 64 128 256)
    WARMUP=10
    ITERS=100
    PADDING_POLICY=longest
    ;;
  online)
    # Online search: run two passes.
    # - short-query grid: seq={16,32,64,128} bs={1,2,4,8,16}
    # - long-query grid:  seq={256,512,1024} bs={1,2,4} (representative point: seq=512, bs=1)
    SEQ_LENGTHS=(16 32 64 128 256 512 1024)
    PADDING_POLICY=max_length

    SHORT_SEQ_LENGTHS=(16 32 64 128)
    SHORT_BATCH_SIZES=(1 2 4 8 16)
    SHORT_WARMUP=30
    SHORT_ITERS=500

    LONG_SEQ_LENGTHS=(256 512 1024)
    LONG_BATCH_SIZES=(1 2 4)
    LONG_WARMUP=50
    LONG_ITERS=500
    ;;
  *)
    log "Unknown PROFILE='${PROFILE}'. Use PROFILE=full or PROFILE=online." >&2
    exit 1
    ;;
esac

python "${SCRIPT_DIR}/scripts/prepare_data.py" --seq-lengths "${SEQ_LENGTHS[@]}" --samples 500

log "Running benchmark..."
MODEL_NAME=${MODEL:-"bizreach-inc/light-splade-japanese-14M"}
log "Using model: ${MODEL_NAME}"

if [[ "${PROFILE}" == "online" ]]; then
  log "Running short-query benchmark..."
  python "${SCRIPT_DIR}/benchmark_splade.py" \
    --model "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --vm-sku "${VM_SKU}" \
    --batch-sizes "${SHORT_BATCH_SIZES[@]}" \
    --seq-lengths "${SHORT_SEQ_LENGTHS[@]}" \
    --warmup "${SHORT_WARMUP}" \
    --iterations "${SHORT_ITERS}" \
    --padding-policy "${PADDING_POLICY}" \
    --log-level DEBUG

  log "Running long-query benchmark (representative: seq=512, bs=1)..."
  python "${SCRIPT_DIR}/benchmark_splade.py" \
    --model "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --vm-sku "${VM_SKU}" \
    --batch-sizes "${LONG_BATCH_SIZES[@]}" \
    --seq-lengths "${LONG_SEQ_LENGTHS[@]}" \
    --warmup "${LONG_WARMUP}" \
    --iterations "${LONG_ITERS}" \
    --padding-policy "${PADDING_POLICY}" \
    --log-level DEBUG
else
  python "${SCRIPT_DIR}/benchmark_splade.py" \
    --model "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --vm-sku "${VM_SKU}" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --seq-lengths "${SEQ_LENGTHS[@]}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERS}" \
    --padding-policy "${PADDING_POLICY}" \
    --log-level DEBUG
fi

if [[ "${PROFILE}" == "online" ]]; then
  JSONS=( $(ls -t "${SCRIPT_DIR}/results/metrics"/*.json 2>/dev/null | head -n 2) )
else
  JSONS=( $(ls -t "${SCRIPT_DIR}/results/metrics"/*.json 2>/dev/null | head -n 1) )
fi

if [[ ${#JSONS[@]} -eq 0 ]]; then
  log "No result JSON found; skipping visualization." >&2
  exit 0
fi

for json in "${JSONS[@]}"; do
  log "Generating plots from ${json}..."
  python "${SCRIPT_DIR}/scripts/visualize_results.py" --input "${json}" --outdir "${SCRIPT_DIR}/results/plots"
done

log "Done. Results in results/metrics and plots in results/plots."
