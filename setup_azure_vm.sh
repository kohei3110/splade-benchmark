#!/usr/bin/env bash
set -euo pipefail

# Azure NC A100 VM bootstrap for Ubuntu 22.04
# - Installs NVIDIA driver via ubuntu-drivers
# - Installs CUDA 12.5 toolkit
# - Installs common utilities (nvtop, htop, build-essential)
# Run as root: sudo ./setup_azure_vm.sh

DEVICE="cuda"
usage() {
  cat <<'USAGE'
Usage: sudo ./setup_azure_vm.sh [--device cpu|cuda]

--device cpu   Install only CPU prerequisites (no NVIDIA/CUDA).
--device cuda  Install NVIDIA driver + CUDA toolkit (default).
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
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be 'cpu' or 'cuda' (got: ${DEVICE})" >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

log() { echo "[setup_azure_vm] $*"; }

log "Updating apt metadata and upgrading base system..."
apt update -y
apt full-upgrade -y

log "Installing base utilities..."
apt install -y --no-install-recommends \
  build-essential \
  wget curl ca-certificates gnupg lsb-release \
  htop \
  python3-venv python3-pip

if [[ "${DEVICE}" == "cpu" ]]; then
  log "Device=cpu selected: skipping NVIDIA driver/CUDA installation."
  log "Done. You can now run ./setup_python_env.sh --device cpu"
  exit 0
fi

log "Installing ubuntu-drivers-common (for GPU driver detection)..."
apt install -y --no-install-recommends ubuntu-drivers-common

log "Detecting Ubuntu version..."
UBUNTU_VERSION=$(lsb_release -rs)
log "Ubuntu version: ${UBUNTU_VERSION}"

log "Installing NVIDIA driver 580 (recommended for A100)..."
apt install -y nvidia-driver-580 nvidia-utils-580

log "Setting up CUDA toolkit..."
CUDA_REPO="ubuntu2204"
if [[ "${UBUNTU_VERSION}" == "24.04" ]]; then
  CUDA_REPO="ubuntu2404"
fi
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb
apt install -y ./cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb
apt update -y
apt install -y cuda-toolkit-12-6 || apt install -y cuda-toolkit-12-5

log "Installing nvtop for GPU monitoring..."
apt install -y nvtop || log "nvtop not available, skipping"

log "Verifying GPU visibility (nvidia-smi)..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "nvidia-smi not yet available (reboot likely required)."
fi

log "CUDA compiler version:"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  log "nvcc not yet available; ensure CUDA path is set after reboot."
fi

log "Done. Reboot the VM to finalize driver/CUDA setup: sudo reboot"
