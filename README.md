# SPLADE Benchmark on Azure NC24ads_A100_v4

Custom benchmarking toolkit for `bizreach-inc/light-splade-japanese-14M` on Azure NC A100. Measures latency (P50/P95/P99), throughput (QPS), sparsity, and GPU metrics across batch sizes and sequence lengths. Files are located directly under `bizreach-inc/`.

## Prerequisites
- Azure NC24ads_A100_v4 (Ubuntu 22.04 or 24.04)
- Sudo access for driver/CUDA install
- Internet access to download models/datasets

## 1) VM Bootstrap (run as root)
```
sudo ./setup_azure_vm.sh
sudo reboot
```

## 2) Python Environment
```
sudo apt-get update && sudo apt-get install -y python3-venv python3.12-venv  # if ensurepip/venv is missing
./setup_python_env.sh  # creates ~/splade_benchmark_env
source ~/splade_benchmark_env/bin/activate
```

## 3) Prepare Data, Run Benchmark, Visualize

### Default (14M model)
```
./run_full_benchmark.sh
```

### Specify different model
```
# 28M model
MODEL="bizreach-inc/light-splade-japanese-28M" ./run_full_benchmark.sh

# 56M model
MODEL="bizreach-inc/light-splade-japanese-56M" ./run_full_benchmark.sh

# Or use Python directly
python benchmark_splade.py --model bizreach-inc/light-splade-japanese-28M
```

### Benchmark multiple models
```bash
for model in "bizreach-inc/light-splade-japanese-14M" \
             "bizreach-inc/light-splade-japanese-28M" \
             "bizreach-inc/light-splade-japanese-56M"; do
  MODEL="${model}" ./run_full_benchmark.sh
done
```

- Data saved under `data/`
- Metrics saved under `results/metrics/*.json` and `.csv`
- Plots saved under `results/plots/`
- Logs under `results/logs/`

## Available Models
- `bizreach-inc/light-splade-japanese-14M` (default, 13.8M params)
- `bizreach-inc/light-splade-japanese-28M` (27.5M params)
- `bizreach-inc/light-splade-japanese-56M` (55.0M params)

## Key Defaults
- Batch sizes: 1, 8, 32, 64, 128, 256
- Sequence lengths: 128, 256, 512, 1024
- Warmup: 10 iterations
- Measure: 100 iterations
- Precision: FP32
- OOM handling: auto-halves batch size on CUDA OOM

## Troubleshooting
- OOM: batch size auto-halves on CUDA OOM.
- Missing GPU: ensure `nvidia-smi` works and drivers are installed.
- Slow downloads: set `HF_HOME` to a fast disk and optionally `HF_HUB_ENABLE_HF_TRANSFER=1`.
