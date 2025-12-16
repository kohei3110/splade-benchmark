#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence

import torch

from models.splade_model import SpladeModelWrapper
from utils.benchmark_runner import run_benchmarks
from utils.logger import setup_logger
from utils.metrics_collector import save_csv, save_json

DEFAULT_BATCH_SIZES = [1, 8, 32, 64, 128, 256]
DEFAULT_SEQ_LENGTHS = [128, 256, 512, 1024]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "metrics"
LOG_DIR = ROOT / "results" / "logs"


def load_texts(seq_length: int) -> List[str]:
    candidates = [DATA_DIR / f"queries_{seq_length}.json", DATA_DIR / "queries.json"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, dict) and "texts" in payload:
                    return payload["texts"]
    raise FileNotFoundError(
        f"No dataset found for seq_length={seq_length}. Run 'python scripts/prepare_data.py' or './run_full_benchmark.sh' from the repository root first."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SPLADE model on CPU or GPU")
    parser.add_argument("--model", type=str, default="bizreach-inc/light-splade-japanese-14M",
                        help="HuggingFace model name (e.g., bizreach-inc/light-splade-japanese-28M)")
    parser.add_argument("--device", type=str, choices=("cpu", "cuda"), default="cuda",
                        help="Execution device. Use 'cpu' to force CPU even if CUDA is available.")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Set torch.set_num_threads(N) (CPU-only tuning; applied for any device).")
    parser.add_argument("--interop-threads", type=int, default=None,
                        help="Set torch.set_num_interop_threads(N) (must be set early).")
    parser.add_argument("--vm-sku", type=str, default="",
                        help="Optional annotation for result metadata (e.g., Standard_D32s_v6, NC24ads_A100_v4).")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--seq-lengths", type=int, nargs="*", default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--padding-policy", type=str, choices=("longest", "max_length"), default="longest",
                        help="Tokenizer padding policy for E2E benchmarking. 'longest' matches current behavior.")
    parser.add_argument("--gpu-monitor", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable best-effort GPU monitoring via NVML (auto-skipped on CPU or if NVML unavailable).")
    parser.add_argument("--monitor-interval", type=float, default=0.5,
                        help="GPU monitor sampling interval (seconds).")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CPU thread settings as early as possible.
    if args.interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(args.interop_threads))
        except (ValueError, RuntimeError) as exc:
            # This must be called before any interop threads are created.
            # We log a warning and continue rather than failing the run.
            import logging

            logging.getLogger(__name__).warning("Failed to set interop threads=%s: %s", args.interop_threads, exc)
    if args.cpu_threads is not None:
        try:
            torch.set_num_threads(int(args.cpu_threads))
        except (ValueError, RuntimeError) as exc:
            import logging

            logging.getLogger(__name__).warning("Failed to set torch threads=%s: %s", args.cpu_threads, exc)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"benchmark_{timestamp}.log"
    logger = setup_logger("benchmark", log_file=log_path, level=getattr(__import__("logging"), args.log_level.upper(), 20))

    logger.info(
        "Starting benchmark: model=%s device=%s batches=%s seqs=%s warmup=%s iterations=%s padding=%s gpu_monitor=%s interval=%s vm_sku=%s",
        args.model,
        args.device,
        args.batch_sizes,
        args.seq_lengths,
        args.warmup,
        args.iterations,
        args.padding_policy,
        args.gpu_monitor,
        args.monitor_interval,
        args.vm_sku or "(unset)",
    )

    model = SpladeModelWrapper(model_name=args.model, device=args.device, padding_policy=args.padding_policy)
    logger.info("Model device info: %s", model.device_stats())

    texts_by_seq = {}
    for seq_len in args.seq_lengths:
        try:
            texts_by_seq[seq_len] = load_texts(seq_len)
        except FileNotFoundError as exc:
            logger.error("Dataset not found for seq_length=%s: %s", seq_len, exc)
            raise

    results = run_benchmarks(
        model=model,
        texts_by_seq_length=texts_by_seq,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        warmup_iters=args.warmup,
        measure_iters=args.iterations,
        device=str(model.device),
        model_name=args.model,
        vm_sku=args.vm_sku,
        cpu_threads=args.cpu_threads,
        interop_threads=args.interop_threads,
        padding_policy=args.padding_policy,
        enable_gpu_monitor=bool(args.gpu_monitor),
        monitor_interval=float(args.monitor_interval),
        logger=logger,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_suffix = args.model.replace("/", "_").replace("-", "_")
    device_suffix = args.device
    pad_suffix = "padmax" if args.padding_policy == "max_length" else "padlong"
    thread_suffix = ""
    if args.cpu_threads is not None or args.interop_threads is not None:
        thread_suffix = f"_t{args.cpu_threads or 'x'}_i{args.interop_threads or 'x'}"
    json_path = RESULTS_DIR / f"benchmark_{model_suffix}_{device_suffix}_{pad_suffix}{thread_suffix}_{timestamp}.json"
    csv_path = RESULTS_DIR / f"benchmark_{model_suffix}_{device_suffix}_{pad_suffix}{thread_suffix}_{timestamp}.csv"
    save_json(json_path, results)
    save_csv(csv_path, results)

    logger.info("Saved results to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
