#!/usr/bin/env python3
from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

from models.splade_model import SpladeModelWrapper
from utils.gpu_monitor import GpuMonitor
from utils.metrics_collector import BenchmarkResult

LOGGER = logging.getLogger(__name__)


def cycle_batches(texts: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    if not texts:
        raise ValueError("No texts provided for benchmarking")
    idx = 0
    n = len(texts)
    while True:
        batch = []
        while len(batch) < batch_size:
            batch.append(texts[idx % n])
            idx += 1
        yield batch


def run_single_config(
    model: SpladeModelWrapper,
    texts: Sequence[str],
    requested_batch_size: int,
    batch_size: int,
    seq_length: int,
    warmup_iters: int,
    measure_iters: int,
    enable_gpu_monitor: bool,
    monitor_interval: float = 0.5,
) -> BenchmarkResult:
    batch_iterator = cycle_batches(texts, batch_size)
    latencies_ms: List[float] = []
    total_samples = 0
    sparsity_sum = 0.0
    sparsity_max = 0

    monitor: Optional[GpuMonitor] = None
    if enable_gpu_monitor and model.device.type == "cuda":
        monitor = GpuMonitor(device_index=0, interval=monitor_interval)
        monitor.start()

    try:
        for _ in range(warmup_iters):
            batch = next(batch_iterator)
            _ = model.encode_batch(batch, max_length=seq_length)
        if model.device.type == "cuda":
            torch.cuda.synchronize()

        for _ in tqdm(range(measure_iters), desc=f"bs={batch_size}, seq={seq_length}", leave=False):
            batch = next(batch_iterator)
            start = time.perf_counter()
            output = model.encode_batch(batch, max_length=seq_length)
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            latencies_ms.append(elapsed_ms)
            total_samples += len(batch)
            sparsity_sum += float(output.nonzero_counts.sum().item())
            sparsity_max = max(sparsity_max, output.max_nonzero)
    finally:
        if monitor is not None:
            monitor.stop()

    latencies_ms_sorted = sorted(latencies_ms)
    total_time_sec = sum(latencies_ms_sorted) / 1000.0
    throughput_qps = total_samples / total_time_sec if total_time_sec > 0 else 0.0
    sparsity_mean = sparsity_sum / total_samples if total_samples else 0.0

    return BenchmarkResult(
        requested_batch_size=requested_batch_size,
        batch_size=batch_size,
        seq_length=seq_length,
        samples=total_samples,
        latency_ms=latencies_ms_sorted,
        throughput_qps=throughput_qps,
        sparsity_mean=sparsity_mean,
        sparsity_max=sparsity_max,
        gpu_metrics=(monitor.summary() if monitor is not None else {}),
        oom_retries=0,
    )


def run_benchmarks(
    model: SpladeModelWrapper,
    texts_by_seq_length: Dict[int, Sequence[str]],
    batch_sizes: Sequence[int],
    seq_lengths: Sequence[int],
    warmup_iters: int,
    measure_iters: int,
    device: str,
    model_name: str,
    vm_sku: str,
    cpu_threads: Optional[int],
    interop_threads: Optional[int],
    padding_policy: str,
    enable_gpu_monitor: bool,
    monitor_interval: float,
    logger: logging.Logger,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for seq_len in seq_lengths:
        texts = texts_by_seq_length.get(seq_len)
        if not texts:
            raise ValueError(f"No texts found for seq_length={seq_len}. Ensure data/queries_{seq_len}.json exists.")
        for bsz in batch_sizes:
            requested_bsz = bsz
            current_bsz = bsz
            oom_retries = 0
            while current_bsz >= 1:
                try:
                    logger.info("Running benchmark | batch_size=%s seq_length=%s", current_bsz, seq_len)
                    result = run_single_config(
                        model=model,
                        texts=texts,
                        requested_batch_size=requested_bsz,
                        batch_size=current_bsz,
                        seq_length=seq_len,
                        warmup_iters=warmup_iters,
                        measure_iters=measure_iters,
                        enable_gpu_monitor=enable_gpu_monitor,
                        monitor_interval=monitor_interval,
                    )
                    result.oom_retries = oom_retries
                    result.device = device
                    result.model_name = model_name
                    result.vm_sku = vm_sku
                    result.cpu_threads = cpu_threads
                    result.interop_threads = interop_threads
                    result.padding_policy = padding_policy
                    result.gpu_monitor_enabled = bool(enable_gpu_monitor and model.device.type == "cuda")
                    results.append(result)
                    break
                except RuntimeError as exc:
                    if model.device.type == "cuda" and "out of memory" in str(exc).lower():
                        logger.warning(
                            "OOM at batch_size=%s seq_length=%s; reducing batch size by half",
                            current_bsz,
                            seq_len,
                        )
                        torch.cuda.empty_cache()
                        current_bsz = current_bsz // 2
                        oom_retries += 1
                        continue
                    raise
            if current_bsz < 1:
                logger.error("Failed to benchmark seq_length=%s even with batch_size=1", seq_len)
    return results
