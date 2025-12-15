#!/usr/bin/env python3
from __future__ import annotations

import logging
import time
from typing import Iterable, List, Sequence

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
    batch_size: int,
    seq_length: int,
    warmup_iters: int,
    measure_iters: int,
    monitor_interval: float = 0.5,
) -> BenchmarkResult:
    batch_iterator = cycle_batches(texts, batch_size)
    latencies_ms: List[float] = []
    total_samples = 0
    sparsity_sum = 0.0
    sparsity_max = 0

    monitor = GpuMonitor(device_index=0, interval=monitor_interval)
    monitor.start()

    try:
        for _ in range(warmup_iters):
            batch = next(batch_iterator)
            _ = model.encode_batch(batch, max_length=seq_length)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for _ in tqdm(range(measure_iters), desc=f"bs={batch_size}, seq={seq_length}", leave=False):
            batch = next(batch_iterator)
            start = time.perf_counter()
            output = model.encode_batch(batch, max_length=seq_length)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            latencies_ms.append(elapsed_ms)
            total_samples += len(batch)
            sparsity_sum += float(output.nonzero_counts.sum().item())
            sparsity_max = max(sparsity_max, output.max_nonzero)
    finally:
        monitor.stop()

    latencies_ms_sorted = sorted(latencies_ms)
    total_time_sec = sum(latencies_ms_sorted) / 1000.0
    throughput_qps = total_samples / total_time_sec if total_time_sec > 0 else 0.0
    sparsity_mean = sparsity_sum / total_samples if total_samples else 0.0

    return BenchmarkResult(
        batch_size=batch_size,
        seq_length=seq_length,
        samples=total_samples,
        latency_ms=latencies_ms_sorted,
        throughput_qps=throughput_qps,
        sparsity_mean=sparsity_mean,
        sparsity_max=sparsity_max,
        gpu_metrics=monitor.summary(),
        oom_retries=0,
    )


def run_benchmarks(
    model: SpladeModelWrapper,
    texts: Sequence[str],
    batch_sizes: Sequence[int],
    seq_lengths: Sequence[int],
    warmup_iters: int,
    measure_iters: int,
    logger: logging.Logger,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for seq_len in seq_lengths:
        for bsz in batch_sizes:
            current_bsz = bsz
            oom_retries = 0
            while current_bsz >= 1:
                try:
                    logger.info("Running benchmark | batch_size=%s seq_length=%s", current_bsz, seq_len)
                    result = run_single_config(
                        model=model,
                        texts=texts,
                        batch_size=current_bsz,
                        seq_length=seq_len,
                        warmup_iters=warmup_iters,
                        measure_iters=measure_iters,
                    )
                    result.oom_retries = oom_retries
                    results.append(result)
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
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
