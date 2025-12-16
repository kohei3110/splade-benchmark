import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BenchmarkResult:
    # Per-config parameters
    requested_batch_size: int
    batch_size: int
    seq_length: int
    samples: int
    latency_ms: List[float]
    throughput_qps: float
    sparsity_mean: float
    sparsity_max: float
    gpu_metrics: Dict[str, float]
    oom_retries: int = 0

    # Run-level metadata (attached per result for easy CSV/JSON export)
    device: str = ""
    model_name: str = ""
    vm_sku: str = ""
    cpu_threads: Optional[int] = None
    interop_threads: Optional[int] = None
    padding_policy: str = ""
    gpu_monitor_enabled: bool = False

    def summary(self) -> Dict:
        percentiles = {p: percentile(self.latency_ms, p) for p in (50, 95, 99)}
        return {
            "device": self.device,
            "model": self.model_name,
            "vm_sku": self.vm_sku,
            "cpu_threads": self.cpu_threads,
            "interop_threads": self.interop_threads,
            "padding_policy": self.padding_policy,
            "gpu_monitor_enabled": bool(self.gpu_monitor_enabled),
            "requested_batch_size": self.requested_batch_size,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "samples": self.samples,
            "throughput_qps": self.throughput_qps,
            "latency_ms_p50": percentiles[50],
            "latency_ms_p95": percentiles[95],
            "latency_ms_p99": percentiles[99],
            "sparsity_mean": self.sparsity_mean,
            "sparsity_max": self.sparsity_max,
            "oom_retries": self.oom_retries,
            **(self.gpu_metrics or {}),
        }


def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[int(k)]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1


def save_json(path: Path, results: List[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.summary() for r in results]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, results: List[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    summaries = [r.summary() for r in results]
    base_order = [
        "device",
        "model",
        "vm_sku",
        "cpu_threads",
        "interop_threads",
        "padding_policy",
        "gpu_monitor_enabled",
        "requested_batch_size",
        "batch_size",
        "seq_length",
        "samples",
        "throughput_qps",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "sparsity_mean",
        "sparsity_max",
        "oom_retries",
    ]
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())
    keys = [k for k in base_order if k in all_keys]
    keys.extend(sorted(all_keys - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean": statistics.mean(latencies),
    }
