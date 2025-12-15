import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class BenchmarkResult:
    batch_size: int
    seq_length: int
    samples: int
    latency_ms: List[float]
    throughput_qps: float
    sparsity_mean: float
    sparsity_max: float
    gpu_metrics: Dict[str, float]
    oom_retries: int = 0

    def summary(self) -> Dict:
        percentiles = {p: percentile(self.latency_ms, p) for p in (50, 95, 99)}
        return {
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
    keys = list(results[0].summary().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r.summary())


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean": statistics.mean(latencies),
    }
