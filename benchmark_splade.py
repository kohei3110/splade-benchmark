#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence

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
    parser = argparse.ArgumentParser(description="Benchmark SPLADE model on GPU")
    parser.add_argument("--model", type=str, default="bizreach-inc/light-splade-japanese-14M",
                        help="HuggingFace model name (e.g., bizreach-inc/light-splade-japanese-28M)")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--seq-lengths", type=int, nargs="*", default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"benchmark_{timestamp}.log"
    logger = setup_logger("benchmark", log_file=log_path, level=getattr(__import__("logging"), args.log_level.upper(), 20))

    logger.info("Starting benchmark: model=%s batches=%s seqs=%s warmup=%s iterations=%s", args.model, args.batch_sizes, args.seq_lengths, args.warmup, args.iterations)

    model = SpladeModelWrapper(model_name=args.model)
    logger.info("Model device info: %s", model.device_stats())

    # Use the longest seq_length dataset if per-length file missing; fallback handled in load_texts.
    texts_cache = {}
    for seq_len in args.seq_lengths:
        try:
            texts_cache[seq_len] = load_texts(seq_len)
        except FileNotFoundError:
            logger.warning("Dataset for seq_length=%s not found; attempting fallback queries.json", seq_len)
            texts_cache[seq_len] = load_texts(DEFAULT_SEQ_LENGTHS[-1])

    # Flatten texts for reuse; runner cycles through list
    shared_texts: Sequence[str] = next(iter(texts_cache.values())) if texts_cache else []

    results = run_benchmarks(
        model=model,
        texts=shared_texts,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        warmup_iters=args.warmup,
        measure_iters=args.iterations,
        logger=logger,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_suffix = args.model.replace("/", "_").replace("-", "_")
    json_path = RESULTS_DIR / f"benchmark_{model_suffix}_{timestamp}.json"
    csv_path = RESULTS_DIR / f"benchmark_{model_suffix}_{timestamp}.csv"
    save_json(json_path, results)
    save_csv(csv_path, results)

    logger.info("Saved results to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
