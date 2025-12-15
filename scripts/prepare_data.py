#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_SEQ_LENGTHS = [128, 256, 512, 1024]
DEFAULT_SAMPLES_PER_LEN = 500
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def normalize_length(text: str, target_chars: int) -> str:
    if len(text) >= target_chars:
        return text[:target_chars]
    if not text:
        text = "日本"
    repeat_times = max(2, (target_chars // max(1, len(text))) + 1)
    expanded = (text + "。") * repeat_times
    return expanded[:target_chars]


def sample_texts(dataset, num_samples: int) -> List[str]:
    subset = dataset.shuffle(seed=42).select(range(min(num_samples * 3, len(dataset))))
    texts = [row.get("query") or row.get("passage") or row.get("text") for row in subset]
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    random.seed(42)
    random.shuffle(texts)
    return texts[:num_samples]


def build_length_buckets(texts: List[str], seq_lengths: List[int]) -> Dict[int, List[str]]:
    buckets: Dict[int, List[str]] = {s: [] for s in seq_lengths}
    for seq_len in seq_lengths:
        target_chars = int(seq_len * 2.2)
        buckets[seq_len] = [normalize_length(t, target_chars) for t in texts]
    return buckets


def save_json(path: Path, texts: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare Japanese query/text data for SPLADE benchmarking")
    parser.add_argument("--seq-lengths", type=int, nargs="*", default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_LEN,
                        help="Number of samples per sequence length")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading mMARCO Japanese dataset (train split)...")
    dataset = load_dataset("unicamp-dl/mmarco", "japanese", split="train")
    print(f"Dataset size: {len(dataset):,}")

    num_samples_total = max(args.samples, 1) * 2
    base_texts = sample_texts(dataset, num_samples_total)
    if not base_texts:
        raise RuntimeError("No texts could be sampled from dataset.")

    buckets = build_length_buckets(base_texts, args.seq_lengths)

    print("Saving datasets:")
    for seq_len, texts in buckets.items():
        limited = texts[: args.samples]
        out_path = DATA_DIR / f"queries_{seq_len}.json"
        save_json(out_path, limited)
        print(f"- {out_path.relative_to(DATA_DIR.parent)}: {len(limited)} samples")

    consolidated_path = DATA_DIR / "queries.json"
    save_json(consolidated_path, base_texts[: args.samples])
    print(f"- {consolidated_path.relative_to(DATA_DIR.parent)}: {min(args.samples, len(base_texts))} base samples")

    print("Done.")


if __name__ == "__main__":
    main()
