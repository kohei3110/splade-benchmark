#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_results(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_heatmap(df: pd.DataFrame, value: str, title: str, out_path: Path):
    pivot = df.pivot(index="batch_size", columns="seq_length", values=value)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title(title)
    plt.ylabel("Batch size")
    plt.xlabel("Sequence length")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_lines(df: pd.DataFrame, y: str, title: str, out_path: Path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="batch_size", y=y, hue="seq_length", marker="o")
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel("Batch size")
    plt.legend(title="Seq length")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize SPLADE benchmark results")
    parser.add_argument("--input", type=Path, required=True, help="Path to benchmark JSON result")
    parser.add_argument("--outdir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    df = load_results(args.input)
    df = df.sort_values(["seq_length", "batch_size"])

    plot_heatmap(df, "latency_ms_p50", "Latency P50 (ms)", args.outdir / "latency_p50_heatmap.png")
    plot_heatmap(df, "latency_ms_p95", "Latency P95 (ms)", args.outdir / "latency_p95_heatmap.png")
    plot_heatmap(df, "throughput_qps", "Throughput (QPS)", args.outdir / "throughput_heatmap.png")

    plot_lines(df, "latency_ms_p50", "Latency P50 vs Batch Size", args.outdir / "latency_p50_lines.png")
    plot_lines(df, "throughput_qps", "Throughput vs Batch Size", args.outdir / "throughput_lines.png")

    gpu_cols = [c for c in df.columns if c.startswith("gpu_util") or c.startswith("mem_used")]
    for col in gpu_cols:
        plot_heatmap(df, col, col, args.outdir / f"{col}_heatmap.png")

    print(f"Plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
