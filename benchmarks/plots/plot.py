#!/usr/bin/env python3
"""Plot benchmark results from CSV files."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from CSV files"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing CSV results (default: ../results relative to script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: same as results-dir)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or (Path(__file__).parent.parent / "results")
    output_dir = args.output_dir or results_dir

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    csv_files = list(results_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    data = {}
    for csv_file in csv_files:
        server_name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            data[server_name] = df
            print(f"Loaded {server_name}: {len(df)} samples")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not data:
        print("No valid data loaded")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Latency vs Concurrency
    fig, ax = plt.subplots(figsize=(12, 8))

    for server_name, df in sorted(data.items()):
        if "concurrency" in df.columns and "latency_p50_ms" in df.columns:
            agg = (
                df.groupby("concurrency")
                .agg(
                    {
                        "latency_p50_ms": "mean",
                        "latency_p90_ms": "mean",
                        "latency_p99_ms": "mean",
                        "throughput_rps": "mean",
                    }
                )
                .reset_index()
            )
            ax.plot(
                agg["concurrency"],
                agg["latency_p50_ms"],
                label=server_name,
                linewidth=2,
                marker="o",
                markersize=3,
            )

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Latency vs Concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    latency_path = output_dir / "latency_vs_concurrency.png"
    plt.savefig(latency_path, dpi=150)
    print(f"Saved: {latency_path}")

    # Plot 2: Throughput vs Concurrency
    fig, ax = plt.subplots(figsize=(12, 8))

    for server_name, df in sorted(data.items()):
        if "concurrency" in df.columns and "throughput_rps" in df.columns:
            agg = (
                df.groupby("concurrency")
                .agg(
                    {
                        "throughput_rps": "mean",
                    }
                )
                .reset_index()
            )
            ax.plot(
                agg["concurrency"],
                agg["throughput_rps"],
                label=server_name,
                linewidth=2,
                marker="o",
                markersize=3,
            )

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    throughput_path = output_dir / "throughput_vs_concurrency.png"
    plt.savefig(throughput_path, dpi=150)
    print(f"Saved: {throughput_path}")

    # Plot 3: P99 Latency vs Concurrency
    fig, ax = plt.subplots(figsize=(12, 8))

    for server_name, df in sorted(data.items()):
        if "concurrency" in df.columns and "latency_p99_ms" in df.columns:
            agg = (
                df.groupby("concurrency")
                .agg(
                    {
                        "latency_p99_ms": "mean",
                    }
                )
                .reset_index()
            )
            ax.plot(
                agg["concurrency"],
                agg["latency_p99_ms"],
                label=server_name,
                linewidth=2,
                marker="o",
                markersize=3,
            )

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency p99 (ms)")
    ax.set_title("P99 Latency vs Concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    p99_path = output_dir / "p99_latency_vs_concurrency.png"
    plt.savefig(p99_path, dpi=150)
    print(f"Saved: {p99_path}")

    # Summary
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(
        f"{'Server':<25} {'Max Throughput':<20} {'Min P50 (ms)':<20} {'Min P99 (ms)':<20}"
    )
    print("-" * 100)

    for server_name, df in sorted(data.items()):
        if "throughput_rps" in df.columns and "latency_p50_ms" in df.columns:
            max_throughput = df.groupby("concurrency")["throughput_rps"].mean().max()
            min_p50 = df.groupby("concurrency")["latency_p50_ms"].mean().min()
            min_p99 = df.groupby("concurrency")["latency_p99_ms"].mean().min()
            print(
                f"{server_name:<25} {max_throughput:<20.2f} {min_p50:<20.2f} {min_p99:<20.2f}"
            )

    print("=" * 100)
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
