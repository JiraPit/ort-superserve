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
            print(f"Loaded {server_name}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not data:
        print("No valid data loaded")
        sys.exit(1)

    colors = {
        "ort-superserve": "blue",
        "actix-with-batching": "green",
        "actix-without-batching": "orange",
        "arc-mutex": "red",
        "batched-fn": "purple",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for server_name, df in sorted(data.items()):
        color = colors.get(server_name, "gray")
        if "latency_p50_ms" in df.columns:
            ax.bar(
                server_name,
                df["latency_p50_ms"].iloc[0],
                label=server_name,
                color=color,
                alpha=0.7,
            )

    ax.set_xlabel("Server")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Latency Comparison (p50)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    latency_path = output_dir / "latency_comparison.png"
    plt.savefig(latency_path, dpi=150)
    print(f"Saved: {latency_path}")

    fig, ax = plt.subplots(figsize=(12, 8))

    for server_name, df in sorted(data.items()):
        color = colors.get(server_name, "gray")
        if "throughput_rps" in df.columns:
            ax.bar(
                server_name,
                df["throughput_rps"].iloc[0],
                label=server_name,
                color=color,
                alpha=0.7,
            )

    ax.set_xlabel("Server")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    throughput_path = output_dir / "throughput_comparison.png"
    plt.savefig(throughput_path, dpi=150)
    print(f"Saved: {throughput_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Server':<25} {'Throughput (req/s)':<20} {'Latency p50 (ms)':<20} {'Latency p99 (ms)':<20}"
    )
    print("-" * 80)

    for server_name, df in sorted(data.items()):
        throughput = (
            df["throughput_rps"].iloc[0] if "throughput_rps" in df.columns else 0
        )
        p50 = df["latency_p50_ms"].iloc[0] if "latency_p50_ms" in df.columns else 0
        p99 = df["latency_p99_ms"].iloc[0] if "latency_p99_ms" in df.columns else 0
        print(f"{server_name:<25} {throughput:<20.2f} {p50:<20.2f} {p99:<20.2f}")

    print("=" * 80)
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
