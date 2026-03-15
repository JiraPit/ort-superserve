# ONNX Session Management Benchmarks

This directory contains benchmarks comparing different approaches to ONNX session management for high-throughput model serving.

## Benchmark Servers

| Server | Port | Description |
|--------|------|-------------|
| `ort-superserve-server` | 3001 | ort-superserve library with dynamic batching and JoinSet parallelism |
| `actix-with-batching-server` | 3002 | Manual actix actors with batching (preprocessing in async handlers) |
| `actix-without-batching-server` | 3003 | Manual actix actor without batching (single worker) |
| `arc-mutex-server` | 3004 | Naive `Arc<Mutex<Session>>` baseline |
| `batched-fn-server` | 3005 | Using `batched-fn` crate for transparent batching (parallel preprocessing in handlers) |

## Architecture Comparison

| Server | Batching | Preprocessing | Postprocessing | Session Count | Intra Threads |
|--------|----------|---------------|----------------|---------------|---------------|
| ort-superserve | ✅ Dynamic | ✅ JoinSet (parallel within batch) | ✅ JoinSet (parallel within batch) | 1 | 4 |
| actix-with-batching | ✅ Manual | ✅ Async handlers (parallel across requests) | ✅ Async handlers (parallel across requests) | 1 | 4 |
| actix-without-batching | ❌ | ✅ Async handlers | ✅ Async handlers | 1 | 4 |
| arc-mutex | ❌ | ✅ Async handlers | ✅ Async handlers | 1 | 4 |
| batched-fn | ✅ Macro | ✅ Async handlers (parallel across requests) | ✅ Async handlers (parallel across requests) | 1 | 4 |

### Preprocessing Parallelism Explained

- **ort-superserve**: When a batch of N items is ready, spawns N async tasks via `JoinSet` to preprocess all items **within the batch** in parallel, then batches and sends to worker.
- **actix-with-batching**: Each HTTP handler preprocesses its own request before sending to the batcher. Multiple handlers run **concurrently**, so preprocessing is parallel **across requests**.
- **batched-fn**: Each HTTP handler preprocesses its own request before sending to the batcher. Multiple handlers run **concurrently**, so preprocessing is parallel **across requests**.

## Prerequisites

1. **Rust** - Install from https://rustup.rs
2. **Python 3** - For download script and plotting
3. **MNIST Model** - Automatically downloaded

## Quick Start

```bash
# Download MNIST data (model and test images)
python3 download_data.py

# Build all servers
cargo build --release

# Run all benchmarks
./run_benchmarks.sh
```

## Manual Benchmarking

```bash
# Start a specific server
cargo run --release --bin ort-superserve-server

# In another terminal, run the benchmark client
cargo run --release --bin bench-client -- \
    --server ort-superserve \
    --port 3001 \
    --output results/ort-superserve.csv \
    --ramp-duration 60 \
    --hold-duration 30 \
    --max-concurrency 2048
```

## Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ramp-duration` | 60 | Seconds to ramp from 1 to max concurrency |
| `--hold-duration` | 30 | Seconds to hold at max concurrency |
| `--max-concurrency` | 2048 | Maximum concurrent requests |
| `--warmup-requests` | 10 | Warmup requests before benchmarking |

## Results

After running benchmarks, results are stored in `results/` as CSV files with:

- `server` - Server name
- `total_requests` - Total requests completed
- `duration_sec` - Total benchmark duration
- `throughput_rps` - Requests per second
- `latency_p50_ms` - Median latency
- `latency_p90_ms` - 90th percentile latency
- `latency_p99_ms` - 99th percentile latency

## Generating Plots

```bash
cd plots
pip install -r requirements.txt
python3 plot.py
```

Plots are saved to `results/`:
- `latency_comparison.png` - Latency comparison bar chart
- `throughput_comparison.png` - Throughput comparison bar chart

## Directory Structure

```
benchmarks/
├── Cargo.toml                    # Workspace definition
├── download_data.py              # MNIST data download script
├── run_benchmarks.sh             # Orchestration script
├── data/
│   ├── mnist-12.onnx            # ONNX model
│   └── images/                   # MNIST test images (PNG)
├── results/                      # Benchmark results (CSV, PNG)
├── plots/
│   ├── plot.py                  # Matplotlib visualization
│   └── requirements.txt         # Python dependencies
├── servers/
│   ├── shared/                  # Common types and utilities
│   ├── arc-mutex-server/        # Arc<Mutex> baseline
│   ├── actix-without-batching-server/  # Actix actor, no batching
│   ├── actix-with-batching-server/     # Actix actors with batching
│   ├── batched-fn-server/       # batched-fn macro approach
│   └── ort-superserve-server/   # ort-superserve library
└── bench-client/               # Benchmark client
```

## Model

Uses MNIST-12 ONNX model from the ONNX Model Zoo:
- Input: `[batch_size, 1, 28, 28]` - Grayscale images
- Output: `[batch_size, 10]` - Class logits

## License

MIT