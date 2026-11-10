# ONNX Session Management Benchmarks

This directory contains benchmarks comparing different approaches to ONNX session management for high-throughput model serving using ResNet50.

## Preprocessing Pipeline

| Step | Description | Where it Happens |
|------|-------------|------------------|
| 1 | **Receive request** - Deserialize JSON with base64 PNG | Axum handler |
| 2 | **Decode + Resize + Crop** - PNG→RGB, scale to 256, center crop to 224x224 | Server (`ImageInput::to_input_array()`) |
| 3 | **Normalize** - Apply ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Server (Axum handler or `Input::preprocess()`) |
| 4 | **Inference** - Run ONNX model | Server |
| 5 | **Postprocess** - Convert logits to class_id + confidence | Server |

## Benchmark Servers

| Server | Port | Description |
|--------|------|-------------|
| `ort-superserve-server` | 3001 | ort-superserve library with dynamic batching |
| `ort-superserve-8-sessions-server` | 3006 | ort-superserve with 8 sessions |
| `ort-superserve-preprocess-server` | 3007 | ort-superserve with eager preprocessing (`preprocess_on_infer`); started by `start_servers.sh` only |
| `actix-with-batching-server` | 3002 | Manual actix actors with batching |
| `actix-without-batching-server` | 3003 | Manual actix actor without batching |
| `arc-mutex-server` | 3004 | Naive `Arc<Mutex<Session>>` baseline |
| `batched-fn-server` | 3005 | Using `batched-fn` crate for transparent batching |

## Quick Start

```bash
# Run all benchmarks (ResNet50 by default, or pass `mobilenet`).
# Downloads the model and test images into <repo>/test_assets/ if they are missing.
./run_benchmarks.sh
./run_benchmarks.sh mobilenet
```

To download the assets yourself, run from the repository root:

```bash
uv run --project download_test_assets download_test_assets/download_test_assets.py --model resnet50
```

This creates `test_assets/resnet50-v1-12-int8/`, containing the model and an `images/` directory. The servers load the model from there.

## Manual Benchmarking

```bash
# Start a specific server
cargo run --release --bin ort-superserve-server

# In another terminal, run the benchmark client
cargo run --release --bin bench-client -- \
    --server ort-superserve \
    --port 3001 \
    --data-dir ../test_assets/resnet50-v1-12-int8/images \
    --output results_resnet50/ort-superserve.csv \
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
| `--data-dir` | `../test_assets/resnet50-v1-12-int8/images` | Directory of PNG images to send |

## Model

ResNet50-v1-12-int8 (ImageNet):
- Input: `[batch_size, 3, 224, 224]` - RGB images normalized
- Output: `[batch_size, 1000]` - ImageNet class logits
