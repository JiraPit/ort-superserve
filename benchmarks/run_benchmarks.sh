#!/bin/bash
set -e

cd "$(dirname "$0")"

# Download data if needed
if [ ! -f "data/resnet50-v1-12-int8.onnx" ] || [ ! -d "data/images" ]; then
    echo "Downloading ResNet50 data..."
    cd download_data
    uv run download-data --data-dir ../data
    cd ..
fi

# Build all servers
echo "Building servers..."
cargo build --release

# Ports
declare -A PORTS=(
    ["ort-superserve"]=3001
    ["actix-with-batching"]=3002
    ["actix-without-batching"]=3003
    ["arc-mutex"]=3004
    ["batched-fn"]=3005
    ["ort-superserve-8-sessions"]=3006
)

# Create results directory
mkdir -p results

# Run benchmarks
for server in ort-superserve ort-superserve-8-sessions actix-with-batching actix-without-batching arc-mutex batched-fn; do
    port=${PORTS[$server]}
    echo ""
    echo "================================================"
    echo "Benchmarking $server on port $port"
    echo "================================================"
    
    # Start server
    echo "Starting $server..."
    cargo run --release --bin ${server}-server &
    SERVER_PID=$!
    
    # Wait for warmup
    sleep 5
    
    # Run benchmark
    echo "Running benchmark..."
    cargo run --release --bin bench-client -- \
        --server $server \
        --port $port \
        --output results/${server}.csv \
        --ramp-duration 60 \
        --hold-duration 120 \
        --max-concurrency 3000
    
    # Stop server
    echo "Stopping $server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    
    echo "Benchmark for $server complete"
    sleep 2
done

echo ""
echo "================================================"
echo "All benchmarks complete!"
echo "================================================"

# Generate plots
echo "Generating plots..."
cd plots
uv run generate-plots --results-dir ../results

echo "Done! Results are in results/ directory."