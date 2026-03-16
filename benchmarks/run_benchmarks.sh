#!/bin/bash
set -e

cd "$(dirname "$0")"

# Default model
MODEL_ARG="${1:-resnet50}"
shift || true

# Optional: specify remote host to benchmark against
HOST="${1:-}"

# Validate model
if [[ "$MODEL_ARG" != "resnet50" && "$MODEL_ARG" != "mobilenet" ]]; then
    echo "Error: Invalid model '$MODEL_ARG'. Choose 'resnet50' or 'mobilenet'"
    exit 1
fi

echo "Using model: $MODEL_ARG"

# Model name mapping (used as environment variable for servers)
case "$MODEL_ARG" in
    resnet50) MODEL_NAME="resnet50-v1-12-int8";;
    mobilenet) MODEL_NAME="mobilenetv2-12-int8";;
    *) MODEL_NAME="$MODEL_ARG";;
esac

export MODEL="$MODEL_NAME"

# Download data if needed
MODEL_FILE="${MODEL_NAME}.onnx"
if [ ! -f "data/$MODEL_FILE" ] || [ ! -d "data/images" ]; then
    echo "Downloading $MODEL_ARG data..."
    cd download_data
    uv run download-data --data-dir ../data --model $MODEL_ARG
    cd ..
fi

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
RESULTS_DIR="results_${MODEL_ARG}"
mkdir -p "$RESULTS_DIR"

# Determine if benchmarking remote host or local servers
if [ -n "$HOST" ]; then
    echo "================================================"
    echo "Benchmarking REMOTE host: $HOST"
    echo "================================================"
    
    for server in ort-superserve ort-superserve-8-sessions actix-with-batching actix-without-batching arc-mutex batched-fn; do
        port=${PORTS[$server]}
        echo ""
        echo "================================================"
        echo "Benchmarking $server on $HOST:$port"
        echo "================================================"
        
        # Run benchmark against remote
        MODEL=$MODEL_NAME cargo run --release --bin bench-client -- \
            --server $server \
            --host $HOST \
            --port $port \
            --output ${RESULTS_DIR}/${server}.csv \
            --ramp-duration 60 \
            --hold-duration 60 \
            --max-concurrency 2048
        
        sleep 2
    done
else
    # Run benchmarks on local servers
    for server in ort-superserve ort-superserve-8-sessions actix-with-batching actix-without-batching arc-mutex batched-fn; do
        port=${PORTS[$server]}
        echo ""
        echo "================================================"
        echo "Benchmarking $server on port $port"
        echo "================================================"
        
        # Start server
        echo "Starting $server with MODEL=$MODEL_NAME..."
        MODEL=$MODEL_NAME cargo run --release --bin ${server}-server &
        SERVER_PID=$!
        
        # Wait for warmup
        sleep 5
        
        # Run benchmark
        echo "Running benchmark..."
        cargo run --release --bin bench-client -- \
            --server $server \
            --port $port \
            --output ${RESULTS_DIR}/${server}.csv \
            --ramp-duration 60 \
            --hold-duration 60 \
            --max-concurrency 2048
        
        # Stop server
        echo "Stopping $server..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        
        echo "Benchmark for $server complete"
        sleep 2
    done
fi

echo ""
echo "================================================"
echo "All benchmarks complete!"
echo "================================================"

# Generate plots
echo "Generating plots..."
cd plots
uv run plot.py --results-dir ../${RESULTS_DIR}

echo "Done! Results are in ${RESULTS_DIR}/ directory."
