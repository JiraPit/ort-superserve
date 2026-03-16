#!/bin/bash
set -e

cd "$(dirname "$0")"

# Default model
MODEL="${1:-resnet50}"

# Optional: specify remote host to benchmark against
HOST="${2:-}"

# Validate model
if [[ "$MODEL" != "resnet50" && "$MODEL" != "mobilenet" ]]; then
    echo "Error: Invalid model '$MODEL'. Choose 'resnet50' or 'mobilenet'"
    exit 1
fi

echo "Starting servers with model: $MODEL"

# Model name mapping
case "$MODEL" in
    resnet50) MODEL_NAME="resnet50-v1-12-int8";;
    mobilenet) MODEL_NAME="mobilenetv2-12-int8";;
    *) MODEL_NAME="$MODEL";;
esac

export MODEL="$MODEL_NAME"

# Build if needed
if [ ! -f "target/release/ort-superserve-server" ]; then
    echo "Building servers..."
    cargo build --release
fi

# Ports and servers
declare -A SERVERS=(
    [3001]="ort-superserve-server"
    [3002]="actix-with-batching-server"
    [3003]="actix-without-batching-server"
    [3004]="arc-mutex-server"
    [3005]="batched-fn-server"
    [3006]="ort-superserve-8-sessions-server"
)

# Start all servers
for port in "${!SERVERS[@]}"; do
    server="${SERVERS[$port]}"
    echo "Starting $server on port $port..."
    MODEL=$MODEL_NAME cargo run --release --bin $server &
done

echo ""
echo "All servers started!"
echo "Ports:"
for port in "${!SERVERS[@]}"; do
    echo "  ${SERVERS[$port]}: $port"
done
echo ""
echo "To stop all servers, run: pkill -f 'ort-superserve-server|actix-|arc-mutex|batched-fn'"
