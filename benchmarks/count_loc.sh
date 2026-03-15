#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERS_DIR="$SCRIPT_DIR/servers"

count_loc() {
    local dir="$1"
    local name="$2"
    local total=0
    
    while IFS= read -r file; do
        local count=$(grep -v '^\s*$' "$file" | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
        total=$((total + count))
    done < <(find "$dir" -name "*.rs" -type f)
    
    echo "$name: $total lines"
    echo "$total"
}

echo "===================================="
echo "Lines of Code (LOC) Comparison"
echo "===================================="
echo ""

echo "Server Implementations:"
echo "----------------------"

ort_superserve=$(find "$SERVERS_DIR/ort-superserve-server" -name "*.rs" -type f -exec cat {} \; | grep -v '^\s*$' | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
echo "ort-superserve-server: $ort_superserve lines"

actix_batch=$(find "$SERVERS_DIR/actix-with-batching-server" -name "*.rs" -type f -exec cat {} \; | grep -v '^\s*$' | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
echo "actix-with-batching-server: $actix_batch lines"

actix_no_batch=$(find "$SERVERS_DIR/actix-without-batching-server" -name "*.rs" -type f -exec cat {} \; | grep -v '^\s*$' | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
echo "actix-without-batching-server: $actix_no_batch lines"

arc_mutex=$(find "$SERVERS_DIR/arc-mutex-server" -name "*.rs" -type f -exec cat {} \; | grep -v '^\s*$' | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
echo "arc-mutex-server: $arc_mutex lines"

batched_fn=$(find "$SERVERS_DIR/batched-fn-server" -name "*.rs" -type f -exec cat {} \; | grep -v '^\s*$' | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
echo "batched-fn-server: $batched_fn lines"

echo ""
echo "Shared Library (used by all servers):"
echo "--------------------------------------"
shared_loc=$(find "$SERVERS_DIR/shared" -name "*.rs" -type f -exec cat {} \; | grep -v '^\s*$' | grep -v '^\s*//' | grep -v '^\s*///' | grep -v '^\s*//!' | wc -l)
echo "shared: $shared_loc lines"

echo ""
echo "Summary"
echo "-------"
echo "Server-specific code only (excludes shared library):"
echo ""
printf "%-35s %5s\n" "ort-superserve-server" "$ort_superserve"
printf "%-35s %5s\n" "actix-with-batching-server" "$actix_batch"
printf "%-35s %5s\n" "actix-without-batching-server" "$actix_no_batch"
printf "%-35s %5s\n" "arc-mutex-server" "$arc_mutex"
printf "%-35s %5s\n" "batched-fn-server" "$batched_fn"
echo ""
echo "Total per server (including shared library):"
echo ""
printf "%-35s %5s\n" "ort-superserve-server" "$((ort_superserve + shared_loc))"
printf "%-35s %5s\n" "actix-with-batching-server" "$((actix_batch + shared_loc))"
printf "%-35s %5s\n" "actix-without-batching-server" "$((actix_no_batch + shared_loc))"
printf "%-35s %5s\n" "arc-mutex-server" "$((arc_mutex + shared_loc))"
printf "%-35s %5s\n" "batched-fn-server" "$((batched_fn + shared_loc))"