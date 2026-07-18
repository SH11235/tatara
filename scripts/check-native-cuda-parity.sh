#!/usr/bin/env bash
# Compare the CUDA C++ kernels with cuda-oxide, then compare the cuda-oxide and portable host
# runtimes in separate test processes. This script requires a CUDA GPU and NVCC.
set -euo pipefail

cd "$(dirname "$0")/.."

hybrid_log=$(mktemp /tmp/tatara-native-hybrid.XXXXXX)
portable_log=$(mktemp /tmp/tatara-native-portable.XXXXXX)
trap 'rm -f -- "$hybrid_log" "$portable_log"' EXIT

echo "== native CUDA C++ kernels vs cuda-oxide =="
cargo test -p nnue-trainer --features native-cuda --release \
    simple_native_ -- --nocapture --test-threads=1

echo "== cuda-oxide host fingerprint =="
cargo test -p nnue-trainer --features native-cuda --release \
    standard_simple_crelu_runs_one_native_training_step -- --nocapture --test-threads=1 \
    2>&1 | tee "$hybrid_log"

echo "== portable host fingerprint =="
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release \
    standard_simple_crelu_runs_one_native_training_step -- --nocapture --test-threads=1 \
    2>&1 | tee "$portable_log"

echo "== portable host CLI smoke =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple

extract_fingerprint() {
    sed -n 's/^.*\[native-host-parity\] //p' "$1" | tail -n 1
}

hybrid_fingerprint=$(extract_fingerprint "$hybrid_log")
portable_fingerprint=$(extract_fingerprint "$portable_log")
if [[ -z "$hybrid_fingerprint" || -z "$portable_fingerprint" ]]; then
    echo "native parity fingerprint was not emitted" >&2
    exit 1
fi
if [[ "$hybrid_fingerprint" != "$portable_fingerprint" ]]; then
    echo "native host parity mismatch" >&2
    echo "  cuda-oxide host: $hybrid_fingerprint" >&2
    echo "  portable host:   $portable_fingerprint" >&2
    exit 1
fi

echo "native host parity matched: $portable_fingerprint"
