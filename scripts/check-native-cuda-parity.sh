#!/usr/bin/env bash
# Compare the CUDA C++ kernels with cuda-oxide, then compare the cuda-oxide and portable host
# runtimes in separate test processes. This script requires a CUDA GPU and NVCC.
set -euo pipefail

cd "$(dirname "$0")/.."

hybrid_log=$(mktemp /tmp/tatara-native-hybrid.XXXXXX)
portable_log=$(mktemp /tmp/tatara-native-portable.XXXXXX)
portable_cli_dir=$(mktemp -d /tmp/tatara-native-cli.XXXXXX)
trap 'rm -f -- "$hybrid_log" "$portable_log"; rm -r -- "$portable_cli_dir"' EXIT

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

echo "== portable host Simple configuration matrix =="
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release \
    complete_simple_native_configuration_matrix_runs_one_step -- --nocapture --test-threads=1

echo "== portable host CLI smoke =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple

echo "== portable host full Simple CLI training =="
portable_cli_args=(
    --data crates/shogi-format/tests/data/sample.psv
    --output "$portable_cli_dir"
    --feature-set halfka-hm-merged --arch 8x2-8-8 --activation pairwise
    --batches-per-superbatch 1 --batch-size 64 --threads 1 --save-rate 1
    --win-rate-model --scale 600 --wrm-nnue2score 600
    --loss-pow-exp 2.5 --loss-qp-asymmetry 0.2
    --loss-weight-boost-w1 1.5 --loss-weight-boost-w2 0.75
    --optimizer adamw --weight-decay 0.0001
    --norm-loss --norm-loss-factor 0.0001 --all-optim
)
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple \
    "${portable_cli_args[@]}" --net-id native-simple-cli --superbatches 1
test -s "$portable_cli_dir/native-simple-cli-1.bin"
test -s "$portable_cli_dir/native-simple-cli-1.ckpt"

echo "== portable host Simple CLI resume =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple \
    "${portable_cli_args[@]}" --net-id native-simple-cli-resume --superbatches 2 \
    --resume "$portable_cli_dir/native-simple-cli-1.ckpt"
test -s "$portable_cli_dir/native-simple-cli-resume-2.bin"
test -s "$portable_cli_dir/native-simple-cli-resume-2.ckpt"

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
