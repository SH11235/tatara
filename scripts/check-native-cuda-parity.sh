#!/usr/bin/env bash
# Compare the CUDA C++ kernels with cuda-oxide, then compare the cuda-oxide and portable host
# runtimes in separate test processes. This script requires a CUDA GPU and NVCC.
set -euo pipefail

cd "$(dirname "$0")/.."

hybrid_log=$(mktemp /tmp/tatara-native-hybrid.XXXXXX)
portable_log=$(mktemp /tmp/tatara-native-portable.XXXXXX)
portable_cli_dir=$(mktemp -d /tmp/tatara-native-cli.XXXXXX)
portable_layerstack_cli_dir=$(mktemp -d /tmp/tatara-native-layerstack-cli.XXXXXX)
trap 'rm -f -- "$hybrid_log" "$portable_log"; rm -r -- "$portable_cli_dir" "$portable_layerstack_cli_dir"' EXIT

echo "== native CUDA artifact and edge-case tests =="
cargo test -p cuda-native-runtime --features native-cuda --release -- \
    --nocapture --test-threads=1

echo "== native CUDA production launch inventory =="
cargo test -p nnue-trainer --features native-cuda --release \
    native_inventory_parser -- --nocapture
cargo test -p nnue-trainer --features native-cuda --release \
    every_production_cuda_launch_is_exported -- --nocapture
cargo test -p nnue-trainer --features native-cuda --release \
    cuda_launch_stays_within_known_production_roots -- --nocapture
cargo test -p nnue-trainer --features native-cuda --release \
    native_bucket_capacity_matches_host -- --nocapture

echo "== native CUDA C++ kernels vs cuda-oxide =="
cargo test -p nnue-trainer --features native-cuda --release \
    simple_native_ -- --nocapture --test-threads=1

echo "== LayerStack CUDA C++ kernels vs cuda-oxide =="
cargo test -p nnue-trainer --features native-cuda --release \
    layerstack_native_ -- --nocapture --test-threads=1

echo "== cuda-oxide host fingerprint =="
cargo test -p nnue-trainer --features native-cuda --release \
    standard_simple_crelu_runs_one_native_training_step -- --nocapture --test-threads=1 \
    2>&1 | tee "$hybrid_log"
cargo test -p nnue-trainer --features native-cuda --release \
    standard_layerstack_runs_one_native_training_step -- --nocapture --test-threads=1 \
    2>&1 | tee -a "$hybrid_log"

echo "== portable host fingerprint =="
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release \
    standard_simple_crelu_runs_one_native_training_step -- --nocapture --test-threads=1 \
    2>&1 | tee "$portable_log"
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release \
    standard_layerstack_runs_one_native_training_step -- --nocapture --test-threads=1 \
    2>&1 | tee -a "$portable_log"

echo "== portable host Simple configuration matrix =="
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release \
    complete_simple_native_configuration_matrix_runs_one_step -- --nocapture --test-threads=1

echo "== portable host LayerStack configuration matrices =="
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release \
    complete_layerstack_native_ -- --nocapture --test-threads=1

echo "== portable host CLI smoke =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- layerstack

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

echo "== portable host LayerStack CLI training =="
portable_layerstack_cli_args=(
    --data crates/shogi-format/tests/data/sample.psv
    --test-data crates/shogi-format/tests/data/sample.psv --test-positions 16
    --output "$portable_layerstack_cli_dir"
    --feature-set halfkp --ft-out 128 --l1 16 --l2 32
    --bucket-mode kingrank9 --num-buckets 9
    --batches-per-superbatch 1 --batch-size 16 --threads 1 --save-rate 1
    --win-rate-model --optimizer ranger
)
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- layerstack \
    "${portable_layerstack_cli_args[@]}" --net-id native-layerstack-cli --superbatches 1
test -s "$portable_layerstack_cli_dir/native-layerstack-cli-1.bin"
test -s "$portable_layerstack_cli_dir/native-layerstack-cli-1.ckpt"

echo "== portable host LayerStack CLI resume =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- layerstack \
    "${portable_layerstack_cli_args[@]}" --net-id native-layerstack-cli-resume --superbatches 2 \
    --resume "$portable_layerstack_cli_dir/native-layerstack-cli-1.ckpt"
test -s "$portable_layerstack_cli_dir/native-layerstack-cli-resume-2.bin"
test -s "$portable_layerstack_cli_dir/native-layerstack-cli-resume-2.ckpt"

echo "== portable host LayerStack CLI eval-only =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- layerstack \
    "${portable_layerstack_cli_args[@]}" --net-id native-layerstack-cli-eval \
    --resume "$portable_layerstack_cli_dir/native-layerstack-cli-1.ckpt" --eval-only

echo "== portable host LayerStack YaneuraOu output =="
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- layerstack \
    "${portable_layerstack_cli_args[@]}" --output-format yaneuraou \
    --net-id native-layerstack-cli-yo --superbatches 1
test -s "$portable_layerstack_cli_dir/native-layerstack-cli-yo-1.bin"
test -s "$portable_layerstack_cli_dir/native-layerstack-cli-yo-1.ckpt"

extract_fingerprint() {
    sed -n 's/^.*\[native-host-parity\] //p' "$1" | tail -n 1
}

extract_layerstack_fingerprint() {
    sed -n 's/^.*\[native-layerstack-host-parity\] //p' "$1" | tail -n 1
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

hybrid_layerstack_fingerprint=$(extract_layerstack_fingerprint "$hybrid_log")
portable_layerstack_fingerprint=$(extract_layerstack_fingerprint "$portable_log")
if [[ -z "$hybrid_layerstack_fingerprint" || -z "$portable_layerstack_fingerprint" ]]; then
    echo "LayerStack native parity fingerprint was not emitted" >&2
    exit 1
fi
if [[ "$hybrid_layerstack_fingerprint" != "$portable_layerstack_fingerprint" ]]; then
    echo "LayerStack native host parity mismatch" >&2
    echo "  cuda-oxide host: $hybrid_layerstack_fingerprint" >&2
    echo "  portable host:   $portable_layerstack_fingerprint" >&2
    exit 1
fi

echo "LayerStack native host parity matched: $portable_layerstack_fingerprint"
