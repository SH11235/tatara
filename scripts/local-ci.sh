#!/usr/bin/env bash
# CUDA + LLVM が揃った開発機で workspace 全体 (GPU crate 含む) の fmt / clippy /
# test を回すスクリプト。GitHub Actions 側 (`.github/workflows/checks.yaml`) は
# GPU crate を exclude しているため、push 前にここで full check を回す必要がある。
#
# test は `--release` で実行する: GPU 数値同等性テストが debug build の f32 fma
# off で tolerance を満たさず fail するため (release は本番経路と同じ codegen)。
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_OXIDE_TARGET:=sm_86}"
: "${LLVM_LINK_BIN:=/usr/bin/llvm-link-22}"
: "${OPT_BIN:=/usr/bin/opt-22}"
: "${LLC_BIN:=/usr/bin/llc-22}"
export CUDA_OXIDE_TARGET LLVM_LINK_BIN OPT_BIN LLC_BIN

echo "== cargo fmt --all -- --check =="
cargo fmt --all -- --check

echo "== cargo clippy --workspace --all-targets -- -D warnings =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== cargo test --workspace --release =="
cargo test --workspace --release

echo "PASS"
