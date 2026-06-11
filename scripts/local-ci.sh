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

# kernel source を編集したあと `cargo-oxide build` を忘れると、kernel loader の
# 鮮度チェックが `.ptx` vs `.ll` の mtime しか見ないため、test も本番 run も古い
# kernel のまま silent に走る。test の前に必ず再生成して artifact を source と
# 同期させる。cargo-oxide は build のたびに bin の main.rs を touch して再
# codegen を強制するため、warm cache でも本 step + 後続 test の bin 再ビルドで
# 数十秒掛かる。
echo "== bash scripts/build-kernels.sh (kernel artifacts) =="
bash scripts/build-kernels.sh

echo "== cargo test --workspace --release =="
cargo test --workspace --release

echo "PASS"
