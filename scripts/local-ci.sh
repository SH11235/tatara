#!/usr/bin/env bash
# scripts/local-ci.sh
# PR 作成 / push 前必須の local check (fmt / clippy / test、全 crate)。
# `.github/workflows/checks.yaml` は GitHub-hosted runner に CUDA / LLVM が無いため
# GPU 依存 crate (gpu-runtime / experiments / progress-kpabs-train / nnue-trainer) を
# clippy / test の workspace から exclude するが、本機 (CUDA + LLVM 22 install 済)
# では **exclude なし全 crate** をチェックする。
#
# test は `--release` で実行する: nnue-trainer の GPU 数値同等性テスト
# (`gpu_cpu_equivalence_tests`) は debug build の f32 fma off で tolerance を満たさず
# fail するため (release は本番経路と同じ codegen)。warm cache で fmt / clippy / test
# 計 ~20s。
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
