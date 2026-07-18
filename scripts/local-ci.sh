#!/usr/bin/env bash
# CUDA + LLVM が揃った開発機で workspace 全体 (GPU crate 含む) の fmt / clippy /
# kernel build / test を回すスクリプト。GitHub Actions 側 (`.github/workflows/checks.yaml`) は
# GPU crate を exclude しているため、push 前にここで full check を回す必要がある。
#
# test は `--release` で実行する: GPU 数値同等性テストが debug build の f32 fma
# off で tolerance を満たさず fail するため (release は本番経路と同じ codegen)。
set -euo pipefail

cd "$(dirname "$0")/.."

# CUDA_OXIDE_TARGET はここでは既定値を与えない: kernel build は
# `scripts/build-kernels.sh` が GPU 世代を auto-detect し、test 側の kernel
# loader も env 未設定なら sm_75 (全世代で前方互換に動く既定) を使う。ここで
# sm_86 等を埋めると sub-Ampere host で auto-detect が無効化され invalid PTX に
# なる。特定 target を試すときだけ呼び出し側の環境変数で明示する。
: "${LLVM_LINK_BIN:=/usr/bin/llvm-link-22}"
: "${OPT_BIN:=/usr/bin/opt-22}"
: "${LLC_BIN:=/usr/bin/llc-22}"
export LLVM_LINK_BIN OPT_BIN LLC_BIN

echo "== cargo fmt --all -- --check =="
cargo fmt --all -- --check

echo "== cargo clippy --workspace --all-targets -- -D warnings =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== native CUDA feature compile coverage =="
cargo check -p nnue-trainer --features native-cuda
cargo check -p nnue-trainer --no-default-features --features native-cuda-host

# kernel source を編集したあと `cargo-oxide build` を忘れると、kernel loader の
# 鮮度チェックが `.ptx` vs `.ll` の mtime しか見ないため、test も本番 run も古い
# kernel のまま silent に走る。test の前に必ず再生成して artifact を source と
# 同期させる。cargo-oxide は build のたびに bin の main.rs を touch して再
# codegen を強制するため、warm cache でも本 step + 後続 test の bin 再ビルドで
# 数十秒掛かる。
echo "== bash scripts/build-kernels.sh (kernel artifacts) =="
bash scripts/build-kernels.sh

echo "== bash scripts/check-native-cuda-parity.sh =="
bash scripts/check-native-cuda-parity.sh

echo "== cargo test --workspace --release =="
cargo test --workspace --release

echo "PASS"
