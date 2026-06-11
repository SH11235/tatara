#!/usr/bin/env bash
# GPU の compute capability を検出して、kernel を持つ全 bin の GPU kernel を
# ビルドする。
#
# cargo-oxide の target auto-detect は kernel features から sm_80 を選ぶ。sm_80
# PTX は Ampere 以降 (sm_80 / 86 / 89 / 90 …) で前方互換に動くため、Ampere+ では
# 環境変数の指定は要らない。sub-Ampere GPU (Turing sm_75 等) のみ
# CUDA_OXIDE_TARGET の明示が要るので、このスクリプトが nvidia-smi で GPU 世代を
# 判定し、必要なときだけ自動設定する。
#
# 前提: cargo-oxide が install 済 (scripts/setup-cuda-oxide.sh)。
# 使い方: リポジトリ root から `bash scripts/build-kernels.sh`
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if ! command -v cargo-oxide >/dev/null 2>&1; then
  echo "error: cargo-oxide が PATH に無い。先に bash scripts/setup-cuda-oxide.sh を実行する。" >&2
  exit 1
fi

# CUDA_OXIDE_TARGET が既に設定済みならそれを尊重する (local-ci.sh 等の呼び出し
# 元が export してくる)。未設定のときだけ GPU の compute capability (例: "8.6")
# を取得し、sub-Ampere (sm < 80) の場合に限り自動設定する。
if [ -n "${CUDA_OXIDE_TARGET:-}" ]; then
  echo "[build-kernels] CUDA_OXIDE_TARGET=$CUDA_OXIDE_TARGET (既設の環境変数) でビルド"
else
  cc=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ' || true)
  fi

  if [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]]; then
    sm=${cc//./}
    if [ "$sm" -lt 80 ]; then
      export CUDA_OXIDE_TARGET="sm_$sm"
      echo "[build-kernels] compute_cap $cc (sub-Ampere) → CUDA_OXIDE_TARGET=sm_$sm を設定"
    else
      echo "[build-kernels] compute_cap $cc (Ampere+) → 既定 (sm_80 PTX、前方互換) でビルド"
    fi
  else
    echo "[build-kernels] warning: GPU 世代を判定できず、既定 (sm_80) でビルドする。" \
         "Turing 等 sub-Ampere GPU では CUDA_OXIDE_TARGET=sm_75 を手動指定すること。" >&2
  fi
fi

# kernel を持つ bin をすべてビルドする。
for bin in nnue_train progress_kpabs_train; do
  echo "[build-kernels] cargo-oxide build: bins/$bin"
  ( cd "bins/$bin" && cargo-oxide build )
done

echo "[build-kernels] 完了。"
