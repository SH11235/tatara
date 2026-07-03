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

# cargo-oxide の codegen backend cache (~/.cargo/cuda-oxide/) は
# scripts/setup-cuda-oxide.sh だけが pin rev に揃える/揃っているか検証する
# (詳細は同スクリプトのコメント参照)。ここでは同スクリプトが書いた stamp と
# Cargo.lock の pin を比べるだけの軽い確認に留め、build 自体では cache を
# 書き換えない (build のたびに fetch/rebuild すると遅くなるため)。stamp が
# 無い/ずれているときに実際に device codegen エラーになるとは限らないが、
# 原因切り分けの入口として fail-fast する。
if [[ -z "${CUDA_OXIDE_BACKEND:-}" ]]; then
  full_rev="$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+#[0-9a-f]+' Cargo.lock | sed -E 's/.*#//' || true)"
  stamp_file="${CARGO_HOME:-$HOME/.cargo}/cuda-oxide/.pin-stamp"
  expected_stamp="$full_rev|$(rustc --version)"
  actual_stamp=""
  [[ -n "$full_rev" && -f "$stamp_file" ]] && actual_stamp="$(cat "$stamp_file")"
  if [[ -z "$full_rev" || "$actual_stamp" != "$expected_stamp" ]]; then
    echo "error: cuda-oxide の codegen backend cache が pin rev / toolchain と一致しません。" >&2
    echo "       bash scripts/setup-cuda-oxide.sh を再実行してから再試行してください。" >&2
    exit 1
  fi
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
