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

# libdevice call と LLVM atomic operation が同じ module にある場合、pre-Blackwell
# 向け legacy NVVM IR は atomic operation を表現できない。modern NVVM IR を一度
# 生成し、libdevice を link してから LLVM NVPTX backend で実行対象の PTX に落とす。
# modern IR の target は利用可能な命令を制限するだけで、最終 PTX target は下の
# llc --mcpu が決める。
ptx_target="${CUDA_OXIDE_TARGET:-sm_80}"
target_number="${ptx_target#sm_}"
target_number="${target_number%%[a-z]*}"
if [[ "$target_number" =~ ^[0-9]+$ && "$target_number" -ge 100 ]]; then
  nvvm_target="$ptx_target"
else
  nvvm_target="sm_100"
fi

find_llvm_tool() {
  local env_name="$1"
  local tool="$2"
  local configured="${!env_name:-}"
  if [[ -n "$configured" ]]; then
    echo "$configured"
    return
  fi
  command -v "$tool-22" || command -v "$tool-21" || command -v "$tool"
}

find_libdevice() {
  local root
  for root in "${CUDA_TOOLKIT_PATH:-}" "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda; do
    if [[ -n "$root" && -f "$root/nvvm/libdevice/libdevice.10.bc" ]]; then
      echo "$root/nvvm/libdevice/libdevice.10.bc"
      return
    fi
  done
  local candidate
  candidate="$(compgen -G '/usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc' | head -1 || true)"
  [[ -n "$candidate" ]] && echo "$candidate"
}

llvm_link="$(find_llvm_tool LLVM_LINK_BIN llvm-link)"
opt_bin="$(find_llvm_tool OPT_BIN opt)"
llc_bin="$(find_llvm_tool LLC_BIN llc)"
libdevice="$(find_libdevice)"
if [[ -z "$llvm_link" || -z "$opt_bin" || -z "$llc_bin" || -z "$libdevice" ]]; then
  echo "error: PTX 生成に必要な llvm-link / opt / llc / libdevice.10.bc が見つかりません。" >&2
  exit 1
fi

# kernel を持つ bin をすべてビルドする。
for bin in nnue_train progress_kpabs_train; do
  echo "[build-kernels] cargo-oxide build: bins/$bin (modern NVVM IR: $nvvm_target)"
  ( cd "bins/$bin" && cargo-oxide build --emit-nvvm-ir --arch "$nvvm_target" )

  ll="$repo_root/$bin.ll"
  if [[ ! -f "$ll" ]]; then
    ll="$repo_root/bins/$bin/$bin.ll"
  fi
  if [[ ! -f "$ll" ]]; then
    echo "error: cargo-oxide が $bin.ll を生成しませんでした。" >&2
    exit 1
  fi

  artifact_dir="$(dirname "$ll")"
  linked_bc="$artifact_dir/$bin.linked.bc"
  opt_bc="$artifact_dir/$bin.opt.bc"
  ptx="$artifact_dir/$bin.ptx"
  "$llvm_link" "$ll" "$libdevice" -o "$linked_bc"
  # internalize が kernel entry を internal 化しても消えないのは、cargo-oxide の
  # emit する .ll が全 kernel を @llvm.used に載せているため (globaldce の生存根拠)。
  "$opt_bin" --passes=nvvm-reflect,internalize,globaldce "$linked_bc" -o "$opt_bc"
  "$llc_bin" --mtriple=nvptx64-nvidia-cuda --mcpu="$ptx_target" -O2 "$opt_bc" -o "$ptx"
  rm -f "$artifact_dir/$bin.options" "$artifact_dir/$bin.target"
  echo "[build-kernels] PTX: $ptx ($(sha256sum "$ptx" | awk '{print $1}'))"
done

echo "[build-kernels] 完了。"
