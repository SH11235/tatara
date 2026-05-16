#!/usr/bin/env bash
# cuda-oxide の kernel ビルドツール `cargo-oxide` を、本リポジトリが pin して
# いる cuda-oxide の git rev に合わせて install するセットアップスクリプト。
#
# GPU kernel は cuda-oxide の rustc codegen backend で PTX 化する。その backend
# を駆動する cargo subcommand `cargo-oxide` は、本リポジトリの依存 (cuda-core 等)
# とは別に利用者の環境へ install する必要がある。cuda-oxide repo の手動 clone は
# 不要で、upstream が公式サポートする `cargo install --git` を使う。rev は
# Cargo.lock が固定している cuda-oxide の rev をそのまま使い、library 側と
# codegen backend 側を同一 rev に揃える (rev ずれは codegen backend の ABI 不一致
# を招く)。
#
# このスクリプトはシステムパッケージ (CUDA / LLVM / clang) を install しない。
# 不足はチェックして報告するだけなので、導入手順は docs/setup.md を参照。
#
# 使い方:
#   bash scripts/setup-cuda-oxide.sh   # cargo-oxide を pin rev で install + 環境診断
#
# cargo-oxide は毎回 pin rev で入れ直すため、cuda-oxide rev を bump した後も
# 同じコマンドを再実行すればよい。
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_OXIDE_GIT="https://github.com/NVlabs/cuda-oxide.git"

# pin している cuda-oxide rev を Cargo.lock から読む。rev の単一情報源は root
# Cargo.toml の [workspace.dependencies] で、Cargo.lock はその解決結果。
rev="$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+' Cargo.lock | sed 's/.*rev=//' || true)"
if [[ -z "$rev" ]]; then
  echo "ERROR: Cargo.lock から cuda-oxide の rev を取得できませんでした。" >&2
  echo "       リポジトリ root で、cuda-oxide を依存に持つ状態で実行してください。" >&2
  exit 1
fi
echo "cuda-oxide pinned rev: $rev"
echo

# ホスト前提のチェック (報告のみ、install はしない)。
missing=0
check() {  # check <表示名> <コマンド> <不足時の一言>
  if command -v "$2" >/dev/null 2>&1; then
    printf '  ok   %-7s %s\n' "$1" "$(command -v "$2")"
  else
    printf '  MISS %-7s %s\n' "$1" "$3"
    missing=1
  fi
}

echo "host 前提のチェック:"
check rustup rustup "rustup 経由で nightly + rust-src / rustc-dev component が要る"
check cargo  cargo  "Rust toolchain が要る"
check git    git    "cargo install --git が git を呼ぶ"
# llc は llc-22 → llc-21 の順で cuda-oxide が探す。どれか 1 つあれば良い。
if command -v llc-22 >/dev/null 2>&1 || command -v llc-21 >/dev/null 2>&1 \
   || command -v llc >/dev/null 2>&1; then
  printf '  ok   %-7s\n' "llc"
else
  printf '  MISS %-7s %s\n' "llc" "LLVM 21+ (llc-22 推奨)。docs/setup.md の「システム install」参照"
  missing=1
fi
check clang clang "clang + libclang-dev が cuda-bindings の bindgen に要る"
check nvcc  nvcc  "CUDA Toolkit 12.x (libcublas 含む) が要る"
echo

if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo が無いため続行できません。docs/setup.md を参照。" >&2
  exit 1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "WARN: 不足している前提があります。cuda-oxide のビルドで失敗する場合は"
  echo "      docs/setup.md の手順で導入してください。続行します。"
  echo
fi

# cargo-oxide を pin rev で install する。既に別 rev が入っていると codegen
# backend の ABI がずれるため、毎回 --force で pin rev に入れ直す (cargo-oxide
# 本体は小さく rebuild は数秒)。skip 判定で別 rev を見逃さないための方針。
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
echo "cargo-oxide を install します: cargo install --git ... --rev $rev"
cargo install --git "$CUDA_OXIDE_GIT" --rev "$rev" cargo-oxide --force
echo

# install 先 ($cargo_bin/cargo-oxide) が PATH 上で解決されるか確認する。
oxide="$cargo_bin/cargo-oxide"
oxide_on_path="$(command -v cargo-oxide || true)"
if [[ -z "$oxide_on_path" ]]; then
  echo "WARN: cargo-oxide が PATH にありません。次を shell rc に追加してください:"
  echo "      export PATH=\"$cargo_bin:\$PATH\""
  echo
elif [[ "$oxide_on_path" != "$oxide" ]]; then
  echo "WARN: PATH 上の cargo-oxide ($oxide_on_path) が install 先 $oxide を"
  echo "      shadow しています。pin rev を使うには $cargo_bin を PATH 前方に置いてください。"
  echo
fi

# 環境診断。install したばかりの pin rev の cargo-oxide で実行する。初回は
# codegen backend を自動で取得・ビルドしてキャッシュするため時間がかかる。
if [[ -x "$oxide" ]]; then
  echo "cargo oxide doctor:"
  doctor_status=0
  "$oxide" doctor || doctor_status=$?
  echo
  if [[ "$doctor_status" -ne 0 ]]; then
    echo "WARN: doctor が問題を報告しました (上記)。docs/setup.md で対処してください。"
  fi
else
  echo "WARN: $oxide が見つからないため doctor を skip しました。"
fi

echo
echo "次のステップ — kernel を PTX 化してビルド (CUDA_OXIDE_TARGET は GPU 世代に合わせる):"
echo "  (cd bins/nnue_train           && CUDA_OXIDE_TARGET=sm_86 cargo-oxide build)"
echo "  (cd bins/progress_kpabs_train && CUDA_OXIDE_TARGET=sm_86 cargo-oxide build)"
echo "GPU 世代と sm_XX の対応は docs/setup.md のサポート GPU マトリクスを参照。"
