#!/usr/bin/env bash
# cuda-oxide の kernel ビルドツール `cargo-oxide` と、その codegen backend
# cache (`~/.cargo/cuda-oxide/`) を、本リポジトリが pin している cuda-oxide の
# git rev に揃えるセットアップスクリプト。
#
# GPU kernel は cuda-oxide の rustc codegen backend で PTX 化する。その backend
# を駆動する cargo subcommand `cargo-oxide` は、本リポジトリの依存 (cuda-core 等)
# とは別に利用者の環境へ install する必要がある。cuda-oxide repo の手動 clone は
# 不要で、upstream が公式サポートする `cargo install --git` を使う。rev は
# Cargo.lock が固定している cuda-oxide の rev をそのまま使い、library 側と
# codegen backend 側を同一 rev に揃える (rev ずれは codegen backend の ABI 不一致
# を招く)。CLI 本体だけでなく codegen backend の cache 自体も pin rev に揃える
# 理由は本文中のコメントを参照。
#
# このスクリプトはシステムパッケージ (CUDA / LLVM / clang) を install しない。
# 不足はチェックして報告するだけなので、導入手順は docs/setup.md を参照。
#
# 使い方:
#   bash scripts/setup-cuda-oxide.sh   # cargo-oxide + backend cache を pin rev に揃えて環境診断
#
# 毎回 pin rev に揃え直す (一致していれば no-op) ため、cuda-oxide rev を
# bump した後や、意味不明な device codegen エラーに遭遇したときは、まず
# 同じコマンドを再実行すればよい。
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_OXIDE_GIT="https://github.com/NVlabs/cuda-oxide.git"

# pin している cuda-oxide rev を Cargo.lock から読む。rev の単一情報源は root
# Cargo.toml の [workspace.dependencies] で、Cargo.lock はその解決結果。short rev
# は `cargo install --rev` に、full rev (40 桁 SHA) は後段の backend cache 検証に使う。
rev="$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+' Cargo.lock | sed 's/.*rev=//' || true)"
full_rev="$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+#[0-9a-f]+' Cargo.lock | sed -E 's/.*#//' || true)"
if [[ -z "$rev" || -z "$full_rev" ]]; then
  echo "ERROR: Cargo.lock から cuda-oxide の rev を取得できませんでした。" >&2
  echo "       リポジトリ root で、cuda-oxide を依存に持つ状態で実行してください。" >&2
  exit 1
fi
echo "cuda-oxide pinned rev: $rev ($full_rev)"
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

cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
oxide="$cargo_bin/cargo-oxide"
cli_stamp_file="$cargo_bin/.cargo-oxide-pin-stamp"
current_cli_stamp=""
if [[ -x "$oxide" && -f "$cli_stamp_file" ]]; then
  current_cli_hash="$(sha256sum "$oxide" | awk '{print $1}')"
  current_cli_stamp="$(cat "$cli_stamp_file")"
fi

# cargo-oxide は CLI binary が backend cache より新しいと cache を無効化する。
# 同じ rev を --force で再 install して binary の更新時刻だけを進めないよう、
# pin rev と binary hash の stamp が一致する場合は install を省略する。
if [[ "$current_cli_stamp" == "$full_rev|${current_cli_hash:-}" ]]; then
  echo "cargo-oxide CLI は pin rev と一致済み ($rev) — 再 install を skip"
else
  echo "cargo-oxide を install します: cargo install --git ... --rev $rev"
  cargo install --git "$CUDA_OXIDE_GIT" --rev "$rev" cargo-oxide --force
  if [[ -x "$oxide" ]]; then
    cli_hash="$(sha256sum "$oxide" | awk '{print $1}')"
    echo "$full_rev|$cli_hash" > "$cli_stamp_file"
  fi
fi
echo

# install 先 ($cargo_bin/cargo-oxide) が PATH 上で解決されるか確認する。
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

# cargo-oxide 本体 (`cargo install --rev`) と codegen backend の実体は別物。
# backend は `~/.cargo/cuda-oxide/` に一度だけ (`main` HEAD の unpinned shallow
# clone で) fetch・build・cache され、以降は存在するかどうかしか見ない —
# つまり CLI を pin rev で入れ直しても、初回 fetch 時点の main HEAD のままの
# backend が黙って使われ続けうる。この不一致は host/GPU/LLVM のバージョン差では
# 出ず、cuda-oxide 自身のこの cache 設計に起因する device codegen の不可解な
# 失敗として現れる。ここでは cache が指す rev + rustc nightly version (backend
# .so は正確な nightly ABI に紐づくため) を stamp file に記録し、pin と食い違う
# ときだけ cache 全体を破棄して pin rev で再取得する。`doctor` が cache 不在を
# `cargo oxide setup` が提供する backend build 経路を使い、独自の build
# コマンドは持たない。
if [[ -n "${CUDA_OXIDE_BACKEND:-}" ]]; then
  echo "WARN: CUDA_OXIDE_BACKEND=$CUDA_OXIDE_BACKEND が設定されています。"
  echo "      この env var は下記の pin 検証より優先されるため、以降の"
  echo "      backend cache 検証は skip されます。"
  echo
fi

backend_cache_dir="${CARGO_HOME:-$HOME/.cargo}/cuda-oxide"
backend_src_dir="$backend_cache_dir/src"
backend_so="$backend_cache_dir/librustc_codegen_cuda.so"
stamp_file="$backend_cache_dir/.pin-stamp"
expected_stamp="$full_rev|$(rustc --version)"

if [[ -z "${CUDA_OXIDE_BACKEND:-}" ]]; then
  current_stamp=""
  [[ -f "$stamp_file" ]] && current_stamp="$(cat "$stamp_file")"
  current_source_rev=""
  if [[ -d "$backend_src_dir" ]]; then
    current_source_rev="$(git -C "$backend_src_dir" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ "$current_stamp" == "$expected_stamp" && "$current_source_rev" == "$full_rev" ]]; then
    echo "cuda-oxide backend cache は pin rev と一致済み ($rev) — 再取得を skip"
  else
    echo "cuda-oxide backend cache が pin rev と不一致 (または未取得) — 破棄して"
    echo "rev $rev で取り直します。"
    rm -rf "$backend_cache_dir"
    mkdir -p "$backend_src_dir"
    git -C "$backend_src_dir" init -q
    git -C "$backend_src_dir" remote add origin "$CUDA_OXIDE_GIT"
    if ! git -C "$backend_src_dir" fetch --depth 1 origin "$full_rev" -q \
        || ! git -C "$backend_src_dir" checkout -q FETCH_HEAD; then
      echo "ERROR: cuda-oxide rev $full_rev の fetch に失敗しました。ネットワーク/rev を確認してください。" >&2
      exit 1
    fi
  fi
  echo
fi

# install した pin rev の source から backend を build する。`doctor` は診断専用で
# backend を build しないため、cache が無い場合は明示的に `setup` を実行する。
if [[ -z "${CUDA_OXIDE_BACKEND:-}" && ! -f "$backend_so" && -x "$oxide" ]]; then
  echo "cuda-oxide codegen backend を build します:"
  "$oxide" setup
  echo
fi

# 環境診断。install したばかりの pin rev の cargo-oxide で実行する。
if [[ -x "$oxide" ]]; then
  echo "cargo-oxide doctor:"
  doctor_status=0
  "$oxide" doctor || doctor_status=$?
  echo
  if [[ "$doctor_status" -ne 0 ]]; then
    echo "WARN: doctor が問題を報告しました (上記)。docs/setup.md で対処してください。"
  fi
else
  echo "WARN: $oxide が見つからないため doctor を skip しました。"
fi

# backend .so が実際に存在するかどうかで stamp を書く。doctor の他項目
# (nvcc 等) の成否とは独立 — .so の有無だけが pin 一致の根拠になる。
if [[ -z "${CUDA_OXIDE_BACKEND:-}" ]]; then
  if [[ -f "$backend_so" ]]; then
    echo "$expected_stamp" > "$stamp_file"
    echo "cuda-oxide backend cache を pin rev ($rev) で確認、stamp を更新しました。"
  else
    echo "WARN: $backend_so が見つからないため pin stamp を書けませんでした。"
    echo "      上記の doctor 出力を確認し、解消後にこのスクリプトを再実行してください。"
  fi
  echo
fi

echo
echo "次のステップ — kernel を PTX 化してビルド (CUDA_OXIDE_TARGET は GPU 世代に合わせる):"
echo "  (cd bins/nnue_train           && CUDA_OXIDE_TARGET=sm_86 cargo-oxide build)"
echo "  (cd bins/progress_kpabs_train && CUDA_OXIDE_TARGET=sm_86 cargo-oxide build)"
echo "GPU 世代と sm_XX の対応は docs/setup.md のサポート GPU マトリクスを参照。"
