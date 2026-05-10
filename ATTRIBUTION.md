# Attribution

このリポジトリは以下のオープンソースプロジェクトから派生・参照しています。

## bullet-shogi (MIT)

- Source: https://github.com/SH11235/bullet-shogi
- Upstream: https://github.com/jw1912/bullet
- Use: PSV reader、ShogiBoard / Hand 等の format 周りを vendor (Stage 1〜)
- License: MIT

### 取り込み済 file (時系列で追記)

#### Stage 1-1 (2026-05-10, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/shogi/types.rs` → `crates/shogi-format/src/types.rs`
  (Color, PieceType, Square, Piece, Hand。完全一致 + `cargo fmt` 適用)
- `crates/bullet_lib/src/shogi/packed_sfen.rs` → `crates/shogi-format/src/packed_sfen.rs`
  (BitStream, PackedSfen, PackedSfenValue, ShogiBoard。完全一致から下記の差分:
  - `unsafe impl crate::value::loader::CanBeDirectlySequentiallyLoaded for PackedSfenValue {}` を削除 (bullet trait 依存を排除)
  - `impl crate::value::loader::LoadableDataType for PackedSfenValue { ... }` を削除し、`fn result(&self) -> crate::GameResult` を **inherent method** として書き直し
  - `cargo fmt` 適用)
- `crates/bullet_lib/src/shogi/bona_piece.rs` → `crates/shogi-format/src/bona_piece.rs`
  (BonaPiece 定数群。完全一致 + `cargo fmt` 適用)

新規追加 (bullet 由来ではない):

- `crates/shogi-format/src/game_result.rs` — bullet `crate::value::loader::GameResult` の最小サブセット (Loss=0, Draw=1, Win=2)。bullet trait に依存しないために自前定義
- `crates/shogi-format/src/lib.rs` — 上記 4 module の宣言と公開型 re-export
- `crates/shogi-format/Cargo.toml` — workspace member として最小設定
- `crates/shogi-format/tests/psv_smoke.rs` + `tests/data/sample.psv` (smoke_progress/smoke.bin の先頭 4000 bytes / 100 records)

#### Stage 1-2 (2026-05-10, bullet-shogi commit `f275eb9`)

- `crates/bullet_lib/src/game/outputs.rs` の `ShogiProgressKPAbs` 周辺
  → `crates/shogi-features/src/progress_kpabs.rs`
  (関連定数 `SHOGI_PROGRESS8_NUM_BUCKETS` `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`
   と static `SHOGI_PROGRESS_KP_ABS_WEIGHTS` `SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS`
   も同 file に同梱。**数値計算 path (for_each_active_index / progress / bucket
   / load_from_bin) は upstream と byte 一致**、下記の差分のみ:
  - `impl OutputBuckets<PackedSfenValue> for ShogiProgressKPAbs { ... }` を削除し、
    `bucket()` を **inherent method** として書き直し (bullet `OutputBuckets` trait
    依存を排除)。失われる `OutputBuckets::BUCKETS` const は
    `ShogiProgressKPAbs::BUCKETS` inherent const で代替
  - import path を `crate::shogi::*` から `shogi_format::*` に書き換え
    (bullet 内部の chess 系 import `bulletformat::*` も削除)
  - module-level および各 method の doc-comment を日本語化・rshogi-nnue
    文脈に合わせて加筆 (英文 upstream → 日本語ローカライズ + 仕様要約追記)
  - `cargo fmt` 適用)

新規追加 (bullet 由来ではない):

- `crates/shogi-features/{Cargo.toml, src/lib.rs}` — workspace member として最小設定、
  shogi-format crate への path dep
- `crates/shogi-features/tests/progress_kpabs_smoke.rs` — shogi-format crate の
  `tests/data/sample.psv` を共有して各 record で `for_each_active_index` /
  `collect_active_indices` / `progress` / `bucket` の挙動を検証 (重み未ロード
  状態で `progress()` が `sigmoid(0)=0.5` / `bucket()` が `4` になることも確認)

## cuda-oxide (Apache-2.0)

- Source: https://github.com/NVlabs/cuda-oxide
- Use: GPU kernel を build-time に PTX 化 (host 側 wrapper も含む)
- License: Apache-2.0
- Dependency style: `Cargo.toml` の git dep + rev pin (vendor せず)
- 採用 rev: **`6de0509`** (NVlabs/cuda-oxide main, 2026-05-08)
  Stage 0-1 で動作確認、Stage 1-3 (#7) で `crates/gpu-runtime` から
  `cuda-core` / `cuda-host` を取り込み

## Pliron (Apache-2.0)

- Source: https://github.com/vaivaswatha/pliron
- Use: cuda-oxide が依存 (transitive)
- License: Apache-2.0

## ライセンス互換性メモ

本リポジトリ自体は MIT。MIT は Apache-2.0 由来コードを含むコンパイル
バイナリ配布と互換。ソース配布時は各依存の `LICENSE` を保持する。
