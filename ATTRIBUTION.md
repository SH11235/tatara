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

## cuda-oxide (Apache-2.0)

- Source: https://github.com/NVlabs/cuda-oxide
- Use: GPU kernel を build-time に PTX 化 (host 側 wrapper も含む)
- License: Apache-2.0
- Dependency style: `Cargo.toml` の git dep + rev pin (vendor せず)

## Pliron (Apache-2.0)

- Source: https://github.com/vaivaswatha/pliron
- Use: cuda-oxide が依存 (transitive)
- License: Apache-2.0

## ライセンス互換性メモ

本リポジトリ自体は MIT。MIT は Apache-2.0 由来コードを含むコンパイル
バイナリ配布と互換。ソース配布時は各依存の `LICENSE` を保持する。
