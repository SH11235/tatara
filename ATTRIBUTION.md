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

#### Stage 1-5 (2026-05-10, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_forward`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn forward`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/forward.rs` の
  `forward_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]` (cuda-oxide `cuda_device`)
  - `int` → `u32` (符号要らない)、`float* preds` → `mut DisjointSlice<f32>`
  - `for (int j; j<max_inds; ++j)` → `while j < max_inds` (cuda-oxide gemm 上流に倣う)
  - `expf(-z)` → `(-z).exp()` (cuda-oxide が `__nv_expf` に lowering する)
  - C++ `preds[pos] = ...` (上の `if (pos >= n_pos) return;` で bounds 保証
    された unconditional write) → Rust `if let Some(p) = preds.get_mut(pos)
    { *p = ... }`。cuda-oxide の `DisjointSlice<T>::get_mut(idx) -> Option`
    は GPU soundness のため Option を返す API、`pos >= n_pos` 早期 return
    と組み合わせると `preds.len() == n_pos` で必ず Some が返り挙動同一。
    `preds.len() < n_pos` の異常入力に対しては C++ は OOB write (UB)、
    Rust は silent skip という **defensive な差分** あり
  - 計算ロジックは上記 5 点の表面的差分以外 **同一**。reference CPU
    (`forward_cpu`) も GPU kernel と同じ式を素直に書き写しただけで、
    `preds.len() == n_pos` を満たす入力に対し同出力 (浮動小数誤差範囲内) を返す
  - 注: kernel 関数を main.rs に直接配置しているのは、cuda-oxide の
    rustc-codegen-cuda backend が **bin entry から到達可能な #[kernel]
    関数のみ NVPTX IR 化** する設計のため (本リポ内検証で lib.rs 内
    kernel は `cargo oxide build` で `.ll` 出力されないことを確認)。
    `.ll` 生成の正しい invocation は **`cd experiments/001-cuda-oxide-kpabs
    && cargo-oxide build`** (cwd を crate dir にする)。workspace root から
    `cargo-oxide build exp-001-cuda-oxide-kpabs` を呼ぶと cargo-oxide 上流
    実装 (`crates/cargo-oxide/src/backend.rs`) の workspace-root 探索が
    standalone path に落ちず IR 出力が silently no-op になる

#### Stage 1-6 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_grad_loss_hist`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn grad`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/grad.rs` の
  `grad_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]` (cuda-oxide `cuda_device`)
  - C++ `int` → Rust `u32` (n_pos / max_inds)、`int idx` は `i32` のまま (-1 padding 検出)
  - C++ `float* grad` / `double* loss_acc` / `unsigned long long* hist` →
    Rust `&[f32]` / `&[f64]` / `&[u64]` (atomicAdd 経由でのみ書く前提)
  - C++ `atomicAdd(&grad[idx], gscale)` (f32) → Rust の
    `unsafe { &*(grad.as_ptr().add(idx) as *const DeviceAtomicF32) }
     .fetch_add(gscale, AtomicOrdering::Relaxed)` (cuda-oxide `cuda_device::atomic`)。
    生成 IR は `atomicrmw fadd ... syncscope("device") monotonic` で確認済み
    (sm_60+ の `atom.add.f32` に lowering される、本リポは sm_75 で動作)
  - 同パターンで `loss_acc` (f64) と `hist[bin]` (u64) も `DeviceAtomicF64` /
    `DeviceAtomicU64` に reinterpret cast して `fetch_add(_, Relaxed)`。
    Relaxed 採用は collection 用途で順序保証不要 (bullet 上流 C++ `atomicAdd`
    の暗黙 ordering と同等)
  - C++ `int b = (int)(p * 8.0f); if (b<0) b=0; if (b>7) b=7;` →
    Rust 側 GPU kernel では verbatim な if-else を維持
    (`#[allow(clippy::manual_clamp)]`)。Rust の `i32::clamp` は内部で
    `assert!(min <= max)` の panic 経路 (`Debug::fmt`) を持ち、cuda-oxide の
    rustc-codegen-cuda backend が現状その lowering 未対応 (実機で再現確認)。
    CPU reference (`grad_cpu`) は host 実行のみのため `i32::clamp` を使う
  - 計算ロジックは上記の atomic API / clamp 表現の差異以外 **同一**。reference
    CPU (`grad_cpu`) は同じ式を素直に書き写しただけで、複数 thread の並列
    更新による浮動小数加算順序の差は生じるが (関連: associative でない f32
    の加算)、host 単一 thread 実行では deterministic な値を返す
  - 注: kernel 関数を main.rs に直接配置している理由は Stage 1-5 entry と同じ

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
