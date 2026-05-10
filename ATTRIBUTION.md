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

#### Stage 1-9 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs` の **host 側ロジック** (kernel
  以外、約 900 行) を以下の単位で `experiments/001-cuda-oxide-kpabs/` に移植。
  bullet-shogi の multi-thread prefetch / pack interleaving / 学習 epoch
  per-checkpoint / val split は Stage 1-9 の受け入れ条件 (1 epoch 完走 +
  progress.bin 出力) に対し過剰なため意図的に削除し、最小実装に絞った。

  - `src/host/games.rs::PackCursor` / `GameIterator` ←→ 上流 `PackCursor` /
    `GameIterator`。PSV ファイルを 1 record ずつ読み、`game_ply` の減少を
    境界として 1 ゲーム単位に切り出す。bullet 上流の `Vec<u8>` バッファ + size
    検証 path も同等
  - `src/host/batch.rs::Batch` ←→ 上流 `Batch`。`push_game` で 1 ゲーム分の
    flat indices / targets / per_pos_norm を埋め、`finalize` で per_pos_norm
    に `1/n_games` を乗じて batch averaging を完成。target は `i / (game_len - 1)`
    の game-relative ラベル (上流と同式)。`MAX_INDS_PER_POS = 80` も同値
  - `src/host/progress_bin.rs::write_progress_bin` / `read_progress_bin` ←→
    上流の同名関数。YaneuraOu 互換の f64 LE × `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`
    形式 (= 1,003,104 bytes)。f32 ↔ f64 cast は wire format 通り
  - `src/host/cli.rs::Args` ←→ 上流 `Args` の核サブセット (`--data` `--output`
    `--init-from` `--games-per-step` `--max-games` `--epochs` `--lr` `--lr-scale`
    `--log-interval-steps` `--device`)。prefetch / val split 関連 flag は削除
  - `src/main.rs::GpuTrainer` ←→ 上流 `GpuTrainer`。`step` で
    forward → grad/loss/hist → adam_step を順次 `cuda_launch!` で起動、
    `eval_forward` は forward → eval kernel。device buffer 確保は cuda-oxide
    `DeviceBuffer<T>` ベースで bullet の `RawBuf` (raw `malloc/memset`) は不要
  - `src/main.rs::train_one_epoch` ←→ 上流 `train_one_epoch`。multi-thread
    prefetch を **single-threaded** に簡素化 (mpsc / JoinHandle なし)、log /
    epoch 集計はそのまま

- **kernel artifact loader** (`load_kernel_module_with_fallback` /
  `compile_ll_to_ptx_via_llc`) は新規 (上流の NVRTC は Rust kernel に使えない)。
  cuda-oxide が出力する opaque pointer NVVM IR (`define void @grad(ptr ...)`)
  は libNVVM が parse できない (実機エラー: `nvvmCompileProgram error 9:
  parse expected type`、`exp_001_cuda_oxide_kpabs.ll:11` 由来) ため、本 PR は
  **`llvm-link-21 + opt-21 (passes='internalize,globaldce,nvvm-reflect') +
  llc-21`** の 3 段 pipeline で `.ll → .ptx` を生成する。kernel symbol を
  `--internalize-public-api-list=grad,forward,adam_step,eval` で保存し、
  libdevice の未使用関数を `globaldce` で除去、`__nvvm_reflect()` を `nvvm-reflect`
  pass で 0/1 に畳み込む。NVCC の `compileToCubin` 相当だが driver 側の JIT
  にも対応した形で生成。`.ptx` には `.extern .func` が残らず ptxas 単体で完結

- 環境前提: WSL2 sm_75 box (RTX 2070 SUPER)、CUDA 12.9、LLVM 21.1.8 (clang-21
  / llvm-link-21 / opt-21 / llc-21)、`/usr/local/cuda-12.9/nvvm/libdevice/
  libdevice.10.bc`。Stage 1-1〜1-8 と同じ。実行確認: `cargo run -p
  exp-001-cuda-oxide-kpabs -- --data <sample.psv> --output <progress.bin>
  --games-per-step 4 --max-games 8` で 1 epoch 完走 + 1003104 bytes
  progress.bin 出力済 (受け入れ条件達成)

#### Stage 1-8 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_eval_loss_hist`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn eval`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/eval.rs` の
  `eval_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]`
  - C++ `const float* preds / targets` / `double* loss_acc` / `unsigned long long* hist` →
    Rust `&[f32]` / `&[f64]` / `&[u64]` (atomicAdd 経由で書く前提)
  - C++ `atomicAdd(loss_acc, (double)err*(double)err)` → Rust
    `DeviceAtomicF64::fetch_add(_, AtomicOrdering::Relaxed)`、IR で
    `atomicrmw fadd ptr ..., double ... syncscope("device") monotonic` 確認
  - C++ `atomicAdd(&hist[b], 1ULL)` → Rust `DeviceAtomicU64::fetch_add(1, Relaxed)`、
    IR で `atomicrmw add ptr ..., i64 1 syncscope("device") monotonic` 確認
  - C++ `(int)(p * 8.0f); if (b<0) b=0; if (b>7) b=7;` → kernel 側は Stage 1-6
    と同じく verbatim if-else (`#[allow(clippy::manual_clamp)]`)、CPU reference は
    `i32::clamp(0, 7)`
  - 計算ロジックは `grad` の **gradient scatter / per_pos_norm を除いたサブセット** で、
    eval 側 `eval_cpu` と grad 側 `grad_cpu` に同じ `(preds, targets, n_pos)` を渡せば
    `loss_acc` / `hist` が完全一致する不変条件をテスト (`tests/eval_smoke.rs::
    eval_output_matches_grad_loss_hist_subset`)

#### Stage 1-7 (2026-05-11, bullet-shogi commit `f275eb9`)

- `examples/shogi_progress_kpabs_train_cuda.rs::KERNELS_SRC::k_adam_step`
  → `experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[kernel] fn adam_step`
  + `experiments/001-cuda-oxide-kpabs/src/kernels/adam_step.rs` の
  `adam_step_cpu` (numerical equivalence test 用 reference)。
  - 言語移植: C++ `__global__` → Rust `#[kernel]` (cuda-oxide `cuda_device`)
  - C++ `float* weights / m / v / grad` (in-place 4 buffer) → Rust
    `mut DisjointSlice<f32>` × 4。1 thread = 1 weight で aliasing なし、
    grad のような scatter は発生しないので **atomics 不要** (Stage 1-6 と
    異なる)。host 側で `len() == n` を保証し、`get_mut(i)` の Option を
    全 4 で揃える `if let (Some, Some, Some, Some) = (...)` パターンで
    silent skip 防御
  - C++ `fmaxf(bc, 1e-30f)` → Rust 側 GPU kernel では verbatim な if-else
    `if bc > 1e-30 { bc } else { 1e-30 }` を維持。Rust の `f32::max` は
    内部で `std::intrinsics::maximum_number_nsz_f32` を呼び、cuda-oxide が
    現状その intrinsic を未解決 (実機エラー: `Symbol
    std__intrinsics__maximum_number_nsz_f32 not found`、`f32.rs:993` 由来)。
    CPU reference (`adam_step_cpu`) は host 実行のみのため `f32::max` を使う
  - C++ `sqrtf(v_hat)` → Rust `v_hat.sqrt()`。cuda-oxide は IR で
    `call float @__nv_sqrtf(...)` に lowering する (libdevice 経由、
    `.ll` 出力で確認済)
  - C++ `int n` → Rust `u32`
  - 計算式は表面的差異 (Option-returning DisjointSlice / max の if-else 表現)
    を除き同一

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
