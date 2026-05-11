//! `nnue-train` crate — HalfKA_hm 1536-16-32 NNUE training pipeline (host 側)。
//!
//! Stage 3 (EPIC #17) の **CPU-only training pipeline library**。
//! GPU `#[kernel]` は持たず、host 側の schedule / dataloader / optimizer state
//! のみを提供する。GPU kernel 定義は `bins/nnue_train/src/main.rs` 側 inline
//! (cuda-oxide rustc-codegen-cuda backend が bin entry 経由でしか `#[kernel]`
//! を NVPTX IR 化しない制約のため、Stage 1-5 で確立)。
//!
//! 本 crate は CI workflow で `--exclude` せず test を通す方針 (`gpu-kernels`
//! と同方針)。host 側ロジックのみで構成し、`gpu-runtime` / `cuda-host` 等
//! GPU build chain には依存しない。
//!
//! ## 提供 module
//!
//! - `schedule` (Stage 3-4, #60): learning-rate / wdl scheduler。bullet-shogi
//!   `crates/bullet_lib/src/trainer/schedule/{lr,wdl}.rs` (commit `f275eb9`)
//!   から vendor、`ansi` color formatter 依存は `Display` impl に置換
//!
//! ## 提供予定 module
//!
//! - `dataloader` (Stage 3-5, #61): PSV file → HalfKA_hm sparse batch + prefetch
//!   ring。bullet-shogi `crates/bullet_lib/src/value/{dataloader,loader}.rs`
//!   を vendor
//! - `optimizer` (Stage 3-6, #62): Ranger / RAdam host state + Stage 2
//!   pointwise kernel の launch helper (host orchestration)。kernel 本体は
//!   Stage 2-3〜2-5 で landed 済 (`crates/gpu-kernels::pointwise`)
//! - `trainer` (Stage 3-8, #65): main training loop (forward → loss_wdl →
//!   backward → optimizer step)。`bins/nnue_train::main` から呼ばれる

pub mod schedule;
