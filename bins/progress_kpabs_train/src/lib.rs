//! `progress-kpabs-train` library surface。
//!
//! Stage 1-11 (#15) で `experiments/001-cuda-oxide-kpabs/` から昇格した
//! 本 crate は GPU 駆動 driver と GPU 非依存の host helper を集約する。
//! reference CPU kernel は `gpu-kernels` crate を引き込んで使う (kernel CPU
//! ロジックは Stage 1-11 で `crates/gpu-kernels/src/progress/` に移動済)。
//!
//! - `host`: GPU 非依存の host loop helper。PSV reader / Batch builder /
//!   progress.bin I/O / CLI Args。bin (`src/main.rs`) と integration test
//!   (`tests/*`) の両方から呼べる
//!
//! GPU 側 `#[kernel]` (forward / grad / adam_step / eval) は cuda-oxide の
//! rustc-codegen-cuda backend 制約 (bin entry から到達可能な kernel のみ
//! NVPTX IR 化) のため `src/main.rs` に inline 配置している。reference CPU
//! 実装は `gpu_kernels::progress::*` を参照する。

pub mod host;
