//! `progress-kpabs-train` library surface。
//!
//! KP-abs progress trainer の GPU 駆動 driver と GPU 非依存 host helper を
//! 集約する crate。reference CPU kernel は `gpu_kernels::progress::*` を参照
//! する。
//!
//! - `host`: GPU 非依存の host loop helper。PSV reader / Batch builder /
//!   progress.bin I/O / CLI Args を提供し、bin (`src/main.rs`) と integration
//!   test (`tests/*`) の両方から呼べる。
//!
//! GPU 側 `#[kernel]` (forward / grad / adam_step / eval) は cuda-oxide の
//! rustc-codegen-cuda backend 制約 (bin entry から到達可能な kernel のみが
//! NVPTX IR 化される) のため `src/main.rs` に inline 配置している。

pub mod host;
