//! `progress-kpabs-train` library surface — KP-absolute progress trainer 用の
//! GPU 非依存 host helper を集約する。
//!
//! - `host`: PSV reader / Batch builder / progress.bin I/O / CLI Args。bin
//!   (`src/main.rs`) と integration test (`tests/*`) の両方から呼べる。
//!
//! GPU 側 `#[kernel]` (forward / grad / adam_step / eval) は cuda-oxide の
//! rustc-codegen-cuda backend 制約 (bin entry から到達可能な kernel のみ
//! NVPTX IR 化) のため `src/main.rs` に inline 配置している。reference CPU
//! 実装は `gpu_kernels::progress::*` を参照する。

pub mod host;
