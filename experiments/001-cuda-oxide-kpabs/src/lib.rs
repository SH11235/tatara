//! experiments/001-cuda-oxide-kpabs library surface。
//!
//! cuda-oxide `#[kernel]` で書く GPU カーネルと、その reference CPU 実装、
//! および host loop で使う GPU 非依存のロジック (PSV reader, batch builder,
//! progress.bin I/O, CLI) を集約する。bin (`src/main.rs`) と integration test
//! (`tests/*`) の両方から呼べるようにするため lib として公開している。
//!
//! - `kernels`: forward / grad / adam_step / eval の reference CPU 実装。
//!   GPU 側 `#[kernel]` は `src/main.rs` に inline (cuda-oxide backend が
//!   bin entry 経由で到達可能な kernel しか PTX 化しないため)
//! - `host`: GPU 非依存の host loop helper。PSV reader / Batch builder /
//!   progress.bin I/O / CLI Args。GPU 操作 (`GpuTrainer`) は `main.rs` に置く

pub mod host;
pub mod kernels;
