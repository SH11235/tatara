//! experiments/001-cuda-oxide-kpabs library surface。
//!
//! cuda-oxide `#[kernel]` で書く GPU カーネルと、その reference CPU 実装
//! を集約する。bin (`src/main.rs`) と integration test (`tests/*`) の
//! 両方から呼べるようにするため lib として公開している。
//!
//! Stage 1 範囲: forward / grad / adam_step / eval。順次 Issue #9〜#12 で増築。

pub mod kernels;
