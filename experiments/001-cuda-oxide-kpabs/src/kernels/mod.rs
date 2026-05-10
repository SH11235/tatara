//! cuda-oxide `#[kernel]` で書く GPU カーネル群。
//!
//! Stage 1 範囲: forward / grad / adam_step / eval の 4 つ。順次 Issue #9〜#12
//! で実装する。本 module は kernel 定義 (device side) と reference CPU 実装
//! (numerical equivalence 検証用) を持つ。

pub mod forward;
pub mod grad;
