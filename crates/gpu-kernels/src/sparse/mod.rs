//! Sparse FT kernel suite (HalfKA_hm 入力層用) の reference CPU 実装。
//!
//! 1 kernel = 1 file で配置する。GPU 側 `#[kernel]` の inline 定義は
//! `bins/nnue_train/src/main.rs` 側に置く (cuda-oxide rustc-codegen-cuda
//! backend は bin entry 経由で到達可能な kernel しか PTX 化しないため)。
//!
//! ## 提供する module
//!
//! - `sparse_ft_forward` — HalfKA_hm sparse feature transform forward
//! - `sparse_ft_backward` — 同 backward、atomics scatter

pub mod sparse_ft_backward;
pub mod sparse_ft_forward;
