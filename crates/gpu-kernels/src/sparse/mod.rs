//! Sparse feature transform (HalfKA_hm) の reference CPU。
//!
//! 1 kernel = 1 file。GPU 側 `#[kernel]` の inline 定義は bin 側
//! (`bins/nnue_train/src/main.rs`) に置く (cuda-oxide rustc-codegen-cuda
//! backend の bin-entry 制約)。
//!
//! - `sparse_ft_forward` — HalfKA_hm sparse feature transform forward
//! - `sparse_ft_backward` — 同 backward (atomic scatter)

pub mod sparse_ft_backward;
pub mod sparse_ft_forward;
