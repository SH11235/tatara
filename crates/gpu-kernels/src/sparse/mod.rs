//! Stage 2 (EPIC #16) で整備する sparse FT kernel suite の reference CPU。
//!
//! ADR-0004 の Pattern table に対応する 2 sparse kernel を、Stage 1 の `progress/`
//! と同じく 1 kernel = 1 file で配置する。GPU 側 `#[kernel]` の inline 定義は
//! `experiments/002-fused-kernels/src/main.rs` (cuda-oxide rustc-codegen-cuda
//! backend の bin-entry 制約、Stage 1-5 で確立)。
//!
//! ## 提供する module (Stage 2-6〜2-7 で順次追加)
//!
//! - `sparse_ft_forward` — HalfKA_hm sparse feature transform forward (Stage 2-6 / #42) — **landed**
//! - `sparse_ft_backward` — 同 backward、atomics scatter (Stage 2-7 / #43)
//!
//! Stage 2-0 scaffold (#36) では module 自体は空。各 kernel 実装 PR で
//! `pub mod <kernel_name>;` を 1 行ずつ追加していく運用。

pub mod sparse_ft_forward;
