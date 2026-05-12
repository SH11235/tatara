//! Stage 2 (EPIC #16) で整備する pointwise fused kernel suite の reference CPU。
//!
//! ADR-0004 の Pattern table に対応する 5 fused kernel を、Stage 1 の `progress/`
//! と同じく 1 kernel = 1 file で配置する。GPU 側 `#[kernel]` の inline 定義は
//! `experiments/002-fused-kernels/src/main.rs` (cuda-oxide rustc-codegen-cuda
//! backend の bin-entry 制約、Stage 1-5 で確立)。
//!
//! ## 提供する module (Stage 2-1〜2-5 + Stage 3 #84 で順次追加)
//!
//! - `screlu_grad` — SCReLU activation gradient (Stage 2-1 / #37) — **landed**
//! - `loss_wdl` — sigmoid + WDL blend + scale (Stage 2-2 / #38) — **landed**
//! - `adamw_step` — AdamW with decay + clip (Stage 2-3 / #39) — **landed**
//! - `radam_step` — RAdam (AdamW + bias correction + denom switch) (Stage 2-4 / #40) — **landed**
//! - `ranger_step` — RAdam + lookahead lerp (Stage 2-5 / #41) — **landed**
//! - `loss_wrm` — bullet win-rate-model loss (Stage 3 / #84、v102 厳密再現用) — **landed**
//!
//! Stage 2-0 scaffold (#36) では module 自体は空。各 kernel 実装 PR で
//! `pub mod <kernel_name>;` を 1 行ずつ追加していく運用。

pub mod adamw_step;
pub mod loss_wdl;
pub mod loss_wrm;
pub mod radam_step;
pub mod ranger_step;
pub mod screlu_grad;
