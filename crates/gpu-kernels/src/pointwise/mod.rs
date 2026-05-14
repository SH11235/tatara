//! Pointwise fused kernel suite の reference CPU。1 kernel = 1 file。GPU 側
//! `#[kernel]` の inline 定義は bin 側 (`bins/nnue_train/src/main.rs`) に置く
//! (cuda-oxide rustc-codegen-cuda backend の bin-entry 制約)。
//!
//! - `screlu_grad` — SCReLU activation gradient
//! - `loss_wdl` — sigmoid + WDL blend + scale
//! - `adamw_step` — AdamW with decay + clip
//! - `radam_step` — RAdam (AdamW + bias correction + denom switch)
//! - `ranger_step` — RAdam + lookahead lerp
//! - `loss_wrm` — bullet win-rate-model loss (NNUE2score / in_scaling)

pub mod adamw_step;
pub mod loss_wdl;
pub mod loss_wrm;
pub mod radam_step;
pub mod ranger_step;
pub mod screlu_grad;
