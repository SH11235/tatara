//! Pointwise fused kernel suite の reference CPU。
//!
//! 1 kernel = 1 file で配置する。GPU 側 `#[kernel]` の inline 定義は
//! `bins/nnue_train/src/main.rs` 側にある (cuda-oxide rustc-codegen-cuda
//! backend の bin-entry 制約)。
//!
//! ## 提供する module
//!
//! - `screlu_fwd` — SCReLU activation forward
//! - `screlu_grad` — SCReLU activation gradient
//! - `loss_wdl` — sigmoid + WDL blend + scale
//! - `adamw_step` — AdamW with decay + clip
//! - `radam_step` — RAdam (AdamW + bias correction + denom switch)
//! - `ranger_step` — RAdam + lookahead lerp
//! - `loss_wrm` — win-rate-model loss (bullet `loss_fn_wrm` を移植)

pub mod adamw_step;
pub mod loss_wdl;
pub mod loss_wrm;
pub mod radam_step;
pub mod ranger_step;
pub mod screlu_fwd;
pub mod screlu_grad;
