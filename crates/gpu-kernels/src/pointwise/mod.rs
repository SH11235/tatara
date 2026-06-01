//! Pointwise fused kernel suite の reference CPU。
//!
//! 1 kernel = 1 file で配置する。GPU 側 `#[kernel]` の定義は
//! `bins/nnue_train/src/kernels/` 側にある (cuda-oxide rustc-codegen-cuda
//! backend の bin-crate 制約)。
//!
//! ## 提供する module
//!
//! - `screlu_fwd` — SCReLU activation forward
//! - `screlu_grad` — SCReLU activation gradient
//! - `loss_wdl` — sigmoid + WDL blend + scale
//! - `adamw_step` — AdamW with decay + clip
//! - `radam_step` — RAdam (AdamW + bias correction + denom switch)
//! - `ranger_step` — RAdam + lookahead lerp
//! - `norm_loss` — per-weight-group L2-norm regularisation
//! - `loss_wrm` — win-rate-model loss

pub mod adamw_step;
pub mod loss_wdl;
pub mod loss_wrm;
pub mod norm_loss;
pub mod radam_step;
pub mod ranger_step;
pub mod screlu_fwd;
pub mod screlu_grad;
