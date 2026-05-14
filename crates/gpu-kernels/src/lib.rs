//! `gpu-kernels` crate — GPU kernel の reference CPU 実装ライブラリ。
//!
//! GPU 側の `#[kernel]` 定義は **bin entry (例: `bins/progress_kpabs_train/
//! src/main.rs`) に inline 配置する制約** が cuda-oxide の rustc-codegen-cuda
//! backend にあるため、本 crate には reference CPU 実装のみを置く。GPU との
//! 数値同等性テストは bin 側が本 crate を引き込む形で行う。
//!
//! ## 提供するもの
//!
//! - `progress`: KP-absolute progress trainer 用 4 reference kernel
//!   - `progress::forward::forward_cpu` — sigmoid 線形 forward
//!   - `progress::grad::grad_cpu` — gradient scatter + loss + histogram (atomic 不要 host 単一 thread)
//!   - `progress::adam_step::adam_step_cpu` — Adam optimizer 1 step (1 weight = 1 thread の reference)
//!   - `progress::eval::eval_cpu` — validation/test 時の loss + histogram
//! - `pointwise`: pointwise fused kernel suite の reference CPU 置き場
//!   (SCReLU grad / WDL loss / WRM loss / AdamW / RAdam / Ranger)
//! - `sparse`: sparse FT kernel suite (forward / backward) の reference CPU
//! - `layerstack`: v102 LayerStack arch 用 ~19 kernel
//!   (`ft_post_perspective` / `dense_mm` (+ bucket) / `crelu` /
//!   `abs_pow2_scale` / `concat_l1sqr_main` / `elementwise` / `slice2d`) の
//!   reference CPU。`bins/nnue_train` の `gpu_cpu_equivalence_tests` が使う。
//!   kernel ↔ CPU ref 対応表は [`layerstack`] module doc 参照
//!
//! GPU kernel は呼び出し側 bin / experiment crate ごとに `#[kernel]` を inline
//! 定義する (cuda-oxide 制約)。

pub mod layerstack;
pub mod pointwise;
pub mod progress;
pub mod sparse;
