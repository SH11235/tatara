//! LayerStack architecture (1536-16-32 + progress8kpabs 9 buckets) で使う
//! kernel の reference CPU 実装。
//!
//! GPU 側 `#[kernel]` 定義は **`bins/nnue_train/src/main.rs` に inline 配置**
//! されている (cuda-oxide rustc-codegen-cuda backend は `#[kernel]` を bin
//! crate entry に置く必要がある)。本 module 群はその `#[kernel]` のロジックを
//! host で素直に書き写した `*_cpu` 関数で、`bins/nnue_train` の
//! `#[cfg(test)] mod gpu_cpu_equivalence_tests` が GPU↔CPU 数値同等性
//! テストの reference として使う。
//!
//! bullet 上流の対応箇所 (`examples/shogi_layerstack.rs` /
//! `crates/trainer/src/model/builder.rs`) は各 `*_cpu` の docstring に記載。
//!
//! ## kernel ↔ CPU reference 対応表
//!
//! | `#[kernel]` (bins/nnue_train) | CPU reference (本 crate) |
//! |---|---|
//! | `ft_post_perspective_fwd`     | [`ft_post_perspective::ft_post_perspective_fwd_cpu`] |
//! | `ft_post_perspective_grad`    | [`ft_post_perspective::ft_post_perspective_grad_cpu`] |
//! | `dense_mm_fwd`                | [`dense_mm::dense_mm_fwd_cpu`] |
//! | `dense_mm_bwd_input`          | [`dense_mm::dense_mm_bwd_input_cpu`] |
//! | `dense_mm_bwd_weight`         | [`dense_mm::dense_mm_bwd_weight_cpu`] |
//! | `bias_grad`                   | [`dense_mm::bias_grad_cpu`] |
//! | `dense_mm_fwd_bucket`         | [`dense_mm_bucket::dense_mm_fwd_bucket_cpu`] |
//! | `dense_mm_bwd_input_bucket`   | [`dense_mm_bucket::dense_mm_bwd_input_bucket_cpu`] |
//! | `dense_mm_bwd_weight_bucket`  | [`dense_mm_bucket::dense_mm_bwd_weight_bucket_cpu`] |
//! | `bias_grad_bucket`            | [`dense_mm_bucket::bias_grad_bucket_cpu`] |
//! | `crelu_fwd`                   | [`crelu::crelu_fwd_cpu`] |
//! | `crelu_grad`                  | [`crelu::crelu_grad_cpu`] |
//! | `abs_pow2_scale_fwd`          | [`abs_pow2_scale::abs_pow2_scale_fwd_cpu`] |
//! | `abs_pow2_scale_grad`         | [`abs_pow2_scale::abs_pow2_scale_grad_cpu`] |
//! | `concat_l1sqr_main_fwd`       | [`concat_l1sqr_main::concat_l1sqr_main_fwd_cpu`] |
//! | `concat_l1sqr_main_grad`      | [`concat_l1sqr_main::concat_l1sqr_main_grad_cpu`] |
//! | `elementwise_add`             | [`elementwise::elementwise_add_cpu`] |
//! | `slice_extract_2d`            | [`slice2d::slice_extract_2d_cpu`] |
//! | `slice_scatter_2d`            | [`slice2d::slice_scatter_2d_cpu`] |
//!
//! その他の reference (`loss_wdl` / `loss_wrm` / `sparse_ft_forward` /
//! `sparse_ft_backward` / `screlu_grad` / `radam_step` / `ranger_step` /
//! `adamw_step`) は `pointwise/` / `sparse/` 配下にある。
//!
//! ## アーキテクチャ定数 (bullet 由来)
//!
//! - `FT_IN = 73305` (`HALFKA_HM_DIMENSIONS`)、`FT_OUT = 1536` (per-perspective)
//! - `COMBINED_DIM = FT_OUT = 1536` (pairwise 1536→768 を 2 perspective concat)
//! - `L1_OUT = 16`、`L1_EFFECTIVE = L1_OUT - 1 = 15`、`L1_SKIP = 1`
//! - `L2_IN = L1_EFFECTIVE * 2 = 30` (l1_sqr.concat(l1_main))、`L2_OUT = 32`
//! - `NUM_BUCKETS = 9` (progress8kpabs)、`MAX_ACTIVE = 40` (nnz)
//! - `FT_POST_SCALE = L1_SQR_SCALE = 127.0/128.0` (`qa = 127` 由来)

pub mod abs_pow2_scale;
pub mod concat_l1sqr_main;
pub mod crelu;
pub mod dense_mm;
pub mod dense_mm_bucket;
pub mod elementwise;
pub mod ft_post_perspective;
pub mod slice2d;
