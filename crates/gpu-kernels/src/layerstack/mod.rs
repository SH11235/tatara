//! LayerStack architecture (FT → L1 16 → L2 32、出力 bucket topology 9-way) で
//! 使う kernel の reference CPU 実装。bucket index は progress8kpabs が局面
//! ごとに 0..=7 を算出する (9 枠中 index 8 は予約で割当対象外、`arch.rs` の
//! `NUM_BUCKETS` 参照)。
//!
//! GPU 側 `#[kernel]` 定義は **`bins/nnue_train/src/kernels/` に配置**
//! されている (cuda-oxide rustc-codegen-cuda backend は `#[kernel]` を bin
//! crate 内に置く必要がある)。本 module 群はその `#[kernel]` のロジックを
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
//! FT 入力次元 (`ft_in`) と 1 perspective あたりの active feature 数
//! (`max_active`) は入力 feature set ごとに異なる runtime 値で、kernel は
//! `cols` / `nnz` / `ft_dim` を launch 引数で受け取る (本 module に固定
//! 定数として持たない)。FT 出力次元 `ft_out` は `--ft-out`、L1 出力次元
//! `l1_out` は `--l1`、L2 出力次元 `l2_out` は `--l2` で選ぶ runtime 値
//! (既定 1536 / 16 / 32) で、kernel はこれらも launch 引数で受ける。以下は
//! LayerStack トポロジで固定の定数 / 派生次元:
//!
//! - `l1_skip = 1`、`l1_effective = l1_out - 1`、`l2_in = l1_effective * 2`
//!   (l1_sqr.concat(l1_main)) — いずれも `l1_out` から導出する派生次元
//! - `NUM_BUCKETS = 9` (progress8kpabs)
//! - `FT_POST_SCALE = L1_SQR_SCALE = 127.0/128.0` (`qa = 127` 由来)

pub mod abs_pow2_scale;
pub mod concat_l1sqr_main;
pub mod crelu;
pub mod dense_mm;
pub mod dense_mm_bucket;
pub mod elementwise;
pub mod ft_post_perspective;
pub mod slice2d;
