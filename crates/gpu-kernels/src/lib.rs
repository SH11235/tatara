//! `gpu-kernels` crate — GPU kernel の reference CPU 実装ライブラリ。
//!
//! GPU 側の `#[kernel]` 定義は **bin entry (例: `bins/progress_kpabs_train/
//! src/main.rs`) に inline 配置する制約** が cuda-oxide の rustc-codegen-cuda
//! backend にあるため、本 crate には reference CPU 実装のみを置く。GPU との数値
//! 同等性テストは bin 側が本 crate を引き込む形で行う。
//!
//! ## 提供するもの
//!
//! - `progress`: KP-absolute progress trainer 用 4 reference kernel
//!   - `progress::forward::forward_cpu` — sigmoid 線形 forward
//!   - `progress::grad::grad_cpu` — gradient scatter + loss + histogram (atomic 不要 host 単一 thread)
//!   - `progress::adam_step::adam_step_cpu` — Adam optimizer 1 step (1 weight = 1 thread の reference)
//!   - `progress::eval::eval_cpu` — validation/test 時の loss + histogram
//! - `pointwise`: Stage 2 (EPIC #16) で整備する pointwise fused kernel suite の
//!   reference CPU 置き場。Stage 2-0 scaffold (#36) では空 module、Stage 2-1〜2-5
//!   で各 kernel ごとに submodule が追加される
//! - `sparse`: Stage 2 (EPIC #16) で整備する sparse FT kernel suite の
//!   reference CPU 置き場。Stage 2-0 scaffold (#36) では空 module、Stage 2-6〜2-7
//!   で sparse_ft_forward / sparse_ft_backward が追加される
//! - `layerstack`: Stage 3-7 (#63) で `bins/nnue_train` に追加した v102 LayerStack
//!   arch 用 ~19 kernel (`ft_post_perspective` / `dense_mm` (+ bucket) / `crelu` /
//!   `abs_pow2_scale` / `concat_l1sqr_main` / `elementwise` / `slice2d`) の
//!   reference CPU。Issue #85 で追加、`bins/nnue_train` の
//!   `gpu_cpu_equivalence_tests` が使う。kernel ↔ CPU ref 対応表は
//!   [`layerstack`] module doc 参照
//!
//! ## 将来の拡張
//!
//! Stage 2 以降の hand-fused kernel suite では、各 module の reference CPU を
//! 同 crate に追加していく。GPU kernel は呼び出し側 bin / experiment crate
//! ごとに `#[kernel]` を inline 定義する慣行を維持する (cuda-oxide 制約)。

pub mod layerstack;
pub mod pointwise;
pub mod progress;
pub mod sparse;
