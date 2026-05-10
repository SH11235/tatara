//! `gpu-kernels` crate — GPU kernel の reference CPU 実装ライブラリ。
//!
//! Stage 1-11 で `experiments/001-cuda-oxide-kpabs/src/kernels/` から昇格。
//! GPU 側の `#[kernel]` 定義は **bin entry (例: `bins/progress_kpabs_train/
//! src/main.rs`) に inline 配置する制約** が cuda-oxide の rustc-codegen-cuda
//! backend にあるため (Stage 1-5 で確立)、本 crate には reference CPU 実装
//! のみを置く。GPU との数値同等性テストは bin 側が本 crate を引き込む形で行う。
//!
//! ## 提供するもの
//!
//! - `progress`: KP-absolute progress trainer 用 4 reference kernel
//!   - `progress::forward::forward_cpu` — sigmoid 線形 forward
//!   - `progress::grad::grad_cpu` — gradient scatter + loss + histogram (atomic 不要 host 単一 thread)
//!   - `progress::adam_step::adam_step_cpu` — Adam optimizer 1 step (1 weight = 1 thread の reference)
//!   - `progress::eval::eval_cpu` — validation/test 時の loss + histogram
//!
//! ## 将来の拡張
//!
//! Stage 2 以降の hand-fused kernel suite では、各 module の reference CPU を
//! 同 crate に追加していく。GPU kernel は呼び出し側 bin / experiment crate
//! ごとに `#[kernel]` を inline 定義する慣行を維持する (cuda-oxide 制約)。

pub mod progress;
