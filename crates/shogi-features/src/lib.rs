//! 将棋 NNUE / progress 学習で使う特徴量計算。
//!
//! - `progress_kpabs`: KP-absolute 特徴 (`81 * FE_OLD_END` 次元) と
//!   logistic regression による 0..=1 progress / 0..=7 bucket。
//!   bullet-shogi (commit `f275eb9`) の
//!   `crates/bullet_lib/src/game/outputs.rs::ShogiProgressKPAbs` から vendor。
//!
//! 詳細な取り込み元・差分は本リポの `ATTRIBUTION.md` を参照。

pub mod progress_kpabs;

pub use progress_kpabs::{
    SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, SHOGI_PROGRESS8_NUM_BUCKETS, ShogiProgressKPAbs,
};
