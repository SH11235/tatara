//! 将棋 NNUE / progress 学習で使う特徴量計算。
//!
//! - `progress_kpabs`: KP-absolute 特徴 (`81 * FE_OLD_END` 次元) と
//!   logistic regression による 0..=1 progress / 0..=7 bucket。
//!   bullet-shogi (commit `f275eb9`) の
//!   `crates/bullet_lib/src/game/outputs.rs::ShogiProgressKPAbs` から vendor。
//! - `halfka_hm`: HalfKA_hm (Half-Mirror) 特徴 (`45 * 1629 = 73_305` 次元、
//!   合法局面の active 数固定 40)、Stage 3 NNUE 1536-16-32 trainer の入力。
//!   bullet-shogi (commit `f275eb9`) の
//!   `crates/bullet_lib/src/game/inputs/shogi_halfka.rs::ShogiHalfKA_hm` から vendor。
//!
//! 詳細な取り込み元・差分は本リポの `ATTRIBUTION.md` を参照。

pub mod halfka_hm;
pub mod progress_kpabs;

pub use halfka_hm::{
    FEATURE_HASH_HM_V2, HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, NUM_KING_BUCKETS, PIECE_INPUTS,
    SHOGI_HALFKA_HM_NUM_ACTIVE_INDICES, SHOGI_HALFKA_HM_NUM_FEATURES, ShogiHalfKA_hm,
};
pub use progress_kpabs::{
    SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, SHOGI_PROGRESS8_NUM_BUCKETS, ShogiProgressKPAbs,
};
