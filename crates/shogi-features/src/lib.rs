//! 将棋 NNUE / progress 学習で使う特徴量計算。
//!
//! - `feature_set`: マルチ feature-set 対応の入力特徴量。2 軸 (玉エンコード ×
//!   玉マス処理) でパラメタライズした 1 本の indexer が公開 5 cell を扱う。
//! - `halfka_hm`: HalfKA_hm (Half-Mirror) 特徴 (`45 * 1629 = 73_305` 次元、
//!   合法局面の active 数固定 40)、NNUE 1536-16-32 trainer の入力。
//! - `progress_kpabs`: KP-absolute 特徴 (`81 * FE_OLD_END` 次元) と
//!   logistic regression による `0..=1` progress / N-bucket 割当 (caller が N を指定)。

pub mod effect_bucket;
pub mod feature_set;
pub mod halfka_hm;
pub mod kingrank9;
pub mod progress_kpabs;
pub mod psqt_material;
mod simd;
pub mod threat;
pub mod threat_exclusion;
pub mod threat_symmetric;

pub use effect_bucket::{
    EffectBucketAttackCounts, EffectBucketConfig, collect_effect_bucket_features_board,
    effect_bucket, effect_bucket_attacker_counts, effect_bucket_index,
    map_effect_bucket_features_board, packed_is_bucketed,
};
pub use feature_set::{FeatureSet, FeatureSetSpec, FtFactorizeMode};
pub use halfka_hm::{
    FEATURE_HASH_HM_V2, HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, NUM_KING_BUCKETS, PIECE_INPUTS,
    SHOGI_HALFKA_HM_NUM_ACTIVE_INDICES, SHOGI_HALFKA_HM_NUM_FEATURES, ShogiHalfKA_hm,
};
pub use kingrank9::{KINGRANK9_NUM_BUCKETS, kingrank9_bucket_board};
pub use progress_kpabs::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs};
pub use psqt_material::{material_cp, psqt_material_values};
pub use threat::{THREAT_MAX_ACTIVE, ThreatClass, ThreatIndexer, threat_dimensions_of};
pub use threat_exclusion::ThreatProfile;
pub use threat_symmetric::{
    RawThreatEdge, for_each_active_threat_edge, is_canonical_dead, is_necessarily_mutual,
};
