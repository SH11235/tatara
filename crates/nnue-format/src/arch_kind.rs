//! `arch_kind` module — NNUE network アーキテクチャの種別。
//!
//! 入力 feature set (`shogi-features` の `FeatureSet`) とは独立した軸で、層構成
//! (bucket / PSQT / skip 接続の有無、weight group 数) と host training pipeline
//! の分岐を決める。学習 artifact / checkpoint が「どのアーキで学習されたか」を
//! 記録し、別アーキの weight を取り違えて読み込まないために、シリアライズ層で
//! ある本 crate に置く。

/// NNUE network のアーキテクチャ種別。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArchKind {
    /// progress8kpabs 9-bucket LayerStack (FT → bucket 化 L1/L2/L3 + L1f skip)。
    LayerStack,
    /// bucket 無しの 4 層 dense アーキ (FT → L1 → L2 → L3)。
    Simple,
}

impl ArchKind {
    /// CLI サブコマンド名 / artifact identity が扱う flat な canonical 名。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            ArchKind::LayerStack => "layerstack",
            ArchKind::Simple => "simple",
        }
    }
}
