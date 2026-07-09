//! マルチ feature-set 対応の NNUE 入力特徴量。
//!
//! 入力 feature set を **独立 2 軸**でパラメタライズした 1 本の indexer で
//! 全 feature set を扱う:
//!
//! - 軸 1 — 玉特徴エンコード (`KingEncoding`): 玉を特徴に含めない /
//!   自玉・敵玉を別 plane に置く / 両玉を 1 plane へ畳む。
//! - 軸 2 — 玉マス処理 (`KingSquareMode`): 玉マスを 81 bucket でそのまま使うか、
//!   6-9 筋を 1-4 筋へ反転して 45 bucket にする (盤駒も筋ミラーする)。
//!
//! 2 軸は作用対象が異なる: 軸 2 はマス座標 (玉マス→bucket、盤駒マス→筋ミラー)、
//! 軸 1 は玉 BonaPiece の plane base に作用する。両者は干渉しない。
//!
//! 2 軸は crate 内部の表現で、公開するのは [`FeatureSet`] の 5 variant のみ。
//! 無効な軸の組み合わせは公開 enum として表現できない。[`FeatureSetSpec`] が
//! 公開 enum と内部 2 軸・次元・hash を結ぶ単一の真実源で、生成は
//! [`FeatureSet::spec`] 経由のみ。

use shogi_format::bona_piece::{E_KING, F_KING, FE_HAND_END, FE_OLD_END};
use shogi_format::types::{Color, HAND_PIECE_TYPES, Square};
use shogi_format::{BonaPiece, PackedSfenValue, ShogiBoard};

use crate::effect_bucket::{EffectBucketAttackCounts, EffectBucketConfig};
use crate::threat::{THREAT_MAX_ACTIVE, ThreatIndexer};
use crate::threat_exclusion::ThreatProfile;

// =============================================================================
// 軸 1 / 軸 2
// =============================================================================

/// 軸 1 — 玉特徴のエンコード方式 (crate 内部表現)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KingEncoding {
    /// 玉を特徴に含めない (HalfKP)。玉位置は bucket index にのみ効く。
    NoKing,
    /// 自玉・敵玉が別々の piece-input ordinal を占有する (HalfKA)。
    SplitPlane,
    /// 両玉を 1 plane へ畳む。敵玉の BonaPiece を 81 引いて自玉 plane に重ねる。
    MergedPlane,
}

/// 軸 2 — 玉マスから king bucket への写像方式 (crate 内部表現)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KingSquareMode {
    /// 玉マス (視点変換後) をそのまま 81 bucket に使う。
    Direct,
    /// 6-9 筋を 1-4 筋へ筋反転して 45 bucket にする。玉が 6-9 筋のときは
    /// 盤駒の筋も反転して左右対称な局面を同一表現に正規化する。
    HorizontalMirror,
}

// =============================================================================
// 公開 feature set
// =============================================================================

/// 公開 feature set。内部の 2 軸組み合わせのうち正式サポートする 5 cell。
#[allow(clippy::enum_variant_names)]
// 全 variant が HalfKP/HalfKA 系列。`Half` は
// Stockfish 由来の確立した語で、prefix を削ると feature set 名として不明瞭になる。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeatureSet {
    /// 玉を特徴化しない・ミラー無し。
    HalfKp,
    /// 両玉を別 plane・ミラー無し。
    HalfKaSplit,
    /// 両玉を 1 plane へ畳む・ミラー無し。
    HalfKaMerged,
    /// 両玉を別 plane・半面ミラー。
    HalfKaHmSplit,
    /// 両玉を 1 plane へ畳む・半面ミラー。
    HalfKaHmMerged,
}

impl FeatureSet {
    /// 全公開 feature set。
    pub const ALL: [FeatureSet; 5] = [
        FeatureSet::HalfKp,
        FeatureSet::HalfKaSplit,
        FeatureSet::HalfKaMerged,
        FeatureSet::HalfKaHmSplit,
        FeatureSet::HalfKaHmMerged,
    ];

    /// CLI / artifact header が扱う flat な canonical 名。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            FeatureSet::HalfKp => "halfkp",
            FeatureSet::HalfKaSplit => "halfka-split",
            FeatureSet::HalfKaMerged => "halfka-merged",
            FeatureSet::HalfKaHmSplit => "halfka-hm-split",
            FeatureSet::HalfKaHmMerged => "halfka-hm-merged",
        }
    }

    /// canonical 名から逆引きする。未知の名前は `None`。
    pub fn from_canonical_name(name: &str) -> Option<FeatureSet> {
        FeatureSet::ALL
            .into_iter()
            .find(|fs| fs.canonical_name() == name)
    }

    /// この feature set の [`FeatureSetSpec`] を返す。
    pub const fn spec(self) -> FeatureSetSpec {
        // 軸 2 が `Direct` のとき 81 bucket、`HorizontalMirror` のとき 45 bucket。
        // piece_inputs は軸 1 で決まる: NoKing = 玉 plane 無し、SplitPlane = 玉
        // plane 2 枚、MergedPlane = 玉 plane 1 枚。
        match self {
            FeatureSet::HalfKp => FeatureSetSpec {
                feature_set: self,
                king_encoding: KingEncoding::NoKing,
                king_square_mode: KingSquareMode::Direct,
                king_buckets: 81,
                piece_inputs: PIECE_INPUTS_NO_KING,
                max_active: MAX_ACTIVE_NO_KING,
                feature_hash: FEATURE_HASH_HALFKP,
                arch_feature_name: "HalfKP",
                ft_factorize: false,
                ft_factorize_mode: FtFactorizeMode::Base,
                threat_profile: None,
                effect_bucket_config: None,
            },
            FeatureSet::HalfKaSplit => FeatureSetSpec {
                feature_set: self,
                king_encoding: KingEncoding::SplitPlane,
                king_square_mode: KingSquareMode::Direct,
                king_buckets: 81,
                piece_inputs: PIECE_INPUTS_SPLIT,
                max_active: MAX_ACTIVE_WITH_KING,
                feature_hash: FEATURE_HASH_HALFKA_SPLIT,
                arch_feature_name: "HalfKaSplit",
                ft_factorize: false,
                ft_factorize_mode: FtFactorizeMode::Base,
                threat_profile: None,
                effect_bucket_config: None,
            },
            FeatureSet::HalfKaMerged => FeatureSetSpec {
                feature_set: self,
                king_encoding: KingEncoding::MergedPlane,
                king_square_mode: KingSquareMode::Direct,
                king_buckets: 81,
                piece_inputs: PIECE_INPUTS_MERGED,
                max_active: MAX_ACTIVE_WITH_KING,
                feature_hash: FEATURE_HASH_HALFKA_MERGED,
                arch_feature_name: "HalfKaMerged",
                ft_factorize: false,
                ft_factorize_mode: FtFactorizeMode::Base,
                threat_profile: None,
                effect_bucket_config: None,
            },
            FeatureSet::HalfKaHmSplit => FeatureSetSpec {
                feature_set: self,
                king_encoding: KingEncoding::SplitPlane,
                king_square_mode: KingSquareMode::HorizontalMirror,
                king_buckets: 45,
                piece_inputs: PIECE_INPUTS_SPLIT,
                max_active: MAX_ACTIVE_WITH_KING,
                feature_hash: FEATURE_HASH_HALFKA_HM_SPLIT,
                arch_feature_name: "HalfKaHmSplit",
                ft_factorize: false,
                ft_factorize_mode: FtFactorizeMode::Base,
                threat_profile: None,
                effect_bucket_config: None,
            },
            FeatureSet::HalfKaHmMerged => FeatureSetSpec {
                feature_set: self,
                king_encoding: KingEncoding::MergedPlane,
                king_square_mode: KingSquareMode::HorizontalMirror,
                king_buckets: 45,
                piece_inputs: PIECE_INPUTS_MERGED,
                max_active: MAX_ACTIVE_WITH_KING,
                feature_hash: FEATURE_HASH_HALFKA_HM_MERGED,
                arch_feature_name: "HalfKaHmMerged",
                ft_factorize: false,
                ft_factorize_mode: FtFactorizeMode::Base,
                threat_profile: None,
                effect_bucket_config: None,
            },
        }
    }
}

// =============================================================================
// 次元・hash 定数
// =============================================================================

/// 玉 plane を持たない piece 入力数 (盤上駒 + 手駒、`FE_OLD_END`)。
const PIECE_INPUTS_NO_KING: usize = FE_OLD_END; // 1548
/// 玉 plane 2 枚を持つ piece 入力数 (`E_KING` + 81)。
const PIECE_INPUTS_SPLIT: usize = E_KING as usize + 81; // 1710
/// 玉 plane 1 枚を持つ piece 入力数 (敵玉を畳んだ後の上限、`E_KING`)。
const PIECE_INPUTS_MERGED: usize = E_KING as usize; // 1629

/// 玉を含めないときの最大 active 特徴数 (合法局面の駒 40 から玉 2 を除く)。
const MAX_ACTIVE_NO_KING: usize = 38;
/// 玉を含めるときの最大 active 特徴数 (合法局面の駒総数)。
const MAX_ACTIVE_WITH_KING: usize = 40;

// 参照実装あり 3 cell の feature hash は nnue-pytorch 系の固定値。
/// halfkp の feature hash。
const FEATURE_HASH_HALFKP: u32 = 0x5D69_D5B8;
/// halfka-split の feature hash。
const FEATURE_HASH_HALFKA_SPLIT: u32 = 0x5F13_4CB8;
/// halfka-hm-merged の feature hash。
const FEATURE_HASH_HALFKA_HM_MERGED: u32 = 0x7F13_4CB8;
// 参照実装なし 2 cell は canonical 名の FNV-1a 32bit hash を feature 定数とする。
// 外部エンジン互換は元々非対象なので reproducible で衝突しない値であれば良い。
/// `halfka-merged` の feature 定数。
const FEATURE_HASH_HALFKA_MERGED: u32 = fnv1a32("halfka-merged");
/// `halfka-hm-split` の feature 定数。
const FEATURE_HASH_HALFKA_HM_SPLIT: u32 = fnv1a32("halfka-hm-split");

/// FNV-1a 32bit hash。
const fn fnv1a32(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash
}

/// threat profile ごとの feature hash 定数。`feature_hash() = base ^ この値` で
/// 合成する。`threat-{profile}` 文字列の FNV-1a 32bit hash を定数化したもので、
/// 全 base × 全 profile の合成が衝突しないことを test で固定する。各 profile は
/// row の意味を変える (compaction) ため、load 時に hash 不一致を弾いて silent な
/// row ずれを防ぐ。
const fn threat_profile_hash(profile: ThreatProfile) -> u32 {
    match profile {
        ThreatProfile::Full => fnv1a32("threat-full"),
        ThreatProfile::SameClass => fnv1a32("threat-same-class"),
        ThreatProfile::SameClassMajorPawn => fnv1a32("threat-same-class-major-pawn"),
        ThreatProfile::StepAttacker => fnv1a32("threat-step-attacker"),
        ThreatProfile::FullSymDedup => fnv1a32("threat-full-symdedup"),
        ThreatProfile::CrossSide => fnv1a32("threat-cross-side"),
    }
}

/// effect bucket config ごとの feature hash 定数。`feature_hash() = base ^ この値` で
/// 合成する。config は row の意味を変えるため load 時に hash / arch token の
/// 両方で取り違えを弾く。
const fn effect_bucket_config_hash(config: EffectBucketConfig) -> u32 {
    match (config.nb, config.king_bucketed) {
        (4, false) => fnv1a32("effect-bucket-2x2-kingfixed"),
        (4, true) => fnv1a32("effect-bucket-2x2-kingbucketed"),
        (9, false) => fnv1a32("effect-bucket-3x3-kingfixed"),
        (9, true) => fnv1a32("effect-bucket-3x3-kingbucketed"),
        _ => panic!("unsupported effect bucket config"),
    }
}

// =============================================================================
// FeatureSetSpec — feature 軸の単一の真実源
// =============================================================================

/// feature set 1 つを完全に記述する runtime spec。
///
/// 公開 enum・内部 2 軸・次元・hash を 1 つにまとめ、CLI パース直後に確定して
/// 以降の全層 (dataloader / trainer / export / checkpoint) が同じ spec を参照
/// する。生成は [`FeatureSet::spec`] のみ — フィールドは private で個別箇所が
/// 定数を再計算しない。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureSetSpec {
    feature_set: FeatureSet,
    king_encoding: KingEncoding,
    king_square_mode: KingSquareMode,
    king_buckets: usize,
    piece_inputs: usize,
    max_active: usize,
    feature_hash: u32,
    arch_feature_name: &'static str,
    /// FT factorizer (学習時のみの仮想特徴) が有効か。有効時は実特徴が共有する
    /// virtual piece-input rows と virtual threat-pair rows を FT weight の後ろに持ち、
    /// export 時に実行へ畳み込む。
    /// 仮想行の forward 寄与と勾配は trainer が dense kernel
    /// (実行との畳み込み / 同じ仮想行に対応する実行勾配の縮約) で配線するため、特徴
    /// emit と active 数 (`max_active`) は factorizer 非依存のまま。次元で
    /// 変わるのは weight 行数 (`train_ft_in`) だけ。export 後の artifact
    /// (次元 / hash / arch 名) も base と同一。
    ft_factorize: bool,
    ft_factorize_mode: FtFactorizeMode,
    /// Threat sparse 特徴を base に連結する profile。`None` で base と
    /// bit-identical。`Some(profile)` のとき base の `ft_in` 直後に
    /// `threat_dims(profile)` 行を連結し、`max_active` / `feature_hash` も
    /// 変える (factorizer の次元不変 modifier とは別カテゴリ = base 次元の拡張)。
    /// factorizer とは併用可 (fold/reduce/coalesce が threat pair ごとの仮想行を
    /// 追加で配線する)。PSQT との併用のみ CLI が hard-error にする。
    threat_profile: Option<ThreatProfile>,
    /// effect bucket で base index 全体を書き換える config。`None` で base と
    /// bit-identical。threat とは同時に使わない。
    effect_bucket_config: Option<EffectBucketConfig>,
}

/// FT factorizer の仮想行と実特徴の対応。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FtFactorizeMode {
    /// Base feature index `kb * piece_inputs + p` は `virtual[p]` を参照する。
    Base,
    /// effect bucket index `(kb * piece_inputs + p) * NB + bucket` は `virtual[p]` を参照する。
    PoolEffectBuckets,
    /// effect bucket index `(kb * piece_inputs + p) * NB + bucket` は `virtual[p, bucket]` を参照する。
    PerEffectBucket,
}

impl FeatureSetSpec {
    /// 公開 feature set。
    pub const fn feature_set(&self) -> FeatureSet {
        self.feature_set
    }

    /// king bucket 数 (81 または 45)。
    pub const fn king_buckets(&self) -> usize {
        self.king_buckets
    }

    /// piece 入力数 (king bucket 1 つあたりの特徴 plane 幅)。
    pub const fn piece_inputs(&self) -> usize {
        self.piece_inputs
    }

    /// base feature の入力次元 (`king_buckets * piece_inputs`、threat 連結前)。
    /// threat index emit の offset と factorizer 仮想行の境界に使う。
    pub const fn base_ft_in(&self) -> usize {
        self.king_buckets * self.piece_inputs
    }

    /// 連結 threat profile (無効時 `None`)。
    pub const fn threat_profile(&self) -> Option<ThreatProfile> {
        self.threat_profile
    }

    /// effect bucket config (無効時 `None`)。
    pub const fn effect_bucket_config(&self) -> Option<EffectBucketConfig> {
        self.effect_bucket_config
    }

    /// 連結 threat 次元数 (threat 無効時 0)。
    pub const fn threat_dims(&self) -> usize {
        match self.threat_profile {
            Some(p) => crate::threat::threat_dimensions_of(p),
            None => 0,
        }
    }

    /// 総入力次元 `ft_in`。effect bucket は base index を bucket 数倍に展開し、threat は
    /// base の直後に sparse row を連結する。
    pub const fn ft_in(&self) -> usize {
        match self.effect_bucket_config {
            Some(cfg) => {
                debug_assert!(self.threat_profile.is_none());
                self.base_ft_in() * cfg.nb
            }
            None => self.base_ft_in() + self.threat_dims(),
        }
    }

    /// 1 局面で同時に active になる最大特徴数。threat 有効時は両視点の threat
    /// edge 上限 (`THREAT_MAX_ACTIVE`) を base に加える。
    pub const fn max_active(&self) -> usize {
        match (self.effect_bucket_config, self.threat_profile) {
            (Some(_), _) => self.max_active,
            (None, Some(_)) => self.max_active + THREAT_MAX_ACTIVE,
            (None, None) => self.max_active,
        }
    }

    /// FT factorizer を有効にした spec を返す (modifier 適用の唯一の経路)。
    ///
    /// 有効化しても base 次元 getter (`ft_in` / `max_active` / `feature_hash` /
    /// `arch_feature_name`) は変わらない — export される artifact が base と
    /// 同一形であることを型レベルで表す。学習側の weight buffer / checkpoint
    /// だけが `train_ft_in` を参照する。
    pub const fn with_ft_factorize(self) -> Self {
        let mode = match self.effect_bucket_config {
            Some(_) => FtFactorizeMode::PoolEffectBuckets,
            None => FtFactorizeMode::Base,
        };
        self.with_ft_factorize_mode(mode)
    }

    /// FT factorizer を指定 mode で有効にした spec を返す。
    pub const fn with_ft_factorize_mode(self, mode: FtFactorizeMode) -> Self {
        match (self.effect_bucket_config, mode) {
            (Some(_), FtFactorizeMode::Base) => {
                panic!("effect bucket feature sets need an effect bucket factorizer mode")
            }
            (None, FtFactorizeMode::PoolEffectBuckets | FtFactorizeMode::PerEffectBucket) => {
                panic!("effect bucket factorizer modes require an effect bucket feature set")
            }
            _ => {}
        }
        FeatureSetSpec {
            ft_factorize: true,
            ft_factorize_mode: mode,
            ..self
        }
    }

    /// Threat profile を連結した spec を返す (modifier 適用の唯一の経路)。
    ///
    /// `ft_factorize` とは違い base 次元を拡張する: `ft_in` / `max_active` /
    /// `feature_hash` がすべて変わる。base offset (`base_ft_in`) は不変で、
    /// threat index の emit はその直後に連結される。`ft_factorize` とは併用可
    /// (fold/reduce/coalesce が threat pair ごとの仮想行を追加で配線する)。PSQT との
    /// 併用のみ呼び出し側 (CLI) が hard-error にする (base 限定 PSQT が未検証のため)。
    pub fn with_threat_profile(self, profile: ThreatProfile) -> Self {
        if self.effect_bucket_config.is_some() {
            panic!("effect bucket feature sets cannot use threat profiles");
        }
        FeatureSetSpec {
            threat_profile: Some(profile),
            ..self
        }
    }

    /// effect bucket feature set を有効にした spec を返す。
    pub fn with_effect_bucket_config(self, config: EffectBucketConfig) -> Self {
        if self.threat_profile.is_some() {
            panic!("effect bucket feature sets cannot use threat profiles");
        }
        let ft_factorize_mode = if self.ft_factorize {
            FtFactorizeMode::PoolEffectBuckets
        } else {
            self.ft_factorize_mode
        };
        FeatureSetSpec {
            effect_bucket_config: Some(config),
            ft_factorize_mode,
            ..self
        }
    }

    /// FT factorizer が有効か。
    pub const fn ft_factorize(&self) -> bool {
        self.ft_factorize
    }

    /// FT factorizer の共有 mode。
    pub const fn ft_factorize_mode(&self) -> FtFactorizeMode {
        self.ft_factorize_mode
    }

    /// FT factorizer の仮想行数。
    pub const fn ft_factorize_virtual_rows(&self) -> usize {
        if !self.ft_factorize {
            return 0;
        }
        let base_rows = match (self.ft_factorize_mode, self.effect_bucket_config) {
            (FtFactorizeMode::Base, _) | (FtFactorizeMode::PoolEffectBuckets, _) => {
                self.piece_inputs
            }
            (FtFactorizeMode::PerEffectBucket, Some(cfg)) => self.piece_inputs * cfg.nb,
            (FtFactorizeMode::PerEffectBucket, None) => {
                panic!("effect bucket per-effect-bucket factorizer needs effect bucket config")
            }
        };
        base_rows + self.threat_factorize_pair_count()
    }

    /// Threat profile に残る pair 数。virtual threat-pair rows の行数に使う。
    pub const fn threat_factorize_pair_count(&self) -> usize {
        match self.threat_profile {
            Some(ThreatProfile::Full) | Some(ThreatProfile::FullSymDedup) => 324,
            Some(ThreatProfile::SameClass) => 288,
            Some(ThreatProfile::SameClassMajorPawn) => 272,
            Some(ThreatProfile::StepAttacker) => 144,
            Some(ThreatProfile::CrossSide) => 144,
            None => 0,
        }
    }

    /// Threat within-pair factorizer 用の pair prefix table。
    ///
    /// 返り値は threat block 内 offset の prefix table で、pair ordinal `p` の範囲は
    /// `[starts[p], starts[p + 1])`。factorizer 無効または threat 無効なら空。
    pub fn threat_factorize_pair_starts(&self) -> Vec<usize> {
        let Some(profile) = self.threat_profile else {
            return Vec::new();
        };
        if !self.ft_factorize {
            return Vec::new();
        }
        let mut starts = Vec::with_capacity(self.threat_factorize_pair_count() + 1);
        starts.push(0);
        crate::threat::for_each_threat_pair_range(profile, |_, _, _, _, base, width| {
            debug_assert_eq!(starts.last().copied(), Some(base));
            starts.push(base + width);
        });
        starts
    }

    /// 学習時の FT weight 行数。factorizer 有効時は mode ごとの virtual piece-input rows
    /// と virtual threat-pair rows が実行の後ろに連結される。無効時は `ft_in` と同値。sparse index の
    /// 範囲と active 数は factorizer に依らず base (`ft_in` / `max_active`) の
    /// まま — 仮想行は trainer の dense kernel 経由でのみ読み書きされる。
    pub const fn train_ft_in(&self) -> usize {
        if self.ft_factorize {
            self.ft_in() + self.ft_factorize_virtual_rows()
        } else {
            self.ft_in()
        }
    }

    /// feature 定数 (artifact identity の `FT_HASH` 導出に使う feature 部分)。
    /// threat 連結時は base hash に profile hash を XOR 合成する (`off` は base
    /// と bit-identical)。全 base × 全 profile の合成 hash が pairwise distinct で
    /// あることは test (`threat_profile_hashes_keep_all_specs_distinct`) で固定。
    pub const fn feature_hash(&self) -> u32 {
        match (self.effect_bucket_config, self.threat_profile) {
            (Some(cfg), _) => self.feature_hash ^ effect_bucket_config_hash(cfg),
            (None, Some(p)) => self.feature_hash ^ threat_profile_hash(p),
            (None, None) => self.feature_hash,
        }
    }

    /// arch 文字列に埋める feature 名。
    pub const fn arch_feature_name(&self) -> &'static str {
        self.arch_feature_name
    }

    /// canonical 名 ([`FeatureSet::canonical_name`])。
    pub const fn canonical_name(&self) -> &'static str {
        self.feature_set.canonical_name()
    }

    /// 玉自体を active 特徴として emit するか (軸 1 が `NoKing` 以外)。
    const fn emits_king_feature(&self) -> bool {
        !matches!(self.king_encoding, KingEncoding::NoKing)
    }

    /// 敵玉 BonaPiece を自玉 plane へ畳むか (軸 1 が `MergedPlane`)。
    const fn folds_enemy_king(&self) -> bool {
        matches!(self.king_encoding, KingEncoding::MergedPlane)
    }

    /// `PackedSfenValue` をデコードして特徴インデックスを列挙する。
    ///
    /// `decode` のコストを避けたい呼び出し側は [`map_features_board`] を使う。
    ///
    /// [`map_features_board`]: Self::map_features_board
    pub fn map_features<F: FnMut(usize, usize)>(&self, pos: &PackedSfenValue, f: F) {
        self.map_features_board(&pos.decode(), f);
    }

    /// decode 済み [`ShogiBoard`] から特徴インデックスを列挙する。
    ///
    /// 各駒について `(stm_idx, nstm_idx)` の 2 視点ペアを `f` に渡す。
    /// 玉位置が無効な局面 (片玉 / 詰将棋データ) は何も emit しない。
    ///
    /// dataloader は 1 局面につき `decode` を 1 回だけ呼び、得た `ShogiBoard` を
    /// 特徴抽出と progress bucket 計算で共有するため、こちらが基本の入口。
    /// 2 軸の解釈 (king bucket / 筋ミラー要否) は 1 局面につき 1 回だけ
    /// `PerspectiveCtx` に畳み、内側の駒走査ループには分岐を持ち込まない。
    /// emit は FT factorizer に依存しない (仮想行は trainer の dense kernel が
    /// 配線するため sparse index 列には現れない)。
    pub fn map_features_board<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        if let Some(config) = self.effect_bucket_config {
            self.map_effect_bucket_features_board_both(board, config, f);
            return;
        }

        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king = board.king_square(stm);
        let nstm_king = board.king_square(nstm);
        if !stm_king.is_valid() || !nstm_king.is_valid() {
            return;
        }

        let stm_ctx = self.perspective_ctx(stm_king, stm);
        let nstm_ctx = self.perspective_ctx(nstm_king, nstm);

        // 盤上の駒 (玉以外)。`ShogiBoard::for_each_board_piece` が board を 1-pass
        // で `(piece_type, color, ascending square)` 順に供給する (`pieces(color, pt)`
        // を 26 通り loop で呼ぶパターンと emit 順は同一、helper doc 参照)。
        board.for_each_board_piece(|piece, sq| {
            let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
            let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
            f(self.index(&stm_ctx, stm_bp), self.index(&nstm_ctx, nstm_bp));
        });

        // 両玉の特徴 (軸 1 が玉を特徴化するときのみ)。
        if self.emits_king_feature() {
            let stm_friend = king_bonapiece(stm_king, stm, true);
            let nstm_friend = king_bonapiece(nstm_king, nstm, true);
            f(
                self.index(&stm_ctx, stm_friend),
                self.index(&nstm_ctx, nstm_friend),
            );

            let stm_enemy = king_bonapiece(nstm_king, stm, false);
            let nstm_enemy = king_bonapiece(stm_king, nstm, false);
            f(
                self.index(&stm_ctx, stm_enemy),
                self.index(&nstm_ctx, nstm_enemy),
            );
        }

        // 手駒の特徴。
        for owner in [Color::Black, Color::White] {
            for &pt in &HAND_PIECE_TYPES {
                let count = board.hand(owner).count(pt);
                for i in 1..=count {
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                    if stm_bp != BonaPiece::ZERO {
                        let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                        f(self.index(&stm_ctx, stm_bp), self.index(&nstm_ctx, nstm_bp));
                    }
                }
            }
        }

        // Threat 特徴 (profile 有効時のみ)。base の `ft_in` (= `base_ft_in`) 直後に
        // 連結するので offset を足して emit する。
        if let Some(profile) = self.threat_profile {
            let base = self.base_ft_in();
            ThreatIndexer::shared(profile).for_each_active_threat_index_pair(
                board,
                stm,
                nstm,
                |s, n| f(base + s, base + n),
            );
        }
    }

    /// 特徴インデックスを `Vec<(stm_idx, nstm_idx)>` として収集する。
    ///
    /// 容量は `max_active` で事前確保する。
    pub fn collect_active_indices(&self, pos: &PackedSfenValue) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.max_active());
        self.map_features(pos, |stm, nstm| out.push((stm, nstm)));
        out
    }

    /// active 特徴 index を `stm_out` / `nstm_out` に直接書込んで件数を返す。
    /// closure 経由 [`map_features_board`] と byte-identical な出力を SIMD store
    /// と直接 fit する形で取り出す。
    ///
    /// 玉位置が無効な局面 (片玉 / 詰将棋) は 0 を返して何も書込まない。
    ///
    /// 戻り値は **実 active 数** で、出力 slice 長 `cap` を超え得る (threat 連結
    /// 時に `THREAT_MAX_ACTIVE` の見積りを超えた場合)。書き込みは `cap` 以内に
    /// 限るが、超過は silent truncation せず戻り値に反映するので、caller は
    /// `count > cap` を overflow として hard-error 検出できる。base 特徴 (board /
    /// king / hand) は合法局面で必ず `cap` 内なので超過は threat phase でのみ起こる。
    ///
    /// HalfKaHmMerged の board phase は [`crate::simd`] (runtime detect で
    /// scalar / AVX-2 / AVX-512 のいずれか)、それ以外 / king / hand phase は
    /// scalar。
    ///
    /// [`map_features_board`]: Self::map_features_board
    pub fn extract_active_features(
        &self,
        board: &ShogiBoard,
        stm_out: &mut [i32],
        nstm_out: &mut [i32],
    ) -> usize {
        debug_assert_eq!(stm_out.len(), nstm_out.len());

        if let Some(config) = self.effect_bucket_config {
            let cap = stm_out.len();
            let mut count = 0usize;
            self.map_effect_bucket_features_board_both(board, config, |stm, nstm| {
                if count < cap {
                    stm_out[count] = stm as i32;
                    nstm_out[count] = nstm as i32;
                }
                count += 1;
            });
            return count;
        }

        let stm = board.side_to_move;
        let nstm = stm.opponent();
        let stm_king = board.king_square(stm);
        let nstm_king = board.king_square(nstm);
        if !stm_king.is_valid() || !nstm_king.is_valid() {
            return 0;
        }

        let stm_ctx = self.perspective_ctx(stm_king, stm);
        let nstm_ctx = self.perspective_ctx(nstm_king, nstm);
        let cap = stm_out.len();
        let mut count = 0usize;

        // board phase
        if crate::simd::spec_is_halfka_hm_merged(self) {
            // (pt, color, sq) を i32 stack 配列に積む。`for_each_board_piece` は
            // 81 マス上限、AVX-512 lane (16) 倍数で 96 確保。
            const STACK_BUF: usize = 96;
            let mut pt_buf = [0i32; STACK_BUF];
            let mut color_buf = [0i32; STACK_BUF];
            let mut sq_buf = [0i32; STACK_BUF];
            let mut n_pieces = 0usize;
            board.for_each_board_piece(|piece, sq| {
                if n_pieces < STACK_BUF {
                    pt_buf[n_pieces] = piece.piece_type as i32;
                    color_buf[n_pieces] = piece.color as i32;
                    sq_buf[n_pieces] = sq.0 as i32;
                    n_pieces += 1;
                }
            });

            let writable = n_pieces.min(cap.saturating_sub(count));
            if writable > 0 {
                let stm_pers = crate::simd::PerspectiveOffset {
                    kb_offset: (stm_ctx.king_bucket * self.piece_inputs) as i32,
                    mirror: stm_ctx.mirror_files,
                    black_persp: if stm == Color::Black { 1 } else { 0 },
                    color_code: stm as i32,
                };
                let nstm_pers = crate::simd::PerspectiveOffset {
                    kb_offset: (nstm_ctx.king_bucket * self.piece_inputs) as i32,
                    mirror: nstm_ctx.mirror_files,
                    black_persp: if nstm == Color::Black { 1 } else { 0 },
                    color_code: nstm as i32,
                };
                crate::simd::extract_halfka_hm_board_phase(crate::simd::BoardPhaseArgs {
                    pt: &pt_buf,
                    color: &color_buf,
                    sq: &sq_buf,
                    n: writable,
                    stm: &stm_pers,
                    nstm: &nstm_pers,
                    stm_out: &mut stm_out[count..count + writable],
                    nstm_out: &mut nstm_out[count..count + writable],
                });
                count += writable;
            }
        } else {
            // HalfKaHmMerged 以外は SIMD 化対象外、closure で 1 駒ずつ計算。
            board.for_each_board_piece(|piece, sq| {
                if count >= cap {
                    return;
                }
                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                stm_out[count] = self.index(&stm_ctx, stm_bp) as i32;
                nstm_out[count] = self.index(&nstm_ctx, nstm_bp) as i32;
                count += 1;
            });
        }

        // king phase
        if self.emits_king_feature() {
            if count < cap {
                let stm_friend = king_bonapiece(stm_king, stm, true);
                let nstm_friend = king_bonapiece(nstm_king, nstm, true);
                stm_out[count] = self.index(&stm_ctx, stm_friend) as i32;
                nstm_out[count] = self.index(&nstm_ctx, nstm_friend) as i32;
                count += 1;
            }
            if count < cap {
                let stm_enemy = king_bonapiece(nstm_king, stm, false);
                let nstm_enemy = king_bonapiece(stm_king, nstm, false);
                stm_out[count] = self.index(&stm_ctx, stm_enemy) as i32;
                nstm_out[count] = self.index(&nstm_ctx, nstm_enemy) as i32;
                count += 1;
            }
        }

        // hand phase
        for owner in [Color::Black, Color::White] {
            for &pt in &HAND_PIECE_TYPES {
                let n_hand = board.hand(owner).count(pt);
                for i in 1..=n_hand {
                    if count >= cap {
                        return count;
                    }
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                    if stm_bp != BonaPiece::ZERO {
                        let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                        stm_out[count] = self.index(&stm_ctx, stm_bp) as i32;
                        nstm_out[count] = self.index(&nstm_ctx, nstm_bp) as i32;
                        count += 1;
                    }
                }
            }
        }

        // Threat phase (profile 有効時のみ)。base は合法局面で必ず cap 内に収まる
        // (king/hand 含め最大 `base max_active`) ため、ここに来る時点で base 分は
        // 書き込み済。threat edge は `THREAT_MAX_ACTIVE` の見積りを超え得るので、
        // 戻り値は cap で頭打ちにせず **真の総数** を返す: 書き込みは `count < cap`
        // の範囲に限り、超過分は数えるだけにする。caller (dataloader) が
        // `total > max_active` を hard-error 検出するための signal。
        let mut total = count;
        if let Some(profile) = self.threat_profile {
            let base = self.base_ft_in();
            ThreatIndexer::shared(profile).for_each_active_threat_index_pair(
                board,
                stm,
                nstm,
                |s, n| {
                    if total < cap {
                        stm_out[total] = (base + s) as i32;
                        nstm_out[total] = (base + n) as i32;
                    }
                    total += 1;
                },
            );
        }

        total
    }

    /// `perspective_ctx` の crate 内 expose (parity test 用)。
    #[cfg(test)]
    pub(crate) fn perspective_ctx_for_test(
        &self,
        king_sq: Square,
        perspective: Color,
    ) -> (usize, bool) {
        let ctx = self.perspective_ctx(king_sq, perspective);
        (ctx.king_bucket, ctx.mirror_files)
    }

    /// 1 視点分の king bucket / 筋ミラー要否を確定する。
    fn perspective_ctx(&self, king_sq: Square, perspective: Color) -> PerspectiveCtx {
        let king_idx = perspective_index(king_sq, perspective);
        match self.king_square_mode {
            KingSquareMode::Direct => PerspectiveCtx {
                king_bucket: king_idx,
                mirror_files: false,
            },
            KingSquareMode::HorizontalMirror => {
                let file = king_idx / 9;
                let rank = king_idx % 9;
                // 6-9 筋 (file >= 5) を 1-4 筋へ反転し、5 筋 × 9 段の 45 bucket に。
                let file_m = if file >= 5 { 8 - file } else { file };
                PerspectiveCtx {
                    king_bucket: file_m * 9 + rank,
                    mirror_files: file >= 5,
                }
            }
        }
    }

    /// 1 視点の context と BonaPiece から特徴インデックスを計算する。
    fn index(&self, ctx: &PerspectiveCtx, bp: BonaPiece) -> usize {
        ctx.king_bucket * self.piece_inputs + self.pack_bonapiece(bp, ctx.mirror_files)
    }

    fn effect_bucket_index(
        &self,
        ctx: &PerspectiveCtx,
        counts: &EffectBucketAttackCounts,
        config: EffectBucketConfig,
        bp: BonaPiece,
        board_piece: Option<(Color, Square)>,
    ) -> usize {
        let packed = self.pack_bonapiece(bp, ctx.mirror_files);
        let base = ctx.king_bucket * self.piece_inputs + packed;
        let bucket = if crate::effect_bucket::packed_is_bucketed(packed, config.king_bucketed) {
            let (color, sq) =
                board_piece.expect("bucketed effect bucket feature must have a board square");
            crate::effect_bucket::effect_bucket(
                counts.get(color.opponent(), sq),
                counts.get(color, sq),
                config.nb,
            )
        } else {
            0
        };
        crate::effect_bucket::effect_bucket_index(base, bucket, config.nb)
    }

    fn map_effect_bucket_features_board_both<F: FnMut(usize, usize)>(
        &self,
        board: &ShogiBoard,
        config: EffectBucketConfig,
        mut f: F,
    ) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king = board.king_square(stm);
        let nstm_king = board.king_square(nstm);
        if !stm_king.is_valid() || !nstm_king.is_valid() {
            return;
        }

        let stm_ctx = self.perspective_ctx(stm_king, stm);
        let nstm_ctx = self.perspective_ctx(nstm_king, nstm);
        let counts = crate::effect_bucket::effect_bucket_attacker_counts(board);

        board.for_each_board_piece(|piece, sq| {
            let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
            let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
            f(
                self.effect_bucket_index(
                    &stm_ctx,
                    &counts,
                    config,
                    stm_bp,
                    Some((piece.color, sq)),
                ),
                self.effect_bucket_index(
                    &nstm_ctx,
                    &counts,
                    config,
                    nstm_bp,
                    Some((piece.color, sq)),
                ),
            );
        });

        if self.emits_king_feature() {
            let stm_friend = king_bonapiece(stm_king, stm, true);
            let nstm_friend = king_bonapiece(nstm_king, nstm, true);
            f(
                self.effect_bucket_index(
                    &stm_ctx,
                    &counts,
                    config,
                    stm_friend,
                    Some((stm, stm_king)),
                ),
                self.effect_bucket_index(
                    &nstm_ctx,
                    &counts,
                    config,
                    nstm_friend,
                    Some((nstm, nstm_king)),
                ),
            );

            let stm_enemy = king_bonapiece(nstm_king, stm, false);
            let nstm_enemy = king_bonapiece(stm_king, nstm, false);
            f(
                self.effect_bucket_index(
                    &stm_ctx,
                    &counts,
                    config,
                    stm_enemy,
                    Some((nstm, nstm_king)),
                ),
                self.effect_bucket_index(
                    &nstm_ctx,
                    &counts,
                    config,
                    nstm_enemy,
                    Some((stm, stm_king)),
                ),
            );
        }

        for owner in [Color::Black, Color::White] {
            for &pt in &HAND_PIECE_TYPES {
                for i in 1..=board.hand(owner).count(pt) {
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                    if stm_bp != BonaPiece::ZERO {
                        let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                        f(
                            self.effect_bucket_index(&stm_ctx, &counts, config, stm_bp, None),
                            self.effect_bucket_index(&nstm_ctx, &counts, config, nstm_bp, None),
                        );
                    }
                }
            }
        }
    }

    /// BonaPiece を piece-input ordinal 内のインデックスへ写す。
    ///
    /// 1. `mirror_files` のとき盤上駒・玉のマスを筋反転する (手駒は対象外)。
    /// 2. 軸 1 が `MergedPlane` のとき敵玉 (`>= E_KING`) を 81 引いて自玉 plane に
    ///    重ねる。
    fn pack_bonapiece(&self, bp: BonaPiece, mirror_files: bool) -> usize {
        let mut pp = bp.value() as usize;

        if mirror_files && pp >= FE_HAND_END {
            // 盤上駒・玉の layout は `FE_HAND_END + piece_index * 81 + sq`。
            let rel = pp - FE_HAND_END;
            let piece_index = rel / 81;
            let sq = rel % 81;
            let file = sq / 9;
            let rank = sq % 9;
            let mirrored_sq = (8 - file) * 9 + rank;
            pp = FE_HAND_END + piece_index * 81 + mirrored_sq;
        }

        if self.folds_enemy_king() && pp >= E_KING as usize {
            pp -= 81;
        }

        pp
    }
}

// =============================================================================
// indexer 内部
// =============================================================================

/// 1 視点分の特徴抽出 context。2 軸の解釈を 1 局面 1 回ここに畳む。
struct PerspectiveCtx {
    /// king bucket index。
    king_bucket: usize,
    /// 盤上駒・玉のマスを筋反転するか。
    mirror_files: bool,
}

/// 玉マスを視点変換したマスインデックス (後手視点は盤を 180 度回転)。
#[inline]
fn perspective_index(sq: Square, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq.index(),
        Color::White => sq.inverse().index(),
    }
}

/// 玉の BonaPiece を生成する。
///
/// 自玉は `F_KING` plane、敵玉は `E_KING` plane を base にし、視点変換した
/// 玉マスを足す。
#[inline]
fn king_bonapiece(king_sq: Square, perspective: Color, is_friend: bool) -> BonaPiece {
    let base = if is_friend { F_KING } else { E_KING };
    BonaPiece::new(base + perspective_index(king_sq, perspective) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_dimensions_match_design_table() {
        // (canonical, king_buckets, piece_inputs, ft_in, max_active)
        let expected = [
            ("halfkp", 81, 1548, 125_388, 38),
            ("halfka-split", 81, 1710, 138_510, 40),
            ("halfka-merged", 81, 1629, 131_949, 40),
            ("halfka-hm-split", 45, 1710, 76_950, 40),
            ("halfka-hm-merged", 45, 1629, 73_305, 40),
        ];
        for (fs, &(name, kb, pi, ft_in, ma)) in FeatureSet::ALL.iter().zip(&expected) {
            let spec = fs.spec();
            assert_eq!(spec.canonical_name(), name);
            assert_eq!(spec.king_buckets(), kb, "{name} king_buckets");
            assert_eq!(spec.piece_inputs(), pi, "{name} piece_inputs");
            assert_eq!(spec.ft_in(), ft_in, "{name} ft_in");
            assert_eq!(spec.ft_in(), kb * pi, "{name} ft_in == kb * pi");
            assert_eq!(spec.max_active(), ma, "{name} max_active");
        }
    }

    #[test]
    fn canonical_name_round_trips() {
        for fs in FeatureSet::ALL {
            assert_eq!(
                FeatureSet::from_canonical_name(fs.canonical_name()),
                Some(fs)
            );
        }
        assert_eq!(FeatureSet::from_canonical_name("halfka"), None);
        assert_eq!(FeatureSet::from_canonical_name(""), None);
    }

    #[test]
    fn feature_hashes_are_pinned_and_distinct() {
        // 各 cell の feature hash を数値で固定する (取り違え / typo 検出)。
        // halfkp / halfka-split / halfka-hm-merged は nnue-pytorch 系の固定値、
        // halfka-merged / halfka-hm-split は canonical 名の FNV-1a 32bit hash。
        let expected: [(FeatureSet, u32); 5] = [
            (FeatureSet::HalfKp, 0x5D69_D5B8),
            (FeatureSet::HalfKaSplit, 0x5F13_4CB8),
            (FeatureSet::HalfKaMerged, 0xACD6_8C97),
            (FeatureSet::HalfKaHmSplit, 0x2A46_AC09),
            (FeatureSet::HalfKaHmMerged, 0x7F13_4CB8),
        ];
        let mut seen = Vec::new();
        for (fs, hash) in expected {
            assert_eq!(fs.spec().feature_hash(), hash, "{}", fs.canonical_name());
            seen.push(hash);
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 5, "feature hash が衝突している");

        // FNV-1a が canonical 名から決定的に上記の値を導くことを確認する。
        assert_eq!(fnv1a32("halfka-merged"), 0xACD6_8C97);
        assert_eq!(fnv1a32("halfka-hm-split"), 0x2A46_AC09);
    }

    #[test]
    fn extract_active_features_matches_map_features_board() {
        // 公開 5 feature set × sample.psv 100 records で
        // `extract_active_features` と `map_features_board` の sparse index 列が
        // byte-identical であることを確認する。
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let Ok(bytes) = std::fs::read(&path) else {
            // fixture が無い CI 構成 (本リポでは同梱) は skip。
            return;
        };
        assert_eq!(bytes.len() % 40, 0);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (size_of test で 40 byte 確認済、align 1)、`bytes.len() % 40 == 0` を
        // 直前で assert。`bytes` の所有 lifetime 内に閉じる slice。
        let records: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };

        for fs in FeatureSet::ALL {
            let spec = fs.spec();
            for (i, psv) in records.iter().enumerate() {
                let board = psv.decode();
                // closure path
                let mut via_closure = Vec::new();
                spec.map_features_board(&board, |stm, nstm| {
                    via_closure.push((stm as i32, nstm as i32));
                });
                // direct-write path
                let mut stm_buf = vec![0i32; spec.max_active()];
                let mut nstm_buf = vec![0i32; spec.max_active()];
                let n = spec.extract_active_features(&board, &mut stm_buf, &mut nstm_buf);
                let via_direct: Vec<(i32, i32)> =
                    (0..n).map(|k| (stm_buf[k], nstm_buf[k])).collect();

                assert_eq!(
                    via_direct,
                    via_closure,
                    "{} record {}: extract_active_features と map_features_board が不一致",
                    fs.canonical_name(),
                    i
                );
            }
        }
    }

    #[test]
    fn factorized_spec_train_dimensions() {
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            let fact = fs.spec().with_ft_factorize();
            // base getter は modifier の影響を受けない (export 形状と sparse
            // emit の不変条件 — active 数も factorizer 非依存)。
            assert_eq!(fact.ft_in(), base.ft_in());
            assert_eq!(fact.max_active(), base.max_active());
            assert_eq!(fact.feature_hash(), base.feature_hash());
            assert_eq!(fact.arch_feature_name(), base.arch_feature_name());
            // train_ft_in は OFF では base と同値、ON でpiece-input 仮想行 を連結。
            assert_eq!(base.train_ft_in(), base.ft_in());
            assert_eq!(fact.train_ft_in(), base.ft_in() + base.piece_inputs());
            // modifier は PartialEq で弁別される (Batch / trainer / weight の
            // spec 照合が on/off 混在を自動 reject する根拠)。適用は冪等。
            assert_ne!(fact, base);
            assert_eq!(fact.with_ft_factorize(), fact);
        }
    }

    #[test]
    fn factorized_emission_matches_base() {
        // 特徴 emit は factorizer 非依存: ON の emit 列 = OFF の実特徴列
        // (仮想行は trainer の dense kernel が配線するため index 列に現れない)。
        // direct-write 経路 (`extract_active_features`) と closure 経路の一致も
        // ON で確認する。
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
        assert_eq!(bytes.len() % 40, 0);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (size_of test で 40 byte 確認済、align 1)、`bytes.len() % 40 == 0` を
        // 直前で assert。`bytes` の所有 lifetime 内に閉じる slice。
        let records: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };

        for fs in FeatureSet::ALL {
            let base = fs.spec();
            let spec = fs.spec().with_ft_factorize();
            for (i, psv) in records.iter().take(20).enumerate() {
                let board = psv.decode();
                let mut real = Vec::new();
                base.map_features_board(&board, |s, n| real.push((s, n)));
                let mut on = Vec::new();
                spec.map_features_board(&board, |s, n| on.push((s, n)));

                assert_eq!(on, real, "{} record {i}", fs.canonical_name());
                for &(s, n) in &on {
                    assert!(s < base.ft_in());
                    assert!(n < base.ft_in());
                }

                let mut stm_buf = vec![0i32; spec.max_active()];
                let mut nstm_buf = vec![0i32; spec.max_active()];
                let cnt = spec.extract_active_features(&board, &mut stm_buf, &mut nstm_buf);
                let direct: Vec<(usize, usize)> = (0..cnt)
                    .map(|k| (stm_buf[k] as usize, nstm_buf[k] as usize))
                    .collect();
                assert_eq!(direct, on, "{} record {i}", fs.canonical_name());
            }
        }
    }

    #[test]
    fn arch_feature_name_uses_pascal_case() {
        // arch_str に embed される keyword は PascalCase 表記で固定する。
        // net を load する側の parser は両綴りを受理し得るが、emit 側は
        // 単一の canonical 形を残す。
        let expected = [
            (FeatureSet::HalfKp, "HalfKP"),
            (FeatureSet::HalfKaSplit, "HalfKaSplit"),
            (FeatureSet::HalfKaMerged, "HalfKaMerged"),
            (FeatureSet::HalfKaHmSplit, "HalfKaHmSplit"),
            (FeatureSet::HalfKaHmMerged, "HalfKaHmMerged"),
        ];
        for (fs, name) in expected {
            assert_eq!(fs.spec().arch_feature_name(), name, "{:?}", fs);
        }
    }

    // ---- Threat 連結 ----

    const ALL_PROFILES: [ThreatProfile; 6] = [
        ThreatProfile::Full,
        ThreatProfile::SameClass,
        ThreatProfile::SameClassMajorPawn,
        ThreatProfile::StepAttacker,
        ThreatProfile::FullSymDedup,
        ThreatProfile::CrossSide,
    ];

    const ALL_EFFECT_BUCKET_CONFIGS: [EffectBucketConfig; 4] = [
        EffectBucketConfig::KINGFIXED_2X2,
        EffectBucketConfig::KINGBUCKETED_2X2,
        EffectBucketConfig::KINGFIXED_3X3,
        EffectBucketConfig::KINGBUCKETED_3X3,
    ];

    #[test]
    fn threat_off_is_bit_identical_to_base() {
        // `with_threat_profile` を呼ばない spec は base と全 getter 一致。
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            assert_eq!(base.threat_profile(), None);
            assert_eq!(base.threat_dims(), 0);
            assert_eq!(base.ft_in(), base.base_ft_in());
        }
    }

    #[test]
    fn threat_getters_extend_base_dimensions() {
        let dims = [
            (ThreatProfile::Full, 216_720usize),
            (ThreatProfile::SameClass, 192_640),
            (ThreatProfile::SameClassMajorPawn, 173_568),
            (ThreatProfile::StepAttacker, 33_408),
            // FullSymDedup は index 空間が Full と同一 (dims 216_720)。
            (ThreatProfile::FullSymDedup, 216_720),
            (ThreatProfile::CrossSide, 96_320),
        ];
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            for (profile, d) in dims {
                let spec = base.with_threat_profile(profile);
                assert_eq!(spec.threat_profile(), Some(profile));
                assert_eq!(spec.threat_dims(), d);
                assert_eq!(spec.base_ft_in(), base.ft_in());
                assert_eq!(spec.ft_in(), base.ft_in() + d);
                assert_eq!(spec.max_active(), base.max_active() + THREAT_MAX_ACTIVE);
                // threat-only は factorizer 無効なので train_ft_in == ft_in。
                assert_eq!(spec.train_ft_in(), spec.ft_in());
            }
        }
    }

    #[test]
    fn threat_factorized_pair_prefix_matches_profile_dimensions() {
        for fs in FeatureSet::ALL {
            for profile in ALL_PROFILES {
                let spec = fs.spec().with_threat_profile(profile).with_ft_factorize();
                let starts = spec.threat_factorize_pair_starts();
                assert_eq!(
                    starts.len(),
                    spec.threat_factorize_pair_count() + 1,
                    "{} {profile} pair prefix length",
                    fs.canonical_name()
                );
                assert_eq!(
                    starts.last().copied(),
                    Some(spec.threat_dims()),
                    "{} {profile} pair prefix terminal dim",
                    fs.canonical_name()
                );
                assert!(
                    starts.windows(2).all(|w| w[0] < w[1]),
                    "{} {profile} pair prefix must be strictly increasing",
                    fs.canonical_name()
                );
                assert_eq!(
                    spec.train_ft_in(),
                    spec.ft_in() + spec.piece_inputs() + spec.threat_factorize_pair_count(),
                    "{} {profile} train_ft_in",
                    fs.canonical_name()
                );
            }
        }
    }

    #[test]
    fn threat_profile_hashes_keep_all_specs_distinct() {
        // 全 base(5) × {off + 6 profile} = 35 通りの合成 feature_hash が pairwise
        // distinct (profile compaction / dedup で row の意味が変わるため load 時に
        // 弾ける。FullSymDedup は dims が Full と同一なので特に hash で判別する)。
        let mut seen = Vec::new();
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            seen.push(base.feature_hash());
            for profile in ALL_PROFILES {
                let h = base.with_threat_profile(profile).feature_hash();
                // off と各 profile も互いに distinct。
                assert_ne!(
                    h,
                    base.feature_hash(),
                    "{} {profile}",
                    base.canonical_name()
                );
                seen.push(h);
            }
        }
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n, "合成 feature_hash に衝突がある (35 通り)");
    }

    #[test]
    fn threat_emit_appends_after_base_with_offset() {
        // threat-on の emit 列 = base 実特徴列 + (base_ft_in offset を足した threat
        // index 列)。closure 経路と direct-write 経路が一致し、全 index が
        // `[0, ft_in())` 内であることを sample.psv で確認する。
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
        assert_eq!(bytes.len() % 40, 0);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (size_of test で 40 byte 確認済、align 1)、`bytes.len() % 40 == 0` を
        // 直前で assert。`bytes` の所有 lifetime 内に閉じる slice。
        let records: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };

        let base = FeatureSet::HalfKaHmMerged.spec();
        for profile in ALL_PROFILES {
            let spec = base.with_threat_profile(profile);
            let base_ft_in = base.ft_in();
            for psv in records.iter().take(20) {
                let board = psv.decode();

                // base 実特徴のみ (closure)。
                let mut base_only = Vec::new();
                base.map_features_board(&board, |s, n| base_only.push((s, n)));

                // threat-on の closure 列。
                let mut on = Vec::new();
                spec.map_features_board(&board, |s, n| on.push((s, n)));

                // 先頭 base 個は base と一致。
                assert_eq!(&on[..base_only.len()], &base_only[..], "{profile}");
                // 残りは threat region (>= base_ft_in)、全体が ft_in 内。
                for &(s, n) in &on {
                    assert!(s < spec.ft_in() && n < spec.ft_in(), "{profile}");
                }
                for &(s, n) in &on[base_only.len()..] {
                    assert!(
                        s >= base_ft_in && n >= base_ft_in,
                        "{profile} threat region"
                    );
                }

                // direct-write 経路 (cap 内) が closure と一致。実 PSV は cap 内に
                // 収まる前提 (収まらなければ overflow test 側で検出)。
                let cap = spec.max_active();
                let mut stm_buf = vec![0i32; cap];
                let mut nstm_buf = vec![0i32; cap];
                let total = spec.extract_active_features(&board, &mut stm_buf, &mut nstm_buf);
                assert!(
                    total <= cap,
                    "{profile}: total {total} > cap {cap} (sample.psv)"
                );
                let direct: Vec<(usize, usize)> = (0..total)
                    .map(|k| (stm_buf[k] as usize, nstm_buf[k] as usize))
                    .collect();
                assert_eq!(direct, on, "{profile}: direct != closure");
            }
        }
    }

    #[test]
    fn extract_returns_true_total_on_threat_overflow() {
        // cap を意図的に小さく (base+1) 渡し、threat edge の超過が silent truncate
        // されず戻り値 (真の active 数) に反映されることを確認する (dataloader の
        // hard-error signal)。startpos は threat edge が 0 でないので超過する。
        let spec = FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(ThreatProfile::Full);
        let board = startpos_for_overflow();

        // 正規 cap での真の active 数。
        let full_cap = spec.max_active();
        let mut s = vec![0i32; full_cap];
        let mut n = vec![0i32; full_cap];
        let true_total = spec.extract_active_features(&board, &mut s, &mut n);
        assert!(
            true_total > 40,
            "startpos should have threat edges beyond base"
        );

        // 小さい cap でも戻り値は真の総数 (cap で頭打ちにならない)。
        let small_cap = 41usize;
        let mut s2 = vec![0i32; small_cap];
        let mut n2 = vec![0i32; small_cap];
        let total = spec.extract_active_features(&board, &mut s2, &mut n2);
        assert_eq!(total, true_total, "戻り値が cap で truncate されている");
        assert!(total > small_cap, "overflow (cap 越え) を検出できる戻り値");
    }

    #[test]
    fn effect_bucket_getters_multiply_base_dims() {
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            for cfg in ALL_EFFECT_BUCKET_CONFIGS {
                let spec = base.with_effect_bucket_config(cfg);
                assert_eq!(spec.effect_bucket_config(), Some(cfg));
                assert_eq!(spec.base_ft_in(), base.base_ft_in());
                assert_eq!(spec.ft_in(), base.base_ft_in() * cfg.nb);
                assert_eq!(spec.max_active(), base.max_active());
                assert_eq!(spec.train_ft_in(), spec.ft_in());
                let fact = spec.with_ft_factorize();
                assert_eq!(fact.ft_factorize_mode(), FtFactorizeMode::PoolEffectBuckets);
                assert_eq!(fact.train_ft_in(), spec.ft_in() + spec.piece_inputs());
                let fact_bucket = spec.with_ft_factorize_mode(FtFactorizeMode::PerEffectBucket);
                assert_eq!(
                    fact_bucket.train_ft_in(),
                    spec.ft_in() + spec.piece_inputs() * cfg.nb
                );
            }
        }
    }

    #[test]
    fn effect_bucket_config_hashes_keep_all_specs_distinct() {
        let mut seen = Vec::new();
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            seen.push(base.feature_hash());
            for cfg in ALL_EFFECT_BUCKET_CONFIGS {
                let h = base.with_effect_bucket_config(cfg).feature_hash();
                assert_ne!(h, base.feature_hash(), "{}", base.canonical_name());
                seen.push(h);
            }
        }
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n, "合成 feature_hash に衝突がある (25 通り)");
    }

    #[test]
    #[should_panic(expected = "effect bucket feature sets cannot use threat profiles")]
    fn with_effect_bucket_config_rejects_threat() {
        FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(ThreatProfile::CrossSide)
            .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2);
    }

    #[test]
    fn with_effect_bucket_config_keeps_factorize_enabled() {
        let spec = FeatureSet::HalfKaHmMerged
            .spec()
            .with_ft_factorize()
            .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2);
        assert!(spec.ft_factorize());
        assert_eq!(spec.ft_factorize_mode(), FtFactorizeMode::PoolEffectBuckets);
    }

    #[test]
    fn effect_bucket_emit_matches_closure_and_uses_effect_bucket_range() {
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
        assert_eq!(bytes.len() % 40, 0);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (size_of test で 40 byte 確認済、align 1)、`bytes.len() % 40 == 0` を
        // 直前で assert。`bytes` の所有 lifetime 内に閉じる slice。
        let records: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };

        let cfg = EffectBucketConfig::KINGFIXED_2X2;
        for fs in FeatureSet::ALL {
            let base = fs.spec();
            let spec = base.with_effect_bucket_config(cfg);
            for (i, psv) in records.iter().take(20).enumerate() {
                let board = psv.decode();
                let mut base_pairs = Vec::new();
                base.map_features_board(&board, |s, n| base_pairs.push((s, n)));
                let mut effect_bucket_pairs = Vec::new();
                spec.map_features_board(&board, |s, n| effect_bucket_pairs.push((s, n)));
                assert_eq!(
                    effect_bucket_pairs.len(),
                    base_pairs.len(),
                    "{} record {i}",
                    fs.canonical_name()
                );
                for (&(effect_bucket_s, effect_bucket_n), &(base_s, base_n)) in
                    effect_bucket_pairs.iter().zip(&base_pairs)
                {
                    assert_eq!(
                        effect_bucket_s / cfg.nb,
                        base_s,
                        "{} record {i}",
                        fs.canonical_name()
                    );
                    assert_eq!(
                        effect_bucket_n / cfg.nb,
                        base_n,
                        "{} record {i}",
                        fs.canonical_name()
                    );
                    assert!(effect_bucket_s < spec.ft_in() && effect_bucket_n < spec.ft_in());
                }

                let mut stm_buf = vec![0i32; spec.max_active()];
                let mut nstm_buf = vec![0i32; spec.max_active()];
                let cnt = spec.extract_active_features(&board, &mut stm_buf, &mut nstm_buf);
                let direct: Vec<(usize, usize)> = (0..cnt)
                    .map(|k| (stm_buf[k] as usize, nstm_buf[k] as usize))
                    .collect();
                assert_eq!(
                    direct,
                    effect_bucket_pairs,
                    "{} record {i}",
                    fs.canonical_name()
                );
            }
        }
    }

    fn startpos_for_overflow() -> ShogiBoard {
        use shogi_format::types::{Piece, PieceType};
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }
        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop);
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop);
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook);
        for file in [0u8, 8] {
            board.board[Square::new(file, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
            board.board[Square::new(file, 0).index()] = Piece::new(Color::White, PieceType::Lance);
        }
        for file in [1u8, 7] {
            board.board[Square::new(file, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
            board.board[Square::new(file, 0).index()] = Piece::new(Color::White, PieceType::Knight);
        }
        for file in [2u8, 6] {
            board.board[Square::new(file, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
            board.board[Square::new(file, 0).index()] = Piece::new(Color::White, PieceType::Silver);
        }
        for file in [3u8, 5] {
            board.board[Square::new(file, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
            board.board[Square::new(file, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        }
        board
    }
}
