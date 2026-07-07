//! Threat sparse 特徴量の index 算出。
//!
//! Threat 特徴量は「ある駒が別の駒に利いている」関係を
//! `attacker_class × attacked_class × 盤上関係` で表す sparse 特徴量。利き計算に
//! Bitboard / movegen は使わず、`board: [Piece; 81]` 上の座標 ray-walk で攻撃先を
//! 列挙する (slider は 2×u64 の [`Occupied`] で遮蔽を判定して break)。
//!
//! index の構成:
//!
//! ```text
//! threat_index =
//!     pair_base[attacker_side][attacker_class][attacked_side][attacked_class]
//!   + from_offset[attack_pattern][from_sq_n]
//!   + attack_order[attack_pattern][from_sq_n][to_sq_n]
//! ```
//!
//! `from_offset` / `attack_order` は空盤面利きから事前計算する index 算出用の
//! table であり movegen ではない。`pair_base` は [`ThreatProfile`] ごとに除外
//! pair を sentinel skip + prefix-sum で詰め直して構築する。マス座標は
//! perspective swap → HM mirror の順で正規化し、STM / NSTM 両視点で別 index を
//! 出す。式・除外規則・golden 値は bullet-shogi 正準ベクタの移植。

use std::sync::LazyLock;

use shogi_format::ShogiBoard;
use shogi_format::types::{Color, PieceType, Square};

use crate::halfka_hm::is_hm_mirror;
pub use crate::threat_exclusion::ThreatProfile;
use crate::threat_symmetric::{RawThreatEdge, is_canonical_dead};

/// ThreatClass の数 (King 除外、9 family)。
pub const NUM_THREAT_CLASSES: usize = 9;

/// Threat 特徴量の per-position active 数の上限。行幅 `base(40) + THREAT_MAX_ACTIVE`
/// を決める。DLSuisho 803M + aoba 103M 局面の Full profile 実測で total active の
/// max は 145 (base 40 + threat ≤ 105)、分布は +1.5〜2 特徴ごとに count が半減する
/// 急峻な指数減衰。128 は実測 max に対し margin 23 を持ち、到達確率は ~1e-13/pos
/// (100B 局面級でも安全)。万一超過しても dataloader は silent truncation せず
/// hard-error で停止する (この定数を上げて再ビルドを促す) ため破損は起きない。
pub const THREAT_MAX_ACTIVE: usize = 128;

/// profile から threat 次元数を返す。`ThreatIndexer::new(profile)` の
/// `threat_dimensions()` と同値だが pair_base table を組まずに済む const fn で、
/// `FeatureSetSpec::ft_in` 等の hot path から軽く呼べる。値は
/// `dimensions_match_indexer` test が indexer の計算結果と一致を固定する。
pub const fn threat_dimensions_of(profile: ThreatProfile) -> usize {
    match profile {
        ThreatProfile::Full => 216_720,
        ThreatProfile::SameClass => 192_640,
        ThreatProfile::SameClassMajorPawn => 173_568,
        ThreatProfile::StepAttacker => 33_408,
        // FullSymDedup は emit の対称重複除去のみで index 空間は Full と同一。
        ThreatProfile::FullSymDedup => 216_720,
        ThreatProfile::CrossSide => 96_320,
    }
}

// =============================================================================
// ThreatClass
// =============================================================================

/// Threat 駒種分類 (King 除外、9 family)。
///
/// discriminant 0-8 は index 算出に直接使うため固定する。Gold / 成駒 (と・成香・
/// 成桂・成銀) は利きが金と同じため [`ThreatClass::GoldLike`] に集約する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ThreatClass {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    GoldLike = 4,
    Bishop = 5,
    Rook = 6,
    Horse = 7,
    Dragon = 8,
}

impl ThreatClass {
    /// PieceType から ThreatClass への変換。King / None は threat に含まないので None。
    #[inline]
    pub fn from_piece_type(pt: PieceType) -> Option<Self> {
        match pt {
            PieceType::Pawn => Some(Self::Pawn),
            PieceType::Lance => Some(Self::Lance),
            PieceType::Knight => Some(Self::Knight),
            PieceType::Silver => Some(Self::Silver),
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => Some(Self::GoldLike),
            PieceType::Bishop => Some(Self::Bishop),
            PieceType::Rook => Some(Self::Rook),
            PieceType::Horse => Some(Self::Horse),
            PieceType::Dragon => Some(Self::Dragon),
            PieceType::King | PieceType::None => None,
        }
    }
}

/// 各 class の空盤面利き総数 (per color、全 81 マス合計)。`pair_base` の各 pair
/// が占める index 幅 = `ATTACKS_PER_COLOR[attacker_class]`。値は
/// `attacks_empty_board` の総和に一致する (test で検証)。
const ATTACKS_PER_COLOR: [usize; NUM_THREAT_CLASSES] = [
    72,   // Pawn
    324,  // Lance
    112,  // Knight
    328,  // Silver
    416,  // GoldLike
    816,  // Bishop
    1296, // Rook
    1104, // Horse
    1552, // Dragon
];

// =============================================================================
// pair_base テーブル
// =============================================================================

/// pair 軸の総数 = 2 (attacker side) × 9 (class) × 2 (attacked side) × 9 (class)。
const NUM_PAIRS: usize = 2 * NUM_THREAT_CLASSES * 2 * NUM_THREAT_CLASSES;

/// 除外された pair の sentinel 値。
const EXCLUDED_PAIR_BASE: usize = usize::MAX;

/// pair 軸を 1 次元 index に畳む。layout は
/// `attacker_side(2) × attacker_class(9) × attacked_side(2) × attacked_class(9)`
/// の row-major (attacker_side が最上位)。
#[inline]
const fn pair_index(attacker_side: usize, ac: usize, attacked_side: usize, dc: usize) -> usize {
    attacker_side * (NUM_THREAT_CLASSES * 2 * NUM_THREAT_CLASSES)
        + ac * (2 * NUM_THREAT_CLASSES)
        + attacked_side * NUM_THREAT_CLASSES
        + dc
}

/// 指定 profile の `pair_base` table と threat dims を構築する。除外 pair は
/// sentinel を入れて prefix-sum から飛ばすので、残った pair が `[0, dims)` を
/// 隙間なく埋める。
fn build_pair_base(profile: ThreatProfile) -> ([usize; NUM_PAIRS], usize) {
    let mut table = [EXCLUDED_PAIR_BASE; NUM_PAIRS];
    let mut cumulative = 0usize;
    for attacker_side in 0..2 {
        for (ac, &attacks) in ATTACKS_PER_COLOR.iter().enumerate() {
            for attacked_side in 0..2 {
                for dc in 0..NUM_THREAT_CLASSES {
                    let idx = pair_index(attacker_side, ac, attacked_side, dc);
                    if profile.is_excluded(attacker_side, ac, attacked_side, dc) {
                        table[idx] = EXCLUDED_PAIR_BASE;
                    } else {
                        table[idx] = cumulative;
                        cumulative += attacks;
                    }
                }
            }
        }
    }
    (table, cumulative)
}

/// `pair_base` から base index を引く。除外 pair は None。
#[inline]
fn lookup_pair_base(
    pair_base: &[usize; NUM_PAIRS],
    attacker_side: usize,
    ac: ThreatClass,
    attacked_side: usize,
    dc: ThreatClass,
) -> Option<usize> {
    let base = pair_base[pair_index(attacker_side, ac as usize, attacked_side, dc as usize)];
    if base == EXCLUDED_PAIR_BASE {
        None
    } else {
        Some(base)
    }
}

/// profile に残る pair ごとに `(attacker_side, attacker_class, attacked_side,
/// attacked_class, base, width)` を呼ぶ。`base` は threat block 内の先頭 feature
/// index、`width` はその pair が占める feature 数 (`ATTACKS_PER_COLOR[attacker_class]`)。
/// 除外 pair は飛ばす。
///
/// 盤面依存の [`ThreatIndexer::for_each_active_threat_index`] と異なり index 空間の
/// 静的構造だけを返すので、threat FT row を pair-class 単位で操作する外部処理 (重み
/// ablation 診断 / pair 別ノルム分解) が index→pair の逆引きなしに使える。
pub fn for_each_threat_pair_range<F>(profile: ThreatProfile, mut f: F)
where
    F: FnMut(usize, ThreatClass, usize, ThreatClass, usize, usize),
{
    let (table, _dims) = build_pair_base(profile);
    for attacker_side in 0..2 {
        for &ac in &ALL_CLASSES {
            for attacked_side in 0..2 {
                for &dc in &ALL_CLASSES {
                    if let Some(base) =
                        lookup_pair_base(&table, attacker_side, ac, attacked_side, dc)
                    {
                        f(
                            attacker_side,
                            ac,
                            attacked_side,
                            dc,
                            base,
                            ATTACKS_PER_COLOR[ac as usize],
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// attack pattern / 空盤面利き
// =============================================================================

/// attack pattern の総数 = 9 (Black 全 class) + 5 (White 方向性駒)。非方向性駒は
/// 色で利きが変わらないので Black の pattern を再利用する。
pub const NUM_ATTACK_PATTERNS: usize = 14;

/// 利きが向きに依存する駒か (歩・香・桂・銀・GoldLike)。これらは White 用に別
/// pattern を持つ。
#[inline]
pub fn is_directional(class: ThreatClass) -> bool {
    matches!(
        class,
        ThreatClass::Pawn
            | ThreatClass::Lance
            | ThreatClass::Knight
            | ThreatClass::Silver
            | ThreatClass::GoldLike
    )
}

/// attack pattern id を返す。方向性駒の White は `9..=13`、それ以外は class
/// discriminant `0..=8`。
#[inline]
pub fn attack_pattern_id(class: ThreatClass, oriented_color: Color) -> usize {
    if oriented_color == Color::White && is_directional(class) {
        NUM_THREAT_CLASSES + class as usize
    } else {
        class as usize
    }
}

/// 全 class の配列 (table 構築 / test 用)。
const ALL_CLASSES: [ThreatClass; NUM_THREAT_CLASSES] = [
    ThreatClass::Pawn,
    ThreatClass::Lance,
    ThreatClass::Knight,
    ThreatClass::Silver,
    ThreatClass::GoldLike,
    ThreatClass::Bishop,
    ThreatClass::Rook,
    ThreatClass::Horse,
    ThreatClass::Dragon,
];

/// 空盤面上で `from` の駒の攻撃先マスを raw 値昇順で返す。返り値は
/// `(マス配列, マス数)`。配列の上限 36 は単一駒の最大利き数 (龍/馬) を十分に
/// 上回る。
///
/// 空盤面なので slider は盤端まで伸び、遮蔽は無い。これは index table 用の
/// 事前計算であって実盤面の利きではない (実盤面は [`for_each_attack`] が
/// [`Occupied`] で遮蔽する)。
fn attacks_empty_board(class: ThreatClass, color: Color, from: Square) -> ([u8; 36], usize) {
    let mut targets = [0u8; 36];
    let mut count = 0;
    let mut push = |sq: Square| {
        targets[count] = sq.0;
        count += 1;
    };
    let pt = piece_type_for_walk(class);
    walk_attacks(pt, color, from, &mut push, |_| false);

    // raw 値昇順で挿入ソート (count は最大 ~20)。attack_order は raw 昇順 0-index。
    for i in 1..count {
        let key = targets[i];
        let mut j = i;
        while j > 0 && targets[j - 1] > key {
            targets[j] = targets[j - 1];
            j -= 1;
        }
        targets[j] = key;
    }

    (targets, count)
}

/// ThreatClass を ray-walk 用の代表 PieceType に戻す。GoldLike は Gold で代表する
/// (成金 5 種は同じ利き)。
#[inline]
fn piece_type_for_walk(class: ThreatClass) -> PieceType {
    match class {
        ThreatClass::Pawn => PieceType::Pawn,
        ThreatClass::Lance => PieceType::Lance,
        ThreatClass::Knight => PieceType::Knight,
        ThreatClass::Silver => PieceType::Silver,
        ThreatClass::GoldLike => PieceType::Gold,
        ThreatClass::Bishop => PieceType::Bishop,
        ThreatClass::Rook => PieceType::Rook,
        ThreatClass::Horse => PieceType::Horse,
        ThreatClass::Dragon => PieceType::Dragon,
    }
}

// =============================================================================
// 占有 bitset と ray-walk
// =============================================================================

/// 81 マスの占有 bitset。slider 遮蔽判定に使う。
pub struct Occupied {
    /// `bits[0]`: sq 0..=63、`bits[1]`: sq 64..=80。
    bits: [u64; 2],
}

impl Occupied {
    /// 盤面から占有 bitset を作る。
    pub fn from_board(board: &ShogiBoard) -> Self {
        let mut bits = [0u64; 2];
        for sq in 0..81u8 {
            if board.board[sq as usize].is_some() {
                if sq < 64 {
                    bits[0] |= 1u64 << sq;
                } else {
                    bits[1] |= 1u64 << (sq - 64);
                }
            }
        }
        Self { bits }
    }

    /// マス `sq` が占有されているか。
    #[inline]
    pub fn is_occupied(&self, sq: u8) -> bool {
        if sq < 64 {
            (self.bits[0] >> sq) & 1 != 0
        } else {
            (self.bits[1] >> (sq - 64)) & 1 != 0
        }
    }
}

/// `pt` の駒の攻撃先を座標 ray-walk で列挙し `emit` へ渡す。slider は各マスを
/// emit した後 `blocked(sq)` が true なら break する (遮蔽はマス自体を含む =
/// 占有駒も攻撃対象)。
fn walk_attacks<E: FnMut(Square), B: Fn(Square) -> bool>(
    pt: PieceType,
    color: Color,
    from: Square,
    emit: &mut E,
    blocked: B,
) {
    let file = from.file() as i8;
    let rank = from.rank() as i8;

    // slider: 各方向に占有駒で止まるまで伸ばす。
    let ray = |df: i8, dr: i8, emit: &mut E| {
        let mut f = file + df;
        let mut r = rank + dr;
        while (0..9).contains(&f) && (0..9).contains(&r) {
            let sq = Square::new(f as u8, r as u8);
            emit(sq);
            if blocked(sq) {
                break;
            }
            f += df;
            r += dr;
        }
    };
    // 単発: 盤内なら 1 マスだけ emit。
    let step = |df: i8, dr: i8, emit: &mut E| {
        let f = file + df;
        let r = rank + dr;
        if (0..9).contains(&f) && (0..9).contains(&r) {
            emit(Square::new(f as u8, r as u8));
        }
    };

    // 先手は前方 = rank 減少。後手は前方 = rank 増加。
    let forward: i8 = if color == Color::Black { -1 } else { 1 };

    match pt {
        PieceType::Pawn => step(0, forward, emit),
        PieceType::Lance => ray(0, forward, emit),
        PieceType::Knight => {
            step(-1, 2 * forward, emit);
            step(1, 2 * forward, emit);
        }
        PieceType::Silver => {
            for (df, dr) in [
                (-1, forward),
                (0, forward),
                (1, forward),
                (-1, -forward),
                (1, -forward),
            ] {
                step(df, dr, emit);
            }
        }
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => {
            for (df, dr) in [
                (-1, forward),
                (0, forward),
                (1, forward),
                (-1, 0),
                (1, 0),
                (0, -forward),
            ] {
                step(df, dr, emit);
            }
        }
        PieceType::Bishop => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                ray(df, dr, emit);
            }
        }
        PieceType::Rook => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                ray(df, dr, emit);
            }
        }
        PieceType::Horse => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                ray(df, dr, emit);
            }
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                step(df, dr, emit);
            }
        }
        PieceType::Dragon => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                ray(df, dr, emit);
            }
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                step(df, dr, emit);
            }
        }
        // King / None は threat に含まない。
        PieceType::King | PieceType::None => {}
    }
}

/// 実盤面上の攻撃先マスを列挙し `callback` を呼ぶ。slider は [`Occupied`] で
/// 遮蔽されるが、遮蔽マス自体は攻撃対象 (駒を取れる) なので emit してから break
/// する。
pub fn for_each_attack<F: FnMut(Square)>(
    pt: PieceType,
    color: Color,
    from: Square,
    occ: &Occupied,
    mut callback: F,
) {
    walk_attacks(pt, color, from, &mut callback, |sq| occ.is_occupied(sq.0));
}

// =============================================================================
// from_offset / attack_order table
// =============================================================================

/// `from_offset[attack_pattern][from_sq]` = その pattern で `from_sq` より前
/// (raw 昇順) の全マスの利き数の累積和。pair_base に加算する from ごとの offset。
pub struct FromOffsetTable {
    data: [[usize; 81]; NUM_ATTACK_PATTERNS],
}

impl FromOffsetTable {
    fn new() -> Self {
        let mut data = [[0usize; 81]; NUM_ATTACK_PATTERNS];
        for &class in &ALL_CLASSES {
            Self::fill(&mut data[class as usize], class, Color::Black);
            if is_directional(class) {
                Self::fill(
                    &mut data[NUM_THREAT_CLASSES + class as usize],
                    class,
                    Color::White,
                );
            }
        }
        Self { data }
    }

    fn fill(slots: &mut [usize; 81], class: ThreatClass, color: Color) {
        let mut cumulative = 0usize;
        for sq_raw in 0..81u8 {
            slots[sq_raw as usize] = cumulative;
            let (_, cnt) = attacks_empty_board(class, color, Square(sq_raw));
            cumulative += cnt;
        }
    }

    #[inline]
    fn get(&self, pattern: usize, from: Square) -> usize {
        self.data[pattern][from.index()]
    }
}

/// `attack_order[attack_pattern][from][to]` = 空盤面上で `from` の駒が `to` を
/// 攻撃する際の raw 昇順 0-index。攻撃しない (from, to) は [`Self::INVALID`]。
pub struct AttackOrderTable {
    data: Box<[[[u8; 81]; 81]; NUM_ATTACK_PATTERNS]>,
}

impl AttackOrderTable {
    /// 攻撃しない (from, to) を表す sentinel。
    pub const INVALID: u8 = u8::MAX;

    fn new() -> Self {
        let mut data = vec![[[Self::INVALID; 81]; 81]; NUM_ATTACK_PATTERNS];
        for &class in &ALL_CLASSES {
            Self::fill(&mut data[class as usize], class, Color::Black);
            if is_directional(class) {
                Self::fill(
                    &mut data[NUM_THREAT_CLASSES + class as usize],
                    class,
                    Color::White,
                );
            }
        }
        // box の内容は NUM_ATTACK_PATTERNS 個の固定長配列なので必ず変換できる。
        let data: Box<[[[u8; 81]; 81]; NUM_ATTACK_PATTERNS]> =
            data.into_boxed_slice().try_into().unwrap();
        Self { data }
    }

    fn fill(table: &mut [[u8; 81]; 81], class: ThreatClass, color: Color) {
        for from_raw in 0..81u8 {
            let (targets, count) = attacks_empty_board(class, color, Square(from_raw));
            for (order, &to_raw) in targets.iter().take(count).enumerate() {
                table[from_raw as usize][to_raw as usize] = order as u8;
            }
        }
    }

    #[inline]
    fn get(&self, pattern: usize, from: Square, to: Square) -> u8 {
        self.data[pattern][from.index()][to.index()]
    }
}

static FROM_OFFSET_TABLE: LazyLock<FromOffsetTable> = LazyLock::new(FromOffsetTable::new);
static ATTACK_ORDER_TABLE: LazyLock<AttackOrderTable> = LazyLock::new(AttackOrderTable::new);

/// 空盤面で `class`/`color` の駒が `from` から `to` を攻撃するか (raw 座標)。
/// index 算出用 [`AttackOrderTable`] の逆引き (O(1)、alloc なし) で、対称重複除去
/// 分類器 ([`crate::threat_symmetric`]) の necessarily-mutual 判定に使う。
#[inline]
pub(crate) fn empty_board_attacks(
    class: ThreatClass,
    color: Color,
    from: Square,
    to: Square,
) -> bool {
    let pattern = attack_pattern_id(class, color);
    ATTACK_ORDER_TABLE.get(pattern, from, to) != AttackOrderTable::INVALID
}

static SHARED_FULL: LazyLock<ThreatIndexer> =
    LazyLock::new(|| ThreatIndexer::new(ThreatProfile::Full));
static SHARED_SAME_CLASS: LazyLock<ThreatIndexer> =
    LazyLock::new(|| ThreatIndexer::new(ThreatProfile::SameClass));
static SHARED_SAME_CLASS_MAJOR_PAWN: LazyLock<ThreatIndexer> =
    LazyLock::new(|| ThreatIndexer::new(ThreatProfile::SameClassMajorPawn));
static SHARED_STEP_ATTACKER: LazyLock<ThreatIndexer> =
    LazyLock::new(|| ThreatIndexer::new(ThreatProfile::StepAttacker));
static SHARED_FULL_SYM_DEDUP: LazyLock<ThreatIndexer> =
    LazyLock::new(|| ThreatIndexer::new(ThreatProfile::FullSymDedup));
static SHARED_CROSS_SIDE: LazyLock<ThreatIndexer> =
    LazyLock::new(|| ThreatIndexer::new(ThreatProfile::CrossSide));

// =============================================================================
// マス正規化と index 計算
// =============================================================================

/// マスを perspective 基準 + HM mirror で正規化する。Black 視点はそのまま、White
/// 視点は 180 度回転 (`inverse`)、その後 `hm_mirror` が立っていれば file を反転
/// する。from / to に同一適用する。
#[inline]
pub fn normalize_sq(sq: Square, perspective: Color, hm_mirror: bool) -> Square {
    let sq_n = if perspective == Color::Black {
        sq
    } else {
        sq.inverse()
    };
    if hm_mirror { sq_n.mirror_file() } else { sq_n }
}

/// 単一 threat edge の index 計算パラメータ (正規化済みマスを渡す)。
struct ThreatParams {
    attacker_side: usize,
    attacker_class: ThreatClass,
    oriented_color: Color,
    attacked_side: usize,
    attacked_class: ThreatClass,
    from_sq_n: Square,
    to_sq_n: Square,
}

/// threat index を計算する。除外 pair は None。
#[inline]
fn threat_index(params: &ThreatParams, pair_base: &[usize; NUM_PAIRS]) -> Option<usize> {
    let base = lookup_pair_base(
        pair_base,
        params.attacker_side,
        params.attacker_class,
        params.attacked_side,
        params.attacked_class,
    )?;
    let pattern = attack_pattern_id(params.attacker_class, params.oriented_color);
    let from_off = FROM_OFFSET_TABLE.get(pattern, params.from_sq_n);
    let attack_ord = ATTACK_ORDER_TABLE.get(pattern, params.from_sq_n, params.to_sq_n);
    debug_assert_ne!(
        attack_ord,
        AttackOrderTable::INVALID,
        "to_sq {} not attacked by pattern {pattern} from {}",
        params.to_sq_n.0,
        params.from_sq_n.0,
    );
    Some(base + from_off + attack_ord as usize)
}

// =============================================================================
// ThreatProfile -> dims / 公開 API
// =============================================================================

/// 構築済みの threat indexer。`pair_base` table と dims を 1 度だけ計算して保持
/// する (`pair_base` は 324 entry × usize で軽量、box せず保持する)。
#[derive(Clone, Debug)]
pub struct ThreatIndexer {
    profile: ThreatProfile,
    pair_base: [usize; NUM_PAIRS],
    threat_dims: usize,
}

impl ThreatIndexer {
    /// 指定 profile で構築する。
    pub fn new(profile: ThreatProfile) -> Self {
        let (pair_base, threat_dims) = build_pair_base(profile);
        Self {
            profile,
            pair_base,
            threat_dims,
        }
    }

    /// profile ごとに 1 度だけ構築した共有 indexer を返す。`pair_base` table を
    /// position ごとに組み直すのを避けるため、dataloader hot path はこれを使う。
    pub fn shared(profile: ThreatProfile) -> &'static ThreatIndexer {
        match profile {
            ThreatProfile::Full => &SHARED_FULL,
            ThreatProfile::SameClass => &SHARED_SAME_CLASS,
            ThreatProfile::SameClassMajorPawn => &SHARED_SAME_CLASS_MAJOR_PAWN,
            ThreatProfile::StepAttacker => &SHARED_STEP_ATTACKER,
            ThreatProfile::FullSymDedup => &SHARED_FULL_SYM_DEDUP,
            ThreatProfile::CrossSide => &SHARED_CROSS_SIDE,
        }
    }

    /// profile を返す。
    pub fn profile(&self) -> ThreatProfile {
        self.profile
    }

    /// threat 特徴量の次元数。
    pub fn threat_dimensions(&self) -> usize {
        self.threat_dims
    }

    /// `perspective` 視点の active threat index を 1 つずつ `f` へ渡す。index は
    /// `[0, threat_dimensions())` の threat 空間内 (base feature の offset は
    /// caller が加える)。除外 pair の edge は呼ばない。玉位置が不正な局面は何も
    /// emit しない。
    pub fn for_each_active_threat_index<F: FnMut(usize)>(
        &self,
        board: &ShogiBoard,
        perspective: Color,
        mut f: F,
    ) {
        let friend = perspective;
        let king_sq = board.king_square(friend);
        let enemy_king_sq = board.king_square(friend.opponent());
        if !king_sq.is_valid() || !enemy_king_sq.is_valid() {
            return;
        }
        let hm = is_hm_mirror(king_sq, perspective);

        let occ = Occupied::from_board(board);

        for sq_raw in 0..81u8 {
            let from_sq = Square(sq_raw);
            let attacker = board.piece_on(from_sq);
            if attacker.is_none() {
                continue;
            }
            let attacker_color = attacker.color;
            let Some(attacker_class) = ThreatClass::from_piece_type(attacker.piece_type) else {
                continue; // King / None
            };

            let attacker_side = usize::from(attacker_color != friend);
            let from_n = normalize_sq(from_sq, perspective, hm);
            let oriented_color = if perspective == Color::Black {
                attacker_color
            } else {
                attacker_color.opponent()
            };

            for_each_attack(
                attacker.piece_type,
                attacker_color,
                from_sq,
                &occ,
                |to_sq| {
                    let target = board.piece_on(to_sq);
                    if target.is_none() {
                        return;
                    }
                    let Some(attacked_class) = ThreatClass::from_piece_type(target.piece_type)
                    else {
                        return; // King / None
                    };
                    if self.profile.drops_canonical_dead()
                        && is_canonical_dead(&RawThreatEdge {
                            attacker_class,
                            attacker_color,
                            from_sq,
                            attacked_class,
                            target_color: target.color,
                            to_sq,
                        })
                    {
                        return;
                    }
                    let attacked_side = usize::from(target.color != friend);
                    let to_n = normalize_sq(to_sq, perspective, hm);

                    let idx = threat_index(
                        &ThreatParams {
                            attacker_side,
                            attacker_class,
                            oriented_color,
                            attacked_side,
                            attacked_class,
                            from_sq_n: from_n,
                            to_sq_n: to_n,
                        },
                        &self.pair_base,
                    );
                    let Some(idx) = idx else {
                        return; // excluded pair
                    };
                    debug_assert!(
                        idx < self.threat_dims,
                        "threat index {idx} out of range {}",
                        self.threat_dims
                    );
                    f(idx);
                },
            );
        }
    }

    /// `perspective` 視点の active threat index を `Vec` で返す (test / 検証用)。
    pub fn active_threat_indices(&self, board: &ShogiBoard, perspective: Color) -> Vec<usize> {
        let mut out = Vec::new();
        self.for_each_active_threat_index(board, perspective, |idx| out.push(idx));
        out
    }

    /// 1 つの threat edge につき `(stm 視点 index, nstm 視点 index)` を pair で
    /// 渡す。base feature の 2 視点 emit と同じ列に詰めるための入口。pair の除外
    /// 判定 (`is_excluded`) は side / class 関係に依り perspective 反転で不変なので、
    /// edge は両視点で同時に active / 同時に excluded になる (片側だけ Some になる
    /// ことはない)。index は threat 空間 `[0, threat_dimensions())` (base offset は
    /// caller 加算)。玉位置が不正な局面は何も emit しない。
    pub fn for_each_active_threat_index_pair<F: FnMut(usize, usize)>(
        &self,
        board: &ShogiBoard,
        stm: Color,
        nstm: Color,
        mut f: F,
    ) {
        let stm_king = board.king_square(stm);
        let nstm_king = board.king_square(nstm);
        if !stm_king.is_valid() || !nstm_king.is_valid() {
            return;
        }
        let stm_hm = is_hm_mirror(stm_king, stm);
        let nstm_hm = is_hm_mirror(nstm_king, nstm);

        let occ = Occupied::from_board(board);

        for sq_raw in 0..81u8 {
            let from_sq = Square(sq_raw);
            let attacker = board.piece_on(from_sq);
            if attacker.is_none() {
                continue;
            }
            let attacker_color = attacker.color;
            let Some(attacker_class) = ThreatClass::from_piece_type(attacker.piece_type) else {
                continue; // King / None
            };

            for_each_attack(
                attacker.piece_type,
                attacker_color,
                from_sq,
                &occ,
                |to_sq| {
                    let target = board.piece_on(to_sq);
                    if target.is_none() {
                        return;
                    }
                    let Some(attacked_class) = ThreatClass::from_piece_type(target.piece_type)
                    else {
                        return; // King / None
                    };
                    let edge = RawThreatEdge {
                        attacker_class,
                        attacker_color,
                        from_sq,
                        attacked_class,
                        target_color: target.color,
                        to_sq,
                    };
                    // canonical-dead drop は raw 座標判定で perspective 非依存なので、
                    // 両視点を同時に落とし both-or-neither を保つ (下の debug_assert)。
                    if self.profile.drops_canonical_dead() && is_canonical_dead(&edge) {
                        return;
                    }
                    let stm_idx = self.edge_index(stm, stm_hm, &edge);
                    let nstm_idx = self.edge_index(nstm, nstm_hm, &edge);
                    // 除外判定 (`is_excluded`) は side / class 関係に依り perspective
                    // 反転で不変なので、edge は両視点で同時に active / 同時に excluded。
                    // 片側だけ Some になるのは side 非対称 profile を将来足した時の
                    // signal で、その時は本 pair 経路の silent drop を見直す必要がある。
                    debug_assert_eq!(
                        stm_idx.is_some(),
                        nstm_idx.is_some(),
                        "threat edge active on only one perspective (asymmetric exclusion?)"
                    );
                    if let (Some(s), Some(n)) = (stm_idx, nstm_idx) {
                        f(s, n);
                    }
                },
            );
        }
    }

    /// 1 edge を 1 視点から見た threat index を計算する。除外 pair は None。
    #[inline]
    fn edge_index(&self, perspective: Color, hm: bool, edge: &RawThreatEdge) -> Option<usize> {
        let friend = perspective;
        let attacker_side = usize::from(edge.attacker_color != friend);
        let attacked_side = usize::from(edge.target_color != friend);
        let oriented_color = if perspective == Color::Black {
            edge.attacker_color
        } else {
            edge.attacker_color.opponent()
        };
        threat_index(
            &ThreatParams {
                attacker_side,
                attacker_class: edge.attacker_class,
                oriented_color,
                attacked_side,
                attacked_class: edge.attacked_class,
                from_sq_n: normalize_sq(edge.from_sq, perspective, hm),
                to_sq_n: normalize_sq(edge.to_sq, perspective, hm),
            },
            &self.pair_base,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::types::Piece;

    // -- table / dims --------------------------------------------------------

    #[test]
    fn attacks_per_color_totals_match_constant() {
        for (i, &class) in ALL_CLASSES.iter().enumerate() {
            let total: usize = (0..81u8)
                .map(|sq| attacks_empty_board(class, Color::Black, Square(sq)).1)
                .sum();
            assert_eq!(
                total, ATTACKS_PER_COLOR[i],
                "{class:?}: expected {} got {total}",
                ATTACKS_PER_COLOR[i]
            );
        }
    }

    #[test]
    fn directional_white_attack_count_mirrors_black() {
        // 方向性駒は White 利き数の総和も Black と一致する (前方向きが反転するだけ)。
        for &class in &ALL_CLASSES {
            if !is_directional(class) {
                continue;
            }
            let black: usize = (0..81u8)
                .map(|s| attacks_empty_board(class, Color::Black, Square(s)).1)
                .sum();
            let white: usize = (0..81u8)
                .map(|s| attacks_empty_board(class, Color::White, Square(s)).1)
                .sum();
            assert_eq!(black, white, "{class:?}");
        }
    }

    #[test]
    fn profile_dimensions_match_canonical() {
        assert_eq!(
            ThreatIndexer::new(ThreatProfile::Full).threat_dimensions(),
            216_720
        );
        assert_eq!(
            ThreatIndexer::new(ThreatProfile::SameClass).threat_dimensions(),
            192_640
        );
        assert_eq!(
            ThreatIndexer::new(ThreatProfile::SameClassMajorPawn).threat_dimensions(),
            173_568
        );
        assert_eq!(
            ThreatIndexer::new(ThreatProfile::StepAttacker).threat_dimensions(),
            33_408
        );
        assert_eq!(
            ThreatIndexer::new(ThreatProfile::FullSymDedup).threat_dimensions(),
            216_720
        );
        assert_eq!(
            ThreatIndexer::new(ThreatProfile::CrossSide).threat_dimensions(),
            96_320
        );
    }

    #[test]
    fn const_dimensions_match_indexer() {
        for profile in [
            ThreatProfile::Full,
            ThreatProfile::SameClass,
            ThreatProfile::SameClassMajorPawn,
            ThreatProfile::StepAttacker,
            ThreatProfile::FullSymDedup,
            ThreatProfile::CrossSide,
        ] {
            assert_eq!(
                threat_dimensions_of(profile),
                ThreatIndexer::new(profile).threat_dimensions(),
                "{profile}"
            );
        }
    }

    #[test]
    fn from_offset_rook_is_uniform() {
        let pattern = attack_pattern_id(ThreatClass::Rook, Color::Black);
        // Rook は全マスで利き 16。
        for sq_raw in 0..81u8 {
            assert_eq!(
                FROM_OFFSET_TABLE.get(pattern, Square(sq_raw)),
                16 * sq_raw as usize
            );
        }
    }

    #[test]
    fn attack_order_invalid_for_non_attacked() {
        let pattern = attack_pattern_id(ThreatClass::Pawn, Color::Black);
        // 先手歩は前方 = rank 減少。rank=0 (1一) は盤端で前進できず利き 0。
        let from = Square::new(0, 0);
        for to_raw in 0..81u8 {
            assert_eq!(
                ATTACK_ORDER_TABLE.get(pattern, from, Square(to_raw)),
                AttackOrderTable::INVALID
            );
        }
    }

    // -- normalize -----------------------------------------------------------

    #[test]
    fn normalize_sq_matches_perspective_then_mirror() {
        let sq = Square::new(4, 4); // 5五
        assert_eq!(normalize_sq(sq, Color::Black, false), sq);
        assert_eq!(normalize_sq(sq, Color::Black, true), sq.mirror_file());
        assert_eq!(normalize_sq(sq, Color::White, false), sq.inverse());
        assert_eq!(
            normalize_sq(sq, Color::White, true),
            sq.inverse().mirror_file()
        );
    }

    // -- range validity ------------------------------------------------------

    /// startpos を手動構築する (両玉 5筋 = HM mirror 無し)。
    fn startpos_board() -> ShogiBoard {
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
        board.board[Square::new(0, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(8, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(0, 0).index()] = Piece::new(Color::White, PieceType::Lance);
        board.board[Square::new(8, 0).index()] = Piece::new(Color::White, PieceType::Lance);
        board.board[Square::new(1, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(7, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(1, 0).index()] = Piece::new(Color::White, PieceType::Knight);
        board.board[Square::new(7, 0).index()] = Piece::new(Color::White, PieceType::Knight);
        board.board[Square::new(2, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(6, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(2, 0).index()] = Piece::new(Color::White, PieceType::Silver);
        board.board[Square::new(6, 0).index()] = Piece::new(Color::White, PieceType::Silver);
        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        board
    }

    /// 玉が file >= 5 で HM mirror が効く局面 (両玉 9筋 = file 8)。
    fn mirrored_king_board() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(8, 8), // 9九
            white_king_sq: Square::new(8, 0), // 9一
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        // 成駒・slider・near-king edge を混ぜて class 網羅を上げる。
        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Dragon);
        board.board[Square::new(7, 6).index()] = Piece::new(Color::White, PieceType::ProPawn);
        board.board[Square::new(6, 7).index()] = Piece::new(Color::Black, PieceType::Horse);
        board.board[Square::new(6, 6).index()] = Piece::new(Color::White, PieceType::Silver);
        board.board[Square::new(8, 6).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(8, 4).index()] = Piece::new(Color::White, PieceType::Pawn);
        board
    }

    fn assert_indices_in_range(profile: ThreatProfile, board: &ShogiBoard) {
        let indexer = ThreatIndexer::new(profile);
        let dims = indexer.threat_dimensions();
        for perspective in [Color::Black, Color::White] {
            for idx in indexer.active_threat_indices(board, perspective) {
                assert!(idx < dims, "profile {profile} idx {idx} >= dims {dims}");
            }
        }
    }

    #[test]
    fn all_profiles_in_range_on_test_boards() {
        for profile in [
            ThreatProfile::Full,
            ThreatProfile::SameClass,
            ThreatProfile::SameClassMajorPawn,
            ThreatProfile::StepAttacker,
            ThreatProfile::FullSymDedup,
            ThreatProfile::CrossSide,
        ] {
            assert_indices_in_range(profile, &startpos_board());
            assert_indices_in_range(profile, &mirrored_king_board());
        }
    }

    /// FullSymDedup は Full の active index の真部分集合を emit し、落とした差分は
    /// ちょうど canonical-dead edge 数に一致する (both-or-neither は pair emit の
    /// debug_assert が担保、ここでは片視点の active 集合で検証)。
    #[test]
    fn full_symdedup_drops_exactly_canonical_dead() {
        use crate::threat_symmetric::{
            RawThreatEdge, for_each_active_threat_edge, is_canonical_dead,
        };

        let full = ThreatIndexer::new(ThreatProfile::Full);
        let dedup = ThreatIndexer::new(ThreatProfile::FullSymDedup);
        for board in [startpos_board(), mirrored_king_board()] {
            for perspective in [Color::Black, Color::White] {
                let full_set = full.active_threat_indices(&board, perspective);
                let dedup_set = dedup.active_threat_indices(&board, perspective);
                assert!(
                    dedup_set.len() <= full_set.len(),
                    "dedup must not add features"
                );
                // 落とした数 = この局面の canonical-dead edge 数。
                let mut dead = 0usize;
                for_each_active_threat_edge(&board, |e: &RawThreatEdge| {
                    if is_canonical_dead(e) {
                        dead += 1;
                    }
                });
                assert_eq!(
                    full_set.len() - dedup_set.len(),
                    dead,
                    "dropped feature count must equal canonical-dead edge count"
                );
            }
        }
    }

    /// FullSymDedup の pair emit は both-or-neither を保つ (dead は raw 判定で両視点
    /// 同時 drop なので `for_each_active_threat_index_pair` の debug_assert が発火
    /// しない)。debug build で panic しなければ pass。
    #[test]
    fn full_symdedup_pair_emit_both_or_neither() {
        let dedup = ThreatIndexer::new(ThreatProfile::FullSymDedup);
        for board in [startpos_board(), mirrored_king_board()] {
            let mut count = 0usize;
            dedup.for_each_active_threat_index_pair(&board, Color::Black, Color::White, |_, _| {
                count += 1;
            });
            // pair emit 数は片視点の active 数と一致する。
            let per_perspective = dedup.active_threat_indices(&board, Color::Black).len();
            assert_eq!(count, per_perspective);
        }
    }

    #[test]
    fn invalid_king_emits_nothing() {
        let board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::NONE,
            ..Default::default()
        };
        let indexer = ThreatIndexer::new(ThreatProfile::Full);
        assert!(
            indexer
                .active_threat_indices(&board, Color::Black)
                .is_empty()
        );
        assert!(
            indexer
                .active_threat_indices(&board, Color::White)
                .is_empty()
        );
    }

    // -- golden --------------------------------------------------------------

    /// bullet-shogi 正準ベクタ移植。startpos (full profile) の sorted threat index
    /// が 34 値と一致し、対称局面ゆえ Black / White 両視点で同一であること。
    #[test]
    fn canonical_startpos_threat_indices() {
        let board = startpos_board();
        let indexer = ThreatIndexer::new(ThreatProfile::Full);

        #[rustfmt::skip]
        let expected: Vec<usize> = vec![
            1330, 1618, 7147, 7148, 7231, 7232, 11047, 11213,
            16475, 16578, 23268, 23270, 24087, 25717, 37487, 40080,
            43974, 112573, 112861, 116503, 116504, 116587, 116588,
            122160, 122650, 128533, 128636, 138321, 138323, 139136,
            140770, 158280, 160871, 164753,
        ];

        let mut black = indexer.active_threat_indices(&board, Color::Black);
        black.sort_unstable();
        assert_eq!(black, expected, "Black perspective canonical mismatch");

        let mut white = indexer.active_threat_indices(&board, Color::White);
        white.sort_unstable();
        assert_eq!(
            white, expected,
            "White perspective canonical mismatch (symmetric pos)"
        );
    }

    /// FullSymDedup の startpos active index (両視点一致、対称局面)。canonical
    /// startpos (full) から canonical-dead 2 edge (11047 / 122160) を落とした 32 値。
    /// rshogi engine の同 profile 実装と index 集合が一致することの cross-repo
    /// アンカー (同一の期待値を両 repo の golden test に持たせる)。
    #[test]
    fn canonical_startpos_symdedup_indices() {
        let board = startpos_board();
        let indexer = ThreatIndexer::new(ThreatProfile::FullSymDedup);

        #[rustfmt::skip]
        let expected: Vec<usize> = vec![
            1330, 1618, 7147, 7148, 7231, 7232, 11213, 16475, 16578, 23268,
            23270, 24087, 25717, 37487, 40080, 43974, 112573, 112861, 116503,
            116504, 116587, 116588, 122650, 128533, 128636, 138321, 138323,
            139136, 140770, 158280, 160871, 164753,
        ];

        let mut black = indexer.active_threat_indices(&board, Color::Black);
        black.sort_unstable();
        assert_eq!(black, expected, "Black symdedup startpos mismatch");

        let mut white = indexer.active_threat_indices(&board, Color::White);
        white.sort_unstable();
        assert_eq!(
            white, expected,
            "White symdedup startpos mismatch (symmetric pos)"
        );
    }
}
