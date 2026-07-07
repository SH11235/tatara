//! 対称 threat pair の静的分類器。
//!
//! Threat 特徴量は各 directed edge `attacker → target` を 1 feature として emit
//! する。ある edge の逆向き edge (`target → attacker`) が「この edge が active な
//! 全局面で必ず active」なら、片側は情報を捨てずに落とせる (mutual-implication
//! による削減)。本 module はその判定を占有非依存の純関数として与える:
//!
//! - [`is_necessarily_mutual`]: 逆向き edge が常に共 active か。
//! - [`is_canonical_dead`]: mutual pair のうち視点非依存の canonical 規則で
//!   「落とす側」か。
//!
//! [`for_each_active_threat_edge`] は 1 局面の active directed edge を raw (視点
//! 非依存) な [`RawThreatEdge`] として列挙する ([`crate::threat::ThreatIndexer`]
//! の index 列挙と同じ駒ループ・利き列挙を使うが、index / profile 除外・HM
//! 正規化は挟まない)。

use shogi_format::ShogiBoard;
use shogi_format::types::{Color, Square};

use crate::threat::{Occupied, ThreatClass, empty_board_attacks, for_each_attack};

/// 1 つの active threat edge を raw (視点非依存) 座標・色で表す。
///
/// index 空間の正規化 (perspective swap / HM mirror) を掛ける前の生データ。
/// canonical 分類が両視点で一致するために、判定はこの raw 表現の上で行う。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawThreatEdge {
    /// 攻め手の class。
    pub attacker_class: ThreatClass,
    /// 攻め手の実際の色 (盤上の所有者)。
    pub attacker_color: Color,
    /// 攻め手のマス (raw)。
    pub from_sq: Square,
    /// 攻められている駒の class。
    pub attacked_class: ThreatClass,
    /// 攻められている駒の実際の色 (盤上の所有者)。
    pub target_color: Color,
    /// 攻められている駒のマス (raw)。
    pub to_sq: Square,
}

/// この edge の逆向き edge (`target → attacker`) が、この edge が active な全局面で
/// 必ず active か。
///
/// active な edge は from→to の直線区間が空であることを含意する (slider の
/// attacker は最初の遮蔽駒 = `to_sq` で止まり、step 駒は隣接する)。逆向き edge は
/// `attacked_class` の駒が `to_sq` から `from_sq` を攻撃するもので、同じ区間が空
/// なので「実盤面で届く」⇔「空盤面で届く」(slider は `from_sq` を最初の遮蔽駒と
/// して掴み、step 駒の到達は占有非依存)。方向性駒 (歩・香・桂・銀・GoldLike) は
/// 色で利き向きが変わるため、逆向き ray の向きは target 自身の色で決める。攻め手
/// 側の class / 色 / マスは判定に不要 (逆向きの到達可否のみで決まる)。
pub fn is_necessarily_mutual(edge: &RawThreatEdge) -> bool {
    empty_board_attacks(
        edge.attacked_class,
        edge.target_color,
        edge.to_sq,
        edge.from_sq,
    )
}

/// この edge が necessarily-mutual pair の「落とす側」(canonical dead) か。
///
/// tie-break は **perspective swap と HM (Half-Mirror) file 反転のどちらでも不変な
/// 量**でなければならない (trainer は両視点 + HM 正規化後の index を突き合わせる)。
/// `ThreatClass` discriminant の大小は駒種の順序でミラー・回転・視点反転いずれにも
/// 不変なので、class が違う pair は大きい class を attacker とする側を残し、小さい側
/// (`ac < dc`) を dead にする。
///
/// 同 class pair は class では区別できず raw マス番号でしか順序付けできないが、raw
/// file 順は HM ミラーで反転する。ミラー等価な 2 局面で「残す側」が逆転し、正規化
/// index 空間で HM 等価性 (同一 index 集合) と dead 保証 (dead の相方が必ず active)
/// が壊れるため、同 class 相互 pair は dedup せず両側を emit する。
pub fn is_canonical_dead(edge: &RawThreatEdge) -> bool {
    is_necessarily_mutual(edge) && (edge.attacker_class as u8) < (edge.attacked_class as u8)
}

/// 1 局面の active directed threat edge を raw 表現で 1 つずつ `f` へ渡す。
///
/// King / None を端点に持つ edge は threat に含めない ([`ThreatClass::from_piece_type`]
/// が None を返す)。index 列挙と違い profile 除外・HM 正規化・玉位置の妥当性検査は
/// 挟まない (edge の active 判定自体は玉位置に依らないため)。slider は [`Occupied`]
/// で遮蔽され、遮蔽マス自体は攻撃対象として emit される。
pub fn for_each_active_threat_edge<F: FnMut(&RawThreatEdge)>(board: &ShogiBoard, mut f: F) {
    let occ = Occupied::from_board(board);
    for sq_raw in 0..81u8 {
        let from_sq = Square(sq_raw);
        let attacker = board.piece_on(from_sq);
        if attacker.is_none() {
            continue;
        }
        let Some(attacker_class) = ThreatClass::from_piece_type(attacker.piece_type) else {
            continue;
        };
        let attacker_color = attacker.color;
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
                let Some(attacked_class) = ThreatClass::from_piece_type(target.piece_type) else {
                    return;
                };
                f(&RawThreatEdge {
                    attacker_class,
                    attacker_color,
                    from_sq,
                    attacked_class,
                    target_color: target.color,
                    to_sq,
                });
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::types::{Piece, PieceType};

    /// 玉だけ置いた空盤面 (edge 列挙は玉位置に依らないので分類の単体検証に十分)。
    fn bare_board() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        board
    }

    fn put(board: &mut ShogiBoard, file: u8, rank: u8, color: Color, pt: PieceType) -> Square {
        let sq = Square::new(file, rank);
        board.board[sq.index()] = Piece::new(color, pt);
        sq
    }

    /// 局面の active directed edge を `(from, to)` 集合として集める。
    fn active_edge_set(board: &ShogiBoard) -> std::collections::HashSet<(u8, u8)> {
        let mut set = std::collections::HashSet::new();
        for_each_active_threat_edge(board, |e| {
            set.insert((e.from_sq.0, e.to_sq.0));
        });
        set
    }

    /// dead 判定された edge は逆向き edge が同一局面で必ず active。
    fn assert_dead_implies_reverse_active(board: &ShogiBoard) {
        let active = active_edge_set(board);
        for_each_active_threat_edge(board, |e| {
            if is_canonical_dead(e) {
                assert!(
                    active.contains(&(e.to_sq.0, e.from_sq.0)),
                    "dead edge {:?}@{}→{:?}@{} lacks active reverse",
                    e.attacker_class,
                    e.from_sq.0,
                    e.attacked_class,
                    e.to_sq.0,
                );
            }
        });
    }

    /// mutual pair は必ずちょうど片方だけが dead (両 dead / 両生存を禁止)。
    fn assert_exactly_one_dead_per_pair(board: &ShogiBoard) {
        let mut dead = std::collections::HashSet::new();
        for_each_active_threat_edge(board, |e| {
            if is_canonical_dead(e) {
                dead.insert((e.from_sq.0, e.to_sq.0));
            }
        });
        for &(from, to) in &dead {
            assert!(
                !dead.contains(&(to, from)),
                "both directions of pair ({from},{to}) marked dead",
            );
        }
    }

    #[test]
    fn same_class_pair_is_mutual_but_never_dead() {
        // 飛↔飛 同士 (同筋)。必ず相互だが同 class なので dedup しない (HM ミラーで
        // raw file 順が反転し「残す側」が正規化空間で一意に定まらないため両側 emit)。
        let mut board = bare_board();
        let lo = put(&mut board, 4, 4, Color::Black, PieceType::Rook);
        let hi = put(&mut board, 4, 6, Color::White, PieceType::Rook);
        let edge_lo_hi = RawThreatEdge {
            attacker_class: ThreatClass::Rook,
            attacker_color: Color::Black,
            from_sq: lo,
            attacked_class: ThreatClass::Rook,
            target_color: Color::White,
            to_sq: hi,
        };
        let edge_hi_lo = RawThreatEdge {
            attacker_class: ThreatClass::Rook,
            attacker_color: Color::White,
            from_sq: hi,
            attacked_class: ThreatClass::Rook,
            target_color: Color::Black,
            to_sq: lo,
        };
        assert!(is_necessarily_mutual(&edge_lo_hi));
        assert!(is_necessarily_mutual(&edge_hi_lo));
        assert!(!is_canonical_dead(&edge_lo_hi));
        assert!(!is_canonical_dead(&edge_hi_lo));
        assert_dead_implies_reverse_active(&board);
        assert_exactly_one_dead_per_pair(&board);
    }

    #[test]
    fn bishop_horse_superset_dead_is_bishop() {
        // 角→馬 は非方向 superset 内包で必ず相互。角 (小 class) を落とす。
        let mut board = bare_board();
        put(&mut board, 2, 4, Color::Black, PieceType::Bishop);
        put(&mut board, 5, 7, Color::White, PieceType::Horse);
        let mut bishop_dead = false;
        let mut horse_dead = false;
        for_each_active_threat_edge(&board, |e| {
            if e.attacker_class == ThreatClass::Bishop
                && e.attacked_class == ThreatClass::Horse
                && is_canonical_dead(e)
            {
                bishop_dead = true;
            }
            if e.attacker_class == ThreatClass::Horse
                && e.attacked_class == ThreatClass::Bishop
                && is_canonical_dead(e)
            {
                horse_dead = true;
            }
        });
        assert!(bishop_dead, "bishop→horse should be dead");
        assert!(!horse_dead, "horse→bishop should be kept");
        assert_dead_implies_reverse_active(&board);
    }

    #[test]
    fn distant_bishop_dragon_not_mutual() {
        // 角↔竜 は隣接時のみ相互。距離のある斜めでは竜が角に届かず non-mutual。
        let mut board = bare_board();
        let bishop = put(&mut board, 1, 1, Color::Black, PieceType::Bishop);
        let dragon = put(&mut board, 4, 4, Color::White, PieceType::Dragon);
        let edge = RawThreatEdge {
            attacker_class: ThreatClass::Bishop,
            attacker_color: Color::Black,
            from_sq: bishop,
            attacked_class: ThreatClass::Dragon,
            target_color: Color::White,
            to_sq: dragon,
        };
        assert!(!is_necessarily_mutual(&edge));
        assert!(!is_canonical_dead(&edge));
    }

    #[test]
    fn adjacent_bishop_dragon_is_mutual() {
        // 隣接斜めなら竜は king-step で角に届くので相互。
        let mut board = bare_board();
        let bishop = put(&mut board, 3, 3, Color::Black, PieceType::Bishop);
        let dragon = put(&mut board, 4, 4, Color::White, PieceType::Dragon);
        let edge = RawThreatEdge {
            attacker_class: ThreatClass::Bishop,
            attacker_color: Color::Black,
            from_sq: bishop,
            attacked_class: ThreatClass::Dragon,
            target_color: Color::White,
            to_sq: dragon,
        };
        assert!(is_necessarily_mutual(&edge));
        assert!(is_canonical_dead(&edge)); // Bishop(5) < Dragon(8)
        assert_dead_implies_reverse_active(&board);
    }

    #[test]
    fn opposing_pawns_mutual_but_not_dead() {
        // 対面する対色の歩は相互だが同 class なので dedup しない (両側 emit)。
        let mut board = bare_board();
        put(&mut board, 4, 5, Color::Black, PieceType::Pawn);
        put(&mut board, 4, 4, Color::White, PieceType::Pawn);
        assert_dead_implies_reverse_active(&board);
        assert_exactly_one_dead_per_pair(&board);
        let mut dead_count = 0;
        for_each_active_threat_edge(&board, |e| {
            if is_canonical_dead(e) {
                dead_count += 1;
            }
        });
        assert_eq!(dead_count, 0, "same-class pair must not be deduped");
    }

    #[test]
    fn same_color_silver_diagonal_mutual_straight_not() {
        // 同色 銀↔銀: 斜め隣接は相互、直進 (縦) は非相互。
        let mut board = bare_board();
        let center = put(&mut board, 4, 4, Color::Black, PieceType::Silver);
        let diag = put(&mut board, 3, 3, Color::Black, PieceType::Silver);
        let straight_from = put(&mut board, 6, 5, Color::Black, PieceType::Silver);
        let straight_to = put(&mut board, 6, 4, Color::Black, PieceType::Silver);
        // 斜め: center(4,4) と diag(3,3) は相互 (両方向 active)。
        let edge_diag = RawThreatEdge {
            attacker_class: ThreatClass::Silver,
            attacker_color: Color::Black,
            from_sq: diag,
            attacked_class: ThreatClass::Silver,
            target_color: Color::Black,
            to_sq: center,
        };
        assert!(is_necessarily_mutual(&edge_diag));
        // 直進: 先手銀 straight_from(6,5) が前方 straight_to(6,4) を攻撃するが、
        // straight_to の銀は前方 = さらに rank 減少方向を攻撃し straight_from へ
        // 戻らないので非相互。
        let edge_straight = RawThreatEdge {
            attacker_class: ThreatClass::Silver,
            attacker_color: Color::Black,
            from_sq: straight_from,
            attacked_class: ThreatClass::Silver,
            target_color: Color::Black,
            to_sq: straight_to,
        };
        assert!(!is_necessarily_mutual(&edge_straight));
        assert_dead_implies_reverse_active(&board);
        assert_exactly_one_dead_per_pair(&board);
    }

    #[test]
    fn bishop_rook_never_mutual() {
        // 角↔飛 は利きが直交するので非対称 (どちらの向きも非相互)。
        let mut board = bare_board();
        let bishop = put(&mut board, 2, 2, Color::Black, PieceType::Bishop);
        let rook = put(&mut board, 2, 5, Color::White, PieceType::Rook);
        // 角は斜めのみ。(2,2)→(2,5) は同筋なので角はそもそも攻撃しない → edge 無し。
        // 飛→角方向のみ active。飛→角が mutual でないことを確認。
        let edge_rook_bishop = RawThreatEdge {
            attacker_class: ThreatClass::Rook,
            attacker_color: Color::White,
            from_sq: rook,
            attacked_class: ThreatClass::Bishop,
            target_color: Color::Black,
            to_sq: bishop,
        };
        assert!(!is_necessarily_mutual(&edge_rook_bishop));
        assert!(!is_canonical_dead(&edge_rook_bishop));
        assert_dead_implies_reverse_active(&board);
    }
}
