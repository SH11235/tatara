//! effect bucket active index emission.
//!
//! effect bucket extends the base HalfKA_hm merged index with a per-piece attack
//! bucket: `effect_bucket_index = base_index * NB + bucket`.

use shogi_format::types::{Color, HAND_PIECE_TYPES, PieceType, Square};
use shogi_format::{BonaPiece, ShogiBoard};

use crate::halfka_hm::{
    MAX_ACTIVE_FEATURES, PIECE_INPUTS, halfka_index, is_hm_mirror, king_bonapiece, king_bucket,
    pack_bonapiece,
};

/// effect bucket configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectBucketConfig {
    /// Number of buckets. Supported values are 4 and 9.
    pub nb: usize,
    /// Whether king features receive effect buckets.
    pub king_bucketed: bool,
}

impl EffectBucketConfig {
    /// 2x2 attack bucket, kings fixed to bucket 0.
    pub const KINGFIXED_2X2: Self = Self {
        nb: 4,
        king_bucketed: false,
    };
    /// 2x2 attack bucket, kings bucketed.
    pub const KINGBUCKETED_2X2: Self = Self {
        nb: 4,
        king_bucketed: true,
    };
    /// 3x3 attack bucket, kings fixed to bucket 0.
    pub const KINGFIXED_3X3: Self = Self {
        nb: 9,
        king_bucketed: false,
    };
    /// 3x3 attack bucket, kings bucketed.
    pub const KINGBUCKETED_3X3: Self = Self {
        nb: 9,
        king_bucketed: true,
    };

    /// effect bucket input dimension.
    pub const fn dimensions(self) -> usize {
        45 * PIECE_INPUTS * self.nb
    }
}

/// Per-color attacker counts for all board squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectBucketAttackCounts {
    counts: [[u8; Square::NONE.0 as usize]; 2],
}

impl EffectBucketAttackCounts {
    /// Returns the number of pieces of `color` attacking `sq`.
    #[inline]
    pub fn get(self, color: Color, sq: Square) -> u8 {
        self.counts[color as usize][sq.index()]
    }
}

const PACKED_HAND_END: usize = 90;
const PACKED_BOARD_END: usize = 1548;

/// Returns whether a packed BonaPiece receives an effect bucket.
#[inline]
pub fn packed_is_bucketed(packed_bp: usize, king_bucketed: bool) -> bool {
    if packed_bp < PACKED_HAND_END {
        false
    } else if packed_bp < PACKED_BOARD_END {
        true
    } else {
        king_bucketed
    }
}

/// Quantizes attacked/defended counts into an effect bucket.
#[inline]
pub fn effect_bucket(attacked: u8, defended: u8, nb: usize) -> usize {
    match nb {
        4 => defended.min(1) as usize * 2 + attacked.min(1) as usize,
        9 => defended.min(2) as usize * 3 + attacked.min(2) as usize,
        _ => unreachable!("unsupported effect bucket count: {nb}"),
    }
}

/// Combines a base HalfKA_hm index and an effect bucket.
#[inline]
pub fn effect_bucket_index(base_index: usize, bucket: usize, nb: usize) -> usize {
    base_index * nb + bucket
}

/// Builds per-square attacker counts from the raw board.
///
/// Counts include king attacks, ignore pins, exclude the piece's own square, and
/// stop sliders at the first occupied square while still counting that square.
pub fn effect_bucket_attacker_counts(board: &ShogiBoard) -> EffectBucketAttackCounts {
    let occ = Occupied::from_board(board);
    let mut counts = [[0u8; Square::NONE.0 as usize]; 2];

    for sq_raw in 0..Square::NONE.0 {
        let from = Square(sq_raw);
        let piece = board.piece_on(from);
        if piece.is_none() {
            continue;
        }
        for_each_attack_with_king(piece.piece_type, piece.color, from, &occ, |to| {
            let slot = &mut counts[piece.color as usize][to.index()];
            *slot = slot.saturating_add(1);
        });
    }

    EffectBucketAttackCounts { counts }
}

/// Emits effect bucket active indices for one physical perspective.
///
/// `perspective` is independent from side-to-move; pass `Color::Black` for the
/// black view and `Color::White` for the white view.
pub fn map_effect_bucket_features_board<F: FnMut(usize)>(
    board: &ShogiBoard,
    config: EffectBucketConfig,
    perspective: Color,
    mut f: F,
) {
    let king_sq = board.king_square(perspective);
    let enemy_king_sq = board.king_square(perspective.opponent());
    if !king_sq.is_valid() || !enemy_king_sq.is_valid() {
        return;
    }

    let ctx = EffectBucketPerspectiveCtx {
        king_bucket: king_bucket(king_sq, perspective),
        hm_mirror: is_hm_mirror(king_sq, perspective),
    };
    let counts = effect_bucket_attacker_counts(board);

    board.for_each_board_piece(|piece, sq| {
        let bp = BonaPiece::from_piece_square(piece, sq, perspective);
        emit_effect_bucket_index(&ctx, &counts, config, bp, Some((piece.color, sq)), &mut f);
    });

    let friend_king = king_bonapiece(perspective_sq_index(king_sq, perspective), true);
    emit_effect_bucket_index(
        &ctx,
        &counts,
        config,
        friend_king,
        Some((perspective, king_sq)),
        &mut f,
    );

    let enemy_king = king_bonapiece(perspective_sq_index(enemy_king_sq, perspective), false);
    emit_effect_bucket_index(
        &ctx,
        &counts,
        config,
        enemy_king,
        Some((perspective.opponent(), enemy_king_sq)),
        &mut f,
    );

    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            for i in 1..=board.hand(owner).count(pt) {
                let bp = BonaPiece::from_hand_piece(perspective, owner, pt, i);
                if bp != BonaPiece::ZERO {
                    emit_effect_bucket_index(&ctx, &counts, config, bp, None, &mut f);
                }
            }
        }
    }
}

/// Collects effect bucket active indices for one perspective.
pub fn collect_effect_bucket_features_board(
    board: &ShogiBoard,
    config: EffectBucketConfig,
    perspective: Color,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(MAX_ACTIVE_FEATURES);
    map_effect_bucket_features_board(board, config, perspective, |idx| out.push(idx));
    out
}

struct EffectBucketPerspectiveCtx {
    king_bucket: usize,
    hm_mirror: bool,
}

#[inline]
fn perspective_sq_index(sq: Square, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq.index(),
        Color::White => sq.inverse().index(),
    }
}

fn emit_effect_bucket_index<F: FnMut(usize)>(
    ctx: &EffectBucketPerspectiveCtx,
    counts: &EffectBucketAttackCounts,
    config: EffectBucketConfig,
    bp: BonaPiece,
    board_piece: Option<(Color, Square)>,
    f: &mut F,
) {
    let packed = pack_bonapiece(bp, ctx.hm_mirror);
    let base = halfka_index(ctx.king_bucket, packed);
    let bucket = if packed_is_bucketed(packed, config.king_bucketed) {
        let (color, sq) =
            board_piece.expect("bucketed effect bucket feature must have a board square");
        effect_bucket(
            counts.get(color.opponent(), sq),
            counts.get(color, sq),
            config.nb,
        )
    } else {
        0
    };
    f(effect_bucket_index(base, bucket, config.nb));
}

struct Occupied {
    bits: [u64; 2],
}

impl Occupied {
    fn from_board(board: &ShogiBoard) -> Self {
        let mut bits = [0u64; 2];
        for sq in 0..Square::NONE.0 {
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

    #[inline]
    fn is_occupied(&self, sq: Square) -> bool {
        if sq.0 < 64 {
            (self.bits[0] >> sq.0) & 1 != 0
        } else {
            (self.bits[1] >> (sq.0 - 64)) & 1 != 0
        }
    }
}

fn for_each_attack_with_king<F: FnMut(Square)>(
    pt: PieceType,
    color: Color,
    from: Square,
    occ: &Occupied,
    mut emit: F,
) {
    let file = from.file() as i8;
    let rank = from.rank() as i8;

    let ray = |df: i8, dr: i8, emit: &mut F| {
        let mut f = file + df;
        let mut r = rank + dr;
        while (0..9).contains(&f) && (0..9).contains(&r) {
            let sq = Square::new(f as u8, r as u8);
            emit(sq);
            if occ.is_occupied(sq) {
                break;
            }
            f += df;
            r += dr;
        }
    };
    let step = |df: i8, dr: i8, emit: &mut F| {
        let f = file + df;
        let r = rank + dr;
        if (0..9).contains(&f) && (0..9).contains(&r) {
            emit(Square::new(f as u8, r as u8));
        }
    };

    let forward = if color == Color::Black { -1 } else { 1 };
    match pt {
        PieceType::Pawn => step(0, forward, &mut emit),
        PieceType::Lance => ray(0, forward, &mut emit),
        PieceType::Knight => {
            step(-1, 2 * forward, &mut emit);
            step(1, 2 * forward, &mut emit);
        }
        PieceType::Silver => {
            for (df, dr) in [
                (-1, forward),
                (0, forward),
                (1, forward),
                (-1, -forward),
                (1, -forward),
            ] {
                step(df, dr, &mut emit);
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
                step(df, dr, &mut emit);
            }
        }
        PieceType::Bishop => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                ray(df, dr, &mut emit);
            }
        }
        PieceType::Rook => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                ray(df, dr, &mut emit);
            }
        }
        PieceType::Horse => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                ray(df, dr, &mut emit);
            }
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                step(df, dr, &mut emit);
            }
        }
        PieceType::Dragon => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                ray(df, dr, &mut emit);
            }
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                step(df, dr, &mut emit);
            }
        }
        PieceType::King => {
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df != 0 || dr != 0 {
                        step(df, dr, &mut emit);
                    }
                }
            }
        }
        PieceType::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeatureSet;
    use shogi_format::types::Piece;

    fn board_with_kings(
        black_king: Square,
        white_king: Square,
        pieces: &[(Color, PieceType, Square)],
    ) -> ShogiBoard {
        let mut board = ShogiBoard {
            black_king_sq: black_king,
            white_king_sq: white_king,
            ..Default::default()
        };
        board.board[black_king.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[white_king.index()] = Piece::new(Color::White, PieceType::King);
        for &(color, pt, sq) in pieces {
            board.board[sq.index()] = Piece::new(color, pt);
        }
        board
    }

    #[test]
    fn dimensions_match_config() {
        assert_eq!(EffectBucketConfig::KINGFIXED_2X2.dimensions(), 73_305 * 4);
        assert_eq!(EffectBucketConfig::KINGFIXED_3X3.dimensions(), 73_305 * 9);
    }

    #[test]
    fn bucketed_predicate_domains() {
        assert!(!packed_is_bucketed(0, true));
        assert!(!packed_is_bucketed(89, true));
        assert!(packed_is_bucketed(90, false));
        assert!(packed_is_bucketed(1547, false));
        assert!(!packed_is_bucketed(1548, false));
        assert!(packed_is_bucketed(1548, true));
    }

    #[test]
    fn bucket_quantization() {
        assert_eq!(effect_bucket(0, 0, 4), 0);
        assert_eq!(effect_bucket(1, 0, 4), 1);
        assert_eq!(effect_bucket(0, 1, 4), 2);
        assert_eq!(effect_bucket(4, 3, 4), 3);
        assert_eq!(effect_bucket(2, 0, 9), 2);
        assert_eq!(effect_bucket(0, 2, 9), 6);
        assert_eq!(effect_bucket(4, 3, 9), 8);
    }

    #[test]
    fn effect_bucket_active_divides_back_to_base_set() {
        let mut board = board_with_kings(
            Square::new(4, 8),
            Square::new(4, 0),
            &[
                (Color::Black, PieceType::Pawn, Square::new(4, 6)),
                (Color::White, PieceType::Pawn, Square::new(4, 2)),
                (Color::Black, PieceType::Rook, Square::new(1, 7)),
                (Color::White, PieceType::Bishop, Square::new(7, 1)),
            ],
        );
        board.black_hand.add(PieceType::Gold, 1);

        for config in [
            EffectBucketConfig::KINGFIXED_2X2,
            EffectBucketConfig::KINGBUCKETED_2X2,
            EffectBucketConfig::KINGFIXED_3X3,
            EffectBucketConfig::KINGBUCKETED_3X3,
        ] {
            for perspective in [Color::Black, Color::White] {
                board.side_to_move = perspective;
                let mut base = Vec::new();
                FeatureSet::HalfKaHmMerged
                    .spec()
                    .map_features_board(&board, |stm, _| base.push(stm));
                base.sort_unstable();

                let mut effect_bucket =
                    collect_effect_bucket_features_board(&board, config, perspective);
                effect_bucket.sort_unstable();
                let mut recovered: Vec<_> =
                    effect_bucket.iter().map(|idx| idx / config.nb).collect();
                recovered.sort_unstable();
                assert_eq!(recovered, base);
            }
        }
    }

    #[test]
    fn king_attacks_are_counted_for_adjacent_piece_bucket() {
        let board = board_with_kings(
            Square::new(4, 8),
            Square::new(4, 0),
            &[(Color::Black, PieceType::Gold, Square::new(4, 7))],
        );
        let counts = effect_bucket_attacker_counts(&board);
        let gold_sq = Square::new(4, 7);
        assert_eq!(counts.get(Color::Black, gold_sq), 1);

        let mut indices = collect_effect_bucket_features_board(
            &board,
            EffectBucketConfig::KINGFIXED_2X2,
            Color::Black,
        );
        indices.sort_unstable();
        assert!(
            indices.iter().any(|idx| idx % 4 == 2 || idx % 4 == 3),
            "black king defense must affect at least one adjacent bucketed feature"
        );
    }
}
