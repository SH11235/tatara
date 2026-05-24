//! board phase SIMD path 共通の precomputed table。

use shogi_format::bona_piece::PIECE_BASE;

/// `PIECE_BASE[pt][is_friend]` を `[i32; 30]` に flatten。gather lookup の
/// base ptr 用 (`pt * 2 + is_friend` を index に取る)。
pub(crate) const PIECE_BASE_FLAT: [i32; 30] = {
    let mut t = [0i32; 30];
    let mut pt = 0;
    while pt < 15 {
        t[pt * 2] = PIECE_BASE[pt][0] as i32;
        t[pt * 2 + 1] = PIECE_BASE[pt][1] as i32;
        pt += 1;
    }
    t
};

/// `sq → file-mirror した sq` の lookup (1筋 ↔ 9筋、段は不変)。
pub(crate) const MIRRORED_SQ: [i32; 81] = {
    let mut t = [0i32; 81];
    let mut i = 0;
    while i < 81 {
        let file = i / 9;
        let rank = i % 9;
        let mirrored_file = 8 - file;
        t[i] = (mirrored_file * 9 + rank) as i32;
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::bona_piece::{E_PAWN, F_DRAGON, F_PAWN};

    #[test]
    fn piece_base_flat_pawn_matches() {
        // PAWN = piece_type 1、index 3 (friend) / 2 (enemy)。
        assert_eq!(PIECE_BASE_FLAT[3] as u16, F_PAWN);
        assert_eq!(PIECE_BASE_FLAT[2], E_PAWN as i32);
    }

    #[test]
    fn piece_base_flat_dragon_matches() {
        assert_eq!(PIECE_BASE_FLAT[29] as u16, F_DRAGON);
    }

    #[test]
    fn mirrored_sq_corners_swap() {
        assert_eq!(MIRRORED_SQ[0], 72); // 1一 → 9一
        assert_eq!(MIRRORED_SQ[72], 0); // 9一 → 1一
        assert_eq!(MIRRORED_SQ[40], 40); // 5五 → 5五 (中央列)
        assert_eq!(MIRRORED_SQ[8], 80); // 1九 → 9九
    }

    #[test]
    fn mirrored_sq_is_involution() {
        for i in 0..81 {
            assert_eq!(MIRRORED_SQ[MIRRORED_SQ[i] as usize], i as i32);
        }
    }
}
