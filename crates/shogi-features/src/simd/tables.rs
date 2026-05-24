//! board phase SIMD path 共通の precomputed table。
//!
//! 各 path (scalar / AVX-2 / AVX-512) が同一の table を参照することで
//! bit-identical 性を担保する。table size は L1 cache に余裕で収まる
//! (PIECE_BASE_FLAT 120 byte、MIRRORED_SQ 324 byte)。

use shogi_format::bona_piece::PIECE_BASE;

/// `PIECE_BASE[pt][is_friend]` を `[i32; 15 * 2]` に flatten した形。
///
/// gather lookup の base ptr に使う (`pt * 2 + is_friend` を index に取れば
/// 元 `PIECE_BASE` の対応 cell を返す)。
pub(crate) const PIECE_BASE_FLAT: [i32; 30] = {
    let mut t = [0i32; 30];
    let mut pt = 0;
    while pt < 15 {
        // is_friend = 0 → enemy 側、1 → friend 側 (`PIECE_BASE` の row layout)
        t[pt * 2] = PIECE_BASE[pt][0] as i32;
        t[pt * 2 + 1] = PIECE_BASE[pt][1] as i32;
        pt += 1;
    }
    t
};

/// `sq_index → file-mirror した sq_index` の lookup table。
///
/// `mirror_files = true` のときの盤上駒 sq 反転 (1筋 ↔ 9筋、段は不変) を
/// `(8 - sq/9) * 9 + sq % 9` で precompute。
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
        // Dragon = piece_type 14、friend index = 29。
        assert_eq!(PIECE_BASE_FLAT[29] as u16, F_DRAGON);
    }

    #[test]
    fn mirrored_sq_corners_swap() {
        // 1一 (sq=0、file=0, rank=0) → 9一 (file=8, rank=0、sq=72)
        assert_eq!(MIRRORED_SQ[0], 72);
        // 9一 → 1一
        assert_eq!(MIRRORED_SQ[72], 0);
        // 5五 (file=4, rank=4、sq=40) → 5五 (mirrored_file=4)
        assert_eq!(MIRRORED_SQ[40], 40);
        // 1九 (sq=8, file=0, rank=8) → 9九 (sq=80)
        assert_eq!(MIRRORED_SQ[8], 80);
    }

    #[test]
    fn mirrored_sq_is_involution() {
        for i in 0..81 {
            assert_eq!(MIRRORED_SQ[MIRRORED_SQ[i] as usize], i as i32);
        }
    }
}
