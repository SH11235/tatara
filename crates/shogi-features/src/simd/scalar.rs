//! scalar reference 実装。lane-width 1 で SIMD path と byte-identical な形に
//! 書く (parity test の信頼性を上げるため、`map_features_board` の式と完全に
//! 同じ算法を踏む)。

use super::{BoardPhaseArgs, MIRRORED_SQ, PIECE_BASE_FLAT};

/// HalfKaHmMerged 専用 board phase の scalar 実装。
///
/// per-piece に次を計算して `(stm_out[i], nstm_out[i])` に書込む:
/// 1. `sq_index_stm = if stm.black_persp == 1 { sq } else { 80 - sq }`
/// 2. `is_friend_stm = (color == stm.color_code) ? 1 : 0` (=`!is_friend_nstm`)
/// 3. `base_*    = PIECE_BASE_FLAT[pt * 2 + is_friend_*]` (gather)
/// 4. `sq_packed_* = stm.mirror ? MIRRORED_SQ[sq_index_*] : sq_index_*`
/// 5. `idx_* = stm.kb_offset + base_* + sq_packed_*`
///
/// `n` が `pt.len()` / output 長を超えないことは caller が保証する。
#[inline]
pub(super) fn extract_halfka_hm_board_phase(args: &mut BoardPhaseArgs<'_>) {
    let stm = args.stm;
    let nstm = args.nstm;
    for i in 0..args.n {
        let p = args.pt[i];
        let c = args.color[i];
        let s = args.sq[i];

        let sq_idx_stm = if stm.black_persp == 1 { s } else { 80 - s };
        let sq_idx_nstm = 80 - sq_idx_stm;

        let is_friend_stm = (c == stm.color_code) as i32;
        let is_friend_nstm = 1 - is_friend_stm;

        let base_stm = PIECE_BASE_FLAT[(p * 2 + is_friend_stm) as usize];
        let base_nstm = PIECE_BASE_FLAT[(p * 2 + is_friend_nstm) as usize];

        let sq_packed_stm = if stm.mirror {
            MIRRORED_SQ[sq_idx_stm as usize]
        } else {
            sq_idx_stm
        };
        let sq_packed_nstm = if nstm.mirror {
            MIRRORED_SQ[sq_idx_nstm as usize]
        } else {
            sq_idx_nstm
        };

        args.stm_out[i] = stm.kb_offset + base_stm + sq_packed_stm;
        args.nstm_out[i] = nstm.kb_offset + base_nstm + sq_packed_nstm;
    }
}
