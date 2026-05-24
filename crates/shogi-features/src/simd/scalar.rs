//! lane-width 1 reference 実装。SIMD path の tail fallback も兼ねるので
//! 算法は `avx2` / `avx512` と完全に一致させる。

use super::{BoardPhaseArgs, MIRRORED_SQ, PIECE_BASE_FLAT};

/// HalfKaHmMerged board phase の scalar 実装。
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
