//! AVX-2 (8 lane × i32) board phase 実装。tail は scalar fallback。

#![cfg(target_arch = "x86_64")]

use super::{BoardPhaseArgs, MIRRORED_SQ, PIECE_BASE_FLAT};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpeq_epi32, _mm256_i32gather_epi32,
    _mm256_loadu_si256, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_storeu_si256,
    _mm256_sub_epi32,
};

const LANES: usize = 8;

/// # Safety
/// caller は AVX-2 が available であることを保証する
/// (`super::BoardPhaseDispatch::detect()` または `super::testing::extract_avx2`
/// の `is_x86_feature_detected!` 経由)。`args` の各 slice は `args.n` 以上。
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn extract_halfka_hm_board_phase(args: &mut BoardPhaseArgs<'_>) {
    let stm = args.stm;
    let nstm = args.nstm;
    unsafe {
        let v_80 = _mm256_set1_epi32(80);
        let v_one = _mm256_set1_epi32(1);
        let v_stm_kb = _mm256_set1_epi32(stm.kb_offset);
        let v_nstm_kb = _mm256_set1_epi32(nstm.kb_offset);
        let v_stm_color = _mm256_set1_epi32(stm.color_code);

        let mut i = 0;
        while i + LANES <= args.n {
            let v_pt = _mm256_loadu_si256(args.pt.as_ptr().add(i) as *const __m256i);
            let v_color = _mm256_loadu_si256(args.color.as_ptr().add(i) as *const __m256i);
            let v_sq = _mm256_loadu_si256(args.sq.as_ptr().add(i) as *const __m256i);

            let v_sq_idx_stm = if stm.black_persp == 1 {
                v_sq
            } else {
                _mm256_sub_epi32(v_80, v_sq)
            };
            let v_sq_idx_nstm = _mm256_sub_epi32(v_80, v_sq_idx_stm);

            // `cmpeq_epi32` は match 時 -1 (0xFFFFFFFF)、それ以外 0 を返すので
            // `& 1` で 0/1 に圧縮する。
            let v_cmp_stm = _mm256_cmpeq_epi32(v_color, v_stm_color);
            let v_is_friend_stm = _mm256_and_si256(v_cmp_stm, v_one);
            let v_is_friend_nstm = _mm256_sub_epi32(v_one, v_is_friend_stm);

            let v_pt_x2 = _mm256_slli_epi32(v_pt, 1);
            let v_base_idx_stm = _mm256_add_epi32(v_pt_x2, v_is_friend_stm);
            let v_base_idx_nstm = _mm256_add_epi32(v_pt_x2, v_is_friend_nstm);

            let v_base_stm = _mm256_i32gather_epi32::<4>(PIECE_BASE_FLAT.as_ptr(), v_base_idx_stm);
            let v_base_nstm =
                _mm256_i32gather_epi32::<4>(PIECE_BASE_FLAT.as_ptr(), v_base_idx_nstm);

            let v_sq_packed_stm = if stm.mirror {
                _mm256_i32gather_epi32::<4>(MIRRORED_SQ.as_ptr(), v_sq_idx_stm)
            } else {
                v_sq_idx_stm
            };
            let v_sq_packed_nstm = if nstm.mirror {
                _mm256_i32gather_epi32::<4>(MIRRORED_SQ.as_ptr(), v_sq_idx_nstm)
            } else {
                v_sq_idx_nstm
            };

            let v_idx_stm =
                _mm256_add_epi32(v_stm_kb, _mm256_add_epi32(v_base_stm, v_sq_packed_stm));
            let v_idx_nstm =
                _mm256_add_epi32(v_nstm_kb, _mm256_add_epi32(v_base_nstm, v_sq_packed_nstm));

            _mm256_storeu_si256(args.stm_out.as_mut_ptr().add(i) as *mut __m256i, v_idx_stm);
            _mm256_storeu_si256(
                args.nstm_out.as_mut_ptr().add(i) as *mut __m256i,
                v_idx_nstm,
            );

            i += LANES;
        }

        if i < args.n {
            let mut tail = BoardPhaseArgs {
                pt: &args.pt[i..args.n],
                color: &args.color[i..args.n],
                sq: &args.sq[i..args.n],
                n: args.n - i,
                stm,
                nstm,
                stm_out: &mut args.stm_out[i..args.n],
                nstm_out: &mut args.nstm_out[i..args.n],
            };
            super::scalar::extract_halfka_hm_board_phase(&mut tail);
        }
    }
}
