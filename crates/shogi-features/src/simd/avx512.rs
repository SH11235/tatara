//! AVX-512 (16 lane × i32) board phase 実装。使用 intrinsic は全て AVX-512F の
//! base set 内で済むので DQ / BW / VL は要求しない。tail は scalar fallback。

#![cfg(target_arch = "x86_64")]

use super::{BoardPhaseArgs, MIRRORED_SQ, PIECE_BASE_FLAT};
use core::arch::x86_64::{
    _mm512_add_epi32, _mm512_cmpeq_epi32_mask, _mm512_i32gather_epi32, _mm512_loadu_si512,
    _mm512_mask_blend_epi32, _mm512_set1_epi32, _mm512_slli_epi32, _mm512_storeu_si512,
    _mm512_sub_epi32,
};

const LANES: usize = 16;

/// # Safety
/// caller は AVX-512F が available であることを保証する
/// (`super::BoardPhaseDispatch::detect()` または `super::testing::extract_avx512`
/// の `is_x86_feature_detected!` 経由)。`args` の各 slice は `args.n` 以上。
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn extract_halfka_hm_board_phase(args: &mut BoardPhaseArgs<'_>) {
    let stm = args.stm;
    let nstm = args.nstm;
    unsafe {
        let v_80 = _mm512_set1_epi32(80);
        let v_one = _mm512_set1_epi32(1);
        let v_zero = _mm512_set1_epi32(0);
        let v_stm_kb = _mm512_set1_epi32(stm.kb_offset);
        let v_nstm_kb = _mm512_set1_epi32(nstm.kb_offset);
        let v_stm_color = _mm512_set1_epi32(stm.color_code);

        let mut i = 0;
        while i + LANES <= args.n {
            let v_pt = _mm512_loadu_si512(args.pt.as_ptr().add(i) as *const _);
            let v_color = _mm512_loadu_si512(args.color.as_ptr().add(i) as *const _);
            let v_sq = _mm512_loadu_si512(args.sq.as_ptr().add(i) as *const _);

            let v_sq_idx_stm = if stm.black_persp == 1 {
                v_sq
            } else {
                _mm512_sub_epi32(v_80, v_sq)
            };
            let v_sq_idx_nstm = _mm512_sub_epi32(v_80, v_sq_idx_stm);

            // AVX-512 の cmpeq は `__mmask16` を返すので `mask_blend(0, 1)` で
            // 0/1 ベクトルに直す (`_mm256_and_si256(cmp, 1)` 相当)。
            let mask_stm = _mm512_cmpeq_epi32_mask(v_color, v_stm_color);
            let v_is_friend_stm = _mm512_mask_blend_epi32(mask_stm, v_zero, v_one);
            let v_is_friend_nstm = _mm512_sub_epi32(v_one, v_is_friend_stm);

            let v_pt_x2 = _mm512_slli_epi32(v_pt, 1);
            let v_base_idx_stm = _mm512_add_epi32(v_pt_x2, v_is_friend_stm);
            let v_base_idx_nstm = _mm512_add_epi32(v_pt_x2, v_is_friend_nstm);

            let v_base_stm = _mm512_i32gather_epi32::<4>(v_base_idx_stm, PIECE_BASE_FLAT.as_ptr());
            let v_base_nstm =
                _mm512_i32gather_epi32::<4>(v_base_idx_nstm, PIECE_BASE_FLAT.as_ptr());

            let v_sq_packed_stm = if stm.mirror {
                _mm512_i32gather_epi32::<4>(v_sq_idx_stm, MIRRORED_SQ.as_ptr())
            } else {
                v_sq_idx_stm
            };
            let v_sq_packed_nstm = if nstm.mirror {
                _mm512_i32gather_epi32::<4>(v_sq_idx_nstm, MIRRORED_SQ.as_ptr())
            } else {
                v_sq_idx_nstm
            };

            let v_idx_stm =
                _mm512_add_epi32(v_stm_kb, _mm512_add_epi32(v_base_stm, v_sq_packed_stm));
            let v_idx_nstm =
                _mm512_add_epi32(v_nstm_kb, _mm512_add_epi32(v_base_nstm, v_sq_packed_nstm));

            _mm512_storeu_si512(args.stm_out.as_mut_ptr().add(i) as *mut _, v_idx_stm);
            _mm512_storeu_si512(args.nstm_out.as_mut_ptr().add(i) as *mut _, v_idx_nstm);

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
