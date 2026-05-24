//! AVX-2 (8 lane × i32) board phase 実装。
//!
//! 関数単位で `#[target_feature(enable = "avx2")]` を付与し、binary 全体は
//! AVX-2 必須化しない (caller の `BoardPhaseDispatch::detect()` が runtime
//! feature check 後にのみ unsafe で呼ぶ)。
//!
//! 内部は `_mm256_loadu_si256` / `_mm256_i32gather_epi32` / `_mm256_storeu_si256`
//! を素朴に並べた lane-major 実装。8 lane に満たない tail は scalar fallback。

#![cfg(target_arch = "x86_64")]

use super::{BoardPhaseArgs, MIRRORED_SQ, PIECE_BASE_FLAT};
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpeq_epi32, _mm256_i32gather_epi32,
    _mm256_loadu_si256, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_storeu_si256,
    _mm256_sub_epi32,
};

const LANES: usize = 8;

/// HalfKaHmMerged 専用 board phase の AVX-2 実装。
///
/// # Safety
/// caller は AVX-2 が available であることを保証する
/// (`super::BoardPhaseDispatch::detect()` で確認済の dispatch 経由、または
/// `super::testing::extract_avx2` の `is_x86_feature_detected!` 経由)。
/// `args` の各 slice は `args.n` 以上の長さを持つこと。
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn extract_halfka_hm_board_phase(args: &mut BoardPhaseArgs<'_>) {
    let stm = args.stm;
    let nstm = args.nstm;
    // SAFETY: target_feature が AVX-2 を許可、caller がランタイム確認済。
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

            // sq_idx_stm = if stm.black { sq } else { 80 - sq }
            let v_sq_idx_stm = if stm.black_persp == 1 {
                v_sq
            } else {
                _mm256_sub_epi32(v_80, v_sq)
            };
            // sq_idx_nstm = 80 - sq_idx_stm
            let v_sq_idx_nstm = _mm256_sub_epi32(v_80, v_sq_idx_stm);

            // is_friend_stm = (color == stm_color) ? 1 : 0
            // `_mm256_cmpeq_epi32` は match 時 0xFFFFFFFF (=-1)、それ以外 0 を
            // 返すので `& 1` で `{-1, 0} → {1, 0}` に圧縮する。
            let v_cmp_stm = _mm256_cmpeq_epi32(v_color, v_stm_color);
            let v_is_friend_stm = _mm256_and_si256(v_cmp_stm, v_one);
            let v_is_friend_nstm = _mm256_sub_epi32(v_one, v_is_friend_stm);

            // base_idx = pt * 2 + is_friend
            let v_pt_x2 = _mm256_slli_epi32(v_pt, 1);
            let v_base_idx_stm = _mm256_add_epi32(v_pt_x2, v_is_friend_stm);
            let v_base_idx_nstm = _mm256_add_epi32(v_pt_x2, v_is_friend_nstm);

            // base_* = PIECE_BASE_FLAT[base_idx_*]
            let v_base_stm = _mm256_i32gather_epi32::<4>(PIECE_BASE_FLAT.as_ptr(), v_base_idx_stm);
            let v_base_nstm =
                _mm256_i32gather_epi32::<4>(PIECE_BASE_FLAT.as_ptr(), v_base_idx_nstm);

            // sq_packed = mirror ? MIRRORED_SQ[sq_idx] : sq_idx
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

            // idx = kb_offset + base + sq_packed
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

        // Tail: scalar formula で埋める (SIMD と同じ算法を踏み bit-identical 保証)。
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
