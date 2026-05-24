//! HalfKaHmMerged board phase の runtime SIMD dispatch。
//!
//! - `scalar`: lane-width 1 reference (常時 available、tail fallback も兼ねる)
//! - `avx2`:   x86_64 + AVX-2 (8 lane × i32)
//! - `avx512`: x86_64 + AVX-512F (16 lane × i32)
//!
//! `BoardPhaseDispatch::detect()` で起動時 1 回判定し `OnceLock` に焼く。

use crate::feature_set::FeatureSetSpec;
use std::sync::OnceLock;

mod scalar;
mod tables;

pub(crate) use tables::{MIRRORED_SQ, PIECE_BASE_FLAT};

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

/// HalfKaHmMerged の 1 視点分 SIMD 入力 (king bucket offset を事前計算)。
#[derive(Clone, Copy)]
pub(crate) struct PerspectiveOffset {
    /// `king_bucket * piece_inputs`。
    pub kb_offset: i32,
    /// 盤駒 sq を file mirror する視点か。
    pub mirror: bool,
    /// `Color::Black` なら 1、`White` なら 0 (sq inverse の SIMD 分岐に使う)。
    pub black_persp: i32,
    /// `Color` を i32 に cast した値 (is_friend 判定に使う)。
    pub color_code: i32,
}

/// board phase SIMD path の入出力。各 path は同じ struct を受ける。
pub(crate) struct BoardPhaseArgs<'a> {
    pub pt: &'a [i32],
    pub color: &'a [i32],
    pub sq: &'a [i32],
    pub n: usize,
    pub stm: &'a PerspectiveOffset,
    pub nstm: &'a PerspectiveOffset,
    pub stm_out: &'a mut [i32],
    pub nstm_out: &'a mut [i32],
}

/// 起動時に検出した SIMD path tag。優先順は AVX-512 → AVX-2 → Scalar。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BoardPhaseDispatch {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl BoardPhaseDispatch {
    pub(crate) fn detect() -> BoardPhaseDispatch {
        static CACHED: OnceLock<BoardPhaseDispatch> = OnceLock::new();
        *CACHED.get_or_init(detect_uncached)
    }
}

fn detect_uncached() -> BoardPhaseDispatch {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return BoardPhaseDispatch::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return BoardPhaseDispatch::Avx2;
        }
    }
    BoardPhaseDispatch::Scalar
}

/// HalfKaHmMerged board phase を dispatch して output slice に直接書込む。
#[inline]
pub(crate) fn extract_halfka_hm_board_phase(mut args: BoardPhaseArgs<'_>) {
    debug_assert!(args.pt.len() >= args.n && args.color.len() >= args.n && args.sq.len() >= args.n);
    debug_assert!(args.stm_out.len() >= args.n && args.nstm_out.len() >= args.n);
    match BoardPhaseDispatch::detect() {
        BoardPhaseDispatch::Scalar => scalar::extract_halfka_hm_board_phase(&mut args),
        #[cfg(target_arch = "x86_64")]
        BoardPhaseDispatch::Avx2 => {
            // SAFETY: detect() が AVX-2 を確認済。
            unsafe { avx2::extract_halfka_hm_board_phase(&mut args) }
        }
        #[cfg(target_arch = "x86_64")]
        BoardPhaseDispatch::Avx512 => {
            // SAFETY: detect() が AVX-512F を確認済。
            unsafe { avx512::extract_halfka_hm_board_phase(&mut args) }
        }
    }
}

pub(crate) fn spec_is_halfka_hm_merged(spec: &FeatureSetSpec) -> bool {
    use crate::FeatureSet;
    matches!(spec.feature_set(), FeatureSet::HalfKaHmMerged)
}

/// test 用 forced-dispatch entry。runtime check 後にだけ unsafe で呼ぶ。
/// runtime check に落ちた場合は silent skip (caller test が dispatch tag を
/// 出力するので silent skip も視認可能)。
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) fn extract_scalar(mut args: BoardPhaseArgs<'_>) {
        scalar::extract_halfka_hm_board_phase(&mut args);
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn extract_avx2(mut args: BoardPhaseArgs<'_>) {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // SAFETY: 直前で AVX-2 を確認している。
        unsafe { avx2::extract_halfka_hm_board_phase(&mut args) }
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn extract_avx512(mut args: BoardPhaseArgs<'_>) {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // SAFETY: 直前で AVX-512F を確認している。
        unsafe { avx512::extract_halfka_hm_board_phase(&mut args) }
    }
}

// =============================================================================
// parity tests (scalar vs AVX-2 vs AVX-512)
// =============================================================================

#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::FeatureSet;
    use shogi_format::types::{Color, Piece, PieceType, Square};
    use shogi_format::{PackedSfenValue, ShogiBoard};

    /// `sample.psv` (100 records) を読み込む共通 fixture。
    fn sample_psv_records() -> Vec<PackedSfenValue> {
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let bytes =
            std::fs::read(&path).expect("sample.psv が読めない (../shogi-format/tests/data/)");
        assert_eq!(bytes.len() % 40, 0);
        assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (不正ビットパターン無し、align = 1)。`bytes.len() % 40 == 0` を直前で
        // 確認済、Vec<u8> の lifetime 内に閉じる。
        let recs: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };
        recs.to_vec()
    }

    fn collect_board_pieces(board: &ShogiBoard) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut pt = Vec::new();
        let mut color = Vec::new();
        let mut sq = Vec::new();
        board.for_each_board_piece(|piece, s| {
            pt.push(piece.piece_type as i32);
            color.push(piece.color as i32);
            sq.push(s.0 as i32);
        });
        (pt, color, sq)
    }

    fn build_perspective(
        spec: &FeatureSetSpec,
        king_sq: Square,
        perspective: Color,
    ) -> PerspectiveOffset {
        let (king_bucket, mirror) = spec.perspective_ctx_for_test(king_sq, perspective);
        PerspectiveOffset {
            kb_offset: (king_bucket * spec.piece_inputs()) as i32,
            mirror,
            black_persp: if perspective == Color::Black { 1 } else { 0 },
            color_code: perspective as i32,
        }
    }

    type PathOutput = (Vec<i32>, Vec<i32>);

    fn run_path<F: FnOnce(BoardPhaseArgs<'_>)>(
        pt: &[i32],
        color: &[i32],
        sq: &[i32],
        n: usize,
        stm: &PerspectiveOffset,
        nstm: &PerspectiveOffset,
        path: F,
    ) -> PathOutput {
        let mut stm_out = vec![0i32; n];
        let mut nstm_out = vec![0i32; n];
        path(BoardPhaseArgs {
            pt,
            color,
            sq,
            n,
            stm,
            nstm,
            stm_out: &mut stm_out,
            nstm_out: &mut nstm_out,
        });
        (stm_out, nstm_out)
    }

    fn run_all_paths(
        pt: &[i32],
        color: &[i32],
        sq: &[i32],
        n: usize,
        stm: &PerspectiveOffset,
        nstm: &PerspectiveOffset,
    ) -> (PathOutput, Option<PathOutput>, Option<PathOutput>) {
        let scalar_out = run_path(pt, color, sq, n, stm, nstm, testing::extract_scalar);

        let avx2_out = {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                Some(run_path(pt, color, sq, n, stm, nstm, testing::extract_avx2))
            } else {
                None
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                None
            }
        };
        let avx512_out = {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx512f") {
                Some(run_path(
                    pt,
                    color,
                    sq,
                    n,
                    stm,
                    nstm,
                    testing::extract_avx512,
                ))
            } else {
                None
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                None
            }
        };

        (scalar_out, avx2_out, avx512_out)
    }

    #[test]
    fn board_phase_paths_match_on_sample_psv() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let records = sample_psv_records();
        let mut checked = 0usize;
        for (i, psv) in records.iter().enumerate() {
            let board = psv.decode();
            let stm = board.side_to_move;
            let nstm = stm.opponent();
            let stm_king = board.king_square(stm);
            let nstm_king = board.king_square(nstm);
            if !stm_king.is_valid() || !nstm_king.is_valid() {
                continue;
            }
            let (pt, color, sq) = collect_board_pieces(&board);
            if pt.is_empty() {
                continue;
            }
            let stm_pers = build_perspective(&spec, stm_king, stm);
            let nstm_pers = build_perspective(&spec, nstm_king, nstm);

            let (scalar_out, avx2, avx512) =
                run_all_paths(&pt, &color, &sq, pt.len(), &stm_pers, &nstm_pers);
            if let Some(avx2_out) = avx2 {
                assert_eq!(scalar_out.0, avx2_out.0, "record {i}: stm scalar vs AVX-2");
                assert_eq!(scalar_out.1, avx2_out.1, "record {i}: nstm scalar vs AVX-2");
            }
            if let Some(avx512_out) = avx512 {
                assert_eq!(
                    scalar_out.0, avx512_out.0,
                    "record {i}: stm scalar vs AVX-512"
                );
                assert_eq!(
                    scalar_out.1, avx512_out.1,
                    "record {i}: nstm scalar vs AVX-512"
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "sample.psv に valid record が無い");
    }

    /// AVX-2 (8) / AVX-512 (16) の lane 境界前後 + tail fallback を網羅する n 群。
    #[test]
    fn board_phase_paths_match_at_lane_boundaries() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let board = make_full_board();
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        let stm_pers = build_perspective(&spec, board.king_square(stm), stm);
        let nstm_pers = build_perspective(&spec, board.king_square(nstm), nstm);
        let (pt_all, color_all, sq_all) = collect_board_pieces(&board);
        assert!(pt_all.len() >= 16);

        for &n in &[0usize, 1, 7, 8, 9, 15, 16, 17, pt_all.len()] {
            if n > pt_all.len() {
                continue;
            }
            let pt = &pt_all[..n];
            let color = &color_all[..n];
            let sq = &sq_all[..n];
            let (scalar_out, avx2, avx512) = run_all_paths(pt, color, sq, n, &stm_pers, &nstm_pers);
            if let Some(avx2_out) = avx2 {
                assert_eq!(scalar_out.0, avx2_out.0, "n={n}: stm scalar vs AVX-2");
                assert_eq!(scalar_out.1, avx2_out.1, "n={n}: nstm scalar vs AVX-2");
            }
            if let Some(avx512_out) = avx512 {
                assert_eq!(scalar_out.0, avx512_out.0, "n={n}: stm scalar vs AVX-512");
                assert_eq!(scalar_out.1, avx512_out.1, "n={n}: nstm scalar vs AVX-512");
            }
        }
    }

    /// 同一 PSV record で `map_features_board` closure + scalar + AVX-2 +
    /// AVX-512 の 4 経路を一括比較する。runtime detect が false の SIMD path は
    /// silent skip するが、最後に各 path の比較件数を stdout に出して visible に。
    #[test]
    fn closure_and_all_simd_paths_agree_on_sample_psv() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let records = sample_psv_records();
        let mut compared_scalar = 0usize;
        let mut compared_avx2 = 0usize;
        let mut compared_avx512 = 0usize;
        let mut closure_only = 0usize;
        for (i, psv) in records.iter().enumerate() {
            let board = psv.decode();
            let stm = board.side_to_move;
            let nstm = stm.opponent();
            let stm_king = board.king_square(stm);
            let nstm_king = board.king_square(nstm);
            if !stm_king.is_valid() || !nstm_king.is_valid() {
                continue;
            }
            let (pt, color, sq) = collect_board_pieces(&board);

            // `map_features_board` は board → king → hand の順に emit するので
            // 先頭 `pt.len()` 要素 (board phase) を切り出して SIMD と比較する。
            let mut via_closure = Vec::new();
            spec.map_features_board(&board, |stm_idx, nstm_idx| {
                via_closure.push((stm_idx as i32, nstm_idx as i32));
            });

            if pt.is_empty() {
                closure_only += 1;
                continue;
            }
            assert!(
                via_closure.len() >= pt.len(),
                "record {i}: closure emit が board piece 数を下回る ({} < {})",
                via_closure.len(),
                pt.len()
            );
            let closure_board: Vec<(i32, i32)> = via_closure[..pt.len()].to_vec();

            let stm_pers = build_perspective(&spec, stm_king, stm);
            let nstm_pers = build_perspective(&spec, nstm_king, nstm);
            let (scalar_out, avx2, avx512) =
                run_all_paths(&pt, &color, &sq, pt.len(), &stm_pers, &nstm_pers);

            let scalar_pairs: Vec<(i32, i32)> = scalar_out
                .0
                .iter()
                .zip(scalar_out.1.iter())
                .map(|(&a, &b)| (a, b))
                .collect();
            assert_eq!(scalar_pairs, closure_board, "record {i}: scalar vs closure");
            compared_scalar += 1;

            if let Some(avx2_out) = avx2 {
                let pairs: Vec<(i32, i32)> = avx2_out
                    .0
                    .iter()
                    .zip(avx2_out.1.iter())
                    .map(|(&a, &b)| (a, b))
                    .collect();
                assert_eq!(pairs, closure_board, "record {i}: AVX-2 vs closure");
                compared_avx2 += 1;
            }
            if let Some(avx512_out) = avx512 {
                let pairs: Vec<(i32, i32)> = avx512_out
                    .0
                    .iter()
                    .zip(avx512_out.1.iter())
                    .map(|(&a, &b)| (a, b))
                    .collect();
                assert_eq!(pairs, closure_board, "record {i}: AVX-512 vs closure");
                compared_avx512 += 1;
            }
        }
        // どの path が test で実際に比較されたか stdout に書き出して、
        // (runtime feature 未対応で) silent skip された組合せが見えるようにする。
        println!(
            "closure_and_all_simd_paths_agree_on_sample_psv: scalar={compared_scalar} \
             avx2={compared_avx2} avx512={compared_avx512} closure_only={closure_only}"
        );
        assert!(
            compared_scalar > 0,
            "scalar parity が 1 record も比較されなかった"
        );
    }

    /// AVX-2 以降を持つ x86_64 で detect が scalar に落ちないこと。Sandy Bridge
    /// 以前は AVX-2 無しで scalar に落ちるので skip。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatch_selects_simd_on_x86_64() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let detected = BoardPhaseDispatch::detect();
        assert!(
            !matches!(detected, BoardPhaseDispatch::Scalar),
            "AVX-2 以降の host で scalar に dispatch している (実 detect: {:?})",
            detected,
        );
    }

    /// 16 lane を超える駒数を持つ full-board fixture (両陣 9 歩 + 4 駒)。
    fn make_full_board() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }
        // 加えて 16 lane を埋めるための盤上駒。
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Bishop);
        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop);
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook);
        board
    }
}
