//! `map_features_board` の board phase 向け runtime SIMD dispatch。
//!
//! HalfKaHmMerged feature set の board piece 1 枚あたりの
//! `(BonaPiece × 2 視点 → packed index × 2 視点)` 計算を SIMD lane に並べる
//! ための path 群と、起動時 1 回判定の dispatch hook を置く。
//!
//! - `scalar`: lane-width 1 の reference 実装、常時 available
//! - `avx2`:   x86_64 + AVX-2 (8 lane × i32)、`#[target_feature]` 制御
//! - `avx512`: x86_64 + AVX-512F (16 lane × i32)、`#[target_feature]` 制御
//!
//! dispatch は `BoardPhaseDispatch::detect()` 経由で起動時 1 回判定 → `OnceLock` に
//! cache、以降は branch なしで関数 pointer 呼び出し。

use crate::feature_set::FeatureSetSpec;
use std::sync::OnceLock;

mod scalar;
mod tables;

pub(crate) use tables::{MIRRORED_SQ, PIECE_BASE_FLAT};

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

/// HalfKaHmMerged 専用 board phase の 1 視点コンテキスト (king bucket × piece
/// inputs を事前計算した縦軸 offset、mirror 要否)。
#[derive(Clone, Copy)]
pub(crate) struct PerspectiveOffset {
    /// `king_bucket * piece_inputs` (= 全 board piece の base offset)。
    pub kb_offset: i32,
    /// 視点 sq → packed sq の file mirror 要否。
    pub mirror: bool,
    /// `1` if perspective == Color::Black else `0`、SIMD で sq 変換に使う。
    pub black_persp: i32,
    /// `0` if perspective == Color::Black else `1`、SIMD で is_friend 判定に使う。
    pub color_code: i32,
}

/// board phase SIMD path の入出力 (clippy::too_many_arguments を避ける
/// 集約 struct、各 path は同じ shape を取る)。
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

/// 起動時に検出した SIMD dispatch tag。`BoardPhaseDispatch::detect()` で 1 回判定して
/// `OnceLock` に焼く。AVX-512 → AVX-2 → Scalar の優先順。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BoardPhaseDispatch {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl BoardPhaseDispatch {
    /// 起動時に CPU feature を 1 回検出して cache する。
    pub(crate) fn detect() -> BoardPhaseDispatch {
        static CACHED: OnceLock<BoardPhaseDispatch> = OnceLock::new();
        *CACHED.get_or_init(detect_uncached)
    }
}

fn detect_uncached() -> BoardPhaseDispatch {
    #[cfg(target_arch = "x86_64")]
    {
        // 本 path で使う intrinsic (`_mm512_loadu_si512` / `_mm512_set1_epi32` /
        // `_mm512_cmpeq_epi32_mask` / `_mm512_mask_blend_epi32` /
        // `_mm512_i32gather_epi32` / `_mm512_storeu_si512` 等) は全て AVX-512F
        // のみで利用可能。DQ / BW / VL は本実装で要求しない (KNL 等 F-only host
        // でも SIMD path に dispatch する)。
        if std::is_x86_feature_detected!("avx512f") {
            return BoardPhaseDispatch::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return BoardPhaseDispatch::Avx2;
        }
    }
    BoardPhaseDispatch::Scalar
}

/// HalfKaHmMerged 専用 board phase を dispatch して output slice に直接書込む。
///
/// `BoardPhaseDispatch::detect()` で起動時に決定した path に dispatch、結果は scalar 経路と
/// byte-identical (parity test 参照)。
#[inline]
pub(crate) fn extract_halfka_hm_board_phase(mut args: BoardPhaseArgs<'_>) {
    debug_assert!(args.pt.len() >= args.n && args.color.len() >= args.n && args.sq.len() >= args.n);
    debug_assert!(args.stm_out.len() >= args.n && args.nstm_out.len() >= args.n);
    match BoardPhaseDispatch::detect() {
        BoardPhaseDispatch::Scalar => scalar::extract_halfka_hm_board_phase(&mut args),
        #[cfg(target_arch = "x86_64")]
        BoardPhaseDispatch::Avx2 => {
            // SAFETY: `BoardPhaseDispatch::detect()` が AVX-2 を確認済の path に来る。
            unsafe { avx2::extract_halfka_hm_board_phase(&mut args) }
        }
        #[cfg(target_arch = "x86_64")]
        BoardPhaseDispatch::Avx512 => {
            // SAFETY: `BoardPhaseDispatch::detect()` が AVX-512F を確認済の path に来る。
            unsafe { avx512::extract_halfka_hm_board_phase(&mut args) }
        }
    }
}

/// HalfKaHmMerged feature set かを判定する小さな helper (dispatch hook が
/// generic feature set には適用できないため)。
pub(crate) fn spec_is_halfka_hm_merged(spec: &FeatureSetSpec) -> bool {
    use crate::FeatureSet;
    matches!(spec.feature_set(), FeatureSet::HalfKaHmMerged)
}

/// dispatch を強制的に scalar / AVX-2 / AVX-512 のいずれかで実行する test 用 hook
/// (cache を介さず直接 path を呼ぶ)。release では使わない。
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) fn extract_scalar(mut args: BoardPhaseArgs<'_>) {
        scalar::extract_halfka_hm_board_phase(&mut args);
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn extract_avx2(mut args: BoardPhaseArgs<'_>) {
        if !std::is_x86_feature_detected!("avx2") {
            // AVX-2 が無いマシンでは test skip。
            return;
        }
        // SAFETY: 直前の `is_x86_feature_detected!` で AVX-2 を確認している。
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

    /// `board.for_each_board_piece` で得る (pt, color, sq) 3 array を構築する。
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

    /// HalfKaHmMerged の perspective 引数を 1 視点分組み立てる helper。
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

    /// SIMD lane 境界 (8 / 16) の前後と tail-fallback path を網羅する size 群で
    /// AVX-2 / AVX-512 / scalar 出力の byte-identical 性を確認する。
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

        // lane width 8 (AVX-2) と 16 (AVX-512) の境界前後を網羅する n 群。
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

    /// 同一 PSV record に対して 4 経路
    /// (`map_features_board` closure / scalar / AVX-2 / AVX-512) すべての
    /// 出力が一致することを 1 つの test で網羅確認する (path 間の局所 parity と
    /// 高位 API 等価性をまとめて 1 経路にチェックして transitive 担保の漏れを
    /// 防ぐ)。SIMD path は runtime detect が false なら該当 path だけ skip し、
    /// どの path を比較したかを stdout に書き出して silent skip を視認可能に。
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

            // 1. closure 経路 (board phase だけを抜き出し、king / hand は除く)
            let mut closure_board = Vec::new();
            {
                let stm_ctx = spec.perspective_ctx_for_test(stm_king, stm);
                let nstm_ctx = spec.perspective_ctx_for_test(nstm_king, nstm);
                let (stm_kb, stm_mirror) = stm_ctx;
                let (nstm_kb, nstm_mirror) = nstm_ctx;
                let pi = spec.piece_inputs();
                use shogi_format::BonaPiece;
                board.for_each_board_piece(|piece, s| {
                    // map_features_board と同じ式を踏む (board piece の
                    // `pack_bonapiece` は `folds_enemy_king` を trigger しない)。
                    let stm_bp = BonaPiece::from_piece_square(piece, s, stm).value() as i32;
                    let nstm_bp = BonaPiece::from_piece_square(piece, s, nstm).value() as i32;
                    let stm_packed = pack_for_test(stm_bp, stm_mirror);
                    let nstm_packed = pack_for_test(nstm_bp, nstm_mirror);
                    let stm_idx = (stm_kb * pi) as i32 + stm_packed;
                    let nstm_idx = (nstm_kb * pi) as i32 + nstm_packed;
                    closure_board.push((stm_idx, nstm_idx));
                });
            }

            if pt.is_empty() {
                closure_only += 1;
                continue;
            }

            // 2. forced SIMD path 群
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

    /// board piece 限定 `pack_bonapiece` (mirror のみ適用、enemy-king fold は
    /// board piece では trigger しないので省略)。closure 比較 test 用 helper。
    fn pack_for_test(bp: i32, mirror: bool) -> i32 {
        use shogi_format::bona_piece::FE_HAND_END;
        if !mirror || bp < FE_HAND_END as i32 {
            return bp;
        }
        let rel = bp - FE_HAND_END as i32;
        let piece_index = rel / 81;
        let sq = rel % 81;
        let file = sq / 9;
        let rank = sq % 9;
        let mirrored_sq = (8 - file) * 9 + rank;
        FE_HAND_END as i32 + piece_index * 81 + mirrored_sq
    }

    /// 起動時 detect が現在 host で「scalar 以外」を選ぶことを確認する
    /// (本機 i9-10900X / Ryzen 5950X 等の AVX-2 以降前提)。x86_64 以外の host
    /// では skip。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatch_selects_simd_on_x86_64() {
        if !std::is_x86_feature_detected!("avx2") {
            return; // 極端に古い x86_64 (Sandy Bridge 以前) は AVX-2 無く scalar fallback。
        }
        let detected = BoardPhaseDispatch::detect();
        assert!(
            !matches!(detected, BoardPhaseDispatch::Scalar),
            "AVX-2 以降の host で scalar に dispatch している (実 detect: {:?})",
            detected,
        );
    }

    /// SIMD lane を埋め切る `for_each_board_piece` 出力を作るための full-board
    /// fixture (両陣 9 歩 + 駒台駒)。
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
