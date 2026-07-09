//! `feature_set` の参照照合・第一原理検証・回帰テスト。
//!
//! - 参照実装あり 3 cell (`halfkp` / `halfka-split` / `halfka-hm-merged`):
//!   nnue-pytorch 系の index 式を移植した oracle / `ShogiHalfKA_hm` と照合。
//! - 参照実装なし 2 cell (`halfka-merged` / `halfka-hm-split`): index 範囲・
//!   injective・玉 plane 衝突・退化一致・perspective swap・mirror involution を
//!   第一原理で検証する。

use std::path::PathBuf;

use shogi_features::EffectBucketConfig;
use shogi_features::feature_set::{FeatureSet, FeatureSetSpec};
use shogi_features::halfka_hm::ShogiHalfKA_hm;
use shogi_format::bona_piece::{E_KING, F_KING, FE_OLD_END};
use shogi_format::types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, PieceType, Square};
use shogi_format::{BonaPiece, PackedSfenValue, ShogiBoard};

// =============================================================================
// fixtures / helpers
// =============================================================================

/// `shogi-format` crate の `tests/data/sample.psv` (100 records × 40 bytes) を
/// `PackedSfenValue` の Vec として読み込む。
fn sample_psv_records() -> Vec<PackedSfenValue> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
    let bytes = std::fs::read(&path).expect("sample.psv が読めない (../shogi-format/tests/data/)");
    assert_eq!(bytes.len() % 40, 0);
    assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
    // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
    // (不正ビットパターン無し、align = 1)。bytes 長は 40 の倍数。
    let recs: &[PackedSfenValue] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
    };
    recs.to_vec()
}

/// 玉マスを視点変換したマスインデックス (後手視点は盤を 180 度回転)。
fn perspective_index(sq: Square, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq.index(),
        Color::White => sq.inverse().index(),
    }
}

/// 特徴インデックスを `(king_bucket, packed_bonapiece)` に分解する。
fn decompose(index: usize, piece_inputs: usize) -> (usize, usize) {
    (index / piece_inputs, index % piece_inputs)
}

/// spec の特徴を `Vec<(stm_idx, nstm_idx)>` で収集する (emission 順序を保持)。
fn collect(spec: &FeatureSetSpec, board: &ShogiBoard) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    spec.map_features_board(board, |stm, nstm| out.push((stm, nstm)));
    out
}

/// 玉 + 任意の駒からなる `ShogiBoard` を組み立てる。
fn build_board(
    stm: Color,
    black_king: Square,
    white_king: Square,
    pieces: &[(Color, PieceType, Square)],
) -> ShogiBoard {
    let mut board = ShogiBoard {
        side_to_move: stm,
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

/// 局面を盤ごと筋ミラーする (盤上駒・玉マスの筋を反転、手番・手駒は不変)。
fn mirror_board(board: &ShogiBoard) -> ShogiBoard {
    let mut mirrored = ShogiBoard {
        side_to_move: board.side_to_move,
        black_hand: board.black_hand,
        white_hand: board.white_hand,
        black_king_sq: board.black_king_sq.mirror_file(),
        white_king_sq: board.white_king_sq.mirror_file(),
        ..Default::default()
    };
    for (idx, &piece) in board.board.iter().enumerate() {
        if piece.is_some() {
            let dst = Square::from_index(idx).mirror_file();
            mirrored.board[dst.index()] = piece;
        }
    }
    mirrored
}

fn collect_effect_bucket_sorted(
    spec: &FeatureSetSpec,
    board: &ShogiBoard,
) -> (Vec<usize>, Vec<usize>) {
    let mut stm = vec![0i32; spec.max_active()];
    let mut nstm = vec![0i32; spec.max_active()];
    let n = spec.extract_active_features(board, &mut stm, &mut nstm);
    let mut stm: Vec<usize> = stm[..n].iter().map(|&x| x as usize).collect();
    let mut nstm: Vec<usize> = nstm[..n].iter().map(|&x| x as usize).collect();
    stm.sort_unstable();
    nstm.sort_unstable();
    (stm, nstm)
}

// =============================================================================
// 参照 oracle (nnue-pytorch 系の index 式を再構成)
// =============================================================================

/// HalfKP index 式の参照 oracle。
/// 玉を特徴に含めず、`king_sq * 1548 + bonapiece`。
fn oracle_halfkp(board: &ShogiBoard) -> Vec<(usize, usize)> {
    const FE_END: usize = 1548;
    let mut out = Vec::new();
    let stm = board.side_to_move;
    let nstm = stm.opponent();
    let stm_king = board.king_square(stm);
    let nstm_king = board.king_square(nstm);
    if !stm_king.is_valid() || !nstm_king.is_valid() {
        return out;
    }
    let stm_ksq = perspective_index(stm_king, stm);
    let nstm_ksq = perspective_index(nstm_king, nstm);

    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);
                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                out.push((
                    stm_ksq * FE_END + stm_bp.value() as usize,
                    nstm_ksq * FE_END + nstm_bp.value() as usize,
                ));
            }
        }
    }
    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            for i in 1..=board.hand(owner).count(pt) {
                let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                out.push((
                    stm_ksq * FE_END + stm_bp.value() as usize,
                    nstm_ksq * FE_END + nstm_bp.value() as usize,
                ));
            }
        }
    }
    out
}

/// `ShogiHalfKA` (Non-Mirror) index 式の参照 oracle。両玉を別 plane に置き
/// `king_sq * 1710 + bonapiece`。
fn oracle_halfka_split(board: &ShogiBoard) -> Vec<(usize, usize)> {
    const PI: usize = 1710;
    let mut out = Vec::new();
    let stm = board.side_to_move;
    let nstm = stm.opponent();
    let stm_king = board.king_square(stm);
    let nstm_king = board.king_square(nstm);
    if !stm_king.is_valid() || !nstm_king.is_valid() {
        return out;
    }
    let stm_kb = perspective_index(stm_king, stm);
    let nstm_kb = perspective_index(nstm_king, nstm);

    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);
                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                out.push((
                    stm_kb * PI + stm_bp.value() as usize,
                    nstm_kb * PI + nstm_bp.value() as usize,
                ));
            }
        }
    }

    let king_bp = |king: Square, persp: Color, friend: bool| {
        let base = if friend { F_KING } else { E_KING } as usize;
        base + perspective_index(king, persp)
    };
    out.push((
        stm_kb * PI + king_bp(stm_king, stm, true),
        nstm_kb * PI + king_bp(nstm_king, nstm, true),
    ));
    out.push((
        stm_kb * PI + king_bp(nstm_king, stm, false),
        nstm_kb * PI + king_bp(stm_king, nstm, false),
    ));

    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            for i in 1..=board.hand(owner).count(pt) {
                let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                if stm_bp != BonaPiece::ZERO {
                    let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                    out.push((
                        stm_kb * PI + stm_bp.value() as usize,
                        nstm_kb * PI + nstm_bp.value() as usize,
                    ));
                }
            }
        }
    }
    out
}

// =============================================================================
// 参照照合 (参照実装あり 3 cell)
// =============================================================================

#[test]
fn crosscheck_halfkp_against_oracle() {
    let spec = FeatureSet::HalfKp.spec();
    let mut nonempty = 0;
    for psv in sample_psv_records() {
        let board = psv.decode();
        let got = collect(&spec, &board);
        let want = oracle_halfkp(&board);
        assert_eq!(got, want, "halfkp が HalfKP oracle と不一致");
        nonempty += usize::from(!got.is_empty());
    }
    assert!(
        nonempty > 0,
        "sample.psv に active feature を持つ record が無い"
    );
}

#[test]
fn crosscheck_halfka_split_against_oracle() {
    let spec = FeatureSet::HalfKaSplit.spec();
    let mut nonempty = 0;
    for psv in sample_psv_records() {
        let board = psv.decode();
        let got = collect(&spec, &board);
        let want = oracle_halfka_split(&board);
        assert_eq!(got, want, "halfka-split が HalfKA oracle と不一致");
        nonempty += usize::from(!got.is_empty());
    }
    assert!(
        nonempty > 0,
        "sample.psv に active feature を持つ record が無い"
    );
}

#[test]
fn regression_halfka_hm_merged_is_bit_identical_to_shogi_halfka_hm() {
    // 回帰保証: halfka-hm-merged は `ShogiHalfKA_hm` と emission 順序含め
    // bit-identical な index を返す。
    let spec = FeatureSet::HalfKaHmMerged.spec();
    let mut nonempty = 0;
    for psv in sample_psv_records() {
        let board = psv.decode();
        let got = collect(&spec, &board);
        let mut want = Vec::new();
        ShogiHalfKA_hm.map_features_board(&board, |stm, nstm| want.push((stm, nstm)));
        assert_eq!(got, want, "halfka-hm-merged が ShogiHalfKA_hm と不一致");
        nonempty += usize::from(!got.is_empty());
    }
    assert!(
        nonempty > 0,
        "sample.psv に active feature を持つ record が無い"
    );
}

#[test]
fn golden_halfka_hm_merged_kings_only() {
    // 既知局面の index 集合を固定する。bk=5九 / wk=5一 / 先手番、盤上駒・手駒なし。
    // 両玉とも視点変換後 file=4 のためミラー無し、king bucket は 44。
    //   自玉 packed = F_KING + 44 = 1592 → index = 44*1629 + 1592 = 73268
    //   敵玉 packed = (E_KING + 36) - 81 = 1584 → index = 44*1629 + 1584 = 73260
    let board = build_board(Color::Black, Square::new(4, 8), Square::new(4, 0), &[]);
    let spec = FeatureSet::HalfKaHmMerged.spec();
    assert_eq!(collect(&spec, &board), vec![(73268, 73268), (73260, 73260)],);
    // 同一局面で `ShogiHalfKA_hm` とも一致する。
    let mut via_shogi_halfka_hm = Vec::new();
    ShogiHalfKA_hm.map_features_board(&board, |stm, nstm| via_shogi_halfka_hm.push((stm, nstm)));
    assert_eq!(via_shogi_halfka_hm, vec![(73268, 73268), (73260, 73260)]);
}

#[test]
fn golden_halfka_merged_kings_only() {
    // bk=1一 / wk=9九 / 先手番。Direct (81 bucket)、両玉とも king bucket = 0。
    //   自玉 packed = F_KING + 0 = 1548 → index = 0*1629 + 1548 = 1548
    //   敵玉 packed = (E_KING + 80) - 81 = 1628 → index = 0*1629 + 1628 = 1628
    let board = build_board(Color::Black, Square::new(0, 0), Square::new(8, 8), &[]);
    let spec = FeatureSet::HalfKaMerged.spec();
    assert_eq!(collect(&spec, &board), vec![(1548, 1548), (1628, 1628)]);
}

#[test]
fn golden_halfka_hm_split_kings_only() {
    // bk=7筋3段 / wk=3筋7段 / 先手番。両玉とも視点変換後 file=6 (>= 5) のため
    // 筋ミラーが効き king bucket は 2*9+2 = 20。SplitPlane は敵玉を畳まない。
    //   自玉 BonaPiece = F_KING + 56 = 1604、筋ミラー後の packed = 1568
    //     → index = 20*1710 + 1568 = 35768
    //   敵玉 BonaPiece = E_KING + 24 = 1653、筋ミラー後の packed = 1689
    //     → index = 20*1710 + 1689 = 35889
    let board = build_board(Color::Black, Square::new(6, 2), Square::new(2, 6), &[]);
    let spec = FeatureSet::HalfKaHmSplit.spec();
    assert_eq!(collect(&spec, &board), vec![(35768, 35768), (35889, 35889)]);
}

// =============================================================================
// index-level 不変条件 (全 5 cell)
// =============================================================================

#[test]
fn all_cells_indices_in_range_and_within_max_active() {
    let records = sample_psv_records();
    for fs in FeatureSet::ALL {
        let spec = fs.spec();
        let ft_in = spec.ft_in();
        for psv in &records {
            let board = psv.decode();
            let features = collect(&spec, &board);
            assert!(
                features.len() <= spec.max_active(),
                "{}: active 数 {} が max_active {} を超過",
                spec.canonical_name(),
                features.len(),
                spec.max_active(),
            );
            for &(stm, nstm) in &features {
                assert!(
                    stm < ft_in,
                    "{}: stm index {stm} >= ft_in {ft_in}",
                    spec.canonical_name()
                );
                assert!(
                    nstm < ft_in,
                    "{}: nstm index {nstm} >= ft_in {ft_in}",
                    spec.canonical_name()
                );
            }
        }
    }
}

#[test]
fn all_cells_indices_are_injective_per_perspective() {
    // 異なる駒は異なる index になる。玉 plane 畳み込み (merged) でも自玉と敵玉が
    // 同一 index へ衝突しないことを含む (両玉は常に別マス)。
    let records = sample_psv_records();
    for fs in FeatureSet::ALL {
        let spec = fs.spec();
        for psv in &records {
            let board = psv.decode();
            let features = collect(&spec, &board);
            let mut stm: Vec<usize> = features.iter().map(|&(s, _)| s).collect();
            let mut nstm: Vec<usize> = features.iter().map(|&(_, n)| n).collect();
            let n = stm.len();
            stm.sort_unstable();
            stm.dedup();
            nstm.sort_unstable();
            nstm.dedup();
            assert_eq!(
                stm.len(),
                n,
                "{}: stm index に重複 (衝突)",
                spec.canonical_name()
            );
            assert_eq!(
                nstm.len(),
                n,
                "{}: nstm index に重複 (衝突)",
                spec.canonical_name()
            );
        }
    }
}

// =============================================================================
// 退化一致 (新規 2 cell を参照照合済み cell に紐づける)
// =============================================================================

/// 軸 1 の退化検証: `merged_fs` は `split_fs` に「敵玉 plane の畳み込み」だけを
/// 加えた cell。軸 2 (king bucket / 筋ミラー) は両者同一なので、全特徴で
/// king bucket は一致し、packed BonaPiece は敵玉 (`>= E_KING`) のみ 81 ずれる。
fn assert_merged_is_split_plus_fold(split_fs: FeatureSet, merged_fs: FeatureSet) {
    let split = split_fs.spec();
    let merged = merged_fs.spec();
    for psv in sample_psv_records() {
        let board = psv.decode();
        let split_f = collect(&split, &board);
        let merged_f = collect(&merged, &board);
        assert_eq!(split_f.len(), merged_f.len(), "emission 長が不一致");
        for (&(ss, sn), &(ms, mn)) in split_f.iter().zip(&merged_f) {
            for (split_idx, merged_idx) in [(ss, ms), (sn, mn)] {
                let (split_kb, split_pp) = decompose(split_idx, split.piece_inputs());
                let (merged_kb, merged_pp) = decompose(merged_idx, merged.piece_inputs());
                assert_eq!(
                    split_kb, merged_kb,
                    "king bucket は king encoding に依存しない"
                );
                if split_pp >= E_KING as usize {
                    assert_eq!(merged_pp, split_pp - 81, "敵玉 plane は 81 畳まれる");
                } else {
                    assert_eq!(merged_pp, split_pp, "玉以外 / 自玉 plane は不変");
                }
            }
        }
    }
}

#[test]
fn degenerate_halfka_merged_is_halfka_split_plus_fold() {
    // halfka-merged (参照実装なし) を halfka-split (oracle 照合済み) に紐づける。
    assert_merged_is_split_plus_fold(FeatureSet::HalfKaSplit, FeatureSet::HalfKaMerged);
}

#[test]
fn degenerate_halfka_hm_split_is_halfka_hm_merged_minus_fold() {
    // halfka-hm-split (参照実装なし) を halfka-hm-merged (`ShogiHalfKA_hm` と
    // 照合済み) に紐づける。hm-merged は hm-split に fold を加えた cell。
    assert_merged_is_split_plus_fold(FeatureSet::HalfKaHmSplit, FeatureSet::HalfKaHmMerged);
}

// =============================================================================
// 玉 plane 衝突 (merged 系)
// =============================================================================

#[test]
fn merged_cells_fold_enemy_king_without_collision() {
    // merged 系で自玉特徴と敵玉特徴が plane 畳み込み後に同一 index へ衝突しない。
    // 自玉 / 敵玉は常に別マスなので、畳み込み後も packed は異なる。
    let positions = [
        build_board(Color::Black, Square::new(4, 8), Square::new(4, 0), &[]),
        build_board(Color::White, Square::new(2, 7), Square::new(6, 1), &[]),
        build_board(Color::Black, Square::new(0, 0), Square::new(8, 8), &[]),
    ];
    for fs in [FeatureSet::HalfKaMerged, FeatureSet::HalfKaHmMerged] {
        let spec = fs.spec();
        for board in &positions {
            let features = collect(&spec, board);
            // 盤上駒・手駒なし → 自玉ペア + 敵玉ペアの 2 特徴のみ。
            assert_eq!(features.len(), 2, "{}: 玉のみの局面", spec.canonical_name());
            let (friend_stm, _) = features[0];
            let (enemy_stm, _) = features[1];
            assert_ne!(
                friend_stm,
                enemy_stm,
                "{}: 自玉と敵玉が衝突",
                spec.canonical_name()
            );
            // 両玉とも自玉 plane ([FE_OLD_END, E_KING)) 内に畳まれている。
            for &(idx, _) in &features {
                let (_, packed) = decompose(idx, spec.piece_inputs());
                assert!(
                    (FE_OLD_END..E_KING as usize).contains(&packed),
                    "{}: 玉 packed {packed} が自玉 plane 外",
                    spec.canonical_name(),
                );
            }
        }
    }
}

// =============================================================================
// semantic 不変条件: perspective swap
// =============================================================================

#[test]
fn perspective_swap_swaps_stm_nstm_pairs() {
    // 手番を入れ替えると stm/nstm 視点が入れ替わり、各 (stm, nstm) ペアが
    // (nstm, stm) に整合して入れ替わる。全 5 cell で成立する。
    let records = sample_psv_records();
    for fs in FeatureSet::ALL {
        let spec = fs.spec();
        for psv in &records {
            let board = psv.decode();
            let original = collect(&spec, &board);
            let mut swapped_board = board.clone();
            swapped_board.side_to_move = board.side_to_move.opponent();
            let swapped = collect(&spec, &swapped_board);

            let mut want: Vec<(usize, usize)> = original.iter().map(|&(s, n)| (n, s)).collect();
            let mut got = swapped.clone();
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(
                got,
                want,
                "{}: perspective swap が非整合",
                spec.canonical_name()
            );
        }
    }
}

// =============================================================================
// semantic 不変条件: mirror involution (HorizontalMirror 系)
// =============================================================================

#[test]
fn horizontal_mirror_cells_are_invariant_under_file_mirror() {
    // HorizontalMirror 系では、局面を盤ごと筋ミラーした像が同一の active index
    // 集合を返す (玉が片側半面に来るよう正規化しているため)。
    //
    // この不変条件は両玉とも視点変換後の筋が中央 (file 4) 以外のときに成立する。
    // file 4 の玉は筋ミラーで動かず正規化が効かないため、玉が file 4 以外に来る
    // 局面で検証する。
    let pieces: &[(Color, PieceType, Square)] = &[
        (Color::Black, PieceType::Pawn, Square::new(2, 6)),
        (Color::Black, PieceType::Gold, Square::new(7, 7)),
        (Color::White, PieceType::Rook, Square::new(1, 1)),
        (Color::White, PieceType::Silver, Square::new(5, 3)),
    ];
    let positions = [
        build_board(Color::Black, Square::new(6, 8), Square::new(2, 0), pieces),
        build_board(Color::White, Square::new(1, 7), Square::new(7, 2), pieces),
        build_board(Color::Black, Square::new(0, 4), Square::new(8, 4), pieces),
    ];
    for fs in [FeatureSet::HalfKaHmSplit, FeatureSet::HalfKaHmMerged] {
        let spec = fs.spec();
        for board in &positions {
            let mut original = collect(&spec, board);
            let mut mirrored = collect(&spec, &mirror_board(board));
            original.sort_unstable();
            mirrored.sort_unstable();
            assert_eq!(
                original,
                mirrored,
                "{}: 筋ミラー像が非不変",
                spec.canonical_name(),
            );
            assert!(!original.is_empty());
        }
    }
}

#[test]
fn direct_cells_are_not_invariant_under_file_mirror() {
    // 対照: Direct 系 (ミラー無し) は筋ミラーで feature 集合が変わる。これにより
    // 上のテストが HorizontalMirror 固有の正規化を検証していることが分かる。
    let pieces: &[(Color, PieceType, Square)] =
        &[(Color::Black, PieceType::Pawn, Square::new(2, 6))];
    let board = build_board(Color::Black, Square::new(6, 8), Square::new(2, 0), pieces);
    for fs in [FeatureSet::HalfKaSplit, FeatureSet::HalfKaMerged] {
        let spec = fs.spec();
        let mut original = collect(&spec, &board);
        let mut mirrored = collect(&spec, &mirror_board(&board));
        original.sort_unstable();
        mirrored.sort_unstable();
        assert_ne!(
            original,
            mirrored,
            "{}: Direct 系なのに筋ミラー不変",
            spec.canonical_name(),
        );
    }
}

#[test]
fn effect_bucket_canonical_golden_halfka_hm_merged_2x2_kingfixed() {
    let pieces: &[(Color, PieceType, Square)] = &[
        (Color::Black, PieceType::Gold, Square::new(3, 7)),
        (Color::Black, PieceType::Silver, Square::new(4, 7)),
        (Color::Black, PieceType::Horse, Square::new(6, 6)),
        (Color::Black, PieceType::Pawn, Square::new(2, 5)),
        (Color::Black, PieceType::Pawn, Square::new(4, 5)),
        (Color::White, PieceType::Gold, Square::new(4, 1)),
        (Color::White, PieceType::Silver, Square::new(5, 1)),
        (Color::White, PieceType::Dragon, Square::new(2, 2)),
        (Color::White, PieceType::Pawn, Square::new(4, 3)),
        (Color::White, PieceType::Lance, Square::new(0, 4)),
        (Color::Black, PieceType::Pawn, Square::new(0, 6)),
    ];
    let board = build_board(Color::Black, Square::new(4, 8), Square::new(4, 0), pieces);
    let spec = FeatureSet::HalfKaHmMerged
        .spec()
        .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2);
    let (stm, nstm) = collect_effect_bucket_sorted(&spec, &board);

    assert_eq!(
        stm,
        vec![
            287_089, 287_157, 287_228, 287_544, 288_052, 289_182, 289_518, 289_794, 290_130,
            291_192, 292_653, 293_040, 293_072,
        ]
    );
    assert_eq!(
        nstm,
        vec![
            287_228, 287_544, 287_617, 287_685, 288_016, 289_146, 289_482, 289_830, 290_166,
            291_356, 292_489, 293_040, 293_072,
        ]
    );
}

/// 学習経路 (`extract_active_features` = `map_effect_bucket_features_board_both`) が cross-repo
/// golden で検証済みの単視点 dumper (`collect_effect_bucket_features_board`) と各視点で index 集合
/// bit 一致する不変条件。両者は別実装なので実局面で突き合わせないと契約が silent に割れる。
#[test]
fn effect_bucket_training_path_matches_golden_dumper_on_real_psv() {
    let configs = [
        EffectBucketConfig::KINGFIXED_2X2,
        EffectBucketConfig::KINGBUCKETED_2X2,
        EffectBucketConfig::KINGFIXED_3X3,
        EffectBucketConfig::KINGBUCKETED_3X3,
    ];
    let records = sample_psv_records();
    let mut checked = 0usize;
    for cfg in configs {
        let spec = FeatureSet::HalfKaHmMerged
            .spec()
            .with_effect_bucket_config(cfg);
        for (i, psv) in records.iter().enumerate() {
            let board = psv.decode();
            let stm = board.side_to_move;
            let (train_stm, train_nstm) = collect_effect_bucket_sorted(&spec, &board);
            let mut gold_stm =
                shogi_features::collect_effect_bucket_features_board(&board, cfg, stm);
            let mut gold_nstm =
                shogi_features::collect_effect_bucket_features_board(&board, cfg, stm.opponent());
            gold_stm.sort_unstable();
            gold_nstm.sort_unstable();
            assert_eq!(
                train_stm, gold_stm,
                "cfg {cfg:?} record {i}: stm 視点が golden dumper と不一致"
            );
            assert_eq!(
                train_nstm, gold_nstm,
                "cfg {cfg:?} record {i}: nstm 視点が golden dumper と不一致"
            );
            if !train_stm.is_empty() {
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "sample.psv に active effect bucket feature を持つ record が無い"
    );
}
