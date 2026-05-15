//! KP-absolute progress feature。
//!
//! 仕様:
//! - 特徴次元: `81 * FE_OLD_END = 81 * 1548 = 125_388` (玉位置 × BonaPiece)
//! - 学習形式: logistic regression (`p = sigmoid(Σ w_i * x_i)`)
//! - bucket 割当: `min(7, floor(p * 8.0))` (8 bucket)
//! - 重み読込: `progress.bin` (f64 little-endian × 125_388 個、`8 * 125_388 = 1_003_104` bytes 固定)
//!
//! 数式 / 定数の出典は bullet-shogi のオリジナル実装 (`ATTRIBUTION.md` 参照)。

use std::path::Path;
use std::sync::OnceLock;

use shogi_format::bona_piece::FE_OLD_END;
use shogi_format::types::{BOARD_PIECE_TYPES, HAND_PIECE_TYPES};
use shogi_format::{BonaPiece, Color, PackedSfenValue, Piece, ShogiBoard};

/// 8 bucket を採用 (progress を 0..=7 にマップ)。
pub const SHOGI_PROGRESS8_NUM_BUCKETS: usize = 8;

/// KP-absolute 特徴の次元数: `81 * FE_OLD_END`。
pub const SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS: usize = 81 * FE_OLD_END;

static SHOGI_PROGRESS_KP_ABS_WEIGHTS: OnceLock<Box<[f32]>> = OnceLock::new();
static SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS: [f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS] =
    [0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];

/// Progress-based 8 bucket assignment using KP-absolute features.
///
/// 重みはプロセス全体で 1 つ (`OnceLock` で保持) のため、本 struct は `Copy`
/// にできる。
#[derive(Clone, Copy, Default)]
pub struct ShogiProgressKPAbs;

impl ShogiProgressKPAbs {
    /// bucket 数 (型レベルで bucket 数を参照したい呼び出し側のための inherent
    /// const、自由関数定数 `SHOGI_PROGRESS8_NUM_BUCKETS` と同一値)。
    pub const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn weights() -> &'static [f32] {
        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .get()
            .map_or(&SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS, |weights| {
                weights.as_ref()
            })
    }

    /// 指定局面の KP-absolute 有効 index を全列挙し、`f` に渡す。
    ///
    /// `progress8kpabs` で使う特徴展開そのもの。
    pub fn for_each_active_index(pos: &PackedSfenValue, f: impl FnMut(usize)) {
        Self::for_each_active_index_board(&pos.decode(), f);
    }

    /// `for_each_active_index` の **decode 済み `ShogiBoard` を直接受ける** 版。
    ///
    /// dataloader が 1 局面につき `PackedSfenValue::decode()` を 1 回だけ呼んで
    /// その `ShogiBoard` を HalfKA_hm 特徴抽出と本 progress 計算の両方で使い回す
    /// ための入口 (`for_each_active_index` は `pos.decode()` 経由で本メソッドを
    /// 呼ぶのと等価)。
    pub fn for_each_active_index_board(board: &ShogiBoard, mut f: impl FnMut(usize)) {
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            return;
        }

        let sq_bk = board.black_king_sq.index();
        let sq_wk = board.white_king_sq.inverse().index();

        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);

                    let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }

        for owner in [Color::Black, Color::White] {
            let hand = if owner == Color::Black {
                board.black_hand
            } else {
                board.white_hand
            };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for c in 1..=count {
                    let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }
    }

    /// `for_each_active_index` の結果を `Vec` に集める。`out` は事前 clear される。
    pub fn collect_active_indices(pos: &PackedSfenValue, out: &mut Vec<usize>) {
        out.clear();
        Self::for_each_active_index(pos, |idx| out.push(idx));
    }

    /// `progress.bin` (f64 LE × `NUM_WEIGHTS`、`1_003_104` bytes 固定) を読み込む。
    ///
    /// プロセスでロード可能な KP-absolute モデルは 1 つだけ (二回目以降は Err)。
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
        if bytes.len() != expected {
            return Err(format!(
                "progress.bin size mismatch: got {} bytes, expected {}",
                bytes.len(),
                expected
            ));
        }

        let weights: Vec<f32> = bytes
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|chunk| {
                f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")) as f32
            })
            .collect();

        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .set(weights.into_boxed_slice())
            .map_err(|_| {
                "KP-absolute progress weights are already loaded in this process".to_string()
            })?;

        Ok(Self)
    }

    /// progress 推定 (`0.0..=1.0`)。重み未ロードでは常に 0.5 (sigmoid(0))。
    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        self.progress_board(&pos.decode())
    }

    /// `progress` の **decode 済み `ShogiBoard` を直接受ける** 版。
    pub fn progress_board(&self, board: &ShogiBoard) -> f32 {
        let weights = Self::weights();
        let mut sum = 0.0f32;
        Self::for_each_active_index_board(board, |idx| sum += weights[idx]);

        let p = 1.0 / (1.0 + (-sum).exp());
        p.clamp(0.0, 1.0)
    }

    /// 8 bucket 割当 (`0..=7`)。
    pub fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        self.bucket_board(&pos.decode())
    }

    /// `bucket` の **decode 済み `ShogiBoard` を直接受ける** 版。
    /// `bucket(&pos)` は `bucket_board(&pos.decode())` と等価。
    pub fn bucket_board(&self, board: &ShogiBoard) -> u8 {
        let p = self.progress_board(board);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// shogi-format crate の `tests/data/sample.psv` (100 records × 40 bytes)。
    fn sample_psv_records() -> Vec<PackedSfenValue> {
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let bytes =
            std::fs::read(&path).expect("sample.psv が読めない (../shogi-format/tests/data/)");
        assert_eq!(bytes.len() % 40, 0);
        assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (不正ビットパターン無し、align = 1)。bytes 長は 40 の倍数。
        let recs: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };
        recs.to_vec()
    }

    #[test]
    fn board_path_matches_legacy_path_on_real_psv() {
        // legacy delegating path (`for_each_active_index(&psv)` / `progress(&psv)` /
        // `bucket(&psv)` は内部で `psv.decode()` → board path) と board path
        // (`*_board(&psv.decode())`) が完全に一致する不変条件を実 PSV で確認する。
        //
        // NOTE: 重みは未ロード前提 (zero weights → progress = sigmoid(0) = 0.5、
        // bucket 4)。本 unit-test binary では `load_from_bin` を一切呼ばないので
        // `SHOGI_PROGRESS_KP_ABS_WEIGHTS` (OnceLock) は未 set のまま。
        let kpabs = ShogiProgressKPAbs;
        let records = sample_psv_records();
        let mut total_active = 0usize;
        for (i, psv) in records.iter().enumerate() {
            let board = psv.decode();

            let mut via_legacy = Vec::new();
            ShogiProgressKPAbs::for_each_active_index(psv, |idx| via_legacy.push(idx));
            let mut via_board = Vec::new();
            ShogiProgressKPAbs::for_each_active_index_board(&board, |idx| via_board.push(idx));
            assert_eq!(
                via_legacy, via_board,
                "record {i}: for_each_active_index の legacy/board path 不一致"
            );
            // 全 index が valid 範囲内。
            for &idx in &via_board {
                assert!(
                    idx < SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
                    "record {i}: idx {idx} 範囲外"
                );
            }
            total_active += via_board.len();

            assert_eq!(
                kpabs.progress(psv),
                kpabs.progress_board(&board),
                "record {i}: progress の legacy/board path 不一致"
            );
            assert_eq!(
                kpabs.bucket(psv),
                kpabs.bucket_board(&board),
                "record {i}: bucket の legacy/board path 不一致"
            );
            // zero weights → progress 厳密に 0.5、bucket 4。
            assert_eq!(kpabs.progress(psv), 0.5, "record {i}: zero-weight progress");
            assert_eq!(kpabs.bucket(psv), 4, "record {i}: zero-weight bucket");
        }
        assert!(
            total_active > 0,
            "sample.psv は実局面なので active index を持つ record があるはず"
        );
    }

    #[test]
    fn bucket_board_matches_bucket_zero_weights() {
        // 最小局面 (両玉のみ) でも legacy/board path が一致し、zero-weight で
        // progress = 0.5 / bucket = 4 になることを確認。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            ..Default::default()
        };
        board.black_king_sq = shogi_format::types::Square::new(4, 8);
        board.white_king_sq = shogi_format::types::Square::new(4, 0);
        let p = ShogiProgressKPAbs;
        assert_eq!(p.progress_board(&board), 0.5);
        assert_eq!(p.bucket_board(&board), 4);
    }
}
