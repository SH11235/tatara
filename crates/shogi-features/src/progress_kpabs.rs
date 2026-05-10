//! KP-absolute progress feature。
//!
//! bullet-shogi (commit `f275eb9`) の
//! `crates/bullet_lib/src/game/outputs.rs::ShogiProgressKPAbs` から vendor。
//!
//! 仕様:
//! - 特徴次元: `81 * FE_OLD_END = 81 * 1548 = 125_388` (玉位置 × BonaPiece)
//! - 学習形式: logistic regression (`p = sigmoid(Σ w_i * x_i)`)
//! - bucket 割当: `min(7, floor(p * 8.0))` (8 bucket)
//! - 重み読込: YaneuraOu 互換 `progress.bin` (f64 little-endian × 125_388 個)
//!
//! bullet 上流からの差分:
//! - bullet `OutputBuckets` trait の `impl` を削除。`bucket()` を inherent
//!   method として残し、bullet trait に依存しないようにする (本リポ
//!   `ATTRIBUTION.md` の「取り込み済 file」セクション参照)。

use std::path::Path;
use std::sync::OnceLock;

use shogi_format::bona_piece::FE_OLD_END;
use shogi_format::types::{BOARD_PIECE_TYPES, HAND_PIECE_TYPES};
use shogi_format::{BonaPiece, Color, PackedSfenValue, Piece};

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
    /// bucket 数。bullet 上流の `OutputBuckets::BUCKETS` 相当を inherent const
    /// として持つ (型レベルで bucket 数を参照したい下流 crate の利便性のため、
    /// 自由関数定数 `SHOGI_PROGRESS8_NUM_BUCKETS` と同一値)。
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
    pub fn for_each_active_index(pos: &PackedSfenValue, mut f: impl FnMut(usize)) {
        let board = pos.decode();
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

    /// YaneuraOu 互換 `progress.bin` (f64 LE × `NUM_WEIGHTS`) を読み込む。
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
        let weights = Self::weights();
        let mut sum = 0.0f32;
        Self::for_each_active_index(pos, |idx| sum += weights[idx]);

        let p = 1.0 / (1.0 + (-sum).exp());
        p.clamp(0.0, 1.0)
    }

    /// 8 bucket 割当 (`0..=7`)。
    ///
    /// bullet-shogi 上流では `OutputBuckets::bucket` trait method 経由で呼ぶが、
    /// 本リポでは bullet trait 依存を避けるため inherent method として提供する。
    pub fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as u8
    }
}
