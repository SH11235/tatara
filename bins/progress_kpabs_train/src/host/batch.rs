//! Batch builder for the KP-abs progress trainer。
//!
//! bullet-shogi 上流 (`shogi_progress_kpabs_train_cuda.rs::Batch`) を移植。
//! 1 batch は K games (= `games_per_step`) 分の position をまとめて、kernel に
//! 渡す flat 配列 (`indices`、`targets`、`per_pos_norm`) を構築する。
//!
//! ## targets の付与
//!
//! game-relative ラベル付け: 1 ゲーム長 `game_len` のうち `i` 番目の position
//! (0-indexed) に対し `y = i / (game_len - 1)`。先頭は 0、終端は 1、
//! 中盤は線形補間。`game_len == 1` (special case) は 0 を返す。
//!
//! ## per_pos_norm
//!
//! batch averaging のための正規化係数。「1 / (game_len * n_games_in_batch)」を
//! position ごとに持つ。各 game の終わりに `push_game` で
//! `1 / game_len` を埋め、`finalize` で全要素に `1 / n_games` を乗じて
//! 最終値にする (n_games が batch 確定までわからないため 2 段階)。

use shogi_features::ShogiProgressKPAbs;
use shogi_format::PackedSfenValue;

use super::MAX_INDS_PER_POS;

/// 1 batch 分の host-side 配列。kernel への送信前に `Vec` のまま GPU に転送する。
#[derive(Debug, Default)]
pub struct Batch {
    /// 長さ `n_positions * MAX_INDS_PER_POS`。padding は `-1`。
    pub indices: Vec<i32>,
    /// 長さ `n_positions`。`y = i / (game_len - 1)`。
    pub targets: Vec<f32>,
    /// 長さ `n_positions`。最終値は `1 / (game_len * n_games)`。
    pub per_pos_norm: Vec<f32>,
    /// position 数 (= 全 game の position 数の合計)。
    pub n_positions: usize,
    /// game 数 (`push_game` した回数のうち空でないもの)。
    pub n_games: usize,
}

impl Batch {
    /// 空の batch を作る。
    pub fn new() -> Self {
        Self::default()
    }

    /// 1 ゲーム分の position を追加する。`scratch` は `for_each_active_index`
    /// 結果の一時バッファ (呼び出し側で使い回し)。
    ///
    /// 空ゲーム (`game.is_empty()`) は何もせず early return。
    pub fn push_game(&mut self, game: &[PackedSfenValue], scratch: &mut Vec<usize>) {
        let game_len = game.len();
        if game_len == 0 {
            return;
        }
        self.n_games += 1;
        for (i, psv) in game.iter().enumerate() {
            let y = if game_len == 1 {
                0.0_f32
            } else {
                i as f32 / (game_len - 1) as f32
            };
            ShogiProgressKPAbs::collect_active_indices(psv, scratch);
            let mut row = [-1_i32; MAX_INDS_PER_POS];
            for (j, &idx) in scratch.iter().take(MAX_INDS_PER_POS).enumerate() {
                row[j] = idx as i32;
            }
            self.indices.extend_from_slice(&row);
            self.targets.push(y);
            // 暫定: 1/game_len。finalize() で 1/n_games を掛けて最終化。
            self.per_pos_norm.push(1.0_f32 / game_len as f32);
            self.n_positions += 1;
        }
    }

    /// `per_pos_norm` を最終形 `1 / (game_len * n_games)` にする。
    pub fn finalize(&mut self) {
        let inv_k = 1.0_f32 / self.n_games.max(1) as f32;
        for n in &mut self.per_pos_norm {
            *n *= inv_k;
        }
    }

    /// batch を空にして再利用できるようにする。allocation は保持。
    pub fn clear(&mut self) {
        self.indices.clear();
        self.targets.clear();
        self.per_pos_norm.clear();
        self.n_positions = 0;
        self.n_games = 0;
    }
}
