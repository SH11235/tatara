//! PSQT shortcut の Material 初期化値計算。
//!
//! HalfKA 特徴量 (`feat = king_bucket * PIECE_INPUTS + packed_bonapiece`) ごとに
//! 駒の centipawn 価値を 1 個割り当て、PSQT shortcut 重みの初期値として書き出す
//! ためのテーブル。bullet-shogi `examples/shogi_layerstack.rs` の `psqt_material`
//! モジュール + `compute_psqt_material_values` を移植したもの (`ATTRIBUTION.md`)。
//!
//! ## 駒価値 (centipawn)
//!
//! - 歩 (Pawn) = 100、香 (Lance) = 300、桂 (Knight) = 320
//! - 銀 (Silver) = 500、金 (Gold) = 550、角 (Bishop) = 850、飛 (Rook) = 1000
//! - 馬 (Horse) = 1020 (= 角 × 1.2)、龍 (Dragon) = 1200 (= 飛 × 1.2)
//! - 玉 (King) = 0
//!
//! 成歩 / 成香 / 成桂 / 成銀は BonaPiece 上で Gold スロットに統合されるため Gold と
//! 同じ価値を共有する。
//!
//! ## 出力 layout
//!
//! [`psqt_material_values`] は長さ `NUM_BUCKETS * input_size` (column-major
//! `[NUM_BUCKETS, input_size]` flatten) の `Vec<f32>` を返す。各 feature index
//! `feat` について `out[feat * NUM_BUCKETS .. feat * NUM_BUCKETS + NUM_BUCKETS]`
//! の全 bucket セルに同一の `material / out_scaling` を書き込む (Material prior は
//! bucket に依らず一定)。

use shogi_format::bona_piece::{
    E_BISHOP, E_DRAGON, E_GOLD, E_HAND_BISHOP, E_HAND_GOLD, E_HAND_KNIGHT, E_HAND_LANCE,
    E_HAND_PAWN, E_HAND_ROOK, E_HAND_SILVER, E_HORSE, E_KNIGHT, E_LANCE, E_PAWN, E_ROOK, E_SILVER,
    F_BISHOP, F_DRAGON, F_GOLD, F_HAND_BISHOP, F_HAND_GOLD, F_HAND_KNIGHT, F_HAND_LANCE,
    F_HAND_PAWN, F_HAND_ROOK, F_HAND_SILVER, F_HORSE, F_KNIGHT, F_LANCE, F_PAWN, F_ROOK, F_SILVER,
};

use crate::halfka_hm::PIECE_INPUTS;

/// 駒の centipawn 価値定数 (bullet-shogi `psqt_material` モジュール由来)。
pub mod material_cp {
    pub const PAWN: f32 = 100.0;
    pub const LANCE: f32 = 300.0;
    pub const KNIGHT: f32 = 320.0;
    pub const SILVER: f32 = 500.0;
    pub const GOLD: f32 = 550.0;
    pub const BISHOP: f32 = 850.0;
    pub const ROOK: f32 = 1000.0;
    /// 馬 = 角 × 1.2 = 1020。
    pub const HORSE: f32 = BISHOP * 1.2;
    /// 龍 = 飛 × 1.2 = 1200。
    pub const DRAGON: f32 = ROOK * 1.2;
}

/// `packed_bonapiece` (0..PIECE_INPUTS) → 駒の Material 値 (符号付き centipawn) の
/// lookup table を構築。`pack_bonapiece` 適用後の値を引数に取る (E_KING は F_KING 平面に
/// 既に畳まれている前提)。
///
/// friend 駒 = `+material`、enemy 駒 = `-material`、玉および空きスロットは 0。
fn build_packed_bp_material_table() -> [f32; PIECE_INPUTS] {
    use material_cp::*;

    let mut table = [0.0_f32; PIECE_INPUTS];

    let fill = |table: &mut [f32], base: u16, count: u16, value: f32| {
        for i in 0..count {
            table[(base + i) as usize] = value;
        }
    };

    // 手駒 (最大枚数: 歩 18 / 香桂銀金 4 / 角飛 2)。
    fill(&mut table, F_HAND_PAWN, 18, PAWN);
    fill(&mut table, E_HAND_PAWN, 18, -PAWN);
    fill(&mut table, F_HAND_LANCE, 4, LANCE);
    fill(&mut table, E_HAND_LANCE, 4, -LANCE);
    fill(&mut table, F_HAND_KNIGHT, 4, KNIGHT);
    fill(&mut table, E_HAND_KNIGHT, 4, -KNIGHT);
    fill(&mut table, F_HAND_SILVER, 4, SILVER);
    fill(&mut table, E_HAND_SILVER, 4, -SILVER);
    fill(&mut table, F_HAND_GOLD, 4, GOLD);
    fill(&mut table, E_HAND_GOLD, 4, -GOLD);
    fill(&mut table, F_HAND_BISHOP, 2, BISHOP);
    fill(&mut table, E_HAND_BISHOP, 2, -BISHOP);
    fill(&mut table, F_HAND_ROOK, 2, ROOK);
    fill(&mut table, E_HAND_ROOK, 2, -ROOK);

    // 盤上駒 (各駒種 81 マス)。成歩 / 成香 / 成桂 / 成銀は BonaPiece 上で Gold スロットに
    // 統合されるため F_GOLD / E_GOLD が成駒 4 種を吸収する。
    fill(&mut table, F_PAWN, 81, PAWN);
    fill(&mut table, E_PAWN, 81, -PAWN);
    fill(&mut table, F_LANCE, 81, LANCE);
    fill(&mut table, E_LANCE, 81, -LANCE);
    fill(&mut table, F_KNIGHT, 81, KNIGHT);
    fill(&mut table, E_KNIGHT, 81, -KNIGHT);
    fill(&mut table, F_SILVER, 81, SILVER);
    fill(&mut table, E_SILVER, 81, -SILVER);
    fill(&mut table, F_GOLD, 81, GOLD);
    fill(&mut table, E_GOLD, 81, -GOLD);
    fill(&mut table, F_BISHOP, 81, BISHOP);
    fill(&mut table, E_BISHOP, 81, -BISHOP);
    fill(&mut table, F_HORSE, 81, HORSE);
    fill(&mut table, E_HORSE, 81, -HORSE);
    fill(&mut table, F_ROOK, 81, ROOK);
    fill(&mut table, E_ROOK, 81, -ROOK);
    fill(&mut table, F_DRAGON, 81, DRAGON);
    fill(&mut table, E_DRAGON, 81, -DRAGON);

    // 玉 (F_KING..F_KING+81) は 0 のまま (table 初期値)。
    table
}

/// PSQT shortcut 重みの Material 初期値を計算する (HalfKA / HalfKA_hm 共通)。
///
/// 戻り値は column-major `[num_buckets, input_size]` flatten された
/// `Vec<f32>` で `out[feat * num_buckets + bucket]` の order。`input_size` は通常
/// `halfka_dim`、`input_size > halfka_dim` (Threat tail 等) のケースは tail を
/// `0` で残す。Material prior は bucket に依らないため `num_buckets` 軸は同一値で埋める。
///
/// # Panics
/// - `out_scaling <= 0.0`
/// - `input_size < halfka_dim`
/// - `halfka_dim % PIECE_INPUTS != 0`
pub fn psqt_material_values(
    halfka_dim: usize,
    input_size: usize,
    num_buckets: usize,
    out_scaling: f32,
) -> Vec<f32> {
    assert!(out_scaling > 0.0, "out_scaling must be positive");
    assert!(input_size >= halfka_dim, "input_size must be >= halfka_dim");
    assert_eq!(
        halfka_dim % PIECE_INPUTS,
        0,
        "halfka_dim must be a multiple of PIECE_INPUTS ({PIECE_INPUTS})"
    );

    let packed_material = build_packed_bp_material_table();
    let num_king_buckets = halfka_dim / PIECE_INPUTS;
    let mut vals = vec![0.0_f32; num_buckets * input_size];

    for kb in 0..num_king_buckets {
        for (bp, &material) in packed_material.iter().enumerate() {
            let feat = kb * PIECE_INPUTS + bp;
            let value = material / out_scaling;
            let base = feat * num_buckets;
            for slot in vals.iter_mut().skip(base).take(num_buckets) {
                *slot = value;
            }
        }
    }

    vals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::halfka_hm::HALFKA_HM_DIMENSIONS;

    const TEST_NUM_BUCKETS: usize = 9;

    #[test]
    fn material_constants_match_bullet() {
        use material_cp::*;
        assert_eq!(PAWN, 100.0);
        assert_eq!(LANCE, 300.0);
        assert_eq!(KNIGHT, 320.0);
        assert_eq!(SILVER, 500.0);
        assert_eq!(GOLD, 550.0);
        assert_eq!(BISHOP, 850.0);
        assert_eq!(ROOK, 1000.0);
        // BISHOP * 1.2 / ROOK * 1.2 は float の丸めで 0.5 ULP ずれるため近似比較。
        assert!((HORSE - 1020.0).abs() < 1e-3);
        assert!((DRAGON - 1200.0).abs() < 1e-3);
    }

    #[test]
    fn packed_table_friend_signs() {
        let t = build_packed_bp_material_table();
        // F_PAWN..+81 が +100、E_PAWN..+81 が -100。
        assert_eq!(t[F_PAWN as usize], 100.0);
        assert_eq!(t[(F_PAWN + 80) as usize], 100.0);
        assert_eq!(t[E_PAWN as usize], -100.0);
        // F_GOLD は成駒も吸収するため Gold 価値が 81 マス分連続。
        assert_eq!(t[F_GOLD as usize], 550.0);
        assert_eq!(t[(E_GOLD + 80) as usize], -550.0);
        // F_DRAGON は 1200。
        assert_eq!(t[F_DRAGON as usize], 1200.0);
        assert_eq!(t[E_DRAGON as usize], -1200.0);
    }

    #[test]
    fn packed_table_king_is_zero() {
        let t = build_packed_bp_material_table();
        // F_KING (1548) 以降の 81 マスは 0 (玉は寄与しない)。
        // E_KING は pack 後 F_KING 平面に畳まれているので range として等価。
        for (i, &v) in t.iter().enumerate().take(PIECE_INPUTS).skip(1548) {
            assert_eq!(v, 0.0, "king/empty slot {i} must be 0");
        }
    }

    #[test]
    fn psqt_material_values_shape() {
        let v = psqt_material_values(
            HALFKA_HM_DIMENSIONS,
            HALFKA_HM_DIMENSIONS,
            TEST_NUM_BUCKETS,
            600.0,
        );
        assert_eq!(v.len(), TEST_NUM_BUCKETS * HALFKA_HM_DIMENSIONS);
    }

    #[test]
    fn psqt_material_values_uniform_across_buckets() {
        let v = psqt_material_values(
            HALFKA_HM_DIMENSIONS,
            HALFKA_HM_DIMENSIONS,
            TEST_NUM_BUCKETS,
            600.0,
        );
        // 任意 feat について 9 bucket 全部同じ値であることを確認 (Material prior は bucket
        // 軸に依らないため)。
        for feat in [0_usize, 100, 1000, 5000, HALFKA_HM_DIMENSIONS - 1] {
            let base = feat * TEST_NUM_BUCKETS;
            let v0 = v[base];
            for b in 1..TEST_NUM_BUCKETS {
                assert_eq!(v[base + b], v0, "feat {feat} bucket {b} mismatch");
            }
        }
    }

    #[test]
    fn psqt_material_values_specific_features() {
        let scaling = 600.0;
        let v = psqt_material_values(
            HALFKA_HM_DIMENSIONS,
            HALFKA_HM_DIMENSIONS,
            TEST_NUM_BUCKETS,
            scaling,
        );
        // king_bucket=0, packed_bp=F_PAWN+0 (=90): friend Pawn = +100/600。
        let kb = 0;
        let feat_pawn = kb * PIECE_INPUTS + F_PAWN as usize;
        assert_eq!(v[feat_pawn * TEST_NUM_BUCKETS], 100.0 / scaling);
        // king_bucket=3, packed_bp=E_DRAGON+0: enemy Dragon = -1200/600 ≈ -2.0
        // (Dragon = ROOK * 1.2 で float 丸めにより 0.5 ULP のずれが生じる)。
        let feat_e_dragon = 3 * PIECE_INPUTS + E_DRAGON as usize;
        assert!((v[feat_e_dragon * TEST_NUM_BUCKETS] - (-2.0)).abs() < 1e-5);
        // king_bucket=10, packed_bp=F_HAND_PAWN+5 (handful 内): +100/600。
        let feat_hp = 10 * PIECE_INPUTS + (F_HAND_PAWN + 5) as usize;
        assert_eq!(v[feat_hp * TEST_NUM_BUCKETS], 100.0 / scaling);
        // king_bucket=20, packed_bp=F_KING+0: 玉 = 0。
        let feat_king = 20 * PIECE_INPUTS + 1548;
        assert_eq!(v[feat_king * TEST_NUM_BUCKETS], 0.0);
    }

    #[test]
    fn psqt_material_values_tail_zero_when_input_larger() {
        // input_size > halfka_dim のとき末尾の Threat tail 領域は 0 で残る。
        let halfka = HALFKA_HM_DIMENSIONS;
        let extra = 100;
        let v = psqt_material_values(halfka, halfka + extra, TEST_NUM_BUCKETS, 600.0);
        assert_eq!(v.len(), TEST_NUM_BUCKETS * (halfka + extra));
        for feat in halfka..halfka + extra {
            for b in 0..TEST_NUM_BUCKETS {
                assert_eq!(v[feat * TEST_NUM_BUCKETS + b], 0.0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "out_scaling must be positive")]
    fn psqt_material_rejects_nonpositive_scaling() {
        let _ = psqt_material_values(
            HALFKA_HM_DIMENSIONS,
            HALFKA_HM_DIMENSIONS,
            TEST_NUM_BUCKETS,
            0.0,
        );
    }

    #[test]
    #[should_panic(expected = "input_size must be >= halfka_dim")]
    fn psqt_material_rejects_smaller_input() {
        let _ = psqt_material_values(
            HALFKA_HM_DIMENSIONS,
            HALFKA_HM_DIMENSIONS - 1,
            TEST_NUM_BUCKETS,
            600.0,
        );
    }
}
