//! PSQT shortcut の Material 初期化値計算。
//!
//! ## 駒価値 (centipawn)
//!
//! - 歩 (Pawn) = 100、香 (Lance) = 300、桂 (Knight) = 320
//! - 銀 (Silver) = 500、金 (Gold) = 550、角 (Bishop) = 850、飛 (Rook) = 1000
//! - 馬 (Horse) = 1020 (= 角 × 1.2)、龍 (Dragon) = 1200 (= 飛 × 1.2)
//! - 玉 (King) = 0
//!
//! 成歩 / 成香 / 成桂 / 成銀は BonaPiece 上で Gold スロットに統合されるため Gold
//! と同じ価値を共有する。
//!
//! ## piece input layout 別の対応
//!
//! 各 feature set の `piece_inputs` 長は `BonaPiece` の plane 構成で決まり、
//! 手駒 + 盤上駒 (`0..FE_OLD_END = 1548`) は全 feature set 共通、玉 plane の
//! 有無と plane 数だけが variant ごとに異なる:
//!
//! | feature set | piece_inputs | 玉 plane |
//! |---|---:|:--|
//! | `HalfKp` | 1548 | 無し |
//! | `HalfKaSplit` / `HalfKaHmSplit` | 1710 | F_KING (81) + E_KING (81) |
//! | `HalfKaMerged` / `HalfKaHmMerged` | 1629 | F_KING (81) のみ (敵玉は -81 で同 plane に畳む) |
//!
//! 玉 = 0 で全 variant 共通なので、玉 plane 領域は default `0.0` のまま残せば
//! 良い。
//!
//! 数式 / 定数の出典は bullet-shogi のオリジナル実装 (`ATTRIBUTION.md` 参照)。

use shogi_format::bona_piece::{
    E_BISHOP, E_DRAGON, E_GOLD, E_HAND_BISHOP, E_HAND_GOLD, E_HAND_KNIGHT, E_HAND_LANCE,
    E_HAND_PAWN, E_HAND_ROOK, E_HAND_SILVER, E_HORSE, E_KNIGHT, E_LANCE, E_PAWN, E_ROOK, E_SILVER,
    F_BISHOP, F_DRAGON, F_GOLD, F_HAND_BISHOP, F_HAND_GOLD, F_HAND_KNIGHT, F_HAND_LANCE,
    F_HAND_PAWN, F_HAND_ROOK, F_HAND_SILVER, F_HORSE, F_KNIGHT, F_LANCE, F_PAWN, F_ROOK, F_SILVER,
    FE_OLD_END,
};

use crate::feature_set::FeatureSetSpec;

/// 駒の centipawn 価値定数。
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

/// `packed_bonapiece` (0..piece_inputs) → 駒 Material 値 (符号付き centipawn) の
/// lookup table。手駒 + 盤上駒部分 (`0..FE_OLD_END`) は全 feature set 共通、
/// 玉 plane (`FE_OLD_END..`) は玉 = 0 で default 0 のまま残す。
fn build_packed_bp_material_table(piece_inputs: usize) -> Vec<f32> {
    use material_cp::*;

    assert!(
        piece_inputs >= FE_OLD_END,
        "piece_inputs ({piece_inputs}) must be >= FE_OLD_END ({FE_OLD_END})"
    );
    let mut table = vec![0.0f32; piece_inputs];

    let fill = |t: &mut [f32], base: u16, count: u16, value: f32| {
        for i in 0..count {
            t[(base + i) as usize] = value;
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

    // 玉 plane (`FE_OLD_END..piece_inputs`) は玉 = 0 で default 0 のまま。
    // HalfKp は piece_inputs = FE_OLD_END で玉 plane を持たない (no-op)。

    table
}

/// PSQT 重みの Material 初期値を `FeatureSetSpec` から計算する。
///
/// 戻り値は長さ `num_buckets * spec.ft_in()` の row-major
/// `out[feat * num_buckets + bucket]`。Material prior は bucket 軸に依らないので
/// `num_buckets` 軸は同一値で埋める。
///
/// # Panics
/// - `out_scaling <= 0.0`
/// - `num_buckets == 0`
/// - `spec.piece_inputs() < FE_OLD_END` (公開 5 feature set はいずれも満たす)
pub fn psqt_material_values(
    spec: &FeatureSetSpec,
    num_buckets: usize,
    out_scaling: f32,
) -> Vec<f32> {
    assert!(out_scaling > 0.0, "out_scaling must be positive");
    assert!(num_buckets > 0, "num_buckets must be positive");

    let piece_inputs = spec.piece_inputs();
    let king_buckets = spec.king_buckets();
    let input_size = spec.ft_in();

    let packed_material = build_packed_bp_material_table(piece_inputs);
    let mut vals = vec![0.0_f32; num_buckets * input_size];

    for kb in 0..king_buckets {
        for (bp, &material) in packed_material.iter().enumerate() {
            let feat = kb * piece_inputs + bp;
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
    use crate::FeatureSet;
    use crate::halfka_hm::PIECE_INPUTS;
    use shogi_format::bona_piece::F_KING;

    const TEST_NUM_BUCKETS: usize = 9;

    #[test]
    fn material_constants_pinned() {
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
    fn packed_table_friend_signs_halfka_merged() {
        let t = build_packed_bp_material_table(PIECE_INPUTS);
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
    fn packed_table_king_plane_is_zero() {
        // HalfKaMerged 系 (piece_inputs = 1629) の F_KING plane は 0。
        let t = build_packed_bp_material_table(PIECE_INPUTS);
        for (i, &v) in t
            .iter()
            .enumerate()
            .take(PIECE_INPUTS)
            .skip(F_KING as usize)
        {
            assert_eq!(v, 0.0, "king slot {i} must be 0");
        }
        // HalfKaSplit 系 (piece_inputs = 1710) の E_KING plane も 0。
        let t = build_packed_bp_material_table(1710);
        for (i, &v) in t.iter().enumerate().take(1710).skip(F_KING as usize) {
            assert_eq!(v, 0.0, "split king slot {i} must be 0");
        }
    }

    #[test]
    fn packed_table_halfkp_no_king_plane() {
        // HalfKp (piece_inputs = 1548 = FE_OLD_END) は玉 plane 無し。
        let t = build_packed_bp_material_table(FE_OLD_END);
        assert_eq!(t.len(), FE_OLD_END);
        // 手駒 / 盤上駒部分は埋まる、末尾の F_PAWN..E_DRAGON が centipawn 値で
        // 終端しているのを抽出 check。末尾 slot (E_DRAGON + 80) が FE_OLD_END
        // - 1 = 1547 になる。
        assert_eq!(t[F_PAWN as usize], 100.0);
        assert_eq!(t[(E_DRAGON + 80) as usize], -1200.0);
        assert_eq!(E_DRAGON as usize + 80 + 1, FE_OLD_END);
    }

    #[test]
    fn psqt_material_values_shape_all_feature_sets() {
        let scaling = 600.0;
        for fs in FeatureSet::ALL {
            let spec = fs.spec();
            let vals = psqt_material_values(&spec, TEST_NUM_BUCKETS, scaling);
            assert_eq!(
                vals.len(),
                TEST_NUM_BUCKETS * spec.ft_in(),
                "{}: vals length mismatch",
                fs.canonical_name()
            );
            // 全要素 finite。
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "{}: non-finite found",
                fs.canonical_name()
            );
        }
    }

    #[test]
    fn psqt_material_values_uniform_across_buckets_all_feature_sets() {
        let scaling = 600.0;
        for fs in FeatureSet::ALL {
            let spec = fs.spec();
            let vals = psqt_material_values(&spec, TEST_NUM_BUCKETS, scaling);
            // 任意 feat について 9 bucket 全部同じ値 (Material prior は bucket 軸に
            // 依らないため)。spec ごとに ft_in が違うので probe 位置も spec 内に
            // 収まるよう min を取る。
            let probes = [0usize, 100, 1000, 5000, spec.ft_in() - 1];
            for &feat in &probes {
                let base = feat * TEST_NUM_BUCKETS;
                let v0 = vals[base];
                for b in 1..TEST_NUM_BUCKETS {
                    assert_eq!(
                        vals[base + b],
                        v0,
                        "{} feat {feat} bucket {b} mismatch",
                        fs.canonical_name()
                    );
                }
            }
        }
    }

    #[test]
    fn psqt_material_values_specific_features_halfka_hm_merged() {
        let scaling = 600.0;
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let v = psqt_material_values(&spec, TEST_NUM_BUCKETS, scaling);
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
        let feat_king = 20 * PIECE_INPUTS + F_KING as usize;
        assert_eq!(v[feat_king * TEST_NUM_BUCKETS], 0.0);
    }

    /// 全 5 feature set で代表 feature の prior が hardcoded な centipawn / scaling
    /// と一致することを direct probe で確認する (helper 内のロジックが drift しても
    /// hardcoded 値とは独立なので oracle として機能する)。
    #[test]
    fn psqt_material_values_hardcoded_probes_all_feature_sets() {
        const SCALING: f32 = 600.0;
        const PAWN_PRIOR: f32 = 100.0 / SCALING; // friend pawn = +100 / 600
        const ROOK_PRIOR: f32 = 1000.0 / SCALING; // friend rook = +1000 / 600
        const E_DRAGON_PRIOR: f32 = -1200.0 / SCALING; // enemy dragon = -1200 / 600

        for fs in FeatureSet::ALL {
            let spec = fs.spec();
            let v = psqt_material_values(&spec, TEST_NUM_BUCKETS, SCALING);
            let pi = spec.piece_inputs();
            let nb = TEST_NUM_BUCKETS;
            let name = fs.canonical_name();

            // king_bucket=0 の F_HAND_PAWN 1 枚目 (+100/600)
            let feat = F_HAND_PAWN as usize;
            assert_eq!(v[feat * nb], PAWN_PRIOR, "{name}: F_HAND_PAWN");

            // king_bucket=0 の F_PAWN @ sq0 (+100/600)
            let feat = F_PAWN as usize;
            assert_eq!(v[feat * nb], PAWN_PRIOR, "{name}: F_PAWN");

            // king_bucket=0 の F_HAND_ROOK 1 枚目 (+1000/600)
            let feat = F_HAND_ROOK as usize;
            assert_eq!(v[feat * nb], ROOK_PRIOR, "{name}: F_HAND_ROOK");

            // king_bucket=0 の E_DRAGON @ sq80 (≈-2.0、float 丸めで 1e-5 内)
            let feat = (E_DRAGON + 80) as usize;
            assert!(
                (v[feat * nb] - E_DRAGON_PRIOR).abs() < 1e-5,
                "{name}: E_DRAGON@80"
            );

            // king_bucket=最後 (king_buckets - 1) の F_PAWN @ sq0 でも同じ prior
            let kb = spec.king_buckets() - 1;
            let feat = kb * pi + F_PAWN as usize;
            assert_eq!(v[feat * nb], PAWN_PRIOR, "{name}: last_kb F_PAWN");
        }
    }

    #[test]
    fn psqt_material_values_halfkp_pawn_and_dragon() {
        // HalfKp (piece_inputs = 1548、玉 plane 無し) でも手駒・盤上駒 prior
        // 初期化が正しく動くこと。
        let scaling = 600.0;
        let spec = FeatureSet::HalfKp.spec();
        let v = psqt_material_values(&spec, TEST_NUM_BUCKETS, scaling);
        let pi = spec.piece_inputs();
        assert_eq!(pi, FE_OLD_END);
        // king_bucket=0 の F_PAWN: +100/600
        let feat_pawn = F_PAWN as usize;
        assert_eq!(v[feat_pawn * TEST_NUM_BUCKETS], 100.0 / scaling);
        // king_bucket=40 の E_DRAGON: ≈ -2.0
        let feat_e_dragon = 40 * pi + E_DRAGON as usize;
        assert!((v[feat_e_dragon * TEST_NUM_BUCKETS] - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn psqt_material_values_split_king_planes_zero() {
        // HalfKaSplit / HalfKaHmSplit (piece_inputs = 1710) の F_KING / E_KING
        // plane が 0 のまま、手駒 / 盤上駒 prior は埋まること。
        let scaling = 600.0;
        for fs in [FeatureSet::HalfKaSplit, FeatureSet::HalfKaHmSplit] {
            let spec = fs.spec();
            let v = psqt_material_values(&spec, TEST_NUM_BUCKETS, scaling);
            let pi = spec.piece_inputs();
            assert_eq!(pi, 1710);
            // king_bucket=0 の F_PAWN: +100/600
            assert_eq!(
                v[F_PAWN as usize * TEST_NUM_BUCKETS],
                100.0 / scaling,
                "{}: F_PAWN init",
                fs.canonical_name()
            );
            // 玉 plane (FE_OLD_END..pi = 1548..1710) は 0。
            for feat_bp in FE_OLD_END..pi {
                let v0 = v[feat_bp * TEST_NUM_BUCKETS];
                assert_eq!(
                    v0,
                    0.0,
                    "{}: king plane bp {feat_bp} must be 0",
                    fs.canonical_name()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "out_scaling must be positive")]
    fn psqt_material_rejects_nonpositive_scaling() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let _ = psqt_material_values(&spec, TEST_NUM_BUCKETS, 0.0);
    }

    #[test]
    #[should_panic(expected = "num_buckets must be positive")]
    fn psqt_material_rejects_zero_buckets() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let _ = psqt_material_values(&spec, 0, 600.0);
    }
}
