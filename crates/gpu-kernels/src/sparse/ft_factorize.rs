//! FT factorizer の fold / reduce kernel (`ft_fold_virtual` /
//! `ft_reduce_virtual_grad`) の reference CPU 実装。
//!
//! GPU 側 (`#[kernel]`) は `bins/nnue_train/src/kernels/` に定義されている
//! (cuda-oxide rustc-codegen-cuda backend は bin entry 経由で到達可能な kernel
//! しか PTX 化しないため)。
//!
//! ## アルゴリズム
//!
//! FT factorizer は学習時のみ virtual piece-input rows と virtual threat-pair rows を
//! FT weight の後ろに持つ。base feature は駒ごとに 1 仮想行を base の
//! king-bucket 数 (HalfKA_hm 系では 45) で共有する。effect bucket は各駒特徴を
//! NB 個の被攻撃×被防御バケットに分割するため、mode が piece-input 仮想行を
//! effect bucket でも pool するかを決める。threat feature は pair prefix table で
//! 同じ pair に属する行を 1 つの virtual threat-pair row へ対応させる。この対応を
//! sparse path に流す代わりに dense kernel 2 本で配線する:
//!
//! - **fold** (forward): base feature には対応する仮想行を畳む。effect bucket の
//!   PoolEffectBuckets は `virtual_row = (feat/NB)%piece_inputs` を使い、
//!   駒ごとに 1 仮想行を全バケットで共有する。PerEffectBucket は
//!   `virtual_row = ((feat/NB)%piece_inputs)*NB + feat%NB` を使い、
//!   (駒, バケット) ごとに仮想行を持つ。threat real 行
//!   (`[base_ft_in, ft_in)`) は同じ pair の仮想行を畳み込む。線形性に
//!   より実行部の `Σ_active (w_real + w_virt) = Σ_active comb` と等価。
//! - **reduce** (backward): 各仮想行の勾配を、同じ仮想行へ対応する **base**
//!   実特徴の勾配和で埋める。仮想特徴の出現列は対応する base 実特徴の出現列の
//!   合併なので、仮想 index を sparse backward に流す直接 gather と数学的に等価
//!   (f32 加算順のみ異なる)。threat pair 仮想行は同じ pair に属する threat 実行の
//!   勾配和で埋める。
//!
//! weight / grad は column-major (`buf[feature * ft_out + ri]`)。`base_ft_in` は
//! 仮想行を持つ base 実行の行数、`ft_in` (= base + threat) が piece-input 仮想行
//! の手前。train 形状は
//! `(ft_in + base_virtual_rows + threat_pair_starts.len() - 1) × ft_out`、
//! `base_ft_in % piece_inputs == 0` が前提。threat 無効時は `base_ft_in == ft_in`。

pub const FT_FACTORIZE_BASE: u32 = 0;
pub const FT_FACTORIZE_POOL_EFFECT_BUCKETS: u32 = 1;
pub const FT_FACTORIZE_PER_EFFECT_BUCKET: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct FtFactorizeLayout<'a> {
    pub base_ft_in: usize,
    pub ft_in: usize,
    pub ft_out: usize,
    pub piece_inputs: usize,
    pub nb: usize,
    pub mode: u32,
    pub threat_pair_starts: &'a [usize],
}

pub fn base_virtual_rows(piece_inputs: usize, nb: usize, mode: u32) -> usize {
    match mode {
        FT_FACTORIZE_PER_EFFECT_BUCKET => piece_inputs * nb,
        FT_FACTORIZE_BASE | FT_FACTORIZE_POOL_EFFECT_BUCKETS => piece_inputs,
        _ => panic!("unknown FT factorizer mode"),
    }
}

fn virtual_row(feature: usize, piece_inputs: usize, nb: usize, mode: u32) -> usize {
    match mode {
        FT_FACTORIZE_BASE => feature % piece_inputs,
        FT_FACTORIZE_POOL_EFFECT_BUCKETS => (feature / nb) % piece_inputs,
        FT_FACTORIZE_PER_EFFECT_BUCKET => ((feature / nb) % piece_inputs) * nb + feature % nb,
        _ => panic!("unknown FT factorizer mode"),
    }
}

/// Reference CPU 実装 (fold)。`comb` (export 形状 `ft_in * ft_out` = base + threat)
/// を全要素 overwrite する。base セル `[0, base_ft_in)` は virtual piece-input rows
/// を加算、threat 行 `[base_ft_in, ft_in)` は virtual threat-pair rows を加算する。
///
/// 入力前提:
/// - `w.len() == (ft_in + base_virtual_rows + threat_pair_starts.len() - 1) * ft_out`
///   (train 形状)
/// - `comb.len() == ft_in * ft_out` (export 形状)
/// - `base_ft_in <= ft_in` かつ `base_ft_in % piece_inputs == 0`
pub fn ft_fold_virtual_cpu(w: &[f32], comb: &mut [f32], layout: FtFactorizeLayout<'_>) {
    let base_vrows = base_virtual_rows(layout.piece_inputs, layout.nb, layout.mode);
    let threat_virtual_base = layout.ft_in + base_vrows;
    for feature in 0..layout.ft_in {
        for ri in 0..layout.ft_out {
            let real = w[feature * layout.ft_out + ri];
            comb[feature * layout.ft_out + ri] = if feature < layout.base_ft_in {
                let vrow = virtual_row(feature, layout.piece_inputs, layout.nb, layout.mode);
                real + w[(layout.ft_in + vrow) * layout.ft_out + ri]
            } else if layout.threat_pair_starts.len() >= 2 {
                let rel = feature - layout.base_ft_in;
                let pair = layout
                    .threat_pair_starts
                    .partition_point(|&start| start <= rel)
                    - 1;
                real + w[(threat_virtual_base + pair) * layout.ft_out + ri]
            } else {
                real
            };
        }
    }
}

/// Reference CPU 実装 (reduce)。`grad` の仮想 block (`ft_in..`) を overwrite する。
/// virtual piece-input rows は **base** 実行部の対応行の和、virtual threat-pair rows は
/// 同じ pair に属する threat 実行の和で埋める。base / threat 実行部は読みのみ。
///
/// 入力前提:
/// - `grad.len() == (ft_in + base_virtual_rows + threat_pair_starts.len() - 1) * ft_out`
///   (train 形状)
/// - `base_ft_in <= ft_in` かつ `base_ft_in % piece_inputs == 0`
pub fn ft_reduce_virtual_grad_cpu(grad: &mut [f32], layout: FtFactorizeLayout<'_>) {
    let base_vrows = base_virtual_rows(layout.piece_inputs, layout.nb, layout.mode);
    for vrow in 0..base_vrows {
        for ri in 0..layout.ft_out {
            let mut sum = 0.0_f32;
            for feature in 0..layout.base_ft_in {
                if virtual_row(feature, layout.piece_inputs, layout.nb, layout.mode) == vrow {
                    sum += grad[feature * layout.ft_out + ri];
                }
            }
            grad[(layout.ft_in + vrow) * layout.ft_out + ri] = sum;
        }
    }
    let threat_virtual_base = layout.ft_in + base_vrows;
    for pair in 0..layout.threat_pair_starts.len().saturating_sub(1) {
        let start = layout.base_ft_in + layout.threat_pair_starts[pair];
        let end = layout.base_ft_in + layout.threat_pair_starts[pair + 1];
        for ri in 0..layout.ft_out {
            let mut sum = 0.0_f32;
            for feature in start..end {
                sum += grad[feature * layout.ft_out + ri];
            }
            grad[(threat_virtual_base + pair) * layout.ft_out + ri] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::sparse_ft_backward::sparse_ft_backward_cpu;
    use crate::sparse::sparse_ft_forward::sparse_ft_forward_cpu;

    // 小次元 fixture: ft_in = 6 (kb=3 × pi=2)、ft_out = 4、train 行数 8。
    const FT_IN: usize = 6;
    const FT_OUT: usize = 4;
    const PI: usize = 2;
    const TRAIN_ROWS: usize = FT_IN + PI;

    fn base_layout() -> FtFactorizeLayout<'static> {
        FtFactorizeLayout {
            base_ft_in: FT_IN,
            ft_in: FT_IN,
            ft_out: FT_OUT,
            piece_inputs: PI,
            nb: 1,
            mode: FT_FACTORIZE_BASE,
            threat_pair_starts: &[],
        }
    }

    fn train_weights() -> Vec<f32> {
        (0..TRAIN_ROWS * FT_OUT)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect()
    }

    /// batch 2 件の実 index 列 (`-1` padding 込み) と、旧意味論 (実 + 仮想を
    /// 1:1 で append、train 次元) の index 列のペア。
    fn real_and_train_indices() -> (Vec<i32>, Vec<i32>, usize, usize) {
        let real_nnz = 3;
        let real: Vec<i32> = vec![0, 4, 5, /* pos1 */ 2, 3, -1];
        let mut train = Vec::new();
        for pos in real.chunks(real_nnz) {
            let mut row: Vec<i32> = pos.to_vec();
            for &idx in pos {
                row.push(if idx >= 0 {
                    FT_IN as i32 + idx % PI as i32
                } else {
                    -1
                });
            }
            train.extend(row);
        }
        (real, train, real_nnz, real_nnz * 2)
    }

    fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{label} len");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol * scale,
                "{label}[{i}]: got {g}, want {w}"
            );
        }
    }

    /// fold した comb で base forward した結果が、train 重みに仮想 index 込みの
    /// 旧意味論 forward と一致する (加算順差のみ、tolerance 比較)。
    #[test]
    fn fold_forward_matches_virtual_index_forward() {
        let w = train_weights();
        let mut comb = vec![0.0_f32; FT_IN * FT_OUT];
        ft_fold_virtual_cpu(&w, &mut comb, base_layout());

        let (real, train, real_nnz, train_nnz) = real_and_train_indices();
        let batch = 2;
        let mut out_fold = vec![0.0_f32; batch * FT_OUT];
        sparse_ft_forward_cpu(
            &comb,
            &real,
            &[real_nnz as i32; 2],
            &mut out_fold,
            batch,
            FT_OUT,
            FT_IN,
            real_nnz,
        );
        let mut out_virtual = vec![0.0_f32; batch * FT_OUT];
        sparse_ft_forward_cpu(
            &w,
            &train,
            &[train_nnz as i32; 2],
            &mut out_virtual,
            batch,
            FT_OUT,
            TRAIN_ROWS,
            train_nnz,
        );
        assert_close("fold forward", &out_fold, &out_virtual, 1e-6);
    }

    /// 実 block のみの backward + reduce が、仮想 index 込みの旧意味論 backward
    /// と一致する: 実 block は完全一致 (同一演算列)、仮想 block は加算順差のみ。
    #[test]
    fn reduce_matches_virtual_index_backward() {
        let (real, train, real_nnz, train_nnz) = real_and_train_indices();
        let batch = 2;
        let grad_out: Vec<f32> = (0..batch * FT_OUT)
            .map(|i| (i as f32 + 1.0) * 0.25)
            .collect();

        let mut grad_new = vec![0.0_f32; TRAIN_ROWS * FT_OUT];
        sparse_ft_backward_cpu(
            &grad_out,
            &real,
            &mut grad_new,
            batch,
            FT_OUT,
            FT_IN,
            real_nnz,
        );
        ft_reduce_virtual_grad_cpu(&mut grad_new, base_layout());

        let mut grad_virtual = vec![0.0_f32; TRAIN_ROWS * FT_OUT];
        sparse_ft_backward_cpu(
            &grad_out,
            &train,
            &mut grad_virtual,
            batch,
            FT_OUT,
            TRAIN_ROWS,
            train_nnz,
        );

        assert_eq!(
            &grad_new[..FT_IN * FT_OUT],
            &grad_virtual[..FT_IN * FT_OUT],
            "実 block は仮想 index の有無に依存しない"
        );
        assert_close(
            "reduce virtual block",
            &grad_new[FT_IN * FT_OUT..],
            &grad_virtual[FT_IN * FT_OUT..],
            1e-6,
        );
    }

    /// fold の基準値: 実行 + 同 p の仮想行の和を 1 cell ずつ検算。
    #[test]
    fn fold_adds_matching_virtual_row() {
        let w = train_weights();
        let mut comb = vec![0.0_f32; FT_IN * FT_OUT];
        ft_fold_virtual_cpu(&w, &mut comb, base_layout());
        for feature in 0..FT_IN {
            let p = feature % PI;
            for ri in 0..FT_OUT {
                let want = w[feature * FT_OUT + ri] + w[(FT_IN + p) * FT_OUT + ri];
                assert_eq!(
                    comb[feature * FT_OUT + ri],
                    want,
                    "feature {feature} ri {ri}"
                );
            }
        }
    }

    /// reduce の基準値: 仮想行 p = kb 方向の実行和、実 block は不変。
    #[test]
    fn reduce_sums_over_king_buckets() {
        let mut grad: Vec<f32> = (0..TRAIN_ROWS * FT_OUT).map(|i| i as f32).collect();
        let real_snapshot = grad[..FT_IN * FT_OUT].to_vec();
        ft_reduce_virtual_grad_cpu(&mut grad, base_layout());
        assert_eq!(&grad[..FT_IN * FT_OUT], &real_snapshot[..]);
        for p in 0..PI {
            for ri in 0..FT_OUT {
                let want: f32 = (0..FT_IN / PI)
                    .map(|kb| ((kb * PI + p) * FT_OUT + ri) as f32)
                    .sum();
                assert_eq!(grad[(FT_IN + p) * FT_OUT + ri], want, "p {p} ri {ri}");
            }
        }
    }

    // ---- threat 同居 (base_ft_in < ft_in) ----
    // base 実行 `[0, B)` の後ろに threat real 行 `[B, FT)`、その後ろに
    // piece-input 仮想行の `[FT, FT+PI)` が並ぶ layout で fold/reduce が range-aware に
    // 動くことを確認する。
    const B: usize = 6; // base (kb=3 × pi=2)
    const THREAT: usize = 4; // threat real 行
    const FT: usize = B + THREAT; // 10 (= 仮想行の手前)
    const COEXIST_ROWS: usize = FT + PI; // train 形状 12

    fn coexist_layout() -> FtFactorizeLayout<'static> {
        FtFactorizeLayout {
            base_ft_in: B,
            ft_in: FT,
            ft_out: FT_OUT,
            piece_inputs: PI,
            nb: 1,
            mode: FT_FACTORIZE_BASE,
            threat_pair_starts: &[],
        }
    }

    fn coexist_weights() -> Vec<f32> {
        (0..COEXIST_ROWS * FT_OUT)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.011)
            .collect()
    }

    #[test]
    fn fold_without_threat_pair_table_leaves_threat_rows_untouched() {
        let w = coexist_weights();
        let mut comb = vec![0.0_f32; FT * FT_OUT];
        ft_fold_virtual_cpu(&w, &mut comb, coexist_layout());
        // base セルは実行 + 同 p 仮想行。
        for feature in 0..B {
            let p = feature % PI;
            for ri in 0..FT_OUT {
                let want = w[feature * FT_OUT + ri] + w[(FT + p) * FT_OUT + ri];
                assert_eq!(comb[feature * FT_OUT + ri], want, "base {feature} ri {ri}");
            }
        }
        // pair table が無い場合、threat 行は素通し (仮想行を加算しない)。
        for feature in B..FT {
            for ri in 0..FT_OUT {
                assert_eq!(
                    comb[feature * FT_OUT + ri],
                    w[feature * FT_OUT + ri],
                    "threat {feature} ri {ri} は不可触のはず"
                );
            }
        }
    }

    #[test]
    fn reduce_without_threat_pair_table_uses_only_base_rows() {
        let mut grad: Vec<f32> = (0..COEXIST_ROWS * FT_OUT)
            .map(|i| (i as f32 + 1.0) * 0.3)
            .collect();
        let real_snapshot = grad[..FT * FT_OUT].to_vec();
        ft_reduce_virtual_grad_cpu(&mut grad, coexist_layout());
        // base + threat 実 block は読みのみ (不変)。
        assert_eq!(
            &grad[..FT * FT_OUT],
            &real_snapshot[..],
            "実 block は reduce で不変"
        );
        // pair table が無い場合、仮想行は base 実行のみの和。
        for p in 0..PI {
            for ri in 0..FT_OUT {
                let want: f32 = (0..B / PI)
                    .map(|kb| real_snapshot[(kb * PI + p) * FT_OUT + ri])
                    .sum();
                assert_eq!(grad[(FT + p) * FT_OUT + ri], want, "virtual p {p} ri {ri}");
            }
        }
    }

    #[test]
    fn fold_without_threat_pair_table_keeps_threat_only_rows() {
        // pair table が無い場合、base + 仮想 = 0、threat != 0 なら fold 後 threat row == 元 threat row。
        let mut w = vec![0.0_f32; COEXIST_ROWS * FT_OUT];
        for feature in B..FT {
            for ri in 0..FT_OUT {
                w[feature * FT_OUT + ri] = ((feature + ri) as f32 + 1.0) * 0.07;
            }
        }
        let mut comb = vec![0.0_f32; FT * FT_OUT];
        ft_fold_virtual_cpu(&w, &mut comb, coexist_layout());
        for feature in 0..B {
            for ri in 0..FT_OUT {
                assert_eq!(comb[feature * FT_OUT + ri], 0.0, "base は 0 のまま");
            }
        }
        for feature in B..FT {
            for ri in 0..FT_OUT {
                assert_eq!(comb[feature * FT_OUT + ri], w[feature * FT_OUT + ri]);
            }
        }
    }

    #[test]
    fn reduce_without_threat_pair_table_ignores_threat_grad() {
        // pair table が無い場合、base grad = 0・threat grad != 0 なら仮想行 grad = 0。
        let mut grad = vec![0.0_f32; COEXIST_ROWS * FT_OUT];
        for feature in B..FT {
            for ri in 0..FT_OUT {
                grad[feature * FT_OUT + ri] = ((feature * ri) as f32 + 1.0) * 0.5;
            }
        }
        ft_reduce_virtual_grad_cpu(&mut grad, coexist_layout());
        for p in 0..PI {
            for ri in 0..FT_OUT {
                assert_eq!(
                    grad[(FT + p) * FT_OUT + ri],
                    0.0,
                    "virtual p {p} ri {ri} は 0"
                );
            }
        }
    }

    #[test]
    fn threat_pair_fold_adds_variable_width_pair_virtual_rows() {
        const STARTS: [usize; 4] = [0, 1, 3, THREAT];
        const ROWS: usize = FT + PI + 3;
        let mut w = vec![0.0_f32; ROWS * FT_OUT];
        for feature in 0..FT {
            for ri in 0..FT_OUT {
                w[feature * FT_OUT + ri] = feature as f32 + ri as f32 * 0.25;
            }
        }
        for p in 0..PI {
            for ri in 0..FT_OUT {
                w[(FT + p) * FT_OUT + ri] = 100.0 + p as f32 + ri as f32 * 0.5;
            }
        }
        for pair in 0..3 {
            for ri in 0..FT_OUT {
                w[(FT + PI + pair) * FT_OUT + ri] = 1000.0 + pair as f32 * 10.0 + ri as f32 * 0.75;
            }
        }

        let mut comb = vec![0.0_f32; FT * FT_OUT];
        ft_fold_virtual_cpu(
            &w,
            &mut comb,
            FtFactorizeLayout {
                threat_pair_starts: &STARTS,
                ..coexist_layout()
            },
        );

        for pair in 0..3 {
            for rel in STARTS[pair]..STARTS[pair + 1] {
                let feature = B + rel;
                for ri in 0..FT_OUT {
                    assert_eq!(
                        comb[feature * FT_OUT + ri],
                        w[feature * FT_OUT + ri] + w[(FT + PI + pair) * FT_OUT + ri]
                    );
                }
            }
        }
    }

    #[test]
    fn threat_pair_reduce_sums_variable_width_pairs() {
        const STARTS: [usize; 4] = [0, 1, 3, THREAT];
        const ROWS: usize = FT + PI + 3;
        let mut grad: Vec<f32> = (0..ROWS * FT_OUT)
            .map(|i| (i as f32 + 1.0) * 0.125)
            .collect();
        let real_snapshot = grad[..FT * FT_OUT].to_vec();
        ft_reduce_virtual_grad_cpu(
            &mut grad,
            FtFactorizeLayout {
                threat_pair_starts: &STARTS,
                ..coexist_layout()
            },
        );

        assert_eq!(&grad[..FT * FT_OUT], &real_snapshot[..]);
        for pair in 0..3 {
            for ri in 0..FT_OUT {
                let want: f32 = (STARTS[pair]..STARTS[pair + 1])
                    .map(|rel| real_snapshot[(B + rel) * FT_OUT + ri])
                    .sum();
                assert_eq!(grad[(FT + PI + pair) * FT_OUT + ri], want);
            }
        }
    }

    #[test]
    fn effect_bucket_fold_maps_virtual_rows_by_mode() {
        const KB: usize = 3;
        const NB: usize = 4;
        const EFFECT_BUCKET_FT: usize = KB * PI * NB;
        let mut w = vec![0.0_f32; (EFFECT_BUCKET_FT + PI * NB) * FT_OUT];
        for feature in 0..EFFECT_BUCKET_FT {
            for ri in 0..FT_OUT {
                w[feature * FT_OUT + ri] = feature as f32 + ri as f32 * 0.25;
            }
        }
        for p in 0..PI {
            for bucket in 0..NB {
                for ri in 0..FT_OUT {
                    w[(EFFECT_BUCKET_FT + p * NB + bucket) * FT_OUT + ri] =
                        1000.0 + (p * 10 + bucket) as f32 + ri as f32 * 0.5;
                }
            }
        }

        let mut attack = vec![0.0_f32; EFFECT_BUCKET_FT * FT_OUT];
        ft_fold_virtual_cpu(
            &w[..(EFFECT_BUCKET_FT + PI) * FT_OUT],
            &mut attack,
            FtFactorizeLayout {
                base_ft_in: EFFECT_BUCKET_FT,
                ft_in: EFFECT_BUCKET_FT,
                ft_out: FT_OUT,
                piece_inputs: PI,
                nb: NB,
                mode: FT_FACTORIZE_POOL_EFFECT_BUCKETS,
                threat_pair_starts: &[],
            },
        );
        let mut bucketed = vec![0.0_f32; EFFECT_BUCKET_FT * FT_OUT];
        ft_fold_virtual_cpu(
            &w,
            &mut bucketed,
            FtFactorizeLayout {
                base_ft_in: EFFECT_BUCKET_FT,
                ft_in: EFFECT_BUCKET_FT,
                ft_out: FT_OUT,
                piece_inputs: PI,
                nb: NB,
                mode: FT_FACTORIZE_PER_EFFECT_BUCKET,
                threat_pair_starts: &[],
            },
        );

        for feature in 0..EFFECT_BUCKET_FT {
            let p = (feature / NB) % PI;
            let bucket = feature % NB;
            for ri in 0..FT_OUT {
                assert_eq!(
                    attack[feature * FT_OUT + ri],
                    w[feature * FT_OUT + ri] + w[(EFFECT_BUCKET_FT + p) * FT_OUT + ri]
                );
                assert_eq!(
                    bucketed[feature * FT_OUT + ri],
                    w[feature * FT_OUT + ri]
                        + w[(EFFECT_BUCKET_FT + p * NB + bucket) * FT_OUT + ri]
                );
            }
        }
    }

    #[test]
    fn effect_bucket_reduce_sums_expected_axes() {
        const KB: usize = 3;
        const NB: usize = 4;
        const EFFECT_BUCKET_FT: usize = KB * PI * NB;
        let mut attack_grad: Vec<f32> = (0..(EFFECT_BUCKET_FT + PI) * FT_OUT)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let attack_snapshot = attack_grad[..EFFECT_BUCKET_FT * FT_OUT].to_vec();
        ft_reduce_virtual_grad_cpu(
            &mut attack_grad,
            FtFactorizeLayout {
                base_ft_in: EFFECT_BUCKET_FT,
                ft_in: EFFECT_BUCKET_FT,
                ft_out: FT_OUT,
                piece_inputs: PI,
                nb: NB,
                mode: FT_FACTORIZE_POOL_EFFECT_BUCKETS,
                threat_pair_starts: &[],
            },
        );
        for p in 0..PI {
            for ri in 0..FT_OUT {
                let mut want = 0.0_f32;
                for kb in 0..KB {
                    for bucket in 0..NB {
                        want += attack_snapshot[((kb * PI + p) * NB + bucket) * FT_OUT + ri];
                    }
                }
                assert_eq!(attack_grad[(EFFECT_BUCKET_FT + p) * FT_OUT + ri], want);
            }
        }

        let mut bucket_grad: Vec<f32> = (0..(EFFECT_BUCKET_FT + PI * NB) * FT_OUT)
            .map(|i| (i as f32 + 1.0) * 0.02)
            .collect();
        let bucket_snapshot = bucket_grad[..EFFECT_BUCKET_FT * FT_OUT].to_vec();
        ft_reduce_virtual_grad_cpu(
            &mut bucket_grad,
            FtFactorizeLayout {
                base_ft_in: EFFECT_BUCKET_FT,
                ft_in: EFFECT_BUCKET_FT,
                ft_out: FT_OUT,
                piece_inputs: PI,
                nb: NB,
                mode: FT_FACTORIZE_PER_EFFECT_BUCKET,
                threat_pair_starts: &[],
            },
        );
        for p in 0..PI {
            for bucket in 0..NB {
                for ri in 0..FT_OUT {
                    let want: f32 = (0..KB)
                        .map(|kb| bucket_snapshot[((kb * PI + p) * NB + bucket) * FT_OUT + ri])
                        .sum();
                    assert_eq!(
                        bucket_grad[(EFFECT_BUCKET_FT + p * NB + bucket) * FT_OUT + ri],
                        want
                    );
                }
            }
        }
    }
}
