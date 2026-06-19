//! FT factorizer の fold / reduce kernel (`ft_fold_virtual` /
//! `ft_reduce_virtual_grad`) の reference CPU 実装。
//!
//! GPU 側 (`#[kernel]`) は `bins/nnue_train/src/kernels/` に定義されている
//! (cuda-oxide rustc-codegen-cuda backend は bin entry 経由で到達可能な kernel
//! しか PTX 化しないため)。
//!
//! ## アルゴリズム
//!
//! FT factorizer は学習時のみ仮想 P plane (`piece_inputs` 行) を FT weight の
//! 後ろに持つ。実特徴 index は全 feature set で `kb * piece_inputs + p` の形
//! なので、実特徴 1 つに対応する仮想特徴は piece plane `p = idx % piece_inputs`
//! で一意に決まる。この対応を sparse path に流す代わりに dense kernel 2 本で
//! 配線する:
//!
//! - **fold** (forward): base king-bucket セル `kb·pi + p` には
//!   `comb = w[同] + w[(ft_in + p)·ft_out + ri]` を畳む。threat real 行
//!   (`[base_ft_in, ft_in)`) は仮想行を持たないので `comb = w` で素通し。線形性に
//!   より base 実行の `Σ_active (w_real + w_virt) = Σ_active comb`、threat は
//!   そのまま。
//! - **reduce** (backward): `grad[(ft_in + p)·ft_out + ri] =
//!   Σ_{kb < base_ft_in/pi} grad[(kb·pi + p)·ft_out + ri]`。各仮想特徴の出現列は
//!   同 p を持つ **base** 実特徴の出現列の合併 (base 実 1 つにつき仮想 1 つ) なので
//!   仮想 index を sparse backward に流す直接 gather と数学的に等価 (f32 加算順
//!   のみ異なる)。threat real 行の勾配は仮想行に寄与せず不可触。
//!
//! weight / grad は column-major (`buf[feature * ft_out + ri]`)。`base_ft_in` は
//! 仮想行を持つ base king-bucket セル数、`ft_in` (= base + threat) が仮想 P plane
//! の手前。train 形状は `(ft_in + piece_inputs) × ft_out`、
//! `base_ft_in % piece_inputs == 0` が前提。threat 無効時は `base_ft_in == ft_in`。

/// Reference CPU 実装 (fold)。`comb` (export 形状 `ft_in * ft_out` = base + threat)
/// を全要素 overwrite する。base セル `[0, base_ft_in)` は仮想行を加算、threat 行
/// `[base_ft_in, ft_in)` は素通し。
///
/// 入力前提:
/// - `w.len() == (ft_in + piece_inputs) * ft_out` (train 形状)
/// - `comb.len() == ft_in * ft_out` (export 形状)
/// - `base_ft_in <= ft_in` かつ `base_ft_in % piece_inputs == 0`
pub fn ft_fold_virtual_cpu(
    w: &[f32],
    comb: &mut [f32],
    base_ft_in: usize,
    ft_in: usize,
    ft_out: usize,
    piece_inputs: usize,
) {
    for feature in 0..ft_in {
        for ri in 0..ft_out {
            let real = w[feature * ft_out + ri];
            comb[feature * ft_out + ri] = if feature < base_ft_in {
                let p = feature % piece_inputs;
                real + w[(ft_in + p) * ft_out + ri]
            } else {
                real // threat real 行: 仮想行を持たない
            };
        }
    }
}

/// Reference CPU 実装 (reduce)。`grad` の仮想 block (`ft_in..ft_in+piece_inputs`
/// 行) を **base** 実 block の king-bucket 方向和で overwrite する。base / threat
/// 実 block は読みのみ (threat 行は仮想行に寄与しない)。
///
/// 入力前提:
/// - `grad.len() == (ft_in + piece_inputs) * ft_out` (train 形状)
/// - `base_ft_in <= ft_in` かつ `base_ft_in % piece_inputs == 0`
pub fn ft_reduce_virtual_grad_cpu(
    grad: &mut [f32],
    base_ft_in: usize,
    ft_in: usize,
    ft_out: usize,
    piece_inputs: usize,
) {
    let n_kb = base_ft_in / piece_inputs;
    for p in 0..piece_inputs {
        for ri in 0..ft_out {
            let mut sum = 0.0_f32;
            for kb in 0..n_kb {
                sum += grad[(kb * piece_inputs + p) * ft_out + ri];
            }
            grad[(ft_in + p) * ft_out + ri] = sum;
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
        ft_fold_virtual_cpu(&w, &mut comb, FT_IN, FT_IN, FT_OUT, PI);

        let (real, train, real_nnz, train_nnz) = real_and_train_indices();
        let batch = 2;
        let mut out_fold = vec![0.0_f32; batch * FT_OUT];
        sparse_ft_forward_cpu(&comb, &real, &mut out_fold, batch, FT_OUT, FT_IN, real_nnz);
        let mut out_virtual = vec![0.0_f32; batch * FT_OUT];
        sparse_ft_forward_cpu(
            &w,
            &train,
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
        ft_reduce_virtual_grad_cpu(&mut grad_new, FT_IN, FT_IN, FT_OUT, PI);

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
        ft_fold_virtual_cpu(&w, &mut comb, FT_IN, FT_IN, FT_OUT, PI);
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
        ft_reduce_virtual_grad_cpu(&mut grad, FT_IN, FT_IN, FT_OUT, PI);
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
    // base king-bucket セル `[0, B)` の後ろに threat real 行 `[B, FT)`、その後ろに
    // 仮想 P plane `[FT, FT+PI)` が並ぶ layout で fold/reduce が range-aware に
    // 動くことを確認する。
    const B: usize = 6; // base (kb=3 × pi=2)
    const THREAT: usize = 4; // threat real 行
    const FT: usize = B + THREAT; // 10 (= 仮想行の手前)
    const COEXIST_ROWS: usize = FT + PI; // train 形状 12

    fn coexist_weights() -> Vec<f32> {
        (0..COEXIST_ROWS * FT_OUT)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.011)
            .collect()
    }

    #[test]
    fn fold_leaves_threat_rows_untouched() {
        let w = coexist_weights();
        let mut comb = vec![0.0_f32; FT * FT_OUT];
        ft_fold_virtual_cpu(&w, &mut comb, B, FT, FT_OUT, PI);
        // base セルは実行 + 同 p 仮想行。
        for feature in 0..B {
            let p = feature % PI;
            for ri in 0..FT_OUT {
                let want = w[feature * FT_OUT + ri] + w[(FT + p) * FT_OUT + ri];
                assert_eq!(comb[feature * FT_OUT + ri], want, "base {feature} ri {ri}");
            }
        }
        // threat 行は素通し (仮想行を加算しない)。
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
    fn reduce_uses_only_base_king_buckets_and_leaves_threat_grad() {
        let mut grad: Vec<f32> = (0..COEXIST_ROWS * FT_OUT)
            .map(|i| (i as f32 + 1.0) * 0.3)
            .collect();
        let real_snapshot = grad[..FT * FT_OUT].to_vec();
        ft_reduce_virtual_grad_cpu(&mut grad, B, FT, FT_OUT, PI);
        // base + threat 実 block は読みのみ (不変)。
        assert_eq!(
            &grad[..FT * FT_OUT],
            &real_snapshot[..],
            "実 block は reduce で不変"
        );
        // 仮想行は **base king-bucket のみ** の和 (threat 行は寄与しない)。
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
    fn fold_separation_threat_only_nonzero() {
        // base + 仮想 = 0、threat ≠ 0 → fold 後 threat row == 元 threat row。
        let mut w = vec![0.0_f32; COEXIST_ROWS * FT_OUT];
        for feature in B..FT {
            for ri in 0..FT_OUT {
                w[feature * FT_OUT + ri] = ((feature + ri) as f32 + 1.0) * 0.07;
            }
        }
        let mut comb = vec![0.0_f32; FT * FT_OUT];
        ft_fold_virtual_cpu(&w, &mut comb, B, FT, FT_OUT, PI);
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
    fn reduce_separation_threat_grad_gives_zero_virtual() {
        // base grad = 0・threat grad ≠ 0 → 仮想行 grad = 0 (threat は寄与しない)。
        let mut grad = vec![0.0_f32; COEXIST_ROWS * FT_OUT];
        for feature in B..FT {
            for ri in 0..FT_OUT {
                grad[feature * FT_OUT + ri] = ((feature * ri) as f32 + 1.0) * 0.5;
            }
        }
        ft_reduce_virtual_grad_cpu(&mut grad, B, FT, FT_OUT, PI);
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
}
