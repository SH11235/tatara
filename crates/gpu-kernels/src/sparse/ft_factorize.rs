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
//! - **fold** (forward): `comb[(kb·pi + p)·ft_out + ri] = w[同] +
//!   w[(ft_in + p)·ft_out + ri]`。線形性により `Σ_active (w_real + w_virt) =
//!   Σ_active comb` で、sparse forward は base の index 列のまま factorizer の
//!   forward 寄与を得る
//! - **reduce** (backward): `grad[(ft_in + p)·ft_out + ri] =
//!   Σ_kb grad[(kb·pi + p)·ft_out + ri]`。各仮想特徴の出現列は同 p を持つ実
//!   特徴の出現列の合併 (実 1 つにつき仮想ちょうど 1 つ) なので、仮想 index を
//!   sparse backward に流す直接 gather と数学的に等価 (f32 加算順のみ異なる)
//!
//! weight / grad は column-major (`buf[feature * ft_out + ri]`)、train 形状は
//! `(ft_in + piece_inputs) × ft_out`、`ft_in % piece_inputs == 0` が前提。

/// Reference CPU 実装 (fold)。`comb` (base 形状 `ft_in * ft_out`) を全要素
/// overwrite する。
///
/// 入力前提:
/// - `w.len() == (ft_in + piece_inputs) * ft_out` (train 形状)
/// - `comb.len() == ft_in * ft_out` (base 形状)
/// - `ft_in % piece_inputs == 0`
pub fn ft_fold_virtual_cpu(
    w: &[f32],
    comb: &mut [f32],
    ft_in: usize,
    ft_out: usize,
    piece_inputs: usize,
) {
    for feature in 0..ft_in {
        let p = feature % piece_inputs;
        for ri in 0..ft_out {
            comb[feature * ft_out + ri] = w[feature * ft_out + ri] + w[(ft_in + p) * ft_out + ri];
        }
    }
}

/// Reference CPU 実装 (reduce)。`grad` の仮想 block (`ft_in..ft_in+piece_inputs`
/// 行) を実 block の king-bucket 方向和で overwrite する。実 block は読みのみ。
///
/// 入力前提:
/// - `grad.len() == (ft_in + piece_inputs) * ft_out` (train 形状)
/// - `ft_in % piece_inputs == 0`
pub fn ft_reduce_virtual_grad_cpu(
    grad: &mut [f32],
    ft_in: usize,
    ft_out: usize,
    piece_inputs: usize,
) {
    let n_kb = ft_in / piece_inputs;
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
        ft_fold_virtual_cpu(&w, &mut comb, FT_IN, FT_OUT, PI);

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
        ft_reduce_virtual_grad_cpu(&mut grad_new, FT_IN, FT_OUT, PI);

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
        ft_fold_virtual_cpu(&w, &mut comb, FT_IN, FT_OUT, PI);
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
        ft_reduce_virtual_grad_cpu(&mut grad, FT_IN, FT_OUT, PI);
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
}
