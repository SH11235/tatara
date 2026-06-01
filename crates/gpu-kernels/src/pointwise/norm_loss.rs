//! Norm loss (per-weight-group L2-norm regularisation) の reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn norm_loss_reduce` / `norm_loss_apply` は呼び出し元 bin
//! entry に inline 定義する (cuda-oxide rustc-codegen-cuda backend の bin-entry
//! 制約)。本 module は GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム (Georgiou et al. 2021, "Norm Loss")
//!
//! 各 weight group g (= 出力ニューロン 1 個に流れ込む重みベクトル、または bias
//! テンソル全体) の L2 norm を 1 (oblique manifold) へ緩く引き寄せる weight decay
//! 代替。optimizer step あたり 1 回、weight に乗法補正を掛ける。
//!
//! ```text
//! reduce (per group g):
//!     norm[g] = sqrt( Σ_pos weight[off(g, pos)]^2 )
//!
//! apply (per element):
//!     corr  = 2 * factor * (1 - 1 / (norm[g] + eps))
//!     mult  = 1 - lr * corr
//!     weight[off(g, pos)] *= mult
//! ```
//!
//! `norm[g] > 1` のとき `corr > 0` で `mult < 1` (縮小)、`norm[g] < 1` で
//! `corr < 0` の `mult > 1` (拡大)。両側から `norm[g] = 1` へ引き寄せる。
//!
//! ## generic group indexing
//!
//! weight tensor のレイアウトは 3 通りあるが、`(group_pitch, elem_stride,
//! group_len)` の 3 パラメータで element offset を統一表現する:
//!
//! ```text
//! off(g, pos) = g * group_pitch + pos * elem_stride   (g < n_groups, pos < group_len)
//! ```
//!
//! - **contiguous row** `[n_groups, group_len]` row-major (dense L1/L2/L3 weight、
//!   bucket 軸も row に畳んだ `[bucket*out, in]`): `group_pitch = group_len`,
//!   `elem_stride = 1`。1 行 = 1 出力ニューロンの入力次元ベクトル。
//! - **strided column** `[group_len, n_groups]` row-major (FT weight
//!   `[ft_in, ft_out]`、shared L1f weight `[ft_out, l1_out]`): `group_pitch = 1`,
//!   `elem_stride = n_groups`。1 列 = 1 出力ニューロンの入力次元ベクトル
//!   (列方向 stride `n_groups`)。
//! - **per-tensor scalar** (bias、1D テンソル全体で 1 norm): `n_groups = 1`,
//!   `group_pitch = 0` (= 任意、g は 0 のみ), `elem_stride = 1`,
//!   `group_len = テンソル長`。
//!
//! reference 実装は nnue-pytorch `ranger21.Ranger21` の `unit_norm` (2D linear は
//! `dim=1`、1D bias は scalar) と同じ粒度を上記パラメータで表す。

/// `n_groups` 個の weight group それぞれの L2 norm を計算する (`norm_loss_reduce`
/// kernel の reference)。
///
/// `norms.len() == n_groups`、各 group の element offset は
/// `g * group_pitch + pos * elem_stride` (`pos < group_len`)。重複・範囲外
/// アクセスが無いことは呼び出し側のパラメータ責務 (module doc の 3 レイアウトは
/// いずれも単射)。
pub fn norm_loss_compute_norms_cpu(
    weight: &[f32],
    norms: &mut [f32],
    n_groups: usize,
    group_pitch: usize,
    elem_stride: usize,
    group_len: usize,
) {
    for (g, norm) in norms.iter_mut().enumerate().take(n_groups) {
        let base = g * group_pitch;
        let mut sumsq = 0.0_f32;
        for pos in 0..group_len {
            let w = weight[base + pos * elem_stride];
            sumsq += w * w;
        }
        *norm = sumsq.sqrt();
    }
}

/// 事前計算した per-group norm を使い、各 weight に norm loss 補正
/// `*= 1 - lr * 2 * factor * (1 - 1/(norm + eps))` を掛ける (`norm_loss_apply`
/// kernel の reference)。
///
/// `lr == 0` または `factor == 0` で no-op (補正係数 1.0)。indexing は
/// [`norm_loss_compute_norms_cpu`] と同一。
#[allow(clippy::too_many_arguments)]
pub fn norm_loss_apply_cpu(
    weight: &mut [f32],
    norms: &[f32],
    factor: f32,
    lr: f32,
    eps: f32,
    n_groups: usize,
    group_pitch: usize,
    elem_stride: usize,
    group_len: usize,
) {
    for (g, &norm) in norms.iter().enumerate().take(n_groups) {
        let correction = 2.0_f32 * factor * (1.0_f32 - 1.0_f32 / (norm + eps));
        let mult = 1.0_f32 - lr * correction;
        let base = g * group_pitch;
        for pos in 0..group_len {
            weight[base + pos * elem_stride] *= mult;
        }
    }
}

/// reduce + apply を 1 関数に畳んだ便宜 reference (1 step 分)。`scratch_norms`
/// は呼び出し側が確保する長さ `n_groups` の作業領域。
#[allow(clippy::too_many_arguments)]
pub fn norm_loss_step_cpu(
    weight: &mut [f32],
    scratch_norms: &mut [f32],
    factor: f32,
    lr: f32,
    eps: f32,
    n_groups: usize,
    group_pitch: usize,
    elem_stride: usize,
    group_len: usize,
) {
    norm_loss_compute_norms_cpu(
        weight,
        scratch_norms,
        n_groups,
        group_pitch,
        elem_stride,
        group_len,
    );
    norm_loss_apply_cpu(
        weight,
        scratch_norms,
        factor,
        lr,
        eps,
        n_groups,
        group_pitch,
        elem_stride,
        group_len,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f64) -> bool {
        ((a as f64) - (b as f64)).abs() <= tol
    }

    /// contiguous row layout `[n_groups, group_len]`: 各行の L2 norm を独立計算。
    #[test]
    fn norms_contiguous_rows() {
        // 2 行 × 3 列。行0 = [3,4,0] (norm 5)、行1 = [0,0,2] (norm 2)。
        let w = vec![3.0, 4.0, 0.0, 0.0, 0.0, 2.0];
        let mut norms = vec![0.0; 2];
        norm_loss_compute_norms_cpu(&w, &mut norms, 2, 3, 1, 3);
        assert!(approx_eq(norms[0], 5.0, 1e-6));
        assert!(approx_eq(norms[1], 2.0, 1e-6));
    }

    /// strided column layout `[group_len, n_groups]`: 列ごとの L2 norm。
    /// FT weight `[ft_in, ft_out]` や L1f weight `[ft_out, l1_out]` の per-output
    /// norm に対応する (列 = 出力ニューロン、stride = n_groups)。
    #[test]
    fn norms_strided_columns() {
        // row-major [3 行, 2 列]:
        //   row0 = [3, 0]
        //   row1 = [4, 0]
        //   row2 = [0, 5]
        // 列0 = [3,4,0] (norm 5)、列1 = [0,0,5] (norm 5)。
        let w = vec![3.0, 0.0, 4.0, 0.0, 0.0, 5.0];
        let mut norms = vec![0.0; 2];
        // n_groups=2 (列数), group_pitch=1 (列の先頭は連続), elem_stride=2 (= 列数),
        // group_len=3 (行数)。
        norm_loss_compute_norms_cpu(&w, &mut norms, 2, 1, 2, 3);
        assert!(approx_eq(norms[0], 5.0, 1e-6));
        assert!(approx_eq(norms[1], 5.0, 1e-6));
    }

    /// per-tensor scalar (bias): テンソル全体で 1 つの norm。
    #[test]
    fn norm_per_tensor_scalar() {
        let w = vec![1.0, 2.0, 2.0]; // norm = 3
        let mut norms = vec![0.0; 1];
        norm_loss_compute_norms_cpu(&w, &mut norms, 1, 0, 1, 3);
        assert!(approx_eq(norms[0], 3.0, 1e-6));
    }

    /// norm > 1 の行は縮小、norm < 1 の行は拡大 (oblique manifold への引き寄せ)。
    #[test]
    fn apply_shrinks_large_norm_grows_small_norm() {
        // 行0 norm 5 (>1) → 縮小、行1 norm 0.5 (<1) → 拡大。
        let mut w = vec![3.0, 4.0, 0.0, 0.5, 0.0, 0.0];
        let mut norms = vec![0.0; 2];
        let factor = 1e-4_f32;
        let lr = 0.1_f32;
        let eps = 1e-8_f32;
        norm_loss_compute_norms_cpu(&w, &mut norms, 2, 3, 1, 3);
        let w_before = w.clone();
        norm_loss_apply_cpu(&mut w, &norms, factor, lr, eps, 2, 3, 1, 3);

        let mult0 = 1.0_f32 - lr * 2.0 * factor * (1.0 - 1.0 / (5.0 + eps));
        let mult1 = 1.0_f32 - lr * 2.0 * factor * (1.0 - 1.0 / (0.5 + eps));
        assert!(mult0 < 1.0, "norm 5 行は縮小 (mult<1)");
        assert!(mult1 > 1.0, "norm 0.5 行は拡大 (mult>1)");
        for i in 0..3 {
            assert!(approx_eq(w[i], w_before[i] * mult0, 1e-9));
        }
        for i in 3..6 {
            assert!(approx_eq(w[i], w_before[i] * mult1, 1e-9));
        }
    }

    /// norm == 1 の行 (oblique manifold 上) はほぼ不変 (eps 由来の極小補正のみ)。
    #[test]
    fn unit_norm_row_is_near_fixed_point() {
        let mut w = vec![0.6, 0.8]; // norm = 1
        let mut norms = vec![0.0; 1];
        let factor = 1e-4_f32;
        let lr = 0.1_f32;
        let eps = 1e-8_f32;
        norm_loss_compute_norms_cpu(&w, &mut norms, 1, 2, 1, 2);
        assert!(approx_eq(norms[0], 1.0, 1e-6));
        let before = w.clone();
        norm_loss_apply_cpu(&mut w, &norms, factor, lr, eps, 1, 2, 1, 2);
        // correction ≈ 2*factor*(1 - 1/(1+eps)) ≈ 2*factor*eps ≈ 0 → ほぼ不変。
        for i in 0..2 {
            assert!(approx_eq(w[i], before[i], 1e-7));
        }
    }

    /// `factor == 0` は no-op (default OFF 経路の bit-identical 担保に対応する性質)。
    #[test]
    fn zero_factor_is_noop() {
        let mut w = vec![3.0, 4.0, 7.0, -2.0];
        let before = w.clone();
        let mut norms = vec![0.0; 2];
        norm_loss_step_cpu(&mut w, &mut norms, 0.0, 0.1, 1e-8, 2, 2, 1, 2);
        assert_eq!(w, before);
    }

    /// `lr == 0` も no-op。
    #[test]
    fn zero_lr_is_noop() {
        let mut w = vec![3.0, 4.0, 7.0, -2.0];
        let before = w.clone();
        let mut norms = vec![0.0; 2];
        norm_loss_step_cpu(&mut w, &mut norms, 1e-4, 0.0, 1e-8, 2, 2, 1, 2);
        assert_eq!(w, before);
    }

    /// 全零 group は norm 0、correction = 2*factor*(1 - 1/eps) で大きな負補正に
    /// なるが、対象 weight も 0 なので結果は 0 のまま (NaN/Inf 汚染しない)。
    #[test]
    fn zero_group_stays_zero() {
        let mut w = vec![0.0, 0.0, 0.0];
        let mut norms = vec![0.0; 1];
        norm_loss_step_cpu(&mut w, &mut norms, 1e-4, 0.1, 1e-8, 1, 3, 1, 3);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
    }
}
