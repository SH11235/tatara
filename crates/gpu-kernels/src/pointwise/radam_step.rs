//! Fused RAdam optimizer step (AdamW + bias correction + denom switch) の
//! reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn radam_step` は呼び出し元 bin entry に inline 定義する
//! (cuda-oxide rustc-codegen-cuda backend の bin-entry 制約)。本 module は
//! GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム
//!
//! RAdam (Rectified Adam, Liu et al. 2019)。Adam の bias correction を `beta2_t`
//! 由来の variance 補正係数 `step_size` に統合し、学習初期 (small `step`) で
//! variance が不安定なときは `1 / sqrt(v)` の正規化を **off** にして m のみで
//! 更新する。
//!
//! ```text
//! host pre-compute (per step、scalar):
//!     beta2_t   = beta2 ^ step
//!     n_sma_max = 2 / (1 - beta2) - 1
//!     n_sma     = n_sma_max - 2 * step * beta2_t / (1 - beta2_t)
//!     bc1       = 1 - beta1 ^ step
//!     if n_sma > n_sma_threshold (5.0):
//!         p1 = (n_sma - 4) / (n_sma_max - 4)
//!         p2 = (n_sma - 2) / n_sma
//!         p3 = n_sma_max / (n_sma_max - 2)
//!         step_size = sqrt((1 - beta2_t) * p1 * p2 * p3) / bc1
//!         denom     = 1
//!     else:
//!         step_size = 1 / bc1
//!         denom     = 0
//!
//! kernel (per element i):
//!     g          = grad[i]
//!     rate       = lr * step_size
//!     weights[i] *= 1 - decay * rate            # weight decay (AdamW base)
//!     m[i]       = beta1 * m[i] + (1 - beta1) * g
//!     v[i]       = beta2 * v[i] + (1 - beta2) * g * g
//!     val        = m[i]
//!     if denom != 0:
//!         val   /= sqrt(v[i]) + eps             # variance 補正 ON
//!     weights[i] -= rate * val
//!     weights[i] = clamp(weights[i], min_w, max_w)
//!     grad[i]    = 0                             # 次 batch 用に reset
//! ```
//!
//! - **bias correction を `step_size` に畳み込む**: `adamw_step` (bias correction
//!   なし) との最大の差分。`step_size` に `bc1 = 1 - beta1^t` の inverse +
//!   variance 補正係数 `sqrt((1-beta2_t)*...)` を取り込み、kernel 側は
//!   `rate = lr * step_size` 1 個で受ける
//! - **denom switch**: 学習初期で variance が不安定なときは `denom = 0` で
//!   `1/sqrt(v)` を **off**、十分に accumulate された後 (n_sma > threshold) は
//!   `denom = 1` で通常 Adam-like update
//! - **n_sma threshold**: 論文 default `5.0`
//! - host pre-compute の `step_size`, `denom` は per-step scalar、kernel に値渡し
//!   (1-element device buffer は使わない設計)
//!
//! ## cuda-oxide 制限
//!
//! - GPU kernel 側は `f32::clamp` を使えないため `if-else` ladder で展開
//! - `f32::sqrt` は `__nv_sqrtf` (libdevice) に lowering される
//! - host pre-compute fn (`radam_compute_step_size_denom`) は host 実行で
//!   `f32::powf` / `f32::sqrt` を自由に使える (kernel 内では使わない)

/// RAdam の host pre-compute: step number から `step_size` と `denom` を計算する。
///
/// 数式は module doc のアルゴリズム節 (RAdam, Liu et al. 2019) と同一。
///
/// 入力前提:
/// - `step >= 1` (1-indexed。`step = 0` で `beta^0 = 1` となり `bc1 = 0`、
///   `step_size = 1/0 = +inf` になるので呼び出し側で避ける)
/// - `0 < beta1 < 1`, `0 < beta2 < 1`
/// - `n_sma_threshold > 0` (論文 default `5.0`)
///
/// 戻り値: `(step_size, denom)` (denom は 0 or 1 の i32、kernel 引数として
/// そのまま渡す前提)。
pub fn radam_compute_step_size_denom(
    step: u64,
    beta1: f32,
    beta2: f32,
    n_sma_threshold: f32,
) -> (f32, i32) {
    let step_f = step as f32;
    let beta2_t = beta2.powf(step_f);
    let n_sma_max = 2.0_f32 / (1.0_f32 - beta2) - 1.0_f32;
    let n_sma = n_sma_max - 2.0_f32 * step_f * beta2_t / (1.0_f32 - beta2_t);
    let bc1 = 1.0_f32 - beta1.powf(step_f);

    let step_size = if n_sma > n_sma_threshold {
        let p1 = (n_sma - 4.0_f32) / (n_sma_max - 4.0_f32);
        let p2 = (n_sma - 2.0_f32) / n_sma;
        let p3 = n_sma_max / (n_sma_max - 2.0_f32);
        ((1.0_f32 - beta2_t) * p1 * p2 * p3).sqrt() / bc1
    } else {
        1.0_f32 / bc1
    };
    let denom = i32::from(n_sma > n_sma_threshold);
    (step_size, denom)
}

/// Reference CPU 実装。
///
/// In-place mutation:
/// - `weights[i]`: weight decay → bias-corrected RAdam update → clamp の 3 段
/// - `m[i]` / `v[i]`: Adam 1 次 / 2 次 moment running average
/// - `grad[i]`: 0.0 にリセット (次 batch の accumulation 用)
///
/// 入力前提:
/// - `weights.len() == m.len() == v.len() == grad.len() == n`
/// - `lr ≥ 0`, `decay ≥ 0`, `0 < beta1 < 1`, `0 < beta2 < 1`, `eps > 0`
/// - `min_w ≤ max_w` (host 側不変条件)
/// - `step_size > 0`, `denom ∈ {0, 1}` (host pre-compute による、
///   `radam_compute_step_size_denom` の戻り値をそのまま渡す前提)
///
/// 引数 14 個は RAdam の host pre-compute 引数を畳み込んだ形。
/// `clippy::too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn radam_step_cpu(
    weights: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &mut [f32],
    lr: f32,
    step_size: f32,
    denom: i32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_w: f32,
    max_w: f32,
    n: usize,
) {
    let rate = lr * step_size;
    for i in 0..n {
        let g = grad[i];
        let mut p = weights[i];
        p *= 1.0_f32 - decay * rate;
        let mi = beta1 * m[i] + (1.0_f32 - beta1) * g;
        let vi = beta2 * v[i] + (1.0_f32 - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        let mut val = mi;
        if denom != 0 {
            val /= vi.sqrt() + eps;
        }
        p -= rate * val;
        weights[i] = p.clamp(min_w, max_w);
        grad[i] = 0.0_f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: f32, b: f32, tol: f64) -> bool {
        ((a as f64) - (b as f64)).abs() <= tol
    }

    /// `radam_compute_step_size_denom`: 学習初期 (step=1) で n_sma が threshold を
    /// 下回り `denom = 0` (`1/sqrt(v)` off)、`step_size = 1 / bc1` になる。
    /// RAdam 式から手計算で確認。
    #[test]
    fn step_size_denom_step_1_disables_variance() {
        let (step_size, denom) = radam_compute_step_size_denom(1, 0.9_f32, 0.999_f32, 5.0_f32);
        // step=1 で beta2_t = 0.999、n_sma_max = 2/0.001 - 1 ≈ 1999、
        // n_sma = 1999 - 2 * 1 * 0.999 / 0.001 = 1999 - 1998 = 1 (< 5.0)
        // bc1 = 1 - 0.9 = 0.1、step_size = 1 / 0.1 = 10
        assert_eq!(denom, 0);
        let bc1 = 1.0_f32 - 0.9_f32.powf(1.0_f32);
        let expected = 1.0_f32 / bc1;
        assert!(
            approx_eq_f32(step_size, expected, 1e-6),
            "step_size: got {step_size} exp {expected}"
        );
    }

    /// `radam_compute_step_size_denom`: 大きな step では n_sma が threshold を
    /// 超え `denom = 1`、`step_size = sqrt((1 - beta2_t) * p1*p2*p3) / bc1` の
    /// 形になる。step=1000 で手計算。
    #[test]
    fn step_size_denom_large_step_enables_variance() {
        let step = 1000_u64;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let n_sma_threshold = 5.0_f32;
        let (step_size, denom) = radam_compute_step_size_denom(step, beta1, beta2, n_sma_threshold);

        // RAdam 式を独立に再計算
        let step_f = step as f32;
        let beta2_t = beta2.powf(step_f);
        let n_sma_max = 2.0_f32 / (1.0_f32 - beta2) - 1.0_f32;
        let n_sma = n_sma_max - 2.0_f32 * step_f * beta2_t / (1.0_f32 - beta2_t);
        assert!(
            n_sma > n_sma_threshold,
            "expected n_sma > {n_sma_threshold}, got {n_sma}"
        );
        let bc1 = 1.0_f32 - beta1.powf(step_f);
        let p1 = (n_sma - 4.0_f32) / (n_sma_max - 4.0_f32);
        let p2 = (n_sma - 2.0_f32) / n_sma;
        let p3 = n_sma_max / (n_sma_max - 2.0_f32);
        let expected_ss = ((1.0_f32 - beta2_t) * p1 * p2 * p3).sqrt() / bc1;
        assert_eq!(denom, 1);
        assert!(
            approx_eq_f32(step_size, expected_ss, 1e-6),
            "step_size: got {step_size} exp {expected_ss}"
        );
    }

    /// 学習初期 (step=1, denom=0) で `1/sqrt(v)` が off、weights は m から
    /// `weights -= rate * m` の形で更新される (variance gating 確認)。
    /// w = 1.0、g = 0.1、lr = 0.1、step_size = 10 → rate = 1.0
    /// m = 0.9*0 + 0.1*0.1 = 0.01
    /// v = 0.999*0 + 0.001*0.01 = 1e-5  (denom=0 で使われない)
    /// val = m = 0.01 (denom=0 なので /= は skip)
    /// w = 1.0 (no decay) - 1.0 * 0.01 = 0.99
    #[test]
    fn one_step_denom_zero_skips_variance() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![0.1_f32];
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1,  // lr
            10.0, // step_size
            0,    // denom (variance off)
            0.0,  // decay
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            1,
        );
        // expected: rate = 0.1 * 10 = 1.0、val = m = 0.01、w = 1 - 1.0 * 0.01 = 0.99
        assert!(
            approx_eq_f32(weights[0], 0.99_f32, 1e-6),
            "got {} exp 0.99",
            weights[0]
        );
        assert!(approx_eq_f32(m[0], 0.01_f32, 1e-7));
        assert!(approx_eq_f32(v[0], 1e-5_f32, 1e-9));
        assert_eq!(grad, vec![0.0_f32]);
    }

    /// 通常領域 (denom=1) で AdamW base + variance 補正がかかる。
    /// w = 1.0、m_init = 0.01、v_init = 1e-5、g = 0.1、lr = 0.1、step_size = 1.0、
    /// decay = 0.0、eps = 1e-8 → rate = 0.1
    /// m = 0.9*0.01 + 0.1*0.1 = 0.019
    /// v = 0.999*1e-5 + 0.001*0.01 = 1.099e-5
    /// val = 0.019 / (sqrt(1.099e-5) + 1e-8) ≈ 0.019 / 0.003315 ≈ 5.732
    /// w = 1.0 - 0.1 * 5.732 ≈ 0.4268
    #[test]
    fn one_step_denom_one_applies_adam() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.01_f32];
        let mut v = vec![1e-5_f32];
        let mut grad = vec![0.1_f32];
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1,
            1.0,
            1,
            0.0,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            1,
        );

        // 期待値を f32 で再計算 → f64 cast (f32 リテラル比較の pitfall 回避)
        let g = 0.1_f32;
        let lr = 0.1_f32;
        let step_size = 1.0_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let rate = lr * step_size;
        let mi = beta1 * 0.01_f32 + (1.0_f32 - beta1) * g;
        let vi = beta2 * 1e-5_f32 + (1.0_f32 - beta2) * g * g;
        let val = mi / (vi.sqrt() + eps);
        let exp_w = 1.0_f32 - rate * val;
        let diff = ((weights[0] as f64) - (exp_w as f64)).abs();
        assert!(diff < 1e-7, "got {} exp {exp_w} diff {diff}", weights[0]);
    }

    /// `decay = 0`、`g = 0`、clip 有効で **clip がかかるとき** weights は
    /// `[min_w, max_w]` の範囲外に出ない (`adamw_step` と同型ガード)。
    #[test]
    fn clamp_pulls_weights_into_range() {
        let mut weights = vec![100.0_f32, -100.0, 0.5];
        let mut m = vec![0.0_f32; 3];
        let mut v = vec![0.0_f32; 3];
        let mut grad = vec![0.0_f32; 3];
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.0,
            1.0,
            1,
            0.0,
            0.9,
            0.999,
            1e-8,
            -1.0,
            1.0,
            3,
        );
        assert_eq!(weights, vec![1.0_f32, -1.0, 0.5]);
    }

    /// 5 step の RAdam を host orchestration で回し、`step_size` / `denom` を
    /// 1 step ずつ更新しながら weights が target = 0 に向かって monotonic に
    /// 近づくことを確認 (簡易 convergence、`adamw_step` と同型)。
    #[test]
    fn five_step_monotonic_descent_with_radam_schedule() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut prev_w = weights[0];
        for step in 1..=5_u64 {
            let mut grad = vec![1.0_f32];
            let (step_size, denom) = radam_compute_step_size_denom(step, 0.9, 0.999, 5.0);
            radam_step_cpu(
                &mut weights,
                &mut m,
                &mut v,
                &mut grad,
                0.1,
                step_size,
                denom,
                0.0,
                0.9,
                0.999,
                1e-8,
                f32::MIN,
                f32::MAX,
                1,
            );
            assert!(
                weights[0] < prev_w,
                "step {step} not decreasing: prev={prev_w} cur={}",
                weights[0]
            );
            prev_w = weights[0];
        }
    }

    /// NaN 入力 (grad = NaN) で weights が NaN に汚染される (`adamw_step` と同型、
    /// optimizer は NaN を握り潰さず伝搬する)。
    #[test]
    fn nan_grad_propagates_into_weights() {
        let mut weights = vec![0.5_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![f32::NAN];
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            1e-3,
            1.0,
            1,
            0.0,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            1,
        );
        assert!(weights[0].is_nan());
        assert!(m[0].is_nan());
        assert!(v[0].is_nan());
        assert_eq!(grad, vec![0.0_f32]);
    }

    /// `min_w == max_w` で weights が単一値に collapse する degenerate clip
    /// (`adamw_step` と同型ガード、kernel 側 if-else ladder の境界扱いが CPU
    /// `f32::clamp` と一致するか)。
    #[test]
    fn collapsed_clip_range_pins_weights_to_single_value() {
        let mut weights = vec![100.0_f32, -100.0, 0.0, 0.5];
        let mut m = vec![0.0_f32; 4];
        let mut v = vec![0.0_f32; 4];
        let mut grad = vec![1.0_f32; 4];
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1, // lr
            1.0, // step_size
            1,   // denom
            0.0, // decay
            0.9,
            0.999,
            1e-8,
            0.5, // min_w
            0.5, // max_w (degenerate)
            4,
        );
        assert_eq!(weights, vec![0.5_f32; 4]);
    }

    /// 空配列 (n = 0) でも panic せず no-op。
    #[test]
    fn empty_input_yields_no_changes() {
        let mut weights: Vec<f32> = vec![];
        let mut m: Vec<f32> = vec![];
        let mut v: Vec<f32> = vec![];
        let mut grad: Vec<f32> = vec![];
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1,
            1.0,
            1,
            0.01,
            0.9,
            0.999,
            1e-8,
            -1.0,
            1.0,
            0,
        );
        assert!(weights.is_empty());
    }
}
