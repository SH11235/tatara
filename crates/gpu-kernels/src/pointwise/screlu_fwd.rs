//! SCReLU (squared clipped ReLU) activation forward kernel の reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn screlu_fwd` は呼び出し元 bin entry (`bins/nnue_train/src/
//! main.rs`) に inline 定義する (cuda-oxide rustc-codegen-cuda backend の
//! bin-entry 制約)。本 module は GPU↔CPU 数値同等性テストの reference 用。
//! `screlu_grad` (gradient) と対になる forward。
//!
//! ## アルゴリズム
//!
//! ```text
//! a    = clip(x[i], 0, 1)
//! y[i] = a * a
//! ```
//!
//! - clip 境界は `[0, 1]` (bullet 上流 `SCReLU` と一致)。`screlu_grad` が前提と
//!   する `forward = clip(x, 0, 1)^2` と整合する。
//! - GPU `#[kernel]` 側は `f32::clamp` を lower できないため `if-else` ladder
//!   (`x < 0 ? 0 : x > 1 ? 1 : x`) で clip を展開する。reference も同じ if-else で
//!   書き、kernel と bit-完全一致させる (数値同等性テストの tolerance は 0)。
//! - NaN: `NaN < 0` も `NaN > 1` も false なので `a = NaN`、`y = NaN * NaN = NaN`。
//!   reference / kernel とも NaN をそのまま伝搬する。

/// SCReLU forward reference — `y[i] = clip(x[i], 0, 1)²`。
///
/// `x.len() == y.len() == n` を host invariant として要求する
/// (`n > x.len()` は panic)。
#[allow(clippy::manual_clamp)] // kernel の if-else 展開と bit-完全一致させるため。
pub fn screlu_fwd_cpu(x: &[f32], y: &mut [f32], n: usize) {
    for i in 0..n {
        let xi = x[i];
        let a = if xi < 0.0_f32 {
            0.0_f32
        } else if xi > 1.0_f32 {
            1.0_f32
        } else {
            xi
        };
        y[i] = a * a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_squares_clipped_value() {
        // x: [-2, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5]
        // a: [ 0,    0, 0.0, 0.3, 0.7, 1.0, 1.0]   (clip [0, 1])
        // y: [ 0,    0, 0.0, 0.09,0.49,1.0, 1.0]   (a²)
        let x = vec![-2.0_f32, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5];
        let mut y = vec![0.0_f32; x.len()];
        screlu_fwd_cpu(&x, &mut y, x.len());
        let expected = [
            0.0_f32,
            0.0,
            0.0,
            0.3_f32 * 0.3_f32,
            0.7_f32 * 0.7_f32,
            1.0,
            1.0,
        ];
        for (i, (got, exp)) in y.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-7, "i={i}: got {got} exp {exp}");
        }
    }

    #[test]
    fn forward_propagates_nan() {
        let x = vec![f32::NAN, 0.5];
        let mut y = vec![0.0_f32; 2];
        screlu_fwd_cpu(&x, &mut y, 2);
        assert!(y[0].is_nan());
        assert_eq!(y[1], 0.25_f32);
    }

    #[test]
    fn saturated_inputs_clip_then_square() {
        // x < 0 → a=0、x > 1 → a=1。±Inf も clip で 0 / 1 に落ちる。
        let x = vec![-100.0_f32, 100.0, f32::NEG_INFINITY, f32::INFINITY];
        let mut y = vec![9.0_f32; 4];
        screlu_fwd_cpu(&x, &mut y, 4);
        assert_eq!(y, vec![0.0_f32, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn empty_is_noop() {
        let mut y: Vec<f32> = vec![];
        screlu_fwd_cpu(&[], &mut y, 0);
        assert!(y.is_empty());
    }
}
