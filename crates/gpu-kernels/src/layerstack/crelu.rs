//! CReLU (clipped ReLU) forward / gradient kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn crelu_fwd` / `crelu_grad`) は `bins/nnue_train/src/
//! main.rs` に inline 定義 (cuda-oxide bin-entry 制約)。bullet 上流の
//! `crelu()` (`crates/trainer/src/model/builder.rs` の activation) と等価で、
//! FT post / l2_pre / l2_out の activation に使われる。
//!
//! ## アルゴリズム
//!
//! ```text
//! forward:  y[i]  = clip(x[i], 0, 1)
//! gradient: dx[i] = dy[i] if 0 < x[i] < 1 else 0
//! ```
//!
//! - clamp 境界は `[0, 1]` (bullet の `CReLU` と一致、`abs_pow(2)` 前提の qa=127
//!   量子化に合わせて 0..1 に丸める)。
//! - gradient は **strict 不等号** (`x == 0` / `x == 1` ちょうどは grad 0)。
//!   GPU kernel が `f32::clamp` lowering 失敗回避の if-else 展開で同じく strict に
//!   書いているので、reference もそれに合わせる。
//! - NaN: `clip(NaN, 0, 1)` — kernel の if-else 展開 (`if x < 0 {0} else if x > 1
//!   {1} else {x}`) では `NaN < 0` も `NaN > 1` も false なので `NaN` がそのまま
//!   通る。reference も同じ挙動 (`!(NaN < 0) && !(NaN > 1)` → x). grad では
//!   `NaN > 0 && NaN < 1` が false なので 0。

/// CReLU forward reference — `y[i] = clip(x[i], 0, 1)`。
///
/// `x.len() == y.len() == n` 前提。`n > x.len()` は panic (host invariant 違反)。
#[allow(clippy::manual_clamp)] // kernel の if-else 展開と bit-完全一致させるため (f32::clamp は cuda-oxide で lowering 失敗)。
pub fn crelu_fwd_cpu(x: &[f32], y: &mut [f32], n: usize) {
    for i in 0..n {
        let xi = x[i];
        // kernel の if-else 展開と bit-完全一致させる (f32::clamp は lowering 失敗)。
        y[i] = if xi < 0.0_f32 {
            0.0_f32
        } else if xi > 1.0_f32 {
            1.0_f32
        } else {
            xi
        };
    }
}

/// CReLU gradient reference — `dx[i] = dy[i] if 0 < x[i] < 1 else 0`。
///
/// `x.len() == dy.len() == dx.len() == n` 前提。
pub fn crelu_grad_cpu(x: &[f32], dy: &[f32], dx: &mut [f32], n: usize) {
    for i in 0..n {
        let xi = x[i];
        dx[i] = if xi > 0.0_f32 && xi < 1.0_f32 {
            dy[i]
        } else {
            0.0_f32
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_clamps_to_unit_interval() {
        let x = vec![-2.0_f32, -0.5, 0.0, 0.3, 1.0, 1.5, 7.0];
        let mut y = vec![0.0_f32; x.len()];
        crelu_fwd_cpu(&x, &mut y, x.len());
        assert_eq!(y, vec![0.0_f32, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn forward_propagates_nan() {
        let x = vec![f32::NAN, 0.5];
        let mut y = vec![0.0_f32; 2];
        crelu_fwd_cpu(&x, &mut y, 2);
        assert!(y[0].is_nan());
        assert_eq!(y[1], 0.5_f32);
    }

    #[test]
    fn grad_is_one_in_interior_zero_at_boundaries() {
        let x = vec![-1.0_f32, 0.0, 0.0001, 0.5, 0.9999, 1.0, 2.0];
        let dy = vec![3.0_f32; x.len()];
        let mut dx = vec![0.0_f32; x.len()];
        crelu_grad_cpu(&x, &dy, &mut dx, x.len());
        assert_eq!(dx, vec![0.0_f32, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn grad_nan_x_yields_zero() {
        let x = vec![f32::NAN];
        let dy = vec![5.0_f32];
        let mut dx = vec![9.0_f32];
        crelu_grad_cpu(&x, &dy, &mut dx, 1);
        assert_eq!(dx[0], 0.0_f32);
    }

    #[test]
    fn empty_is_noop() {
        let mut y: Vec<f32> = vec![];
        crelu_fwd_cpu(&[], &mut y, 0);
        assert!(y.is_empty());
        let mut dx: Vec<f32> = vec![];
        crelu_grad_cpu(&[], &[], &mut dx, 0);
        assert!(dx.is_empty());
    }
}
