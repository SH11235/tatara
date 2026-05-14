//! `abs_pow(2) * scale` forward / gradient kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn abs_pow2_scale_fwd` / `_grad`) は `bins/nnue_train/src/
//! main.rs` に inline 定義 (cuda-oxide bin-entry 制約)。bullet 上流の
//! `abs_pow(2.0)` (`crates/trainer/src/model/builder.rs`) × scale 乗算で、
//! `l1_sqr = l1_main^2 * (127/128)` (`shogi_layerstack.rs:2260` 付近) に使う。
//!
//! ## アルゴリズム
//!
//! ```text
//! forward:  y[i]  = x[i] * x[i] * scale       # |x|^2 == x^2 なので abs 不要
//! gradient: dx[i] = 2 * x[i] * scale * dy[i]  # d(x^2*scale)/dx = 2 x scale
//! ```
//!
//! - bullet `abs_pow(2.0)` は数学的に `|x|^2 = x^2` なので abs 演算は不要
//!   (kernel も同じ判断で `x * x` を直書きしている)。
//! - scale は `127.0/128.0` (= `L1_SQR_SCALE`)、qa=127 量子化由来。
//! - NaN は伝搬する (`NaN * NaN * scale = NaN`、`2 * NaN * ... = NaN`)。

/// `abs_pow(2) * scale` forward reference — `y[i] = x[i] * x[i] * scale`。
///
/// `x.len() == y.len() == n` 前提。
pub fn abs_pow2_scale_fwd_cpu(x: &[f32], y: &mut [f32], scale: f32, n: usize) {
    for i in 0..n {
        let xi = x[i];
        y[i] = xi * xi * scale;
    }
}

/// `abs_pow(2) * scale` gradient reference — `dx[i] = 2 * x[i] * scale * dy[i]`。
///
/// `x.len() == dy.len() == dx.len() == n` 前提。
pub fn abs_pow2_scale_grad_cpu(x: &[f32], dy: &[f32], dx: &mut [f32], scale: f32, n: usize) {
    for i in 0..n {
        let xi = x[i];
        dx[i] = 2.0_f32 * xi * scale * dy[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_squares_and_scales() {
        let x = vec![-3.0_f32, 0.0, 0.5, 2.0];
        let mut y = vec![0.0_f32; 4];
        let scale = 127.0_f32 / 128.0_f32;
        abs_pow2_scale_fwd_cpu(&x, &mut y, scale, 4);
        for i in 0..4 {
            let exp = x[i] * x[i] * scale;
            assert_eq!(y[i], exp, "i={i}");
        }
        // x = -3: 9 * 127/128
        assert_eq!(y[0], 9.0_f32 * (127.0_f32 / 128.0_f32));
        assert_eq!(y[1], 0.0_f32);
    }

    #[test]
    fn grad_is_two_x_scale_dy() {
        let x = vec![1.0_f32, -2.0, 0.0, 4.0];
        let dy = vec![0.5_f32, 1.0, 3.0, -1.0];
        let mut dx = vec![0.0_f32; 4];
        let scale = 2.0_f32;
        abs_pow2_scale_grad_cpu(&x, &dy, &mut dx, scale, 4);
        for i in 0..4 {
            let exp = 2.0_f32 * x[i] * scale * dy[i];
            assert_eq!(dx[i], exp, "i={i}");
        }
        // x=1, dy=0.5, scale=2 → 2*1*2*0.5 = 2
        assert_eq!(dx[0], 2.0_f32);
        // x=0 → grad 0 regardless of dy
        assert_eq!(dx[2], 0.0_f32);
    }

    /// scale=1 で grad = 2x·dy が autodiff の `d(x^2)/dx · dy` と一致する性質。
    #[test]
    fn grad_matches_fwd_derivative_scale_one() {
        let x = vec![0.7_f32, -1.3, 2.5];
        let dy = vec![1.0_f32; 3];
        let mut dx = vec![0.0_f32; 3];
        abs_pow2_scale_grad_cpu(&x, &dy, &mut dx, 1.0, 3);
        for i in 0..3 {
            assert_eq!(dx[i], 2.0_f32 * x[i]);
        }
    }

    #[test]
    fn nan_propagates() {
        let x = vec![f32::NAN];
        let mut y = vec![0.0_f32];
        abs_pow2_scale_fwd_cpu(&x, &mut y, 1.0, 1);
        assert!(y[0].is_nan());
        let dy = vec![1.0_f32];
        let mut dx = vec![0.0_f32];
        abs_pow2_scale_grad_cpu(&x, &dy, &mut dx, 1.0, 1);
        assert!(dx[0].is_nan());
    }

    #[test]
    fn empty_is_noop() {
        let mut y: Vec<f32> = vec![];
        abs_pow2_scale_fwd_cpu(&[], &mut y, 1.0, 0);
        assert!(y.is_empty());
    }
}
