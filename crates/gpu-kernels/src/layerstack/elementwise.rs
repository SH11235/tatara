//! Elementwise add kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn elementwise_add`) は `bins/nnue_train/src/main.rs` に
//! inline 定義 (cuda-oxide bin-entry 制約)。forward の `l1_total =
//! l1_bucket + l1f_out`、`net_output = l3_out + l1_skip` と、backward の
//! gradient-combine (`dl1_main = dl1_main_from_concat + dl1_main_from_sqr`、
//! `dcombined = dcombined_from_l1 + dcombined_from_l1f`) に使われる
//! (bullet 上流では `+` operator overload に相当)。
//!
//! ## アルゴリズム
//!
//! ```text
//! c[i] = a[i] + b[i]
//! ```
//!
//! - NaN / Inf は IEEE 754 のまま伝搬。

/// Elementwise add reference — `c[i] = a[i] + b[i]`。
///
/// `a.len() == b.len() == c.len() == n` 前提。
pub fn elementwise_add_cpu(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        c[i] = a[i] + b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_elementwise() {
        let a = vec![1.0_f32, -2.0, 3.5, 0.0];
        let b = vec![0.5_f32, 2.0, -1.5, 7.0];
        let mut c = vec![0.0_f32; 4];
        elementwise_add_cpu(&a, &b, &mut c, 4);
        assert_eq!(c, vec![1.5_f32, 0.0, 2.0, 7.0]);
    }

    #[test]
    fn nan_propagates() {
        let a = vec![f32::NAN, 1.0];
        let b = vec![1.0_f32, 2.0];
        let mut c = vec![0.0_f32; 2];
        elementwise_add_cpu(&a, &b, &mut c, 2);
        assert!(c[0].is_nan());
        assert_eq!(c[1], 3.0_f32);
    }

    #[test]
    fn empty_is_noop() {
        let mut c: Vec<f32> = vec![];
        elementwise_add_cpu(&[], &[], &mut c, 0);
        assert!(c.is_empty());
    }
}
