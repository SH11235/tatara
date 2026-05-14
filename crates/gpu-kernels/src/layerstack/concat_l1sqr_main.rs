//! `concat(l1_sqr, l1_main)` forward / gradient kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn concat_l1sqr_main_fwd` / `_grad`) は `bins/nnue_train/
//! src/main.rs` に inline 定義 (cuda-oxide bin-entry 制約)。bullet 上流の
//! `l1_sqr.concat(l1_main)` (`shogi_layerstack.rs:1434` 付近、`l2_input` を作る前段)
//! に等価。`a_dim == b_dim == L1_EFFECTIVE (= 15)`、`out_dim = 30`。
//!
//! ## アルゴリズム
//!
//! ```text
//! forward (per batch bi):
//!     out[bi][0 .. a_dim]            = a[bi][..]          # l1_sqr
//!     out[bi][a_dim .. a_dim+b_dim]  = b[bi][..]          # l1_main
//!
//! gradient (precondition: a_dim == b_dim == dim):
//!     da[bi][..] = dout[bi][0 .. dim]
//!     db[bi][..] = dout[bi][dim .. 2*dim]
//! ```
//!
//! - layout は **row-major** (`a` は `batch × a_dim`、`b` は `batch × b_dim`、
//!   `out` は `batch × (a_dim+b_dim)`)。前半が `a`、後半が `b`。
//! - backward kernel は `a_dim == b_dim` を前提に 1 thread = 1 (batch, dim_index)
//!   で `da[tid]` と `db[tid]` の両方を書く。reference もそれに合わせる。
//! - 値はそのままコピー (NaN/Inf も透過)。

/// `concat(a, b)` forward reference — row-major、前半 `a`・後半 `b`。
///
/// `a.len() == batch * a_dim`、`b.len() == batch * b_dim`、
/// `out.len() == batch * (a_dim + b_dim)` 前提。
pub fn concat_l1sqr_main_fwd_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    a_dim: usize,
    b_dim: usize,
) {
    let out_dim = a_dim + b_dim;
    for bi in 0..batch {
        for oi in 0..out_dim {
            out[bi * out_dim + oi] = if oi < a_dim {
                a[bi * a_dim + oi]
            } else {
                b[bi * b_dim + (oi - a_dim)]
            };
        }
    }
}

/// `concat(a, b)` gradient reference — `da = dout[..dim]`、`db = dout[dim..2*dim]`。
///
/// **Precondition: `a_dim == b_dim == dim`** (両方 `L1_EFFECTIVE`)。
/// `dout.len() == batch * 2 * dim`、`da.len() == db.len() == batch * dim` 前提。
pub fn concat_l1sqr_main_grad_cpu(
    dout: &[f32],
    da: &mut [f32],
    db: &mut [f32],
    batch: usize,
    dim: usize,
) {
    let out_dim = 2 * dim;
    for bi in 0..batch {
        for ii in 0..dim {
            da[bi * dim + ii] = dout[bi * out_dim + ii];
            db[bi * dim + ii] = dout[bi * out_dim + dim + ii];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_concats_row_major() {
        // batch=2, a_dim=3, b_dim=2
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0_f32; 2 * 5];
        concat_l1sqr_main_fwd_cpu(&a, &b, &mut out, 2, 3, 2);
        assert_eq!(
            out,
            vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 4.0, 5.0, 6.0, 30.0, 40.0]
        );
    }

    #[test]
    fn grad_splits_back() {
        // batch=2, dim=3 → out_dim=6
        let dout = vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ];
        let mut da = vec![0.0_f32; 2 * 3];
        let mut db = vec![0.0_f32; 2 * 3];
        concat_l1sqr_main_grad_cpu(&dout, &mut da, &mut db, 2, 3);
        assert_eq!(da, vec![1.0_f32, 2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(db, vec![4.0_f32, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    /// forward → grad の round-trip: dout が forward の output と同 layout なら
    /// da == a 部分の grad, db == b 部分の grad (concat は線形なので grad は split)。
    #[test]
    fn round_trip_v102_shape() {
        // a_dim = b_dim = 15, batch = 3
        let batch = 3;
        let dim = 15;
        let a: Vec<f32> = (0..batch * dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..batch * dim).map(|i| -(i as f32) - 0.5).collect();
        let mut out = vec![0.0_f32; batch * 2 * dim];
        concat_l1sqr_main_fwd_cpu(&a, &b, &mut out, batch, dim, dim);
        // pretend dout == out (identity downstream grad)
        let mut da = vec![0.0_f32; batch * dim];
        let mut db = vec![0.0_f32; batch * dim];
        concat_l1sqr_main_grad_cpu(&out, &mut da, &mut db, batch, dim);
        assert_eq!(da, a);
        assert_eq!(db, b);
    }

    #[test]
    fn nan_propagates() {
        let a = vec![f32::NAN, 1.0];
        let b = vec![2.0_f32, 3.0];
        let mut out = vec![0.0_f32; 4];
        concat_l1sqr_main_fwd_cpu(&a, &b, &mut out, 2, 1, 1);
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[2.0_f32, 1.0, 3.0]);
    }

    #[test]
    fn empty_is_noop() {
        let mut out: Vec<f32> = vec![];
        concat_l1sqr_main_fwd_cpu(&[], &[], &mut out, 0, 3, 2);
        assert!(out.is_empty());
    }
}
