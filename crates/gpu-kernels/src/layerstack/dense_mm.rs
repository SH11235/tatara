//! Regular (non-bucket) dense matmul forward / backward + bias-grad kernel の
//! reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn dense_mm_fwd` / `dense_mm_bwd_input` / `dense_mm_bwd_weight`
//! / `bias_grad`) は `bins/nnue_train/src/main.rs` に inline 定義 (cuda-oxide
//! bin-entry 制約)。bullet 上流の `Affine` / dense `linear` (`crates/trainer/src/
//! model/builder.rs`) に等価で、**shared factorized L1f** layer
//! (`combined (B×1536) → l1f_out (B×16)`) の forward / backward に使われる
//! (`shogi_layerstack.rs:2244` 付近の `l1f` Affine)。
//!
//! ## Layout 規約 (kernel と完全一致させる — テストの核心)
//!
//! - `x`: row-major `batch × in_dim` (`x[b * in_dim + i]`)
//! - `w`: row-major `in_dim × out_dim` (`w[i * out_dim + o]`) — **in-major**
//! - `y`: row-major `batch × out_dim` (`y[b * out_dim + o]`)
//! - `bias`: `out_dim` (`bias[o]`)
//!
//! ## アルゴリズム
//!
//! ```text
//! forward:       y[b][o]  = bias[o] + sum_i x[b][i] * w[i][o]
//! bwd_input:     dx[b][i] = sum_o dy[b][o] * w[i][o]
//! bwd_weight:    dw[i][o] = sum_b x[b][i] * dy[b][o]        # overwrite (atomics 不要)
//! bias_grad:     grad_bias[o] += sum_b dy[b][o]              # accumulate
//! ```
//!
//! - `bwd_weight` は kernel が「1 thread = 1 weight cell + batch loop」で
//!   overwrite するため reference も overwrite (`grad_w` を上書き)。
//! - `bias_grad` は kernel が atomic accumulate (host が呼出前に 0 初期化) なので
//!   reference は **既存値に加算** (`grad_bias[o] += ...`)。GPU↔CPU テストでは
//!   CPU 側も 0 初期化済 buffer に渡せば一致する。
//! - 累積順序は `i` / `o` / `b` の昇順 (f32 round-off の累積を kernel と揃える)。
//! - NaN/Inf は伝搬。

/// dense matmul forward reference — `y[b][o] = bias[o] + sum_i x[b][i] * w[i][o]`。
///
/// `x.len() == batch*in_dim`、`w.len() == in_dim*out_dim`、`bias.len() == out_dim`、
/// `y.len() == batch*out_dim` 前提。
pub fn dense_mm_fwd_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) {
    for bi in 0..batch {
        for oi in 0..out_dim {
            let mut sum = bias[oi];
            for k in 0..in_dim {
                sum += x[bi * in_dim + k] * w[k * out_dim + oi];
            }
            y[bi * out_dim + oi] = sum;
        }
    }
}

/// dense matmul backward wrt input — `dx[b][i] = sum_o dy[b][o] * w[i][o]`。
///
/// `dy.len() == batch*out_dim`、`w.len() == in_dim*out_dim`、`dx.len() == batch*in_dim` 前提。
pub fn dense_mm_bwd_input_cpu(
    dy: &[f32],
    w: &[f32],
    dx: &mut [f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) {
    for bi in 0..batch {
        for ii in 0..in_dim {
            let mut sum = 0.0_f32;
            for o in 0..out_dim {
                sum += dy[bi * out_dim + o] * w[ii * out_dim + o];
            }
            dx[bi * in_dim + ii] = sum;
        }
    }
}

/// dense matmul backward wrt weight — `dw[i][o] = sum_b x[b][i] * dy[b][o]` (overwrite)。
///
/// `x.len() == batch*in_dim`、`dy.len() == batch*out_dim`、`grad_w.len() == in_dim*out_dim` 前提。
pub fn dense_mm_bwd_weight_cpu(
    x: &[f32],
    dy: &[f32],
    grad_w: &mut [f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) {
    for ii in 0..in_dim {
        for oi in 0..out_dim {
            let mut sum = 0.0_f32;
            for b in 0..batch {
                sum += x[b * in_dim + ii] * dy[b * out_dim + oi];
            }
            grad_w[ii * out_dim + oi] = sum;
        }
    }
}

/// bias gradient reference — `grad_bias[o] += sum_b dy[b][o]` (accumulate into existing)。
///
/// `dy.len() == batch*out_dim`、`grad_bias.len() == out_dim` 前提。kernel と同じ
/// accumulate semantics (host が呼出前に 0 初期化)。
pub fn bias_grad_cpu(dy: &[f32], grad_bias: &mut [f32], batch: usize, out_dim: usize) {
    for bi in 0..batch {
        for oi in 0..out_dim {
            grad_bias[oi] += dy[bi * out_dim + oi];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手計算: batch=2, in_dim=2, out_dim=2。
    /// x = [[1,2],[3,4]], w = [[5,6],[7,8]] (row-major in-major: w[0][0]=5,w[0][1]=6,w[1][0]=7,w[1][1]=8),
    /// bias = [10, 20].
    /// y[0][0] = 10 + 1*5 + 2*7 = 10+5+14 = 29
    /// y[0][1] = 20 + 1*6 + 2*8 = 20+6+16 = 42
    /// y[1][0] = 10 + 3*5 + 4*7 = 10+15+28 = 53
    /// y[1][1] = 20 + 3*6 + 4*8 = 20+18+32 = 70
    #[test]
    fn forward_hand_computed() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![5.0_f32, 6.0, 7.0, 8.0];
        let bias = vec![10.0_f32, 20.0];
        let mut y = vec![0.0_f32; 4];
        dense_mm_fwd_cpu(&x, &w, &bias, &mut y, 2, 2, 2);
        assert_eq!(y, vec![29.0_f32, 42.0, 53.0, 70.0]);
    }

    /// bwd_input: dx[b][i] = sum_o dy[b][o] * w[i][o].
    /// dy = [[1,1],[1,1]], w as above.
    /// dx[0][0] = 1*5 + 1*6 = 11; dx[0][1] = 1*7 + 1*8 = 15
    /// dx[1][0] = 11; dx[1][1] = 15
    #[test]
    fn bwd_input_hand_computed() {
        let dy = vec![1.0_f32, 1.0, 1.0, 1.0];
        let w = vec![5.0_f32, 6.0, 7.0, 8.0];
        let mut dx = vec![0.0_f32; 4];
        dense_mm_bwd_input_cpu(&dy, &w, &mut dx, 2, 2, 2);
        assert_eq!(dx, vec![11.0_f32, 15.0, 11.0, 15.0]);
    }

    /// bwd_weight: dw[i][o] = sum_b x[b][i] * dy[b][o].
    /// x = [[1,2],[3,4]], dy = [[10,20],[30,40]].
    /// dw[0][0] = 1*10 + 3*30 = 10+90 = 100
    /// dw[0][1] = 1*20 + 3*40 = 20+120 = 140
    /// dw[1][0] = 2*10 + 4*30 = 20+120 = 140
    /// dw[1][1] = 2*20 + 4*40 = 40+160 = 200
    #[test]
    fn bwd_weight_hand_computed() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let dy = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut dw = vec![999.0_f32; 4]; // pre-filled garbage → overwrite check
        dense_mm_bwd_weight_cpu(&x, &dy, &mut dw, 2, 2, 2);
        assert_eq!(dw, vec![100.0_f32, 140.0, 140.0, 200.0]);
    }

    /// bias_grad: grad_bias[o] += sum_b dy[b][o]. dy = [[10,20],[30,40]] →
    /// col sums = [40, 60]. accumulate into [1, 2] → [41, 62].
    #[test]
    fn bias_grad_accumulates() {
        let dy = vec![10.0_f32, 20.0, 30.0, 40.0];
        let mut grad_bias = vec![1.0_f32, 2.0];
        bias_grad_cpu(&dy, &mut grad_bias, 2, 2);
        assert_eq!(grad_bias, vec![41.0_f32, 62.0]);
    }

    /// chain-rule consistency: with bias=0, dx from bwd_input must equal the
    /// vector-Jacobian product of forward. Pick random x, w, dy; compute y, then
    /// verify d(<dy,y>)/dx[b][i] == dx[b][i] analytically: <dy,y> is linear in x
    /// so the gradient is exactly bwd_input. We just cross-check magnitudes via a
    /// finite-difference-free identity: sum_b sum_o dy[b][o]*y[b][o]
    ///   == sum_b sum_i x[b][i]*dx[b][i]  (+ bias term, here 0).
    #[test]
    fn fwd_bwd_input_inner_product_identity() {
        let batch = 3;
        let in_dim = 4;
        let out_dim = 5;
        let x: Vec<f32> = (0..batch * in_dim)
            .map(|i| (i as f32) * 0.1 - 0.7)
            .collect();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (i as f32) * 0.05 + 0.2)
            .collect();
        let bias = vec![0.0_f32; out_dim];
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| (i as f32) * 0.3 - 1.1)
            .collect();

        let mut y = vec![0.0_f32; batch * out_dim];
        dense_mm_fwd_cpu(&x, &w, &bias, &mut y, batch, in_dim, out_dim);
        let mut dx = vec![0.0_f32; batch * in_dim];
        dense_mm_bwd_input_cpu(&dy, &w, &mut dx, batch, in_dim, out_dim);

        let lhs: f64 = (0..batch * out_dim)
            .map(|i| dy[i] as f64 * y[i] as f64)
            .sum();
        let rhs: f64 = (0..batch * in_dim)
            .map(|i| x[i] as f64 * dx[i] as f64)
            .sum();
        assert!((lhs - rhs).abs() < 1e-3, "lhs={lhs} rhs={rhs}");
    }

    #[test]
    fn empty_batch_is_noop_fwd_uses_bias_only() {
        // batch=0: y empty, nothing written
        let mut y: Vec<f32> = vec![];
        dense_mm_fwd_cpu(&[], &[], &[1.0, 2.0], &mut y, 0, 0, 2);
        assert!(y.is_empty());
        // bias_grad with batch=0: unchanged
        let mut grad_bias = vec![3.0_f32, 4.0];
        bias_grad_cpu(&[], &mut grad_bias, 0, 2);
        assert_eq!(grad_bias, vec![3.0_f32, 4.0]);
    }
}
