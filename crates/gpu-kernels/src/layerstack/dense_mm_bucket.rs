//! Per-bucket dense matmul forward / backward + bias-grad kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn dense_mm_fwd_bucket` / `dense_mm_bwd_input_bucket` /
//! `dense_mm_bwd_weight_bucket` / `bias_grad_bucket`) は `bins/nnue_train/src/
//! main.rs` に inline 定義 (cuda-oxide bin-entry 制約)。bullet 上流の **LayerStack
//! per-bucket Affine** (`crates/trainer/src/model/builder.rs` の `layer_stack` /
//! `select(bucket)`、`shogi_layerstack.rs:2244-2275` の L1 / L2 / L3) に等価。
//! progress8kpabs の 9 bucket × {L1 (16,1536), L2 (32,30), L3 (1,32)} を
//! per-position の `bucket_idx[b]` で select する。
//!
//! ## Layout 規約 (kernel と完全一致させる — テストの核心)
//!
//! - `w`: row-major `(num_buckets * out_dim) × in_dim` — **bucket-major, その中 out-major**
//!   (`w[buc * out_dim * in_dim + oi * in_dim + k]`)。`dense_mm_fwd_bucket` /
//!   `dense_mm_bwd_*_bucket` で共通。`dense_mm_bwd_weight_bucket` の `grad_w` も同 layout
//!   (= `tid == grad_w index`)。
//! - `bias`: `num_buckets * out_dim` — bucket-major (`bias[buc * out_dim + oi]`)。
//! - `bucket_idx`: `batch` 要素の `i32`。`< 0` または `>= num_buckets` は out-of-range:
//!   - forward: `y[b][..] = 0`
//!   - bwd_input: `dx[b][..] = 0`
//!   - bwd_weight: その position はどの bucket cell にも match せず無視
//!   - bias_grad: その position は無視
//! - `x`/`y`/`dy`/`dx`: row-major `batch × {in_dim|out_dim}` (perspective 共有なし)。
//!
//! ## アルゴリズム
//!
//! ```text
//! forward (bucket g = bucket_idx[b]):
//!     y[b][o] = bias[g][o] + sum_i x[b][i] * w[g][o][i]
//! bwd_input (bucket g = bucket_idx[b]):
//!     dx[b][i] = sum_o dy[b][o] * w[g][o][i]
//! bwd_weight:
//!     grad_w[g][o][i] = sum_{b : bucket_idx[b] == g} x[b][i] * dy[b][o]   # overwrite
//! bias_grad:
//!     grad_bias[g][o] += sum_{b : bucket_idx[b] == g} dy[b][o]            # accumulate
//! ```
//!
//! - `bwd_weight` は kernel が「1 thread = 1 (bucket, o, i) cell + batch loop」で
//!   overwrite (atomics 不要) なので reference も overwrite。累積順は `b` 昇順。
//! - `bias_grad` は kernel が atomic accumulate (host が呼出前に 0 初期化) なので
//!   reference は **既存値に加算**。
//! - NaN/Inf は伝搬。

/// per-bucket dense matmul forward — `y[b][o] = bias[g][o] + sum_i x[b][i] * w[g][o][i]`。
/// out-of-range bucket は `y[b][..] = 0`。
#[allow(clippy::too_many_arguments)]
pub fn dense_mm_fwd_bucket_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    bucket_idx: &[i32],
    y: &mut [f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    num_buckets: usize,
) {
    for bi in 0..batch {
        let buc = bucket_idx[bi];
        if buc < 0 || (buc as usize) >= num_buckets {
            for oi in 0..out_dim {
                y[bi * out_dim + oi] = 0.0_f32;
            }
            continue;
        }
        let g = buc as usize;
        for oi in 0..out_dim {
            let w_row_base = g * out_dim * in_dim + oi * in_dim;
            let mut sum = bias[g * out_dim + oi];
            for k in 0..in_dim {
                sum += x[bi * in_dim + k] * w[w_row_base + k];
            }
            y[bi * out_dim + oi] = sum;
        }
    }
}

/// per-bucket dense matmul backward wrt input — `dx[b][i] = sum_o dy[b][o] * w[g][o][i]`。
/// out-of-range bucket は `dx[b][..] = 0`。
#[allow(clippy::too_many_arguments)]
pub fn dense_mm_bwd_input_bucket_cpu(
    dy: &[f32],
    w: &[f32],
    bucket_idx: &[i32],
    dx: &mut [f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    num_buckets: usize,
) {
    for bi in 0..batch {
        let buc = bucket_idx[bi];
        if buc < 0 || (buc as usize) >= num_buckets {
            for ii in 0..in_dim {
                dx[bi * in_dim + ii] = 0.0_f32;
            }
            continue;
        }
        let g = buc as usize;
        for ii in 0..in_dim {
            let mut sum = 0.0_f32;
            for o in 0..out_dim {
                let w_idx = g * out_dim * in_dim + o * in_dim + ii;
                sum += dy[bi * out_dim + o] * w[w_idx];
            }
            dx[bi * in_dim + ii] = sum;
        }
    }
}

/// per-bucket dense matmul backward wrt weight (overwrite) —
/// `grad_w[g][o][i] = sum_{b : bucket_idx[b]==g} x[b][i] * dy[b][o]`。
#[allow(clippy::too_many_arguments)]
pub fn dense_mm_bwd_weight_bucket_cpu(
    x: &[f32],
    dy: &[f32],
    bucket_idx: &[i32],
    grad_w: &mut [f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    num_buckets: usize,
) {
    let per_bucket = out_dim * in_dim;
    for g in 0..num_buckets {
        let target = g as i32;
        for oi in 0..out_dim {
            for ii in 0..in_dim {
                let mut sum = 0.0_f32;
                for b in 0..batch {
                    if bucket_idx[b] == target {
                        sum += x[b * in_dim + ii] * dy[b * out_dim + oi];
                    }
                }
                grad_w[g * per_bucket + oi * in_dim + ii] = sum;
            }
        }
    }
}

/// per-bucket bias gradient (accumulate) — `grad_bias[g][o] += sum_{b : bucket_idx[b]==g} dy[b][o]`。
/// out-of-range bucket の position は無視。kernel と同じ accumulate semantics。
pub fn bias_grad_bucket_cpu(
    dy: &[f32],
    bucket_idx: &[i32],
    grad_bias: &mut [f32],
    batch: usize,
    out_dim: usize,
    num_buckets: usize,
) {
    for bi in 0..batch {
        let buc = bucket_idx[bi];
        if buc < 0 || (buc as usize) >= num_buckets {
            continue;
        }
        let g = buc as usize;
        for oi in 0..out_dim {
            grad_bias[g * out_dim + oi] += dy[bi * out_dim + oi];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手計算: num_buckets=2, in_dim=2, out_dim=1, batch=2.
    /// w layout (bucket-major, out-major, in last):
    ///   bucket0 out0: [w000=1, w001=2]
    ///   bucket1 out0: [w100=3, w101=4]
    /// bias = [bucket0 out0 = 10, bucket1 out0 = 20]
    /// x = [[1,1],[2,2]], bucket_idx = [0, 1]
    /// batch0 (bucket0): y = 10 + 1*1 + 1*2 = 13
    /// batch1 (bucket1): y = 20 + 2*3 + 2*4 = 20+6+8 = 34
    #[test]
    fn forward_hand_computed() {
        let x = vec![1.0_f32, 1.0, 2.0, 2.0];
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bias = vec![10.0_f32, 20.0];
        let bucket_idx = vec![0_i32, 1];
        let mut y = vec![0.0_f32; 2];
        dense_mm_fwd_bucket_cpu(&x, &w, &bias, &bucket_idx, &mut y, 2, 2, 1, 2);
        assert_eq!(y, vec![13.0_f32, 34.0]);
    }

    #[test]
    fn forward_out_of_range_bucket_yields_zero() {
        let x = vec![5.0_f32, 5.0];
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bias = vec![10.0_f32, 20.0];
        let bucket_idx = vec![-1_i32]; // out of range
        let mut y = vec![999.0_f32];
        dense_mm_fwd_bucket_cpu(&x, &w, &bias, &bucket_idx, &mut y, 1, 2, 1, 2);
        assert_eq!(y, vec![0.0_f32]);
        // also >= num_buckets
        let bucket_idx2 = vec![2_i32];
        let mut y2 = vec![999.0_f32];
        dense_mm_fwd_bucket_cpu(&x, &w, &bias, &bucket_idx2, &mut y2, 1, 2, 1, 2);
        assert_eq!(y2, vec![0.0_f32]);
    }

    /// bwd_input hand-computed. num_buckets=2, in_dim=2, out_dim=1.
    /// w same as above. dy = [[1],[1]], bucket_idx = [0, 1].
    /// batch0 (bucket0): dx[0] = 1 * w000 = 1; dx[1] = 1 * w001 = 2
    /// batch1 (bucket1): dx[0] = 1 * w100 = 3; dx[1] = 1 * w101 = 4
    #[test]
    fn bwd_input_hand_computed() {
        let dy = vec![1.0_f32, 1.0];
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bucket_idx = vec![0_i32, 1];
        let mut dx = vec![0.0_f32; 4];
        dense_mm_bwd_input_bucket_cpu(&dy, &w, &bucket_idx, &mut dx, 2, 2, 1, 2);
        assert_eq!(dx, vec![1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bwd_input_out_of_range_yields_zero() {
        let dy = vec![1.0_f32];
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bucket_idx = vec![-5_i32];
        let mut dx = vec![7.0_f32, 7.0];
        dense_mm_bwd_input_bucket_cpu(&dy, &w, &bucket_idx, &mut dx, 1, 2, 1, 2);
        assert_eq!(dx, vec![0.0_f32, 0.0]);
    }

    /// bwd_weight hand-computed. num_buckets=2, in_dim=2, out_dim=1, batch=3.
    /// x = [[1,1],[2,2],[3,3]], dy = [[10],[20],[30]], bucket_idx = [0, 0, 1].
    /// bucket0 collects b=0,1: grad_w[0][0][0] = 1*10 + 2*20 = 50; grad_w[0][0][1] = 1*10 + 2*20 = 50
    /// bucket1 collects b=2:   grad_w[1][0][0] = 3*30 = 90; grad_w[1][0][1] = 3*30 = 90
    #[test]
    fn bwd_weight_hand_computed() {
        let x = vec![1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let dy = vec![10.0_f32, 20.0, 30.0];
        let bucket_idx = vec![0_i32, 0, 1];
        let mut grad_w = vec![-1.0_f32; 4]; // overwrite check
        dense_mm_bwd_weight_bucket_cpu(&x, &dy, &bucket_idx, &mut grad_w, 3, 2, 1, 2);
        assert_eq!(grad_w, vec![50.0_f32, 50.0, 90.0, 90.0]);
    }

    /// out-of-range bucket position contributes to no weight cell.
    #[test]
    fn bwd_weight_out_of_range_position_ignored() {
        let x = vec![1.0_f32, 1.0, 100.0, 100.0];
        let dy = vec![10.0_f32, 10.0];
        let bucket_idx = vec![0_i32, -1]; // second position out of range
        let mut grad_w = vec![0.0_f32; 4];
        dense_mm_bwd_weight_bucket_cpu(&x, &dy, &bucket_idx, &mut grad_w, 2, 2, 1, 2);
        // only b=0 in bucket 0: grad_w[0][0][0] = 1*10 = 10, [0][0][1] = 10; bucket 1 = 0
        assert_eq!(grad_w, vec![10.0_f32, 10.0, 0.0, 0.0]);
    }

    /// bias_grad_bucket hand-computed (accumulate). num_buckets=2, out_dim=2, batch=3.
    /// dy = [[1,2],[3,4],[5,6]], bucket_idx = [0, 1, 0].
    /// bucket0 collects b=0,2: [1+5, 2+6] = [6, 8]
    /// bucket1 collects b=1:   [3, 4]
    /// grad_bias starts [10,20,30,40] → [16, 28, 33, 44]
    #[test]
    fn bias_grad_bucket_accumulates() {
        let dy = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bucket_idx = vec![0_i32, 1, 0];
        let mut grad_bias = vec![10.0_f32, 20.0, 30.0, 40.0];
        bias_grad_bucket_cpu(&dy, &bucket_idx, &mut grad_bias, 3, 2, 2);
        assert_eq!(grad_bias, vec![16.0_f32, 28.0, 33.0, 44.0]);
    }

    #[test]
    fn bias_grad_bucket_skips_out_of_range() {
        let dy = vec![1.0_f32, 2.0, 99.0, 99.0];
        let bucket_idx = vec![0_i32, 9]; // 9 >= num_buckets=2
        let mut grad_bias = vec![0.0_f32; 4];
        bias_grad_bucket_cpu(&dy, &bucket_idx, &mut grad_bias, 2, 2, 2);
        // only b=0 in bucket 0
        assert_eq!(grad_bias, vec![1.0_f32, 2.0, 0.0, 0.0]);
    }

    /// chain-rule consistency for one bucket: with bias=0, sum over a single bucket
    /// <dy, y> == <x, dx>. Use all positions in bucket 0.
    #[test]
    fn fwd_bwd_input_inner_product_identity_single_bucket() {
        let num_buckets = 3;
        let in_dim = 4;
        let out_dim = 5;
        let batch = 3;
        let w: Vec<f32> = (0..num_buckets * out_dim * in_dim)
            .map(|i| (i as f32) * 0.03 - 0.4)
            .collect();
        let bias = vec![0.0_f32; num_buckets * out_dim];
        let x: Vec<f32> = (0..batch * in_dim)
            .map(|i| (i as f32) * 0.07 + 0.1)
            .collect();
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| (i as f32) * 0.2 - 0.9)
            .collect();
        let bucket_idx = vec![1_i32, 1, 1]; // all bucket 1

        let mut y = vec![0.0_f32; batch * out_dim];
        dense_mm_fwd_bucket_cpu(
            &x,
            &w,
            &bias,
            &bucket_idx,
            &mut y,
            batch,
            in_dim,
            out_dim,
            num_buckets,
        );
        let mut dx = vec![0.0_f32; batch * in_dim];
        dense_mm_bwd_input_bucket_cpu(
            &dy,
            &w,
            &bucket_idx,
            &mut dx,
            batch,
            in_dim,
            out_dim,
            num_buckets,
        );
        let lhs: f64 = (0..batch * out_dim)
            .map(|i| dy[i] as f64 * y[i] as f64)
            .sum();
        let rhs: f64 = (0..batch * in_dim)
            .map(|i| x[i] as f64 * dx[i] as f64)
            .sum();
        assert!((lhs - rhs).abs() < 1e-3, "lhs={lhs} rhs={rhs}");
    }

    #[test]
    fn empty_batch_is_noop() {
        let mut y: Vec<f32> = vec![];
        dense_mm_fwd_bucket_cpu(&[], &[], &[], &[], &mut y, 0, 2, 1, 2);
        assert!(y.is_empty());
        let mut grad_bias = vec![1.0_f32; 4];
        bias_grad_bucket_cpu(&[], &[], &mut grad_bias, 0, 2, 2);
        assert_eq!(grad_bias, vec![1.0_f32; 4]);
    }
}
