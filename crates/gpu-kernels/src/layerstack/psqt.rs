//! PSQT shortcut kernel (HalfKA 入力層 per-feature × per-bucket スカラー prior) の
//! reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn psqt_diff_sparse_fwd_inplace` /
//! `#[kernel] fn psqt_diff_sparse_bwd`) は `bins/nnue_train/src/kernels/
//! layerstack.rs` に inline 定義されている。本 module の `*_cpu` は GPU と同じ
//! ロジックを host に書き写したもので、GPU↔CPU 数値同等性テストの reference に
//! 使う。
//!
//! ## アルゴリズム
//!
//! dual-perspective psqt の per-bucket スカラー prior。layout は row-major
//! `psqt_w[feat * num_buckets + bucket]` (.bin save format と同じ feature-major
//! order)。
//!
//! ```text
//! forward (in-place add):
//!   bucket = bucket_idx[b]
//!   if 0 <= bucket < num_buckets:
//!     sum_stm  = Σ_ni psqt_w[stm_indices[b, ni],  bucket] (idx>=0 かつ <ft_in)
//!     sum_nstm = Σ_ni psqt_w[nstm_indices[b, ni], bucket]
//!     net_output[b] += 0.5 * (sum_stm - sum_nstm)
//!
//! backward (accumulate semantics, host 0 初期化):
//!   for (b, ni < nnz_arr[b]):
//!     bucket = bucket_idx[b]; if invalid: skip
//!     g = 0.5 * dnet[b]
//!     psqt_w_grad[stm_indices[b, ni],  bucket] += +g
//!     psqt_w_grad[nstm_indices[b, ni], bucket] += -g
//! ```

/// forward (in-place add)。
///
/// `net_output[b]` に `0.5 * (sum_stm - sum_nstm)` を加算する。caller は `net_output`
/// を事前に `l3_out + l1_skip` 等で初期化済の前提。`bucket_idx[b] < 0` または
/// `>= num_buckets` は skip。`indices` の `-1` padding / `idx >= ft_in` も skip。
#[allow(clippy::too_many_arguments)]
pub fn psqt_diff_sparse_fwd_inplace_cpu(
    psqt_w: &[f32],
    stm_indices: &[i32],
    nstm_indices: &[i32],
    nnz_arr: &[i32],
    bucket_idx: &[i32],
    net_output: &mut [f32],
    batch: usize,
    max_active: usize,
    num_buckets: usize,
    ft_in: usize,
) {
    for b in 0..batch {
        let bucket = bucket_idx[b];
        if bucket < 0 || (bucket as usize) >= num_buckets {
            continue;
        }
        let bucket_u = bucket as usize;
        let base = b * max_active;
        let mut sum_stm = 0.0_f32;
        let mut sum_nstm = 0.0_f32;
        for ni in 0..nnz_arr[b] as usize {
            let idx_s = stm_indices[base + ni];
            if idx_s >= 0 && (idx_s as usize) < ft_in {
                sum_stm += psqt_w[(idx_s as usize) * num_buckets + bucket_u];
            }
            let idx_n = nstm_indices[base + ni];
            if idx_n >= 0 && (idx_n as usize) < ft_in {
                sum_nstm += psqt_w[(idx_n as usize) * num_buckets + bucket_u];
            }
        }
        net_output[b] += 0.5_f32 * (sum_stm - sum_nstm);
    }
}

/// backward (accumulate semantics)。
///
/// `psqt_w_grad` は呼出前に 0 初期化されている前提 (GPU は `memset_async(0)`)。stm
/// 側に `+0.5*dnet[b]`、nstm 側に `-0.5*dnet[b]` を、対応 `(feat, bucket)` cell に
/// add する。重複 index は累積 (atomic sum)。`nnz_arr[b]` は position b の実 active
/// 数で、`ni >= nnz_arr[b]` の padding slot は走査しない (GPU kernel の early-out と
/// 同契約。`nnz` は index の row stride、`0 <= nnz_arr[b] <= nnz`)。
#[allow(clippy::too_many_arguments)]
pub fn psqt_diff_sparse_bwd_cpu(
    dnet: &[f32],
    stm_indices: &[i32],
    nstm_indices: &[i32],
    nnz_arr: &[i32],
    bucket_idx: &[i32],
    psqt_w_grad: &mut [f32],
    batch: usize,
    nnz: usize,
    num_buckets: usize,
    ft_in: usize,
) {
    for b in 0..batch {
        let bucket = bucket_idx[b];
        if bucket < 0 || (bucket as usize) >= num_buckets {
            continue;
        }
        let bucket_u = bucket as usize;
        let half_g = 0.5_f32 * dnet[b];
        for ni in 0..nnz_arr[b] as usize {
            let idx_s = stm_indices[b * nnz + ni];
            if idx_s >= 0 && (idx_s as usize) < ft_in {
                psqt_w_grad[(idx_s as usize) * num_buckets + bucket_u] += half_g;
            }
            let idx_n = nstm_indices[b * nnz + ni];
            if idx_n >= 0 && (idx_n as usize) < ft_in {
                psqt_w_grad[(idx_n as usize) * num_buckets + bucket_u] -= half_g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 単一 position、単一 bucket、active feature 1 個ずつで「+w_stm − w_nstm」の
    /// 0.5× が net_output に乗ることを確認。
    #[test]
    fn fwd_single_position_basic() {
        // ft_in=3, num_buckets=2, batch=1, nnz=1
        // psqt_w[feat * 2 + bucket]:
        //   feat 0: bucket 0 = 1.0, bucket 1 = 2.0
        //   feat 1: bucket 0 = 4.0, bucket 1 = 8.0
        //   feat 2: bucket 0 = 0.0, bucket 1 = 0.0
        let psqt_w = vec![1.0, 2.0, 4.0, 8.0, 0.0, 0.0];
        let stm = vec![1_i32]; // feat 1
        let nstm = vec![0_i32]; // feat 0
        let bucket = vec![1_i32]; // bucket=1
        let mut out = vec![10.0_f32]; // pre-existing net_output value
        psqt_diff_sparse_fwd_inplace_cpu(&psqt_w, &stm, &nstm, &[1], &bucket, &mut out, 1, 1, 2, 3);
        // delta = 0.5 * (psqt_w[1,1] - psqt_w[0,1]) = 0.5 * (8 - 2) = 3
        assert_eq!(out[0], 13.0);
    }

    /// padding (-1) と out-of-range は skip される。
    #[test]
    fn fwd_padding_and_oob_skipped() {
        let psqt_w = vec![1.0, 2.0, 3.0, 4.0];
        let stm = vec![-1_i32, 99]; // 全 invalid
        let nstm = vec![-1_i32, 99];
        let bucket = vec![0_i32];
        let mut out = vec![5.0_f32];
        psqt_diff_sparse_fwd_inplace_cpu(&psqt_w, &stm, &nstm, &[2], &bucket, &mut out, 1, 2, 2, 2);
        assert_eq!(out[0], 5.0, "delta = 0 when all indices invalid");
    }

    /// bucket_idx 不正 (`< 0` or `>= num_buckets`) は skip。
    #[test]
    fn fwd_invalid_bucket_skipped() {
        let psqt_w = vec![1.0, 2.0];
        let stm = vec![0_i32];
        let nstm = vec![-1_i32];
        let bucket_neg = vec![-1_i32];
        let bucket_oob = vec![9_i32];
        let mut out_neg = vec![7.0_f32];
        let mut out_oob = vec![7.0_f32];
        psqt_diff_sparse_fwd_inplace_cpu(
            &psqt_w,
            &stm,
            &nstm,
            &[1],
            &bucket_neg,
            &mut out_neg,
            1,
            1,
            1,
            2,
        );
        psqt_diff_sparse_fwd_inplace_cpu(
            &psqt_w,
            &stm,
            &nstm,
            &[1],
            &bucket_oob,
            &mut out_oob,
            1,
            1,
            1,
            2,
        );
        assert_eq!(out_neg[0], 7.0);
        assert_eq!(out_oob[0], 7.0);
    }

    /// backward の符号 + 0.5 scale が stm/nstm 別々に効く。
    #[test]
    fn bwd_signs_and_scale() {
        let dnet = vec![4.0_f32];
        let stm = vec![1_i32];
        let nstm = vec![2_i32];
        let bucket = vec![0_i32];
        let mut grad = vec![0.0_f32; 3 * 2]; // ft_in=3, num_buckets=2
        psqt_diff_sparse_bwd_cpu(&dnet, &stm, &nstm, &[1], &bucket, &mut grad, 1, 1, 2, 3);
        // stm += 0.5*4 = 2 at (feat=1, bucket=0)
        // nstm += -0.5*4 = -2 at (feat=2, bucket=0)
        assert_eq!(grad[2], 2.0); // feat=1, bucket=0 → 1*2 + 0
        assert_eq!(grad[4], -2.0); // feat=2, bucket=0 → 2*2 + 0
        assert_eq!(grad[0], 0.0); // feat=0, bucket=0
        assert_eq!(grad[1], 0.0); // feat=0, bucket=1
    }

    /// 同一 feature が複数 (b, ni) で重複ヒットすると勾配が和算される (atomic 想定)。
    #[test]
    fn bwd_duplicate_indices_accumulate() {
        let dnet = vec![10.0_f32, 20.0_f32];
        // batch 0: stm = [3, 3], nstm = [-1, -1] (feat 3 を 2 回)
        // batch 1: stm = [3, -1], nstm = [-1, -1] (feat 3 を 1 回)
        let stm = vec![3_i32, 3, 3, -1];
        let nstm = vec![-1_i32, -1, -1, -1];
        let bucket = vec![0_i32, 0];
        let mut grad = vec![0.0_f32; 5]; // ft_in=5, num_buckets=1 → 5*1 = 5
        psqt_diff_sparse_bwd_cpu(&dnet, &stm, &nstm, &[2, 2], &bucket, &mut grad, 2, 2, 1, 5);
        // feat 3 + bucket 0: b=0 が 2 回 (0.5*10*2=10) + b=1 が 1 回 (0.5*20=10) = 20
        assert_eq!(grad[3], 20.0);
    }

    /// forward/backward 一貫性: forward の delta を grad として backward に流すと、
    /// 既存 weight 上に矛盾なく蓄積される (sanity check)。
    #[test]
    fn fwd_bwd_consistency() {
        let psqt_w = vec![1.0_f32, -2.0, 3.0, 4.0]; // ft_in=2, num_buckets=2
        let stm = vec![0_i32];
        let nstm = vec![1_i32];
        let bucket = vec![1_i32];
        let mut out = vec![0.0_f32];
        psqt_diff_sparse_fwd_inplace_cpu(&psqt_w, &stm, &nstm, &[1], &bucket, &mut out, 1, 1, 2, 2);
        // delta = 0.5 * (psqt_w[0,1] - psqt_w[1,1]) = 0.5 * (-2 - 4) = -3
        assert_eq!(out[0], -3.0);
        // backward に dnet=delta を入れると stm 側 += 0.5*-3 = -1.5、nstm 側 += +1.5。
        let mut grad = vec![0.0_f32; 4];
        psqt_diff_sparse_bwd_cpu(
            &[-3.0_f32],
            &stm,
            &nstm,
            &[1],
            &bucket,
            &mut grad,
            1,
            1,
            2,
            2,
        );
        assert_eq!(grad[1], -1.5); // feat=0, bucket=1 → 0*2 + 1
        assert_eq!(grad[3], 1.5); // feat=1, bucket=1 → 1*2 + 1
    }
}
