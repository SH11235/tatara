//! Sparse feature transform backward kernel (HalfKA_hm 用、atomic scatter) の
//! reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn sparse_ft_backward`) は `bins/nnue_train/src/main.rs` に
//! inline 定義されている (cuda-oxide rustc-codegen-cuda backend は bin entry
//! 経由で到達可能な kernel しか PTX 化しないため)。本 module の
//! `sparse_ft_backward_cpu` は GPU と同じロジックを host に書き写したもので、
//! GPU↔CPU 数値同等性テストの reference に使う。
//!
//! ## アルゴリズム (bullet 上流 `linear/sparse.rs::SparseMatmulBwd::evaluate` に等価)
//!
//! sparse forward の対 backward。column-major weight layout
//! (`grad_weight[idx * rows + ri]`) は forward と同型。複数 (bi, ni) が同じ
//! (idx, ri) cell に書き込むため **atomic scatter** で実装する (GPU 側のみ、
//! CPU reference は host single-thread で sequential add)。
//!
//! ```text
//! per (batch_index bi, row_index ri):
//!     g = grad_out[bi * rows + ri]
//!     for ni in 0..nnz:
//!         idx = indices[bi * nnz + ni]
//!         if idx >= 0 && (idx as usize) < cols:
//!             grad_weight[idx * rows + ri] += g       # GPU: atomicAdd, CPU: 累積
//! ```
//!
//! - **`grad_weight` の初期化**: 本 fn は **accumulate** semantics で既存値に
//!   add する。host が呼び出し前に `device.memset(0)` (or `vec![0.0; ...]`) で
//!   zero clear する責務
//! - bullet 上流 `evaluate` は test 用に冒頭で `o.write(idx, zero)` を行うが、
//!   本 fn は production kernel semantics (accumulate) に揃えるため zero clear を
//!   含めない。test 側で初期化することで bullet 上流の expected 値を再現する
//! - layout: column-major weight (forward と同 `weight[idx * rows + ri]`)
//!
//! ## bullet 上流との差分
//!
//! - bullet `evaluate_bwd` test (batch=2/rows=3/cols=3/nnz=4、
//!   grad_out=[0,1,2,3,4,5]、indices=[0,1,-1,-1,2,2,1,0]、
//!   expected=[3,5,7,3,5,7,6,8,10]) と同 fixture を CPU test で 1:1 再現
//! - thread 配置: bullet 上流は PointwiseIR で per-row reduction、batch 軸別 unroll
//!   想定。本実装は **flat 1D `tid = bi * rows + ri`** で batch 軸も込み (forward と
//!   同型 idiom、atomic scatter で衝突を吸収)
//! - `SparseMatmulBwdMulti` (複数 backward 集約) は対象外
//!
//! ## cuda-oxide 制限
//!
//! GPU kernel は `+` / `*` / i32 比較 + `DeviceAtomicF32::fetch_add` で cuda-oxide
//! 制限非該当。`f32::clamp` / `sqrt` 等は使わない。

/// Reference CPU 実装。
///
/// In-place accumulate: `grad_weight` に既存値を add する (host が呼び出し前に
/// 0 で初期化する責務)。
///
/// 入力前提:
/// - `grad_out.len() == batch * rows`
/// - `indices.len() == batch * nnz` (`-1` padding 許容、`>= cols` も silent skip)
/// - `grad_weight.len() == rows * cols` (column-major、`grad_weight[col * rows + row]`)
///
/// 引数数 (7) は bullet 上流 evaluate と同型のため
/// `clippy::too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn sparse_ft_backward_cpu(
    grad_out: &[f32],
    indices: &[i32],
    grad_weight: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
    nnz: usize,
) {
    for bi in 0..batch {
        for ri in 0..rows {
            let g = grad_out[bi * rows + ri];
            for ni in 0..nnz {
                let idx = indices[bi * nnz + ni];
                if idx >= 0 && (idx as usize) < cols {
                    grad_weight[(idx as usize) * rows + ri] += g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bullet 上流 (`linear/sparse.rs::tests::evaluate_bwd`) と同じ shape
    /// (batch=2, rows=3, cols=3, nnz=4) で完全一致。期待値
    /// [3,5,7,3,5,7,6,8,10] は bullet の test と同じ。本テストが緑なら bullet
    /// 上流と同レイアウト + accumulate semantics で動作することが保証される。
    #[test]
    fn matches_bullet_upstream_evaluate_bwd_test() {
        let grad_out = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0_i32, 1, -1, -1, 2, 2, 1, 0];
        let mut grad_weight = vec![0.0_f32; 9]; // rows * cols = 3*3
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 2, 3, 3, 4);
        assert_eq!(
            grad_weight,
            vec![3.0_f32, 5.0, 7.0, 3.0, 5.0, 7.0, 6.0, 8.0, 10.0]
        );
    }

    /// 全 padding (-1) なら grad_weight は変化なし (初期 0 のまま)。
    #[test]
    fn all_padding_yields_no_accumulation() {
        let grad_out = vec![1.0_f32, 2.0, 3.0]; // batch=1, rows=3
        let indices = vec![-1_i32; 4]; // 全 padding
        let mut grad_weight = vec![0.0_f32; 6]; // rows=3, cols=2
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 1, 3, 2, 4);
        assert_eq!(grad_weight, vec![0.0_f32; 6]);
    }

    /// `idx >= cols` の異常入力は silent skip (bullet 上流と同型 defensive)。
    #[test]
    fn out_of_range_index_is_silently_skipped() {
        let grad_out = vec![1.0_f32, 1.0]; // batch=1, rows=2
        let indices = vec![0_i32, 5, -1]; // idx=5 は cols=2 を超える、skip
        let mut grad_weight = vec![0.0_f32; 4]; // rows=2, cols=2
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 1, 2, 2, 3);
        // idx=0 のみ効く: grad_weight[0*2+0] += 1.0、grad_weight[0*2+1] += 1.0
        assert_eq!(grad_weight, vec![1.0_f32, 1.0, 0.0, 0.0]);
    }

    /// 同 index の重複 (`indices = [0, 0, 1]`) は **重み合計** に accumulate
    /// (bullet 上流 `evaluate` と同型、kernel は atomic scatter で衝突を吸収)。
    #[test]
    fn duplicate_indices_are_summed() {
        let grad_out = vec![10.0_f32, 20.0]; // batch=1, rows=2
        let indices = vec![0_i32, 0, 1]; // idx=0 を 2 回 + idx=1 を 1 回
        let mut grad_weight = vec![0.0_f32; 4]; // rows=2, cols=2
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 1, 2, 2, 3);
        // idx=0 が 2 回: grad_weight[0*2+0] += 10*2 = 20、grad_weight[0*2+1] += 20*2 = 40
        // idx=1 が 1 回: grad_weight[1*2+0] += 10、grad_weight[1*2+1] += 20
        assert_eq!(grad_weight, vec![20.0_f32, 40.0, 10.0, 20.0]);
    }

    /// **accumulate semantics**: 既存値に add される (host が呼び出し前に 0 init
    /// する責務)。先行値 1.0 を入れて呼び出し、結果が 「先行値 + 新加算分」 に
    /// なることを確認。
    #[test]
    fn accumulates_into_existing_grad_weight() {
        let grad_out = vec![5.0_f32]; // batch=1, rows=1
        let indices = vec![0_i32]; // idx=0
        let mut grad_weight = vec![1.0_f32]; // 先行値 1.0、cols=1
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 1, 1, 1, 1);
        // accumulate: 1.0 + 5.0 = 6.0
        assert_eq!(grad_weight, vec![6.0_f32]);
    }

    /// 0 dim edge case (panic せず no-op)。
    #[test]
    fn zero_dimension_is_no_op() {
        let grad_out: Vec<f32> = vec![];
        let indices: Vec<i32> = vec![];
        let mut grad_weight: Vec<f32> = vec![];
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 0, 0, 0, 0);
        assert!(grad_weight.is_empty());

        // nnz=0: padding ループが 0 周で grad_weight は変化なし
        let grad_out = vec![1.0_f32, 2.0];
        let indices: Vec<i32> = vec![];
        let mut grad_weight = vec![0.0_f32, 0.0];
        sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_weight, 1, 2, 1, 0);
        assert_eq!(grad_weight, vec![0.0_f32, 0.0]);
    }

    /// **forward / backward 整合性 (gradient check)**: forward で得た sum を
    /// 直接 backward に流すと、重複 index 数に応じて grad_weight が累積される。
    /// forward `out[bi*rows+ri] = Σ weight[idx*rows+ri]`、backward
    /// `grad_weight[idx*rows+ri] += grad_out[bi*rows+ri]` の対応が崩れていない
    /// ことを確認 (autograd 経路の sanity check 用途)。
    #[test]
    fn forward_backward_consistency() {
        use crate::sparse::sparse_ft_forward::sparse_ft_forward_cpu;

        let weight = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0_i32, 1, -1, -1, 2, 2, 1, 0];
        let batch = 2;
        let rows = 2;
        let cols = 3;
        let nnz = 4;

        let mut out = vec![0.0_f32; batch * rows];
        sparse_ft_forward_cpu(&weight, &indices, &mut out, batch, rows, cols, nnz);
        // forward の結果を grad_out として backward に渡し、grad_weight を計算
        let mut grad_weight = vec![0.0_f32; rows * cols];
        sparse_ft_backward_cpu(&out, &indices, &mut grad_weight, batch, rows, cols, nnz);

        // 期待値: out = [2, 4, 10, 14] (forward test 同 fixture)
        // 各 (bi, ri) で out[bi*rows+ri] が active なすべての idx に scatter
        // batch 0 (out[0]=2, out[1]=4): idx in [0, 1] 2 つ
        //   col 0 row 0: += 2、col 0 row 1: += 4
        //   col 1 row 0: += 2、col 1 row 1: += 4
        // batch 1 (out[2]=10, out[3]=14): idx in [2, 2, 1, 0] (重複 idx=2 が 2 回)
        //   col 2 row 0: += 10*2 = 20、col 2 row 1: += 14*2 = 28
        //   col 1 row 0: += 10、col 1 row 1: += 14
        //   col 0 row 0: += 10、col 0 row 1: += 14
        // 合計 (column-major、col j row i は grad_weight[j*rows+i]):
        //   col 0: row 0 = 2+10 = 12、row 1 = 4+14 = 18
        //   col 1: row 0 = 2+10 = 12、row 1 = 4+14 = 18
        //   col 2: row 0 = 20、row 1 = 28
        // → grad_weight = [12, 18, 12, 18, 20, 28]
        assert_eq!(grad_weight, vec![12.0_f32, 18.0, 12.0, 18.0, 20.0, 28.0]);
    }
}
