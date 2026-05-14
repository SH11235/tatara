//! Sparse feature transform forward kernel (HalfKA_hm 用) の reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn sparse_ft_forward` は bin entry
//! (`bins/nnue_train/src/main.rs`) に inline 定義されている (cuda-oxide
//! rustc-codegen-cuda backend の bin-entry 制約)。本 module の
//! `sparse_ft_forward_cpu` は GPU と同じロジックを host で素直に書き写したもので、
//! GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム (bullet 上流 `linear/sparse.rs::SparseMatmul::evaluate` に等価)
//!
//! NNUE 入力層の sparse feature transform forward。`indices` は per-position の
//! active feature index 配列 (`-1` を padding とする)、`weight` を column-major
//! で持つ。1 thread = 1 (batch, row) tuple で:
//!
//! ```text
//! per (batch_index bi, row_index ri):
//!     sum = 0
//!     for ni in 0..nnz:
//!         idx = indices[bi * nnz + ni]
//!         if idx >= 0 && (idx as usize) < cols:
//!             sum += weight[idx * rows + ri]      # column-major
//!     out[bi * rows + ri] = sum
//! ```
//!
//! - `weight` (size `rows * cols`): **column-major** (bullet 上流 `evaluate` の
//!   `d.read(rows * idx + ri)` と同型)。column j の row i は
//!   `weight[j * rows + i]` 位置
//! - `indices` (size `batch * nnz`): per position の active feature index、
//!   `-1` は padding (skip)。`>= cols` の値も silent skip (bullet 上流の
//!   defensive check と同型)
//! - `out` (size `batch * rows`): per position の FT output、row-major で
//!   `out[bi * rows + ri]` 位置
//!
//! ## bullet 上流との対応 / divergence
//!
//! - bullet `SparseMatmul::evaluate` (`linear/sparse.rs:61-91`) は `DValue` 経由
//!   の generic な dtype を受けるが、本実装は `f32` 固定 (NNUE training の hot
//!   path で f32 のみを扱う)
//! - bullet は `nnz` ループを runtime fusion で吸収する。本実装は `nnz` を
//!   build-time の引数として受けて kernel 内 `for ni in 0..nnz` を素直に展開
//!   (memory bandwidth bound なので unroll の差は小さい想定)
//! - **silent skip on `idx >= cols`**: bullet 上流も `if idx >= 0 && (idx as
//!   usize) < cols` で defensive にチェックする (`linear/sparse.rs:82`)。本実装
//!   も同型に揃える
//!
//! ## cuda-oxide 制限
//!
//! - GPU kernel 側は `+` / `*` のみ + indexing (i32 比較含む) で cuda-oxide 制限
//!   に当たらない (Stage 1-5 forward と同等の単純 pointwise op)
//! - i32 比較 `idx >= 0` / `idx as usize < cols` は cuda-oxide で問題なく
//!   compile される (Stage 1-6 grad の `if b < 0` / `if b > 7` と同型)

/// Reference CPU 実装。
///
/// Out-of-place 出力: `out[bi * rows + ri]` を 1 entry ずつ埋める。
///
/// 入力前提:
/// - `weight.len() == rows * cols` (column-major、`weight[col * rows + row]`)
/// - `indices.len() == batch * nnz` (`-1` padding 許容、`>= cols` も silent skip)
/// - `out.len() == batch * rows` (row-major、`out[batch * rows + row]`)
///
/// 引数数 (7) は bullet 上流 evaluate の入出力 + sparse 形状を漏れなく渡すため
/// `clippy::too_many_arguments` を allow (Stage 1 / Stage 2 と同 convention)。
#[allow(clippy::too_many_arguments)]
pub fn sparse_ft_forward_cpu(
    weight: &[f32],
    indices: &[i32],
    out: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
    nnz: usize,
) {
    for bi in 0..batch {
        for ri in 0..rows {
            let mut sum = 0.0_f32;
            for ni in 0..nnz {
                let idx = indices[bi * nnz + ni];
                if idx >= 0 && (idx as usize) < cols {
                    sum += weight[(idx as usize) * rows + ri];
                }
            }
            out[bi * rows + ri] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bullet 上流 (`linear/sparse.rs::tests::evaluate`、`:243-253`) と同じ
    /// shape (batch=2, rows=2, cols=3, nnz=4) で完全一致。期待値 [2, 4, 10, 14]
    /// は bullet の test と同じ。本テストが緑なら bullet 上流と同レイアウトで
    /// 動作することが保証される。
    #[test]
    fn matches_bullet_upstream_evaluate_test() {
        let weights = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        // 2 batches × 4 nnz = 8 indices
        let indices = vec![0_i32, 1, -1, -1, 2, 2, 1, 0];
        let mut out = vec![0.0_f32; 4]; // batch * rows = 2*2

        sparse_ft_forward_cpu(&weights, &indices, &mut out, 2, 2, 3, 4);

        assert_eq!(out, vec![2.0_f32, 4.0, 10.0, 14.0]);
    }

    /// 全 padding (-1) の position は output が 0 になる。
    #[test]
    fn all_padding_yields_zero() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0]; // rows=2, cols=2
        let indices = vec![-1_i32; 6]; // 1 batch × 6 nnz、全 padding
        let mut out = vec![999.0_f32, 999.0]; // 初期値で汚染、kernel が上書きする
        sparse_ft_forward_cpu(&weights, &indices, &mut out, 1, 2, 2, 6);
        assert_eq!(out, vec![0.0_f32, 0.0]);
    }

    /// `idx >= cols` の異常入力 (bullet 上流 `evaluate` で `idx as usize < cols`
    /// チェックされる経路) は silent skip。kernel が OOB read しないこと。
    #[test]
    fn out_of_range_index_is_silently_skipped() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0]; // rows=2, cols=2
        // idx=5 は cols=2 を超える → skip される
        let indices = vec![0_i32, 5, -1];
        let mut out = vec![0.0_f32; 2];
        sparse_ft_forward_cpu(&weights, &indices, &mut out, 1, 2, 2, 3);
        // idx=0 だけが効く: row 0 → w[0*2+0] = 1.0、row 1 → w[0*2+1] = 2.0
        assert_eq!(out, vec![1.0_f32, 2.0]);
    }

    /// 同 index の重複 (`indices = [0, 0, 1]`) は **重み合計** になる
    /// (bullet 上流 `evaluate` の `for ni in 0..nnz` ループで重複 idx を別々に
    /// 加算するのと同型)。
    #[test]
    fn duplicate_indices_are_summed() {
        let weights = vec![1.0_f32, 10.0, 2.0, 20.0]; // rows=2, cols=2
        // 1 batch × 3 nnz、idx=0 を 2 回 + idx=1 を 1 回
        let indices = vec![0_i32, 0, 1];
        let mut out = vec![0.0_f32; 2];
        sparse_ft_forward_cpu(&weights, &indices, &mut out, 1, 2, 2, 3);
        // row 0: w[0*2+0]*2 + w[1*2+0] = 1*2 + 2 = 4
        // row 1: w[0*2+1]*2 + w[1*2+1] = 10*2 + 20 = 40
        assert_eq!(out, vec![4.0_f32, 40.0]);
    }

    /// batch=0 / rows=0 / cols=0 / nnz=0 のいずれか edge case でも panic せず
    /// no-op。`out.len() == batch * rows == 0` も許容される。
    #[test]
    fn zero_dimension_is_no_op() {
        let weights: Vec<f32> = vec![];
        let indices: Vec<i32> = vec![];
        let mut out: Vec<f32> = vec![];
        sparse_ft_forward_cpu(&weights, &indices, &mut out, 0, 0, 0, 0);
        assert!(out.is_empty());

        // nnz=0 のみ: padding ループが 0 周で sum = 0 になる
        let weights = vec![1.0_f32, 2.0];
        let indices: Vec<i32> = vec![];
        let mut out = vec![999.0_f32, 999.0];
        sparse_ft_forward_cpu(&weights, &indices, &mut out, 1, 2, 1, 0);
        assert_eq!(out, vec![0.0_f32, 0.0]);
    }
}
