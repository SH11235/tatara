//! Sparse feature transform forward kernel (HalfKA_hm 用) の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn sparse_ft_forward`) は `bins/nnue_train/src/kernels/` に
//! 定義されている (cuda-oxide rustc-codegen-cuda backend は bin entry
//! 経由で到達可能な kernel しか PTX 化しないため)。本 module の
//! `sparse_ft_forward_cpu` は GPU と同じロジックを host に書き写したもので、
//! GPU↔CPU 数値同等性テストの reference に使う。
//!
//! ## アルゴリズム
//!
//! NNUE 入力層の sparse feature transform forward。`indices` は per-position の
//! active feature index 配列 (`-1` を padding とする)、`weight` を column-major
//! で持つ。1 thread = 1 (batch, row) tuple で:
//!
//! ```text
//! per (batch_index bi, row_index ri):
//!     sum = 0
//!     for ni in 0..nnz_arr[bi]:
//!         idx = indices[bi * max_active + ni]
//!         if idx >= 0 && (idx as usize) < cols:
//!             sum += weight[idx * rows + ri]      # column-major
//!     out[bi * rows + ri] = sum
//! ```
//!
//! - `weight` (size `rows * cols`): **column-major**。column j の row i は
//!   `weight[j * rows + i]` 位置
//! - `indices` (size `batch * max_active`): per position の active feature index、
//!   `-1` は padding (skip)。`>= cols` の値も defensive に silent skip
//! - `nnz_arr` (size `batch`): position ごとの実 active feature 数
//! - `out` (size `batch * rows`): per position の FT output、row-major で
//!   `out[bi * rows + ri]` 位置
//!
//! `f32` 固定 (NNUE training hot path で f32 のみ扱う)。
//!
//! ## cuda-oxide 制限
//!
//! GPU kernel 側は `+` / `*` + indexing (i32 比較含む) のみで cuda-oxide 制限に
//!当たらない。`idx >= 0` / `(idx as usize) < cols` の i32 比較も問題なく compile
//! される。

/// Reference CPU 実装。
///
/// Out-of-place 出力: `out[bi * rows + ri]` を 1 entry ずつ埋める。
///
/// 入力前提:
/// - `weight.len() == rows * cols` (column-major、`weight[col * rows + row]`)
/// - `indices.len() == batch * max_active` (`-1` padding 許容、`>= cols` も silent skip)
/// - `nnz_arr.len() == batch`、各値は `0..=max_active`
/// - `out.len() == batch * rows` (row-major、`out[batch * rows + row]`)
///
/// 引数数 (7) は入出力 + sparse 形状を漏れなく渡すため
/// `clippy::too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn sparse_ft_forward_cpu(
    weight: &[f32],
    indices: &[i32],
    nnz_arr: &[i32],
    out: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
    max_active: usize,
) {
    for bi in 0..batch {
        for ri in 0..rows {
            let mut sum = 0.0_f32;
            for ni in 0..nnz_arr[bi] as usize {
                let idx = indices[bi * max_active + ni];
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

    /// shape (batch=2, rows=2, cols=3, nnz=4) で期待値 [2, 4, 10, 14] を再現
    /// する基準テスト (column-major weight + `-1` padding + duplicate idx の
    /// 三点を 1 ケースで cover)。
    #[test]
    fn matches_reference_evaluate_test() {
        let weights = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        // 2 batches × 4 nnz = 8 indices
        let indices = vec![0_i32, 1, -1, -1, 2, 2, 1, 0];
        let mut out = vec![0.0_f32; 4]; // batch * rows = 2*2

        sparse_ft_forward_cpu(&weights, &indices, &[2, 4], &mut out, 2, 2, 3, 4);

        assert_eq!(out, vec![2.0_f32, 4.0, 10.0, 14.0]);
    }

    /// 全 padding (-1) の position は output が 0 になる。
    #[test]
    fn all_padding_yields_zero() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0]; // rows=2, cols=2
        let indices = vec![-1_i32; 6]; // 1 batch × 6 nnz、全 padding
        let mut out = vec![999.0_f32, 999.0]; // 初期値で汚染、kernel が上書きする
        sparse_ft_forward_cpu(&weights, &indices, &[6], &mut out, 1, 2, 2, 6);
        assert_eq!(out, vec![0.0_f32, 0.0]);
    }

    /// `idx >= cols` の異常入力は silent skip。kernel が OOB read しないこと。
    #[test]
    fn out_of_range_index_is_silently_skipped() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0]; // rows=2, cols=2
        // idx=5 は cols=2 を超える → skip される
        let indices = vec![0_i32, 5, -1];
        let mut out = vec![0.0_f32; 2];
        sparse_ft_forward_cpu(&weights, &indices, &[3], &mut out, 1, 2, 2, 3);
        // idx=0 だけが効く: row 0 → w[0*2+0] = 1.0、row 1 → w[0*2+1] = 2.0
        assert_eq!(out, vec![1.0_f32, 2.0]);
    }

    /// 同 index の重複 (`indices = [0, 0, 1]`) は **重み合計** になる
    /// (`for ni in 0..nnz` ループで重複 idx を別々に加算する仕様)。
    #[test]
    fn duplicate_indices_are_summed() {
        let weights = vec![1.0_f32, 10.0, 2.0, 20.0]; // rows=2, cols=2
        // 1 batch × 3 nnz、idx=0 を 2 回 + idx=1 を 1 回
        let indices = vec![0_i32, 0, 1];
        let mut out = vec![0.0_f32; 2];
        sparse_ft_forward_cpu(&weights, &indices, &[3], &mut out, 1, 2, 2, 3);
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
        sparse_ft_forward_cpu(&weights, &indices, &[], &mut out, 0, 0, 0, 0);
        assert!(out.is_empty());

        // nnz=0 のみ: padding ループが 0 周で sum = 0 になる
        let weights = vec![1.0_f32, 2.0];
        let indices: Vec<i32> = vec![];
        let mut out = vec![999.0_f32, 999.0];
        sparse_ft_forward_cpu(&weights, &indices, &[0], &mut out, 1, 2, 1, 0);
        assert_eq!(out, vec![0.0_f32, 0.0]);
    }
}
