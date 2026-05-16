//! 2D row-slice extract / scatter kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn slice_extract_2d` / `slice_scatter_2d`) は
//! `bins/nnue_train/src/main.rs` に inline 定義 (cuda-oxide bin-entry 制約)。
//! bullet 上流の `slice_rows(start, end)` (`crates/trainer/src/model/builder.rs`)
//! に等価で、`l1_total (B×16)` から `l1_main (B×15)` (offset 0) と
//! `l1_skip (B×1)` (offset 15) を切り出す forward と、その backward で
//! `dl1_main` / `dl1_skip` を `dl1_total (B×16)` に書き戻すのに使う。
//!
//! ## アルゴリズム
//!
//! ```text
//! extract (per batch bi, out col oi in 0..out_dim):
//!     dst[bi * out_dim + oi] = src[bi * src_stride + src_offset + oi]
//!
//! scatter (per batch bi, in col ii in 0..in_dim):
//!     dst[bi * dst_stride + dst_offset + ii] = src[bi * in_dim + ii]
//! ```
//!
//! - `src`/`dst` の "stride" は **その方向の 1 行の幅** (`l1_total` は 16)。
//!   slice する側 (`dst` of extract / `src` of scatter) は packed (= `out_dim` /
//!   `in_dim` 幅) を仮定する。
//! - scatter は `dst` を呼出側が事前に 0 (or 適切値) で初期化する責務 (kernel と
//!   同じ accumulate ではなく overwrite だが、書かれない範囲は呼出側責任)。
//!   reference は `dst` を渡された状態のまま、書く範囲だけ上書きする。
//! - 値はそのままコピー (NaN/Inf 透過)。

/// 2D row slice extract reference — `dst[bi][oi] = src[bi*src_stride + src_offset + oi]`。
///
/// `dst.len() == batch * out_dim`、`src.len() >= batch * src_stride` かつ
/// `src_offset + out_dim <= src_stride` 前提。
pub fn slice_extract_2d_cpu(
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    src_stride: usize,
    src_offset: usize,
    out_dim: usize,
) {
    for bi in 0..batch {
        for oi in 0..out_dim {
            dst[bi * out_dim + oi] = src[bi * src_stride + src_offset + oi];
        }
    }
}

/// 2D row slice scatter reference — `dst[bi*dst_stride + dst_offset + ii] = src[bi][ii]`。
///
/// `src.len() == batch * in_dim`、`dst.len() >= batch * dst_stride` かつ
/// `dst_offset + in_dim <= dst_stride` 前提。`dst` の書かれない範囲は呼出側責任
/// (kernel の host 契約「dst を事前 0 初期化」と同じ)。
pub fn slice_scatter_2d_cpu(
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    in_dim: usize,
    dst_stride: usize,
    dst_offset: usize,
) {
    for bi in 0..batch {
        for ii in 0..in_dim {
            dst[bi * dst_stride + dst_offset + ii] = src[bi * in_dim + ii];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// forward step: l1_total (B×16) → l1_main (B×15) at offset 0,
    /// l1_skip (B×1) at offset 15。
    #[test]
    fn extract_l1_main_and_skip_layerstack() {
        // batch=2, src_stride=16
        let src: Vec<f32> = (0..2 * 16).map(|i| i as f32).collect();
        let mut l1_main = vec![0.0_f32; 2 * 15];
        slice_extract_2d_cpu(&src, &mut l1_main, 2, 16, 0, 15);
        // batch 0 rows 0..15, batch 1 rows 16..31
        let exp_main: Vec<f32> = (0..15).chain(16..31).map(|i: usize| i as f32).collect();
        assert_eq!(l1_main, exp_main);

        let mut l1_skip = vec![0.0_f32; 2];
        slice_extract_2d_cpu(&src, &mut l1_skip, 2, 16, 15, 1);
        assert_eq!(l1_skip, vec![15.0_f32, 31.0]);
    }

    /// scatter round-trips with extract: scatter dl1_main at 0 + dl1_skip at 15
    /// then extract back equals the originals.
    #[test]
    fn scatter_then_extract_round_trip_layerstack() {
        let batch = 3;
        let dst_stride = 16;
        let dl1_main: Vec<f32> = (0..batch * 15).map(|i| i as f32 + 0.25).collect();
        let dl1_skip: Vec<f32> = (0..batch).map(|i| -(i as f32) - 1.0).collect();
        let mut dl1_total = vec![0.0_f32; batch * dst_stride];
        slice_scatter_2d_cpu(&dl1_main, &mut dl1_total, batch, 15, dst_stride, 0);
        slice_scatter_2d_cpu(&dl1_skip, &mut dl1_total, batch, 1, dst_stride, 15);

        let mut back_main = vec![0.0_f32; batch * 15];
        slice_extract_2d_cpu(&dl1_total, &mut back_main, batch, dst_stride, 0, 15);
        assert_eq!(back_main, dl1_main);
        let mut back_skip = vec![0.0_f32; batch];
        slice_extract_2d_cpu(&dl1_total, &mut back_skip, batch, dst_stride, 15, 1);
        assert_eq!(back_skip, dl1_skip);
    }

    #[test]
    fn scatter_preserves_unwritten_dst() {
        let src = vec![1.0_f32, 2.0];
        // dst stride 4, batch 1, write in_dim=2 at offset 1 → dst[1..3] overwritten
        let mut dst = vec![9.0_f32, 9.0, 9.0, 9.0];
        slice_scatter_2d_cpu(&src, &mut dst, 1, 2, 4, 1);
        assert_eq!(dst, vec![9.0_f32, 1.0, 2.0, 9.0]);
    }

    #[test]
    fn nan_propagates() {
        let src = vec![f32::NAN, 1.0, 2.0, 3.0];
        let mut dst = vec![0.0_f32; 4];
        slice_extract_2d_cpu(&src, &mut dst, 2, 2, 0, 2);
        assert!(dst[0].is_nan());
    }

    #[test]
    fn empty_is_noop() {
        let mut dst: Vec<f32> = vec![];
        slice_extract_2d_cpu(&[], &mut dst, 0, 16, 0, 15);
        assert!(dst.is_empty());
        let mut dst2: Vec<f32> = vec![];
        slice_scatter_2d_cpu(&[], &mut dst2, 0, 15, 16, 0);
        assert!(dst2.is_empty());
    }
}
