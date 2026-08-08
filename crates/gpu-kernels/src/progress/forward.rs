//! Forward pass の reference CPU 実装。
//!
//! GPU 側は CUDA C++ の `progress_forward`。本 module の `forward_cpu` は
//! GPU と同じロジックを host に書き写したもので、numerical equivalence test
//! の reference に使う。
//!
//! ## アルゴリズム
//!
//! 1 thread = 1 position に対し、`max_inds` (typically 80) 個の flat index
//! 配列の `>= 0` 要素に対応する weight を累積し、`sigmoid(z)` を取る:
//!
//! ```text
//! preds[pos] = sigmoid( Σ_{j: idx[base+j] >= 0} weights[idx[base+j]] )
//! ```
//!
//! `base = pos * max_inds`、padding 値 `-1` は skip。
//!
//! ## 実装メモ
//!
//! - GPU 側の `expf` と比較するため、reference も f32 の指数関数を使う。

/// Reference CPU 実装。
///
/// 戻り値: `Vec<f32>` of length `n_pos`。
pub fn forward_cpu(indices: &[i32], weights: &[f32], n_pos: usize, max_inds: usize) -> Vec<f32> {
    let mut preds = vec![0.0f32; n_pos];
    for (pos, p) in preds.iter_mut().enumerate() {
        let mut z = 0.0f32;
        let base = pos * max_inds;
        for j in 0..max_inds {
            let idx = indices[base + j];
            if idx >= 0 {
                z += weights[idx as usize];
            }
        }
        *p = 1.0f32 / (1.0f32 + (-z).exp());
    }
    preds
}
