//! Forward pass の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn forward`) は `src/main.rs` に inline 定義されている
//! (cuda-oxide の rustc-codegen-cuda backend は bin entry 経由で到達可能な
//! kernel しか PTX 化しないため)。本 module の `forward_cpu` は GPU と同じ
//! ロジックを host に書き写したもので、host loop の numerical equivalence test
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
//! - `(-z).exp()` は cuda-oxide が libdevice 経由で `__nv_expf` に lowering する。

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
