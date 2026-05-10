//! Forward pass の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn forward`) は **`src/main.rs` に inline 定義** されている
//! (cuda-oxide の rustc-codegen-cuda backend は bin entry 経由で到達可能な
//! kernel しか PTX 化しないため)。本 module の `forward_cpu` は GPU と
//! 同じロジックを host で素直に書き写したもので、Stage 1-9 (#13) で
//! host loop が組まれたときの numerical equivalence test の reference 用。
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
//! ## bullet 上流 (`KERNELS_SRC::k_forward`) との対応
//!
//! - C++ `extern "C" __global__ void k_forward(...)` → Rust `#[kernel] fn forward(...)`
//! - C++ `const int*` / `const float*` → Rust `&[i32]` / `&[f32]`
//! - C++ output `float* preds` → Rust `mut DisjointSlice<f32>`
//! - C++ `int n_pos` / `int max_inds` → `u32` (符号要らないので素直に)
//! - C++ `for (int j = 0; j < max_inds; ++j)` → Rust `while j < max_inds` (gemm 上流に倣う)
//! - C++ `expf(-z)` → Rust `(-z).exp()` (cuda-oxide では libdevice 経由で `__nv_expf` に lowering)
//!
//! 計算ロジックは byte 単位で同一。

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
