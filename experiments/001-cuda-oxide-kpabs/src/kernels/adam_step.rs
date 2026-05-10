//! Adam optimizer step kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn adam_step`) は **`src/main.rs` に inline 定義** されている
//! (Stage 1-5 で確立した backend 仕様)。本 module の `adam_step_cpu` は GPU と
//! 同じ更新式を host で素直に書き写したもの。
//!
//! ## アルゴリズム
//!
//! 1 thread = 1 weight。Adam の m / v 更新 + bias correction + weight 更新 +
//! grad リセットを 1 step 内で完結:
//!
//! ```text
//! g          = grad[i]
//! m[i]       = beta1 * m[i] + (1 - beta1) * g
//! v[i]       = beta2 * v[i] + (1 - beta2) * g^2
//! m_hat      = m[i] / max(bc1, 1e-30)             // bc1 = 1 - beta1^t
//! v_hat      = v[i] / max(bc2, 1e-30)             // bc2 = 1 - beta2^t
//! weights[i] -= lr * m_hat / (sqrt(v_hat) + eps)
//! grad[i]    = 0
//! ```
//!
//! `bc1`/`bc2` は host 側で step 番号 `t` から事前計算して渡す (`1 - beta^t`)。
//! 1e-30 floor は学習初期 (small `t`) で `bc` が 0 に潰れるのを防ぐ。
//!
//! ## bullet 上流 (`KERNELS_SRC::k_adam_step`) との対応
//!
//! - C++ `extern "C" __global__ void k_adam_step(...)` → Rust `#[kernel] fn adam_step(...)`
//! - C++ output `float* weights / m / v / grad` (in-place) → Rust `mut DisjointSlice<f32>` × 4。
//!   1 thread = 1 index で aliasing なし、atomics 不要 (Stage 1-6 grad とは異なる)
//! - C++ `fmaxf(bc1, 1e-30f)` → 本 CPU reference では `bc1.max(1e-30f32)`、
//!   GPU kernel 側は `f32::max` が cuda-oxide で lowering 失敗するため
//!   `if bc1 > 1e-30 { bc1 } else { 1e-30 }` に展開している (詳細は
//!   `ATTRIBUTION.md` Stage 1-7 entry および `src/main.rs::adam_step` の comment)
//! - C++ `sqrtf(v_hat)` → Rust `v_hat.sqrt()` (cuda-oxide が `__nv_sqrtf` に lowering)
//! - C++ `int n` → Rust `u32` (符号要らない)
//!
//! 計算式は表面的差異 (Option-returning DisjointSlice / `len() < n` で silent skip)
//! を除き同一。`weights.len() == m.len() == v.len() == grad.len() == n_weights`
//! を host 側 invariant とすれば結果は同等。

/// Reference CPU 実装。
///
/// In-place mutation:
/// - `weights[i]`: 学習率 `lr` でスケールした正規化勾配で更新
/// - `m[i]` / `v[i]`: Adam 1次/2次 moment running average
/// - `grad[i]`: 0.0 にリセット (次 batch の accumulation 用)
///
/// 入力前提:
/// - `weights.len() == m.len() == v.len() == grad.len() == n`
///
/// 引数数 (10) は bullet 上流 `k_adam_step` と 1:1 対応のため
/// clippy `too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn adam_step_cpu(
    weights: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    n: usize,
) {
    for i in 0..n {
        let g = grad[i];
        let mi = beta1 * m[i] + (1.0f32 - beta1) * g;
        let vi = beta2 * v[i] + (1.0f32 - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        let m_hat = mi / bc1.max(1e-30f32);
        let v_hat = vi / bc2.max(1e-30f32);
        weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        grad[i] = 0.0f32;
    }
}
