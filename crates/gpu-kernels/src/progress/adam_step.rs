//! Adam optimizer step kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn adam_step`) は `src/main.rs` に inline 定義されている
//! (cuda-oxide rustc-codegen-cuda backend は bin entry 経由でしか PTX 化しない
//! ため)。本 module の `adam_step_cpu` は GPU と同じ更新式を host に書き写した
//! もの。
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
//! ## 実装メモ
//!
//! - 1 thread = 1 index で aliasing なし、atomics 不要 (`grad` kernel の scatter
//!   path とは異なる)。
//! - `bc.max(1e-30)` は本 CPU reference では `bc.max(1e-30f32)`。GPU kernel 側は
//!   `f32::max` が cuda-oxide で lowering 失敗するため `if bc > 1e-30 { bc }
//!   else { 1e-30 }` に展開している (`src/main.rs::adam_step` の comment 参照)。
//! - `v_hat.sqrt()` は cuda-oxide が `__nv_sqrtf` (libdevice) に lowering する。
//! - `weights.len() == m.len() == v.len() == grad.len() == n` を host 側
//!   invariant として要求する。

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
/// 引数数 (10) は kernel 側と 1:1 対応のため clippy `too_many_arguments` を
/// allow する。
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
