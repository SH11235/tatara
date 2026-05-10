//! Backward (loss + gradient + histogram) kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn grad`) は **`src/main.rs` に inline 定義** されている
//! (cuda-oxide の rustc-codegen-cuda backend は bin entry 経由で到達可能な
//! kernel しか PTX 化しないため、Stage 1-5 で確立した配置)。本 module の
//! `grad_cpu` は GPU と同じロジックを host で素直に書き写したもので、
//! Stage 1-9 (#13) で host loop が組まれたときの numerical equivalence test
//! の reference 用。
//!
//! ## アルゴリズム
//!
//! 1 thread = 1 position に対し、forward の `preds[pos]` と target からスカラー
//! gradient `gscale` を計算し、`max_inds` 個の active index へ atomicAdd で
//! scatter する。同時に loss (`err^2`) と prediction histogram bucket を更新する:
//!
//! ```text
//! err     = preds[pos] - targets[pos]
//! norm    = per_pos_norm[pos]
//! gscale  = 2 * err * preds[pos] * (1 - preds[pos]) * norm     // d(err^2)/dz の chain rule (sigmoid 微分込み)
//! grad[idx[base+j]] += gscale     for j s.t. idx[base+j] >= 0  // atomic
//! loss_acc          += err * err  (f64 で累積、precision loss 防止)
//! hist[clamp(int(p*8), 0, 7)] += 1                             // u64、epoch 全体で overflow しない
//! ```
//!
//! `base = pos * max_inds`、padding 値 `-1` は skip。
//!
//! ## bullet 上流 (`KERNELS_SRC::k_grad_loss_hist`) との対応
//!
//! - C++ `extern "C" __global__ void k_grad_loss_hist(...)` → Rust `#[kernel] fn grad(...)`
//! - C++ `const int*` / `const float*` / `double*` / `unsigned long long*` →
//!   Rust `&[i32]` / `&[f32]` / `&[f64]` / `&[u64]`
//! - C++ `atomicAdd(&grad[idx], gscale)` (f32) → Rust `DeviceAtomicF32::fetch_add(gscale, Relaxed)`
//! - C++ `atomicAdd(loss_acc, err*err)` (f64) → Rust `DeviceAtomicF64::fetch_add(..., Relaxed)`
//! - C++ `atomicAdd(&hist[b], 1ULL)` (u64) → Rust `DeviceAtomicU64::fetch_add(1, Relaxed)`
//! - C++ `(int)(p * 8.0f)` の truncate-toward-zero → Rust `(p * 8.0f32) as i32`
//!   (Rust は saturating cast だが clamp [0,7] が後段にあるため範囲内では同値)
//! - C++ `for (int j = 0; j < max_inds; ++j)` → Rust `while j < max_inds` (forward と同じ)
//!
//! 上記の表面的差異 (atomic API、float→int の saturating cast、clamp 表現) を
//! 除き計算ロジックは同一で、有限な finite 入力に対し同じ結果を返す。
//! NaN / 非有限 `p` では Rust 側 `as i32` が i32 範囲に saturate するのに対し
//! C++ 側 `(int)` は UB だが、後段の clamp [0,7] がいずれも 0 か 7 に丸めるため
//! sigmoid 出力 (`forward` 側の値域 (0,1)) を流す production path では差は出ない。
//!
//! ## 並列化と atomics
//!
//! GPU 版は 1 thread = 1 position で並列実行され、複数 position から同じ weight
//! index への加算が衝突するため `grad` への shoot は **device-scope atomicAdd**
//! が必須。`loss_acc` (single cell) と `hist[0..8]` も bin 衝突するため atomic。
//! Reference CPU 実装は単一スレッドで素直に shared mutable buffer を更新する。
//! `Relaxed` ordering は collection 用途では十分 (順序は問わず最終結果のみ重要)。

/// Reference CPU 実装。
///
/// In-place mutation:
/// - `grad`: 長さ `n_weights`。`indices` で参照される weight index に `gscale` 加算
/// - `loss_acc`: 単一 f64 cell。`err^2` を batch 全体で累積 (epoch loss の構成要素)
/// - `hist`: 長さ 8 の u64。`p` を 8 等分した bin にカウント
///
/// 入力前提:
/// - `indices.len() == n_pos * max_inds`
/// - `preds.len() == targets.len() == per_pos_norm.len() == n_pos`
/// - `indices` の非 padding 要素は `0..grad.len()` に収まる
/// - `hist.len() == 8`
///
/// 引数数 (9) は bullet 上流 `k_grad_loss_hist` と 1:1 対応するため
/// clippy `too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn grad_cpu(
    indices: &[i32],
    preds: &[f32],
    targets: &[f32],
    per_pos_norm: &[f32],
    grad: &mut [f32],
    loss_acc: &mut f64,
    hist: &mut [u64; 8],
    n_pos: usize,
    max_inds: usize,
) {
    for pos in 0..n_pos {
        let p = preds[pos];
        let y = targets[pos];
        let err = p - y;
        let norm = per_pos_norm[pos];
        let gscale = 2.0f32 * err * p * (1.0f32 - p) * norm;

        let base = pos * max_inds;
        for j in 0..max_inds {
            let idx = indices[base + j];
            if idx >= 0 {
                grad[idx as usize] += gscale;
            }
        }

        *loss_acc += (err as f64) * (err as f64);

        let b = ((p * 8.0f32) as i32).clamp(0, 7);
        hist[b as usize] += 1;
    }
}
