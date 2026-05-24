//! Backward (loss + gradient + histogram) kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn grad`) は `src/main.rs` に inline 定義されている
//! (cuda-oxide の rustc-codegen-cuda backend は bin entry 経由で到達可能な
//! kernel しか PTX 化しないため)。本 module の `grad_cpu` は GPU と同じ
//! ロジックを host に書き写したもので、host loop の numerical equivalence test
//! の reference に使う。
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
//! ## 実装メモ
//!
//! - `(p * 8.0f32) as i32` は Rust の saturating cast。NaN は 0 になる。後段の
//!   clamp [0,7] が掛かるため、sigmoid 出力の値域 (0,1) であれば必ず 0..=7 に
//!   収まる。
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
/// 引数数 (9) は kernel 側と 1:1 対応するため clippy `too_many_arguments` を
/// allow する。
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
