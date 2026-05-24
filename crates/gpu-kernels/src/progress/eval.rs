//! Evaluation (loss + histogram only) kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn eval`) は `src/main.rs` に inline 定義されている
//! (cuda-oxide rustc-codegen-cuda backend は bin entry 経由でしか PTX 化しない
//! ため)。本 module の `eval_cpu` は GPU と同じロジックを host に書き写した
//! もの。
//!
//! ## アルゴリズム
//!
//! 1 thread = 1 position。`grad` kernel から **gradient scatter を除いた**
//! 評価専用 path。weight 更新には使わず、validation/test 時の loss と prediction
//! 分布を取るために使う:
//!
//! ```text
//! err               = preds[pos] - targets[pos]
//! loss_acc         += err * err           (f64 で累積)
//! hist[clamp(int(p*8), 0, 7)] += 1        (u64)
//! ```
//!
//! `eval` は `grad` の真のサブセット (gradient scatter / per_pos_norm を除いた path)
//! なので、同一 `(preds, targets, n_pos)` に対する `loss_acc` / `hist` 結果は
//! `grad` を **scatter 無効化 (`indices` 全 `-1` padding)** で呼んだ結果と
//! 一致するはず。`tests/eval_smoke.rs::eval_output_matches_grad_loss_hist_subset`
//! で確認している。

/// Reference CPU 実装。
///
/// In-place mutation:
/// - `loss_acc`: 単一 f64 cell。`err^2` を batch 全体で累積
/// - `hist`: 長さ 8 の u64。`p` を 8 等分した bin にカウント
///
/// 入力前提:
/// - `preds.len() == targets.len() == n_pos`
/// - `hist.len() == 8`
pub fn eval_cpu(
    preds: &[f32],
    targets: &[f32],
    loss_acc: &mut f64,
    hist: &mut [u64; 8],
    n_pos: usize,
) {
    for pos in 0..n_pos {
        let p = preds[pos];
        let y = targets[pos];
        let err = p - y;
        *loss_acc += (err as f64) * (err as f64);

        let b = ((p * 8.0f32) as i32).clamp(0, 7);
        hist[b as usize] += 1;
    }
}
