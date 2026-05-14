//! Fused Ranger optimizer (RAdam + Lookahead) の reference CPU 実装。
//!
//! Ranger (Wright 2019、bullet `ranger.rs` と同等) は **RAdam + Lookahead** の
//! 2 段構成。RAdam で fast params (`weights`) を更新しつつ、`step % k == 0` の
//! ときだけ Lookahead lerp (Zhang et al. 2019) で **slow params (`s`) との SMA**
//! を取る:
//!
//! ```text
//! per RAdam step:
//!     radam_step_cpu(weights, m, v, grad, ..., step_size, denom, ...)
//! every k steps (step % k == 0):
//!     ranger_lookahead_lerp_cpu(weights, slow, alpha)
//! ```
//!
//! GPU 側は **2 kernel** 構成: `radam_step` を毎 step 呼び、`step % k == 0` の
//! ときだけ `#[kernel] fn ranger_lookahead_lerp` を追加で呼ぶ host orchestration。
//! kernel 自体は scalar 1 thread = 1 weight の単純 pointwise (atomics 不要)。
//!
//! ## アルゴリズム
//!
//! ```text
//! Lookahead lerp (per element i):
//!     new_w = alpha * weights[i] + (1 - alpha) * slow[i]
//!     weights[i] = new_w
//!     slow[i]    = new_w        # weights / slow が同期する
//! ```
//!
//! - `alpha ∈ [0, 1]`: lookahead blend (典型値 0.5)
//! - `k`: lookahead step (典型値 6)
//! - `slow` の初期値は `0.0`
//! - lerp 後は **weights == slow** で完全同期
//!
//! ## 設計メモ
//!
//! - **slow params の checkpoint**: bullet 上流は `slow.bin` に書き出して resume
//!   時に復元する。本 reference は orchestration までは含まず、trainer 側で扱う
//!
//! ## cuda-oxide 制限
//!
//! - lerp kernel は `+` / `*` のみで `f32::clamp` も `sqrt` も使わないため
//!   cuda-oxide 制限に当たらない

/// Lookahead lerp の reference CPU 実装。
///
/// In-place mutation:
/// - `weights[i] = alpha * weights[i] + (1 - alpha) * slow[i]`
/// - `slow[i] = weights[i]` (= 上の new_w、両者を同期させる)
///
/// 入力前提:
/// - `weights.len() == slow.len() == n`
/// - `alpha ∈ [0, 1]` (host 側不変条件、典型値 0.5)
pub fn ranger_lookahead_lerp_cpu(weights: &mut [f32], slow: &mut [f32], alpha: f32, n: usize) {
    let one_minus_alpha = 1.0_f32 - alpha;
    for i in 0..n {
        let new_w = alpha * weights[i] + one_minus_alpha * slow[i];
        weights[i] = new_w;
        slow[i] = new_w;
    }
}

/// Ranger orchestration の reference CPU 実装 (1 step)。
///
/// `step` 番目の RAdam update を実行し、`step % k == 0` (`k > 0` 前提) の
/// ときだけ Lookahead lerp を続けて実行する。trainer 側 RangerScheduler が
/// GPU 上で同 sequence を組むときの reference 動作。
///
/// 入力前提:
/// - `weights.len() == m.len() == v.len() == grad.len() == slow.len() == n`
/// - `step >= 1` (1-indexed、`radam_compute_step_size_denom` が要求する制約)
/// - `k >= 1` (lookahead step 間隔)
///
/// 引数 16 個は RangerParams を CPU 側で展開した形。
/// `clippy::too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
pub fn ranger_step_cpu(
    weights: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &mut [f32],
    slow: &mut [f32],
    lr: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_w: f32,
    max_w: f32,
    n_sma_threshold: f32,
    alpha: f32,
    step: u64,
    k: u64,
    n: usize,
) {
    use super::radam_step::{radam_compute_step_size_denom, radam_step_cpu};

    debug_assert!(k >= 1, "Ranger lookahead step k must be >= 1 (got 0)");
    let (step_size, denom) = radam_compute_step_size_denom(step, beta1, beta2, n_sma_threshold);
    radam_step_cpu(
        weights, m, v, grad, lr, step_size, denom, decay, beta1, beta2, eps, min_w, max_w, n,
    );

    // `k != 0` ガード: CPU の `step % 0` は panic、`u64::is_multiple_of(0)` は
    // step==0 のとき true / それ以外 false で panic しない (両者の動作差を埋め、
    // 呼び出し側の `RangerParams` が `k >= 1` を保証しない経路でも安全に no-op
    // にする)。
    if k != 0 && step.is_multiple_of(k) {
        ranger_lookahead_lerp_cpu(weights, slow, alpha, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: f32, b: f32, tol: f64) -> bool {
        ((a as f64) - (b as f64)).abs() <= tol
    }

    /// alpha = 1.0 で lerp は **weights を維持** + slow を weights と同期。
    /// new_w = 1.0 * w + 0.0 * s = w、weights は不変、slow = weights。
    #[test]
    fn lerp_alpha_one_keeps_weights_syncs_slow() {
        let mut weights = vec![1.0_f32, 2.0, 3.0];
        let mut slow = vec![10.0_f32, 20.0, 30.0];
        ranger_lookahead_lerp_cpu(&mut weights, &mut slow, 1.0, 3);
        assert_eq!(weights, vec![1.0_f32, 2.0, 3.0]);
        assert_eq!(slow, vec![1.0_f32, 2.0, 3.0]);
    }

    /// alpha = 0.0 で lerp は **slow に引き戻し**。
    /// new_w = 0 * w + 1 * s = s、weights = slow (両者とも slow の値)。
    #[test]
    fn lerp_alpha_zero_pulls_weights_to_slow() {
        let mut weights = vec![1.0_f32, 2.0, 3.0];
        let mut slow = vec![10.0_f32, 20.0, 30.0];
        ranger_lookahead_lerp_cpu(&mut weights, &mut slow, 0.0, 3);
        assert_eq!(weights, vec![10.0_f32, 20.0, 30.0]);
        assert_eq!(slow, vec![10.0_f32, 20.0, 30.0]);
    }

    /// alpha = 0.5 (典型値) で lerp は **平均**。
    /// new_w = 0.5 * w + 0.5 * s = (w + s) / 2
    /// w=2, s=4 → new_w = 3、weights/slow 両方 3 に。
    #[test]
    fn lerp_alpha_half_takes_midpoint() {
        let mut weights = vec![2.0_f32, -4.0, 0.0];
        let mut slow = vec![4.0_f32, 4.0, 10.0];
        ranger_lookahead_lerp_cpu(&mut weights, &mut slow, 0.5, 3);
        assert_eq!(weights, vec![3.0_f32, 0.0, 5.0]);
        assert_eq!(slow, vec![3.0_f32, 0.0, 5.0]);
    }

    /// lerp 後は weights == slow で同期する性質テスト。任意の `(weights, slow,
    /// alpha)` で post-condition `weights == slow` が成り立つ。
    #[test]
    fn lerp_synchronizes_weights_and_slow() {
        let mut weights = vec![0.5_f32, -1.5, 2.7, 100.0];
        let mut slow = vec![-0.5_f32, 1.5, -2.7, -100.0];
        ranger_lookahead_lerp_cpu(&mut weights, &mut slow, 0.3, 4);
        assert_eq!(weights, slow, "post-lerp weights/slow must be equal");
    }

    /// 空配列 (n = 0) でも panic せず no-op。
    #[test]
    fn lerp_empty_input_yields_no_changes() {
        let mut weights: Vec<f32> = vec![];
        let mut slow: Vec<f32> = vec![];
        ranger_lookahead_lerp_cpu(&mut weights, &mut slow, 0.5, 0);
        assert!(weights.is_empty());
        assert!(slow.is_empty());
    }

    /// `ranger_step_cpu`: `step % k == 0` のときだけ lerp が走り、それ以外は
    /// RAdam 単独。k=2 で step=1 (RAdam only) → step=2 (RAdam + lerp) の挙動を
    /// 確認。lerp は step=2 のとき weights を slow=0 と blend するため、weights
    /// は (alpha * w_radam + (1-alpha) * 0) = alpha * w_radam に縮む。
    #[test]
    fn step_cpu_invokes_lerp_only_at_k_step() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![0.1_f32];
        let mut slow = vec![0.0_f32];

        // step=1 (k=2): RAdam のみ実行、lerp 走らない。weights は RAdam 後の値、
        // slow は 0 のまま (RangerLookahead 初期値 0.0 を踏襲)
        ranger_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            &mut slow,
            0.1,   // lr
            0.0,   // decay
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // eps
            f32::MIN,
            f32::MAX,
            5.0, // n_sma_threshold
            0.5, // alpha
            1,   // step
            2,   // k (step=1 % 2 != 0 なので lerp 走らない)
            1,
        );
        let w_after_step1 = weights[0];
        assert_eq!(slow[0], 0.0_f32, "slow should be unchanged at step=1");

        // step=2: 再 RAdam (grad は kernel 内で 0 に reset 済みなので 0.0) + lerp
        // が両方走る。grad = 0 なので RAdam 部分は decay = 0 で no-op (m/v 更新だけ
        // で weights 自体は変わらない)、その後 lerp で weights = 0.5 * w_after_step1
        // + 0.5 * 0 = w_after_step1 / 2
        let mut grad = vec![0.0_f32];
        ranger_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            &mut slow,
            0.1,
            0.0,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            5.0,
            0.5,
            2,
            2,
            1,
        );
        // lerp 後は weights == slow で同期、両者は w_after_step1 / 2 (grad=0 なので
        // RAdam の weights update がほぼ無く、おおむね w_after_step1 のまま lerp に入る)
        // Note: RAdam 内で grad=0 のとき m_new = beta1*m_old + 0、val = m_new/sqrt(v_new+eps)
        // で weights -= rate*val なので weights は m_old != 0 だと若干変わる
        // ここでは厳密値より 「lerp が走った結果 weights == slow」 の同期だけを assert
        assert!(approx_eq_f32(weights[0], slow[0], 1e-7));
        // weights は w_after_step1 から十分縮んでいる (lerp で 0.5x + small RAdam delta)
        assert!(
            weights[0].abs() < w_after_step1.abs(),
            "lerp should pull weights closer to slow=0: w_after_step1={w_after_step1} cur={}",
            weights[0]
        );
    }

    /// `k = 1` (lerp が毎 step 走る境界値): step % 1 == 0 が常に true なので
    /// 毎 step lerp、slow が weights に毎 step 追従する。grad = 0 で RAdam が
    /// no-op になる構成にして、lerp の効果のみ観察。
    /// w=1.0、slow=0.5、alpha=0.5 → new_w = 0.5*1.0 + 0.5*0.5 = 0.75
    #[test]
    fn step_cpu_k_equals_one_lerps_every_step() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![0.0_f32];
        let mut slow = vec![0.5_f32];
        ranger_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            &mut slow,
            0.1,
            0.0,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            5.0,
            0.5,
            1, // step
            1, // k = 1 (毎 step lerp)
            1,
        );
        // RAdam 部分は g=0 / m=v=0 / decay=0 で weights 不変 (1.0 のまま)、
        // 続く lerp で new_w = 0.5*1.0 + 0.5*0.5 = 0.75
        assert!(approx_eq_f32(weights[0], 0.75_f32, 1e-6));
        assert!(approx_eq_f32(slow[0], 0.75_f32, 1e-6));
        // post-condition: weights == slow
        assert_eq!(weights[0], slow[0]);
    }
}
