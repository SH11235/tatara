//! Win-rate-model (WRM) loss kernel の reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn loss_wrm` は呼び出し元 bin entry に inline 定義する
//! (cuda-oxide rustc-codegen-cuda backend の bin-entry 制約)。本 module は
//! GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム
//!
//! 教師 score と net 出力の双方を win-rate に変換し、その二乗誤差を loss とする。
//! target / prediction 双方で sigmoid 対称差を取る:
//!
//! ```text
//! per position i:
//!     # --- target (offset/scaling は caller 指定、既定 270 / 380) ---
//!     pt   = (score[i]  - target_offset) / target_scaling
//!     pmt  = (-score[i] - target_offset) / target_scaling
//!     target_wrm = 0.5 * (1 + sigmoid(pt) - sigmoid(pmt))
//!     target = lambda * wdl[i] + (1 - lambda) * target_wrm
//!     # --- prediction (scorenet = out * nnue2score) ---
//!     scorenet = out[i] * nnue2score
//!     q   = sigmoid((scorenet  - 270) / in_scaling)
//!     qm  = sigmoid((-scorenet - 270) / in_scaling)
//!     qf  = 0.5 * (1 + q - qm)
//!     err = qf - target
//!     loss_acc += err^2                          # un-normalized sum
//!     # chain rule: dq/dout = q(1-q) * nnue2score/in_scaling,
//!     #             dqm/dout = -qm(1-qm) * nnue2score/in_scaling
//!     #             dqf/dout = 0.5 * (nnue2score/in_scaling) * (q(1-q) + qm(1-qm))
//!     #             dL/dout  = 2*err * dqf/dout  → 2 と 0.5 が打ち消し合う
//!     dl_dout[i] = err * (nnue2score / in_scaling) * (q(1-q) + qm(1-qm)) * per_pos_norm[i]
//! ```
//!
//! ## `loss_wdl` (sigmoid-MSE) との違い / なぜ WRM が要るか
//!
//! [`super::loss_wdl::loss_wdl_cpu`] は `p = sigmoid(out * scale)` で net_output に
//! `scale = 1/scale_param` を掛けるため net_output は **cp 単位** (`out ≈ cp`) で
//! 収束する。一方 WRM loss は `scorenet = out * nnue2score` (= `out * 600`) を
//! cp 単位とみなすため、net_output は **`out ≈ cp / nnue2score` (O(1))** で収束する。
//! `crates/nnue-format` の量子化 (`QA=127 / QB=64 / FV_SCALE=28`) は `out ≈ cp/600`
//! スケールを前提とするので、その量子化フォーマット向けの net を学習するには
//! WRM loss を使う (sigmoid-MSE で学習した net は byte レイアウトは互換だが
//! 数値スケールが ~600× ずれて量子化後に破綻する)。
//!
//! ## 定数
//!
//! - target 側 `target_offset` / `target_scaling` は caller 指定 (CLI
//!   `--wrm-target-offset` / `--wrm-target-scaling`、既定 270 / 380)。既定 270/380 は
//!   chess の評価値分布向けの値で、score 分布が異なるドメインでは再調整する
//! - prediction 側 offset 270 はハードコード (caller から可変なのは target 側のみ)
//! - prediction 側 `in_scaling` は CLI `--wrm-in-scaling` (既定 340)、target 側と
//!   独立に指定する
//! - `lambda` (WDL blend) は典型的には 0.0 (target = target_wrm のみ) だが、
//!   WdlScheduler 互換のため引数として残す (`lambda = 1.0` で純 WDL)
//!
//! ## 実装メモ
//!
//! WRM target + WDL blend を kernel 内に畳み込み、`score` (raw cp) と `wdl`
//! ({0, 0.5, 1}) を 2 buffer で渡す (`loss_wdl` と同じ trade-off)。
//!
//! ## NaN / Inf 挙動
//!
//! - `out[i] = NaN` / `score[i] = NaN` → sigmoid 経由で NaN 伝搬 (`loss_wdl` と同じ、
//!   学習中の NaN を loss 経路で気付ける)
//! - `|score|` が非常に大きい場合 (例: ±32000 の mate-stamp) `(score -
//!   target_offset)/target_scaling` が既定 380 で ±84 程度になり sigmoid が 0/1 に
//!   飽和する。`exp(±84)` は f32 範囲内 (`exp(88.7) ≈ 3.4e38`) なので overflow せず、
//!   target_wrm は 0 か 1 に張り付くだけで NaN にならない。`q*(1-q)` も飽和時は 0 に
//!   なり grad が消える

/// Reference CPU 実装。
///
/// In-place 出力:
/// - `dl_dout[i]`: per-position grad (`per_pos_norm` 込み)
/// - `loss_acc`: per-position `err^2` の host 単一-thread 累積 (atomic 不要)
///
/// 入力前提:
/// - `out.len() == score.len() == wdl.len() == per_pos_norm.len() == dl_dout.len() == n`
/// - `nnue2score > 0` / `in_scaling > 0` / `target_scaling > 0` (CLI
///   `--wrm-nnue2score` / `--wrm-in-scaling` / `--wrm-target-scaling` は正値、
///   host 側で保証)、`target_offset` は任意の有限値
/// - `lambda ∈ [0, 1]` (1.0 で純 WDL ターゲット、0.0 で純 WRM ターゲット)
/// - `wdl[i] ∈ {0.0, 0.5, 1.0}` (loss / draw / win)
///
/// 引数 12 個は host invariant を漏れなく渡すため。`clippy::too_many_arguments`
/// を allow する。
#[allow(clippy::too_many_arguments)]
pub fn loss_wrm_cpu(
    out: &[f32],
    score: &[f32],
    wdl: &[f32],
    per_pos_norm: &[f32],
    dl_dout: &mut [f32],
    loss_acc: &mut f64,
    lambda: f32,
    nnue2score: f32,
    in_scaling: f32,
    target_offset: f32,
    target_scaling: f32,
    n: usize,
) {
    // prediction 側 offset はハードコード。target 側 offset / scaling のみ caller 指定
    // (`target_offset` / `target_scaling`)。
    const PRED_OFFSET: f32 = 270.0;
    for i in 0..n {
        // target: WRM applied to raw cp score
        let s = score[i];
        let pt = (s - target_offset) / target_scaling;
        let pmt = (-s - target_offset) / target_scaling;
        let target_wrm = 0.5_f32 * (1.0_f32 + sigmoid_f32(pt) - sigmoid_f32(pmt));
        let target = lambda * wdl[i] + (1.0_f32 - lambda) * target_wrm;

        // prediction: WRM applied to net output (scorenet = out * nnue2score)
        let scorenet = out[i] * nnue2score;
        let q = sigmoid_f32((scorenet - PRED_OFFSET) / in_scaling);
        let qm = sigmoid_f32((-scorenet - PRED_OFFSET) / in_scaling);
        let qf = 0.5_f32 * (1.0_f32 + q - qm);

        let err = qf - target;
        let norm = per_pos_norm[i];
        *loss_acc += (err as f64) * (err as f64);
        dl_dout[i] =
            err * (nnue2score / in_scaling) * (q * (1.0_f32 - q) + qm * (1.0_f32 - qm)) * norm;
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// `score = 0` のとき target_wrm = `0.5 * (1 + sigmoid(-270/380) - sigmoid(-270/380))`
    /// = `0.5` (pt == pmt)。同様に `out = 0` のとき qf = `0.5 * (1 + sigmoid(-270/340)
    /// - sigmoid(-270/340))` = `0.5`。よって err = 0、loss = 0、grad = 0。
    #[test]
    fn zero_input_yields_half_target_and_prediction_zero_loss() {
        let out = vec![0.0_f32];
        let score = vec![0.0_f32];
        let wdl = vec![0.5_f32];
        let per_pos_norm = vec![1.0_f32];
        let mut dl_dout = vec![123.0_f32];
        let mut loss_acc = 0.0_f64;
        loss_wrm_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            0.0,
            600.0,
            340.0,
            270.0,
            380.0,
            1,
        );
        assert_eq!(loss_acc, 0.0, "err must be exactly zero at score=out=0");
        assert_eq!(dl_dout[0], 0.0_f32);
    }

    /// `lambda = 1` で WRM target が消え、target は純 WDL ({0, 0.5, 1})。
    /// `score = 999` は target に効かない。`out = 0` → qf = 0.5、`wdl = 0.5` (draw)
    /// → err = 0、loss = 0、grad = 0。
    #[test]
    fn lambda_one_uses_pure_wdl_target() {
        let out = vec![0.0_f32];
        let score = vec![999.0_f32];
        let wdl = vec![0.5_f32];
        let per_pos_norm = vec![1.0_f32];
        let mut dl_dout = vec![0.0_f32];
        let mut loss_acc = 0.0_f64;
        loss_wrm_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            1.0,
            600.0,
            340.0,
            270.0,
            380.0,
            1,
        );
        assert_eq!(dl_dout[0], 0.0_f32);
        assert_eq!(loss_acc, 0.0);
    }

    /// loss / grad が docstring の式と一致することを、同じ式を独立に書き直して
    /// 照合する (期待値は f32 計算 → f64 cast、f32 リテラル比較の pitfall 回避)。
    #[test]
    fn matches_wrm_formula() {
        let out = vec![0.3_f32, -0.8, 2.5, -0.05];
        let score = vec![150.0_f32, -1200.0, 30.0, 5000.0];
        let wdl = vec![1.0_f32, 0.0, 0.5, 1.0];
        let per_pos_norm = vec![1.0_f32; 4];
        let lambda = 0.0_f32;
        let nnue2score = 600.0_f32;
        let in_scaling = 340.0_f32;

        let mut dl_dout = vec![0.0_f32; 4];
        let mut loss_acc = 0.0_f64;
        loss_wrm_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            lambda,
            nnue2score,
            in_scaling,
            270.0,
            380.0,
            4,
        );

        // 式を独立に再計算 (WRM target + WRM prediction)
        let sig = |x: f32| 1.0_f32 / (1.0_f32 + (-x).exp());
        let mut exp_loss = 0.0_f64;
        for i in 0..4 {
            let pt = (score[i] - 270.0) / 380.0;
            let pmt = (-score[i] - 270.0) / 380.0;
            let target_wrm = 0.5 * (1.0 + sig(pt) - sig(pmt));
            let target = lambda * wdl[i] + (1.0 - lambda) * target_wrm;
            let scorenet = out[i] * nnue2score;
            let q = sig((scorenet - 270.0) / in_scaling);
            let qm = sig((-scorenet - 270.0) / in_scaling);
            let qf = 0.5 * (1.0 + q - qm);
            let err = qf - target;
            exp_loss += (err as f64) * (err as f64);
            let exp_grad = err * (nnue2score / in_scaling) * (q * (1.0 - q) + qm * (1.0 - qm));
            let diff = ((dl_dout[i] as f64) - (exp_grad as f64)).abs();
            assert!(
                diff < 1e-7,
                "i={i}: got {} exp {exp_grad} diff {diff}",
                dl_dout[i]
            );
        }
        assert!(
            approx_eq_f64(loss_acc, exp_loss, 1e-10),
            "loss: got {loss_acc} exp {exp_loss}"
        );
    }

    /// 解析勾配が数値微分 (中心差分) と一致することを確認する。`per_pos_norm = 1`
    /// なので `dl_dout[i] = dL_i/dout[i]` (L_i = err_i^2)。
    #[test]
    fn analytic_grad_matches_finite_difference() {
        let outs = [0.2_f32, -1.3, 0.75, 3.1, -0.4];
        let score_v = [400.0_f32, -50.0, 1800.0, -3000.0, 12.0];
        let nnue2score = 600.0_f32;
        let in_scaling = 340.0_f32;
        let lambda = 0.0_f32;

        let loss_only = |o: f32, s: f32| -> f64 {
            let mut dl = [0.0_f32];
            let mut acc = 0.0_f64;
            loss_wrm_cpu(
                &[o],
                &[s],
                &[1.0],
                &[1.0],
                &mut dl,
                &mut acc,
                lambda,
                nnue2score,
                in_scaling,
                270.0,
                380.0,
                1,
            );
            acc
        };

        for (&o, &s) in outs.iter().zip(score_v.iter()) {
            let mut dl = [0.0_f32];
            let mut acc = 0.0_f64;
            loss_wrm_cpu(
                &[o],
                &[s],
                &[1.0],
                &[1.0],
                &mut dl,
                &mut acc,
                lambda,
                nnue2score,
                in_scaling,
                270.0,
                380.0,
                1,
            );
            // 中心差分は f64 で評価して打ち切り誤差を抑える
            let h = 1.0e-3_f64;
            let lp = loss_only((o as f64 + h) as f32, s);
            let lm = loss_only((o as f64 - h) as f32, s);
            let num_grad = (lp - lm) / (2.0 * h);
            let diff = ((dl[0] as f64) - num_grad).abs();
            let scale = num_grad.abs().max(1e-6);
            // f32 で評価した loss の中心差分なので tol は緩め (符号 / 係数 (×2 / ÷0.5)
            // のミスを捕まえるのが目的)。
            assert!(
                diff / scale < 1e-2,
                "out={o} score={s}: analytic {} numeric {num_grad} rel-diff {}",
                dl[0],
                diff / scale
            );
        }
    }

    /// `per_pos_norm` は grad にだけ乗り loss_acc には乗らない convention (`loss_wdl` と同型)。
    #[test]
    fn per_pos_norm_scales_grad_but_not_loss() {
        let out = vec![1.5_f32; 3];
        let score = vec![800.0_f32; 3];
        let wdl = vec![1.0_f32; 3];

        let mut dl_a = vec![0.0_f32; 3];
        let mut acc_a = 0.0_f64;
        loss_wrm_cpu(
            &out,
            &score,
            &wdl,
            &[1.0_f32; 3],
            &mut dl_a,
            &mut acc_a,
            0.0,
            600.0,
            340.0,
            270.0,
            380.0,
            3,
        );
        let mut dl_b = vec![0.0_f32; 3];
        let mut acc_b = 0.0_f64;
        loss_wrm_cpu(
            &out,
            &score,
            &wdl,
            &[0.25_f32; 3],
            &mut dl_b,
            &mut acc_b,
            0.0,
            600.0,
            340.0,
            270.0,
            380.0,
            3,
        );

        assert!(approx_eq_f64(acc_a, acc_b, 1e-12));
        for i in 0..3 {
            let quarter = dl_a[i] * 0.25;
            let diff = ((dl_b[i] as f64) - (quarter as f64)).abs();
            assert!(
                diff < 1e-7,
                "i={i}: quarter={quarter} got {} diff {diff}",
                dl_b[i]
            );
        }
    }

    /// 大きな score (mate-stamp 帯) でも NaN/Inf にならず target が 0/1 に飽和する。
    #[test]
    fn large_score_saturates_without_nan() {
        let out = vec![10.0_f32, -10.0];
        let score = vec![32000.0_f32, -32000.0];
        let wdl = vec![1.0_f32, 0.0];
        let per_pos_norm = vec![1.0_f32; 2];
        let mut dl_dout = vec![0.0_f32; 2];
        let mut loss_acc = 0.0_f64;
        loss_wrm_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            0.0,
            600.0,
            340.0,
            270.0,
            380.0,
            2,
        );
        assert!(
            loss_acc.is_finite(),
            "loss_acc must be finite, got {loss_acc}"
        );
        assert!(
            dl_dout.iter().all(|g| g.is_finite()),
            "grads must be finite: {dl_dout:?}"
        );
    }

    /// `target_offset` / `target_scaling` が target_wrm に実際に効くことを、既定値
    /// (270 / 380) と非既定値 (200 / 500) で loss を計算して照合する。`out = 0` で
    /// prediction qf = 0.5 固定なので、loss = `(0.5 - target)^2` が target_wrm の
    /// 違いをそのまま反映する。期待 target は同じ式を独立に書き直して照合。
    #[test]
    fn custom_target_params_change_target_wrm() {
        let out = vec![0.0_f32]; // qf = 0.5 固定 → loss は target のみに依存
        let score = vec![600.0_f32]; // 非ゼロ score: target_wrm が offset/scaling に依存
        let wdl = vec![0.5_f32];
        let sig = |x: f32| 1.0_f32 / (1.0_f32 + (-x).exp());
        let target_wrm = |off: f32, sc: f32| {
            0.5_f32 * (1.0_f32 + sig((600.0 - off) / sc) - sig((-600.0 - off) / sc))
        };

        let run = |off: f32, sc: f32| -> f64 {
            let mut dl = vec![0.0_f32];
            let mut acc = 0.0_f64;
            loss_wrm_cpu(
                &out,
                &score,
                &wdl,
                &[1.0_f32],
                &mut dl,
                &mut acc,
                0.0,
                600.0,
                340.0,
                off,
                sc,
                1,
            );
            acc
        };

        // 既定 270/380 と非既定 200/500 で target_wrm は異なる → loss も異なる。
        let loss_default = run(270.0, 380.0);
        let loss_custom = run(200.0, 500.0);
        assert!(
            (loss_default - loss_custom).abs() > 1e-6,
            "target params must change the loss: default={loss_default} custom={loss_custom}"
        );

        // それぞれ独立式 `(0.5 - target_wrm)^2` と一致することを確認。
        for (off, sc) in [(270.0_f32, 380.0_f32), (200.0, 500.0)] {
            let err = 0.5_f32 - target_wrm(off, sc);
            let expected = (err as f64) * (err as f64);
            let got = run(off, sc);
            assert!(
                approx_eq_f64(got, expected, 1e-10),
                "off={off} sc={sc}: got {got} expected {expected}"
            );
        }
    }
}
