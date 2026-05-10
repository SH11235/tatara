//! Sigmoid + WDL blend + scale loss kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn loss_wdl`) は **`experiments/002-fused-kernels/
//! src/main.rs` に inline 定義** されている (Stage 1-5 で確立した「`#[kernel]`
//! は bin entry に inline」制約)。本 module の `loss_wdl_cpu` は GPU と
//! 同じロジックを host で素直に書き写したもので、Stage 2-2 (#38) GPU↔CPU 数値
//! 同等性テストの reference 用。
//!
//! ## アルゴリズム
//!
//! NNUE training の `MSE-on-sigmoid + WDL blend` を 1 fused kernel にまとめる。
//! bullet 上流 (`crates/bullet_lib/src/value/loader.rs::301-316` の data-layer
//! WDL blend と `crates/compiler/src/tensor/operation/autograd/dfo.rs::Sigmoid`)
//! の loss path を NNUE 専用に hand-fuse する。
//!
//! ```text
//! per position i:
//!     p   = sigmoid(out[i] * scale)            # NNUE eval を確率に
//!     ys  = sigmoid(score[i] * scale)          # 教師 cp score を確率に
//!     y   = lambda * wdl[i] + (1 - lambda) * ys     # WDL blend
//!     err = p - y
//!     loss_acc += err^2                        # un-normalized sum (Stage 1-6 grad と同型)
//!     dl_dout[i] = 2 * err * p * (1 - p) * scale * per_pos_norm[i]
//! ```
//!
//! - `out`, `score` は同じ cp scale を共有する前提 (`scale = 1/scale_param`、
//!   bullet 上流 `loader.rs::310` の `sigmoid(rscale * score)` と同流)
//! - `lambda = 1.0` で純 WDL ターゲット、`lambda = 0.0` で純 score sigmoid。
//!   bullet 上流の WdlScheduler が batch / superbatch 単位で動的に決める想定
//! - `wdl[i] ∈ {0.0, 0.5, 1.0}` (loss=0, draw=0.5, win=1)、bullet 上流
//!   `loader.rs::312` の `result() as u8 / 2.0` と同型
//! - `per_pos_norm[i]` は Stage 1 grad と同型 (`1 / (n_games * game_len)`)。
//!   loss_acc 側には乗らず、grad だけに乗る (Stage 1-6 grad と同 convention)
//!
//! ## bullet 上流との対応 / divergence
//!
//! - bullet は **data layer で blend を pre-compute** する (`loader.rs::315`
//!   `blend * result + (1 - blend) * score` の式が実体)。kernel 側は
//!   pre-blended target に対する `output.sigmoid().squared_error(target)` (`builder.rs::505`)
//!   を runtime PointwiseIR で fuse するだけ
//! - 本実装は **kernel 内に WDL blend を畳み込む** ことで data layer から
//!   sigmoid(score) と blend を消す。`score` と `wdl` を 2 buffer で渡す
//!   trade-off だが、batch 1 度しか転送しないため total memory traffic は
//!   bullet と同等以下
//! - chain rule で sigmoid(out * scale) の `out` 微分には `* scale` が乗る
//!   (`d/du sigmoid(u) = p (1-p)`、`u = out * scale`)。bullet 上流で `out`
//!   が既に scale 済の cp 単位を持っている場合 (`scale = 1.0` 相当) は本式の
//!   `* scale` 項が消えるだけで一致する
//!
//! ## NaN / Inf 挙動
//!
//! - `out[i] = NaN` → `p = sigmoid(NaN) = NaN`、`err = NaN`、`loss = NaN`、
//!   `dl_dout = NaN` (NaN を伝搬)。bullet 上流も sigmoid path で同等
//! - `score[i] = NaN` (= 教師の異常値) も同様に NaN 伝搬
//! - `wdl[i] ∈ {0, 0.5, 1}` の invariant が破れた場合 (例えば `wdl = 2.0`)
//!   は target が `[0, 1]` を超え、err も範囲外になるが kernel 側で潰さない
//!   (host invariant 違反の検出は loader 側で行う前提)
//! - SCReLU (Stage 2-1) と異なり本 kernel は NaN を握り潰さず伝搬するため、
//!   学習中の NaN は loss 経路で気付ける (これは upstream と一致)

/// Reference CPU 実装。
///
/// In-place 出力:
/// - `dl_dout[i]`: per-position grad (`per_pos_norm` 込み)
/// - `loss_acc`: per-position `err^2` の host 単一-thread 累積 (atomic 不要)
///
/// 入力前提:
/// - `out.len() == score.len() == wdl.len() == per_pos_norm.len() == dl_dout.len() == n`
/// - `scale > 0`。負の scale は `p` の sigmoid を反転させ chain rule の `* scale`
///   項で grad の符号も反転させてしまい、別 loss を計算することになる。NNUE の
///   `1/scale_param` 用法 (`scale_param > 0`) では常に成立、host (loader) 側で
///   保証する前提
/// - `lambda ∈ [0, 1]` (1.0 で純 WDL ターゲット、0.0 で純 score sigmoid)
/// - `wdl[i] ∈ {0.0, 0.5, 1.0}` (loss / draw / win、bullet 上流 `pos.result()
///   as u8 / 2.0` と同型)
///
/// 引数数 (9) は host invariant を漏れなく渡すため bullet 上流より多い。
/// clippy `too_many_arguments` を allow する (Stage 1 grad と同 convention)。
#[allow(clippy::too_many_arguments)]
pub fn loss_wdl_cpu(
    out: &[f32],
    score: &[f32],
    wdl: &[f32],
    per_pos_norm: &[f32],
    dl_dout: &mut [f32],
    loss_acc: &mut f64,
    lambda: f32,
    scale: f32,
    n: usize,
) {
    for i in 0..n {
        let p = sigmoid_f32(out[i] * scale);
        let ys = sigmoid_f32(score[i] * scale);
        let y = lambda * wdl[i] + (1.0_f32 - lambda) * ys;
        let err = p - y;
        let norm = per_pos_norm[i];
        *loss_acc += (err as f64) * (err as f64);
        dl_dout[i] = 2.0_f32 * err * p * (1.0_f32 - p) * scale * norm;
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

    /// `lambda = 0` で WDL 項が消えるとき、kernel は `sigmoid(out*scale)` と
    /// `sigmoid(score*scale)` の MSE になり、bullet 上流の plain
    /// `output.sigmoid().squared_error(sigmoid(score*scale))` と一致する。
    /// 期待値は f32 計算 → f64 cast (Stage 1-10 で確立した f32 リテラル比較
    /// pitfall 回避)。
    #[test]
    fn lambda_zero_is_pure_sigmoid_mse() {
        let out = vec![0.5_f32, -0.3, 1.0];
        let score = vec![0.0_f32, 0.0, 0.5];
        let wdl = vec![1.0_f32; 3];
        let per_pos_norm = vec![1.0_f32; 3];
        let mut dl_dout = vec![0.0_f32; 3];
        let mut loss_acc = 0.0_f64;
        let lambda = 0.0_f32;
        let scale = 1.0_f32;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            lambda,
            scale,
            3,
        );

        let expected_loss: f64 = (0..3)
            .map(|i| {
                let p = sigmoid_f32(out[i]);
                let ys = sigmoid_f32(score[i]);
                let err = (p - ys) as f64;
                err * err
            })
            .sum();
        assert!(
            approx_eq_f64(loss_acc, expected_loss, 1e-12),
            "loss: got {loss_acc} exp {expected_loss}"
        );

        for i in 0..3 {
            let p = sigmoid_f32(out[i]);
            let ys = sigmoid_f32(score[i]);
            let err = p - ys;
            let exp_grad = 2.0_f32 * err * p * (1.0_f32 - p);
            let diff = ((dl_dout[i] as f64) - (exp_grad as f64)).abs();
            assert!(
                diff < 1e-7,
                "i={i}: got {} exp {} diff {diff}",
                dl_dout[i],
                exp_grad
            );
        }
    }

    /// `lambda = 1` で score 項が消え、target は純 WDL ({0, 0.5, 1}) になる。
    /// `out = 0` のとき `p = 0.5`、`wdl = 0.5` (draw) なら err = 0、loss = 0、
    /// dl_dout = 0 になる端点を確認。
    #[test]
    fn lambda_one_with_draw_target_yields_zero_at_p_half() {
        let out = vec![0.0_f32];
        let score = vec![999.0_f32]; // 影響しない (lambda=1)
        let wdl = vec![0.5_f32];
        let per_pos_norm = vec![1.0_f32];
        let mut dl_dout = vec![0.0_f32];
        let mut loss_acc = 0.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            1.0,
            1.0,
            1,
        );
        assert_eq!(dl_dout[0], 0.0_f32);
        assert_eq!(loss_acc, 0.0);
    }

    /// `per_pos_norm` は **grad にだけ** 乗り、`loss_acc` には乗らないことを
    /// ガード (Stage 1 grad と同 convention)。同入力 norm = 1.0 と norm = 0.5
    /// で loss_acc が同じ、dl_dout が半分になることで検証。
    #[test]
    fn per_pos_norm_scales_grad_but_not_loss() {
        let out = vec![1.0_f32; 4];
        let score = vec![0.0_f32; 4];
        let wdl = vec![1.0_f32; 4];

        let per_pos_norm_a = vec![1.0_f32; 4];
        let per_pos_norm_b = vec![0.5_f32; 4];

        let mut dl_dout_a = vec![0.0_f32; 4];
        let mut loss_acc_a = 0.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm_a,
            &mut dl_dout_a,
            &mut loss_acc_a,
            0.0,
            1.0,
            4,
        );

        let mut dl_dout_b = vec![0.0_f32; 4];
        let mut loss_acc_b = 0.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm_b,
            &mut dl_dout_b,
            &mut loss_acc_b,
            0.0,
            1.0,
            4,
        );

        assert!(approx_eq_f64(loss_acc_a, loss_acc_b, 1e-12));
        for i in 0..4 {
            let half = dl_dout_a[i] * 0.5;
            let diff = ((dl_dout_b[i] as f64) - (half as f64)).abs();
            assert!(
                diff < 1e-7,
                "i={i}: half={half} got {} diff {diff}",
                dl_dout_b[i]
            );
        }
    }

    /// `scale` は `out * scale` で sigmoid 内に入るとともに、chain rule で
    /// dl_dout の `* scale` 項として現れる。`scale = 0` で grad 全消え、
    /// `p = sigmoid(0) = 0.5` で loss は固定値になる。
    #[test]
    fn scale_zero_zeroes_grad_and_freezes_p_at_half() {
        let out = vec![100.0_f32, -100.0, 0.0];
        let score = vec![1.0_f32, -1.0, 0.5];
        let wdl = vec![1.0_f32, 0.0, 0.5];
        let per_pos_norm = vec![1.0_f32; 3];
        let mut dl_dout = vec![0.0_f32; 3];
        let mut loss_acc = 0.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            0.5,
            0.0,
            3,
        );
        for &g in &dl_dout {
            assert_eq!(g, 0.0_f32);
        }
        // p = 0.5、ys = sigmoid(0) = 0.5、y = 0.5*wdl + 0.5*0.5 = 0.5*wdl + 0.25
        let expected_loss: f64 = (0..3)
            .map(|i| {
                let y = 0.5_f32 * wdl[i] + 0.5_f32 * 0.5_f32;
                let err = (0.5_f32 - y) as f64;
                err * err
            })
            .sum();
        assert!(
            approx_eq_f64(loss_acc, expected_loss, 1e-12),
            "loss: got {loss_acc} exp {expected_loss}"
        );
    }

    /// 空配列 (n = 0) でも panic せず、loss_acc も dl_dout も変化なし。
    #[test]
    fn empty_input_yields_no_changes() {
        let mut dl_dout: Vec<f32> = vec![];
        let mut loss_acc = 7.0_f64;
        loss_wdl_cpu(&[], &[], &[], &[], &mut dl_dout, &mut loss_acc, 0.5, 1.0, 0);
        assert!(dl_dout.is_empty());
        assert_eq!(loss_acc, 7.0);
    }

    /// loss_acc は **既存値に加算**される (Stage 1 grad と同型、batch 跨ぎ
    /// 累積を host 側で持つため)。先行値 1.0 + 新加算分の合計になることを確認。
    #[test]
    fn loss_acc_accumulates_into_existing_value() {
        let out = vec![1.0_f32];
        let score = vec![0.0_f32];
        let wdl = vec![1.0_f32];
        let per_pos_norm = vec![1.0_f32];
        let mut dl_dout = vec![0.0_f32];
        let mut loss_acc = 1.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout,
            &mut loss_acc,
            0.0,
            1.0,
            1,
        );
        let p = sigmoid_f32(1.0);
        let ys = sigmoid_f32(0.0);
        let err = (p - ys) as f64;
        let expected = 1.0 + err * err;
        assert!(
            approx_eq_f64(loss_acc, expected, 1e-12),
            "loss_acc: got {loss_acc} exp {expected}"
        );
    }
}
