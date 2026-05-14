//! Fused AdamW optimizer step (decay + clip 込み) の reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn adamw_step` は bin entry (`bins/nnue_train/src/main.rs`)
//! に inline 定義されている (cuda-oxide rustc-codegen-cuda backend の bin-entry
//! 制約)。本 module の `adamw_step_cpu` は GPU と同じ更新式を host で書き写した
//! もので、GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム (bullet 上流 `optimiser/adam.rs::AdamWParams::build` に等価)
//!
//! 1 thread = 1 weight、atomics 不要 (Stage 1 `progress::adam_step` と同型)。
//!
//! ```text
//! per element i:
//!     g          = grad[i]
//!     weights[i] *= 1 - decay * lr               # weight decay (AdamW の本質)
//!     m[i]       = beta1 * m[i] + (1 - beta1) * g
//!     v[i]       = beta2 * v[i] + (1 - beta2) * g * g
//!     val        = m[i] / (sqrt(v[i]) + eps)
//!     weights[i] -= lr * val
//!     weights[i] = clamp(weights[i], min_w, max_w)
//!     grad[i]    = 0                             # 次 batch 用に reset (Stage 1 慣行)
//! ```
//!
//! - **bias correction なし**: bullet 上流 AdamW (`adam.rs:34-50`) は意図的に
//!   bias correction (`bc1 = 1 - beta1^t` 等) を含まず、`val = m / (sqrt(v) + eps)`
//!   を直接使う。RAdam と分岐させる前の "AdamW base" 形 (Stage 2-4 で RAdam に
//!   bias correction + denom switch を加える)。Stage 1 `progress::adam_step` は
//!   bc1/bc2 を host pre-compute して渡す形 (元の bullet `KERNELS_SRC::k_adam_step`)
//!   と異なる convention で、混同しないよう本 docstring で明示
//! - **decay + clip**: AdamW の差分。`decay = 0.0` で plain Adam (bullet 上流式)、
//!   `min_w = f32::MIN, max_w = f32::MAX` で clip 無効
//! - **grad reset**: Stage 1 `progress::adam_step` 慣行を踏襲。bullet 上流は
//!   `gradients` を `const float*` で受けて reset しないが、本リポは host loop が
//!   次 batch の `atomicAdd` 累積に向けて kernel 内で reset する設計
//!
//! ## bullet 上流との対応 / divergence
//!
//! - bullet `KernelSrc` は `float4` vectorize path を持つ (`size % 4 == 0` なら
//!   1 thread = 4 weights を unroll)。本 PR は **scalar 1 thread = 1 weight** で
//!   素直に書く (Stage 1 慣行、`size` が 4 の倍数でないケースの分岐コストが
//!   学習律速にはならない、Stage 2-8 で必要に応じ optimize 候補)
//! - bullet `adj * grad` は host buffer 経由 (`adj_ptr` 1-element)。本実装は
//!   `lr` を直接渡す (Stage 1 同型)。`adj` (gradient_factor) が必要になる時は
//!   後で追加する想定
//!
//! ## cuda-oxide 制限
//!
//! - GPU kernel 側は `f32::clamp` / `min` / `max` を使えない (Stage 1-7 で確認、
//!   `Symbol std__intrinsics__maximum_number_nsz_f32 not found`)。**`if-else`
//!   ladder で展開** する (`x < min_w ? min_w : x > max_w ? max_w : x`)。
//!   CPU reference は host 実行で `f32::clamp` を使用
//! - `v.sqrt()` は cuda-oxide が `__nv_sqrtf` (libdevice) に lowering する。
//!   Stage 1-7 で動作確認済

/// Reference CPU 実装。
///
/// In-place mutation:
/// - `weights[i]`: weight decay → Adam update → clamp の 3 段
/// - `m[i]` / `v[i]`: Adam 1 次 / 2 次 moment running average
/// - `grad[i]`: 0.0 にリセット (次 batch の accumulation 用、Stage 1 同 convention)
///
/// 入力前提:
/// - `weights.len() == m.len() == v.len() == grad.len() == n`
/// - `lr ≥ 0`, `decay ≥ 0`, `0 < beta1 < 1`, `0 < beta2 < 1`, `eps > 0`
/// - `min_w ≤ max_w` (host 側不変条件、`f32::clamp` の panic 経路を避けるため)
///
/// 引数数 (12) は bullet 上流 `adam.rs::OP` 引数 + AdamW 拡張のため
/// `clippy::too_many_arguments` を allow (Stage 1 `progress::adam_step` と同方針)。
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_cpu(
    weights: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &mut [f32],
    lr: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_w: f32,
    max_w: f32,
    n: usize,
) {
    for i in 0..n {
        let g = grad[i];
        let mut p = weights[i];
        p *= 1.0_f32 - decay * lr;
        let mi = beta1 * m[i] + (1.0_f32 - beta1) * g;
        let vi = beta2 * v[i] + (1.0_f32 - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        let val = mi / (vi.sqrt() + eps);
        p -= lr * val;
        weights[i] = p.clamp(min_w, max_w);
        grad[i] = 0.0_f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `decay = 0`、clip 無効 (`-INF, +INF`) で plain Adam (bullet 上流の
    /// `KERNELS_SRC::k_adam_step` の bias-correction を取り除いた形) と一致する。
    /// `g = 0` (gradient ゼロ) なら m / v / weights は 1 step で変化なし
    /// (m_init = 0、v_init = 0、val = 0 / eps = 0)。
    #[test]
    fn zero_grad_zero_decay_yields_no_change() {
        let mut weights = vec![0.5_f32, -0.3, 1.0];
        let mut m = vec![0.0_f32; 3];
        let mut v = vec![0.0_f32; 3];
        let mut grad = vec![0.0_f32; 3];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            1e-3,
            0.0,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            3,
        );
        assert_eq!(weights, vec![0.5_f32, -0.3, 1.0]);
        assert_eq!(m, vec![0.0_f32; 3]);
        assert_eq!(v, vec![0.0_f32; 3]);
        // grad は 0 にリセット (元から 0 だったが convention 確認)
        assert_eq!(grad, vec![0.0_f32; 3]);
    }

    /// 既知入力 1 step の hand-calc。
    /// w = 1.0、g = 0.1、lr = 0.1、decay = 0.01、beta1 = 0.9、beta2 = 0.999、eps = 1e-8
    /// w_after_decay = 1.0 * (1 - 0.01 * 0.1) = 0.999
    /// m = 0.9*0 + 0.1*0.1 = 0.01
    /// v = 0.999*0 + 0.001*0.01 = 1e-5
    /// val = 0.01 / (sqrt(1e-5) + 1e-8) ≈ 0.01 / 0.0031623 ≈ 3.1623
    /// w = 0.999 - 0.1 * 3.1623 ≈ 0.6828
    #[test]
    fn one_step_hand_calc() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![0.1_f32];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1,
            0.01,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            1,
        );

        // 期待値は f32 で再計算 → f64 cast (Stage 1-10 pitfall 回避)
        let g = 0.1_f32;
        let lr = 0.1_f32;
        let decay = 0.01_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let mut p = 1.0_f32;
        p *= 1.0_f32 - decay * lr;
        let mi = beta1 * 0.0_f32 + (1.0_f32 - beta1) * g;
        let vi = beta2 * 0.0_f32 + (1.0_f32 - beta2) * g * g;
        let val = mi / (vi.sqrt() + eps);
        p -= lr * val;

        let diff = ((weights[0] as f64) - (p as f64)).abs();
        assert!(diff < 1e-7, "got {} exp {p} diff {diff}", weights[0]);
        assert!(((m[0] as f64) - (mi as f64)).abs() < 1e-7);
        assert!(((v[0] as f64) - (vi as f64)).abs() < 1e-7);
        assert_eq!(grad, vec![0.0_f32]);
    }

    /// `decay = 0`、`g = 0`、`lr = 0` で weights のみ操作される設定で、
    /// **clip が範囲外 weight を引き戻す** ことを確認 (Adam update 経路を
    /// 殺して clamp 単独の効果を見る、これが破れると clamp の if-else 展開で
    /// `>` / `<` の境界扱いが GPU と CPU で divergent になる)。
    /// weights = [100, -100, 0.5]、min_w = -1.0、max_w = 1.0
    /// → weights = clamp(100, -1, 1) = 1.0、clamp(-100, -1, 1) = -1.0、interior 0.5 維持
    #[test]
    fn clamp_pulls_weights_into_range() {
        let mut weights = vec![100.0_f32, -100.0, 0.5];
        let mut m = vec![0.0_f32; 3];
        let mut v = vec![0.0_f32; 3];
        let mut grad = vec![0.0_f32; 3];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.0,
            0.0,
            0.9,
            0.999,
            1e-8,
            -1.0,
            1.0,
            3,
        );
        assert_eq!(weights, vec![1.0_f32, -1.0, 0.5]);
    }

    /// `decay > 0`、`lr > 0`、`g = 0` で weight decay 単独効果を確認。
    /// w = 1.0、decay = 0.1、lr = 0.5
    /// w *= 1 - 0.1 * 0.5 = 0.95
    /// w -= 0.5 * (0 / (sqrt(0) + eps)) = 0.95 - 0 = 0.95
    #[test]
    fn pure_weight_decay_with_zero_grad() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![0.0_f32];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.5,
            0.1,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            1,
        );
        let expected = 1.0_f32 * (1.0_f32 - 0.1_f32 * 0.5_f32);
        let diff = ((weights[0] as f64) - (expected as f64)).abs();
        assert!(diff < 1e-7, "got {} exp {expected} diff {diff}", weights[0]);
    }

    /// 5 step を host 実行で回し、最終 weights が monotonic に target に近づくこと
    /// (簡易的な convergence 確認)。target = 0、initial weight = 1、grad = 1.0 を
    /// 与え続ける (= w を 0 に向ける gradient)。lr = 0.1、decay = 0、bias-correction
    /// なしなので monotonic decreasing になるはず。
    #[test]
    fn five_step_monotonic_descent_toward_zero() {
        let mut weights = vec![1.0_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut prev_w = weights[0];
        for _ in 0..5 {
            // 各 step で grad = 1.0 を与える (kernel 内で reset → 次 step で再 set)
            let mut grad = vec![1.0_f32];
            adamw_step_cpu(
                &mut weights,
                &mut m,
                &mut v,
                &mut grad,
                0.1,
                0.0,
                0.9,
                0.999,
                1e-8,
                f32::MIN,
                f32::MAX,
                1,
            );
            assert!(
                weights[0] < prev_w,
                "step did not decrease: prev={prev_w} cur={}",
                weights[0]
            );
            prev_w = weights[0];
        }
        // initial 1.0 から 5 step で十分に 0 に近づいている
        assert!(weights[0] < 0.7, "after 5 steps, w={}", weights[0]);
    }

    /// NaN 入力 (grad = NaN) で weights が NaN に汚染されることを確認 (loss_wdl
    /// と同型で NaN を伝搬する。学習中の NaN 検出は loss/optimizer 経路で気付ける
    /// 必要があり、SCReLU のような握り潰しは optimizer では行わない)。
    /// `m = beta1 * 0 + (1-beta1) * NaN = NaN`、`v` も `* NaN^2` で NaN、
    /// `m / sqrt(v)` も NaN、`weights -= lr * NaN = NaN`、最後の `clamp` で
    /// f32::clamp は NaN 入力を NaN のまま返す (IEEE 754 仕様 + Rust spec)。
    #[test]
    fn nan_grad_propagates_into_weights() {
        let mut weights = vec![0.5_f32];
        let mut m = vec![0.0_f32];
        let mut v = vec![0.0_f32];
        let mut grad = vec![f32::NAN];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            1e-3,
            0.0,
            0.9,
            0.999,
            1e-8,
            f32::MIN,
            f32::MAX,
            1,
        );
        assert!(weights[0].is_nan(), "expected NaN, got {}", weights[0]);
        assert!(m[0].is_nan());
        assert!(v[0].is_nan());
        // grad は kernel 内で 0 にリセットされる (NaN 元 grad が次 batch に
        // 持ち越されない、これは設計上の自衛 = optimizer step 後の grad は
        // 必ず 0 になる)
        assert_eq!(grad, vec![0.0_f32]);
    }

    /// `min_w == max_w` で weights が単一値に collapse する degenerate clip。
    /// host 側不変条件 (`min_w ≤ max_w`) 内で許される境界 (`<` ではなく `≤`)。
    /// kernel の if-else ladder (`p < min_w ? min_w : p > max_w ? max_w : p`)
    /// で `min_w == max_w` のとき `p > max_w` が true なら max_w、false なら p
    /// (`p < min_w` も false の経路) になる挙動が CPU の `f32::clamp` と一致する
    /// ことを確認。
    #[test]
    fn collapsed_clip_range_pins_weights_to_single_value() {
        let mut weights = vec![100.0_f32, -100.0, 0.0, 0.5];
        let mut m = vec![0.0_f32; 4];
        let mut v = vec![0.0_f32; 4];
        let mut grad = vec![1.0_f32; 4];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1,
            0.0,
            0.9,
            0.999,
            1e-8,
            0.5,
            0.5,
            4,
        );
        assert_eq!(weights, vec![0.5_f32; 4]);
    }

    /// 空配列 (n = 0) でも panic せず no-op。
    #[test]
    fn empty_input_yields_no_changes() {
        let mut weights: Vec<f32> = vec![];
        let mut m: Vec<f32> = vec![];
        let mut v: Vec<f32> = vec![];
        let mut grad: Vec<f32> = vec![];
        adamw_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            0.1,
            0.01,
            0.9,
            0.999,
            1e-8,
            -1.0,
            1.0,
            0,
        );
        assert!(weights.is_empty());
    }
}
