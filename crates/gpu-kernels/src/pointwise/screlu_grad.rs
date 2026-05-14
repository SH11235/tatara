//! SCReLU activation gradient (fused) reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn screlu_grad` は bin entry (`bins/nnue_train/src/main.rs`)
//! に inline 定義されている (cuda-oxide rustc-codegen-cuda backend の bin-entry
//! 制約)。本 module の `screlu_grad_cpu` は GPU と同じロジックを host で書き写した
//! もので、GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム
//!
//! bullet 上流 (`crates/compiler/src/tensor/operation/autograd/dfo.rs::SCReLU`)
//! の forward / backward から fused gradient を導出する:
//!
//! ```text
//! SCReLU.forward(x)  = clamp(x, 0, 1)^2
//! SCReLU.backward(y) = 2 * sqrt(y) * IsPositive(1 - sqrt(y))
//!                    = 2 * a * (a > 0 && a < 1)        (a = clamp(x, 0, 1))
//! ```
//!
//! 入力 `x` を直接受け取る fused 形式に hand-fuse すると 1 thread = 1 element
//! で:
//!
//! ```text
//! a    = clamp(x, 0, 1)
//! dydx = if a > 0 && a < 1 { 2 * a } else { 0 }
//! dL_dx = dL_dy * dydx
//! ```
//!
//! 端点 `x = 0` (a = 0) と `x = 1` (a = 1) では SCReLU の forward が piecewise
//! 連続な極小・極大に達するため derivative は 0 (劣微分 0)。bullet 上流の
//! `IsPositive` も `> 0` (strict) なので一致する。
//!
//! ## bullet 上流との対応
//!
//! - bullet `SCReLU::backward(output: y)` は forward の output を入力に取る形
//!   (recompute なし、PointwiseIR が forward と同 batch でつなぐ前提) だが、
//!   本 fused kernel は **forward の input `x` を直接受け取る**形にしている:
//!   - CPU memory 上には x がそのまま残っているので追加 traffic 不要
//!   - `clamp(x, 0, 1)` の評価で a を再計算するコストは 1 op、`sqrt(y)`
//!     を取る方が高くつく (intrinsics call)
//!   - bullet 流 `2 * sqrt(y) * IsPositive(1 - sqrt(y))` (y = a^2) は内部で
//!     `a^2` と `sqrt(a^2)` を経由するため interior 値で 2 回の中間丸めが
//!     入る。x からの再評価は `clamp` 1 回で済み、interior の `2 * a` を
//!     直接出すので中間丸め (`sqrt(a*a)` 経由) を避けられる
//! - 結果は数値同値: bullet `2 * sqrt(y) * IsPositive(1 - sqrt(y))` と本
//!   実装 `2 * a * (a > 0 && a < 1)` は数学的に同一。float round-off も
//!   x-based のほうが小さい (`sqrt(a*a)` の中間丸めが無い)
//!
//! ## cuda-oxide 制限
//!
//! - host CPU reference では `f32::clamp(0.0, 1.0)` と `if-else` を使い分けて
//!   readable に書く (host 実行で `f32::clamp` は問題なく動く)
//! - GPU `#[kernel]` 側は `f32::clamp` も内部で `f32::max` / `f32::min` を呼ぶ
//!   ため lowering 失敗するリスクあり (Stage 1-7 で確認した `f32::max` lowering
//!   未対応問題)。kernel 側は **`if-else` ladder で展開**する (`x < 0 ? 0 : x > 1
//!   ? 1 : x`)
//! - `IsPositive` (a > 0 && a < 1) も bool → f32 cast を介さず `if-else` で書く

/// Reference CPU 実装。
///
/// In-place 出力 (`dl_dx`) を 1 element ずつ埋める。`x.len() / dl_dy.len() /
/// dl_dx.len() == n` を host 側 invariant として要求する。
///
/// 計算式:
///
/// ```text
/// a    = clamp(x[i], 0.0, 1.0)
/// dydx = if 0.0 < a && a < 1.0 { 2.0 * a } else { 0.0 }
/// dl_dx[i] = dl_dy[i] * dydx
/// ```
///
/// ## NaN / Inf 挙動 (bullet 上流との divergence)
///
/// - `x[i] = NaN`: `clamp(NaN, 0, 1)` の結果は `NaN`、続く `0 < a` / `a < 1`
///   比較は IEEE 754 仕様で **両方 false に倒れる** ため `dydx = 0`、`dl_dx = 0`
///   (ただし `dl_dy[i] = NaN` なら `0 * NaN = NaN` で伝搬)
/// - `x[i] = +Inf`: clamp で `a = 1.0`、`a < 1` false → `dydx = 0`
/// - `x[i] = -Inf`: clamp で `a = 0.0`、`0 < a` false → `dydx = 0`
///
/// bullet 上流の `2 * sqrt(y) * IsPositive(1 - sqrt(y))` (y = a^2) は NaN
/// 入力で `2 * sqrt(NaN) * 0 = NaN` を伝搬する点で本実装と **divergent**。
/// 本実装は NaN を 0 grad に握り潰すが、学習中の NaN 検出は forward / loss 等の
/// 上流 op で行う前提なので致命的ではない。NaN を渡された SCReLU.backward 自体は
/// 病態 (forward が NaN を出している) で、grad の経路で気付くより forward
/// 経路での guard が本筋。
pub fn screlu_grad_cpu(x: &[f32], dl_dy: &[f32], dl_dx: &mut [f32], n: usize) {
    for i in 0..n {
        let a = x[i].clamp(0.0_f32, 1.0_f32);
        let dydx = if a > 0.0_f32 && a < 1.0_f32 {
            2.0_f32 * a
        } else {
            0.0_f32
        };
        dl_dx[i] = dl_dy[i] * dydx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手計算可能な少数 element を入れて期待値と一致するか検証する。
    /// 期待値は f32 で計算して f64 cast (Stage 1-10 で確立した f32 リテラル比較
    /// pitfall: 例 `(0.7 - 1.0)^2 != 0.09` を回避)。
    #[test]
    fn small_known_input_matches_hand_calculation() {
        // x:    [-0.5, 0.0, 0.3, 0.7, 1.0, 1.5]
        // a:    [0.0,  0.0, 0.3, 0.7, 1.0, 1.0]   (clamp [0, 1])
        // dydx: [0.0,  0.0, 0.6, 1.4, 0.0, 0.0]   (端点 a=0/1 で 0)
        // dl_dy: [1, 2, 3, 4, 5, 6]
        // → dl_dx: [0, 0, 1.8, 5.6, 0, 0]
        let x = vec![-0.5_f32, 0.0, 0.3, 0.7, 1.0, 1.5];
        let dl_dy = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dl_dx = vec![0.0_f32; 6];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx, 6);

        let expected = [
            0.0_f32,
            0.0_f32,
            (3.0_f32 * (2.0_f32 * 0.3_f32)),
            (4.0_f32 * (2.0_f32 * 0.7_f32)),
            0.0_f32,
            0.0_f32,
        ];
        for (i, (got, exp)) in dl_dx.iter().zip(expected.iter()).enumerate() {
            let diff = ((*got as f64) - (*exp as f64)).abs();
            assert!(diff < 1e-7, "i={i}: got {got} exp {exp} diff {diff}",);
        }
    }

    /// 端点ちょうど (a = 0 or a = 1) では derivative 0 (`>` strict、bullet
    /// `IsPositive` と同義)。これが破れると forward と backward の整合性が
    /// 崩れる (PointwiseIR との数値ドリフト) のでガード。
    #[test]
    fn boundary_a_equals_zero_or_one_yields_zero_grad() {
        let x = vec![0.0_f32, 1.0_f32];
        let dl_dy = vec![5.0_f32, 7.0_f32];
        let mut dl_dx = vec![1.0_f32; 2];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx, 2);
        assert_eq!(dl_dx, vec![0.0_f32, 0.0_f32]);
    }

    /// クランプの飽和域 (x < 0 or x > 1) も derivative 0。
    #[test]
    fn saturated_inputs_yield_zero_grad() {
        let x = vec![-100.0_f32, 100.0_f32, -1e-9, 1.0 + 1e-9];
        let dl_dy = vec![1.0_f32; 4];
        let mut dl_dx = vec![999.0_f32; 4];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx, 4);
        assert_eq!(dl_dx, vec![0.0_f32; 4]);
    }

    /// dL/dy を伝播する: dl_dy の値が dl_dx に線形に乗る (上流 grad pass-through
    /// の検証)。a = 0.5 で dydx = 1.0、dl_dx = dl_dy が成り立つ。
    #[test]
    fn dl_dy_propagates_linearly_when_dydx_equals_one() {
        let x = vec![0.5_f32; 4];
        let dl_dy = vec![1.0_f32, -2.0_f32, 3.5_f32, 0.0_f32];
        let mut dl_dx = vec![0.0_f32; 4];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx, 4);
        // a=0.5 → dydx = 2 * 0.5 = 1.0、dl_dx = dl_dy
        for (got, exp) in dl_dx.iter().zip(dl_dy.iter()) {
            assert!((got - exp).abs() < 1e-7, "got {got} exp {exp}");
        }
    }

    /// 空配列 (n = 0) でも panic せず no-op になる。
    #[test]
    fn empty_input_yields_empty_output() {
        let mut dl_dx: Vec<f32> = vec![];
        screlu_grad_cpu(&[], &[], &mut dl_dx, 0);
        assert!(dl_dx.is_empty());
    }

    /// NaN / Inf 入力では `dl_dx = 0` になる (bullet 上流は NaN を伝搬するが、
    /// 本実装は `clamp + (0 < a && a < 1)` の比較が NaN/Inf で false に倒れるため
    /// 自然と 0 に落ちる。docstring の "NaN / Inf 挙動" セクション参照)。
    /// 学習中の NaN 検出は上流 (forward / loss) で行う前提。
    #[test]
    fn nan_and_inf_inputs_yield_zero_grad() {
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let dl_dy = vec![1.0_f32; 3];
        let mut dl_dx = vec![999.0_f32; 3];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx, 3);
        assert_eq!(dl_dx, vec![0.0_f32; 3]);
    }
}
