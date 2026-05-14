//! SCReLU activation gradient (fused) の reference CPU 実装。
//!
//! GPU 側 `#[kernel] fn screlu_grad` は呼び出し元 bin entry に inline 定義する
//! (cuda-oxide rustc-codegen-cuda backend の bin-entry 制約)。本 module は
//! GPU↔CPU 数値同等性テストの reference 用。
//!
//! ## アルゴリズム
//!
//! SCReLU の forward / backward は次の通り (bullet 上流 `SCReLU` 同等):
//!
//! ```text
//! SCReLU.forward(x)  = clamp(x, 0, 1)^2
//! SCReLU.backward(y) = 2 * sqrt(y) * IsPositive(1 - sqrt(y))
//!                    = 2 * a * (a > 0 && a < 1)        (a = clamp(x, 0, 1))
//! ```
//!
//! 入力 `x` を直接受け取る fused 形式に書き直すと 1 thread = 1 element で:
//!
//! ```text
//! a    = clamp(x, 0, 1)
//! dydx = if a > 0 && a < 1 { 2 * a } else { 0 }
//! dL_dx = dL_dy * dydx
//! ```
//!
//! 端点 `x = 0` (a = 0) と `x = 1` (a = 1) では derivative は 0 (劣微分 0)。
//!
//! ## 入力 x を受け取る理由 (出力 y を受け取る形との対比)
//!
//! - 出力 y からの再構成 `2 * sqrt(y) * IsPositive(1 - sqrt(y))` (y = a^2) は
//!   `sqrt(a*a)` の中間丸めが入る。x から `clamp` 1 回で再計算した方が float
//!   round-off が小さく、`sqrt` intrinsic call も避けられる
//! - 結果は数学的に同値 (`2 * a * (a > 0 && a < 1)`)
//!
//! ## cuda-oxide 制限
//!
//! - GPU `#[kernel]` 側は `f32::clamp` / `f32::max` / `f32::min` を lower でき
//!   ないため `if-else` ladder で展開する (`x < 0 ? 0 : x > 1 ? 1 : x`)
//! - `IsPositive` (a > 0 && a < 1) も bool → f32 cast を介さず `if-else` で書く
//! - host CPU reference では `f32::clamp` を使う (host 実行では問題なく動く)

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
/// ## NaN / Inf 挙動
///
/// - `x[i] = NaN`: `clamp(NaN, 0, 1)` は `NaN`、続く `0 < a` / `a < 1` 比較は
///   IEEE 754 で両方 false に倒れるため `dydx = 0`、`dl_dx = 0`
///   (ただし `dl_dy[i] = NaN` なら `0 * NaN = NaN` で伝搬)
/// - `x[i] = +Inf`: clamp で `a = 1.0`、`a < 1` false → `dydx = 0`
/// - `x[i] = -Inf`: clamp で `a = 0.0`、`0 < a` false → `dydx = 0`
///
/// NaN を 0 grad に握り潰すため学習中の NaN 検出はできないが、SCReLU.backward に
/// NaN が渡る時点で forward が病態なので forward / loss 経路で気付く前提。
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
    /// 期待値は f32 で計算して f64 cast (f32 リテラル比較の pitfall
    /// 例 `(0.7 - 1.0)^2 != 0.09` を回避)。
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

    /// 端点ちょうど (a = 0 or a = 1) では derivative 0 (strict `>`)。
    /// これが破れると forward と backward の整合性が崩れる (clamp の境界で
    /// 数値ドリフトする) のでガード。
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

    /// NaN / Inf 入力では `dl_dx = 0` になる (`clamp + (0 < a && a < 1)` の
    /// 比較が NaN/Inf で false に倒れるため自然に 0 に落ちる、docstring の
    /// "NaN / Inf 挙動" セクション参照)。学習中の NaN 検出は上流 (forward /
    /// loss) で行う前提。
    #[test]
    fn nan_and_inf_inputs_yield_zero_grad() {
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let dl_dy = vec![1.0_f32; 3];
        let mut dl_dx = vec![999.0_f32; 3];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx, 3);
        assert_eq!(dl_dx, vec![0.0_f32; 3]);
    }
}
