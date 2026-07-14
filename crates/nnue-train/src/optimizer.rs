//! optimizer のハイパーパラメータと host 側事前計算 helper。
//!
//! [`OptimizerKind`] / [`RangerParams`] と、GPU `radam_step` kernel の計算に
//! 一致する [`radam_compute_step_size_denom`] を提供する。

pub use gpu_kernels::pointwise::radam_step::radam_compute_step_size_denom;

// =============================================================================
// optimizer 種別
// =============================================================================

/// optimizer 種別。
///
/// 3 種とも element 更新は同一の GPU `radam_step` kernel を共用し、host 側で
/// 渡す per-step scalar (`step_size`, `denom`)・`beta1`・lookahead lerp の有無
/// だけが異なる:
///
/// | kind   | (step_size, denom)                  | beta1 | lookahead |
/// |--------|-------------------------------------|-------|-----------|
/// | ranger | RAdam rectified schedule            | 0.99  | あり      |
/// | radam  | RAdam rectified schedule            | 0.9   | なし      |
/// | adamw  | 常に (1, 1) = bias correction なし  | 0.9   | なし      |
///
/// `adamw` は bullet (`trainer/src/optimiser/adam.rs`) の AdamW と同じく
/// bias correction を持たない `p -= lr * m / (sqrt(v) + eps)` 形。beta1 の
/// 既定値も bullet の各 optimiser default (Ranger 0.99 / RAdam・AdamW 0.9)
/// に合わせている。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    /// RAdam + Lookahead。
    Ranger,
    /// rectified Adam (Liu et al. 2019)。lookahead なし。
    RAdam,
    /// decoupled weight decay + weight clamp 付き Adam。bias correction なし。
    AdamW,
}

impl OptimizerKind {
    /// CLI 文字列 (case-insensitive) から解決する。未知の名前は `None`。
    pub fn parse(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("ranger") {
            Some(Self::Ranger)
        } else if s.eq_ignore_ascii_case("radam") {
            Some(Self::RAdam)
        } else if s.eq_ignore_ascii_case("adamw") {
            Some(Self::AdamW)
        } else {
            None
        }
    }

    /// 正規化名 (CLI / experiment.json 記録用の小文字表記)。
    pub fn name(self) -> &'static str {
        match self {
            Self::Ranger => "ranger",
            Self::RAdam => "radam",
            Self::AdamW => "adamw",
        }
    }

    /// 1st moment EMA decay。
    pub fn beta1(self) -> f32 {
        match self {
            Self::Ranger => RangerParams::DEFAULT.beta1,
            Self::RAdam | Self::AdamW => 0.9,
        }
    }

    /// Ranger の slow-weight lookahead lerp を行うか。
    pub fn uses_lookahead(self) -> bool {
        matches!(self, Self::Ranger)
    }

    /// `radam_step` kernel に渡す per-step scalar `(step_size, denom)`。
    ///
    /// ranger / radam は RAdam の rectified schedule
    /// ([`radam_compute_step_size_denom`]、`step >= 1` 前提)。adamw は
    /// bias correction を行わないため step によらず `(1.0, 1)` で、kernel の
    /// 更新式は `p -= lr * m / (sqrt(v) + eps)` に退化する。
    pub fn step_size_denom(self, step: u64, beta2: f32, n_sma_threshold: f32) -> (f32, i32) {
        match self {
            Self::Ranger | Self::RAdam => {
                radam_compute_step_size_denom(step, self.beta1(), beta2, n_sma_threshold)
            }
            Self::AdamW => (1.0, 1),
        }
    }
}

// =============================================================================
// パラメータ
// =============================================================================

/// Ranger optimizer のハイパパラメータ。
///
/// default (decay=0.01, beta1=0.99, beta2=0.999, alpha=0.5, k=6) は本 trainer の
/// 標準設定。`radam_step` kernel が要求する eps + n_sma_threshold を field 化して
/// いる。weight clip 範囲は layer (テンソル) ごとに量子化定数から導出する別概念
/// なので本 struct には持たせない (kernel launch 時に per-group で渡す)。
#[derive(Clone, Copy, Debug)]
pub struct RangerParams {
    /// weight decay 係数 (AdamW-style decoupled decay)。
    pub decay: f32,
    /// 1st moment EMA decay。
    pub beta1: f32,
    /// 2nd moment EMA decay。
    pub beta2: f32,
    /// 数値安定化用 epsilon (`1/sqrt(v)+eps`)。
    pub eps: f32,
    /// Lookahead lerp 係数 (`weights = alpha * weights + (1-alpha) * slow`)。
    pub alpha: f32,
    /// Lookahead lerp 周期 (`step % k == 0` で lerp 起動)。
    pub k: usize,
    /// RAdam variance 補正の閾値 (n_sma > threshold で `1/sqrt(v)` 経路を on)。
    pub n_sma_threshold: f32,
}

impl RangerParams {
    /// `Default::default()` と同値の `const` 定数。`const` 文脈で field を直接
    /// 参照したい呼び出し側 (kernel launch 引数) はこれを single source of
    /// truth として使うことで const 値の二重定義を防ぐ。
    pub const DEFAULT: Self = Self {
        decay: 0.01,
        beta1: 0.99,
        beta2: 0.999,
        eps: 1e-8,
        alpha: 0.5,
        k: 6,
        n_sma_threshold: 5.0,
    };
}

impl Default for RangerParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranger_params_default_values() {
        let p = RangerParams::default();
        // 標準 default: decay=0.01, beta1=0.99, beta2=0.999, alpha=0.5, k=6,
        // eps=1e-8, n_sma_threshold=5.0。weight clip 範囲は per-layer で別途渡す
        // ので本 struct には含まない。
        assert_eq!(p.decay, 0.01);
        assert_eq!(p.beta1, 0.99);
        assert_eq!(p.beta2, 0.999);
        assert_eq!(p.alpha, 0.5);
        assert_eq!(p.k, 6);
        assert_eq!(p.eps, 1e-8);
        assert_eq!(p.n_sma_threshold, 5.0);
    }

    #[test]
    fn optimizer_kind_parse_accepts_known_names_case_insensitive() {
        assert_eq!(OptimizerKind::parse("ranger"), Some(OptimizerKind::Ranger));
        assert_eq!(OptimizerKind::parse("Ranger"), Some(OptimizerKind::Ranger));
        assert_eq!(OptimizerKind::parse("radam"), Some(OptimizerKind::RAdam));
        assert_eq!(OptimizerKind::parse("RAdam"), Some(OptimizerKind::RAdam));
        assert_eq!(OptimizerKind::parse("adamw"), Some(OptimizerKind::AdamW));
        assert_eq!(OptimizerKind::parse("AdamW"), Some(OptimizerKind::AdamW));
        assert_eq!(OptimizerKind::parse("sgd"), None);
        assert_eq!(OptimizerKind::parse(""), None);
    }

    #[test]
    fn optimizer_kind_beta1_and_lookahead() {
        assert_eq!(OptimizerKind::Ranger.beta1(), 0.99);
        assert_eq!(OptimizerKind::RAdam.beta1(), 0.9);
        assert_eq!(OptimizerKind::AdamW.beta1(), 0.9);
        assert!(OptimizerKind::Ranger.uses_lookahead());
        assert!(!OptimizerKind::RAdam.uses_lookahead());
        assert!(!OptimizerKind::AdamW.uses_lookahead());
    }

    /// adamw は step によらず `(1.0, 1)` (bias correction なし)。
    #[test]
    fn adamw_step_size_denom_is_constant() {
        for step in [1_u64, 2, 6, 1000] {
            assert_eq!(
                OptimizerKind::AdamW.step_size_denom(step, 0.999, 5.0),
                (1.0, 1)
            );
        }
    }

    /// adamw の per-step scalar `(1.0, 1)` を `radam_step` の CPU reference に
    /// 渡すと、bias correction なしの AdamW 更新式
    /// `p = (p * (1 - decay*lr)) - lr * m / (sqrt(v) + eps)` (+ clamp) に
    /// 一致する。独立に書き下した式と 1 step 突き合わせて固定する。
    #[test]
    fn adamw_scalars_reduce_radam_step_to_plain_adamw() {
        use gpu_kernels::pointwise::radam_step::radam_step_cpu;

        let (beta1, beta2, eps, lr, decay) = (
            OptimizerKind::AdamW.beta1(),
            0.999_f32,
            1e-8_f32,
            0.001_f32,
            0.01_f32,
        );
        let (min_w, max_w) = (-1.98_f32, 1.98_f32);
        let (w0, m0, v0, g) = (0.5_f32, 0.02_f32, 3e-4_f32, 0.1_f32);

        let mut weights = vec![w0];
        let mut m = vec![m0];
        let mut v = vec![v0];
        let mut grad = vec![g];
        let (step_size, denom) = OptimizerKind::AdamW.step_size_denom(1, beta2, 5.0);
        radam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            lr,
            step_size,
            denom,
            decay,
            beta1,
            beta2,
            eps,
            min_w,
            max_w,
            1,
        );

        let m1 = beta1 * m0 + (1.0 - beta1) * g;
        let v1 = beta2 * v0 + (1.0 - beta2) * g * g;
        let expected =
            ((w0 * (1.0 - decay * lr)) - lr * m1 / (v1.sqrt() + eps)).clamp(min_w, max_w);
        assert_eq!(m[0], m1);
        assert_eq!(v[0], v1);
        assert!(
            (weights[0] - expected).abs() < 1e-7,
            "got {} exp {expected}",
            weights[0]
        );
    }

    /// ranger / radam は各自の beta1 で rectified schedule を計算する。
    /// beta1 は `step_size` の bias correction 項 `1 - beta1^t` にのみ影響し、
    /// `denom` は beta2 のみで決まる。
    #[test]
    fn ranger_and_radam_step_size_follow_rectified_schedule() {
        for step in [1_u64, 10, 1000] {
            let ranger = OptimizerKind::Ranger.step_size_denom(step, 0.999, 5.0);
            let radam = OptimizerKind::RAdam.step_size_denom(step, 0.999, 5.0);
            assert_eq!(
                ranger,
                radam_compute_step_size_denom(step, 0.99, 0.999, 5.0)
            );
            assert_eq!(radam, radam_compute_step_size_denom(step, 0.9, 0.999, 5.0));
            assert_eq!(ranger.1, radam.1);
        }
    }
}
