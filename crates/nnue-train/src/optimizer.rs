//! Ranger optimizer のハイパーパラメータと host 側事前計算 helper。
//!
//! [`RangerParams`] と、GPU `radam_step` kernel の計算に一致する
//! [`radam_compute_step_size_denom`] を提供する。

pub use gpu_kernels::pointwise::radam_step::radam_compute_step_size_denom;

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
}
