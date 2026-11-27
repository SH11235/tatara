use shogi_features::FeatureSetSpec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OptimGroupKind {
    Ft,
    L1,
    L1Shared,
    Dense,
    Bias,
}

/// 1 param-group の weight_decay (絶対値) と lr_mult (scheduled lr への倍率)。
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OptimGroupParams {
    pub(crate) weight_decay: f32,
    pub(crate) lr_mult: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OptimGroupConfig {
    pub(crate) ft: OptimGroupParams,
    pub(crate) ft_piece_lr_mult: f32,
    pub(crate) threat_lr_mult: f32,
    pub(crate) dense: OptimGroupParams,
    l1_lr_mult: Option<f32>,
    l1_shared_weight_decay: Option<f32>,
    pub(crate) bias: OptimGroupParams,
}

impl OptimGroupConfig {
    pub(crate) fn resolve(
        global_weight_decay: f32,
        ft_weight_decay: Option<f32>,
        dense_weight_decay: Option<f32>,
        bias_weight_decay: Option<f32>,
        ft_lr_mult: Option<f32>,
        dense_lr_mult: Option<f32>,
        bias_lr_mult: Option<f32>,
    ) -> Self {
        let group = |wd: Option<f32>, lr_mult: Option<f32>| OptimGroupParams {
            weight_decay: wd.unwrap_or(global_weight_decay),
            lr_mult: lr_mult.unwrap_or(1.0),
        };
        OptimGroupConfig {
            ft: group(ft_weight_decay, ft_lr_mult),
            ft_piece_lr_mult: 1.0,
            threat_lr_mult: 1.0,
            dense: group(dense_weight_decay, dense_lr_mult),
            bias: group(bias_weight_decay, bias_lr_mult),
            l1_lr_mult: None,
            l1_shared_weight_decay: None,
        }
    }

    pub(crate) fn with_l1_shared_weight_decay(mut self, decay: Option<f32>) -> Self {
        self.l1_shared_weight_decay = decay;
        self
    }

    pub(crate) fn with_l1_lr_mult(mut self, l1_lr_mult: Option<f32>) -> Self {
        self.l1_lr_mult = l1_lr_mult;
        self
    }

    pub(crate) fn with_ft_row_lr_mults(mut self, piece: Option<f32>, threat: Option<f32>) -> Self {
        self.ft_piece_lr_mult = piece.unwrap_or(1.0);
        self.threat_lr_mult = threat.unwrap_or(1.0);
        self
    }

    #[cfg(any(feature = "native", test))]
    pub(crate) fn ft_row_learning_rates(&self, ft_lr: f32) -> (f32, f32) {
        (ft_lr * self.ft_piece_lr_mult, ft_lr * self.threat_lr_mult)
    }

    pub(crate) fn uniform_ft_lr_mult(&self, feature_set: FeatureSetSpec) -> Option<f32> {
        let mult = self.ft_piece_lr_mult;
        if (feature_set.threat_dims() == 0 || self.threat_lr_mult == mult)
            && (!feature_set.ft_factorize() || mult == 1.0)
        {
            Some(mult)
        } else {
            None
        }
    }

    fn params(&self, kind: OptimGroupKind) -> OptimGroupParams {
        match kind {
            OptimGroupKind::Ft => self.ft,
            OptimGroupKind::L1 => OptimGroupParams {
                lr_mult: self.l1_lr_mult.unwrap_or(self.dense.lr_mult),
                ..self.dense
            },
            OptimGroupKind::L1Shared => OptimGroupParams {
                weight_decay: self
                    .l1_shared_weight_decay
                    .unwrap_or(self.dense.weight_decay),
                lr_mult: self.l1_lr_mult.unwrap_or(self.dense.lr_mult),
            },
            OptimGroupKind::Dense => self.dense,
            OptimGroupKind::Bias => self.bias,
        }
    }

    /// group の `radam_step` launch に渡す `(weight_decay, lr)` を返す。lr は
    /// schedule が決めた毎 step の `scheduled_lr` に group の lr_mult を掛けた値
    /// (`lr_for(group) = scheduled_lr × lr_mult`)。
    pub(crate) fn effective(&self, kind: OptimGroupKind, scheduled_lr: f32) -> (f32, f32) {
        let p = self.params(kind);
        (p.weight_decay, scheduled_lr * p.lr_mult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// per-group flag を一つも指定しない (`None`) と、全 group で weight_decay = 大域値・
    /// lr_mult = 1.0 に resolve され、`effective` の lr は `scheduled_lr × 1.0` で
    /// scheduled_lr と bit-identical (= 既定 launch 引数が単一 weight_decay 経路と一致)。
    #[test]
    fn shared_decay_overrides_only_shared_weights_and_preserves_l1_rate() {
        let base = OptimGroupConfig::resolve(0.1, None, Some(0.2), None, None, Some(3.0), None)
            .with_l1_lr_mult(Some(0.5));
        assert_eq!(
            base.effective(OptimGroupKind::L1Shared, 0.02),
            base.effective(OptimGroupKind::L1, 0.02)
        );
        let changed = base.with_l1_shared_weight_decay(Some(0.7));
        assert_eq!(
            changed.effective(OptimGroupKind::L1Shared, 0.02),
            (0.7, 0.01)
        );
        for kind in [
            OptimGroupKind::Ft,
            OptimGroupKind::L1,
            OptimGroupKind::Dense,
            OptimGroupKind::Bias,
        ] {
            assert_eq!(base.effective(kind, 0.02), changed.effective(kind, 0.02));
        }
    }

    #[test]
    fn resolve_without_overrides_falls_back_to_global() {
        let cfg = OptimGroupConfig::resolve(0.125, None, None, None, None, None, None);
        let expected = OptimGroupParams {
            weight_decay: 0.125,
            lr_mult: 1.0,
        };
        assert_eq!(cfg.ft, expected);
        assert_eq!(cfg.dense, expected);
        assert_eq!(cfg.bias, expected);
        let scheduled_lr = 0.0123_f32;
        for kind in [
            OptimGroupKind::Ft,
            OptimGroupKind::L1,
            OptimGroupKind::Dense,
            OptimGroupKind::Bias,
        ] {
            let (wd, lr) = cfg.effective(kind, scheduled_lr);
            assert_eq!(wd, 0.125);
            // `× 1.0` は finite f32 で恒等なので scheduled_lr と bit-identical。
            assert_eq!(lr.to_bits(), scheduled_lr.to_bits());
        }
    }

    /// per-group override は group ごとに resolve され、未指定 group のみ大域値 /
    /// lr_mult=1.0 に落ちる。`effective` の lr は `scheduled_lr × lr_mult`。
    #[test]
    fn resolve_with_overrides_is_per_group() {
        // ft: wd 上書きのみ (lr_mult は default 1.0)、dense: lr_mult 上書きのみ
        // (wd は大域)、bias: wd=0 opt-in + lr_mult 上書き。
        let cfg = OptimGroupConfig::resolve(
            0.1,        // global weight_decay
            Some(0.5),  // ft_weight_decay
            None,       // dense_weight_decay → 大域 0.1
            Some(0.0),  // bias_weight_decay (opt-in wd=0)
            None,       // ft_lr_mult → 1.0
            Some(2.0),  // dense_lr_mult
            Some(0.25), // bias_lr_mult
        );
        assert_eq!(
            cfg.ft,
            OptimGroupParams {
                weight_decay: 0.5,
                lr_mult: 1.0
            }
        );
        assert_eq!(
            cfg.dense,
            OptimGroupParams {
                weight_decay: 0.1,
                lr_mult: 2.0
            }
        );
        assert_eq!(
            cfg.bias,
            OptimGroupParams {
                weight_decay: 0.0,
                lr_mult: 0.25
            }
        );
        let lr = 0.01_f32;
        assert_eq!(cfg.effective(OptimGroupKind::Ft, lr), (0.5, 0.01));
        assert_eq!(cfg.effective(OptimGroupKind::Dense, lr), (0.1, 0.02));
        assert_eq!(cfg.effective(OptimGroupKind::Bias, lr), (0.0, 0.0025));

        let (l1_wd, l1_lr) = cfg.effective(OptimGroupKind::L1, lr);
        let (dense_wd, dense_lr) = cfg.effective(OptimGroupKind::Dense, lr);
        assert_eq!(l1_wd.to_bits(), dense_wd.to_bits());
        assert_eq!(l1_lr.to_bits(), dense_lr.to_bits());
    }

    #[test]
    fn ft_row_multipliers_compose_with_ft_rate_only() {
        let baseline = OptimGroupConfig::resolve(
            0.125,
            Some(0.25),
            None,
            None,
            Some(4.0),
            Some(2.0),
            Some(0.75),
        );
        let cfg = baseline.with_ft_row_lr_mults(Some(0.5), Some(0.0));
        let lr = 0.0123_f32;
        let (wd, ft_lr) = cfg.effective(OptimGroupKind::Ft, lr);
        assert_eq!(wd, 0.25);
        assert_eq!(ft_lr, lr * 4.0);
        assert_eq!(cfg.ft_row_learning_rates(ft_lr), (ft_lr * 0.5, 0.0));
        assert_eq!(baseline.ft_row_learning_rates(ft_lr), (ft_lr, ft_lr));
        let base = shogi_features::FeatureSet::HalfKaHmMerged.spec();
        assert_eq!(cfg.uniform_ft_lr_mult(base), Some(0.5));
        assert_eq!(cfg.uniform_ft_lr_mult(base.with_ft_factorize()), None);
        let effect =
            base.with_effect_bucket_config(shogi_features::EffectBucketConfig::KINGFIXED_2X2);
        assert_eq!(cfg.uniform_ft_lr_mult(effect), Some(0.5));
        assert_eq!(cfg.uniform_ft_lr_mult(effect.with_ft_factorize()), None);
        let threat = base.with_threat_profile(shogi_features::ThreatProfile::CrossSide);
        assert_eq!(cfg.uniform_ft_lr_mult(threat), None);
        let equal = cfg.with_ft_row_lr_mults(Some(0.5), Some(0.5));
        assert_eq!(equal.uniform_ft_lr_mult(threat), Some(0.5));
        assert_eq!(equal.uniform_ft_lr_mult(threat.with_ft_factorize()), None);
        for kind in [
            OptimGroupKind::Ft,
            OptimGroupKind::L1,
            OptimGroupKind::Dense,
            OptimGroupKind::Bias,
        ] {
            assert_eq!(cfg.effective(kind, lr), baseline.effective(kind, lr));
        }
    }

    #[test]
    fn l1_override_replaces_dense_multiplier_and_inherits_dense_decay() {
        let baseline = OptimGroupConfig::resolve(
            0.125,
            None,
            Some(0.25),
            None,
            Some(4.0),
            Some(2.0),
            Some(0.75),
        );
        let cfg = baseline.with_l1_lr_mult(Some(0.5));
        let lr = 0.0123_f32;
        assert_eq!(cfg.effective(OptimGroupKind::L1, lr), (0.25, lr * 0.5));
        for kind in [
            OptimGroupKind::Ft,
            OptimGroupKind::Dense,
            OptimGroupKind::Bias,
        ] {
            assert_eq!(cfg.effective(kind, lr), baseline.effective(kind, lr));
        }
    }
}
