#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WdlBlend {
    pub lambda: f64,
    pub sample_std: f32,
    pub sample_key: u64,
    pub clamp: bool,
}

impl From<f32> for WdlBlend {
    fn from(lambda: f32) -> Self {
        Self {
            lambda: f64::from(lambda),
            ..Self::default()
        }
    }
}

impl WdlBlend {
    pub fn row_lambda(self, wdl: f32, ignore_draws: bool, row: u32) -> f32 {
        if wdl < 0.0 || (ignore_draws && wdl == 0.5) {
            return 0.0;
        }
        if self.clamp {
            jittered_lambda(self.lambda, self.sample_std, self.sample_key, row)
        } else {
            self.lambda as f32
        }
    }
}

#[inline(always)]
fn mix_bits(mut value: u64) -> u64 {
    // SplitMix64 finalizer avalanches both halves of the seed and counter.
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn step_key(seed: u64, step: u64) -> u64 {
    mix_bits(seed ^ mix_bits(step.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

#[inline(always)]
fn unit_uniform(bits: u64) -> f32 {
    // The 23-bit midpoint is exactly representable and keeps log away from zero.
    (((bits >> 41) as u32) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

#[inline(always)]
pub fn standard_normal(key: u64, row: u64) -> f32 {
    let counter = row.wrapping_mul(2);
    let u1 = unit_uniform(mix_bits(key ^ counter));
    let u2 = unit_uniform(mix_bits(key ^ (counter | 1)));
    (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
}

#[inline(always)]
pub fn jittered_lambda(lambda: f64, sample_std: f32, sample_key: u64, row: u32) -> f32 {
    let mut perturbed = lambda;
    if sample_std != 0.0 {
        // A finite f32 STD can still overflow a float when multiplied by a normal deviate.
        perturbed += sample_std as f64 * standard_normal(sample_key, row as u64) as f64;
    }
    perturbed.clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_blend_preserves_lambda_bits() {
        for base in [-0.0_f32, 0.3, 1.0] {
            let blend = WdlBlend::from(base);
            assert_eq!(blend.row_lambda(1.0, false, 3).to_bits(), base.to_bits());
        }
    }

    #[test]
    fn sample_jitter_is_deterministic_with_requested_mean_and_deviation() {
        let blend = WdlBlend {
            lambda: 0.5,
            sample_std: 0.07,
            sample_key: step_key(101, 130),
            clamp: true,
        };
        let count = 65_536;
        let (mut sum, mut sum_squared) = (0.0, 0.0);
        for row in 0..count {
            let lambda = f64::from(blend.row_lambda(1.0, false, row));
            assert!(lambda > 0.0 && lambda < 1.0);
            sum += lambda;
            sum_squared += lambda * lambda;
        }
        let mean = sum / f64::from(count);
        let std = (sum_squared / f64::from(count) - mean * mean).sqrt();
        assert!((mean - blend.lambda).abs() < 0.001);
        assert!((std - f64::from(blend.sample_std)).abs() < 0.001);

        let first = blend.row_lambda(1.0, false, 42);
        let replay = WdlBlend {
            sample_key: step_key(101, 130),
            ..blend
        };
        assert_eq!(first.to_bits(), replay.row_lambda(1.0, false, 42).to_bits());
        let next_step = WdlBlend {
            sample_key: step_key(101, 131),
            ..blend
        };
        assert_ne!(
            first.to_bits(),
            next_step.row_lambda(1.0, false, 42).to_bits()
        );
        assert_ne!(first.to_bits(), blend.row_lambda(1.0, false, 43).to_bits());
    }

    #[test]
    fn clamps_once_after_summing_and_preserves_score_only_rows() {
        let key = step_key(27, 44);
        let below = WdlBlend {
            lambda: -0.2,
            clamp: true,
            ..WdlBlend::default()
        };
        let above = WdlBlend {
            lambda: 1.2,
            ..below
        };
        assert_eq!(below.row_lambda(1.0, false, 0), 0.0);
        assert_eq!(above.row_lambda(1.0, false, 0), 1.0);

        let blend = WdlBlend {
            sample_std: 0.4,
            sample_key: key,
            ..below
        };
        let row = (0..128)
            .find(|&row| (0.6..1.5).contains(&standard_normal(key, u64::from(row))))
            .unwrap();
        let expected = (blend.lambda
            + f64::from(blend.sample_std) * f64::from(standard_normal(key, u64::from(row))))
            as f32;
        assert_eq!(blend.row_lambda(1.0, false, row), expected);
        assert_eq!(blend.row_lambda(-1.0, false, row), 0.0);
        assert_eq!(blend.row_lambda(0.5, true, row), 0.0);
        assert_eq!(blend.row_lambda(0.5, false, row), expected);
    }
}
