use gpu_kernels::pointwise::wdl_jitter::{WdlBlend, standard_normal, step_key};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct WdlJitter {
    pub sample_std: f32,
    pub batch_std: f32,
    pub batch_decay: f64,
    pub cycle: bool,
    pub seed: u64,
}

impl WdlJitter {
    pub fn enabled(self) -> bool {
        self.sample_std > 0.0 || self.batch_std > 0.0
    }

    pub(crate) fn is_disabled(&self) -> bool {
        !self.enabled()
    }

    pub fn next(self, base: f32, cycle_factor: f32, step: u64, batch_state: &mut f64) -> WdlBlend {
        let key = step_key(self.seed, step);
        if self.batch_std > 0.0 {
            // Sample rows come from u32, so the batch counter has a disjoint pair of uniforms.
            let innovation = f64::from(standard_normal(key, u64::MAX));
            let scale = f64::from(self.batch_std) * (1.0 - self.batch_decay.powi(2)).sqrt();
            *batch_state = self.batch_decay * *batch_state + scale * innovation;
        }
        let factor = if self.cycle { cycle_factor } else { 1.0 };
        let batch_offset = if self.batch_std > 0.0 {
            *batch_state * f64::from(factor)
        } else {
            0.0
        };
        WdlBlend {
            lambda: f64::from(base) + batch_offset,
            sample_std: self.sample_std * factor,
            sample_key: key,
            clamp: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader_state::{LoaderState, WdlJitterState};

    #[test]
    fn zero_decay_depends_only_on_seed_and_step() {
        let jitter = WdlJitter {
            batch_std: 0.2,
            seed: 173,
            ..WdlJitter::default()
        };
        let mut state = 15.0;
        let mut other = -9.0;
        let first = jitter.next(0.5, 1.0, 10, &mut state);
        assert_eq!(first, jitter.next(0.5, 1.0, 10, &mut other));
        assert_eq!(state, other);
        assert_ne!(first.lambda, jitter.next(0.5, 1.0, 11, &mut state).lambda);
    }

    #[test]
    fn batch_jitter_has_stationary_deviation_and_correlation() {
        let jitter = WdlJitter {
            batch_std: 0.2,
            batch_decay: 0.85,
            seed: 57311,
            ..WdlJitter::default()
        };
        let mut state = 0.0;
        let (mut sum, mut sum_squared, mut sum_lagged) = (0.0, 0.0, 0.0);
        let count = 80_000;
        for step in 0..count + 1000 {
            let previous = state;
            jitter.next(0.5, 1.0, step, &mut state);
            if step >= 1000 {
                sum += state;
                sum_squared += state * state;
                sum_lagged += state * previous;
            }
        }
        let mean = sum / count as f64;
        let variance = sum_squared / count as f64 - mean * mean;
        let correlation = (sum_lagged / count as f64 - mean * mean) / variance;
        assert!(mean.abs() < 0.006);
        assert!((variance.sqrt() - f64::from(jitter.batch_std)).abs() < 0.006);
        assert!((correlation - jitter.batch_decay).abs() < 0.015);
    }

    #[test]
    fn checkpoint_round_trip_replays_batch_and_sample_jitter() {
        let jitter = WdlJitter {
            sample_std: 0.15,
            batch_std: 0.2,
            batch_decay: 0.9,
            seed: 195,
            cycle: true,
        };
        let mut uninterrupted = 0.0;
        let mut replay = 0.0;
        for step in 0..50 {
            assert_eq!(
                jitter.next(0.5, 0.25, step, &mut uninterrupted),
                jitter.next(0.5, 0.25, step, &mut replay),
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("net.ckpt");
        let mut metadata = LoaderState {
            version: 1,
            superbatch: 1,
            teachers: Vec::new(),
            settings: serde_json::json!({}),
            wdl_jitter_state: None,
        };
        metadata.save(&path).unwrap();
        assert!(
            LoaderState::load(&path)
                .unwrap()
                .unwrap()
                .wdl_jitter_state
                .is_none()
        );
        metadata.wdl_jitter_state = Some(WdlJitterState {
            completed_steps: 50,
            batch_state_bits: uninterrupted.to_bits(),
        });
        metadata.save(&path).unwrap();
        let saved = LoaderState::load(&path)
            .unwrap()
            .unwrap()
            .wdl_jitter_state
            .unwrap();
        let mut restored = f64::from_bits(saved.batch_state_bits);
        let muted = jitter.next(0.5, 0.0, saved.completed_steps, &mut uninterrupted);
        let full = jitter.next(0.5, 1.0, saved.completed_steps, &mut restored);
        assert_eq!(muted.lambda, 0.5);
        assert_eq!(muted.sample_std, 0.0);
        assert_eq!(full.sample_std, jitter.sample_std);
        assert_eq!(uninterrupted.to_bits(), restored.to_bits());
        for step in saved.completed_steps + 1..100 {
            assert_eq!(
                jitter.next(0.5, 0.25, step, &mut uninterrupted),
                jitter.next(0.5, 0.25, step, &mut restored),
            );
        }
    }
}
