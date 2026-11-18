use std::io;

use serde::{Deserialize, Serialize};

/// Training-only sampling probabilities. Teacher IDs are stream ordinals, independent of paths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampling {
    /// Runtime logging opt-in, excluded from sampling decisions and recipe identity.
    #[serde(skip)]
    pub log_enabled: bool,
    pub keep_rate: f64,
    #[serde(skip_serializing_if = "is_one")]
    pub tree_sample_accept: f64,
    pub min_ply: Option<u16>,
    pub ply_keep_curve: Vec<(f64, f64)>,
    pub bucket_keep_rates: Option<Vec<f64>>,
    pub skip_seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            log_enabled: false,
            keep_rate: 1.0,
            tree_sample_accept: 1.0,
            min_ply: None,
            ply_keep_curve: Vec::new(),
            bucket_keep_rates: None,
            skip_seed: 0,
        }
    }
}

impl Sampling {
    pub fn is_default(&self) -> bool {
        let mut effective = self.clone();
        effective.log_enabled = false;
        effective == Self::default()
    }

    pub(crate) fn settings(&self, tree_sample_eval_only: bool) -> serde_json::Value {
        let mut settings = serde_json::json!(self);
        let fields = settings
            .as_object_mut()
            .expect("sampling settings are an object");
        // At rate 1, bypassing game-record sampling only matters when a filter is configured.
        if tree_sample_eval_only
            && (self.tree_sample_accept != 1.0
                || self.keep_rate != 1.0
                || self.min_ply.is_some()
                || !self.ply_keep_curve.is_empty()
                || self.bucket_keep_rates.is_some())
        {
            fields.insert("tree_sample_accept".into(), self.tree_sample_accept.into());
        } else {
            fields.remove("tree_sample_accept");
        }
        settings
    }

    pub fn validate(
        &self,
        has_ply: bool,
        compute_bucket: bool,
        num_buckets: usize,
    ) -> io::Result<()> {
        let error = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if !valid_rate(self.keep_rate) || self.keep_rate == 0.0 {
            return Err(error("--keep-rate must be finite and in (0, 1]"));
        }
        if !valid_rate(self.tree_sample_accept) {
            return Err(error("--tree-sample-accept must be finite and in [0, 1]"));
        }
        if !has_ply && (self.min_ply.is_some() || !self.ply_keep_curve.is_empty()) {
            return Err(error(
                "--min-ply and --ply-keep-curve require PSV; HCPE has no game ply",
            ));
        }
        if self
            .ply_keep_curve
            .iter()
            .any(|&(x, y)| !x.is_finite() || x < 0.0 || !valid_rate(y))
            || self.ply_keep_curve.windows(2).any(|p| p[0].0 >= p[1].0)
        {
            return Err(error(
                "--ply-keep-curve needs increasing finite nonnegative plies and probabilities in [0, 1]",
            ));
        }
        if let Some(rates) = &self.bucket_keep_rates {
            if !compute_bucket {
                return Err(error(
                    "--bucket-keep-rates requires a LayerStack bucket path",
                ));
            }
            if rates.len() != num_buckets || rates.iter().any(|&r| !valid_rate(r)) {
                return Err(error(
                    "--bucket-keep-rates must contain one finite probability in [0, 1] per bucket",
                ));
            }
        }
        Ok(())
    }

    pub fn ply_rate(&self, ply: u16) -> f64 {
        if self.min_ply.is_some_and(|min| ply <= min) {
            return 0.0;
        }
        let x = f64::from(ply);
        let Some(&(first_x, first_y)) = self.ply_keep_curve.first() else {
            return 1.0;
        };
        if x <= first_x {
            return first_y;
        }
        for pair in self.ply_keep_curve.windows(2) {
            let [(a, ra), (b, rb)] = pair else {
                unreachable!()
            };
            if x <= *b {
                return ra + (rb - ra) * (x - a) / (b - a);
            }
        }
        self.ply_keep_curve.last().unwrap().1
    }

    pub fn accepts(&self, teacher: u64, epoch: u64, record: u64, probability: f64) -> bool {
        if probability >= 1.0 {
            return true;
        }
        if probability <= 0.0 {
            return false;
        }
        let hash = mix(mix(mix(mix(self.skip_seed) ^ teacher) ^ epoch) ^ record);
        let uniform = (hash >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        uniform < probability
    }
}

fn is_one(rate: &f64) -> bool {
    *rate == 1.0
}

pub(crate) fn valid_rate(rate: f64) -> bool {
    rate.is_finite() && (0.0..=1.0).contains(&rate)
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

pub fn parse_curve(value: &str) -> Result<Vec<(f64, f64)>, String> {
    let points: Result<Vec<_>, String> = value
        .split(',')
        .map(|point| {
            let (x, y) = point.split_once(':').ok_or("expected ply:probability")?;
            Ok((
                x.parse().map_err(|_| "invalid ply")?,
                y.parse().map_err(|_| "invalid probability")?,
            ))
        })
        .collect();
    let sampling = Sampling {
        ply_keep_curve: points?,
        ..Sampling::default()
    };
    sampling
        .validate(true, false, 1)
        .map_err(|e| e.to_string())?;
    Ok(sampling.ply_keep_curve)
}

#[derive(Clone, Debug)]
pub struct PlyKeepCurve(pub Vec<(f64, f64)>);

impl std::str::FromStr for PlyKeepCurve {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_curve(value).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_default_accepts_every_record() {
        let sampling = Sampling::default();
        for record in 0..1000 {
            assert!(sampling.accepts(
                0,
                0,
                record,
                sampling.keep_rate * sampling.ply_rate(record as u16)
            ));
        }
    }

    #[test]
    fn sampling_default_serialization_preserves_loader_settings_bytes() {
        let legacy = r#"{"keep_rate":1.0,"min_ply":null,"ply_keep_curve":[],"bucket_keep_rates":null,"skip_seed":0}"#;
        let sampling: Sampling = serde_json::from_str(legacy).unwrap();
        assert_eq!(sampling.tree_sample_accept, 1.0);
        assert_eq!(serde_json::to_string(&sampling).unwrap(), legacy);
    }

    #[test]
    fn sampling_settings_record_tree_accept_for_each_game_filter() {
        let unfiltered = Sampling {
            log_enabled: true,
            skip_seed: 42,
            ..Sampling::default()
        };
        assert!(
            unfiltered
                .settings(true)
                .get("tree_sample_accept")
                .is_none()
        );
        for sampling in [
            Sampling {
                keep_rate: 0.5,
                ..Sampling::default()
            },
            Sampling {
                min_ply: Some(0),
                ..Sampling::default()
            },
            Sampling {
                ply_keep_curve: vec![(0.0, 0.5)],
                ..Sampling::default()
            },
            Sampling {
                bucket_keep_rates: Some(vec![0.5; 9]),
                ..Sampling::default()
            },
        ] {
            assert_eq!(sampling.settings(true)["tree_sample_accept"], 1.0);
            assert!(sampling.settings(false).get("tree_sample_accept").is_none());
        }
        let custom = Sampling {
            tree_sample_accept: 0.25,
            ..Sampling::default()
        };
        assert_eq!(custom.settings(true)["tree_sample_accept"], 0.25);
        assert!(custom.settings(false).get("tree_sample_accept").is_none());
    }

    #[test]
    fn sampling_validates_tree_sample_accept() {
        for rate in [0.0, 0.25, 1.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            let sampling = Sampling {
                tree_sample_accept: rate,
                ..Sampling::default()
            };
            assert_eq!(sampling.validate(true, false, 1).is_ok(), valid_rate(rate));
        }
    }

    #[test]
    fn sampling_is_statistical_and_reproducible() {
        let sampling = Sampling {
            skip_seed: 1234,
            ..Sampling::default()
        };
        let decisions = |teacher, epoch| {
            (0..100_000)
                .map(|record| sampling.accepts(teacher, epoch, record, 0.25))
                .collect::<Vec<_>>()
        };
        let first = decisions(0, 0);
        assert_eq!(first, decisions(0, 0));
        let parallel = std::thread::scope(|scope| {
            let even = scope.spawn(|| {
                (0..100_000)
                    .step_by(2)
                    .map(|i| sampling.accepts(0, 0, i, 0.25))
                    .collect::<Vec<_>>()
            });
            let odd = scope.spawn(|| {
                (1..100_000)
                    .step_by(2)
                    .map(|i| sampling.accepts(0, 0, i, 0.25))
                    .collect::<Vec<_>>()
            });
            even.join()
                .unwrap()
                .into_iter()
                .zip(odd.join().unwrap())
                .flat_map(|(a, b)| [a, b])
                .collect::<Vec<_>>()
        });
        assert_eq!(first, parallel);
        assert_ne!(first, decisions(0, 1));
        assert_ne!(first, decisions(1, 0));
        let kept = first.iter().filter(|&&keep| keep).count();
        assert!((24_300..25_700).contains(&kept), "kept={kept}");
        let changed = Sampling {
            skip_seed: 1235,
            ..sampling.clone()
        };
        assert!(
            (0..100).any(|i| sampling.accepts(0, 0, i, 0.25) != changed.accepts(0, 0, i, 0.25))
        );
        assert!(!sampling.accepts(0, 0, 0, 0.0));
    }

    #[test]
    fn sampling_ply_interpolation_and_limits() {
        let sampling = Sampling {
            min_ply: Some(18),
            ply_keep_curve: parse_curve("0:0.025,22:0.05,25.5:0.2,29.5:0.8,32:1").unwrap(),
            ..Sampling::default()
        };
        assert_eq!(sampling.ply_rate(18), 0.0);
        assert!((sampling.ply_rate(19) - (0.025 + 0.025 * 19.0 / 22.0)).abs() < 1e-12);
        assert!((sampling.ply_rate(27) - 0.425).abs() < 1e-12);
        assert_eq!(sampling.ply_rate(32), 1.0);
        assert_eq!(sampling.ply_rate(100), 1.0);
        let sampling = Sampling {
            ply_keep_curve: parse_curve("10:0.3").unwrap(),
            ..Sampling::default()
        };
        assert_eq!(sampling.ply_rate(0), 0.3);
        assert_eq!(sampling.ply_rate(100), 0.3);
        for invalid in [
            "", "0:1,0:0", "1:0,0:1", "NaN:0.1", "0:NaN", "-1:1", "0:1.1",
        ] {
            assert!(parse_curve(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn sampling_rejects_invalid_formats_buckets_and_rates() {
        for sampling in [
            Sampling {
                min_ply: Some(0),
                ..Sampling::default()
            },
            Sampling {
                ply_keep_curve: vec![(0.0, 1.0)],
                ..Sampling::default()
            },
        ] {
            assert!(
                sampling
                    .validate(false, true, 9)
                    .unwrap_err()
                    .to_string()
                    .contains("HCPE")
            );
        }
        let sampling = Sampling {
            bucket_keep_rates: Some(vec![1.0; 9]),
            ..Sampling::default()
        };
        assert!(sampling.validate(true, true, 8).is_err());
        assert!(sampling.validate(true, false, 9).is_err());
        sampling.validate(false, true, 9).unwrap();
        for rate in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(
                Sampling {
                    keep_rate: rate,
                    ..Sampling::default()
                }
                .validate(true, false, 1)
                .is_err()
            );
        }
        for rate in [-0.1, 1.1, f64::NAN] {
            assert!(
                Sampling {
                    bucket_keep_rates: Some(vec![rate]),
                    ..Sampling::default()
                }
                .validate(true, true, 1)
                .is_err()
            );
        }
    }
}
