//! Dense-only quantization contract. FT accumulation remains floating point.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum QatMode {
    #[default]
    Off,
    Dense,
}

impl QatMode {
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn active(self, warmup: Option<u64>, completed_steps: u64) -> bool {
        self == Self::Dense && completed_steps >= warmup.unwrap_or(0)
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Dense => "dense",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{ArchCommand, Cli};
    use clap::Parser;

    #[test]
    fn qat_cli_defaults_to_inheritance_and_accepts_explicit_switches() {
        for (args, expected) in [
            (
                vec![
                    "nnue-train",
                    "layerstack",
                    "--layerstack-head",
                    "split-skip",
                ],
                None,
            ),
            (
                vec![
                    "nnue-train",
                    "layerstack",
                    "--layerstack-head",
                    "split-skip",
                    "--qat",
                    "off",
                ],
                Some(QatMode::Off),
            ),
            (
                vec![
                    "nnue-train",
                    "layerstack",
                    "--layerstack-head",
                    "split-skip",
                    "--qat",
                    "dense",
                ],
                Some(QatMode::Dense),
            ),
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            let ArchCommand::LayerStack(args) = cli.arch else {
                panic!("LayerStack expected")
            };
            assert_eq!(args.qat, expected);
        }
        assert!(Cli::try_parse_from(["nnue-train", "simple", "--qat", "dense"]).is_err());
        assert!(
            Cli::try_parse_from([
                "nnue-train",
                "layerstack",
                "--layerstack-head",
                "split-skip",
                "--qat",
                "full"
            ])
            .is_err()
        );
    }
}

#[cfg(any(feature = "gpu", test))]
pub(crate) fn resolve_warmup(
    requested: Option<u64>,
    saved: Option<u64>,
    mode: QatMode,
    resume: bool,
    forward_only: bool,
) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    if forward_only && resume {
        return Ok(saved);
    }
    if resume && requested.is_some() && requested != saved {
        return Err("--qat-warmup-steps does not match the resumed checkpoint".into());
    }
    let warmup = if resume { saved } else { requested };
    if warmup.is_some() && mode != QatMode::Dense {
        return Err("--qat-warmup-steps requires --qat dense".into());
    }
    Ok(warmup)
}

#[cfg(test)]
mod warmup_tests {
    use super::*;
    use crate::cli::{ArchCommand, Cli};
    use clap::Parser;

    #[test]
    fn parse_and_validate_warmup() {
        for n in ["0", "2", "18446744073709551615"] {
            let cli = Cli::try_parse_from([
                "nnue-train",
                "layerstack",
                "--qat",
                "dense",
                "--qat-warmup-steps",
                n,
            ])
            .unwrap();
            let ArchCommand::LayerStack(args) = cli.arch else {
                panic!()
            };
            let steps = Some(n.parse().unwrap());
            assert_eq!(args.qat_warmup_steps, steps);
            assert_eq!(
                resolve_warmup(steps, None, args.qat.unwrap(), false, false).unwrap(),
                steps
            );
        }
        for arch in ["simple", "layerstack"] {
            assert!(
                Cli::try_parse_from(["nnue-train", arch, "--qat-warmup-steps", "-1",]).is_err()
            );
        }
        assert!(Cli::try_parse_from(["nnue-train", "simple", "--qat-warmup-steps", "2"]).is_err());
        assert!(resolve_warmup(Some(0), None, QatMode::Off, false, false).is_err());
        assert!(resolve_warmup(Some(2), None, QatMode::Off, false, false).is_err());
    }

    #[test]
    fn resume_inherits_and_rejects_changed_warmup() {
        for saved in [None, Some(0), Some(2)] {
            assert_eq!(
                resolve_warmup(None, saved, QatMode::Dense, true, false).unwrap(),
                saved
            );
            assert_eq!(
                resolve_warmup(saved, saved, QatMode::Dense, true, false).unwrap(),
                saved
            );
            assert!(resolve_warmup(Some(3), saved, QatMode::Dense, true, false).is_err());
            assert_eq!(
                resolve_warmup(Some(3), saved, QatMode::Off, true, true).unwrap(),
                saved
            );
        }
        assert!(resolve_warmup(None, Some(2), QatMode::Off, true, false).is_err());
        assert_eq!(
            resolve_warmup(Some(3), Some(2), QatMode::Dense, false, false).unwrap(),
            Some(3)
        );
    }

    #[test]
    fn switch_uses_completed_optimizer_updates() {
        for step in 0..4 {
            assert_eq!(QatMode::Dense.active(Some(2), step), step >= 2);
            assert!(QatMode::Dense.active(None, step));
            assert!(QatMode::Dense.active(Some(0), step));
            assert!(!QatMode::Off.active(Some(2), step));
        }
    }
}
