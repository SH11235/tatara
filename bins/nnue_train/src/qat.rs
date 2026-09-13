//! Dense-only quantization contract. FT accumulation remains floating point.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum QatMode {
    #[default]
    Off,
    Dense,
}

impl QatMode {
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
            (vec!["nnue-train", "layerstack"], None),
            (
                vec!["nnue-train", "layerstack", "--qat", "off"],
                Some(QatMode::Off),
            ),
            (
                vec!["nnue-train", "layerstack", "--qat", "dense"],
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
        assert!(Cli::try_parse_from(["nnue-train", "layerstack", "--qat", "full"]).is_err());
    }
}
