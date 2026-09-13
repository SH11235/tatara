use nnue_train::experiment::TrainerBuild;

#[cfg(any(feature = "gpu", test))]
pub(crate) const BUILD_COMMIT: &str = env!("TATARA_BUILD_COMMIT");

pub(crate) fn trainer_build() -> TrainerBuild {
    TrainerBuild {
        commit: match env!("TATARA_BUILD_FULL_COMMIT") {
            "unknown" => None,
            commit => Some(commit.into()),
        },
        dirty: match env!("TATARA_BUILD_DIRTY") {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        backend: if cfg!(feature = "native") {
            "native"
        } else if cfg!(feature = "oxide-parity") {
            "oxide-parity"
        } else if cfg!(feature = "oxide") {
            "oxide"
        } else {
            "cpu-only"
        }
        .into(),
        rustc: env!("TATARA_BUILD_RUSTC").into(),
        target: env!("TATARA_BUILD_TARGET").into(),
        profile: env!("TATARA_BUILD_PROFILE").into(),
        opt_level: env!("TATARA_BUILD_OPT_LEVEL").into(),
        debug: env!("TATARA_BUILD_DEBUG").into(),
    }
}

#[cfg(any(feature = "gpu", test))]
pub(crate) fn experiment_commit() -> Option<String> {
    let build = trainer_build();
    match (build.commit, build.dirty) {
        (Some(commit), Some(false)) => Some(commit),
        (Some(commit), Some(true)) => Some(format!("{commit}-dirty")),
        _ => None,
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn runtime_backend() -> &'static str {
    #[cfg(any(feature = "native", feature = "oxide-parity"))]
    if crate::kernel_module::native_backend_requested() {
        return "native";
    }
    "oxide"
}

#[cfg(test)]
mod tests {
    #[test]
    fn legacy_and_full_identity_agree() {
        let build = super::trainer_build();
        match (build.commit, build.dirty) {
            (Some(full), Some(dirty)) => {
                let short = super::BUILD_COMMIT.trim_end_matches("-dirty");
                assert!(full.starts_with(short));
                assert_eq!(super::BUILD_COMMIT.ends_with("-dirty"), dirty);
                assert!(super::experiment_commit().unwrap().starts_with(&full));
            }
            _ => assert_eq!(super::BUILD_COMMIT, "unknown"),
        }
    }
}
