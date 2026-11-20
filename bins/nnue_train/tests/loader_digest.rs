use std::{path::Path, process::Command};

#[test]
fn split_keep_bucket_options_preserve_loader_digest_and_monitoring() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/shogi-format/tests/data/sample.psv");
    let run = |before: &[&str], after: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_nnue-train"))
            .args(before)
            .args([
                "loader-digest",
                "--batches",
                "2",
                "--batch-size",
                "16",
                "--data",
            ])
            .arg(&data)
            .args(after)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output
    };
    let rates = ["--bucket-keep-rates", "1,1,1"];
    let spec_and_monitor = ["--keep-bucket-spec", "handmajor3", "--monitor-keep-buckets"];
    let together = run(&[], &[rates.as_slice(), &spec_and_monitor].concat());
    for (before, after) in [
        (rates.as_slice(), spec_and_monitor.as_slice()),
        (spec_and_monitor.as_slice(), rates.as_slice()),
    ] {
        let split = run(before, after);
        assert_eq!(split.stdout, together.stdout);
        assert_eq!(split.stderr, together.stderr);
    }
}

#[test]
fn mixed_loader_digest_distinguishes_inherited_bucket_rates_from_overrides() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/shogi-format/tests/data/sample.psv");
    let digest = |rates, clamp, override_args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_nnue-train"))
            .args(["loader-digest", "--data"])
            .arg(&data)
            .arg("--mix-data")
            .arg(&data)
            .args([
                "--batch-size",
                "16",
                "--batches",
                "8",
                "--keep-bucket-spec",
                "handmajor3",
                "--bucket-keep-rates",
                rates,
                "--mix-rate",
                "0.2",
                "--mix-score-drop-abs",
                "32000",
                "--mix-score-clamp-abs",
                clamp,
            ])
            .args(override_args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .find(|line| line.starts_with("digest="))
            .expect("digest line")
            .to_owned()
    };

    let inherited = digest("0.5,0.5,0.5", "5", &[]);
    assert_eq!(
        inherited,
        digest("0.5,0.5,0.5", "5", &["--mix-keep-rate", "inherit"])
    );
    assert_ne!(
        inherited,
        digest("0.5,0.5,0.5", "5", &["--mix-keep-rate", "1"])
    );
    assert_ne!(inherited, digest("0.5,0.5,0.5", "1", &[]));
    assert_eq!(
        digest("1,1,1", "5", &[]),
        digest("1,1,1", "5", &["--mix-keep-rate", "1"])
    );
}
