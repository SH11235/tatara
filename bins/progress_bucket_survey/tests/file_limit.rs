#[cfg(target_os = "linux")]
#[test]
fn survey_scans_more_files_than_the_descriptor_limit() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/shogi-format/tests/data/sample.psv");
    let mut command = std::process::Command::new("bash");
    command.args(["-c", "ulimit -n 32 && exec \"$@\"", "survey"]);
    command.arg(env!("CARGO_BIN_EXE_progress-bucket-survey"));
    command.args(["--bucket-spec", "handmajor3", "--samples", "64", "--tsv"]);
    for _ in 0..64 {
        command.arg("--data").arg(&data);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("total\tall\tall\t64\t1.000000000")
    );
}
