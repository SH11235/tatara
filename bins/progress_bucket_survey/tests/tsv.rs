use std::{path::Path, process::Command};

#[test]
fn tsv_distinguishes_a_file_named_total_from_the_aggregate() {
    let root =
        std::env::temp_dir().join(format!("progress-bucket-survey-tsv-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/shogi-format/tests/data/sample.psv");
    std::fs::copy(&fixture, root.join("total")).unwrap();
    std::fs::copy(&fixture, root.join("teacher.psv")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_progress-bucket-survey"))
        .current_dir(&root)
        .args([
            "--data",
            "total",
            "--data",
            "teacher.psv",
            "--bucket-spec",
            "handmajor3",
            "--samples",
            "4",
            "--per-pack",
            "--tsv",
        ])
        .output()
        .unwrap();
    std::fs::remove_dir_all(&root).unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        output.lines().next(),
        Some("file\tbucket_a\tbucket_b\tcount\tfraction\tscope")
    );
    let totals: Vec<_> = output
        .lines()
        .filter(|line| line.starts_with("total\tall\tall\t"))
        .collect();
    assert_eq!(
        totals,
        [
            "total\tall\tall\t2\t1.000000000\tpack",
            "total\tall\tall\t4\t1.000000000\taggregate",
        ]
    );
}
