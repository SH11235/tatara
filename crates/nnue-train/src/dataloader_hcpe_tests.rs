//! `sample.hcpe` は `sample.psv` と同じ 100 局面を同じ順序で HCPE にしたもの。
use super::*;
use shogi_features::FeatureSet;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shogi-format/tests/data")
        .join(name)
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nnue-train-hcpe-{}-{name}", std::process::id()))
}

/// score, wdl, stm indices, nstm indices, bucket。
type PositionRow = (f32, f32, Vec<i32>, Vec<i32>, i32);

/// 1 worker・shuffle なしで `batches` 個の batch を読み、局面ごとの内容を平らに並べる。
fn loader_contents(
    path: &Path,
    batch_size: usize,
    batches: usize,
    end_offset: u64,
    score_source: Option<ScoreSource<&Path>>,
) -> Vec<PositionRow> {
    shuffled_loader_contents(path, batch_size, batches, end_offset, score_source, 0, None)
}

/// `shuffle_seed` が `Some` なら window を shuffle する。
fn shuffled_loader_contents(
    path: &Path,
    batch_size: usize,
    batches: usize,
    end_offset: u64,
    score_source: Option<ScoreSource<&Path>>,
    buffer_mib: usize,
    shuffle_seed: Option<u64>,
) -> Vec<PositionRow> {
    let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
        path,
        batch_size,
        None,
        None,
        1,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        end_offset,
        false,
        score_source,
        buffer_mib,
        shuffle_seed.is_some(),
        shuffle_seed.unwrap_or(0),
    )
    .unwrap();
    let mut out = Vec::new();
    for _ in 0..batches {
        let (batch, buckets) = loader.next_batch().unwrap().unwrap();
        for (i, &bucket) in buckets.iter().enumerate().take(batch.n_positions) {
            let row = i * batch.max_active;
            let nnz = batch.nnz[i] as usize;
            out.push((
                batch.score[i],
                batch.wdl[i],
                batch.stm_indices[row..row + nnz].to_vec(),
                batch.nstm_indices[row..row + nnz].to_vec(),
                bucket,
            ));
        }
        loader.recycle((batch, buckets));
    }
    out
}

#[test]
fn hcpe_training_stream_matches_psv_of_the_same_positions() {
    // 100 局面を 25 局面 × 6 batch で読み、epoch の折り返しも含めて比較する。
    let psv = loader_contents(&fixture("sample.psv"), 25, 6, 100 * PSV_RECORD_BYTES, None);
    let hcpe = loader_contents(&fixture("sample.hcpe"), 25, 6, 100 * 38, None);
    assert_eq!(psv.len(), 150);
    assert_eq!(psv, hcpe);
}

#[test]
fn hcpe_shuffle_window_matches_psv_with_the_same_seed() {
    // 窓の局面数は形式によらず 40 byte 単位なので、同じ seed なら同じ並びになる。
    let read =
        |name: &str, end| shuffled_loader_contents(&fixture(name), 25, 8, end, None, 1, Some(7));
    let psv = read("sample.psv", 100 * PSV_RECORD_BYTES);
    let hcpe = read("sample.hcpe", 100 * 38);
    assert_eq!(psv, hcpe);
    let sequential = loader_contents(&fixture("sample.hcpe"), 25, 8, 100 * 38, None);
    assert_ne!(hcpe, sequential, "the window must actually be shuffled");
}

#[test]
fn hcpe_training_range_counts_records_of_38_bytes() {
    let psv = loader_contents(&fixture("sample.psv"), 10, 2, 10 * PSV_RECORD_BYTES, None);
    let hcpe = loader_contents(&fixture("sample.hcpe"), 10, 2, 10 * 38, None);
    assert_eq!(psv, hcpe);
    let Err(error) = BucketedPrefetchedLoader::spawn(
        &fixture("sample.hcpe"),
        10,
        None,
        None,
        1,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        10 * PSV_RECORD_BYTES,
        false,
    ) else {
        panic!("a PSV-aligned end offset is not an HCPE record boundary");
    };
    assert!(
        error
            .to_string()
            .contains("not aligned to HCPE record size")
    );
}

#[test]
fn hcpe_score_sidecar_is_aligned_by_record_count() {
    let scores = temp_path("sidecar.scores");
    let replacement: Vec<u8> = (0..100_i16).flat_map(|i| (i * 7).to_le_bytes()).collect();
    std::fs::write(&scores, &replacement).unwrap();
    let source = Some(ScoreSource::Sidecar {
        scores: scores.as_path(),
        mask: None,
    });
    let psv = loader_contents(
        &fixture("sample.psv"),
        25,
        4,
        100 * PSV_RECORD_BYTES,
        source,
    );
    let hcpe = loader_contents(&fixture("sample.hcpe"), 25, 4, 100 * 38, source);
    let _ = std::fs::remove_file(&scores);
    assert_eq!(psv, hcpe);
    let expected: Vec<f32> = (0..100_i16).map(|i| f32::from(i * 7)).collect();
    let got: Vec<f32> = hcpe.iter().map(|row| row.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn hcpe_rejects_dual_label_scores() {
    let Err(error) = PsvEpochReader::new_range(
        &fixture("sample.hcpe"),
        0,
        100 * 38,
        None,
        None,
        Some(ScoreSource::DualLabel(DualLabelMode::All)),
    ) else {
        panic!("dual-label needs the PSV reserved fields");
    };
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(
        error
            .to_string()
            .contains("--dual-label-psv requires PSV input")
    );
}

#[test]
fn corrupt_hcpe_record_stops_the_loader() {
    let path = temp_path("corrupt.hcpe");
    let mut bytes = std::fs::read(fixture("sample.hcpe")).unwrap();
    // 先頭 record の HuffmanCodedPos を全 bit 1 にすると玉の升が 81 以上になる。
    bytes[..32].fill(0xff);
    std::fs::write(&path, &bytes).unwrap();
    let mut loader = BucketedPrefetchedLoader::spawn(
        &path,
        10,
        None,
        None,
        1,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        bytes.len() as u64,
        false,
    )
    .unwrap();
    let Err(error) = loader.next_batch() else {
        panic!("corrupt record must fail");
    };
    drop(loader);
    let _ = std::fs::remove_file(&path);
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(
        error.to_string().contains(path.to_str().unwrap()),
        "{error}"
    );
}

#[cfg(unix)]
#[test]
fn symlink_whose_name_and_target_disagree_on_the_format_is_rejected() {
    let psv_alias = temp_path("alias.psv");
    let hcpe_alias = temp_path("alias.hcpe");
    let _ = std::fs::remove_file(&psv_alias);
    let _ = std::fs::remove_file(&hcpe_alias);
    std::os::unix::fs::symlink(fixture("sample.hcpe"), &psv_alias).unwrap();
    std::os::unix::fs::symlink(fixture("sample.hcpe"), &hcpe_alias).unwrap();
    let mismatch = TeacherFormat::for_path(&psv_alias);
    let consistent = TeacherFormat::for_path(&hcpe_alias);
    let _ = std::fs::remove_file(&psv_alias);
    let _ = std::fs::remove_file(&hcpe_alias);
    let error = mismatch.expect_err("a .psv name over an HCPE file is ambiguous");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(consistent.unwrap(), TeacherFormat::Hcpe);
}
