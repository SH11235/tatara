use std::fs;
use std::io;
use std::path::Path;

use nnue_train::dataloader::{
    Batch, BucketMode, BucketedLoaderOptions, BucketedPrefetchedLoader, DualLabelMode,
    LabelOptions, PrefetchedLoader, PsvFileLoader, ScoreSource, TeacherPosition,
};
use nnue_train::validation::HeldoutSet;
use shogi_features::FeatureSet;

const SAMPLE: &[u8] = include_bytes!("../../shogi-format/tests/data/sample.psv");
const HCPE: [u8; 38] = [
    0x8d, 0xb8, 0x09, 0x15, 0x06, 0x00, 0x00, 0x80, 0x85, 0xf0, 0x6e, 0x4a, 0xfc, 0x62, 0x2b, 0xe1,
    0x45, 0x89, 0xe3, 0x13, 0xfe, 0x5e, 0x61, 0xa2, 0x04, 0x03, 0x00, 0x60, 0x8c, 0x0f, 0x67, 0xbd,
    0xe9, 0x07, 0x2b, 0x1a, 0x02, 0x00,
];

fn training_loader(path: &Path, buffer_mib: usize) -> io::Result<BucketedPrefetchedLoader> {
    training_loader_with_source(path, buffer_mib, None)
}

fn training_loader_with_source(
    path: &Path,
    buffer_mib: usize,
    score_source: Option<ScoreSource<&Path>>,
) -> io::Result<BucketedPrefetchedLoader> {
    BucketedPrefetchedLoader::spawn_with_score_sources(
        &[path],
        4,
        None,
        None,
        1,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        fs::metadata(path)?.len(),
        false,
        score_source,
        buffer_mib,
        false,
        0,
        Default::default(),
    )
}

#[test]
fn sidecar_and_dual_labels_keep_feature_payload_with_windowed_training() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("teacher.psv");
    let dual_path = dir.path().join("dual.psv");
    let score_path = dir.path().join("scores.i16");
    let labels = [111_i16, -222, 333, -444];
    let mut dual_bytes = SAMPLE[..160].to_vec();
    let mut scores = Vec::new();
    for (record, label) in dual_bytes.chunks_exact_mut(40).zip(labels) {
        record[34..36].copy_from_slice(&label.to_le_bytes());
        record[39] = 0;
        scores.extend(label.to_le_bytes());
    }
    fs::write(&path, &SAMPLE[..160]).unwrap();
    fs::write(&dual_path, dual_bytes).unwrap();
    fs::write(&score_path, scores).unwrap();
    let mut finite = PsvFileLoader::new(&path).unwrap();
    let mut expected = Batch::with_capacity(4, FeatureSet::HalfKaHmMerged.spec());
    finite.fill_batch(&mut expected).unwrap();
    for (score, label) in expected.score[..4].iter_mut().zip(labels) {
        *score = f32::from(label);
    }
    for window in [0, 1] {
        for (input, source) in [
            (
                path.as_path(),
                ScoreSource::Sidecar {
                    scores: score_path.as_path(),
                    mask: None,
                },
            ),
            (
                dual_path.as_path(),
                ScoreSource::DualLabel(DualLabelMode::All),
            ),
        ] {
            let mut loader = training_loader_with_source(input, window, Some(source)).unwrap();
            for _ in 0..2 {
                let (batch, buckets) = loader.next_batch().unwrap().unwrap();
                assert_eq!(payload(&batch), payload(&expected));
                loader.recycle((batch, buckets));
            }
        }
    }
    let heldout = HeldoutSet::load_from_range_with_score_sources(
        &[&dual_path],
        40,
        160,
        1,
        None,
        None,
        3,
        &BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        9,
        Some(ScoreSource::DualLabel(DualLabelMode::All)),
    )
    .unwrap();
    assert_eq!(heldout.n_positions(), 3);
}

#[test]
fn aligned_hcpe_is_read_as_hcpe_records_and_rejected_by_psv_only_readers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("teacher.HCPE");
    // record ごとに score を変え、epoch の折り返し位置を score の並びで確かめる。
    let mut bytes = HCPE.repeat(20);
    for (index, record) in bytes.chunks_exact_mut(38).enumerate() {
        record[32..34].copy_from_slice(&(index as i16).to_le_bytes());
    }
    fs::write(&path, &bytes).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), 760);
    let heldout = HeldoutSet::load(
        &path,
        4,
        None,
        None,
        20,
        &BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        9,
    )
    .expect("HCPE heldout is supported");
    assert_eq!(heldout.n_positions(), 20);
    assert_eq!(heldout.n_batches(), 5);
    let mut expected = Batch::with_capacity(4, FeatureSet::HalfKaHmMerged.spec());
    let mut record = shogi_format::HuffmanCodedPosAndEval::default();
    record.as_bytes_mut().copy_from_slice(&HCPE);
    let board = record.decode().unwrap();
    for _ in 0..4 {
        assert!(expected.push_decoded(&board).unwrap());
    }
    let expected_scores: Vec<f32> = (0..24).map(|index| (index % 20) as f32).collect();
    for window in [0, 1] {
        // 20 HCPE records, not 19 PSV records: 6 batches of 4 wrap one epoch boundary.
        let mut loader = training_loader(&path, window).expect("HCPE training input");
        let mut scores = Vec::new();
        for chunk in expected_scores.chunks(4) {
            let (batch, buckets) = loader.next_batch().unwrap().unwrap();
            expected.score[..4].copy_from_slice(chunk);
            assert_eq!(payload(&batch), payload(&expected));
            scores.extend_from_slice(&batch.score[..4]);
            loader.recycle((batch, buckets));
        }
        assert_eq!(scores, expected_scores);
    }
    assert!(PsvFileLoader::new(&path).is_err());
    assert!(PrefetchedLoader::spawn(&path, 4, FeatureSet::HalfKaHmMerged.spec(), 1).is_err());
}

fn payload(batch: &Batch) -> Vec<u8> {
    let mut bytes = Vec::new();
    for index in 0..batch.n_positions {
        bytes.extend(batch.score[index].to_le_bytes());
        bytes.extend(batch.wdl[index].to_le_bytes());
        bytes.extend(batch.nnz[index].to_le_bytes());
        let start = index * batch.max_active;
        let end = start + batch.nnz[index] as usize;
        for feature in batch.stm_indices[start..end]
            .iter()
            .chain(&batch.nstm_indices[start..end])
        {
            bytes.extend(feature.to_le_bytes());
        }
    }
    bytes
}

#[test]
fn direct_and_windowed_training_preserve_digest_payload_and_epoch_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("teacher.psv");
    fs::write(&path, &SAMPLE[..160]).unwrap();
    let mut finite = PsvFileLoader::new(&path).unwrap();
    let mut expected = Batch::with_capacity(4, FeatureSet::HalfKaHmMerged.spec());
    assert_eq!(finite.fill_batch(&mut expected).unwrap(), 4);
    for window in [0, 1] {
        let mut loader = training_loader(&path, window).unwrap();
        for _ in 0..3 {
            let (batch, buckets) = loader
                .next_batch()
                .unwrap()
                .expect("full batch across epochs");
            assert_eq!(payload(&batch), payload(&expected));
            assert_eq!(buckets.len(), 4);
            loader.recycle((batch, buckets));
        }
    }
    let heldout = HeldoutSet::load(
        &path,
        4,
        None,
        None,
        4,
        &BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        9,
    )
    .unwrap();
    assert_eq!(heldout.n_positions(), 4);
    let mut prefetched =
        PrefetchedLoader::spawn(&path, 4, FeatureSet::HalfKaHmMerged.spec(), 1).unwrap();
    assert_eq!(
        payload(&prefetched.next_batch().unwrap().unwrap()),
        payload(&expected)
    );
    assert!(prefetched.next_batch().unwrap().is_none());
}

#[test]
fn independent_sampling_matches_routing_with_the_same_definition() {
    use nnue_train::sampling::{KeepBucket, Sampling};
    use shogi_features::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("teacher.psv");
    fs::write(&path, SAMPLE).unwrap();
    let coeff = dir.path().join("progress.bin");
    let weights: Vec<u8> = (0..SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS)
        .flat_map(|index| (((index % 29) as f64 - 11.0) * 0.02).to_le_bytes())
        .collect();
    fs::write(&coeff, &weights).unwrap();
    ShogiProgressKPAbs::load_from_bytes(&weights).unwrap();

    let spec = "ownking:2/3/7+oppking:2/3/7".parse().unwrap();
    for (mode, count, definition) in [
        (
            BucketMode::Spec(spec),
            16,
            KeepBucket::Spec {
                spec: spec.to_string(),
            },
        ),
        (
            BucketMode::ProgressKpAbs,
            8,
            KeepBucket::ProgressKpAbs {
                progress_coeff: coeff,
                num_buckets: 8,
            },
        ),
    ] {
        let open =
            |keep_bucket, monitor_keep_buckets, workers, resume: Option<&TeacherPosition>| {
                BucketedPrefetchedLoader::spawn_mixed_resumable(
                    &[&path],
                    BucketedLoaderOptions {
                        sampling: Sampling {
                            keep_rate: 0.9,
                            ply_keep_curve: vec![(0.0, 0.8), (80.0, 1.0)],
                            bucket_keep_rates: Some(
                                (0..count)
                                    .map(|b| 0.3 + b as f64 / (2 * count) as f64)
                                    .collect(),
                            ),
                            monitor_keep_buckets,
                            keep_bucket,
                            skip_seed: 123,
                            ..Default::default()
                        },
                        batch_size: 13,
                        score_drop_abs: None,
                        score_clamp_abs: None,
                        num_workers: workers,
                        bucket_mode: mode,
                        feature_set: FeatureSet::HalfKaHmMerged.spec(),
                        compute_bucket: true,
                        num_buckets: count,
                        train_end_offset: SAMPLE.len() as u64,
                        monitor_active: false,
                        labels: LabelOptions::default(),
                        teacher_shuffle_buffer_mib: 1,
                        teacher_shuffle: true,
                        teacher_shuffle_seed: 456,
                    },
                    &[],
                    0,
                    resume,
                )
                .unwrap()
            };
        let collect = |loader: &mut BucketedPrefetchedLoader, batches| {
            std::iter::repeat_with(|| {
                let slot = loader.next_batch().unwrap().unwrap();
                let batch = &slot.0;
                let rows = (0..batch.n_positions)
                    .map(|i| {
                        let start = i * batch.max_active;
                        let end = start + batch.nnz[i] as usize;
                        (
                            batch.stm_indices[start..end].to_vec(),
                            batch.nstm_indices[start..end].to_vec(),
                            batch.score[i].to_bits(),
                            batch.wdl[i].to_bits(),
                            batch.per_pos_norm[i].to_bits(),
                            slot.1[i],
                        )
                    })
                    .collect::<Vec<_>>();
                loader.recycle(slot);
                rows
            })
            .take(batches)
            .collect::<Vec<_>>()
        };
        let routing = collect(&mut open(None, false, 1, None), 10);
        for workers in [1, 3] {
            let mut independent = open(Some(definition.clone()), false, workers, None);
            assert_eq!(collect(&mut independent, 10), routing);
            assert!(independent.sampling_bucket_counts().is_none());
        }

        let mut monitored = open(Some(definition.clone()), true, 1, None);
        let mut resumed_batches = collect(&mut monitored, 1);
        while monitored.position().bucket_skip == 0 && resumed_batches.len() < 9 {
            resumed_batches.extend(collect(&mut monitored, 1));
        }
        let position = monitored.position().clone();
        assert!(position.window_offset > 0);
        assert!(position.bucket_skip > 0);
        let counts = monitored.sampling_bucket_counts().unwrap();
        assert_eq!(counts.len(), count);
        assert!(counts.iter().any(|&(read, kept)| read > kept && kept > 0));
        drop(monitored);
        let mut resumed = open(Some(definition), false, 3, Some(&position));
        let remaining = 10 - resumed_batches.len();
        resumed_batches.extend(collect(&mut resumed, remaining));
        assert_eq!(resumed_batches, routing);
    }
}

#[test]
fn independent_sampling_works_without_network_routing_buckets() {
    use nnue_train::sampling::{KeepBucket, Sampling};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("teacher.psv");
    fs::write(&path, SAMPLE).unwrap();
    let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
        &[&path],
        13,
        None,
        None,
        2,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        false,
        1,
        SAMPLE.len() as u64,
        false,
        None,
        0,
        false,
        0,
        Sampling {
            bucket_keep_rates: Some(vec![0.3, 0.7, 0.9, 1.0]),
            keep_bucket: Some(KeepBucket::Spec {
                spec: "handmajor4".into(),
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let (batch, buckets) = loader.next_batch().unwrap().unwrap();
    assert_eq!(batch.n_positions, 13);
    assert!(buckets.is_empty());
}
