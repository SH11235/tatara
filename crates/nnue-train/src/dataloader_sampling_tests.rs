use super::*;
use crate::sampling::Sampling;
use shogi_features::FeatureSet;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shogi-format/tests/data")
        .join(name)
}

fn reader(sampling: Sampling) -> PsvEpochReader {
    let path = fixture("sample.psv");
    let mut reader = PsvEpochReader::new_range(&[&path], 0, 4000, None, None, None).unwrap();
    reader.sampling = sampling;
    reader
}

fn epoch_scores(reader: &mut PsvEpochReader) -> Vec<i16> {
    let mut scores = Vec::new();
    while let Some(record) = reader.next_in_epoch().unwrap() {
        scores.push(record.score());
    }
    scores
}

#[test]
fn sampling_default_preserves_physical_stream_and_epoch_wrap() {
    let path = fixture("sample.psv");
    let mut physical = TeacherFileLoader::new_range(&[path], 0, 4000).unwrap();
    let mut expected = Vec::new();
    while let Some(record) = physical.next_record().unwrap() {
        expected.push(record.score());
    }
    let mut reader = reader(Sampling::default());
    assert_eq!(epoch_scores(&mut reader), expected);
    assert_eq!(epoch_scores(&mut reader), expected);
    assert_eq!(reader.epoch, 2);
    assert_eq!(reader.sampling_counts.read.load(Ordering::Relaxed), 200);
    assert_eq!(reader.sampling_counts.kept.load(Ordering::Relaxed), 200);
}

#[test]
fn sampling_matches_record_identity_and_can_replay_from_mid_epoch() {
    let sampling = Sampling {
        keep_rate: 0.5,
        ply_keep_curve: vec![(0.0, 0.5)],
        skip_seed: 42,
        ..Default::default()
    };
    let mut original = reader(sampling.clone());
    let path = fixture("sample.psv");
    let mut physical = TeacherFileLoader::new_range(&[&path], 0, 4000).unwrap();
    let mut expected = Vec::new();
    let mut index = 0;
    while let Some(record) = physical.next_record().unwrap() {
        if sampling.accepts(0, 0, index, 0.25) {
            expected.push(record.score());
        }
        index += 1;
    }
    assert_eq!(epoch_scores(&mut original), expected);
    let mut resumed = PsvEpochReader::new_range(&[&path], 2000, 4000, None, None, None).unwrap();
    resumed.sampling = sampling.clone();
    resumed.epoch = 3;
    let mut full = reader(sampling);
    full.epoch = 3;
    full.loader = TeacherFileLoader::new_range(&[&path], 2000, 4000).unwrap();
    full.record_index = 50;
    assert_eq!(epoch_scores(&mut resumed), epoch_scores(&mut full));
}

fn bucket_loader(sampling: Sampling, workers: usize, shuffle: bool) -> BucketedPrefetchedLoader {
    let mut source = reader(sampling.clone());
    source.bucket_sampling = Some(Arc::new(BucketSampling {
        sampling,
        counts: Arc::clone(&source.sampling_counts),
        buckets: (0..9).map(|_| SamplingCounts::default()).collect(),
        progress: Mutex::new(BucketProgress::default()),
        accepted: Mutex::new(Vec::new()),
    }));
    let counts = Arc::clone(&source.sampling_counts);
    let input = if shuffle {
        TrainingPsvReader::Windowed(WindowedPsvReader::spawn(source, 17, true, 123))
    } else {
        TrainingPsvReader::Direct(Box::new(source))
    };
    let mut loader = BucketedPrefetchedLoader::spawn_with_reader(
        8,
        workers,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        false,
        input,
        &fixture("sample.psv"),
    );
    loader.sampling_counts = Some(counts);
    loader
}

#[test]
fn sampling_bucket_and_ply_filters_reach_barren_guard() {
    let mut reader = reader(Sampling {
        min_ply: Some(u16::MAX),
        ..Default::default()
    });
    assert!(
        reader
            .next()
            .err()
            .unwrap()
            .to_string()
            .contains("5 full passes")
    );
    assert_eq!(reader.sampling_counts.read.load(Ordering::Relaxed), 500);
    for workers in [1, 4] {
        for shuffle in [false, true] {
            let mut loader = bucket_loader(
                Sampling {
                    bucket_keep_rates: Some(vec![0.0; 9]),
                    ..Default::default()
                },
                workers,
                shuffle,
            );
            assert!(
                loader
                    .next_batch()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("5 full passes")
            );
            assert_eq!(loader.sampling_counts().1, 0);
            let counts = loader.sampling_bucket_counts().unwrap();
            assert!(counts.iter().map(|c| c.0).sum::<u64>() >= 500);
            assert!(counts.iter().all(|c| c.1 == 0));
        }
    }
}

#[test]
fn sampling_bucket_rates_follow_record_identity_across_workers_and_shuffle() {
    let sampling = Sampling {
        keep_rate: 0.8,
        ply_keep_curve: vec![(0.0, 0.9)],
        bucket_keep_rates: Some(vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.2, 0.4, 0.6, 0.8]),
        skip_seed: 42,
        ..Default::default()
    };
    let mut physical = reader(Sampling::default());
    let mut expected = Vec::new();
    for epoch in 0..2 {
        while let Some(record) = physical.next_in_epoch().unwrap() {
            assert_eq!(record.epoch, epoch);
            let board = record.decode().unwrap();
            let bucket = usize::from(BucketMode::KingRank9.bucket_board(&board, 9));
            if sampling.accepts(
                0,
                epoch,
                record.record_index,
                0.8 * 0.9 * sampling.bucket_keep_rates.as_ref().unwrap()[bucket],
            ) {
                expected.push((epoch, record.record_index));
            }
        }
    }
    assert!(expected.iter().any(|&(epoch, _)| epoch == 0));
    assert!(expected.iter().any(|&(epoch, _)| epoch == 1));
    assert!(expected.len() < 200);
    for workers in [1, 4] {
        for shuffle in [false, true] {
            let mut loader = bucket_loader(sampling.clone(), workers, shuffle);
            let state = loader.bucket_sampling.as_ref().unwrap().clone();
            let mut completed = false;
            for _ in 0..1000 {
                let (batch, buckets) = loader.next_batch().unwrap().unwrap();
                assert_eq!(batch.n_positions, 8);
                assert_eq!(buckets.len(), 8);
                loader.recycle((batch, buckets));
                if state
                    .progress
                    .lock()
                    .unwrap()
                    .epochs
                    .first_key_value()
                    .is_some_and(|(&epoch, _)| epoch > 1)
                {
                    completed = true;
                    break;
                }
            }
            assert!(completed, "two epochs did not complete");
            let mut actual: Vec<_> = state
                .accepted
                .lock()
                .unwrap()
                .iter()
                .copied()
                .filter(|&(epoch, _)| epoch < 2)
                .collect();
            actual.sort_unstable();
            assert_eq!(actual, expected, "workers={workers} shuffle={shuffle}");
        }
    }
}

#[test]
fn sampling_barren_guard_waits_for_out_of_order_completions() {
    let mut progress = BucketProgress::default();
    for epoch in 0..5 {
        progress.epochs.insert(
            epoch,
            EpochSampling {
                pending: u64::from(epoch == 0),
                kept: 0,
                closed: true,
            },
        );
    }
    progress.check().unwrap();
    assert_eq!(progress.barren, 0);
    let first = progress.epochs.get_mut(&0).unwrap();
    first.pending = 0;
    first.kept = 1;
    progress.check().unwrap();
    assert_eq!(progress.barren, 4);
    progress.epochs.insert(
        5,
        EpochSampling {
            pending: 0,
            kept: 0,
            closed: true,
        },
    );
    assert!(progress.check().is_err());
}

#[test]
fn sampling_loader_rejects_hcpe_ply_options_before_starting_workers() {
    let path = fixture("sample.hcpe");
    for sampling in [
        Sampling {
            min_ply: Some(18),
            ..Default::default()
        },
        Sampling {
            ply_keep_curve: vec![(0.0, 1.0)],
            ..Default::default()
        },
    ] {
        let result = BucketedPrefetchedLoader::spawn_with_score_sources(
            &[&path],
            8,
            None,
            None,
            2,
            BucketMode::KingRank9,
            FeatureSet::HalfKaHmMerged.spec(),
            true,
            9,
            3800,
            false,
            None,
            0,
            false,
            0,
            sampling,
        );
        assert!(result.err().unwrap().to_string().contains("HCPE"));
    }
}
