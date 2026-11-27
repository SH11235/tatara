use super::*;
use shogi_features::FeatureSet;

fn teacher(dir: &Path, name: &str, scores: &[i16]) -> PathBuf {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
    let bytes = std::fs::read(fixture).unwrap();
    let base = &bytes[..PSV_RECORD_BYTES as usize];
    let mut output = Vec::new();
    for score in scores {
        let mut record = base.to_vec();
        record[32..34].copy_from_slice(&score.to_le_bytes());
        output.extend(record);
    }
    let path = dir.join(name);
    std::fs::write(&path, output).unwrap();
    path
}

fn main_reader(path: &Path, drop: Option<i32>, clamp: Option<i16>) -> PsvEpochReader {
    PsvEpochReader::new_range(
        &[path],
        0,
        teacher_data_bytes(&[path]).unwrap(),
        drop,
        clamp,
        None,
    )
    .unwrap()
}

fn mixed(path: &Path, sources: &[MixSource], seed: u64) -> MixedTeacherReader {
    MixedTeacherReader::new(main_reader(path, None, None), sources, seed).unwrap()
}

#[test]
fn rate_validation_rejects_invalid_values_and_totals() {
    for rates in [
        vec![0.0],
        vec![-0.1],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1.0],
        vec![0.7, 0.3],
        vec![0.8, 0.3],
    ] {
        let sources = rates
            .into_iter()
            .map(|rate| MixSource::new("unused.psv".into(), rate))
            .collect::<Vec<_>>();
        assert!(validate_mix_sources(&sources).is_err());
    }
    assert!(validate_mix_sources(&[]).is_ok());
    assert!(validate_mix_sources(&[MixSource::new("unused.psv".into(), 0.03)]).is_ok());
}

#[test]
fn mixing_matches_rates_and_seed_and_wraps_each_teacher_independently() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1, 2, 3]);
    let first = teacher(dir.path(), "first.psv", &[11, 12]);
    let second = teacher(dir.path(), "second.psv", &[21, 22, 23, 24, 25]);
    let sources = [MixSource::new(first, 0.03), MixSource::new(second, 0.2)];
    let mut a = mixed(&main, &sources, 123);
    let mut b = mixed(&main, &sources, 123);
    let mut c = mixed(&main, &sources, 124);
    let mut counts = [0usize; 3];
    let mut different = false;
    let scores = [&[1, 2, 3][..], &[11, 12][..], &[21, 22, 23, 24, 25][..]];
    for _ in 0..30_000 {
        let next = a.next().unwrap();
        let again = b.next().unwrap();
        assert_eq!(
            (next.source(), next.record.score()),
            (again.source(), again.record.score())
        );
        assert_eq!(
            next.record.score(),
            scores[next.source()][counts[next.source()] % scores[next.source()].len()]
        );
        counts[next.source()] += 1;
        different |= next.source() != c.next().unwrap().source();
    }
    assert!(different);
    for (count, expected) in counts.into_iter().zip([0.77, 0.03, 0.2]) {
        assert!((count as f64 / 30_000.0 - expected).abs() < 0.01);
    }
    for (reader, count) in a.sources.iter().zip(counts) {
        let len = ((reader.end_offset - reader.start_offset) / reader.record_bytes) as usize;
        assert_eq!(reader.epoch as usize, (count - 1) / len);
        assert_eq!(reader.record_index as usize, (count - 1) % len + 1);
    }
}

#[test]
fn mixing_rejects_mismatched_formats() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1]);
    let hcpe = dir.path().join("other.hcpe");
    std::fs::write(&hcpe, [0u8; HCPE_RECORD_BYTES]).unwrap();
    let sources = [MixSource::new(hcpe, 0.03)];
    let error = MixedTeacherReader::new(main_reader(&main, None, None), &sources, 0)
        .err()
        .unwrap();
    assert!(error.to_string().contains("format"));
}

fn assert_mixed_scores(reader: &mut MixedTeacherReader, scores: [&[i16]; 2]) {
    let mut counts = [0; 2];
    for _ in 0..100 {
        let next = reader.next().unwrap();
        let source = next.source();
        assert_eq!(
            next.record.score(),
            scores[source][counts[source] % scores[source].len()]
        );
        counts[source] += 1;
    }
    assert!(counts.into_iter().all(|n| n > 2));
}

#[test]
fn mixed_dual_labels_apply_only_to_the_main_teacher() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[10, 20]);
    let extra = teacher(dir.path(), "extra.psv", &[100, 200, 300]);
    let mut bytes = std::fs::read(&main).unwrap();
    for (record, (score, padding)) in bytes
        .chunks_exact_mut(PSV_RECORD_BYTES as usize)
        .zip([(30_i16, 2), (40, 3)])
    {
        record[34..36].copy_from_slice(&score.to_le_bytes());
        record[39] = padding;
    }
    std::fs::write(&main, bytes).unwrap();
    let sources = [MixSource::new(extra, 0.4)];
    for (mode, expected) in [
        (DualLabelMode::All, [30, 40]),
        (DualLabelMode::Gated, [30, 20]),
    ] {
        let main = PsvEpochReader::new_range(
            &[&main],
            0,
            2 * PSV_RECORD_BYTES,
            None,
            None,
            LabelOptions {
                score_source: Some(ScoreSource::DualLabel(mode)),
                tree_sample_eval_only: true,
            },
        )
        .unwrap();
        let mut reader = MixedTeacherReader::new(main, &sources, 0).unwrap();
        assert_mixed_scores(&mut reader, [&expected, &[100, 200, 300]]);
        let TeacherRecord::Psv(psv) = reader.sources[0].next().unwrap().record else {
            panic!("expected a PSV record");
        };
        assert_eq!(psv.game_result(), EVAL_ONLY_GAME_RESULT);
    }
}

#[test]
fn mixed_score_sidecar_and_mask_apply_only_to_concatenated_main_teacher() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[10]);
    let continuation = teacher(dir.path(), "main-part2.psv", &[20, 30]);
    let extra = teacher(dir.path(), "extra.psv", &[100, 200]);
    let scores = dir.path().join("scores.i16");
    let mask = dir.path().join("scores.mask");
    std::fs::write(&scores, [40_i16, 50, 60].map(i16::to_le_bytes).concat()).unwrap();
    std::fs::write(&mask, [0b010]).unwrap();
    let main = PsvEpochReader::new_range(
        &[&main, &continuation],
        0,
        3 * PSV_RECORD_BYTES,
        None,
        None,
        Some(ScoreSource::Sidecar {
            scores: scores.as_path(),
            mask: Some(mask.as_path()),
        }),
    )
    .unwrap();
    let sources = [MixSource::new(extra, 0.4)];
    let mut reader = MixedTeacherReader::new(main, &sources, 0).unwrap();
    assert_mixed_scores(&mut reader, [&[40, 20, 60], &[100, 200]]);
}

#[test]
fn mixed_teachers_apply_drop_before_clamp_with_overrides_or_inheritance() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1000, 500, 150, 1]);
    let extra = teacher(dir.path(), "extra.psv", &[-1000, -500, -150, -1]);
    for (drop, clamp, expected) in [
        (None, None, &[-300, -150, -1][..]),
        (Some(500), None, &[-150, -1]),
        (None, Some(100), &[-100, -100, -1]),
        (Some(500), Some(100), &[-100, -1]),
    ] {
        let sources = [MixSource {
            score_drop_abs: drop,
            score_clamp_abs: clamp,
            ..MixSource::new(extra.clone(), 0.4)
        }];
        let mut reader =
            MixedTeacherReader::new(main_reader(&main, Some(1000), Some(300)), &sources, 0)
                .unwrap();
        assert_mixed_scores(&mut reader, [&[300, 150, 1], expected]);
    }

    let sources = [MixSource {
        score_drop_abs: Some(300),
        ..MixSource::new(extra, 0.4)
    }];
    let error = MixedTeacherReader::new(main_reader(&main, Some(1000), Some(300)), &sources, 0)
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("extra.psv"));
    assert!(error.to_string().contains("must be < score_drop_abs"));
}

#[test]
fn window_shuffle_follows_mixing_and_preserves_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1, 2, 3]);
    let extra = teacher(dir.path(), "extra.psv", &[11, 12]);
    let sources = [MixSource::new(extra, 0.4)];
    let mut sequential = mixed(&main, &sources, 55);
    let mut windowed = WindowedPsvReader::spawn(mixed(&main, &sources, 55), 17, true, 99);
    let mut differs = false;
    for _ in 0..4 {
        let expected = (0..17)
            .map(|_| {
                let r = sequential.next().unwrap();
                (r.source(), r.record.score())
            })
            .collect::<Vec<_>>();
        let actual = (0..17)
            .map(|_| {
                let r = windowed.next().unwrap();
                (r.source(), r.record.score())
            })
            .collect::<Vec<_>>();
        differs |= expected != actual;
        let mut expected = expected;
        let mut actual = actual;
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(expected, actual);
    }
    assert!(differs);
}

#[test]
fn batches_count_only_delivered_records_and_unmixed_order_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1, 2, 3]);
    let extra = teacher(dir.path(), "extra.psv", &[11, 12]);
    let sources = [MixSource::new(extra, 0.4)];
    for shuffle in [false, true] {
        let mut loader = BucketedPrefetchedLoader::spawn_with_mixing(
            &[&main],
            BucketedLoaderOptions {
                sampling: Default::default(),
                batch_size: 16,
                score_drop_abs: None,
                score_clamp_abs: None,
                num_workers: 2,
                bucket_mode: BucketMode::KingRank9,
                feature_set: FeatureSet::HalfKaHmMerged.spec(),
                compute_bucket: false,
                num_buckets: 1,
                train_end_offset: 3 * PSV_RECORD_BYTES,
                monitor_active: false,
                labels: LabelOptions::default(),
                teacher_shuffle_buffer_mib: usize::from(shuffle),
                teacher_shuffle: shuffle,
                teacher_shuffle_seed: 99,
            },
            &sources,
            55,
        )
        .unwrap();
        for _ in 0..3 {
            let (batch, buckets) = loader.next_batch().unwrap().unwrap();
            let expected_mix = batch.score[..batch.n_positions]
                .iter()
                .filter(|&&score| score >= 11.0)
                .count() as u64;
            assert_eq!(batch.teacher_counts, [16 - expected_mix, expected_mix]);
            loader.recycle((batch, buckets));
        }
    }
    let mut loader = BucketedPrefetchedLoader::spawn_with_mixing(
        &[&main],
        BucketedLoaderOptions {
            sampling: Default::default(),
            batch_size: 8,
            score_drop_abs: None,
            score_clamp_abs: None,
            num_workers: 1,
            bucket_mode: BucketMode::KingRank9,
            feature_set: FeatureSet::HalfKaHmMerged.spec(),
            compute_bucket: false,
            num_buckets: 1,
            train_end_offset: 3 * PSV_RECORD_BYTES,
            monitor_active: false,
            labels: LabelOptions::default(),
            teacher_shuffle_buffer_mib: 0,
            teacher_shuffle: false,
            teacher_shuffle_seed: 0,
        },
        &[],
        99,
    )
    .unwrap();
    let (batch, _) = loader.next_batch().unwrap().unwrap();
    assert_eq!(&batch.score[..8], &[1., 2., 3., 1., 2., 3., 1., 2.]);
    assert!(batch.teacher_counts.is_empty());
}

#[test]
fn cancelling_mixed_filtered_scan_stops_every_teacher() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1]);
    let extra = teacher(dir.path(), "extra.psv", &[999]);
    let sources = [MixSource::new(extra, 0.99)];
    let mut reader =
        MixedTeacherReader::new(main_reader(&main, Some(100), None), &sources, 0).unwrap();
    let stop = reader.stop_flag();
    reader.sources[1].after_record = Some(Box::new(move || stop.store(true, Ordering::Relaxed)));
    let error = reader.next().err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
}

#[test]
fn matching_hcpe_and_psv_mixtures_decode_identically() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data");
    let read = |extension: &str| {
        let path = fixture.join(format!("sample.{extension}"));
        let mut source = MixedTeacherReader::new(
            main_reader(&path, None, None),
            &[MixSource::new(path.clone(), 0.3)],
            72,
        )
        .unwrap();
        (0..400)
            .map(|_| {
                let r = source.next().unwrap();
                let board = r.record.decode().unwrap();
                let mut batch = Batch::with_capacity(1, FeatureSet::HalfKaHmMerged.spec());
                batch.push_decoded(&board).unwrap();
                (
                    r.source(),
                    batch.score,
                    batch.wdl,
                    batch.stm_indices,
                    batch.nstm_indices,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(read("psv"), read("hcpe"));
}

#[test]
fn mixed_sampling_preserves_teacher_identity_rates_and_bucket_sets_across_workers() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
    let sampling = crate::sampling::Sampling {
        keep_rate: 0.65,
        ply_keep_curve: vec![(0.0, 0.8)],
        bucket_keep_rates: Some(vec![0.5; 9]),
        skip_seed: 42,
        ..Default::default()
    };
    let mixes = [MixSource::new(fixture.clone(), 0.3)];
    let make = || {
        let mut main = PsvEpochReader::new_range(&[&fixture], 0, 4000, None, None, None).unwrap();
        main.configure_sampling(sampling.clone(), None);
        MixedTeacherReader::new(main, &mixes, 123).unwrap()
    };
    let mut a = make();
    let mut b = make();
    let mut counts = [0; 2];
    for _ in 0..10_000 {
        let r = a.next().unwrap();
        let again = b.next().unwrap();
        assert_eq!(
            (r.source(), r.record.epoch, r.record.record_index),
            (
                again.source(),
                again.record.epoch,
                again.record.record_index
            )
        );
        assert_eq!(r.record.teacher, r.source() as u16);
        assert!(sampling.accepts(
            u64::from(r.record.teacher),
            r.record.epoch,
            r.record.record_index,
            0.65 * 0.8
        ));
        counts[r.source()] += 1;
    }
    assert!((counts[1] as f64 / 10_000.0 - 0.3).abs() < 0.02);
    let mut expected = vec![Vec::new(); 2];
    for (teacher, expected) in expected.iter_mut().enumerate() {
        for epoch in 0..2 {
            for record in 0..100 {
                if sampling.accepts(teacher as u64, epoch, record, 0.65 * 0.8 * 0.5) {
                    expected.push((epoch, record));
                }
            }
        }
    }
    assert_ne!(expected[0], expected[1]);
    for workers in [1, 4] {
        for shuffle in [false, true] {
            let mixed = make();
            let states = mixed.bucket_samplings();
            let input = if shuffle {
                TrainingPsvReader::MixedWindowed(WindowedPsvReader::spawn(mixed, 17, true, 99), 2)
            } else {
                TrainingPsvReader::Mixed(Box::new(mixed))
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
                &fixture,
            );
            let mut completed = false;
            for _ in 0..1000 {
                let (batch, buckets) = loader.next_batch().unwrap().unwrap();
                assert_eq!(batch.teacher_counts.iter().sum::<u64>(), 8);
                loader.recycle((batch, buckets));
                if states.iter().all(|state| {
                    state
                        .as_ref()
                        .unwrap()
                        .progress
                        .lock()
                        .unwrap()
                        .epochs
                        .first_key_value()
                        .is_some_and(|(&epoch, _)| epoch > 1)
                }) {
                    completed = true;
                    break;
                }
            }
            assert!(completed);
            for (teacher, state) in states.iter().enumerate() {
                let mut actual = state
                    .as_ref()
                    .unwrap()
                    .accepted
                    .lock()
                    .unwrap()
                    .iter()
                    .copied()
                    .filter(|&(epoch, _)| epoch < 2)
                    .collect::<Vec<_>>();
                actual.sort_unstable();
                assert_eq!(
                    actual, expected[teacher],
                    "teacher={teacher} workers={workers} shuffle={shuffle}"
                );
            }
        }
    }
}

fn mark_tree_samples(path: &Path, stride: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    for (i, record) in bytes
        .chunks_exact_mut(PSV_RECORD_BYTES as usize)
        .enumerate()
    {
        record[39] = if i.is_multiple_of(stride) { 2 } else { 0 };
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn marked_sampling_uses_its_own_probability_before_buckets() {
    let dir = tempfile::tempdir().unwrap();
    let path = teacher(dir.path(), "tree.psv", &(1..=4000).collect::<Vec<_>>());
    mark_tree_samples(&path, 2);
    for (rate, expected_range) in [(0.0, 0..=0), (0.25, 430..=570), (1.0, 2000..=2000)] {
        let sampling = crate::sampling::Sampling {
            tree_sample_accept: rate,
            keep_rate: 0.01,
            min_ply: Some(u16::MAX),
            ply_keep_curve: vec![(0.0, 0.0)],
            bucket_keep_rates: Some(vec![0.0; 9]),
            skip_seed: 42,
            ..Default::default()
        };
        let mut reader = main_reader(&path, None, None);
        reader.tree_sample_eval_only = true;
        reader.configure_sampling(sampling.clone(), None);
        let buckets = reader.bucket_sampling.clone().unwrap();
        let mut actual = Vec::new();
        while let Some(record) = reader.next_in_epoch().unwrap() {
            assert!(record.tree_sample);
            assert!(buckets.finish(&record, 0).unwrap());
            actual.push(record.record_index);
        }
        let expected = (0..4000)
            .step_by(2)
            .filter(|&record| sampling.accepts(0, 0, record, rate))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(expected_range.contains(&actual.len()));
        assert_eq!(reader.sampling_counts.read.load(Ordering::Relaxed), 4000);
        assert_eq!(
            reader.sampling_counts.kept.load(Ordering::Relaxed),
            actual.len() as u64
        );
        assert_eq!(
            buckets.buckets[0].read.load(Ordering::Relaxed),
            actual.len() as u64
        );
        assert_eq!(
            buckets.buckets[0].kept.load(Ordering::Relaxed),
            actual.len() as u64
        );
    }
}

#[test]
fn mix_keep_rate_bypasses_game_sampling_only_for_its_teacher() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[10, 11]);
    let inherited = teacher(dir.path(), "inherited.psv", &[20, 21]);
    let overridden = teacher(dir.path(), "overridden.psv", &[30, 31]);
    for path in [&main, &inherited, &overridden] {
        mark_tree_samples(path, 2);
    }
    let mixes = [
        MixSource::new(inherited, 0.20),
        MixSource {
            keep_rate: Some(0.5),
            ..MixSource::new(overridden, 0.30)
        },
    ];
    let sampling = crate::sampling::Sampling {
        tree_sample_accept: 0.25,
        min_ply: Some(u16::MAX),
        ply_keep_curve: vec![(0.0, 0.0)],
        bucket_keep_rates: Some(vec![0.0; 9]),
        skip_seed: 42,
        ..Default::default()
    };
    // Each epoch provides enough independent draws to avoid barren-pass errors.
    for path in [&main, &mixes[0].path, &mixes[1].path] {
        let bytes = std::fs::read(path).unwrap();
        std::fs::write(path, bytes.repeat(64)).unwrap();
    }
    let mut reader = main_reader(&main, None, None);
    reader.tree_sample_eval_only = true;
    reader.configure_sampling(sampling.clone(), None);
    let mut mixed = MixedTeacherReader::new(reader, &mixes, 123).unwrap();
    let buckets = mixed.bucket_samplings();
    assert!(buckets[2].is_none());
    let mut counts = [0; 3];
    let mut mix_unmarked = 0;
    for _ in 0..4000 {
        let next = mixed.next().unwrap();
        let source = next.source();
        let record = next.record;
        counts[source] += 1;
        if record.tree_sample {
            assert!(sampling.accepts(source as u64, record.epoch, record.record_index, 0.25));
        } else {
            assert_eq!(source, 2);
            assert!(sampling.accepts(2, record.epoch, record.record_index, 0.5));
            mix_unmarked += 1;
        }
        if let Some(bucket) = &buckets[source] {
            assert!(bucket.finish(&record, 0).unwrap());
        }
    }
    for (count, expected) in counts.into_iter().zip([0.5, 0.2, 0.3]) {
        assert!((f64::from(count) / 4000.0 - expected).abs() < 0.025);
    }
    assert!(mix_unmarked > counts[2] / 2);
}

#[test]
fn delivered_teacher_mix_is_weighted_by_bucket_acceptance() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[10; 512]);
    let marked = teacher(dir.path(), "marked.psv", &[20; 512]);
    let overridden = teacher(dir.path(), "overridden.psv", &[30; 512]);
    mark_tree_samples(&marked, 1);
    let rates = [0.5, 0.25, 0.25];
    let bucket_acceptance = [0.25, 1.0, 1.0];
    let mixes = [
        MixSource::new(marked, rates[1]),
        MixSource {
            keep_rate: Some(0.5),
            ..MixSource::new(overridden, rates[2])
        },
    ];
    let mut reader = main_reader(&main, None, None);
    reader.tree_sample_eval_only = true;
    reader.configure_sampling(
        crate::sampling::Sampling {
            bucket_keep_rates: Some(vec![bucket_acceptance[0]; 9]),
            skip_seed: 42,
            ..Default::default()
        },
        None,
    );
    let mixed = MixedTeacherReader::new(reader, &mixes, 123).unwrap();
    let mut loader = BucketedPrefetchedLoader::spawn_with_reader(
        64,
        4,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        false,
        TrainingPsvReader::Mixed(Box::new(mixed)),
        &main,
    );
    let mut delivered = [0; 3];
    for _ in 0..128 {
        let slot = loader.next_batch().unwrap().unwrap();
        assert_eq!(slot.0.teacher_counts.iter().sum::<u64>(), 64);
        for (total, count) in delivered.iter_mut().zip(&slot.0.teacher_counts) {
            *total += count;
        }
        loader.recycle(slot);
    }
    let total = delivered.iter().sum::<u64>();
    let weights = std::array::from_fn::<_, 3, _>(|i| rates[i] * bucket_acceptance[i]);
    let total_weight = weights.iter().sum::<f64>();
    for (count, weight) in delivered.into_iter().zip(weights) {
        let actual = count as f64 / total as f64;
        let expected = weight / total_weight;
        assert!((actual - expected).abs() < 0.025, "{actual} vs {expected}");
    }
    let buckets = loader.sampling_bucket_counts().unwrap();
    let (read, kept) = buckets.into_iter().find(|&(read, _)| read != 0).unwrap();
    assert!(kept >= total, "bucket kept={kept}, delivered={total}");
    assert!(read > kept, "bucket read={read}, kept={kept}");
}

#[test]
fn sampled_mixed_window_resume_crosses_windows_without_losing_records() {
    let dir = tempfile::tempdir().unwrap();
    let primary = teacher(dir.path(), "primary.psv", &(1..=2000).collect::<Vec<_>>());
    let mut sources = [
        MixSource::new(
            teacher(
                dir.path(),
                "extra.psv",
                &(10001..=12000).collect::<Vec<_>>(),
            ),
            0.25,
        ),
        MixSource::new(
            teacher(
                dir.path(),
                "another.psv",
                &(20001..=22000).collect::<Vec<_>>(),
            ),
            0.35,
        ),
    ];
    sources[0].keep_rate = Some(0.6);
    for path in [&primary, &sources[0].path, &sources[1].path] {
        mark_tree_samples(path, 4);
    }
    let open = |workers, window_mib, batch_size, position: Option<&TeacherPosition>| {
        BucketedPrefetchedLoader::spawn_mixed_resumable(
            &[&primary],
            BucketedLoaderOptions {
                sampling: crate::sampling::Sampling {
                    keep_rate: 0.8,
                    tree_sample_accept: 0.35,
                    bucket_keep_rates: Some(vec![0.7; 9]),
                    skip_seed: 19,
                    ..Default::default()
                },
                batch_size,
                score_drop_abs: None,
                score_clamp_abs: None,
                num_workers: workers,
                bucket_mode: BucketMode::KingRank9,
                feature_set: FeatureSet::HalfKaHmMerged.spec(),
                compute_bucket: true,
                num_buckets: 9,
                train_end_offset: 2000 * PSV_RECORD_BYTES,
                monitor_active: false,
                labels: LabelOptions {
                    tree_sample_eval_only: true,
                    ..Default::default()
                },
                teacher_shuffle_buffer_mib: window_mib,
                teacher_shuffle: true,
                teacher_shuffle_seed: 31,
            },
            &sources,
            79,
            position,
        )
        .unwrap()
    };
    let collect = |loader: &mut BucketedPrefetchedLoader, batches| {
        let mut scores = Vec::new();
        for _ in 0..batches {
            let slot = loader.next_batch().unwrap().unwrap();
            scores.extend(slot.0.score[..slot.0.n_positions].iter().map(|&s| s as i16));
            loader.recycle(slot);
        }
        scores
    };
    for window_mib in [0, 1] {
        let mut reference = open(1, window_mib, 128, None);
        let expected = collect(&mut reference, 350);
        drop(reference);
        for workers in [1, 4] {
            for cut in [1, 151] {
                let mut first = open(workers, window_mib, 128, None);
                let actual = collect(&mut first, cut);
                let position: TeacherPosition =
                    serde_json::from_slice(&serde_json::to_vec(first.position()).unwrap()).unwrap();
                assert_eq!(position.mixes.len(), 2);
                assert!(position.bucket_skip > 0);
                if cut == 151 {
                    if window_mib != 0 {
                        assert!(position.window_index.unwrap() > 0);
                    }
                    assert!(position.epoch > 0);
                }
                drop(first);
                for batch_size in [64, 128] {
                    let mut resumed = open(workers, window_mib, batch_size, Some(&position));
                    let mut joined = actual.clone();
                    joined.extend(collect(&mut resumed, (350 - cut) * 128 / batch_size));
                    assert_eq!(joined, expected);
                }
            }
        }
    }
}

#[test]
fn mixed_tree_samples_use_eval_only_labels_when_marked() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[100]);
    let extra = teacher(dir.path(), "tree.psv", &[200, 201]);
    for enabled in [false, true] {
        for (path, marked) in [(&main, false), (&extra, enabled)] {
            let mut bytes = std::fs::read(path).unwrap();
            for (i, record) in bytes
                .chunks_exact_mut(PSV_RECORD_BYTES as usize)
                .enumerate()
            {
                record[38] = 1;
                // Bit 0 has no dual-label meaning on a mixed teacher and must be ignored.
                record[39] = match (*path == extra, marked && i == 0) {
                    (true, true) => 3,
                    (true, false) => 1,
                    (false, _) => 0,
                };
            }
            std::fs::write(path, bytes).unwrap();
        }
        for shuffle in [false, true] {
            let mut loader = BucketedPrefetchedLoader::spawn_with_mixing(
                &[&main],
                BucketedLoaderOptions {
                    sampling: Default::default(),
                    batch_size: 1024,
                    score_drop_abs: None,
                    score_clamp_abs: None,
                    num_workers: 2,
                    bucket_mode: BucketMode::KingRank9,
                    feature_set: FeatureSet::HalfKaHmMerged.spec(),
                    compute_bucket: false,
                    num_buckets: 1,
                    train_end_offset: PSV_RECORD_BYTES,
                    monitor_active: false,
                    labels: LabelOptions {
                        score_source: None,
                        tree_sample_eval_only: enabled,
                    },
                    teacher_shuffle_buffer_mib: usize::from(shuffle),
                    teacher_shuffle: shuffle,
                    teacher_shuffle_seed: 99,
                },
                &[MixSource::new(extra.clone(), 0.03)],
                55,
            )
            .unwrap();
            let (batch, _) = loader.next_batch().unwrap().unwrap();
            let mut counts = [0; 3];
            for (&score, &wdl) in batch.score.iter().zip(&batch.wdl) {
                let source = match score as i16 {
                    100 => 0,
                    200 => 1,
                    201 => 2,
                    other => panic!("unexpected score: {other}"),
                };
                counts[source] += 1;
                assert_eq!(
                    wdl,
                    if enabled && source == 1 {
                        EVAL_ONLY_WDL
                    } else {
                        1.0
                    }
                );
            }
            assert!(counts.into_iter().all(|count| count > 0));
        }
    }
}

#[test]
fn mixed_tree_samples_reject_reserved_padding_bits() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[100]);
    let extra = teacher(dir.path(), "invalid.psv", &[200]);
    let sources = [MixSource::new(extra.clone(), 0.4)];
    let mut bytes = std::fs::read(&extra).unwrap();
    for (padding, enabled) in [(4, true), (6, true)] {
        bytes[39] = padding;
        std::fs::write(&extra, &bytes).unwrap();
        let mut main = main_reader(&main, None, None);
        main.tree_sample_eval_only = enabled;
        let mut reader = MixedTeacherReader::new(main, &sources, 0).unwrap();
        let error = (0..100)
            .find_map(|_| reader.next().err())
            .expect("mixed teacher must be read");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("reserved padding bits"));
        assert!(error.to_string().contains(extra.to_str().unwrap()));
    }
    // Bits 0 and 1 are accepted with the flag; without it the padding is not interpreted.
    for (padding, enabled) in [(1, true), (3, true), (1, false), (2, false), (7, false)] {
        bytes[39] = padding;
        std::fs::write(&extra, &bytes).unwrap();
        let mut main = main_reader(&main, None, None);
        main.tree_sample_eval_only = enabled;
        let mut reader = MixedTeacherReader::new(main, &sources, 0).unwrap();
        for _ in 0..100 {
            reader.next().unwrap();
        }
    }
}
