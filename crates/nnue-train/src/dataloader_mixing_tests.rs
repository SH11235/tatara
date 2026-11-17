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
            .map(|rate| MixSource {
                path: "unused.psv".into(),
                rate,
            })
            .collect::<Vec<_>>();
        assert!(validate_mix_sources(&sources).is_err());
    }
    assert!(validate_mix_sources(&[]).is_ok());
    assert!(
        validate_mix_sources(&[MixSource {
            path: "unused.psv".into(),
            rate: 0.03
        }])
        .is_ok()
    );
}

#[test]
fn mixing_matches_rates_and_seed_and_wraps_each_teacher_independently() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1, 2, 3]);
    let first = teacher(dir.path(), "first.psv", &[11, 12]);
    let second = teacher(dir.path(), "second.psv", &[21, 22, 23, 24, 25]);
    let sources = [
        MixSource {
            path: first,
            rate: 0.03,
        },
        MixSource {
            path: second,
            rate: 0.2,
        },
    ];
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
            (next.source, next.record.score()),
            (again.source, again.record.score())
        );
        assert_eq!(
            next.record.score(),
            scores[next.source][counts[next.source] % scores[next.source].len()]
        );
        counts[next.source] += 1;
        different |= next.source != c.next().unwrap().source;
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
fn mixing_rejects_formats_and_score_sources() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1]);
    let hcpe = dir.path().join("other.hcpe");
    std::fs::write(&hcpe, [0u8; HCPE_RECORD_BYTES]).unwrap();
    let sources = [MixSource {
        path: hcpe,
        rate: 0.03,
    }];
    let error = MixedTeacherReader::new(main_reader(&main, None, None), &sources, 0)
        .err()
        .unwrap();
    assert!(error.to_string().contains("format"));
    let mut source = main_reader(&main, None, None);
    source.dual_label_psv = Some(DualLabelMode::All);
    let sources = [MixSource {
        path: main.clone(),
        rate: 0.03,
    }];
    assert!(MixedTeacherReader::new(source, &sources, 0).is_err());
}

#[test]
fn mixed_teachers_apply_drop_before_clamp() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1, 1000, 150]);
    let extra = teacher(dir.path(), "extra.psv", &[1000, -150, -1]);
    let sources = [MixSource {
        path: extra,
        rate: 0.4,
    }];
    let mut reader =
        MixedTeacherReader::new(main_reader(&main, Some(1000), Some(100)), &sources, 0).unwrap();
    let mut counts = [0usize; 2];
    for _ in 0..100 {
        let next = reader.next().unwrap();
        assert_eq!(
            next.record.score(),
            [vec![1, 100], vec![-100, -1]][next.source][counts[next.source] % 2]
        );
        counts[next.source] += 1;
    }
    assert!(counts.into_iter().all(|n| n > 0));
}

#[test]
fn window_shuffle_follows_mixing_and_preserves_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let main = teacher(dir.path(), "main.psv", &[1, 2, 3]);
    let extra = teacher(dir.path(), "extra.psv", &[11, 12]);
    let sources = [MixSource {
        path: extra,
        rate: 0.4,
    }];
    let mut sequential = mixed(&main, &sources, 55);
    let mut windowed = WindowedPsvReader::spawn(mixed(&main, &sources, 55), 17, true, 99);
    let mut differs = false;
    for _ in 0..4 {
        let expected = (0..17)
            .map(|_| {
                let r = sequential.next().unwrap();
                (r.source, r.record.score())
            })
            .collect::<Vec<_>>();
        let actual = (0..17)
            .map(|_| {
                let r = windowed.next().unwrap();
                (r.source, r.record.score())
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
    let sources = [MixSource {
        path: extra,
        rate: 0.4,
    }];
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
                score_source: None,
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
            score_source: None,
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
    let sources = [MixSource {
        path: extra,
        rate: 0.99,
    }];
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
            &[MixSource {
                path: path.clone(),
                rate: 0.3,
            }],
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
                    r.source,
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
    let mixes = [MixSource {
        path: fixture.clone(),
        rate: 0.3,
    }];
    let make = || {
        let mut main = PsvEpochReader::new_range(&[&fixture], 0, 4000, None, None, None).unwrap();
        main.configure_sampling(sampling.clone());
        MixedTeacherReader::new(main, &mixes, 123).unwrap()
    };
    let mut a = make();
    let mut b = make();
    let mut counts = [0; 2];
    for _ in 0..10_000 {
        let r = a.next().unwrap();
        let again = b.next().unwrap();
        assert_eq!(
            (r.source, r.record.epoch, r.record.record_index),
            (again.source, again.record.epoch, again.record.record_index)
        );
        assert_eq!(r.record.teacher, r.source as u64);
        assert!(sampling.accepts(
            r.record.teacher,
            r.record.epoch,
            r.record.record_index,
            0.65 * 0.8
        ));
        counts[r.source] += 1;
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

#[test]
fn sampled_mixed_window_resume_crosses_windows_without_losing_records() {
    let dir = tempfile::tempdir().unwrap();
    let primary = teacher(dir.path(), "primary.psv", &(1..=2000).collect::<Vec<_>>());
    let sources = [
        MixSource {
            path: teacher(
                dir.path(),
                "extra.psv",
                &(10001..=12000).collect::<Vec<_>>(),
            ),
            rate: 0.25,
        },
        MixSource {
            path: teacher(
                dir.path(),
                "another.psv",
                &(20001..=22000).collect::<Vec<_>>(),
            ),
            rate: 0.35,
        },
    ];
    let open = |workers, position: Option<&TeacherPosition>| {
        BucketedPrefetchedLoader::spawn_mixed_resumable(
            &[&primary],
            BucketedLoaderOptions {
                sampling: crate::sampling::Sampling {
                    keep_rate: 0.8,
                    bucket_keep_rates: Some(vec![0.7; 9]),
                    skip_seed: 19,
                    ..Default::default()
                },
                batch_size: 128,
                score_drop_abs: None,
                score_clamp_abs: None,
                num_workers: workers,
                bucket_mode: BucketMode::KingRank9,
                feature_set: FeatureSet::HalfKaHmMerged.spec(),
                compute_bucket: true,
                num_buckets: 9,
                train_end_offset: 2000 * PSV_RECORD_BYTES,
                monitor_active: false,
                score_source: None,
                teacher_shuffle_buffer_mib: 1,
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
    let mut reference = open(1, None);
    let expected = collect(&mut reference, 350);
    drop(reference);
    for workers in [1, 4] {
        let mut first = open(workers, None);
        let mut actual = collect(&mut first, 151);
        let position: TeacherPosition =
            serde_json::from_slice(&serde_json::to_vec(first.position()).unwrap()).unwrap();
        assert_eq!(position.mixes.len(), 2);
        assert!(position.window_index.unwrap() > 0);
        assert!(position.epoch > 0);
        drop(first);
        let mut resumed = open(workers, Some(&position));
        actual.extend(collect(&mut resumed, 199));
        if workers == 1 {
            assert_eq!(actual, expected);
        }
        actual.sort_unstable();
        let mut sorted = expected.clone();
        sorted.sort_unstable();
        assert_eq!(actual, sorted);
    }
}
