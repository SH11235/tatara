use super::*;
use shogi_features::FeatureSet;

fn sample_record(score: i16) -> PackedSfenValue {
    let bytes = include_bytes!("../../shogi-format/tests/data/sample.psv");
    let mut psv = PackedSfenValue::default();
    psv.as_bytes_mut()
        .copy_from_slice(&bytes[..PSV_RECORD_BYTES as usize]);
    psv.set_score(score);
    psv
}

#[test]
fn cancellation_during_filtered_scan_stops_before_next_record_or_wrap() {
    for records in [1, 3] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filtered.psv");
        std::fs::write(&path, sample_record(30000).as_bytes().repeat(records)).unwrap();
        let end = records as u64 * PSV_RECORD_BYTES;
        let mut reader = PsvEpochReader::new_range(&path, 0, end, Some(1000), None, None).unwrap();
        let stop = Arc::clone(&reader.stop);
        reader.after_record = Some(Box::new(move || stop.store(true, Ordering::Relaxed)));
        let error = match reader.next_in_epoch() {
            Err(error) => error,
            Ok(_) => panic!("cancelled filtered scan must stop"),
        };
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(error.to_string().contains(path.to_str().unwrap()));
        assert_eq!(reader.record_index, 1);
        assert_eq!(reader.loader.remaining_bytes, end - PSV_RECORD_BYTES);
        assert_eq!(reader.barren_passes, 0);
    }
}

#[test]
fn direct_and_windowed_readers_share_cancellation_with_epoch_scan() {
    for windowed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filtered.psv");
        std::fs::write(&path, sample_record(30000).as_bytes().repeat(3)).unwrap();
        let mut source =
            PsvEpochReader::new_range(&path, 0, 3 * PSV_RECORD_BYTES, Some(1000), None, None)
                .unwrap();
        let stop = Arc::clone(&source.stop);
        source.after_record = Some(Box::new(move || stop.store(true, Ordering::Relaxed)));
        let mut reader = if windowed {
            TrainingPsvReader::Windowed(WindowedPsvReader::spawn(source, 2, false, 0))
        } else {
            TrainingPsvReader::Direct(Box::new(source))
        };
        let shared_stop = reader.stop_flag();
        assert!(
            reader.next().is_err(),
            "cancelled scan must not supply a position"
        );
        assert!(shared_stop.load(Ordering::Relaxed));
    }
}

#[test]
fn cancellation_after_eof_does_not_reopen_or_report_barren_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filtered.psv");
    std::fs::write(&path, sample_record(30000).as_bytes()).unwrap();
    let mut reader =
        PsvEpochReader::new_range(&path, 0, PSV_RECORD_BYTES, Some(1000), None, None).unwrap();
    reader.barren_passes = MAX_BARREN_PASSES - 1;
    let stop = Arc::clone(&reader.stop);
    reader.after_eof = Some(Box::new(move || stop.store(true, Ordering::Relaxed)));
    let error = match reader.next_in_epoch() {
        Err(error) => error,
        Ok(_) => panic!("EOF cancellation must precede reopen and barren failure"),
    };
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    assert_eq!(reader.loader.remaining_bytes, 0);
    assert_eq!(reader.record_index, 1);
    assert_eq!(reader.barren_passes, MAX_BARREN_PASSES - 1);
}

fn spawn_controlled_reader(path: &Path, reader: TrainingPsvReader) -> BucketedPrefetchedLoader {
    BucketedPrefetchedLoader::spawn_with_reader(
        path,
        1,
        1,
        BucketMode::KingRank9,
        FeatureSet::HalfKaHmMerged.spec(),
        true,
        9,
        false,
        reader,
    )
}

#[test]
fn drop_during_filtered_scan_joins_workers_without_recording_an_error() {
    for windowed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filtered.psv");
        std::fs::write(&path, sample_record(30000).as_bytes().repeat(3)).unwrap();
        let mut source =
            PsvEpochReader::new_range(&path, 0, 3 * PSV_RECORD_BYTES, Some(1000), None, None)
                .unwrap();
        let source_stop = Arc::clone(&source.stop);
        let scanned = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed_scans = Arc::clone(&scanned);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        source.after_record = Some(Box::new(move || {
            if observed_scans.fetch_add(1, Ordering::Relaxed) == 0 {
                ready_tx.send(()).unwrap();
                release_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("test releases the scan after Drop sets stop");
                assert!(source_stop.load(Ordering::Relaxed));
            }
        }));
        let reader = if windowed {
            TrainingPsvReader::Windowed(WindowedPsvReader::spawn(source, 2, false, 0))
        } else {
            TrainingPsvReader::Direct(Box::new(source))
        };
        let mut loader = spawn_controlled_reader(&path, reader);
        let stop = Arc::clone(&loader.stop);
        let error_slot = Arc::clone(&loader.err_slot);
        let (stopped_tx, stopped_rx) = mpsc::channel();
        loader.after_stop = Some(Box::new(move || stopped_tx.send(()).unwrap()));
        ready_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("reader must enter the filtered scan before Drop");
        let (completed_tx, completed_rx) = mpsc::channel();
        let task = thread::spawn(move || {
            drop(loader);
            completed_tx.send(()).unwrap();
        });
        stopped_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("Drop must set the shared stop flag without the reader lock");
        assert!(stop.load(Ordering::Relaxed));
        release_tx.send(()).unwrap();
        completed_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("Drop must join the reader and workers after releasing the scan");
        task.join().unwrap();
        assert_eq!(scanned.load(Ordering::Relaxed), 1);
        assert!(
            error_slot.lock().unwrap().is_none(),
            "worker must not record a cancelled filtered scan as an input failure"
        );
    }
}

#[test]
fn cancellation_after_record_read_skips_worker_decode_and_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("duplicate-kings.psv");
    let mut corrupt = sample_record(0);
    let bytes = corrupt.as_bytes_mut();
    bytes[0] = (bytes[0] & 1) | ((bytes[1] & 127) << 1);
    std::fs::write(&path, corrupt.as_bytes()).unwrap();
    let mut source =
        PsvEpochReader::new_range(&path, 0, PSV_RECORD_BYTES, None, None, None).unwrap();
    let stop = Arc::clone(&source.stop);
    source.after_record = Some(Box::new(move || stop.store(true, Ordering::Relaxed)));
    let mut loader = spawn_controlled_reader(&path, TrainingPsvReader::Direct(Box::new(source)));
    assert!(
        loader.next_batch().unwrap().is_none(),
        "cancelled worker must not decode a malformed record or publish a batch"
    );
    assert!(loader.err_slot.lock().unwrap().is_none());
}

#[test]
fn partial_consumption_drop_joins_direct_and_windowed_workers() {
    for buffer_mib in [0, 1] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teacher.psv");
        let passing = sample_record(0);
        let filtered = sample_record(30000);
        let mut bytes = passing.as_bytes().to_vec();
        bytes.extend(filtered.as_bytes().repeat(128));
        std::fs::write(&path, &bytes).unwrap();
        let (completed_tx, completed_rx) = mpsc::channel();
        let task = thread::spawn(move || {
            let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
                &path,
                2,
                Some(1000),
                None,
                2,
                BucketMode::KingRank9,
                FeatureSet::HalfKaHmMerged.spec(),
                true,
                9,
                bytes.len() as u64,
                false,
                None,
                buffer_mib,
                false,
                0,
            )
            .unwrap();
            let slot = loader.next_batch().unwrap().expect("first batch");
            assert_eq!(slot.0.n_positions, 2);
            drop(slot);
            let stop = Arc::clone(&loader.stop);
            let error_slot = Arc::clone(&loader.err_slot);
            drop(loader);
            assert!(stop.load(Ordering::Relaxed));
            assert!(
                error_slot.lock().unwrap().is_none(),
                "Drop cancellation is not an input failure"
            );
            completed_tx.send(()).unwrap();
        });
        completed_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("Drop must join all loader workers and the window producer");
        task.join().unwrap();
    }
}
