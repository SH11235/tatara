use nnue_train::dataloader::{
    BucketMode, BucketedLoaderOptions, BucketedPrefetchedLoader, LabelOptions,
};
use sha2::{Digest, Sha256};
use shogi_features::FeatureSet;

use crate::cli::{Cli, LoaderDigestArgs, TeacherShuffleBufferMib};

pub(crate) fn run(cli: &Cli, args: &LoaderDigestArgs) -> Result<(), Box<dyn std::error::Error>> {
    if cli.test_tail_positions.is_some() {
        return Err("loader-digest does not support --test-tail-positions".into());
    }
    if cli.test_data.is_some() {
        return Err("loader-digest does not support --test-data".into());
    }
    if cli.data.is_empty() {
        return Err("loader-digest requires --data".into());
    }
    let data = cli.data.as_slice();
    if cli.batch_size == 0 {
        return Err("--batch-size must be >= 1".into());
    }
    if args.batches == 0 {
        return Err("--batches must be >= 1".into());
    }
    let feature_set = FeatureSet::from_canonical_name(&cli.feature_set)
        .ok_or_else(|| format!("unknown feature set: {}", cli.feature_set))?
        .spec();
    // record 長の倍数検査は loader の open 時に入力形式ごとに行う。
    let file_size = nnue_train::dataloader::teacher_data_bytes(data)?;
    // `auto` は machine のメモリ量に依存し digest が machine 間で比較できなくなる
    // ため、digest では 0 (直接逐次読み) に解決する。明示値はそのまま使う。
    let teacher_shuffle_buffer_mib = match cli.teacher_shuffle_buffer_mib {
        TeacherShuffleBufferMib::Explicit(mib) => mib,
        TeacherShuffleBufferMib::Auto => 0,
    };
    let teacher_shuffle = TeacherShuffleBufferMib::effective_shuffle(
        teacher_shuffle_buffer_mib,
        cli.no_teacher_shuffle,
    );
    println!(
        "[digest] teacher shuffle window: {} | decode workers pinned to 1 (--threads is not used)",
        cli.teacher_shuffle_buffer_mib.describe(
            teacher_shuffle_buffer_mib,
            teacher_shuffle,
            cli.teacher_shuffle_seed
        )
    );
    let sampling = cli.sampling();
    if let Some(definition) = &sampling.keep_bucket {
        println!(
            "[digest] keep_bucket={}",
            serde_json::to_string(definition)?
        );
    }
    let mut loader = BucketedPrefetchedLoader::spawn_with_mixing(
        data,
        BucketedLoaderOptions {
            sampling,
            batch_size: cli.batch_size,
            score_drop_abs: cli.score_drop_abs,
            score_clamp_abs: cli.score_clamp_abs,
            num_workers: 1,
            bucket_mode: args
                .routing_bucket_spec
                .map_or(BucketMode::ProgressKpAbs, BucketMode::Spec),
            feature_set,
            compute_bucket: args.routing_bucket_spec.is_some(),
            num_buckets: args
                .routing_bucket_spec
                .map_or(1, |spec| spec.num_buckets()),
            train_end_offset: file_size,
            monitor_active: false,
            labels: LabelOptions {
                score_source: cli.score_source(),
                tree_sample_eval_only: cli.tree_sample_eval_only,
            },
            teacher_shuffle_buffer_mib,
            teacher_shuffle,
            teacher_shuffle_seed: cli.teacher_shuffle_seed,
        },
        &cli.mix_sources(),
        cli.mix_seed,
    )?;

    let mut hasher = Sha256::new();
    let mut positions = 0_usize;
    let start_time = std::time::Instant::now();
    for _ in 0..args.batches {
        let (batch, buckets) = loader
            .next_batch()?
            .ok_or("dataloader ended before the requested batch count")?;
        positions += batch.n_positions;
        if args.throughput_only {
            loader.recycle((batch, buckets));
            continue;
        }
        for bucket in &buckets {
            hasher.update(bucket.to_le_bytes());
        }
        for position in 0..batch.n_positions {
            hasher.update(batch.score[position].to_le_bytes());
            hasher.update(batch.wdl[position].to_le_bytes());
            hasher.update(batch.nnz[position].to_le_bytes());
            let nnz = batch.nnz[position] as usize;
            let start = position * batch.max_active;
            for &index in &batch.stm_indices[start..start + nnz] {
                hasher.update(index.to_le_bytes());
            }
            for &index in &batch.nstm_indices[start..start + nnz] {
                hasher.update(index.to_le_bytes());
            }
        }
        loader.recycle((batch, buckets));
    }

    if args.throughput_only {
        let seconds = start_time.elapsed().as_secs_f64();
        println!(
            "batches={} positions={positions} seconds={seconds:.3} positions_per_second={:.0}",
            args.batches,
            positions as f64 / seconds,
        );
        return Ok(());
    }
    let digest = hasher.finalize();
    let digest_hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    println!(
        "digest={digest_hex} batches={} positions={positions}",
        args.batches
    );
    Ok(())
}
