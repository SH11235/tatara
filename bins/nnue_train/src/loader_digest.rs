use nnue_train::dataloader::{BucketMode, BucketedPrefetchedLoader, DualLabelMode};
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
    let data = cli.data.as_deref().ok_or("loader-digest requires --data")?;
    if cli.batch_size == 0 {
        return Err("--batch-size must be >= 1".into());
    }
    if args.batches == 0 {
        return Err("--batches must be >= 1".into());
    }
    let feature_set = FeatureSet::from_canonical_name(&cli.feature_set)
        .ok_or_else(|| format!("unknown feature set: {}", cli.feature_set))?
        .spec();
    // record 長の倍数検査は loader (PsvFileLoader::open_range) が行う。
    let file_size = std::fs::metadata(data)?.len();
    let dual_label_psv = cli.dual_label_psv.map(DualLabelMode::from);
    // `auto` は machine のメモリ量に依存し digest が machine 間で比較できなくなる
    // ため、digest では 0 (直接逐次読み) に解決する。明示値はそのまま使う。
    let teacher_shuffle_buffer_mib = match cli.teacher_shuffle_buffer_mib {
        TeacherShuffleBufferMib::Explicit(mib) => mib,
        TeacherShuffleBufferMib::Auto => 0,
    };
    let teacher_shuffle = !cli.no_teacher_shuffle && teacher_shuffle_buffer_mib != 0;
    let window_note = match cli.teacher_shuffle_buffer_mib {
        TeacherShuffleBufferMib::Auto => " (auto -> 0)",
        TeacherShuffleBufferMib::Explicit(_) => "",
    };
    println!(
        "[digest] teacher window {teacher_shuffle_buffer_mib} MiB x2{window_note}, shuffle {teacher_shuffle}, seed {} | decode workers pinned to 1 (--threads is not used)",
        cli.teacher_shuffle_seed
    );
    let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
        data,
        cli.batch_size,
        cli.score_drop_abs,
        cli.score_clamp_abs,
        1,
        BucketMode::Progress8KpAbs,
        feature_set,
        false,
        1,
        file_size,
        false,
        cli.score_override.as_deref(),
        cli.score_override_mask.as_deref(),
        teacher_shuffle_buffer_mib,
        teacher_shuffle,
        cli.teacher_shuffle_seed,
        dual_label_psv,
    )?;

    let mut hasher = Sha256::new();
    let mut positions = 0_usize;
    for _ in 0..args.batches {
        let (batch, buckets) = loader
            .next_batch()?
            .ok_or("dataloader ended before the requested batch count")?;
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
        positions += batch.n_positions;
        loader.recycle((batch, buckets));
    }

    let digest = hasher.finalize();
    let digest_hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    println!(
        "digest={digest_hex} batches={} positions={positions}",
        args.batches
    );
    Ok(())
}
