use std::path::Path;

use gpu_runtime::CudaContext;
use nnue_format::LayerStackWeights;
use nnue_format::{SimpleActivation, SimpleId, SimpleWeights};
use nnue_train::experiment::{DataInfo, ExperimentDoc, ExperimentLogger, Lineage, Params};
use nnue_train::schedule::{ConstantWDL, StepLR};
use nnue_train::trainer::{LossKind, TrainingConfig};
use shogi_features::progress_kpabs::ShogiProgressKPAbs;
use shogi_features::{FeatureSet, FeatureSetSpec};

use crate::{arch::*, cli::*, trainer_layerstack::*, trainer_simple::*};

pub(crate) fn run_training(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // アーキ種別で host pipeline を分岐する。Simple は別 driver
    // ([`run_simple_training`]) で受け、LayerStack 側はそのまま既存の flow を継続する。
    let layerstack = match &cli.arch {
        ArchCommand::LayerStack(args) => args,
        ArchCommand::Simple(args) => return run_simple_training(cli, args),
    };

    let data = cli.data.as_ref().expect("run_training called with --data");

    // 入力 feature set を CLI から一度だけ決める (以降の buffer 確保 / kernel launch /
    // dataloader / checkpoint identity が参照する単一の真実源)。
    let feature_set = FeatureSet::from_canonical_name(&cli.feature_set)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            let names: Vec<&str> = FeatureSet::ALL
                .iter()
                .map(|fs| fs.canonical_name())
                .collect();
            format!(
                "--feature-set '{}' is not a known feature set (expected one of: {})",
                cli.feature_set,
                names.join(", ")
            )
            .into()
        })?
        .spec();

    // --- 未実装オプション値の reject ---
    if layerstack.bucket_mode != "progress8kpabs" {
        return Err(format!(
            "--bucket-mode '{}' is not implemented (only 'progress8kpabs')",
            layerstack.bucket_mode
        )
        .into());
    }
    if !cli.optimizer.eq_ignore_ascii_case("ranger") {
        return Err(format!(
            "--optimizer '{}' is not implemented (only 'ranger')",
            cli.optimizer
        )
        .into());
    }
    // --ft-fp16-out は weight FP16 path の上に積む拡張なので --ft-fp16 を要求する。
    // `--all-optim` は両 flag を含意するため実効値で判定する
    // ([`ft_fp16_out_missing_ft_fp16`]、`--all-optim --ft-fp16-out` を false-positive
    // reject しない)。
    if ft_fp16_out_missing_ft_fp16(layerstack.ft_fp16_out, cli.ft_fp16, cli.all_optim) {
        return Err(
            "--ft-fp16-out requires --ft-fp16 (FT activation FP16 builds on the weight \
                    FP16 path)"
                .into(),
        );
    }
    // NaN / 範囲外を kernel に流さない (TrainingConfig::validate は loss params のみ見る)。
    if !(cli.lr.is_finite() && cli.lr > 0.0) {
        return Err(format!("--lr must be finite and > 0 (got {})", cli.lr).into());
    }
    if !cli.lr_gamma.is_finite() || cli.lr_gamma <= 0.0 {
        return Err(format!("--lr-gamma must be finite and > 0 (got {})", cli.lr_gamma).into());
    }
    if !cli.wdl.is_finite() || !(0.0..=1.0).contains(&cli.wdl) {
        return Err(format!("--wdl must be finite and in [0.0, 1.0] (got {})", cli.wdl).into());
    }
    if !cli.weight_decay.is_finite() || cli.weight_decay < 0.0 {
        return Err(format!(
            "--weight-decay must be finite and >= 0 (got {})",
            cli.weight_decay
        )
        .into());
    }
    // tiled dense matmul kernels (`dense_mm_fwd_bucket_tiled_l1` / `dense_mm_fwd_tiled_l1f`
    // / `dense_mm_bwd_input_tiled` / `dense_mm_bwd_weight_*_tiled_*`) は grid 計算が
    // `b / 16` で partial tile を切り捨てる前提なので、`b % 16 != 0` だと末尾 (b mod 16)
    // position の forward / backward が 走らず loss / gradient が corrupt する。`debug_assert!`
    // は release で消えるので CLI で early reject する。
    if !cli.batch_size.is_multiple_of(16) {
        return Err(format!(
            "--batch-size must be a multiple of 16 (got {}); tiled dense matmul kernels \
             require b % 16 == 0 (block_dim=256 × grid_dim=b/16)",
            cli.batch_size
        )
        .into());
    }
    // loss kernel の選択: --win-rate-model → loss_wrm、未指定 → loss_wdl。
    let loss = if cli.win_rate_model {
        if !(cli.wrm_in_scaling.is_finite() && cli.wrm_in_scaling > 0.0) {
            return Err(format!(
                "--wrm-in-scaling must be finite and > 0 (got {})",
                cli.wrm_in_scaling
            )
            .into());
        }
        if !(cli.wrm_nnue2score.is_finite() && cli.wrm_nnue2score > 0.0) {
            return Err(format!(
                "--wrm-nnue2score must be finite and > 0 (got {})",
                cli.wrm_nnue2score
            )
            .into());
        }
        if !cli.wrm_target_offset.is_finite() {
            return Err(format!(
                "--wrm-target-offset must be finite (got {})",
                cli.wrm_target_offset
            )
            .into());
        }
        if !(cli.wrm_target_scaling.is_finite() && cli.wrm_target_scaling > 0.0) {
            return Err(format!(
                "--wrm-target-scaling must be finite and > 0 (got {})",
                cli.wrm_target_scaling
            )
            .into());
        }
        LossKind::Wrm {
            nnue2score: cli.wrm_nnue2score,
            in_scaling: cli.wrm_in_scaling,
            target_offset: cli.wrm_target_offset,
            target_scaling: cli.wrm_target_scaling,
        }
    } else {
        if !(cli.scale.is_finite() && cli.scale > 0.0) {
            return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
        }
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };
    if cli.threads == 0 {
        return Err("--threads must be >= 1".into());
    }
    if cli.init_from.is_some() && cli.resume.is_some() {
        return Err("--init-from and --resume are mutually exclusive (--init-from injects weights but resets the Ranger optimizer state; --resume preserves it)".into());
    }
    if cli.superbatches == 0 {
        return Err("--superbatches must be >= 1".into());
    }
    if let Some(0) = cli.keep_checkpoints {
        return Err(
            "--keep-checkpoints must be >= 1 when set (0 would delete every raw checkpoint)".into(),
        );
    }
    // FT 出力次元は backward の gather kernel が grid を `ft_out / 128` で launch する
    // ため 128 の倍数でなければ末尾行の勾配が計算されない。
    if layerstack.ft_out == 0 || !layerstack.ft_out.is_multiple_of(128) {
        return Err(format!(
            "--ft-out must be a positive multiple of 128 (got {})",
            layerstack.ft_out
        )
        .into());
    }
    // L1 出力次元は skip 1 dim を除いた残りが L2 入力になるので、`l1_effective >= 1`
    // (= `l1_out >= 2`) を要求する。`l1_out == 16` で従来の専用 matmul kernel、それ以外で
    // 汎用 matmul kernel が走る (どちらも任意の `l1_out >= 2` で正しい)。
    if layerstack.l1 < 2 {
        return Err(format!(
            "--l1 must be >= 2 (got {}); the L1 output reserves 1 skip dim and the rest \
             feeds L2",
            layerstack.l1
        )
        .into());
    }

    std::fs::create_dir_all(&cli.output)?;

    // progress8kpabs weights (process-global; 未指定なら zero → 全 bucket 4)
    let progress = match &layerstack.progress_coeff {
        Some(p) => {
            println!("[train] loading progress8kpabs coeff: {}", p.display());
            ShogiProgressKPAbs::load_from_bin(p).map_err(|e| -> Box<dyn std::error::Error> {
                format!("failed to load --progress-coeff {}: {e}", p.display()).into()
            })?
        }
        None => {
            eprintln!(
                "[train] note: --progress-coeff not given; all positions map to bucket 4 (sigmoid(0) = 0.5)"
            );
            ShogiProgressKPAbs
        }
    };

    let ctx = CudaContext::new(0)?;
    println!("[train] CUDA context ready, building GpuTrainer (LayerStack)...");
    // `--all-optim` は 4 risky 速度 flag を一括 ON にする shortcut (個別 flag と OR)。
    // 実効値は起動時 log に展開出力し reproducibility 確保。
    let ft_fp16 = cli.ft_fp16 || cli.all_optim;
    let fp16_opt_state = cli.fp16_opt_state || cli.all_optim;
    let ft_fp16_out = layerstack.ft_fp16_out || cli.all_optim;
    let tf32 = layerstack.tf32 || cli.all_optim;
    if cli.all_optim {
        println!(
            "[train] --all-optim → ft_fp16={ft_fp16} ft_fp16_out={ft_fp16_out} \
             fp16_opt_state={fp16_opt_state} tf32={tf32}"
        );
    }
    // workspace を batch_size 分で確保 (partial 末尾 batch は grow-only で対応)。
    let mut trainer = GpuTrainer::new(
        &ctx,
        cli.batch_size,
        layerstack.ft_out,
        layerstack.l1,
        tf32,
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        feature_set,
        cli.weight_decay,
    )?;
    // resume / init-from の処理 → 開始 superbatch と (resume なら) 親 run id を決める。
    let (resumed_superbatch, resume_parent_id): (Option<usize>, Option<String>) =
        if let Some(init) = &cli.init_from {
            println!(
                "[train] injecting pretrained weights from {} (optimizer state reset)",
                init.display()
            );
            let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
            let weights = LayerStackWeights::load_quantised(
                &mut reader,
                feature_set,
                layerstack.ft_out,
                layerstack.l1,
            )?;
            trainer.load_layerstack_weights(&weights)?;
            (None, None)
        } else if let Some(ckpt) = &cli.resume {
            let (sb, parent_id) = trainer.load_raw_checkpoint(ckpt)?;
            println!(
                "[train] resuming from {} at superbatch {}",
                ckpt.display(),
                sb + 1
            );
            if parent_id.is_none() {
                println!(
                    "[train] note: {} predates producer run id embedding; \
                     experiment.json lineage.parent_id will be omitted",
                    ckpt.display()
                );
            }
            (Some(sb), parent_id)
        } else {
            (None, None)
        };

    // start_superbatch の決定 + 範囲チェック (1 <= start <= --superbatches)。
    let start_superbatch = match cli.start_superbatch {
        Some(n) => n,
        None => match resumed_superbatch {
            Some(sb) => sb + 1,
            None => 1,
        },
    };
    if start_superbatch == 0 {
        return Err("--start-superbatch must be >= 1 (1-indexed)".into());
    }
    if start_superbatch > cli.superbatches {
        return Err(format!(
            "--start-superbatch {start_superbatch} > --superbatches {} (nothing to train); pass a larger --superbatches or a smaller start",
            cli.superbatches
        )
        .into());
    }
    if cli.resume.is_some() && cli.start_superbatch.is_some() {
        println!(
            "[train] (--start-superbatch {start_superbatch} overrides the resumed checkpoint's superbatch+1)"
        );
    }

    let lr_scheduler = StepLR {
        start: cli.lr,
        gamma: cli.lr_gamma,
        step: cli.lr_step.max(1),
    };
    let wdl_scheduler = ConstantWDL { value: cli.wdl };
    let cfg = TrainingConfig {
        net_id: cli.net_id.clone(),
        feature_set,
        output_dir: cli.output.clone(),
        start_superbatch,
        end_superbatch: cli.superbatches,
        batches_per_superbatch: cli.batches_per_superbatch,
        batch_size: cli.batch_size,
        save_rate: cli.save_rate,
        keep_raw_checkpoints: cli.keep_checkpoints,
        loss,
        score_drop_abs: cli.score_drop_abs,
        threads: cli.threads,
        test_data: cli.test_data.clone(),
        test_positions: cli.test_positions,
        compute_bucket: true,
    };

    // `--ft-fp16` の FP16 weight mirror を学習開始時の `ft_w` (init / --init-from /
    // --resume いずれか) と一度同期する。以降は optimizer が step ごとに維持する。
    trainer.sync_ft_w_h_mirror()?;

    let mut experiment = build_experiment_logger(
        cli,
        layerstack,
        feature_set,
        start_superbatch,
        resumed_superbatch,
        resume_parent_id,
        data,
    );
    println!("[train] experiment log: {}", experiment.path().display());

    let result = nnue_train::trainer::run(
        &mut trainer,
        data,
        &progress,
        &lr_scheduler,
        &wdl_scheduler,
        &cfg,
        Some(&mut experiment),
    );
    if result.is_err() {
        // run が error 終了したことを experiment.json に残す (status は "running"
        // のまま、results.interrupted を立てる)。`run` は正常終了時のみ
        // status を "completed" にする。
        experiment.mark_interrupted();
        if let Err(e) = experiment.write() {
            eprintln!(
                "[train] warning: failed to write experiment log {}: {e}",
                experiment.path().display()
            );
        }
    }
    result?;
    Ok(())
}

/// PSV 教師データ 1 局面のバイト数 (`shogi_format::PackedSfenValue` = `[u8; 40]`)。
pub(crate) const PSV_RECORD_BYTES: u64 = 40;

/// LayerStack network の architecture 記述子 (FT → L1 → L2、progress8kpabs 9 bucket)。
/// experiment.json `params.architecture` に記録する。FT 出力次元は `--ft-out`、L1
/// 出力次元は `--l1` で可変、L2 出力次元 / bucket 数は固定。
pub(crate) fn layerstack_architecture(ft_out: usize, l1_out: usize) -> String {
    format!("LayerStack-{ft_out}-{l1_out}-{L2_OUT}-{NUM_BUCKETS}bucket")
}

/// 非有限な f32 (NaN / inf) を `0.0` に丸める。experiment.json の数値フィールド
/// に使う。JSON は非有限値を表現できず、混入すると serialise が丸ごと失敗して
/// 構造化ログが 1 件も書けなくなる。`--scale` は `--win-rate-model` 指定時に
/// CLI 側の finite 検証を経ないため防御する。
pub(crate) fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// `path` の basename を `String` で返す。file_name が取れなければ path 全体の
/// 表示文字列で代替する。
pub(crate) fn file_basename(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// tatara の git revision を best-effort で取得する。git が見つからない、
/// または git repository 外で実行された場合は `None`。working tree に未 commit
/// の変更があれば `-dirty` を付ける。
pub(crate) fn git_commit() -> Option<String> {
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !rev.status.success() {
        return None;
    }
    let commit = String::from_utf8(rev.stdout).ok()?.trim().to_string();
    if commit.is_empty() {
        return None;
    }
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok();
    let is_dirty = dirty.is_some_and(|out| out.status.success() && !out.stdout.is_empty());
    Some(if is_dirty {
        format!("{commit}-dirty")
    } else {
        commit
    })
}

/// 学習 run の experiment.json ロガーを CLI 設定から組み立てる。書き込み先は
/// `{--output}/experiments/{id}.json`、`id` は `{net_id}-{UTC 開始時刻}`。
pub(crate) fn build_experiment_logger(
    cli: &Cli,
    layerstack: &LayerstackArgs,
    feature_set: FeatureSetSpec,
    start_superbatch: usize,
    resumed_superbatch: Option<usize>,
    resume_parent_id: Option<String>,
    data: &Path,
) -> ExperimentLogger {
    let start_secs = nnue_train::experiment::now_epoch_secs();
    // id 末尾に process id を付ける。同一 net_id / output で複数プロセスが同一
    // 秒に開始しても (sweep / retry script 等)、pid が異なるため experiment.json
    // の書き込み先 path が衝突せず、incremental write の上書き喪失が起きない。
    let id = format!(
        "{}-{}-{}",
        cli.net_id,
        nnue_train::experiment::format_utc_compact(start_secs),
        std::process::id()
    );
    let name = cli.experiment_name.clone().unwrap_or_else(|| {
        if cli.resume.is_some() {
            format!("{} (resume @sb{start_superbatch})", cli.net_id)
        } else {
            cli.net_id.clone()
        }
    });

    let lineage = cli.resume.as_ref().map(|ckpt| Lineage {
        // resume 元 `*.ckpt` (format version 3+) に埋め込まれた親 run の
        // experiment.json `id`。version 1/2 の `*.ckpt` には無く `None` になり、
        // その resume run の lineage は checkpoint 参照のみになる。
        parent_id: resume_parent_id.clone(),
        resumed_from_checkpoint: file_basename(ckpt),
        resumed_from_superbatch: resumed_superbatch.unwrap_or(start_superbatch.saturating_sub(1)),
    });

    let is_wrm = cli.win_rate_model;
    let params = Params {
        architecture: layerstack_architecture(layerstack.ft_out, layerstack.l1),
        feature_set: feature_set.canonical_name().to_string(),
        ft_in: feature_set.ft_in(),
        l0: layerstack.ft_out,
        l1: layerstack.l1,
        l2: L2_OUT,
        num_buckets: Some(NUM_BUCKETS),
        optimizer: cli.optimizer.clone(),
        bucket_mode: Some(layerstack.bucket_mode.clone()),
        activation: None,
        progress_coeff: layerstack.progress_coeff.as_deref().map(file_basename),
        lr: finite_or_zero(cli.lr),
        lr_gamma: finite_or_zero(cli.lr_gamma),
        lr_step: cli.lr_step.max(1),
        batch_size: cli.batch_size,
        batches_per_superbatch: cli.batches_per_superbatch,
        superbatches: cli.superbatches,
        start_superbatch,
        wdl: finite_or_zero(cli.wdl),
        scale: finite_or_zero(cli.scale),
        weight_decay: finite_or_zero(cli.weight_decay),
        qa: nnue_format::layerstack_weights::QA,
        qb: nnue_format::layerstack_weights::QB,
        loss_kind: if is_wrm { "wrm" } else { "sigmoid" }.to_string(),
        wrm_in_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_in_scaling)),
        wrm_nnue2score: is_wrm.then(|| finite_or_zero(cli.wrm_nnue2score)),
        wrm_target_offset: is_wrm.then(|| finite_or_zero(cli.wrm_target_offset)),
        wrm_target_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_target_scaling)),
        score_drop_abs: cli.score_drop_abs,
        init_from: cli.init_from.as_deref().map(file_basename),
        // test_data / test_positions は `--test-data` 指定時のみ記録する
        // (未指定 run の experiment.json では両フィールドとも省略)。
        test_data: cli.test_data.as_deref().map(file_basename),
        test_positions: cli.test_data.as_ref().map(|_| cli.test_positions),
        // 実効値を記録 (`--all-optim` 経由で ON になった場合も true として残す、
        // raw 個別 flag が false でも experiment.json から再現可能)。
        tf32: layerstack.tf32 || cli.all_optim,
        ft_fp16: cli.ft_fp16 || cli.all_optim,
        ft_fp16_out: layerstack.ft_fp16_out || cli.all_optim,
        fp16_opt_state: cli.fp16_opt_state || cli.all_optim,
        threads: cli.threads,
    };

    let dataset_positions = std::fs::metadata(data)
        .map(|m| m.len() / PSV_RECORD_BYTES)
        .unwrap_or(0);
    let data_info = DataInfo {
        name: file_basename(data),
        positions: dataset_positions,
        total_positions: 0,
        dataset_passes: 0.0,
    };

    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let json_path = cli.output.join("experiments").join(format!("{id}.json"));
    let doc = ExperimentDoc::new(
        id,
        name,
        start_secs,
        git_commit(),
        command,
        lineage,
        params,
        data_info,
    );
    ExperimentLogger::new(json_path, doc)
}

/// Simple アーキ用の experiment.json ロガーを CLI 設定から組み立てる。
/// LayerStack 用 [`build_experiment_logger`] と並ぶ Simple 用 helper で、
/// `Params` の bucket / progress / TF32 / FT-FP16 系フィールドは Simple では
/// 概念が無い (`bucket_mode` / `num_buckets` / `progress_coeff` は `None`、
/// `tf32` / `ft_fp16` / `ft_fp16_out` / `fp16_opt_state` は `false`)。
/// 量子化 multiplier (`qa` / `qb`) は活性化と `simple_weights` の固定値から決める。
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_experiment_logger_simple(
    cli: &Cli,
    id: SimpleId,
    start_superbatch: usize,
    resumed_superbatch: Option<usize>,
    resume_parent_id: Option<String>,
    data: &Path,
    ft_fp16: bool,
    ft_fp16_out: bool,
    fp16_opt_state: bool,
    tf32: bool,
) -> ExperimentLogger {
    let start_secs = nnue_train::experiment::now_epoch_secs();
    let net_id_compact = format!(
        "{}-{}-{}",
        cli.net_id,
        nnue_train::experiment::format_utc_compact(start_secs),
        std::process::id()
    );
    let name = cli.experiment_name.clone().unwrap_or_else(|| {
        if cli.resume.is_some() {
            format!("{} (resume @sb{start_superbatch})", cli.net_id)
        } else {
            cli.net_id.clone()
        }
    });

    let lineage = cli.resume.as_ref().map(|ckpt| Lineage {
        parent_id: resume_parent_id.clone(),
        resumed_from_checkpoint: file_basename(ckpt),
        resumed_from_superbatch: resumed_superbatch.unwrap_or(start_superbatch.saturating_sub(1)),
    });

    let architecture = format!(
        "simple-{}-{}x2-{}-{}-{}",
        id.feature_set.canonical_name(),
        id.ft_out,
        id.l1_out,
        id.l2_out,
        id.activation.canonical_name(),
    );

    let is_wrm = cli.win_rate_model;
    let params = Params {
        architecture,
        feature_set: id.feature_set.canonical_name().to_string(),
        ft_in: id.ft_in(),
        l0: id.ft_out,
        l1: id.l1_out,
        l2: id.l2_out,
        num_buckets: None,
        optimizer: cli.optimizer.clone(),
        bucket_mode: None,
        activation: Some(id.activation.canonical_name().to_string()),
        progress_coeff: None,
        lr: finite_or_zero(cli.lr),
        lr_gamma: finite_or_zero(cli.lr_gamma),
        lr_step: cli.lr_step.max(1),
        batch_size: cli.batch_size,
        batches_per_superbatch: cli.batches_per_superbatch,
        superbatches: cli.superbatches,
        start_superbatch,
        wdl: finite_or_zero(cli.wdl),
        scale: finite_or_zero(cli.scale),
        weight_decay: finite_or_zero(cli.weight_decay),
        qa: id.activation.qa(),
        qb: nnue_format::simple_weights::QB,
        loss_kind: if is_wrm { "wrm" } else { "sigmoid" }.to_string(),
        wrm_in_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_in_scaling)),
        wrm_nnue2score: is_wrm.then(|| finite_or_zero(cli.wrm_nnue2score)),
        wrm_target_offset: is_wrm.then(|| finite_or_zero(cli.wrm_target_offset)),
        wrm_target_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_target_scaling)),
        score_drop_abs: cli.score_drop_abs,
        init_from: cli.init_from.as_deref().map(file_basename),
        test_data: cli.test_data.as_deref().map(file_basename),
        test_positions: cli.test_data.as_ref().map(|_| cli.test_positions),
        // 実効値を記録 (`--all-optim` 展開込み、caller `run_simple_training` 経由)。
        tf32,
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        threads: cli.threads,
    };

    let dataset_positions = std::fs::metadata(data)
        .map(|m| m.len() / PSV_RECORD_BYTES)
        .unwrap_or(0);
    let data_info = DataInfo {
        name: file_basename(data),
        positions: dataset_positions,
        total_positions: 0,
        dataset_passes: 0.0,
    };

    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let json_path = cli
        .output
        .join("experiments")
        .join(format!("{net_id_compact}.json"));
    let doc = ExperimentDoc::new(
        net_id_compact,
        name,
        start_secs,
        git_commit(),
        command,
        lineage,
        params,
        data_info,
    );
    ExperimentLogger::new(json_path, doc)
}

/// Simple アーキの層次元 preset 文字列 (`"<ft_out>x2-<l1_out>-<l2_out>"`) を
/// `(ft_out, l1_out, l2_out)` にパースする。bullet-shogi 由来の表記。
///
/// 例: `"256x2-32-32"` → `(256, 32, 32)`、`"1024x2-128-64"` → `(1024, 128, 64)`。
/// 形式不一致や非整数は `--arch` の不正値として `InvalidInput` で返す。
pub(crate) fn parse_simple_preset(
    s: &str,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let (head, tail) = s
        .split_once('-')
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!(
                "--arch '{s}' must look like '<ft_out>x2-<l1_out>-<l2_out>' (e.g. '256x2-32-32')"
            )
            .into()
        })?;
    let ft_out_str = head
        .strip_suffix("x2")
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': leading FT block must end with 'x2' (e.g. '256x2-32-32')").into()
        })?;
    let ft_out: usize = ft_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{ft_out_str}' is not a non-negative integer FT dimension")
                .into()
        })?;
    let (l1_out_str, l2_out_str) =
        tail.split_once('-')
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!(
                    "--arch '{s}': trailing block must look like '<l1_out>-<l2_out>' (got '{tail}')"
                )
                .into()
            })?;
    let l1_out: usize = l1_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{l1_out_str}' is not a non-negative integer L1 dimension")
                .into()
        })?;
    let l2_out: usize = l2_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{l2_out_str}' is not a non-negative integer L2 dimension")
                .into()
        })?;
    Ok((ft_out, l1_out, l2_out))
}

/// Simple 4 層アーキの training driver。`run_training` から `ArchCommand::Simple`
/// 分岐で呼ばれる。LayerStack 側 (`run_training` 本体) と並ぶ単独 entrypoint で、
/// trainer 構築・init_from / resume・lr / wdl スケジューラ・superbatch loop は
/// 同じ `nnue_train::trainer::run` driver を使う。
pub(crate) fn run_simple_training(
    cli: &Cli,
    simple_args: &SimpleArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = cli
        .data
        .as_ref()
        .expect("run_simple_training called with --data");

    let feature_set = FeatureSet::from_canonical_name(&cli.feature_set)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            let names: Vec<&str> = FeatureSet::ALL
                .iter()
                .map(|fs| fs.canonical_name())
                .collect();
            format!(
                "--feature-set '{}' is not a known feature set (expected one of: {})",
                cli.feature_set,
                names.join(", ")
            )
            .into()
        })?
        .spec();

    if !cli.optimizer.eq_ignore_ascii_case("ranger") {
        return Err(format!(
            "--optimizer '{}' is not implemented (only 'ranger')",
            cli.optimizer
        )
        .into());
    }
    // `--ft-fp16-out` は FP16 weight mirror 経路の上に積む拡張で、`--ft-fp16` を要求する。
    // `--all-optim` は両 flag を含意するため実効値で判定する ([`ft_fp16_out_missing_ft_fp16`]、
    // `--all-optim --ft-fp16-out` の冗長指定を false-positive reject しない)。
    if ft_fp16_out_missing_ft_fp16(simple_args.ft_fp16_out, cli.ft_fp16, cli.all_optim) {
        return Err(
            "--ft-fp16-out requires --ft-fp16 (FT activation FP16 builds on the weight \
             FP16 path)"
                .into(),
        );
    }
    if !(cli.lr.is_finite() && cli.lr > 0.0) {
        return Err(format!("--lr must be finite and > 0 (got {})", cli.lr).into());
    }
    if !cli.lr_gamma.is_finite() || cli.lr_gamma <= 0.0 {
        return Err(format!("--lr-gamma must be finite and > 0 (got {})", cli.lr_gamma).into());
    }
    if !cli.wdl.is_finite() || !(0.0..=1.0).contains(&cli.wdl) {
        return Err(format!("--wdl must be finite and in [0.0, 1.0] (got {})", cli.wdl).into());
    }
    if !cli.weight_decay.is_finite() || cli.weight_decay < 0.0 {
        return Err(format!(
            "--weight-decay must be finite and >= 0 (got {})",
            cli.weight_decay
        )
        .into());
    }
    if !cli.batch_size.is_multiple_of(16) {
        return Err(format!(
            "--batch-size must be a multiple of 16 (got {})",
            cli.batch_size
        )
        .into());
    }
    if cli.threads == 0 {
        return Err("--threads must be >= 1".into());
    }
    if cli.init_from.is_some() && cli.resume.is_some() {
        return Err("--init-from and --resume are mutually exclusive".into());
    }
    if cli.superbatches == 0 {
        return Err("--superbatches must be >= 1".into());
    }
    if let Some(0) = cli.keep_checkpoints {
        return Err(
            "--keep-checkpoints must be >= 1 when set (0 would delete every raw checkpoint)".into(),
        );
    }

    // 層次元の決定: --arch preset + --l1 / --l2 / --l3 override。
    let (preset_ft_out, preset_l1_out, preset_l2_out) = parse_simple_preset(&simple_args.arch)?;
    let ft_out = simple_args.l1.unwrap_or(preset_ft_out);
    let l1_out = simple_args.l2.unwrap_or(preset_l1_out);
    let l2_out = simple_args.l3.unwrap_or(preset_l2_out);
    let activation = SimpleActivation::from_canonical_name(&simple_args.activation).ok_or_else(
        || -> Box<dyn std::error::Error> {
            format!(
                "--activation '{}' is not implemented (expected one of: crelu, screlu, pairwise)",
                simple_args.activation
            )
            .into()
        },
    )?;
    let id = SimpleId {
        feature_set,
        activation,
        ft_out,
        l1_out,
        l2_out,
    };

    // Simple は loss kind に関わらず `cli.scale` を量子化 `fv_scale` の算出で参照
    // するため、WRM 経路でも finite / 正値を要求する (LayerStack は WRM 時に scale
    // を参照しないので sigmoid 経路でのみ検証していた)。
    if !(cli.scale.is_finite() && cli.scale > 0.0) {
        return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
    }
    let loss = if cli.win_rate_model {
        if !(cli.wrm_in_scaling.is_finite() && cli.wrm_in_scaling > 0.0) {
            return Err(format!(
                "--wrm-in-scaling must be finite and > 0 (got {})",
                cli.wrm_in_scaling
            )
            .into());
        }
        if !(cli.wrm_nnue2score.is_finite() && cli.wrm_nnue2score > 0.0) {
            return Err(format!(
                "--wrm-nnue2score must be finite and > 0 (got {})",
                cli.wrm_nnue2score
            )
            .into());
        }
        if !cli.wrm_target_offset.is_finite() {
            return Err(format!(
                "--wrm-target-offset must be finite (got {})",
                cli.wrm_target_offset
            )
            .into());
        }
        if !(cli.wrm_target_scaling.is_finite() && cli.wrm_target_scaling > 0.0) {
            return Err(format!(
                "--wrm-target-scaling must be finite and > 0 (got {})",
                cli.wrm_target_scaling
            )
            .into());
        }
        LossKind::Wrm {
            nnue2score: cli.wrm_nnue2score,
            in_scaling: cli.wrm_in_scaling,
            target_offset: cli.wrm_target_offset,
            target_scaling: cli.wrm_target_scaling,
        }
    } else {
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };

    std::fs::create_dir_all(&cli.output)?;

    // Simple は bucket-aware progress を持たない: dataloader に渡す
    // `ShogiProgressKPAbs` は zero-weight default (全 position が bucket 4)、
    // TrainerBackend::train_step 内で bucket index は無視される。
    let progress = ShogiProgressKPAbs;

    let ctx = CudaContext::new(0)?;
    println!("[train] CUDA context ready, building SimpleGpuTrainer...");
    // 推論側 evaluation scale。FT 活性化出力は活性化に依らず 127-scale のため
    // fv_scale も活性化非依存 (round(FT_OUTPUT_QA × QB / 学習 scale))。`cli.scale`
    // は前段で有限・正値を保証済。
    let fv_scale = nnue_format::simple_weights::simple_fv_scale(cli.scale);
    // `--all-optim` は 4 risky 速度 flag を一括 ON にする shortcut (個別 flag と OR)。
    // 実効値は起動時 log に展開出力し reproducibility 確保 (--all-optim だけでなく
    // どの flag が ON になったかを後で `tail train.log` で見て experiment.json の
    // 設定再現に使う)。
    let ft_fp16 = cli.ft_fp16 || cli.all_optim;
    let fp16_opt_state = cli.fp16_opt_state || cli.all_optim;
    let ft_fp16_out = simple_args.ft_fp16_out || cli.all_optim;
    let tf32 = simple_args.tf32 || cli.all_optim;
    if cli.all_optim {
        println!(
            "[train] --all-optim → ft_fp16={ft_fp16} ft_fp16_out={ft_fp16_out} \
             fp16_opt_state={fp16_opt_state} tf32={tf32}"
        );
    }
    let mut trainer = SimpleGpuTrainer::new(
        &ctx,
        cli.batch_size,
        id,
        cli.weight_decay,
        fv_scale,
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        tf32,
    )?;

    let (resumed_superbatch, resume_parent_id): (Option<usize>, Option<String>) =
        if let Some(init) = &cli.init_from {
            println!(
                "[train] injecting pretrained weights from {} (optimizer state reset)",
                init.display()
            );
            let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
            let weights = SimpleWeights::load(&mut reader, id)?;
            trainer.load_simple_weights(&weights)?;
            (None, None)
        } else if let Some(ckpt) = &cli.resume {
            let (sb, parent_id) = trainer.load_raw_checkpoint(ckpt)?;
            println!(
                "[train] resuming from {} at superbatch {}",
                ckpt.display(),
                sb + 1
            );
            (Some(sb), parent_id)
        } else {
            (None, None)
        };

    // `--ft-fp16` の FP16 weight mirror を学習開始時の `ft_w` (init / --init-from /
    // --resume いずれか) と一度同期する。以降は optimizer が step ごとに維持する。
    // `--ft-fp16` 未指定なら no-op。
    trainer.sync_ft_w_h_mirror()?;

    let start_superbatch = match cli.start_superbatch {
        Some(n) => n,
        None => match resumed_superbatch {
            Some(sb) => sb + 1,
            None => 1,
        },
    };
    if start_superbatch == 0 {
        return Err("--start-superbatch must be >= 1 (1-indexed)".into());
    }
    if start_superbatch > cli.superbatches {
        return Err(format!(
            "--start-superbatch {start_superbatch} > --superbatches {} (nothing to train)",
            cli.superbatches
        )
        .into());
    }

    let lr_scheduler = StepLR {
        start: cli.lr,
        gamma: cli.lr_gamma,
        step: cli.lr_step.max(1),
    };
    let wdl_scheduler = ConstantWDL { value: cli.wdl };
    let cfg = TrainingConfig {
        net_id: cli.net_id.clone(),
        feature_set,
        output_dir: cli.output.clone(),
        start_superbatch,
        end_superbatch: cli.superbatches,
        batches_per_superbatch: cli.batches_per_superbatch,
        batch_size: cli.batch_size,
        save_rate: cli.save_rate,
        keep_raw_checkpoints: cli.keep_checkpoints,
        loss,
        score_drop_abs: cli.score_drop_abs,
        threads: cli.threads,
        test_data: cli.test_data.clone(),
        test_positions: cli.test_positions,
        compute_bucket: false,
    };

    let mut experiment = build_experiment_logger_simple(
        cli,
        id,
        start_superbatch,
        resumed_superbatch,
        resume_parent_id,
        data,
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        tf32,
    );
    println!("[train] experiment log: {}", experiment.path().display());

    let result = nnue_train::trainer::run(
        &mut trainer,
        data,
        &progress,
        &lr_scheduler,
        &wdl_scheduler,
        &cfg,
        Some(&mut experiment),
    );
    if result.is_err() {
        experiment.mark_interrupted();
        if let Err(e) = experiment.write() {
            eprintln!(
                "[train] warning: failed to write experiment log {}: {e}",
                experiment.path().display()
            );
        }
    }
    result?;
    Ok(())
}
