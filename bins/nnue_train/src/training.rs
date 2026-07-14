#[cfg(feature = "gpu")]
use std::path::Path;

#[cfg(feature = "gpu")]
use gpu_runtime::CudaContext;
#[cfg(feature = "gpu")]
use nnue_format::LayerStackWeights;
#[cfg(feature = "gpu")]
use nnue_format::{SimpleActivation, SimpleId, SimpleWeights};
#[cfg(feature = "gpu")]
use nnue_train::experiment::{DataInfo, ExperimentDoc, ExperimentLogger, Lineage, Params};
#[cfg(feature = "gpu")]
use nnue_train::init::{LayerStackInit, SimpleInit, WeightLayer};
#[cfg(feature = "gpu")]
use nnue_train::schedule::WdlScheduler;
#[cfg(any(feature = "gpu", test))]
use nnue_train::schedule::{LrSchedulerEnum, WdlSchedulerEnum};
#[cfg(feature = "gpu")]
use nnue_train::trainer::{LossKind, TrainingConfig};
#[cfg(feature = "gpu")]
use shogi_features::progress_kpabs::ShogiProgressKPAbs;
#[cfg(feature = "gpu")]
use shogi_features::{EffectBucketConfig, FtFactorizeMode, ThreatProfile};
#[cfg(any(feature = "gpu", test))]
use shogi_features::{FeatureSet, FeatureSetSpec};

#[cfg(any(feature = "gpu", test))]
use crate::cli::*;
#[cfg(feature = "gpu")]
use crate::{arch::*, trainer_common::PrecisionFlags, trainer_layerstack::*, trainer_simple::*};

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Default)]
struct OomFeatureConfig<'a> {
    threat_profile: Option<&'a str>,
    effect_bucket_config: Option<&'a str>,
}

/// GPU buffer 確保が OOM した時の actionable error。tunable な current config と
/// メモリ削減手段を列挙する。`gpu_runtime::is_out_of_memory` で OOM と判定した
/// ときだけ呼ぶ。`accumulator_flag` は FT 出力次元を下げる flag (LayerStack
/// `--ft-out` / Simple `--l1`)、feature 拡張 config は LayerStack のみ Some。
/// `--ft-fp16` / `--ft-fp16-out` は棋力 trade-off のため remedy には挙げず
/// (診断用に current 値だけ表示)、メモリ専用の `--fp16-opt-state` を推奨する。
#[cfg(feature = "gpu")]
fn gpu_oom_error(
    batch_size: usize,
    ft_fp16: bool,
    ft_fp16_out: bool,
    fp16_opt_state: bool,
    accumulator_flag: &str,
    accumulator_dim: usize,
    feature_config: OomFeatureConfig<'_>,
) -> Box<dyn std::error::Error> {
    use std::fmt::Write as _;
    let mut msg = String::from("GPU out of memory while allocating training buffers (");
    let _ = write!(msg, "batch-size={batch_size}");
    let _ = write!(msg, ", {accumulator_flag}={accumulator_dim}");
    let _ = write!(
        msg,
        ", ft-fp16={ft_fp16}, ft-fp16-out={ft_fp16_out}, fp16-opt-state={fp16_opt_state}"
    );
    if let Some(p) = feature_config.threat_profile {
        let _ = write!(msg, ", threat-profile={p}");
    }
    if let Some(c) = feature_config.effect_bucket_config {
        let _ = write!(msg, ", effect-bucket={c}");
    }
    msg.push_str(").\nReduce GPU memory by one or more of:\n");
    if !fp16_opt_state {
        msg.push_str("  - add `--fp16-opt-state` (optimizer の moment buffer を f16 化し約半減)\n");
    }
    let _ = writeln!(
        msg,
        "  - lower `{accumulator_flag}` (FT 出力次元。buffer はこれに概ね線形)"
    );
    if matches!(feature_config.threat_profile, Some(p) if p != "off") {
        msg.push_str(
            "  - smaller `--threat-profile` (dims 降順: full > same-class > same-class-major-pawn > cross-side > step-attacker、または off)\n",
        );
    }
    if matches!(feature_config.effect_bucket_config, Some(c) if c != "off") {
        msg.push_str("  - smaller `--effect-bucket` (2x2 uses fewer FT rows than 3x3, or off)\n");
    }
    msg.push_str("  - smaller `--batch-size`");
    msg.into()
}

#[cfg(any(feature = "gpu", test))]
#[derive(Clone, Copy)]
struct SharedPrecisionFlags {
    ft_fp16: bool,
    ft_fp16_out: bool,
    fp16_opt_state: bool,
    tf32: bool,
}

#[cfg(any(feature = "gpu", test))]
struct SharedCliValidation {
    feature_set: FeatureSetSpec,
    precision: SharedPrecisionFlags,
}

#[cfg(feature = "gpu")]
fn parse_effect_bucket_config(
    name: &str,
) -> Result<Option<EffectBucketConfig>, Box<dyn std::error::Error>> {
    let config = match name {
        "off" => None,
        "2x2-kingfixed" => Some(EffectBucketConfig::KINGFIXED_2X2),
        "2x2-kingbucketed" => Some(EffectBucketConfig::KINGBUCKETED_2X2),
        "3x3-kingfixed" => Some(EffectBucketConfig::KINGFIXED_3X3),
        "3x3-kingbucketed" => Some(EffectBucketConfig::KINGBUCKETED_3X3),
        _ => {
            return Err(format!(
                "--effect-bucket '{name}' is not a known config (expected one of: off, \
                 2x2-kingfixed, 2x2-kingbucketed, 3x3-kingfixed, 3x3-kingbucketed)"
            )
            .into());
        }
    };
    Ok(config)
}

#[cfg(any(feature = "gpu", test))]
impl SharedCliValidation {
    fn start_superbatch(
        &self,
        cli: &Cli,
        resumed_superbatch: Option<usize>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let start_superbatch = cli
            .start_superbatch
            .unwrap_or_else(|| resumed_superbatch.map_or(1, |sb| sb + 1));
        if start_superbatch == 0 {
            return Err("--start-superbatch must be >= 1 (1-indexed)".into());
        }
        Ok(start_superbatch)
    }
}

#[cfg(any(feature = "gpu", test))]
fn validate_shared_cli(
    cli: &Cli,
    ft_fp16_out_raw: bool,
    tf32_raw: bool,
) -> Result<SharedCliValidation, Box<dyn std::error::Error>> {
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
    // `--ft-fp16-out` は weight FP16 path の上に積む拡張なので `--ft-fp16` を要求する。
    if ft_fp16_out_missing_ft_fp16(ft_fp16_out_raw, cli.ft_fp16, cli.all_optim) {
        return Err(
            "--ft-fp16-out requires --ft-fp16 (FT activation FP16 builds on the weight \
             FP16 path)"
                .into(),
        );
    }
    // NaN / 範囲外を kernel に流さない。WDL 値の範囲は [`build_wdl_scheduler`] が検証する。
    if !(cli.lr.is_finite() && cli.lr > 0.0) {
        return Err(format!("--lr must be finite and > 0 (got {})", cli.lr).into());
    }
    if !cli.lr_gamma.is_finite() || cli.lr_gamma <= 0.0 {
        return Err(format!("--lr-gamma must be finite and > 0 (got {})", cli.lr_gamma).into());
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
            "--batch-size must be a multiple of 16 (got {}); tiled dense matmul kernels \
             require b % 16 == 0 (block_dim=256 × grid_dim=b/16)",
            cli.batch_size
        )
        .into());
    }
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

    Ok(SharedCliValidation {
        feature_set,
        precision: SharedPrecisionFlags {
            ft_fp16: cli.ft_fp16 || cli.all_optim,
            ft_fp16_out: ft_fp16_out_raw || cli.all_optim,
            fp16_opt_state: cli.fp16_opt_state || cli.all_optim,
            tf32: tf32_raw || cli.all_optim,
        },
    })
}

#[cfg(feature = "gpu")]
pub(crate) fn run_training(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // アーキ種別で host pipeline を分岐する。Simple は別 driver
    // ([`run_simple_training`]) で受け、LayerStack 側はそのまま既存の flow を継続する。
    let layerstack = match &cli.arch {
        ArchCommand::LayerStack(args) => args,
        ArchCommand::Simple(args) => return run_simple_training(cli, args),
    };

    // --data は通常学習と --eval-only --test-tail-positions では必須だが、
    // --threat-norm-dump と --eval-only --test-data は学習データを読まないので任意。
    // 必須経路では参照点で明示 error にする (None のまま誤って学習に進ませない)。
    let data = cli.data.as_ref();

    if (cli.threat_ablate.is_some() || cli.threat_norm_dump) && cli.init_from.is_none() {
        return Err(
            "--threat-ablate / --threat-norm-dump require --init-from (a quantised \
                    threat .bin to load and mask)"
                .into(),
        );
    }
    if cli.eval_only && cli.init_from.is_none() && cli.resume.is_none() {
        return Err("--eval-only requires --init-from or --resume (weights to evaluate)".into());
    }

    let shared = validate_shared_cli(cli, layerstack.ft_fp16_out, layerstack.tf32)?;
    let feature_set = shared.feature_set;

    // Threat profile の解決。`off` は base と bit-identical (None)。
    let threat_profile = match layerstack.threat_profile.as_str() {
        "off" => None,
        name => Some(ThreatProfile::from_cli(name).ok_or_else(
            || -> Box<dyn std::error::Error> {
                format!(
                    "--threat-profile '{name}' is not a known profile (expected one of: off, {})",
                    ThreatProfile::available()
                )
                .into()
            },
        )?),
    };
    let effect_bucket_config = parse_effect_bucket_config(&layerstack.effect_bucket_config)?;
    if threat_profile.is_some() && effect_bucket_config.is_some() {
        return Err("--effect-bucket is mutually exclusive with --threat-profile".into());
    }
    // PSQT shortcut は全 FT row に material prior を載せる設計で、base row 限定の
    // 境界処理が要る feature 拡張とは同時に使わない。factorizer と threat の併用は
    // threat path で解禁する。
    if threat_profile.is_some() && layerstack.psqt {
        return Err(
            "--threat-profile is mutually exclusive with --psqt for now (the PSQT shortcut \
             must be restricted to base FT rows when threat features are concatenated, which \
             is not yet wired)"
                .into(),
        );
    }
    if effect_bucket_config.is_some() && layerstack.psqt {
        return Err(
            "--effect-bucket is mutually exclusive with --psqt for now (the PSQT shortcut \
             must be restricted to effect bucket FT rows explicitly)"
                .into(),
        );
    }
    // spec の modifier 適用 (確定はこの 1 箇所)。threat と factorizer は独立に
    // 連結でき、両 ON の FT layout は `[base real | threat real | virtual piece-input rows]`。
    // effect bucket は base row 全体を bucket 数倍に展開し、factorizer は共有 mode に応じた
    // piece-input 仮想行 を後ろへ連結する。factorizer は default ON で `--no-ft-factorize` が opt-out。`--init-from` は
    // factorizer と排他で auto-suppress する (量子化 .bin は仮想行を持たないため
    // 初期化元にできない)。threat profile は init-from でも保持する (threat row は
    // .bin に書かれており初期化できる)。
    let feature_set = match (threat_profile, effect_bucket_config) {
        (Some(profile), None) => {
            println!("[train] --threat-profile {profile} → FT input extended by threat dims");
            feature_set.with_threat_profile(profile)
        }
        (None, Some(config)) => {
            println!(
                "[train] --effect-bucket {} → FT input extended by effect buckets",
                layerstack.effect_bucket_config
            );
            feature_set.with_effect_bucket_config(config)
        }
        (None, None) => feature_set,
        (Some(_), Some(_)) => {
            unreachable!("effect bucket and threat are rejected before spec construction")
        }
    };
    let feature_set = if cli.ft_factorize_enabled() {
        if cli.init_from.is_some() {
            println!(
                "[train] --init-from set → ft-factorizer disabled (a quantised .bin has no \
                 virtual factorizer rows)"
            );
            feature_set
        } else {
            let mode = if feature_set.effect_bucket_config().is_some() {
                match layerstack.ft_factorize_effect_bucket_share {
                    EffectBucketFactorizeShare::PoolBuckets => FtFactorizeMode::PoolEffectBuckets,
                    EffectBucketFactorizeShare::PerBucket => FtFactorizeMode::PerEffectBucket,
                }
            } else {
                FtFactorizeMode::Base
            };
            feature_set.with_ft_factorize_mode(mode)
        }
    } else {
        feature_set
    };

    // --- 未実装オプション値の reject ---
    if layerstack.bucket_mode != "progress8kpabs" {
        return Err(format!(
            "--bucket-mode '{}' is not implemented (only 'progress8kpabs')",
            layerstack.bucket_mode
        )
        .into());
    }
    // per-group override flags は wd / lr_mult とも (指定時) finite かつ >= 0。lr_mult=0
    // はその group の radam 更新を無効化する opt-in (clamp と norm loss apply は lr_mult
    // 非依存に掛かる)、bias wd=0 と同様に許容する。
    for (name, v) in per_group_optim_flags(cli) {
        if let Some(v) = v
            && (!v.is_finite() || v < 0.0)
        {
            return Err(format!("{name} must be finite and >= 0 (got {v})").into());
        }
    }
    if cli.norm_loss && (!cli.norm_loss_factor.is_finite() || cli.norm_loss_factor < 0.0) {
        return Err(format!(
            "--norm-loss-factor must be finite and >= 0 (got {})",
            cli.norm_loss_factor
        )
        .into());
    }
    // loss kernel の選択: --win-rate-model → loss_wrm、未指定 → loss_wdl。
    let loss = if cli.win_rate_model {
        build_wrm_loss(cli)?
    } else {
        if !(cli.scale.is_finite() && cli.scale > 0.0) {
            return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
        }
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };
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
    // (= `l1_out >= 2`) を要求する。上限 256 は bias backward kernel の shared-mem
    // accumulator (PARTIAL) 固定容量。L1 系 tiled dense matmul kernel は出力次元を
    // 16 幅の out-tile に分割して扱うため、`l1_out` は 16 の倍数でなくてよい。
    if layerstack.l1 < 2 || layerstack.l1 > 256 {
        return Err(format!(
            "--l1 must be in [2, 256] (got {}); the L1 output reserves 1 skip dim and the \
             rest feeds L2",
            layerstack.l1
        )
        .into());
    }
    // L2 出力次元の上限 256 は bias backward kernel の shared-mem accumulator
    // (PARTIAL) 固定容量。L2 / L3 kernel は出力次元を runtime 引数で受けるため
    // `l2_out` は特定数の倍数でなくてよい。
    if layerstack.l2 < 2 || layerstack.l2 > 256 {
        return Err(format!(
            "--l2 must be in [2, 256] (got {}); it is the L2 per-bucket dense output width",
            layerstack.l2
        )
        .into());
    }
    // bucket 数の下限 2 は progress binning が意味を持つ最小値。上限 9 は L2 / L3
    // per-bucket weight backward kernel の固定 9-register accumulator 容量
    // (`MAX_SUPPORTED_NUM_BUCKETS`)。larger N would need the per-bucket weight
    // backward kernels' register fan-out to be generalised.
    if !(2..=MAX_SUPPORTED_NUM_BUCKETS).contains(&layerstack.num_buckets) {
        return Err(format!(
            "--num-buckets must be in [2, {MAX_SUPPORTED_NUM_BUCKETS}] (got {}); larger N \
             requires the per-bucket weight backward kernels to be generalised",
            layerstack.num_buckets
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

    // norm-dump は load した threat FT 重みの host 側 L2 分解だけで GPU を要さない。
    // CUDA context / GpuTrainer の構築前に load → dump → return し、GPU 非搭載の
    // ホストでも回せるようにする (--threat-ablate / --eval-only は GPU で評価するため対象外)。
    if cli.threat_norm_dump {
        let init = cli
            .init_from
            .as_ref()
            .ok_or("--threat-norm-dump requires --init-from")?;
        let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
        let weights = LayerStackWeights::load_quantised_with_psqt(
            &mut reader,
            feature_set,
            layerstack.ft_out,
            layerstack.l1,
            layerstack.l2,
            layerstack.num_buckets,
            layerstack.psqt,
        )?;
        crate::threat_ablate::norm_dump(&weights, layerstack.ft_out);
        return Ok(());
    }

    let ctx = CudaContext::new(0)?;
    println!("[train] CUDA context ready, building GpuTrainer (LayerStack)...");
    // `--all-optim` は 4 risky 速度 flag を一括 ON にする shortcut (個別 flag と OR)。
    // 実効値は起動時 log に展開出力し reproducibility 確保。
    let SharedPrecisionFlags {
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        tf32,
    } = shared.precision;
    if cli.all_optim {
        println!(
            "[train] --all-optim → ft_fp16={ft_fp16} ft_fp16_out={ft_fp16_out} \
             fp16_opt_state={fp16_opt_state} tf32={tf32}"
        );
    }
    let norm_loss_factor = if cli.norm_loss {
        println!(
            "[train] norm loss active (factor = {})",
            cli.norm_loss_factor
        );
        Some(cli.norm_loss_factor)
    } else {
        None
    };
    // PSQT shortcut の初期 weight (`--psqt` 有効時のみ確保)。`zeroed` は全 0、`material`
    // は piece centipawn / out_scaling で全 bucket 同値を書く (Stockfish 系の prior)。
    // out_scaling 規約: WRM 有効時は `wrm_nnue2score` (= net_output が logit(WRM(cp)) の
    // domain、PSQT も同 scale で寄与する)、無効時は `scale` (= sigmoid 経路の cp → logit
    // 変換係数)。
    let psqt_init_vec: Option<Vec<f32>> = if layerstack.psqt {
        let n = feature_set.ft_in() * layerstack.num_buckets;
        let vec = match layerstack.psqt_init {
            PsqtInit::Zeroed => vec![0.0_f32; n],
            PsqtInit::Material => {
                let out_scaling = if cli.win_rate_model {
                    cli.wrm_nnue2score
                } else {
                    cli.scale
                };
                if !(out_scaling.is_finite() && out_scaling > 0.0) {
                    return Err(format!(
                        "--psqt-init material requires a positive out_scaling \
                         (got {} from {})",
                        out_scaling,
                        if cli.win_rate_model {
                            "--wrm-nnue2score"
                        } else {
                            "--scale"
                        }
                    )
                    .into());
                }
                shogi_features::psqt_material_values(
                    &feature_set,
                    layerstack.num_buckets,
                    out_scaling,
                )
            }
        };
        println!(
            "[train] PSQT shortcut: enabled (init={:?}, out_scaling={})",
            layerstack.psqt_init,
            if cli.win_rate_model {
                cli.wrm_nnue2score
            } else {
                cli.scale
            }
        );
        Some(vec)
    } else {
        println!("[train] PSQT shortcut: disabled");
        None
    };

    let init_spec = build_layerstack_init_spec(cli);
    // optimizer の param-group (ft / dense / bias) ごとの weight_decay と lr_mult を
    // CLI から resolve する。per-group flag 未指定の group は大域 --weight-decay と
    // lr_mult=1.0 にフォールバック → 全 flag 未指定なら従来挙動と bit-identical。
    let optim_groups = OptimGroupConfig::resolve(
        cli.weight_decay,
        cli.ft_weight_decay,
        cli.dense_weight_decay,
        cli.bias_weight_decay,
        cli.ft_lr_mult,
        cli.dense_lr_mult,
        cli.bias_lr_mult,
    );
    let per_group_recorded = per_group_optim_overridden(cli);
    if per_group_recorded {
        println!(
            "[train] per-group optim: ft(wd={}, lr_mult={}) dense(wd={}, lr_mult={}) \
             bias(wd={}, lr_mult={})",
            optim_groups.ft.weight_decay,
            optim_groups.ft.lr_mult,
            optim_groups.dense.weight_decay,
            optim_groups.dense.lr_mult,
            optim_groups.bias.weight_decay,
            optim_groups.bias.lr_mult,
        );
    }
    // workspace を batch_size 分で確保 (partial 末尾 batch は grow-only で対応)。
    let mut trainer = GpuTrainer::new(
        &ctx,
        cli.batch_size,
        layerstack.ft_out,
        layerstack.l1,
        layerstack.l2,
        layerstack.num_buckets,
        PrecisionFlags {
            tf32,
            ft_fp16,
            ft_fp16_out,
            fp16_opt_state,
        },
        feature_set,
        optim_groups,
        norm_loss_factor,
        psqt_init_vec.as_deref(),
        &init_spec,
    )
    .map_err(|e| {
        if gpu_runtime::is_out_of_memory(e.as_ref()) {
            gpu_oom_error(
                cli.batch_size,
                ft_fp16,
                ft_fp16_out,
                fp16_opt_state,
                "--ft-out",
                layerstack.ft_out,
                OomFeatureConfig {
                    threat_profile: Some(layerstack.threat_profile.as_str()),
                    effect_bucket_config: Some(layerstack.effect_bucket_config.as_str()),
                },
            )
        } else {
            e
        }
    })?;
    // resume / init-from の処理 → 開始 superbatch と (resume なら) 親 run id /
    // 保存済 LR horizon を決める。
    let (resumed_superbatch, resume_parent_id, resumed_lr_horizon): (
        Option<usize>,
        Option<String>,
        Option<usize>,
    ) = if let Some(init) = &cli.init_from {
        println!(
            "[train] injecting pretrained weights from {} (optimizer state reset)",
            init.display()
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
        let mut weights = LayerStackWeights::load_quantised_with_psqt(
            &mut reader,
            feature_set,
            layerstack.ft_out,
            layerstack.l1,
            layerstack.l2,
            layerstack.num_buckets,
            layerstack.psqt,
        )?;
        if let Some(spec) = cli.threat_ablate.as_deref() {
            let stats = crate::threat_ablate::apply(&mut weights, layerstack.ft_out, spec)
                .map_err(std::io::Error::other)?;
            println!(
                "[ablate] spec={spec} zeroed_dims={} zeroed_pairs={} (threat_dims={})",
                stats.zeroed_dims,
                stats.zeroed_pairs,
                weights.feature_set.threat_dims(),
            );
        }
        trainer.load_layerstack_weights(&weights)?;
        (None, None, None)
    } else if let Some(ckpt) = &cli.resume {
        let (sb, parent_id, lr_horizon) = trainer.load_raw_checkpoint(ckpt)?;
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
        (Some(sb), parent_id, lr_horizon)
    } else {
        (None, None, None)
    };

    // start_superbatch の決定 + 範囲チェック (1 <= start <= --superbatches)。
    let start_superbatch = shared.start_superbatch(cli, resumed_superbatch)?;
    // eval-only は学習しないので train-range 制約を課さない。これを課すと最終
    // checkpoint (start = saved_sb + 1 > --superbatches) を --resume で評価できない。
    if start_superbatch > cli.superbatches && !cli.eval_only {
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

    let lr_scheduler = build_lr_scheduler(cli, resumed_lr_horizon)?;
    let wdl_scheduler = build_wdl_scheduler(cli)?;
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
        score_clamp_abs: cli.score_clamp_abs,
        threads: cli.threads,
        test_data: cli.test_data.clone(),
        test_positions: cli.test_positions,
        test_tail_positions: cli.test_tail_positions,
        compute_bucket: true,
        num_buckets: layerstack.num_buckets,
        monitor_fp16_clamps: cli.monitor_fp16_clamps,
        monitor_active_features: cli.monitor_active_features,
    };

    // forward 用 FT weight (`--ft-fp16` の mirror / factorizer の comb) を学習
    // 開始時の `ft_w` (init / --init-from / --resume いずれか) と一度同期する。
    // 以降は optimizer (mirror) または step 末の fold (comb) が維持する。
    trainer.sync_ft_forward_weights()?;

    if cli.eval_only {
        // 通常学習は trainer::run 内の cfg.validate() が held-out 数を検証するが、
        // eval-only はそこへ到達せず HeldoutSet::load を直接呼ぶため、ここで等価の
        // 下限を弾く (test_positions==0 は 1 batch に切上げられ無言で縮退する)。
        // cfg.validate() 丸ごとは呼べない: eval-only では resume 由来の
        // start_superbatch が --superbatches を超えていても正当なため。
        // score_clamp_abs は CLI parser (i16 + range 1..) が値域を保証する。
        if cfg.test_positions == 0 {
            return Err("--eval-only requires --test-positions >= 1 (held-out batch count)".into());
        }
        let wdl_lambda = wdl_scheduler.blend(0, cfg.end_superbatch, cfg.end_superbatch);
        let set = match (&cfg.test_data, cfg.test_tail_positions) {
            (Some(test_path), None) => nnue_train::validation::HeldoutSet::load(
                test_path,
                cfg.batch_size,
                cfg.score_drop_abs,
                cfg.score_clamp_abs,
                cfg.test_positions,
                &progress,
                cfg.feature_set,
                cfg.num_buckets,
            )?,
            (None, Some(n)) => {
                if n == 0 {
                    return Err("--test-tail-positions must be >= 1".into());
                }
                let data = data.ok_or(
                    "--eval-only --test-tail-positions requires --data (the file whose tail is the held-out set)",
                )?;
                let file_size = std::fs::metadata(data)?.len();
                let tail_bytes = n.checked_mul(PSV_RECORD_BYTES).ok_or_else(|| {
                    std::io::Error::other("--test-tail-positions * PSV record size overflows u64")
                })?;
                if tail_bytes >= file_size {
                    return Err("--test-tail-positions leaves no data to evaluate".into());
                }
                nnue_train::validation::HeldoutSet::load_from_range(
                    data,
                    file_size - tail_bytes,
                    file_size,
                    cfg.batch_size,
                    cfg.score_drop_abs,
                    cfg.score_clamp_abs,
                    cfg.test_positions,
                    &progress,
                    cfg.feature_set,
                    cfg.num_buckets,
                )?
            }
            _ => {
                return Err(
                    "--eval-only requires exactly one of --test-tail-positions / --test-data"
                        .into(),
                );
            }
        };
        let report = set.evaluate(&mut trainer, wdl_lambda, loss)?;
        println!(
            "[eval-only] test_loss={:.8} test_accuracy={:.6} n_positions={} n_counted={} \
             wdl_lambda={wdl_lambda}",
            report.mean_loss, report.accuracy, report.n_positions, report.n_counted,
        );
        return Ok(());
    }

    // ここに到達 = 通常学習 (eval-only / norm-dump は上で return 済)。学習には --data が要る。
    let data = data.ok_or("training requires --data")?;

    let mut experiment = build_experiment_logger(
        cli,
        layerstack,
        feature_set,
        start_superbatch,
        resumed_superbatch,
        resume_parent_id,
        data,
        lr_scheduler.to_string(),
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
/// crate 側 `nnue_train::dataloader::PSV_RECORD_BYTES` を re-export している。
#[cfg(feature = "gpu")]
pub(crate) use nnue_train::dataloader::PSV_RECORD_BYTES;

/// LayerStack network の architecture 記述子 (FT → L1 → L2、progress N-bucket)。
/// experiment.json `params.architecture` に記録する。FT 出力次元は `--ft-out`、L1
/// 出力次元は `--l1`、L2 出力次元は `--l2`、bucket 数は `--num-buckets` で可変。
#[cfg(feature = "gpu")]
pub(crate) fn layerstack_architecture(
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
) -> String {
    format!("LayerStack-{ft_out}-{l1_out}-{l2_out}-{num_buckets}bucket")
}

/// `--lr-schedule` と関連 flag から runtime LR scheduler を構築する。`--lr` /
/// `--lr-gamma` の finite/正値検証は caller 側で済ませている前提で、ここでは
/// schedule 固有 flag (decay 終端 / one-cycle 係数 / warmup) を検証する。
/// LayerStack / Simple 両 driver が同じ配線を共有するための単一エントリポイント。
///
/// linear/cosine/exponential の終端 (`--lr-final-superbatch` 未指定時) と one-cycle
/// の total horizon は `--superbatches` から決まるため、resume で `--superbatches` を
/// 変えると同じ superbatch でも返す lr が変わる (曲線が伸縮する)。step は horizon を
/// 持たず `--superbatches` に依存しないので影響を受けない。
///
/// `resumed_horizon` は v5+ checkpoint から復元した保存済 horizon ([`crate::ckpt`])。
/// 指定時は `--superbatches` 由来の default より優先され、curve が `--superbatches`
/// から独立に再現される。優先順位は [`resolve_lr_horizon`] 参照。
#[cfg(any(feature = "gpu", test))]
pub(crate) fn build_lr_scheduler(
    cli: &Cli,
    resumed_horizon: Option<usize>,
) -> Result<LrSchedulerEnum, Box<dyn std::error::Error>> {
    use nnue_train::schedule::*;

    let lr = cli.lr;
    // linear / cosine / exponential の減衰終端 superbatch。優先順位は explicit
    // --lr-final-superbatch > resume した保存 horizon > --superbatches。horizon を
    // 使う schedule の arm 内でのみ解決する (horizon を持たない step / constant /
    // drop で resume horizon の note を出さないため)。
    let decay_horizon =
        || resolve_lr_horizon(cli.lr_final_superbatch, resumed_horizon, cli.superbatches);

    let base = match cli.lr_schedule {
        LrScheduleArg::Step => LrSchedulerEnum::Step(StepLR {
            start: lr,
            gamma: cli.lr_gamma,
            step: cli.lr_step.max(1),
        }),
        LrScheduleArg::Constant => LrSchedulerEnum::Constant(ConstantLR { value: lr }),
        LrScheduleArg::Drop => LrSchedulerEnum::Drop(DropLR {
            start: lr,
            gamma: cli.lr_gamma,
            drop: cli.lr_step,
        }),
        LrScheduleArg::Linear => {
            let final_superbatch = decay_horizon();
            validate_decay(cli.lr_final, final_superbatch, false)?;
            LrSchedulerEnum::LinearDecay(LinearDecayLR {
                initial_lr: lr,
                final_lr: cli.lr_final,
                final_superbatch,
            })
        }
        LrScheduleArg::Cosine => {
            let final_superbatch = decay_horizon();
            validate_decay(cli.lr_final, final_superbatch, false)?;
            LrSchedulerEnum::CosineDecay(CosineDecayLR {
                initial_lr: lr,
                final_lr: cli.lr_final,
                final_superbatch,
            })
        }
        LrScheduleArg::Exponential => {
            let final_superbatch = decay_horizon();
            validate_decay(cli.lr_final, final_superbatch, true)?;
            LrSchedulerEnum::ExponentialDecay(ExponentialDecayLR {
                initial_lr: lr,
                final_lr: cli.lr_final,
                final_superbatch,
            })
        }
        LrScheduleArg::OneCycle => {
            if !cli.lr_warmup_pct.is_finite() || !(0.0..=1.0).contains(&cli.lr_warmup_pct) {
                return Err(format!(
                    "--lr-warmup-pct must be finite and in [0.0, 1.0] (got {})",
                    cli.lr_warmup_pct
                )
                .into());
            }
            if !(cli.lr_div_factor.is_finite() && cli.lr_div_factor >= 1.0) {
                return Err(format!(
                    "--lr-div-factor must be finite and >= 1 (got {}); the initial LR is \
                     --lr / --lr-div-factor and must not exceed the peak --lr",
                    cli.lr_div_factor
                )
                .into());
            }
            if !(cli.lr_final_div_factor.is_finite() && cli.lr_final_div_factor > 0.0) {
                return Err(format!(
                    "--lr-final-div-factor must be finite and > 0 (got {})",
                    cli.lr_final_div_factor
                )
                .into());
            }
            // one-cycle は専用の horizon flag を持たないので、explicit 引数は常に
            // None。resume した保存 horizon があればそれを、無ければ --superbatches。
            let total = resolve_lr_horizon(None, resumed_horizon, cli.superbatches).max(1);
            LrSchedulerEnum::OneCycle(OneCycleLR::new(
                lr,
                cli.lr_warmup_pct,
                cli.lr_div_factor,
                cli.lr_final_div_factor,
                total,
            ))
        }
    };

    match cli.lr_warmup_steps {
        Some(_) if matches!(cli.lr_schedule, LrScheduleArg::OneCycle) => Err(
            "--lr-warmup-steps cannot be combined with --lr-schedule one-cycle \
             (one-cycle carries its own warmup)"
                .into(),
        ),
        Some(w) => Ok(base.with_warmup(w)),
        None => Ok(base),
    }
}

/// LR schedule の horizon (decay の `final_superbatch` / one-cycle の
/// `total_superbatch`) を解決する。優先順位:
///
/// 1. `explicit` — resume か否かに関わらず明示された CLI horizon flag
///    (decay の `--lr-final-superbatch`)。one-cycle は専用 flag が無いため常に `None`。
/// 2. `resumed` — v5+ checkpoint から復元した保存済 horizon。curve を
///    `--superbatches` から独立に再現させる。
/// 3. `default` — `--superbatches` 由来の fallback (新規 run / 保存 horizon 無し)。
///
/// resume 時に保存 horizon が default を上書きする / 明示 flag が保存 horizon を
/// 上書きする場合は operator 向けに 1 行 note を出す。
#[cfg(any(feature = "gpu", test))]
fn resolve_lr_horizon(explicit: Option<usize>, resumed: Option<usize>, default: usize) -> usize {
    match (explicit, resumed) {
        (Some(e), Some(saved)) => {
            println!(
                "[train] note: explicit --lr-final-superbatch {e} overrides the resumed \
                 checkpoint LR horizon {saved}"
            );
            e
        }
        (Some(e), None) => e,
        (None, Some(saved)) => {
            println!(
                "[train] using saved LR horizon {saved} from checkpoint \
                 (schedule curve stays independent of --superbatches)"
            );
            saved
        }
        (None, None) => default,
    }
}

/// linear / cosine / exponential 減衰の終端パラメータを検証する。`require_positive`
/// は exponential 用 (幾何補間 `(final/initial)^lambda` のため `final_lr > 0` を要求)。
#[cfg(any(feature = "gpu", test))]
fn validate_decay(
    final_lr: f32,
    final_superbatch: usize,
    require_positive: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !final_lr.is_finite() || final_lr < 0.0 {
        return Err(format!("--lr-final must be finite and >= 0 (got {final_lr})").into());
    }
    if require_positive && final_lr <= 0.0 {
        return Err(
            "--lr-schedule exponential requires --lr-final > 0 (geometric interpolation)".into(),
        );
    }
    if final_superbatch == 0 {
        return Err("--lr-final-superbatch must be >= 1".into());
    }
    Ok(())
}

/// 非有限な f32 (NaN / inf) を `0.0` に丸める。experiment.json の数値フィールド
/// に使う。JSON は非有限値を表現できず、混入すると serialise が丸ごと失敗して
/// 構造化ログが 1 件も書けなくなる。`--scale` は `--win-rate-model` 指定時に
/// CLI 側の finite 検証を経ないため防御する。
#[cfg(feature = "gpu")]
pub(crate) fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// per-group optimizer override flag の `(CLI 名, 指定値)` 一覧。layerstack 経路の
/// 値 validation と simple 経路の reject が同じ表を参照する (flag 追加時の漏れ防止)。
#[cfg(any(feature = "gpu", test))]
pub(crate) fn per_group_optim_flags(cli: &Cli) -> [(&'static str, Option<f32>); 6] {
    [
        ("--ft-weight-decay", cli.ft_weight_decay),
        ("--dense-weight-decay", cli.dense_weight_decay),
        ("--bias-weight-decay", cli.bias_weight_decay),
        ("--ft-lr-mult", cli.ft_lr_mult),
        ("--dense-lr-mult", cli.dense_lr_mult),
        ("--bias-lr-mult", cli.bias_lr_mult),
    ]
}

/// per-group optimizer override flag が一つでも指定されているか。`true` のとき
/// log と experiment.json に有効 per-group 値を記録する (全 `None` の既定 run では
/// 記録を省き、大域 `weight_decay` フィールドのみで足りる)。
#[cfg(feature = "gpu")]
pub(crate) fn per_group_optim_overridden(cli: &Cli) -> bool {
    per_group_optim_flags(cli).iter().any(|(_, v)| v.is_some())
}

/// simple サブコマンドが解釈しない global フラグを明示エラーで弾く (silent no-op 回避)。
///
/// simple driver ([`run_simple_training`]) と `SimpleGpuTrainer` は、global フラグの
/// 大半 (data / feature-set / LR・WDL schedule / loss・WRM パラメータ / precision
/// (`--ft-fp16` / `--fp16-opt-state` / `--all-optim`) / init (`--init-ft` / `--init-l1..3`) /
/// checkpoint / monitor / `--ft-factorize` / `--norm-loss`) を消費する。layerstack 専用で
/// simple が解釈しないのは本関数が弾く 2 群だけ:
///   - `--eval-only` / `--threat-ablate` / `--threat-norm-dump`: layerstack の eval / threat
///     経路専用。
///   - per-group optimizer override (`--ft/dense/bias-weight-decay` / `--ft/dense/bias-lr-mult`):
///     simple は単一 `weight_decay` 経路のみ。
///
/// 他の layerstack 専用 flag の enforce は別所が担う: `--optimizer` は
/// [`validate_shared_cli`] が `ranger` 以外を reject、`--init-l1f` は
/// [`build_simple_init_spec`] が reject (simple に L1f 層は無い)。
#[cfg(any(feature = "gpu", test))]
pub(crate) fn reject_simple_unsupported_flags(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.eval_only || cli.threat_ablate.is_some() || cli.threat_norm_dump {
        return Err(
            "--eval-only / --threat-ablate / --threat-norm-dump are only supported with the \
             layerstack subcommand"
                .into(),
        );
    }
    if let Some((name, _)) = per_group_optim_flags(cli).iter().find(|(_, v)| v.is_some()) {
        return Err(format!(
            "{name} is only supported with the layerstack subcommand, not the simple trainer \
             (simple uses a single --weight-decay path)"
        )
        .into());
    }
    Ok(())
}

/// `--win-rate-model` 指定時の WRM loss パラメータを検証して [`LossKind::Wrm`] を作る。
/// CLI フラグの finite / 正値チェックは利用者向けのエラーメッセージのため、
/// layerstack / simple 両 entry で共有するこの helper で前段に行う。
#[cfg(feature = "gpu")]
pub(crate) fn build_wrm_loss(cli: &Cli) -> Result<LossKind, Box<dyn std::error::Error>> {
    if !(cli.wrm_in_scaling.is_finite() && cli.wrm_in_scaling > 0.0) {
        return Err(format!(
            "--wrm-in-scaling must be finite and > 0 (got {})",
            cli.wrm_in_scaling
        )
        .into());
    }
    if !cli.wrm_in_offset.is_finite() {
        return Err(format!("--wrm-in-offset must be finite (got {})", cli.wrm_in_offset).into());
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
    // pow_exp は誤差 |err| の冪。grad は |err|^(pow_exp-1) を含むので pow_exp >= 1 が要る。
    if !(cli.loss_pow_exp.is_finite() && cli.loss_pow_exp >= 1.0) {
        return Err(format!(
            "--loss-pow-exp must be finite and >= 1 (got {})",
            cli.loss_pow_exp
        )
        .into());
    }
    // qp_asymmetry は過大評価の追加ペナルティで >= 0。<= -1 では asym <= 0 となり当該
    // 局面の loss が負・勾配が反転するため reject する。
    if !(cli.loss_qp_asymmetry.is_finite() && cli.loss_qp_asymmetry >= 0.0) {
        return Err(format!(
            "--loss-qp-asymmetry must be finite and >= 0 (got {})",
            cli.loss_qp_asymmetry
        )
        .into());
    }
    // weight boost は w1/w2 >= 0 (重み増幅用途)。w2 < 0 は weight base 0 で `0^負 = inf`
    // を生み、w1 < 0 は de-emphasis で boost の意図に反する。w1,w2 >= 0 で weight >= 1、
    // Σw >= n > 0 が保証される。
    if !(cli.loss_weight_boost_w1.is_finite() && cli.loss_weight_boost_w1 >= 0.0) {
        return Err(format!(
            "--loss-weight-boost-w1 must be finite and >= 0 (got {})",
            cli.loss_weight_boost_w1
        )
        .into());
    }
    if !(cli.loss_weight_boost_w2.is_finite() && cli.loss_weight_boost_w2 >= 0.0) {
        return Err(format!(
            "--loss-weight-boost-w2 must be finite and >= 0 (got {})",
            cli.loss_weight_boost_w2
        )
        .into());
    }
    Ok(LossKind::Wrm {
        nnue2score: cli.wrm_nnue2score,
        in_scaling: cli.wrm_in_scaling,
        in_offset: cli.wrm_in_offset,
        target_offset: cli.wrm_target_offset,
        target_scaling: cli.wrm_target_scaling,
        pow_exp: cli.loss_pow_exp,
        qp_asymmetry: cli.loss_qp_asymmetry,
        weight_boost_w1: cli.loss_weight_boost_w1,
        weight_boost_w2: cli.loss_weight_boost_w2,
    })
}

/// CLI フラグから WDL lambda scheduler を構築する。`--start-wdl` と `--end-wdl`
/// を両方指定すると `start → end` の線形 taper、いずれも未指定なら `--wdl` の
/// 一定 lambda になる。片方だけの指定は error。`--wdl` と `--start-wdl` /
/// `--end-wdl` の同時指定は clap の `conflicts_with` で parse 時に reject される。
/// すべての値が finite かつ `[0.0, 1.0]` であることを要求する (kernel に NaN /
/// 範囲外を流さない)。
#[cfg(any(feature = "gpu", test))]
pub(crate) fn build_wdl_scheduler(
    cli: &Cli,
) -> Result<WdlSchedulerEnum, Box<dyn std::error::Error>> {
    fn check(name: &str, value: f32) -> Result<f32, Box<dyn std::error::Error>> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("{name} must be finite and in [0.0, 1.0] (got {value})").into());
        }
        Ok(value)
    }

    match (cli.start_wdl, cli.end_wdl) {
        (Some(start), Some(end)) => Ok(WdlSchedulerEnum::linear(
            check("--start-wdl", start)?,
            check("--end-wdl", end)?,
        )),
        (Some(_), None) | (None, Some(_)) => {
            Err("--start-wdl and --end-wdl must be set together for a linear WDL taper".into())
        }
        (None, None) => Ok(WdlSchedulerEnum::constant(check("--wdl", cli.wdl)?)),
    }
}

/// `path` の basename を `String` で返す。file_name が取れなければ path 全体の
/// 表示文字列で代替する。
#[cfg(feature = "gpu")]
pub(crate) fn file_basename(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// tatara の git revision を best-effort で取得する。git が見つからない、
/// または git repository 外で実行された場合は `None`。working tree に未 commit
/// の変更があれば `-dirty` を付ける。
#[cfg(feature = "gpu")]
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

/// 初期化方式を experiment.json 用に要約する。override が無い既定の run、および
/// `--init-from` / `--resume` で重みが上書きされる run では `None` を返す (初期化
/// 選択が実 weight に効かないので記録しても reader を混乱させるだけ)。override が
/// あれば差し替えた層名を記す。
#[cfg(feature = "gpu")]
pub(crate) fn init_summary_for_log(cli: &Cli) -> Option<String> {
    if cli.init_from.is_some() || cli.resume.is_some() {
        return None;
    }
    let overridden: Vec<&str> = [
        ("ft", cli.init_ft.is_some()),
        ("l1", cli.init_l1.is_some()),
        ("l1f", cli.init_l1f.is_some()),
        ("l2", cli.init_l2.is_some()),
        ("l3", cli.init_l3.is_some()),
    ]
    .into_iter()
    .filter_map(|(name, set)| set.then_some(name))
    .collect();
    if overridden.is_empty() {
        return None;
    }
    Some(format!("overrides: {}", overridden.join(",")))
}

/// LayerStack の weight 初期化 spec を CLI から組み立てる (既定値 + per-layer override)。
#[cfg(feature = "gpu")]
pub(crate) fn build_layerstack_init_spec(cli: &Cli) -> LayerStackInit {
    let mut spec = LayerStackInit::default_uniform();
    if let Some(ov) = cli.init_ft {
        spec.apply_weight_override(WeightLayer::Ft, ov);
    }
    if let Some(ov) = cli.init_l1 {
        spec.apply_weight_override(WeightLayer::L1, ov);
    }
    if let Some(ov) = cli.init_l1f {
        spec.apply_weight_override(WeightLayer::L1f, ov);
    }
    if let Some(ov) = cli.init_l2 {
        spec.apply_weight_override(WeightLayer::L2, ov);
    }
    if let Some(ov) = cli.init_l3 {
        spec.apply_weight_override(WeightLayer::L3, ov);
    }
    spec
}

/// Simple の weight 初期化 spec を CLI から組み立てる。`--init-l1f` は L1f を持たない
/// Simple では error。
#[cfg(feature = "gpu")]
pub(crate) fn build_simple_init_spec(cli: &Cli) -> Result<SimpleInit, Box<dyn std::error::Error>> {
    let mut spec = SimpleInit::default_uniform();
    if let Some(ov) = cli.init_ft {
        spec.apply_weight_override(WeightLayer::Ft, ov)?;
    }
    if let Some(ov) = cli.init_l1 {
        spec.apply_weight_override(WeightLayer::L1, ov)?;
    }
    if let Some(ov) = cli.init_l1f {
        spec.apply_weight_override(WeightLayer::L1f, ov)?;
    }
    if let Some(ov) = cli.init_l2 {
        spec.apply_weight_override(WeightLayer::L2, ov)?;
    }
    if let Some(ov) = cli.init_l3 {
        spec.apply_weight_override(WeightLayer::L3, ov)?;
    }
    Ok(spec)
}

/// 学習 run の experiment.json ロガーを CLI 設定から組み立てる。書き込み先は
/// `{--output}/experiments/{id}.json`、`id` は `{net_id}-{UTC 開始時刻}`。
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gpu")]
pub(crate) fn build_experiment_logger(
    cli: &Cli,
    layerstack: &LayerstackArgs,
    feature_set: FeatureSetSpec,
    start_superbatch: usize,
    resumed_superbatch: Option<usize>,
    resume_parent_id: Option<String>,
    data: &Path,
    lr_schedule: String,
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
    // per-group override 指定時のみ experiment.json に有効 per-group 値を記録する。
    // resolve は CLI のみに依存する純関数 (`run_training` の trainer 構築と同じ入力)。
    let per_group_recorded = per_group_optim_overridden(cli);
    let optim_groups = OptimGroupConfig::resolve(
        cli.weight_decay,
        cli.ft_weight_decay,
        cli.dense_weight_decay,
        cli.bias_weight_decay,
        cli.ft_lr_mult,
        cli.dense_lr_mult,
        cli.bias_lr_mult,
    );
    let params = Params {
        architecture: layerstack_architecture(
            layerstack.ft_out,
            layerstack.l1,
            layerstack.l2,
            layerstack.num_buckets,
        ),
        feature_set: feature_set.canonical_name().to_string(),
        ft_in: feature_set.ft_in(),
        ft_factorize: feature_set.ft_factorize().then_some(true),
        l0: layerstack.ft_out,
        l1: layerstack.l1,
        l2: layerstack.l2,
        num_buckets: Some(layerstack.num_buckets),
        optimizer: cli.optimizer.clone(),
        bucket_mode: Some(layerstack.bucket_mode.clone()),
        activation: None,
        progress_coeff: layerstack.progress_coeff.as_deref().map(file_basename),
        lr: finite_or_zero(cli.lr),
        lr_gamma: finite_or_zero(cli.lr_gamma),
        lr_step: cli.lr_step.max(1),
        lr_schedule,
        batch_size: cli.batch_size,
        batches_per_superbatch: cli.batches_per_superbatch,
        superbatches: cli.superbatches,
        start_superbatch,
        wdl: finite_or_zero(cli.wdl),
        start_wdl: cli.start_wdl.map(finite_or_zero),
        end_wdl: cli.end_wdl.map(finite_or_zero),
        scale: finite_or_zero(cli.scale),
        weight_decay: finite_or_zero(cli.weight_decay),
        // per-group override 指定時のみ resolve 済の有効値を記録 (全未指定の既定 run
        // では省略、大域 weight_decay フィールドで足りる)。
        ft_weight_decay: per_group_recorded.then_some(optim_groups.ft.weight_decay),
        dense_weight_decay: per_group_recorded.then_some(optim_groups.dense.weight_decay),
        bias_weight_decay: per_group_recorded.then_some(optim_groups.bias.weight_decay),
        ft_lr_mult: per_group_recorded.then_some(optim_groups.ft.lr_mult),
        dense_lr_mult: per_group_recorded.then_some(optim_groups.dense.lr_mult),
        bias_lr_mult: per_group_recorded.then_some(optim_groups.bias.lr_mult),
        norm_loss_factor: cli
            .norm_loss
            .then_some(cli.norm_loss_factor)
            .map(finite_or_zero),
        qa: nnue_format::layerstack_weights::QA,
        qb: nnue_format::layerstack_weights::QB,
        loss_kind: if is_wrm { "wrm" } else { "sigmoid" }.to_string(),
        wrm_in_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_in_scaling)),
        wrm_in_offset: is_wrm.then(|| finite_or_zero(cli.wrm_in_offset)),
        wrm_nnue2score: is_wrm.then(|| finite_or_zero(cli.wrm_nnue2score)),
        wrm_target_offset: is_wrm.then(|| finite_or_zero(cli.wrm_target_offset)),
        wrm_target_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_target_scaling)),
        wrm_pow_exp: is_wrm.then(|| finite_or_zero(cli.loss_pow_exp)),
        wrm_qp_asymmetry: is_wrm.then(|| finite_or_zero(cli.loss_qp_asymmetry)),
        wrm_weight_boost_w1: is_wrm.then(|| finite_or_zero(cli.loss_weight_boost_w1)),
        wrm_weight_boost_w2: is_wrm.then(|| finite_or_zero(cli.loss_weight_boost_w2)),
        score_drop_abs: cli.score_drop_abs,
        score_clamp_abs: cli.score_clamp_abs.map(i32::from),
        init_from: cli.init_from.as_deref().map(file_basename),
        init_preset: init_summary_for_log(cli),
        // test_data / test_positions / test_tail_positions は対応する CLI フラグ
        // 指定時のみ Some を記録する (未指定 run の experiment.json では省略)。
        // test_data と test_tail_positions は clap conflicts_with で同時指定不能。
        test_data: cli.test_data.as_deref().map(file_basename),
        test_positions: (cli.test_data.is_some() || cli.test_tail_positions.is_some())
            .then_some(cli.test_positions),
        test_tail_positions: cli.test_tail_positions,
        // 実効値を記録 (`--all-optim` 経由で ON になった場合も true として残す、
        // raw 個別 flag が false でも experiment.json から再現可能)。
        tf32: layerstack.tf32 || cli.all_optim,
        ft_fp16: cli.ft_fp16 || cli.all_optim,
        ft_fp16_out: layerstack.ft_fp16_out || cli.all_optim,
        fp16_opt_state: cli.fp16_opt_state || cli.all_optim,
        threads: cli.threads,
    };

    let data_info = build_data_info(cli, data);

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

/// `--data` の raw record 数から `--test-tail-positions` 分を差し引いて
/// training-only な局面数を返す。両 builder (LayerStack / Simple) が同じ
/// 算出ロジックを使うための単一エントリポイント。`data` の metadata 読み
/// 出しに失敗したときは `0`、`--test-tail-positions` が raw 件数以上の場合は
/// raw record 数をそのまま返す (`trainer::run` 側で `validate` 経由 reject
/// される前提の defensive fallback)。
#[cfg(feature = "gpu")]
pub(crate) fn build_data_info(cli: &Cli, data: &Path) -> DataInfo {
    let total_records = std::fs::metadata(data)
        .map(|m| m.len() / PSV_RECORD_BYTES)
        .unwrap_or(0);
    let train_records = match cli.test_tail_positions {
        Some(n) if n < total_records => total_records - n,
        _ => total_records,
    };
    DataInfo {
        name: file_basename(data),
        positions: train_records,
        total_positions: 0,
        dataset_passes: 0.0,
    }
}

/// Simple アーキ用の experiment.json ロガーを CLI 設定から組み立てる。
/// LayerStack 用 [`build_experiment_logger`] と並ぶ Simple 用 helper で、
/// `Params` の bucket / progress / TF32 / FT-FP16 系フィールドは Simple では
/// 概念が無い (`bucket_mode` / `num_buckets` / `progress_coeff` は `None`、
/// `tf32` / `ft_fp16` / `ft_fp16_out` / `fp16_opt_state` は `false`)。
/// 量子化 multiplier (`qa` / `qb`) は活性化と `simple_weights` の固定値から決める。
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gpu")]
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
    lr_schedule: String,
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
        ft_factorize: id.feature_set.ft_factorize().then_some(true),
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
        lr_schedule,
        batch_size: cli.batch_size,
        batches_per_superbatch: cli.batches_per_superbatch,
        superbatches: cli.superbatches,
        start_superbatch,
        wdl: finite_or_zero(cli.wdl),
        start_wdl: cli.start_wdl.map(finite_or_zero),
        end_wdl: cli.end_wdl.map(finite_or_zero),
        scale: finite_or_zero(cli.scale),
        weight_decay: finite_or_zero(cli.weight_decay),
        // simple は per-group optimizer 非対応 ([`reject_simple_unsupported_flags`] が弾く)。
        ft_weight_decay: None,
        dense_weight_decay: None,
        bias_weight_decay: None,
        ft_lr_mult: None,
        dense_lr_mult: None,
        bias_lr_mult: None,
        // `--norm-loss` 有効時のみ factor を記録する (無効時 None)。
        norm_loss_factor: cli.norm_loss.then_some(cli.norm_loss_factor),
        qa: id.activation.qa(),
        qb: nnue_format::simple_weights::QB,
        loss_kind: if is_wrm { "wrm" } else { "sigmoid" }.to_string(),
        wrm_in_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_in_scaling)),
        wrm_in_offset: is_wrm.then(|| finite_or_zero(cli.wrm_in_offset)),
        wrm_nnue2score: is_wrm.then(|| finite_or_zero(cli.wrm_nnue2score)),
        wrm_target_offset: is_wrm.then(|| finite_or_zero(cli.wrm_target_offset)),
        wrm_target_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_target_scaling)),
        wrm_pow_exp: is_wrm.then(|| finite_or_zero(cli.loss_pow_exp)),
        wrm_qp_asymmetry: is_wrm.then(|| finite_or_zero(cli.loss_qp_asymmetry)),
        wrm_weight_boost_w1: is_wrm.then(|| finite_or_zero(cli.loss_weight_boost_w1)),
        wrm_weight_boost_w2: is_wrm.then(|| finite_or_zero(cli.loss_weight_boost_w2)),
        score_drop_abs: cli.score_drop_abs,
        score_clamp_abs: cli.score_clamp_abs.map(i32::from),
        init_from: cli.init_from.as_deref().map(file_basename),
        init_preset: init_summary_for_log(cli),
        test_data: cli.test_data.as_deref().map(file_basename),
        test_positions: (cli.test_data.is_some() || cli.test_tail_positions.is_some())
            .then_some(cli.test_positions),
        test_tail_positions: cli.test_tail_positions,
        // 実効値を記録 (`--all-optim` 展開込み、caller `run_simple_training` 経由)。
        tf32,
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        threads: cli.threads,
    };

    let data_info = build_data_info(cli, data);

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

/// Simple アーキの層次元 preset 文字列 (`"<l1>x2-<l2>-<l3>"`、`<l1>` = FT 出力、
/// `<l2>` / `<l3>` = 隠れ層。`--arch` の help と同表記) を
/// `(ft_out, l1_out, l2_out)` にパースする。
///
/// 例: `"256x2-32-32"` → `(256, 32, 32)`、`"1024x2-128-64"` → `(1024, 128, 64)`。
/// 形式不一致や非整数は `--arch` の不正値として `InvalidInput` で返す。
#[cfg(any(feature = "gpu", test))]
pub(crate) fn parse_simple_preset(
    s: &str,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let (head, tail) = s
        .split_once('-')
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("--arch '{s}' must look like '<l1>x2-<l2>-<l3>' (e.g. '256x2-32-32')").into()
        })?;
    let ft_out_str = head
        .strip_suffix("x2")
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': leading FT block must end with 'x2' (e.g. '256x2-32-32')").into()
        })?;
    let ft_out: usize = ft_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!(
                "--arch '{s}': '{ft_out_str}' is not a non-negative integer for the <l1> (FT) block"
            )
            .into()
        })?;
    let (l1_out_str, l2_out_str) =
        tail.split_once('-')
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("--arch '{s}': trailing block must look like '<l2>-<l3>' (got '{tail}')")
                    .into()
            })?;
    let l1_out: usize = l1_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{l1_out_str}' is not a non-negative integer for the <l2> block")
                .into()
        })?;
    let l2_out: usize = l2_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{l2_out_str}' is not a non-negative integer for the <l3> block")
                .into()
        })?;
    Ok((ft_out, l1_out, l2_out))
}

/// Simple 4 層アーキの training driver。`run_training` から `ArchCommand::Simple`
/// 分岐で呼ばれる。LayerStack 側 (`run_training` 本体) と並ぶ単独 entrypoint で、
/// trainer 構築・init_from / resume・lr / wdl スケジューラ・superbatch loop は
/// 同じ `nnue_train::trainer::run` driver を使う。
#[cfg(feature = "gpu")]
pub(crate) fn run_simple_training(
    cli: &Cli,
    simple_args: &SimpleArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    // simple が解釈しない layerstack 専用 global フラグ (eval/threat 系・per-group optim) を
    // silent no-op せず明示 reject する。`--data` 展開より前に弾く: main は `--eval-only`
    // 等では `--data` 不在でも本 driver に dispatch するため、後段の `data.expect` より前で
    // reject しないと clean error が panic に化ける。監査内容は関数 doc を参照。
    reject_simple_unsupported_flags(cli)?;
    let data = cli
        .data
        .as_ref()
        .expect("run_simple_training called with --data");

    let shared = validate_shared_cli(cli, simple_args.ft_fp16_out, simple_args.tf32)?;
    let feature_set = shared.feature_set;
    if cli.norm_loss && (!cli.norm_loss_factor.is_finite() || cli.norm_loss_factor < 0.0) {
        return Err(format!(
            "--norm-loss-factor must be finite and >= 0 (got {})",
            cli.norm_loss_factor
        )
        .into());
    }
    // 層次元の決定: --arch preset + --l1 / --l2 / --l3 override。
    let (preset_ft_out, preset_l1_out, preset_l2_out) = parse_simple_preset(&simple_args.arch)?;
    let ft_out = simple_args.l1.unwrap_or(preset_ft_out);
    let l1_out = simple_args.l2.unwrap_or(preset_l1_out);
    let l2_out = simple_args.l3.unwrap_or(preset_l2_out);
    // `SimpleGpuTrainer::new` の検査は `ft_out % 4 == 0` のみで 0 を素通しする
    // (`0 % 4 == 0`)。0 次元は層が機能しない退化アーキのまま学習が走ってしまう
    // ので、CLI で分かる error にして reject する。
    if ft_out == 0 || !ft_out.is_multiple_of(4) {
        return Err(format!(
            "Simple FT output dimension must be a positive multiple of 4 (got {ft_out}); \
             set it via --arch '<l1>x2-<l2>-<l3>' (the <l1> block) or --l1"
        )
        .into());
    }
    if l1_out == 0 || l2_out == 0 {
        return Err(format!(
            "Simple hidden layer dimensions must be >= 1 (got <l2>={l1_out}, <l3>={l2_out}); \
             set them via --arch '<l1>x2-<l2>-<l3>' or --l2 / --l3"
        )
        .into());
    }
    let activation = SimpleActivation::from_canonical_name(&simple_args.activation).ok_or_else(
        || -> Box<dyn std::error::Error> {
            format!(
                "--activation '{}' is not implemented (expected one of: crelu, screlu, pairwise)",
                simple_args.activation
            )
            .into()
        },
    )?;
    // FT factorizer は default ON で `--no-ft-factorize` が opt-out。Simple の feature set は
    // threat / effect bucket を持たないので mode は Base 固定 (piece-input 仮想行のみ)。仮想行を
    // 持つ .bin は無いため `--init-from` とは排他で auto-suppress する。量子化 .bin は仮想行を
    // 実行へ畳み込んで shape が非 factorize と同形になるので推論側の変更は不要。
    let feature_set = if cli.ft_factorize_enabled() {
        if cli.init_from.is_some() {
            println!(
                "[train] --init-from set → ft-factorizer disabled (a quantised .bin has no \
                 virtual factorizer rows)"
            );
            feature_set
        } else {
            println!(
                "[train] ft-factorizer ON (virtual piece-input rows folded into the exported .bin)"
            );
            feature_set.with_ft_factorize_mode(FtFactorizeMode::Base)
        }
    } else {
        feature_set
    };
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
        build_wrm_loss(cli)?
    } else {
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };

    // `--init-l1f` は Simple では受け付けないため CUDA 初期化より前に解決して
    // 早期 reject する (CUDA context 作成のコストを払わせない)。
    let init_spec = build_simple_init_spec(cli)?;

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
    let norm_loss_factor = if cli.norm_loss {
        println!(
            "[train] norm loss active (factor = {})",
            cli.norm_loss_factor
        );
        Some(cli.norm_loss_factor)
    } else {
        None
    };
    // `--all-optim` は 4 risky 速度 flag を一括 ON にする shortcut (個別 flag と OR)。
    // 実効値は起動時 log に展開出力し reproducibility 確保 (--all-optim だけでなく
    // どの flag が ON になったかを後で `tail train.log` で見て experiment.json の
    // 設定再現に使う)。
    let SharedPrecisionFlags {
        ft_fp16,
        ft_fp16_out,
        fp16_opt_state,
        tf32,
    } = shared.precision;
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
        norm_loss_factor,
        fv_scale,
        PrecisionFlags {
            tf32,
            ft_fp16,
            ft_fp16_out,
            fp16_opt_state,
        },
        &init_spec,
    )
    .map_err(|e| {
        if gpu_runtime::is_out_of_memory(e.as_ref()) {
            gpu_oom_error(
                cli.batch_size,
                ft_fp16,
                ft_fp16_out,
                fp16_opt_state,
                "--l1",
                ft_out,
                OomFeatureConfig::default(),
            )
        } else {
            e
        }
    })?;

    let (resumed_superbatch, resume_parent_id, resumed_lr_horizon): (
        Option<usize>,
        Option<String>,
        Option<usize>,
    ) = if let Some(init) = &cli.init_from {
        println!(
            "[train] injecting pretrained weights from {} (optimizer state reset)",
            init.display()
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
        let weights = SimpleWeights::load(&mut reader, id)?;
        trainer.load_simple_weights(&weights)?;
        (None, None, None)
    } else if let Some(ckpt) = &cli.resume {
        let (sb, parent_id, lr_horizon) = trainer.load_raw_checkpoint(ckpt)?;
        println!(
            "[train] resuming from {} at superbatch {}",
            ckpt.display(),
            sb + 1
        );
        (Some(sb), parent_id, lr_horizon)
    } else {
        (None, None, None)
    };

    // forward が読む FT weight (factorizer の comb / `--ft-fp16` mirror) を学習開始時の
    // `ft_w` (init / --init-from / --resume いずれか) から一度同期する。以降は factorizer
    // の comb は step 末の fold、`--ft-fp16` mirror は optimizer が維持する。factorizer 無効
    // かつ `--ft-fp16` 未指定なら no-op。
    trainer.sync_ft_forward_weights()?;

    let start_superbatch = shared.start_superbatch(cli, resumed_superbatch)?;
    if start_superbatch > cli.superbatches {
        return Err(format!(
            "--start-superbatch {start_superbatch} > --superbatches {} (nothing to train)",
            cli.superbatches
        )
        .into());
    }

    let lr_scheduler = build_lr_scheduler(cli, resumed_lr_horizon)?;
    let wdl_scheduler = build_wdl_scheduler(cli)?;
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
        score_clamp_abs: cli.score_clamp_abs,
        threads: cli.threads,
        test_data: cli.test_data.clone(),
        test_positions: cli.test_positions,
        test_tail_positions: cli.test_tail_positions,
        compute_bucket: false,
        // Simple アーキは bucket-less で compute_bucket=false により bucket 計算
        // 自体が skip される。値は dataloader の `num_buckets >= 1` assertion を
        // 通すための placeholder。
        num_buckets: 1,
        monitor_fp16_clamps: cli.monitor_fp16_clamps,
        monitor_active_features: cli.monitor_active_features,
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
        lr_scheduler.to_string(),
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

#[cfg(test)]
mod shared_cli_tests {
    use super::*;
    use clap::Parser;

    fn parse(extra: &[&str]) -> Cli {
        let mut argv = vec!["nnue-trainer"];
        argv.extend_from_slice(extra);
        // global init flag を subcommand の前に置く。layerstack は追加必須引数なし。
        argv.push("layerstack");
        Cli::try_parse_from(argv).expect("cli parse")
    }

    fn shared_cli_error(extra: &[&str], ft_fp16_out: bool) -> String {
        match validate_shared_cli(&parse(extra), ft_fp16_out, false) {
            Ok(_) => panic!("shared CLI validation unexpectedly accepted input"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn shared_cli_validation_rejects_invalid_common_values() {
        assert!(
            shared_cli_error(&["--feature-set", "unknown"], false)
                .contains("--feature-set 'unknown' is not a known feature set")
        );
        assert!(
            shared_cli_error(&["--optimizer", "adam"], false)
                .contains("--optimizer 'adam' is not implemented")
        );
        assert!(shared_cli_error(&["--lr", "0"], false).contains("--lr must be finite and > 0"));
        assert!(
            shared_cli_error(&["--lr-gamma", "NaN"], false)
                .contains("--lr-gamma must be finite and > 0")
        );
        assert!(
            shared_cli_error(&["--weight-decay=-1"], false)
                .contains("--weight-decay must be finite and >= 0")
        );
        assert_eq!(
            shared_cli_error(&["--threads", "0"], false),
            "--threads must be >= 1"
        );
        assert_eq!(
            shared_cli_error(&["--superbatches", "0"], false),
            "--superbatches must be >= 1"
        );
        assert!(
            shared_cli_error(&["--keep-checkpoints", "0"], false)
                .contains("--keep-checkpoints must be >= 1 when set")
        );
    }

    #[test]
    fn shared_cli_validation_pins_unified_errors() {
        let batch_error = shared_cli_error(&["--batch-size", "17"], false);
        assert!(batch_error.contains("--batch-size must be a multiple of 16 (got 17)"));
        assert!(batch_error.contains("tiled dense matmul kernels require b % 16 == 0"));

        let init_resume_error = shared_cli_error(
            &["--init-from", "base.bin", "--resume", "state.ckpt"],
            false,
        );
        assert!(init_resume_error.contains("--init-from and --resume are mutually exclusive"));
        assert!(init_resume_error.contains("--init-from injects weights"));
        assert!(init_resume_error.contains("--resume preserves it"));
    }

    #[test]
    fn shared_cli_validation_checks_fp16_dependency_and_expands_all_optim() {
        assert!(shared_cli_error(&[], true).contains("--ft-fp16-out requires --ft-fp16"));

        let shared =
            validate_shared_cli(&parse(&["--all-optim"]), false, false).expect("valid shared CLI");
        assert_eq!(shared.feature_set, FeatureSet::HalfKaHmMerged.spec());
        assert!(shared.precision.ft_fp16);
        assert!(shared.precision.ft_fp16_out);
        assert!(shared.precision.fp16_opt_state);
        assert!(shared.precision.tf32);
    }

    #[test]
    fn shared_cli_validation_resolves_start_superbatch() {
        let cli = parse(&[]);
        let shared = validate_shared_cli(&cli, false, false).expect("valid shared CLI");
        assert_eq!(shared.start_superbatch(&cli, None).unwrap(), 1);
        assert_eq!(shared.start_superbatch(&cli, Some(6)).unwrap(), 7);

        let cli = parse(&["--start-superbatch", "3"]);
        let shared = validate_shared_cli(&cli, false, false).expect("valid shared CLI");
        assert_eq!(shared.start_superbatch(&cli, Some(6)).unwrap(), 3);

        let cli = parse(&["--start-superbatch", "0"]);
        let shared = validate_shared_cli(&cli, false, false).expect("valid shared CLI");
        assert_eq!(
            shared.start_superbatch(&cli, None).unwrap_err().to_string(),
            "--start-superbatch must be >= 1 (1-indexed)"
        );
    }

    #[test]
    fn parse_simple_preset_accepts_valid_presets() {
        assert_eq!(parse_simple_preset("256x2-32-32").unwrap(), (256, 32, 32));
        assert_eq!(
            parse_simple_preset("1024x2-128-64").unwrap(),
            (1024, 128, 64)
        );
        assert_eq!(parse_simple_preset("512x2-8-64").unwrap(), (512, 8, 64));
    }

    #[test]
    fn parse_simple_preset_rejects_malformed_input() {
        // 'x2' suffix 欠落 / block 不足 / 非整数 / 空文字列。
        assert!(parse_simple_preset("256-32-32").is_err());
        assert!(parse_simple_preset("256x2-32").is_err());
        assert!(parse_simple_preset("ax2-32-32").is_err());
        assert!(parse_simple_preset("256x2-a-32").is_err());
        assert!(parse_simple_preset("256x2-32-a").is_err());
        assert!(parse_simple_preset("").is_err());
    }

    #[test]
    fn parse_simple_preset_passes_zero_dims_to_caller_validation() {
        // parse 自体は 0 を通す (型上は非負整数)。0 次元の reject は
        // `run_simple_training` の次元検証が担う。
        assert_eq!(parse_simple_preset("0x2-0-0").unwrap(), (0, 0, 0));
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(extra: &[&str]) -> Cli {
        let mut argv = vec!["nnue-trainer"];
        argv.extend_from_slice(extra);
        argv.push("layerstack");
        Cli::try_parse_from(argv).expect("cli parse")
    }

    #[test]
    fn default_run_logs_no_init_summary() {
        assert_eq!(init_summary_for_log(&parse(&[])), None);
    }

    #[test]
    fn override_logs_overridden_layers() {
        let cli = parse(&["--init-ft", "uniform:fanin"]);
        assert_eq!(init_summary_for_log(&cli).as_deref(), Some("overrides: ft"));
    }

    #[test]
    fn multiple_overrides_are_listed() {
        let cli = parse(&["--init-ft", "uniform:fanin", "--init-l2", "zero"]);
        assert_eq!(
            init_summary_for_log(&cli).as_deref(),
            Some("overrides: ft,l2")
        );
    }

    #[test]
    fn init_from_run_logs_no_summary() {
        // `--init-from` は重みを上書きするので override 指定は実 weight に効かない。
        let cli = parse(&["--init-ft", "uniform:fanin", "--init-from", "base.bin"]);
        assert_eq!(init_summary_for_log(&cli), None);
    }

    /// per-group optimizer flag の検出 (`per_group_optim_flags` /
    /// `per_group_optim_overridden`)。既定 run では未指定、いずれか 1 つでも指定すると
    /// overridden になり、simple 経路の reject が指定 flag 名を特定できる。
    #[test]
    fn per_group_optim_flag_detection() {
        let cli = parse(&[]);
        assert!(!per_group_optim_overridden(&cli));
        assert!(per_group_optim_flags(&cli).iter().all(|(_, v)| v.is_none()));

        let cli = parse(&["--bias-weight-decay", "0"]);
        assert!(per_group_optim_overridden(&cli));
        let specified: Vec<&str> = per_group_optim_flags(&cli)
            .iter()
            .filter(|(_, v)| v.is_some())
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(specified, ["--bias-weight-decay"]);

        // 6 flag 全指定で表の (CLI 名 → Cli field) 配線を値で照合する (table の
        // copy-paste 取り違えを検出)。
        let cli = parse(&[
            "--ft-weight-decay",
            "0.1",
            "--dense-weight-decay",
            "0.2",
            "--bias-weight-decay",
            "0.3",
            "--ft-lr-mult",
            "1.5",
            "--dense-lr-mult",
            "2.0",
            "--bias-lr-mult",
            "0.5",
        ]);
        let values: Vec<f32> = per_group_optim_flags(&cli)
            .iter()
            .map(|(_, v)| v.expect("all six flags set"))
            .collect();
        assert_eq!(values, [0.1, 0.2, 0.3, 1.5, 2.0, 0.5]);
    }

    #[test]
    fn gpu_oom_error_lists_relevant_remedies() {
        // fp16-opt-state off + threat on: current 値を出し、全 remedy を提示。
        let m = gpu_oom_error(
            65536,
            false,
            false,
            false,
            "--ft-out",
            1536,
            OomFeatureConfig {
                threat_profile: Some("full"),
                effect_bucket_config: Some("off"),
            },
        )
        .to_string();
        assert!(m.contains("--ft-out=1536"));
        assert!(m.contains("threat-profile=full"));
        assert!(m.contains("add `--fp16-opt-state`"));
        assert!(m.contains("lower `--ft-out`"));
        assert!(m.contains("smaller `--threat-profile`"));
        assert!(m.contains("smaller `--batch-size`"));

        // fp16-opt-state 既に on + threat off (Simple): 該当しない remedy は省く。
        let m2 = gpu_oom_error(
            4096,
            true,
            false,
            true,
            "--l1",
            256,
            OomFeatureConfig::default(),
        )
        .to_string();
        assert!(m2.contains("--l1=256"));
        assert!(!m2.contains("add `--fp16-opt-state`"));
        assert!(!m2.contains("smaller `--threat-profile`"));
        assert!(m2.contains("lower `--l1`"));
        assert!(m2.contains("smaller `--batch-size`"));
    }
}
