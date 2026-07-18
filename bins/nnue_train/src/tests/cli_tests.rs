//! CLI 構成テスト (clap、GPU 不要)。

use std::path::PathBuf;

use clap::Parser;
use nnue_format::{ArchKind, SimpleActivation};

use crate::cli::*;
use crate::training::{
    per_group_optim_flags, reject_simple_unsupported_flags, require_simple_win_rate_model,
    validate_bucket_mode, validate_output_format,
};

use clap::CommandFactory;

/// `nnue-train <global flags...> simple` を parse して `Cli` を返す helper。
fn simple_cli(argv: &[&str]) -> Cli {
    let mut full = vec!["nnue-train"];
    full.extend_from_slice(argv);
    full.push("simple");
    Cli::try_parse_from(full).expect("simple cli should parse")
}

#[test]
fn cli_definition_is_valid() {
    // clap derive の構成 (global 引数 + 必須サブコマンド) が破綻していないこと。
    Cli::command().debug_assert();
}

/// `--ft-factorize` / `--no-ft-factorize` は global flag。任意 subcommand の後ろに
/// 付けても global 引数として parse される (層は `simple` でも `layerstack` でも同じ)。
fn cli_with_factorize(argv: &[&str]) -> Cli {
    let mut full = vec!["nnue-train", "simple"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("cli should parse")
}

/// simple が reject すべき layerstack 専用 per-group optimizer フラグの期待セット。
/// production テーブル ([`per_group_optim_flags`]) から独立に固定し、テーブル側の
/// 追加・脱落を検出する。
const SIMPLE_REJECTED_PER_GROUP: [&str; 6] = [
    "--ft-weight-decay",
    "--dense-weight-decay",
    "--bias-weight-decay",
    "--ft-lr-mult",
    "--dense-lr-mult",
    "--bias-lr-mult",
];

#[test]
fn simple_rejects_layerstack_only_global_flags() {
    // eval / threat 系は layerstack の eval・threat 経路専用 → simple では reject。
    assert!(reject_simple_unsupported_flags(&simple_cli(&["--eval-only"])).is_err());
    assert!(reject_simple_unsupported_flags(&simple_cli(&["--threat-ablate", "all"])).is_err());
    assert!(reject_simple_unsupported_flags(&simple_cli(&["--threat-norm-dump"])).is_err());
    // per-group optimizer override 6 種すべて reject (独立の期待リストで固定)。
    for name in SIMPLE_REJECTED_PER_GROUP {
        assert!(
            reject_simple_unsupported_flags(&simple_cli(&[name, "0.1"])).is_err(),
            "{name} must be rejected on the simple subcommand"
        );
    }
    // production テーブルが期待リストと完全一致することも固定する (テーブルからの
    // 脱落・順序変更・追加を検出)。
    let table: Vec<&str> = per_group_optim_flags(&simple_cli(&[]))
        .iter()
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(
        table, SIMPLE_REJECTED_PER_GROUP,
        "per_group_optim_flags table drifted from the rejected set"
    );
}

#[test]
fn simple_accepts_consumed_global_flags() {
    // 既定 simple run は reject されない。
    assert!(reject_simple_unsupported_flags(&simple_cli(&[])).is_ok());
    // --norm-loss / precision 系は simple が消費するので reject されない。
    assert!(reject_simple_unsupported_flags(&simple_cli(&["--norm-loss"])).is_ok());
    assert!(
        reject_simple_unsupported_flags(&simple_cli(&[
            "--norm-loss",
            "--norm-loss-factor",
            "1e-4"
        ]))
        .is_ok()
    );
    assert!(reject_simple_unsupported_flags(&simple_cli(&["--ft-fp16", "--all-optim"])).is_ok());
}

#[test]
fn simple_rejects_loss_wdl_requires_win_rate_model() {
    // --win-rate-model 無し = loss_wdl 経路。dense int8 clamp と非整合なので reject。
    assert!(require_simple_win_rate_model(&simple_cli(&[])).is_err());
    // WRM の出力 scale と export の scale が不一致なら reject。
    let err = require_simple_win_rate_model(&simple_cli(&["--win-rate-model"]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("--scale (290) must equal --wrm-nnue2score (600)"));
    // 一致する WRM は accept (identity 退化の設定でも受理される)。
    assert!(
        require_simple_win_rate_model(&simple_cli(&["--win-rate-model", "--scale", "600"])).is_ok()
    );
    assert!(
        require_simple_win_rate_model(&simple_cli(&[
            "--win-rate-model",
            "--scale",
            "600",
            "--wrm-in-offset",
            "0",
            "--wrm-target-offset",
            "0",
            "--wrm-in-scaling",
            "600",
            "--wrm-target-scaling",
            "600",
            "--wrm-nnue2score",
            "600",
        ]))
        .is_ok()
    );
}

/// main は `--eval-only` 等の診断フラグでは `--data` 不在でも `run_training` へ dispatch
/// する。reject が `data.expect` より前に走らないと clean error が panic に化けるため、
/// 実経路 (`run_training` → `run_simple_training`) で clean な `Err` になることを固定する。
#[cfg(feature = "gpu")]
#[test]
fn simple_eval_only_without_data_errors_cleanly() {
    let cli = Cli::try_parse_from(["nnue-train", "--eval-only", "simple"])
        .expect("--eval-only simple should parse");
    assert!(
        crate::training::run_training(&cli).is_err(),
        "--eval-only on simple without --data must return a clean Err, not panic"
    );
}

#[test]
fn ft_factorize_defaults_on_and_no_flag_disables() {
    // default は ON (flag 無し)。`--ft-factorize` は back-compat の明示 ON。
    assert!(cli_with_factorize(&[]).ft_factorize_enabled());
    assert!(cli_with_factorize(&["--ft-factorize"]).ft_factorize_enabled());
    // `--no-ft-factorize` で OFF。
    assert!(!cli_with_factorize(&["--no-ft-factorize"]).ft_factorize_enabled());
    // overrides_with: command-line 後勝ち。
    assert!(!cli_with_factorize(&["--ft-factorize", "--no-ft-factorize"]).ft_factorize_enabled());
    assert!(cli_with_factorize(&["--no-ft-factorize", "--ft-factorize"]).ft_factorize_enabled());
    // layerstack subcommand の後ろに置いても global flag として parse される (back-compat)。
    assert!(
        !Cli::try_parse_from(["nnue-train", "layerstack", "--no-ft-factorize"])
            .expect("layerstack --no-ft-factorize should parse")
            .ft_factorize_enabled()
    );
    // `--psqt` と factorizer は併用可 (PSQT 行も同じ fold を通る)。clap で衝突せず
    // parse できることだけ確認する (auto-suppress するのは `--init-from` のみ)。
    assert!(
        Cli::try_parse_from(["nnue-train", "layerstack", "--psqt"]).is_ok(),
        "--psqt coexists with the factorizer"
    );
}

#[test]
fn layerstack_subcommand_parses() {
    let cli = Cli::try_parse_from(["nnue-train", "layerstack"]).expect("layerstack subcommand");
    assert_eq!(cli.arch.kind(), ArchKind::LayerStack);
    assert_eq!(cli.output_format, OutputFormatArg::Tatara);
}

#[test]
fn bench_pos_subcommand_uses_tracked_and_local_defaults() {
    let cli = Cli::try_parse_from([
        "nnue-train",
        "bench-pos",
        "--case",
        "layerstack-fp32",
        "--case",
        "simple-halfkp-fp32",
    ])
    .expect("bench-pos subcommand should parse");
    let ArchCommand::BenchPos(args) = cli.arch else {
        panic!("expected bench-pos subcommand");
    };
    assert_eq!(args.profile, PathBuf::from("bench-pos.toml"));
    assert_eq!(args.local_config, PathBuf::from("bench-pos.local.toml"));
    assert_eq!(args.cases, ["layerstack-fp32", "simple-halfkp-fp32"]);
    assert!(!args.allow_dirty);
}

#[test]
#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
fn native_bench_subcommand_has_fixed_v1_defaults() {
    let cli = Cli::try_parse_from(["nnue-train", "native-bench"])
        .expect("native-bench subcommand should parse");
    let ArchCommand::NativeBench(args) = cli.arch else {
        panic!("expected native-bench subcommand");
    };
    assert_eq!(args.profile, NativeBenchProfileArg::V1);
    assert_eq!(args.architecture, NativeBenchArchitectureArg::All);
    assert_eq!(args.precision, NativeBenchPrecisionArg::All);
    assert_eq!(args.mode, NativeBenchModeArg::NativeOnly);
    assert_eq!(cli.batch_size, 16_384);
    assert_eq!(args.warmup_steps, 3);
    assert_eq!(args.steps, 100);
    assert_eq!(args.runs, 3);
    assert_eq!(args.device, 0);
    assert!(!args.allow_dirty);
}

#[test]
fn yaneuraou_output_format_parses_and_simple_rejects_it() {
    let cli = Cli::try_parse_from(["nnue-train", "--output-format", "yaneuraou", "layerstack"])
        .expect("YaneuraOu LayerStack output should parse");
    assert_eq!(cli.output_format, OutputFormatArg::Yaneuraou);

    let simple = simple_cli(&["--output-format", "yaneuraou"]);
    let error = reject_simple_unsupported_flags(&simple).unwrap_err();
    assert!(error.to_string().contains("only with the layerstack"));

    validate_output_format(
        OutputFormatArg::Yaneuraou,
        nnue_train::dataloader::BucketMode::KingRank9,
    )
    .expect("KingRank9 should support YaneuraOu output");
    let error = validate_output_format(
        OutputFormatArg::Yaneuraou,
        nnue_train::dataloader::BucketMode::Progress8KpAbs,
    )
    .unwrap_err();
    assert!(error.to_string().contains("--bucket-mode kingrank9"));
}

fn layerstack_args(argv: &[&str]) -> LayerstackArgs {
    let mut full = vec!["nnue-train", "layerstack"];
    full.extend_from_slice(argv);
    match Cli::try_parse_from(full)
        .expect("layerstack CLI should parse")
        .arch
    {
        ArchCommand::LayerStack(args) => args,
        ArchCommand::Simple(_) => unreachable!("layerstack subcommand was requested"),
        ArchCommand::BenchPos(_) => unreachable!("layerstack subcommand was requested"),
        #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
        ArchCommand::NativeBench(_) => unreachable!("layerstack subcommand was requested"),
    }
}

#[test]
fn kingrank9_bucket_mode_validation() {
    let valid = layerstack_args(&["--bucket-mode", "kingrank9", "--num-buckets", "9"]);
    assert!(validate_bucket_mode(&valid).is_ok());

    let wrong_count = layerstack_args(&["--bucket-mode", "kingrank9", "--num-buckets", "8"]);
    let err = validate_bucket_mode(&wrong_count).unwrap_err().to_string();
    assert!(err.contains("must be 9"), "{err}");

    let progress_coeff = layerstack_args(&[
        "--bucket-mode",
        "kingrank9",
        "--progress-coeff",
        "progress.bin",
    ]);
    let err = validate_bucket_mode(&progress_coeff)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not used"), "{err}");

    let unknown = layerstack_args(&["--bucket-mode", "unknown"]);
    let err = validate_bucket_mode(&unknown).unwrap_err().to_string();
    assert!(err.contains("unknown"), "{err}");
    assert!(err.contains("progress8kpabs"), "{err}");
    assert!(err.contains("kingrank9"), "{err}");
}

#[test]
fn simple_subcommand_parses() {
    let cli = Cli::try_parse_from(["nnue-train", "simple"]).expect("simple subcommand");
    assert_eq!(cli.arch.kind(), ArchKind::Simple);
}

#[test]
fn all_optim_meta_flag_parses_for_both_subcommands() {
    // `--all-optim` は global、`simple` / `layerstack` どちらの subcommand でも accept。
    // 個別 4 flag (`--ft-fp16` / `--fp16-opt-state` / `--ft-fp16-out` / `--tf32`) を
    // 一括 ON にする shortcut (dispatch 経路で OR 結合)。
    let cli_simple = Cli::try_parse_from(["nnue-train", "--all-optim", "simple"])
        .expect("simple should accept --all-optim");
    assert!(cli_simple.all_optim);
    assert!(matches!(cli_simple.arch, ArchCommand::Simple(_)));

    let cli_layerstack = Cli::try_parse_from(["nnue-train", "--all-optim", "layerstack"])
        .expect("layerstack should accept --all-optim");
    assert!(cli_layerstack.all_optim);
    assert!(matches!(cli_layerstack.arch, ArchCommand::LayerStack(_)));

    // global なので subcommand 後置も accept (clap の global=true 標準動作)。
    // 両 subcommand 後置 case を確認 (`global = true` の対称性保証)。
    let cli_postfix_simple = Cli::try_parse_from(["nnue-train", "simple", "--all-optim"])
        .expect("--all-optim should be accepted after `simple` (global)");
    assert!(cli_postfix_simple.all_optim);

    let cli_postfix_layerstack = Cli::try_parse_from(["nnue-train", "layerstack", "--all-optim"])
        .expect("--all-optim should be accepted after `layerstack` (global)");
    assert!(cli_postfix_layerstack.all_optim);
}

#[test]
fn ft_fp16_out_requires_ft_fp16_uses_effective_values() {
    // `--ft-fp16-out` が `--ft-fp16` を要求する制約は実効値 (`--all-optim` 含意込み)
    // で判定する。`ft_fp16_out_missing_ft_fp16(ft_fp16_out, ft_fp16, all_optim)` が
    // `true` を返すと制約違反 = error。
    //
    // arg 順: (ft_fp16_out_raw, ft_fp16_raw, all_optim)。

    // --ft-fp16-out 単独 (--ft-fp16 / --all-optim なし) → 制約違反 (error)。
    assert!(ft_fp16_out_missing_ft_fp16(true, false, false));
    // --ft-fp16-out --ft-fp16 → OK。
    assert!(!ft_fp16_out_missing_ft_fp16(true, true, false));
    // --all-optim 単独 (raw flag は両方 false) → OK (--all-optim が両方含意)。
    assert!(!ft_fp16_out_missing_ft_fp16(false, false, true));
    // --all-optim --ft-fp16-out (冗長指定) → OK。all_optim=true なら ft_fp16 も実効
    // ON のため制約は充足、helper は常に false を返す。
    assert!(!ft_fp16_out_missing_ft_fp16(true, false, true));
    // flag なし → OK (ft_fp16_out が OFF なら制約は無関係)。
    assert!(!ft_fp16_out_missing_ft_fp16(false, false, false));
    // --ft-fp16 単独 (ft_fp16_out OFF) → OK。
    assert!(!ft_fp16_out_missing_ft_fp16(false, true, false));
}

#[test]
fn subcommand_is_required() {
    // サブコマンド未指定はエラー (clap サブコマンド必須化により CLI 文字列互換は破壊)。
    assert!(Cli::try_parse_from(["nnue-train"]).is_err());
}

#[test]
fn shared_args_are_global_around_subcommand() {
    // 共有 (global) 引数は値付き / フラグ いずれもサブコマンドの後ろに置ける。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "layerstack",
        "--ft-fp16",
        "--data",
        "x.psv",
        "--batch-size",
        "4096",
    ])
    .expect("global args after subcommand");
    assert!(cli.ft_fp16);
    assert_eq!(cli.data.as_deref(), Some(std::path::Path::new("x.psv")));
    assert_eq!(cli.batch_size, 4096);
}

#[test]
fn monitor_fp16_clamps_flag_defaults_off_and_parses_global() {
    // default は false (新規 opt-in flag、`--ft-fp16-out` 経路の cap 監視 log を gate)。
    let cli =
        Cli::try_parse_from(["nnue-train", "simple"]).expect("simple subcommand should parse");
    assert!(!cli.monitor_fp16_clamps);
    // 指定すれば true、`global = true` なので subcommand 前後どちらでも accept。
    let cli_pre = Cli::try_parse_from(["nnue-train", "--monitor-fp16-clamps", "simple"])
        .expect("--monitor-fp16-clamps before subcommand");
    assert!(cli_pre.monitor_fp16_clamps);
    let cli_post = Cli::try_parse_from(["nnue-train", "layerstack", "--monitor-fp16-clamps"])
        .expect("--monitor-fp16-clamps after subcommand");
    assert!(cli_post.monitor_fp16_clamps);
}

#[test]
fn monitor_active_features_flag_defaults_off_and_parses_global() {
    // default は false (新規 opt-in flag、実 active feature 数 histogram の log を gate)。
    let cli =
        Cli::try_parse_from(["nnue-train", "simple"]).expect("simple subcommand should parse");
    assert!(!cli.monitor_active_features);
    // 指定すれば true、`global = true` なので subcommand 前後どちらでも accept。
    let cli_pre = Cli::try_parse_from(["nnue-train", "--monitor-active-features", "simple"])
        .expect("--monitor-active-features before subcommand");
    assert!(cli_pre.monitor_active_features);
    let cli_post = Cli::try_parse_from(["nnue-train", "layerstack", "--monitor-active-features"])
        .expect("--monitor-active-features after subcommand");
    assert!(cli_post.monitor_active_features);
}

#[test]
fn simple_accepts_tf32_flag() {
    // `--tf32` は LayerStack / Simple 両 subcommand で受理される (両方 cuBLAS handle
    // に同 flag を渡す opt-in)。default OFF / 渡せば ON で TF32 TC 有効化。
    let cli = Cli::try_parse_from(["nnue-train", "simple", "--tf32"])
        .expect("simple should accept --tf32");
    match cli.arch {
        ArchCommand::Simple(args) => assert!(args.tf32),
        ArchCommand::LayerStack(_) => panic!("expected Simple subcommand"),
        ArchCommand::BenchPos(_) => panic!("expected Simple subcommand"),
        #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
        ArchCommand::NativeBench(_) => panic!("expected Simple subcommand"),
    }
}

#[test]
fn layerstack_specific_arg_rejected_before_subcommand() {
    // layerstack 固有引数 (--progress-coeff) は global ではないので、
    // サブコマンドより前には置けずエラーになる。
    assert!(
        Cli::try_parse_from(["nnue-train", "--progress-coeff", "p.bin", "layerstack"]).is_err()
    );
}

#[test]
fn lr_schedule_defaults_to_step_bit_identical() {
    use nnue_train::schedule::{LrScheduler, LrSchedulerEnum, StepLR};

    // `--lr-schedule` 省略時は従来の StepLR と一致する (default step、bit-identical)。
    let cli = Cli::try_parse_from(["nnue-train", "layerstack"]).expect("layerstack");
    assert_eq!(cli.lr_schedule, LrScheduleArg::Step);
    let sched = crate::training::build_lr_scheduler(&cli, None).expect("build step scheduler");
    assert!(matches!(sched, LrSchedulerEnum::Step(_)));

    let reference = StepLR {
        start: cli.lr,
        gamma: cli.lr_gamma,
        step: cli.lr_step.max(1),
    };
    for (batch, sb) in [(0, 1), (0, 2), (3, 5), (0, 50)] {
        assert_eq!(sched.lr(batch, sb), reference.lr(batch, sb));
    }
}

#[test]
fn lr_schedule_one_cycle_builds_and_records_warmup_boundary() {
    use nnue_train::schedule::LrSchedulerEnum;

    // one-cycle は --superbatches を total として warmup 境界を解決する
    // (warmup_pct 0.2 × 10 = 2)。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "one-cycle",
        "--superbatches",
        "10",
        "--lr",
        "1e-3",
        "layerstack",
    ])
    .expect("one-cycle parse");
    let sched = crate::training::build_lr_scheduler(&cli, None).expect("build one-cycle");
    match sched {
        LrSchedulerEnum::OneCycle(oc) => {
            assert_eq!(oc.total_superbatch, 10);
            assert_eq!(oc.warmup_superbatch, 2);
            assert_eq!(oc.max_lr, 1e-3);
        }
        other => panic!("expected OneCycle, got {other}"),
    }
}

#[test]
fn lr_schedule_cosine_defaults_final_superbatch_to_superbatches() {
    use nnue_train::schedule::LrSchedulerEnum;

    // --lr-final-superbatch 省略時は --superbatches に解決される。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "cosine",
        "--superbatches",
        "400",
        "--lr-final",
        "1e-6",
        "layerstack",
    ])
    .expect("cosine parse");
    match crate::training::build_lr_scheduler(&cli, None).expect("build cosine") {
        LrSchedulerEnum::CosineDecay(c) => {
            assert_eq!(c.final_superbatch, 400);
            assert_eq!(c.final_lr, 1e-6);
        }
        other => panic!("expected CosineDecay, got {other}"),
    }
}

#[test]
fn resumed_horizon_overrides_superbatches_default_for_decay_and_one_cycle() {
    use nnue_train::schedule::LrSchedulerEnum;

    // cosine: --lr-final-superbatch 省略 + --superbatches 400 でも、resume した
    // 保存 horizon 100 が default を上書きして curve を pin する。
    let cosine = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "cosine",
        "--superbatches",
        "400",
        "layerstack",
    ])
    .expect("cosine parse");
    match crate::training::build_lr_scheduler(&cosine, Some(100)).expect("build cosine") {
        LrSchedulerEnum::CosineDecay(c) => assert_eq!(c.final_superbatch, 100),
        other => panic!("expected CosineDecay, got {other}"),
    }

    // one-cycle: 専用 horizon flag が無いので、保存 horizon が --superbatches を
    // 上書きして total を pin する。
    let one_cycle = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "one-cycle",
        "--superbatches",
        "400",
        "layerstack",
    ])
    .expect("one-cycle parse");
    match crate::training::build_lr_scheduler(&one_cycle, Some(100)).expect("build one-cycle") {
        LrSchedulerEnum::OneCycle(oc) => assert_eq!(oc.total_superbatch, 100),
        other => panic!("expected OneCycle, got {other}"),
    }
}

#[test]
fn explicit_final_superbatch_flag_wins_over_resumed_horizon() {
    use nnue_train::schedule::LrSchedulerEnum;

    // 明示した --lr-final-superbatch は resume した保存 horizon より優先される。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "linear",
        "--superbatches",
        "400",
        "--lr-final-superbatch",
        "250",
        "layerstack",
    ])
    .expect("linear parse");
    match crate::training::build_lr_scheduler(&cli, Some(100)).expect("build linear") {
        LrSchedulerEnum::LinearDecay(l) => assert_eq!(l.final_superbatch, 250),
        other => panic!("expected LinearDecay, got {other}"),
    }
}

#[test]
fn resumed_horizon_does_not_affect_step_schedule() {
    use nnue_train::schedule::{LrScheduler, LrSchedulerEnum};

    // step は horizon を持たないので、resume した保存 horizon を渡しても curve は
    // 不変 (--lr-step / --lr-gamma のみで決まる)。
    let cli = Cli::try_parse_from(["nnue-train", "--lr-schedule", "step", "layerstack"])
        .expect("step parse");
    let with_horizon =
        crate::training::build_lr_scheduler(&cli, Some(100)).expect("build step (resumed)");
    let without = crate::training::build_lr_scheduler(&cli, None).expect("build step (fresh)");
    assert!(matches!(with_horizon, LrSchedulerEnum::Step(_)));
    for (batch, sb) in [(0, 1), (0, 5), (3, 50), (0, 400)] {
        assert_eq!(with_horizon.lr(batch, sb), without.lr(batch, sb));
    }
}

#[test]
fn lr_schedule_drop_and_linear_build_expected_variants() {
    use nnue_train::schedule::LrSchedulerEnum;

    let drop = Cli::try_parse_from(["nnue-train", "--lr-schedule", "drop", "layerstack"])
        .expect("drop parse");
    assert!(matches!(
        crate::training::build_lr_scheduler(&drop, None).expect("build drop"),
        LrSchedulerEnum::Drop(_)
    ));

    let linear = Cli::try_parse_from(["nnue-train", "--lr-schedule", "linear", "layerstack"])
        .expect("linear parse");
    assert!(matches!(
        crate::training::build_lr_scheduler(&linear, None).expect("build linear"),
        LrSchedulerEnum::LinearDecay(_)
    ));
}

#[test]
fn lr_warmup_steps_wraps_in_warmup() {
    use nnue_train::schedule::LrSchedulerEnum;

    let cli = Cli::try_parse_from(["nnue-train", "--lr-warmup-steps", "200", "layerstack"])
        .expect("warmup-steps parse");
    assert!(matches!(
        crate::training::build_lr_scheduler(&cli, None).expect("build warmup"),
        LrSchedulerEnum::Warmup(_)
    ));
}

#[test]
fn lr_warmup_steps_rejected_with_one_cycle() {
    // one-cycle は自前の warmup を持つので --lr-warmup-steps との併用は reject。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "one-cycle",
        "--lr-warmup-steps",
        "200",
        "layerstack",
    ])
    .expect("parse ok");
    assert!(crate::training::build_lr_scheduler(&cli, None).is_err());
}

#[test]
fn lr_schedule_exponential_rejects_zero_final() {
    // exponential は (final/initial)^lambda の幾何補間のため final > 0 を要求。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "exponential",
        "--lr-final",
        "0",
        "layerstack",
    ])
    .expect("parse ok");
    assert!(crate::training::build_lr_scheduler(&cli, None).is_err());
}

#[test]
fn lr_schedule_one_cycle_rejects_div_factor_below_one() {
    // div_factor < 1 だと initial_lr = max_lr / div_factor が peak --lr を超え、
    // warmup が下りになるので reject。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "one-cycle",
        "--lr-div-factor",
        "0.5",
        "layerstack",
    ])
    .expect("parse ok");
    assert!(crate::training::build_lr_scheduler(&cli, None).is_err());
}

#[test]
fn lr_schedule_one_cycle_rejects_out_of_range_warmup_pct() {
    let cli = Cli::try_parse_from([
        "nnue-train",
        "--lr-schedule",
        "one-cycle",
        "--lr-warmup-pct",
        "1.5",
        "layerstack",
    ])
    .expect("parse ok");
    assert!(crate::training::build_lr_scheduler(&cli, None).is_err());
}

#[test]
fn simple_activation_arg_parses_and_maps() {
    // `--activation` は crelu / screlu / pairwise を受理し、それぞれ
    // `SimpleActivation` variant へ写る (未知値は run_simple_training が reject)。
    for (name, want) in [
        ("crelu", SimpleActivation::CReLU),
        ("screlu", SimpleActivation::SCReLU),
        ("pairwise", SimpleActivation::Pairwise),
    ] {
        let cli = Cli::try_parse_from(["nnue-train", "simple", "--activation", name])
            .expect("simple should accept --activation");
        let act = match cli.arch {
            ArchCommand::Simple(args) => args.activation,
            ArchCommand::LayerStack(_) => panic!("expected Simple subcommand"),
            ArchCommand::BenchPos(_) => panic!("expected Simple subcommand"),
            #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
            ArchCommand::NativeBench(_) => panic!("expected Simple subcommand"),
        };
        assert_eq!(SimpleActivation::from_canonical_name(&act), Some(want));
    }
}

#[test]
fn wdl_taper_flags_parse_global_and_conflict_with_wdl() {
    use nnue_train::schedule::WdlSchedulerEnum;

    use crate::training::build_wdl_scheduler;

    // 未指定なら `--wdl` の constant lambda (default 0.0)。
    let cli = Cli::try_parse_from(["nnue-train", "layerstack"]).expect("defaults parse");
    assert!(cli.start_wdl.is_none() && cli.end_wdl.is_none());
    match build_wdl_scheduler(&cli).expect("constant scheduler") {
        WdlSchedulerEnum::Constant(c) => assert_eq!(c.value, 0.0),
        WdlSchedulerEnum::Linear(_) => panic!("expected constant WDL"),
    }

    // 両指定で linear taper。`global = true` なので subcommand 後置でも accept。
    // 端点 0.0 / 1.0 (`[0.0, 1.0]` の境界) はいずれも valid。
    let cli = Cli::try_parse_from([
        "nnue-train",
        "simple",
        "--start-wdl",
        "1.0",
        "--end-wdl",
        "0.0",
    ])
    .expect("linear taper flags parse");
    match build_wdl_scheduler(&cli).expect("linear scheduler") {
        WdlSchedulerEnum::Linear(l) => {
            assert_eq!(l.start, 1.0);
            assert_eq!(l.end, 0.0);
        }
        WdlSchedulerEnum::Constant(_) => panic!("expected linear WDL"),
    }

    // `--wdl` と `--start-wdl` / `--end-wdl` の同時指定は parse 時に reject。
    assert!(
        Cli::try_parse_from(["nnue-train", "simple", "--wdl", "0.5", "--start-wdl", "0.0"])
            .is_err()
    );
    assert!(
        Cli::try_parse_from(["nnue-train", "simple", "--wdl", "0.5", "--end-wdl", "0.5"]).is_err()
    );
}

#[test]
fn build_wdl_scheduler_rejects_partial_and_out_of_range() {
    use crate::training::build_wdl_scheduler;

    // 片方だけの指定は error (両指定で linear、未指定で constant の二択)。
    let only_start =
        Cli::try_parse_from(["nnue-train", "simple", "--start-wdl", "0.0"]).expect("parses");
    assert!(build_wdl_scheduler(&only_start).is_err());
    let only_end =
        Cli::try_parse_from(["nnue-train", "simple", "--end-wdl", "0.5"]).expect("parses");
    assert!(build_wdl_scheduler(&only_end).is_err());

    // 範囲外 (`[0.0, 1.0]` 外) は error。
    let out_of_range = Cli::try_parse_from([
        "nnue-train",
        "simple",
        "--start-wdl",
        "0.0",
        "--end-wdl",
        "1.5",
    ])
    .expect("parses");
    assert!(build_wdl_scheduler(&out_of_range).is_err());

    // 非有限値 (NaN) も error (kernel に NaN lambda を流さない)。
    let nan = Cli::try_parse_from([
        "nnue-train",
        "simple",
        "--start-wdl",
        "nan",
        "--end-wdl",
        "0.5",
    ])
    .expect("parses");
    assert!(build_wdl_scheduler(&nan).is_err());
}
