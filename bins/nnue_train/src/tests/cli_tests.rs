//! CLI 構成テスト (clap、GPU 不要)。

use clap::Parser;
use nnue_format::{ArchKind, SimpleActivation};

use crate::cli::*;

use clap::CommandFactory;

#[test]
fn cli_definition_is_valid() {
    // clap derive の構成 (global 引数 + 必須サブコマンド) が破綻していないこと。
    Cli::command().debug_assert();
}

#[test]
fn layerstack_subcommand_parses() {
    let cli = Cli::try_parse_from(["nnue-train", "layerstack"]).expect("layerstack subcommand");
    assert_eq!(cli.arch.kind(), ArchKind::LayerStack);
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
fn simple_accepts_tf32_flag() {
    // `--tf32` は LayerStack / Simple 両 subcommand で受理される (両方 cuBLAS handle
    // に同 flag を渡す opt-in)。default OFF / 渡せば ON で TF32 TC 有効化。
    let cli = Cli::try_parse_from(["nnue-train", "simple", "--tf32"])
        .expect("simple should accept --tf32");
    match cli.arch {
        ArchCommand::Simple(args) => assert!(args.tf32),
        ArchCommand::LayerStack(_) => panic!("expected Simple subcommand"),
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
        };
        assert_eq!(SimpleActivation::from_canonical_name(&act), Some(want));
    }
}
