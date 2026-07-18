use gpu_runtime::CudaContext;
use nnue_format::{SimpleActivation, SimpleId};
use nnue_train::{
    dataloader::BucketMode,
    init::{LayerStackInit, SimpleInit},
    optimizer::OptimizerKind,
    trainer::LossKind,
};
use shogi_features::{EffectBucketConfig, FeatureSet, FtFactorizeMode, ThreatProfile};

#[cfg(feature = "native-cuda")]
use crate::kernel_module::with_test_native_backend;
#[cfg(feature = "native-cuda")]
use crate::trainer_simple::SimpleRawCheckpointState;
use crate::{
    arch::{SMOKE_BATCH, SMOKE_LOSS_WRM},
    trainer_common::{BatchData, PrecisionFlags},
    trainer_layerstack::{GpuTrainer as LayerStackGpuTrainer, OptimGroupConfig},
    trainer_simple::SimpleGpuTrainer,
};

// NVCC の FMA による mul-add ごとの丸め差を許容するため、CUDA C++ と cuda-oxide は
// bit 一致ではなくこの相対 tolerance で比較する。
const NATIVE_PARITY_TOLERANCE: f64 = 2.0e-6;
const NATIVE_WEIGHT_ABS_TOLERANCE: f32 = 2.0e-9;
const NATIVE_FIRST_MOMENT_ABS_TOLERANCE: f32 = 2.0e-10;
const NATIVE_SECOND_MOMENT_ABS_TOLERANCE: f32 = 8.0e-12;

#[cfg(feature = "native-cuda")]
#[derive(Clone, Copy)]
struct StateTolerances {
    weight: f32,
    first_moment: f32,
    second_moment: f32,
}

#[cfg(feature = "native-cuda")]
const LARGE_BATCH_STATE_TOLERANCES: StateTolerances = StateTolerances {
    weight: 5.0e-8,
    first_moment: 5.0e-7,
    second_moment: 1.0e-8,
};

fn standard_id() -> SimpleId {
    SimpleId {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    }
}

fn cuda_launch_symbols(source: &str) -> std::collections::BTreeSet<String> {
    // attribute (`#[...]` の bracket stream) が item を **非 test build で必ず除去する** か。
    // `#[test]` (bare) と `#[cfg(test)]` / `#[cfg(all(test, ...))]` / `#[cfg(any(test))]` は
    // test 専用なので除外する。一方 `#[cfg(not(test))]` (production 専用) や
    // `#[cfg(any(test, feature = "x"))]` (feature 有効な production でも compile) は production
    // launch を含み得るので **除外しない**。誤って production を除外すると native export 検査の
    // 走査から実 launch が漏れる (穴が素通り) ため、判定を誤らない側に倒す。
    #[derive(Clone, Copy, PartialEq)]
    enum Tri {
        True,
        False,
        Unknown,
    }

    fn attribute_marks_test_only(stream: proc_macro2::TokenStream) -> bool {
        let tokens: Vec<proc_macro2::TokenTree> = stream.into_iter().collect();
        // bare `#[test]`。
        if let [proc_macro2::TokenTree::Ident(ident)] = tokens.as_slice() {
            return ident == "test";
        }
        // `#[cfg(<predicate>)]` の predicate を test=false で 3 値評価し、False なら test 専用。
        for window in tokens.windows(2) {
            if let [
                proc_macro2::TokenTree::Ident(ident),
                proc_macro2::TokenTree::Group(group),
            ] = window
            {
                if ident == "cfg" && group.delimiter() == proc_macro2::Delimiter::Parenthesis {
                    return eval_predicate_group(group.stream()) == Tri::False;
                }
            }
        }
        false
    }

    // group の中身を top-level `,` で分割し、各 predicate を評価する。
    fn split_predicates(stream: proc_macro2::TokenStream) -> Vec<Vec<proc_macro2::TokenTree>> {
        let mut parts = vec![Vec::new()];
        for token in stream {
            match &token {
                proc_macro2::TokenTree::Punct(punct) if punct.as_char() == ',' => {
                    parts.push(Vec::new());
                }
                _ => parts.last_mut().unwrap().push(token),
            }
        }
        parts.into_iter().filter(|part| !part.is_empty()).collect()
    }

    // `test` は false、`not` / `all` / `any` を再帰評価、その他 (feature = "x" 等) は Unknown。
    fn eval_predicate(tokens: &[proc_macro2::TokenTree]) -> Tri {
        match tokens {
            [proc_macro2::TokenTree::Ident(ident)] => {
                if ident == "test" {
                    Tri::False
                } else {
                    Tri::Unknown
                }
            }
            [
                proc_macro2::TokenTree::Ident(ident),
                proc_macro2::TokenTree::Group(group),
            ] if group.delimiter() == proc_macro2::Delimiter::Parenthesis => {
                let inner = split_predicates(group.stream());
                if ident == "not" {
                    match inner.as_slice() {
                        [only] => match eval_predicate(only) {
                            Tri::True => Tri::False,
                            Tri::False => Tri::True,
                            Tri::Unknown => Tri::Unknown,
                        },
                        _ => Tri::Unknown,
                    }
                } else if ident == "all" {
                    let mut result = Tri::True;
                    for predicate in &inner {
                        match eval_predicate(predicate) {
                            Tri::False => return Tri::False,
                            Tri::Unknown => result = Tri::Unknown,
                            Tri::True => {}
                        }
                    }
                    result
                } else if ident == "any" {
                    let mut result = Tri::False;
                    for predicate in &inner {
                        match eval_predicate(predicate) {
                            Tri::True => return Tri::True,
                            Tri::Unknown => result = Tri::Unknown,
                            Tri::False => {}
                        }
                    }
                    result
                } else {
                    Tri::Unknown
                }
            }
            _ => Tri::Unknown,
        }
    }

    fn eval_predicate_group(stream: proc_macro2::TokenStream) -> Tri {
        match split_predicates(stream).as_slice() {
            [only] => eval_predicate(only),
            _ => Tri::Unknown,
        }
    }

    fn visit(stream: proc_macro2::TokenStream, symbols: &mut std::collections::BTreeSet<String>) {
        let mut tokens = stream.into_iter().peekable();
        // `#[cfg(test)]` を見たら次に来る item 本体 (brace group) か `;` までを production
        // launch 走査から除外する。inline test module 内の launch を数えないため。
        let mut skip_next_item_body = false;
        while let Some(token) = tokens.next() {
            if skip_next_item_body {
                match token {
                    proc_macro2::TokenTree::Group(group)
                        if group.delimiter() == proc_macro2::Delimiter::Brace =>
                    {
                        skip_next_item_body = false;
                    }
                    proc_macro2::TokenTree::Punct(punct) if punct.as_char() == ';' => {
                        skip_next_item_body = false;
                    }
                    _ => {}
                }
                continue;
            }
            match token {
                proc_macro2::TokenTree::Punct(punct) if punct.as_char() == '#' => {
                    if let Some(proc_macro2::TokenTree::Group(group)) = tokens.peek() {
                        if group.delimiter() == proc_macro2::Delimiter::Bracket
                            && attribute_marks_test_only(group.stream())
                        {
                            skip_next_item_body = true;
                            tokens.next();
                        }
                    }
                }
                proc_macro2::TokenTree::Ident(ident) if ident == "cuda_launch" => {
                    let is_macro = matches!(
                        tokens.peek(),
                        Some(proc_macro2::TokenTree::Punct(punct)) if punct.as_char() == '!'
                    );
                    if !is_macro {
                        continue;
                    }
                    tokens.next();
                    let Some(proc_macro2::TokenTree::Group(body)) = tokens.next() else {
                        panic!("cuda_launch! must be followed by a delimited macro body");
                    };
                    let mut body_tokens = body.stream().into_iter().peekable();
                    let mut kernel_symbol = None;
                    while let Some(body_token) = body_tokens.next() {
                        let proc_macro2::TokenTree::Ident(field) = body_token else {
                            continue;
                        };
                        if field != "kernel" {
                            continue;
                        }
                        let has_colon = matches!(
                            body_tokens.next(),
                            Some(proc_macro2::TokenTree::Punct(punct)) if punct.as_char() == ':'
                        );
                        if !has_colon {
                            continue;
                        }
                        if let Some(proc_macro2::TokenTree::Ident(symbol)) = body_tokens.next() {
                            kernel_symbol = Some(symbol.to_string());
                        }
                        break;
                    }
                    symbols.insert(
                        kernel_symbol.expect("cuda_launch! must contain `kernel: <identifier>`"),
                    );
                }
                proc_macro2::TokenTree::Group(group) => visit(group.stream(), symbols),
                _ => {}
            }
        }
    }

    let mut symbols = std::collections::BTreeSet::new();
    visit(
        source.parse().expect("Rust source must tokenize"),
        &mut symbols,
    );
    symbols
}

/// source が `cuda_launch!` を実際に **invoke** しているか (macro ident + `!`)。tokenize して
/// 判定するので、コメント / string literal 中の文字列は誤検出しない。tokenize できない source
/// は raw 部分列一致に fallback する (誤検出側に倒し、実 launch を見逃さない)。
fn source_invokes_cuda_launch(source: &str) -> bool {
    fn scan(stream: proc_macro2::TokenStream) -> bool {
        let mut tokens = stream.into_iter().peekable();
        while let Some(token) = tokens.next() {
            match &token {
                proc_macro2::TokenTree::Ident(ident) if ident == "cuda_launch" => {
                    if matches!(
                        tokens.peek(),
                        Some(proc_macro2::TokenTree::Punct(punct)) if punct.as_char() == '!'
                    ) {
                        return true;
                    }
                }
                proc_macro2::TokenTree::Group(group) if scan(group.stream()) => return true,
                _ => {}
            }
        }
        false
    }
    match source.parse::<proc_macro2::TokenStream>() {
        Ok(stream) => scan(stream),
        Err(_) => source.contains("cuda_launch!"),
    }
}

fn production_cuda_launch_symbols() -> std::collections::BTreeSet<String> {
    fn visit(path: &std::path::Path, symbols: &mut std::collections::BTreeSet<String>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().and_then(|name| name.to_str()) != Some("tests") {
                    visit(&path, symbols);
                }
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                symbols.extend(cuda_launch_symbols(&source));
            }
        }
    }

    let mut symbols = std::collections::BTreeSet::new();
    visit(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut symbols,
    );
    symbols
}

/// `//` 行コメントと `/* ... */` ブロックコメントを空白へ潰す (改行と string literal は保持)。
/// ブロックコメントは複数行に跨るので、行頭アンカーだけではコメント化された宣言を弾けない。
fn strip_cuda_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    #[derive(PartialEq)]
    enum State {
        Code,
        Line,
        Block,
        Str,
    }
    let mut state = State::Code;
    while i < bytes.len() {
        let b = bytes[i];
        let next = bytes.get(i + 1).copied();
        match state {
            State::Code => {
                if b == b'/' && next == Some(b'/') {
                    state = State::Line;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else if b == b'/' && next == Some(b'*') {
                    state = State::Block;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else {
                    if b == b'"' {
                        state = State::Str;
                    }
                    out.push(b as char);
                    i += 1;
                }
            }
            State::Str => {
                out.push(b as char);
                if b == b'\\' {
                    if let Some(escaped) = next {
                        out.push(escaped as char);
                        i += 2;
                        continue;
                    }
                } else if b == b'"' {
                    state = State::Code;
                }
                i += 1;
            }
            State::Line => {
                if b == b'\n' {
                    state = State::Code;
                    out.push('\n');
                } else {
                    out.push(' ');
                }
                i += 1;
            }
            State::Block => {
                if b == b'*' && next == Some(b'/') {
                    state = State::Code;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else {
                    out.push(if b == b'\n' { '\n' } else { ' ' });
                    i += 1;
                }
            }
        }
    }
    out
}

/// `native_kernels.cu` の行頭に `extern "C" __global__ void <name>(` がある export だけを
/// 集める。コメントを潰してから行頭マッチするので、`//` / `/* ... */` (複数行含む) で
/// コメント化された宣言は「存在する」と誤判定されない (穴を素通りさせないため)。
fn native_source_exports() -> std::collections::BTreeSet<String> {
    let native = include_str!("../../../../crates/cuda-native-runtime/kernels/native_kernels.cu");
    let prefix = "extern \"C\" __global__ void ";
    strip_cuda_comments(native)
        .lines()
        .filter_map(|line| line.strip_prefix(prefix).map(str::to_owned))
        .filter_map(|declaration| declaration.split_once('(').map(|(name, _)| name.to_owned()))
        .collect()
}

fn assert_native_exports(required: &std::collections::BTreeSet<String>) {
    let exported = native_source_exports();
    let missing: Vec<_> = required.difference(&exported).collect();
    assert!(missing.is_empty(), "native CUDA is missing: {missing:?}");
}

#[test]
fn native_inventory_parser_accepts_inline_and_multiline_launches() {
    let source = "cuda_launch! { kernel: inline, stream: s }\n\
                  cuda_launch!(\n  kernel : multiline,\n  stream: s\n);";
    assert_eq!(
        cuda_launch_symbols(source),
        ["inline".to_owned(), "multiline".to_owned()]
            .into_iter()
            .collect()
    );
}

#[test]
fn native_inventory_parser_ignores_comments_and_string_delimiters() {
    let source = r#"
        fn launch() {
            let _ignored = "cuda_launch! { kernel: string_fake, args: [\"]\"] }";
            // cuda_launch! { kernel: line_comment_fake }
            /* cuda_launch![kernel: block_comment_fake] */
            cuda_launch! /* comment with } ] ) */ {
                kernel: comment_separated,
                args: ["}", "]", ")"],
            };
            cuda_launch![
                kernel: bracket_delimited,
                args: [call("}"), nested({ 1 })],
            ];
        }
    "#;
    assert_eq!(
        cuda_launch_symbols(source),
        [
            "bracket_delimited".to_owned(),
            "comment_separated".to_owned()
        ]
        .into_iter()
        .collect()
    );
}

#[test]
#[should_panic(expected = "cuda_launch! must contain `kernel: <identifier>`")]
fn native_inventory_parser_rejects_unrecognized_launches() {
    let _ = cuda_launch_symbols("cuda_launch! { stream: stream, args: [] }");
}

#[test]
fn native_inventory_parser_excludes_only_test_only_launches() {
    let source = r#"
        #[cfg(test)]
        mod tests {
            fn t() { cuda_launch! { kernel: cfg_test_excluded, stream: s } }
        }
        #[cfg(all(test, feature = "x"))]
        fn all_test() { cuda_launch! { kernel: all_test_excluded, stream: s } }
        #[test]
        fn bare_test() { cuda_launch! { kernel: bare_test_excluded, stream: s } }

        #[cfg(not(test))]
        fn prod_only() { cuda_launch! { kernel: not_test_kept, stream: s } }
        #[cfg(any(test, feature = "x"))]
        fn maybe_prod() { cuda_launch! { kernel: any_test_feature_kept, stream: s } }
        fn always() { cuda_launch! { kernel: ungated_kept, stream: s } }
    "#;
    assert_eq!(
        cuda_launch_symbols(source),
        [
            "any_test_feature_kept".to_owned(),
            "not_test_kept".to_owned(),
            "ungated_kept".to_owned(),
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn every_simple_native_kernel_is_exported() {
    let launch_sources = [
        include_str!("../trainer_simple.rs"),
        include_str!("../ft_factorize_host.rs"),
    ];
    let mut required = std::collections::BTreeSet::new();
    for source in launch_sources {
        required.extend(cuda_launch_symbols(source));
    }
    assert_eq!(required.len(), 49, "Simple kernel inventory changed");
    assert_native_exports(&required);
}

#[test]
fn every_layerstack_native_kernel_is_exported() {
    let launch_sources = [
        include_str!("../trainer_layerstack.rs"),
        include_str!("../ft_factorize_host.rs"),
    ];
    let mut required = std::collections::BTreeSet::new();
    for source in launch_sources {
        required.extend(cuda_launch_symbols(source));
    }
    assert_eq!(required.len(), 61, "LayerStack kernel inventory changed");
    assert_native_exports(&required);
}

#[test]
fn every_production_cuda_launch_is_exported() {
    let required = production_cuda_launch_symbols();
    assert_eq!(required.len(), 77, "production kernel inventory changed");
    assert_native_exports(&required);
}

/// `production_cuda_launch_symbols` は nnue_train crate の `src` だけを走査する。native path
/// を持ち得る production launch が別 crate / 別 binary に移ると、この走査から外れて 77 本の
/// inventory tripwire も発火しなくなる。workspace 全体を走査し、`cuda_launch!` を含む
/// production source が既知の許可 path 配下だけにあることを固定する。
#[test]
fn cuda_launch_stays_within_known_production_roots() {
    // native trainer が launch する crate。ここ以外に `cuda_launch!` が現れたら、native
    // export 検査 (`every_production_cuda_launch_is_exported`) の走査外になっていないか確認する。
    const NATIVE_LAUNCH_ROOT: &str = "bins/nnue_train/src";
    // `cuda_launch!` を含むが native path ではない既知の場所。gpu-runtime は macro 定義と
    // example、progress_kpabs_train は cuda-oxide 固定 backend。
    const NON_NATIVE_LAUNCH_ROOTS: [&str; 2] =
        ["crates/gpu-runtime", "bins/progress_kpabs_train/src"];

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root is two levels above the crate manifest")
        .to_owned();

    fn visit(path: &std::path::Path, offenders: &mut Vec<String>, root: &std::path::Path) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                let name = path.file_name().and_then(|name| name.to_str());
                if !matches!(name, Some("target") | Some("tests") | Some(".git")) {
                    visit(&path, offenders, root);
                }
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                // token 走査でコメント / string literal 中の `cuda_launch!` を誤検出しない。
                if !source_invokes_cuda_launch(&source) {
                    continue;
                }
                let relative = path.strip_prefix(root).unwrap_or(&path);
                let relative = relative.to_string_lossy().replace('\\', "/");
                let known = relative.starts_with(NATIVE_LAUNCH_ROOT)
                    || NON_NATIVE_LAUNCH_ROOTS
                        .iter()
                        .any(|allowed| relative.starts_with(allowed));
                if !known {
                    offenders.push(relative);
                }
            }
        }
    }

    let mut offenders = Vec::new();
    visit(&workspace_root, &mut offenders, &workspace_root);
    assert!(
        offenders.is_empty(),
        "cuda_launch! appears outside known roots (update production inventory scan): {offenders:?}"
    );
}

/// `native_kernels.cu` の per-bucket accumulator 容量 (`kMaxSupportedNumBuckets`) が host 側の
/// `MAX_SUPPORTED_NUM_BUCKETS` と一致することを固定する。host が上回ると kernel が
/// `min(num_buckets, kMaxSupportedNumBuckets)` で clamp し、上位 bucket の勾配が黙って落ちる。
#[test]
fn native_bucket_capacity_matches_host() {
    let native = include_str!("../../../../crates/cuda-native-runtime/kernels/native_kernels.cu");
    let marker = "constexpr unsigned int kMaxSupportedNumBuckets = ";
    let value = native
        .lines()
        .find_map(|line| line.trim_start().strip_prefix(marker))
        .and_then(|rest| rest.split(';').next())
        .map(str::trim)
        .expect("native_kernels.cu must define kMaxSupportedNumBuckets")
        .parse::<usize>()
        .expect("kMaxSupportedNumBuckets must be an integer");
    assert_eq!(
        value,
        crate::arch::MAX_SUPPORTED_NUM_BUCKETS,
        "native kMaxSupportedNumBuckets must match host MAX_SUPPORTED_NUM_BUCKETS"
    );
}

#[derive(Clone, Copy)]
struct LayerStackTestOptions {
    feature_set: shogi_features::FeatureSetSpec,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
    bucket_mode: BucketMode,
    precision: PrecisionFlags,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    psqt: bool,
}

impl LayerStackTestOptions {
    fn standard() -> Self {
        Self {
            feature_set: FeatureSet::HalfKp.spec(),
            ft_out: 128,
            l1_out: 16,
            l2_out: 32,
            num_buckets: 2,
            bucket_mode: BucketMode::Progress8KpAbs,
            precision: PrecisionFlags::default(),
            optimizer: OptimizerKind::Ranger,
            norm_loss_factor: None,
            psqt: false,
        }
    }
}

fn create_layerstack_trainer(
    context: &std::sync::Arc<CudaContext>,
    native: bool,
) -> Result<LayerStackGpuTrainer, Box<dyn std::error::Error>> {
    create_layerstack_trainer_with_options(context, native, LayerStackTestOptions::standard())
}

fn create_layerstack_trainer_with_options(
    context: &std::sync::Arc<CudaContext>,
    native: bool,
    options: LayerStackTestOptions,
) -> Result<LayerStackGpuTrainer, Box<dyn std::error::Error>> {
    create_layerstack_trainer_with_batch(context, native, options, SMOKE_BATCH)
}

fn create_layerstack_trainer_with_batch(
    context: &std::sync::Arc<CudaContext>,
    native: bool,
    options: LayerStackTestOptions,
    batch_size: usize,
) -> Result<LayerStackGpuTrainer, Box<dyn std::error::Error>> {
    let psqt_init = options
        .psqt
        .then(|| vec![0.0_f32; options.feature_set.ft_in() * options.num_buckets]);
    let operation = || {
        LayerStackGpuTrainer::new(
            context,
            batch_size,
            options.ft_out,
            options.l1_out,
            options.l2_out,
            options.num_buckets,
            options.bucket_mode,
            options.precision,
            options.feature_set,
            options.optimizer,
            OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
            options.norm_loss_factor,
            psqt_init.as_deref(),
            &LayerStackInit::default_uniform(),
        )
    };
    #[cfg(feature = "native-cuda")]
    return with_test_native_backend(native, operation);
    #[cfg(feature = "native-cuda-host")]
    {
        assert!(
            native,
            "native-host build cannot create a cuda-oxide trainer"
        );
        operation()
    }
}

#[test]
fn standard_layerstack_runs_one_native_training_step() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let mut trainer = create_layerstack_trainer(&context, true)?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, FeatureSet::HalfKp.spec());
    for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
        *bucket = (row % 2) as i32;
    }
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);
    let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let loss = trainer.validate(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?.loss;
    assert!(
        loss.is_finite(),
        "native LayerStack loss is not finite: {loss}"
    );
    trainer.assert_all_weights_finite()?;
    let state = trainer.raw_checkpoint_state_to_host()?;
    let mut fingerprint = 0xcbf29ce484222325_u64;
    fingerprint ^= state.0;
    fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    for (_, group) in &state.1 {
        for value in group
            .0
            .iter()
            .chain(&group.1)
            .chain(&group.2)
            .chain(&group.3)
        {
            let quantized = (f64::from(*value) * 1.0e6).round() as i64;
            fingerprint ^= quantized as u64;
            fingerprint = fingerprint.wrapping_mul(0x100000001b3);
        }
    }
    eprintln!(
        "[native-layerstack-host-parity] loss_bits={:016x}, state_fingerprint_1e6={fingerprint:016x}",
        loss.to_bits()
    );
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn standard_layerstack_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_layerstack_native_matches_cuda_oxide(
        LayerStackTestOptions::standard(),
        SMOKE_LOSS_WRM,
        1,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn large_batch_multistep_layerstack_native_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    assert_layerstack_native_matches_cuda_oxide_with_batch(
        LayerStackTestOptions {
            num_buckets: 9,
            ..LayerStackTestOptions::standard()
        },
        SMOKE_LOSS_WRM,
        3,
        16_384,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn large_batch_multistep_all_optim_layerstack_native_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    assert_layerstack_native_matches_cuda_oxide_with_batch(
        LayerStackTestOptions {
            num_buckets: 9,
            precision: PrecisionFlags {
                tf32: true,
                ft_fp16: true,
                ft_fp16_out: true,
                fp16_opt_state: true,
            },
            ..LayerStackTestOptions::standard()
        },
        SMOKE_LOSS_WRM,
        3,
        16_384,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn complete_layerstack_native_configuration_matrix_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    for (name, options, loss) in layerstack_configuration_matrix() {
        eprintln!("[native-parity] LayerStack configuration: {name}");
        assert_layerstack_native_matches_cuda_oxide(options, loss, 1)?;
    }
    Ok(())
}

#[test]
fn complete_layerstack_native_configuration_matrix_runs_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    for (name, options, loss) in layerstack_configuration_matrix() {
        eprintln!("[native-host] LayerStack configuration: {name}");
        let mut trainer = create_layerstack_trainer_with_options(&context, true, options)?;
        let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, options.feature_set);
        for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
            *bucket = (row % options.num_buckets) as i32;
        }
        batch.score.fill(200.0);
        batch.wdl.fill(0.8);
        let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
        let validation = trainer.validate(&batch.as_ref(), 0.0, loss)?;
        assert!(validation.loss.is_finite());
        trainer.assert_all_weights_finite()?;
    }
    Ok(())
}

fn layerstack_configuration_matrix() -> Vec<(&'static str, LayerStackTestOptions, LossKind)> {
    let standard = LayerStackTestOptions::standard();
    let extended_wrm = match SMOKE_LOSS_WRM {
        LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            ..
        } => LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            pow_exp: 2.5,
            qp_asymmetry: 0.2,
            weight_boost_w1: 1.5,
            weight_boost_w2: 0.75,
        },
        LossKind::Sigmoid { .. } => unreachable!(),
    };
    vec![
        (
            "tf32",
            LayerStackTestOptions {
                precision: PrecisionFlags {
                    tf32: true,
                    ..PrecisionFlags::default()
                },
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "all-fp16",
            LayerStackTestOptions {
                precision: PrecisionFlags {
                    ft_fp16: true,
                    ft_fp16_out: true,
                    fp16_opt_state: true,
                    ..PrecisionFlags::default()
                },
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "radam-norm-sigmoid",
            LayerStackTestOptions {
                optimizer: OptimizerKind::RAdam,
                norm_loss_factor: Some(0.25),
                ..standard
            },
            LossKind::Sigmoid { scale: 1.0 / 600.0 },
        ),
        (
            "adamw-extended-wrm",
            LayerStackTestOptions {
                optimizer: OptimizerKind::AdamW,
                ..standard
            },
            extended_wrm,
        ),
        (
            "psqt",
            LayerStackTestOptions {
                psqt: true,
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "factorized",
            LayerStackTestOptions {
                feature_set: standard.feature_set.with_ft_factorize(),
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "threat-factorized",
            LayerStackTestOptions {
                feature_set: standard
                    .feature_set
                    .with_threat_profile(ThreatProfile::CrossSide)
                    .with_ft_factorize(),
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "effect-factorized-pool",
            LayerStackTestOptions {
                feature_set: standard
                    .feature_set
                    .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2)
                    .with_ft_factorize_mode(FtFactorizeMode::PoolEffectBuckets),
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "effect-factorized-per-bucket",
            LayerStackTestOptions {
                feature_set: standard
                    .feature_set
                    .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2)
                    .with_ft_factorize_mode(FtFactorizeMode::PerEffectBucket),
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "kingrank9",
            LayerStackTestOptions {
                num_buckets: 9,
                bucket_mode: BucketMode::KingRank9,
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "minimum-hidden",
            LayerStackTestOptions {
                l1_out: 2,
                l2_out: 2,
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
        (
            "maximum-hidden",
            LayerStackTestOptions {
                l1_out: 256,
                l2_out: 256,
                ..standard
            },
            SMOKE_LOSS_WRM,
        ),
    ]
}

#[test]
fn complete_layerstack_native_feature_matrix_runs_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let base = FeatureSet::HalfKp.spec();
    let threats = [
        ThreatProfile::Full,
        ThreatProfile::SameClass,
        ThreatProfile::SameClassMajorPawn,
        ThreatProfile::StepAttacker,
        ThreatProfile::FullSymDedup,
        ThreatProfile::CrossSide,
    ];
    let effects = [
        EffectBucketConfig::KINGFIXED_2X2,
        EffectBucketConfig::KINGBUCKETED_2X2,
        EffectBucketConfig::KINGFIXED_3X3,
        EffectBucketConfig::KINGBUCKETED_3X3,
    ];
    let mut configurations = Vec::new();
    for feature_set in FeatureSet::ALL {
        configurations.push(("base", feature_set.spec()));
    }
    for profile in threats {
        let spec = base.with_threat_profile(profile);
        configurations.push(("threat", spec));
        configurations.push(("threat-factorized", spec.with_ft_factorize()));
    }
    for effect in effects {
        let spec = base.with_effect_bucket_config(effect);
        configurations.push(("effect", spec));
        configurations.push((
            "effect-factorized-pool",
            spec.with_ft_factorize_mode(FtFactorizeMode::PoolEffectBuckets),
        ));
        configurations.push((
            "effect-factorized-per-bucket",
            spec.with_ft_factorize_mode(FtFactorizeMode::PerEffectBucket),
        ));
    }

    for (name, feature_set) in configurations {
        eprintln!(
            "[native-host] LayerStack feature configuration: {name}, {}",
            feature_set.arch_feature_name()
        );
        let options = LayerStackTestOptions {
            feature_set,
            l1_out: 2,
            l2_out: 2,
            ..LayerStackTestOptions::standard()
        };
        let mut trainer = create_layerstack_trainer_with_options(&context, true, options)?;
        let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
            *bucket = (row % options.num_buckets) as i32;
        }
        let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
        trainer.assert_all_weights_finite()?;
    }
    Ok(())
}

#[cfg(feature = "native-cuda")]
fn assert_layerstack_native_matches_cuda_oxide(
    options: LayerStackTestOptions,
    loss: LossKind,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_layerstack_native_matches_cuda_oxide_with_batch(options, loss, steps, SMOKE_BATCH)
}

#[cfg(feature = "native-cuda")]
fn assert_layerstack_native_matches_cuda_oxide_with_batch(
    options: LayerStackTestOptions,
    loss: LossKind,
    steps: usize,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let mut oxide = create_layerstack_trainer_with_batch(&context, false, options, batch_size)?;
    let mut native = create_layerstack_trainer_with_batch(&context, true, options, batch_size)?;
    let mut batch = BatchData::smoke_dummy(batch_size, options.feature_set);
    for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
        *bucket = (row % options.num_buckets) as i32;
    }
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    for _ in 0..steps {
        let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
        let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    }
    let oxide_loss = oxide.validate(&batch.as_ref(), 0.0, loss)?.loss;
    let native_loss = native.validate(&batch.as_ref(), 0.0, loss)?.loss;
    let loss_difference = (oxide_loss - native_loss).abs();
    assert!(
        loss_difference <= NATIVE_PARITY_TOLERANCE * (1.0 + oxide_loss.abs()),
        "LayerStack loss differs: oxide={oxide_loss}, native={native_loss}, diff={loss_difference}"
    );
    let oxide_state = oxide.raw_checkpoint_state_to_host()?;
    let native_state = native.raw_checkpoint_state_to_host()?;
    if batch_size > SMOKE_BATCH {
        assert_checkpoint_state_close_with_tolerances(
            "LayerStack large batch",
            &oxide_state,
            &native_state,
            LARGE_BATCH_STATE_TOLERANCES,
        );
    } else {
        assert_checkpoint_state_close("LayerStack", &oxide_state, &native_state);
    }
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_threat_and_effect_profiles_layerstack_native_match_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    let base = FeatureSet::HalfKp.spec();
    let threats = [
        ThreatProfile::Full,
        ThreatProfile::SameClass,
        ThreatProfile::SameClassMajorPawn,
        ThreatProfile::StepAttacker,
        ThreatProfile::FullSymDedup,
        ThreatProfile::CrossSide,
    ];
    let effects = [
        EffectBucketConfig::KINGFIXED_2X2,
        EffectBucketConfig::KINGBUCKETED_2X2,
        EffectBucketConfig::KINGFIXED_3X3,
        EffectBucketConfig::KINGBUCKETED_3X3,
    ];
    let mut feature_sets = Vec::new();
    for profile in threats {
        feature_sets.push(("threat", base.with_threat_profile(profile)));
    }
    for effect in effects {
        feature_sets.push(("effect", base.with_effect_bucket_config(effect)));
    }
    for (name, feature_set) in feature_sets {
        eprintln!(
            "[native-parity] LayerStack {name}: {}",
            feature_set.arch_feature_name()
        );
        let options = LayerStackTestOptions {
            feature_set,
            l1_out: 2,
            l2_out: 2,
            ..LayerStackTestOptions::standard()
        };
        assert_layerstack_native_matches_cuda_oxide(options, SMOKE_LOSS_WRM, 1)?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn production_ft_width_layerstack_native_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    // 既定の parity 構成は ft_out=128 のみで、`ft_out / 16` tile/grid や大 stride でだけ出る
    // 差を取り逃す。production 既定幅 (1536) で 1 ケース比較する。
    assert_layerstack_native_matches_cuda_oxide(
        LayerStackTestOptions {
            ft_out: 1536,
            ..LayerStackTestOptions::standard()
        },
        SMOKE_LOSS_WRM,
        1,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn tf32_invalid_bucket_layerstack_native_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    // TF32 L1 経路は有効 bucket segment のみ cuBLAS で上書きする。ある行の bucket が
    // 有効→invalid に変わると、その行の sorted 出力は更新されず前 step 値が残り得る。
    // 手書き kernel / cuda-oxide は同じ行を 0 で埋めるので、native TF32 もそれに一致する
    // ことを 2-step で確認する。
    let context = CudaContext::new(0)?;
    let options = LayerStackTestOptions {
        precision: PrecisionFlags {
            tf32: true,
            ..PrecisionFlags::default()
        },
        ..LayerStackTestOptions::standard()
    };
    let mut oxide = create_layerstack_trainer_with_options(&context, false, options)?;
    let mut native = create_layerstack_trainer_with_options(&context, true, options)?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, options.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    // Step 1: 全行 valid bucket。
    for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
        *bucket = (row % options.num_buckets) as i32;
    }
    let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;

    // Step 2: 一部の行を invalid bucket (>= num_buckets) にして、有効→invalid 遷移を作る。
    for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
        *bucket = if row % 4 == 0 {
            options.num_buckets as i32
        } else {
            (row % options.num_buckets) as i32
        };
    }
    let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;

    // native/oxide は同じ host trainer を共有し kernel module だけ切り替わるので、上の
    // checkpoint 比較は両経路共有の 0 初期化 (trainer_layerstack.rs の TF32 分岐) が消えても
    // 両者一致で通ってしまう。修正を守るには「sorted L1 出力の padding 行が prior step の残差を
    // 持たず 0」を backend 非依存に固定する。step 1 で real row だった sorted 位置が step 2 で
    // padding になると、0 初期化が無ければ step 1 の値が残る。
    let (l1_bucket_sorted, bucket_idx_sorted) =
        native.l1_bucket_sorted_and_index_to_host_for_test()?;
    let l1_out = l1_bucket_sorted.len() / bucket_idx_sorted.len();
    let mut padding_rows = 0usize;
    for (sorted_row, &bucket) in bucket_idx_sorted.iter().enumerate() {
        if bucket >= 0 {
            continue;
        }
        padding_rows += 1;
        let row = &l1_bucket_sorted[sorted_row * l1_out..(sorted_row + 1) * l1_out];
        for (column, &value) in row.iter().enumerate() {
            assert_eq!(
                value, 0.0,
                "padding sorted row {sorted_row} col {column} carried a stale L1 output",
            );
        }
    }
    assert!(
        padding_rows > 0,
        "expected invalid buckets to create padding sorted rows",
    );

    let oxide_state = oxide.raw_checkpoint_state_to_host()?;
    let native_state = native.raw_checkpoint_state_to_host()?;
    assert_checkpoint_state_close(
        "LayerStack tf32 invalid bucket",
        &oxide_state,
        &native_state,
    );
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn checkpoint_resume_layerstack_native_matches_cuda_oxide_in_both_directions()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let options = LayerStackTestOptions {
        feature_set: FeatureSet::HalfKp.spec().with_ft_factorize(),
        precision: PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
        psqt: true,
        ..LayerStackTestOptions::standard()
    };
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, options.feature_set);
    for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
        *bucket = (row % options.num_buckets) as i32;
    }
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    for source_is_native in [false, true] {
        let source_name = if source_is_native { "native" } else { "oxide" };
        let path = std::env::temp_dir().join(format!(
            "tatara-layerstack-native-resume-{source_name}-{}.ckpt",
            std::process::id()
        ));
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let mut source =
                create_layerstack_trainer_with_options(&context, source_is_native, options)?;
            for _ in 0..5 {
                let _ = source.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
            }
            let checkpoint_state = source.raw_checkpoint_state_to_host()?;
            assert_eq!(checkpoint_state.0, 5);
            source.save_raw_checkpoint(&path, 17, source_name, Some(42))?;
            let _ = source.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
            let oracle_state = source.raw_checkpoint_state_to_host()?;
            assert_eq!(oracle_state.0, 6);
            drop(source);

            let mut oxide = create_layerstack_trainer_with_options(&context, false, options)?;
            let mut native = create_layerstack_trainer_with_options(&context, true, options)?;
            for (backend_name, trainer) in [("oxide", &mut oxide), ("native", &mut native)] {
                let (superbatch, producer, horizon) = trainer.load_raw_checkpoint(&path)?;
                assert_eq!(superbatch, 17);
                assert_eq!(producer.as_deref(), Some(source_name));
                assert_eq!(horizon, Some(42));
                let loaded_state = trainer.raw_checkpoint_state_to_host()?;
                assert_checkpoint_state_bit_identical(
                    &format!("LayerStack {source_name} to {backend_name} load"),
                    &checkpoint_state,
                    &loaded_state,
                );
                trainer.sync_ft_forward_weights()?;
                let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
                let resumed_state = trainer.raw_checkpoint_state_to_host()?;
                assert_checkpoint_state_delta_close(
                    &format!("LayerStack {source_name} to {backend_name}"),
                    &checkpoint_state,
                    &oracle_state,
                    &resumed_state,
                );
            }
            Ok(())
        })();
        let _ = std::fs::remove_file(&path);
        result?;
    }
    Ok(())
}

fn create_trainer(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    native: bool,
    batch_size: usize,
) -> Result<SimpleGpuTrainer, Box<dyn std::error::Error>> {
    create_trainer_with_options(
        context,
        id,
        native,
        batch_size,
        OptimizerKind::Ranger,
        None,
        PrecisionFlags::default(),
    )
}

fn create_trainer_with_options(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    native: bool,
    batch_size: usize,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
) -> Result<SimpleGpuTrainer, Box<dyn std::error::Error>> {
    #[cfg(feature = "native-cuda")]
    let result = with_test_native_backend(native, || {
        SimpleGpuTrainer::new(
            context,
            batch_size,
            id,
            optimizer,
            0.0,
            norm_loss_factor,
            16,
            precision,
            &SimpleInit::default_uniform(),
        )
    });
    #[cfg(feature = "native-cuda-host")]
    let result = {
        assert!(
            native,
            "native-host build cannot create a cuda-oxide trainer"
        );
        SimpleGpuTrainer::new(
            context,
            batch_size,
            id,
            optimizer,
            0.0,
            norm_loss_factor,
            16,
            precision,
            &SimpleInit::default_uniform(),
        )
    };
    let mut trainer = result?;
    // Production synchronizes the FP16 mirror / factorizer comb before the first batch.
    // Keep parity and configuration tests on that same initialized forward path.
    trainer.sync_ft_forward_weights()?;
    Ok(trainer)
}

#[test]
fn standard_simple_crelu_runs_one_native_training_step() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let id = standard_id();
    let mut trainer = create_trainer(&context, id, true, SMOKE_BATCH)?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    let _lagged_loss = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let loss = trainer.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
    assert!(loss.is_finite(), "native Simple loss is not finite: {loss}");
    trainer.assert_all_weights_finite()?;
    let weights = trainer.to_simple_weights()?;
    let mut fingerprint = 0xcbf29ce484222325_u64;
    for value in weights
        .ft_w
        .iter()
        .chain(&weights.ft_b)
        .chain(&weights.l1_w)
        .chain(&weights.l1_b)
        .chain(&weights.l2_w)
        .chain(&weights.l2_b)
        .chain(&weights.l3_w)
        .chain(&weights.l3_b)
    {
        fingerprint ^= u64::from(value.to_bits());
        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    }
    eprintln!(
        "[native-host-parity] loss_bits={:016x}, weight_fingerprint={fingerprint:016x}",
        loss.to_bits()
    );
    Ok(())
}

#[test]
fn complete_simple_native_configuration_matrix_runs_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let all_fp16 = PrecisionFlags {
        ft_fp16: true,
        ft_fp16_out: true,
        fp16_opt_state: true,
        ..PrecisionFlags::default()
    };
    let extended_wrm = match SMOKE_LOSS_WRM {
        LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            ..
        } => LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            pow_exp: 2.5,
            qp_asymmetry: 0.2,
            weight_boost_w1: 1.5,
            weight_boost_w2: 0.75,
        },
        LossKind::Sigmoid { .. } => unreachable!(),
    };

    for (activation, optimizer, norm_loss, loss) in [
        (
            SimpleActivation::CReLU,
            OptimizerKind::Ranger,
            None,
            SMOKE_LOSS_WRM,
        ),
        (
            SimpleActivation::SCReLU,
            OptimizerKind::RAdam,
            None,
            LossKind::Sigmoid { scale: 1.0 / 600.0 },
        ),
        (
            SimpleActivation::Pairwise,
            OptimizerKind::AdamW,
            Some(0.25),
            extended_wrm,
        ),
    ] {
        let mut id = standard_id();
        id.activation = activation;
        assert_native_configuration_runs(&context, id, optimizer, norm_loss, all_fp16, loss)?;
    }

    assert_native_configuration_runs(
        &context,
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags {
            ft_fp16: true,
            ..PrecisionFlags::default()
        },
        SMOKE_LOSS_WRM,
    )?;

    let mut factorized = standard_id();
    factorized.feature_set = factorized.feature_set.with_ft_factorize();
    assert_native_configuration_runs(
        &context,
        factorized,
        OptimizerKind::Ranger,
        None,
        all_fp16,
        SMOKE_LOSS_WRM,
    )?;

    let mut wide = standard_id();
    wide.l1_out = 257;
    wide.l2_out = 257;
    assert_native_configuration_runs(
        &context,
        wide,
        OptimizerKind::Ranger,
        None,
        PrecisionFlags {
            tf32: true,
            ..PrecisionFlags::default()
        },
        SMOKE_LOSS_WRM,
    )?;

    for feature_set in FeatureSet::ALL {
        let mut id = standard_id();
        id.feature_set = feature_set.spec();
        id.ft_out = 32;
        id.l1_out = 16;
        id.l2_out = 16;
        assert_native_configuration_runs(
            &context,
            id,
            OptimizerKind::Ranger,
            None,
            PrecisionFlags::default(),
            SMOKE_LOSS_WRM,
        )?;
    }
    Ok(())
}

fn assert_native_configuration_runs(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
    loss: LossKind,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = create_trainer_with_options(
        context,
        id,
        true,
        SMOKE_BATCH,
        optimizer,
        norm_loss_factor,
        precision,
    )?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);
    let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    let forward_loss = trainer.forward(&batch.as_ref(), 0.0, loss)?;
    assert!(forward_loss.is_finite());
    trainer.assert_all_weights_finite()?;
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn standard_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step(standard_id())
}

#[test]
#[cfg(feature = "native-cuda")]
fn factorized_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn screlu_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    let mut id = standard_id();
    id.activation = SimpleActivation::SCReLU;
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn pairwise_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.activation = SimpleActivation::Pairwise;
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn wide_hidden_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.l1_out = 257;
    id.l2_out = 257;
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn radam_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::RAdam,
        PrecisionFlags::default(),
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn adamw_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::AdamW,
        PrecisionFlags::default(),
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn tf32_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            tf32: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_out_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_out_screlu_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.activation = SimpleActivation::SCReLU;
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_out_pairwise_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.activation = SimpleActivation::Pairwise;
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn fp16_optimizer_state_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_fp16_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn factorized_all_fp16_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_fp16_adamw_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::AdamW,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_fp16_ranger_lookahead_simple_native_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_with_training_options_and_steps(
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
        SMOKE_LOSS_WRM,
        6,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_feature_sets_simple_native_match_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    for feature_set in FeatureSet::ALL {
        let mut id = standard_id();
        id.feature_set = feature_set.spec();
        id.ft_out = 32;
        id.l1_out = 16;
        id.l2_out = 16;
        assert_native_matches_cuda_oxide_after_one_step(id)?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn norm_loss_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        standard_id(),
        OptimizerKind::Ranger,
        Some(0.25),
        PrecisionFlags::default(),
        SMOKE_LOSS_WRM,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn sigmoid_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags::default(),
        LossKind::Sigmoid { scale: 1.0 / 600.0 },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn extended_wrm_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let extended = match SMOKE_LOSS_WRM {
        LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            ..
        } => LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            pow_exp: 2.5,
            qp_asymmetry: 0.2,
            weight_boost_w1: 1.5,
            weight_boost_w2: 0.75,
        },
        LossKind::Sigmoid { .. } => unreachable!(),
    };
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags::default(),
        extended,
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_after_one_step(
    id: SimpleId,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags::default(),
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_after_one_step_with_options(
    id: SimpleId,
    optimizer: OptimizerKind,
    precision: PrecisionFlags,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        id,
        optimizer,
        None,
        precision,
        SMOKE_LOSS_WRM,
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_after_one_step_with_training_options(
    id: SimpleId,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
    loss: LossKind,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_with_training_options_and_steps(
        id,
        optimizer,
        norm_loss_factor,
        precision,
        loss,
        1,
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_with_training_options_and_steps(
    id: SimpleId,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
    loss: LossKind,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let mut oxide = create_trainer_with_options(
        &context,
        id,
        false,
        SMOKE_BATCH,
        optimizer,
        norm_loss_factor,
        precision,
    )?;
    let mut native = create_trainer_with_options(
        &context,
        id,
        true,
        SMOKE_BATCH,
        optimizer,
        norm_loss_factor,
        precision,
    )?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    for _ in 0..steps {
        let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
        let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    }
    let oxide_loss = oxide.forward(&batch.as_ref(), 0.0, loss)?;
    let native_loss = native.forward(&batch.as_ref(), 0.0, loss)?;
    assert_trainers_close(id, oxide_loss, native_loss, &oxide, &native)
}

#[cfg(feature = "native-cuda")]
fn assert_trainers_close(
    id: SimpleId,
    oxide_loss: f64,
    native_loss: f64,
    oxide: &SimpleGpuTrainer,
    native: &SimpleGpuTrainer,
) -> Result<(), Box<dyn std::error::Error>> {
    if id.feature_set.ft_factorize() {
        let oxide_master = oxide.ft_w_to_host()?;
        let native_master = native.ft_w_to_host()?;
        assert_weight_group_close("ft_w_master", &oxide_master, &native_master);
    }
    let oxide_weights = oxide.to_simple_weights()?;
    let native_weights = native.to_simple_weights()?;

    let loss_difference = (oxide_loss - native_loss).abs();
    assert!(
        loss_difference <= NATIVE_PARITY_TOLERANCE * (1.0 + oxide_loss.abs()),
        "loss differs: oxide={oxide_loss}, native={native_loss}, diff={loss_difference}"
    );
    for (name, oxide_group, native_group) in [
        ("ft_w", &oxide_weights.ft_w, &native_weights.ft_w),
        ("ft_b", &oxide_weights.ft_b, &native_weights.ft_b),
        ("l1_w", &oxide_weights.l1_w, &native_weights.l1_w),
        ("l1_b", &oxide_weights.l1_b, &native_weights.l1_b),
        ("l2_w", &oxide_weights.l2_w, &native_weights.l2_w),
        ("l2_b", &oxide_weights.l2_b, &native_weights.l2_b),
        ("l3_w", &oxide_weights.l3_w, &native_weights.l3_w),
        ("l3_b", &oxide_weights.l3_b, &native_weights.l3_b),
    ] {
        assert_weight_group_close(name, oxide_group, native_group);
    }
    Ok(())
}

/// raw checkpoint が backend 非依存であることを、optimizer state と Ranger の
/// `step_count` まで含めて固定する。片方の backend で 5 step 進めて保存し、同じ
/// checkpoint を cuda-oxide / CUDA C++ の両方で読んだ直後の全 state を bit 比較する。
/// 続く 6 step 目 (lookahead 発火点) の結果も無中断の 6 step 状態と比較する。保存元も
/// 両 backend を試すため、双方向の resume を覆う。
#[test]
#[cfg(feature = "native-cuda")]
fn checkpoint_resume_simple_native_matches_cuda_oxide_in_both_directions()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let id = SimpleId {
        feature_set: FeatureSet::HalfKaHmMerged.spec().with_ft_factorize(),
        activation: SimpleActivation::Pairwise,
        ft_out: 8,
        l1_out: 8,
        l2_out: 8,
    };
    let precision = PrecisionFlags {
        ft_fp16: true,
        ft_fp16_out: true,
        fp16_opt_state: true,
        ..PrecisionFlags::default()
    };
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    for source_is_native in [false, true] {
        let source_name = if source_is_native { "native" } else { "oxide" };
        let path = std::env::temp_dir().join(format!(
            "tatara-simple-native-resume-{source_name}-{}.ckpt",
            std::process::id()
        ));
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let mut source = create_trainer_with_options(
                &context,
                id,
                source_is_native,
                SMOKE_BATCH,
                OptimizerKind::Ranger,
                None,
                precision,
            )?;
            for _ in 0..5 {
                let _ = source.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
            }
            let checkpoint_state = source.raw_checkpoint_state_to_host()?;
            assert_eq!(checkpoint_state.0, 5, "checkpoint source step count");
            source.save_raw_checkpoint(&path, 17, source_name, Some(42))?;
            let _ = source.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
            let oracle_state = source.raw_checkpoint_state_to_host()?;
            assert_eq!(oracle_state.0, 6, "uninterrupted oracle step count");
            drop(source);

            let mut oxide = create_trainer_with_options(
                &context,
                id,
                false,
                SMOKE_BATCH,
                OptimizerKind::Ranger,
                None,
                precision,
            )?;
            let mut native = create_trainer_with_options(
                &context,
                id,
                true,
                SMOKE_BATCH,
                OptimizerKind::Ranger,
                None,
                precision,
            )?;
            for (backend_name, trainer) in [("oxide", &mut oxide), ("native", &mut native)] {
                let (superbatch, producer, horizon) = trainer.load_raw_checkpoint(&path)?;
                assert_eq!(superbatch, 17);
                assert_eq!(producer.as_deref(), Some(source_name));
                assert_eq!(horizon, Some(42));
                let loaded_state = trainer.raw_checkpoint_state_to_host()?;
                assert_checkpoint_state_bit_identical(
                    &format!("{source_name} to {backend_name} load"),
                    &checkpoint_state,
                    &loaded_state,
                );
                trainer.sync_ft_forward_weights()?;
                let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
                let resumed_state = trainer.raw_checkpoint_state_to_host()?;
                assert_checkpoint_state_delta_close(
                    &format!("{source_name} to {backend_name}"),
                    &checkpoint_state,
                    &oracle_state,
                    &resumed_state,
                );
            }
            let oxide_loss = oxide.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
            let native_loss = native.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
            assert_trainers_close(id, oxide_loss, native_loss, &oxide, &native)
        })();
        let _ = std::fs::remove_file(&path);
        result?;
    }
    Ok(())
}

#[cfg(feature = "native-cuda")]
fn assert_checkpoint_state_bit_identical(
    path: &str,
    expected: &SimpleRawCheckpointState,
    actual: &SimpleRawCheckpointState,
) {
    assert_eq!(actual.0, expected.0, "{path}: step count differs");
    assert_eq!(
        actual.1.len(),
        expected.1.len(),
        "{path}: group count differs"
    );
    for ((expected_name, expected_group), (actual_name, actual_group)) in
        expected.1.iter().zip(&actual.1)
    {
        assert_eq!(actual_name, expected_name, "{path}: group name differs");
        for (state_name, expected_values, actual_values) in [
            ("weight", &expected_group.0, &actual_group.0),
            ("first moment", &expected_group.1, &actual_group.1),
            ("second moment", &expected_group.2, &actual_group.2),
            ("slow weight", &expected_group.3, &actual_group.3),
        ] {
            assert_eq!(
                expected_values.len(),
                actual_values.len(),
                "{path} {expected_name} {state_name}: length differs"
            );
            for (index, (&expected, &actual)) in
                expected_values.iter().zip(actual_values).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{path} {expected_name} {state_name}[{index}]: expected={expected:?}, actual={actual:?}"
                );
            }
        }
    }
}

#[cfg(feature = "native-cuda")]
fn assert_checkpoint_state_close(
    path: &str,
    expected: &SimpleRawCheckpointState,
    actual: &SimpleRawCheckpointState,
) {
    assert_checkpoint_state_close_with_tolerances(
        path,
        expected,
        actual,
        StateTolerances {
            weight: NATIVE_WEIGHT_ABS_TOLERANCE,
            first_moment: NATIVE_FIRST_MOMENT_ABS_TOLERANCE,
            second_moment: NATIVE_SECOND_MOMENT_ABS_TOLERANCE,
        },
    );
}

#[cfg(feature = "native-cuda")]
fn assert_checkpoint_state_close_with_tolerances(
    path: &str,
    expected: &SimpleRawCheckpointState,
    actual: &SimpleRawCheckpointState,
    tolerances: StateTolerances,
) {
    assert_eq!(actual.0, expected.0, "{path}: step count differs");
    assert_eq!(
        actual.1.len(),
        expected.1.len(),
        "{path}: group count differs"
    );
    for ((expected_name, expected_group), (actual_name, actual_group)) in
        expected.1.iter().zip(&actual.1)
    {
        assert_eq!(actual_name, expected_name, "{path}: group name differs");
        for (state_name, expected_values, actual_values) in [
            ("weight", &expected_group.0, &actual_group.0),
            ("first moment", &expected_group.1, &actual_group.1),
            ("second moment", &expected_group.2, &actual_group.2),
            ("slow weight", &expected_group.3, &actual_group.3),
        ] {
            assert_weight_group_close_with_abs(
                &format!("{path} {expected_name} {state_name}"),
                expected_values,
                actual_values,
                state_abs_tolerance(state_name, tolerances),
            );
        }
    }
}

#[cfg(feature = "native-cuda")]
fn assert_checkpoint_state_delta_close(
    path: &str,
    baseline: &SimpleRawCheckpointState,
    expected: &SimpleRawCheckpointState,
    actual: &SimpleRawCheckpointState,
) {
    assert_eq!(actual.0, expected.0, "{path}: step count differs");
    assert_eq!(baseline.1.len(), expected.1.len());
    assert_eq!(actual.1.len(), expected.1.len());
    for (
        ((baseline_name, baseline_group), (expected_name, expected_group)),
        (actual_name, actual_group),
    ) in baseline.1.iter().zip(&expected.1).zip(&actual.1)
    {
        assert_eq!(
            baseline_name, expected_name,
            "{path}: baseline group differs"
        );
        assert_eq!(actual_name, expected_name, "{path}: actual group differs");
        for (state_name, baseline_values, expected_values, actual_values) in [
            (
                "weight",
                &baseline_group.0,
                &expected_group.0,
                &actual_group.0,
            ),
            (
                "first moment",
                &baseline_group.1,
                &expected_group.1,
                &actual_group.1,
            ),
            (
                "second moment",
                &baseline_group.2,
                &expected_group.2,
                &actual_group.2,
            ),
            (
                "slow weight",
                &baseline_group.3,
                &expected_group.3,
                &actual_group.3,
            ),
        ] {
            assert_eq!(baseline_values.len(), expected_values.len());
            assert_eq!(actual_values.len(), expected_values.len());
            let absolute_tolerance = state_abs_tolerance(
                state_name,
                StateTolerances {
                    weight: NATIVE_WEIGHT_ABS_TOLERANCE,
                    first_moment: NATIVE_FIRST_MOMENT_ABS_TOLERANCE,
                    second_moment: NATIVE_SECOND_MOMENT_ABS_TOLERANCE,
                },
            );
            let mut maximum_expected_delta = 0.0_f32;
            let mut maximum_delta_difference = 0.0_f32;
            for (index, ((&baseline, &expected), &actual)) in baseline_values
                .iter()
                .zip(expected_values)
                .zip(actual_values)
                .enumerate()
            {
                let expected_delta = expected - baseline;
                let actual_delta = actual - baseline;
                let difference = (expected_delta - actual_delta).abs();
                let bound =
                    absolute_tolerance + NATIVE_PARITY_TOLERANCE as f32 * expected_delta.abs();
                maximum_expected_delta = maximum_expected_delta.max(expected_delta.abs());
                maximum_delta_difference = maximum_delta_difference.max(difference);
                assert!(
                    difference <= bound,
                    "{path} {expected_name} {state_name}[{index}] delta differs: baseline={baseline}, expected={expected}, actual={actual}, expected_delta={expected_delta}, actual_delta={actual_delta}, diff={difference}, bound={bound}"
                );
            }
            eprintln!(
                "[native-parity] {path} {expected_name} {state_name} delta: max_expected={maximum_expected_delta:.3e}, max_diff={maximum_delta_difference:.3e}"
            );
        }
    }
}

#[cfg(feature = "native-cuda")]
fn state_abs_tolerance(state_name: &str, tolerances: StateTolerances) -> f32 {
    match state_name {
        "first moment" => tolerances.first_moment,
        "second moment" => tolerances.second_moment,
        "weight" | "slow weight" => tolerances.weight,
        _ => panic!("unknown checkpoint state: {state_name}"),
    }
}

#[cfg(feature = "native-cuda")]
fn assert_weight_group_close(name: &str, expected: &[f32], actual: &[f32]) {
    assert_weight_group_close_with_abs(name, expected, actual, NATIVE_WEIGHT_ABS_TOLERANCE);
}

#[cfg(feature = "native-cuda")]
fn assert_weight_group_close_with_abs(
    name: &str,
    expected: &[f32],
    actual: &[f32],
    absolute_tolerance: f32,
) {
    assert_eq!(expected.len(), actual.len());
    let mut maximum_difference = 0.0_f32;
    let mut maximum_bound = 0.0_f32;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let difference = (expected - actual).abs();
        let bound = absolute_tolerance + NATIVE_PARITY_TOLERANCE as f32 * expected.abs();
        maximum_difference = maximum_difference.max(difference);
        maximum_bound = maximum_bound.max(bound);
        assert!(
            difference <= bound,
            "{name} differs: expected={expected}, actual={actual}, diff={difference}, bound={bound}"
        );
    }
    eprintln!(
        "[native-parity] {name}: max_abs_diff={maximum_difference:.3e}, max_bound={maximum_bound:.3e}"
    );
}
