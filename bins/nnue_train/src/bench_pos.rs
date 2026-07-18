//! End-to-end training throughput benchmark driven by tracked and machine-local TOML files.
#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::cli::BenchPosArgs;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_WORK_DIR: &str = "target/benchmark-work/bench-pos";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkProfile {
    schema_version: u32,
    profile: String,
    runs: usize,
    superbatches: usize,
    warmup_superbatches: usize,
    batches_per_superbatch: usize,
    batch_size: usize,
    learning_rate: f64,
    score_drop_abs: i32,
    cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkCase {
    id: String,
    architecture: Architecture,
    feature_set: String,
    #[serde(default)]
    global_args: Vec<String>,
    #[serde(default)]
    architecture_args: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Architecture {
    Layerstack,
    Simple,
}

impl Architecture {
    fn name(self) -> &'static str {
        match self {
            Self::Layerstack => "layerstack",
            Self::Simple => "simple",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalConfig {
    schema_version: u32,
    data: PathBuf,
    data_id: String,
    progress_coeff: Option<PathBuf>,
    progress_id: Option<String>,
    threads: usize,
    #[serde(default)]
    lock_gpu_clock: bool,
    work_dir: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    timestamp_unix_ms: u64,
    profile: String,
    parameters: ParametersReport,
    inputs: InputsReport,
    environment: EnvironmentReport,
    cases: Vec<CaseReport>,
    summaries: Vec<Summary>,
}

#[derive(Debug, Serialize)]
struct ParametersReport {
    runs: usize,
    superbatches: usize,
    warmup_superbatches: usize,
    batches_per_superbatch: usize,
    batch_size: usize,
    threads: usize,
    learning_rate: f64,
    score_drop_abs: i32,
    lock_gpu_clock: bool,
}

#[derive(Debug, Serialize)]
struct InputsReport {
    data_id: String,
    data_file_name: Option<String>,
    data_bytes: u64,
    progress_id: Option<String>,
    progress_file_name: Option<String>,
    progress_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct EnvironmentReport {
    platform: &'static str,
    os: &'static str,
    architecture: &'static str,
    gpu: Option<String>,
    driver: Option<String>,
    cuda_toolkit: Option<String>,
    rustc: Option<String>,
    git_commit: Option<String>,
    dirty: Option<bool>,
    cargo_features: Vec<&'static str>,
    command_line: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    id: String,
    architecture: Architecture,
    feature_set: String,
    training_arguments: Vec<String>,
    runs: Vec<RunReport>,
}

#[derive(Debug, Serialize)]
struct RunReport {
    run: usize,
    order_in_run: usize,
    elapsed_ms: u64,
    superbatches: Vec<SuperbatchMeasurement>,
    measured_mean_pos_per_sec: f64,
    log_file: String,
}

#[derive(Debug, Clone, Serialize)]
struct SuperbatchMeasurement {
    superbatch: usize,
    pos_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    case_id: String,
    runs: Vec<f64>,
    mean_pos_per_sec: f64,
    median_pos_per_sec: f64,
    sample_sd_pos_per_sec: f64,
    min_pos_per_sec: f64,
    max_pos_per_sec: f64,
    coefficient_of_variation_percent: f64,
}

struct ResolvedLocalConfig {
    data: PathBuf,
    data_id: String,
    progress_coeff: Option<PathBuf>,
    progress_id: Option<String>,
    threads: usize,
    lock_gpu_clock: bool,
    work_dir: PathBuf,
}

struct GpuClockGuard {
    locked: bool,
}

impl GpuClockGuard {
    fn acquire(requested: bool) -> Self {
        if !requested {
            return Self { locked: false };
        }
        let Some(max_clock) = command_output(
            "nvidia-smi",
            &["--query-supported-clocks=gr", "--format=csv,noheader"],
        )
        .and_then(|output| {
            output
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().next())
                .map(str::to_owned)
        }) else {
            eprintln!("[bench-pos] warning: could not query a supported GPU graphics clock");
            return Self { locked: false };
        };
        let status = Command::new("nvidia-smi")
            .args(["-lgc", max_clock.as_str()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if status.is_ok_and(|status| status.success()) {
            eprintln!("[bench-pos] locked GPU graphics clock to {max_clock} MHz");
            Self { locked: true }
        } else {
            eprintln!(
                "[bench-pos] warning: GPU clock lock failed; continuing without a clock lock"
            );
            Self { locked: false }
        }
    }
}

impl Drop for GpuClockGuard {
    fn drop(&mut self) {
        if self.locked {
            let _ = Command::new("nvidia-smi")
                .arg("-rgc")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            eprintln!("[bench-pos] released GPU graphics clock lock");
        }
    }
}

pub(crate) fn run(args: &BenchPosArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_invocation_args(std::env::args().skip(1))?;
    let profile = read_toml::<BenchmarkProfile>(&args.profile, "benchmark profile")?;
    let local = read_toml::<LocalConfig>(&args.local_config, "machine-local config")?;
    validate_profile(&profile)?;
    let local = resolve_local_config(&args.local_config, local)?;
    validate_local_config(&local)?;

    let dirty = git_dirty();
    if dirty == Some(true) && !args.allow_dirty {
        return Err(
            "working tree is dirty; commit changes or pass --allow-dirty (the report records dirty=true)"
                .into(),
        );
    }

    let selected = select_cases(&profile.cases, &args.cases)?;
    if selected
        .iter()
        .any(|case| case.architecture == Architecture::Layerstack)
        && local.progress_coeff.is_none()
    {
        return Err(
            "selected LayerStack case requires progress_coeff in bench-pos.local.toml".into(),
        );
    }

    fs::create_dir_all(&local.work_dir)?;
    fs::create_dir_all(&args.output_dir)?;
    let timestamp_unix_ms = unix_timestamp_ms()?;
    let inputs = capture_inputs(&local)?;
    let environment = capture_environment(dirty);
    let log_dir = args.output_dir.join(format!("{timestamp_unix_ms}-logs"));
    fs::create_dir_all(&log_dir)?;
    let executable = std::env::current_exe()?;
    let _clock_guard = GpuClockGuard::acquire(local.lock_gpu_clock);

    eprintln!(
        "[bench-pos] profile={}, cases={}, runs={}, superbatches={} (warmup={}), batches/sb={}, batch_size={}, threads={}",
        profile.profile,
        selected
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>()
            .join(","),
        profile.runs,
        profile.superbatches,
        profile.warmup_superbatches,
        profile.batches_per_superbatch,
        profile.batch_size,
        local.threads,
    );
    if let Some(gpu_state) = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=temperature.gpu,utilization.gpu",
            "--format=csv,noheader",
        ],
    ) {
        eprintln!("[bench-pos] GPU start: {gpu_state}");
    }

    let mut case_reports = selected
        .iter()
        .map(|case| CaseReport {
            id: case.id.clone(),
            architecture: case.architecture,
            feature_set: case.feature_set.clone(),
            training_arguments: display_training_arguments(&profile, &local, case),
            runs: Vec::new(),
        })
        .collect::<Vec<_>>();
    for run_index in 1..=profile.runs {
        let case_indices = case_indices_for_run(selected.len(), run_index);
        for (order_index, case_index) in case_indices.into_iter().enumerate() {
            let case = selected[case_index];
            let command = training_command(&executable, &profile, &local, case);
            let started = Instant::now();
            let output = execute(command)?;
            let elapsed_ms = u64::try_from(started.elapsed().as_millis())?;
            let log_path = log_dir.join(format!("{}-run-{run_index}.log", case.id));
            write_process_log(&log_path, &output)?;
            if !output.status.success() {
                print_process_output(&output);
                return Err(format!(
                    "case {} run {run_index} failed with {}; log: {}",
                    case.id,
                    output.status,
                    log_path.display()
                )
                .into());
            }
            let measurements = parse_superbatch_measurements(&output)?;
            validate_measurements(&profile, &measurements)?;
            let measured = measurements
                .iter()
                .filter(|measurement| measurement.superbatch > profile.warmup_superbatches)
                .map(|measurement| measurement.pos_per_sec)
                .collect::<Vec<_>>();
            let mean = statistics(&measured).mean;
            eprintln!(
                "[bench-pos-run] case={}, run={}/{}, order={}, sb{}-{} mean={:.0} pos/s, elapsed_ms={elapsed_ms}",
                case.id,
                run_index,
                profile.runs,
                order_index + 1,
                profile.warmup_superbatches + 1,
                profile.superbatches,
                mean,
            );
            case_reports[case_index].runs.push(RunReport {
                run: run_index,
                order_in_run: order_index + 1,
                elapsed_ms,
                superbatches: measurements,
                measured_mean_pos_per_sec: mean,
                log_file: report_relative_path(&args.output_dir, &log_path),
            });
        }
    }

    let summaries = summarize_cases(&case_reports);
    for summary in &summaries {
        eprintln!(
            "[bench-pos-summary] case={}, runs={}, mean={:.0}, median={:.0}, sample_sd={:.0}, min={:.0}, max={:.0} pos/s, CV={:.2}%",
            summary.case_id,
            summary
                .runs
                .iter()
                .map(|value| format!("{value:.0}"))
                .collect::<Vec<_>>()
                .join("/"),
            summary.mean_pos_per_sec,
            summary.median_pos_per_sec,
            summary.sample_sd_pos_per_sec,
            summary.min_pos_per_sec,
            summary.max_pos_per_sec,
            summary.coefficient_of_variation_percent,
        );
    }

    let report = BenchmarkReport {
        schema_version: SCHEMA_VERSION,
        timestamp_unix_ms,
        profile: profile.profile,
        parameters: ParametersReport {
            runs: profile.runs,
            superbatches: profile.superbatches,
            warmup_superbatches: profile.warmup_superbatches,
            batches_per_superbatch: profile.batches_per_superbatch,
            batch_size: profile.batch_size,
            threads: local.threads,
            learning_rate: profile.learning_rate,
            score_drop_abs: profile.score_drop_abs,
            lock_gpu_clock: local.lock_gpu_clock,
        },
        inputs,
        environment,
        cases: case_reports,
        summaries,
    };
    let report_path = write_report(&args.output_dir, timestamp_unix_ms, &report)?;
    eprintln!("[bench-pos] report={}", report_path.display());
    Ok(())
}

fn case_indices_for_run(case_count: usize, run_index: usize) -> Vec<usize> {
    if run_index.is_multiple_of(2) {
        (0..case_count).rev().collect()
    } else {
        (0..case_count).collect()
    }
}

fn validate_invocation_args(
    arguments: impl IntoIterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let Some(subcommand_index) = arguments
        .iter()
        .position(|argument| argument == "bench-pos")
    else {
        return Err("bench-pos subcommand is missing from the process arguments".into());
    };
    if subcommand_index != 0 {
        return Err("place every bench-pos option after the bench-pos subcommand".into());
    }
    let mut index = 1;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "--allow-dirty" {
            index += 1;
            continue;
        }
        if ["--profile", "--local-config", "--case", "--output-dir"]
            .iter()
            .any(|option| argument == option)
        {
            if index + 1 >= arguments.len() {
                return Err(format!("{argument} requires a value").into());
            }
            index += 2;
            continue;
        }
        if ["--profile=", "--local-config=", "--case=", "--output-dir="]
            .iter()
            .any(|prefix| argument.starts_with(prefix))
        {
            index += 1;
            continue;
        }
        return Err(format!(
            "{argument} is a training option, not a bench-pos runner option; put benchmark settings in the tracked profile"
        )
        .into());
    }
    Ok(())
}

fn read_toml<T: for<'de> Deserialize<'de>>(
    path: &Path,
    description: &str,
) -> Result<T, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("could not read {description} {}: {error}", path.display()))?;
    toml::from_str(&text)
        .map_err(|error| format!("invalid {description} {}: {error}", path.display()).into())
}

fn validate_profile(profile: &BenchmarkProfile) -> Result<(), Box<dyn std::error::Error>> {
    if profile.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported benchmark profile schema_version {}; expected {SCHEMA_VERSION}",
            profile.schema_version
        )
        .into());
    }
    if !valid_id(&profile.profile) {
        return Err(
            "benchmark profile name must contain only ASCII letters, digits, '.', '-', or '_'"
                .into(),
        );
    }
    if profile.runs == 0
        || profile.superbatches == 0
        || profile.batches_per_superbatch == 0
        || profile.batch_size == 0
    {
        return Err(
            "runs, superbatches, batches_per_superbatch, and batch_size must be positive".into(),
        );
    }
    if profile.warmup_superbatches >= profile.superbatches {
        return Err("warmup_superbatches must be smaller than superbatches".into());
    }
    if !profile.learning_rate.is_finite() || profile.learning_rate <= 0.0 {
        return Err("learning_rate must be finite and positive".into());
    }
    if profile.score_drop_abs <= 0 {
        return Err("score_drop_abs must be positive".into());
    }
    if profile.cases.is_empty() {
        return Err("benchmark profile must contain at least one case".into());
    }
    let mut ids = BTreeSet::new();
    for case in &profile.cases {
        if !valid_id(&case.id) {
            return Err(format!(
                "case id {:?} must contain only ASCII letters, digits, '.', '-', or '_'",
                case.id
            )
            .into());
        }
        if !ids.insert(case.id.as_str()) {
            return Err(format!("duplicate benchmark case id: {}", case.id).into());
        }
        if case.feature_set.trim().is_empty() {
            return Err(format!("case {} has an empty feature_set", case.id).into());
        }
        validate_extra_args(case)?;
    }
    Ok(())
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn validate_extra_args(case: &BenchmarkCase) -> Result<(), Box<dyn std::error::Error>> {
    const CONTROLLED_GLOBAL: &[&str] = &[
        "--data",
        "--feature-set",
        "--output",
        "--net-id",
        "--superbatches",
        "--batches-per-superbatch",
        "--batch-size",
        "--lr",
        "--score-drop-abs",
        "--save-rate",
        "--threads",
    ];
    for argument in case.global_args.iter().chain(&case.architecture_args) {
        if CONTROLLED_GLOBAL.iter().any(|controlled| {
            argument == controlled || argument.starts_with(&format!("{controlled}="))
        }) {
            return Err(format!(
                "case {} may not override runner-controlled option {argument}",
                case.id
            )
            .into());
        }
    }
    if case
        .architecture_args
        .iter()
        .any(|argument| argument == "--progress-coeff" || argument.starts_with("--progress-coeff="))
    {
        return Err(format!(
            "case {} architecture_args may not override machine-local --progress-coeff",
            case.id
        )
        .into());
    }
    Ok(())
}

fn resolve_local_config(
    config_path: &Path,
    local: LocalConfig,
) -> Result<ResolvedLocalConfig, Box<dyn std::error::Error>> {
    if local.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported local config schema_version {}; expected {SCHEMA_VERSION}",
            local.schema_version
        )
        .into());
    }
    let base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(ResolvedLocalConfig {
        data: resolve_path(base, local.data),
        data_id: local.data_id,
        progress_coeff: local.progress_coeff.map(|path| resolve_path(base, path)),
        progress_id: local.progress_id,
        threads: local.threads,
        lock_gpu_clock: local.lock_gpu_clock,
        work_dir: local.work_dir.map_or_else(
            || base.join(DEFAULT_WORK_DIR),
            |path| resolve_path(base, path),
        ),
    })
}

fn resolve_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn validate_local_config(local: &ResolvedLocalConfig) -> Result<(), Box<dyn std::error::Error>> {
    if local.threads == 0 {
        return Err("threads must be positive".into());
    }
    if local.data_id.trim().is_empty() {
        return Err("data_id must not be empty".into());
    }
    require_file(&local.data, "data")?;
    if let Some(progress) = &local.progress_coeff {
        require_file(progress, "progress_coeff")?;
        if local
            .progress_id
            .as_deref()
            .is_none_or(|id| id.trim().is_empty())
        {
            return Err("progress_id is required when progress_coeff is set".into());
        }
    } else if local.progress_id.is_some() {
        return Err("progress_id requires progress_coeff".into());
    }
    Ok(())
}

fn require_file(path: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !path.is_file() {
        return Err(format!("{name} file does not exist: {}", path.display()).into());
    }
    if fs::metadata(path)?.len() == 0 {
        return Err(format!("{name} file is empty: {}", path.display()).into());
    }
    Ok(())
}

fn select_cases<'a>(
    cases: &'a [BenchmarkCase],
    requested: &[String],
) -> Result<Vec<&'a BenchmarkCase>, Box<dyn std::error::Error>> {
    if requested.is_empty() {
        return Ok(cases.iter().collect());
    }
    let by_id = cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();
    for id in requested {
        let Some(case) = by_id.get(id.as_str()) else {
            return Err(format!("unknown benchmark case {id:?}").into());
        };
        if seen.insert(id.as_str()) {
            selected.push(*case);
        }
    }
    Ok(selected)
}

fn training_command(
    executable: &Path,
    profile: &BenchmarkProfile,
    local: &ResolvedLocalConfig,
    case: &BenchmarkCase,
) -> Command {
    let mut command = Command::new(executable);
    command.args(training_arguments(profile, local, case, false));
    command
}

fn display_training_arguments(
    profile: &BenchmarkProfile,
    local: &ResolvedLocalConfig,
    case: &BenchmarkCase,
) -> Vec<String> {
    training_arguments(profile, local, case, true)
        .into_iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect()
}

fn training_arguments(
    profile: &BenchmarkProfile,
    local: &ResolvedLocalConfig,
    case: &BenchmarkCase,
    redact_paths: bool,
) -> Vec<OsString> {
    let data = if redact_paths {
        OsString::from("<DATA>")
    } else {
        local.data.as_os_str().to_owned()
    };
    let output = if redact_paths {
        OsString::from("<WORK_DIR>")
    } else {
        local.work_dir.as_os_str().to_owned()
    };
    let mut arguments = vec![
        "--data".into(),
        data,
        "--feature-set".into(),
        case.feature_set.clone().into(),
        "--output".into(),
        output,
        "--net-id".into(),
        format!("bench-pos-{}", case.id).into(),
        "--superbatches".into(),
        profile.superbatches.to_string().into(),
        "--batches-per-superbatch".into(),
        profile.batches_per_superbatch.to_string().into(),
        "--batch-size".into(),
        profile.batch_size.to_string().into(),
        "--lr".into(),
        profile.learning_rate.to_string().into(),
        "--win-rate-model".into(),
        "--score-drop-abs".into(),
        profile.score_drop_abs.to_string().into(),
        "--save-rate".into(),
        profile.superbatches.to_string().into(),
        "--threads".into(),
        local.threads.to_string().into(),
    ];
    arguments.extend(case.global_args.iter().map(OsString::from));
    arguments.push(case.architecture.name().into());
    if case.architecture == Architecture::Layerstack {
        arguments.push("--progress-coeff".into());
        arguments.push(if redact_paths {
            OsString::from("<PROGRESS_COEFF>")
        } else {
            local
                .progress_coeff
                .as_ref()
                .expect("validated LayerStack progress coefficient")
                .as_os_str()
                .to_owned()
        });
    }
    arguments.extend(case.architecture_args.iter().map(OsString::from));
    arguments
}

fn execute(mut command: Command) -> Result<Output, Box<dyn std::error::Error>> {
    command
        .output()
        .map_err(|error| format!("could not start training process: {error}").into())
}

fn write_process_log(path: &Path, output: &Output) -> Result<(), Box<dyn std::error::Error>> {
    let mut log = Vec::new();
    log.extend_from_slice(b"[stdout]\n");
    log.extend_from_slice(&output.stdout);
    if !output.stdout.ends_with(b"\n") {
        log.push(b'\n');
    }
    log.extend_from_slice(b"[stderr]\n");
    log.extend_from_slice(&output.stderr);
    if !output.stderr.ends_with(b"\n") {
        log.push(b'\n');
    }
    fs::write(path, log)?;
    Ok(())
}

fn print_process_output(output: &Output) {
    eprintln!(
        "[bench-pos] child stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    eprintln!(
        "[bench-pos] child stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn parse_superbatch_measurements(
    output: &Output,
) -> Result<Vec<SuperbatchMeasurement>, Box<dyn std::error::Error>> {
    let mut measurements = Vec::new();
    for bytes in [&output.stdout, &output.stderr] {
        for line in String::from_utf8_lossy(bytes).split(['\n', '\r']) {
            let Some(rest) = line.strip_prefix("[train] superbatch ") else {
                continue;
            };
            let mut fields = rest.split_whitespace();
            let Some(superbatch_field) = fields.next() else {
                continue;
            };
            let Some(superbatch) = superbatch_field
                .split_once('/')
                .and_then(|(current, _)| current.parse::<usize>().ok())
            else {
                continue;
            };
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let Some(index) = tokens.iter().position(|token| *token == "pos/s") else {
                continue;
            };
            let Some(value) = index
                .checked_sub(1)
                .and_then(|value_index| tokens.get(value_index))
                .and_then(|value| value.parse::<f64>().ok())
            else {
                continue;
            };
            measurements.push(SuperbatchMeasurement {
                superbatch,
                pos_per_sec: value,
            });
        }
    }
    measurements.sort_by_key(|measurement| measurement.superbatch);
    if measurements.is_empty() {
        return Err("training output contained no parseable superbatch throughput lines".into());
    }
    Ok(measurements)
}

fn validate_measurements(
    profile: &BenchmarkProfile,
    measurements: &[SuperbatchMeasurement],
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = measurements
        .iter()
        .map(|measurement| measurement.superbatch)
        .collect::<Vec<_>>();
    let expected = (1..=profile.superbatches).collect::<Vec<_>>();
    if actual != expected {
        return Err(format!(
            "expected one throughput line for each superbatch {expected:?}, got {actual:?}"
        )
        .into());
    }
    if measurements
        .iter()
        .any(|measurement| !measurement.pos_per_sec.is_finite() || measurement.pos_per_sec <= 0.0)
    {
        return Err("superbatch throughput must be finite and positive".into());
    }
    Ok(())
}

fn summarize_cases(cases: &[CaseReport]) -> Vec<Summary> {
    cases
        .iter()
        .map(|case| {
            let runs = case
                .runs
                .iter()
                .map(|run| run.measured_mean_pos_per_sec)
                .collect::<Vec<_>>();
            let stats = statistics(&runs);
            Summary {
                case_id: case.id.clone(),
                runs,
                mean_pos_per_sec: stats.mean,
                median_pos_per_sec: stats.median,
                sample_sd_pos_per_sec: stats.sample_sd,
                min_pos_per_sec: stats.min,
                max_pos_per_sec: stats.max,
                coefficient_of_variation_percent: if stats.mean > 0.0 {
                    stats.sample_sd * 100.0 / stats.mean
                } else {
                    0.0
                },
            }
        })
        .collect()
}

struct Statistics {
    mean: f64,
    median: f64,
    sample_sd: f64,
    min: f64,
    max: f64,
}

fn statistics(values: &[f64]) -> Statistics {
    assert!(!values.is_empty());
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sample_sd = if values.len() > 1 {
        let squared = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (squared / (values.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    Statistics {
        mean,
        median,
        sample_sd,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
    }
}

fn capture_inputs(local: &ResolvedLocalConfig) -> Result<InputsReport, Box<dyn std::error::Error>> {
    Ok(InputsReport {
        data_id: local.data_id.clone(),
        data_file_name: file_name(&local.data),
        data_bytes: fs::metadata(&local.data)?.len(),
        progress_id: local.progress_id.clone(),
        progress_file_name: local.progress_coeff.as_deref().and_then(file_name),
        progress_bytes: local
            .progress_coeff
            .as_deref()
            .map(fs::metadata)
            .transpose()?
            .map(|metadata| metadata.len()),
    })
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
}

fn capture_environment(dirty: Option<bool>) -> EnvironmentReport {
    EnvironmentReport {
        platform: platform_name(),
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        gpu: command_output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]),
        driver: command_output(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"],
        ),
        cuda_toolkit: command_output("nvcc", &["--version"]),
        rustc: command_output("rustc", &["--version"]),
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        dirty,
        cargo_features: [
            cfg!(feature = "cuda-oxide").then_some("cuda-oxide"),
            cfg!(feature = "native-cuda").then_some("native-cuda"),
            cfg!(feature = "native-cuda-host").then_some("native-cuda-host"),
        ]
        .into_iter()
        .flatten()
        .collect(),
        command_line: sanitized_command_line(std::env::args()),
    }
}

fn report_relative_path(output_dir: &Path, path: &Path) -> String {
    path.strip_prefix(output_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn sanitized_command_line(arguments: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut sanitized = Vec::new();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--local-config" => {
                sanitized.push(argument);
                if arguments.next().is_some() {
                    sanitized.push("<LOCAL_CONFIG>".into());
                }
            }
            "--output-dir" => {
                sanitized.push(argument);
                if arguments.next().is_some() {
                    sanitized.push("<OUTPUT_DIR>".into());
                }
            }
            _ if argument.starts_with("--local-config=") => {
                sanitized.push("--local-config=<LOCAL_CONFIG>".into());
            }
            _ if argument.starts_with("--output-dir=") => {
                sanitized.push("--output-dir=<OUTPUT_DIR>".into());
            }
            _ => sanitized.push(argument),
        }
    }
    sanitized
}

fn platform_name() -> &'static str {
    if cfg!(target_os = "windows") {
        return "windows";
    }
    if cfg!(target_os = "linux")
        && (std::env::var_os("WSL_DISTRO_NAME").is_some()
            || fs::read_to_string("/proc/sys/kernel/osrelease")
                .is_ok_and(|release| release.to_ascii_lowercase().contains("microsoft")))
    {
        return "wsl";
    }
    std::env::consts::OS
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn git_dirty() -> Option<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    output.status.success().then_some(!output.stdout.is_empty())
}

fn unix_timestamp_ms() -> Result<u64, Box<dyn std::error::Error>> {
    Ok(u64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
    )?)
}

fn write_report(
    output_dir: &Path,
    timestamp_unix_ms: u64,
    report: &BenchmarkReport,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = output_dir.join(format!(
        "{timestamp_unix_ms}-{}-{}-v{SCHEMA_VERSION}.json",
        report.environment.platform, report.profile
    ));
    let temporary = path.with_extension("json.tmp");
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    fs::write(&temporary, json)?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::ExitStatus;

    #[cfg(unix)]
    fn success_status() -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(0)
    }

    #[cfg(windows)]
    fn success_status() -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(0)
    }

    fn output(stdout: &str) -> Output {
        Output {
            status: success_status(),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    #[test]
    fn tracked_profile_parses_and_has_unique_cases() {
        let profile: BenchmarkProfile =
            toml::from_str(include_str!("../../../bench-pos.toml")).expect("profile parses");
        validate_profile(&profile).expect("profile validates");
        assert_eq!(profile.cases.len(), 4);
    }

    #[test]
    fn profile_rejects_controlled_options_in_both_argument_lists() {
        let base = BenchmarkCase {
            id: "test".into(),
            architecture: Architecture::Simple,
            feature_set: "halfkp".into(),
            global_args: vec!["--batch-size=1024".into()],
            architecture_args: Vec::new(),
        };
        assert!(validate_extra_args(&base).is_err());
        let architecture_override = BenchmarkCase {
            global_args: Vec::new(),
            architecture_args: vec!["--threads".into(), "1".into()],
            ..base
        };
        assert!(validate_extra_args(&architecture_override).is_err());
    }

    #[test]
    fn superbatch_parser_collects_complete_lines() {
        let output = output(
            "[train] superbatch 1/3 | loss 0.1 | 1000000 pos/s | lr 1e-3\n\
             [train] superbatch 2/3 | loss 0.1 | 1200000 pos/s | lr 1e-3\r\n\
             [train] superbatch 3/3 | loss 0.1 | 1400000 pos/s | lr 1e-3\n",
        );
        let values = parse_superbatch_measurements(&output).expect("parse");
        assert_eq!(values.len(), 3);
        assert_eq!(values[1].superbatch, 2);
        assert_eq!(values[1].pos_per_sec, 1_200_000.0);
    }

    #[test]
    fn statistics_match_sample_standard_deviation_and_cv_inputs() {
        let stats = statistics(&[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert!((stats.sample_sd - 1.825_741_858_350_553_8).abs() < 1.0e-12);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn report_arguments_redact_machine_local_paths() {
        let profile: BenchmarkProfile =
            toml::from_str(include_str!("../../../bench-pos.toml")).expect("profile parses");
        let local = ResolvedLocalConfig {
            data: PathBuf::from("/secret/teacher.psv"),
            data_id: "teacher-v1".into(),
            progress_coeff: Some(PathBuf::from("/secret/progress.bin")),
            progress_id: Some("progress-v1".into()),
            threads: 16,
            lock_gpu_clock: false,
            work_dir: PathBuf::from("/secret/work"),
        };
        let arguments = display_training_arguments(&profile, &local, &profile.cases[0]);
        let joined = arguments.join(" ");
        assert!(joined.contains("<DATA>"));
        assert!(joined.contains("<PROGRESS_COEFF>"));
        assert!(joined.contains("<WORK_DIR>"));
        assert!(!joined.contains("/secret"));
    }

    #[test]
    fn runner_command_line_redacts_local_and_output_paths() {
        let arguments = [
            "nnue-train",
            "bench-pos",
            "--local-config",
            "/secret/local.toml",
            "--output-dir=/secret/results",
        ]
        .into_iter()
        .map(str::to_owned);
        let sanitized = sanitized_command_line(arguments).join(" ");
        assert!(sanitized.contains("--local-config <LOCAL_CONFIG>"));
        assert!(sanitized.contains("--output-dir=<OUTPUT_DIR>"));
        assert!(!sanitized.contains("/secret"));
    }

    #[test]
    fn invocation_rejects_inherited_training_options() {
        validate_invocation_args(
            ["bench-pos", "--case", "layerstack-fp32"]
                .into_iter()
                .map(str::to_owned),
        )
        .expect("runner options should pass");
        let error = validate_invocation_args(
            ["bench-pos", "--batch-size", "1024"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert!(error.to_string().contains("tracked profile"));
    }

    #[test]
    fn case_order_reverses_on_even_runs() {
        assert_eq!(case_indices_for_run(4, 1), [0, 1, 2, 3]);
        assert_eq!(case_indices_for_run(4, 2), [3, 2, 1, 0]);
        assert_eq!(case_indices_for_run(4, 3), [0, 1, 2, 3]);
    }

    #[test]
    fn serialized_report_conforms_to_versioned_json_schema() {
        let case = CaseReport {
            id: "simple-halfkp-fp32".into(),
            architecture: Architecture::Simple,
            feature_set: "halfkp".into(),
            training_arguments: vec!["--data".into(), "<DATA>".into(), "simple".into()],
            runs: vec![RunReport {
                run: 1,
                order_in_run: 1,
                elapsed_ms: 2_000,
                superbatches: vec![
                    SuperbatchMeasurement {
                        superbatch: 1,
                        pos_per_sec: 1_000_000.0,
                    },
                    SuperbatchMeasurement {
                        superbatch: 2,
                        pos_per_sec: 1_200_000.0,
                    },
                ],
                measured_mean_pos_per_sec: 1_200_000.0,
                log_file: "123-logs/simple-halfkp-fp32-run-1.log".into(),
            }],
        };
        let report = BenchmarkReport {
            schema_version: SCHEMA_VERSION,
            timestamp_unix_ms: 1_784_400_000_000,
            profile: "test-v1".into(),
            parameters: ParametersReport {
                runs: 1,
                superbatches: 2,
                warmup_superbatches: 1,
                batches_per_superbatch: 2,
                batch_size: 16_384,
                threads: 16,
                learning_rate: 8.75e-4,
                score_drop_abs: 32_000,
                lock_gpu_clock: false,
            },
            inputs: InputsReport {
                data_id: "teacher-v1".into(),
                data_file_name: Some("teacher.psv".into()),
                data_bytes: 40_000,
                progress_id: None,
                progress_file_name: None,
                progress_bytes: None,
            },
            environment: EnvironmentReport {
                platform: "windows",
                os: "windows",
                architecture: "x86_64",
                gpu: Some("NVIDIA GeForce RTX 5090".into()),
                driver: Some("596.36".into()),
                cuda_toolkit: Some("CUDA 12.9".into()),
                rustc: Some("rustc test".into()),
                git_commit: Some("0123456789abcdef".into()),
                dirty: Some(false),
                cargo_features: vec!["native-cuda-host"],
                command_line: vec!["nnue-train".into(), "bench-pos".into()],
            },
            summaries: summarize_cases(std::slice::from_ref(&case)),
            cases: vec![case],
        };
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/schemas/bench-pos-v1.schema.json"
        ))
        .expect("benchmark schema must be valid JSON");
        let instance = serde_json::to_value(report).expect("benchmark report must serialize");
        let validator = jsonschema::validator_for(&schema).expect("benchmark schema must compile");
        let errors = validator
            .iter_errors(&instance)
            .map(|error| error.to_string())
            .collect::<Vec<_>>();
        assert!(errors.is_empty(), "schema validation failed: {errors:#?}");
    }
}
