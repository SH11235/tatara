//! Fixed-fixture native CUDA throughput benchmark.
//!
//! Timing, backend ordering, statistics, environment capture, and JSON output live here so the
//! same implementation runs on Linux/WSL and native Windows. OS-specific shell setup is not part
//! of the benchmark contract.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use gpu_runtime::CudaContext;
use nnue_format::{SimpleActivation, SimpleId};
use nnue_train::{
    dataloader::BucketMode,
    init::{LayerStackInit, SimpleInit},
    optimizer::OptimizerKind,
    trainer::TrainerBackend,
};
use serde::Serialize;
use shogi_features::FeatureSet;

use crate::{
    arch::SMOKE_LOSS_WRM,
    cli::{
        NativeBenchArchitectureArg, NativeBenchArgs, NativeBenchModeArg, NativeBenchPrecisionArg,
        NativeBenchProfileArg,
    },
    trainer_common::{BatchData, PrecisionFlags},
    trainer_layerstack::{GpuTrainer as LayerStackGpuTrainer, OptimGroupConfig},
    trainer_simple::SimpleGpuTrainer,
};

#[cfg(feature = "native-cuda")]
use crate::kernel_module::with_native_backend;

const SCHEMA_VERSION: u32 = 1;
const PROFILE_NAME: &str = "v1";
const DEFAULT_BATCH_SIZE: usize = 16_384;
const DEFAULT_WARMUP_STEPS: usize = 3;
const DEFAULT_STEPS: usize = 100;
const DEFAULT_RUNS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    fn fixture_id(self) -> &'static str {
        match self {
            Self::Layerstack => "layerstack-halfka-hm-merged-factorized-v1",
            Self::Simple => "simple-halfkp-factorized-v1",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PrecisionSpec {
    name: &'static str,
    flags: PrecisionFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Backend {
    CudaOxide,
    NativeCuda,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::CudaOxide => "cuda-oxide",
            Self::NativeCuda => "native-cuda",
        }
    }

    fn is_native(self) -> bool {
        self == Self::NativeCuda
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct PrecisionFlagsReport {
    tf32: bool,
    ft_fp16: bool,
    ft_fp16_out: bool,
    fp16_opt_state: bool,
}

impl From<PrecisionFlags> for PrecisionFlagsReport {
    fn from(flags: PrecisionFlags) -> Self {
        Self {
            tf32: flags.tf32,
            ft_fp16: flags.ft_fp16,
            ft_fp16_out: flags.ft_fp16_out,
            fp16_opt_state: flags.fp16_opt_state,
        }
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkParameters {
    batch_size: usize,
    warmup_steps: usize,
    steps: usize,
    runs: usize,
    device: usize,
    customized: bool,
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

#[derive(Debug, Clone, Serialize)]
struct Measurement {
    architecture: &'static str,
    fixture_id: &'static str,
    precision: &'static str,
    precision_flags: PrecisionFlagsReport,
    backend: Backend,
    run: usize,
    order_in_run: usize,
    positions: u64,
    elapsed_ns: u64,
    pos_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    architecture: &'static str,
    fixture_id: &'static str,
    precision: &'static str,
    backend: Backend,
    runs: Vec<f64>,
    mean_pos_per_sec: f64,
    median_pos_per_sec: f64,
    sample_sd_pos_per_sec: f64,
    min_pos_per_sec: f64,
    max_pos_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct BackendComparison {
    architecture: &'static str,
    precision: &'static str,
    paired_deltas_percent: Vec<f64>,
    mean_paired_delta_percent: f64,
    sample_sd_paired_delta_percent: f64,
    native_over_oxide_ratio: f64,
}

#[derive(Debug, Serialize)]
struct PrecisionSpeedup {
    architecture: &'static str,
    backend: Backend,
    all_optim_over_fp32_ratio: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    timestamp_unix_ms: u64,
    profile: &'static str,
    mode: &'static str,
    parameters: BenchmarkParameters,
    environment: EnvironmentReport,
    measurements: Vec<Measurement>,
    summaries: Vec<Summary>,
    backend_comparisons: Vec<BackendComparison>,
    precision_speedups: Vec<PrecisionSpeedup>,
}

trait BenchTrainer {
    fn bench_step(&mut self, batch: &BatchData<'_>) -> Result<(), Box<dyn std::error::Error>>;
    fn bench_flush(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

impl BenchTrainer for LayerStackGpuTrainer {
    fn bench_step(&mut self, batch: &BatchData<'_>) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.step(batch, 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
        Ok(())
    }

    fn bench_flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = TrainerBackend::flush_pending_loss(self)?;
        Ok(())
    }
}

impl BenchTrainer for SimpleGpuTrainer {
    fn bench_step(&mut self, batch: &BatchData<'_>) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.step(batch, 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
        Ok(())
    }

    fn bench_flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = TrainerBackend::flush_pending_loss(self)?;
        Ok(())
    }
}

struct TimedRun {
    positions: u64,
    elapsed_ns: u64,
    pos_per_sec: f64,
}

pub(crate) fn run(
    args: &NativeBenchArgs,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_args(args, batch_size)?;
    if args.mode == NativeBenchModeArg::Compare && !cfg!(feature = "native-cuda") {
        return Err("--mode compare requires --features native-cuda (cuda-oxide is unavailable in native-cuda-host builds)".into());
    }

    let timestamp_unix_ms = unix_timestamp_ms()?;
    let environment = capture_environment(args.device);
    if environment.dirty == Some(true) && !args.allow_dirty {
        return Err(
            "working tree is dirty; commit changes or pass --allow-dirty (the report records dirty=true)"
                .into(),
        );
    }

    let architectures = selected_architectures(args.architecture);
    let precisions = selected_precisions(args.precision);
    let context = CudaContext::new(args.device)?;
    let mut measurements = Vec::new();

    eprintln!(
        "[native-bench] profile={PROFILE_NAME}, mode={}, architectures={}, precisions={}, batch={}, warmup={}, steps={}, runs={}, device={}",
        mode_name(args.mode),
        architectures
            .iter()
            .map(|architecture| architecture.name())
            .collect::<Vec<_>>()
            .join(","),
        precisions
            .iter()
            .map(|precision| precision.name)
            .collect::<Vec<_>>()
            .join(","),
        batch_size,
        args.warmup_steps,
        args.steps,
        args.runs,
        args.device,
    );

    for architecture in architectures {
        benchmark_architecture(
            &context,
            architecture,
            &precisions,
            args,
            batch_size,
            &mut measurements,
        )?;
    }

    let summaries = build_summaries(&measurements);
    let backend_comparisons = build_backend_comparisons(&measurements, &summaries);
    let precision_speedups = build_precision_speedups(&summaries);
    print_summaries(&summaries, &backend_comparisons, &precision_speedups);

    let report = BenchmarkReport {
        schema_version: SCHEMA_VERSION,
        timestamp_unix_ms,
        profile: PROFILE_NAME,
        mode: mode_name(args.mode),
        parameters: BenchmarkParameters {
            batch_size,
            warmup_steps: args.warmup_steps,
            steps: args.steps,
            runs: args.runs,
            device: args.device,
            customized: batch_size != DEFAULT_BATCH_SIZE
                || args.warmup_steps != DEFAULT_WARMUP_STEPS
                || args.steps != DEFAULT_STEPS
                || args.runs != DEFAULT_RUNS,
        },
        environment,
        measurements,
        summaries,
        backend_comparisons,
        precision_speedups,
    };
    let output = write_report(&args.output_dir, timestamp_unix_ms, args.mode, &report)?;
    eprintln!("[native-bench] report={}", output.display());
    Ok(())
}

fn validate_args(
    args: &NativeBenchArgs,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.profile != NativeBenchProfileArg::V1 {
        return Err("unsupported native benchmark profile".into());
    }
    if batch_size == 0 {
        return Err("--batch-size must be greater than zero".into());
    }
    if args.steps == 0 {
        return Err("--steps must be greater than zero".into());
    }
    if args.runs == 0 {
        return Err("--runs must be greater than zero".into());
    }
    Ok(())
}

fn selected_architectures(selection: NativeBenchArchitectureArg) -> Vec<Architecture> {
    match selection {
        NativeBenchArchitectureArg::Layerstack => vec![Architecture::Layerstack],
        NativeBenchArchitectureArg::Simple => vec![Architecture::Simple],
        NativeBenchArchitectureArg::All => vec![Architecture::Layerstack, Architecture::Simple],
    }
}

fn selected_precisions(selection: NativeBenchPrecisionArg) -> Vec<PrecisionSpec> {
    let fp32 = PrecisionSpec {
        name: "fp32",
        flags: PrecisionFlags::default(),
    };
    let all_optim = PrecisionSpec {
        name: "all-optim",
        flags: PrecisionFlags {
            tf32: true,
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
        },
    };
    match selection {
        NativeBenchPrecisionArg::Fp32 => vec![fp32],
        NativeBenchPrecisionArg::AllOptim => vec![all_optim],
        NativeBenchPrecisionArg::All => vec![fp32, all_optim],
    }
}

fn mode_name(mode: NativeBenchModeArg) -> &'static str {
    match mode {
        NativeBenchModeArg::NativeOnly => "native-only",
        NativeBenchModeArg::Compare => "compare",
    }
}

fn benchmark_architecture(
    context: &std::sync::Arc<CudaContext>,
    architecture: Architecture,
    precisions: &[PrecisionSpec],
    args: &NativeBenchArgs,
    batch_size: usize,
    measurements: &mut Vec<Measurement>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut owned = BatchData::smoke_dummy(batch_size, fixture_feature_set(architecture));
    if architecture == Architecture::Layerstack {
        for (row, bucket) in owned.bucket_idx.iter_mut().enumerate() {
            *bucket = (row % 9) as i32;
        }
    }
    owned.score.fill(200.0);
    owned.wdl.fill(0.8);
    let batch = owned.as_ref();

    for run in 0..args.runs {
        let precision_order: Box<dyn Iterator<Item = &PrecisionSpec>> = if run.is_multiple_of(2) {
            Box::new(precisions.iter())
        } else {
            Box::new(precisions.iter().rev())
        };
        for precision in precision_order {
            let backends = match args.mode {
                NativeBenchModeArg::NativeOnly => [Some(Backend::NativeCuda), None],
                NativeBenchModeArg::Compare if run.is_multiple_of(2) => {
                    [Some(Backend::CudaOxide), Some(Backend::NativeCuda)]
                }
                NativeBenchModeArg::Compare => {
                    [Some(Backend::NativeCuda), Some(Backend::CudaOxide)]
                }
            };
            for (order, backend) in backends.into_iter().flatten().enumerate() {
                let timed = benchmark_one(
                    context,
                    architecture,
                    precision.flags,
                    backend,
                    &batch,
                    args.warmup_steps,
                    args.steps,
                )?;
                eprintln!(
                    "[native-bench-run] architecture={}, precision={}, run={}, order={}, backend={}, elapsed_ms={:.3}, pos_per_sec={:.0}",
                    architecture.name(),
                    precision.name,
                    run + 1,
                    order + 1,
                    backend.name(),
                    timed.elapsed_ns as f64 / 1.0e6,
                    timed.pos_per_sec,
                );
                measurements.push(Measurement {
                    architecture: architecture.name(),
                    fixture_id: architecture.fixture_id(),
                    precision: precision.name,
                    precision_flags: precision.flags.into(),
                    backend,
                    run: run + 1,
                    order_in_run: order + 1,
                    positions: timed.positions,
                    elapsed_ns: timed.elapsed_ns,
                    pos_per_sec: timed.pos_per_sec,
                });
            }
        }
    }
    Ok(())
}

fn fixture_feature_set(architecture: Architecture) -> shogi_features::FeatureSetSpec {
    match architecture {
        Architecture::Layerstack => FeatureSet::HalfKaHmMerged.spec().with_ft_factorize(),
        Architecture::Simple => FeatureSet::HalfKp.spec().with_ft_factorize(),
    }
}

fn benchmark_one(
    context: &std::sync::Arc<CudaContext>,
    architecture: Architecture,
    precision: PrecisionFlags,
    backend: Backend,
    batch: &BatchData<'_>,
    warmup_steps: usize,
    steps: usize,
) -> Result<TimedRun, Box<dyn std::error::Error>> {
    match architecture {
        Architecture::Layerstack => {
            let trainer = create_layerstack_trainer(context, backend, batch.n_pos, precision)?;
            time_trainer(trainer, batch, warmup_steps, steps)
        }
        Architecture::Simple => {
            let trainer = create_simple_trainer(context, backend, batch.n_pos, precision)?;
            time_trainer(trainer, batch, warmup_steps, steps)
        }
    }
}

fn create_layerstack_trainer(
    context: &std::sync::Arc<CudaContext>,
    backend: Backend,
    batch_size: usize,
    precision: PrecisionFlags,
) -> Result<LayerStackGpuTrainer, Box<dyn std::error::Error>> {
    create_with_backend(backend, || {
        LayerStackGpuTrainer::new(
            context,
            batch_size,
            1536,
            16,
            32,
            9,
            BucketMode::Progress8KpAbs,
            precision,
            fixture_feature_set(Architecture::Layerstack),
            OptimizerKind::Ranger,
            OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
            None,
            None,
            &LayerStackInit::default_uniform(),
        )
    })
}

fn create_simple_trainer(
    context: &std::sync::Arc<CudaContext>,
    backend: Backend,
    batch_size: usize,
    precision: PrecisionFlags,
) -> Result<SimpleGpuTrainer, Box<dyn std::error::Error>> {
    let id = SimpleId {
        feature_set: fixture_feature_set(Architecture::Simple),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    let mut trainer = create_with_backend(backend, || {
        SimpleGpuTrainer::new(
            context,
            batch_size,
            id,
            OptimizerKind::Ranger,
            0.0,
            None,
            16,
            precision,
            &SimpleInit::default_uniform(),
        )
    })?;
    trainer.sync_ft_forward_weights()?;
    Ok(trainer)
}

fn create_with_backend<T>(
    backend: Backend,
    operation: impl FnOnce() -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    #[cfg(feature = "native-cuda")]
    return with_native_backend(backend.is_native(), operation);

    #[cfg(feature = "native-cuda-host")]
    {
        if !backend.is_native() {
            return Err("cuda-oxide is unavailable in native-cuda-host builds".into());
        }
        operation()
    }
}

fn time_trainer<T: BenchTrainer>(
    mut trainer: T,
    batch: &BatchData<'_>,
    warmup_steps: usize,
    steps: usize,
) -> Result<TimedRun, Box<dyn std::error::Error>> {
    for _ in 0..warmup_steps {
        trainer.bench_step(batch)?;
    }
    trainer.bench_flush()?;

    let start = Instant::now();
    for _ in 0..steps {
        trainer.bench_step(batch)?;
    }
    trainer.bench_flush()?;
    let elapsed = start.elapsed();
    let elapsed_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    let positions = u64::try_from(batch.n_pos)?
        .checked_mul(u64::try_from(steps)?)
        .ok_or("position count overflow")?;
    Ok(TimedRun {
        positions,
        elapsed_ns,
        pos_per_sec: positions as f64 / elapsed.as_secs_f64(),
    })
}

fn build_summaries(measurements: &[Measurement]) -> Vec<Summary> {
    let mut groups: BTreeMap<(&'static str, &'static str, &'static str, Backend), Vec<f64>> =
        BTreeMap::new();
    for measurement in measurements {
        groups
            .entry((
                measurement.architecture,
                measurement.fixture_id,
                measurement.precision,
                measurement.backend,
            ))
            .or_default()
            .push(measurement.pos_per_sec);
    }
    groups
        .into_iter()
        .map(|((architecture, fixture_id, precision, backend), runs)| {
            let stats = statistics(&runs);
            Summary {
                architecture,
                fixture_id,
                precision,
                backend,
                runs,
                mean_pos_per_sec: stats.mean,
                median_pos_per_sec: stats.median,
                sample_sd_pos_per_sec: stats.sample_sd,
                min_pos_per_sec: stats.min,
                max_pos_per_sec: stats.max,
            }
        })
        .collect()
}

fn build_backend_comparisons(
    measurements: &[Measurement],
    summaries: &[Summary],
) -> Vec<BackendComparison> {
    type RunKey = (&'static str, &'static str, usize);
    type BackendPair = (Option<f64>, Option<f64>);
    let mut groups: BTreeMap<RunKey, BackendPair> = BTreeMap::new();
    for measurement in measurements {
        let entry = groups
            .entry((
                measurement.architecture,
                measurement.precision,
                measurement.run,
            ))
            .or_default();
        match measurement.backend {
            Backend::CudaOxide => entry.0 = Some(measurement.pos_per_sec),
            Backend::NativeCuda => entry.1 = Some(measurement.pos_per_sec),
        }
    }

    let mut deltas: BTreeMap<(&'static str, &'static str), Vec<f64>> = BTreeMap::new();
    for ((architecture, precision, _), (oxide, native)) in groups {
        if let (Some(oxide), Some(native)) = (oxide, native) {
            deltas
                .entry((architecture, precision))
                .or_default()
                .push((native / oxide - 1.0) * 100.0);
        }
    }
    deltas
        .into_iter()
        .map(|((architecture, precision), paired_deltas_percent)| {
            let delta_stats = statistics(&paired_deltas_percent);
            let oxide = summary_mean(summaries, architecture, precision, Backend::CudaOxide)
                .expect("paired comparison has cuda-oxide summary");
            let native = summary_mean(summaries, architecture, precision, Backend::NativeCuda)
                .expect("paired comparison has native summary");
            BackendComparison {
                architecture,
                precision,
                paired_deltas_percent,
                mean_paired_delta_percent: delta_stats.mean,
                sample_sd_paired_delta_percent: delta_stats.sample_sd,
                native_over_oxide_ratio: native / oxide,
            }
        })
        .collect()
}

fn build_precision_speedups(summaries: &[Summary]) -> Vec<PrecisionSpeedup> {
    let mut result = Vec::new();
    for architecture in ["layerstack", "simple"] {
        for backend in [Backend::CudaOxide, Backend::NativeCuda] {
            if let (Some(fp32), Some(all_optim)) = (
                summary_mean(summaries, architecture, "fp32", backend),
                summary_mean(summaries, architecture, "all-optim", backend),
            ) {
                result.push(PrecisionSpeedup {
                    architecture,
                    backend,
                    all_optim_over_fp32_ratio: all_optim / fp32,
                });
            }
        }
    }
    result
}

fn summary_mean(
    summaries: &[Summary],
    architecture: &str,
    precision: &str,
    backend: Backend,
) -> Option<f64> {
    summaries
        .iter()
        .find(|summary| {
            summary.architecture == architecture
                && summary.precision == precision
                && summary.backend == backend
        })
        .map(|summary| summary.mean_pos_per_sec)
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
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
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

fn print_summaries(
    summaries: &[Summary],
    comparisons: &[BackendComparison],
    speedups: &[PrecisionSpeedup],
) {
    for summary in summaries {
        eprintln!(
            "[native-bench-summary] architecture={}, precision={}, backend={}, runs={}, mean={:.0}, median={:.0}, sample_sd={:.0}, min={:.0}, max={:.0} pos/s",
            summary.architecture,
            summary.precision,
            summary.backend.name(),
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
        );
    }
    for comparison in comparisons {
        eprintln!(
            "[native-bench-compare] architecture={}, precision={}, paired_delta={:.3}±{:.3}%, native_over_oxide={:.4}",
            comparison.architecture,
            comparison.precision,
            comparison.mean_paired_delta_percent,
            comparison.sample_sd_paired_delta_percent,
            comparison.native_over_oxide_ratio,
        );
    }
    for speedup in speedups {
        eprintln!(
            "[native-bench-precision] architecture={}, backend={}, all_optim_over_fp32={:.4}",
            speedup.architecture,
            speedup.backend.name(),
            speedup.all_optim_over_fp32_ratio,
        );
    }
}

fn capture_environment(device: usize) -> EnvironmentReport {
    let device_arg = format!("--id={device}");
    EnvironmentReport {
        platform: platform_name(),
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        gpu: command_output(
            "nvidia-smi",
            &[
                device_arg.as_str(),
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
        ),
        driver: command_output(
            "nvidia-smi",
            &[
                device_arg.as_str(),
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
        ),
        cuda_toolkit: nvcc_output(),
        rustc: command_output("rustc", &["--version"]),
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        dirty: git_dirty(),
        cargo_features: [
            cfg!(feature = "cuda-oxide").then_some("cuda-oxide"),
            cfg!(feature = "native-cuda").then_some("native-cuda"),
            cfg!(feature = "native-cuda-host").then_some("native-cuda-host"),
        ]
        .into_iter()
        .flatten()
        .collect(),
        command_line: std::env::args().collect(),
    }
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

fn nvcc_output() -> Option<String> {
    if let Some(path) = std::env::var_os("NVCC")
        && let Some(output) = command_output(path.to_string_lossy().as_ref(), &["--version"])
    {
        return Some(output);
    }
    let executable = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };
    for variable in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            let candidate = PathBuf::from(root).join("bin").join(executable);
            if let Some(output) =
                command_output(candidate.to_string_lossy().as_ref(), &["--version"])
            {
                return Some(output);
            }
        }
    }
    if let Some(output) = command_output(executable, &["--version"]) {
        return Some(output);
    }
    if cfg!(target_os = "linux") {
        let mut candidates = fs::read_dir("/usr/local")
            .ok()?
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("cuda-"))
            .map(|entry| entry.path().join("bin/nvcc"))
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        candidates.sort();
        for candidate in candidates.into_iter().rev() {
            if let Some(output) =
                command_output(candidate.to_string_lossy().as_ref(), &["--version"])
            {
                return Some(output);
            }
        }
    }
    None
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
    mode: NativeBenchModeArg,
    report: &BenchmarkReport,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let filename = format!(
        "{timestamp_unix_ms}-{}-{}-{PROFILE_NAME}.json",
        report.environment.platform,
        mode_name(mode),
    );
    let path = output_dir.join(filename);
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

    #[test]
    fn statistics_uses_sample_sd_and_even_median() {
        let stats = statistics(&[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert!((stats.sample_sd - 1.825_741_858_350_553_8).abs() < 1.0e-12);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn v1_precision_matrix_expands_all_optim_flags() {
        let precisions = selected_precisions(NativeBenchPrecisionArg::All);
        assert_eq!(precisions.len(), 2);
        assert_eq!(precisions[0].name, "fp32");
        assert!(!precisions[0].flags.tf32);
        assert_eq!(precisions[1].name, "all-optim");
        assert!(precisions[1].flags.tf32);
        assert!(precisions[1].flags.ft_fp16);
        assert!(precisions[1].flags.ft_fp16_out);
        assert!(precisions[1].flags.fp16_opt_state);
    }

    #[test]
    fn serialized_v1_report_conforms_to_versioned_json_schema() {
        let measurement = Measurement {
            architecture: "simple",
            fixture_id: Architecture::Simple.fixture_id(),
            precision: "fp32",
            precision_flags: PrecisionFlags::default().into(),
            backend: Backend::NativeCuda,
            run: 1,
            order_in_run: 1,
            positions: 16_384,
            elapsed_ns: 1_000_000,
            pos_per_sec: 16_384_000.0,
        };
        let report = BenchmarkReport {
            schema_version: SCHEMA_VERSION,
            timestamp_unix_ms: 1_784_400_000_000,
            profile: PROFILE_NAME,
            mode: "compare",
            parameters: BenchmarkParameters {
                batch_size: DEFAULT_BATCH_SIZE,
                warmup_steps: DEFAULT_WARMUP_STEPS,
                steps: DEFAULT_STEPS,
                runs: DEFAULT_RUNS,
                device: 0,
                customized: false,
            },
            environment: EnvironmentReport {
                platform: "wsl",
                os: "linux",
                architecture: "x86_64",
                gpu: Some("NVIDIA GeForce RTX 5090".into()),
                driver: Some("596.36".into()),
                cuda_toolkit: Some("CUDA 12.9".into()),
                rustc: Some("rustc test".into()),
                git_commit: Some("0123456789abcdef".into()),
                dirty: Some(false),
                cargo_features: vec!["native-cuda"],
                command_line: vec!["nnue-train".into(), "native-bench".into()],
            },
            measurements: vec![measurement],
            summaries: vec![Summary {
                architecture: "simple",
                fixture_id: Architecture::Simple.fixture_id(),
                precision: "fp32",
                backend: Backend::NativeCuda,
                runs: vec![16_384_000.0],
                mean_pos_per_sec: 16_384_000.0,
                median_pos_per_sec: 16_384_000.0,
                sample_sd_pos_per_sec: 0.0,
                min_pos_per_sec: 16_384_000.0,
                max_pos_per_sec: 16_384_000.0,
            }],
            backend_comparisons: vec![BackendComparison {
                architecture: "simple",
                precision: "fp32",
                paired_deltas_percent: vec![1.0],
                mean_paired_delta_percent: 1.0,
                sample_sd_paired_delta_percent: 0.0,
                native_over_oxide_ratio: 1.01,
            }],
            precision_speedups: vec![PrecisionSpeedup {
                architecture: "simple",
                backend: Backend::NativeCuda,
                all_optim_over_fp32_ratio: 1.25,
            }],
        };
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/schemas/native-cuda-benchmark-v1.schema.json"
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
