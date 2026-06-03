use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use nnue_format::ArchKind;

use crate::arch::*;

// ===========================================================================
// CLI (clap)
// ===========================================================================

/// Shogi NNUE trainer.
///
/// Pick the NNUE architecture to train with the `layerstack` / `simple`
/// subcommands. Shared options are global arguments and may be placed before or
/// after the subcommand. Passing `--data <PSV>` runs the training loop;
/// omitting it runs a GPU smoke test that exercises the forward/backward path.
#[derive(Parser, Debug)]
#[command(name = "nnue-train", about = "Shogi NNUE trainer")]
pub(crate) struct Cli {
    /// Training data PSV file (`PackedSfenValue` x N, 40 bytes each). When omitted, runs a GPU smoke test.
    #[arg(long, global = true)]
    pub(crate) data: Option<PathBuf>,

    /// PSV file for held-out validation. Pass positions that are never used for
    /// a gradient update, separate from the training `--data`. When set, a
    /// forward-only validation pass runs at the end of each superbatch and
    /// reports test_loss (mean held-out loss) and test_accuracy (agreement
    /// between the output sign and the game result) in the training log and
    /// experiment.json. Used for early detection of divergence and overfitting.
    /// Mutually exclusive with `--test-tail-positions`.
    #[arg(long, global = true, conflicts_with = "test_tail_positions")]
    pub(crate) test_data: Option<PathBuf>,

    /// Reserve the last N positions of `--data` as a same-file held-out
    /// validation set: training reads positions in `[0, file_end - N * 40)`
    /// and validation reads `[file_end - N * 40, file_end)`. Useful when the
    /// training PSV is too large to split into separate files. Mutually
    /// exclusive with `--test-data`. The first `--test-positions` of the
    /// reserved tail are evaluated at the end of each superbatch.
    #[arg(long, global = true, conflicts_with = "test_data")]
    pub(crate) test_tail_positions: Option<u64>,

    /// Number of positions per held-out validation pass. Takes this many
    /// positions from the start of the held-out source (either `--test-data`
    /// or the tail reserved by `--test-tail-positions`) and rounds up to a
    /// whole `--batch-size` multiple to form full batches. Used only when
    /// one of those flags is set.
    #[arg(long, default_value_t = 10000, global = true)]
    pub(crate) test_positions: usize,

    /// Output directory for checkpoints (writes `{net_id}-{superbatch}.bin`).
    #[arg(long, default_value = "checkpoints", global = true)]
    pub(crate) output: PathBuf,

    /// Network id (used in checkpoint file names).
    #[arg(long, default_value = "rshogi", global = true)]
    pub(crate) net_id: String,

    /// Input feature set. One of: halfkp, halfka-split, halfka-merged,
    /// halfka-hm-split, halfka-hm-merged. Determines the FT input dimension and
    /// the number of active features. The default halfka-hm-merged is
    /// king-symmetric merged HalfKA.
    #[arg(long, default_value = "halfka-hm-merged", global = true)]
    pub(crate) feature_set: String,

    /// `name` field in experiment.json (display name in the experiment-tracking
    /// UI). Defaults to net_id, or `{net_id} (resume @sb{start superbatch})`
    /// when `--resume` is used.
    #[arg(long, global = true)]
    pub(crate) experiment_name: Option<String>,

    /// Number of superbatches to train (runs 1..=superbatches). The default of
    /// 10 is for smoke testing; use a much larger value for real training.
    #[arg(long, default_value_t = 10, global = true)]
    pub(crate) superbatches: usize,

    /// Number of batches per superbatch.
    #[arg(long, default_value_t = 6104, global = true)]
    pub(crate) batches_per_superbatch: usize,

    /// Number of positions per batch. Affects both GPU throughput and training
    /// dynamics. The default of 16384 is for smoke testing.
    #[arg(long, default_value_t = 16384, global = true)]
    pub(crate) batch_size: usize,

    /// Initial learning rate.
    #[arg(long, default_value_t = 8.75e-4, global = true)]
    pub(crate) lr: f32,

    /// Learning-rate schedule shape. The schedule maps a superbatch index to a
    /// learning rate; --lr is the starting (or, for one-cycle, the peak) rate.
    ///
    /// - step (default): multiply --lr by --lr-gamma every --lr-step
    ///   superbatches. Bit-identical to the historical behaviour.
    /// - constant: hold --lr for the whole run (--lr-gamma / --lr-step ignored).
    /// - drop: hold --lr, then multiply by --lr-gamma once after --lr-step
    ///   superbatches.
    /// - linear / cosine / exponential: decay from --lr to --lr-final by
    ///   --lr-final-superbatch (defaults to --superbatches), then hold
    ///   --lr-final. exponential requires --lr-final > 0.
    /// - one-cycle: warm up from --lr/--lr-div-factor to the peak --lr over the
    ///   first --lr-warmup-pct of --superbatches, then cosine-anneal to
    ///   --lr/--lr-div-factor/--lr-final-div-factor.
    #[arg(long, value_enum, default_value_t = LrScheduleArg::Step, global = true)]
    pub(crate) lr_schedule: LrScheduleArg,

    /// LR gamma. For --lr-schedule step, multiplies the LR every --lr-step
    /// superbatches; for drop, the one-shot multiplier applied after --lr-step
    /// superbatches. Ignored by the other schedules.
    #[arg(long, default_value_t = 0.992, global = true)]
    pub(crate) lr_gamma: f32,

    /// LR step. For --lr-schedule step, the superbatch interval at which the LR
    /// is multiplied by --lr-gamma; for drop, the superbatch after which the LR
    /// drops once. Ignored by the other schedules.
    #[arg(long, default_value_t = 1, global = true)]
    pub(crate) lr_step: usize,

    /// Final learning rate for the linear / cosine / exponential decay
    /// schedules. The LR decays from --lr to this value by
    /// --lr-final-superbatch and then holds. exponential requires a value > 0
    /// (it interpolates the LR geometrically). Ignored by the other schedules.
    #[arg(long, default_value_t = 1e-5, global = true)]
    pub(crate) lr_final: f32,

    /// Superbatch by which the linear / cosine / exponential decay reaches
    /// --lr-final. When omitted, the horizon is taken from (in priority order):
    /// the saved horizon in a v5+ --resume checkpoint, else --superbatches.
    /// Passing this flag explicitly always wins, even over a resumed
    /// checkpoint's saved horizon. one-cycle uses the same precedence for its
    /// total horizon but has no explicit flag, so on resume its saved horizon
    /// wins over --superbatches. A checkpoint written before v5 (or by a
    /// schedule without a horizon) carries none, so resume falls back to
    /// --superbatches. Ignored by the other schedules.
    #[arg(long, global = true)]
    pub(crate) lr_final_superbatch: Option<usize>,

    /// Warm up the learning rate over the first N batches of the first
    /// superbatch, ramping from a small fraction of the scheduled LR up to it,
    /// on top of any --lr-schedule. Applies to every schedule except one-cycle
    /// (which carries its own warmup). When omitted, no batch-level warmup.
    #[arg(long, global = true)]
    pub(crate) lr_warmup_steps: Option<usize>,

    /// one-cycle only: fraction of --superbatches spent warming up from the
    /// initial LR to the peak --lr before annealing. Must be in [0.0, 1.0].
    #[arg(long, default_value_t = 0.2, global = true)]
    pub(crate) lr_warmup_pct: f32,

    /// one-cycle only: the initial LR is --lr divided by this factor (the peak
    /// is --lr). Must be >= 1 so the initial LR does not exceed the peak.
    #[arg(long, default_value_t = 25.0, global = true)]
    pub(crate) lr_div_factor: f32,

    /// one-cycle only: the final LR is the initial LR (--lr / --lr-div-factor)
    /// divided by this factor. Must be > 0.
    #[arg(long, default_value_t = 1e4, global = true)]
    pub(crate) lr_final_div_factor: f32,

    /// WDL blend lambda (constant). Mutually exclusive with the linear-taper
    /// pair `--start-wdl` / `--end-wdl`.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) wdl: f32,

    /// Start of a linear WDL lambda taper, used at the first superbatch. Requires
    /// `--end-wdl`; the lambda interpolates linearly from `--start-wdl` to
    /// `--end-wdl` across superbatches. Conflicts with `--wdl`.
    #[arg(long, global = true, conflicts_with = "wdl")]
    pub(crate) start_wdl: Option<f32>,

    /// End of a linear WDL lambda taper, reached at the final superbatch. Requires
    /// `--start-wdl`. Conflicts with `--wdl`.
    #[arg(long, global = true, conflicts_with = "wdl")]
    pub(crate) end_wdl: Option<f32>,

    /// Score scale for the sigmoid loss (`loss_scale = 1 / scale`). Not used
    /// when `--win-rate-model` is set (WRM loss uses the `--wrm-*` scaling
    /// instead).
    #[arg(long, default_value_t = 290.0, global = true)]
    pub(crate) scale: f32,

    /// Write a checkpoint every `save_rate` superbatches (and at the end).
    #[arg(long, default_value_t = 20, global = true)]
    pub(crate) save_rate: usize,

    /// Exclude positions with `|score| >= score_drop_abs` from the loss.
    #[arg(long, global = true)]
    pub(crate) score_drop_abs: Option<i32>,

    /// Inject weights from a quantised NNUE binary before training starts
    /// (pretrained start). The optimizer state (Ranger m/v/slow/step) is
    /// **reset** — use `--resume` for a true resume (`--init-from` and
    /// `--resume` are mutually exclusive).
    #[arg(long, global = true)]
    pub(crate) init_from: Option<PathBuf>,

    /// Resume training by restoring weights + Ranger optimizer state
    /// (m/v/slow/step) from a raw checkpoint (`{net_id}-{sb}.ckpt`) — a true
    /// resume. Mutually exclusive with `--init-from` (which injects weights only
    /// and resets the optimizer). When `--start-superbatch` is omitted, resumes
    /// from the superbatch recorded in the checkpoint + 1. A v5+ checkpoint also
    /// stores the LR-schedule horizon; on resume the saved horizon is restored
    /// so the LR curve is reproduced independently of --superbatches (see
    /// --lr-final-superbatch for the full precedence).
    #[arg(long, global = true)]
    pub(crate) resume: Option<PathBuf>,

    /// Superbatch number to start training from (1-indexed, inclusive). When
    /// omitted: with `--resume`, the checkpoint's superbatch + 1; otherwise 1.
    /// An error if outside `1 <= N <= --superbatches` (may be set explicitly to
    /// redo past superbatches on resume).
    #[arg(long, global = true)]
    pub(crate) start_superbatch: Option<usize>,

    /// Keep only the most recent N raw checkpoints (`*.ckpt`) to save disk
    /// space. When omitted, all are kept. Raw state is large (~1.8GB each) and
    /// piles up over long runs, so setting this is recommended. Quantised `.bin`
    /// files (~116MB) are always kept regardless of this setting (inference
    /// artifacts).
    #[arg(long, global = true)]
    pub(crate) keep_checkpoints: Option<usize>,

    /// Use the win-rate-model loss. When set, uses the `loss_wrm` kernel
    /// (applies WRM to both prediction and target); otherwise uses `loss_wdl`
    /// (plain sigmoid-MSE + `--scale`). The net_output scale becomes
    /// `out ≈ cp / --wrm-nnue2score`, matching the scale that quantisation
    /// (`QA=127/QB=64/FV_SCALE=28`) assumes.
    #[arg(long, global = true)]
    pub(crate) win_rate_model: bool,
    /// In-scaling for the WRM prediction side (default 340). Independent of the
    /// target-side scaling (`--wrm-target-scaling`). Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 340.0, global = true)]
    pub(crate) wrm_in_scaling: f32,
    /// Center offset of the WRM prediction win-rate sigmoid, subtracted from the
    /// scaled net score inside `sigmoid((net*nnue2score - offset)/in_scaling)`
    /// (default 270). Independent of the target-side offset
    /// (`--wrm-target-offset`). Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 270.0, global = true)]
    pub(crate) wrm_in_offset: f32,
    /// WRM nnue2score (`scorenet = net_output * --wrm-nnue2score`, default 600).
    /// Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 600.0, global = true)]
    pub(crate) wrm_nnue2score: f32,
    /// Center offset of the WRM target sigmoid (the score at which `target` is
    /// 0.5, default 270). Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 270.0, global = true)]
    pub(crate) wrm_target_offset: f32,
    /// Input scale of the WRM target sigmoid (inverse of steepness, default
    /// 380). The defaults 270/380 are tuned for the chess score distribution;
    /// retune them if your score distribution differs. Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 380.0, global = true)]
    pub(crate) wrm_target_scaling: f32,
    /// Exponent of the WRM error term `|qf - target|^pow_exp` (default 2.0, plain
    /// squared error). nnue-pytorch uses 2.5. Must be >= 1 (the gradient contains
    /// `|err|^(pow_exp-1)`). Used only when `--win-rate-model` is set; the default
    /// 2.0 keeps the loss kernel on its bit-identical squared-error path.
    #[arg(long, default_value_t = 2.0, global = true)]
    pub(crate) loss_pow_exp: f32,
    /// Asymmetric penalty for overprediction (default 0.0 = symmetric; must be
    /// non-negative). When set, positions where the prediction `qf` exceeds the
    /// target are weighted by `1 + qp_asymmetry`. Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) loss_qp_asymmetry: f32,
    /// Weight-boost parameter w1 (default 0.0, must be >= 0). Per-position loss
    /// weight is `1 + (2^w1 - 1) * ((pf-0.5)^2 * pf*(1-pf))^w2`, amplifying
    /// decisive positions; `w1 = 0` gives uniform weight 1 (no boost). The total
    /// loss is then normalised by the sum of weights. Used only when
    /// `--win-rate-model` is set.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) loss_weight_boost_w1: f32,
    /// Weight-boost parameter w2 (default 0.5, the exponent in the weight
    /// formula, must be >= 0). Has no effect when `--loss-weight-boost-w1` is 0.
    /// Used only when `--win-rate-model` is set.
    #[arg(long, default_value_t = 0.5, global = true)]
    pub(crate) loss_weight_boost_w2: f32,
    /// Optimizer name (only "ranger" is implemented).
    #[arg(long, default_value = "ranger", global = true)]
    pub(crate) optimizer: String,
    /// Weight decay coefficient for the Ranger optimizer (AdamW-style decoupled
    /// weight decay). The default 0.0 means no decay. A non-zero value slightly
    /// decays the weights of every weight group toward 0 on each step.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) weight_decay: f32,
    /// Enable norm loss (per-weight-group L2-norm regularisation, Georgiou et
    /// al. 2021). With the default `false`, the optimizer step is bit-identical
    /// to the baseline. When enabled, each step (just before the Ranger update)
    /// every targeted weight group is nudged so its L2 norm relaxes toward 1
    /// (the oblique manifold): the 2D layer weights per output neuron (FT /
    /// L1f / L1 / L2 / L3), the PSQT shortcut weights per output bucket (when
    /// --psqt is enabled), and 1D biases by their whole-tensor norm. Strength
    /// is set by --norm-loss-factor. An opt-in regularizer whose playing-strength
    /// effect must be confirmed by SPRT.
    #[arg(long, global = true)]
    pub(crate) norm_loss: bool,
    /// Norm loss strength (only used when --norm-loss is set). Each step every
    /// targeted weight w is multiplied by
    /// `1 - lr * 2 * factor * (1 - 1 / (||w_group||_2 + eps))`. The default 1e-4
    /// matches the Ranger21 reference.
    #[arg(long, default_value_t = 1e-4, global = true)]
    pub(crate) norm_loss_factor: f32,
    /// Number of dataloader prefetch workers. Each worker does PSV parsing +
    /// HalfKA_hm sparse extraction + progress8kpabs bucket computation in a
    /// single `decode()` call and supplies positions ahead of time. `1` gives
    /// deterministic sequential reads; `>= 2` parses in parallel (position order
    /// within an epoch is non-deterministic, which is fine for training).
    #[arg(long, default_value_t = 16, global = true)]
    pub(crate) threads: usize,

    /// Fast mode that runs the FT weight (`ft_w`) forward pass through an FP16
    /// mirror. With the default `false`, it is bit-identical to the FP32 path.
    /// `true` halves the weight DRAM bandwidth of `sparse_ft_forward`, but
    /// quantisation error may shift playing strength (an opt-in option for
    /// quick, fast training; default OFF until production quality is confirmed
    /// by SPRT).
    ///
    /// FT weights stay small in practice — small-scale initialization plus
    /// AdamW weight decay keep the master FP32 magnitude orders of magnitude
    /// below the FP16 finite range (`|x| <= 65504`), so the mirror conversion is
    /// not expected to overflow to ±inf under normal training. There is no hard
    /// clamp on FT, so `--ft-fp16` remains an opt-in precision trade-off.
    #[arg(long, global = true)]
    pub(crate) ft_fp16: bool,

    /// Fast mode that keeps the feature transformer (FT) optimizer state in
    /// FP16. With the default `false`, it is bit-identical to the FP32 path.
    ///
    /// The FT is the largest layer in this network, and its optimizer update is
    /// memory-bandwidth bound on the state read/write. Halving the precision of
    /// the state reduces the memory traffic of the optimizer step and raises
    /// training throughput. The state values are extremely small, so they are
    /// multiplied by a fixed factor to bring them into the FP16 usable range
    /// before being stored.
    ///
    /// An independent flag from `--ft-fp16` / `--ft-fp16-out`. Quantisation
    /// error may shift playing strength, so it is default OFF and production
    /// quality is not guaranteed until confirmed by SPRT (an opt-in option for
    /// sanity checks and quick, fast training).
    #[arg(long, global = true)]
    pub(crate) fp16_opt_state: bool,

    /// Shortcut to opt into all four risky speed flags at once. Effectively
    /// turns on `--ft-fp16` / `--fp16-opt-state` / (on the subcommand)
    /// `--ft-fp16-out` / `--tf32` together (OR-combined with the individual
    /// flags; works with both subcommands).
    ///
    /// Default OFF (all flags OFF gives a pure FP32, bit-identical path). When
    /// set, the effective values are expanded in the startup log
    /// (`[train] --all-optim → ft_fp16=true ft_fp16_out=true fp16_opt_state=true
    /// tf32=true`) to keep experiment.json reproducible. Default OFF because
    /// quantisation / TF32 error may shift playing strength.
    ///
    /// For fine-grained control (turning on only some), do not use this flag;
    /// list the four individual flags instead.
    #[arg(long, global = true)]
    pub(crate) all_optim: bool,

    /// Log the number of FP16 clamp events (`|x| > 65504` cap to `±65504`) in the
    /// FT activation backward kernels (the `--ft-fp16-out` write path) at the end
    /// of every superbatch. Used to gauge how often the loss-scaled gradient
    /// saturates the FP16 finite range — a high rate suggests the loss scale
    /// should be retuned, as systematic clamping caps gradient magnitudes and
    /// can shift playing strength.
    ///
    /// The clamp counter is always active in the FP16 path; this flag only gates
    /// the host-side D2H read and log line (`[fp16-clamp] sb=... clamps=...
    /// delta=... elems=... ratio=...`). With `--ft-fp16-out` off, the counter
    /// stays at zero because the clamp kernels are not launched.
    #[arg(long, global = true)]
    pub(crate) monitor_fp16_clamps: bool,

    /// Override the feature-transformer (L0) weight initialiser. Applies to a
    /// fresh run; ignored when `--init-from` / `--resume` loads weights. The
    /// default weight init is `[-0.01, 0.01]` uniform.
    ///
    /// Grammar: `zero`, `<uniform|normal>:abs:<value>`, or
    /// `<uniform|normal>:fanin[:<gain>[:<effective>]]` where the magnitude is
    /// `sqrt(gain / effective_or_fan_in)` (half-width for uniform, std for
    /// normal). Examples: `uniform:fanin`, `normal:fanin:2:32`
    /// (`sqrt(2/32) = 0.25`), `uniform:abs:0.01` (the default). Applies to the
    /// weight only; the bias keeps the default.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_ft: Option<nnue_train::init::LayerInitOverride>,

    /// Override the L1 weight initialiser. Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l1: Option<nnue_train::init::LayerInitOverride>,

    /// Override the shared factorised L1f weight initialiser (layerstack only).
    /// Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l1f: Option<nnue_train::init::LayerInitOverride>,

    /// Override the L2 weight initialiser. Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l2: Option<nnue_train::init::LayerInitOverride>,

    /// Override the L3 (output) weight initialiser. Same grammar as `--init-ft`.
    #[arg(long, global = true, value_name = "SPEC", value_parser = nnue_train::init::parse_layer_init_spec)]
    pub(crate) init_l3: Option<nnue_train::init::LayerInitOverride>,

    /// Subcommand selecting the NNUE architecture to train (`layerstack` / `simple`).
    #[command(subcommand)]
    pub(crate) arch: ArchCommand,
}

/// `--lr-schedule` の選択肢。lib 側 schedule 型への runtime selection。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum LrScheduleArg {
    #[default]
    Step,
    Constant,
    Drop,
    Linear,
    Cosine,
    Exponential,
    #[value(name = "one-cycle")]
    OneCycle,
}

/// `--ft-fp16-out` が `--ft-fp16` を要求する制約を **実効値** (`--all-optim` の含意込み)
/// で検証する。`true` を返したら制約違反 = error (FT activation FP16 が ON だが
/// FT weight FP16 が OFF)。
///
/// `--all-optim` は `--ft-fp16` / `--ft-fp16-out` の双方を ON 相当にするため、`--all-optim`
/// が指定されていれば制約は常に満たされる (両 flag が実効 ON)。よって制約違反は
/// 「`--ft-fp16-out` が raw 指定されていて、`--all-optim` も無く、`--ft-fp16` も raw 指定
/// されていない」ときのみ。これにより `--all-optim --ft-fp16-out` (冗長指定) を
/// false-positive reject しない。
pub(crate) fn ft_fp16_out_missing_ft_fp16(
    ft_fp16_out_raw: bool,
    ft_fp16_raw: bool,
    all_optim: bool,
) -> bool {
    ft_fp16_out_raw && !all_optim && !ft_fp16_raw
}

/// 学習対象の NNUE アーキを選ぶサブコマンド。アーキ固有の引数を持つ。
#[derive(Subcommand, Debug)]
pub(crate) enum ArchCommand {
    /// progress-kpabs N-bucket LayerStack architecture (FT → L1 → L2; layer dimensions set by --ft-out / --l1 / --l2; bucket count by --num-buckets).
    #[command(name = "layerstack")]
    LayerStack(LayerstackArgs),
    /// Simple 4-layer dense architecture (no buckets / PSQT / skip).
    Simple(SimpleArgs),
}

impl ArchCommand {
    /// サブコマンドに対応する [`ArchKind`]。
    pub(crate) fn kind(&self) -> ArchKind {
        match self {
            ArchCommand::LayerStack(_) => ArchKind::LayerStack,
            ArchCommand::Simple(_) => ArchKind::Simple,
        }
    }
}

/// LayerStack アーキ固有の引数。
#[derive(Args, Debug)]
pub(crate) struct LayerstackArgs {
    /// progress8kpabs coefficient file (`progress.bin`; f64 LE x 125388 = 81
    /// king squares x 1548 KP-abs piece inputs). When omitted, every position
    /// falls in bucket 4 (zero weights → `sigmoid(0) = 0.5`).
    #[arg(long)]
    pub(crate) progress_coeff: Option<PathBuf>,

    /// Bucket mode (only "progress8kpabs" is implemented).
    #[arg(long, default_value = "progress8kpabs")]
    pub(crate) bucket_mode: String,

    /// Output dimension of the FT (feature transformer) per perspective. Must be
    /// a positive multiple of 128. The default value keeps the network
    /// bit-identical to the standard layout and resume-compatible with existing
    /// checkpoints.
    #[arg(long, default_value_t = DEFAULT_FT_OUT)]
    pub(crate) ft_out: usize,

    /// Output dimension of the L1 (per-bucket dense) layer. Specify a value in
    /// [2, 256]. The default keeps the network bit-identical to the standard
    /// layout and resume-compatible with existing checkpoints. Every value runs
    /// on the same per-bucket tiled matmul kernels — the output dimension is
    /// processed in 16-wide tiles — so non-default widths are not penalized.
    #[arg(long, default_value_t = DEFAULT_L1_OUT)]
    pub(crate) l1: usize,

    /// Output dimension of the L2 (per-bucket dense) layer. Specify a value in
    /// [2, 256]; the upper bound is the fixed shared-memory accumulator capacity
    /// of the per-bucket bias-gradient kernel. The default keeps the network
    /// bit-identical to the standard layout and resume-compatible with existing
    /// checkpoints. The L2 / L3 kernels take the output dimension as a runtime
    /// argument, so non-default widths are not penalized.
    #[arg(long, default_value_t = DEFAULT_L2_OUT)]
    pub(crate) l2: usize,

    /// LayerStack output bucket count. Each position is routed to bucket
    /// `min(N-1, floor(p * N))` where `p` is the progress estimate. Specify a
    /// value in `[2, 9]`; the upper bound is the fixed 9-register accumulator
    /// in the per-bucket weight backward kernels. The default 9 keeps the
    /// binning and weight-buffer shape identical to the standard layout and
    /// resume-compatible with existing checkpoints. The historical 8-bucket
    /// progress emission used `floor(p * 8)` on a 9-slot layout, leaving slot 8
    /// unused; the unified design here means setting `--num-buckets 9` (the
    /// default) actually emits index 8 — existing 9-bucket distributed nets
    /// have an untrained slot 8 and may see a short-term eval shift on the
    /// `p in [8/9, 1]` tail until continued training catches up.
    #[arg(long, default_value_t = DEFAULT_NUM_BUCKETS)]
    pub(crate) num_buckets: usize,

    /// Opt-in flag to use Ampere+ Tensor Cores in TF32 mode. `true` calls cuBLAS
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`, rounding the
    /// FP32 Sgemm inputs to 10-bit-mantissa TF32 and running TC mma → FP32
    /// accumulate (~3 significant decimal digits of mantissa; exponent range
    /// same as FP32). With the default `false`, it runs `CUBLAS_DEFAULT_MATH` (a
    /// pure FP32 path, no Tensor Cores).
    ///
    /// Dropping 13 mantissa bits affects the numerics of the `fwd_L1f` /
    /// `bwd_L1f` Sgemm, so it is conservatively default OFF for quality.
    #[arg(long)]
    pub(crate) tf32: bool,

    /// Also keep the FT activation (the `ft_*_out` forward output and the
    /// `dft_*_out` backward gradient) in FP16. Requires `--ft-fp16` (an
    /// extension stacked on top of the weight FP16 path).
    ///
    /// `ft_*_out` is the output of `sparse_ft_forward`; making it FP16 halves
    /// the bandwidth of the subsequent read + inverse-index gather (`phD`, the
    /// heaviest DRAM read in a step). dft is a tiny value proportional to
    /// `1/batch` from batch normalization, so when made FP16 it is lifted into
    /// the normal range by loss scaling (a factor proportional to batch) before
    /// being stored.
    ///
    /// Split into a separate flag from weight FP16 (`--ft-fp16`) so that SPRT
    /// can isolate the strength impact in two steps:
    /// FP32 → `--ft-fp16` → `--ft-fp16 --ft-fp16-out`. Quantisation error may
    /// shift playing strength, so it is default OFF and production quality is
    /// not guaranteed until confirmed by SPRT.
    #[arg(long)]
    pub(crate) ft_fp16_out: bool,

    /// Enable the PSQT (Piece-Square Table) shortcut layer.
    ///
    /// PSQT is a per-feature × per-bucket scalar prior layered in parallel with
    /// the dense `FT -> L1 -> L2 -> L3` path: `net_output +=
    /// 0.5 * (Σ stm_active psqt_w[f, bucket] - Σ nstm_active psqt_w[f, bucket])`.
    /// Stockfish SFNNv10 style — the dense path then only has to learn
    /// non-material structure on top of the material prior carried by PSQT.
    ///
    /// Default OFF for bit-identical compatibility with non-PSQT checkpoints.
    /// When enabled the saved `.bin` carries an extra `PSQT=9,` token in the
    /// arch string plus an i32 PSQT block (scale `QA * QB = 8128`).
    #[arg(long)]
    pub(crate) psqt: bool,

    /// PSQT shortcut weight initialiser: `zeroed` (default) or `material`.
    ///
    /// - `zeroed`: every PSQT weight starts at 0; the dense path absorbs
    ///   material information first and PSQT only picks up the residual
    ///   correction. Known to leave a long plateau early in training.
    /// - `material`: PSQT weights are pre-loaded with centipawn piece values
    ///   divided by `--wrm-nnue2score` (or `--scale` when WRM is off) so the
    ///   shortcut already encodes piece material from step 0. The dense path
    ///   then specialises in non-material structure (positional/tactical
    ///   patterns).
    ///
    /// Requires `--psqt`. Material init additionally requires the loss to know
    /// the centipawn → logit scaling: either use `--win-rate-model` with
    /// `--wrm-in-scaling` (and `--wrm-nnue2score`) set, or use the sigmoid path
    /// where `--scale` provides the conversion factor.
    #[arg(long, value_enum, default_value_t = PsqtInit::Zeroed, requires = "psqt")]
    pub(crate) psqt_init: PsqtInit,
}

/// PSQT shortcut の初期化方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PsqtInit {
    /// Start every PSQT weight at zero (PSQT is initially inert).
    Zeroed,
    /// Pre-load PSQT with centipawn piece values / out_scaling (Material prior).
    Material,
}

/// Simple 4 層アーキ固有の引数。
#[derive(Args, Debug)]
pub(crate) struct SimpleArgs {
    /// Layer-dimension preset (`<l1>x2-<l2>-<l3>`). l1 is the accumulator (FT
    /// output) dimension; l2 / l3 are the hidden-layer dimensions. Each can be
    /// overridden individually with `--l1` / `--l2` / `--l3`.
    #[arg(long, default_value = "256x2-32-32")]
    pub(crate) arch: String,

    /// Accumulator (FT output) dimension. Defaults to the `--arch` preset value.
    #[arg(long)]
    pub(crate) l1: Option<usize>,

    /// Dimension of hidden layer 1. Defaults to the `--arch` preset value.
    #[arg(long)]
    pub(crate) l2: Option<usize>,

    /// Dimension of hidden layer 2. Defaults to the `--arch` preset value.
    #[arg(long)]
    pub(crate) l3: Option<usize>,

    /// FT post-activation function ("crelu" / "screlu" / "pairwise").
    /// "pairwise" multiplies the corresponding indices of the first and second
    /// halves, halving the L1 input dimension (the L1 / L2 dense layers use
    /// CReLU activation).
    #[arg(long, default_value = "crelu")]
    pub(crate) activation: String,

    /// Also keep the FT activation (the `ft_*_out` forward output and the
    /// `dft_*_out` backward gradient) in FP16. Requires the global `--ft-fp16`
    /// (supports crelu / screlu / pairwise).
    ///
    /// `ft_*_out` is the output of `sparse_ft_forward`; making it FP16 halves
    /// the bandwidth of the subsequent read + the `sparse_ft_backward` read. dft
    /// is a tiny value proportional to `1/batch` from batch normalization, so
    /// when made FP16 it is lifted into the normal range by loss scaling
    /// (proportional to batch) before being stored.
    ///
    /// An opt-in option: quantisation error may shift playing strength, so it is
    /// default OFF and production quality is not guaranteed until confirmed by
    /// SPRT.
    #[arg(long)]
    pub(crate) ft_fp16_out: bool,

    /// Opt-in flag to use Ampere+ Tensor Cores in TF32 mode. `true` calls cuBLAS
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`, rounding the
    /// FP32 inputs of the L1/L2/L3 dense Sgemm to 10-bit-mantissa TF32 and
    /// running TC mma → FP32 accumulate (~3 significant decimal digits of
    /// mantissa; exponent range same as FP32). With the default `false`, it runs
    /// `CUBLAS_DEFAULT_MATH` (a pure FP32 path, no Tensor Cores).
    ///
    /// Dropping 13 mantissa bits affects the numerics of the dense Sgemm, so it
    /// is conservatively default OFF for quality. Same policy as LayerStack
    /// `--tf32` (an opt-in flag with a playing-strength risk).
    #[arg(long)]
    pub(crate) tf32: bool,
}
