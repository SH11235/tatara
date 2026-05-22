use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use nnue_format::ArchKind;

use crate::arch::*;

// ===========================================================================
// CLI (clap) — 引数群は bullet-shogi `examples/shogi_layerstack.rs` に対応
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
    #[arg(long, global = true)]
    pub(crate) test_data: Option<PathBuf>,

    /// Number of positions per held-out validation pass. Takes this many
    /// positions from the start of the test PSV and rounds up to a whole
    /// `--batch-size` multiple to form full batches. Used only when
    /// `--test-data` is set.
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

    /// LR gamma (multiplies the LR by gamma every `lr_step` superbatches).
    #[arg(long, default_value_t = 0.995, global = true)]
    pub(crate) lr_gamma: f32,

    /// LR step (superbatch interval at which the LR is multiplied by gamma).
    #[arg(long, default_value_t = 1, global = true)]
    pub(crate) lr_step: usize,

    /// WDL blend lambda (constant).
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) wdl: f32,

    /// Score scale for the sigmoid loss (`loss_scale = 1 / scale`). Not used
    /// when `--win-rate-model` is set (WRM loss uses the `--wrm-*` scaling
    /// instead).
    #[arg(long, default_value_t = 290.0, global = true)]
    pub(crate) scale: f32,

    /// Write a checkpoint every `save_rate` superbatches (and at the end).
    #[arg(long, default_value_t = 20, global = true)]
    pub(crate) save_rate: usize,

    /// Exclude positions with `|score| >= score_drop_abs` from the loss (bullet `--score-drop-abs`).
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
    /// from the superbatch recorded in the checkpoint + 1.
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
    /// Optimizer name (only "ranger" is implemented).
    #[arg(long, default_value = "ranger", global = true)]
    pub(crate) optimizer: String,
    /// Weight decay coefficient for the Ranger optimizer (AdamW-style decoupled
    /// weight decay). The default 0.0 means no decay. A non-zero value slightly
    /// decays the weights of every weight group toward 0 on each step.
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) weight_decay: f32,
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
    /// FT weights are small on every path — initialization, the optimizer's
    /// MIN_W/MAX_W clamp (`|w| <= 1.98`), and the quantised checkpoint — so they
    /// fit comfortably within the FP16 finite range (`|x| <= 65504`) and the
    /// mirror conversion does not overflow to ±inf.
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

    /// Subcommand selecting the NNUE architecture to train (`layerstack` / `simple`).
    #[command(subcommand)]
    pub(crate) arch: ArchCommand,
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
    /// progress8kpabs 9-bucket LayerStack architecture (FT → L1 → L2 32; FT dimension set by --ft-out, L1 dimension by --l1).
    #[command(name = "layerstack")]
    LayerStack(LayerstackArgs),
    /// Simple 4-layer architecture, ported from bullet-shogi.
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

    /// Output dimension of the L1 (per-bucket dense) layer. Specify `>= 2`. The
    /// default value keeps the network bit-identical to the standard layout and
    /// resume-compatible with existing checkpoints, and runs the fastest
    /// dedicated matmul kernel. A non-default value switches to a generic matmul
    /// kernel (numerically equivalent, but slower).
    #[arg(long, default_value_t = DEFAULT_L1_OUT)]
    pub(crate) l1: usize,

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
