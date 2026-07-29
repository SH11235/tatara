//! `progress-kpabs-train` binary entry point。
//!
//! host loop は PSV file 群 → batch → forward → grad → adam_step を駆動し、最終
//! weight を `progress.bin` (f64 LE × N) に出力する。
//!
//! ## 設計
//!
//! - **kernels** (forward / grad / adam_step / eval) は CUDA C++ で実装し、
//!   NNUE 学習用 kernel と独立した fatbin にまとめる。
//! - **GpuTrainer** は本 file。device buffer (weights / m / v / grad / loss_acc /
//!   hist + scratch) を所有し、`step` / `eval_forward` で 1 batch 分の
//!   forward → grad/eval → (training なら) adam_step を launch する。
//! - **host helper** (Batch builder / PSV reader / progress.bin I/O / CLI) は
//!   GPU 非依存なので `lib.rs` の `host` module に置く。host helper は
//!   `cargo test -p progress-kpabs-train` で GPU なしで単体テストできる。
//!
//! ## 使い方
//!
//! ```bash
//! # 1 epoch の動作確認 (smoke):
//! cargo run -p progress-kpabs-train -- \
//!     --data crates/shogi-format/tests/data/sample.psv \
//!     --output /tmp/progress.bin \
//!     --games-per-step 4 --max-games 8 --lr 1e-3
//!
//! # 実データで (--val-fraction で held-out 検証 loss も出力):
//! cargo run --release -p progress-kpabs-train -- \
//!     --data <path/to/training.bin> \
//!     --output progress.bin --epochs 1 --val-fraction 0.05
//! ```
//!
use std::process::ExitCode;
use std::time::Instant;
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    ptr,
};

use clap::Parser;
use cuda_native_runtime::{Context, DeviceBuffer, Event, Function, PROGRESS_KERNEL_FATBIN, Stream};
use progress_kpabs_train::host::{
    ADAM_BETA1, ADAM_BETA2, ADAM_EPS, MAX_INDS_PER_POS,
    batch::Batch,
    cli::Args,
    games::{GameIterator, PackCursor},
    progress_bin::{read_progress_bin, write_progress_bin},
};
use shogi_features::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;

const INITIAL_POSITIONS_PER_GAME: usize = 256;
const BLOCK_DIM: u32 = 256;

fn arg<T>(value: &mut T) -> *mut c_void {
    ptr::from_mut(value).cast()
}

fn grid_dim_1d(n: usize) -> (u32, u32, u32) {
    ((n as u32).div_ceil(BLOCK_DIM), 1, 1)
}

// ---------------------------------------------------------------------------
// Host driver (GpuTrainer + main)
// ---------------------------------------------------------------------------

/// GPU 上で 4 kernel を順次起動する trainer。
///
/// device buffer は内部所有:
/// - `weights / m / v / grad`: `DeviceBuffer<f32>` (size = `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`)
/// - `loss_acc`: `DeviceBuffer<f64>` (size = 1)
/// - `hist`: `DeviceBuffer<u64>` (size = 8)
///
/// 入力 (`indices` / `targets` / `per_pos_norm`) と `preds` は起動時に確保し、
/// 各 batch では同一 stream 上の H2D と kernel launch に再利用する。
struct GpuTrainer {
    context: Context,
    stream: Stream,
    forward: Function,
    grad_kernel: Function,
    eval: Function,
    adam_step: Function,

    weights: DeviceBuffer<f32>,
    m: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    grad: DeviceBuffer<f32>,
    loss_acc: DeviceBuffer<f64>,
    hist: DeviceBuffer<u64>,
    indices: DeviceBuffer<i32>,
    targets: DeviceBuffer<f32>,
    per_pos_norm: DeviceBuffer<f32>,
    preds: DeviceBuffer<f32>,
    input_capacity: usize,
    input_upload_done: Event,

    /// Adam の `beta^t` 累積値 (`bc1 = 1 - beta1_pow` を kernel に渡す)。
    beta1_pow: f32,
    beta2_pow: f32,
}

impl GpuTrainer {
    /// CUDA context を作成し、kernel module を load し、device buffer を確保する。
    fn new(
        ctx: &Context,
        init_weights: Option<&[f32]>,
        games_per_step: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.create_stream()?;
        let module = ctx.load_module(PROGRESS_KERNEL_FATBIN)?;
        let forward = module.function(c"progress_forward")?;
        let grad_kernel = module.function(c"progress_grad")?;
        let eval = module.function(c"progress_eval")?;
        let adam_step = module.function(c"progress_adam_step")?;

        let n = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
        let weights = match init_weights {
            Some(init) => {
                if init.len() != n {
                    return Err(
                        format!("init_weights length {} != expected {}", init.len(), n).into(),
                    );
                }
                DeviceBuffer::from_slice(ctx, init)?
            }
            None => DeviceBuffer::<f32>::zeroed(ctx, n)?,
        };
        let m = DeviceBuffer::<f32>::zeroed(ctx, n)?;
        let v = DeviceBuffer::<f32>::zeroed(ctx, n)?;
        let grad = DeviceBuffer::<f32>::zeroed(ctx, n)?;
        let loss_acc = DeviceBuffer::<f64>::zeroed(ctx, 1)?;
        let hist = DeviceBuffer::<u64>::zeroed(ctx, 8)?;
        let input_capacity = games_per_step
            .checked_mul(INITIAL_POSITIONS_PER_GAME)
            .ok_or("input buffer capacity overflow")?
            .max(1);
        let indices_len = input_capacity
            .checked_mul(MAX_INDS_PER_POS)
            .ok_or("index buffer capacity overflow")?;
        let indices = DeviceBuffer::<i32>::zeroed(ctx, indices_len)?;
        let targets = DeviceBuffer::<f32>::zeroed(ctx, input_capacity)?;
        let per_pos_norm = DeviceBuffer::<f32>::zeroed(ctx, input_capacity)?;
        let preds = DeviceBuffer::<f32>::zeroed(ctx, input_capacity)?;
        let input_upload_done = ctx.create_event()?;

        Ok(Self {
            context: ctx.clone(),
            stream,
            forward,
            grad_kernel,
            eval,
            adam_step,
            weights,
            m,
            v,
            grad,
            loss_acc,
            hist,
            indices,
            targets,
            per_pos_norm,
            preds,
            input_capacity,
            input_upload_done,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        })
    }

    /// `loss_acc` / `hist` を 0 に reset する (epoch 開始時 / log 区間切り替え時)。
    fn zero_loss_hist(&mut self) -> cuda_native_runtime::Result<()> {
        self.loss_acc.zero_async(&self.stream)?;
        self.hist.zero_async(&self.stream)
    }

    fn ensure_input_capacity(&mut self, n_pos: usize) -> Result<(), Box<dyn std::error::Error>> {
        if n_pos <= self.input_capacity {
            return Ok(());
        }
        self.stream.synchronize()?;
        let capacity = n_pos
            .checked_next_power_of_two()
            .ok_or("input buffer capacity overflow")?;
        let indices_len = capacity
            .checked_mul(MAX_INDS_PER_POS)
            .ok_or("index buffer capacity overflow")?;
        self.indices = DeviceBuffer::<i32>::zeroed(&self.context, indices_len)?;
        self.targets = DeviceBuffer::<f32>::zeroed(&self.context, capacity)?;
        self.per_pos_norm = DeviceBuffer::<f32>::zeroed(&self.context, capacity)?;
        self.preds = DeviceBuffer::<f32>::zeroed(&self.context, capacity)?;
        self.input_capacity = capacity;
        Ok(())
    }

    /// 1 step (= 1 batch 分の forward → grad/loss/hist accumulate → adam_step) を実行する。
    fn step(&mut self, batch: &Batch, lr: f32) -> Result<(), Box<dyn std::error::Error>> {
        let n_pos = batch.n_positions;
        if n_pos == 0 {
            return Ok(());
        }

        self.ensure_input_capacity(n_pos)?;
        // SAFETY: batch storage remains live and immutable until input_upload_done is synchronized.
        unsafe {
            self.indices.copy_from_async(&batch.indices, &self.stream)?;
            self.targets.copy_from_async(&batch.targets, &self.stream)?;
            self.per_pos_norm
                .copy_from_async(&batch.per_pos_norm, &self.stream)?;
        }
        self.input_upload_done.record(&self.stream)?;

        let n_pos_u32 = n_pos as u32;
        let max_inds_u32 = MAX_INDS_PER_POS as u32;
        self.launch_forward(n_pos_u32, max_inds_u32)?;
        self.launch_grad(n_pos_u32, max_inds_u32)?;

        // Adam step
        self.beta1_pow *= ADAM_BETA1;
        self.beta2_pow *= ADAM_BETA2;
        let bc1 = 1.0_f32 - self.beta1_pow;
        let bc2 = 1.0_f32 - self.beta2_pow;
        let beta1 = ADAM_BETA1;
        let beta2 = ADAM_BETA2;
        let eps = ADAM_EPS;
        let n_w = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
        let n_w_u32 = n_w as u32;
        self.launch_adam(lr, beta1, beta2, eps, bc1, bc2, n_w_u32)?;

        Ok(())
    }

    /// 評価 path: forward → eval kernel (loss + histogram のみ、weight 不変)。
    fn eval_forward(&mut self, batch: &Batch) -> Result<(), Box<dyn std::error::Error>> {
        let n_pos = batch.n_positions;
        if n_pos == 0 {
            return Ok(());
        }

        self.ensure_input_capacity(n_pos)?;
        // SAFETY: batch storage remains live and immutable until input_upload_done is synchronized.
        unsafe {
            self.indices.copy_from_async(&batch.indices, &self.stream)?;
            self.targets.copy_from_async(&batch.targets, &self.stream)?;
        }
        self.input_upload_done.record(&self.stream)?;

        let n_pos_u32 = n_pos as u32;
        let max_inds_u32 = MAX_INDS_PER_POS as u32;
        self.launch_forward(n_pos_u32, max_inds_u32)?;
        self.launch_eval(n_pos_u32)?;

        Ok(())
    }

    /// 直前の `step` / `eval_forward` が発行した入力 H2D の完了を host で待つ。
    fn synchronize_input_upload(&self) -> cuda_native_runtime::Result<()> {
        self.input_upload_done.synchronize()
    }

    fn read_loss_hist(&self) -> cuda_native_runtime::Result<(f64, [u64; 8])> {
        self.stream.synchronize()?;
        let mut loss_vec = [0.0_f64; 1];
        self.loss_acc.copy_to(&mut loss_vec)?;
        let mut hist_vec = [0_u64; 8];
        self.hist.copy_to(&mut hist_vec)?;
        let mut hist_arr = [0_u64; 8];
        hist_arr.copy_from_slice(&hist_vec[..8]);
        Ok((loss_vec[0], hist_arr))
    }

    fn read_weights(&self) -> cuda_native_runtime::Result<Vec<f32>> {
        self.stream.synchronize()?;
        let mut weights = vec![0.0; self.weights.len()];
        self.weights.copy_to(&mut weights)?;
        Ok(weights)
    }

    fn launch_forward(&self, mut n_pos: u32, mut max_inds: u32) -> cuda_native_runtime::Result<()> {
        let mut indices = self.indices.device_ptr();
        let mut indices_len = self.indices.len() as u64;
        let mut weights = self.weights.device_ptr();
        let mut weights_len = self.weights.len() as u64;
        let mut preds = self.preds.device_ptr();
        let mut preds_len = self.preds.len() as u64;
        let mut args = [
            arg(&mut indices),
            arg(&mut indices_len),
            arg(&mut weights),
            arg(&mut weights_len),
            arg(&mut preds),
            arg(&mut preds_len),
            arg(&mut n_pos),
            arg(&mut max_inds),
        ];
        // SAFETY: arguments match progress_forward and the logical sizes bound every allocation.
        unsafe {
            self.forward.launch(
                &self.stream,
                grid_dim_1d(n_pos as usize),
                (BLOCK_DIM, 1, 1),
                0,
                &mut args,
            )
        }
    }

    fn launch_grad(&self, mut n_pos: u32, mut max_inds: u32) -> cuda_native_runtime::Result<()> {
        let mut indices = self.indices.device_ptr();
        let mut indices_len = self.indices.len() as u64;
        let mut preds = self.preds.device_ptr();
        let mut preds_len = self.preds.len() as u64;
        let mut targets = self.targets.device_ptr();
        let mut targets_len = self.targets.len() as u64;
        let mut norm = self.per_pos_norm.device_ptr();
        let mut norm_len = self.per_pos_norm.len() as u64;
        let mut grad = self.grad.device_ptr();
        let mut grad_len = self.grad.len() as u64;
        let mut loss = self.loss_acc.device_ptr();
        let mut loss_len = self.loss_acc.len() as u64;
        let mut hist = self.hist.device_ptr();
        let mut hist_len = self.hist.len() as u64;
        let mut args = [
            arg(&mut indices),
            arg(&mut indices_len),
            arg(&mut preds),
            arg(&mut preds_len),
            arg(&mut targets),
            arg(&mut targets_len),
            arg(&mut norm),
            arg(&mut norm_len),
            arg(&mut grad),
            arg(&mut grad_len),
            arg(&mut loss),
            arg(&mut loss_len),
            arg(&mut hist),
            arg(&mut hist_len),
            arg(&mut n_pos),
            arg(&mut max_inds),
        ];
        // SAFETY: arguments match progress_grad and the logical sizes bound every allocation.
        unsafe {
            self.grad_kernel.launch(
                &self.stream,
                grid_dim_1d(n_pos as usize),
                (BLOCK_DIM, 1, 1),
                0,
                &mut args,
            )
        }
    }

    fn launch_eval(&self, mut n_pos: u32) -> cuda_native_runtime::Result<()> {
        let mut preds = self.preds.device_ptr();
        let mut preds_len = self.preds.len() as u64;
        let mut targets = self.targets.device_ptr();
        let mut targets_len = self.targets.len() as u64;
        let mut loss = self.loss_acc.device_ptr();
        let mut loss_len = self.loss_acc.len() as u64;
        let mut hist = self.hist.device_ptr();
        let mut hist_len = self.hist.len() as u64;
        let mut args = [
            arg(&mut preds),
            arg(&mut preds_len),
            arg(&mut targets),
            arg(&mut targets_len),
            arg(&mut loss),
            arg(&mut loss_len),
            arg(&mut hist),
            arg(&mut hist_len),
            arg(&mut n_pos),
        ];
        // SAFETY: arguments match progress_eval and the logical sizes bound every allocation.
        unsafe {
            self.eval.launch(
                &self.stream,
                grid_dim_1d(n_pos as usize),
                (BLOCK_DIM, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_adam(
        &self,
        mut lr: f32,
        mut beta1: f32,
        mut beta2: f32,
        mut eps: f32,
        mut bc1: f32,
        mut bc2: f32,
        mut n: u32,
    ) -> cuda_native_runtime::Result<()> {
        let mut weights = self.weights.device_ptr();
        let mut weights_len = self.weights.len() as u64;
        let mut momentum = self.m.device_ptr();
        let mut momentum_len = self.m.len() as u64;
        let mut velocity = self.v.device_ptr();
        let mut velocity_len = self.v.len() as u64;
        let mut grad = self.grad.device_ptr();
        let mut grad_len = self.grad.len() as u64;
        let mut args = [
            arg(&mut weights),
            arg(&mut weights_len),
            arg(&mut momentum),
            arg(&mut momentum_len),
            arg(&mut velocity),
            arg(&mut velocity_len),
            arg(&mut grad),
            arg(&mut grad_len),
            arg(&mut lr),
            arg(&mut beta1),
            arg(&mut beta2),
            arg(&mut eps),
            arg(&mut bc1),
            arg(&mut bc2),
            arg(&mut n),
        ];
        // SAFETY: arguments match progress_adam_step and the logical sizes bound every allocation.
        unsafe {
            self.adam_step.launch(
                &self.stream,
                grid_dim_1d(n as usize),
                (BLOCK_DIM, 1, 1),
                0,
                &mut args,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Training driver
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct EpochStats {
    samples: usize,
    games: usize,
    steps: usize,
    mean_loss: f64,
    bucket_hist: [u64; 8],
}

/// データ走査 1 pass の役割。`Train` は訓練 game で `step` (forward → grad →
/// adam_step、weight 更新あり)、`Validate` は検証 game で `eval_forward`
/// (forward → eval kernel のみ、weight 不変) を回す。`Train` が保持する
/// `f32` は実効学習率。
#[derive(Clone, Copy)]
enum PassMode {
    Train(f32),
    Validate,
}

/// `game_seq` 番目 (非空 game を出現順に 0-indexed 採番) の game を検証へ回す
/// か判定する。`stride` が `None` (検証無効) なら常に `false`。stride `N` の
/// とき `N` game ごとに 1 個 (`game_seq % N == 0`) を検証に割り当てる。
fn is_val_game(game_seq: u64, stride: Option<u64>) -> bool {
    match stride {
        None => false,
        Some(n) => game_seq.is_multiple_of(n),
    }
}

/// 1 batch を `mode` に応じて実行する。`Train` は forward → grad → adam_step
/// (weight 更新あり)、`Validate` は forward → eval kernel のみ。
fn run_batch(
    trainer: &mut GpuTrainer,
    batch: &Batch,
    mode: PassMode,
) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        PassMode::Train(lr) => trainer.step(batch, lr),
        PassMode::Validate => trainer.eval_forward(batch),
    }
}

/// data file 群を 1 回走査し、`mode` に応じて訓練 / 検証の片側を実行する。
///
/// `--val-fraction` 有効時は非空 game を出現順に 0-indexed で採番し
/// (`game_seq`)、`is_val_game` で訓練用 / 検証用に振り分ける。`Train` pass は
/// 検証 game を、`Validate` pass は訓練 game を skip する。両 pass は同じ
/// `game_seq` 採番と同じ `max_games` 上限を使うので、同一 `args` で順に 2 回
/// 呼べば排他かつ網羅的な game 分割になる。検証 loss は epoch 完了後の固定
/// weight で測る必要があるため、呼び出し側は `Train` pass 完了後に
/// `Validate` pass を回す。
fn run_data_pass(
    trainer: &mut GpuTrainer,
    data_files: &[PathBuf],
    args: &Args,
    epoch: usize,
    mode: PassMode,
) -> Result<EpochStats, Box<dyn std::error::Error>> {
    trainer.zero_loss_hist()?;

    let max_games = if args.max_games > 0 {
        Some(args.max_games as u64)
    } else {
        None
    };
    let val_stride = args.val_stride();
    let want_val = matches!(mode, PassMode::Validate);

    let mut batch = Batch::new();
    let mut scratch: Vec<usize> = Vec::with_capacity(96);
    let mut samples_total = 0_usize;
    let mut games_total = 0_usize;
    let mut steps = 0_usize;
    let mut game_seq = 0_u64;
    let start = Instant::now();

    'outer: for path in data_files {
        let cursor = PackCursor::open(path)?;
        let mut gi = GameIterator::new(cursor);
        while let Some(game) = gi.next_game()? {
            if game.is_empty() {
                continue;
            }
            // `max_games` は走査した非空 game 数 (訓練 + 検証 合算) の上限。
            // 両 pass が同じ閾値を使うことで採番と分割が一致する。
            if let Some(limit) = max_games
                && game_seq >= limit
            {
                break 'outer;
            }
            let seq = game_seq;
            game_seq += 1;
            // この pass が担当しない側の game は採番だけ進めて skip する
            // (pass 間で採番を一致させるため skip 側も game_seq を消費する)。
            if is_val_game(seq, val_stride) != want_val {
                continue;
            }
            batch.push_game(&game, &mut scratch);
            if batch.n_games >= args.games_per_step {
                batch.finalize();
                games_total += batch.n_games;
                samples_total += batch.n_positions;
                run_batch(trainer, &batch, mode)?;
                steps += 1;
                // H2D 完了後だけ転送元 `Batch` storage を再利用する。event は kernel
                // より前に record 済みなので、GPU compute の完了は待たない。
                trainer.synchronize_input_upload()?;
                batch.clear();

                if matches!(mode, PassMode::Train(_))
                    && args.log_interval_steps > 0
                    && steps.is_multiple_of(args.log_interval_steps)
                {
                    let (loss_sum, _) = trainer.read_loss_hist()?;
                    let avg = if samples_total > 0 {
                        loss_sum / samples_total as f64
                    } else {
                        0.0
                    };
                    let elapsed = start.elapsed().as_secs_f64();
                    let games_per_sec = games_total as f64 / elapsed.max(1e-9);
                    println!(
                        "epoch {} steps {} games {} samples {} avg_loss {:.6} games/s {:.0}",
                        epoch, steps, games_total, samples_total, avg, games_per_sec
                    );
                }
            }
        }
    }
    // 残り (n_games < games_per_step) も 1 batch として処理。
    if batch.n_games > 0 {
        batch.finalize();
        games_total += batch.n_games;
        samples_total += batch.n_positions;
        run_batch(trainer, &batch, mode)?;
        steps += 1;
    }

    let (loss_sum, hist) = trainer.read_loss_hist()?;
    let mean_loss = if samples_total > 0 {
        loss_sum / samples_total as f64
    } else {
        0.0
    };
    Ok(EpochStats {
        samples: samples_total,
        games: games_total,
        steps,
        mean_loss,
        bucket_hist: hist,
    })
}

/// `--output` の path から epoch `epoch` の checkpoint path を導く。拡張子の
/// 手前に `.e{epoch}` を挿入する (`out/foo.bin` → `out/foo.e3.bin`、拡張子の
/// 無い `out/foo` → `out/foo.e3`)。
fn epoch_checkpoint_path(output: &Path, epoch: usize) -> PathBuf {
    let mut name = output.file_stem().unwrap_or_default().to_os_string();
    name.push(format!(".e{epoch}"));
    if let Some(ext) = output.extension() {
        name.push(".");
        name.push(ext);
    }
    output.with_file_name(name)
}

fn run_training(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.epochs == 0 {
        return Err("--epochs must be >= 1".into());
    }
    if args.games_per_step == 0 {
        return Err("--games-per-step must be >= 1".into());
    }
    // 0.0..=0.5 の外 (負値 / 0.5 超 / NaN / inf) は reject。NaN/inf は
    // `RangeInclusive::contains` が false を返すのでこの 1 条件で弾ける。
    if !(0.0..=0.5).contains(&args.val_fraction) {
        return Err(format!(
            "--val-fraction must be within 0.0..=0.5 (got {}); 0.0 disables held-out validation",
            args.val_fraction
        )
        .into());
    }

    let data_paths = args.data_paths();
    if data_paths.is_empty() {
        return Err("--data is required (comma-separated PSV files)".into());
    }
    for p in &data_paths {
        if !p.exists() {
            return Err(format!("data file not found: {}", p.display()).into());
        }
    }

    let init_weights = args
        .init_from
        .as_deref()
        .map(read_progress_bin)
        .transpose()?;

    let device = i32::try_from(args.device).map_err(|_| "--device exceeds CUDA ordinal range")?;
    let ctx = Context::new(device)?;
    println!(
        "CUDA device {} ready, kernel module loading...",
        args.device
    );
    let mut trainer = GpuTrainer::new(&ctx, init_weights.as_deref(), args.games_per_step)?;
    println!(
        "GpuTrainer ready: {} weights, batch={} games, lr={} (effective={})",
        SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
        args.games_per_step,
        args.lr,
        args.effective_lr()
    );

    let lr = args.effective_lr();
    let val_enabled = args.val_stride().is_some();
    for epoch in 1..=args.epochs {
        let train = run_data_pass(&mut trainer, &data_paths, &args, epoch, PassMode::Train(lr))?;
        // 訓練 position が 0 のまま進むと未学習の weight を progress.bin に書き
        // 出してしまうので、その前に明示エラーにする。検証有効時は game_seq 0 が
        // 必ず検証側へ回る (`is_val_game`: 0 は任意 stride の倍数) ため、走査
        // game 数が少なすぎると訓練側だけが空になりうる。
        if train.samples == 0 {
            return Err(if val_enabled {
                "--val-fraction holdout left no training games; at least 2 games \
                 are required (raise --max-games or use a larger dataset)"
            } else {
                "no training positions: the data files contain no non-empty games"
            }
            .into());
        }
        if val_enabled {
            // 検証 loss は epoch 完了後の固定 weight で測る必要があるため、訓練
            // pass の後にデータをもう一度走査して検証 game だけ評価する。訓練側に
            // game があれば game_seq 0 は検証側に入るので `val.samples > 0` も保証。
            let val = run_data_pass(&mut trainer, &data_paths, &args, epoch, PassMode::Validate)?;
            println!(
                "EPOCH {} DONE: train_games={} val_games={} train_samples={} val_samples={} \
                 train_steps={} train_loss={:.6} val_loss={:.6} train_hist={:?} val_hist={:?}",
                epoch,
                train.games,
                val.games,
                train.samples,
                val.samples,
                train.steps,
                train.mean_loss,
                val.mean_loss,
                train.bucket_hist,
                val.bucket_hist,
            );
        } else {
            println!(
                "EPOCH {} DONE: games={} samples={} steps={} mean_loss={:.6} hist={:?}",
                epoch, train.games, train.samples, train.steps, train.mean_loss, train.bucket_hist
            );
        }

        // epoch ごとの checkpoint を書き出し、どの epoch を採用するか後から
        // 選べるようにする。最終 epoch の重みはループ後に `--output` へも書く。
        let epoch_weights = trainer.read_weights()?;
        let checkpoint = epoch_checkpoint_path(&args.output, epoch);
        write_progress_bin(&checkpoint, &epoch_weights)?;
        println!("wrote epoch {epoch} checkpoint: {}", checkpoint.display());
    }

    let weights = trainer.read_weights()?;
    write_progress_bin(&args.output, &weights)?;
    println!(
        "wrote progress.bin: {} ({} weights, {} bytes)",
        args.output.display(),
        weights.len(),
        weights.len() * std::mem::size_of::<f64>()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run_training(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

/// 訓練 / 検証 game の振り分けロジックの単体テスト (GPU 不要)。
#[cfg(test)]
mod driver_logic_tests {
    use super::{epoch_checkpoint_path, is_val_game};

    #[test]
    fn epoch_checkpoint_path_inserts_before_extension() {
        assert_eq!(
            epoch_checkpoint_path(std::path::Path::new("out/progress/foo.bin"), 3),
            std::path::PathBuf::from("out/progress/foo.e3.bin")
        );
    }

    #[test]
    fn epoch_checkpoint_path_handles_missing_extension() {
        assert_eq!(
            epoch_checkpoint_path(std::path::Path::new("foo"), 1),
            std::path::PathBuf::from("foo.e1")
        );
    }

    #[test]
    fn val_disabled_routes_every_game_to_train() {
        for seq in 0..256 {
            assert!(!is_val_game(seq, None), "seq {seq} must not be val");
        }
    }

    #[test]
    fn stride_routes_every_nth_game_to_val() {
        let stride = Some(20);
        assert!(is_val_game(0, stride));
        assert!(is_val_game(20, stride));
        assert!(is_val_game(40, stride));
        assert!(!is_val_game(1, stride));
        assert!(!is_val_game(19, stride));
        assert!(!is_val_game(21, stride));
        // 1000 game 中ちょうど 1/20 (= 50 個) が検証へ回る。
        let val_count = (0..1000_u64).filter(|&s| is_val_game(s, stride)).count();
        assert_eq!(val_count, 50);
    }
}

// ---------------------------------------------------------------------------
