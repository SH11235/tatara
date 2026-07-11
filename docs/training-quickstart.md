**English** | [日本語](training-quickstart.ja.md)

# Training quickstart

The shortest path to training a shogi NNUE from scratch with `nnue-train`. GPUs:
Ampere+ (sm_80+) is official, Turing uses `CUDA_OXIDE_TARGET=sm_75`. For setting
up the toolchain and CUDA / LLVM, see [docs/setup.md](setup.md).

A trained NNUE is defined by choosing an architecture (`simple` / `layerstack`)
and an input feature set (for the options, see "What you can train" in the
[README](../README.md)). This page shows the steps using two configurations as
examples:

- **Example 1: HalfKP NNUE** (`simple` architecture) — the minimal
  configuration. No buckets, so little preparation
- **Example 2: LayerStack NNUE** — a configuration that uses the 9 game-progress
  buckets

## Required inputs

| File | Format | Purpose | Approx. size |
|---|---|---|---:|
| Training data PSV | `PackedSfenValue` × N (fixed 40 bytes / position) | Passed via `--data` | Hundreds of GB |
| progress coefficients | `progress.bin` (f64 LE; 81 king squares × 1548 KP-abs piece inputs = fixed `1_003_104` bytes) | Passed via `--progress-coeff`. For LayerStack's 9-bucket assignment (not needed for simple) | 1.0 MB |
| (optional) pretrained NNUE | quantised `.bin` (`save_quantised` format) | Injects weights via `--init-from` (the optimizer is reset) | — |

## Example 1: Training a HalfKP NNUE (simple architecture)

The `simple` architecture has no buckets, so it does not need `progress.bin`.
The minimal configuration:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --feature-set halfkp \
  --superbatches <N> \
  --threads <N> \
  simple
```

`simple` defaults to `--arch 256x2-32-32` / `--activation crelu`. For how to
choose `--superbatches` and the additional options you can pass, see "Key
options" below.

## Example 2: Training a LayerStack NNUE

The `layerstack` architecture uses the 9 game-progress buckets, so prepare the
bucket coefficients `progress.bin` first — train it with `progress-kpabs-train`
and check the bucket split with `progress-bucket-survey`, as described in
[Game-progress buckets: preparing `progress.bin`](progress-bin.md).

### Training

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> \
  --threads <N> \
  layerstack --progress-coeff <path/to/progress.bin>
```

`layerstack` defaults to `--feature-set halfka-hm-merged` / 9 buckets. The FT
output dimension can be changed with `--ft-out` (a multiple of 128, default
1536).

## Key options

`nnue-train`'s CLI defaults are small, for smoke testing. The ones you mainly
change for real training are:

| Option | CLI default | Description |
|---|---:|---|
| `--superbatches` | 10 | Number of superbatches to train. The default 10 is for smoke testing; use a much larger value for real training (see "How much to train" below) |
| `--batch-size` | 16384 | Number of positions per gradient update. A training hyperparameter that affects both GPU throughput and training dynamics (gradient variance, number of updates) |
| `--feature-set` | halfka-hm-merged | Input feature set. Choose from `halfkp` / `halfka-split` / `halfka-merged` / `halfka-hm-split` / `halfka-hm-merged` (see the [README](../README.md)) |
| `--keep-checkpoints` | keep all | Keep the most recent N raw `.ckpt` files (weight + optimizer state). The default of keeping all is the safe choice for tracking training failures. Note that disk usage adds up: with `--save-rate 20` over a 400-superbatch run you accumulate 20 `.ckpt` files × ~1.8 GB (default LayerStack arch) ≈ 36 GB. Limit it if disk space is tight. Quantised `.bin` files are always kept |
| `--win-rate-model` | OFF | WRM (win-rate-model) loss. Converges to `net_output ≈ cp/600`, consistent with quantisation (`QA=127 / QB=64 / FV_SCALE=28`). Add it if you are training a net for quantised inference (without it, plain sigmoid-MSE). See [Tuning the WRM loss](wrm-loss-tuning.md) for the tuning parameters |
| `--score-drop-abs` | none | Exclude positions with `|score| >=` this value from the loss (rejects extreme evaluations near mate) |
| `--score-clamp-abs` | none | Saturate surviving positions' scores to `[-N, N]` (normalises teacher files whose encode variants clip at different ceilings) |
| `--threads` | 16 | **Always set this.** Because GPU processing is fast, the CPU dataloader is easily the bottleneck; a larger value is recommended. Use your CPU's physical core count as a starting point — a small value (e.g. 1) will cause a large drop in pos/s. Use `NNUE_TRAIN_STEP_PROFILE=1` to see the h2d / fwd / bwd / optimizer breakdown and tune accordingly |
| `--test-tail-positions` | none | Reserve the last N positions of `--data` as a held-out validation set in the same file (see "Held-out validation" below). Recommended whenever you want held-out validation |
| `--test-positions` | 10000 | Number of positions evaluated each superbatch from the held-out source. Used only with `--test-tail-positions` or `--test-data` |
| `--num-buckets` (`layerstack`) | 9 | LayerStack output bucket count, an integer in `[2, 9]`. Each position is routed to `min(N-1, floor(progress * N))`. Lower values trade per-bucket specialisation for more samples per bucket; the default 9 keeps the binning identical to existing distributed nets |

`--batches-per-superbatch` (6104) / `--lr` (8.75e-4) / `--save-rate` (20)
and the like can be left at their defaults; pass them only when you want to
change them.

**How much to train**: 1 superbatch = `batches-per-superbatch × batch-size`
positions. With the default `batch-size`, 1 superbatch ≈ 100 million positions,
roughly the same scale as one epoch of the upstream chess NNUE trainer
[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) (default
`--epoch-size` = 100 million positions). nnue-pytorch's default is 800 epochs.
Decide `--superbatches` by balancing the amount of training data against
overfitting.

The time it takes varies greatly with the GPU and the configuration (whether
FP16 modes are on).

## Held-out validation

To watch for overfitting and divergence (NaN) without waiting for SPRT
self-play, add held-out validation: positions that are never used for a
gradient update, scored each superbatch and reported as `test_loss` /
`test_acc`. Enable it with `--test-tail-positions` (or `--test-data`) plus
`--test-positions`; see [Held-out validation](held-out-validation.md) for the
flags, how to pick the held-out source, and how to read the metrics.

## Interrupting and resuming training

A raw `.ckpt` saves everything: **weights + Ranger optimizer state
(m / v / slow / step) + the current superbatch number**. Even if it stops on a
power loss or a GPU error, you can fully resume. Add `--resume` to the same
options + architecture subcommand used for training:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --feature-set halfkp --superbatches <N> --keep-checkpoints 4 \
  --resume checkpoints/<run-name>/<run-name>-<sb>.ckpt \
  simple
```

With `--resume` (and `--start-superbatch` omitted), it resumes from the
checkpoint's sb + 1; specifying `--start-superbatch N` explicitly lets you redo
past superbatches.

For LR schedules with a horizon, the checkpoint records the resolved horizon so
the curve is reproduced on resume independently of `--superbatches`; see
[Horizon and resuming](training-schedule.md#horizon-and-resuming) for the
precedence rules.

> **Difference between `--resume` and `--init-from`**: `--init-from` injects
> only the weights from a quantised `.bin` and **resets** the optimizer state
> (fine-tuning / continued training); `--resume` restores both weights and
> optimizer from a raw `.ckpt` (a true resume). The two are mutually exclusive.

## Reading the output artifacts

After training, what appears under `checkpoints/<run-name>/`:

| File | Format | Purpose |
|---|---|---|
| `<run-name>-<sb>.bin` | quantised NNUE binary | **The artifact to feed to the inference engine** (for the binary layout, see `crates/nnue-format`) |
| `<run-name>-<sb>.ckpt` | raw f32 + optimizer state | For `--resume`; not used for inference (pruned by `--keep-checkpoints`) |

`<run-name>-<final sb>.bin` is the final net. It is loaded by the
[rshogi](https://github.com/SH11235/rshogi) engine — not by other shogi engines
such as YaneuraOu (see
[Using the trained net](../README.md#using-the-trained-net)). Measure playing
strength by integrating it into the engine.

## Smoke test

If you want to check just the GPU path before preparing data, add the
architecture subcommand and omit `--data`: a smoke test runs that executes the
`GpuTrainer` forward / backward path for a single step:

```bash
target/release/nnue-train simple
# → if the run ends with a "[smoke/simple] PASSED" line, the GPU path is healthy
```

Or run the whole pipeline in a few seconds with a small run (1 sb × 3 batches):

```bash
target/release/nnue-train --data <PSV> \
  --output /tmp/smoke --net-id smoke \
  --superbatches 1 --batches-per-superbatch 3 \
  --save-rate 1 --threads 4 \
  simple
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `kernel artifact nnue_train.{cubin,ptx,ll} not found` | On the first build you need to generate the `.ll` with `cd bins/nnue_train && cargo-oxide build`. For details, see [docs/setup.md](setup.md) |
| `libcublas.so` link / load errors | The CUDA Toolkit is in none of `/usr/local/cuda` / `CUDA_HOME` / `CUDA_PATH`. Specify it explicitly with `CUDA_TOOLKIT_PATH=/path/to/cuda-12.x` (both build.rs and runtime resolve via the same chain) |
| `CUDA_ERROR_INVALID_PTX` (driver error 218) | On a sub-Ampere GPU (sm_75) with `CUDA_OXIDE_TARGET` unset. Export `CUDA_OXIDE_TARGET=sm_75`, then rebuild and rerun |
| pos/s extremely low (< 500K on an RTX 3080 Ti) | Increase `--threads` (start from your physical core count, see "Key options") and check whether the dataloader's prefetch is keeping up. `NNUE_TRAIN_STEP_PROFILE=1` prints the ms spent in each phase (h2d / fwd / bwd / optimizer) to stderr so you can see the breakdown |
| rejected with `--batch-size must be a multiple of 16` | The tiled dense matmul kernels require `b % 16 == 0`, so the CLI rejects other values at startup. Pass a multiple of 16 (the default 16384 satisfies the condition) |

## Related

- [docs/setup.md](setup.md) — toolchain (LLVM / CUDA / cuda-oxide) setup
- [Game-progress buckets: preparing `progress.bin`](progress-bin.md) — training and surveying the LayerStack bucket coefficients
- [Held-out validation](held-out-validation.md) — `test_loss` / `test_acc` setup and how to read the metrics
- [Training schedules](training-schedule.md) — learning-rate and WDL lambda scheduling
- [Tuning the WRM loss](wrm-loss-tuning.md) — the WRM transform and its tuning options (win-rate transform + generalized loss form)
