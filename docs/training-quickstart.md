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
  --keep-checkpoints 4 \
  simple
```

`simple` defaults to `--arch 256x2-32-32` / `--activation crelu`. For how to
choose `--superbatches` and the additional options you can pass, see "Key
options" below.

## Example 2: Training a LayerStack NNUE

The `layerstack` architecture uses the 9 game-progress buckets, so prepare the
bucket coefficients `progress.bin` first.

### Generating progress.bin

Train the progress coefficients with `progress-kpabs-train`. The idea of
learning game progress and assigning it to output buckets is based on
[a post by nodchip](https://nodchip.hatenablog.com/entry/2026/02/04/000000).

> **Do not shuffle the data.** Pass `progress-kpabs-train`'s `--data` a PSV of
> **consecutive games** (positions in game order, with games following one
> after another). The progress coefficients learn "how far a position has
> advanced within a single game", and `progress-kpabs-train` reads the data one
> game at a time (detecting game boundaries by `game_ply`) and labels each
> position with its relative position within that game. With a shuffled PSV the
> game boundaries break, the labels become meaningless, and correct coefficients
> cannot be learned. The `nnue-train` examples for the main training use a
> shuffled PSV — the requirement is the opposite, so do not mix them up.

Specify the total number of epochs with `--epochs`; a `<run-name>.e<N>.bin` is
written per epoch.

```bash
target/release/progress-kpabs-train \
  --data <path/to/consecutive-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

Which epoch's output (`<run-name>.e<N>.bin`) to use takes some trial and error
(progress.bin is a coefficient that decides bucket assignment and is independent
of NNUE training convergence, so how many epochs you need is data-dependent).

### Training

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> \
  --keep-checkpoints 4 \
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
| `--keep-checkpoints` | keep all | Keep the most recent N raw `.ckpt` files (weight + optimizer state, large in size). Start small (e.g. 4); once training is stable you can raise it to 20–100. Quantised `.bin` files are always kept |
| `--win-rate-model` | OFF | WRM (win-rate-model) loss. Converges to `net_output ≈ cp/600`, consistent with quantisation (`QA=127 / QB=64 / FV_SCALE=28`). Add it if you are training a net for quantised inference (without it, plain sigmoid-MSE) |
| `--score-drop-abs` | none | Exclude positions with `|score| >=` this value from the loss (rejects extreme evaluations near mate) |

`--batches-per-superbatch` (6104) / `--lr` (8.75e-4) / `--save-rate` (20) /
`--threads` (16) and the like can be left at their defaults; pass them only when
you want to change them.

**How much to train**: 1 superbatch = `batches-per-superbatch × batch-size`
positions. With the default `batch-size`, 1 superbatch ≈ 100 million positions,
roughly the same scale as one epoch of the upstream chess NNUE trainer
[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) (default
`--epoch-size` = 100 million positions). nnue-pytorch's default is 800 epochs.
Decide `--superbatches` by balancing the amount of training data against
overfitting.

The time it takes varies greatly with the GPU and the configuration (whether
FP16 modes are on).

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

`<run-name>-<final sb>.bin` is the final net. Measure playing strength by
integrating it into the shogi engine.

## Smoke test

If you want to check just the GPU path before preparing data, add the
architecture subcommand and omit `--data`: a smoke test runs that executes the
`GpuTrainer` forward / backward path for a single step:

```bash
target/release/nnue-train simple
# → if a log to the effect of "[smoke] forward + backward OK" appears, the GPU path is healthy
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
| pos/s extremely low (< 500K on an RTX 3080 Ti) | Set `--threads` to about half your CPU core count and check whether the dataloader's prefetch is keeping up. `NNUE_TRAIN_STEP_PROFILE=1` prints the ms spent in each phase (h2d / fwd / bwd / optimizer) to stderr so you can see the breakdown |
| rejected with `--batch-size % 16 != 0` | The tiled L1 kernel requires `b % 16 == 0` (fails via `debug_assert!`). Pass a multiple of 16 (the default 16384 satisfies the condition) |

## Related

- [docs/setup.md](setup.md) — toolchain (LLVM / CUDA / cuda-oxide) setup
