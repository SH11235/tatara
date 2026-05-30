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
> cannot be learned. As a general rule the main NNUE training (`nnue-train`)
> benefits from a *shuffled* PSV; progress training is the opposite, so do not
> reuse the same file for both.

Specify the total number of epochs with `--epochs`. Each epoch writes a
`<run-name>.e<N>.bin` checkpoint, and the final epoch is also written to the
`--output` path.

```bash
target/release/progress-kpabs-train \
  --data <path/to/consecutive-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

Which epoch's output (`<run-name>.e<N>.bin`) to use takes some trial and error
(progress.bin is a coefficient that decides bucket assignment and is independent
of NNUE training convergence, so how many epochs you need is data-dependent).

To help decide which epoch to use, add `--val-fraction <f>` (e.g. `0.05`): it
holds out roughly that fraction of games — every Nth game in input order, since
the data must stay in consecutive-game order — and reports a held-out `val_loss`
at the end of each epoch. This adds one extra pass over the data per epoch.

Treat `val_loss` as a sanity check and an epoch-selection hint, not a precise
quality score. The progress model is simple — per-feature weights summed and
passed through a sigmoid — so it overfits little: a small `train_loss`/`val_loss`
gap is normal, and a clearly widening gap is what to watch for. Because the real
goal is a good bucket split, which a plain MSE only approximates, prefer an
epoch where `val_loss` levels off over chasing its exact minimum, and judge the
final `progress.bin` by the strength of the LayerStack NNUE trained with it.

### Checking the bucket distribution

`progress-bucket-survey` reports how a `progress.bin` assigns positions to the
progress buckets. A roughly even spread is healthy; if one bucket dominates, the
LayerStack output buckets are trained on very uneven amounts of data.

```bash
cargo build --release -p progress-bucket-survey
target/release/progress-bucket-survey \
  --data <path/to/consecutive-psv.bin> \
  --progress output/progress/<run-name>.e5.bin \
  --samples 200000
```

It prints a per-bucket count and percentage plus the top bucket's share. Only
one `progress.bin` can be loaded per run, so to compare epochs run it once per
`<run-name>.e<N>.bin` and compare the outputs.

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
| `--keep-checkpoints` | keep all | Keep the most recent N raw `.ckpt` files (weight + optimizer state). The default of keeping all is the safe choice for tracking training failures. Note that disk usage adds up: with `--save-rate 20` over a 400-superbatch run you accumulate 20 `.ckpt` files × ~100 MB ≈ 2 GB. Limit it if disk space is tight. Quantised `.bin` files are always kept |
| `--win-rate-model` | OFF | WRM (win-rate-model) loss. Converges to `net_output ≈ cp/600`, consistent with quantisation (`QA=127 / QB=64 / FV_SCALE=28`). Add it if you are training a net for quantised inference (without it, plain sigmoid-MSE). See [Tuning the WRM loss](wrm-loss-tuning.md) for the tuning parameters |
| `--score-drop-abs` | none | Exclude positions with `|score| >=` this value from the loss (rejects extreme evaluations near mate) |
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
gradient update. Each superbatch ends with a forward-only pass over them; the
training log prints `test_loss` / `test_acc` (compact field name for
console), and `experiment.json` records the same numbers as `test_loss` /
`test_accuracy`.

### Three related flags

| Flag | Role | Type |
|---|---|---|
| `--test-tail-positions <N>` | Held-out **source**: the last N positions of `--data` itself | source A |
| `--test-data <PATH>` | Held-out **source**: a separate PSV file | source B |
| `--test-positions <K>` | How many positions are **evaluated** per superbatch from the chosen source | evaluation size, shared |

`--test-tail-positions` and `--test-data` are alternative held-out sources and
are mutually exclusive — pick one (or neither, to disable held-out
validation). `--test-positions` is a separate parameter that determines how
many positions of that chosen source actually get scored every superbatch
(drawn from the start of the source and rounded up to a whole `--batch-size`
multiple).

### Which source to pick

- **`--test-tail-positions <N>` (recommended)**: reserve the last N positions
  of `--data` itself. Training reads `[0, file_end - N * 40)` and validation
  reads `[file_end - N * 40, file_end)`; the two ranges are disjoint by
  construction so contamination cannot occur. One file does both jobs, which
  removes the need to prepare a separate held-out PSV and to keep two files in
  sync. The only cost is that training loses N positions from its pool — for
  `N << total positions` (e.g. 1e6 reserved out of 1e9 trained) this is well
  under 0.1% and not noticeable.
- **`--test-data <path>`**: a separate PSV file used only for validation. Use
  it when you have a held-out set that is meaningfully independent of `--data`
  (different generator, different time period, etc.) and you want that
  independence preserved. For ergonomic reasons alone there is no benefit to
  splitting `--data` into two files.

### Example

Reserve the last 1M positions as held-out, evaluate 10K of them each superbatch:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --test-tail-positions 1000000 \
  --test-positions 10000 \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> --threads <N> \
  layerstack --progress-coeff <path/to/progress.bin>
```

### Reading the metrics

`test_loss` uses the same loss kernel (sigmoid-MSE or WRM) and the same
`--wdl` blend as `train_loss`, so the two are on the same scale and can be
compared directly within a superbatch. A widening `test_loss − train_loss`
gap signals overfitting; an early `test_loss` divergence often catches NaN
issues before `train_loss` looks abnormal.

`test_acc` / `test_accuracy` is the sign-agreement between net output and the
game result (draws excluded from the denominator). It is scale-invariant, so
it can be compared across runs and configurations that have different loss
scales.

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
- [Tuning the WRM loss](wrm-loss-tuning.md) — the WRM transform and its 5 tuning options
