**English** | [日本語](training-quickstart.ja.md)

# Training quickstart

The shortest path to training a shogi NNUE from scratch with `nnue-train`. GPUs:
Ampere+ (sm_80+) is official; the default build detects the compute capability
of the installed GPU (Turing included) with `nvidia-smi` automatically. For
setting up the toolchain and CUDA / NVCC, see [docs/setup.md](setup.md).

A trained NNUE is defined by choosing an architecture (`simple` / `layerstack`)
and an input feature set (for the options, see "What you can train" in the
[README](../README.md)). This page shows the steps using two configurations as
examples:

- **Example 1: HalfKP NNUE** (`simple` architecture) — the minimal
  configuration. No buckets, so little preparation
- **Example 2: LayerStack NNUE** — a configuration that uses game-progress
  buckets (8 by default)

## Required inputs

| File | Format | Purpose | Approx. size |
|---|---|---|---:|
| Training data | PSV: `PackedSfenValue` × N (fixed 40 bytes / position), or HCPE: `HuffmanCodedPosAndEval` × N (fixed 38 bytes / position, `.hcpe` extension) | Passed via `--data` | Hundreds of GB |
| progress coefficients | `progress.bin` (f64 LE; 81 king squares × 1548 KP-abs piece inputs = fixed `1_003_104` bytes) | Passed via `--progress-coeff` for LayerStack `progresskpabs` / `progresskpabsq16` mode. Not used by `kingrank9` or `simple` | 1.0 MB |
| (optional) pretrained NNUE | quantised `.bin` (`save_quantised` format) | Injects weights via `--init-from` (the optimizer is reset) | — |

`--data` takes one file; repeat it to read several files of one format in the given order as one
data set. The format is chosen by the extension: `.hcpe` (case-insensitive)
is read as HCPE, anything else as PSV. For a symlink, the link name and the target name must imply
the same format. Variable-length hcpe3 files cannot be read directly; convert them to PSV first
with `hcpe3_to_psv` from [rshogi's tools](https://github.com/SH11235/rshogi/tree/main/crates/tools). To train on a mix of
formats, convert them to PSV first (`hcpe_to_psv` for HCPE). Shuffle the data in advance.

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
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple
```

The `simple` trainer requires `--win-rate-model`: its int8 output layer cannot
represent the centipawn-scale output that the plain sigmoid loss converges to,
so the plain-sigmoid path is rejected up front. The `--wrm-*` values above
degenerate the WRM to a plain sigmoid, so this configuration uses the same value
for `--scale`, every `--wrm-*-scaling`, and `--wrm-nnue2score`. When
`simple --fv-scale` is omitted, `--scale` must equal `--wrm-nnue2score` whether
or not weights are loaded. The exported `fv_scale` comes from a v9+ `--resume`
checkpoint when available, then from the `--init-from` input net, and otherwise
from `round(127 × QB / --scale)`. Set `simple --fv-scale <N>` to always write
`N` and waive that equality requirement.

The derived value is a nominal scale based on the training labels, not
necessarily the playing-strength optimum. Determine the optimum through games
and bake it in with `--fv-scale`, or set the engine-side option.

`simple` defaults to `--arch 256x2-32-32` / `--activation crelu`. For how to
choose `--superbatches` and the additional options you can pass, see "Key
options" below.

## Example 2: Training a LayerStack NNUE

The default LayerStack bucket mode is `progresskpabs`. It
uses 1–64 game-progress buckets. Prepare `progress.bin`
with `progress-kpabs-train` and inspect it with `progress-bucket-survey`, as
described in [Game-progress buckets: preparing `progress.bin`](progress-bin.md).
`progresskpabsq16` uses the same `progress.bin` and bucket count but sums the
coefficients as integers (`round(w * 65536)`) and compares against integer
thresholds, the routing used by rshogi `LS_BUCKET_MODE=progresskpabsq16`. Its
buckets differ from `progresskpabs` only for positions very close to a bucket
boundary, where the coefficient rounding decides the side. Train, rescore and
run inference with the same mode.
`kingrank9` matches YaneuraOu KingRank9: it normalises both king ranks to the
side-to-move perspective and selects one of a fixed 3x3 grid of 9 buckets. It
does not use `progress.bin`.

### Training

For new LayerStack training, choose `--layerstack-head split-skip` or
`--layerstack-head dense-concat`. Resume and eval-only use the saved head.
Any explicit head must match it. See [the architecture format](arch-string.md).

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> \
  --threads <N> \
  layerstack --layerstack-head split-skip --progress-coeff <path/to/progress.bin>
```

For KingRank9, replace the final line with:

```bash
  layerstack --layerstack-head split-skip --bucket-mode kingrank9
```

In `kingrank9` mode, omitting `--num-buckets` selects 9 (any other value is
rejected).

`layerstack` defaults to `--feature-set halfka-hm-merged` / 8 buckets. The FT
output dimension can be changed with `--ft-out` (a multiple of 128, default
1536).

Use `--bucket-mode spec --bucket-spec 'ownking:2/3/7+oppking:2/3/7'` to route by
both kings' ranks into 16 buckets. Join up to three axes with `+`. The leftmost
axis is most significant. For example, `handmajor4+ownking:2/3/7+oppking:2/3/7`
gives 64 buckets. `ownking` / `oppking` measure ranks from each king's home rank (1).
`ownking:2/3` gives bands 1, 2, and 3–9: each cut less than or equal to the
rank advances the band. `handmajor3` gives 0 when neither or both sides hold a
rook or bishop, 1 when only the side to move does, and 2 when only its opponent
does. `handmajor4` separates both sides holding a major piece into value 3.
The bucket count is the product of axis sizes (at most 64). An explicit
`--num-buckets` must match; `--progress-coeff` is rejected.

Use `--init-from <net.bin> --init-ft-only --freeze-ft` to load and fix the FT
(including Threat and PSQT) while initializing the output layers with
`--init-l1` / `--init-l1-shared` / `--init-l2` / `--init-l3`. Match the source
feature set and FT width. PSQT has bucket columns, so importing it requires
the same bucket count and spec. For the control, use the same initialization and freeze
options with a progress mode and its coefficients. Saved `.bin` files carry an
`ls_bucket=` token; `--eval-only --init-from <net.bin>` inherits that routing.
Other consumers need a reader that supports spec routing.

## Key options

`nnue-train`'s CLI defaults are small, for smoke testing. The ones you mainly
change for real training are:

| Option | CLI default | Description |
|---|---:|---|
| `--superbatches` | 10 | Number of superbatches to train. The default 10 is for smoke testing; use a much larger value for real training (see "How much to train" below) |
| `--batch-size` | 16384 | Number of positions per gradient update. A training hyperparameter that affects both GPU throughput and training dynamics (gradient variance, number of updates) |
| `--feature-set` | halfka-hm-merged | Input feature set. Choose from `halfkp` / `halfka-split` / `halfka-merged` / `halfka-hm-split` / `halfka-hm-merged` (see the [README](../README.md)) |
| `--keep-checkpoints` | keep all | Keep the most recent N raw `.ckpt` files (weight + optimizer state). The default of keeping all is the safe choice for tracking training failures. Note that disk usage adds up: with `--save-rate 20` over a 400-superbatch run you accumulate 20 `.ckpt` files × ~1.8 GB (default LayerStack arch) ≈ 36 GB. Limit it if disk space is tight. Quantised `.bin` files are always kept |
| `--win-rate-model` | OFF (layerstack) / required (simple) | WRM (win-rate-model) loss. Converges to `net_output ≈ cp / --wrm-nnue2score`. On `layerstack` it is optional (without it, plain sigmoid-MSE). The `simple` trainer requires it; when `simple --fv-scale` is omitted, it also requires `--scale = --wrm-nnue2score` whether or not weights are loaded. The example above uses 290 for both, producing `FV_SCALE=28`. See [Tuning the WRM loss](wrm-loss-tuning.md) for the tuning parameters |
| `--fv-scale` (`simple`) | none | Positive integer evaluation scale written to the exported architecture string. An explicit value always wins. When omitted, the value comes from a v9+ `--resume` checkpoint, then from the `--init-from` input net, and otherwise from `round(127 × QB / --scale)`; omission requires `--scale = --wrm-nnue2score` whether or not weights are loaded. A v9+ checkpoint needs no respecification; with v8 or earlier, repeat an override or the value falls back to the derived value |
| `--optimizer` | ranger | `ranger`, `radam`, `adamw`, or `sgd` (SGD with classical momentum 0.9). For an ordinary raw `.ckpt` resume, select the original optimizer. |
| `--transfer-radam-momentum` | off | Use with `--optimizer sgd --resume <RAdam .ckpt> --lr <SGD learning rate>`. Imports the unquantized weights and the first gradient moment. Starts the SGD step count, learning-rate schedule, and teacher input from the beginning. Confirm the source run used `--optimizer radam` and select a new output directory or net ID. |
| `--score-drop-abs` | none | Exclude positions with `|score| >=` this value from the loss (rejects extreme evaluations near mate) |
| `--score-clamp-abs` | none | Saturate surviving positions' scores to `[-N, N]` (normalises teacher files whose encode variants clip at different ceilings) |
| `--mix-data PATH` / `--mix-rate R` | none | Repeat in matching order to sample additional teachers; each finite rate is positive and their sum is less than 1. The remainder comes from the concatenated `--data` stream. All teachers must use the same format |
| `--mix-seed` | 0 | Stateless teacher-selection seed, independent of the shuffle seed |
| `--threads` | 16 | **Always set this.** Because GPU processing is fast, the CPU dataloader is easily the bottleneck; a larger value is recommended. Use your CPU's physical core count as a starting point — a small value (e.g. 1) will cause a large drop in pos/s. Use `NNUE_TRAIN_STEP_PROFILE=1` to see the h2d / fwd / bwd / optimizer breakdown and tune accordingly |
| `--teacher-shuffle-buffer-mib` | 0 | Teacher-data read-ahead and shuffle window size in MiB, per window. The default 0 reads the teacher sequentially without windows (teachers are expected to be pre-shuffled on disk). A positive value enables double-buffered read-ahead plus per-epoch in-window shuffle; `auto` uses 1/16 of the smaller of total RAM and the cgroup limit (including nested cgroup v2 limits), capped at 4096 MiB per window. Two windows are used, so additional window memory (including record metadata) is approximately twice the effective value |
| `--no-teacher-shuffle` | OFF | Disables only the within-window shuffle while retaining double-buffered sequential read-ahead. Use it to separate the effects of I/O read-ahead and reordering. Takes effect only when `--teacher-shuffle-buffer-mib > 0` |
| `--teacher-shuffle-seed` | 0 | Base seed combined with the dataset epoch and window index (takes effect only when `--teacher-shuffle-buffer-mib > 0`). The same value reproduces each window permutation; training delivers decoded batches in input order even with multiple workers |
| `--test-tail-positions` | none | Reserve the last N positions of `--data` as a held-out validation set in the same file (see "Held-out validation" below). Recommended whenever you want held-out validation |
| `--test-positions` | 10000 | Number of positions evaluated each superbatch from the held-out source. Used only with `--test-tail-positions` or `--test-data` |
| `--bucket-mode` (`layerstack`) | progresskpabs | `progresskpabs` routes by the KP-absolute progress estimate. `progresskpabsq16` routes by the same estimate in integer Q16 arithmetic. `kingrank9` matches YaneuraOu KingRank9, requires 9 buckets, and rejects `--progress-coeff` |
| `--num-buckets` (`layerstack`) | 8 (9 for `kingrank9`) | In `progresskpabs` / `progresskpabsq16` mode, an integer in `[1, 64]`; positions route to `min(N-1, floor(progress * N))`. In `kingrank9` mode this must be 9. Pass `--num-buckets 9` to `--resume` / `--init-from` a 9-bucket checkpoint or `.bin` |

### Training position sampling

Sampling is opt-in and applies only to training records; `--test-data` and
`--test-tail-positions` are unaffected. Defaults preserve the original position stream.

| Option | Default | Meaning |
|--------|---------|---------|
| `--keep-rate R` | 1 | Keep probability, finite and in `(0, 1]` |
| `--min-ply N` | none | Discard PSV records with `game_ply <= N`; invalid for HCPE |
| `--ply-keep-curve POINTS` | none | Comma-separated `ply:probability` points, strictly increasing nonnegative plies and probabilities in `[0,1]`; invalid for HCPE |
| `--bucket-keep-rates r0,r1,...` | none | Probabilities in `[0,1]`, one per sampling bucket. Defaults to LayerStack routing buckets |
| `--skip-seed N` | 0 | Fixed seed for stateless position sampling |

For example, `--keep-rate 0.25 --min-ply 18 --ply-keep-curve
"0:0.025,22:0.05,25.5:0.2,29.5:0.8,32:1"` combines random and early-ply skipping.
Curve values are linearly interpolated, with constant values outside the endpoints.
All applicable probabilities are multiplied; score filtering also applies.
Sampling hashes the seed, teacher ID (0 for concatenated `--data`, 1 onward for
`--mix-data` in argument order), zero-based epoch and physical record index. Multiple `--data` files form one stream
in argument order. Decisions are independent of shuffle windows and decode worker
scheduling, including `--threads >= 2`.
Bucket sampling runs in the parallel decode workers and reuses the decoded board
for feature extraction. Rejected records are replaced to keep batches full.
After five consecutive empty passes the loader returns an error.
When any sampling option is specified, each superbatch logs cumulative `read` and
`kept` counts for each teacher and the total, including prefetched records.
Routing-bucket sampling also logs per-bucket reads (decoded candidates surviving reader
filters) and survivors. Marked tree samples in the bucket log count as both read and kept.
The effective sampling configuration
is stored in `experiment.json` under `params.sampling` and participates in the recipe fingerprint.
The all-default configuration is omitted to preserve existing recipe fingerprints (including its implicit seed 0).

#### Choosing a separate sampling bucket

Use `--keep-progress-coeff progress.bin --keep-num-buckets 8` to thin by game
progress while training with king-rank routing.

```bash
nnue-train --data teacher.psv \
  --keep-progress-coeff progress.bin --keep-num-buckets 8 \
  --bucket-keep-rates 1,1,1,1,0.9,0.7,0.5,0.3 \
  layerstack --layerstack-head split-skip --bucket-mode spec --bucket-spec 'ownking:2/3/7+oppking:2/3/7'
```

For spec-based thinning, use `--keep-bucket-spec 'handmajor3'` with three rates.
A spec determines the bucket count. Progress uses eight buckets by default
and accepts one to 64. A separate sampling definition also works with `simple`.
Use the [bucket survey](progress-bin.md#checking-the-bucket-distribution) to see
the input distribution and how it overlaps the routing buckets.

Each mixed teacher inherits the sampling definition unless it has a numeric
`--mix-keep-rate`. Use `--monitor-keep-buckets` to log exact reads and survivors
as `[sampling]` rows labeled `keep_bucket` for teachers using the independent
definition. Without monitoring, the loader skips classifying a position when
the same stateless draw and the lowest or highest keep probability already
determine the outcome. For PSV, this also skips decoding when the draw
rules out every bucket. Exact monitoring classifies every candidate that
passes the reader filters and uses the same draw for the keep decision.
`[teacher-mix]` fractions measure the positions delivered after thinning.
For ordinary positions, the selected bucket
rate multiplies `--keep-rate` and the applicable ply-curve rate after the
`--min-ply` cutoff. Positions marked for `--tree-sample-eval-only` follow
`--tree-sample-accept` and bypass this multiplication. Dual-label PSV uses the
same bucket for a board with either label.

`experiment.json` records the spec or the coefficient filename and bucket count
under `params.sampling.keep_bucket`. Resuming saved input positions checks the
sampling definition and the contents of the coefficient file.

### Mixing teachers

```bash
nnue-train --data main.psv --mix-data tree.psv --mix-rate 0.03 --mix-seed 42 layerstack --layerstack-head split-skip
```

This samples approximately 3% of training positions from `tree.psv` and 97% from
`main.psv`. Add more `--mix-data` / `--mix-rate` pairs in matching order; keep each
repeated option on one side of the subcommand. Rates describe the expected fraction
after score and reader sampling filters, rather than a fixed quota per batch.
Worker-side bucket rejection retries teacher selection. Each teacher's delivered share
is proportional to its selection rate multiplied by the fraction of its reader
candidates accepted at the bucket stage. Marked tree samples and records from
teachers with a numeric `--mix-keep-rate` pass this stage with probability 1, so
bucket rejection on the main teacher can raise the delivered share of these teachers
above their configured rates. Use the delivered counts in the `[teacher-mix]` log
to tune `--mix-rate`.
Each teacher cycles independently. To set score limits for the additional teachers,
repeat `--mix-score-drop-abs N` or
`--mix-score-clamp-abs N` once per `--mix-data` file in the same order. Use
`inherit` in a teacher's position to use the corresponding main threshold, or
omit the option to inherit it for all teachers. Each teacher selects its
score, drops records at or above its absolute drop threshold, then clamps the scores
of the surviving records. If both limits are set for a teacher, the clamp must be
below the drop threshold.
Repeat `--mix-keep-rate F` once per additional teacher in the same order
to use F (from 0 to 1) as the sole sampling probability for its unmarked
positions. Use `inherit` to apply the main teacher's sampling filters to that
teacher, or omit the option to inherit them for all teachers.
Explicit rates are recorded as `keep_rate` in `experiment.json` under `params.mix_data`.
An empty teacher or one with no surviving positions is an error when sampled.
`--test-tail-positions` reserves a tail from the main teacher only; additional teachers
are read in full and must exclude held-out positions themselves.

Teacher selection hashes `--mix-seed` with the mixed-stream position number; it does
not maintain random-generator state. A shuffle window, when enabled, is filled from
the mixed stream and shuffled afterward. Mixed windows cross individual teacher epoch
boundaries. The record count per window is calculated from the actual in-memory
record size, including source metadata, and rounded down to a multiple of the batch
size. Each window stays within the specified MiB budget, including when mixing;
the two windows together use at most twice that budget. Training receives batches in reader order even with multiple decode
workers. With bucket sampling, workers read numbered raw chunks under the input lock,
then decode, extract features and filter outside the lock. Accepted positions are assembled
in chunk order into full batches; leftovers carry into the next batch. Assembly copies
fixed-size sparse indices and labels, without decoding again. Lower keep rates still
require decoding more candidates per delivered position. Without bucket sampling,
each raw chunk remains one batch.

`--dual-label-psv` and `--score-override` (including its optional mask) apply
only to the concatenated `--data` stream. Put `--data` and its score-source
options on the same side of the subcommand. Additional teachers use their ordinary
stored score and never treat padding as a dual-label gate.

For positions without a game result, encode result 0 (draw) and use
`--wdl-ignore-draws` to train on the score only. This also affects genuine draws in every teacher, including the main teacher;
it does not ignore decisive game results. `--wdl 0` trains every position on score only.

Each superbatch logs `[teacher-mix]` counts and observed fractions for the batches
actually passed to training, excluding unread prefetched positions. Source 0 is the
main teacher; sources 1 onward follow `--mix-data` order. `experiment.json` records
ordered teacher basenames, rates and any per-teacher score limits in `params.mix_data`.
The seed is stored in `params.mix_seed`. These fields participate in the recipe fingerprint. With no mixing,
the input sequence and recipe fingerprint remain unchanged. Old checkpoints without `.loader.json`
do not restore teacher reading positions; reproducing an interrupted mixed
stream also requires its output ordinal and each teacher's epoch and next record index.

### Teacher-data read-ahead and shuffle

The default (`--teacher-shuffle-buffer-mib 0`) reads the teacher sequentially without windows,
assuming teachers are pre-shuffled on disk. Passing a positive value or `auto` enables windowing;
`auto` sizes each window to 1/16 of the smaller of total RAM and the cgroup memory limit (including
nested cgroup v2 limits), capped at 4096 MiB per window, so both windows together stay within 1/8
of available memory. With windowing enabled, a producer thread reads teacher records sequentially into
windows of that size. While the CPU dataloader consumes one window, the producer prepares the next. A completed window is reordered
with Fisher–Yates. A partial window at physical EOF is processed without mixing records from the
next dataset epoch, and the next epoch uses a different seed that includes its epoch number. This
avoids repeating exactly the same order when training makes multiple passes over an already
pre-shuffled PSV.

As a guide, machines limited to 8/16/32/64 GiB or more resolve to 512 MiB/1 GiB/2 GiB/4 GiB per
window. Capacity is rounded down to whole batches using the in-memory record size,
including identity metadata, for both PSV and HCPE. The startup `N MiB x2 (auto)` message and `experiment.json` record the resolved value.
The calculation does not use currently available memory, so other processes and page-cache state
do not change the result for a fixed memory limit.

Additional window memory (including record metadata) is approximately twice the effective value; an explicit
`--teacher-shuffle-buffer-mib 4096` adds about 8 GiB. A larger window can absorb longer
HDD stalls, but it also increases the delay before the first batch, CPU time spent shuffling, and
random memory traffic. When windowing is enabled, a small explicit value such as 256 keeps the
first-batch delay, memory, and shuffle cost low. `--threads` controls decode workers; it does not parallelise the
producer's window shuffle.

```bash
# 1 GiB per window (up to about 2 GiB of additional window memory), with a fixed base seed
target/release/nnue-train ... \
  --teacher-shuffle-buffer-mib 1024 --teacher-shuffle-seed 42 \
  layerstack --layerstack-head split-skip ...

# Measure double-buffered I/O without reordering
target/release/nnue-train ... \
  --teacher-shuffle-buffer-mib 256 --no-teacher-shuffle \
  layerstack --layerstack-head split-skip ...

# Memory-sized read-ahead window plus shuffle (opt-in for slow storage)
target/release/nnue-train ... \
  --teacher-shuffle-buffer-mib auto \
  layerstack --layerstack-head split-skip ...
```

Score sidecars, `--score-drop-abs`, and `--score-clamp-abs` are applied in original file order
before records enter the shuffle window. On `--resume`, the adjacent `.loader.json` restores
its epoch, input position, and consumption within the shuffle window. Prefetched but unused
records are excluded. One worker reproduces the uninterrupted position sequence; parallel
decode also consumes a contiguous input range.

Training input (`--data`), `--test-data`, and `loader-digest` accept PSV and HCPE. Rescoring
(`--rescore-input`) requires PSV and rejects files named `.hcpe`. A score sidecar
(`--score-override`) is aligned by record count, so it works with either format. `--dual-label-psv`
reads the second label from PSV-only fields and is rejected for HCPE.
Use teacher data produced by a generator that guarantees valid records and positions. Training
trusts that content: it does not scan the entire file at startup or structurally validate each
decoded PSV position. An HCPE record whose Huffman code or piece counts are invalid stops
training with an error when it is decoded. Keep the input file unchanged throughout training, including across dataset
epochs; write regenerated data to a different file.
PSV rescoring retains its structural checks for king placement and total piece count before
writing scores. These checks do not verify move legality or every possible format corruption.

For training-only WDL lambda jitter, see [Training schedules](training-schedule.md#training-jitter).

`--tree-sample-eval-only` trains PSV positions marked by padding bit 1 (byte 39)
on the score term only (WDL weight 0), regardless of the WDL schedule. It requires
`--data` in PSV format and applies to all teachers. The main teacher permits
padding bits 0 and 1 with this flag, and bit 0 retains its dual-label gate meaning.
Additional teachers also permit bits 0 and 1 but never read bit 0 as a gate. Marked positions are rejected in external
`--test-data` and `--test-tail-positions`.
With this flag, `--tree-sample-accept F` keeps marked positions from any teacher
with probability F (from 0 to 1, default 1). Marked positions bypass
`--keep-rate`, `--min-ply`, `--ply-keep-curve`, `--bucket-keep-rates`, and
`--mix-keep-rate`.

Dropping the training loader cancels both direct and windowed readers at record boundaries,
including while high-score records are skipped and before the next dataset epoch is opened.
Cancellation during normal shutdown is not an experiment failure. A read already blocked
inside the operating system must return before shutdown can finish.

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

## Rescoring a teacher pool (score sidecar)

`--rescore-input` relabels every record of a PSV file with the forward pass of a
trained network and writes a little-endian i16 score sidecar
(`<input name>.scores.i16`, one score per input record, in input order):

```bash
nnue-train   --rescore-input /path/to/pool.psv   --rescore-output /path/to/scores/   --rescore-score-scale 1200   --batch-size 65536   layerstack   --init-from /path/to/net.bin   --ft-out 3072 --l1 16 --l2 32 --num-buckets 9   --bucket-mode progresskpabs --progress-coeff /path/to/progress.bin
```

- `--rescore-score-scale` has no default: pass the `--wrm-nnue2score` value the
  net generation was trained with (`score = net_output * scale`, rounded and
  clipped to `+/- --rescore-score-clip`).
- Weights come from `--init-from` (a quantised `.bin`; labels go through the
  quantise/dequantise round trip) or `--resume` (a raw `.ckpt`; fp32 master
  labels). The sidecar's `.meta.json` records which one was used.
- The run is resumable: an `.in-progress` marker holds a fingerprint of every
  label-affecting condition (net sha256, routing, progress coefficients,
  score conversion, build commit), and a matching interrupted run appends from
  the recorded record count. Any changed condition regenerates the sidecar
  from scratch. Two honest limits: content hashing covers the `--init-from`
  `.bin` and the progress coefficients (hashed from the loaded bytes); the
  `--resume` `.ckpt` and the input PSV are guarded by size + mtime checks
  only, so a same-size replacement with a restored mtime is not detected.
  A binary built from a dirty or unknown git tree disables completion skip
  and resume entirely (the fingerprint gets a per-run nonce) — build from a
  clean checkout for campaign runs.
- Training-side filters (`--score-drop-abs` etc.) and reduced-precision flags
  (`--tf32`, `--ft-fp16`, ...) are rejected: the sidecar covers every record
  and labels always use the strict FP32 forward path.

## Held-out validation

To watch for overfitting and divergence (NaN) without waiting for SPRT
self-play, add held-out validation: positions that are never used for a
gradient update, scored each superbatch and reported as `test_loss` /
`test_acc`. Enable it with `--test-tail-positions` (or `--test-data`) plus
`--test-positions`; see [Held-out validation](held-out-validation.md) for the
flags, how to pick the held-out source, and how to read the metrics.

## Interrupting and resuming training

Quantised `.bin` exports, raw `.ckpt` checkpoints, experiment JSON, and rescore
completion markers are written to a uniquely named temporary file in the
destination directory, flushed and synced, then renamed over the destination.
A failure before the rename keeps the previous complete artifact and removes
the temporary file. The rescore score sidecar itself is still appended in place
and resumed through its in-progress marker.

On Unix the parent directory is synced after the rename. If that sync fails,
the complete new file is already in place, so training continues and a
`[artifact] warning:` line is printed; the rename may not survive a crash or
power loss until the filesystem persists it by itself. Filesystems that do not
support directory sync skip the step without a warning. On Windows no directory
sync is attempted.

Consequences of replacing by rename:

- The output directory must be writable, even when the destination file itself
  already exists and is writable.
- Replacing an existing file needs room for the old and the new file at the
  same time.
- A destination symlink is replaced by a regular file, and other hardlinks keep
  the old contents.
- Unix permission bits of an existing regular file are preserved; ownership and
  ACLs become those of the writing process. New files follow the current umask.
- An existing `.bin` export that cannot be opened for writing (for example after
  `chmod a-w`) is not replaced and the save fails with a permission error.
  `.ckpt`, experiment JSON and completion markers are replaced regardless of
  their own write permission, so protect those through the directory.

A forced kill, Ctrl-C during a save, OOM kill or power loss can leave the
temporary file behind. It is hidden and named after its destination:
`.<destination file name>.<random>.tatara-tmp` (for example
`.rshogi-100.ckpt.Ab3xYz.tatara-tmp`), and can be as large as the artifact.
Leftovers are never read as checkpoints and are not removed automatically,
because another writer may still be using them; delete them once no trainer is
writing to that directory.

Replacement on Windows goes through the standard library rename. It has not
been exercised by automated tests; if a program holding the destination open
blocks the replacement, close it and retry.

A raw `.ckpt` saves the full training state: **weights + optimizer state
(m / v / slow / step) + the current superbatch number**. Format v9 and later
also save `fv_scale`, so it does not need to be respecified. When resuming a v8
or earlier checkpoint, repeat `simple --fv-scale <N>` to preserve an override;
if omitted, the value falls back to the `--scale`-derived value. In every
format, omitting `--fv-scale` requires `--scale = --wrm-nnue2score` because this
validation runs before the checkpoint is read. Even after a power loss or a GPU
error, you can fully resume by adding `--resume` to the same options +
architecture subcommand used for training:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --feature-set halfkp --superbatches <N> --keep-checkpoints 4 \
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
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

### Stopping at a superbatch boundary

To shorten a running job without interrupting a superbatch, write `control.json`
under its output directory:

```json
{"target_superbatches": 1000}
```

The trainer polls this file at the start of each superbatch. The target is
clamped between the superbatch being started and the original `--superbatches`
value, so it cannot extend the run. The selected final superbatch runs to
completion and always writes both `.bin` and `.ckpt`, even when it is not a
`--save-rate` boundary. Training then follows the normal completion path and
marks `experiment.json` as `completed`.

Applied changes are appended to `control_history.jsonl`. A missing control file
is normal and ignored. Read or parse failures produce a warning but do not stop
training. LR and WDL schedules retain their original horizon; only progress and
ETA use the shortened target.

Before resuming in the same output directory, remove `control.json` or replace
it with the new target. A target below the resumed run's starting superbatch is
treated as stale state from the previous run, ignored with a warning, and not
recorded in `control_history.jsonl`.

The default path is `control.json` under `--output`. Set
`--control-file /path/to/control.json` to read only that path instead of the default.
The selected path is logged once at startup. The trainer never rewrites the file.
Write a temporary file and rename it to avoid partial JSON.
Applied targets are also recorded in `experiment.json` as `results.target_superbatches`.

Add `"pause": true` to the same file to pause at the end of the superbatch where
the trainer reads `pause=true` in a file rewritten after startup (changed modification
time or contents).
It saves weights and optimizer state in `.ckpt`, input positions in `.loader.json`,
and inference weights in `.bin`, regardless of `--save-rate`, then exits with code **75**.
Pause is polled at superbatch ends and its application is appended to `control_history.jsonl`.

```json
{"target_superbatches": 1000, "pause": true}
```

Startup `pause=true` is ignored with a warning. Leaving it true does not pause.
To pause again, rewrite the file with `pause=true`; intermediate false values do not
need to be observed. A pause received at the final superbatch is ignored with a log
message, and training completes normally with exit code 0. `experiment.json` marks paused runs interrupted, records applied
pauses in `control_history`, and records the ending superbatch and pause flag in `stopped`.
SIGINT and SIGTERM are not converted into pause requests; use the control file for a
checkpointed stop.

### Pausing and resuming the input stream

Keep each `.ckpt` together with its matching `.loader.json`. Multiple `--data` files form
one teacher in command-line order. Each `--mix-data` teacher retains its own epoch and next
record; the sidecar also restores the mixing selection counter and shuffle window position.
Version 1 sidecars store the number of accepted positions already delivered from
a partially consumed chunk. Resume rereads that chunk and skips those accepted positions;
prefetch cannot advance the saved boundary. Unknown sidecar versions are rejected.
When windows are enabled, the sidecar stores the actual record count per window
(`window_records`); resume rejects a mismatch with the count calculated from the
current settings because it changes the meaning of positions within a window.
Resume rejects changes to any teacher's basename/size, sampling probabilities/seeds,
mixing rates/seed, batch size, training range, score processing, or shuffle settings: it
stops at startup and lists the differing items. Absolute teacher paths are recorded for
reference and may change. When bucket keep rates are enabled, bucket mode/count and
progress coefficient file size/SHA-256 must also match. To resume with changed teachers
or settings, add `--resume-input restart`: weights and optimizer state are restored and
the teachers are read from the beginning. Content changes of the same size are not
detected, so keep teacher files immutable when restoring input positions.
Old checkpoints without a sidecar warn once and restart input from the beginning.
Passing `--start-superbatch` with `--resume` warns and restarts teacher input from
the beginning, without restoring loader positions. Batches per superbatch may change.

Restart with the same arguments plus `--resume <last.ckpt>`, keeping its matching
`.loader.json`. Startup pause is ignored, so leaving it true also allows resuming.

> **Difference between `--resume` and `--init-from`**: `--init-from` injects
> only the weights from a quantised `.bin` and **resets** the optimizer state
> (fine-tuning / continued training); `--resume` restores both weights and
> optimizer from a raw `.ckpt` (a true resume). The two are mutually exclusive.

## Reading the output artifacts

After training, what appears under `checkpoints/<run-name>/`:

| File | Format | Purpose |
|---|---|---|
| `<run-name>-<sb>.bin` | quantised NNUE binary | **The artifact to feed to the inference engine** (for the binary layout, see `crates/nnue-format`) |
| `<run-name>-<sb>.loader.json` | Teacher identity and input position | Keep with `.ckpt`; pruned together by `--keep-checkpoints` |
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
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `kernel artifact nnue_train.{cubin,ptx,ll} not found` | Opt-in `oxide` builds only: on the first build you need to generate the `.ll` with `cd bins/nnue_train && cargo-oxide build`. For details, see [docs/setup.md](setup.md) |
| `libcublas.so` link / load errors | The CUDA Toolkit is in none of `/usr/local/cuda` / `CUDA_HOME` / `CUDA_PATH`. Specify it explicitly with `CUDA_TOOLKIT_PATH=/path/to/cuda-12.x` (both build.rs and runtime resolve via the same chain) |
| `CUDA_ERROR_INVALID_PTX` (driver error 218) | The embedded PTX does not match the GPU. The default build detects the target with `nvidia-smi`, but the detected value is not tracked by Cargo — after swapping the GPU, rebuild with `cargo clean -p cuda-native-runtime` first. On opt-in `oxide` builds with a sub-Ampere GPU (sm_75), export `CUDA_OXIDE_TARGET=sm_75`, then rebuild and rerun |
| pos/s extremely low (< 500K on an RTX 3080 Ti) | Increase `--threads` (start from your physical core count, see "Key options") and check whether the dataloader's prefetch is keeping up. `NNUE_TRAIN_STEP_PROFILE=1` prints the ms spent in each phase (h2d / fwd / bwd / optimizer) to stderr so you can see the breakdown |
| rejected with `--batch-size must be a multiple of 16` | The tiled dense matmul kernels require `b % 16 == 0`, so the CLI rejects other values at startup. Pass a multiple of 16 (the default 16384 satisfies the condition) |

## Related

- [docs/setup.md](setup.md) — toolchain (CUDA / NVCC, opt-in cuda-oxide) setup
- [Game-progress buckets: preparing `progress.bin`](progress-bin.md) — training and surveying the LayerStack bucket coefficients
- [Held-out validation](held-out-validation.md) — `test_loss` / `test_acc` setup and how to read the metrics
- [Training schedules](training-schedule.md) — learning-rate and WDL lambda scheduling
- [Tuning the WRM loss](wrm-loss-tuning.md) — the WRM transform and its tuning options (win-rate transform + generalized loss form)
