**English** | [日本語](held-out-validation.ja.md)

# Held-out validation

To watch for overfitting and divergence (NaN) without waiting for SPRT
self-play, add held-out validation: positions that are never used for a
gradient update. Each superbatch ends with a forward-only pass over them; the
training log prints `test_loss` / `test_acc` (compact field name for
console), and `experiment.json` records the same numbers as `test_loss` /
`test_accuracy`. This is opt-in; for the overall training flow see
[docs/training-quickstart.md](training-quickstart.md).

## Three related flags

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

## Which source to pick

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

## Example

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

## Reading the metrics

`test_loss` uses the same loss kernel (sigmoid-MSE or WRM) and the same WDL
lambda blend as `train_loss`, so the two are on the same scale and can be
compared directly within a superbatch. For how to set that blend — a constant
`--wdl` or a linear `--start-wdl` / `--end-wdl` taper — see
[Training schedules](training-schedule.md). A widening `test_loss − train_loss`
gap signals overfitting; an early `test_loss` divergence often catches NaN
issues before `train_loss` looks abnormal.

`test_acc` / `test_accuracy` is the sign-agreement between net output and the
game result (draws excluded from the denominator). It is scale-invariant, so
it can be compared across runs and configurations that have different loss
scales.
