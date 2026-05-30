**English** | [日本語](training-schedule.ja.md)

# Training schedules

How to schedule the **WDL lambda** across a training run. For the per-run
training steps themselves, see
[docs/training-quickstart.md](training-quickstart.md); for the exact flag
syntax, ranges, and defaults, run `nnue-train --help` (the help text is the
authoritative per-flag reference).

## What the WDL lambda controls

Every position has two targets the net can be trained against:

- the **teacher score** (the engine evaluation in centipawns, passed through the
  loss's sigmoid / win-rate transform), and
- the **game result** (WDL: `0.0` loss / `0.5` draw / `1.0` win).

The loss blends them with a single scalar `lambda`:

```
target = lambda * (game result) + (1 - lambda) * (teacher score)
```

So `lambda = 0` trains purely on the teacher evaluation, and `lambda = 1` trains
purely on the game outcome. The blend is identical for the plain sigmoid-MSE
loss and the win-rate-model loss (`--win-rate-model`); only the teacher-score
side of the formula differs between them.

`lambda` is in `[0.0, 1.0]`.

## Constant lambda (`--wdl`)

`--wdl <value>` holds `lambda` fixed for the whole run. The default is `0.0`
(train purely on the teacher score). Raise it to always mix in some weight on
the game result.

```bash
target/release/nnue-train --data <psv> --wdl 0.3 ... simple
```

## Linear taper (`--start-wdl` / `--end-wdl`)

`--start-wdl <a> --end-wdl <b>` interpolates `lambda` linearly across the run:
it is `a` at the first superbatch and `b` at the last (`--superbatches`),
moving by an equal step each superbatch in between. On `--resume`, the taper
continues from the resumed superbatch rather than restarting.

- Both flags must be given together; either one alone is an error.
- They are mutually exclusive with `--wdl` (passing both is rejected at parse
  time).

The common use is a curriculum that starts evaluation-heavy and ends
result-heavy — train early on the dense teacher score for a stable signal, then
shift weight onto the sparser game outcome:

```bash
target/release/nnue-train --data <psv> --start-wdl 0.0 --end-wdl 0.5 ... simple
```

This linear scheduling of the blend follows nnue-pytorch's
`start_lambda → end_lambda` taper
([model/lambda_utils.py](https://github.com/official-stockfish/nnue-pytorch/blob/e215624/model/lambda_utils.py)).

A single-superbatch run (`--superbatches 1`) has no interval to interpolate
over, so the taper collapses to `--start-wdl`.

## Where the value is recorded

The effective schedule is written to the run's `experiment.json` under
`params`: `wdl` for the constant case, and `start_wdl` / `end_wdl` for a linear
taper (omitted when unused). `test_loss` is computed with the same `lambda` as
`train_loss` at each superbatch, so the two stay on one scale (see "Reading the
metrics" in the [quickstart](training-quickstart.md)).
