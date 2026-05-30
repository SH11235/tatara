**English** | [日本語](training-schedule.ja.md)

# Training schedules

A run has two independent schedulers: the **learning rate** (`--lr-schedule`) and
the **WDL lambda** (`--wdl` / `--start-wdl` / `--end-wdl`). Both are recomputed
each run as a function of the CLI arguments and the superbatch index. For the
per-run training steps themselves see
[docs/training-quickstart.md](training-quickstart.md); for the exact flag syntax,
ranges, and defaults run `nnue-train --help` (the help text is the authoritative
per-flag reference). This document covers the cross-flag behaviour the help text
can't: what each scheduler does and how it interacts with `--resume`.

## Learning rate schedule (`--lr-schedule`)

`--lr-schedule` selects how the learning rate moves across the run. The default
`step` is bit-identical to the historical `StepLR`. The options:

- `step` (default) — multiply the LR by `--lr-gamma` every `--lr-step`
  superbatches.
- `constant` — hold `--lr` for the whole run.
- `drop` — one-shot multiply by `--lr-gamma` after `--lr-step` superbatches.
- `linear` / `cosine` / `exponential` — decay from `--lr` to `--lr-final` by
  `--lr-final-superbatch`, then hold.
- `one-cycle` — warm up over the first `--lr-warmup-pct` of the horizon, then
  cosine-anneal down.

`--lr-warmup-steps` additionally wraps any schedule except `one-cycle` (which
carries its own warmup) in a batch-level warmup over the first superbatch.

### Horizon and resuming

The schedules with a **horizon** — the superbatch at which the curve reaches its
terminal LR — are the `linear` / `cosine` / `exponential` decays (their
`--lr-final-superbatch`) and `one-cycle` (its total). When `--lr-final-superbatch`
is omitted the horizon defaults to `--superbatches`, so a stateless rebuild would
stretch or shrink the curve whenever you resume with a different `--superbatches`.

To keep the curve reproducible, a checkpoint records the resolved horizon and
`--resume` restores it. On resume the horizon is taken from, in priority order:

1. an explicit `--lr-final-superbatch` (decay schedules) — always wins;
2. the horizon saved in the checkpoint;
3. `--superbatches` — the fallback.

`one-cycle` has no explicit horizon flag, so on resume its saved horizon always
wins over `--superbatches`. Checkpoints written before this was recorded (or by
`step` / `constant` / `drop`, which have no horizon) carry none and fall back to
`--superbatches`. `step` therefore reproduces the same curve on resume regardless
of `--superbatches`, since it has no horizon to begin with.

## WDL lambda schedule

The WDL lambda controls what target the net is trained against. Every position
has two:

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
side of the formula differs between them. `lambda` is in `[0.0, 1.0]`.

### Constant lambda (`--wdl`)

`--wdl <value>` holds `lambda` fixed for the whole run. The default is `0.0`
(train purely on the teacher score). Raise it to always mix in some weight on
the game result.

```bash
target/release/nnue-train --data <psv> --wdl 0.3 ... simple
```

### Linear taper (`--start-wdl` / `--end-wdl`)

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

## Where the values are recorded

The effective schedules are written to the run's `experiment.json` under
`params`. `lr_schedule` holds the resolved LR schedule string, including the
effective horizon (so a resumed run shows the restored horizon, not the
`--superbatches`-derived default). The `wdl` field is always present (the
constant lambda); a linear taper additionally records `start_wdl` / `end_wdl`
(omitted otherwise), and in that case the scheduler derives `lambda` from those
endpoints and ignores `wdl`. `test_loss` is computed with the same `lambda` as
`train_loss` at each superbatch, so the two stay on one scale (see "Reading the
metrics" in [Held-out validation](held-out-validation.md)).
