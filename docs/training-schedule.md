**English** | [日本語](training-schedule.ja.md)

# Training schedules

A run has two independent schedulers: the **learning rate** (`--lr-schedule`) and
the **WDL lambda** (`--wdl` / `--start-wdl` / `--end-wdl` /
`--wdl-cycle-delta`). Both are recomputed
each run from the CLI arguments and restored checkpoint state. For the
per-run training steps themselves see
[docs/training-quickstart.md](training-quickstart.md); for the exact flag syntax,
ranges, and defaults run `nnue-train --help` (the help text is the authoritative
per-flag reference). This document covers the cross-flag behaviour the help text
can't: what each scheduler does and how it interacts with `--resume`.

## QAT warmup (`--qat-warmup-steps`)

LayerStack `--qat dense --qat-warmup-steps N` disables QAT for the first N
optimizer updates, then switches it on. The counter is cumulative across resume
and independent of the LR schedule. An omitted value inherits the checkpoint;
an explicit mismatch is rejected. Older checkpoints have no warmup.
Validation uses the current QAT state; integer export always quantizes the masters.
See [Dense QAT](dense-qat.md#warmup) for scope and forward-only resume behavior.

## Learning rate schedule (`--lr-schedule`)

`--lr-schedule` selects how the learning rate moves across the run. The options:

- `step` (default) — multiply the LR by `--lr-gamma` every `--lr-step`
  superbatches.
- `constant` — hold `--lr` for the whole run.
- `drop` — one-shot multiply by `--lr-gamma` after `--lr-step` superbatches.
- `linear` / `cosine` / `exponential` — decay from `--lr` to `--lr-final` by
  `--lr-final-superbatch`, then hold.
- `one-cycle` — warm up over the first `--lr-warmup-pct` of the horizon, then
  cosine-anneal down.

With `--lr-schedule one-cycle --lr-total-steps N`, the cycle advances once per
optimizer update, independently of `--superbatches`. N must be positive; other
schedules reject this option. The initial LR is `--lr / --lr-div-factor`, and the
final LR is initial LR divided by `--lr-final-div-factor`. Warmup ends at
a floating-point phase boundary at `--lr-warmup-pct * N - 1`, without rounding.
The zero-based update index follows PyTorch's half-cosine phases: warmup ends
at that boundary and cooldown ends at `N - 1`. The warmup percentage is stored
as f32 and converted to f64 for the boundary calculation. With a nonintegral
boundary, no update necessarily lands exactly on max LR. With warmup percentage
1.0, update `N - 1` reaches max LR. Updates `N` and later hold the value of update
`N - 1`. Zero warmup uses cooldown directly; a zero-length warmup phase at
update 0 uses max LR.

The checkpoint saves N and the completed optimizer update count. On resume, an
omitted N inherits the saved value; a different N or schedule is rejected. A
checkpoint without an update-based horizon cannot switch to this mode on resume.
Changing batch size or batches per superbatch preserves the saved update count
and LR curve. Keep the other one-cycle options unchanged on resume. Omitting
`--lr-total-steps` on a fresh run retains superbatch scheduling.

`--lr-warmup-steps` additionally wraps any schedule except `one-cycle` (which
carries its own warmup) in a batch-level warmup over the first superbatch.

`--l1-lr-mult` multiplies the scheduled rate for the first dense layer’s weights,
including shared weights in both LayerStacks heads. When omitted, L1 uses
`--dense-lr-mult` in LayerStacks and 1.0 in simple. L1 uses the dense weight
decay coefficient in LayerStacks (except an explicit shared-L1 override) and the global coefficient in simple.
On resume, the multiplier comes from the current command line and is part of
the recipe comparison.

`--l1-shared-weight-decay` overrides decay only for the shared L1 weights in
LayerStack, including both heads. Bucket-specific L1 weights, other layers and
biases keep their existing coefficients. The shared weights keep the L1 learning
rate from `--l1-lr-mult` (or the dense multiplier when omitted). The option must
be finite and nonnegative; simple rejects it. Omission preserves dense decay.

`--adam-bias-correction` requires `--optimizer adamw`. It enables
`m_hat = m / (1 - beta1^t)` and `v_hat = v / (1 - beta2^t)`, with update
`w = w * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + epsilon)` followed by
tatara's existing weight clamp. Omission preserves AdamW without correction.
Both native CUDA (including fused FT) and cuda-oxide use this correction, also
with FP16 optimizer states; FP16 storage retains its rounding error.
The checkpoint records `--adam-bias-correction`; resume requires the same flag
value. Checkpoints without this field imply correction disabled. Epsilon
and decay are rescaled along with the gradient step so the shared optimizer
kernel preserves the effective epsilon and decoupled decay rate.

Supply these options again on resume: the checkpoint restores weights, moments
and the accumulated optimizer step `t`, while the current CLI selects the
options. Explicit values appear in startup logs, experiment parameters and the
recipe fingerprint; omitted options add no parameter fields.

In LayerStacks training, `--ft-piece-lr-mult` scales the learning rate for the
real piece rows across all effect buckets, and `--threat-lr-mult` scales it for
real Threat rows, including compact Threat rows. The Threat option requires
`--threat-profile`. Both default to 1.0 and multiply the scheduled rate after
`--ft-lr-mult`. Virtual rows and PSQT weights keep the FT group rate. Each row
uses the FT weight decay coefficient, so decoupled decay follows its optimizer
rate. Norm loss uses the scheduled rate. On resume, supply the multipliers on
the command line.

Different rates within the FT weights require the native CUDA fused optimizer.
For that combination, disable PSQT and precision diagnostics and use a multiple
of 8 for any compact Threat width.

### Changing batch size on resume

You can change `--batch-size` and `--batches-per-superbatch` together while
keeping their product constant. Keep the checkpoint and its `.loader.json`
together. Accepted-record offsets, including partially consumed sampling
chunks, are restored with the new batch size. Teacher identities and input
settings must still match. If changing batch size changes the effective
shuffle window size (`window_records`), input restoration is rejected: use a
compatible batch size, or explicitly choose `--resume-input restart` to read
from the beginning.

Learning-rate schedules use superbatch numbers unless one-cycle has `--lr-total-steps`.
The saved horizon is restored, so changing the batch count does not compress
the curve. Keep the other schedule options unchanged; they come from the CLI.
Optimizer moments and step count continue from the checkpoint; smaller batches
produce more optimizer steps per superbatch. Batch-level warmup and batch WDL
jitter remain measured in batches, so their duration in positions can change.

### Horizon and resuming

The schedules with a **horizon** — the superbatch at which the curve reaches its
terminal LR — are the `linear` / `cosine` / `exponential` decays (their
`--lr-final-superbatch`) and `one-cycle` without `--lr-total-steps`. When `--lr-final-superbatch`
is omitted the horizon defaults to `--superbatches`, so a stateless rebuild would
stretch or shrink the curve whenever you resume with a different `--superbatches`.

To keep the curve reproducible, a checkpoint records the resolved horizon and
`--resume` restores it. On resume the horizon is taken from, in priority order:

1. an explicit `--lr-final-superbatch` (decay schedules) — always wins;
2. the horizon saved in the checkpoint;
3. `--superbatches` — the fallback.

Without `--lr-total-steps`, one-cycle uses a superbatch horizon; its saved horizon always
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

The `... simple` invocations below abbreviate a full command; on the simple
trainer that always includes the required `--win-rate-model` flags (see
[the quickstart](training-quickstart.md#example-1-training-a-halfkp-nnue-simple-architecture)).

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

### Cosine cycle (`--wdl-cycle-delta`)

`--wdl-cycle-delta <d>` uses `--wdl` as the base and applies a half-cosine
warmup to `base + d`, followed by a half-cosine cooldown to the base. The
warmup occupies `--wdl-cycle-warmup-pct` of the horizon (default `0.25`). The
delta may be negative, provided `--wdl` plus the delta remains in `[0, 1]`. Set
`--wdl-schedule-superbatch` to make the horizon explicit; otherwise it is
`--superbatches`. The schedule is a function of the superbatch index, so resume
continues the same curve as long as the horizon inputs are unchanged; unlike
the learning-rate horizon it is not stored in the checkpoint, so a run that may
be extended with a larger `--superbatches` on resume should pass
`--wdl-schedule-superbatch` explicitly to pin the curve. A single-superbatch
horizon leaves no room for the cycle, so lambda stays at the base. With
`--wdl-cycle-warmup-pct 1.0` there is no cooldown: lambda returns to the base
at the horizon instead of easing back.

### Training jitter

The training loss can perturb the scheduled WDL weight per position:

```text
lambda_i = clamp(base + batch_jitter + sample_jitter_i, 0, 1)
state = D * state + batch_STD * sqrt(1 - D^2) * N(0, 1)
```

- `--wdl-jitter-sample STD` adds independent Gaussian noise per position.
- `--wdl-jitter-batch STD` adds the shared batch state. Its stationary standard deviation is STD.
- `--wdl-jitter-batch-decay D` sets the correlation in `[0, 1)`. A decay of 0 gives independent batches, and a nonzero decay requires positive batch STD.
- `--wdl-jitter-cycle` requires `--wdl-cycle-delta` and multiplies both noise terms by the cycle's cosine coefficient.
- `--wdl-jitter-seed N` selects the deterministic noise sequence.

Both standard deviations must be finite and nonnegative. The numeric flags default to 0 and cycle scaling defaults to off. Setting the seed or enabling cycle scaling requires at least one positive standard deviation. Score-only rows and draws covered by `--wdl-ignore-draws` still get a WDL weight of 0.

The CPU reference, cuda-oxide, and native CUDA use the same integer hash and uniform conversion. The Gaussian samples can differ slightly due to rounding in the Box–Muller logarithm and cosine.

The training formula corresponds to Stockfish nnue-pytorch's lambda jitter, with tatara's lambda weighting WDL and Stockfish's weighting the score. Stockfish also applies jitter during evaluation. tatara uses the unperturbed schedule or `--validation-wdl` for validation and eval-only runs so validation losses can be compared across runs.

tatara computes the cycle coefficient per superbatch and sets it to zero at and beyond the schedule horizon, muting jitter. Stockfish's `lambda_cycle_jitter` uses a per-step coefficient that stays at 1 at and beyond the horizon when `lambda_cycle_warmup_pct = 1.0`.

Keep the checkpoint and its `.loader.json` together to resume correlated batch jitter. `--resume-input restart` restarts teacher input and still restores batch jitter state when it matches the checkpoint. If the batch jitter state is missing or was saved at a different optimizer step, training warns and restarts it from zero.

### Draws and validation

`--wdl-ignore-draws` makes positions whose WDL label is `0.5` use lambda zero,
so their target is derived from the teacher score. `--validation-wdl <value>`
fixes lambda for held-out validation and eval-only runs; without it, validation
uses the training schedule.

`--tree-sample-eval-only` trains PSV positions marked by padding bit 1 (byte 39)
on the score term only (WDL weight 0), regardless of the WDL schedule. It requires
`--data` in PSV format. Bit 0 retains its independent dual-label gate meaning.
Records with padding bits other than bit 0 and bit 1 are rejected. Marked positions
are rejected in external `--test-data` and `--test-tail-positions`.
With this flag, `--tree-sample-accept F` keeps marked positions from any teacher
with probability F (from 0 to 1, default 1). Marked positions bypass
`--keep-rate`, `--min-ply`, `--ply-keep-curve`, `--bucket-keep-rates`, and
`--mix-keep-rate`.

## Where the values are recorded

The effective schedules are written to the run's `experiment.json` under
`params`. `lr_schedule` holds the resolved LR schedule string, including the
effective horizon (so a resumed run shows the restored horizon, not the
`--superbatches`-derived default). The `wdl` field is always present (the
constant lambda); a linear taper additionally records `start_wdl` / `end_wdl`
(omitted otherwise), and in that case the scheduler derives `lambda` from those
endpoints and ignores `wdl`. Unless `--validation-wdl` is set, `test_loss` uses the
unperturbed WDL schedule at each superbatch (see
[Held-out validation](held-out-validation.md)).
Cycle runs also record `wdl_cycle_delta`, `wdl_cycle_warmup_pct`, and
`wdl_schedule_superbatch` (the latter only when explicitly set).
`validation_wdl` is recorded only when specified; `wdl_ignore_draws` and
`tree_sample_eval_only` are recorded only when true.
When at least one jitter standard deviation is positive, the settings are recorded under `wdl_jitter`.
