**English** | [日本語](wrm-loss-tuning.ja.md)

# Tuning the WRM loss

Enabling `--win-rate-model` switches to the WRM (win-rate-model) loss, which
converts both the teacher score and the net output to a win-rate and minimises
the squared error between them. For *why* you would use WRM (the net output
converges to a `cp / 600` scale that matches the quantisation format, etc.) see
`--win-rate-model` in the [training Quickstart](training-quickstart.md).

This page explains the WRM transform and the 5 CLI options that tune it. All of
them take effect only when `--win-rate-model` is set. The defaults work as-is,
so you only need to change them when adapting the loss to your score
distribution.

## The WRM transform

Let `sigmoid(x) = 1 / (1 + e^(-x))`. Per position, the prediction (net output)
and the target (teacher score, in centipawns) are converted to a win-rate
separately:

```text
# prediction side (net output)
scorenet = net_output * nnue2score
q   = sigmoid((scorenet  - in_offset) / in_scaling)
qm  = sigmoid((-scorenet - in_offset) / in_scaling)
qf  = 0.5 * (1 + q - qm)

# target side (teacher score)
pt         = (score  - target_offset) / target_scaling
pmt        = (-score - target_offset) / target_scaling
target_wrm = 0.5 * (1 + sigmoid(pt) - sigmoid(pmt))
target     = lambda * wdl + (1 - lambda) * target_wrm   # lambda is --wdl (default 0)

loss = mean((qf - target)^2)
```

`q` / `qm` model "win" and "loss" as one-sided sigmoids; their symmetric
difference becomes the final win-rate `qf`. The `offset` is the centre of that
one-sided sigmoid (the score at which the one-sided win-rate is 0.5), and
`scaling` is the input scale (the inverse of the slope — smaller means the
win-rate reacts more sharply to the score). The prediction side and the target
side take independent offset / scaling values.

## The 5 options

| Option | Default | Side | Role |
|---|---:|---|---|
| `--wrm-nnue2score` | 600 | shared | Factor that maps the net output back to a centipawn scale (`scorenet = net_output * this`). The net output converges to `cp / nnue2score` |
| `--wrm-in-scaling` | 340 | prediction | Input scale (inverse slope) of the prediction one-sided win-rate sigmoid |
| `--wrm-in-offset` | 270 | prediction | Centre offset of the prediction one-sided win-rate sigmoid (one-sided win-rate is 0.5 at `scorenet ==` this) |
| `--wrm-target-offset` | 270 | target | Centre offset of the target one-sided win-rate sigmoid |
| `--wrm-target-scaling` | 380 | target | Input scale of the target one-sided win-rate sigmoid |

`--wdl` (the `lambda` above) blends the target between the WRM win-rate and the
WDL label ({0, 0.5, 1}). At the default 0 the target is `target_wrm` only; at 1
it is pure WDL.

## Defaults and retuning

The defaults (offset 270 / target scaling 380 / in scaling 340 / nnue2score 600)
are tuned for the chess centipawn distribution. If your shogi score distribution
differs, the win-rate transform may saturate the score too much or too little,
so consider retuning. The prediction side (`in_*`) and target side (`target_*`)
are independent, so you can fit the teacher's win-rate curve (target) and the
win-rate curve the net should learn (prediction) separately.

Whether a retune helps cannot be judged from the loss value alone — validate it
by comparing playing strength with SPRT self-play.

The WRM values actually used are recorded in `experiment.json`
(`wrm_in_scaling` / `wrm_in_offset` / `wrm_nnue2score` / `wrm_target_offset` /
`wrm_target_scaling`, only when `loss_kind` is `"wrm"`).

## Related

- [training Quickstart](training-quickstart.md) — the main options, including `--win-rate-model`
- [experiment.json schema](decisions/2026-05-17-experiment-json.md) — how the WRM parameters are recorded
