**English** | [日本語](wrm-loss-tuning.ja.md)

# Tuning the WRM loss

Enabling `--win-rate-model` switches to the WRM (win-rate-model) loss, which
converts both the teacher score and the net output to a win-rate and minimises
the squared error between them. For *why* you would use WRM (the net output
converges to a `cp / 600` scale that matches the quantisation format, etc.) see
`--win-rate-model` in the [training Quickstart](training-quickstart.md).

This page explains the WRM transform and the CLI options that tune it: the
win-rate transform geometry (5 options) and the generalized loss form (4
options, following nnue-pytorch). All of them take effect only when
`--win-rate-model` is set. The defaults work as-is, so you only need to change
them when adapting the loss to your score distribution.

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

loss = mean((qf - target)^2)   # default; generalized by the options below
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

## Generalized loss form (pow_exp / asymmetry / weight boost)

By default the WRM loss is the plain squared error `mean((qf - target)^2)`. Four
more options generalize the error term, following nnue-pytorch's loss. At their
defaults the loss is bit-identical to that squared error, so leaving them unset
changes nothing.

Per position, with `pf = target_wrm` (the score-based win-rate before the WDL
blend):

```text
err    = qf - target
weight = 1 + (2^w1 - 1) * ((pf - 0.5)^2 * pf * (1 - pf))^w2
asym   = (qf > target) ? 1 + qp_asymmetry : 1     # penalize overprediction only
loss_i = asym * |err|^pow_exp * weight
loss   = sum(loss_i) / sum(weight)                # normalized by the weight sum
```

| Option | Default | Role |
|---|---:|---|
| `--loss-pow-exp` | 2.0 | Exponent of the error term `\|qf - target\|^pow_exp`. 2.0 is squared error; nnue-pytorch uses 2.5. Must be >= 1 |
| `--loss-qp-asymmetry` | 0.0 | Extra penalty when the prediction exceeds the target (`qf > target`): that position's loss is multiplied by `1 + this`. 0 is symmetric |
| `--loss-weight-boost-w1` | 0.0 | Strength of the per-position weight boost. `0` gives uniform weight 1 (no boost); larger emphasizes decisive positions (`pf` near 0/1). Must be >= 0 |
| `--loss-weight-boost-w2` | 0.5 | Exponent in the weight formula (curve shape). No effect when `w1` is 0. Must be >= 0 |

When any of `pow_exp` / `qp_asymmetry` / `w1` is non-default, the total loss is
normalized by the sum of the per-position weights (`Σ weight`), matching
nnue-pytorch. `w1` and `w2` are constrained to `>= 0` so every weight is `>= 1`
and the normalization is well-defined. The option names `w1` / `w2` mirror
nnue-pytorch so its recommended values carry over.

## Defaults and retuning

The defaults (offset 270 / target scaling 380 / in scaling 340 / nnue2score 600)
are tuned for the chess centipawn distribution. If your shogi score distribution
differs, the win-rate transform may saturate the score too much or too little,
so consider retuning. The prediction side (`in_*`) and target side (`target_*`)
are independent, so you can fit the teacher's win-rate curve (target) and the
win-rate curve the net should learn (prediction) separately.

Whether a retune helps cannot be judged from the loss value alone — validate it
by comparing playing strength with SPRT self-play.

The WRM values actually used are recorded in `experiment.json` (only when
`loss_kind` is `"wrm"`): the transform parameters `wrm_in_scaling` /
`wrm_in_offset` / `wrm_nnue2score` / `wrm_target_offset` / `wrm_target_scaling`,
and the generalized-loss parameters `wrm_pow_exp` / `wrm_qp_asymmetry` /
`wrm_weight_boost_w1` / `wrm_weight_boost_w2`.

Enabling the generalized loss changes the gradient, not just the reported loss
value, so its effect on playing strength must likewise be validated with SPRT.

## Related

- [training Quickstart](training-quickstart.md) — the main options, including `--win-rate-model`
- [experiment.json schema](decisions/2026-05-17-experiment-json.md) — how the WRM parameters are recorded
