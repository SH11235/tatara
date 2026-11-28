# Dense QAT continuation

Use `layerstack --qat dense` to continue an existing raw checkpoint with dense
quantization. `--qat off` runs ordinary continuation. Neither changes the NNUE
architecture, integer export format, optimizer history, or FT precision settings.
Both 1536×16×32 and 3072×16×32 use the same mode.

The option defaults to off for fresh training and raw checkpoints through version 9.
Version 10 raw checkpoints record the effective mode. On resume, omitting `--qat`
inherits it; explicitly supplying `dense` or `off` overrides it. Experiment
`params.qat` records the effective mode and the command records the explicit choice.
The source checkpoint remains unchanged. Older trainers reject version 10 raw files.

For a source at superbatch 1200, choose a small endpoint such as 1202 on **both**
branches. Use the same complete architecture, feature, teacher, optimizer, precision
and learning-rate arguments as the source. For example, append these choices to
the existing command, keeping global options before `layerstack`:

```text
--resume /path/to/source.ckpt --output /path/to/control --superbatches 1202 ... layerstack --ft-out 1536 --l1 16 --l2 32 --qat off
--resume /path/to/source.ckpt --output /path/to/qat     --superbatches 1202 ... layerstack --ft-out 1536 --l1 16 --l2 32 --qat dense
```

`--superbatches` is the absolute endpoint, not the number of added superbatches.
Use matching explicit learning-rate settings and, for schedules with a horizon,
`--lr-final-superbatch`. Do not change precision flags along with QAT when testing
the QAT effect. Resume does not restore an exact data cursor; equal source and loader
settings do not promise identical input order across runs.

## Warmup

For fresh LayerStack training, add `--qat dense --qat-warmup-steps N`.
The first N optimizer updates use the same forward/backward path as `--qat off`;
once the number of completed updates reaches N, the next update uses dense QAT.
This is an abrupt switch, like nnue-pytorch's `fake_quant_warmup_steps`, independent
of the learning-rate schedule. Zero enables QAT immediately. The option requires
dense QAT and is unavailable for Simple.

Warmup gates dense weight/bias quantization, quantized activation/output forward,
and the corresponding STE backward. FT precision settings and optimizer weight
clamps stay unchanged: those clamps also apply with `--qat off`.

Held-out loss and accuracy use the QAT state at the current completed-update
count without advancing it. For example, validation after update N uses dense
QAT. Integer `.bin` export always quantizes the master weights, including during
warmup, so its predictions can differ from the floating-point validation path.
Raw checkpoints preserve the masters, optimizer update count, and warmup setting.

On training resume, an omitted warmup option inherits the saved value; an explicit
different value is rejected. A checkpoint with QAT warmup cannot resume training with `--qat off`.
Checkpoints without the field mean no warmup and
cannot acquire one through ordinary resume. Warmup checkpoints use raw format 18;
older readers reject them. Without this option, the writer retains the existing
format selection. Eval-only and rescore resume skip warmup-setting comparison;
eval-only uses the saved count and warmup with the effective QAT mode. Rescore
still requires `--qat off`. Forward-only resume with `--qat off` omits the warmup
setting from startup logs, params, and the recipe fingerprint.

When present, the effective setting is printed at startup and recorded as
`params.qat_warmup_steps`, which participates in the recipe fingerprint. Without
a requested or inherited setting, the field and warmup log are omitted.

## Scope and arithmetic

This is **partial dense QAT** on the native CUDA backend (Linux or Windows).
Cuda-oxide, oxide-parity, rescore, PSQT, and optional L2/L3 shared-delta are rejected when
QAT is enabled. Ordinary L1 shared weights are supported.

- FT weights, factorizer folding, accumulator and pairwise calculation retain the
  existing floating-point training path. Its combined output is floored onto the
  1/127 grid before dense layers. This does not emulate integer FT accumulation.
- L1 bucket and shared parameters are summed in f32 before one quantization, as
  in export. Dense weights use f64 scaling, half-away rounding and i8 clipping
  to [-128,127] at scale 64. Biases use scale 8128 and i32 clipping.
- Dense forward uses integer products and sums. L1 square activation uses
  `clamp((i64(z)*z)>>19,0,127)` including negative inputs; ordinary activations use
  `clamp(z>>6,0,127)`. Outputs are dequantized for the existing loss.
- The un-clipped L1 skip is added in integer units. Final engine division by
  FV_SCALE, which truncates toward zero, is **outside** the training forward.
  Supply the same FV_SCALE to the consuming engine for both exported networks.
- Rounding uses STE; activation clipping and the square derivative follow the
  continuous training expressions. Dense weight quantization, including clipping,
  uses an identity surrogate. Input gradients use quantized weights; shared and
  bucket components both receive the effective L1 gradient.

FP32 master weights are restored before optimizer updates and on forward/backward
errors. Raw saves and integer export always read the masters. QAT uses additional
dense scratch buffers and integer kernels; training throughput is not assumed to
match ordinary continuation. Its effect on playing strength requires a separate
evaluation.
