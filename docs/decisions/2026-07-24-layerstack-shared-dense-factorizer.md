# LayerStack shared dense factorizer

## Decision

LayerStack keeps its existing shared L1 term and can opt in to shared L2 and L3
terms with `--stack-factorize-all`. Each enabled layer is trained as:

`effective_bucket_parameter = bucket_parameter + shared_parameter`

The shared L2/L3 parameters start at zero. Forward and input-backward use a
folded buffer, while parameter gradients are reduced across buckets for the
shared optimizer group. Quantized export writes only the folded per-bucket
parameters, preserving the existing LayerStack file format and engine ABI.

## Consequences

- The control path is unchanged when the flag is absent.
- Enabling the flag is step-zero equivalent to the control initialization.
- The implementation works for every supported bucket count and is not tied to
  KingRank9; Progress8 uses the same runtime bucket count.
- Raw checkpoints append the four shared parameter groups and therefore retain
  their optimizer and lookahead state across resume.
- A quantized network cannot recover the factorized decomposition. Loading one
  initializes the folded value in the bucket term and resets the shared term to
  zero.
