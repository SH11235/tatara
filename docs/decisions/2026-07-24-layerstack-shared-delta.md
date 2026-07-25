# LayerStack shared-delta parameterization

## Decision

LayerStack keeps its existing L1 shared term and can opt in to L2/L3 shared
terms with `--stack-shared-delta`. Each enabled layer is trained as:

`effective_bucket_parameter = bucket_parameter + shared_parameter`

The shared L2/L3 parameters start at zero. Forward and input-backward use
folded buffers, while parameter gradients are reduced across buckets for the
shared optimizer groups. Quantized export writes only the folded per-bucket
parameters, preserving the existing LayerStack file format and engine ABI.

## Consequences

- The control path is unchanged when the flag is absent.
- Enabling the flag is step-zero equivalent to the control initialization.
- The implementation works for every supported bucket count and is not tied to
  KingRank9; Progress8 uses the same runtime bucket count.
- Raw checkpoints append the four shared parameter groups
  (`l2_shared_weight`, `l2_shared_bias`, `l3_shared_weight`, and
  `l3_shared_bias`) and therefore retain
  their optimizer and lookahead state across resume.
- A quantized network cannot recover the shared-delta decomposition. Loading one
  initializes the folded value in the bucket term and resets the shared term to
  zero.
