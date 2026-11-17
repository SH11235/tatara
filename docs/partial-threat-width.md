# Partial Threat width

LayerStack training can keep the piece feature transformer at its full output width while giving Threat rows fewer columns:

```text
nnue-train --data /path/to/train.psv --output /path/to/run layerstack --ft-out 1536 --threat-profile full --threat-width 512
```

Use the normal training data, output and schedule options for your run. `--threat-width` is a LayerStack option, requires a Threat profile and accepts a positive even number no larger than `--ft-out`. Omitting it uses the full FT output width. An explicit full width has the same training and artifact layout as omission. `experiment.json` records a partial width as `params.threat_width`; the field is omitted for full width.

For FT output width L and Threat width T, compact Threat column j maps to FT column j for j < T/2, and to L/2 + j - T/2 otherwise. Thus each pairwise half receives its first T/2 Threat columns. Piece rows and FT bias retain width L. Feature indexing, Threat profile, quantization scale and PSQT behavior are unchanged.

The training matrix is packed as `[piece real × L, Threat real × T, piece virtual × L, Threat virtual × T]`. Virtual rows exist only with the FT factorizer. Optimizer moments and lookahead weights use the same packed layout. Compact forward reads the master weights and folds factorizer contributions as it accumulates. FP16 weight mode rounds the folded weight to FP16 before accumulation. FP16 outputs and optimizer states are supported. Compact training uses separate optimizer and gradient kernels; its throughput should be measured for the chosen batch and precision settings. FT precision sampling with `--precision-steps` requires full width.

## Quantized export

The LayerStack version remains `0x7AF32F21`. All integers are little endian. The header writer is `LayerStackWeights::write_quantised_header`:

1. Version, u32.
2. Existing network hash, u32.
3. Architecture string byte length, u32.
4. Architecture string bytes.
5. Number of output buckets, u32.
6. Existing feature-transform hash, u32.

For partial width only, the architecture string ends with `,ThreatWidth=T`, using a decimal integer. This is the only additional width field; there is no binary width word in the payload. The existing `Threat=<feature count>` and optional `ThreatProfile=<id>` fields retain their meanings. Consumers must validate the width token before interpreting the weight payload. Full width omits the token and retains the existing bytes.

The payload keeps the existing order: compressed FT biases, compressed piece FT weights, optional PSQT, Threat weights, then the existing LayerStack dense layers. A nonzero Threat profile id is still written as u32 immediately before the Threat weights. Threat weights are raw signed i8 in feature-major order, with T consecutive columns per feature. Their scale remains QA=127 and their clamp remains [-127,127]. No Threat bias is added. Full width stores L columns per Threat feature as before.

`eval-sfen` detects `ThreatWidth` from the architecture string and uses the same column mapping. A partial-width file requires an engine reader that understands the token and compact rows.

## Checkpoints

Full-width raw checkpoints use version 11 for Ranger or RAdam and version 12 for SGD. Partial-width raw checkpoints use version 13 for Ranger or RAdam and version 14 for SGD. Both partial-width versions add one u64 Threat width immediately after the feature hash and before the producer run id. Other header fields and parameter-group serialization retain their existing order. Resume validates the width and rejects changes of width. Restarting from a quantized file likewise requires the matching profile and `--threat-width`.
