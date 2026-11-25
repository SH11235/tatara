# Partial Threat width

LayerStack training can keep the piece feature transformer at its full output width while giving Threat rows fewer columns:

```text
nnue-train --data /path/to/train.psv --output /path/to/run layerstack --layerstack-head split-skip --ft-out 1536 --threat-profile full --threat-width 512
```

Use the normal training data, output and schedule options for your run. `--threat-width` is a LayerStack option, requires a Threat profile and accepts a positive even number no larger than `--ft-out`. Omitting it uses the full FT output width. An explicit full width has the same training and artifact layout as omission. `experiment.json` records a partial width as `params.threat_width`; the field is omitted for full width.

For FT output width L and Threat width T, compact Threat column j maps to FT column j for j < T/2, and to L/2 + j - T/2 otherwise. Thus each pairwise half receives its first T/2 Threat columns. Piece rows and FT bias retain width L. Feature indexing, Threat profile, quantization scale and PSQT behavior are unchanged.

The training matrix is packed as `[piece real × L, Threat real × T, piece virtual × L, Threat virtual × T]`. Virtual rows exist only with the FT factorizer. Optimizer moments and lookahead weights use the same packed layout. Forward reads a packed cache with factorizer contributions folded once after each optimizer step and after checkpoint loading. FP16 weight mode rounds the folded weight to FP16 before caching; the compact cache stores those rounded values as FP32. FP16 outputs and optimizer states are supported. Native forward groups positions by king bucket and loads four columns together when both widths are divisible by eight. The native fused optimizer updates the packed piece and Threat blocks with their respective row widths when the Threat width is divisible by eight and the FT width is divisible by four. Other widths and the cuda-oxide backend use separate optimizer kernels. FT norm reduction partitions rows across the GPU; fused updates accumulate the norms for the following step. Gradient gathering keeps the packed layout. FT precision sampling with `--precision-steps` requires full width.

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

Full-width raw checkpoints use version 11 for Ranger, RAdam, or AdamW and version 12 for SGD. Partial-width raw checkpoints use version 13 for Ranger and AdamW, version 14 for SGD, and version 15 for RAdam. Transferring momentum from a version 13 checkpoint to SGD requires producer metadata identifying RAdam. All partial-width versions add one u64 Threat width immediately after the feature hash and before the producer run id. Other header fields and parameter-group serialization retain their existing order. Resume validates the width and rejects changes of width. Restarting from a quantized file likewise requires the matching profile and `--threat-width`.

## Throughput comparison

Use the same representative small PSV and progress coefficients for both widths. About one to four million PSV records is sufficient; the loader wraps at EOF. Keep other GPU workloads stopped. For an RTX 5090, select the native CUDA target explicitly so Cargo also replaces any artifacts built without GPU detection:

```bash
TATARA_CUDA_COMPUTE=120 cargo build -p nnue-trainer --release -j 4
```

For another GPU, use its numeric compute capability instead of `120`.

The following runs 384 batches per process, twice per width, with the factorizer left at its default enabled setting. Set `SHOGI_DATA` to the shared data root before running. The output directory is unique to avoid overwriting checkpoints:

```bash
set -euo pipefail
psv=/path/to/small.psv
progress=/path/to/progress.bin
mkdir -p "${SHOGI_DATA:?}/runs/tatara"
bench_root=$(mktemp -d "$SHOGI_DATA/runs/tatara/threat-width-bench.XXXXXX")
unset NNUE_TRAIN_STEP_PROFILE
export NNUE_TRAIN_BATCH_PROGRESS_EVERY=0
for repeat in 1 2; do
  for width in full 512; do
    width_args=()
    if [[ "$width" != full ]]; then
      width_args=(--threat-width "$width")
    fi
    target/release/nnue-train \
      --data "$psv" --output "$bench_root/width-$width-run-$repeat" \
      --progress-coeff "$progress" --batch-size 65536 \
      --superbatches 3 --batches-per-superbatch 128 \
      --teacher-shuffle-buffer-mib 0 --optimizer ranger --norm-loss \
      layerstack --layerstack-head split-skip --ft-out 1536 --l1 16 --l2 64 --num-buckets 8 \
      --qat dense --threat-profile full "${width_args[@]}" \
      2>&1 | tee "$bench_root/width-$width-run-$repeat.log"
  done
done
```

Compare the `pos/s` values from superbatches 2 and 3; the first superbatch warms up the kernels and data pipeline. These timings cover training and data loading, and exclude checkpoint writing. Report both repetitions and the GPU model. Losses between different widths are not a numerical equivalence check. For a before/after comparison of the trainer, run the same 512-width command with each trainer revision.

GPU validation of the compact kernels, optimizer equivalence, and checkpoint resume is separate from timing:

```bash
cargo test -p nnue-trainer --release -j 4 partial_threat_ -- --test-threads=1
cargo test -p nnue-trainer --release -j 4 fused_ft_optimizer_matches_separate_launches -- --test-threads=1
cargo test -p nnue-trainer --release -j 4 explicit_full_threat_width_preserves_training -- --test-threads=1
```
