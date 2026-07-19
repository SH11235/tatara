# Native CUDA benchmark

`nnue-train native-bench` is the reproducible, fixed-fixture throughput benchmark for the CUDA
C++ backend. The benchmark contract, timing, run ordering, statistics, environment capture, and
JSON output are implemented in Rust and are shared by Linux/WSL and native Windows.

This is different from `nnue-train bench-pos`: that runner measures an end-to-end training job
with real data, while `native-bench` measures trainer/GPU throughput using a deterministic
in-memory batch and excludes trainer construction and batch generation.

## Fixed v1 profile

The defaults are batch size 16,384, 3 warm-up steps, 100 timed steps, and 3 independent runs.
Changing a fixture default requires a new profile version.

| Architecture | Fixture |
|---|---|
| LayerStack | `layerstack-halfka-hm-merged-factorized-v1`: factorized HalfKaHmMerged, FT 1536, L1 16, L2 32, 9-bucket Progress8KpAbs trainer shape |
| Simple | `simple-halfkp-factorized-v1`: factorized HalfKP, CReLU, FT 256, L1 32, L2 32 |

Each architecture is measured with both precision configurations by default:

- `fp32`: every precision flag off.
- `all-optim`: TF32, FP16 FT weights/output, and FP16 optimizer state on.

The precision order alternates between runs. In `compare` mode the cuda-oxide/CUDA C++ backend
order also alternates, so a systematic first-run advantage does not belong to one backend.

LayerStack does not read `progress.bin`: the in-memory benchmark batch assigns deterministic
round-robin bucket indices (`row % 9`). This isolates trainer/GPU throughput. Use
`nnue-train bench-pos` when progress calculation and real-data loading should be included.

## Commands

CUDA C++ only, supported on Windows and Linux/WSL:

```sh
cargo run -p nnue-trainer --release --no-default-features --features native-cuda-host -- \
  native-bench --architecture all --precision all
```

The same command in PowerShell is:

```powershell
cargo run -p nnue-trainer --release --no-default-features --features native-cuda-host -- `
  native-bench --architecture all --precision all
```

cuda-oxide versus CUDA C++, supported on Linux/WSL:

```sh
bash scripts/build-kernels.sh
cargo run -p nnue-trainer --release --features native-cuda -- \
  native-bench --mode compare --architecture all --precision all
```

Use `--architecture layerstack` or `simple`, and `--precision fp32` or `all-optim`, to select a
subset. Timing can be overridden with `--batch-size`, `--warmup-steps`, `--steps`, and `--runs`;
the JSON report then records `parameters.customized: true`.

Run the command from a Visual Studio Developer PowerShell on Windows if the C++ build tools are
not already in `PATH`. The CUDA toolkit containing `nvcc` must also be installed.

If warning C4819 is followed by `identifier ... is undefined` on Windows, NVCC's host compiler
has interpreted the UTF-8 kernel source using the default code page. The current build script
passes `/utf-8` to MSVC automatically. For an older commit, set `$env:CL = '/utf-8'` in Developer
PowerShell before running the same benchmark command.

## Results

Reports are written to `target/benchmark-results/native-cuda/` by default and conform to
[`native-cuda-benchmark-v1.schema.json`](schemas/native-cuda-benchmark-v1.schema.json). They
contain every run, mean/median/sample standard deviation/min/max, paired backend deltas,
all-optim/FP32 speedups, expanded precision flags, commit and dirty state, platform (Windows,
WSL, or Linux), GPU, driver, CUDA
toolkit, rustc, Cargo features, and the full command line.

Normal CPU-only CI compiles the versioned schema and validates the checked-in representative
fixture. Native-feature tests additionally serialize the real Rust report types and validate that
document against the same schema.

The runner rejects a dirty working tree by default. `--allow-dirty` is useful while developing,
and the report still records `dirty: true`. Raw reports remain under the gitignored `target/`
tree. Copy or summarize only representative merge/release results in `rshogi-notes`, together
with the tatara commit and exact command.
