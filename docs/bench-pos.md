# End-to-end training benchmark

`nnue-train bench-pos` measures training throughput with real PSV data, the production
dataloader, progress bucket calculation, and the selected GPU backend. The Rust runner is shared
by Linux/WSL and native Windows.

This benchmark has a different contract from `nnue-train native-bench`. `native-bench` uses a
deterministic in-memory batch to isolate trainer and GPU throughput. `bench-pos` reads real data
and is intended to catch end-to-end changes in data loading, feature extraction, bucket routing,
and training.

## Configuration

The configuration is split into two files:

- [`bench-pos.toml`](../bench-pos.toml) is tracked and defines the measurement contract and case
  matrix.
- `bench-pos.local.toml` is gitignored and contains machine-specific paths and hardware settings.

Create the local file once:

```sh
cp bench-pos.local.toml.example bench-pos.local.toml
```

The same file format works in PowerShell:

```powershell
Copy-Item bench-pos.local.toml.example bench-pos.local.toml
```

Set `data` and `progress_coeff` to local files. Relative paths are resolved from the local config
file's directory. `data_id` and `progress_id` are stable logical names written to the report, so
two machines can confirm that they intended to use the same inputs without publishing absolute
paths. The report also records each input basename and byte size.

`progress_coeff` and `progress_id` may be omitted when only Simple cases are selected. A
LayerStack case requires them. The repository does not carry a sample `progress.bin`; use the
coefficient file used by the training setup being measured.

`threads` is machine-specific because it controls dataloader workers. Set `lock_gpu_clock = true`
to ask `nvidia-smi` to lock the graphics clock at its supported maximum. Failure to acquire the
required permission is reported as a warning and does not abort the benchmark.

## Fixed standard profile

The tracked `standard-v1` profile runs two independent training processes per case. Each process
uses five superbatches of 200 batches at batch size 65,536. Superbatch 1 warms the file cache and
GPU; the mean of superbatches 2–5 is the run value. Case order is reversed on every second run so
thermal or clock drift does not always favor the same case.

The matrix contains:

- LayerStack factorized HalfKaHmMerged, FP32
- LayerStack factorized HalfKaHmMerged, all-optim
- Simple factorized HalfKP CReLU 256x2-32-32, FP32
- Simple factorized HalfKP CReLU 256x2-32-32, all-optim

Changing the standard contract requires a new profile name. Machine-specific paths, thread count,
and clock-lock permission do not belong in the tracked profile.

## Running

Build the backend to measure. On Linux/WSL, the default build measures cuda-oxide:

```sh
cargo build -p nnue-trainer --release
target/release/nnue-train bench-pos
```

On native Windows, build the CUDA C++ portable host backend:

```powershell
cargo build -p nnue-trainer --release --no-default-features --features native-cuda-host
target\release\nnue-train.exe bench-pos
```

The same portable host backend can be measured on Linux/WSL:

```sh
cargo build -p nnue-trainer --release --no-default-features --features native-cuda-host
target/release/nnue-train bench-pos
```

Run a subset with repeatable `--case` options:

```sh
target/release/nnue-train bench-pos --case layerstack-fp32
target/release/nnue-train bench-pos --case simple-halfkp-fp32 --case simple-halfkp-all-optim
```

Use `--profile` or `--local-config` to select non-default files. The runner rejects a dirty
working tree unless `--allow-dirty` is passed. Training options inherited from the parent CLI are
rejected by this subcommand; put measurement and training settings in the tracked profile so they
cannot silently differ between hosts.

## Results

JSON reports and child-process logs are written below `target/benchmark-results/bench-pos/`.
Reports conform to [`bench-pos-v1.schema.json`](schemas/bench-pos-v1.schema.json).
Reports include every superbatch value, the measured mean for each run, mean/median/sample standard
deviation/min/max/CV across runs, the exact training arguments with machine paths redacted, input
identities, commit/dirty state, backend features, GPU, driver, CUDA toolkit, rustc, and OS.

Build time is outside the benchmark. The per-superbatch `pos/s` values come from the production
training loop and include real-data loading and feature/bucket processing. The report separately
records total child-process wall time for each run.
