**English** | [日本語](README.ja.md)

# tatara

**A fast Rust trainer for shogi NNUE evaluation networks.**

tatara trains shogi NNUE (Efficiently Updatable Neural Network) evaluation
networks on the GPU. It is written in **Rust end to end**, from host to device:
GPU kernels are compiled to PTX at build time by
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) (NVIDIA Labs' Rust → PTX
rustc backend) — no C / C++ / CUDA C++ anywhere in the pipeline.

Hand-fusing the GPU kernels makes it **very fast** — it out-throughputs its
upstream CUDA C++ trainer
[bullet-shogi](https://github.com/SH11235/bullet-shogi). Measured on an RTX 3080
Ti (LayerStack), even the bit-identical default path is **+37%** over
bullet-shogi, and stacking the opt-in FP16 modes reaches up to **~2.1×**.

*tatara (踏鞴) is the traditional Japanese furnace that smelts iron sand into
tamahagane steel — forging a net out of raw data.*

> **NVIDIA only** — because cuda-oxide only generates PTX, ROCm / AMD is out of
> scope. To train comparable shogi NNUE nets on an AMD GPU, see the upstream
> [bullet-shogi](https://github.com/SH11235/bullet-shogi), which has both CUDA
> and HIP backends.

## What you can train

A trained NNUE is defined by two independent choices: the **architecture**
(a subcommand), which fixes the network structure, and the **input feature set**
(`--feature-set`), which fixes how a board position is turned into an input
vector.

### Architecture

| Architecture | Subcommand | Structure |
|---|---|---|
| **LayerStack** | `layerstack` | Specializes the output layer per bucket by game progress (9 buckets; the same idea as Stockfish's "LayerStacks"). FT output `--ft-out` (default 1536) → 16 → 32 |
| **Simple** | `simple` | A plain NNUE with no bucket split (FT → 2 hidden layers → single output). Layer dimensions are set with `--arch <l1>x2-<l2>-<l3>` (`l1` = FT output, `l2`/`l3` = hidden layers; default `256x2-32-32`); activation crelu / screlu / pairwise |

### Input feature set

`--feature-set` selects one of five (default `halfka-hm-merged`). They differ in
how the kings are included as features:

| `--feature-set` | King handling |
|---|---|
| `halfkp` | Kings themselves are not included as piece features |
| `halfka-split` | Kings are included; own-king and enemy-king features have separate slots |
| `halfka-merged` | Kings are included; own-king and enemy-king features share one slot |
| `halfka-hm-split` | `halfka-split` plus a left-right mirror so the king always sits on one side of the board, compressing king squares 81 → 45 |
| `halfka-hm-merged` (default) | `halfka-merged` plus the same left-right mirror king-square compression |

The default `halfka-hm-merged` applies to shogi the same design as Stockfish's
**HalfKAv2_hm** (left-right king-square mirroring + own-king/enemy-king features
sharing one slot).

A separate binary, `progress-kpabs-train`, is a KP-abs progress trainer that
produces `progress.bin`, the bucket coefficients for LayerStack. The approach of
learning game progress and assigning it to output buckets is based on an idea
from [a post by nodchip](https://nodchip.hatenablog.com/entry/2026/02/04/000000).

## Setup

### Requirements

- **OS** — Linux is first-class; Windows is supported via WSL2; macOS cannot
  build the GPU crates
- **NVIDIA GPU** (Ampere and later / sm_80+ is officially supported; Turing /
  sm_75 also runs simple kernels with the `CUDA_OXIDE_TARGET=sm_75` environment
  variable)
- **CUDA Toolkit 12.x** (verified with 12.9)
- **LLVM 21+** (`llc-21` is the floor; `llc-22` is recommended because it is
  needed for fully correct atomics syncscope)
- **Rust nightly** (`rust-toolchain.toml` tracks the cuda-oxide upstream
  channel; do not change the channel yourself, since it depends on the rustc
  internal ABI)

To set up `cargo-oxide`, which builds the GPU kernels, run
`bash scripts/setup-cuda-oxide.sh`. For detailed installation steps, per-OS
guidance, and the supported-GPU matrix, see [docs/setup.md](docs/setup.md).

### Build and train

For building the kernels and running the smoke test, see
[docs/setup.md](docs/setup.md); for how to run training, see
[docs/training-quickstart.md](docs/training-quickstart.md).

## Documentation

- [Setup guide](docs/setup.md) — per-OS guidance, CUDA / LLVM / `cargo-oxide`
  setup, supported-GPU matrix, CUDA toolkit root resolution
- [Training quickstart](docs/training-quickstart.md) — per-architecture training
  examples + key CLI options + resume / checkpoint workflow
- [ADR (Architecture Decision Records)](docs/decisions/) — design decisions and
  their rationale
- [Fused kernel catalog](docs/kernels/fused-pattern-catalog.md) — which kernel
  does what
- [Arch string](docs/arch-string.md) — how the architecture-description string
  embedded in the quantised `.bin` header is assembled and checked at load time

## Glossary

| Abbreviation | Meaning |
|---|---|
| **NNUE** | Efficiently Updatable Neural Network — a lightweight evaluation function used by shogi / chess engines |
| **FT** | Feature Transformer — the NNUE's sparse-input → dense layer |
| **PSV** | PackedSfenValue — a training-data format from bullet-shogi (one position + score + WDL) |
| **KP / KP-abs** | King-Piece relative feature and its absolute-value variant (for progress / entering-king detection) |
| **bucket** | Per-output-bucket weight separation (branching by game phase / progress) |
| **CReLU / SCReLU / Pairwise** | NNUE activation functions. CReLU = Clipped ReLU, SCReLU = Squared Clipped ReLU, Pairwise = elementwise product of the first and second halves, halving the input dimension. Selected by `--activation` on the `simple` architecture |
| **RAdam / Ranger** | Rectified Adam / Ranger optimizer (Ranger = RAdam + lookahead) |
| **WRM** | Win-rate model loss (from bullet `--win-rate-model`) |
| **SPRT** | Sequential Probability Ratio Test — a method that plays two nets against each other and sequentially tests the strength difference. Used to confirm the quality of a trained net |
| **superbatch** | A bullet term: the unit of "multiple batches treated as one, advancing the lr/wdl scheduler" |
| **PTX** | Parallel Thread Execution — a virtual ISA for NVIDIA GPUs. CUDA C++ / Rust → PTX (`.ptx` text) → the CUDA driver's JIT compiles it to SASS (real machine code) for execution. It is portable across generations (PTX built for sm_80 runs forward-compatibly on sm_86/89/90). See the supported-GPU matrix in `docs/setup.md` |
| **SASS** | Per-generation real machine code for NVIDIA GPUs. The terminal form the CUDA driver JIT produces from PTX. This repository does not handle it directly |
| **sm_XX** | The compute capability of an NVIDIA GPU (e.g. sm_75 = Turing, sm_86 = Ampere RTX 30xx). Used to specify the target architecture when generating PTX (`CUDA_OXIDE_TARGET=sm_86`, etc.) |

## Related repositories

- [rshogi](https://github.com/SH11235/rshogi) — the shogi engine that loads and plays with the NNUE trained here
- [bullet](https://github.com/jw1912/bullet) — upstream (NNUE training framework)
- [bullet-shogi](https://github.com/SH11235/bullet-shogi) — a shogi-oriented fork of bullet
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — the Rust → PTX rustc backend

## License

MIT (see [LICENSE](LICENSE)).
For the scope of code taken from bullet-shogi / bullet / cuda-oxide and license
compatibility, see [ATTRIBUTION.md](ATTRIBUTION.md).
