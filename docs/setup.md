**English** | [日本語](setup.ja.md)

# Development environment setup

tatara is built around **cuda-oxide** (NVIDIA Labs' Rust → PTX rustc backend),
so you need to set up both the host (LLVM 21+, ideally LLVM 22) and the GPU
(sm_80+ officially supported). Ampere (sm_86) is the primary target; Turing
(sm_75) also works via the `CUDA_OXIDE_TARGET=sm_75` environment variable (for
the constraints, see the "sub-Ampere GPU" section and the GPU matrix at the
end).

## Supported OSes

| OS | Status | Steps |
|---|---|---|
| Linux | First-class support (verified on Ubuntu 22.04 / 24.04) | Follow the steps in this file directly |
| Windows | Supported via WSL2 (Ubuntu). Native Windows is officially unsupported by cuda-oxide (only CPU-only crates build natively) | Do "Preparing Windows (WSL2)" first, then follow the Linux steps |
| macOS | GPU builds unsupported | Work on a remote Linux machine with an NVIDIA GPU (see below) |

cuda-oxide and this repo's GPU crates require an **NVIDIA GPU + the CUDA
Toolkit**. macOS cannot host an NVIDIA GPU and the CUDA Toolkit is not available
for it (neither Apple Silicon nor Intel Macs), so macOS alone cannot build the
GPU crates (`gpu-runtime` / `bins/*`). To develop from macOS, build and train on
a remote Linux machine with an NVIDIA GPU (an in-house server or a GPU cloud
instance) and use your local macOS for SSH / editing. Editing a CPU-only crate
(`shogi-format` / `shogi-features` / `nnue-format`, etc.) on its own and running
`cargo test -p <crate>` works on macOS, but running `cargo build` across the
whole workspace fails on the cuda-oxide-dependent build.

## Preparing Windows (WSL2)

Building cuda-oxide on native Windows is **officially unsupported upstream**.
The cuda-oxide installation documentation (linked under "Related" at the end)
explicitly states "cuda-oxide currently targets Linux only. Windows is not
supported." (as of 2026-05; there is no Windows-support issue or work either).
On top of that, cuda-oxide is an experimental backend tied directly to the rustc
internal ABI, and this repo's `build.rs` also resolves the CUDA toolkit root
using Linux paths (`/usr/local/cuda` / `lib64/libcublas.so`). So for the GPU
crates (`gpu-runtime` / `bins/*`) on Windows, use **WSL2 + Ubuntu**. NVIDIA GPUs
are visible through CUDA from WSL2, so inside WSL2 the Linux steps in this file
work as-is (cuda-oxide is also officially tested on Ubuntu 24.04).

Note that the CPU-only crates (`shogi-format` / `shogi-features` /
`gpu-kernels` / `nnue-format` / `nnue-train`) pass `cargo test` even on native
Windows (the MSVC toolchain) (242 tests verified green on Windows 11 +
nightly-2026-04-03 in 2026-05). You can run the same scope as the GitHub Actions
CPU check from your local Windows PowerShell:

```powershell
cargo test --workspace --exclude gpu-runtime --exclude progress-kpabs-train --exclude nnue-trainer
```

1. **Install the NVIDIA GPU driver on the Windows host.** WSL2's CUDA uses the
   Windows-side driver; do not install a GPU driver inside WSL2.
2. **Install WSL2 and Ubuntu.** In PowerShell (as administrator):

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

3. **Confirm `nvidia-smi` shows the GPU inside WSL2.** If it does not, update the
   Windows-side driver and the WSL kernel (`wsl --update`) to the latest.
4. **Install the CUDA Toolkit inside WSL2.** Use the WSL toolkit package and do
   not install the driver (`cuda-drivers`) — the driver is on the Windows side.
   From NVIDIA's "CUDA on WSL" distribution, install only `cuda-toolkit-12-x`.
5. From here on, run the steps from "System install" in this file in the WSL2
   shell as-is.

Watch out for WSL2 disk usage (see "WSL2 disk note" at the end).

## System requirements

| Item | Requirement | Notes |
|---|---|---|
| OS | Linux / WSL2 (Windows) | See "Supported OSes" |
| CUDA Toolkit | 12.x (verified with 12.9) | nvcc, libNVVM, nvJitLink, **libcublas** |
| LLVM | **21+ (floor), 22 recommended** | apt.llvm.org provides LLVM 20/21/22 for both jammy / noble. If `llc-22` is on PATH, cuda-oxide prefers it |
| Clang | **clang-21 or 22** + `libclang-common-{21,22}-dev` | Needed by `cuda-bindings`' bindgen (even on LLVM 22, one of clang-21/22 is required) |
| Rust | nightly-2026-04-03 (cuda-oxide pinned) | Pinned by `rust-toolchain.toml` |
| GPU | **Official: Ampere+ (sm_80+)**. Turing (sm_75) also works with `CUDA_OXIDE_TARGET=sm_75` | RTX 30/40/50, A100, H100, B200, etc. |

## Resolving the CUDA toolkit root

`bins/nnue_train` dynamically links against **libcublas** (it runs the L1f
weight backward with `cublasSgemm_v2`). Both build.rs and the runtime look for
the CUDA toolkit root in the following order of precedence:

1. `CUDA_TOOLKIT_PATH` env (a legacy alias used only by build.rs; highest priority)
2. `CUDA_HOME` env (shared by build / runtime)
3. `CUDA_PATH` env (same)
4. Default paths: `/usr/local/cuda` → `/usr/local/cuda-13.2` → `/usr/local/cuda-12.9` → `/opt/cuda`

It uses the first root where `<root>/lib64/libcublas.so` exists. No extra
configuration is needed if the CUDA Toolkit is installed in a standard path. If
you keep it in a non-standard path:

```bash
export CUDA_HOME=/path/to/cuda-12.9    # picked up by both build.rs and runtime
# or, to point only build.rs:
export CUDA_TOOLKIT_PATH=/path/to/cuda-12.9
```

If build.rs cannot find `libcublas.so`, it emits a `cargo:warning` and falls
back to emitting `/usr/local/cuda/lib64` (the build itself does not fail; it
stops only if ld ultimately cannot resolve `-lcublas`).

> **LLVM 22 and atomics syncscope**: cuda-oxide's `atomics` example README
> states "Atomic operations require llc-22 or newer for correct syncscope".
> The example completes on LLVM 21 too, but for production kernels that need
> precise PTX around `memory_order`, upgrading to `llc-22` is preferable.
> cuda-oxide's `cargo-oxide build` (Rust → `.ll`) auto-discovers in the order
> `llc-22` → `llc-21` (can be pinned with `CUDA_OXIDE_LLC=/path`). The
> `.ll`→`.ptx` conversion at binary startup is a separate path that uses env
> vars such as `LLC_BIN` (see "Smoke test").

## System install

Common to Linux / WSL2. Assumes Ubuntu (apt):

```bash
# Basic tools
sudo apt-get update
sudo apt-get install -y wget gnupg lsb-release

# The full LLVM 21 series (apt.llvm.org). To install LLVM 22, replace `21` with `22`
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
chmod +x /tmp/llvm.sh
sudo /tmp/llvm.sh 21
sudo apt-get install -y clang-21 libclang-common-21-dev

# Make clang reachable under its vanilla name
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-21   100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100

# Verify
which llc-21 clang
llc-21 --version | grep nvptx
```

For Rust, use rustup. `rust-toolchain.toml` pins a nightly, so running `cargo`
inside the repository automatically installs the matching toolchain (and the
`rust-src` / `rustc-dev` components).

## Setting up cuda-oxide

Install `cargo-oxide`, the cargo subcommand that compiles GPU kernels to PTX.
You do not need to manually clone the cuda-oxide repo — upstream officially
supports `cargo install --git`, and `cargo-oxide` automatically fetches, builds,
and caches the codegen backend on first run.

The easiest way is to use this repository's wrapper script:

```bash
bash scripts/setup-cuda-oxide.sh
```

The script does the following:

- Reads the cuda-oxide rev pinned by `Cargo.lock` (to keep the library side and
  the codegen backend side on the same rev — a rev mismatch causes a backend ABI
  mismatch)
- Checks for the host prerequisites (rustup / cargo / llc / clang / nvcc) and
  reports their presence (it does not install system packages)
- Runs `cargo install --git ... cargo-oxide` at that rev
- Diagnoses the environment with `cargo-oxide doctor`

To install manually without the script, match the cuda-oxide rev in
`Cargo.lock`:

```bash
rev=$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+' Cargo.lock | sed 's/.*rev=//')
cargo install --git https://github.com/NVlabs/cuda-oxide.git --rev "$rev" --force cargo-oxide
```

Make sure `~/.cargo/bin` is on your PATH. The script reinstalls `cargo-oxide` at
the pinned rev every time, so when you bump the cuda-oxide rev (when you update
the library-side `Cargo.toml`), just rerun `bash scripts/setup-cuda-oxide.sh` the
same way.

## Smoke test

If `cargo-oxide doctor` shows ✓ for every item, the host side is OK. Next, build
the actual kernels. From the repository root:

```bash
bash scripts/build-kernels.sh
```

This detects the GPU generation with `nvidia-smi` and builds every binary that
has kernels (`nnue_train` / `progress_kpabs_train`) with `cargo-oxide build`.
Ampere+ uses the default (sm_80 PTX, forward-compatible) and Turing (sm_75) has
`CUDA_OXIDE_TARGET` set automatically, so you do not need to type the
environment variable by hand.

To build only a specific binary, you can do it manually:

```bash
cd bins/nnue_train
cargo-oxide build
```

For Ampere and later (sm_80+), this is all you need. **Only for Turing
(sm_75)**, prefix it with `CUDA_OXIDE_TARGET=sm_75 cargo-oxide build` (see
"sub-Ampere GPU" below).

`cargo-oxide build` compiles `#[kernel]` to NVPTX IR (`.ll`). At startup the
binary links this `.ll` with libdevice to produce `.ptx` and loads a CudaModule.
The `.ll`→`.ptx` step uses LLVM 21+ `llvm-link` / `opt` / `llc`, auto-searching
`-22` → `-21` (can be pinned with the `LLVM_LINK_BIN` / `OPT_BIN` / `LLC_BIN`
env vars). If training starts without errors, the pipeline is working.

## sub-Ampere GPU (sm_75 Turing)

On its own, `cargo-oxide`'s auto-detect (`select_target()`) picks a target from
the kernel features, and the Basic fallback picks `sm_80`. Even if you pass
`--arch=sm_75`, auto-detect overrides it, so the PTX header stays
`.target sm_80` and loading fails on a Turing GPU with `CUDA_ERROR_INVALID_PTX`
(driver error 218).

The workaround is the **`CUDA_OXIDE_TARGET=sm_75` environment variable**. It is a
first-class override that bypasses `select_target()` and flows all the way
through to `llc -mcpu=sm_75`:

```bash
cd bins/progress_kpabs_train
CUDA_OXIDE_TARGET=sm_75 cargo-oxide build
```

If typing it every time is tedious, export it in your shell rc.

### sm_75 limitations

`CUDA_OXIDE_TARGET=sm_75` only works when the LLVM IR contains no sm_80+-only
ops:

- `cp.async` — asynchronous global → shared copy (sm_80+)
- `wgmma` — warpgroup matrix-multiply-accumulate (sm_90+ Hopper)
- `tcgen05` — 5th-gen tensor cores (sm_100+ Blackwell)
- `tma.*` — Tensor Memory Accelerator (sm_90+)
- `cluster.*` — Thread Block Cluster (sm_90+)

Compiling IR that contains these to sm_75 PTX fails either at `llc` or at the
CUDA driver's JIT load stage. The simple KP-abs progress kernels (forward / grad
scatter / adam_step / eval) are within sm_75's scope. Kernels that use a fused
optimizer step or async copy / Hopper-only ops require an sm_80+ GPU.

You can grep the `.ll` that `cargo-oxide build` produces (it appears in the
binary's directory where you ran the build) to check for sm_80+ ops:

```bash
grep -E '(cp\.async|wgmma|tcgen05|tma\.|cluster\.)' \
  bins/progress_kpabs_train/progress_kpabs_train.ll
# (no output = OK)
```

## Run-only users (not modifying the kernels)

A binary looks for the kernel module in the order `<name>.ll` → `<name>.cubin`
→ `<name>.ptx`; if a `.ll` is present it links it with libdevice and converts it
to `.ptx`, otherwise it loads a prebuilt `.cubin` / `.ptx` directly. It searches
both the binary's crate directory and the workspace root.

So if you already have a `.ptx` for your target GPU, you can load the kernels
without `cargo-oxide` or LLVM just by placing it in the binary's crate directory
or the workspace root. The CUDA driver JITs the `.ptx` into SASS, so a `.ptx`
generated for `sm_80` runs forward-compatibly on Ampere and later
(sm_86/89/90/100…) (Turing sm_75 needs a separate sm_75 `.ptx`).

That said, a `.ptx` goes stale when the kernel source changes, and it is
`.gitignore`d so it is not in git. This repo does not currently distribute
prebuilt `.ptx`, so even users who do not modify the kernels need
`cargo-oxide build` on the first run.

## Supported GPU matrix

| Generation | sm | Representative GPUs | Works with a standard build | `CUDA_OXIDE_TARGET=sm_XX` |
|---|---|---|---|---|
| Pascal | sm_60/61 | GTX 10xx, P100 | ✗ | Untested (LLVM IR compatibility also unverified) |
| Volta | sm_70 | V100, Titan V | ✗ | May work (untested) |
| Turing | sm_75 | RTX 2070 SUPER, GTX 16xx, T4 | ✗ | ✅ Verified |
| Ampere | sm_80 | A100, A30 | ✅ | n/a |
| Ampere | sm_86 | RTX 3080 Ti, RTX 30xx, A40, A10 | ✅ Verified (primary) | n/a |
| Ada | sm_89 | RTX 40xx | ✅ | n/a |
| Hopper | sm_90 | H100, H200 | ✅ | n/a |
| Blackwell | sm_100/120 | B100, B200, RTX 50xx | ✅ | n/a |

The cuda-oxide rev is pinned in this repository's `Cargo.toml`
(`[workspace.dependencies]`), and `scripts/setup-cuda-oxide.sh` keeps
`cargo-oxide` on the same rev. LLVM works on either 21 (sm_75) or 22 (sm_86).

## WSL2 disk note

In a WSL2 environment, `/` (ext4) is actually backed by a **sparse vhdx on the
C: drive**. `df -h /` shows the virtual capacity; the physical limit is bound by
the Avail of `df -h /mnt/c`. To avoid filling up C:, it is recommended to keep
the hundreds-of-GB training data (PSV, checkpoints, logs) and Rust build
artifacts (`target/`) on a separate drive:

- Keep the training data in a working directory on a separate drive and symlink
  to it from `data/` inside the repo
- Point `CARGO_TARGET_DIR` at a path on a separate drive

## Related

- [cuda-oxide adoption ADR](decisions/2026-05-09-cuda-oxide-adoption.md) —
  the adoption decision and its Consequences
- [cuda-oxide upstream](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide-book installation requirements](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
- [cuda-oxide atomics example README (the basis for LLVM 22 syncscope)](https://github.com/NVlabs/cuda-oxide/blob/main/crates/rustc-codegen-cuda/examples/atomics/README.md)
