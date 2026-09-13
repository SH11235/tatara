# Trainer build provenance

Training logs store compiler and source provenance in `params.trainer_build`.
`nnue-train build-info` prints the same object without loading data or initializing
CUDA. Launching one executable from another repository or outside a repository does
not change this object. Both LayerStack and Simple training use it.

- `commit`: full source commit embedded by Cargo, or `null` when unavailable.
- `dirty`: `true` for tracked or non-ignored untracked changes observed at build-script
  execution, `false` for a clean tree, `null` if Git status cannot be determined.
- `backend`: compiled configuration (`native`, `oxide`, `oxide-parity`, or `cpu-only`).
- `rustc`, `target`, `profile`, `opt_level`, `debug`: compiler and Cargo build settings.

The top-level `commit` remains an optional string, now containing the full trainer
commit with a `-dirty` suffix when appropriate. It is omitted when the source or its
cleanliness is unknown. `params.trainer_backend` separately records the backend
selected for the run using the same selector as kernel loading. An `oxide-parity`
build can select native CUDA kernels at runtime without changing its build identity.

The recipe repository and runtime working directory are not inferred from the
trainer commit. `command` retains the invocation; record recipe provenance in the
run recipe separately. A historical experiment without `trainer_build` has no
verified embedded build provenance: its `commit` may describe the launch directory.
Do not rewrite that field by assuming the current executable ran the old experiment.

## Regeneration and limits

`bins/nnue_train/build.rs` obtains Git information from the trainer workspace root.
An unpacked source tree nested in another repository is reported as unknown.
The build script watches itself, its identity helper, the workspace `.git` marker,
worktree `HEAD` and `index`, and common `refs` and `packed-refs`. This covers linked
worktrees, detached HEAD, branch commit/ref updates, and packed refs. Missing watched
paths cause Cargo to rerun the script on subsequent builds; they remain watched so
creation of previously absent Git metadata cannot leave an unknown identity cached.

Git cleanliness is a snapshot when the build script runs. Cargo does not guarantee
rerunning it for every unstaged source edit, new untracked file, ignored file, or
external dependency change. An ordinary recompilation can therefore retain a stale
dirty flag. Build artifacts do not update when a repository changes after building.
Use a clean committed checkout for provenance-sensitive runs. To force refresh after
uncommitted edits, use `cargo clean -p nnue-trainer` and rebuild with the intended
features. Editing files concurrently with a build is outside this guarantee.

This object is source/compiler provenance, not a content hash of the executable or
a complete environment manifest. Linker flags, dependency overrides and dynamic
CUDA/cuBLAS libraries can change independently. Benchmark reports already collect
runtime tool versions; they are not substituted for compiler build information.
The native runtime embeds its generated `NATIVE_KERNEL_FATBIN`; oxide loads external
kernel artifacts using `gpu-runtime`'s existing loader. This metadata does not hash
or certify those external artifacts and introduces no second artifact-discovery
mechanism. Keep the executable and kernel artifacts with a reproducible build recipe
when exact binary reproduction matters.

## Compatibility

The experiment schema version is unchanged. `params` is a passthrough object in
nnue-lab's upload schema, so the optional provenance fields survive normalization;
`commit` still satisfies its optional string contract. Existing experiments without
the fields remain valid. No checkpoint or resume state changes are involved.

Rescore continues using `TATARA_BUILD_COMMIT` from the same capture, retaining its
short commit and `-dirty` representation and its fingerprint keys. Dirty or unknown
builds still receive a nonce, disabling completion skip and resume. A failed Git
status query is unknown rather than silently clean. No new build field is added to
rescore's fingerprint, so its existing compatibility and limitations remain.

## CPU verification

```sh
cargo test -p nnue-trainer --no-default-features
cargo test -p nnue-train --release
cargo clippy -p nnue-trainer --all-targets --no-default-features -- -D warnings
```

The integration tests run the executable from the trainer tree, an unrelated
repository and a directory outside Git. A dependency-free Cargo fixture exercises
the actual build script through linked worktree commits, ref packing, detached HEAD,
staged dirty changes, clean commits and source archives inside/outside Git.
