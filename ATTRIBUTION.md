# Attribution

This repository derives from and references the open-source projects below.
Each project's copyright notices and license are retained as in the original.

## bullet-shogi / bullet (MIT)

- bullet-shogi: <https://github.com/SH11235/bullet-shogi> (a shogi-oriented fork of jw1912/bullet)
- bullet (upstream): <https://github.com/jw1912/bullet>
- License: MIT

The NNUE trainer algorithm is ported from bullet-shogi / bullet.

## cuda-oxide (Apache-2.0)

- Source: <https://github.com/NVlabs/cuda-oxide>
- License: Apache-2.0
- How it is used: a git dependency in `Cargo.toml` pinned to a commit rev (not
  vendored). As the rustc backend that compiles GPU kernels to PTX at build
  time, `crates/gpu-runtime` and the GPU-dependent binaries reference
  `cuda-core` / `cuda-host` / `cuda-device`.

## Pliron (Apache-2.0)

- Source: <https://github.com/vaivaswatha/pliron>
- License: Apache-2.0
- How it is used: a transitive crate that cuda-oxide depends on.

## License compatibility

This repository itself is MIT. MIT is compatible with distributing compiled
binaries that include Apache-2.0-derived code. When distributing source, retain
the `LICENSE` file of each dependency.
