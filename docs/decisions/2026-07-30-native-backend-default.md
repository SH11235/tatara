# native CUDA C++ backend を既定にする

- **Status**: Accepted
- **Date**: 2026-07-30

## Context

GPU kernel は当初 cuda-oxide (Rust → PTX の rustc codegen backend) だけで書く方針
だった (`2026-05-09-cuda-oxide-adoption.md`)。その後、Windows native 対応と導入障壁の
低減のために CUDA C++ kernel + CUDA Driver API の native backend を並行提供した
(`2026-07-18-native-cuda-backend.md`)。

native backend は kernel coverage が揃い、Linux/WSL2 で本番学習に使われている。一方
cuda-oxide 経路は nightly Rust、LLVM 21+、`cargo-oxide` codegen backend cache という
追加の toolchain を要求し、kernel 成果物 (`.ll` → PTX) の事前生成も要る。既定がこちらで
ある限り、新規利用者は使わない backend の環境構築を強いられる。

## Decision

`nnue-trainer` の既定 backend を native CUDA C++ + portable Driver API host runtime に
する。cuda-oxide は**廃止せず opt-in で残す**。

feature 名は backend を直接指す語彙に揃える。

| feature | 内容 |
|---|---|
| `native` (既定) | CUDA C++ kernel + portable Driver API host runtime |
| `oxide` | Rust `#[kernel]` → PTX + cuda-oxide host runtime |
| `oxide-parity` | CUDA C++ kernel を cuda-oxide host から launch する比較オラクル |

`bins/progress_kpabs_train` の kernel も CUDA C++ へ移植する。cuda-oxide 固定の
member が workspace に残ると、Cargo の feature unification で 2 つの backend が同時に
有効化され、`cargo check --workspace` が相互排他の `compile_error!` に抵触するため。

## Consequences

- 既定ビルドの前提が LLVM / `cargo-oxide` から NVCC へ変わる。
- cuda-oxide を残す目的は GPU/CPU 等価テスト資産 (`gpu_cpu_equivalence_tests` /
  `ft_factorize_tests`) を実行できる状態に保つこと。既定構成ではこれらが feature gate で
  skip されるため、`scripts/local-ci.sh` が `oxide` 構成でも明示的にテストを走らせる。
- 2 backend の相互排他は変わらない。`oxide` 系構成は `--no-default-features` を伴う。
- 埋め込む PTX の compute capability は `nvidia-smi` で検出し、NVCC の対応上限へ丸める。
  検出値は Cargo の追跡入力ではないため、GPU 交換時は明示的な再ビルドが要る。
