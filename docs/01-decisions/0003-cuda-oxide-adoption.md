# ADR-0003: GPU カーネルは cuda-oxide で書く

- **Status**: Accepted
- **Date**: 2026-05-09

## Context

bullet-shogi (および上流 bullet) は CUDA C++ を NVRTC で runtime コンパイル
し、PointwiseIR で fused kernel を組み立てる構成。これは強力だが:

- CUDA C++ と Rust で言語が分裂する (host 側のみ Rust)
- 上流の API 変動に追随する責務が継続的に発生する
- ROCm/HIP backend 維持の追加コスト

別の選択肢として **cuda-oxide** (NVIDIA Labs、2026-05 公開) がある。これは
Rust ソースを **build-time に PTX に compile** する rustc backend で、
host から device まで Rust 一言語で書ける。

## Decision

GPU kernel は **cuda-oxide で書く**。NVCC / NVRTC は使わない。

## Rationale

- ROCm 不要なので NVIDIA 専用での割り切りができる (ADR-0006)
- Rust 一言語で完結する設計は個人 learning value に合致
- alpha リスクは新規リポなので局所化できる (既存資産を壊さない)
- bullet-shogi 上流追従の責務から解放される

## Consequences

- LLVM 22 (cuda-oxide の `llc-22`) が build に必要
- nightly Rust (`rust-toolchain.toml` に pin) が必要
- runtime fusion (bullet-gpu の PointwiseIR) は失われる → ADR-0004 で代替策
- sm_86 (Ampere) で動くかは Stage 0 で実機検証する必要がある
- cuda-oxide が production-ready でないと判明した場合は abandon しやすいよう、
  experiments/00N で個別検証してから昇格させる方針 (ADR-0005)
