# GPU カーネルは cuda-oxide で書く

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

- NVIDIA 専用で割り切れる (AMD / ROCm 対応は scope 外、AMD ユーザは CUDA / HIP 両 backend を持つ上流 bullet-shogi へ)
- Rust 一言語で完結する設計
- alpha リスクは新規リポなので局所化できる (既存資産を壊さない)
- bullet-shogi 上流追従の責務から解放される

## Consequences

- **LLVM 21+ (`llc-21`) が floor、`llc-22` 推奨** — pipeline は `llc-22` → `llc-21` の順で auto-discover する (`CUDA_OXIDE_LLC` で固定可)。cuda-oxide の `atomics` example README は LLVM 22 を「Atomic operations require llc-22 or newer for correct syncscope」と推奨。LLVM 21 でも smoke は通るが、本番 kernel で `memory_order` の正確性を求めるなら 22 に上げる。Ubuntu 24.04 では `apt.llvm.org/llvm.sh` で導入
- **`clang` (vanilla 名)** が `cuda-bindings` の bindgen に必要 — `update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100`
- nightly Rust (`rust-toolchain.toml` に pin) が必要 — cuda-oxide の `nightly-2026-04-03` に整合
- runtime fusion (bullet-gpu の PointwiseIR) は失われる → 代替策は `2026-05-09-fused-kernel-strategy.md`
- **GPU 要件: cuda-oxide 公式は Ampere+ (sm_80+)**。Turing (sm_75) は
  `CUDA_OXIDE_TARGET=sm_75` 環境変数で公式パスのまま動く:
  - `--arch=sm_75` flag は cuda-oxide 内部の `select_target()` (auto-detect) に
    override されてしまい、`Basic` フォールバックの `sm_80` が選ばれる。結果と
    して PTX header は `.target sm_80` になり Turing で
    `CUDA_ERROR_INVALID_PTX` (driver error 218)
  - 一方 `CUDA_OXIDE_TARGET=sm_75` (env var) は cuda-oxide pipeline で
    `select_target()` をバイパスして `llc -mcpu=sm_75` までそのまま流れる
  - **適用範囲**: 単純な pointwise / sparse / matmul kernel は OK。LLVM IR に
    sm_80+ 専用 op (`cp.async`, `wgmma`, `tcgen05`, `tma.*`, `cluster.*`) が
    含まれていると `llc` か CUDA driver 段階で失敗するため、fused optimizer
    step や Hopper 専用 ops を使う kernel は sm_80+ 実機が必要
- cuda-oxide が alpha 段階のため、新機能は最小スコープで切り出して検証してから
  本流に取り込む段階的アプローチを採る。
