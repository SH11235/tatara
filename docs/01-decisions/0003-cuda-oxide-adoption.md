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

- **LLVM 21+ (`llc-21`) が floor、`llc-22` 推奨** — pipeline は `llc-22` → `llc-21` の順で auto-discover する (`CUDA_OXIDE_LLC` で固定可)。cuda-oxide の `atomics` example README は LLVM 22 を「Atomic operations require llc-22 or newer for correct syncscope」と推奨。LLVM 21 でも smoke は通るが、本番 kernel で `memory_order` の正確性を求めるなら 22 に上げる。Ubuntu 24.04 では `apt.llvm.org/llvm.sh` で導入
- **`clang` (vanilla 名)** が `cuda-bindings` の bindgen に必要 — `update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100`
- nightly Rust (`rust-toolchain.toml` に pin) が必要 — cuda-oxide の `nightly-2026-04-03` に整合
- runtime fusion (bullet-gpu の PointwiseIR) は失われる → ADR-0004 で代替策
- **GPU 要件: cuda-oxide 公式は Ampere+ (sm_80+)**。Turing (sm_75) は **`CUDA_OXIDE_TARGET=sm_75` 環境変数で公式パスのまま動く** (Stage 0 で確認済み):
  - `--arch=sm_75` flag は cuda-oxide 内部の `select_target()` (auto-detect) に override されてしまい、`Basic` フォールバックの `sm_80` が選ばれる。結果として PTX header は `.target sm_80` になり Turing で `CUDA_ERROR_INVALID_PTX` (driver error 218)
  - 一方 `CUDA_OXIDE_TARGET=sm_75` (env var) は `mir-importer/src/pipeline.rs:803` で読まれ、`select_target()` をバイパスして `llc -mcpu=sm_75` までそのまま流れる (doc-comment は同ファイル `:34`)
  - **適用範囲**: vecadd / atomics 程度の単純 kernel は OK。LLVM IR に sm_80+ 専用 op (`cp.async`, `wgmma`, `tcgen05`, `tma.*`, `cluster.*`) が含まれていると `llc` か CUDA driver 段階で失敗する。Stage 1 KP-abs (forward / grad scatter / adam_step / eval) は適用範囲内見込み。Stage 2+ で fused / async copy / cluster ops を使うと CUDA_OXIDE_TARGET でも回避不能になり sm_80+ 実機が必要
- **sm_86 (Ampere) 実機検証は Stage 1 着手前の follow-up issue で実施**: 本マシンは sm_75 で sh11235 は当面 GPU 占有中のため Stage 0 では sm_75 確認のみ。sm_86 で同 example が公式パスで通ることを確認するまで「cuda-oxide が rshogi-nnue ターゲット環境で動く」最終保証は未到達
- cuda-oxide が production-ready でないと判明した場合は abandon しやすいよう、
  experiments/00N で個別検証してから昇格させる方針 (ADR-0005)

## Stage 0 検証結果 (2026-05-09)

- cuda-oxide commit `6de0509` (NVlabs/cuda-oxide main, 2026-05-08) で動作確認
- 環境: WSL2 Ubuntu 24.04, RTX 2070 SUPER (sm_75), CUDA 12.9, LLVM 21.1.8, rustc nightly-2026-04-03
- `cargo oxide doctor`: 全項目 ✓
- `CUDA_OXIDE_TARGET=sm_75 cargo oxide run vecadd`: `✓ SUCCESS: All 1024 elements correct!` (PTX header `.target sm_75`)
- `CUDA_OXIDE_TARGET=sm_75 cargo oxide run atomics`: 20/20 tests passed (F32/F64/U64 atomicAdd 含む)
- sm_86 実機での同手順検証は **未実施** (sh11235 解放後の follow-up issue で対応)
- 詳細: `docs/setup.md`
