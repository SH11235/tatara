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

- **LLVM 21+** (NVPTX backend 付き、`llc-21` を `PATH` に) が build に必要 — 公式 README は `LLVM 21+` を要件とする (`llc-22` も discoverable)。Ubuntu 24.04 では `apt.llvm.org/llvm.sh` 経由で `llvm-21` + `clang-21` + `libclang-common-21-dev` を入れる
- **`clang` (vanilla 名)** が `cuda-bindings` の bindgen に必要 — `update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100`
- nightly Rust (`rust-toolchain.toml` に pin) が必要 — cuda-oxide の `nightly-2026-04-03` に整合
- runtime fusion (bullet-gpu の PointwiseIR) は失われる → ADR-0004 で代替策
- **GPU 要件: cuda-oxide 公式は Ampere+ (sm_80+) のみ**。Turing (sm_75) と Pascal (sm_60/61) は **codegen が `--mcpu=sm_80` 固定** のため、`cargo oxide run` 直接では `CUDA_ERROR_INVALID_PTX` (driver error 218) になる
  - **Workaround (Stage 0 で確認済み)**: cuda-oxide pipeline で生成された `.ll` を `llc-21 --mcpu=sm_75 -mattr=+ptx70` に流して PTX を再生成、`load_module_from_file` 用の `<example>.ptx` を差し替える方式で sm_75 GPU 実行可能。`scripts/build_for_sm75.sh` に自動化済み
  - **適用範囲**: vecadd / atomics 程度の単純 kernel ならば LLVM IR に sm_80 専用 op (`cp.async`, `wgmma`, `tcgen05`, `tma.*`, `cluster.*`) が含まれず動作する。Stage 1 KP-abs (forward / grad scatter / adam) は適用範囲内見込み。Stage 2+ で fused / async copy / cluster ops を使うと workaround 不能の可能性
  - 実機 sm_86 (sh11235) 検証は GPU 解放後の follow-up
- cuda-oxide が production-ready でないと判明した場合は abandon しやすいよう、
  experiments/00N で個別検証してから昇格させる方針 (ADR-0005)

## Stage 0 検証結果 (2026-05-09)

- cuda-oxide commit `6de0509` (NVlabs/cuda-oxide main, 2026-05-08) で動作確認
- 環境: WSL2 Ubuntu 24.04, RTX 2070 SUPER (sm_75), CUDA 12.9, LLVM 21.1.8, rustc nightly-2026-04-03
- `cargo oxide doctor`: 全項目 ✓
- `vecadd` (sm_75 PTX swap): `✓ SUCCESS: All 1024 elements correct!`
- `atomics` (sm_75 PTX swap): 20/20 tests passed (F32/F64/U64 atomicAdd 含む)
- 詳細: `docs/setup.md`, `scripts/build_for_sm75.sh`
