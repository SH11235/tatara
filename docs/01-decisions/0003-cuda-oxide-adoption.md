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
- **sm_86 (Ampere) 実機検証は Stage 0-1 follow-up (#20, 2026-05-11) で完了**: RTX 3080 Ti / Ubuntu 22.04 jammy / LLVM 22.1.6 で `cargo oxide run vecadd` / `atomics` が `CUDA_OXIDE_TARGET` override **なし**で完走、PTX header `.target sm_80` (auto-detect Basic フォールバック) で sm_86 driver が forward-compat で JIT compile し実行。これで「cuda-oxide が rshogi-nnue 公式 target 環境で動く」最終保証が得られた
- cuda-oxide が production-ready でないと判明した場合は abandon しやすいよう、
  experiments/00N で個別検証してから昇格させる方針 (ADR-0005)

## Stage 0 検証結果

cuda-oxide commit `6de0509` (NVlabs/cuda-oxide main, 2026-05-08) を 2 環境で動作確認。
`(noble, jammy)` × `(LLVM 21, 22)` × `(sm_75 hack, sm_86 公式)` の対角ペアが全て pass。

### 2026-05-09 — sm_75 hack (WSL2 Ubuntu 24.04 noble + LLVM 21.1.8 + RTX 2070 SUPER)

- 環境: WSL2 Ubuntu 24.04, RTX 2070 SUPER (sm_75), CUDA 12.9, LLVM 21.1.8,
  rustc nightly-2026-04-03
- `cargo oxide doctor`: 全項目 ✓
- `CUDA_OXIDE_TARGET=sm_75 cargo oxide run vecadd`: `✓ SUCCESS: All 1024 elements correct!`
  (PTX header `.target sm_75`)
- `CUDA_OXIDE_TARGET=sm_75 cargo oxide run atomics`: 20/20 tests passed
  (F32/F64/U64 atomicAdd 含む)
- sm_75 hack の cuda-oxide 内部挙動: `mir-importer/src/pipeline.rs:803` で
  `CUDA_OXIDE_TARGET` 読込、`select_target()` をバイパス
- 詳細: `docs/setup.md` / Issue #19

### 2026-05-11 — sm_86 公式パス (Native Ubuntu 22.04 jammy + LLVM 22.1.6 + RTX 3080 Ti) ✅ primary

- 環境: Ubuntu 22.04.5 LTS (jammy) Native Linux, RTX 3080 Ti (sm_86, 12 GB),
  CUDA toolkit 12.9 / driver 580.126.09, LLVM 22.1.6 (`llc-22`), clang 22.1.6,
  rustc nightly 1.96.0 (55e86c996 2026-04-02)
- `cargo oxide doctor`: 全 11 項目 ✓
- `cargo oxide run vecadd` (override **なし**): `✓ SUCCESS: All 1024 elements correct!`
  (PTX header `.target sm_80` を auto-detect Basic フォールバックで生成、
  sm_86 driver が forward-compat で JIT compile し実行)
- `cargo oxide run atomics` (override なし): 20/20 tests passed
  (F32/F64/U64 atomicAdd, core::sync::atomic 含む)
- 生成 PTX の sm_80+ 専用 op (`cp.async`, `wgmma`, `tcgen05`, `tma.*`, `cluster.*`)
  混入チェック: 両 example とも 0 件 → Stage 1 KP-abs と同範囲
- 詳細: `docs/setup.md` / Issue #20
