# experiments/002-fused-kernels

Stage 2 (EPIC #16) の hand-fused kernel suite を build-time PTX 化するための
受け皿 experiment crate。Stage 1 の `experiments/001-cuda-oxide-kpabs/` と同じ
役割で、cuda-oxide rustc-codegen-cuda backend の制約 (= `#[kernel]` は bin
entry に inline 配置する) を満たすために存在する。

## 配置 (Stage 2-1〜2-7 で順次埋まる)

| Issue | Kernel | reference CPU 配置 | GPU `#[kernel]` 配置 |
|---|---|---|---|
| #37 (2-1) | `fused_screlu_grad` | `crates/gpu-kernels/src/pointwise/screlu_grad.rs` | 本 crate `src/main.rs` |
| #38 (2-2) | `fused_loss_wdl` | `pointwise/loss_wdl.rs` | 本 crate `src/main.rs` |
| #39 (2-3) | `fused_adamw_step` | `pointwise/adamw_step.rs` | 本 crate `src/main.rs` |
| #40 (2-4) | `fused_radam_step` | `pointwise/radam_step.rs` | 本 crate `src/main.rs` |
| #41 (2-5) | `fused_ranger_step` | `pointwise/ranger_step.rs` | 本 crate `src/main.rs` |
| #42 (2-6) | `sparse_ft_forward`  | `crates/gpu-kernels/src/sparse/sparse_ft_forward.rs`  | 本 crate `src/main.rs` |
| #43 (2-7) | `sparse_ft_backward` | `sparse/sparse_ft_backward.rs` | 本 crate `src/main.rs` |

各 kernel に対し GPU↔CPU 数値同等性 smoke test を **`src/main.rs` の
`#[cfg(test)] mod gpu_cpu_equivalence_tests`** に配置する (Stage 1-10 で確立した
"kernel symbol が bin にしか存在しないため `tests/*.rs` から届かず、`#[cfg(test)]
mod` を main.rs inline に置く" pattern。`bins/progress_kpabs_train` と同方針)。

## ベンチ (Stage 2-8 / #44)

`src/main.rs::bench_tests::bench_all_seven_kernels` で **7 kernel の絶対
samples/sec** を 1024 element / 50 step で計測 (Stage 1-10 の `samples/sec`
ベンチ pattern を踏襲)。Stage 2-0 scaffold 段階では「naive baseline 比 +
bullet runtime-fused 比」の 2 軸計測を計画していたが、Stage 2-1〜2-7 の
kernel 実装が完了した時点で内訳分析の結果:

- **naive baseline 比**: 4/7 kernel (`screlu_grad` / `ranger_lookahead_lerp` /
  `sparse_ft_*`) は naive 分解が degenerate / 適用不能、残る 3 件
  (`loss_wdl` / `adamw_step` / `radam_step`) は **Stage 3 trainer
  integration の actual training throughput** で測る方が training-context
  bandwidth boundedness を反映する → **Stage 3 follow-up issue #53** で deferred
- **bullet runtime-fused 比 (EPIC #16 完了 gate)**: GPU/OS/driver 差で
  apples-to-apples 不可、sh11235 (sm_86 RTX 3080 Ti) GPU 占有解放後に手動
  計測 → **follow-up issue #54** で deferred

→ 詳細は `docs/kernels/fused-pattern-catalog.md` の bench セクション参照。

## 使い方

```bash
# .ll 生成 (Stage 2-1 以降):
cd experiments/002-fused-kernels && \
    CUDA_OXIDE_TARGET=sm_75 \
    /mnt/e/cuda-oxide-target/release/cargo-oxide build

# GPU↔CPU 等価性テスト (要 GPU、ローカル sm_75 box):
cargo test -p exp-002-fused-kernels --bin exp-002-fused-kernels --release \
    -- gpu_cpu_equivalence_tests --test-threads=1

# 絶対 samples/sec ベンチ (sm_75、--nocapture で結果 print):
cargo test -p exp-002-fused-kernels --bin exp-002-fused-kernels --release \
    -- bench_tests --test-threads=1 --nocapture
```

## CI

本 crate は `cuda-host` 経由で transitive に `cuda.h` を要求するため
GitHub-hosted runner では build できない。`.github/workflows/checks.yaml` の
`--exclude` リストに `exp-002-fused-kernels` が追加済 (Stage 1-9 で
`exp-001-cuda-oxide-kpabs` を exclude したのと同じ理由)。host helper や
reference CPU は `gpu-kernels` crate 側に置くことで CI でも検証可能。
