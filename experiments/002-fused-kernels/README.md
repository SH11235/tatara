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

各 kernel に対し GPU↔CPU 数値同等性 smoke test を `tests/<kernel>_smoke.rs` に
配置する (Stage 1 の `experiments/001-cuda-oxide-kpabs/tests/*.rs` と同列)。

## ベンチ (Stage 2-8 / #44)

各 fused kernel と naive (1 op = 1 kernel) baseline の `samples/sec` を
`tests/<kernel>_bench.rs` で計測する (Stage 1-10 の `samples/sec` ベンチ pattern を
踏襲)。bullet 本家との直接比較は GPU/OS/driver 差で apples-to-apples にならない
ため、本 crate では naive baseline 比 (memory-traffic 削減効果の検証) のみを
mandatory にする。

## 使い方

```bash
# .ll 生成 (Stage 2-1 以降):
cd experiments/002-fused-kernels && \
    CUDA_OXIDE_TARGET=sm_75 \
    /mnt/e/cuda-oxide-target/release/cargo-oxide build

# host 側 smoke (cargo build / test、CUDA toolkit 必須):
cargo test -p exp-002-fused-kernels
```

## CI

本 crate は `cuda-host` 経由で transitive に `cuda.h` を要求するため
GitHub-hosted runner では build できない。`.github/workflows/checks.yaml` の
`--exclude` リストに `exp-002-fused-kernels` が追加済 (Stage 1-9 で
`exp-001-cuda-oxide-kpabs` を exclude したのと同じ理由)。host helper や
reference CPU は `gpu-kernels` crate 側に置くことで CI でも検証可能。
