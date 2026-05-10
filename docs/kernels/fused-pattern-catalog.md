# Fused kernel pattern catalog (Stage 2 / EPIC #16)

ADR-0004 で「runtime fusion を build-time hand-fused kernel で代替する」と
決めた 7 fused kernel の **責務 / op 数 / 上流 (bullet) との対応 / 配置 file
path / 進捗** を一覧する。各 kernel issue (Stage 2-1〜2-7) で landed PR と
ベンチ結果を埋めていく。

## Pointwise fused kernels (`crates/gpu-kernels/src/pointwise/`)

| Pattern | Op 数 | 用途 | 上流 (bullet) | reference CPU 配置 | GPU `#[kernel]` 配置 | Issue / PR | Status |
|---|---|---|---|---|---|---|---|
| `fused_screlu_grad` | 2-3 | activation gradient (forward 経路と組合せ) | `crates/compiler/src/tensor/operation/autograd/dfo.rs::SCReLU` | `pointwise/screlu_grad.rs` | `experiments/002-fused-kernels/src/main.rs::screlu_grad` | #37 | 実装済み (PR pending) |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale | `crates/bullet_lib/src/value/loader.rs` (data-layer blend) + `dfo::Sigmoid` | `pointwise/loss_wdl.rs` | `experiments/002-fused-kernels/src/main.rs::loss_wdl` | #38 | 実装済み (PR pending) |
| `fused_adamw_step` | 5 | AdamW (decay + clip 込み) | `crates/trainer/src/optimiser/adam.rs::AdamWParams` | `pointwise/adamw_step.rs` | `experiments/002-fused-kernels/src/main.rs::adamw_step` | #39 | 実装済み (PR pending) |
| `fused_radam_step` | 5+host | RAdam (AdamW + bias correction + denom switch) | `crates/trainer/src/optimiser/radam.rs::RAdamParams` | `pointwise/radam_step.rs` (TBD) | 同上 | #40 | 未実装 |
| `fused_ranger_step` | RAdam + lookahead | Ranger (RAdam + slow params lerp、k-step periodic) | `crates/trainer/src/optimiser/ranger.rs` | `pointwise/ranger_step.rs` (TBD) | 同上 | #41 | 未実装 |

## Sparse FT kernels (`crates/gpu-kernels/src/sparse/`)

| Pattern | Op 数 | 用途 | 上流 (bullet) | reference CPU 配置 | GPU `#[kernel]` 配置 | Issue / PR | Status |
|---|---|---|---|---|---|---|---|
| `sparse_ft_forward` | matmul | HalfKA_hm sparse feature transform forward | `crates/compiler/src/tensor/operation/linear/sparse.rs::SparseMatmul` | `sparse/sparse_ft_forward.rs` (TBD) | `experiments/002-fused-kernels/src/main.rs` | #42 | 未実装 |
| `sparse_ft_backward` | atomic scatter | 同 backward | `linear/sparse.rs::SparseMatmulBwd(Multi)` | `sparse/sparse_ft_backward.rs` (TBD) | 同上 | #43 | 未実装 |

## ベンチ (Stage 2-8 / #44)

2 つの比較軸を併記する:

1. **naive baseline 比 (mandatory local regression gate)** — 各 fused kernel と
   同 experiments crate に並置する naive (1 op = 1 kernel に展開した参考実装) との
   `samples/sec` 比。Stage 2-8 で全 kernel について **fused が naive 比 ≥ 1.0x**
   (memory traffic 削減効果が出ている) を必達確認する。Stage 1-10 で確立した
   `samples/sec` ベンチ pattern を踏襲、計測環境はローカル sm_75 (RTX 2070 SUPER)
   で固定可能なので CI-like な regression gate として使える。
2. **bullet runtime-fused 比 (EPIC #16 completion target)** — ADR-0004 が定めた
   EPIC 完了条件「bullet runtime-fused 比 ≥ 90% (目標 ±0%)」を満たすかの
   検証。bullet 本家との直接比較は GPU (sm_75 / sm_86 / sm_100)・OS・driver・
   NVRTC バージョン差で apples-to-apples にならないため、計測は **手動・別環境で**
   sh11235 (sm_86 RTX 3080 Ti) 解放後に取って手記録する。記録欠落時は EPIC close
   までに最低 1 kernel について sm_86 で確認するスタイルを基本とする。

| Pattern | naive baseline 比 (local sm_75) | bullet runtime-fused 比 (manual sm_86) | 計測環境 | 計測 PR |
|---|---|---|---|---|
| `fused_screlu_grad`   | (TBD) | (TBD) | (TBD) | (TBD) |
| `fused_loss_wdl`      | (TBD) | (TBD) | (TBD) | (TBD) |
| `fused_adamw_step`    | (TBD) | (TBD) | (TBD) | (TBD) |
| `fused_radam_step`    | (TBD) | (TBD) | (TBD) | (TBD) |
| `fused_ranger_step`   | (TBD) | (TBD) | (TBD) | (TBD) |
| `sparse_ft_forward`   | (TBD) | (TBD) | (TBD) | (TBD) |
| `sparse_ft_backward`  | (TBD) | (TBD) | (TBD) | (TBD) |

## 運用方針

- 1 kernel = 1 file = 1 PR で landed させる (Stage 1-5〜1-8 と同流儀)
- 各 PR で Status を「実装済み (PR #N)」に更新 + Issue リンクを埋める
- ベンチは Stage 2-8 (#44) で wrap-up、ただし各 kernel PR でも naive 比較を
  PR 本文に貼ることは推奨 (catalog 側更新は wrap-up でまとめても可)
- 新規 fused kernel を Stage 2 以降に追加する場合は本 catalog にまず entry を
  追加してから着手する (ADR-0004 の "新しい optimizer や activation を試す時は
  パターンを追加する必要がある" を運用化)
