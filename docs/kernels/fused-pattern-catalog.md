# Fused kernel pattern catalog

[fused kernel strategy ADR](../decisions/2026-05-09-fused-kernel-strategy.md)
で「runtime fusion を build-time hand-fused kernel で代替する」と決めた fused
kernel の **責務 / op 数 / 配置ファイル** を一覧する。

## Pointwise fused kernels

配置: `crates/gpu-kernels/src/pointwise/` (reference CPU 実装) +
`bins/nnue_train/src/kernels/` の `#[kernel]` 定義 (device 側、cuda-oxide の
bin-crate reachability 制約により bin crate 内に置く)。

| Pattern | Op 数 | 用途 |
|---|---|---|
| `fused_screlu_grad` | 2-3 | SCReLU activation gradient (forward 経路と組合せ) |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale (`--win-rate-model` 未指定時の loss) |
| `fused_loss_wrm` | 5-6 | WRM (win-rate-model) loss、prediction / target 双方に WRM 適用 |
| `fused_adamw_step` | 5 | AdamW (decay + clip 込み) |
| `fused_radam_step` | 5+host | RAdam (AdamW + bias correction + denom switch) |
| `fused_ranger_step` | RAdam + lookahead | Ranger (RAdam + slow params lerp、k-step periodic) |

## Sparse FT kernels

配置: `crates/gpu-kernels/src/sparse/`。

| Pattern | Op 数 | 用途 |
|---|---|---|
| `sparse_ft_forward` | matmul | HalfKA_hm sparse feature transform forward |
| `sparse_ft_backward` | atomic scatter | 同 backward (per-position の partial gradient を atomic で集約) |

## ベンチ手法

各 kernel には CPU reference 実装と数値同等性テストが併設されており、
`scripts/local-ci.sh` の release build test 経由で常時検証される。

absolute throughput は単一 kernel の micro-bench より、学習 step 全体での
throughput (`bins/nnue_train` の pos/s ログ) で測る。単一 kernel を小さい
element 数で micro-bench すると、kernel 実行時間より `cuStreamSynchronize` の
host-side wait (launch overhead) が支配的になり、training の実 batch size
(≥ 8K) で bandwidth-bound に寄る本番挙動を反映しないため。

## 新規 fused kernel を追加するとき

fused kernel strategy ADR が想定する「新しい optimizer / activation を試す時
はパターンを追加する必要がある」運用に従う。手順:

1. 本 catalog にエントリを追加
2. `crates/gpu-kernels/` 配下に reference CPU 実装 + 数値同等性テスト
3. `bins/nnue_train/src/kernels/` (`common` / `layerstack` / `simple` から
   適切な file) に `#[kernel]` device 実装を追加
4. trainer の host 側 (`crates/nnue-train/src/trainer.rs` の `LossKind` 等)
   に enum branch を追加して switch できるようにする
