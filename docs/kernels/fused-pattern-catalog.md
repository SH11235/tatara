# Fused kernel pattern catalog

[fused kernel strategy ADR](../01-decisions/2026-05-09-fused-kernel-strategy.md)
で「runtime fusion を build-time hand-fused kernel で代替する」と決めた fused
kernel の **責務 / op 数 / 配置ファイル** を一覧する。
upstream (bullet) のどの関数を hand-fuse したかは各 kernel のソースコメント
を参照。

## Pointwise fused kernels

配置: `crates/gpu-kernels/src/pointwise/` (reference CPU 実装) +
`bins/nnue_train/src/main.rs` の `#[kernel]` ブロック (device 側、cuda-oxide
の bin-entry inline 制約による)。

| Pattern | Op 数 | 用途 |
|---|---|---|
| `fused_screlu_grad` | 2-3 | SCReLU activation gradient (forward 経路と組合せ) |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale (旧 v101 loss) |
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
absolute throughput (M samples/sec) は単一 kernel の micro-bench より、
学習 step 全体での throughput (`bins/nnue_train` の pos/s ログ) で測る方が
現実的 (training context では memory bandwidth が真に律速)。

参考値 (RTX 2070 SUPER / sm_75、n_elements=1024、kernel-only timing、50 step
平均):

| Pattern | per-step | absolute |
|---|---|---|
| `fused_screlu_grad`         | ~32 µs | 32.1 M elements/sec |
| `fused_loss_wdl`            | ~34 µs | 29.9 M elements/sec |
| `fused_adamw_step`          | ~30 µs | 34.3 M elements/sec |
| `fused_radam_step`          | ~30 µs | 34.9 M elements/sec |
| `fused_ranger_step` (lerp)  | ~30 µs | 34.6 M elements/sec |
| `sparse_ft_forward`         | ~35 µs | 29.6 M elements/sec |
| `sparse_ft_backward`        | ~32 µs | 31.7 M elements/sec |

1024 element 程度では kernel 実行時間 (< 50 µs) より `cuStreamSynchronize` の
host-side wait overhead が支配的になる (launch overhead dominant)。training
の実 batch size (≥ 8K) では launch overhead が薄まり bandwidth-bound に寄る
ため、上の数値は regression detection 用 baseline として位置付ける。

## 新規 fused kernel を追加するとき

fused kernel strategy ADR が想定する「新しい optimizer / activation を試す時
はパターンを追加する必要がある」運用に従う。手順:

1. 本 catalog にエントリを追加
2. `crates/gpu-kernels/` 配下に reference CPU 実装 + 数値同等性テスト
3. `bins/nnue_train/src/main.rs` の `#[kernel]` ブロックに device 実装を追加
   (kernel 名を `kernel_names` リストに登録)
4. trainer の host 側 (`crates/nnue-train/src/trainer.rs` の `LossKind` 等)
   に enum branch を追加して switch できるようにする
