# runtime fusion を build-time hand-fused kernel で代替する

- **Status**: Accepted
- **Date**: 2026-05-09

## Context

bullet-gpu は runtime に PointwiseIR を組み立てて NVRTC で fused kernel を
作る。これが element-wise シーケンス (optimizer step、activation gradient、
loss + WDL blending 等) で **memory traffic を 1/N に削る**重要な機構。

cuda-oxide は build-time コンパイラなので **runtime fusion は不可**。
naive port (1 op = 1 kernel) なら memory bandwidth bound で
**−20〜−40% の slowdown** が出る可能性がある。

## Decision

本リポジトリは shogi NNUE 専用で architecture が固定のため、必要な fused
kernel パターンは 3〜5 種類で打ち止めできる。それらを cuda-oxide で
build-time に hand-code すれば bullet runtime-fused 相当の memory traffic を
達成できる。

### Fused kernel カタログ

| Pattern | Op 数 | 用途 |
|---|---|---|
| `fused_radam_step` | 5 | RAdam の m, v 更新 + bias correction + weight 更新 |
| `fused_ranger_step` | RAdam + lookahead | Ranger optimizer (RAdam + slow params の lerp) |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale |
| `fused_screlu_grad` | 2-3 | activation gradient (forward 経路と組合せ) |
| `fused_adamw_step` | 5 | AdamW (decay 込み) |

ハンドコード労力は合計 100〜300 行程度。

## Consequences

- 性能ギャップ ~±0% を狙える。「runtime fusion 喪失で −20〜−40%」の懸念は
  naive port 限定の話で、設計でカバーする。
- 実装は `crates/gpu-kernels/pointwise/` 配下に Pattern 1 個 = 1 ファイルで配置
- 各 fused kernel に CPU reference 実装と数値同等性テストを併設する
- 新しい optimizer や activation を試すときはパターンを追加する必要がある
  (固定コスト)
- 詳細カタログは `docs/kernels/fused-pattern-catalog.md`
