# ADR-0005: 4 段階で進める

- **Status**: Accepted
- **Date**: 2026-05-09

## Context

bullet-shogi 相当の NNUE training を cuda-oxide で一気に書き起こすのは
リスクが大きい (cuda-oxide alpha のリスク + アーキ固定の重さ)。
小さく試して段階的に build up する方が安全。

## Decision

4 段階で進める。各 stage は前 stage の完了を**絶対前提とせず**、
`experiments/00N/` で個別検証してから main の `crates/` に昇格させる。

| Stage | スコープ | 目的 | 成果物 |
|---|---|---|---|
| **Stage 1** | `experiments/001/` で cuda-oxide で KP-abs progress trainer を実装 | cuda-oxide / Rust GPU の習熟、最小スコープでの稼働確認 | `progress.bin` (rshogi 互換)、性能ベンチ、技術メモ |
| **Stage 2** | `crates/gpu-kernels/` に hand-fused 学習カーネルを整備 (RAdam, Ranger, SCReLU, sparse FT 等) | NNUE training への足場作り | カーネルライブラリ + 単体テスト |
| **Stage 3** | `crates/nnue-train/` で NNUE training pipeline を構築 (HalfKA_hm 1536-16-32 等) | bullet-shogi 相当の training を Rust 単一言語で再現 | `shogi_nnue_train` binary、自己対局検証 |
| **Stage 4** | 改良路線 (PSQT、Threat、新アーキテクチャ等)、cuda-oxide が成熟したら検討 | research playground | 各種実験記録 |

## Consequences

- 各 stage で「小さく動くもの」を確実に作る。途中で abandon する選択肢を残せる
- Stage 1 がうまく行かなければ cuda-oxide adoption 自体を見直す判断材料が得られる
- experiments → crates への昇格パスを確立 (refactoring/cleanup を伴うが価値がある)
- 各 stage に対応する GitHub milestone を切り、進捗を追跡する
