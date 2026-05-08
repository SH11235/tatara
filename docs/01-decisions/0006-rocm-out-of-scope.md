# ADR-0006: ROCm / AMD GPU サポートは永久に対象外

- **Status**: Accepted
- **Date**: 2026-05-09

## Context

bullet (上流) は CUDA / HIP 両 backend を持っているが、本リポは
それを引き継がない。cuda-oxide が NVIDIA only であり、ROCm 対応するなら
HIP backend が別途必要。

## Decision

ROCm / AMD GPU サポートは **永久に対象外** とする。

## Rationale

- cuda-oxide が NVIDIA only (PTX コンパイラとして実装されている)
- AMD 対応するなら HIP backend が必要だが cuda-oxide には無い
- 本リポは個人 ML playground として割り切り、対応 platform を絞ることで
  開発速度を確保
- AMD で動かしたい場合は bullet-shogi 上流を使う、という棲み分け

## Consequences

- すべての kernel コードは PTX 前提で書ける (条件分岐や abstraction layer 不要)
- `gpu-runtime` crate は CUDA Driver API のみ対象
- README / ドキュメントで「NVIDIA only」を明示する
- 将来 AMD 対応の必要が出たら、別リポ・別プロジェクトとして派生させる
