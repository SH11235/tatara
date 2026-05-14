# 依存戦略 — vendor と git dependency の使い分け

- **Status**: Accepted
- **Date**: 2026-05-09

## Context

外部コードの取り込み方には主に 2 通りある:

1. **Vendor**: 必要部分を手動で copy し、自リポ内で管理する
2. **依存**: `Cargo.toml` に dependency を書いて取り込む (crates.io / git)

bullet-shogi 由来コード (PSV reader、ShogiBoard 等) と cuda-oxide では
更新頻度・編集自由度・上流追従コストが大きく違う。一律な戦略では合わない。

## Decision

- **bullet-shogi 由来のコードは vendor**
  - 必要 file だけ手動 copy、最初の commit に元 commit hash を記載
  - `ATTRIBUTION.md` に取り込み元・元ライセンス・取り込み済 file 一覧
- **cuda-oxide 由来は git dependency + rev pin**
  - `[dependencies] cuda-core = { git = "...", rev = "..." }`
  - alpha 期間は API 不安定なので **rev pin 必須**
  - 自分の検証が済んだ rev に固定し、定期的に手動 bump

## Consequences

### vendor (bullet-shogi)

- 利点: 自由に編集できる、外部依存ゼロ、CI 高速、bullet-shogi の API 変動から独立
- 欠点: 上流の bug fix を手動で sync (frequency: 半年〜年単位で十分)
- 将棋データ format は安定しているのでコストは小さい

### git dep (cuda-oxide)

- 利点: 上流の改善を rev bump で取り込める、vendor 不可能 (build process 全体に乗っている)
- 欠点: alpha 期は API 破壊が起こり得る → rev pin で局所化
- `crates/rustc-codegen-cuda` は cuda-oxide の build process に乗る (cargo subcommand `cargo oxide` 経由)

### Pliron (transitive)

- cuda-oxide が pin している rev を継承
- 直接触る予定無し
