# ADR-0001: License を MIT にする

- **Status**: Accepted
- **Date**: 2026-05-09

## Context

新規リポは vendor 元の `bullet-shogi` (MIT) と、中核技術 `cuda-oxide`
(Apache-2.0) を取り込む構成になる。配布形態として

- ソース配布
- ビルド済みバイナリ (`bins/`) の配布
- 将来 crates.io 公開する可能性

を想定するため、license の選択を上記すべてと整合させたい。

## Decision

本リポジトリ自体は **MIT** とする。

- bullet-shogi 由来の vendor コードは MIT のまま
- cuda-oxide 由来は Apache-2.0 (依存として取り込み、再配布は dependency
  ツリーの一員として扱う)
- 個人実験リポだが商用利用も含めて寛容な license にしておく

## Consequences

- vendor 取り込み時には元 file 冒頭の copyright 表記をそのまま保持し、
  `ATTRIBUTION.md` で出所を明示する。
- cuda-oxide の Apache-2.0 NOTICE はソース配布時に保持する必要がある
  (現状は git dependency 経由なので `Cargo.lock` で参照されるのみ)。
- ライセンス互換性メモは `ATTRIBUTION.md` に残す。
