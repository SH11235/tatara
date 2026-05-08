# Attribution

このリポジトリは以下のオープンソースプロジェクトから派生・参照しています。

## bullet-shogi (MIT)

- Source: https://github.com/SH11235/bullet-shogi
- Upstream: https://github.com/jw1912/bullet
- Use: PSV reader、ShogiBoard / Hand 等の format 周りを vendor (Stage 1〜)
- License: MIT

### 取り込み済 file (時系列で追記)

- _Stage 1 で更新予定_

## cuda-oxide (Apache-2.0)

- Source: https://github.com/NVlabs/cuda-oxide
- Use: GPU kernel を build-time に PTX 化 (host 側 wrapper も含む)
- License: Apache-2.0
- Dependency style: `Cargo.toml` の git dep + rev pin (vendor せず)

## Pliron (Apache-2.0)

- Source: https://github.com/vaivaswatha/pliron
- Use: cuda-oxide が依存 (transitive)
- License: Apache-2.0

## ライセンス互換性メモ

本リポジトリ自体は MIT。MIT は Apache-2.0 由来コードを含むコンパイル
バイナリ配布と互換。ソース配布時は各依存の `LICENSE` を保持する。
