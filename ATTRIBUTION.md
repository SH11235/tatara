# Attribution

このリポジトリは以下のオープンソースプロジェクトから派生・参照しています。
各プロジェクトの著作権表示とライセンスは原本に従い保持されます。

## bullet-shogi / bullet (MIT)

- bullet-shogi: <https://github.com/SH11235/bullet-shogi> (jw1912/bullet の将棋向け fork)
- bullet (upstream): <https://github.com/jw1912/bullet>
- License: MIT

NNUE 学習器のアルゴリズムは bullet-shogi / bullet から移植しています。

## cuda-oxide (Apache-2.0)

- Source: <https://github.com/NVlabs/cuda-oxide>
- License: Apache-2.0
- 取り込み方: `Cargo.toml` の git dep + commit rev pin (vendor せず)。GPU
  kernel を build-time に PTX 化する rustc backend として `crates/gpu-runtime`
  と GPU 依存 bin が `cuda-core` / `cuda-host` / `cuda-device` を参照します。

## Pliron (Apache-2.0)

- Source: <https://github.com/vaivaswatha/pliron>
- License: Apache-2.0
- 取り込み方: cuda-oxide が依存する transitive crate。

## ライセンス互換性

本リポジトリ自体は MIT。MIT は Apache-2.0 由来コードを含むコンパイル
バイナリ配布と互換です。ソース配布時は各依存の `LICENSE` を保持してください。
