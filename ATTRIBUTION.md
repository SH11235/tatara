# Attribution

このリポジトリは以下のオープンソースプロジェクトから派生・参照しています。
各プロジェクトの著作権表示とライセンスは原本に従い保持されます。

## bullet-shogi / bullet (MIT)

- bullet-shogi: <https://github.com/SH11235/bullet-shogi> (jw1912/bullet の将棋向け fork)
- bullet (upstream): <https://github.com/jw1912/bullet>
- License: MIT

**派生範囲**: 以下のファイル群は bullet-shogi / bullet からの移植 (vendor) を含み、
オリジナルのアルゴリズム選択 / 数式 / 定数の出典が bullet にあります。

- `crates/shogi-format/` — PackedSfenValue (PSV) reader、ShogiBoard / Hand 型、
  Stockfish 系の `bona_piece` 定数
- `crates/shogi-features/` — HalfKA_hm 特徴抽出と progress8kpabs バケット定義
- `crates/gpu-kernels/` — pointwise / sparse / layerstack カーネルの hand-fused
  実装。`loss_wdl` / `loss_wrm` / `ranger_step` / `radam_step` / `adamw_step` /
  `screlu_grad` / sparse FT forward/backward / dense_mm bucket variants 等の
  数式と定数 (WRM `in_scaling=380`, offset `270` 等の bullet ハードコード値含む)
- `crates/nnue-format/src/layerstack_weights.rs` — LayerStack 量子化 binary
  format (`QA=127 / QB=64 / FV_SCALE=28`、`FC_HASH` の compute 規則)
- `crates/nnue-train/src/trainer.rs` — superbatch training loop と
  `--score-drop-abs` / WDL blend / Ranger lookahead の挙動

具体的な対応関係 (どの kernel が bullet のどの関数を hand-fuse したか) は各
source ファイルの module doc コメントに記載しています。

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
