# YaneuraOu 用 LayerStack net 変換

`net_to_yo` は tatara の 9 bucket LayerStack `.bin` を YaneuraOu の SFNN 評価
ファイルへ変換する。feature set と FT 出力 / 隠れ層の次元は入力 `.bin` の
`arch_str` から自動検出するため、追加の指定は要らない。

```bash
cargo run --release -p net-to-yo -- \
  --input /path/to/tatara.bin \
  --output /path/to/eval/nn.bin \
  --assume-kingrank9
```

## 対応する feature set と次元

tatara の 5 feature set はいずれも YaneuraOu SFNN feature と同一の
Apery-BonaPiece index 規約で恒等一致するため、重みを並べ替えずそのまま移送できる。

| tatara feature set | YaneuraOu feature | `nnue_arch_gen.py` キー |
|---|---|---|
| `HalfKp` | `HalfKP(Friend)` | `halfkp` |
| `HalfKaSplit` | `HalfKA1(Friend)` | `halfka1` |
| `HalfKaMerged` | `HalfKA2(Friend)` | `halfka2` |
| `HalfKaHmSplit` | `HalfKA_hm1(Friend)` | `halfkahm1` |
| `HalfKaHmMerged` | `HalfKA_hm2(Friend)` | `halfkahm2` |

FT 出力次元 (`ft_out`)・L1 出力 (`l1_out`)・L2 出力 (`l2_out`) は任意で、YaneuraOu 側
は対応するアーキで build する (`YANEURAOU_ENGINE_SFNN_<key>_<ft_out>_<l1_out-1>_<l2_out>_k3k3`)。
`HalfKaHmMerged` の 1536-16-32 だけは YaneuraOu の既定 `SFNN-1536` 構造名を出力し、
それ以外は生成器と同じ `SFNN_<key>_<ft_out>_<l1_out-1>_<l2_out>_k3k3` 名を出力する。

## 変換できない入力

YaneuraOu SFNN に受け皿が無いため、次を含む `.bin` は明示的にエラーにする。

- PSQT / Threat / EffectBucket block を持つ net (`arch_str` に該当トークンがある)
- 9 以外の bucket 数 (YaneuraOu SFNN は KingRank9 = 9 bucket 固定)

量子化 `.bin` は bucket routing mode を記録しないため、変換前に学習時の
`--bucket-mode kingrank9` を確認し、`--assume-kingrank9` で明示する。既定の
`progress8kpabs` で学習した 9 bucket net は、YaneuraOu と bucket の選択規則が
異なるため変換できない。

前提として、YaneuraOu 側は `DISTINGUISH_GOLDS` 無効 (既定) で build する。有効
build は成駒を別 plane に置き feature 次元が変わるため index が一致しない。

## ファイル形式

YaneuraOu SFNNwoPSQT loader が要求する 4 つのハッシュ (version / top-level /
feature-transformer / network) は feature set・次元に依らず固定定数で、この値を
書き出す。version 以外の不一致は YaneuraOu 側で警告扱いになるが、ここでは生成済み
YaneuraOu ビルドと byte 一致する値を出力する。

FT の bias と weight はそれぞれ signed LEB128 block、dense 層は bias の i32 LE、
続いて canonical row-major (32 境界へ 0 padding) の i8 weight を読む。dense weight
と FT weight は YaneuraOu がロード時に実行用 SIMD layout へ並べ替えるため、変換
ファイルには並べ替え前の順序で格納する。

量子化 scale は両形式とも FT が QA=127、dense weight が QB=64、dense bias が
QA×QB=8128 であり、変換時の scale 変更は行わない。architecture string に
`fv_scale` は含めない。YaneuraOu では読み込み前に `setoption name FV_SCALE value 28`
を指定する。
