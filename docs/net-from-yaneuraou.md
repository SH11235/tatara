# YaneuraOu SFNN net の tatara への逆変換

`net_from_yo` は YaneuraOu の SFNN 評価ファイル (nn.bin) を tatara の 9 bucket
LayerStack `.bin` へ逆変換する (`net_to_yo` の逆)。feature set と FT 出力 / 隠れ層の
次元は YaneuraOu の arch 文字列から自動検出するため、追加の指定は要らない。

```bash
cargo run --release -p net-from-yo -- \
  --input /path/to/yaneuraou/nn.bin \
  --output /path/to/tatara.bin \
  --assume-kingrank9
```

## 対応する feature set と次元

`net_to_yo` と対称に、YaneuraOu SFNN の 5 feature set をいずれも tatara feature set
へ逆写像する (index は同一 Apery-BonaPiece 規約で恒等一致、並べ替え不要)。

| YaneuraOu feature | tatara feature set |
|---|---|
| `HalfKP(Friend)` | `HalfKp` |
| `HalfKA1(Friend)` | `HalfKaSplit` |
| `HalfKA2(Friend)` | `HalfKaMerged` |
| `HalfKA_hm1(Friend)` | `HalfKaHmSplit` |
| `HalfKA_hm2(Friend)` | `HalfKaHmMerged` |

FT 出力・隠れ層次元は arch 文字列から検出する。`Features=<name>(Friend)[<in>-><ft>x2]`
から feature と FT 出力を、`Network=` から L1/L2 出力を取る (既定 `SFNN-1536`(-V2) は
`HalfKA_hm2`/1536 専用で L1=16/L2=32、それ以外は生成器同形の
`SFNN_<key>_<ft>_<h1>_<h2>_k3k3` から `l1_out = h1 + 1`, `l2_out = h2`)。

## 変換できない入力

- version / top-level / feature-transformer / 各 LayerStack ハッシュのいずれかが
  YaneuraOu SFNNwoPSQT の固定値と一致しない (SFNN 以外・破損ファイル)
- `ModelType=SFNNWithoutPsqt` でない (PSQT 付き) / 未知 feature
- LayerStack が 9 でない (YaneuraOu SFNN KingRank9 は常に 9)
- Network 名の FT 次元が Features の FT 次元と食い違う、baseline 名なのに
  `HalfKA_hm2`/1536 でない等、arch 内の不整合
- FT 出力が 32 の倍数でない / 上限超過等、健全性ガード範囲外

量子化 `.bin` は bucket routing mode を記録しないため、変換前に元 net が KingRank9
routing であることを確認し、`--assume-kingrank9` で明示する。生成器の Network 名
(`SFNN-1536` / `*_k3k3`) はいずれも 3x3=9 bucket を意味するが、routing 規則自体は
ファイルから判別できないため、フラグでの明示を必須にしている。

## ファイル形式

YaneuraOu SFNNwoPSQT の 4 ハッシュ (version / top-level / feature-transformer /
network) は feature set・次元に依らず固定定数で、これらを検証する。FT は
signed LEB128 block (`COMPRESSED_LEB128` magic、実 YaneuraOu net では先頭に任意の
ASCII marker `_` が付く)、dense 層は bias の i32 LE、続いて 32 境界へ 0 padding した
canonical row-major i8 weight を読む。dense weight は YaneuraOu が実行用 SIMD layout へ
並べ替える前の順序で格納されているため、逆量子化してそのまま取り込む。

量子化 scale は FT が QA=127、dense weight が QB=64、dense bias が QA×QB=8128。
逆量子化は `net_to_yo` の量子化と対称で、量子化格子上の値は厳密に復元される
(tatara → `net_to_yo` → `net_from_yo` の往復は byte 完全一致)。
