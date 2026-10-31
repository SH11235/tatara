# YaneuraOu SFNN net の tatara への逆変換

`net_from_yo` は YaneuraOu の SFNN 評価ファイル (nn.bin) を tatara の
LayerStack `.bin` へ逆変換する (`net_to_yo` の逆)。feature set と FT 出力 / 隠れ層の
次元は YaneuraOu の arch 文字列から自動検出するため、追加の指定は要らない。

```bash
cargo run --release -p net-from-yo -- \
  --input /path/to/yaneuraou/nn.bin \
  --output /path/to/tatara.bin \
  --assume-kingrank9   # progress kp abs の net は --assume-progresskpabs
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
`SFNN_<key>_<ft>_<h1>_<h2>[_k3k3 | _progress<N>]` から `l1_out = h1 + 1`, `l2_out = h2`)。
LayerStack 数は末尾 token から決まり、`_k3k3` は 9、`_progress<N>` は N、token なしは 1。

## 変換できない入力

- version / top-level / feature-transformer / 各 LayerStack ハッシュのいずれかが
  YaneuraOu SFNNwoPSQT の固定値と一致しない (SFNN 以外・破損ファイル)
- `ModelType=SFNNWithoutPsqt` でない (PSQT 付き) / 未知 feature
- `LayerStack=<n>` が Network 名の末尾 token の示す数と食い違う、`_k3k3_progress<N>` の
  ような複合 token、hand / 玉位置 9x9 等の他の token
- routing の主張と合わない (`--assume-kingrank9` に既定名 `SFNN-1536`(-V2) と `_k3k3` 以外、
  `--assume-progresskpabs` に 10 以上の LayerStack)
- Network 名の FT 次元が Features の FT 次元と食い違う、baseline 名なのに
  `HalfKA_hm2`/1536 でない等、arch 内の不整合
- FT 出力が 32 の倍数でない / 上限超過等、健全性ガード範囲外

YaneuraOu の派生実装には、9 stack の `_k3k3` 名の評価ファイルを読み、先頭 N 枠を進行度で
選んで使う build がある。このため Network 名だけでは routing を判別できず、
`--assume-kingrank9` か `--assume-progresskpabs` のどちらか一方の明示を必須にしている。
`--assume-progresskpabs` では `_progress<N>`・token なし・9 stack の `_k3k3` を受理し、
ファイルにある LayerStack 数のまま書き出す。進行度係数 (YaneuraOu では `nn.bin` と同じ
directory の `progress.bin`) と、9 stack の `_k3k3` 名で書かれた net の元の progress bucket 数は
ファイルに記録されない。tatara のツールは保存された bucket 数で進行度を等分するため、
後者の net は tatara では元の評価を再現できない (rshogi では bucket 数を別に指定できる)。
詳細は [tatara ↔ YaneuraOu 変換](net-to-yaneuraou.md) の「bucket routing」節。

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
