# tatara ↔ YaneuraOu LayerStack net 変換

LayerStack net を tatara と YaneuraOu SFNN の間で相互変換する 2 つのツール:
`net_to_yo` (tatara → YaneuraOu) と `net_from_yo` (YaneuraOu → tatara)。どちらも
feature set と FT 出力 / 隠れ層の次元を入力の `arch_str` から自動検出するため、
追加の指定は要らない。対応 feature set / 次元とファイル形式は両ツール共通で、以下の
各節にまとめる。

## `net_to_yo` — tatara → YaneuraOu

tatara の LayerStack `.bin` を YaneuraOu の SFNN 評価ファイルへ変換する。routing の
flag は下の「bucket routing」節を参照。

```bash
cargo run --release -p net-to-yo -- \
  --input /path/to/tatara.bin \
  --output /path/to/eval/nn.bin \
  --assume-kingrank9   # progresskpabs で学習した net は --assume-progresskpabs
```

## `nnue-train` から直接出力

LayerStack を KingRank9 で学習する場合は、推論用 checkpoint を最初から YaneuraOu
SFNN 形式で出力できる。この出力は、同じ weights を tatara `.bin` に export してから
`net_to_yo` で変換した出力と byte 一致する。

```bash
cargo run --release -p nnue-trainer -- \
  --data /path/to/training.psv \
  --output /path/to/checkpoints \
  --output-format yaneuraou \
  layerstack \
  --bucket-mode kingrank9
```

`--output-format` の既定値は `tatara`。`yaneuraou` は plain LayerStack と KingRank9 の
組合せだけを受理し、`progresskpabs`、Simple、PSQT、Threat、EffectBucket は明示的に
エラーにする。学習 config から bucket routing mode を判定できるため、この経路では
`--assume-kingrank9` は不要。`progresskpabs` で学習した net は tatara `.bin` で出力し、
`net_to_yo --assume-progresskpabs` で変換する。

## `net_from_yo` — YaneuraOu → tatara

`net_to_yo` の逆変換。YaneuraOu 形式で export された SFNN 評価ファイルを tatara の
quantised LayerStack `.bin` へ変換する。他の学習器が YaneuraOu 形式で出力した SFNN net
を tatara / rshogi でロードして評価・対局するのに使う。

```bash
cargo run --release -p net-from-yo -- \
  --input /path/to/eval/nn.bin \
  --output /path/to/tatara.bin \
  --assume-kingrank9   # progress kp abs の net は --assume-progresskpabs
```

feature set と次元は入力の `arch_str` (`ModelType=SFNNWithoutPsqt;Features=<name>…`)
から検出し、既定の `SFNN-1536-V2` と生成名の両綴りを受理する。`SFNNWithoutPsqt` で
ない (PSQT 付き)・未知 `Features`・`LayerStack` 数が Network 名の layer stack token と
食い違う入力は reject する。routing の flag は下の「bucket routing」節を参照。FT block の先頭が `_` 始まり (`_COMPRESSED_LEB128`) の入力も
受理する (先頭の `_` を 1 byte 読み飛ばし、共通の `COMPRESSED_LEB128` magic から読む)。
dense weight の 32 境界 padding 列は復元時に除去する。`net_to_yo` と同一の量子化・
レイアウト規則を用いるため、現行 format かつ `fv_scale=28` の `.bin` については
`tatara → net_to_yo → net_from_yo` の round-trip が元と byte 一致する。`fv_scale` を
省略した入力や 28 以外にした入力は、YaneuraOu SFNN 形式に値を保持する場所がないため、
`net_from_yo` が `fv_scale=28` を付けて書き戻す。legacy version header の `.bin` も
現行 header で書き直すため一致しない。

## bucket routing

量子化 `.bin` も YaneuraOu SFNN ファイルも bucket の選択規則を完全には自己記述しないため、
変換時は学習時の `--bucket-mode` に合わせて次のどちらか一方を必ず指定する。

| 学習時の `--bucket-mode` | flag | YaneuraOu の Network 名 | LayerStack 数 |
|---|---|---|---|
| `kingrank9` | `--assume-kingrank9` | `SFNN-1536` / `SFNN_<key>_<ft>_<h1>_<h2>_K3K3` | 9 |
| `progresskpabs` | `--assume-progresskpabs` | `SFNN_<key>_<ft>_<h1>_<h2>_PROGRESS<N>` (N=1 は suffix なし) | N (1〜9) |

progress kp abs の評価ファイルを YaneuraOu で使うときは、学習に使った進行度係数
(`progress.bin`、[進行度係数の用意](progress-bin.md)) を `nn.bin` と同じ directory に置き、
`YANEURAOU_ENGINE_SFNN_<key>_<ft>_<h1>_<h2>_progress<N>` で build する。YaneuraOu の
アーキ生成器が受理する progress bucket 数は 2 / 3 / 4 / 8 / 16 / 32 で、tatara の 2〜9 との
共通部分 (2 / 3 / 4 / 8) 以外は YaneuraOu 側に対応 build が無い。N=1 は suffix なしの
`YANEURAOU_ENGINE_SFNN_<key>_<ft>_<h1>_<h2>` (単一 LayerStack) で使え、`progress.bin` は要らない。
進行度の計算方法 (f32 か、係数と閾値を 65536 倍した整数か) は YaneuraOu の build によって
異なり、整数で計算する build では bucket 境界ちょうど付近の局面で tatara の f32 計算と
選ぶ bucket が異なり得る。

`_progress<N>` build は N stack だけを読み、残りがあるファイルはロードに失敗する。一方、
YaneuraOu の派生実装には、9 stack の `_K3K3` 名の評価ファイルを読み、先頭 N 枠を進行度で
選ぶ build もある。`net_from_yo --assume-progresskpabs` はこの形式も受理し、9 bucket の
`.bin` として書き出す。このとき元の progress bucket 数 N はどちらのファイルにも記録されない。
tatara の評価・学習ツールは保存された bucket 数 (9) で進行度を等分するため、N < 9 の net では
選ぶ bucket が元の build と変わり、元の評価を再現できない。rshogi では `LS_PROGRESS_BUCKETS=<N>`
で保存数より少ない bucket 数を指定でき、`LS_BUCKET_MODE=progresskpabsq16` で整数の進行度
計算を選べる。

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
は対応するアーキで build する (`YANEURAOU_ENGINE_SFNN_<key>_<ft_out>_<l1_out-1>_<l2_out>_k3k3`、
progress kp abs は末尾が `_progress<N>`)。KingRank9 の `HalfKaHmMerged` 1536-16-32 だけは
YaneuraOu の既定 `SFNN-1536` 構造名を出力し、それ以外は生成器と同じ名前を大文字で出力する。

## 変換できない入力

YaneuraOu SFNN に受け皿が無いため、次を含む `.bin` は明示的にエラーにする。

- PSQT / Threat / EffectBucket block を持つ net (`arch_str` に該当トークンがある)
- routing と合わない bucket 数 (`--assume-kingrank9` は 9、`--assume-progresskpabs` は 1〜9)

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

## FV_SCALE

量子化 scale は両形式とも FT が QA=127、dense weight が QB=64、dense bias が
QA×QB=8128 であり、変換時の scale 変更は行わない。YaneuraOu SFNN の architecture
string には `fv_scale` を保持する場所がないため、`net_to_yo` は値を移送しない。

tatara の LayerStack export は、`--fv-scale N` 指定時は `fv_scale=N` を書く。未指定の
plain sigmoid-MSE 学習では `round(QA×QB / --scale)` を書き、既定の `--scale 290`
なら 28 になる。WRM 学習では、評価値の換算係数が loss だけから一意に決まらないため
`fv_scale` token を省略する。値の確定後は `net-fv-scale` で tatara `.bin` の arch
string に直接書き込める。

`nnue-train --output-format yaneuraou` でも YaneuraOu SFNN ファイル内に `fv_scale` は
保存されないため、`--fv-scale` の有無は出力 byte に影響しない。評価時は YaneuraOu の
`FV_SCALE` option を目的の値に設定する。

```bash
cargo run --release -p net-fv-scale -- \
  --input /path/to/input.bin \
  --output /path/to/output.bin \
  --fv-scale N
```

token を削除する場合は `--fv-scale N` の代わりに `--remove` を指定する。`.bin` を
書き換えない場合は、読み込み前に YaneuraOu または rshogi へ
`setoption name FV_SCALE value N` を指定する必要がある。rshogi は token がなくても
エラーにせず `FV_SCALE_HALFKA=16` へ fallback するため、値を指定し忘れると評価値が
意図せず raw score ÷ 16 になる。`net_from_yo` は YaneuraOu SFNN から値を復元できない
ため、生成する tatara net には従来値の `fv_scale=28` を書く。
