# アーキテクチャ記述文字列 (arch string)

量子化 NNUE binary (`*.bin`) の header に埋め込まれる、ネットワーク構造を人間可読
かつ機械照合可能に表す文字列。本ドキュメントは arch string の組み立てロジックと
構成別の実例をまとめる。

## 役割

- 量子化 binary が「どのアーキ・どの feature set・どの層次元・どの活性化」で学習
  されたかを self-describing に記録する。
- load 時に、要求するネットワーク spec から同じ手順で arch string を組み立て直し、
  ファイル内の文字列と照合する。不一致なら別物の weight を取り違えて読み込む前に
  reject する (`load` / `load_quantised` の reject 契約)。

組み立ては `crates/nnue-format` の 2 箇所:

- `layerstack_weights::build_arch_str` — N-bucket (`--num-buckets`、既定 9) LayerStack アーキ
- `simple_weights::build_arch_str` — bucket 無し Simple 4 層アーキ

## 全体構造

arch string は 3 部をカンマで連結した 1 行:

```
Features=<...>,Network=<...>,fv_scale=<N>
```

LayerStack で `--psqt` を有効にした場合のみ、Features と Network の間に
`PSQT=<num_buckets>,` トークンが挿入される (`.bin` 内の PSQT block の存在と
bucket 数を自己記述する):

```
Features=<...>,PSQT=<N>,Network=<...>,fv_scale=<N>
```

EffectBucket feature set では、Features と Network の間に `EffectBucket=<config>,`
トークンが挿入される。`<config>` は `2x2fixed` / `2x2bucketed` /
`3x3fixed` / `3x3bucketed` のいずれかで、bucket 数と玉 feature の bucket 化有無を
表す:

```
Features=<...>,EffectBucket=2x2fixed,Network=<...>,fv_scale=<N>
```

### Features トークン

特徴変換器 (FT) の入出力を表す:

```
Features=<feature_name>(Friend)[<input_size>-><ft_out>x2]
```

- `<feature_name>`: feature set のアーキ名 (`HalfKaHmMerged` / `HalfKP` など)。
- `(Friend)`: FT を視点ごとに適用することを示す marker。
- `[<input_size>-><ft_out>x2]`: FT 入力次元 → FT 出力次元。`x2` は手番側 (stm) と
  相手側 (nstm) の 2 視点ぶんの FT 出力を concat することを表す。

Simple の `--activation pairwise` のみ FT ブロックが別形式になる:

```
Features=<feature_name>(Friend)[<input_size>-><ft_out>/2x2]-Pairwise
```

pairwise 乗算で FT 出力が半減することを `/2` と `-Pairwise` suffix で表し、
推論エンジンはこの suffix で pairwise を識別する
(`simple_weights::arch_identity`)。

### Network 式

層スタックを **入れ子の関数適用**として、最内 (FT 出力) から最外 (スカラ出力) へ
向けて書く。最内が `InputSlice`、それを `AffineTransform` と活性化トークンが順に
包み、最外の `AffineTransform[1<-...]` が 1 次元のネットワーク出力を作る。

| トークン | 意味 |
|---|---|
| `InputSlice[<d>(0:<d>)]` | FT 出力 (`<d>` = `ft_out × 2`) をネットワーク入力として切り出す |
| `AffineTransform[<out><-<in>](...)` | dense (affine) 層: `<in>` 入力 → `<out>` 出力 |
| `AffineTransformSparseInput[<out><-<in>](...)` | sparse 入力版の dense 層 (FT 出力を直接受ける第 1 dense 層) |
| `ClippedReLU[<d>](...)` | clipped ReLU 活性化 (`<d>` 要素) |
| `SqrClippedReLU[<d>](...)` | squared clipped ReLU 活性化 (`<d>` 要素) |

### fv_scale

`fv_scale=<N>` は推論時に評価値スケールへ戻す係数 (`round(127 × QB / 学習 scale)`。
活性化出力が常に 127-scale のため活性化非依存)。
学習の `--scale` 由来なので、同じ topology でも学習設定が違えば値が変わる。この
ため **identity 照合には含めない** (後述)。

## LayerStack の例

`HalfKaHmMerged` feature set・FT 出力 1536・`fv_scale=28` の場合:

```
Features=HalfKaHmMerged(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-30](SqrClippedReLU[30](AffineTransform[16<-3072](InputSlice[3072(0:3072)]))))),fv_scale=28
```

最内から読むと:

| トークン | 内容 |
|---|---|
| `InputSlice[3072(0:3072)]` | FT 出力 3072 = 1536 × 2 視点 |
| `AffineTransform[16<-3072]` | 第 1 dense 層、3072 → 16 |
| `SqrClippedReLU[30]` | 30 要素の squared clipped ReLU |
| `AffineTransform[32<-30]` | 30 → 32 |
| `ClippedReLU[32]` | 32 要素の clipped ReLU |
| `AffineTransform[1<-32]` | 出力層、32 → 1 |

LayerStack の arch string は dense 層チェーンの要約で、L1f skip 接続や
pairwise・per-bucket 構造は文字列に現れない (bucket 数は `--psqt` 有効時の
`PSQT=<N>` トークンにのみ現れる)。EffectBucket feature set の config は
`EffectBucket=<config>` トークンに現れる。LayerStack の完全なアーキ記述は
`crates/nnue-format/src/layerstack_weights.rs` の module doc を参照。

## Simple の例

Simple は bucket 無しの 4 層アーキ (FT + dense 3 層) で、層次元が文字列上の dense 層チェーンと
そのまま対応する。`HalfKaHmMerged` feature set・FT 出力 256・隠れ層 32/32 の場合。

活性化 `crelu` (`fv_scale=13`):

```
Features=HalfKaHmMerged(Friend)[73305->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransformSparseInput[32<-512](InputSlice[512(0:512)]))))),fv_scale=13
```

活性化 `screlu` (`fv_scale=27`):

```
Features=HalfKaHmMerged(Friend)[73305->256x2],Network=AffineTransform[1<-32](SqrClippedReLU[32](AffineTransform[32<-32](SqrClippedReLU[32](AffineTransformSparseInput[32<-512](InputSlice[512(0:512)]))))),fv_scale=27
```

最内から読むと (crelu の例):

| トークン | 内容 |
|---|---|
| `InputSlice[512(0:512)]` | FT 出力 512 = 256 × 2 視点 |
| `AffineTransformSparseInput[32<-512]` | L1 dense、512 → 32 |
| `ClippedReLU[32]` | L1 出力の活性化 |
| `AffineTransform[32<-32]` | L2 dense、32 → 32 |
| `ClippedReLU[32]` | L2 出力の活性化 |
| `AffineTransform[1<-32]` | L3 出力層、32 → 1 |

Simple 固有の点:

- 活性化トークンは `--activation` で決まる。`crelu` → `ClippedReLU`、`screlu` →
  `SqrClippedReLU`。LayerStack と違い 2 つの活性化位置とも同じトークンになる。
  `pairwise` は dense 層を CReLU で活性化するため Network 式のトークンは
  `ClippedReLU` のままで、Features トークンの `-Pairwise` suffix だけが弁別点
  (上記「Features トークン」参照)。
- FT 直後の第 1 dense 層を `AffineTransformSparseInput` と書く (LayerStack は同位置を
  `AffineTransform` と書く)。

## load 時の identity 照合

`,fv_scale=` の手前までが **identity 部**で、feature set 名・FT 入出力次元・dense
各層の次元・活性化を含む。load は要求する spec から同じ手順で identity 部を組み立て、
ファイル内の文字列と照合する:

- LayerStack: `Features=...` トークンの前方一致 + network hash で照合する。
- Simple: identity 部全体の一致で照合する。

`fv_scale` は学習 `--scale` 由来で同一アーキでも変動するため、identity 照合の対象に
しない (ファイルには記録するが照合では無視する)。

## 関連

- `crates/nnue-format/src/layerstack_weights.rs` — LayerStack の `build_arch_str` と
  load reject 契約
- `crates/nnue-format/src/simple_weights.rs` — Simple の `build_arch_str` と
  load reject 契約
