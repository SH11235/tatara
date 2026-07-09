# NNUE 入力 feature-set の 2 軸モデルと公開 5 cell

- **Status**: Accepted
- **Date**: 2026-05-19
- **設計レビュー**: Codex / Claude 両者 APPROVE

## Context

tatara は NNUE 入力 feature set として HalfKA_hm (merged-king-plane,
73,305 次元) 1 種類を compile-time const で固定していた。総入力次元 `FT_IN` /
最大 active 数 `MAX_ACTIVE` / feature hash が `shogi-features` /
`nnue-format` / `gpu-kernels` / `bins/nnue_train` に分散定数として埋まっている。

bullet-shogi 同様に HalfKP / HalfKA / HalfKA_hm を実行時選択したい。さらに
Stockfish の HalfKA → HalfKAv2 に当たる「玉 plane を 2 枚持つか 1 枚に畳むか」
のバリエーションも扱いたい。Stockfish 由来の feature-set 系譜は次の 2 つの
独立した違いに分解できる:

| 違い | 内容 |
|---|---|
| 玉の特徴化 | 玉を特徴に含めない (HalfKP) / 両玉を別 plane (HalfKA) / 両玉を 1 plane に畳む (HalfKAv2) |
| 玉マス処理 | 玉マスを全 81 マスで使う / 左右ミラーで 45 バケットに圧縮 (HalfKAv2_hm) |

HalfKA / HalfKAv2 / HalfKAv2_hm の **いずれも両玉の位置を active feature として
emit する**。HalfKA → HalfKAv2 の差は「玉を 1 つ捨てる」ことではなく、玉用の
piece-input ordinal を 2 枚持つか 1 枚に畳むかという次元上の違いである。

## Decision

### 内部表現は独立 2 軸モデル

feature set を 2 つの直交する軸で内部表現する (`feature_set.rs`):

- **軸 1 — 玉特徴エンコード** (`KingEncoding`): `NoKing` (玉を特徴化しない) /
  `SplitPlane` (玉 plane 2 枚) / `MergedPlane` (玉 plane 1 枚、敵玉 BonaPiece を
  81 引いて自玉 plane へ重ねる)。
- **軸 2 — 玉マス処理** (`KingSquareMode`): `Direct` (81 bucket) /
  `HorizontalMirror` (45 bucket、6-9 筋を 1-4 筋へ反転し盤駒も筋ミラー)。

**2 軸が直交する根拠**: 軸 2 はマス座標 (玉マス→bucket、盤駒マス→筋ミラー) に
作用し、軸 1 は玉 BonaPiece の plane base に作用する。両者は作用対象が異なる
ため合成しても干渉しない。これにより N 個の feature set モジュールを複製せず、
**1 本のパラメタライズド indexer**で全 cell を扱える (盤駒・手駒走査の大半は
feature set 間で共通で、差分は玉 plane 畳み込み・筋ミラー・玉特徴化の有無のみ)。

### 公開 API は閉じた 5-variant enum (二層構造)

公開層 (`FeatureSet` enum) と内部層 (2 軸) を分ける:

| canonical 名 | 軸 1 | 軸 2 | king bucket | piece inputs | ft_in | 参照実装 |
|---|---|---|---|---|---|---|
| `halfkp` | NoKing | Direct | 81 | 1548 | 125,388 | bullet-shogi `ShogiHalfKP` |
| `halfka-split` | SplitPlane | Direct | 81 | 1710 | 138,510 | bullet-shogi `ShogiHalfKA` |
| `halfka-merged` | MergedPlane | Direct | 81 | 1629 | 131,949 | 無し (本リポ新規定義) |
| `halfka-hm-split` | SplitPlane | HorizontalMirror | 45 | 1710 | 76,950 | 無し (本リポ新規定義) |
| `halfka-hm-merged` | MergedPlane | HorizontalMirror | 45 | 1629 | 73,305 | bullet-shogi `ShogiHalfKA_hm` |

CLI・artifact header は `FeatureSet` の 5 variant だけを扱う。2 軸の組み合わせ
としては `HalfKP_hm` 等も表現できるが、**公開しない・サポート保証もしない**。
無効・非保証な組み合わせを公開 enum として表現不能にすることで、サポート範囲を
型レベルで閉じ、曖昧さを排除する。

### `FeatureSetSpec` を feature 軸の単一の真実源にする

feature set 1 つを完全に記述する runtime spec を `FeatureSetSpec` に集約する。
生成は `FeatureSet::spec` のみ、フィールドは private。spec を後から変形する
経路は modifier method (`with_ft_factorize`、設計は
[2026-06-12-ft-factorizer.md](2026-06-12-ft-factorizer.md)) に限り、modifier は
2 軸と直交して base 次元 getter を変えない (学習側 getter `train_*` のみ分岐)。**WHY**: 次元・hash が
複数 crate に分散定数として埋まると個別箇所が再計算してズレる。spec 1 つを
CLI パース直後に確定し、以降の全層 (dataloader / trainer / export / checkpoint)
が同じ spec を参照する。

`map_features_board` (decode 済み `ShogiBoard` 直受け) を基本の入口にする。
**WHY**: dataloader は 1 局面につき `decode` を 1 回だけ呼び、得た `ShogiBoard`
を特徴抽出と progress bucket 計算で共有する。`PackedSfenValue` 直受けの入口
だけにすると decode が hot path で 2 回走る。

2 軸の解釈 (king bucket / 筋ミラー要否) は 1 局面につき 1 回だけ
`PerspectiveCtx` に畳み、内側の駒走査ループには分岐を持ち込まない。

### 命名は content 語

「玉 plane 2 枚 / 1 枚」を `split-king-plane` / `merged-king-plane` と呼ぶ。
**WHY**: CLAUDE.md は外部バージョンタグを識別子に使うことを禁止する。加えて
Stockfish 由来の「v1 / v2」を「両玉 / 片玉」と捉えるのは不正確で、merged も
両玉を emit する (plane 数の違いに過ぎない)。CLI option 名は `--feature-set`
とする (`--features` は Cargo の feature flag と紛らわしい)。

### 参照実装なし 2 cell の正当性

`halfka-merged` / `halfka-hm-split` は bullet-shogi も Stockfish も持たない
組み合わせで、2 軸の直交性から導出した cell である。「直交だから掛け算で
導出した cell も自明に正しい」とはせず、第一原理で index 仕様を定義し検証する:

- index 範囲 / injective (玉 plane 畳み込み後の自玉・敵玉衝突を含む)。
- 退化一致 — `halfka-merged` は `halfka-split` に敵玉 fold だけを加えた cell、
  `halfka-hm-split` は `halfka-hm-merged` から fold を除いた cell であることを
  特徴ごとに照合し、参照照合済みの cell に紐づける。
- semantic 不変条件 — perspective swap で stm/nstm ペアが整合して入れ替わる、
  `HorizontalMirror` 系では盤ごと筋ミラーした像が同一 index 集合を返す。

### 回帰保証

`halfka-hm-merged` の出力 index は現行 `ShogiHalfKA_hm` と emission 順序含め
bit-identical でなければならない (既存学習資産・checkpoint 互換)。

## Rejected alternatives

- **公開 API も 2 軸 (補助フラグ `--king-plane` 等で軸を個別指定)**: 無効な軸の
  組み合わせを実行時に弾く検証が増える。閉じた enum + flat canonical 名を採用。
- **5 cell ぶんの feature set モジュールを個別実装**: 盤駒・手駒走査の大半が
  共通で、N 個の複製は保守コストが高い。2 軸でパラメタライズした 1 本の
  indexer にする。
- **v1 / v2 命名**: 外部バージョンタグの識別子化は規約違反、かつ不正確。
- **feature hash を全 cell で nnue-pytorch 互換にする**: 参照実装あり 3 cell は
  bullet-shogi の hash 値を使えるが、新規 2 cell は互換元が存在しない。新規
  2 cell は canonical 名の FNV-1a 32bit hash を feature 定数とする (外部
  エンジン互換は元々非対象で、reproducible かつ衝突しない値であれば足りる)。

## Consequences

- 入力次元が runtime 値になり、1 本の indexer が 5 種の feature set を扱える。
- 公開 5 cell のうち 3 cell は参照実装と相互照合、2 cell は第一原理で検証、
  `halfka-hm-merged` は現 production と bit-identical。検証は `shogi-features`
  の CPU テストで完結し GPU を要さない。
- compile-time const の廃止と全層への spec 配線、artifact identity / checkpoint
  互換、Simple アーキ対応は本 ADR の範囲外で、設計 Issue の後続フェーズが扱う。
