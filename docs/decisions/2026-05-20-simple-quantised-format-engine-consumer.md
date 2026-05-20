# Simple 量子化 binary フォーマットと推論エンジン consumer 契約

- **Status**: Accepted
- **Date**: 2026-05-20

## Context

`nnue_train` の Simple 4 層アーキ用量子化 binary
(`SimpleWeights::save_quantised`、`crates/nnue-format/src/simple_weights.rs`) は、
本リポ単独では生成・回帰テストできる cross-process 同等性に閉じる format だが、
後続の対局検証 (cf. `docs/experiments/`) では別リポの将棋エンジン (rshogi-core、
独立 codebase) が **唯一の consumer** として本 `.bin` を読み込む。エンジン側
`network.rs` は既存の HalfKP / HalfKA_hm の LayerStack-equivalent format を独自に
size-driven detection で受理しており、Simple format はそれと **別レイアウト**
(`pad32` パディング込み i8 dense + nnue-pytorch 風 nested arch 文字列) である。

エンジン側 session の調査 (engine repo の同名 ADR
`docs/decisions/2026-05-20-simple-quantised-format-engine-consumer.md`) で
YaneuraOu upstream の versioning パターンが精査され、本リポ側の version magic 選定
についても見直し勧告が来た。本 ADR はその確定回答。

### 上流調査で判明した事実

エンジン側 ADR が `git show` で YaneuraOu master を精査した結果:

1. YaneuraOu `source/eval/nnue/nnue_common.h` の `kVersion = 0x7AF32F16` は
   **全 NNUE ファイル共通の単一値**。`evaluate_nnue.cpp` の `ReadHeader` が
   `version != kVersion` を `FileMismatch` で reject する。version magic は
   **形式世代スタンプであり、アーキの弁別子ではない**。
2. アーキは `evaluate_nnue.h` の `kHashValue = FeatureTransformer::GetHashValue()
   ^ Network::GetHashValue()` で識別する (= 計算ハッシュ)。
3. YaneuraOu の `clipped_relu.h` と `sqr_clipped_relu.h` は同一ハッシュ
   `0x538D24C7` を持ち、**hash は CReLU / SCReLU を区別しない**。活性化は arch
   文字列トークン (`ClippedReLU` / `SqrClippedReLU`) で識別する。
4. 本リポの `SimpleWeights::compute_fc_hash` が使う層ハッシュ定数
   (`0xCC03DAE4` = affine, `0xEC42E90D` = input slice, `0x538D24C7` = clipped
   relu) は YaneuraOu の値と一致 → トレーナーの `network_hash` は YaneuraOu
   `kHashValue` 機構の移植。
5. bullet-shogi の bucket-less モデル (`examples/shogi_simple.rs
   --output-format standard` の出力) も version `0x7AF32F16` を使う =
   bucket-less lineage は `0x7AF32F16` で揃っている。
6. rshogi エンジンの version 定数: `NNUE_VERSION = 0x7AF32F16` (HalfKP、
   YaneuraOu lineage) / `NNUE_VERSION_HALFKA = 0x7AF32F20` (nnue-pytorch /
   LayerStack lineage)。

## Decision

### 1. version magic は YaneuraOu の `kVersion = 0x7AF32F16` を採用する

`simple_weights::NNUE_VERSION` を `0x7AF32F16` に揃える。理由:

- YaneuraOu / bullet-shogi bucket-less 系列の正しい lineage 値。
- 本リポの `network_hash` 機構は YaneuraOu `kHashValue` の移植であり、version
  選定もそれに揃えるのが整合性が取れる (層ハッシュ定数だけ YaneuraOu を引き、
  version は別系列を使う、という捻れを解消)。
- format に手を入れる場合は version magic を bump する不変条件で運用する
  (YaneuraOu と同じ「形式世代スタンプ」のルール)。

### 2. アーキ弁別は **hash + arch_str** に閉じる (format 側責務)

rshogi エンジンの既存 `NNUE_VERSION` (HalfKP) と Simple `NNUE_VERSION` は同値
だが、本リポ format の責務は version 弁別ではなく hash + arch 文字列での
弁別を提供することに置く:

- `network_hash` を YaneuraOu `kHashValue` 互換 (`compute_fc_hash ^ ft_hash`)
  に保つ。
- `arch_str` の `arch_identity` で活性化を自己記述する (`load` が string
  equality で reject 済、`load_rejects_activation_mismatch` テスト)。

エンジン側 dispatcher は現状 version で HalfKP 経路に流す構造なので、Simple
を実投入するには hash 駆動 dispatcher への切替が必要。これはエンジン側
スコープ (本 ADR Consequences の「配布 gate」を参照)。

### 3. TODO A (`network_hash` に活性化を XOR) は採用しない

理由:

- YaneuraOu 自身、`kHashValue` に活性化を含めない (CReLU / SCReLU 同ハッシュ)。
  `compute_fc_hash` はそれを忠実に移植している。
- 活性化は `arch_identity` 文字列に自己記述済みで `load` が reject 済。XOR は
  情報の二重化であり format break。
- 採用するとエンジンが「YaneuraOu パターンに依拠」と言いつつ独自 hash に
  逸脱することになる。

### 4. TODO B (`arch_feature_name` 改名) は採用しない

理由:

- `FeatureSet::canonical_name` (`crates/shogi-features/src/feature_set.rs`) が
  既に `halfka-split` / `halfka-hm-merged` 等の flat な kebab-case 名を提供。
- `arch_feature_name` の非対称 (`HalfKA` = split / `HalfKA_hm` = merged) は
  nnue-pytorch arch 文字列の互換トークンで、ファイル内 arch_str に出力する
  ためのもの。改名すると nnue-pytorch / bullet-shogi 出力との文字列互換を失う。
- エンジンは feature set を `canonical_name` / `feature_hash` で識別する。

### 5. 推論エンジン互換は **本 ADR を契約の単一の真実源とする**

Simple アーキ実装当初の作業境界 (本リポは学習側、エンジンは別 repo) では
「推論エンジン互換は非ゴール」という整理だったが、その後の対局検証実験で
rshogi-core が唯一の consumer になることが確定した。今後の方針として:

- **format 安定契約**: `SimpleWeights` の wire format (header / hash /
  quantised byte レイアウト) は本 ADR 以降「単独 consumer = rshogi-core」を
  前提とした契約として扱う。byte 仕様を変える場合は version magic を bump し、
  エンジン側 ローダーと同 commit / 同 release で更新する (= YaneuraOu と同じ
  「形式世代スタンプ」運用)。
- 本 ADR が format 契約の単一の真実源で、コードコメント・Issue 本文・README に
  「非ゴール」相当の古い記述が残っていても本 ADR の決定が優先する。

### 6. `shogi-features` クレートを共有方向で扱う (TODO G)

理由:

- `shogi-features` は純粋クレート (依存は `shogi-format` のみ) でエンジンから
  path / git 依存で取り込める。
- 共有すれば 5 feature set の indexing parity が定義上保証され、エンジン側で
  再実装が不要になる。
- エンジン側 verify-nnue の Golden Forward は feature index bit 一致を要求する
  ため、共有が最も低リスクな選択。

ただし配線 (Cargo の path / git dep 切替、CI 連携) はエンジン側 PR の作業に
なる。本リポ側は **`shogi-features` の API / 内部 invariant を勝手に壊さない**
(= consumer が増えた前提でメンテする) という運用契約だけ持つ。

## Consequences

### 直接の効果

- version magic が YaneuraOu / bullet-shogi bucket-less と同系列に揃った。
  既存の `network_hash` 機構と一貫した lineage を持つ。
- 活性化 (CReLU / SCReLU) の弁別は `arch_str` の `arch_identity` に閉じる
  (hash や header に重複させない、YaneuraOu パターン準拠)。
- `shogi-features` が事実上 2 リポの共有資産になることを宣言した。本リポ
  単独の都合で indexing を変えるときはエンジン側の影響を考慮する。

### 副作用 / 将来課題

- **エンジン consumer への配布 gate**: rshogi エンジンの現状 dispatcher は
  HalfKP も `0x7AF32F16` を使う前提で、version 一致時に HalfKP ローダー経路
  へ流れる。Simple `.bin` をその dispatcher に投入すると HalfKP として誤
  解釈され silent corruption になる。**Simple `.bin` をエンジン consumer 向け
  に publish するのは、エンジン側で hash 駆動 dispatcher (HalfKP / Simple 等を
  `kHashValue` で振り分ける ReadHeader 経路) が landed したあと**。それまでは
  Simple `.bin` は本リポ内テスト fixture と内部検証用途に限定する。
- 本 ADR で `shogi-features` 共有を推奨したが、実際の path / git 依存配線は
  エンジン側の PR が担当する。本リポは「壊さない」コミット以上のことはしない。
- エンジン側 `network.rs` の Simple ローダー実装 (hash 駆動 dispatcher への
  切替、Simple loader 本体、feature index bit 一致検証、未対応 feature set
  2 種の追加) は本リポのスコープ外。エンジン側で実装が完了したら、必要なら
  本 ADR を追記する (Status を維持したまま consequences に commit hash 等を
  足す)。
- feature set `halfka-merged` / `halfka-hm-split` がエンジン側で未対応で
  ある点はエンジン側 ADR に記録済で、対局検証実験で勝者が出るまでは保留。
- バイトレイアウト比較 (`SimpleWeights` 出力 vs bullet-shogi
  `--output-format standard`) は engine 側で別途実施。一致すればエンジン側
  ローダー実装が大幅に縮小する。

## Notes

- 本 ADR の motivation source: engine repo の同名 ADR
  (`docs/decisions/2026-05-20-simple-quantised-format-engine-consumer.md`)。
  本リポ側は engine ADR の精査 (YaneuraOu master 実証) を採用し、version magic
  を YaneuraOu lineage に揃える形で連動した。両 ADR は相互参照する。
- version magic 値の lineage 整理: `0x7AF32F16` = YaneuraOu `kVersion` /
  bullet-shogi bucket-less / 本 Simple format / rshogi engine HalfKP。
  `0x7AF32F20` = nnue-pytorch / LayerStack 系列 / rshogi engine HalfKA_hm /
  bullet-shogi LayerStack。
