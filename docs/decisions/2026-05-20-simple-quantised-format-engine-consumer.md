# Simple 量子化 binary フォーマットと推論エンジン consumer 契約

- **Status**: Accepted
- **Date**: 2026-05-20

## Context

`nnue_train` の Simple 4 層アーキ用量子化 binary (`SimpleWeights::save_quantised`、
`crates/nnue-format/src/simple_weights.rs`) は、本リポ単独では生成・回帰テスト
できる cross-process 同等性に閉じる format だが、後続の対局検証 (cf.
`docs/experiments/`) では別リポの将棋エンジン (rshogi-core、独立 codebase) が
**唯一の consumer** として本 `.bin` を読み込む。エンジン側 `network.rs` は
既存の HalfKP / HalfKA_hm の LayerStack-equivalent format を独自に
size-driven detection で受理しており、Simple format はそれと **別レイアウト**
(`pad32` パディング込み i8 dense + nnue-pytorch 風 nested arch 文字列)
である。

エンジン側 session (`/home/sh11235/development/rshogi/docs/experiments/
20260520_simple_arch_engine_integration.md`) で format 突き合わせを行い、
次の 4 TODO が学習側 (本リポ) に上がっていた:

- **TODO A**: `network_hash` に活性化 (CReLU / SCReLU) 由来の項を XOR で足す。
  サイズ判定では活性化を弁別できない (CReLU/SCReLU は weight セクションの byte
  数を変えない) ため、エンジンが活性化込みで file を 1 通りに同定するための
  弁別子をハッシュに織り込みたい、という提案。
- **TODO B**: `arch_feature_name` の命名非対称 (`HalfKA`=split / `HalfKA_hm`=merged)
  を解消し、flat な `canonical_name` を識別に使える形にする。
- **TODO C**: Simple アーキ実装当初の Issue 本文に書かれた「推論エンジン互換
  非ゴール」記述を見直し、rshogi エンジンを対局検証で使う前提を明記する。
- **TODO G** (横断): `shogi-features` クレートを rshogi-core から path/git
  依存で共有する是非。共有すれば feature index 定義が物理的に一致し、エンジン
  側で 5 種の再実装が不要になる。

加えて、本 ADR を書いている session で気付いた問題:

- 既存 `NNUE_VERSION = 0x7AF32F20` (Simple format magic) が rshogi-core
  `NNUE_VERSION_HALFKA = 0x7AF32F20` と **完全衝突**。エンジンが magic だけで
  format をディスパッチした場合、Simple `.bin` を誤って HalfKA_hm
  LayerStack ローダーへ流して silent corruption になる。

## Decision

### 1. version magic を `0x7AF32F21` に変更

`simple_weights::NNUE_VERSION` を `0x7AF32F20` → `0x7AF32F21`。エンジン側 HalfKP
(`0x7AF32F16`) / HalfKA_hm (`0x7AF32F20`) のいずれとも非衝突な値を選ぶ。Simple
アーキの training pipeline が landed した直後の本 ADR 時点で実モデル流通は
まだ無く、format break の影響は本リポ内テストの fixture 更新のみで完結する。
これ以降、format に手を入れる場合は magic を bump する不変条件で運用する。

### 2. TODO A (network_hash に活性化 XOR) は **不採用**

理由:

- 現行 `arch_str` (`build_arch_str` の `arch_identity` 部) が活性化トークン
  (`ClippedReLU` / `SqrClippedReLU`) を 2 箇所に埋め込む形ですでに自己記述
  しており、`SimpleWeights::load` は `arch_identity` を expected と
  string equality で照合して mismatch を `InvalidData` で reject する
  (テスト `load_rejects_activation_mismatch`、`crates/nnue-format/src/
  simple_weights.rs`)。
- ハッシュへの XOR 追加は format break であり、`arch_str` を使った既存の
  reject 経路を **置き換える** ものではなく **重複させる** だけ。
- エンジン側で「活性化を含めて 1 通りに同定したい」要件は、引数 (`expected:
  SimpleId`) として活性化を渡してから `load` に投げる現行 API で満たせる。
  size-driven 列挙とハッシュ probe の合成が必要になるのはエンジン側の
  detection 戦略の問題で、format に新フィールドを足す前にエンジン側 API
  (`expected` の供給経路) で吸収する余地がある。

### 3. TODO B (`arch_feature_name` 改名) は **やらない**

理由:

- 既に `FeatureSet::canonical_name` (`crates/shogi-features/src/feature_set.rs`)
  が `halfka-split` / `halfka-hm-merged` 等の flat な kebab-case 名を提供して
  おり、エンジンはこの値で feature set を識別できる。
- `arch_feature_name` の非対称 (`HalfKA`=split / `HalfKA_hm`=merged) は
  bullet-shogi / nnue-pytorch 系の `arch_str` 慣習に揃えるための内部表記で、
  ファイルレイアウトの一部 (arch 文字列に出力) として保ったまま、エンジン
  側は `canonical_name` / `feature_hash` を使う、という分業で衝突しない。
- 改名すると nnue-pytorch 互換性 (将来 bullet-shogi 出力と相互 import する
  可能性) を失う side effect が大きい。

### 4. 推論エンジン互換は **本 ADR を契約の単一の真実源とする**

Simple アーキ実装当初の作業境界 (本リポは学習側、エンジンは別 repo) では
「推論エンジン互換は非ゴール」という整理だったが、その後の対局検証実験で
rshogi-core が唯一の consumer になることが確定した。今後の方針として:

- **format 安定契約**: `SimpleWeights` の wire format (header / hash / quantised
  byte レイアウト) は本 ADR 以降「単独 consumer = rshogi-core」を前提とした
  契約として扱う。`SimpleWeights::save_quantised` / `load` の byte 仕様
  (位置・型・スケール) を変える場合は version magic を bump し、エンジン側
  ローダーと同 commit / 同 release で更新する。
- 本 ADR が format 契約の単一の真実源で、コードコメント・Issue 本文・README に
  「非ゴール」相当の古い記述が残っていても本 ADR の決定が優先する。

### 5. TODO G (`shogi-features` 共有) は **共有方向を推奨**

理由:

- `shogi-features` クレートは依存が `shogi-format` のみ (純粋 CPU、GPU 非
  依存)。エンジン (`rshogi-core`) から path/git 依存で取り込める。
- 共有すれば 5 種の feature set indexing (BonaPiece 並び、玉バケット、
  HalfKa hm の左右対称化) が定義上一致し、エンジン側の再実装と恒久 drift
  監視が不要になる。
- エンジン側 verify-nnue の Golden Forward は (TODO E) feature index bit
  一致を要求するため、共有が最も低リスクな選択。

ただし配線 (Cargo の path / git dep 切替、CI 連携) はエンジン側 PR の作業に
なる。本リポ側は **`shogi-features` の API / 内部 invariant を勝手に壊さない**
(= consumer が増えた前提でメンテする) という運用契約だけ持つ。

## Consequences

### 直接の効果

- Simple `.bin` がエンジン側で誤検出される経路 (magic 衝突) を閉じた。
- 将来 format 拡張するときの bump ルール (magic + ADR) が明文化された。
- エンジン側の活性化弁別は file の `arch_str` 照合で完結し、ハッシュ追加に
  伴う format break を避けた。
- `shogi-features` が事実上 2 リポの共有資産になることを宣言した。本リポ
  単独の都合で indexing を変えるときはエンジン側の影響を考慮する。

### 副作用 / 将来課題

- 本 ADR の決定 5 で `shogi-features` 共有を推奨したが、実際の path/git 依存
  配線はエンジン側の PR が担当する。本リポは「壊さない」コミット以上のことは
  しない。
- エンジン側 `network.rs` の Simple ローダー実装 (engine doc の TODO D / E /
  F) は本リポのスコープ外。エンジン側で実装が完了したら、必要なら本 ADR を
  追記する (Status を維持したまま consequences に commit hash 等を足す)。
- feature set `halfka-merged` / `halfka-hm-split` がエンジン側で未対応で
  ある点は engine doc に記録済で、対局検証実験で勝者が出るまでは保留。

## Notes

- 本 ADR の motivation source: engine session 側 memo
  `/home/sh11235/development/rshogi/docs/experiments/
  20260520_simple_arch_engine_integration.md`。同 memo の TODO A/B/C/G に
  対する正式回答が本 ADR。
- version magic の選定: `0x7AF32F16` (HalfKP) / `0x7AF32F20` (HalfKA_hm 等
  LayerStack-equivalent) と非衝突で、bullet-shogi 系の magic とも被らない
  値域から `0x7AF32F21` を採用。
