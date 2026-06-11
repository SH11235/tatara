# FT factorizer (学習時仮想特徴) の設計

- **Status**: Accepted

## Context

HalfKP / HalfKA 系 feature set の FT 重みは (king bucket × piece plane) の疎
テーブルで、実戦の玉位置が囲いに偏在するためセルの大半は勾配がほとんど
届かず初期値近傍に留まる。nnue-pytorch は学習時のみ仮想特徴 (king bucket
非依存の piece plane) を追加し、export 時に実重みへ畳み込む factorizer で
この偏りを補っている。本リポの LayerStack は L1 に同型の機構 (`l1f` shared
+ per-bucket delta) を持つが、FT には無かった。

仮想 P plane の行は king bucket を問わず全局面から勾配を受ける (実効データ
~king-bucket 数倍) ため、玉非依存の成分を高速に学習し、レア玉位置のセルは
export 時に共有 prior を継承する。export で畳み込むため出力 artifact
(次元 / hash / arch 文字列) は base と同一で、推論エンジン側の変更はない。
正則化なしの loss の下では到達可能な関数空間も不変で、変わるのは最適化軌跡
のみ (norm loss 併用時の例外は Decision 7)。

## Decision

### 1. CLI gate `--ft-factorize` (layerstack subcommand 限定、既定 OFF)

既定 OFF は非 factorized 経路と bit-identical。以下の組合せは明示エラーで
reject する:

- **`--psqt` 併用**: PSQT kernel は FT と同じ sparse index 列を消費するため、
  仮想 index が PSQT 勾配バッファ範囲外への atomic write になるか、PSQT を
  学習次元に広げると export PSQT block (`ft_in × num_buckets` 前提) の形式が
  崩れる。両立には PSQT 側にも畳み込みの設計が要る (Rejected alternatives)
- **Simple アーキ**: flag を layerstack subcommand 配下に置くことで構造的に
  到達しない (Simple の export に畳み込みが無いため)
- **`--init-from`**: 量子化 `.bin` は仮想行を持たないため初期化元にできない

### 2. 仮想特徴は P factor のみ

実特徴 index は全 feature set で `kb * piece_inputs + p` の形
(筋ミラー / 敵玉 fold は p 側に折込済み) なので、emit 済み index から
`p = idx % piece_inputs` で仮想 index `ft_in + p` を導出できる。特徴抽出の
盤駒 / 玉 / 手駒の走査は factorizer 非依存のまま、emit 後の post-pass で
仮想分を追加する。

- 学習時次元: `train_ft_in = ft_in + piece_inputs`、
  `train_max_active = max_active × 2`
- nnue-pytorch HalfKP 系の第 3 因子 (玉マス単独、実特徴ごとの重複 emit で
  active ×3) は採用しない — 現行 nnue-pytorch も piece-plane factor 単独で
  運用しており、throughput 代金に見合わない

### 3. spec modifier — 公開 enum は増やさない

公開 `FeatureSet` enum (閉じた 5 variant) は変えず、`FeatureSetSpec` に
modifier (`with_ft_factorize`) を持たせる。export される artifact が base と
同一なので、名前空間を分裂させない。

getter は fail-safe に命名する: `ft_in()` / `max_active()` は **base
(export / format / checkpoint 互換層の意味)** のまま、学習側の消費者
(dataloader 容量 / workspace / kernel 起動 / checkpoint header) だけが
`train_ft_in()` / `train_max_active()` を参照する。既存の format 経路は
無変更でコンパイルされ、train 側だけを意図的に書き換える形になる。spec は
`PartialEq` で Batch / trainer / weight の照合に使われるため、modifier 込み
の不一致は既存の照合がそのまま reject する。

### 4. checkpoint format に factorizer flag (寸法照合が primary guard)

raw checkpoint header の feature-set 節に factorizer flag を追加する
(format version bump、旧版読込みは flag 無し = 無効扱い)。header の
`ft_in` / `max_active` には学習側の値を書くため、on/off を跨ぐ resume は
寸法不一致としても必ず reject される — flag は原因が読めるエラーを先に出す
ためのもの。experiment.json にも flag を記録する。

### 5. export は量子化・飽和検査の前に畳み込む

`W_real[(kb, p)] += W_virtual[p]` の畳み込みを base 形状の host buffer 構築
として先に行い、その配列に i16 飽和検査 → 量子化を掛ける (`l1f` merge と
同型の操作)。畳み込み後の weight 表現は spec も base に落とす — 出力物が
plain な base net であることを型で表す。

### 6. 初期化は「実 block を base 形状で sample → 仮想 block を zero append」

学習次元で一括 sample すると (a) 仮想行に noise が入る、(b) fan-in が変わり
実 row の半値幅がずれる、(c) RNG 消費数が変わり実 row の乱数列が OFF 構成と
不一致になる。base 形状 sample + zero append により、**学習開始時点の
forward が OFF 構成と一致**し、有効/無効の差が学習ダイナミクスだけに閉じる。

### 7. norm loss は仮想行を group に含める

FT の norm loss group (per-output-column × 全行) には学習次元を渡し、仮想行
も正則化対象に含める。king 非依存成分が仮想行へ寄ると同一関数の達成可能
norm が変わるため、norm loss 併用時は目的関数が parametrization 依存になる
— これは factorization の prior と整合する方向であり、weight decay 0 の
レシピでは norm loss が仮想行唯一の magnitude 制御であることから採用する。
norm の apply は group 内一律乗算なので畳み込みと可換、zero 行は zero の
まま。

### 8. FP16 optimizer state の scale headroom

`--fp16-opt-state` の固定 scale (m: 2^28 / v: 2^40) は実 row の実測に基づく。
仮想 P 行は同一 p を持つ全実 row の勾配和を受ける (最大 ~king-bucket 数倍、
v は二乗オーダー) ため、headroom を超えると f16 格納時の silent clamp で
仮想行の実効 step が静かに歪むリスクがある。運用では本番 run の前に
`--fp16-opt-state` 抜きの短い run で仮想 block の |m| / |v| 上限を実測して
headroom 内であることを確認する。超過する場合の設計上の逃げ道は optimizer
step を実 / 仮想の 2 領域 launch に分けて仮想側にだけ別 scale を渡すこと
(scale は kernel 引数のため kernel 改修は不要)。

## コスト

- 学習 throughput: active 特徴数に比例する phase (sparse FT forward /
  inverse-index backward) が ~2 倍になり、全体で 3〜4 割の低下を見込む。
  推論側はゼロ (export 物が base と同形)
- メモリ: FT 系 buffer が `piece_inputs / ft_in` (数 %) 増
- kernel 改修: 不要 (sparse kernel は次元・active 数とも runtime 引数)

## リスク / 検証

- FT weight は学習中 clamp されない (quant 由来 clamp は dense 層のみ) ため、
  畳み込み後の和が i16 飽和域に入る可能性は export 時の飽和検査で監視する
- sparse kernel は範囲外 index を silent skip する規約のため、配線ミスは
  「仮想行が学習されない」型の静かな破綻になる — 学習開始時 export の
  bit 一致 / 1 step 後の仮想行更新 / 量子化 export の base ロード可否を
  テストで能動検出する (`ft_factorize_tests`)
- 効果はサンプル効率型のため短期評価は過大に出やすく、採否は長期 run の
  対局評価で判定する

## Rejected alternatives

- **新 FeatureSet variant の公開**: export 物が base と同一なのに名前が
  分裂し、「同じ .bin を指す名前が 2 つ」になる
- **玉マス factor の同時実装**: active ×3 の throughput 代金に対し、現行
  nnue-pytorch が piece-plane factor 単独で運用している実績を優先
- **推論側で factorized arch を受理**: 推論エンジンは coalesced-only を
  要求する既存方針を維持する理由しかない
- **PSQT との併用対応**: PSQT block の畳み込み + 量子化検査の追加設計が
  必要で、PSQT 自体の採用が見送られている現状ではコストに見合わない
