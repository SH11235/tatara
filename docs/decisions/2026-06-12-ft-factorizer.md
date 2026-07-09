# FT factorizer (学習時仮想特徴) の設計

- **Status**: Accepted

## Context

HalfKP / HalfKA 系 feature set の FT 重みは (king bucket × piece-input ordinal) の疎
テーブルで、実戦の玉位置が囲いに偏在するためセルの大半は勾配がほとんど
届かず初期値近傍に留まる。nnue-pytorch は学習時のみ仮想特徴 (king bucket
非依存の piece-input ordinal) を追加し、export 時に実重みへ畳み込む factorizer で
この偏りを補っている。本リポの LayerStack は L1 に同型の機構 (`l1f` shared
+ per-bucket delta) を持つが、FT には無かった。

piece-input 仮想行 の行は king bucket を問わず全局面から勾配を受ける (実効データ
~king-bucket 数倍) ため、玉非依存の成分を高速に学習し、レア玉位置のセルは
export 時に共有 prior を継承する。export で畳み込むため出力 artifact
(次元 / hash / arch 文字列) は base と同一で、推論エンジン側の変更はない。
正則化なしの loss の下では到達可能な関数空間も不変で、変わるのは最適化軌跡
のみ (norm loss 併用時の例外は Decision 7)。

## Decision

### 1. CLI gate (layerstack subcommand 限定、**既定 ON** / `--no-ft-factorize` で OFF)

棋力評価で neutral〜微 + (回帰なし)・throughput +52.8%・収束加速が確認でき、
chess 系 nnue-pytorch (Stockfish 系統) も default feature set `HalfKAv2_hm^` で
factorizer を本番 ON にしている (`model/features/__init__.py` の
`_default_feature_set_name`) ことから、tatara も **既定 ON** とする。
`--ft-factorize` は back-compat の明示 ON (既定と冗長)、`--no-ft-factorize` が
opt-out。`overrides_with` で command-line 後勝ち。

- **`--psqt` 併用**: **併用可**。PSQT shortcut も FT と同じpiece-input 仮想行を持ち、
  forward は畳み込み済み comb (`PsqtState::w_fold`、base 形状)、backward は
  実 grad の king-bucket 方向縮約で仮想 grad を埋める。FT の fold/reduce kernel
  (`ft_fold_virtual` / `ft_reduce_virtual_grad`) と export の
  `coalesce_ft_factorized` は列数が runtime 引数なので、「列 = num_buckets」で
  そのまま再利用する (新規 kernel 不要)。nnue-pytorch が PSQT を FT 出力列に
  持ち `coalesce_ft_weights_inplace` で全列を畳むのと等価な配線を、tatara の
  別 block PSQT に対して行ったもの
- **Simple アーキ**: flag を layerstack subcommand 配下に置くことで構造的に
  到達しない (Simple の export に畳み込みが無いため)
- **`--init-from`**: factorizer と排他なので **auto-suppress** (起動 log に
  明記、silent ではない)。量子化 `.bin` (coalesce 済) は仮想行を持たないため
  初期化元にできない。from-scratch の「実 block sample + 仮想 block zero」と
  同型の load 経路を足せば併用可能だが未実装 (将来の拡張余地)

### 2. 仮想特徴は P factor のみ、sparse path には流さない (fold + reduce)

実特徴 index は全 feature set で `kb * piece_inputs + p` の形
(筋ミラー / 敵玉 fold は p 側に折込済み) なので、実特徴と仮想特徴は
`p = idx % piece_inputs` で 1:1 対応する。この対応を仮想 index として
sparse path に流す (nnue-pytorch 方式、active ×2) 代わりに、dense kernel
2 本で配線する:

- **forward (fold)**: `comb[(kb·pi+p)] = W_real[(kb·pi+p)] + W_virt[p]` を
  optimizer step 後に毎 step 1 回 materialize し (`ft_fold_virtual` /
  `--ft-fp16` 系では f16 cast を融合した `ft_fold_virtual_f16`)、sparse
  forward は comb (base 形状) を base の index 列のまま読む。線形性により
  `Σ_active (W_real + W_virt) = Σ_active comb` で出力は同一
- **backward (reduce)**: 実特徴 1 つにつき仮想特徴ちょうど 1 つが対応する
  ため、仮想行の勾配は `grad_virt[p] = Σ_kb grad_real[(kb·pi+p)]` で厳密に
  求まる (`ft_reduce_virtual_grad`、inverse-index pipeline の実 block gather
  完了後に 1 launch)

これにより特徴 emit・dataloader・H2D 転送・sparse forward / backward は
factorizer 非依存 (base 次元) のまま、仮想行のコストは dense pass 2 本
(weight / grad の全行読み、計 ~1-2 ms/step @ 1536×73K) に置き換わる。
active 数に比例する DRAM 飽和 phase (sparse forward / inverse-index phD) を
2 倍にする方式と数学的に等価で、f32 加算順 / 丸め位置のみ異なる。

- 学習時次元: `train_ft_in = ft_in + piece_inputs` (weight 行数のみ。active
  数 / index 範囲は base のまま)
- nnue-pytorch HalfKP 系の第 3 因子 (玉マス単独) は採用しない — 現行
  nnue-pytorch も piece-plane factor 単独で運用しており、実績を優先する
  (fold + reduce 構成では追加 factor の代金は dense kernel 1 組に下がるため、
  必要になれば throughput とは独立に評価できる)

### 3. spec modifier — 公開 enum は増やさない

公開 `FeatureSet` enum (閉じた 5 variant) は変えず、`FeatureSetSpec` に
modifier (`with_ft_factorize`) を持たせる。export される artifact が base と
同一なので、名前空間を分裂させない。

getter は fail-safe に命名する: `ft_in()` / `max_active()` は **base
(export / format / checkpoint 互換層と sparse path の意味)** のまま、FT
weight の行数を扱う消費者 (weight buffer / optimizer / norm loss /
checkpoint header) だけが `train_ft_in()` を参照する。既存の format 経路は
無変更でコンパイルされ、weight 行数側だけを意図的に書き換える形になる。
spec は `PartialEq` で Batch / trainer / weight の照合に使われるため、
modifier 込みの不一致は既存の照合がそのまま reject する。

### 4. checkpoint format に factorizer flag (寸法照合が primary guard)

raw checkpoint header の feature-set 節に factorizer flag を追加する
(format version bump、旧版読込みは flag 無し = 無効扱い)。header の
`ft_in` には学習側の行数 (`train_ft_in`) を書くため、on/off を跨ぐ resume
は寸法不一致としても必ず reject される — flag は原因が読めるエラーを先に
出すためのもの。`max_active` は base 値 (sparse path が factorizer 非依存の
ため学習側もこれが実値)。experiment.json にも flag を記録する。

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
piece-input 仮想行は同一 p を持つ全実 row の勾配和を受ける (最大 ~king-bucket 数倍、
v は二乗オーダー) ため、headroom を超えると f16 格納時の silent clamp で
仮想行の実効 step が静かに歪むリスクがある。運用では本番 run の前に
`--fp16-opt-state` 抜きの短い run で仮想 block の |m| / |v| 上限を実測して
headroom 内であることを確認する。超過する場合の設計上の逃げ道は optimizer
step を実 / 仮想の 2 領域 launch に分けて仮想側にだけ別 scale を渡すこと
(scale は kernel 引数のため kernel 改修は不要)。

## コスト

- 学習 throughput: sparse path (forward / inverse-index backward / H2D /
  dataloader emit) は base 次元のまま無増。仮想行の代金は dense pass 2 本
  (fold: FT weight 全行読み + base 形状書き、reduce: FT grad 実 block 全行
  読み + 仮想 block 書き、計 ~1-2 ms/step @ 1536×73K) と optimizer / norm
  loss の行数 +`piece_inputs / ft_in` (数 %)。推論側はゼロ (export 物が
  base と同形)
- メモリ: FT 系 buffer が `piece_inputs / ft_in` (数 %) 増。加えて forward
  用 comb (base 形状) — `--ft-fp16` 系では既存 mirror が comb を兼ねるため
  追加なし、FP32 構成のみ f32 buffer 1 本 (`ft_in × ft_out`、1536×73K で
  ~430 MB) を追加確保する
- kernel 改修: fold / reduce の専用 kernel 2 種 3 本 (いずれも 1 thread =
  1 要素の単純 dense kernel)。sparse kernel 群は無変更

## リスク / 検証

- FT weight は学習中 clamp されない (quant 由来 clamp は dense 層のみ) ため、
  畳み込み後の和が i16 飽和域に入る可能性は export 時の飽和検査で監視する
- 仮想行の配線は fold / reduce kernel に集約される — 配線ミスは「仮想行が
  学習されない / forward に乗らない」型の静かな破綻になる。学習開始時
  export の bit 一致 / 1 step 後の仮想行更新 / 量子化 export の base ロード
  可否 (`ft_factorize_tests`) に加え、fold・reduce が仮想 index 方式の
  sparse 計算と一致することを CPU reference 同士
  (`gpu_kernels::sparse::ft_factorize`) と GPU↔CPU equivalence test で
  能動検出する
- fold / reduce 方式は仮想 index を sparse path に流す方式と f32 加算順 /
  丸め位置が異なる (数学的には forward / 勾配とも等価)。bit 再現が要る
  bisect 等では両実装の checkpoint は互換しない (header の `max_active` 値
  も異なり resume は reject される)
- 効果はサンプル効率型のため短期評価は過大に出やすく、採否は長期 run の
  対局評価で判定する

## Rejected alternatives

- **新 FeatureSet variant の公開**: export 物が base と同一なのに名前が
  分裂し、「同じ .bin を指す名前が 2 つ」になる
- **仮想 index を sparse path に流す (nnue-pytorch 方式)**: active ×2 で
  DRAM 飽和 phase (sparse forward / inverse-index phD) がそのまま 2 倍になり
  実測 ~37% の throughput 低下 (1.40M → 882K pos/s、RTX 3080 Ti、LayerStack
  1536x16x32 hm-merged `--all-optim`)。fold + reduce は同じ最適化問題を
  dense pass ~1-2 ms/step で配線できるため置き換えた
- **玉マス factor の同時実装**: 現行 nnue-pytorch が piece-plane factor
  単独で運用している実績を優先
- **推論側で factorized arch を受理**: 推論エンジンは coalesced-only を
  要求する既存方針を維持する理由しかない
