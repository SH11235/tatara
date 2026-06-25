# Threat 特徴量 (駒間利き関係 sparse 特徴) の学習器設計

- **Status**: Accepted

## Context

Threat 特徴量 = 「ある駒が別の駒に利いている」関係を attacker_class ×
attacked_class × 盤上関係で表す sparse 特徴量。base feature set (HalfKP /
HalfKA 系) の入力に連結して FT へ流す。

過去に別の学習器・別エンジン (bullet-shogi / rshogi) で同種を検証済みで、
結論は二面的だった (詳細は bullet-shogi `docs/experiments/v89_*〜v95_*`):

- **depth 固定対局 (NPS 差を排除した eval 品質比較) では base を上回った**。
  Threat の eval 寄与そのものは正。
- **byoyomi (固定時間) では NPS 劣位で探索が浅くなり eval 利得を相殺し負け越し**。
  特徴量増による推論コスト (将棋特有の「NPS 課金」: 手駒が active に残るため
  特徴追加がチェスより高くつく) が支配的だった。
- 次元削減 (same-class 除外 / cross-side 除外) で NPS gap を詰める方向も試したが、
  実戦強度では埋めきれなかった。

本リポ (tatara) でこれを再実験する。評価する独立変数は **profile による次元
間引き** で、full threat を基準に部分集合を runtime 切替して eval/throughput
の trade-off を測る。推論エンジン (rshogi) の実装は本設計の対象外 (学習側を
先行)。ただし将来エンジン評価へ繋ぐため index 算出式は正準式に揃え、bit-exact
化の余地を残す。

### 移植の donor と前提 (利き計算に movegen は不要)

Threat は利き関係なので「占有を考慮した利き列挙」が要る。**しかし bitboard
attack table や magic は不要**。正準リファレンス bullet-shogi
`crates/bullet_lib/src/game/inputs/shogi_halfka_hm_threat.rs` は Bitboard を
使わず、`board: [Piece; 81]` 上の **座標 ray-walk** (`for_each_attack`: 駒種
ごとに file/rank を進め、slider は 2×u64 の `Occupied` で遮蔽 break) で利きを
列挙する。

tatara は同 donor と (a) 同じ PSV / `ShogiBoard` (`board:[Piece;81]`)、(b) 同じ
Square layout (`file*9+rank`、`inverse()` / `mirror_file()` 既存)、(c) 同じ
PieceType を持つ (確認済)。よって **bullet-shogi の threat module をほぼ
そのまま移植でき、movegen / 81bit Bitboard インフラの新設は不要**。

donor は **bullet-shogi を一次** とする (PSV / ShogiBoard / Square layout が一致、
runtime profile dispatch 実装済、golden vector 保持)。rshogi
(`crates/rshogi-core/src/nnue/threat_features.rs` 等、sibling repo
`/mnt/nvme1/development/rshogi`) は engine 都合の compile-time flag + bitboard
実装で tatara から遠いため、**正準式と golden の cross-check 用** に留める。

### tatara 固有の制約

1. 学習データ PSV は **固定 40 byte**。dataloader は range スライス・epoch
   wrap・`--test-tail-positions` をすべて 40 byte 固定ストライドのオフセット
   計算に依存しており、可変長レコード化は破壊的。
2. dataloader は既知の CPU 律速箇所で、過去の特徴追加が毎回 throughput を
   大きく落としてきた。「最初から速い学習器」が要件。threat 抽出は全 scalar
   (base の `HalfKaHmMerged` SIMD 経路には乗らない) ため、CPU コストは実測で
   確認する。
3. **GPU メモリが本設計の主リスク** ([Decision 8](#8-gpu-メモリ試算-profile-が本機-3080-ti-で乗るか) 参照)。
   movegen ではなくここが gating factor。

参照: チェス nnue-pytorch は `FullThreats` を実装し threat を毎エポック
on-the-fly 計算しても高速 (compile-time pseudo-attack テーブル + popcount)。
ただし C++ + 駒少で、tatara (scalar Rust + 駒多) への速度転用は未検証。
nnue-pytorch から採るのは「factorizer 不要・per-pair 独立」「PSQT 列 zero init」
の設計思想のみで、max_active や dtype は将棋の bullet-shogi 値を正とする。

[2026-05-19 の two-axis model](2026-05-19-nnue-feature-set-two-axis-model.md)
における「base getter (ft_in / max_active / feature_hash) は modifier で不変、
train_* だけ分岐」という不変条件に対し、**Threat はこの 3 つすべてを変える**
ため、factorizer 型の次元不変 modifier とは**別カテゴリ (base 次元を拡張する
連結特徴)** である。

## Decision

### 1. Threat は base に連結する「次元拡張型」sparse 特徴、profile は runtime

base feature set (`--feature-set`) はそのまま。Threat は独立 flag
`--threat-profile` で有効化し、active 時は base の `ft_in` 直後に offset 連結
する (nnue-pytorch ComposedFeatureTransformer / bullet-shogi append と同型)。

```
--threat-profile {off, full, same-class, same-class-major-pawn, step-attacker, cross-side}
```

default は `off` (base と bit-identical)。profile は **runtime enum dispatch**
(bullet-shogi 流)。compile-time feature gate (rshogi 流) は sweep が rebuild に
なるため採らない。

`FeatureSetSpec` に `threat_profile: ThreatProfile` フィールドを足し、getter が
threat を加算する (次元不変 modifier ではないことを型で表す):

- `ft_in() = base_ft_in + threat_dims(profile)`
- `max_active() = base_max_active + THREAT_MAX_ACTIVE`
- `feature_hash() = base_hash ^ threat_profile_hash(profile)`

### 2. index 算出式・profile 間引きは bullet-shogi を移植 (rshogi で cross-check)

```
threat_index =
    pair_base[attacker_side][attacker_class][attacked_side][attacked_class]
  + from_offset[attack_pattern][from_sq_n]
  + attack_order[attack_pattern][from_sq_n][to_sq_n]
```

- `ThreatClass` 9 種 (Pawn0/Lance1/Knight2/Silver3/GoldLike4/Bishop5/Rook6/
  Horse7/Dragon8。discriminant 0-8 厳守。King 除外、Gold+ProPawn+ProLance+
  ProKnight+ProSilver → GoldLike)。Rook=6 / Horse=7 の取り違えは golden で検出。
- `from_offset[14][81]` / `attack_order[14][81][81]` は空盤面利きから事前計算
  (NUM_ATTACK_PATTERNS=14: Black 0-8 + White 方向性駒 9-13、非方向性は Black
  再利用)。これは **index 算出用の事前計算 table であり movegen ではない**。
- 実盤面の利きは `Occupied` (2×u64) + `for_each_attack` の座標 ray-walk。
- profile 間引きは `is_excluded(as, ac, ds, dc)` 1 関数に集約し、`pair_base` を
  profile ごとに sentinel skip + prefix-sum で詰め直す:
  - `full` (id 0): 除外なし → 216,720 dims
  - `same-class` (id 1): `ac == dc` → 192,640 dims
  - `same-class-major-pawn` (id 2): `ac == dc || (ac >= 5 && dc == 0)` → 173,568 dims
  - `step-attacker` (id 3): `ac == 1 || ac >= 5` (occupancy 依存 slider = 香角飛馬竜 を
    attacker から全除外、単発利き駒 歩桂銀金 のみ attacker に残す) → 33,408 dims
  - `cross-side` (id 10): `as == ds || ac == dc` → 96,320 dims
- normalize (perspective swap → HM mirror、from/to に同一適用) も donor に合わせる。
  STM / NSTM 両視点で別 index を出す。
- profile 命名・id は donor (`shogi_threat_exclusion.rs`) に合わせる
  (full:0 / same-class:1 / same-class-major-pawn:2 / cross-side:10)。`step-attacker`
  (id 3) は donor に無い engine-native profile: slider attacker pair を index 空間から
  除く (dims 33,408)。狙いは engine 側で slider attacker を early-prune し利き ray 列挙
  (利き計算の主コスト) を省くこと — pair-class late filter では届かない利き列挙 floor を
  NPS から削れる。trainer 自体は他 profile と同様 emit を間引くだけで ray 列挙は省かない
  (engine 対応は follow-up)。eval 寄与の根拠は
  attacker-class 別 ablation 診断 (slider attacker は per-dim eval 効率が最低、step 駒は
  最高密)。golden は donor に無いため tatara ↔ rshogi 間で index 一致を直接検証する。

### 3. on-the-fly 移植 = PoC、forward+backward+GPU メモリを計測して方式確定

bullet-shogi threat module (Occupied + for_each_attack + 上記 index) を
dataloader 抽出経路へ移植し **on-the-fly 計算**する。この移植 (数百行) 自体が
PoC を兼ねる (別途の小 PoC を先行させない)。

**計測ゲート (実装を先に進める前提条件)**: profile 別に以下を計測する:
- **forward + backward の 1 full step pos/s** (forward だけでなく backward も。
  inverse-index backward の phase B prefix-sum が ft_in 4× で律速化しないか含む)
- **GPU メモリ実測** ([Decision 8](#8-gpu-メモリ試算-profile-が本機-3080-ti-で乗るか))

on-the-fly の CPU コストが許容内、かつ対象 profile が本機で乗るなら **data
形式変更なし**で確定。

precompute は fallback。**ただし precompute は CPU movegen を消すだけで GPU
メモリ・FT gather/scatter コストは下げない**点に注意。precompute が必要に
なる場合は固定長拡張レコード `.psvt` (40 byte + edge_count + 固定 K スロット、
K = THREAT_MAX_ACTIVE) として別途設計する (可変長は dataloader 固定ストライド
前提を壊すため不採用)。

### 4. max_active は 320 起点 + overflow を hard-error 検出 (silent truncation 禁止)

`THREAT_MAX_ACTIVE` の起点は **donor bullet-shogi の 320** (40+320=360)。
chess の 128 は採らない (将棋は手駒・成駒で利きが多い)。Full profile の実 PSV で
per-position の実 max を計測して確定する。

tatara dataloader は `extract_active_features` の戻り値を捨て max_active 超過を
silent skip する。threat edge が黙って欠落すると loss だけ見ても気付けないため、
**越え検出を counter / hard-error 経路に変更**する。**release 学習を含む全 build
で hard-error** とする (学習は release で回るため `debug_assert` に閉じない)。

### 5. factorizer は threat に張らない、export は base FT と同じ i16/LEB128

- factorizer (virtual feature) を threat 重みに張らない (nnue-pytorch 準拠、
  各 attacker×attacked ペアが独立で共有価値が低い)。
- **export dtype は base FT block と同じ i16 / LEB128** (QA=127)。chess
  nnue-pytorch の int8 は採らない (tatara FT 形式と不整合、専用 dequant 経路の
  新設になる)。
- PSQT 併用時、threat 由来の material 列は zero init し、PSQT は base row のみに
  作用させる。**初期実装では PSQT × threat も factorizer と同様 CLI 相互排他**と
  し (GPU PSQT row 境界が未検証のため risk-cut)、base-row 限定境界を入れてから
  併用解禁する。

### 6. FT row range の所有権を明示 (factorizer × threat の整合)

row layout は 3 ケースで定義する (相互に矛盾しないこと):

- **threat-only** (factorizer OFF): `ft_in = base_ft_in + threat_dims`、
  layout `[base real | threat real]`。
- **factorizer-only** (threat OFF): `train_ft_in = ft_in + base_piece_inputs`、
  layout `[base real | base virtual(P)]` (現行どおり仮想 P plane が末尾)。
- **両方 ON** (将来。現状は CLI 相互排他で禁止): `train_ft_in() = ft_in() +
  base_piece_inputs` の自然な append 順に従うと layout は
  `[base real | threat real | base virtual(P)]` になる。この順では
  `ft_fold_virtual` / `ft_reduce_virtual_grad` / PSQT zero-init が使う base row
  境界 (現状 `base_ft_in` 前提) を「base real = `[0, base_ft_in)`、base virtual =
  `[base_ft_in + threat_dims, …)`」へ更新する必要があり未検証。

- 不変条件: fold / reduce / PSQT zero-init は **base row (real + virtual) のみ**に
  作用し、threat real row には触れない (threat virtual は無し)。test で保証する。
- 初期実装では **factorizer × threat を CLI で相互排他**にして risk を切り、
  境界整合 test が緑になってから併用解禁してよい。**factorizer は default ON**
  のため、`--threat-profile` 有効かつ factorizer が明示 OFF でない起動は
  **hard-error** とし (利用者に `--no-ft-factorize` を要求)、暗黙 disable は
  採らない (独立変数が黙って動き baseline 比較の帰属が崩れるため)。

### 7. nnue-format は profile identity を 1 表で確定 (前方互換、エンジンは後追い)

直列化契約 (既存の `PSQT={num_buckets},` arch_str token 方式を踏襲):

- **profile は arch_str の `Threat={profile_name},` token で符号化**する
  (`build_arch_str` 1 箇所で生成、`off` は token を出さない)。新規の binary
  header field は足さない。profile から threat dims は一意に導出できる。
- 現行 loader は既に `arch_str.contains("Threat=")` を「unsupported arch」として
  reject する (`crates/nnue-format/src/layerstack_weights.rs`)。**この reject を
  parse 経路に置換**するのが threat-aware 化の本体。engine 後追いの間は reject の
  ままで安全。
- block 順: header → num_buckets → ft_hash → FT bias/weight → (PSQT block) →
  **(Threat block)** → LayerStacks。LayerStacks は末尾のまま、Threat block は
  PSQT block の直後・LayerStacks の直前に挿入する (現行は PSQT 直後が
  LayerStacks)。threat weight block は base FT と同じ i16/LEB128。
- feature hash = `base_feature_hash ^ threat_profile_hash`。profile ごとの hash
  定数を 1 表で定義し、**全 base × 全 profile の合成 hash が pairwise distinct**
  であることを test で固定する。profile compaction は row の意味を変えるため、
  arch_str token / hash の不一致は load 時に必ず弾く (silent 破損防止)。

### 8. GPU メモリ試算 (profile が本機 3080 Ti で乗るか)

base HM-merged ft_in=73,305、ft_out=1536 default。FT 系は {w, m, v, slow, grad}
の 5 buffer (各 f32)。base で ft_w_grad 単体 ~450MB。**threat 連結で ft_in が
増え、FT 系メモリが線形に増える**:

| profile | threat dims | 連結 ft_in | 倍率 | FT 5buffer 概算 | 3080 Ti (12GB) |
|---------|------------:|-----------:|-----:|----------------:|----------------|
| off     | 0           | 73,305     | 1.0× | ~2.2GB          | 余裕 |
| step-attacker | 33,408 | 106,713    | 1.46× | ~3.2GB         | 余裕 (最小 profile) |
| cross-side | 96,320   | 169,625    | 2.3× | ~5.2GB          | 乗る見込み |
| same-class-major-pawn | 173,568 | 246,873 | 3.4× | ~7.5GB | 要実測 (L1/L2/workspace 込みで tight) |
| same-class | 192,640  | 265,945    | 3.6× | ~8.0GB          | 要実測 (OOM 懸念) |
| full    | 216,720     | 290,025    | 4.0× | ~9.0GB          | **OOM 濃厚** |

(概算。L1/L2/L3・workspace・inverse-index buffer が更に乗る。)

なお間引き profile (cross-side 等) が乗りやすいのは研究目的 (NPS / 次元削減の
trade-off) と整合する。

### 計測結果 (確定、RTX 3080 Ti / base halfka-hm-merged / ft-out 1536 / --no-ft-factorize)

| profile | 連結 ft_in | pos/s | peak GPU | fp32 opt-state で fit |
|---------|-----------:|------:|---------:|---------------------|
| off (base) | 73,305 | ~957K | 5.67GB | ✓ |
| step-attacker | 106,713 | 未計測 | 未計測 | ✓ (cross-side より小、fp32 で確実) |
| cross-side | 169,625 | ~721K (−25%) | 8.90GB | ✓ |
| same-class-major-pawn | 246,873 | ~524K (−45%) | 11.17GB | ✓ (tight) |
| same-class | 265,945 | — | OOM | ✗ |
| full | 290,025 | — | OOM | ✗ |

`--fp16-opt-state` (Ranger m/v を f16、過去 allfp16 実験で NaN 無実証明済) が必要なのは
連結 ft_in が大きい same-class / full (~10.15GB/~504K、~10.73GB/~493K) で、これらも
**ft-out 1536 で fit**。step-attacker / cross-side / same-class-major-pawn はより小さく
**fp32 opt-state のまま fit** (step-attacker は最小ゆえ最も余裕)。逃げ道 `--ft-out 512`
では full が 4.71GB/~1.39M (容量は下がる)。

**結論 (計測ゲート)**:
- GPU util は cross-side で median/max 100% = **GPU 律速**。pos/s 低下は ft_in 増に
  よる FT gather/scatter の GPU 仕事増が支配で dataloader CPU は律速でない。よって
  **precompute は効かない (CPU ray-walk を消しても GPU 飽和)。on-the-fly 確定・
  データ形式変更なし**で Decision 3 を closed。
- 本機では **同 ft-out で全 profile 学習可能** (same-class / full は `--fp16-opt-state`
  必須)。間引き profile は fp32 opt-state でも fit。Full vs 間引きの eval 比較は本機で
  成立する。
- 学習 throughput は base (factorizer OFF) 比 −25%〜−49%、factorizer ON の通常 base
  (~1.38M) 比 ~0.35〜0.52×。低下は ft_in 増による GPU FT 仕事増 (GPU 律速) が支配。
  threat 抽出 (CPU) は bullet-shogi の算法を bit-exact に移植したもので algorithmic
  な速度優位は無く、throughput は cuda-oxide trainer と ft_in で決まる (同 feature
  なら bullet-shogi も同等の GPU コストになる)。

## 実装着手点 (tatara files)

- `crates/shogi-features/src/feature_set.rs` — `FeatureSetSpec` に
  `threat_profile`、getter 加算、on-the-fly emit (新規 `threat` module を追加)。
- `crates/nnue-train/src/dataloader.rs` — max_active overflow の hard-error 化。
- `bins/nnue_train/src/cli.rs` — `--threat-profile`、factorizer 相互排他の gate。
- `bins/nnue_train/src/trainer_layerstack.rs` — ft_in/max_active 伝播、threat
  weight 初期化、row range 境界。
- `crates/nnue-format/src/layerstack_weights.rs` — arch_str `Threat=` token の
  parse 化、Threat block I/O、hash 表。

donor (read-only 参照): bullet-shogi
`crates/bullet_lib/src/game/inputs/shogi_halfka_hm_threat.rs` /
`shogi_threat_exclusion.rs`。cross-check: sibling repo
`/mnt/nvme1/development/rshogi` `crates/rshogi-core/src/nnue/threat_features.rs`。

## Consequences

- base (`--threat-profile off`) は bit-identical を維持。既存 run / artifact に
  影響なし。
- **scope は当初想定より大幅に小さい**: movegen / Bitboard 移植は不要、
  bullet-shogi threat module の移植 (ray-walk + index table + profile) が主。
- 新規実験軸: `full` / `same-class` / `same-class-major-pawn` / `cross-side` を
  rebuild なしで sweep 可能。
- throughput は active 数増 (320) + ft_in 増 (最大 4×) で必ず落ちる。許容ラインと
  本機実行可否は計測で確定。Full は本機で学習不可なら間引き profile に絞る。
- host 側 unsafe があれば妥当性をコメント。cuda-oxide / nightly 構成には触れない。

## Alternatives considered

- **movegen / Bitboard を移植**: 不要と判明 (donor は座標 ray-walk + 2×u64
  Occupied、tatara は Square layout 一致)。新設しない。
- **donor を rshogi にする**: engine 都合の compile-time flag + bitboard で
  tatara から遠い。bullet-shogi を一次 donor とし rshogi は golden cross-check。
- **precompute を最初から確定**: PSV 固定 40 byte 制約で encoder/converter +
  ディスク増が要り、かつ **GPU メモリ問題は解決しない**。fallback に降格。
- **engine (rshogi) 再利用で precompute**: data-prep が別リポ依存で「本ツール内
  完結」から外れる。不採用。
- **compile-time profile gate**: profile ごと rebuild で sweep が遅い。runtime
  dispatch を採用。
- **threat factorizer**: nnue-pytorch が明示的に不要としており不採用。
- **export int8**: tatara FT 形式 (i16) と不整合。i16/LEB128 を継ぐ。
- **可変長レコード**: dataloader 固定ストライド前提を全面改修するため不採用。

## 既知の制約 (deferred design)

- **factorizer × threat / PSQT × threat は CLI 相互排他**。factorizer は dense な
  fold/reduce (sparse 経路には virtual を出さず、weight 行列上で `feature % piece_inputs`
  により縮約・畳み込み) で実装されており、現状 `ft_fold_virtual` / `ft_reduce_virtual_grad`
  は bucketed 領域として `ft_in()` を渡す。threat 連結時は `ft_in()` が threat row を
  含むため、virtual が threat row に modulo aliasing で漏れ込む / threat 勾配が virtual に
  混ざる。併用解禁には fold/reduce/coalesce を **range-aware** にし (bucketed 領域 =
  `base_ft_in`、`base_ft_in % piece_inputs == 0` を仮定、threat row `[base_ft_in, ft_in())`
  は不可触で通す)、かつ **export/coalesce が threat row を保持** する (現状
  `coalesce_ft_factorized` は base_ft_in 行しか返さず、threat+factorizer だと threat を
  silent に切り落とす) よう atomic に修正する必要がある。等価性テスト (train 経路
  forward == fold 後 forward, threat active 時) + 勾配/前向き分離テスト + export shape
  で整合を確認してから解禁する。棋力比較で base+factorizer と threat を同条件で並べたい
  時に必要。

## Golden / test 方針

bit-exact の足場として、startpos の sorted threat index (bullet-shogi 正準
ベクタ、34 値) を golden に置く。ただし **startpos だけでは不十分** (成駒・
Horse/Dragon・HM mirror 境界・side 反転・profile 除外を網羅しない) ため、
構築局面の golden を加える: 両視点・玉を HM 両側・全 9 class (成駒含む)・
slider blocker・same-class / cross-side 除外・profile 別 dims/hash。加えて
random 局面で bullet-shogi (必要なら rshogi) リファレンスとの差分テストを置く。
feature_hash は `base ^ profile_hash` の XOR 合成のため、**全 base × 全 profile
の合成 hash が pairwise distinct** であることを test で固定する (既存
feature_hash pin/distinct test の拡張)。
