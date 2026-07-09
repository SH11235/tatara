# ADR: EffectBucket 特徴族 — cross-repo index 契約

status: accepted / date: 2026-07-09
対象: rshogi 推論側 + rshogi-nnue(tatara) 学習側 / 出口: Golden Forward bit 一致まで (学習は別)
関連: PoC `rshogi-notes/rshogi/20260709-t2t3-changed-status-poc/`、plans/20260702 (T2)、
先例 ADR `rshogi-nnue/docs/decisions/2026-06-19-threat-feature-set.md` (契約構造を踏襲)

## 1. Context / 決定

threat pair は等深さ +75 nElo だが −45% NPS。T3 (差分利き表で列挙 floor 置換) は PoC で NO-GO。
残る候補が **T2 = E-bucket**: threat pair を捨て、base 特徴 index を各駒マスの「被攻撃数×被防御数」
バケットで拡張し **active 特徴数を base のまま (pair 列挙を回避)** に保つ。本 ADR は新 feature 族
**EffectBucket** を rshogi/tatara 両実装で **bit 一致**で追加する index 契約を確定する。学習・SPRT は範囲外。

**effect bucket は threat とも base とも別カテゴリ**:
- threat = `base_ft_in` offset の**追加行** (base index に不触)。
- effect bucket = **全 base index を `base*NB+bucket` に書き換える** index-space 拡張。別 accumulator は不要だが、
  **base emit / refresh / cache / multi-ply / SIMD の各経路を effect bucket 特化する必要がある** (§D5/D6/D9)。
  「base パイプラインをそのまま流用」は誤り (weight_row=index*L1 の算術は流用可だが、index を出す
  経路は全て effect bucket 化が要る)。

## 2. 確定事実 (両 repo 実測)

- base index bit 一致確定: `halfka_index(kb,packed)=kb*1629+packed`、kb∈[0,45), packed∈[0,1629),
  dims=73,305 (tatara `tests/feature_set.rs` regression、rshogi `bona_piece_halfka_hm_merged.rs`)。
- rshogi FT: `weights:i16`, `weight_row=index*L1` は DIMENSIONS 非依存 (`feature_transformer_layer_stacks.rs`)。
  → 算術は流用可、ただし index を出す **active/refresh/cache/multi-ply の各経路が effect bucket 化必要** (§D9)。
- rshogi count: `pos.board_effect(color,sq):u8` (`pos.rs` の `board_effect`)。**self 非包含**、occupancy 認識、
  **玉利きを含む** (`board_effect.rs` の `king_effect` と `short_effects_from` に King)。material 共有
  インフラなので玉抜きに変更不可。count は `BoardEffects.counts` の full 利き由来で **LongEffects は
  count に無関係** (dirs テーブルのみ、`board_effect.rs`) → effect bucket は `board_effect.effect()` のみ参照。
- tatara: base emit `halfka_hm.rs`、count 材料 `Occupied`/`walk_attacks`/`for_each_attack`
  (`threat.rs`)。ただし **walk_attacks は King を emit しない** → 玉抜き count。
  base board phase は **SIMD kernel** `extract_halfka_hm_board_phase` が base index を直接算出
  (`feature_set.rs`) → effect bucket の bucket 差し込みは **kernel の effect bucket 特化が必要** (threat 型 append 不可)。
- **effect bucket は両 repo donor 不在** (bullet-shogi にも無い) → Golden は **tatara↔rshogi 相互 index 一致**。

## 3. 設計決定

### D1. バケット定義 — config 駆動で NB∈{4,9} を全対応
owner 決定: **実装は全 variant を config で対応** (GPU 占有で学習は逐次のため、コードは family を
包含し学習だけ選ぶ)。`EffectBucketConfig { nb: 4|9, king_bucketed: bool }`。
- **2×2 (NB=4)**: attacked=min(敵利き,1), defended=min(自利き,1)、bucket=defended*2+attacked∈[0,4)。
- **3x3 (NB=9)**: attacked=min(敵利き,2), defended=min(自利き,2)、bucket=defended*3+attacked∈[0,9)。
- 各 config = **別 feature-set** (dims=73,305*NB、hash/arch-token は {nb,king_bucketed} で区別、
  golden 個別)。**Golden gate MVP = 2×2 × D2b** を 1 本で harness 検証、
  code は全 4 config 対応。学習フェーズで 2×2→3x3・D2b→D2a を逐次 A/B。

### D2. バケット化する base 範囲 — config `king_bucketed` で全対応
- **盤上非王駒**: 常に bucket 化。**手駒**: マス無し → 常に bucket0。
- **玉**: `king_bucketed` config で D2a(bucket 化)/D2b(bucket0 固定) を切替。両実装。
- **uniform layout (D4) では D2a/D2b は等メモリ** (dims=73,305*NB は nb のみ依存、玉 bucket は
  同じ行空間の「訓練される/されない」差)。→ Opus の「D2b で FT 行を節約」は uniform layout では
  成立せず、**D2 は等メモリの expressiveness つまみ** (玉の被利き状態を index に入れるか)。
  MVP は最小の D2b (玉 bucket0=非玉のみ bucket 化) で「effect bucket が効くか」を clean に ablation、
  D2a は king-safety 上積みの follow-up A/B。
- 注意: **玉が attacker として他駒の count に寄与するか (D3) とは別問題** — D3 では玉利きは
  D2a/D2b いずれでも必ず count に入る。

### D3. count 意味論 (cross-repo #1 hazard、bit 破綻源) — 玉包含を明示
駒 (物理色 own, マス sq): **defended=board_effect(own,sq)**, **attacked=board_effect(!own,sq)**。
定義: 「現占有下で sq を攻撃する own/敵 駒数、**玉利きを含む**、self 非包含、pin 無視 (raw 盤利き)」。
- **玉包含が必須の契約点**: rshogi board_effect は玉利きを数える (変更不可)。**tatara の count 関数は
  King を含める新規実装が要る** (threat の walk_attacks は玉を落とすので流用不可)。玉隣接駒は盤上常時
  多数存在し、玉抜きだと 1 マスで count が 1/0 に割れ bucket 反転 → golden 不一致。
- 遮蔽: 両者 occupancy 認識、slider は遮蔽マスまで (遮蔽マス自体は利きに含む)。
- **正規化不変性**: count は物理量で鏡像・回転・視点反転に不変。**own/敵は駒の実色で取る** (視点の
  friend で取ると壊れる) — 両 repo 厳守。effect bucket index = f(mirror 済 base index, bucket(raw count))。
- effect bucket は `board_effect.effect()` のみ参照 (LongEffects は count 無関係、参照禁止)。

### D4. effect bucket index 合成式 — uniform layout (donor 不在なので本 ADR が正準)
- **`effect_bucket_index = base_index * NB + bucket`** (base-major, bucket-minor)。**dims=73,305*NB (nb のみ依存)**。
- **bucket 決定 (config `king_bucketed` 依存の predicate)**: packed BonaPiece の域で分岐 —
  hand [0,90) → **常に bucket=0**、盤上非王駒 [90,1548) → bucket=quantize(attacked,defended)、
  玉 [1548,1629) → `king_bucketed ? quantize : bucket0`。両 repo は pack_bonapiece 後の同一域判定で一致。
- **uniform (dead-row 許容) を採る理由**: 単一式で cross-repo 契約面が最小 (partition remap 無し) =
  bit 一致の最優先。hand と (D2b 時) 玉の bucket 1..NB-1 は dead row (weight 0・never gather、
  storage のみ)。compact remap は FT を数% 節約するが契約複雑化 → 実験では不採用。
- bucket<NB で単射、衝突無し。tatara nnue-format の FT 行順と rshogi `weight_row=effect_bucket_index*L1` が
  本式で一致必須。weight block は**追加でなく FT 行数 (DIMENSIONS) の拡張**、別 block を足さない。

Header wire token は旧 `E4=` から `EffectBucket=` へ改名する。旧ヘッダーの net は net header 書換ツールで
arch token と feature hash を移行し、tensor payload は再学習せず byte 不変に保つ。

### D5. 差分更新 (bucket-diff) — effect bucket 新規の芯
`feature_index`/trait メソッドは pos/board_effect を受けない (`ls_feature_spec.rs`)。よって:
- **pos を取る専用関数 `append_changed_effect_bucket_indices(pos, dirty_piece, perspective, king_sq, removed, added)`**
  を accumulator update から呼ぶ (threat 同型)。
- 変化源 = (a) DirtyPiece の動/取られ駒 ∪ (b) **count が変わったマスに乗る駒** (bucket 遷移)。
- **(b) の実装 (b1、既定)**: board_effect の **両 add_delta site を hook** して変化を収集 —
  短利き `apply_bitboard` (`board_effect.rs`) **と** 長利き `update_long_effect_from` の ray
  (`board_effect.rs`) の両方。長利き site を漏らすと discovered attack の bucket 変化を silent に落とす。
- **変更前 count が必須**: bucket 遷移判定に before/after 両 count が要る。board_effect は do_move 中に
  in-place 更新されるので、**触れる前に old count を退避** (touched (sq,color) の before を snapshot、
  または hook が (sq,color,before,after) を記録)。net-zero (inc→dec) で touched でも bucket 不変なら弾く。
- **実 bucket 遷移のみ emit** (count 変化でも clip で bucket 不変なら skip)。DirtyPiece 由来と union、
  重複駒 1 回。overflow 時 bool false → full refresh。
- **正当性ゲート (決定論)**: full recompute (`append_active_effect_bucket`) == 差分維持後 が全合法手 bit 一致 verify。

### D6. tatara 学習側 emit — SIMD base-phase の effect bucket 特化 (append でない)
- effect bucket は **全 base index を書き換える**ので、tatara の base emit を board/king/hand **各 phase で** effect bucket 化。
  特に SIMD kernel `extract_halfka_hm_board_phase` (`feature_set.rs`) に count→bucket を差し込む
  (or effect bucket 時は kernel を bypass してスカラ effect bucket 経路へ)。threat の「`base_ft_in` offset に連結」ひな形は
  **effect bucket に使えない** (base index 不触の追加行方式だから)。漏れると silent に bucket0 のまま出る。
- 新規: **King を含む per-square 2ch count 関数** (D3、`for_each_attack`+King 集計、~数十行)。
- `FeatureSetSpec` に `effect_bucket_config: Option<EffectBucketConfig>` + getter 加算 (ft_in/max_active/feature_hash)、
  `feature_hash=base ^ fnv1a32("effect-bucket-{config}")`、arch_str token `EffectBucket={config},`、CLI parse、hash pairwise
  distinct test、token/hash 不一致は load 時 hard reject (threat ADR Decision 7 同型)。**別 weight block は
  足さない** (FT DIMENSIONS の拡張のみ)。

### D7. Golden Forward (bit 一致ゲート) — donor 不在 → 相互 golden
- canonical index golden (両 repo): startpos + 数局面で effect bucket active index 集合 (sorted) を pin、Black/White。
- 相互 cross-check: N 局面 (PSV) で tatara emission と rshogi 推論の effect bucket index 集合を直接 bit 照合。
- **golden 局面に必須で含める**: 玉隣接密集 / 成駒 (馬竜の step) / slider 遮蔽 / near-king。← 玉包含差
  (D3) と bucket 境界はここでしか出ない。
- 学習後に verify-nnue の eval スカラー一致。

### D8. サイズ試算
FT weights = DIMENSIONS×L1×2B。base(73k)×1024=150MB、2×2(293k)=600MB、3x3(660k)=1.35GB。
active 数は base 並 (~40) で gather 回数は増えないが、テーブル大で residency 悪化 → 実 gather コストは
別途実測。**2×2 先行の主因**。

### D9. accumulator 経路の effect bucket 特化 — index を出す全経路
`feature_index(bp,perspective,king_sq)` は pos を受けないため、effect bucket は base の以下経路を**全て**特化:
- **fast-diff** `try_apply_dirty_piece_fast`: effect bucket で無効化 (bucket 不可)。
- **Finny cache / cache-refresh** (`accumulator_layer_stacks.rs`, `feature_transformer_layer_stacks.rs`,
  `refresh_perspective_with_cache`): effect bucket で**無効化** (cache は未移動 slot の bucket 変化を反映できず誤る。
  threat Finny cache が revert された事情と同じ)。full refresh は pos 付き `append_active_effect_bucket` 専用経路へ。
- **multi-ply ancestor update** (`feature_transformer_layer_stacks.rs`): 中間局面が無く effect bucket
  bucket-diff 計算不能 → **path≥2 は full refresh に落とす** (`nnue-threat` の MAX_DEPTH=1 cfg-gate と同型、
  effect bucket でも MAX_DEPTH=1)。
- board_effect **常時 on** を effect bucket edition の feature 依存で強制 (material 非依存 build で off になるのを防ぐ)。

## 4. 実装分割 (Codex 委譲単位)

- **R1 (rshogi コア)**: `effect_bucket.rs` (index 式 D4 + `append_active_effect_bucket` + `append_changed_effect_bucket_indices` D5)、
  count=`board_effect` 玉込 (D3)、**D9 の全経路特化** (fast-diff/Finny/cache-refresh 無効化、MAX_DEPTH=1、
  effect 常時 on)、before-count snapshot + 両 add_delta hook (D5)、決定論 verify test。
- **R2 (rshogi 配線)**: `EffectBucketSpec`/constants/Cargo(ft+edition)/network alias/arch dims 照合。
- **T1 (tatara)**: **King 込 per-square count 関数** (D3)、**SIMD base-phase の effect bucket 特化** (D6、append 不可)、
  `FeatureSetSpec` 拡張 + CLND + hash/token + canonical golden test。
- **G (両)**: 相互 golden cross-check (D7、玉隣接/成駒/遮蔽 必須)。
- 学習 → 等深さ SPRT (effect bucket eval ゲイン主 kill-gate) → 実 gather NPS → 実 TC。

## 5. owner 判断
- **D1/D2 とも config 駆動で全 variant 実装** (`EffectBucketConfig { nb = 4|9, king_bucketed = bool }`、GPU 占有で
  学習は逐次 → コードは family を包含し学習フェーズで選ぶ)。
- **Golden gate MVP = 2×2 × D2b** (harness を 1 config で通す)。他 3 config も同コードで
  golden 可能にしておく。学習・A/B は本 ADR の範囲外。

## 6. gate / review
- rshogi: fmt+clippy(warn0)+test+abs-path。
- tatara: `scripts/local-ci.sh` PASS。
- Golden gate: canonical index golden と相互 cross-check が一致。

## 7. bit 破綻を招く落とし穴 (cross-repo hazard の要点)
1. **玉包含の count 相違** (D3): rshogi は玉を数え tatara for_each_attack は数えない → bit 破綻。玉込を明示・
   tatara に King 集計追加。
2. **effect bucket は base index 書き換え** で tatara SIMD board phase 要改修 (D6): 「連結」framing は scope 過小。
3. **disable 対象に Finny/cache-refresh/multi-ply が抜け** (D9): fast-diff だけ不十分。全経路 pos 付きへ。
4. **b1 は long add_delta hook + old-count snapshot 必須** (D5): 漏らすと discovered attack/net-zero で silent 誤り。
5. **block-I/O 矛盾** (D4/D6): effect bucket は FT 行数拡張、別 block を足さない。
6. **D2 既定を D2b に** / LongEffects は count 無関係 (参照禁止明記) / golden に玉隣接・成駒・遮蔽必須。
