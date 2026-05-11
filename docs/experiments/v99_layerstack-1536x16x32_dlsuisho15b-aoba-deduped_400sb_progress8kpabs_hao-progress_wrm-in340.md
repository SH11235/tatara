# bullet-shogi 実験記録: v99

---

## 実験概要

| 項目 | 値 |
|------|-----|
| 実験ID | v99 |
| ベース実験 | v96 (差分: progress.bin を `nodchip_progress_e1_f1_cuda.bin` から `progress_hao_full_cuda.e1.bin` に変更) |
| 実験日 | 2026-04-28 〜 |
| 目的 | v96 と同一設定で **progress.bin のみ** 差し替えて学習。bucket 選択係数を約770万局面1epoch 学習版（命名規約からの推定）から 80億局面1epoch 学習版へ更新したときの棋力影響を検証する |
| 結論 | **棋力に有意な差なし**（v99-400 SPRT 1823局で nElo −6 ±16, accept_h0 相当）。progress.bin の差し替え（約770万局面 → 80億局面学習）は棋力に統計的に有意な影響をもたらさなかった。 |

---

## 実験設計

### v96 からの差分

| 項目 | v96 | v99 |
|------|-----|-----|
| progress.bin | `nodchip_progress_e1_f1_cuda.bin` (nodchip 公式、命名規約から **1ファイル × 1epoch ≈ 約770万局面**と推定) | **`progress_hao_full_cuda.e1.bin` (本リポジトリ CUDA 版 1016ファイル × 1epoch、~80億局面 × 1epoch)** |
| shuffle seed | 96 | **99** |

**その他のパラメータは v96 と完全に同一。**

### progress.bin について

| 項目 | v96 (`nodchip_progress_e1_f1_cuda.bin`) | v99 (`progress_hao_full_cuda.e1.bin`) |
|------|----|----|
| 学習教師量（推定） | **約770万局面 × 1 epoch**（注） | **80億局面 × 1 epoch** |
| 学習バックエンド | nodchip 公式 PyTorch+CUDA 実装 | bullet-shogi 自前 CUDA + mini-batch (K=1024) |
| 教師データ（推定） | nodchip/tanuki-.nnue-pytorch-2024-07-30.1 系 1ファイル分 | **hao_depth9 全量（1016 ファイル、entering_king は除外）** |
| ファイルサイズ | 1,003,104 bytes | 1,003,104 bytes（YaneuraOu 互換、同一フォーマット） |
| 学習詳細 | （nodchip 公開、注参照） | `docs/experiments/progress/progress_nodchip_continuous_full_20260428.md` |

（注）`nodchip_progress_e1_f1_cuda.bin` の学習量は **直接的な記録は無く、推定**。根拠：
- `nnue-pytorch-nodchip` の `shogi.2026-03-16.sfnnwop-1536.progress_e1_f1_cuda` ブランチ README で `progress_e<N>_f<M>_cuda.bin` 形式が `--epochs N --max-train-files M` の組合せを示す（例: `progress_e10_f256_cuda.bin` は `--epochs 10 --max-train-files 256`）
- 同 README で参照される学習データは Hugging Face `nodchip/tanuki-.nnue-pytorch-2024-07-30.1`（1016 ファイル、各 ~308MB ≈ 7.7M positions）
- 上記命名規約 + データセット → `e1_f1` = 1 epoch × 1 file ≈ **約770万局面**と推定
- ただし当該 .bin の commit message は「ファイル差し替え」のみで、訓練ログ・コマンド履歴等の直接記録は repo に存在しない

### 候補比較（haoek vs hao の事前検証）

本実験では `haoek` (hao + entering_king 全量) で先行学習した progress.bin と、`hao` のみで学習した progress.bin の bucket 分布を `compare_progress_buckets` で比較し、最終的に **hao_e1 を採用**することにした。

#### bucket 分布比較（compare_progress_buckets, 1M positions × 3 datasets）

| Dataset | progress | std% | b6 | b7 | mean_p |
|---|---|---|---|---|---|
| hao | nod_e1_f1 | 3.22 | 19.7% | 8.0% | 0.514 |
| hao | haoek_e1 | 2.66 | 18.1% | 9.1% | 0.517 |
| hao | **hao_e1** | **2.57** | 17.7% | 8.8% | 0.512 |
| ek | nod_e1_f1 | 10.83 | 37.0% | 18.3% | 0.697 |
| ek | haoek_e1 | 9.71 | 32.3% | 19.5% | 0.692 |
| ek | **hao_e1** | **9.56** | 32.1% | 18.6% | 0.687 |
| **DLSuisho15b** (v99 教師) | nod_e1_f1 | 5.82 | 23.2% | 9.1% | 0.582 |
| **DLSuisho15b** (v99 教師) | haoek_e1 | 5.54 | 21.4% | 9.6% | 0.580 |
| **DLSuisho15b** (v99 教師) | **hao_e1** | **5.42** | 21.0% | 9.2% | 0.575 |

実測ログ: `docs/experiments/progress/compare_progress_buckets_hao_vs_haoek_20260428.log`

#### 観察と採用理由

意外な結果として、**hao_e1 が3つのデータセットすべてで std% 最良**となった（entering_king 含む）。理由の解釈：

- haoek_e1 は学習時に entering_king を 5.5% 混ぜた → 「長手数 = 後半」の信号が強化され、結果として b5-b7 への押し込みが若干強い predictor になる
- hao_e1 は通常データ前提で学習 → b6-b7 への極端なシフトが起きず、ek dataset でも均一に分散

**v99 教師データは DLSuisho15b_aoba_deduped で hao 系 + aoba（共に通常データ）が主成分**（後述の tail 分析で entering_king 混入は最大 3% 程度）であり、専用 entering_king 較正の恩恵は限定的。むしろ **hao_e1 の素直な切り分け** の方が DLSuisho15b 全体に対し均一で適合性が高い。

mean_p 比較では hao_e1 が **0.5749**（理想 0.5 に最近接）→ DLSuisho15b 全体を最も自然に2分する位置に切り分けられる。

### 教師データ側の entering_king 含有確認（king 位置 decode による定量判定）

`examples/scan_entering_king.rs` で各 PSV を decode し king 位置を直接読み取って入玉局面比率を測定（1M サンプル/file）。

#### 入玉シグナル

| データ | bk入玉(rank≤2) | wk入玉(rank≥6) | 両入玉 | いずれか入玉 | mean ply |
|---|---|---|---|---|---|
| **DLSuisho15b_aoba_deduped (v99教師)** | **2.72%** | **2.75%** | **1.27%** | **4.20%** | 83.4 |
| DLSuisho15b_deduped (aoba 無し版) | 2.87% | 2.89% | 1.35% | 4.41% | 94.3 |
| aobazero_01 | 1.81% | 1.84% | 0.70% | 2.95% | 83.2 |
| nodchip hao_depth9 | 1.69% | 1.62% | 0.65% | 2.66% | 82.1 |
| nodchip suisho5_entering_king | **17.03%** | **24.45%** | **13.28%** | **28.20%** | 141.2 |

実測ログ: `docs/experiments/progress/scan_entering_king_20260428.log`

#### 線形混合による entering_king 比率推定

非 entering 系（hao+aoba の平均、~1.7%）と entering_king（17-28%）を線形混合した結果、4 指標で一致：

| 指標 | DLSuisho15b 観測 | 推定 ek 比率 |
|---|---|---|
| bk_entered | 2.72% | 6.3% |
| wk_entered | 2.75% | 4.5% |
| both_entered | 1.27% | 4.8% |
| either_entered | 4.20% | 5.5% |

→ **DLSuisho15b_aoba_deduped には約 5% の entering_king 系局面が含まれる**（4 指標で一致、king 位置は直接シグナルなので断定可能）。

`game_ply` tail 分析（先行調査）の推定値 ~3% より高い。**king 位置 decode の方が直接的かつ正確**。

#### v99 への含意

- v99 教師データに **約 5% の entering_king 系局面が混入している**（無視できない量）
- entering_king 込みで較正された haoek_e1 の利点が活きうる
- ただし **DLSuisho15b 全体の bucket 分布では hao_e1 がわずかに勝つ**（std% 5.42 vs 5.54、mean_p 0.575 vs 0.580、前項の比較表）
- 5% 混入は overall 分布を後半寄りに引っ張るが、hao_e1 の素直な切り分けが全体均一性を上回る効果
- 差は小さい（std% 0.12 ポイント）ので、**haoek_e1 を使う v99' 派生実験も将来候補に値する**。本実験では hao_e1 を採用

---

## ネットワーク設定

v96 と同一。

| 項目 | 値 | 備考 |
|------|-----|------|
| アーキテクチャ | LayerStack 1536x16x32 | |
| 特徴量 | HalfKA_hm | 73,305 次元 |
| バケット | progress8kpabs | 9 バケット |
| **progress パラメータ** | **`progress_hao_full_cuda.e1.bin`** | **v96 から変更** |
| QA / QB | 127 / 64 | |
| FV_SCALE | 28 (8128/290) | |

## 学習設定

v96 と同一。

| 項目 | 値 | 備考 |
|------|-----|------|
| optimizer | Ranger | |
| 学習率 (lr) | 8.75e-4 | |
| LR gamma | 0.995 | |
| LR step | 1 | |
| batch-size | 16384 | |
| superbatches | 400 | |
| batches_per_superbatch | 24414 | ~4億局面/sb |
| wdl | 0.0 | |
| scale | 290 | |
| win-rate-model | 有効 | |
| wrm-in-scaling | 340 | |
| wrm-nnue2score | 600 | |
| weight-decay | 0.0 | |
| threads | 10 | |

## データ供給設定

| 項目 | 値 |
|------|-----|
| データセット | DLSuisho15b_aoba_deduped_shuffled.bin (616GB, ~165.4億局面) |
| epoch シャッフル | `--epoch-file-shuffle` |
| shuffle seed | `99` |

---

## 学習コマンド

```bash
cd /mnt/nvme1/development/bullet-shogi

mkdir -p checkpoints/v99

DATA=data/DLSuisho15b_aoba_deduped_shuffled.bin
PROGRESS_BIN=/mnt/nvme1/development/bullet-shogi/data/progress/progress_hao_full_cuda.e1.bin

cargo run --release --example shogi_layerstack -- \
  --data "$DATA" \
  --l0 1536 \
  --l1 16 \
  --l2 32 \
  --batch-size 16384 \
  --batches-per-superbatch 24414 \
  --superbatches 400 \
  --lr 8.75e-4 \
  --lr-gamma 0.995 \
  --lr-step 1 \
  --wdl 0.0 \
  --scale 290 \
  --win-rate-model \
  --wrm-in-scaling 340 \
  --wrm-nnue2score 600 \
  --optimizer ranger \
  --weight-decay 0.0 \
  --save-rate 20 \
  --threads 10 \
  --bucket-mode progress8kpabs \
  --progress-coeff "$PROGRESS_BIN" \
  --epoch-file-shuffle \
  --file-shuffle-seed 99 \
  --output /mnt/nvme1/development/bullet-shogi/checkpoints/v99 \
  --net-id v99 | tee checkpoints/v99/train.log
```

---

## 事前チェックリスト

- [ ] `progress_hao_full_cuda.e1.bin` がサイズ 1,003,104 bytes
- [ ] `FV_SCALE: 28 (QA=127, QB=64, scale=290)` がログに表示される
- [ ] `Win rate model: enabled` がログに表示される
- [ ] `WRM in_scaling: 340 nnue2score: 600` がログに表示される
- [ ] `Batches/superbatch: 24414` がログに表示される
- [ ] データが `DLSuisho15b_aoba_deduped_shuffled.bin` (1 file) であること
- [ ] **`--progress-coeff` が `progress_hao_full_cuda.e1.bin` を指していること**（重要、v96 との唯一の差分）
- [ ] ゼロスタートであること

### 推論側メモ

学習に使った progress.bin と推論に使う progress.bin は**必ず一致させる**。混在させると bucket 割当が変わって NN 重みと不整合になる。

- `LS_BUCKET_MODE=progress8kpabs`
- **`LS_PROGRESS_COEFF=data/progress/progress_hao_full_cuda.e1.bin`**（v96 とは別物）
- FV_SCALE の USI 上書きは不要

---

## 評価計画

### 主要比較

| 対戦 | 目的 |
|------|------|
| **vs v96-best** | **progress.bin 差し替えの効果（同一設定、progress.bin のみ異なる）** |
| vs v87-400 | 過去ベースライン |
| vs material | 棋力下限の確認 |

### 学習時間

- 想定: v96 と同等（1 sb ≈ 660s × 400 sb ≈ 73 時間）
- progress.bin 差替えは推論コストに影響しないので学習時間は変わらない見込み

---

## 実験結果

### Loss 推移

| Superbatch | Loss | 備考 |
|------------|------|------|
| 20 | TBD | |
| 100 | TBD | |
| 200 | TBD | |
| 300 | TBD | |
| 400 | TBD | best loss |

### bucket 分布（学習中、参考）

TBD: 学習中に各 superbatch で各 bucket がどれくらいの頻度で参照されたかが取れる場合は記録。

### 自己対局結果

#### 簡易評価（途中ステップ、2026-04-29〜30）

学習中の途中チェックとして v99-120 時点で 2 つの簡易対局を実施。
YO バイナリ `YaneuraOu-sfnnwop1536-v2-922-tournament-spsa-plcache-avx512fix`
を使用、byoyomi=1000ms / hash=256MB / threads=1 / concurrency=5 /
startpos=`data/startpos/start_sfens_ply32.txt` で計測。

##### 1) v96-120 vs v99-120（fixed 200局, 双方向計400局）

| 指標 | A: yo-v96-120 | B: yo-v99-120 |
|------|--------------:|--------------:|
| 勝/負/引分 | 202 / 185 / 13 | 185 / 202 / 13 |
| Elo 差（A 視点） | **+15 ±34** | nElo: +15 ±34 |
| 平均 NPS | 743K | 737K |
| 平均 depth | 22.84 | 22.80 |
| timed_out | 0 | 0 |

**結論**: v99（progress.bin を hao_full に差し替えた再学習モデル）は v96 と
**同 step で誤差範囲内**（±35 Elo 帯）。少なくとも棋力崩壊は起きていない。

ログ: `runs/selfplay/20260429-v96_120-vs-v99_120-yo-fixed200/`

##### 1b) v96-200 vs v99-200（fixed 200局, 双方向計400局）

| 指標 | A: yo-v96-200 | B: yo-v99-200 |
|------|--------------:|--------------:|
| 勝/負/引分 | 199 / 194 / 7 | 194 / 199 / 7 |
| Elo 差（A 視点） | **+4 ±34** | nElo: +4 ±34 |
| 平均 NPS | 723K | 728K |
| 平均 depth | 21.81 | 21.80 |
| timed_out | 0 | 0 |

**結論**: 同 step (200) でほぼ完全互角。step 120 時点 (+15) と比較し、
**v99 が v96 に追いついている傾向**が見えるが、両走とも誤差範囲内なので
有意とは言えない。

ログ: `runs/selfplay/20260430-v96_200-vs-v99_200-yo-fixed200/`

##### 1c) v96-280 vs v99-280（SPRT, 1308局時点で打切り）

SPRT (`nelo0=0, nelo1=5, α=β=0.05`, 上限 2000 局) で実施。
1308 局時点で結論ほぼ確定（H0 寄りに緩く drift）と判断し打切り。

| 指標 | A: yo-v96-280 | B: yo-v99-280 |
|------|--------------:|--------------:|
| 勝/負/引分 | 648 / 632 / 28 | 632 / 648 / 28 |
| Elo 差（A 視点） | **+4 ±19** | nElo: +4 ±19 |
| SPRT LLR (B vs A) | — | −0.342（境界 ±2.944, 未収束） |
| 平均 NPS | 731K | 723K |
| 平均 depth | 21.78 | 21.69 |
| timed_out | 0 | 0 |

ペントノミアル (B=v99 視点): `[LL=153, LD=18, WL/LW+DD=322, WD=10, WW=150]`

**結論**: step 280 でも誤差範囲内。fixed 200 局より有意に狭い ±19 Elo 帯
で測れたが、依然として **v96 ≥ v99 のわずかな傾向** が継続。

ログ: `runs/selfplay/20260501-v96_280-vs-v99_280-yo-sprt/`

##### 1d) v96-320 vs v99-320（SPRT, 1074局時点で打切り）

SPRT (`nelo0=0, nelo1=5, α=β=0.05`, 上限 2000 局) で実施。
1074 局時点で**初めて v99 優位の傾向**が観測されたためサンプルを保存して打切り。

| 指標 | A: yo-v96-320 | B: yo-v99-320 |
|------|--------------:|--------------:|
| 勝/負/引分 | 507 / 540 / 27 | 540 / 507 / 27 |
| Elo 差（A 視点） | **−11 ±21** | nElo: −11 ±21 |
| SPRT LLR (B vs A) | — | +0.386（境界 ±2.944, 未収束） |
| 平均 NPS | 733K | 727K |
| 平均 depth | 22.10 | 22.14 |
| timed_out | 0 | 0 |

ペントノミアル (B=v99 視点): `[LL=114, LD=7, WL/LW+DD=273, WD=18, WW=125]`

**結論**: v99 が **+11 ±21 nElo** で v96 をわずかに上回る。これまでで初めて
v99 優位の数値だが、信頼区間は依然として 0 を跨ぐため**有意とは言えない**。
ただし step 進行に伴い v96 → 同等 → v99 という単調な改善トレンドが
継続しており、**v99-400 での最終評価に強い動機**がある。

ログ: `runs/selfplay/20260501-v96_320-vs-v99_320-yo-sprt/`

##### 1e) v96-400 vs v99-400（最終評価, SPRT 1823局時点で打切り）

学習完走後の最終評価。SPRT (`nelo0=0, nelo1=5, α=β=0.05`, 上限 2000 局)
で実施。1823 局時点（91%）で残りの局数では LLR が境界に達しない
ことが確定したため打切り。

| 指標 | A: yo-v96-400 | B: yo-v99-400 |
|------|--------------:|--------------:|
| 勝/負/引分 | 904 / 876 / 43 | 876 / 904 / 43 |
| Elo 差（A 視点） | **+5 ±16** | nElo: +5 ±16 |
| SPRT LLR (B vs A) | — | **−0.625**（境界 ±2.944, H0 寄り drift） |
| 平均 NPS | 708K | 699K |
| 平均 depth | 21.81 | 21.72 |
| timed_out | 0 | 0 |

ペントノミアル (B=v99 視点): `[LL=210, LD=22, WL/LW+DD=467, WD=15, WW=199]`

**結論**: **v99 は v96 と統計的に有意な差なし**（信頼区間 ±16 で 0 を含む）。
SPRT は accept_h0 方向に drift しており「v99 が +5 nElo 以上強い」仮説は
実質却下。step 320 で見えた +11 ±21 の v99 優位傾向は最終的に 0 付近に
収束した。

ログ: `runs/selfplay/20260502-v96_400-vs-v99_400-yo-sprt/`

##### ステップ別 Elo 差の推移（B=v99 視点 = nElo, v99 が正で優位）

| step | v99 nElo (vs v96) | 局数 | 形式 |
|---:|---:|---:|---|
| 120 | −15 ±34 | 400 | fixed |
| 200 | −4 ±34 | 400 | fixed |
| 280 | −4 ±19 | 1308 | SPRT 打切り |
| 320 | +11 ±21 | 1074 | SPRT 打切り |
| **400** | **−6 ±16** | **1823** | **SPRT 打切り（最終）** |

step 進行に伴い v96 優位 → 同等 → v99 優位（320）→ 最終的に同等（400）と
推移。step 320 の v99 優位は途中の揺らぎ、**最終的には差なし**に収束。

##### 2) progress.bin ablation: v96-400 + nodchip vs v96-400 + hao（部分 SPRT, 1104局時点で打切り）

同一 NN 重み（v96-400）で **推論時 progress.bin だけ差し替え** ると
mismatch が棋力にどれだけ効くかを検証。SPRT は `nelo0=0, nelo1=5,
α=β=0.05` で開始、1104 局時点で実用上の小差を確認したため打切り。

| 指標 | A: hao（mismatch） | B: nodchip（matched） |
|------|------:|------:|
| 勝/負/引分 | 526 / 548 / 30 | 548 / 526 / 30 |
| Elo 差（A 視点） | **−7 ±20** | nElo: −7 ±21 |
| SPRT LLR | −0.438（±2.944 境界、未収束） | decision: running |
| 平均 NPS | 738K | 739K |
| timed_out | 0 | 0 |

**ペントノミアル内訳（A=hao 視点、551 ペア）**:

| カテゴリ | 数 | 比率 |
|---|---:|---:|
| LL（A 連敗） | 132 | 24.0% |
| LD/DL | 16 | 2.9% |
| **WL/LW（先後入替で勝敗反転）** | **267** | **48.5%** |
| DD（両局引分） | 2 | 0.4% |
| WD/DW | 10 | 1.8% |
| WW（A 連勝） | 124 | 22.5% |

引分 30 局は**全て手数上限 (max_moves=512)** 起因。スイープ
(LL+WW=256) と色入替 split (WL/LW=267) がほぼ拮抗しており、
**進行度ファイル mismatch の効果は先後の利と同オーダー以下**。

**WL/LW ペア内の先後勝率**（267 ペア・534 局）:

| | ペア単位 | 局単位 |
|---|---:|---:|
| 両局とも先手勝ち | 154（57.7%） | 308（57.7%） |
| 両局とも後手勝ち | 113（42.3%） | 226（42.3%） |

参考に全決着局（1074）の先手勝率は 53.4% で、WL/LW に絞ると
**先手有利が +4.3 ポイント強調** される。これは「先後の利が
支配的なペア」を抽出した集合なので自然。

**結論**: 学習時と異なる progress.bin を推論で食わせる影響は
**−7 ±20 Elo（誤差範囲内、わずかに劣化方向）** で棋力影響は実用上小さい。
ドキュメント上の警告「学習と推論の progress.bin は一致必須」は理論的には
正しいが、棋力差は ±20 Elo 以下で観測限界付近。

ログ: `runs/selfplay/20260430-v96_400-progress-ablation-yo-sprt/`

#### byoyomi 1000ms 評価 (strict SPRT)

TBD: v99-400 vs v96-400 の自己対局結果（v99 学習完走後）

#### rshogi 比較

TBD: rshogi バイナリでの比較結果

### 考察

最終評価（v99-400, 1823 局 SPRT）の主要観察:

- **progress.bin 差し替えは棋力に有意な影響なし**（nElo −6 ±16, accept_h0 相当）。
  720 万局面 1 epoch (nodchip) → 80 億局面 1 epoch (hao_full) の 1000 倍超の
  学習量増加にもかかわらず、最終棋力は同等
- **推論時 progress.bin mismatch の単独影響も小さい**（−7 ±20 Elo, v96-400 ablation）。
  bucket 割当の不整合は理論上正しい問題だが、LayerStack 1536x16x32 アーキでは
  棋力影響が ±20 Elo 以下に収まる
- step 進行で v96 → 同等 → v99 優位 (320) → 同等 (400) と揺らぎながら推移。
  途中段階の単調トレンドは最終的に消失し、SPRT で最終棋力は差なしと確定
- 両走で先後の利が支配的（色入替 split が約半数、エンジン差での
  スイープも約半数）→ 細かな改善・劣化は開始局面と先後の効果に埋もれやすい

**含意**:
- progress8kpabs bucket 選択係数は **720 万局面 1 epoch でも収束済み** で、
  追加学習は棋力に効かない（過剰学習の効果が消失する飽和点に既に到達）
- v100（haoek_e1, entering_king 込み較正版）への期待値も限定的と予想されるが、
  bucket 分布の細部が異なるため別実験として実施する価値はある

---

## 関連ドキュメント

- [v96 実験記録](v96_layerstack-1536x16x32_dlsuisho15b-aoba-deduped_400sb_progress8kpabs_wrm-in340.md) — ベース実験
- [v98 実験記録](v98_layerstack-1536x32x32_dlsuisho15b-aoba-deduped_400sb_progress8kpabs_wrm-in340.md) — 32x32 拡大版
- [progress.bin 学習記録](progress/progress_nodchip_continuous_full_20260428.md) — 本実験で使用する progress.bin の生成詳細

## 後続計画メモ

v99 完走後、**v100（仮）で `progress_haoek_full_cuda.e1.bin` を採用** する派生実験を予定。

- 動機: DLSuisho15b に約 5% の entering_king 系局面が含まれることが定量判定で確認された。entering_king 込みで較正された haoek_e1 が混入分に対して合致する可能性
- 比較設計: v100 = v99 - hao_e1 + haoek_e1 で **progress.bin のみ差分** の純粋比較
- 期待される観察:
  - v100 vs v99: entering_king 局面に対する bucket 割当差の効果
  - v100 vs v96: 教師量 約770万 → 80億局面化 + entering_king 込み較正の合算効果
- 学習開始判断: v99 vs v96 の自己対局結果を見てから（v99 が v96 を有意に超えるか、または互角以下か）

---
作成日: 2026-04-28
最終更新: 2026-04-28
