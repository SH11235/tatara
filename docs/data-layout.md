# データ配置規約

rshogi-nnue が扱う **PSV / Pack / progress.bin / .nnue / checkpoint / ログ**
の配置・命名・bullet-shogi 互換性を定義する。Stage 1 以降で生成する自前
データもこの規約に従う。

## 物理配置 (WSL2 + E:)

- リポルート `~/git-repos/rshogi-nnue/` は `/` (C: 上の ext4 vhdx) に置く
- 大容量データは **`/mnt/e/rshogi-nnue/`** (E: ドライブ, NTFS DrvFs, 物理別 SSD) に配置
  - WSL2 の `/` は C: 上 sparse vhdx で物理空きが C: に縛られる (詳細は [docs/setup.md](setup.md#wsl2-ディスク注意))
- リポ内 `data/` は **`/mnt/e/rshogi-nnue/data` への symlink** で実体は E: 側
- 学習出力・ログも E: 側に置く方針 (`/mnt/e/rshogi-nnue/output/`, `.../logs/`)。`output/` `logs/` の symlink 化は Stage 1 着手で実体を作る時に判断 (現状は `data/` のみ symlink 済)

## ディレクトリ構成

```
data/                                    # → /mnt/e/rshogi-nnue/data symlink
├── nodchip_hao_depth9/                  # 連続 PSV (1016 files / 299 GB)
│   └── kifu.tag=train.depth=9.num_positions=1000000000.start_time=<unix>.thread_index=NNN.bin
├── nodchip_suisho5_entering_king/       # 連続 PSV 入玉特化 (127 files / 19 GB)
├── DLSuisho15b_aoba_deduped_shuffled.bin  # shuffle 済 (NNUE 学習本番, 616 GB)*
├── DLSuisho15b_deduped_shuffled.bin       # 同上 aobazero 抜き (547 GB)*
├── aobazero_kd_20240329/                # アオバゼロ知識蒸留教師 (連続)*
├── progress/                            # bullet-shogi 由来 progress.bin
└── smoke_progress/                      # smoke 用 PSV + 出力比較対象

output/                                  # rshogi-nnue 自前の学習出力 (Stage 1+ で運用)
├── progress/                            # 自前 progress.bin
├── nnue/                                # 最終 .nnue
└── checkpoints/                         # 中間 checkpoint (optimizer state 等)

logs/                                    # 学習・実験 log (Stage 1+ で運用)
```

`*` 印は環境依存で存在しない場合あり。本マシンは Stage 0 時点で
`nodchip_hao_depth9/` + `nodchip_suisho5_entering_king/` + `progress/` +
`smoke_progress/` の合計 318 GB のみ転送済み。`DLSuisho15b_*` と
`aobazero_*` は Stage 3 (NNUE training) 着手時に判断。

## ファイル命名規約

### 連続 (game-relative) PSV

bullet-shogi (上流 nodchip) の命名をそのまま採用:

```
kifu.tag=<dataset>.depth=<N>.num_positions=<N>.start_time=<unix>.thread_index=NNN.bin
```

| フィールド | 意味 | 例 |
|---|---|---|
| `dataset` | データセット種別 | `train`, `suisho5.entering_king` |
| `depth` | gensfen 探索深さ | `9` |
| `num_positions` | gensfen の上限値 (生成停止後は数百MBに縮小) | `1000000000` |
| `start_time` | gensfen 起動 unix 時刻 (ファイル群の区別) | `1695340981` |
| `thread_index` | 並列スレッドインデックス (3 桁ゼロ埋め) | `000`, `127` |

`game_ply` 単調減少で対局境界を検出するため **シャッフルせず** に保存する。
`docs/experiments/progress/progress_nodchip_continuous_full_20260428.md`
で bullet-shogi 側の使用例を詳述している。

### Shuffle 済 PSV

```
<source>[_<modifier>]_shuffled.bin
<source>[_<modifier>]_deduped_shuffled.bin
```

例:

| ファイル | 内容 |
|---|---|
| `DLSuisho15b_deduped_shuffled.bin` | DLSuisho15b 教師、dedup + shuffle 済 |
| `DLSuisho15b_aoba_deduped_shuffled.bin` | 上記 + aobazero 教師混合 |

**ファイル名に `_shuffled` を含めば shuffle 済**。`game-relative` モードでは
原理的に使えない (game_ply 単調減少が崩れる) ので NNUE 本番学習
(HalfKA_hm 1536-16-32 等、shuffle が望ましい) 専用。

### Pack ファイル

将来追加予定 (bullet 由来)。`*.pack` で統一、shuffle 済かは中身に metadata
で持たせる方針。Stage 2 以降で詳細化。

### progress.bin

bullet-shogi 命名を踏襲:

```
<data_label>_<scope>[_<backend>].bin           # 最終 (= 最大 epoch と同内容)
<data_label>_<scope>[_<backend>].e<N>.bin      # epoch N の checkpoint
```

| フィールド | 意味 | 例 |
|---|---|---|
| `data_label` | 教師データ構成 | `haoek` (hao + entering_king), `nodchip` (nodchip 公式) |
| `scope` | データ範囲 | `full`, `e1_f1` (1 epoch × 1 file) |
| `backend` | 学習バックエンド | `cuda` (GPU 版), 無印 (CPU 単スレッド版) |
| `eN` | epoch checkpoint | `e1`, `e2`, ..., `e5` |

例:

- `data/progress/nodchip_progress_e1_f1_cuda.bin` — nodchip 公式 1 file × 1 epoch
- `data/progress/progress_haoek_full_cuda.bin` — bullet-shogi 1143 files × 5 epoch
- `data/progress/progress_haoek_full_cuda.e1.bin` — 上記の epoch 1 checkpoint

rshogi-nnue が **自前で生成** する progress.bin は `output/progress/` 配下に
同じ命名規約で出す (上流由来と path で分離)。サイズは YaneuraOu 互換の
**1,003,104 bytes** で固定 (異なれば異常)。

### .nnue

NNUE 重み (YaneuraOu 互換 binary)。

```
output/nnue/<model_id>.nnue
```

`model_id` は実験 ID + 時刻 (例: `v100_20260601_223344`)。Stage 3 で詳細確定。

### Checkpoint

学習途中の中間状態 (重み + optimizer state 等)。

```
output/checkpoints/<run_id>/<step>.ckpt
```

形式 (`*.ckpt` の中身) は Stage 3 で確定。学習中断・再開やアンサンブル取得に使う。

### ログ

```
logs/<experiment_id>/<event>.log
```

Stage 1 で初出 (cuda-oxide kernel の per-step loss、bench、numerical
equivalence diff 等)。詳細は Stage 1 着手時に決定。

## bullet-shogi との互換性

- **連続 PSV はファイル名・配置とも bullet-shogi 完全一致** → bullet-shogi の
  `generate_file_list.sh` 等のスクリプト・既存の cuda 学習プロセスがそのまま
  使える (Stage 1 で numerical equivalence 検証時にも便利)
- **Shuffle 済 PSV (DLSuisho15b 等) も同名で配置** — 持ち込み方は別 issue
- **bullet-shogi 由来の progress.bin** (`progress_haoek_full_cuda.bin` 等) は
  `data/progress/` で参照可、Stage 1 で比較対象として使う

rshogi-nnue が **新規生成** するファイル (自前 progress.bin / .nnue /
checkpoint) は **`output/` 配下に分離** し、上流由来データと混ざらないようにする。

## .gitignore 方針

PSV / Pack / .nnue / progress.bin / checkpoint / ログ系は **すべて git 管理外**:

- `data/` 全体 → 既に `/data` + `**/*.bin` + `**/*.psv` + `**/*.pack` +
  `**/*.nnue` で除外済み
- `output/` 全体 → 本 PR で追加
- `logs/` 全体 → 本 PR で追加
- `checkpoints/` 全体 → 本 PR で追加 (`output/checkpoints/` がメインだが、
  実験ごとにルート直下に作る場合もあるので両カバー)

詳細は本リポの `.gitignore` を参照。

## 関連

- [docs/setup.md](setup.md) — WSL2 + E: 物理配置の根拠
- `docs/experiments/progress/progress_nodchip_continuous_full_20260428.md` —
  bullet-shogi 側 PSV / 学習仕様の詳述 (連続 PSV 命名・進行度学習プロセス)
- `docs/experiments/v99_layerstack-1536x16x32_dlsuisho15b-aoba-deduped_400sb_progress8kpabs_hao-progress_wrm-in340.md` —
  shuffle 済 PSV (DLSuisho15b_aoba_deduped) を使った v99 NNUE 学習の実例
