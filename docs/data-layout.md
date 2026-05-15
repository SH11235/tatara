# データ配置規約

rshogi-nnue が扱う PSV / progress.bin / .nnue / checkpoint / ログの配置・
命名・bullet-shogi 互換性を定義する。

## 物理配置

- リポルートはどこに置いてもよい (`~/git-repos/rshogi-nnue/` 等を推奨)。
- 学習データ (PSV、教師、shuffle 済 bin 等) は大容量 (合計数百 GB) のため、
  別ドライブを推奨。リポ内 `data/` から symlink を張ると扱いやすい。
- 学習出力 (progress.bin / .nnue / checkpoint) とログも別ドライブ推奨。
- WSL2 環境では `/` (C: ドライブ上 sparse vhdx) の物理空き制約があり、
  別物理ドライブ (例: `/mnt/e/`) に置くのが特に重要。詳細は
  [docs/setup.md](setup.md#wsl2-ディスク注意)。

## ディレクトリ構成

```
data/                                    # 学習データ (.gitignore 対象)
├── <connected_psv_a>/                   # 連続 (game-relative) PSV、数百 GB 規模
├── <connected_psv_b>/                   # 連続 PSV (別データセット)
├── <name>_shuffled.bin                  # shuffle 済 PSV (NNUE 本番、数百 GB 規模)
├── progress/                            # 既存 progress.bin (比較対象)
└── smoke_progress/                      # smoke 用 PSV + 出力比較対象

output/                                  # 学習出力 (.gitignore 対象)
├── progress/                            # 自前 progress.bin
├── nnue/                                # 最終 .nnue
└── checkpoints/                         # 中間 checkpoint (optimizer state 含む)

logs/                                    # 学習・実験ログ (.gitignore 対象)
```

## ファイル命名規約

### 連続 (game-relative) PSV

bullet-shogi の生成ファイル命名規約をそのまま採用:

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

### Shuffle 済 PSV

```
<source>[_<modifier>]_shuffled.bin
<source>[_<modifier>]_deduped_shuffled.bin
```

ファイル名に `_shuffled` を含むものは shuffle 済 (`game-relative` モードでは
原理的に使えない)。HalfKA_hm 等 shuffle が望ましい学習に使う。`_deduped_`
は事前 dedup 済を示す optional modifier。

### progress.bin

bullet-shogi 命名を踏襲:

```
<data_label>_<scope>[_<backend>].bin           # 最終 (= 最大 epoch と同内容)
<data_label>_<scope>[_<backend>].e<N>.bin      # epoch N の checkpoint
```

| フィールド | 意味 | 例 |
|---|---|---|
| `data_label` | 教師データ構成 | `haoek` (hao + entering_king) 等 |
| `scope` | データ範囲 | `full`, `e1_f1` (1 epoch × 1 file) |
| `backend` | 学習バックエンド | `cuda` (GPU 版), 無印 (CPU 単スレッド版) |
| `eN` | epoch checkpoint | `e1`, `e2`, ..., `e5` |

自前生成する progress.bin は `output/progress/` 配下に同じ命名規約で出す。
サイズは **1,003,104 bytes** で固定 (`f64 LE × 81 × FE_OLD_END`、異なれば異常)。

### .nnue

NNUE 重み binary。`output/nnue/<model_id>.nnue` 形式で配置する。

### Checkpoint

学習途中の中間状態 (重み + optimizer state)。`output/checkpoints/<run_id>/<step>.ckpt`。
学習中断・再開とアンサンブル取得に使う。具体的なフォーマットは
`crates/nnue-train/src/optimizer.rs` の host state binary format
(magic `b"RNGR"`) と `bins/nnue_train` の raw checkpoint (magic `b"RNRC"`) を
参照。

### ログ

`logs/<experiment_id>/<event>.log` (per-step loss、bench、numerical equivalence
diff 等)。

## bullet-shogi との互換性

- **連続 PSV はファイル名・配置とも bullet-shogi と一致** — bullet-shogi の
  生成スクリプト・既存 cuda 学習プロセスがそのまま使え、numerical equivalence
  検証時にも便利。
- **Shuffle 済 PSV** も同じ命名で配置する。
- **bullet-shogi 由来の progress.bin** は `data/progress/` で参照可。

自前生成するファイル (自前 progress.bin / .nnue / checkpoint) は `output/`
配下に分離し、上流由来データと混ざらないようにする。

## .gitignore 方針

PSV / .nnue / progress.bin / checkpoint / ログ系は **すべて git 管理外**:

- `data/` 全体 (PSV / Pack / .nnue / bin)
- `output/` 全体
- `logs/` 全体
- `checkpoints/` 全体 (実験ごとにルート直下に作る場合もあるので別途カバー)

詳細はリポルートの `.gitignore` を参照。
