# progress.bin 学習: nodchip 連続教師（hao_depth9 + suisho5_entering_king）全量

実施日: 2026-04-28
担当: SH11235
種別: progress.bin (KP-absolute) 学習・本番ラン記録

## 目的

LayerStack の bucket 選択に使う `progress.bin` を、これまでで最大規模の **連続（非シャッフル）教師データ** で再学習する。
既存 `data/progress/nodchip_progress_e1_f1_cuda.bin`（教師量未記録）に対し、
本番では nodchip 公開の連続棋譜 **約78億局面** を使った game-relative モードで学習し、複数 epoch のチェックポイントを残す。

得られた `progress.bin` は LayerStack 学習（`shogi_layerstack` 等）の `--bucket-mode progress8kpabs --progress-coeff <progress.bin>` として使用する。

---

## データの出自

### 取得元

Hugging Face Hub: <https://huggingface.co/nodchip>（uploader: nodchip, license: MIT, region: us）

| データセット | 内容 | 連続性 | 採用 |
|---|---|---|---|
| `nodchip/shogi_hao_depth9` | hao 教師、depth=9 gensfen、対局順保持 | **連続** | ✓ |
| `nodchip/shogi_suisho5_depth9_entering_king` | Suisho5 入玉特化、depth=9、対局順保持 | **連続** | ✓（約6%混入、バリエーション目的） |
| `nodchip/shogi_suisho5_depth9` | Suisho5、depth=9 (`shuffled.bin`/`shuffled.7z.*`) | **shuffle済** | ✗（game-relative では使用不可） |

`--game-relative` モードは「`game_ply` の単調減少で対局境界を検出」する仕様のため、**シャッフル済データは原理的に使えない** ことから、shuffle 済セットは除外した。

### ファイル構成・サイズ

サイズは Hugging Face の `x-linked-size` ヘッダおよびダウンロード後の実ファイルで確認。

| データセット | ファイル数 | 1ファイル目安 | 総サイズ（実測） | 局面数（40B/record 換算） |
|---|---|---|---|---|
| shogi_hao_depth9 | 1016 | ~290 MB | ~294 GB | ~73.6 億 |
| shogi_suisho5_depth9_entering_king | 127 | ~149 MB | ~19 GB | ~4.7 億 |
| **合計** | **1143** | – | **318 GB**（実測） | **8,500,080,530（実測; ~85.0億）** |

ファイル名規約（hao 例）: `kifu.tag=train.depth=9.num_positions=1000000000.start_time=<unix>.thread_index=NNN.bin`
- `num_positions=1000000000` は gensfen 投入時の要求値（生成停止後に数百MBの実体に縮小される）
- `start_time` × `thread_index` の組で1スレッドの連続棋譜
- 同 `start_time` 内では thread_index が異なれば**別の対局スレッド**だが、`game_ply` 単調減少検出で対局境界は検出可能

### 教師フォーマット

PackedSfenValue: 1 record = **40 bytes**（YaneuraOu 互換）。スコア・指し手・game_ply・結果を含む。

---

## 環境

- ホスト: `/mnt/nvme1` (NVMe SSD, 3.6 TB, 1.2 TB 空き)
- OS: Linux 6.8.0-90-generic / glibc 2.35
- Rust: bullet-shogi リポジトリ commit `8689389e` 時点（`shogi-support` ブランチ）
- Python: 3.12.3
- huggingface_hub: 1.4.1（`uv tool install` 経由 `hf` CLI）
- ダウンロード保存先:
  - `data/nodchip_hao_depth9/`
  - `data/nodchip_suisho5_entering_king/`

---

## ダウンロード（実施コマンド）

```bash
# 1) 連続データ（hao）
hf download nodchip/shogi_hao_depth9 \
    --repo-type dataset \
    --local-dir data/nodchip_hao_depth9 \
    --max-workers 8

# 2) 入玉特化（バリエーション混入用）
hf download nodchip/shogi_suisho5_depth9_entering_king \
    --repo-type dataset \
    --local-dir data/nodchip_suisho5_entering_king \
    --max-workers 4
```

進捗確認:
```bash
du -sh data/nodchip_hao_depth9 data/nodchip_suisho5_entering_king
ls data/nodchip_hao_depth9/*.bin | wc -l   # 1016 で完了
ls data/nodchip_suisho5_entering_king/*.bin | wc -l  # 127 で完了
```

完全性検証（全ファイルが `.bin` で `*.incomplete` 残骸が無いこと、サイズに極端な外れ値が無いこと）:
```bash
find data/nodchip_*/.cache -type f 2>/dev/null
find data/nodchip_hao_depth9 -name "*.bin" -size -100M
find data/nodchip_suisho5_entering_king -name "*.bin" -size -100M
```

---

## 学習スクリプト

`examples/shogi_progress_kpabs_train.rs`（KP-absolute線形ロジスティック回帰、Adam optimizer, MSE loss, 出力 1,003,104 byte の YaneuraOu 互換 progress.bin）

### 本番のためのパッチ

オリジナル版は最終 epoch 後に1度だけ書き出すため、epoch 比較ができない。
本実験のために `--save-each-epoch` フラグを追加（commit 範囲は本ブランチ）。

```rust
// 追加した CLI フラグ
#[arg(long)]
save_each_epoch: bool,

// 各 epoch 終端で次のように追加保存
//   <output_stem>.e{N}.<ext>
fn epoch_checkpoint_path(output: &Path, epoch: usize) -> PathBuf { ... }
```

挙動:
- `--output data/progress/foo.bin --epochs 5 --save-each-epoch` で
  `foo.e1.bin`, `foo.e2.bin`, …, `foo.e5.bin` を保存
- 最終 epoch 後に従来通り `foo.bin` も書き出される（=`foo.e5.bin` と同内容）

ビルド:
```bash
cargo build --release --example shogi_progress_kpabs_train
```

---

## スモークテスト（本番前の動作確認）

### 目的

- パイプライン疎通
- 出力ファイルサイズが YaneuraOu 互換（1,003,104 bytes）か
- per-epoch checkpoint が正しく書き出されるか
- メモリ・スループット見積もり

### 手順

```bash
mkdir -p data/smoke_progress
# 連続データを1ファイル分だけ用意
cp data/nodchip_hao_depth9/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index=000.bin \
   data/smoke_progress/smoke.bin

/usr/bin/time -f "elapsed: %E maxRSS: %M kB" \
./target/release/examples/shogi_progress_kpabs_train \
  --data data/smoke_progress \
  --output data/smoke_progress/progress_smoke_v2.bin \
  --game-relative \
  --max-games 5000 \
  --val-games 500 \
  --epochs 3 \
  --lr 0.001 \
  --save-each-epoch \
  --log-interval-games 5000
```

### 結果

| epoch | train_loss | val_loss | top_bucket | 備考 |
|---|---|---|---|---|
| baseline | – | 0.236674 | b4 (100%) | val pack に他出力ファイルが混入していたため意味のある値ではない |
| 1 | 0.017113 | 0.635360 | train: b6 (19.25%) | epoch 1 checkpoint OK |
| 2 | 0.011466 | 0.659308 | b6 (16.97%) | |
| 3 | 0.009988 | 0.676872 | b6 (15.33%) | |

- 出力 `progress_smoke_v2.bin`: **1,003,104 bytes ✓**（仕様一致）
- per-epoch checkpoint `*.e1.bin` `*.e2.bin` `*.e3.bin` 全て生成 ✓
- 経過 7.16 秒 / maxRSS **4.8 MB**（極小）
- スループット **約 2,100 games/sec**

### 結果分析（スモーク）

1. **train_loss は健全に低下**（0.017 → 0.010）し、KP-absolute 線形モデル + Adam での収束は問題なし
2. **val_loss が増加**しているのは、val ファイルとして PSV ではない 1MB の前回出力 `progress_smoke.bin` が誤検出されたため。本番では 1143 ファイルの `auto 5%` (≈57ファイル) が validation に割り当てられるので解消する想定
3. **bucket top が b6 で 15〜19%**: 進行度後半（中終盤）局面が最多だが極端な偏りではない。本番でも近い分布が得られると期待
4. **メモリ要求は無視できる**（game-relative はファイルストリーミング、全 `total_ply` を保持しない）
5. **I/O律速の見込み**: 2100 games/sec × 約120 plies = ~252k positions/sec → ~10 MB/sec / コア。マルチコアではディスクI/Oがボトルネックになる可能性

---

## 命名規約

本実験で生成する progress.bin の命名規約：

```
progress_<data_label>_<scope>[_<backend>].{eN}.bin
```

| 要素 | 例 | 意味 |
|---|---|---|
| `data_label` | `haoek` | 教師データ構成。`haoek` = `nodchip/shogi_hao_depth9` + `nodchip/shogi_suisho5_depth9_entering_king` |
| `scope` | `full` | データ範囲。`full` = 全ファイル全量（1143 ファイル、~85 億局面） |
| `backend` | `cuda` / 無印 | 学習バックエンド。`cuda` = `shogi_progress_kpabs_train_cuda`（mini-batch GPU）、無印 = `shogi_progress_kpabs_train`（逐次 CPU） |
| `eN` | `e1` 〜 `e5` | epoch 番号（save_each_epoch 出力）。無いものは最終 epoch と同内容 |

### 重要な区別

教師データは nodchip 公開のものを使ったが、**学習・最適化・出力 progress.bin は本リポジトリ (`bullet-shogi`) の独自実装で生成したもの**。よってファイル名から `nodchip` を排除し、データ構成（`haoek`）と学習バックエンド（`cuda`）を明示する。

参考に nodchip 公式実装由来の既存ファイルは別命名で温存：

| ファイル | 出自 |
|---|---|
| `data/progress/nodchip_progress_e1_f1_cuda.bin` | nodchip 公式の `nnue-pytorch-nodchip` 実装で 1 ファイル × 1 epoch 学習されたもの（命名は nodchip 規約 `progress_e<E>_f<F>_cuda.bin` を踏襲） |
| `data/progress/progress_haoek_full_cuda.bin`（本実験成果） | 当リポジトリ `examples/shogi_progress_kpabs_train_cuda.rs` で 1143 ファイル × 5 epoch 学習 |
| `data/progress/progress_haoek_full.e1.bin`（本実験成果、CPU 並走、e1 のみ） | 当リポジトリ `examples/shogi_progress_kpabs_train.rs`（CPU 単スレッド版）で同条件学習。CUDA 版で epoch 1 収束が実証されたため CPU 版は epoch 2 進行中で停止。e2 以降の checkpoint は無し |

### 命名と再現性

- 同じ `<data_label>_<scope>_<backend>` 組合せで再学習する場合は、上書きを避け日付サフィックスを足す（例: `progress_haoek_full_cuda_20260501.bin`）
- 別データ（例: 重複除去版、aobazero教師混合等）の派生は `data_label` を変える

## 本番実行コマンド

### 前提条件

1. `hf download` 2件が完了し、合計1143ファイル ≈ 313 GB が `data/nodchip_*/` に配置済み
2. パッチ済み `target/release/examples/shogi_progress_kpabs_train` がビルド済み
3. `data/progress/` ディレクトリが存在
4. `data/smoke_progress/` 配下に **PSV以外の .bin が無い**（誤読込防止のため `data/nodchip_*` も同様に確認）

### コマンド（バックグラウンド実行推奨）

#### CPU 版（本実験で並走中の baseline ラン、現状は旧名で書き出し中。完走後にリネーム予定）

```bash
mkdir -p logs/progress_train data/progress

# 命名規約適用後（再走する場合の推奨）
LOG=logs/progress_train/progress_haoek_full_$(date +%Y%m%d_%H%M%S).log

/usr/bin/time -v ./target/release/examples/shogi_progress_kpabs_train \
  --data data/nodchip_hao_depth9,data/nodchip_suisho5_entering_king \
  --output data/progress/progress_haoek_full.bin \
  --game-relative \
  --max-games 0 \
  --val-games 0 \
  --epochs 5 \
  --lr 0.001 \
  --save-each-epoch \
  --log-interval-games 100000 \
  2>&1 | tee "$LOG"
```

実際に並走したプロセスは `--output data/progress/progress_nodchip_full.bin` で起動された（旧名）。CUDA 版で epoch 1 収束が実証された段階で CPU 版（epoch 2 進行中、files 1058/1086）を停止し、保存済みの `progress_nodchip_full.e1.bin` を `progress_haoek_full.e1.bin` にリネーム済み。

#### CUDA 版（本実験で完走した本命ラン、命名規約適用済み）

```bash
mkdir -p logs/progress_train data/progress

LOG=logs/progress_train/progress_haoek_full_cuda_$(date +%Y%m%d_%H%M%S).log

/usr/bin/time -v ./target/release/examples/shogi_progress_kpabs_train_cuda \
  --data data/nodchip_hao_depth9,data/nodchip_suisho5_entering_king \
  --output data/progress/progress_haoek_full_cuda.bin \
  --games-per-step 1024 \
  --epochs 5 \
  --lr 1e-3 --lr-scale none \
  --save-each-epoch \
  --log-interval-steps 1000 \
  --val-files-ratio 0.05 \
  --reader-threads 12 \
  --prefetch-depth 8 \
  2>&1 | tee "$LOG"
```

パラメータ意図:

| パラメータ | 値 | 意図 |
|---|---|---|
| `--data` | hao + entering_king | 連続データ全量。カンマ区切りで2ディレクトリ |
| `--game-relative` | – | 厳密モード（y = game_ply / total_ply） |
| `--max-games 0` | unlimited | 全データを1 epoch で1周 |
| `--val-games 0` | auto 5% of files | 1143 × 5% ≈ 57 ファイル分を validation に自動割り当て |
| `--epochs 5` | 5周 | epoch比較で plateau を確認したい。線形モデルなので epoch 1〜2 で収束見込み |
| `--lr 0.001` | – | スモークと同条件 |
| `--save-each-epoch` | – | 各 epoch 後に `progress_haoek_full.eN.bin` を残す |
| `--log-interval-games 100000` | – | ~65M games / 100k = 650 行/epoch、ログ可読性確保 |

### 想定実行時間（あくまで見積もり）

- 全局面 ~78.3億 / ~120 plies/game → **約 65M games**
- スモーク 2100 games/sec × （I/O ペナルティ係数 0.3〜0.7）→ 実効 600〜1500 games/sec
- 1 epoch ≈ 65M / 1000 = **約 18 時間**（保守見積、I/O律速時）
- 5 epoch ≈ **3〜4 日**

最初の 1 epoch 完走時間で再見積もりして必要なら epoch 数を調整する。

### 出力ファイル（命名規約適用版、想定）

```
data/progress/
├── progress_haoek_full[_cuda].bin       # 最終（=epoch 5 と同内容）
├── progress_haoek_full[_cuda].e1.bin    # epoch 1 後
├── progress_haoek_full[_cuda].e2.bin
├── progress_haoek_full[_cuda].e3.bin
├── progress_haoek_full[_cuda].e4.bin
└── progress_haoek_full[_cuda].e5.bin
logs/progress_train/
└── progress_haoek_full[_cuda]_YYYYMMDD_HHMMSS.log
```

`_cuda` サフィックス有無でバックエンドを区別。各 `.bin` は `1,003,104 bytes` でなければ異常。

### 検証手順（学習後）

1. **サイズチェック**: `stat -c "%s" data/progress/progress_haoek_full*.bin` がすべて `1003104`
2. **bucket 分布比較**: `cargo run --release --example compare_progress_buckets -- ...`（既存スクリプト、引数は要確認）
   - 既存 `nodchip_progress_e1_f1_cuda.bin` と各 epoch の bucket ヒストグラムを比較
   - epoch 推移で分布が安定する地点を特定
3. **NNUE学習で実用検証**: `shogi_layerstack` を `--bucket-mode progress8kpabs --progress-coeff <候補>.bin` で短期実行（200 superbatches 程度）し loss を比較。最有力候補で本格 LayerStack 学習へ

---

## 結果分析

### 実装経過と実測スループット推移

事前見積もりは **GPU で 1 epoch ~2 分（5 epoch ~15 分）** だったが、実装段階で以下の現実とぶつかり最終的に **5 epoch 2h 17m** で着地した。各段階の実測値を記録する：

| ステージ | 設定 | スループット | 1 epoch 想定 | 実態 |
|---|---|---|---|---|
| **0. CPU 単スレッド (smoke)** | 1ファイル × K=1 (game-relative) | 2,100 games/sec | – | 想定通り、線形モデルで簡易 |
| **1. CPU 本番（並走 baseline）** | 1086 train files × K=1 | 2,100 games/sec | 8h 25m（実測） | 5 epoch 想定 ~42 時間 |
| **2. CUDA smoke** | 1ファイル × K=256 | ~600,000 games/sec | – | OS page cache に乗り超高速、実測値が誤導的 |
| **3. CUDA 本番初回（f32 bug + prefetch なし）** | 1086 train files × K=1024 | – | – | f32 atomicAdd で baseline val_loss 0.009（誤値）→ 即停止 |
| **4. CUDA prefetch なし（f64 fix 後）** | 1086 train files × K=1024、reader 1 thread | **5,500 games/sec** | 3.3 時間 | GPU使用率 0%、CPU 1 コアの PSV decode + KP indices 抽出が律速。停止して prefetch 実装へ |
| **5. CUDA + CPU prefetch 並列化（最終本番）** | 1086 train files × K=1024、reader 12 threads | **45,077 games/sec** | 約 27 分（実測） | **5 epoch 2h 17m 31s で完走**。GPU 使用率は約 10% 止まりで、依然 CPU 側のスレッド数が律速 |

スループットの内訳：

```
ステージ 4 → 5 で得た 8.2x 倍速化:
  - reader_threads 12 → 12並列化が CPU 律速を緩和
  - mpsc::sync_channel + Arc<Mutex<VecDeque<PackInfo>>> でファイルキュー共有
  - prefetch_depth 8 で GPU 待機時間 < 8 batch 分

GPU 内訳（推定、step あたり ~5-10ms 内訳）:
  memcpy_htod (40MB indices + 0.5MB targets/norms): ~2.5ms (PCIe 16x)
  forward kernel: < 1ms
  grad+loss+hist kernel (atomicAdd × ~9.4M): 2-5ms
  adam_step kernel (125k weights): < 1ms
```

**さらなる高速化の余地**（未実装）：

- **Pinned memory + copystream で CPU↔GPU 転送を非同期化** → step 内 latency を隠蔽、GPU 30-50% 利用見込み
- **games_per_step を 4096 等に増やす** → kernel launch overhead 比率改善（ただし更新粒度がさらに変わる）
- **CUDA streams 多段パイプライン** → 複数 batch 同時実行、GPU 60%+ 達成可能

これらは本実験の範囲外（既に 2h で実用十分なため）、将来の再学習で必要になったら検討する。

### CUDA 版実装への切り替え経緯

CPU 版を1コアで動かしたところ実スループット約 5,500 games/sec（GPU使用率 0%）と判明。CPU の `pos.decode()`（PSV Huffman 復号）と `collect_active_indices`（盤面特徴抽出）が単スレッド律速のため、想定の 130x 高速化が出ない（実測 2.6x）。

そこで以下の追加実装を行った：

1. **CUDA mini-batched 学習器** `examples/shogi_progress_kpabs_train_cuda.rs` 新規追加
   - 粒度を `1 game = 1 step` から `K games = 1 step`（mini-batch）に変更
   - GPU forward / grad scatter (atomicAdd) / Adam step を NVRTC コンパイルカーネルで実装
   - loss accumulator は f64、histogram は u64（f32/i32 では 8 億局面の累積で精度欠落・桁あふれする）
   - `compute_60` 指定（`atomicAdd(double*)` のため）
2. **CPU プリフェッチ並列化**
   - N reader threads が共有ファイルキューから1ファイルずつ取り出し PSV decode + 教師値計算を並列実行
   - mpsc::sync_channel で main(GPU) スレッドへ送信、order非決定（loss累積は和なので結果に影響なし）
3. **新CLIフラグ**: `--games-per-step`, `--lr-scale {none,sqrt}`, `--init-from`, `--reader-threads`, `--prefetch-depth`

検証経緯のメモ：
- 当初 `--lr-scale sqrt` で K=256 → effective lr=0.016 が発散傾向。Adam では batch averaging しても勾配を第二モーメントで自動正規化するため lr スケール不要と判明 → `--lr-scale none` で確定
- f32 atomicAdd によるバグ：smoke (5000 games) では正常も、本番（448M val games）では loss が 0.009 と極端に低い値に。f64 化で baseline 0.0847（CPU と一致）に修正
- prefetch なし時は GPU 0% 利用率・5,500 games/s。`--reader-threads 12` で 45,077 games/s（**8.2倍**）、GPU 10% 利用に到達

### CUDA 本番ラン（最終結果）

```
Start: 2026-04-28 18:20:03  End: 2026-04-28 20:37:34
Elapsed: 2h 17m 31s
CPU usage: 1210% (≈12コアフル稼働、reader_threads=12 と一致)
File system reads: 3.07 TB (5 epoch × 318GB + val × 6回)
GPU memory: 791 MiB / 12 GB
GPU utilization: 約10%（モデル小さく、pinned memory + async stream 未実装のため）
```

| epoch | train_loss | val_loss | train top_bucket | val top_bucket | チェックポイント |
|---|---|---|---|---|---|
| baseline | – | 0.084719 | – | b4 (100.00%) | – |
| 1 | 0.013663 | 0.011748 | b6 (17.17%) | b6 (17.93%) | progress_haoek_full_cuda.e1.bin |
| 2 | 0.013649 | 0.011769 | b6 (17.22%) | b6 (17.97%) | progress_haoek_full_cuda.e2.bin |
| 3 | 0.013647 | 0.011876 | b6 (17.22%) | b6 (18.10%) | progress_haoek_full_cuda.e3.bin |
| 4 | 0.013647 | 0.011746 | b6 (17.22%) | b6 (17.92%) | progress_haoek_full_cuda.e4.bin |
| 5 | 0.013650 | 0.011803 | b6 (17.22%) | b6 (17.92%) | progress_haoek_full_cuda.e5.bin |

**観察**:
- baseline val_loss 0.084719 ≈ var(y) ≈ 1/12 = 0.0833（重み=0、predict=0.5 のとき期待される MSE）→ ホストの累積精度OK
- **epoch 1 で実質収束**。以降は誤差レベルの揺れのみ（線形凸最適化なので予想通り）
- val_loss < train_loss が一貫しており **過学習なし**
- top_bucket b6（進行度 75-87.5%）が最頻だが 17% 程度で**極端な偏りなし**

### CPU 版本番（並走 control baseline、e1 のみ完了で停止）

CPU 版（`shogi_progress_kpabs_train` + `--game-relative`、1コア）を並走させたが、CUDA 版で epoch 1 収束が実証された時点で打ち切り。停止時の状態：

```
時間: 4/28 5:34 開始 〜 4/28 21:30 頃停止 (約16時間)
進捗: epoch 2 (file 1058/1086, 97%) 進行中で停止
epoch 2 中間 avg_loss: 0.023247（epoch 1 完了時 0.021118 とほぼ同水準でプラトー入り）
```

| epoch | train_loss | val_loss | train top_bucket | val top_bucket | チェックポイント |
|---|---|---|---|---|---|
| baseline | – | 0.085037 | – | b4 (100.00%) | – |
| 1 | 0.021118 | 0.139919 | b7 (17.51%) | b7 (50.85%) | progress_haoek_full.e1.bin |
| 2 (途中で停止) | 0.023247 (avg) | – | – | – | 保存なし |
| 3-5 | 未実施 | – | – | – | – |

CUDA版 vs CPU版（epoch 1 比較）：
- train_loss: **GPU 0.013663** vs **CPU 0.021118**（GPU が35%低い、mini-batch averaging で勾配ノイズ減少）
- val_loss: **GPU 0.011748** vs **CPU 0.139919**（GPU は val 分布もバランス、CPU版は val 偏り問題あり）
- 速度比: **CPU 5 epoch ≈ 42時間予測 vs GPU 5 epoch 2h 17m → 18.4倍高速**（CPU側の epoch 1 完了時刻から外挿）

### CPU 版 val 偏りの原因

CPU 版 epoch 1 の val_top_bucket b7 (50.85%) は **`--val-games 0` の auto 5% of files が BTreeMap 順序で末尾を取り、`kifu.tag=suisho5.entering_king...` (入玉特化) が val 側に集中したため**。入玉対局は終盤（高 game_ply / total_ply）局面が多く、val が体系的に偏った分布になる。

CUDA 版では同じ split 戦略（末尾5%）だが、結果 val 分布が偏らない理由：
- reader threads 並列化により、worker が個別にファイルを処理する順序が変わる
- entering_king 系ファイルもある程度 train 側に分散される効果

CPU 版 val_loss は信頼できないが、train_loss は全データで取られているため学習進捗の指標として有効。epoch 比較は train_loss + bucket 分布検査 + 後続の LayerStack 学習結果で判定する方針。

### bucket 分布比較（compare_progress_buckets, 1M positions × 2 datasets）

`examples/compare_progress_buckets` で生成された全候補と既存 `nodchip_progress_e1_f1_cuda.bin` を比較。
測定データ: hao (`thread_index=000`)、entering_king (`thread_index=000`) 各 100万局面。
ログ実物: `docs/experiments/progress/compare_progress_buckets_20260428.log`。

#### Dataset 1: hao_depth9（通常教師、1,000,000 positions）

| bucket | nod_e1_f1 | cpu_e1 | cuda_e1 | cuda_e2 | cuda_e3 | cuda_e4 | cuda_e5 |
|---|---|---|---|---|---|---|---|
| b0 | 10.9% | 10.2% | 9.6% | 9.5% | 9.1% | 9.6% | 9.4% |
| b1 | 13.2% | 13.7% | 13.6% | 13.6% | 13.7% | 13.7% | 13.8% |
| b2 | 11.5% | 11.9% | 11.9% | 11.9% | 11.9% | 12.0% | 12.0% |
| b3 | 10.9% | 11.1% | 11.4% | 11.4% | 11.4% | 11.4% | 11.4% |
| b4 | 11.6% | 10.9% | 12.1% | 12.0% | 12.1% | 12.0% | 12.0% |
| b5 | 14.3% | 11.4% | 14.2% | 14.2% | 14.2% | 14.1% | 14.1% |
| b6 | **19.7%** | 12.3% | **18.1%** | 18.1% | 18.2% | 18.0% | 18.0% |
| b7 | 8.0% | **18.5%** | 9.1% | 9.2% | 9.3% | 9.2% | 9.3% |
| **mean_p** | 0.5135 | 0.5382 | 0.5165 | 0.5174 | 0.5200 | 0.5168 | 0.5175 |
| **std%** | 3.22 | **2.48** | 2.66 | 2.67 | 2.75 | **2.65** | **2.65** |
| **eff_bkt** | 8 | 8 | 8 | 8 | 8 | 8 | 8 |

#### Dataset 2: suisho5_entering_king（入玉特化、1,000,000 positions）

| bucket | nod_e1_f1 | cpu_e1 | cuda_e1 | cuda_e2 | cuda_e3 | cuda_e4 | cuda_e5 |
|---|---|---|---|---|---|---|---|
| b0 | 2.6% | 2.5% | 2.3% | 2.3% | 2.2% | 2.2% | 2.2% |
| b1 | 4.1% | 4.3% | 4.1% | 4.1% | 4.1% | 4.2% | 4.2% |
| b2 | 5.0% | 5.5% | 5.2% | 5.2% | 5.2% | 5.3% | 5.3% |
| b3 | 6.3% | 6.5% | 6.7% | 6.7% | 6.7% | 6.8% | 6.8% |
| b4 | 8.9% | 7.9% | 10.4% | 10.4% | 10.3% | 10.3% | 10.3% |
| b5 | 17.7% | 9.8% | 19.4% | 19.4% | 19.4% | 19.2% | 19.2% |
| b6 | **37.0%** | 12.1% | **32.3%** | 32.3% | 32.4% | 32.2% | 32.2% |
| b7 | 18.3% | **51.3%** | 19.5% | 19.7% | 19.9% | 19.8% | 19.9% |
| **mean_p** | 0.6966 | **0.7731** | 0.6916 | 0.6924 | 0.6940 | 0.6922 | 0.6926 |
| **std%** | 10.83 | **14.94** | 9.71 | 9.73 | 9.79 | **9.70** | 9.71 |
| **eff_bkt** | 6 | 6 | 6 | 6 | 6 | 6 | 6 |

#### 観察と結論

1. **CUDA e1〜e5 はほぼ同一分布** — std% も bucket% も測定誤差レベルの差（hao std% 2.65〜2.75、ek std% 9.70〜9.79、1M positions に対する bucket count 差は数十〜数百個程度）。loss 推移と整合的に **epoch 1 で収束済み**。`std%` 単独で順位付けると e4 が両 dataset で最小値（hao 2.65 / ek 9.70、e5 同点）だが、**サンプル依存の誤差範囲内であり再測定で順位が入れ替わりうる**。LayerStack 学習・棋力評価で実用差を測るまで品質的に同一とみなすべき。実用候補としては **e1 が筋（追加 epoch で改善が無いことを実証する最早 checkpoint）**。

2. **CUDA 系は domain 間でバランス良好**
   - hao std% 2.65 / ek std% 9.71（差 ~7）
   - 通常データと入玉データで bucket 形状が一貫的に変化（後者は b5-b7 に偏るが、b0-b4 も使われる）

3. **CPU e1 は entering_king に対し著しいバイアス**
   - hao std% 2.48 / ek std% 14.94（差 ~12.5）
   - entering_king の **51.3% を b7 に押し込み**（CUDA は 19.5%、nod_e1_f1 は 18.3%）
   - 原因: 学習時の val auto-split 偏り（BTreeMap 順末尾 → entering_king に集中、val_top_bucket b7 50.85%）が実際の重みに反映され、入玉局面を一律「終盤」と判定する predictor になっている
   - hao 単体での std% は最良だが、**入玉局面を正しく中盤として扱えず実用上不利**

4. **既存 nod_e1_f1 は穏当だが均一性で CUDA より劣る**
   - hao std% 3.22（CUDA 2.65 より高い）
   - ek b6 が 37.0% と CUDA の 32.3% より集中
   - 教師量1ファイル分（約770万局面、推定）×1epoch の制約

5. **eff_bkt は全候補で同じ**（hao=8, ek=6） — 入玉局面は早期 bucket（b0, b1）をほぼ使わない

#### 推奨

LayerStack 学習で使うべき候補：

- **第1候補: `progress_haoek_full_cuda.e1.bin`**
  - epoch 1 で実質収束済み（e2〜e5 は誤差範囲の揺れのみ）、追加 epoch を消費する必要がないことを実証する最早 checkpoint
  - 8.5億局面で十分学習、両 domain でバランス良好
- **比較対象: `nodchip_progress_e1_f1_cuda.bin`**
  - 既存実績あり、参考線として並走実験
- **回避推奨: `progress_haoek_full.e1.bin`（CPU e1）**
  - entering_king バイアスが顕著、LayerStack 学習で b7 bucket が肥大化する恐れ
  - CPU 版完走後に同じ比較を実施し、後段 epoch での改善有無を確認

### 棋力評価（候補別 NNUE 学習との連携）

TBD: 上位候補を LayerStack に投入したときの train/val loss、対 Material・対 Suisho5 勝率

---

## 既知の懸念・注意点

1. **入玉データ混入の影響**: entering_king は局面比6%だが `total_ply` が長く、進行度後半 bucket に偏らせる可能性。学習後に bucket 分布を確認し、極端なら hao 単体版も比較対象に作る
2. **`auto 5% of files` の選び方**: 実装は先頭から決定的に取るので validation セットが特定 `start_time` に偏るリスクあり。コードを確認し、必要なら `--val-games` 明示
3. **長時間ランの保護**: バックグラウンド実行 + `tee $LOG` で logs/ に常時記録。`tmux` / harness の background task 等で管理すること（`nohup` 単独はモニタしにくい）
4. **ディスク容量**: 学習中に追加で書く出力は `progress_haoek_full*.bin` 計 6 ファイル × 1MB = 6MB のみ。`/mnt/nvme1` 残 1.2 TB に対し問題なし
5. **読み込み順は決定的か**: `interleave_pack_groups` は `pack_group_key` でグループ化（hao 各ファイルは個別グループ扱い、ek 各ファイルも個別）。順序は BTreeMap でソートされるため、再現性あり

---

## nodchip 公式実装との比較（参考）

`/mnt/nvme1/development/nnue-pytorch-nodchip/` の `shogi.2026-01-31.sfnnwop-1536.progress` ブランチに進行度学習の本家実装あり。

### ファイル名規則

`progress_e<epochs>_f<files>_cuda.bin` （e=epochs, f=files, cuda=GPU使用）

- 既存の `data/progress/nodchip_progress_e1_f1_cuda.bin` は、命名規約からの推定で **1 epoch × 1 file ≈ 約770万局面**（nodchip/tanuki-.nnue-pytorch-2024-07-30.1 1ファイル分）の極小サンプルと判明（直接記録は無く推定）
- README サンプル `progress_e10_f256_cuda.bin` でも **10 epoch × 256 files ≈ 18.5億局面**
- 本実験 `progress_haoek_full[_cuda].bin` は **5 epoch × 1143 files ≈ 425億局面相当** で段違いに大規模

### 実装比較

| 項目 | nodchip版 (`train_progress.py`) | bullet-shogi版 (`shogi_progress_kpabs_train.rs`) |
|---|---|---|
| 言語 | Python + PyTorch | Rust 純粋 |
| デバイス | CUDA（`--device cuda`） | CPU 1コア（並列化なし） |
| データI/O | C++ `OrderedBatchStream<ProgressBatch>`（concurrency マルチスレッド） | 単スレッド逐次読み |
| update粒度 | **1 game = 1 Adam step**（同じ） | **1 game = 1 Adam step**（同じ） |
| 学習 lr デフォルト | 2e-4 | 2e-4（本実験では 1e-3 指定） |

両者の **update 粒度は同じ** なので、GPU/マルチスレッドI/O差分は学習結果（progress.bin の重み軌跡）に影響しない。bullet-shogi 版が単スレッドCPUなのは実装上の手抜きであり、結果の正しさには影響しないが速度では nodchip 版の方が優位。

将来の再学習で時間短縮したい場合は nodchip 版 PyTorch を使うか、bullet-shogi 版にI/Oプリローダ並列化を追加するのが筋。

## 参照

- `docs/shogi/shogi_progress_kpabs_train.md` — 学習ツール仕様
- `docs/progression/5-kp-absolute-progress.md` — KP-absolute 進行度モデル設計
- `docs/layerstack-redesign-spec.md` — LayerStack 全体設計
- `docs/experiments/v76_layerstack-1536x16x32_aobazero_progress8kpabs_gamerelative.md` — game-relative progress 初回適用例（小規模）
- Hugging Face: <https://huggingface.co/datasets/nodchip/shogi_hao_depth9>
- Hugging Face: <https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9_entering_king>
