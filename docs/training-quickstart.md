# 学習 Quickstart

`nnue-train` で HalfKA_hm 1536-16-32 LayerStack NNUE を 1 から学習するための
最短手順。GPU は Ampere+ (sm_80+) 公式、Turing は `CUDA_OXIDE_TARGET=sm_75`。
toolchain と CUDA / LLVM の準備は [docs/setup.md](setup.md) を参照。

## 必要な入力 3 種

| ファイル | 形式 | 用途 | サイズ目安 |
|---|---|---|---:|
| 教師データ PSV | `PackedSfenValue` × N (40 bytes 固定 / 局面) | `--data` で渡す | 数百 GB |
| progress 係数 | `progress.bin` (f64 LE × 81 × `FE_OLD_END` = `1_003_104` bytes 固定) | `--progress-coeff` で渡す。9 bucket 振り分けに使う | 1.0 MB |
| (任意) pretrained NNUE | 量子化 `.bin` (`save_quantised` 形式) | `--init-from` で weight 注入 (optimizer は reset) | ~116 MB |

PSV / `.bin` / checkpoint の命名規約と配置は
[docs/data-layout.md](data-layout.md) を参照 (`data/` 配下に symlink を貼る
運用を推奨)。

## Step 1: progress.bin を生成 (まだ無い場合)

`progress-kpabs-train` で先に進行度係数を学習する。`--epochs` で総 epoch
数を指定し、epoch ごとに `<run-name>.e<N>.bin` が出力される。

```bash
target/release/progress-kpabs-train \
  --data <path/to/shuffled-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

`nnue-train` には任意の epoch checkpoint (`<run-name>.e<N>.bin`) を
`--progress-coeff` で渡す。どの epoch を採用するかは試行錯誤になる
(progress.bin は bucket 割当を決める係数で、NNUE 学習の収束とは独立な
ため何 epoch 必要かはデータ依存)。

## Step 2: nnue-train で本体を学習 (400 sb full run)

典型的な full run (400 superbatches × 6104 batches × 65536 positions
= ~160 GB 相当の position 通過):

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --progress-coeff <path/to/progress.e5.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches 400 --batches-per-superbatch 6104 --batch-size 65536 \
  --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
  --save-rate 20 --keep-checkpoints 4 \
  --threads 16 --bucket-mode progress8kpabs
```

| option | 目的 |
|---|---|
| `--win-rate-model` | WRM (win-rate-model) loss (`loss_wrm` kernel)。`net_output ≈ cp/600` で収束、量子化 (`QA=127 / QB=64 / FV_SCALE=28`) と整合。**量子化推論と整合する net を学習するなら必須**。未指定なら plain sigmoid-MSE。 |
| `--score-drop-abs 32000` | `|score| >= 32000` の局面を loss から除外 (詰み近傍の極端な値を弾く) |
| `--save-rate 20` | 20 sb ごとに `{net_id}-{sb}.bin` (量子化) + `{net_id}-{sb}.ckpt` (raw resume) を書き出す |
| `--keep-checkpoints 4` | raw `.ckpt` (~1.8GB/個) を直近 4 個だけ残す。量子化 `.bin` (~116MB) は常に全保持 |
| `--threads 16` | dataloader prefetch worker 数。各 worker が PSV decode + sparse 抽出 + bucket 計算を 1 回で済ませて先読み |

データ量の目安:

- 1 sb = 6104 batches × 65536 positions = 400M positions
- 400 sb full run = 160B positions

所要時間は GPU と構成 (FP16 モード有無) で変わる。RTX 3080 Ti での throughput・
400 sb ETA・GPU 機種別と構成別の目安は [docs/performance.md](performance.md) を参照。

## Step 3: 学習中断・再開

raw `.ckpt` は **weight + Ranger optimizer state (m / v / slow / step) + 現在
の superbatch 番号** を全部保存する。電源断や GPU エラーで止まっても完全
再開できる:

```bash
target/release/nnue-train \
  --data ... --progress-coeff ... \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches 400 --batches-per-superbatch 6104 --batch-size 65536 \
  --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
  --save-rate 20 --threads 16 --bucket-mode progress8kpabs \
  --resume checkpoints/<run-name>/<run-name>-180.ckpt
```

`--resume` あり (`--start-superbatch` 省略) なら checkpoint の sb +1 から
再開、`--start-superbatch N` 明示で過去 sb をやり直すことも可。

> **`--resume` と `--init-from` の違い**: `--init-from` は量子化 `.bin` から
> weight だけ注入し optimizer state を **reset** する (fine-tuning / continued
> training)、`--resume` は raw `.ckpt` から weight + optimizer 両方復元する
> (真の resume)。両者は排他指定。

## Step 4: 出力 artifact の見方

学習後 `checkpoints/<run-name>/` に出るもの:

| ファイル | 形式 | 用途 |
|---|---|---|
| `<run-name>-<sb>.bin` | 量子化 NNUE binary | **推論側に投入する artifact** (LayerStack format、`crates/nnue-format/src/layerstack_weights.rs` 参照) |
| `<run-name>-<sb>.ckpt` | raw f32 + optimizer state | `--resume` 用、推論には使わない (`--keep-checkpoints` で淘汰) |

`<run-name>-400.bin` が最終 net。棋力検証は将棋エンジン側に組み込んで
測定する。

## 動作確認 (smoke)

データ準備前に GPU 経路だけ確認したい場合は `--data` を省略すると `GpuTrainer`
の forward / backward path を 1 step だけ実行する smoke test が走る:

```bash
target/release/nnue-train
# → "[smoke] forward + backward OK" の趣旨のログが出れば GPU 経路は健全
```

または小規模 run (1 sb × 3 batches) で全 pipeline を 5 秒程度で回す:

```bash
target/release/nnue-train --data <PSV> --progress-coeff <progress.bin> \
  --output /tmp/smoke --net-id smoke \
  --superbatches 1 --batches-per-superbatch 3 --batch-size 65536 \
  --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
  --save-rate 1 --threads 4 --bucket-mode progress8kpabs
```

## トラブルシューティング

| 症状 | 原因 / 対応 |
|---|---|
| `kernel artifact nnue_train.{cubin,ptx,ll} not found` | 初回ビルド時 `cd bins/nnue_train && cargo-oxide build` で `.ll` を生成する必要がある。詳細は [docs/setup.md](setup.md) |
| `libcublas.so` 系の link / load エラー | CUDA Toolkit が `/usr/local/cuda` / `CUDA_HOME` / `CUDA_PATH` のいずれにも無い。`CUDA_TOOLKIT_PATH=/path/to/cuda-12.x` で明示する (build.rs / runtime 両方が同じ chain で解決) |
| `CUDA_ERROR_INVALID_PTX` (driver error 218) | sub-Ampere GPU (sm_75) で `CUDA_OXIDE_TARGET` 未設定。`CUDA_OXIDE_TARGET=sm_75` を export してから再ビルド + 実行 |
| pos/s が極端に低い (< 500K on RTX 3080 Ti) | `--threads` を CPU コア数の半分程度に設定、dataloader が prefetch 不足になっていないか確認。`NNUE_TRAIN_STEP_PROFILE=1` で phase breakdown を見る ([docs/performance.md](performance.md)) |
| `--batch-size % 16 != 0` で reject | tiled L1 kernel が `b % 16 == 0` を要求 (`debug_assert!` で fail)。16 の倍数を渡す (65536 なら確実に充足) |

## 関連

- [docs/setup.md](setup.md) — toolchain (LLVM / CUDA / cuda-oxide) セットアップ
- [docs/data-layout.md](data-layout.md) — PSV / progress.bin / .nnue 命名・配置
- [docs/performance.md](performance.md) — pos/s 期待値 + step profile 診断
