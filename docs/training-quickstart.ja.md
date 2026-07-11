[English](training-quickstart.md) | **日本語**

# 学習 Quickstart

`nnue-train` で将棋 NNUE を 1 から学習するための最短手順。GPU は Ampere+
(sm_80+) 公式、Turing は `CUDA_OXIDE_TARGET=sm_75`。toolchain と CUDA / LLVM の
準備は [docs/setup.ja.md](setup.ja.md) を参照。

学習する NNUE はアーキテクチャ(`simple` / `layerstack`)と入力 feature set を
選んで決める(選択肢は [README](../README.ja.md) の「学習できる NNUE」を参照)。
本ページは 2 つの構成を例に手順を示す:

- **例 1: HalfKP NNUE**(`simple` アーキ)— 最小構成。bucket を使わず前準備が少ない
- **例 2: LayerStack NNUE** — 局面進行度の 9 bucket を使う構成

## 必要な入力

| ファイル | 形式 | 用途 | サイズ目安 |
|---|---|---|---:|
| 教師データ PSV | `PackedSfenValue` × N (40 bytes 固定 / 局面) | `--data` で渡す | 数百 GB |
| progress 係数 | `progress.bin` (f64 LE、玉 81 マス × KP-abs 駒入力 1548 = `1_003_104` bytes 固定) | `--progress-coeff` で渡す。LayerStack の 9 bucket 振り分け用 (simple では不要) | 1.0 MB |
| (任意) pretrained NNUE | 量子化 `.bin` (`save_quantised` 形式) | `--init-from` で weight 注入 (optimizer は reset) | — |

## 例 1: HalfKP NNUE を学習 (simple アーキ)

`simple` アーキは bucket を持たないので `progress.bin` が要らない。最小構成:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --feature-set halfkp \
  --superbatches <N> \
  --threads <N> \
  simple
```

`simple` は既定で `--arch 256x2-32-32` / `--activation crelu`。`--superbatches`
の決め方と、追加で指定できる option は下記「主な option」を参照。

## 例 2: LayerStack NNUE を学習

`layerstack` アーキは局面進行度の 9 bucket を使うため、先に bucket 係数
`progress.bin` を用意する——`progress-kpabs-train` で学習し、`progress-bucket-survey`
で bucket 分布を確認する。手順は
[局面進行度 bucket: `progress.bin` の用意](progress-bin.ja.md) を参照。

### 学習

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> \
  --threads <N> \
  layerstack --progress-coeff <path/to/progress.bin>
```

`layerstack` は既定で `--feature-set halfka-hm-merged` / 9 bucket。FT 出力次元は `--ft-out`(128 の倍数、既定 1536)で変えられる。

## 主な option

`nnue-train` の CLI 既定値は動作確認 (smoke) 向けに小さい。本番学習で主に
変更するのは:

| option | CLI 既定 | 説明 |
|---|---:|---|
| `--superbatches` | 10 | 学習する superbatch 数。既定 10 は smoke 用、本番はもっと大きくする (下記「学習量の目安」) |
| `--batch-size` | 16384 | 勾配更新 1 回あたりの局面数。GPU throughput と学習特性 (勾配のばらつき・更新回数) の両方に効く学習ハイパーパラメータ |
| `--feature-set` | halfka-hm-merged | 入力 feature set。`halfkp` / `halfka-split` / `halfka-merged` / `halfka-hm-split` / `halfka-hm-merged` から選ぶ ([README](../README.ja.md) 参照) |
| `--keep-checkpoints` | 全保持 | raw `.ckpt` (weight + optimizer state) を直近 N 個に保つ。既定の全保持が学習失敗の追跡には無難。ただし `--save-rate 20` で 400 sb 学習すると `.ckpt` 20 本 × 約 1.8 GB (既定 LayerStack アーキ) ≈ 36 GB になるため、ストレージが逼迫する場合は制限する。量子化 `.bin` は常に全保持 |
| `--win-rate-model` | OFF | WRM (win-rate-model) loss。`net_output ≈ cp/600` で収束し量子化 (`QA=127 / QB=64 / FV_SCALE=28`) と整合する。量子化推論向けの net を学習するなら追加する (未指定なら plain sigmoid-MSE)。loss の調整パラメータは [WRM loss のチューニング](wrm-loss-tuning.ja.md) を参照 |
| `--score-drop-abs` | なし | `|score| >=` この値の局面を loss から除外する (詰み近傍の極端な評価値を弾く) |
| `--score-clamp-abs` | なし | drop を生き残った局面の score を `[-N, N]` に飽和させる (教師の clip 上限違いを単一上限へ正規化する) |
| `--threads` | 16 | **必ず設定する。** GPU 処理が高速なため CPU データローダーが律速になりやすく、大き目の値を推奨。CPU 物理コア数を目安にし、小さい値 (例: 1) だと pos/s が大幅に低下する。`NNUE_TRAIN_STEP_PROFILE=1` で h2d / fwd / bwd / optimizer の内訳を確認しながら調整する |
| `--test-tail-positions` | なし | `--data` の末尾 N 局面を同一ファイル内の held-out 検証集合として確保する (下記「held-out validation」参照)。held-out validation を有効化したいときの推奨経路 |
| `--test-positions` | 10000 | held-out source から毎 superbatch 評価する局面数。`--test-tail-positions` または `--test-data` 指定時のみ有効 |
| `--num-buckets` (`layerstack`) | 9 | LayerStack の output bucket 数、`[2, 9]` の整数。各局面は `min(N-1, floor(progress * N))` で routing される。低い N は bucket 1 個あたりのサンプル数が増える代わりに局面別特殊化が緩む。既定 9 は既存配布 net と同じ binning |

`--batches-per-superbatch` (6104) / `--lr` (8.75e-4) / `--save-rate` (20)
などは既定のままでよく、変えたいときだけ渡す。

**学習量の目安**: 1 superbatch = `batches-per-superbatch × batch-size` 局面。
既定の `batch-size` では 1 superbatch ≈ 1 億局面で、これは上流のチェス向け NNUE
トレーナー [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) の
1 epoch (既定 `--epoch-size` = 1 億局面) とほぼ同じ規模。nnue-pytorch の既定は
800 epoch。`--superbatches` は教師データ量と過学習の兼ね合いを見て決める。

所要時間は GPU と構成 (FP16 モード有無) で大きく変わる。

## held-out validation

過学習や数値発散 (NaN) を SPRT 自己対局を待たずに早期検知するには、held-out
validation を有効化する。勾配更新に一切使わない局面を毎 superbatch 末に集計し、
`test_loss` / `test_acc` として出力する。`--test-tail-positions`(または
`--test-data`)+ `--test-positions` で有効化する。flag・held-out source の選び方・
指標の読み方は [held-out validation](held-out-validation.ja.md) を参照。

## 学習中断・再開

raw `.ckpt` は **weight + Ranger optimizer state (m / v / slow / step) + 現在の
superbatch 番号** を全部保存する。電源断や GPU エラーで止まっても完全に再開
できる。学習時と同じ option + アーキ サブコマンドに `--resume` を足す:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --feature-set halfkp --superbatches <N> --keep-checkpoints 4 \
  --resume checkpoints/<run-name>/<run-name>-<sb>.ckpt \
  simple
```

`--resume` あり (`--start-superbatch` 省略) なら checkpoint の sb +1 から再開、
`--start-superbatch N` 明示で過去 sb をやり直すことも可。

horizon を持つ LR schedule では、checkpoint に解決済 horizon が保存され、resume
時に `--superbatches` に依らず curve が再現される。優先順位の詳細は
[Horizon と resume](training-schedule.ja.md#horizon-と-resume) を参照。

> **`--resume` と `--init-from` の違い**: `--init-from` は量子化 `.bin` から
> weight だけ注入し optimizer state を **reset** する (fine-tuning / continued
> training)、`--resume` は raw `.ckpt` から weight + optimizer 両方復元する
> (真の resume)。両者は排他指定。

## 出力 artifact の見方

学習後 `checkpoints/<run-name>/` に出るもの:

| ファイル | 形式 | 用途 |
|---|---|---|
| `<run-name>-<sb>.bin` | 量子化 NNUE binary | **推論エンジンに投入する artifact** (binary layout は `crates/nnue-format` 参照) |
| `<run-name>-<sb>.ckpt` | raw f32 + optimizer state | `--resume` 用、推論には使わない (`--keep-checkpoints` で淘汰) |

`<run-name>-<最終 sb>.bin` が最終 net。これは [rshogi](https://github.com/SH11235/rshogi)
エンジンでロードする — YaneuraOu など他の将棋エンジンでは読めない
([学習した net の使い方](../README.ja.md#学習した-net-の使い方) 参照)。棋力検証は
エンジンに組み込んで測定する。

## 動作確認 (smoke)

データ準備前に GPU 経路だけ確認したい場合は、アーキ サブコマンドを付けて
`--data` を省略すると `GpuTrainer` の forward / backward path を 1 step だけ
実行する smoke test が走る:

```bash
target/release/nnue-train simple
# → 末尾に "[smoke/simple] PASSED" の行が出れば GPU 経路は健全
```

または小規模 run (1 sb × 3 batches) で全 pipeline を数秒で回す:

```bash
target/release/nnue-train --data <PSV> \
  --output /tmp/smoke --net-id smoke \
  --superbatches 1 --batches-per-superbatch 3 \
  --save-rate 1 --threads 4 \
  simple
```

## トラブルシューティング

| 症状 | 原因 / 対応 |
|---|---|
| `kernel artifact nnue_train.{cubin,ptx,ll} not found` | 初回ビルド時 `cd bins/nnue_train && cargo-oxide build` で `.ll` を生成する必要がある。詳細は [docs/setup.ja.md](setup.ja.md) |
| `libcublas.so` 系の link / load エラー | CUDA Toolkit が `/usr/local/cuda` / `CUDA_HOME` / `CUDA_PATH` のいずれにも無い。`CUDA_TOOLKIT_PATH=/path/to/cuda-12.x` で明示する (build.rs / runtime 両方が同じ chain で解決) |
| `CUDA_ERROR_INVALID_PTX` (driver error 218) | sub-Ampere GPU (sm_75) で `CUDA_OXIDE_TARGET` 未設定。`CUDA_OXIDE_TARGET=sm_75` を export してから再ビルド + 実行 |
| pos/s が極端に低い (< 500K on RTX 3080 Ti) | `--threads` を増やし (CPU 物理コア数を目安に、上記「主な option」参照)、dataloader の prefetch が間に合っているか確認。`NNUE_TRAIN_STEP_PROFILE=1` で各 phase (h2d / fwd / bwd / optimizer) の所要 ms を stderr に出して内訳を確認できる |
| `--batch-size must be a multiple of 16` で reject | tiled dense matmul kernel が `b % 16 == 0` を要求するため、CLI が起動時に明示エラーで reject する。16 の倍数を渡す (既定の 16384 は条件を満たす) |

## 関連

- [docs/setup.ja.md](setup.ja.md) — toolchain (LLVM / CUDA / cuda-oxide) セットアップ
- [局面進行度 bucket: `progress.bin` の用意](progress-bin.ja.md) — LayerStack の bucket 係数の学習と確認
- [held-out validation](held-out-validation.ja.md) — `test_loss` / `test_acc` の有効化と指標の読み方
- [学習スケジュール](training-schedule.ja.md) — 学習率と WDL lambda のスケジューリング
- [WRM loss のチューニング](wrm-loss-tuning.ja.md) — WRM の変換式と調整引数 (勝率変換 + loss の一般化)
