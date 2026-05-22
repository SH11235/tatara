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
  --keep-checkpoints 4 \
  simple
```

`simple` は既定で `--arch 256x2-32-32` / `--activation crelu`。`--superbatches`
の決め方と、追加で指定できる option は下記「主な option」を参照。

## 例 2: LayerStack NNUE を学習

`layerstack` アーキは局面進行度の 9 bucket を使うため、先に bucket 係数 `progress.bin` を用意する。

### progress.bin を生成

`progress-kpabs-train` で進行度係数を学習する。進行度を学習して出力 bucket に
割り当てる発想は [nodchip 氏の記事](https://nodchip.hatenablog.com/entry/2026/02/04/000000) に基づく。

> **データはシャッフルしないこと。** `progress-kpabs-train` の `--data` には
> **連続した対局**の PSV(局面が対局順に並び、対局が次々と続くもの)を渡す。
> 進行度係数は「1 局の中で局面がどこまで進んだか」を学習するもので、
> `progress-kpabs-train` はデータを 1 局単位で読み(`game_ply` で対局境界を検出)、
> 各局面にその局内での相対位置をラベル付けする。シャッフル済み PSV だと対局境界が
> 壊れてラベルが無意味になり、正しい係数が学習できない。一般に本体の NNUE 学習
> (`nnue-train`) はシャッフル済み PSV が望ましいが、進行度学習は逆で、シャッフル
> すると正しく学習できない。同じファイルを両方に使い回さないこと。

`--epochs` で総 epoch 数を指定する。epoch ごとに `<run-name>.e<N>.bin` の
checkpoint が出力され、最終 epoch は `--output` の path にも書き出される。

```bash
target/release/progress-kpabs-train \
  --data <path/to/consecutive-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

どの epoch の出力 (`<run-name>.e<N>.bin`) を使うかは試行錯誤になる
(progress.bin は bucket 割当を決める係数で、NNUE 学習の収束とは独立なため
何 epoch 必要かはデータ依存)。

どの epoch を使うか決める助けに `--val-fraction <f>`(例 `0.05`)を渡せる。
おおよそ指定割合の対局を入力順の N 局ごとに検証用へ取り分け(データは連続
対局順を保つ必要があるためシャッフルはしない)、各 epoch 末に held-out な
`val_loss` を出力する。有効にすると epoch ごとにデータ走査が 1 回増える。

`val_loss` は健全性チェックと epoch 選びの目安であって、品質の精密な指標では
ない。進行度モデルは単純(特徴ごとの重みを総和して sigmoid に通すだけ)なので
過学習しにくく、`train_loss` と `val_loss` の差は小さいのが正常で、明確に広がる
差が注意すべきサイン。また真の目的は良い bucket 分割で、素の MSE はその近似に
すぎないので、`val_loss` の厳密な最小値を追うより頭打ちになった epoch を選び、
最終的な `progress.bin` の良し悪しはそれで学習した LayerStack NNUE の棋力で
判断する。

### bucket 分布の確認

`progress-bucket-survey` は `progress.bin` が局面を進行度 bucket にどう割り当てる
かを集計する。分布がおおむね均等なら健全で、特定の bucket に偏っていると
LayerStack の出力 bucket ごとの学習データ量が大きく不均衡になる。

```bash
cargo build --release -p progress-bucket-survey
target/release/progress-bucket-survey \
  --data <path/to/consecutive-psv.bin> \
  --progress output/progress/<run-name>.e5.bin \
  --samples 200000
```

bucket ごとの件数・割合と top bucket の占有率を表示する。1 回の実行で読み込める
`progress.bin` は 1 つなので、epoch を比較するときは `<run-name>.e<N>.bin` ごとに
実行して出力を見比べる。

### 学習

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> \
  --keep-checkpoints 4 \
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
| `--keep-checkpoints` | 全保持 | raw `.ckpt` (weight + optimizer state、サイズが大きい) を直近 N 個に保つ。最初は小さめ (例 4)、学習が安定したら 20〜100 に増やしてよい。量子化 `.bin` は常に全保持 |
| `--win-rate-model` | OFF | WRM (win-rate-model) loss。`net_output ≈ cp/600` で収束し量子化 (`QA=127 / QB=64 / FV_SCALE=28`) と整合する。量子化推論向けの net を学習するなら追加する (未指定なら plain sigmoid-MSE) |
| `--score-drop-abs` | なし | `|score| >=` この値の局面を loss から除外する (詰み近傍の極端な評価値を弾く) |

`--batches-per-superbatch` (6104) / `--lr` (8.75e-4) / `--save-rate` (20) /
`--threads` (16) などは既定のままでよく、変えたいときだけ渡す。

**学習量の目安**: 1 superbatch = `batches-per-superbatch × batch-size` 局面。
既定の `batch-size` では 1 superbatch ≈ 1 億局面で、これは上流のチェス向け NNUE
トレーナー [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) の
1 epoch (既定 `--epoch-size` = 1 億局面) とほぼ同じ規模。nnue-pytorch の既定は
800 epoch。`--superbatches` は教師データ量と過学習の兼ね合いを見て決める。

所要時間は GPU と構成 (FP16 モード有無) で大きく変わる。

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
# → "[smoke] forward + backward OK" の趣旨のログが出れば GPU 経路は健全
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
| pos/s が極端に低い (< 500K on RTX 3080 Ti) | `--threads` を CPU コア数の半分程度に設定、dataloader の prefetch が間に合っているか確認。`NNUE_TRAIN_STEP_PROFILE=1` で各 phase (h2d / fwd / bwd / optimizer) の所要 ms を stderr に出して内訳を確認できる |
| `--batch-size % 16 != 0` で reject | tiled L1 kernel が `b % 16 == 0` を要求 (`debug_assert!` で fail)。16 の倍数を渡す (既定の 16384 は条件を満たす) |

## 関連

- [docs/setup.ja.md](setup.ja.md) — toolchain (LLVM / CUDA / cuda-oxide) セットアップ
