[English](README.md) | **日本語**

# tatara

![tatara — Rust で鍛える将棋 NNUE](docs/tatara-hero.jpg)

**将棋 NNUE 評価関数を高速に学習する Rust 製トレーナー。**

tatara は将棋の NNUE (Efficiently Updatable Neural Network) 評価関数を NVIDIA
GPU で学習するツール。trainer、data pipeline、CUDA Driver API runtime は
Rust で書かれ、native GPU backend は hand-fuse した CUDA C++ kernel を NVCC で
fat binary に compile して実行ファイルへ埋め込む。

[cuda-oxide](https://github.com/NVlabs/cuda-oxide) の Rust → PTX backend も Cargo の
既定 feature および native backend の数値・性能 reference として保守している。
`native-cuda-host` build は cuda-oxide を使わず、NVCC で build した kernel と
portable Rust host runtime だけを使う。`native-cuda` feature は数値 parity と
benchmark 比較のため両方の device backend を有効化するが、build されるのは
NVCC fat binary のみ — cuda-oxide 側の PTX は事前に
`bash scripts/build-kernels.sh` で生成しておく
([docs/native-cuda-benchmark.md](docs/native-cuda-benchmark.md) 参照)。

GPU kernel を hand-fuse することで **極めて高速** — 実測した cuda-oxide backend は
上流の CUDA C++ trainer [bullet-shogi](https://github.com/SH11235/bullet-shogi)
を上回る throughput を出し、native backend も同じ fused training 設計を使う。

**cuda-oxide vs bullet-shogi (RTX 3080 Ti 実測)**: LayerStack は bit-identical な既定経路でも
**+37%**、opt-in の FP16 モードを積むと最大 **~2.1×**。Simple (HalfKP
`512x2-8-64`) は既定経路で約 **+20%**、`--all-optim`(FP16/TF32) で約 **+55%**。

**実効 throughput** (`--batch-size 65536`, pos/s; fp32 → `--all-optim`(fp16/tf32))。
`--all-optim` の効果は帯域律速の強い旧世代 GPU ほど、また net が大きいほど大きい:

| arch / 構成 (feature) | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| LayerStack `--ft-out 1536` (`halfka-hm-merged`) | 2.45M → 3.13M (+28%) | 0.99M → 1.59M (+61%) |
| LayerStack `--ft-out 768` (`halfka-hm-merged`) | 4.24M → 5.10M (+20%) | 2.05M → 2.97M (+45%) |
| Simple `256x2-32-32` (`halfkp`) | 11.29M → 13.37M (+18%) | 7.30M → 10.10M (+38%) |

計測コマンド (教師データ / progress は環境依存 path、`--all-optim` の有無で pos/s を比較):

```sh
# LayerStack (halfka-hm-merged, progress 必要)
cargo run --release --bin nnue-train -- \
  --data /path/to/teacher.psv --progress-coeff /path/to/progress.bin \
  --feature-set halfka-hm-merged --batch-size 65536 --superbatches 8 [--all-optim] \
  layerstack --ft-out {1536|768} --l1 {16|8} --l2 32

# HalfKP (simple, progress 不要; simple トレーナは --win-rate-model 必須)
cargo run --release --bin nnue-train -- \
  --data /path/to/teacher.psv \
  --feature-set halfkp --batch-size 65536 --superbatches 8 [--all-optim] \
  --scale 290 --win-rate-model \
  --wrm-in-offset 0 --wrm-target-offset 0 \
  --wrm-in-scaling 290 --wrm-target-scaling 290 --wrm-nnue2score 290 \
  simple --arch 256x2-32-32
```

simple トレーナは win-rate-model loss のみ対応 (int8 出力層が plain sigmoid loss の
収束する centipawn スケール出力を表現できない)。上記の `--wrm-*` は WRM を plain
sigmoid へ恒等退化させる設定で、`--scale` と各 `--wrm-*-scaling` / `--wrm-nnue2score`
は同じ値に揃える (`--scale` が書き出す `fv_scale` も決めるため)。

*tatara(踏鞴)、砂鉄（raw material）から玉鋼を精錬する日本の伝統的なたたら炉 — raw data から net を鍛え上げる。*

> **NVIDIA only** — どちらの GPU backend も NVIDIA CUDA を対象とし、ROCm / AMD は対象外。
> AMD GPU で類似の NNUE 学習を行いたい場合は CUDA / HIP 両 backend を持つ
> 上流の [bullet-shogi](https://github.com/SH11235/bullet-shogi) を参照。

## 学習できる NNUE

学習する NNUE は、ネットワーク構造を決める **アーキテクチャ**(サブコマンド)と、
盤面をどう入力ベクトルに変換するかを決める **入力 feature set**(`--feature-set`)を
独立に選ぶ。

### アーキテクチャ

| アーキ | サブコマンド | 構造 |
|---|---|---|
| **LayerStack** | `layerstack` | 局面の進行度で出力層を bucket 別に専用化(`--num-buckets`、既定 9。Stockfish の "LayerStacks" と同じ発想)。FT 出力 `--ft-out`(既定 1536)→ `--l1`(既定 16)→ `--l2`(既定 32)|
| **Simple** | `simple` | bucket 分割のない素の NNUE(FT → 隠れ 2 層 → 単一出力)。層次元は `--arch <l1>x2-<l2>-<l3>` で指定(`l1` = FT 出力、`l2`/`l3` = 隠れ層、既定 `256x2-32-32`)、活性化 crelu / screlu / pairwise |

### 入力 feature set

`--feature-set` で 5 種から選べる(既定 `halfka-hm-merged`)。玉をどう特徴に含めるかが違う:

| `--feature-set` | 玉の扱い |
|---|---|
| `halfkp` | 玉自体は駒の特徴に含めない |
| `halfka-split` | 玉も含める。自玉用と敵玉用の特徴枠を別々に持つ |
| `halfka-merged` | 玉も含める。自玉と敵玉で特徴枠を 1 つに共有する |
| `halfka-hm-split` | `halfka-split` に加え、玉が常に盤の片側へ来るよう左右反転し、玉マスを 81 → 45 に圧縮 |
| `halfka-hm-merged`(既定) | `halfka-merged` + 同じ左右反転による玉マス圧縮 |

既定の `halfka-hm-merged` は、Stockfish の **HalfKAv2_hm**(玉マスの左右反転 +
自玉・敵玉の特徴枠を 1 つに共有)と同じ設計を将棋に適用したもの。

別バイナリ `progress-kpabs-train` は LayerStack の bucket 係数 `progress.bin` を
生成する KP-abs progress trainer。進行度を学習して出力 bucket に割り当てる手法は
[nodchip 氏の記事](https://nodchip.hatenablog.com/entry/2026/02/04/000000) の
アイデアに基づく。

## セットアップ

### 環境要件

- **OS** — Linux 一級サポート、Windows は WSL2 経由に加え
  `native-cuda-host` で native Windows を実験的にサポート、macOS は GPU ビルド非対応
- **NVIDIA GPU** (backend 別の対応表は `docs/setup.ja.md` 参照)
- **CUDA Toolkit 12.x** (12.9 で動作確認)
- `native-cuda-host` は **NVCC**、既定の cuda-oxide backend は **LLVM 21+** と
  `cargo-oxide`
- **Rust nightly** (`rust-toolchain.toml` で固定)

native CUDA C++ の build command、cuda-oxide のセットアップ、OS 別の詳細手順、
サポート GPU マトリクスは [docs/setup.ja.md](docs/setup.ja.md) を参照。

### ビルドと学習

kernel のビルドと smoke test は [docs/setup.ja.md](docs/setup.ja.md)、学習の回し方は
[docs/training-quickstart.ja.md](docs/training-quickstart.ja.md) を参照。

## ドキュメント

- [Setup guide](docs/setup.ja.md) — OS 別の案内、native CUDA / cuda-oxide の build
  セットアップ、サポート GPU マトリクス、CUDA toolkit root 解決
- [Training quickstart](docs/training-quickstart.ja.md) — アーキ別の学習例 + 主要
  CLI option + resume / checkpoint 運用
- [局面進行度 bucket: `progress.bin` の用意](docs/progress-bin.ja.md) — LayerStack
  の bucket 係数の学習と bucket 分布の確認
- [held-out validation](docs/held-out-validation.ja.md) — `test_loss` / `test_acc`
  の有効化、held-out source の選び方、指標の読み方
- [Training schedules](docs/training-schedule.ja.md) — 学習率 (`--lr-schedule`) と
  WDL lambda (`--wdl` / `--start-wdl` / `--end-wdl`) のスケジューリング
- [ADR (Architecture Decision Records)](docs/decisions/) — 設計判断とその
  rationale
- [Fused kernel catalog](docs/kernels/fused-pattern-catalog.md) — どの kernel
  が何を担うか
- [Arch string](docs/arch-string.md) — 量子化 `.bin` header に埋め込むアーキ
  記述文字列の組み立てと load 時照合 (日本語のみ)
- [YaneuraOu 用 LayerStack net 変換](docs/net-to-yaneuraou.md) —
  `net_to_yo` の使い方、対応アーキ、binary layout

## 学習した net の使い方

tatara が出力する量子化 `.bin` は [rshogi](https://github.com/SH11235/rshogi)
エンジンでロードする前提の format。`.bin` header と SCReLU / Pairwise 活性化は
本プロジェクト固有なので、YaneuraOu など他の将棋エンジンでそのまま読めるとは
限らず、アーキテクチャによっては推論部の追加実装をしないとロードできない。
対応する 1536-16-32 LayerStack は
[`net_to_yo`](docs/net-to-yaneuraou.md) で変換できる。
学習済みの参考 net は
[GitHub Releases](https://github.com/SH11235/tatara/releases) に添付している。
自分で net を学習する場合の環境構築は [docs/setup.ja.md](docs/setup.ja.md) を参照。

## 用語 (glossary)

| 略語 | 意味 |
|---|---|
| **NNUE** | Efficiently Updatable Neural Network — 将棋 / チェスエンジンで使われる軽量評価関数 |
| **FT** | Feature Transformer — NNUE の入力 sparse → dense 層 |
| **L1f** | LayerStack アーキの bucket 非依存 (全 bucket 共有) L1 dense 層。出力は per-bucket L1 の出力に加算される |
| **PSV** | PackedSfenValue — bullet-shogi 由来の学習データ format (1 局面 + score + WDL) |
| **HCPE** | HuffmanCodedPosAndEval — Apery / dlshogi の 38-byte 局面・score・指し手・対局結果 format |
| **KP / KP-abs** | King-Piece relative feature と絶対値版 (progress / 入玉判定用) |
| **bucket** | per-output-bucket 重み分離 (game phase / progress で分岐) |
| **PSQT** | Piece-Square Table — 駒種×マスごとの線形評価テーブル。LayerStack の `--psqt` で per-bucket PSQT 出力を network 出力に加算し、dense 経路は非マテリアル構造の学習に専念できる |
| **CReLU / SCReLU / Pairwise** | NNUE の活性化関数。CReLU = Clipped ReLU、SCReLU = Squared Clipped ReLU、Pairwise = 前半と後半の要素積で入力次元を半減。`simple` アーキの `--activation` で選択 |
| **RAdam / Ranger** | Rectified Adam / Ranger optimizer (Ranger = RAdam + lookahead) |
| **WRM** | Win-rate model loss (bullet `--win-rate-model` 由来) |
| **QA / QB / FV_SCALE** | 量子化スケール定数。QA = FT weight / bias の量子化 multiplier (`simple` アーキでは活性化で決まる: CReLU / Pairwise は 127、SCReLU は 255)、QB = dense weight の scale (64)。活性化出力は活性化関数に依らず常に 127-scale のため、FV_SCALE = `round(127 × QB / 学習 scale)` が net 出力を centipawn 評価値へ戻す係数になる |
| **WDL** | Win/Draw/Loss — 対局結果ターゲット (1.0 / 0.5 / 0.0)。WDL lambda で教師 score と blend する。[docs/training-schedule.ja.md](docs/training-schedule.ja.md) を参照 |
| **SPRT** | Sequential Probability Ratio Test — 2 つの net を対局させ棋力差を逐次検定する手法。学習済 net の品質確認に使う |
| **superbatch** | bullet 用語で「複数 batch を 1 単位として lr/wdl scheduler を進める」単位 |
| **PTX** | Parallel Thread Execution — NVIDIA GPU 向け仮想 ISA。CUDA C++ / Rust → PTX (`.ptx` テキスト) → CUDA driver の JIT が SASS (実機機械語) に compile して実行。世代非依存に配布可 (sm_80 向け PTX を sm_86/89/90 が forward-compat で実行できる)。`docs/setup.ja.md` のサポート GPU マトリクス参照 |
| **SASS** | NVIDIA GPU の世代別実機機械語。PTX から CUDA driver JIT が生成する終端形式。本リポでは直接扱わない |
| **sm_XX** | NVIDIA GPU の compute capability (例: sm_75 = Turing、sm_86 = Ampere RTX 30xx)。PTX 生成時の target アーキ指定 (`CUDA_OXIDE_TARGET=sm_86` 等) に使う |

## 関連リポジトリ

- [rshogi](https://github.com/SH11235/rshogi) — 本リポで学習した NNUE をロードして対局する将棋エンジン
- [bullet](https://github.com/jw1912/bullet) — 上流 (NNUE training framework)
- [bullet-shogi](https://github.com/SH11235/bullet-shogi) — bullet の将棋向け fork
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — Rust → PTX rustc backend

## License

MIT (see [LICENSE](LICENSE))。
bullet-shogi / bullet / cuda-oxide からの取り込み範囲とライセンス互換性は
[ATTRIBUTION.md](ATTRIBUTION.md) を参照。
