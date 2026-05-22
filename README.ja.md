[English](README.md) | **日本語**

# tatara

**将棋 NNUE 評価関数を高速に学習する Rust 製トレーナー。**

tatara は将棋の NNUE (Efficiently Updatable Neural Network) 評価関数を GPU で
学習するツール。host から device まで **Rust 一言語**で書かれ、GPU kernel は
[cuda-oxide](https://github.com/NVlabs/cuda-oxide)(NVIDIA Labs の Rust → PTX
rustc backend)で build-time に PTX 化する — C / C++ / CUDA C++ を一切介さない。

GPU kernel を hand-fuse することで **極めて高速** — 上流の CUDA C++ trainer
[bullet-shogi](https://github.com/SH11235/bullet-shogi) を上回る throughput を
出す。RTX 3080 Ti 実測(LayerStack)で、bit-identical な既定経路でも bullet-shogi
比 **+37%**、opt-in の FP16 モードを積むと最大 **~2.1×**。

*tatara(踏鞴)は砂鉄から玉鋼を精錬する日本の伝統的なたたら炉 — 生のデータからnet を鍛え上げる。*

> **NVIDIA only** — cuda-oxide が PTX 生成専用なため ROCm / AMD は対象外。
> AMD GPU で類似の NNUE 学習を行いたい場合は CUDA / HIP 両 backend を持つ
> 上流の [bullet-shogi](https://github.com/SH11235/bullet-shogi) を参照。

## 学習できる NNUE

学習する NNUE は、ネットワーク構造を決める **アーキテクチャ**(サブコマンド)と、
盤面をどう入力ベクトルに変換するかを決める **入力 feature set**(`--feature-set`)を
独立に選ぶ。

### アーキテクチャ

| アーキ | サブコマンド | 構造 |
|---|---|---|
| **LayerStack** | `layerstack` | 局面の進行度で出力層を bucket 別に専用化(9 bucket、Stockfish の "LayerStacks" と同じ発想)。FT 出力 `--ft-out`(既定 1536)→ 16 → 32 |
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

- **OS** — Linux 一級サポート、Windows は WSL2 経由、macOS は GPU ビルド非対応
- **NVIDIA GPU** (Ampere 以降 / sm_80+ を公式サポート、Turing / sm_75 も
  `CUDA_OXIDE_TARGET=sm_75` 環境変数で単純な kernel は動作)
- **CUDA Toolkit 12.x** (12.9 で動作確認)
- **LLVM 21+** (`llc-21` が floor、`llc-22` が atomics syncscope の完全性に
  必要なので推奨)
- **Rust nightly** (`rust-toolchain.toml` で cuda-oxide upstream の channel
  に追従、rustc internal ABI に依存するため channel を勝手に変えない)

GPU kernel をビルドする `cargo-oxide` のセットアップは
`bash scripts/setup-cuda-oxide.sh`。詳細なインストール手順・OS 別の案内・
サポート GPU マトリクスは [docs/setup.ja.md](docs/setup.ja.md) を参照。

### ビルドと学習

kernel のビルドと smoke test は [docs/setup.ja.md](docs/setup.ja.md)、学習の回し方は
[docs/training-quickstart.ja.md](docs/training-quickstart.ja.md) を参照。

## ドキュメント

- [Setup guide](docs/setup.ja.md) — OS 別の案内、CUDA / LLVM / `cargo-oxide` の
  セットアップ、サポート GPU マトリクス、CUDA toolkit root 解決
- [Training quickstart](docs/training-quickstart.ja.md) — アーキ別の学習例 + 主要
  CLI option + resume / checkpoint 運用
- [ADR (Architecture Decision Records)](docs/decisions/) — 設計判断とその
  rationale
- [Fused kernel catalog](docs/kernels/fused-pattern-catalog.md) — どの kernel
  が何を担うか
- [Arch string](docs/arch-string.md) — 量子化 `.bin` header に埋め込むアーキ
  記述文字列の組み立てと load 時照合

## 用語 (glossary)

| 略語 | 意味 |
|---|---|
| **NNUE** | Efficiently Updatable Neural Network — 将棋 / チェスエンジンで使われる軽量評価関数 |
| **FT** | Feature Transformer — NNUE の入力 sparse → dense 層 |
| **PSV** | PackedSfenValue — bullet-shogi 由来の学習データ format (1 局面 + score + WDL) |
| **KP / KP-abs** | King-Piece relative feature と絶対値版 (progress / 入玉判定用) |
| **bucket** | per-output-bucket 重み分離 (game phase / progress で分岐) |
| **CReLU / SCReLU / Pairwise** | NNUE の活性化関数。CReLU = Clipped ReLU、SCReLU = Squared Clipped ReLU、Pairwise = 前半と後半の要素積で入力次元を半減。`simple` アーキの `--activation` で選択 |
| **RAdam / Ranger** | Rectified Adam / Ranger optimizer (Ranger = RAdam + lookahead) |
| **WRM** | Win-rate model loss (bullet `--win-rate-model` 由来) |
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
