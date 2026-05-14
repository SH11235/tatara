# rshogi-nnue

将棋 NNUE (Efficiently Updatable Neural Network) 学習を **Rust 一言語で**
完結させる個人プロジェクト。GPU kernel は
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) (NVIDIA Labs の Rust → PTX
rustc backend) で build-time に PTX 化し、host から device まで C++ / CUDA C++
を介さない。学習対象は HalfKA_hm 1536-16-32 (LayerStack + PSQT、v102 layout)。

> **NVIDIA only** — cuda-oxide が PTX 生成専用なため ROCm / AMD は対象外。
> AMD GPU で類似の NNUE 学習を行いたい場合は CUDA / HIP 両 backend を持つ
> 上流の [bullet-shogi](https://github.com/SH11235/bullet-shogi) を参照。

## クイックスタート

### 環境要件

- **NVIDIA GPU** (Ampere 以降 / sm_80+ を公式サポート、Turing / sm_75 も
  `CUDA_OXIDE_TARGET=sm_75` 環境変数で単純な kernel は動作)
- **CUDA Toolkit 12.x** (12.9 で動作確認)
- **LLVM 21+** (`llc-21` が floor、`llc-22` が atomics syncscope の完全性に
  必要なので推奨)
- **Rust nightly** (`rust-toolchain.toml` で cuda-oxide upstream の channel
  に追従、rustc internal ABI に依存するため channel を勝手に変えない)

詳細なインストール手順とサポート GPU マトリクスは
[docs/setup.md](docs/setup.md) を参照。

### Build & test

push / PR 前の必須チェック (GPU 依存 crate を含む全 crate の fmt / clippy /
release test):

```bash
bash scripts/local-ci.sh
```

GitHub Actions (`.github/workflows/checks.yaml`) は CUDA / LLVM が無いため
GPU crate を exclude した CPU-only check のみ走らせる。

## リポジトリ構成

| ディレクトリ | 役割 |
|---|---|
| `crates/shogi-format/` | PackedSfenValue (PSV) reader、ShogiBoard / Hand 型 |
| `crates/shogi-features/` | HalfKA_hm 特徴抽出、progress8kpabs bucket |
| `crates/gpu-runtime/` | host 側 CUDA wrapper (cuda-host の薄ラッパ) |
| `crates/gpu-kernels/` | kernel 実装 (`pointwise/` / `sparse/` / `layerstack/` / `progress/`) と CPU reference + 数値同等性テスト |
| `crates/nnue-train/` | CPU-only training pipeline (schedule / dataloader / optimizer host state / superbatch loop driver) |
| `crates/nnue-format/` | NNUE 重みファイル binary IO (header / halfka_psqt / v102 layerstack) |
| `bins/nnue_train/` | NNUE 本番 trainer (GPU `#[kernel]` 定義はここに inline) |
| `bins/progress_kpabs_train/` | KP-abs progress trainer (eval 用) |
| `docs/` | ADR / setup / data layout / kernel catalog |

## ドキュメント

- [Setup guide](docs/setup.md) — CUDA / LLVM / rustup のインストール、サポート
  GPU マトリクス、CUDA toolkit root 解決
- [Training quickstart](docs/training-quickstart.md) — PSV データ準備 + 主要
  CLI option + 400 sb full run + resume / checkpoint 運用
- [Performance guide](docs/performance.md) — GPU 機種別 throughput 目安 +
  `NNUE_TRAIN_STEP_PROFILE` での自己診断手順
- [Data layout](docs/data-layout.md) — PSV / progress.bin / .nnue / checkpoint
  の配置・命名規約
- [ADR (Architecture Decision Records)](docs/decisions/) — 設計判断とその
  rationale
- [Fused kernel catalog](docs/kernels/fused-pattern-catalog.md) — どの kernel
  が何を担うか
- [v102 binary save format](docs/bullet_v102_save_format_report.md) — bullet
  v102 LayerStack `quantised.bin` の binary layout 仕様

## 用語 (glossary)

| 略語 | 意味 |
|---|---|
| **NNUE** | Efficiently Updatable Neural Network — 将棋 / チェスエンジンで使われる軽量評価関数 |
| **HalfKA_hm** | Half-Mirror 版 HalfKA 特徴量 (キング × 駒位置で sparse encode) |
| **FT** | Feature Transformer — NNUE の入力 sparse → dense 層 |
| **PSV** | PackedSfenValue — bullet-shogi 由来の学習データ format (1 局面 + score + WDL) |
| **KP / KP-abs** | King-Piece relative feature と絶対値版 (progress / 入玉判定用) |
| **bucket** | per-output-bucket 重み分離 (game phase / progress で分岐) |
| **SCReLU** | Squared Clipped ReLU — NNUE で広く使われる activation |
| **RAdam / Ranger** | Rectified Adam / Ranger optimizer (Ranger = RAdam + lookahead) |
| **WRM** | Win-rate model loss (bullet `--win-rate-model` 由来) |
| **superbatch** | bullet 用語で「複数 batch を 1 単位として lr/wdl scheduler を進める」単位 |
| **PTX** | Parallel Thread Execution — NVIDIA GPU 向け仮想 ISA。CUDA C++ / Rust → PTX (`.ptx` テキスト) → CUDA driver の JIT が SASS (実機機械語) に compile して実行。世代非依存に配布可 (sm_80 向け PTX を sm_86/89/90 が forward-compat で実行できる)。`docs/setup.md` のサポート GPU マトリクス参照 |
| **SASS** | NVIDIA GPU の世代別実機機械語。PTX から CUDA driver JIT が生成する終端形式。本リポでは直接扱わない |
| **sm_XX** | NVIDIA GPU の compute capability (例: sm_75 = Turing、sm_86 = Ampere RTX 30xx)。PTX 生成時の target アーキ指定 (`CUDA_OXIDE_TARGET=sm_86` 等) に使う |

## 関連リポジトリ

- [bullet](https://github.com/jw1912/bullet) — 上流 (NNUE training framework)
- [bullet-shogi](https://github.com/SH11235/bullet-shogi) — bullet の将棋向け fork
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — Rust → PTX rustc backend

## License

MIT (see [LICENSE](LICENSE))。
bullet-shogi / bullet / cuda-oxide からの取り込み範囲とライセンス互換性は
[ATTRIBUTION.md](ATTRIBUTION.md) を参照。
