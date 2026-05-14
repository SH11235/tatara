# Architecture Overview

rshogi-nnue は将棋 NNUE 学習を **Rust 一言語で** 完結させる個人プロジェクトです。
GPU kernel は [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (NVIDIA Labs の
Rust → PTX rustc backend) で build-time に PTX 化し、host から device まで C++ /
CUDA C++ を介さない構成にしています。

## スコープ

- 学習 input: bullet-shogi 由来の PackedSfenValue (PSV) format
- ネット構成: HalfKA_hm 1536-16-32 (LayerStack + PSQT、v102 layout)
- optimizer: RAdam / Ranger (RAdam + lookahead) / AdamW
- 損失: WRM (win-rate-model)、sigmoid + WDL blend
- platform: NVIDIA GPU のみ (sm_75 以降で検証、sm_86 が主環境)。
  ROCm / AMD GPU はサポートしません。

## リポジトリ構成

```
rshogi-nnue/
├── crates/
│   ├── shogi-format/     PackedSfenValue, ShogiBoard, BonaPiece 等
│   ├── shogi-features/   HalfKA_hm 特徴抽出、progress8kpabs bucket
│   ├── gpu-runtime/      host 側 CUDA wrapper (cuda-host 薄ラッパ)
│   ├── gpu-kernels/      kernel 実装 + CPU reference
│   │   ├── pointwise/    fused optimizer step / loss / activation grad
│   │   ├── sparse/       HalfKA_hm sparse FT forward / backward
│   │   ├── layerstack/   dense_mm / crelu / concat / slice etc.
│   │   └── progress/     KP-abs progress kernel (eval ツール用)
│   ├── nnue-train/       CPU-only training pipeline (schedule / dataloader
│   │                     / optimizer host state / superbatch loop driver)
│   └── nnue-format/      NNUE binary IO (header / halfka_psqt / v102 layerstack)
├── bins/
│   ├── nnue_train/       NNUE 本番 trainer (GPU `#[kernel]` 定義はここに inline)
│   └── progress_kpabs_train/   KP-abs progress trainer
└── docs/
    ├── 00-overview.md    (このファイル)
    ├── 01-decisions/     ADR (Architecture Decision Records)
    ├── data-layout.md    weight buffer / device memory layout
    ├── setup.md          開発環境構築 (CUDA / LLVM / rustup)
    └── kernels/          kernel 設計メモ
```

## 設計の核 — runtime fusion を build-time hand-fusion で代替する

bullet (upstream) は runtime に PointwiseIR を組み立てて NVRTC で fused kernel を
JIT する。これが element-wise 列 (optimizer step / activation gradient / loss +
WDL blend) で memory traffic を 1/N に削るための要となっている。

cuda-oxide は build-time コンパイラなので runtime fusion はできない。一方で
本リポは「shogi NNUE 専用」でネット構成が固定されており、必要な fused pattern は
有限種で打ち止めできる:

| Pattern | 用途 |
|---|---|
| `fused_radam_step` / `fused_ranger_step` / `fused_adamw_step` | optimizer (m, v 更新 + bias correction + weight 更新、Ranger は lookahead 込み) |
| `fused_loss_wdl` / `fused_loss_wrm` | sigmoid + WDL blend + scale / WRM (win-rate-model) loss |
| `fused_screlu_grad` | activation gradient (forward 経路と組合せ) |
| sparse FT forward / backward | HalfKA_hm の sparse feature transformer |
| dense_mm bucket variants | layerstack の per-bucket dense matmul |

これらを build-time に書いておけば runtime-fused 版と同等の memory traffic を
実現できる。各 kernel には CPU reference 実装と数値同等性テストを併設して
ある (`scripts/local-ci.sh` で release ビルドの test 経由で常時検証)。

詳細は `docs/01-decisions/2026-05-09-fused-kernel-strategy.md`。

## 依存関係

- **bullet-shogi / bullet (MIT)**: PSV format / shogi 型 / feature 定義 / loss
  数式 / 量子化定数を vendor (`ATTRIBUTION.md` 参照)。vendor を選んだ理由と
  cuda-oxide を git dep にする理由は `docs/01-decisions/2026-05-09-vendor-vs-dep.md`。
- **cuda-oxide (Apache-2.0)**: rev pin の git dep。alpha 期の API breakage を
  局所化するために commit ピンを必須としている。`rust-toolchain.toml` の
  channel は upstream cuda-oxide に追従する (rustc internal ABI に依存する
  ため channel ずれは link-time fail)。
- **Pliron (Apache-2.0)**: cuda-oxide の transitive 依存。

## 開発フロー

- workspace 全体 (GPU 依存 crate 含む) の fmt / clippy / test は
  `bash scripts/local-ci.sh` で実行 (push / PR 前必須)。
- GitHub Actions は GPU crate を exclude した CPU-only check のみ。
- 設計判断は ADR (`docs/01-decisions/`) に、性能計測 / 仮説検証経緯は
  `docs/experiments/` に記録する方針。
