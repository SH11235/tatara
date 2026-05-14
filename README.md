# rshogi-nnue

Personal Rust shogi NNUE training lab using **cuda-oxide**
(NVIDIA Labs の rustc → PTX backend).

`bullet-shogi` (jw1912/bullet 将棋フォーク) とは別系統で、自前で育てる将棋
NNUE 学習プロジェクト。GPU カーネルを Rust で書き、host から device まで
言語を統一する。

## ビジョン

- **NVIDIA only** で割り切る (ROCm 永久対象外: ADR-0006)
- bullet-shogi 上流追従の責務から解放
- alpha 段階の cuda-oxide のリスクは個人の learning value で相殺

## ロードマップ

| Stage | スコープ |
|---|---|
| 1 | `bins/progress_kpabs_train/` で KP-abs progress trainer (4 kernel) を cuda-oxide 化 |
| 2 | `crates/gpu-kernels/{pointwise,sparse}/` に hand-fused kernel reference (CPU) を整備 (GPU 側 `#[kernel]` は bin 側 inline) |
| 3 | `crates/nnue-train/` + `bins/nnue_train/` で HalfKA_hm 1536-16-32 training pipeline |
| 4 | research playground (PSQT, Threat, 新アーキテクチャ) |

詳細は [docs/00-overview.md](docs/00-overview.md) と
[docs/01-decisions/](docs/01-decisions/) を参照。

## 環境

- NVIDIA GPU
  - 公式要件: **Ampere+ / sm_80+** — Stage 0 で **RTX 3080 Ti (sm_86)** /
    Ubuntu 22.04 jammy / LLVM 22.1.6 を primary に動作確認 (2026-05-11)
  - Turing (sm_75) は `CUDA_OXIDE_TARGET=sm_75` を渡せば公式パスで動作 —
    Stage 1 KP-abs 程度の単純 kernel まで (RTX 2070 SUPER + WSL2 noble +
    LLVM 21.1.8 で確認、2026-05-09)
  - 詳細は [docs/setup.md](docs/setup.md)
- CUDA Toolkit 12+ (12.9 で確認)
- LLVM 21+ (`llc-21` floor、`llc-22` 推奨 — atomics の syncscope 完全性に必要)
  - **LLVM 22.1.6** (Native Linux Ubuntu 22.04) と **LLVM 21.1.8**
    (WSL2 Ubuntu 24.04) の両方で smoke 通過確認済
- Rust nightly (`rust-toolchain.toml` に pin)

セットアップ手順は **[docs/setup.md](docs/setup.md)** を参照。

## 関連リポジトリ

- bullet-shogi (vendor 元): https://github.com/SH11235/bullet-shogi
- bullet (上流): https://github.com/jw1912/bullet
- cuda-oxide (中核技術): https://github.com/NVlabs/cuda-oxide
- rshogi (将棋エンジン本体): NNUE 推論実装の参照先

## License

MIT (see [LICENSE](LICENSE)).
bullet-shogi / cuda-oxide からの取り込みは [ATTRIBUTION.md](ATTRIBUTION.md) を参照。
