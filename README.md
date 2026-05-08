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
| 1 | `experiments/001-cuda-oxide-kpabs/` で KP-abs progress trainer (4 kernel) を cuda-oxide 化 |
| 2 | `crates/gpu-kernels/{pointwise,sparse}/` に hand-fused kernel 整備 |
| 3 | `crates/nnue-train/` で HalfKA_hm 1536-16-32 training pipeline |
| 4 | research playground (PSQT, Threat, 新アーキテクチャ) |

詳細は [docs/00-overview.md](docs/00-overview.md) と
[docs/01-decisions/](docs/01-decisions/) を参照。

## 環境

- NVIDIA GPU (sm_86 / Ampere 想定。Hopper / Blackwell も後で検証)
- CUDA Toolkit 12+
- LLVM 22 (cuda-oxide の `llc-22` 要求)
- Rust nightly (`rust-toolchain.toml` に pin)

## 関連リポジトリ

- bullet-shogi (vendor 元): https://github.com/SH11235/bullet-shogi
- bullet (上流): https://github.com/jw1912/bullet
- cuda-oxide (中核技術): https://github.com/NVlabs/cuda-oxide
- rshogi (将棋エンジン本体): NNUE 推論実装の参照先

## License

MIT (see [LICENSE](LICENSE)).
bullet-shogi / cuda-oxide からの取り込みは [ATTRIBUTION.md](ATTRIBUTION.md) を参照。
