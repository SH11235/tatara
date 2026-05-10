# experiments/001-cuda-oxide-kpabs

Stage 1 の experiment スレッド: bullet-shogi `shogi_progress_kpabs_train_cuda`
(KP-abs progress 学習) を cuda-oxide で書き直す。最終目標は
**bullet-shogi 版と numerical equivalence な `progress.bin`** を出力する
host loop が回ること。

## 動機

- ADR-0003 で cuda-oxide 採用を決めた最初の実機実証
- bullet-shogi cuda 版 (commit `f275eb9`, ~1100 行) を参照しつつ、
  forward / grad / adam_step / eval の 4 kernel を Rust 一言語で書き起こす
- 出力 `progress.bin` を bullet-shogi 版と bit-exact (or 高精度な
  数値等価) に揃えるところまで持っていく

## Scope (Stage 1)

- 本 experiment は **`experiments/001-cuda-oxide-kpabs/` の中で完結**。
  `crates/` への昇格は Stage 1-15 / Issue #15 を待つ
- target GPU: sm_75 (本マシン RTX 2070 SUPER) で開発、sm_86 (sh11235、
  解放後) で再現性を取る
- 教師データ:
  - smoke 用: `crates/shogi-format/tests/data/sample.psv` (100 records)
  - 本番用: `data/nodchip_hao_depth9/` (1016 files / 299 GB) +
    `data/nodchip_suisho5_entering_king/` (127 files / 19 GB)
  - 本マシンの `data/` は `/mnt/e/rshogi-nnue/data` への symlink
    (`docs/data-layout.md`)

## 受け入れ条件 (本 PR / Stage 1-4 / Issue #8)

- [x] `cargo build -p exp-001-cuda-oxide-kpabs` が通る
- [x] dummy main が PSV を 1 batch 読み込み、先頭数 record の主要フィールド
  (score / game_ply / game_result) を print
- [x] shogi-format crate を import している (Stage 1-1 への依存)

## 後続 Issue / 順序

| # | スコープ | Stage |
|---|---|---|
| ✅ #5 | shogi-format vendor (PSV reader / types) | 1-1 |
| ✅ #6 | shogi-features vendor (ShogiProgressKPAbs) | 1-2 |
| ✅ #7 | gpu-runtime (cuda-oxide host wrapper) | 1-3 |
| **✅ #8 (本 PR)** | **experiments/001 scaffold + dummy PSV reader** | **1-4** |
| #9 | forward kernel (cuda-oxide `#[kernel]` 初出) | 1-5 |
| #10 | grad kernel | 1-6 |
| #11 | adam_step kernel | 1-7 |
| #12 | eval kernel | 1-8 |
| #13 | host loop 統合 | 1-9 |
| #14 | numerical equivalence + 性能ベンチ | 1-10 |
| #15 | bins/ + crates/gpu-kernels/ への昇格 | 1-11 |

## 結果記録 (Stage 1 進行中に追記)

- (Stage 1-9 以降で kernel 性能 / loss 推移 / bullet-shogi cuda 版との
  数値比較を時系列で追記する)

## 得られた知見

- (Stage 1 が進む中で随時記入)

## 参照

- 移植元: `bullet-shogi/examples/shogi_progress_kpabs_train_cuda.rs`
  (commit `f275eb9`, ~1100 行)
- ADR-0003: `docs/01-decisions/0003-cuda-oxide-adoption.md`
- データ配置: `docs/data-layout.md`
