# experiments/001-cuda-oxide-kpabs

> **⚠️ Archive notice (Stage 1-11 / #15)**: 本 experiment で実装した kernel と
> host loop は **2026-05-11** に以下 2 crate に昇格済:
>
> - **`crates/gpu-kernels/`** — 4 reference CPU kernel (`forward_cpu` /
>   `grad_cpu` / `adam_step_cpu` / `eval_cpu`)
> - **`bins/progress_kpabs_train/`** — GPU `#[kernel]` (forward/grad/
>   adam_step/eval) + GpuTrainer + host driver (= bullet-shogi
>   `shogi_progress_kpabs_train_cuda` 相当)
>
> 新規実装・運用は上記 crate を使ってください (`cargo run --release -p
> progress-kpabs-train -- ...`)。本 directory は **Stage 1 進行中の試行錯誤
> 履歴** として残し、今後の修正対象は新 crate のみとする。

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
| ✅ #14 | numerical equivalence + 性能ベンチ | 1-10 |
| ✅ #15 | bins/ + crates/gpu-kernels/ への昇格 (本 directory はここで archive) | 1-11 |

## 結果記録 (Stage 1 完走、2026-05-11)

- Stage 1-5..1-8: 4 GPU kernel が `.ll` 段階で IR 出力確認 (atomicrmw 含む)
- Stage 1-9: `.ll → libdevice link → .ptx` pipeline を host loader が組み、
  実機 sm_75 (RTX 2070 SUPER) で 1 epoch 完走 + YaneuraOu 互換 progress.bin
  (1,003,104 bytes) を出力
- Stage 1-10: GPU kernel ↔ CPU reference 数値同等性を 4 kernel × 1+ test で
  確認、samples/sec baseline = ~220k samples/sec を記録
- Stage 1-11 (本 PR): `bins/progress_kpabs_train` + `crates/gpu-kernels` に
  昇格、experiments は archive 化

詳細は `docs/experiments/001-stage1-10-numerical-equivalence.md` 参照。

## 得られた知見

cuda-oxide rev `6de0509` で遭遇した制限と workaround:

- `Ord::clamp` (i32) は内部で `assert!(min <= max)` の panic 経路 (`Debug::fmt`)
  を持ち、cuda-oxide が lowering 未対応 → kernel は verbatim if-else (Stage 1-6)
- `f32::max` は `std::intrinsics::maximum_number_nsz_f32` を呼び未対応 →
  kernel は `if-else` (Stage 1-7)
- libNVVM が opaque pointer NVVM IR (`define void @grad(ptr ...)`) を parse
  できない → `llvm-link-21 + opt-21 + llc-21` の 3 段 pipeline で `.ll → .ptx`
  を host loader 側で組む (Stage 1-9)
- cargo-oxide の `.ll` 出力先が cwd 依存で workspace root に出る場合がある →
  loader は CARGO_MANIFEST_DIR と workspace root の両方を probe (Stage 1-9)

これらは `docs/experiments/001-stage1-10-numerical-equivalence.md` の
"cuda-oxide 不具合 / 制限の文書化" にも転記済み。

## 参照

- 移植元: `bullet-shogi/examples/shogi_progress_kpabs_train_cuda.rs`
  (commit `f275eb9`, ~1100 行)
- ADR-0003: `docs/01-decisions/0003-cuda-oxide-adoption.md`
- データ配置: `docs/data-layout.md`
