# Fused kernel pattern catalog (Stage 2 / EPIC #16)

ADR-0004 で「runtime fusion を build-time hand-fused kernel で代替する」と
決めた 7 fused kernel の **責務 / op 数 / 上流 (bullet) との対応 / 配置 file
path / 進捗** を一覧する。各 kernel issue (Stage 2-1〜2-7) で landed PR と
ベンチ結果を埋めていく。

## Pointwise fused kernels (`crates/gpu-kernels/src/pointwise/`)

| Pattern | Op 数 | 用途 | 上流 (bullet) | reference CPU 配置 | GPU `#[kernel]` 配置 | Issue / PR | Status |
|---|---|---|---|---|---|---|---|
| `fused_screlu_grad` | 2-3 | activation gradient (forward 経路と組合せ) | `crates/compiler/src/tensor/operation/autograd/dfo.rs::SCReLU` | `pointwise/screlu_grad.rs` | `experiments/002-fused-kernels/src/main.rs::screlu_grad` | #37 (PR #46) | ✅ 実装済み |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale | `crates/bullet_lib/src/value/loader.rs` (data-layer blend) + `dfo::Sigmoid` | `pointwise/loss_wdl.rs` | `experiments/002-fused-kernels/src/main.rs::loss_wdl` | #38 (PR #47) | ✅ 実装済み |
| `fused_adamw_step` | 5 | AdamW (decay + clip 込み) | `crates/trainer/src/optimiser/adam.rs::AdamWParams` | `pointwise/adamw_step.rs` | `experiments/002-fused-kernels/src/main.rs::adamw_step` | #39 (PR #48) | ✅ 実装済み |
| `fused_radam_step` | 5+host | RAdam (AdamW + bias correction + denom switch) | `crates/trainer/src/optimiser/radam.rs::RAdamParams` | `pointwise/radam_step.rs` | `experiments/002-fused-kernels/src/main.rs::radam_step` | #40 (PR #49) | ✅ 実装済み |
| `fused_ranger_step` | RAdam + lookahead | Ranger (RAdam + slow params lerp、k-step periodic) | `crates/trainer/src/optimiser/ranger.rs` | `pointwise/ranger_step.rs` | `experiments/002-fused-kernels/src/main.rs::ranger_lookahead_lerp` (+ Stage 2-4 `radam_step` 再利用) | #41 (PR #50) | ✅ 実装済み |

## Sparse FT kernels (`crates/gpu-kernels/src/sparse/`)

| Pattern | Op 数 | 用途 | 上流 (bullet) | reference CPU 配置 | GPU `#[kernel]` 配置 | Issue / PR | Status |
|---|---|---|---|---|---|---|---|
| `sparse_ft_forward` | matmul | HalfKA_hm sparse feature transform forward | `crates/compiler/src/tensor/operation/linear/sparse.rs::SparseMatmul` | `sparse/sparse_ft_forward.rs` | `experiments/002-fused-kernels/src/main.rs::sparse_ft_forward` | #42 (PR #51) | ✅ 実装済み |
| `sparse_ft_backward` | atomic scatter | 同 backward | `linear/sparse.rs::SparseMatmulBwd(Multi)` | `sparse/sparse_ft_backward.rs` | `experiments/002-fused-kernels/src/main.rs::sparse_ft_backward` | #43 (PR #52) | ✅ 実装済み |

## ベンチ (Stage 2-8 / #44)

### 計測軸の絞り込み (Stage 2-8 wrap-up で確定)

Stage 2-0 scaffold で 2 軸 (naive / bullet) で計測する設計だったが、7 kernel
全部が landed した時点で内訳を見直し、本 catalog では **絶対 samples/sec
(local sm_75)** を mandatory 計測値とし、naive / bullet 比は Stage 3 trainer
integration の actual training throughput で判断する deferred 扱いに整理する:

- **naive baseline 比**: 当初 mandatory として設計したが、実装し終わった
  7 kernel の内訳:
  - `screlu_grad`: 2-3 op、naive 分解で kernel 2 個になるが memory traffic
    は元から少なく fused/naive 比は degenerate (~1.0x)
  - `ranger_lookahead_lerp`: 単純 lerp、naive 分解不可能
  - `sparse_ft_forward` / `sparse_ft_backward`: 単一 matmul / scatter で
    naive 分解の意味なし
  - `loss_wdl` / `adamw_step` / `radam_step`: 5-10 op、naive 分解で kernel
    4-5 個に分かれる、fusion 効果が顕在化 → ただし kernel 単独 micro-bench
    より Stage 3 trainer integration の actual training throughput で測る
    方が妥当 (training context で memory bandwidth が真に律速)
  → **Stage 3 trainer integration 後追いで計測** に方針変更
- **bullet runtime-fused 比 (EPIC #16 completion target)**: GPU / OS / driver
  / NVRTC バージョン差で apples-to-apples にならないため、sh11235 (sm_86
  RTX 3080 Ti) で手動測定する deferred 扱いを継続 (Stage 2-0 scaffold の
  方針と同じ)

### 絶対 samples/sec ベンチ結果 (sm_75 ローカル、RTX 2070 SUPER)

実行コマンド (詳細は `experiments/002-fused-kernels/src/main.rs::bench_tests`
docstring 参照):

```bash
cd experiments/002-fused-kernels
CUDA_OXIDE_TARGET=sm_75 /mnt/e/cuda-oxide-target/release/cargo-oxide build
cargo test -p exp-002-fused-kernels --bin exp-002-fused-kernels --release \
    -- bench_tests --test-threads=1 --nocapture
```

**初出計測 (2026-05-11、Stage 2-8 / PR #44 commit)** — n_elements/step ×
50 steps、kernel-only timing (memcpy 含めない):

| Pattern | n_elements/step | 計測時間 | absolute samples/sec |
|---|---|---|---|
| `fused_screlu_grad`        | 1024 | 1.59 ms | 32.1 M elements/sec |
| `fused_loss_wdl`           | 1024 | 1.71 ms | 29.9 M elements/sec |
| `fused_adamw_step`         | 1024 | 1.49 ms | 34.3 M elements/sec |
| `fused_radam_step`         | 1024 | 1.47 ms | 34.9 M elements/sec |
| `fused_ranger_step` (lerp) | 1024 | 1.48 ms | 34.6 M elements/sec |
| `sparse_ft_forward`        | 1024 | 1.73 ms | 29.6 M elements/sec |
| `sparse_ft_backward`       | 1024 | 1.61 ms | 31.7 M elements/sec |

ノート:
- **総時間 vs per-step 時間**: 50 step 合計 ~1.5〜1.7 ms = **per step ~30 µs**。
  内訳は kernel 自体の実行が数 µs、残りは `cuStreamSynchronize` の host-side
  wait overhead。1024 element の単純 pointwise/sparse 程度では kernel 実行
  時間 (< 50 µs) より sync overhead が支配的な **launch overhead dominant**
  プロファイル
- training の実 batch size (≥ 8K) で launch overhead は薄まり bandwidth-bound
  に寄せるため、本数値は **regression detection baseline** としての性格が強い
- Stage 3 trainer integration で 8K〜64K element の train batch を回す段階で
  改めて per-kernel breakdown を取り、bullet 上流 sm_86 と比較する想定

### Deferred 計測軸 (Stage 3 / sm_86 後追い)

| Pattern | naive baseline 比 (Stage 3 follow-up #53) | bullet runtime-fused 比 (sm_86 follow-up #54) |
|---|---|---|
| `fused_screlu_grad`   | degenerate (2-3 op、naive 分解で memory traffic 改善ほぼなし) | #54 で測定 |
| `fused_loss_wdl`      | #53 で actual training 内で測定 | #54 で測定 |
| `fused_adamw_step`    | #53 で actual training 内で測定 | #54 で測定 |
| `fused_radam_step`    | #53 で actual training 内で測定 | #54 で測定 |
| `fused_ranger_step`   | lerp は単純 pointwise、degenerate | #54 で測定 |
| `sparse_ft_forward`   | matmul のため naive 分解不能 | #54 で測定 |
| `sparse_ft_backward`  | atomic scatter のため naive 分解不能 | #54 で測定 |

## EPIC #16 完了条件 status

Stage 2-8 wrap-up 時点 (PR #N、本 PR):

| 完了条件 | Status | 根拠 |
|---|---|---|
| 7 kernel が build-time PTX 化される | ✅ 達成 | Stage 2-1〜2-7 (#46-#52) で 7 kernel が `experiments/002-fused-kernels/src/main.rs` に inline 配置済、`exp_002_fused_kernels.ll` に 7 関数体 (`@screlu_grad / @loss_wdl / @adamw_step / @radam_step / @ranger_lookahead_lerp / @sparse_ft_forward / @sparse_ft_backward`) inline 確認 |
| benchmark で bullet runtime-fused 比 ≥ 90% | ⏳ deferred (#54) | sm_86 (sh11235 RTX 3080 Ti) GPU 占有解放後に手動計測。本 PR では sm_75 absolute samples/sec を baseline として記録、bullet 比は #54 で後追い |
| Stage 3 (nnue-train) から呼び出せる API 形 | ✅ 部分達成 | reference CPU は `crates/gpu-kernels/{pointwise,sparse}/` に `pub fn` で公開、Stage 3 `bins/nnue_train` は cuda-oxide constraint により本 experiments crate の `#[kernel]` を直接 import せず **同 algorithm spec + reference CPU を参照して bins/nnue_train/main.rs に inline 配置する pattern** (Stage 1-5 の `bins/progress_kpabs_train` と同型)。Stage 3 着手時に host launch wrapper を `gpu-runtime` crate に昇格させる loader refactor (本 catalog の 運用方針 セクション 別 issue 参照) と合わせて完成 |

→ **EPIC #16 close は #54 (bullet 比) 達成後**。本 PR は wrap-up infrastructure
(bench harness + catalog finalization) を整備するに留め、EPIC は open のまま
維持する。

## 運用方針

- 1 kernel = 1 file = 1 PR で landed させる (Stage 1-5〜1-8 と同流儀)
- 各 PR で Status を「実装済み (PR #N)」に更新 + Issue リンクを埋める
- 上記 Stage 2-1〜2-7 PR (#46〜#52) で 7 kernel が EPIC #16 完了条件
  「7 kernel が build-time PTX 化される」を達成済み (Stage 2-7 / PR #52 で
  最後の kernel landed、`.ll` 内に 7 関数体 inline 化を確認)
- ベンチは Stage 2-8 (#44) で wrap-up、絶対 samples/sec は本 catalog 上の表
  に記録、naive 比 / bullet 比は Stage 3 / sm_86 で deferred 計測
- 新規 fused kernel を Stage 2 以降に追加する場合は本 catalog にまず entry を
  追加してから着手する (ADR-0004 の "新しい optimizer や activation を試す時は
  パターンを追加する必要がある" を運用化)
