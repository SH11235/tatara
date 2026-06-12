# Fused kernel pattern catalog

本リポジトリの GPU kernel の **責務 / 配置ファイル** を一覧する。pointwise の
fused kernel は [fused kernel strategy ADR](../decisions/2026-05-09-fused-kernel-strategy.md)
が定めた「runtime fusion を build-time hand-fused kernel で代替する」方針の産物。
LayerStack dense kernel と progress trainer kernel もここに併記する。

GPU 側 `#[kernel]` 定義は cuda-oxide の bin-crate reachability 制約により bin
crate 内に置く: `nnue-train` は `bins/nnue_train/src/kernels/` の 3 file
(`common` / `layerstack` / `simple`)、`progress-kpabs-train` は
`bins/progress_kpabs_train/src/main.rs`。主要 kernel には `crates/gpu-kernels/`
配下に reference CPU 実装があり、GPU↔CPU の数値同等性テストは
`bins/nnue_train/src/tests/gpu_cpu_equivalence_tests.rs` が持つ
(progress trainer kernel の同等性テストは `bins/progress_kpabs_train/src/main.rs`
内のテスト)。ただし **全 kernel が同等性テストで照合されているわけではない**
(Simple 専用 kernel の多くと、trainer から launch されない compile-reach 維持
kernel は smoke と学習 run での検証のみ)。照合カバレッジの真実源は同テストファイル。

## Pointwise fused kernels

reference CPU: `crates/gpu-kernels/src/pointwise/`。

| Pattern | 用途 |
|---|---|
| `screlu_grad` | SCReLU activation gradient (forward 経路と組合せ) |
| `loss_wdl` | sigmoid + WDL blend + scale (`--win-rate-model` 未指定時の loss) |
| `loss_wrm` / `wrm_weight_sum` | WRM (win-rate-model) loss、prediction / target 双方に WRM 適用。重み付き loss の分母は `wrm_weight_sum` が集計 |
| `adamw_step` | AdamW (decay + clip 込み) |
| `radam_step` | RAdam (AdamW + bias correction + denom switch)。FP16 mirror / FP16 opt-state の variant (`_fp16_mirror` / `_f16state` / `_f16state_mirror`) を持つ |
| `ranger_lookahead_lerp` | Ranger の lookahead (slow params lerp、k-step periodic)。FP16 mirror variant (`_fp16_mirror`) を持ち、`radam_step` と 2 kernel の組で Ranger を構成する |
| `norm_loss_reduce` / `norm_loss_finalize` / `norm_loss_apply` | per-weight-group L2-norm 正則化 (`--norm-loss`) の norm 集計 → 係数化 → weight への適用 |

## Sparse FT kernels

reference CPU: `crates/gpu-kernels/src/sparse/`。

| Pattern | 用途 |
|---|---|
| `sparse_ft_forward` | sparse feature transform forward。FP16 variant (`_fp16` / `_fp16_out`) を持つ |
| `sparse_ft_backward` | 同 backward (per-position の partial gradient を atomic で集約) |
| `build_feature_counts` / `exclusive_prefix_sum_small` / `scatter_positions` / `gather_and_sum_per_feature_*` | inverse-index 版 FT backward: 出現 feature ごとに position を gather し、atomic 競合なしで weight gradient を集約する |

## LayerStack dense kernels

LayerStack アーキの FT 後処理と、bucket 別重み行列 (bucket 数は
`--num-buckets`、既定 9) を選択する per-bucket dense 層。reference CPU:
`crates/gpu-kernels/src/layerstack/`。

| Pattern | 用途 |
|---|---|
| `ft_post_perspective_fwd` / `_grad` | FT 出力後処理を 1 kernel に集約 (bias add → CReLU → pairwise_mul → ×127/128)、両 perspective まとめて combined 出力 |
| `dense_mm_fwd` / `_bwd_input` / `_bwd_weight` / `bias_grad` | bucket 非依存 dense 層 (L1f shared weight) の forward / backward |
| `dense_mm_fwd_bucket` / `_bwd_input_bucket` / `_bwd_weight_bucket` / `bias_grad_bucket` | per-bucket dense 層 (L1 / L2 / L3) の forward / backward。position ごとに bucket の重み行列を選ぶ |
| `count_buckets` / `exclusive_scan_aligned` / `scatter_bucket_perm` / `permute_rows_f32` / `inverse_permute_rows_f32` | batch を bucket 順 (16-align padding) に並べ替える batch sort。sorted 系 dense kernel の前処理と出力の逆 permute |
| `psqt_diff_sparse_fwd_inplace` / `psqt_diff_sparse_bwd` | per-bucket PSQT shortcut (`--psqt`) の forward 加算 / weight gradient |
| `crelu_fwd` / `crelu_grad` | CReLU 活性化 forward / backward |
| `abs_pow2_scale_fwd` / `_grad` | l1_main を二乗 + scale して l1_sqr を作る |
| `concat_l1sqr_main_fwd` / `_grad` | l1_sqr と l1_main を concat して L2 入力 (`2×(l1_out−1)` 次元、既定 30) を組む |
| `bias_add_per_row` | 行列 (batch × n) の各行へ bias を加算 |
| `elementwise_add` | `net_output = l3_out + l1_skip` 等の要素加算 |
| `slice_extract_2d` / `slice_scatter_2d` | 2D buffer の行 slice 抽出 / 書き戻し |

device 側の実体は tile / FP16 / sorted などの variant を持つ
(`dense_mm_fwd_bucket_tiled_l1_sorted` など)。アーキ上の繋がりは
`bins/nnue_train/src/kernels/mod.rs` の module doc を参照。

## Simple アーキ専用 kernels

`bins/nnue_train/src/kernels/simple.rs` の `simple_*` kernel 群。FT 後処理 +
活性化 (crelu / screlu 別 variant)・FT backward・bias gradient 集約を Simple
アーキの 2 視点 layout に合わせて fuse したもの。一覧と各 kernel の役割は
同 file の `#[kernel]` 定義と doc コメントを参照。

## Progress trainer kernels

別バイナリ `progress-kpabs-train` (LayerStack の bucket 係数 `progress.bin` を
学習する KP-abs progress trainer) の kernel。reference CPU:
`crates/gpu-kernels/src/progress/`。

| Pattern | 用途 |
|---|---|
| `forward` | KP-abs sparse feature の sigmoid 線形 forward |
| `grad` | gradient scatter + loss + histogram |
| `adam_step` | Adam optimizer 1 step |
| `eval` | validation / test 時の loss + histogram |

## ベンチ手法

数値同等性テストを持つ kernel は、CPU reference 実装との照合が
`scripts/local-ci.sh` の release build test 経由で常時検証される (対象 kernel
は `gpu_cpu_equivalence_tests` を参照)。

absolute throughput は単一 kernel の micro-bench より、学習 step 全体での
throughput (`bins/nnue_train` の pos/s ログ) で測る。単一 kernel を小さい
element 数で micro-bench すると、kernel 実行時間より `cuStreamSynchronize` の
host-side wait (launch overhead) が支配的になり、training の実 batch size
(≥ 8K) で bandwidth-bound に寄る本番挙動を反映しないため。

## 新規 fused kernel を追加するとき

fused kernel strategy ADR が想定する「新しい optimizer / activation を試す時はパターンを追加する必要がある」運用に従う。手順:

1. 本 catalog にエントリを追加
2. `crates/gpu-kernels/` 配下に reference CPU 実装 + 数値同等性テスト
3. `bins/nnue_train/src/kernels/` (`common` / `layerstack` / `simple` から適切な file) に `#[kernel]` device 実装を追加
4. trainer の host 側 (`crates/nnue-train/src/trainer.rs` の `LossKind` 等)
   に enum branch を追加して switch できるようにする
