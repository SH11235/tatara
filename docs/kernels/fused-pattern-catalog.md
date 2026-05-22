# Fused kernel pattern catalog

本リポジトリの GPU kernel の **責務 / 配置ファイル** を一覧する。pointwise の
fused kernel は [fused kernel strategy ADR](../decisions/2026-05-09-fused-kernel-strategy.md)
が定めた「runtime fusion を build-time hand-fused kernel で代替する」方針の産物。
LayerStack dense kernel と progress trainer kernel もここに併記する。

GPU 側 `#[kernel]` 定義は cuda-oxide の bin-crate reachability 制約により bin
crate 内に置く: `nnue-train` は `bins/nnue_train/src/kernels/` の 3 file
(`common` / `layerstack` / `simple`)、`progress-kpabs-train` は
`bins/progress_kpabs_train/src/main.rs`。各 kernel には `crates/gpu-kernels/`
配下に reference CPU 実装と数値同等性テストが併設されている。

## Pointwise fused kernels

reference CPU: `crates/gpu-kernels/src/pointwise/`。

| Pattern | Op 数 | 用途 |
|---|---|---|
| `fused_screlu_grad` | 2-3 | SCReLU activation gradient (forward 経路と組合せ) |
| `fused_loss_wdl` | 3-5 | sigmoid + WDL blend + scale (`--win-rate-model` 未指定時の loss) |
| `fused_loss_wrm` | 5-6 | WRM (win-rate-model) loss、prediction / target 双方に WRM 適用 |
| `fused_adamw_step` | 5 | AdamW (decay + clip 込み) |
| `fused_radam_step` | 5+host | RAdam (AdamW + bias correction + denom switch) |
| `fused_ranger_step` | RAdam + lookahead | Ranger (RAdam + slow params lerp、k-step periodic) |

## Sparse FT kernels

reference CPU: `crates/gpu-kernels/src/sparse/`。

| Pattern | Op 数 | 用途 |
|---|---|---|
| `sparse_ft_forward` | matmul | HalfKA_hm sparse feature transform forward |
| `sparse_ft_backward` | atomic scatter | 同 backward (per-position の partial gradient を atomic で集約) |

## LayerStack dense kernels

LayerStack アーキの FT 後処理と、9 bucket 別重み行列を選択する per-bucket
dense 層。reference CPU: `crates/gpu-kernels/src/layerstack/`。

| Pattern | 用途 |
|---|---|
| `ft_post_perspective_fwd` / `_grad` | FT 出力後処理を 1 kernel に集約 (bias add → CReLU → pairwise_mul → ×127/128)、両 perspective まとめて combined 出力 |
| `dense_mm_fwd` / `_bwd_input` / `_bwd_weight` / `bias_grad` | bucket 非依存 dense 層 (L1f shared weight) の forward / backward |
| `dense_mm_fwd_bucket` / `_bwd_input_bucket` / `_bwd_weight_bucket` / `bias_grad_bucket` | per-bucket dense 層 (L1 / L2 / L3) の forward / backward。position ごとに 9 bucket から重み行列を選ぶ |
| `crelu_fwd` / `crelu_grad` | CReLU 活性化 forward / backward |
| `abs_pow2_scale_fwd` / `_grad` | l1_main を二乗 + scale して l1_sqr を作る |
| `concat_l1sqr_main_fwd` / `_grad` | l1_sqr と l1_main を concat して L2 入力 (30 dim) を組む |
| `elementwise_add` | `net_output = l3_out + l1_skip` 等の要素加算 |
| `slice_extract_2d` / `slice_scatter_2d` | 2D buffer の行 slice 抽出 / 書き戻し |

device 側の実体は tile / FP16 などの variant を持つ (`dense_mm_fwd_bucket_tiled_l1`
など)。アーキ上の繋がりは `bins/nnue_train/src/kernels/mod.rs` の module doc を参照。

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

数値同等性テストは各 kernel の CPU reference 実装と対で
`scripts/local-ci.sh` の release build test 経由で常時検証される。

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
