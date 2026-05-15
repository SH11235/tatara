# Performance ガイド

`nnue-train` の throughput (pos/s) 期待値、GPU 機種別目安、`NNUE_TRAIN_STEP_PROFILE`
での自己診断手順。

## 計測手順 (基準)

`--threads 16` + bullet v102 互換 hyper-param で 5 sb × 200 batches × bs=65536
を 2 回実行、sb 2-5 mean (sb 1 は cold cache outlier として除外) で評価する:

```bash
DATA=/path/to/PSV
PROG=/path/to/progress.bin
target/release/nnue-train --data "$DATA" --progress-coeff "$PROG" \
  --output /tmp/bench --net-id bench \
  --superbatches 5 --batches-per-superbatch 200 --batch-size 65536 \
  --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
  --save-rate 5 --threads 16 --bucket-mode progress8kpabs
```

1 回 1m30s 程度、合計 3 分で 5 sb 分の `pos/s` ログが出る。

## GPU 機種別 throughput 目安

| GPU | sm | DRAM BW | 期待 pos/s | 400 sb ETA | 出典 |
|---|---|---:|---:|---:|---|
| RTX 3080 Ti | 86 | 912 GB/s | **~827K** | ~53 h | 本リポジトリ実測 |
| RTX 4090 | 89 | 1008 GB/s | ~1.0-1.1M (推定) | ~40 h | DRAM BW 比 1.10× + clock 比、未実測 |
| A100 40GB | 80 | 1555 GB/s | ~1.3M (推定) | ~32 h | DRAM 比だが int8 倍精度等は無関係、未実測 |
| H100 SXM | 90 | 3 TB/s | ~2M? (推定) | ~20 h? | Hopper TC 未活用なので DRAM 律速ライン、未実測 |
| RTX 2070 SUPER | 75 | 448 GB/s | 動く範囲で測定要 | — | `CUDA_OXIDE_TARGET=sm_75` 必須、cuBLAS は OK |

> **注**: 上記推定は `fwd_ft` + `bwd_L1f` の memory bandwidth 律速モデル
> (DRAM BW 比例) + L2 reuse / launch overhead からの外挿。Ampere+ では cuBLAS
> Sgemm が TF32 TC (`cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)`) で動く。
> FP16 / BF16 cast 経路は本リポジトリ未実装、TF32 のみ。

bullet-shogi (上流、CUDA C++ 実装) と本リポジトリの違い:

- 本リポジトリ (RTX 3080 Ti、5 sb × 200 batches × bs=65536): **~827K pos/s**
- bullet-shogi v102 同条件 (CUDA C++ + NVRTC runtime fusion): **~691K pos/s**
- 本リポジトリは bullet 比 **+20%** (sparse FT 系の bounds check 除去 + cuBLAS
  L1f bwd 化 + async loss readback + fwd_L1f TF32 TC の累積)

## Step phase profile 診断

`NNUE_TRAIN_STEP_PROFILE=1` で各 phase (h2d / fwd_ft / fwd_L1 / bwd_* /
optimizer) の境界で `stream.synchronize()` + 経過 ms を stderr に出す。
profile-on は 25-33% の overhead を伴うので throughput 計測時は外す:

```bash
NNUE_TRAIN_STEP_PROFILE=1 target/release/nnue-train \
  --data "$DATA" --progress-coeff "$PROG" \
  --output /tmp/prof --net-id prof \
  --superbatches 1 --batches-per-superbatch 5 --batch-size 65536 \
  --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
  --save-rate 1 --threads 16 --bucket-mode progress8kpabs \
  2>&1 | grep step-profile
```

batch 0 は cuBLAS JIT init 等で warmup する (`bwd_L1f` だけ ~70 ms)、
batch 1 以降の steady-state を見る。

### 本リポジトリの steady-state 内訳 (RTX 3080 Ti、bs=65536、profile-on)

| phase | 時間 (ms) | 内容 |
|---|---:|---|
| `h2d+reset` | ~3.0 | 入力 5 buffer の H2D + loss_acc / grad reset |
| `fwd_ft` (×2 perspectives) | ~22.7 | `sparse_ft_forward` (HalfKA_hm sparse → 1536-dim per perspective、4-row threading) |
| `fwd_ftpost` | ~1.5 | `ft_post_perspective_fwd` (bias add + CReLU + pairwise + scale) |
| `fwd_L1` | ~7.5 | `dense_mm_fwd_bucket_tiled_l1` |
| `fwd_L1f` | ~0.55 | `cublasSgemm_v2` (TF32 TC) + `bias_add_per_row` |
| `fwd_L1tail` + `fwd_L2` + `forward` | ~0.5 | L3 + loss kernel |
| `bwd_L3` + `bwd_L2` + `bwd_L1eff` | ~1.5 | |
| `bwd_L1f` | **~4.3** | `cublasSgemm_v2` (l1f weight grad) |
| `bwd_L1_inB` | ~4.4 | `dense_mm_bwd_input_tiled` |
| `bwd_L1_wB` | ~3.1 | `dense_mm_bwd_weight_bucket_tiled_l1` |
| `bwd_L1` | ~1.5 | L1 grad その他 |
| `bwd_ftpost` | ~3.9 | `ft_post_perspective_grad` |
| `phA_count` + `phB_psum` + `phC_scat` | ~0.5 | sparse_ft_backward の前半 3 phase |
| `phD_stm` | ~11.3 | `gather_and_sum_per_feature_overwrite` (stm 側) |
| `phD_nstm` | ~10.7 | 同上 (nstm 側) |
| `optimizer` | ~4.5 | `radam_step` × 10 + `ranger_lookahead_lerp` × 10 |
| **合計 (profile-on)** | **~81 ms** | (profile-off の steady-state では ~79 ms ≒ 827K pos/s) |

### 想定外の遅さを見つけたら

1. **`fwd_ft` が 30 ms 以上**: `sparse_ft_forward` の 4-row threading
   になっていない可能性。`bins/nnue_train/nnue_train.ptx` を `awk '/.entry
   sparse_ft_forward\(/,/^}/'` で見て inner loop に `ld.b32 ... +0/+4/+8/+12` の
   4 連続 load が出ているか確認。
2. **`bwd_L1f` が 8 ms 以上**: cuBLAS link が外れている。`ldd target/release/nnue-train
   | grep cublas` で `libcublas.so.12` 由来の path が出るか確認。出なければ
   `CUDA_HOME` / `CUDA_PATH` を見直して再ビルド (`bash scripts/local-ci.sh`)。
3. **`phD_stm` + `phD_nstm` が 30 ms 以上**: sparse_ft_backward の inverse-index
   pipeline (Phase 1-4) のどこかで遅延、`feat_counts` / `feat_offsets` /
   `feat_positions` のサイズが極端に大きい (`batch * MAX_ACTIVE = 65536 × 40 =
   2.6M` を超える) なら workspace を確認。
4. **`h2d+reset` が 6 ms 以上**: dataloader prefetch が間に合っていない。
   `--threads` を CPU 物理コアの半分程度に上げる、PSV ファイルを SSD に置く、
   または別ドライブから symlink を張り直す。
5. **pos/s が profile-off で 700K を切る**: 上記 phase いずれかの inflation +
   GPU other load 競合の可能性。`nvidia-smi` で GPU 使用率と温度確認、別
   process が GPU を占有していないか調べる。

## 関連

- [docs/training-quickstart.md](training-quickstart.md) — 学習を回す手順
- [docs/setup.md](setup.md) — toolchain + CUDA toolkit root 解決
