# GpuTrainer step profiling (Issue #76)

Stage 3-7/3-8 で組んだ `bins/nnue_train::GpuTrainer::step` (v102 LayerStack 1536-16-32,
9 progress8kpabs buckets, forward 15 step + backward ~16 step + Ranger 10 group) の
1 step がどこに時間を使っているかを計測し、EPIC #75 (GPU 性能最適化) のサブ issue 優先度を
data で裏付ける。

## 環境

- GPU: NVIDIA GeForce RTX 2070 SUPER (sm_75, 8 GB), Driver 591.86 / CUDA 13.1
- OS: WSL2 (Linux 6.6.114.1-microsoft-standard-WSL2)
- nsys 2025.1.3, ncu (Nsight Compute 2026.1.0) — いずれもインストール済
- bin: `cargo-oxide build` (`CUDA_OXIDE_TARGET=sm_75`) → `nnue_train.ll` → llc → `.ptx`、`cargo build -p nnue-trainer --release`
- 入力: `crates/shogi-format/tests/data/sample.psv` (100 records)。GPU step の計測には十分
  (batch_size まで wrap して埋める)。但し dataloader 速度は実 660 GB ファイルとは異なる
  (page cache 上の小ファイルを 655× 再読み + bucket=4 固定 → 実データより速いはず)

## 1. ベースライン throughput

```
./target/release/nnue-train --data sample.psv --batch-size 65536 \
    --superbatches 1 --batches-per-superbatch 6 --lr 1e-3
→ 74,000 〜 76,000 pos/s (6 batch × 65536 pos を ~5.3 s)
```

- v102 full run = 400 superbatch × 6104 batch × 65536 pos = 1.6e11 position-feed
  → 1.6e11 / (7.4〜7.6e4) ≈ **2.1〜2.2e6 s ≈ 約 24〜25 日** (この GPU で現状コードのまま)
- GPU step 単体は ~154k pos/s (下記 phase 合計 ~0.42 s/step)。**つまり wall time の
  ~半分は single-thread dataloader** (各 position で `pos.decode()` を 2 回 ——
  `ShogiHalfKA_hm::map_features` と `ShogiProgressKPAbs::for_each_active_index` が
  別々に呼ぶ)。実 660 GB データではここがさらに重くなる見込み → **#89 (dataloader を
  bucket-aware prefetch / 並列パース化) だけで ~2× 詰まる可能性**

## 2. GPU step の phase breakdown (粗い event timing)

`NNUE_TRAIN_STEP_PROFILE=1` をセットすると `GpuTrainer::step` が forward / backward /
optimizer の境界で `stream.synchronize()` + 経過時間を stderr に出す (このフラグ無しでは
追加 sync ゼロ)。WSL2 では ncu の GPU perf counter (後述) が使えないため、この粗い計測が代替。

batch_size = 65536, 6 step の中央的な 1 step:

| phase | 時間 | 割合 | 内訳 |
|---|---:|---:|---|
| `h2d+reset` | ~6 ms | ~1.5% | 入力 6 buffer の H2D + `loss_acc` 再 alloc |
| `forward` | ~128 ms | ~30% | sparse_ft_forward ×2、ft_post_perspective_fwd、L1/L1f/L2/L3 dense、CReLU/concat/slice、loss_wdl |
| **`backward`** | **~280 ms** | **~66%** | 全 `*_grad` kernel + grad buffer の 0 clear。**forward の 2.2×、ここが支配的** |
| `optimizer` | ~11 ms | ~2.5% | radam_step ×10 + ranger_lookahead_lerp ×10 |
| **合計** | **~425 ms/step** | | |

→ **backward が支配的**。forward と backward の中で何の kernel が重いか (dense matmul tiling
#79 か sparse_ft_backward #80 か、bias_grad の atomic か) の内訳は ncu が要る (下記制約)。
optimizer は ~2.5% なので **#83 の optimizer-fuse 部分は低優先**。

## 3. CUDA API breakdown (nsys)

```
nsys profile --trace=cuda,nvtx,osrt --sample=none -o nnue_prof \
    ./target/release/nnue-train --data sample.psv --batch-size 65536 \
    --superbatches 1 --batches-per-superbatch 5 --lr 1e-3
nsys stats --report cuda_api_sum nnue_prof.nsys-rep
```

| Time % | Total (ns) | Calls | Name | コメント |
|---:|---:|---:|---|---|
| 54.2 | 1.65e9 | 20 | `cuStreamSynchronize` | GPU compute 待ち (= phase breakdown の合計に対応) |
| 14.7 | 4.46e8 | 15 | `cuMemcpyDtoHAsync` | 大半は checkpoint save の weight DtoH (一過性、per-step ではない)。per-step は `loss_acc` 8 byte read のみ |
| **13.0** | **3.94e8** | 286 | **`cuMemFree`** | per-step buffer 解放 (~57 alloc/free per step) |
| **10.1** | **3.07e8** | 286 | **`cuMemAllocAsync`** | per-step buffer 確保。**#78 (P4): 中間/grad buffer を永続化すればこの ~23% (≈140 ms/step) が消える** |
| 7.7 | 2.33e8 | 40 | `cuMemcpyHtoDAsync` | 入力 6 buffer/step + init 時の weight upload |
| 0.2 | 7.3e6 | 246 | `cuMemsetD8Async` | grad buffer の 0 clear (~50/step) |
| 0.1 | 3.2e6 | 250 | `cuLaunchKernel` | ~50 launch/step、12 µs/launch。**launch overhead は無視できる → #82 (launch 数削減) は低優先** |

**WSL2 の制約**: nsys は CUDA *API* (host 側) は取れるが **GPU-side の kernel timeline は
取れない** (`cuda_gpu_kern_sum` / `cuda_gpu_trace` は `does not contain ... data`)。

## 4. ncu (kernel-level metrics) — このマシンでは取得不可

```
ncu --metrics gpu__time_duration.sum,... ./target/release/nnue-train ...
→ ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
  NVIDIA GPU Performance Counters on the target device 0.
```

WSL2 では GPU perf counter へのアクセスに **Windows ホスト側ドライバの "Developer mode"
(NVIDIA Control Panel) or レジストリ設定**が要り、WSL 内からは有効化できない。
→ **kernel ごとの occupancy / atomic スループット / coalescing / DRAM 帯域到達率は
sm_86 box (sh11235、native Linux) で取る**。本リポ box (WSL2) では §2 の粗い phase timing が
取得可能な上限。

sm_86 box で取るべきもの (issue #76 残タスク):
- `nsys profile --trace=cuda` で GPU kernel timeline → forward/backward の中の支配 kernel
- `ncu --set basic`(または `full`)で `dense_mm_bwd_input_bucket` / `dense_mm_bwd_weight_bucket`
  (#77 で atomic 撤廃済) / `sparse_ft_backward` / `bias_grad`(L3 は out_dim=1 で全 batch が
  1 cell に衝突) / `loss_wdl` (f64 atomic) / `radam_step(ft_w)` (DRAM 帯域) の metrics
- batch=65536 と batch=4096 など複数 batch で

## 5. EPIC #75 サブ issue 優先度 (この計測を踏まえて更新)

| issue | 内容 | 計測の裏付け | 優先度 |
|---|---|---|---|
| #78 | P4: 中間/grad buffer 永続化 + grad memset reset | malloc/free が CUDA API 時間の ~23% (≈140 ms/step) | **高** (低リスク・高 ROI) |
| #89 | dataloader を bucket-aware prefetch / 並列パース化 | wall time の ~半分が single-thread dataloader (`pos.decode()` ×2/position) | **高** (~2× 見込み) |
| #79 | P1: dense matmul の shared-mem tiling | forward 128 ms + backward の dense 部 (内訳は sm_86 で要計測) | 高 (effort 大) |
| #80 | P3: `sparse_ft_backward` の reduction 化 | backward 280 ms の一部 (内訳は sm_86 で要計測)。FT が一番大きい層 | 中〜高 |
| #81 | P5: pinned + async H2D + 2-stream + ring buffer | h2d は ~1.5% だが #89 (dataloader) と合流すべき | 中 |
| #82 | P6: kernel launch 数削減 | launch overhead は 0.6 ms/step (~50 launch) — **低い** | 低 |
| #83 | optimizer fuse + bias/loss block reduce | optimizer は ~2.5%。bias_grad / loss_wdl の atomic は要 ncu (sm_86) | 低〜中 (bias/loss 次第) |

**推奨着手順**: #78 ≈ #89 (両方 ~大、低リスク) → #80 / #79 (backward の 280 ms) → 残り。
ただし #79/#80 の前に **sm_86 box で ncu/nsys-gpu を取って backward の内訳を確定**するべき
(でないと「dense vs sparse どっちが重いか」が分からないまま手を入れることになる)。

## 再現方法

```bash
cd bins/nnue_train && CUDA_OXIDE_TARGET=sm_75 /mnt/e/cuda-oxide-target/release/cargo-oxide build
cd ../.. && cargo build -p nnue-trainer --release
# phase breakdown:
NNUE_TRAIN_STEP_PROFILE=1 ./target/release/nnue-train \
    --data crates/shogi-format/tests/data/sample.psv --output /tmp/x --net-id x \
    --superbatches 1 --batches-per-superbatch 6 --batch-size 65536 --lr 1e-3 --save-rate 999
# CUDA API breakdown:
nsys profile --trace=cuda,nvtx,osrt --sample=none -o /tmp/nnue_prof --force-overwrite=true \
    ./target/release/nnue-train --data crates/shogi-format/tests/data/sample.psv \
    --output /tmp/x --net-id x --superbatches 1 --batches-per-superbatch 5 --batch-size 65536 --lr 1e-3
nsys stats --report cuda_api_sum /tmp/nnue_prof.nsys-rep
```
