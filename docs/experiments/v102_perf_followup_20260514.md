# v102 perf follow-up 計測ログ (2026-05-14)

EPIC #75 で 292K → 665K pos/s を達成 (commit `f2e3d34`) した直後の、3 並列レビュー
(performance / code quality / module design) で挙げられた追加 perf 修正候補
P1–P7 の **個別計測結果**。

## 計測環境

- **GPU**: NVIDIA GeForce RTX 3080 Ti (sm_86, 12 GiB)
- **Driver**: 580.126.09 / CUDA 13.1
- **OS**: Ubuntu 22.04 (Linux 6.8.0-90-generic, native)
- **Toolchain**: LLVM 22.1.6 (llvm-link-22 / opt-22 / llc-22)、Rust nightly-2026-04-03
- **cuda-oxide**: `/mnt/nvme1/development/cuda-oxide/target/release/cargo-oxide`

## 計測コマンド (全測定共通)

```bash
export CUDA_OXIDE_TARGET=sm_86 \
       LLVM_LINK_BIN=/usr/bin/llvm-link-22 \
       OPT_BIN=/usr/bin/opt-22 \
       LLC_BIN=/usr/bin/llc-22

# build (commit ごとに必要)
cd bins/nnue_train && /mnt/nvme1/development/cuda-oxide/target/release/cargo-oxide build
cd ../.. && cargo build -p nnue-trainer --release

# 実測 (5 sb × 200 batches × bs=65536 で steady-state)
DATA=/mnt/nvme1/development/bullet-shogi/data/DLSuisho15b_aoba_deduped_shuffled.bin
PROG=/mnt/nvme1/development/bullet-shogi/data/progress/progress_hao_full_cuda.e1.bin
target/release/nnue-train --data "$DATA" --progress-coeff "$PROG" \
  --output /tmp/v102_run --net-id v102_run \
  --superbatches 5 --batches-per-superbatch 200 --batch-size 65536 \
  --lr 8.75e-4 --win-rate-model --score-drop-abs 32000 \
  --save-rate 5 --threads 16 --bucket-mode progress8kpabs --epoch-file-shuffle
```

## 計測結果サマリ

| commit | 内容 | sb1 | sb2 | sb3 | sb4 | sb5 | 平均 | vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `f2e3d34` | **EPIC #75 baseline** (292K → 665K perf sprint) | 667,194 | 676,530 | 671,547 | 667,636 | 666,562 | **669,894** | — |
| `f66acdb` | **P6**: ft_w_grad redundant memset 削除 | 661,627 | 671,162 | 669,628 | 667,111 | 667,305 | **667,367** | −0.4% (noise) |

bullet-shogi v102 baseline (`/home/sh11235/development/bullet-shogi/checkpoints/v102/train.log` から
cumulative pos/sec 抽出): n=19600 / min=544,390 / max=707,091 / **avg=691,267**。
rshogi-nnue は **bullet 比 96.9%**。

## loss 軌跡 (相互照合)

| commit | sb1 | sb2 | sb3 | sb4 | sb5 |
|---|---:|---:|---:|---:|---:|
| bullet-shogi v102 | 0.153 | 0.098 | 0.073 | 0.067 | 0.065 |
| `f2e3d34` | 0.153484 | 0.098336 | 0.072798 | 0.067479 | 0.064870 |
| `f66acdb` | 0.153480 | 0.098322 | 0.072800 | 0.067472 | 0.064831 |

f32 atomic 順序差以内、3 系統で一致。

## レビュー予測 vs 実測 (P1–P7)

3 並列レビューでは sparse_ft_forward の break + pinned H2D 等で +8–13 ms/step (~800K
pos/s = bullet 超え) を予測したが、**実測ではほぼ全て効果ゼロ** または見落とした
本当の bottleneck が異なる場所にあった。

| # | 内容 | レビュー予測 | 実測 | 判断 |
|---|---|---:|---:|---|
| P1 | sparse_ft_forward `break` on `-1` padding | +3–7 ms | ~0 ms | **revert**: shogi の avg active は ~38/40 で 5% 削減止まり、PTX で `break` lower 確認したが measured win なし |
| P5 | loss_wdl/wrm block tree reduction | +0.3–0.5 ms | ~0 ms | **revert**: 30+ 行コード追加 vs 効果ゼロ、loss kernel は元から小 work |
| P6 | ft_w_grad redundant memset 削除 | +0.3 ms | ~0 ms (-0.4%) | **commit**: perf zero だが 450 MB の no-op 排除という論理整理 |
| P2 | pinned H2D for batch buffers | +2–3 ms | 未計測 | **skip**: GPU compute が完全 bottleneck (~98 ms/batch wall) でデータパスは ~3–10 ms prefetch overlap 済 |
| P3 | async loss readback (ring) | +1–2 ms | 未計測 | **skip**: 同上、sync 自体が ~1 ms と推定 |
| P4 | ft_post_perspective_grad shared bias reduce | +0.5–1 ms | 未計測 | **skip**: review で `S` 想定だったが grid 構造変更が必要で実は `M`、ROI 不明 |
| P7 | compact feature list for gather kernel | +0.5–1 ms | 未計測 | **要再評価**: phase D の真の bottleneck は random gather access pattern の可能性、empty block 排除では不十分 |

## レビューが見落としていた真の bottleneck

`NNUE_TRAIN_STEP_PROFILE=1` の prof_tick! ラベルがミスリーディングで、`phA_reset
14.7 ms × 2 = 29 ms` の実体は **前 iter の phase D (`gather_and_sum_per_feature_overwrite/
add`) compute 時間** (prof_tick が `stream.synchronize()` を呼ぶため、次の tick が前
phase の compute を含む)。

phase breakdown (`f2e3d34` profile-on、batch=65536):

| phase | 時間 (ms) | 備考 |
|---|---:|---|
| `fwd_ft` (sparse_ft_forward × 2) | 28.04 | BW-bound、float4 vectorize で +5–15% 可能 (cuda-oxide で `*const float4` 要 raw ptr) |
| **phase D × 2 (gather_and_sum)** | **~30** | random gather pattern、~700K/880K blocks が empty |
| `fwd_L1` (dense_mm_fwd_bucket_tiled_l1) | 7.42 | 既に shared-mem tiled、cuBLAS なら ~80% peak (cuda-oxide に binding 無し) |
| `bwd_L1f` (dense_mm_bwd_weight_tiled) | 8.50 | 同上 |
| `bwd_L1_inB` | 4.32 | |
| `optimizer` (radam × 10 + lookahead × 10) | 4.45 | memory-bound (ft_w 450MB の DRAM) |
| 他 | <2 ea | |

## 次の最適化候補 (実測ベースで再評価)

| 候補 | 推定効果 | 工数 | リスク | 備考 |
|---|---:|---|---|---|
| **phase D access pattern 再設計** (batch-major scan, shared-mem reduce) | 5–10 ms | H | 中 | 真の最大 bottleneck。grid 構造 + memory traffic 両方再考必要 |
| **sparse_ft_forward float4 vectorize** | 3–5 ms | H | 高 | cuda-oxide で `*const float4` 経由 unsafe raw ptr、Rust の bounds check と相性悪い可能性 |
| **dense matmul を cuBLAS bind** | 1–3 ms | H | 高 | cuda-oxide に cuBLAS binding 無し、別途 FFI shim 必要 |
| **多 stream + CUDA Graph capture** | 0.5–1 ms | M-H | 中 | GPU bottleneck 状態では ROI 低 |

## 教訓

1. **review の bandwidth 計算は traffic 仮定が誤っていた**: sparse_ft_forward の active
   feature 数を 30 と仮定したが、shogi の盤面ピース数は ~38 なので削減代は 5% (1.5 ms)
   止まり。
2. **prof_tick! ラベルが phase D を隠していた**: `phA_reset` という名前で 14.7 ms 計上
   されているが、実体は前 phase の compute 待ち。**phase D 直後に `prof_tick!("phD")`
   を入れる** だけでも次の review が大いに正確になる (P-test として残課題)。
3. **micro-optimization のレビューは実測なしには空振りしやすい**: P1/P5 のような 1
   kernel level の予測は ±数 ms で実測してみないと正味判断できない。

## commit hash anchor

- `f2e3d34` — EPIC #75 v102 perf 292K → 665K
- `f66acdb` — P6: ft_w_grad redundant memset 削除 (perf neutral)
