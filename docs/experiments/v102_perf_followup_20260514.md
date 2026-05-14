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

## phase D v2 試作 (1 block per feature) — **失敗、revert**

`phD_stm 14.58 ms + phD_nstm 14.56 ms = 29.14 ms` を target に、新版 `_v2` kernel を
試作:
- 構造: grid=(FT_IN, 1, 1)=73K blocks (旧版 880K の 12× 削減) × block_dim=256
- 各 thread が register accumulator 6 個 (acc0..acc5) を持ち、bi loop 内で grad_out
  の 1 row を 8 warps × 6 sweeps で sequential 読み (coalesced)
- 単一 block per feature により nstm の atomic add を non-atomic RMW に置換
- 期待効果 5–10 ms 削減

**実測 (v2 適用、未 commit、1 sb × 3 batches × bs=65536)**:
- phD_stm: **19.463 ms** (旧版 14.58 ms → +33%)
- phD_nstm: **19.736 ms** (旧版 14.56 ms → +36%)
- phase D 合計 **39.20 ms / step** (旧版 29.14 ms より +10.06 ms 悪化)
- 1 sb の pos/s 比較: 252,864 vs 旧版 269,036 (-6%)、loss 0.161902 は一致 (correctness OK)

**敗因仮説**:
1. 6 register accumulator が cuda-oxide で local memory spill されている可能性
   (確認には PTX 検査 + ncu のレジスタ使用量計測が必要)
2. 1 block per feature への集約で per-block 内 work が増え、warp scheduler が他 block
   の latency を隠せる余地が減った
3. 旧版の `if sum != 0.0_f32` 早期 skip が消え、empty feature でも 1536 cell の 0
   書き込みを実施
4. 6 sweep × 14 bi = 84 indirect array access per thread の bounds check 累積コスト

**判断**: revert (v2 kernel 削除、host launch も旧版に戻し)。phase D は 1 block per
feature 構造では性能改善せず、別アプローチが必要:
- compact appearing-features list を構築して空 block を完全排除
  (要 stream compaction、device→host n_appearing 取得 or 最大 grid + early-exit)
- shared mem reduction (multi-warp cooperation per row)
- グリッド構造はそのままで grad_out の access pattern を transpose
  (例: bi-major 内側、feature-major 外側)

これら全て要 0.5–1 日の試作 + 計測 cycle で、確実な勝ち筋なし。実 work では本
領域は cuda-oxide の register allocator の癖を確認しないと方針が立たない。

## 仮説検証 (PTX + ptxas + variant 計測) — 2026-05-14 追補

phase D v2 の negative 結果に対する 4 仮説 (spill / occupancy / early-exit / bounds
check) を、再投入 + PTX 検査 + variant kernel で個別検証した。

### H1 (register spill) — **REFUTED**

PTX inspection (`grep ".local"`, `ptxas -v`):
- v2 PTX: `.reg .pred %p<5>; .reg .b32 %r<15>; .reg .b64 %rd<25>` (h4 with raw-ptr)
- ptxas: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads` で OLD/v2 共
- 6 個の f32 accumulator (acc0..acc5) は全 register 内に配置、local memory への spill 一切なし

### H2 (occupancy/warp scheduling) — **REFUTED**

ptxas -v + sm_86 capacity:
| | regs/thread | block_dim | max blocks/SM | active warps/SM |
|---|---:|---:|---:|---:|
| OLD overwrite/add | 20 | 128 | 12 (cap by 1536 threads/SM) | 12×4 = **48** |
| v2 (h4 with raw-ptr) overwrite/add | 40 | 256 | 6 (cap by 1536 threads/SM) | 6×8 = **48** |

両者 sm_86 の最大 active warps/SM = 48 で full occupancy。occupancy 単独で説明不能。

### H3 (empty-feature 早期 skip 消失) — **論理矛盾で testable でない**

v2 overwrite は memset_zero 除去 (P6, f66acdb) 前提で empty feature でも 1536 cell に
0 を書き切る必要があり、早期 skip と相容れない。v2 add は元から早期 skip 有。

### H4 (bounds check 累積) — **CONFIRMED (-3.8 ms / -10%)**

v2 を **`grad_out[bi * 1536 + tid + K*256]` の bounds-checked indexing から
`grad_out_ptr.add(...).read()` の raw pointer read へ置換** して再計測:
- v2 (bounds-checked): phD_stm 19.46 + phD_nstm 19.74 = **39.20 ms**
- v2 (h4, raw-ptr):    phD_stm 17.7  + phD_nstm 17.8  = **35.4 ms** (-3.8 ms)

PTX で `setp.ge.u64; @%p bra` (loop 内 6 連続) が消失、loop body が `ld.b32 + add.rn.f32`
× 6 + ループ制御だけになることを確認。

### v2 vs OLD の残り 6.3 ms 差の原因 (ncu 無しでは厳密確認不可)

v2 (h4, 35.4 ms) と OLD (29.14 ms) の差は memory pipeline 由来と推測:
- v2: 1 iter で **8 warps × 6 cache lines = 48 cache line read 同時要求**、6 KB 即時
  fetch
- OLD: 1 iter で **4 warps × 1 cache line = 4 cache line read**、512 byte fetch
- 同 SM の queue depth / L2 bandwidth に対する instantaneous demand が v2 で大きい

### **真の win: OLD 構造 + raw-ptr 化 → bullet 超え達成 (b8d2abf)**

最大の発見は **既存 OLD kernel にも同じ bounds check が 3 箇所/iter ある** こと。OLD
の構造 (12 blocks × 4 warps × 1 cache line/iter) を維持して bounds check のみ raw-ptr
化したところ:

| 計測 | sb1 | sb2 | sb3 | sb4 | sb5 | 平均 | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (f2e3d34) | 667,194 | 676,530 | 671,547 | 667,636 | 666,562 | **669,894** | — |
| OLD+rawptr (run 2) | 704,148 | 715,510 | 713,748 | 712,986 | 712,359 | **711,750** | **+6.1%** |
| bullet-shogi v102 avg | — | — | — | — | — | **691,267** | rshogi 103% |

phase D 時間: phD_stm 14.58 → 11.36, phD_nstm 14.56 → 10.81、合計 29.14 → **22.20 ms**
(-6.94 ms = -24%)。step 全体への寄与: 6 ms × ~1000 steps/sb × 5 sb / 5 sb = +6.1% pos/s。

loss 軌跡完全一致 (sb5 差 = 4e-5、f32 atomic 順序差以内)。

ptxas reg 使用量: overwrite 20 → 26、add 20 → 28 (raw_ptr 値の register 保持分)。
occupancy 不変 (max threads/SM = 1536 で cap)。

### 教訓

1. **review の "高 ROI 候補" は仮説止まりで、実測なしには結論できない** — phase D の
   v2 試作は H1-H2 で説明可能と踏んだが、PTX 検査で両仮説とも REFUTED となり、H4
   の bounds check のみ寄与と判明。
2. **既存 kernel に同じ bounds check 問題が潜んでいる** — v2 のために導入した raw-ptr
   化を OLD に逆適用したのが当セッション最大の win。コードベースに分散している同
   パターンを系統的に raw-ptr 化すれば追加 win が期待できる (要 unsafe 妥当性監査)。
3. **PTX + ptxas -v は cuda-oxide の挙動を観測する最短経路** — ncu 不可の WSL2 / 制限
   環境でも、PTX inspection だけで spill / 占有 register / bounds check の有無を確認
   できる。今後の最適化サイクルに組み込む価値あり。

## commit hash anchor

- `f2e3d34` — EPIC #75 v102 perf 292K → 665K (perf sprint baseline)
- `f66acdb` — P6: ft_w_grad redundant memset 削除 (perf neutral)
- `0972c9b` — measurement log doc (本ファイル) 追加
- `0a071b1` — phase D に phD_stm / phD_nstm 独立 prof_tick 追加
- `b31dce8` — phase D v2 試作の negative 結果記録 (revert 済)
- **`b8d2abf`** — **phase D gather kernel の bounds check を raw-ptr 化 (+6.1% pos/s, bullet 超え)**
