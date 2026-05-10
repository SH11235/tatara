# Stage 1-10: experiments/001 numerical equivalence + 性能ベンチ

Issue: [#14](https://github.com/SH11235/rshogi-nnue/issues/14)

bullet-shogi `shogi_progress_kpabs_train_cuda` (NVRTC + raw `__global__`) を Stage 1-5..1-9 で cuda-oxide `#[kernel]` に書き直した実装が、**数値的に同等** で **性能が許容範囲内** であることを確認するためのまとめ。

## 自動化された数値同等性 (本 PR で追加)

`experiments/001-cuda-oxide-kpabs/src/main.rs` の `#[cfg(test)] mod gpu_cpu_equivalence_tests` で、4 GPU kernel を対応する CPU reference (`kernels/{forward,grad,adam_step,eval}.rs::*_cpu`) と直接比較する。

### 実行方法

```bash
cd experiments/001-cuda-oxide-kpabs
CUDA_OXIDE_TARGET=sm_75 \
    /mnt/e/cuda-oxide-target/release/cargo-oxide build  # .ll 生成
cargo test -p exp-001-cuda-oxide-kpabs \
    --bin exp-001-cuda-oxide-kpabs --release \
    -- --test-threads=1 --nocapture
```

### スコープ

- `forward_kernel_matches_cpu_reference`: 16 positions × 8 max_inds × 64 weights、padding 混在の小規模 batch で `forward` の preds 出力を比較。許容 1e-5
- `grad_kernel_matches_cpu_reference`: 同上で `grad` の `grad[idx]` (atomic scatter)、`loss_acc` (f64 atomic)、`hist[bin]` (u64 atomic) を比較。grad は 1e-5、loss は 1e-8、hist は完全一致を要求
- `eval_kernel_matches_cpu_reference`: 24 positions の preds/targets を直接渡して `eval` の loss/hist を比較
- `adam_step_kernel_matches_cpu_reference`: 32 weights の Adam 1 step 後の `weights/m/v/grad` を比較
- `samples_per_sec_baseline_on_sample_psv`: sample.psv の先頭ゲームから 4 games × 8 positions = 32 positions の batch を 50 step 回し、samples/sec を `println!` で記録 (assert は緩く `> 1.0` のみ)

### sm_75 (RTX 2070 SUPER) ローカル測定値 (2026-05-11)

| Test | Result |
|---|---|
| forward ↔ forward_cpu (16 pos × 8 inds) | PASS (max diff < 1e-5) |
| grad ↔ grad_cpu (16 pos × 8 inds) | PASS |
| eval ↔ eval_cpu (24 pos) | PASS |
| adam_step ↔ adam_step_cpu (32 weights, 1 step) | PASS |
| **samples/sec baseline** | **218,245 samples/sec** (32 pos/batch × 50 steps in 7.3 ms) |

> CPU reference は single-thread 実装なので `grad[idx]` への atomic add 順序が GPU と完全一致しない可能性があるが、本 test の小規模 batch (≤ 32 positions) では f32 round-off が 1e-5 以内に収まることを確認済み。

## bullet-shogi 上流とのクロス検証 (manual procedure)

bullet-shogi のローカル build と同一 PSV データが必要なため、本 PR では **手動検証手順** として記録する (Stage 1-11 以降で必要に応じて自動化する想定)。

### 前提

- bullet-shogi **rev `f275eb9`** (Stage 1-1..1-10 の vendor 元と同じ) が clone 済み、
  CUDA + NVRTC が動く環境で
  `cargo run --release -p bullet-shogi --example shogi_progress_kpabs_train_cuda` が実行可能
- 同一 PSV データ (例: `/mnt/e/rshogi-nnue/data/<some>.bin`)
- 両実装で同一 hyperparam: `--games-per-step 1024 --epochs 1 --lr 1e-3 --lr-scale sqrt`

### 1. loss 推移の比較

```bash
# bullet-shogi 側
cd ~/git-repos/bullet-shogi
cargo run --release --example shogi_progress_kpabs_train_cuda -- \
    --data /mnt/e/rshogi-nnue/data/<some>.bin \
    --output /tmp/bullet_progress.bin \
    --epochs 1 --games-per-step 1024 --lr 1e-3 --lr-scale sqrt \
    --log-interval-steps 100 \
    > /tmp/bullet_log.txt 2>&1

# rshogi-nnue 側 (本リポ)
cd ~/git-repos/rshogi-nnue/experiments/001-cuda-oxide-kpabs
cargo run --release -p exp-001-cuda-oxide-kpabs -- \
    --data /mnt/e/rshogi-nnue/data/<some>.bin \
    --output /tmp/rshogi_progress.bin \
    --epochs 1 --games-per-step 1024 --lr 1e-3 --lr-scale sqrt \
    --log-interval-steps 100 \
    > /tmp/rshogi_log.txt 2>&1
```

両 log の `avg_loss` を step 単位で比較。許容 drift は **1e-3 程度** (受け入れ条件)。差分が大きい場合は

- bullet-shogi 側 `interleave_pack_groups` で game 順序が変わる可能性 (rshogi 単一 thread vs bullet multi-thread prefetch の違い)
- f32 atomic 加算順序の差 (kernel scatter で衝突したときの round-off)

を疑う。

### 2. progress.bin の比較

```bash
# byte 一致しないので浮動小数誤差で判定
python3 - <<'EOF'
import struct
with open('/tmp/bullet_progress.bin', 'rb') as f1, open('/tmp/rshogi_progress.bin', 'rb') as f2:
    b1, b2 = f1.read(), f2.read()
assert len(b1) == len(b2) == 1_003_104
n = len(b1) // 8
max_diff = 0.0
for i in range(n):
    v1 = struct.unpack('<d', b1[i*8:(i+1)*8])[0]
    v2 = struct.unpack('<d', b2[i*8:(i+1)*8])[0]
    max_diff = max(max_diff, abs(v1 - v2))
print(f'max abs diff = {max_diff}')
EOF
```

許容: max abs diff < **1e-3** (受け入れ条件、weight magnitude は ~0.0..2.0 程度なので相対誤差 ~1e-3)。

### 3. samples/sec 比較

両 log の `samples N` と elapsed time から samples/sec を算出し、rshogi が bullet 比 80% 以上であることを確認 (受け入れ条件)。

> 既知の perf ボトルネック: Stage 1-9 では `step` ごとに `DeviceBuffer::from_host` で `indices/targets/per_pos_norm/preds` を新規 allocate する path で、bullet-shogi の persistent scratch buffer (`upload_prefix` 経由) より overhead が大きい。Stage 1-10 / Stage 2 で `Stream::memcpy_h2d` 経由の scratch reuse を導入する想定 (TODO、`src/main.rs::GpuTrainer::step` 参照)。

## cuda-oxide 不具合 / 制限の文書化

Stage 1 を通じて遭遇した cuda-oxide rev `6de0509` の不具合 / 制限。受け入れ条件「cuda-oxide 不具合に当たった場合は何が動かなかったか文書化」に該当。

### 1. `Ord::clamp` (i32) lowering 失敗

- 症状: kernel で `i32::clamp(0, 7)` を使うと build 時に
  `Type translation not yet implemented for: ... std::fmt::Debug::fmt`
- 原因: `core::cmp::Ord::clamp` は内部で `assert!(min <= max)` を含み、その panic
  経路に `Debug::fmt` 参照が残る。cuda-oxide rustc-codegen-cuda は現状 `Debug::fmt`
  を NVPTX に lowering できない
- workaround: kernel 側は verbatim `if-else` で書く (`#[allow(clippy::manual_clamp)]`)。
  CPU reference は host 実行なので `i32::clamp` で OK
- 該当: Stage 1-6 (grad)、Stage 1-8 (eval)

### 2. `f32::max` lowering 失敗

- 症状: kernel で `bc.max(1e-30f32)` を使うと build 時に
  `Symbol std__intrinsics__maximum_number_nsz_f32 not found`
- 原因: `f32::max` は `core::intrinsics::maximum_number_nsz_f32` を呼び、cuda-oxide
  はその intrinsic を未対応
- workaround: kernel 側で `if bc > 1e-30 { bc } else { 1e-30 }` に展開。
- 該当: Stage 1-7 (adam_step)

### 3. libNVVM opaque pointer NVVM IR を parse できない

- 症状: cuda-oxide 出力の `.ll` を `cuda_host::ltoir::build_cubin_from_ll` に渡すと
  `libnvvm error 9: parse expected type` で reject
- 原因: cuda-oxide rev `6de0509` が出力する NVVM IR は LLVM 21 系の opaque
  pointer 形式 (`define void @grad(ptr ...)`) で、libNVVM の内蔵 LLVM が古く
  opaque pointer 未対応
- workaround: libNVVM を bypass して `llvm-link-21 + opt-21 (passes='nvvm-reflect,
  internalize,globaldce') + llc-21` の 3 段 pipeline を host 側で組む。
  `experiments/001-cuda-oxide-kpabs/src/main.rs::compile_ll_to_ptx_via_llc`
  に実装
- 該当: Stage 1-9 (host loop integration、kernel module load 時)

### 4. cargo-oxide build の `.ll` 出力先

- 症状: `cargo-oxide build` を experiment crate dir で実行すると `.ll` が
  workspace root に出力される (`load_kernel_module` が CARGO_MANIFEST_DIR を
  読むのと不整合)
- workaround: host loader が CARGO_MANIFEST_DIR と workspace root の両方を probe
- 該当: Stage 1-9
