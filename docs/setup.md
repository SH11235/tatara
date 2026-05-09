# 開発環境セットアップ

rshogi-nnue は **cuda-oxide** (NVIDIA Labs の Rust → PTX rustc backend) を中核
に据えるため、host (LLVM 21+, できれば LLVM 22) と GPU (sm_80+ 公式) の両方を
整える。本リポは **WSL2 Ubuntu 24.04 + RTX 2070 SUPER (sm_75 Turing, 8 GB)** で
動作確認しており、Turing GPU でも 1 つの環境変数 (`CUDA_OXIDE_TARGET=sm_75`)
で Stage 0-1 の smoke test まで通る。

## システム要件

| 項目 | 要件 | 備考 |
|---|---|---|
| OS | Linux (Ubuntu 24.04 確認) | WSL2 含む |
| CUDA Toolkit | 12.x (12.9 で確認) | nvcc, libNVVM, nvJitLink |
| LLVM | **21+ (floor)、22 推奨** | Ubuntu 24.04 noble は標準 apt が LLVM 20 まで → apt.llvm.org 追加。`llc-22` が PATH にあれば cuda-oxide が優先する |
| Clang | **clang-21** + `libclang-common-21-dev` | `cuda-bindings` の bindgen に必要 (LLVM 22 にしても clang-21/22 のどちらかが要る) |
| Rust | nightly-2026-04-03 (cuda-oxide pinned) | `rust-toolchain.toml` で固定 |
| GPU | **公式: Ampere+ (sm_80+)**。Turing (sm_75) も `CUDA_OXIDE_TARGET=sm_75` で動作 | RTX 30/40/50, A100, H100, B200 等 |

> **LLVM 22 と atomics の syncscope**: cuda-oxide の `atomics` example README
> は「Atomic operations require llc-22 or newer for correct syncscope」と記載。
> LLVM 21 でも例題は完走するが、`memory_order` まわりの正確な PTX を求める
> 場面 (Stage 1+ の本番 kernel) では `llc-22` への昇格が望ましい。pipeline は
> `llc-22` → `llc-21` の順で auto-discover する (`CUDA_OXIDE_LLC=/path` で固定可)。

## システム install

```bash
# 基本ツール
sudo apt-get update
sudo apt-get install -y wget gnupg lsb-release

# LLVM 21 系一式 (apt.llvm.org)。LLVM 22 を入れるなら `21` を `22` に置換
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
chmod +x /tmp/llvm.sh
sudo /tmp/llvm.sh 21
sudo apt-get install -y clang-21 libclang-common-21-dev

# clang を vanilla 名で参照可能に
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-21   100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100

# 確認
which llc-21 clang
llc-21 --version | grep nvptx
```

## cuda-oxide のセットアップ

cuda-oxide は **外部 repo として参照** (本リポには vendor しない):

```bash
git clone https://github.com/NVlabs/cuda-oxide.git ~/git-repos/cuda-oxide
cd ~/git-repos/cuda-oxide

# 動作確認した commit に固定 (任意、main 追従でも OK)
git checkout 6de0509

# cargo-oxide ツールをビルド (cuda-oxide の rust-toolchain.toml が active になる)
cargo build -p cargo-oxide --release

# 環境チェック
./target/release/cargo-oxide doctor
```

`cargo oxide doctor` 全項目 ✓ になれば host 側は OK。

## Smoke test

### Ampere+ GPU の場合 (公式パス)

```bash
cd ~/git-repos/cuda-oxide
./target/release/cargo-oxide run vecadd
# → "✓ SUCCESS: All 1024 elements correct!"

./target/release/cargo-oxide run atomics
# → "=== SUCCESS: All 20 atomic tests passed! ==="
```

### sub-Ampere GPU (sm_70/75) の場合: `CUDA_OXIDE_TARGET` 上書き

`cargo oxide` 単独では auto-detect (cuda-oxide の `select_target()` 関数) が
kernel features から target を選び、Basic フォールバックでは `sm_80` を選ぶ。
`--arch=sm_75` を渡しても auto-detect が override してしまうため、PTX header は
`.target sm_80` のままになり、Turing GPU では `CUDA_ERROR_INVALID_PTX` (driver
error 218) で load が失敗する。

回避策は **`CUDA_OXIDE_TARGET=sm_75` 環境変数** を渡すこと。これは
`mir-importer/src/pipeline.rs` で `select_target()` をバイパスする一級 override。

```bash
cd ~/git-repos/cuda-oxide
CUDA_OXIDE_TARGET=sm_75 ./target/release/cargo-oxide run vecadd
# → PTX header `.target sm_75` で生成される
# → "✓ SUCCESS: All 1024 elements correct!"

CUDA_OXIDE_TARGET=sm_75 ./target/release/cargo-oxide run atomics
# → 20/20 tests passed (F32/F64/U64 atomicAdd 全て含む)
```

毎回打つのが面倒なら shell rc に export しておくか、本リポ内の experiment
スクリプトで `env CUDA_OXIDE_TARGET=$target_arch ...` を埋め込む。

### sub-Ampere の限界

`CUDA_OXIDE_TARGET=sm_75` で **動く** のは「LLVM IR に sm_80+ 専用 op が含まれて
いない場合」に限る。具体的には:

- `cp.async` — asynchronous global → shared copy (sm_80+)
- `wgmma` — warpgroup matrix-multiply-accumulate (sm_90+ Hopper)
- `tcgen05` — 5th-gen tensor cores (sm_100+ Blackwell)
- `tma.*` — Tensor Memory Accelerator (sm_90+)
- `cluster.*` — Thread Block Cluster (sm_90+)

これらが含まれた IR を sm_75 PTX に compile しても、`llc` の段階か CUDA driver
の JIT load 段階で失敗する。**Stage 1 KP-abs (forward / grad scatter /
adam_step / eval) は適用範囲内見込み**。Stage 2+ で fused / async copy /
cluster ops を使い始めたら、その時点で sm_80+ GPU (sh11235 等) が必要。

`CUDA_OXIDE_TARGET=sm_75` で生成された IR に sm_80+ op が混入していないかは、
example dir の `<name>.ll` を grep して確認できる:

```bash
grep -E '(cp\.async|wgmma|tcgen05|tma\.|cluster\.)' \
  ~/git-repos/cuda-oxide/crates/rustc-codegen-cuda/examples/vecadd/vecadd.ll
# (no output = OK)
```

### 動作確認 (2026-05-09)

- 環境: WSL2 Ubuntu 24.04, RTX 2070 SUPER (sm_75), CUDA 12.9, LLVM 21.1.8,
  rustc nightly-2026-04-03
- cuda-oxide commit `6de0509` (NVlabs/cuda-oxide main, 2026-05-08)
- vecadd  (`CUDA_OXIDE_TARGET=sm_75`): `✓ SUCCESS: All 1024 elements correct!`
- atomics (`CUDA_OXIDE_TARGET=sm_75`): 20/20 tests passed
- 公式 sm_86 (Ampere) 実機検証は GPU 解放後の follow-up issue で再実施

## サポート GPU マトリクス

| 世代 | sm | 代表的な GPU | `cargo oxide run` 直接 | `CUDA_OXIDE_TARGET=sm_XX` |
|---|---|---|---|---|
| Pascal | sm_60/61 | GTX 10xx, P100 | ✗ | 未検証 (LLVM IR 互換性も要確認) |
| Volta | sm_70 | V100, Titan V | ✗ | 動く可能性 (未検証) |
| Turing | sm_75 | **RTX 2070 SUPER** (本リポ確認), GTX 16xx, T4 | ✗ | ✅ 確認済み |
| Ampere | sm_80 | A100, A30 | ✅ | n/a |
| Ampere | sm_86 | RTX 30xx, A40, A10 | ✅ | n/a |
| Ada | sm_89 | RTX 40xx | ✅ | n/a |
| Hopper | sm_90 | H100, H200 | ✅ | n/a |
| Blackwell | sm_100/120 | B100, B200, RTX 50xx | ✅ | n/a |

## WSL2 ディスク注意

WSL2 環境では `/` (ext4) は **C: ドライブ上の sparse vhdx** が実体。`df -h /` の
表示は仮想容量で、物理は `df -h /mnt/c` の Avail に縛られる。本リポでは:

- 数百 GB 級データ (PSV、checkpoint、ログ) は **`/mnt/e/rshogi-nnue/`** に置く
- `data/` ディレクトリは `/mnt/e/rshogi-nnue/data` への symlink
- cuda-oxide の build artifact (`target/`) も C: 圧迫しないよう
  **`CARGO_TARGET_DIR=/mnt/e/cuda-oxide-target`** を推奨

> **Caveat**: cuda-oxide の sub-workspace (`crates/rustc-codegen-cuda`) は
> in-tree target を要求するため、CARGO_TARGET_DIR を export したまま
> `cargo oxide doctor` を走らせると codegen `.so` 探索が失敗する。症状が出たら
> sub-workspace の `librustc_codegen_cuda.so` を期待パスに symlink する:
>
> ```bash
> ln -sf $CARGO_TARGET_DIR/debug/librustc_codegen_cuda.so \
>        ~/git-repos/cuda-oxide/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
> ```

## 関連

- [ADR-0003 cuda-oxide adoption](01-decisions/0003-cuda-oxide-adoption.md) —
  採用判断と Consequences
- [cuda-oxide upstream](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide-book installation requirements](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
- [cuda-oxide atomics example README (LLVM 22 syncscope の根拠)](https://github.com/NVlabs/cuda-oxide/blob/main/crates/rustc-codegen-cuda/examples/atomics/README.md)
