# 開発環境セットアップ

rshogi-nnue は **cuda-oxide** (NVIDIA Labs の Rust → PTX rustc backend) を中核
に据えるため、host (LLVM 21+, できれば LLVM 22) と GPU (sm_80+ 公式) の両方を
整える必要がある。Ampere (sm_86) を primary に、Turing (sm_75) も
`CUDA_OXIDE_TARGET=sm_75` 環境変数で動作確認している (制約はこのファイル末尾
の GPU マトリクス参照)。

## システム要件

| 項目 | 要件 | 備考 |
|---|---|---|
| OS | Linux (Ubuntu 22.04 jammy / 24.04 noble の両方で確認) | WSL2 含む |
| CUDA Toolkit | 12.x (12.9 で確認) | nvcc, libNVVM, nvJitLink |
| LLVM | **21+ (floor)、22 推奨** | apt.llvm.org が jammy / noble の両方に LLVM 20/21/22 を提供。`llc-22` が PATH にあれば cuda-oxide が優先する |
| Clang | **clang-21 or 22** + `libclang-common-{21,22}-dev` | `cuda-bindings` の bindgen に必要 (LLVM 22 にしても clang-21/22 のどちらかが要る) |
| Rust | nightly-2026-04-03 (cuda-oxide pinned) | `rust-toolchain.toml` で固定 |
| GPU | **公式: Ampere+ (sm_80+)**。Turing (sm_75) も `CUDA_OXIDE_TARGET=sm_75` で動作 | RTX 30/40/50, A100, H100, B200 等 |

> **LLVM 22 と atomics の syncscope**: cuda-oxide の `atomics` example README
> は「Atomic operations require llc-22 or newer for correct syncscope」と記載。
> LLVM 21 でも例題は完走するが、`memory_order` まわりの正確な PTX が必要な本番
> kernel では `llc-22` への昇格が望ましい。pipeline は `llc-22` → `llc-21` の順
> で auto-discover する (`CUDA_OXIDE_LLC=/path` で固定可)。

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

cuda-oxide は **外部 repo として参照** (本リポジトリには vendor しない):

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

### sub-Ampere GPU (例: sm_75 Turing) の場合: `CUDA_OXIDE_TARGET` 上書き

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

毎回打つのが面倒なら shell rc に export しておくか、本リポジトリ内の experiment
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
の JIT load 段階で失敗する。KP-abs progress 系の単純な kernel (forward / grad
scatter / adam_step / eval) は sm_75 hack の適用範囲内。fused optimizer step や
async copy / Hopper cluster ops を使う kernel は sm_80+ GPU が必要。

`CUDA_OXIDE_TARGET=sm_75` で生成された IR に sm_80+ op が混入していないかは、
example dir の `<name>.ll` を grep して確認できる:

```bash
grep -E '(cp\.async|wgmma|tcgen05|tma\.|cluster\.)' \
  ~/git-repos/cuda-oxide/crates/rustc-codegen-cuda/examples/vecadd/vecadd.ll
# (no output = OK)
```

## サポート GPU マトリクス

| 世代 | sm | 代表的な GPU | `cargo oxide run` 直接 | `CUDA_OXIDE_TARGET=sm_XX` |
|---|---|---|---|---|
| Pascal | sm_60/61 | GTX 10xx, P100 | ✗ | 未検証 (LLVM IR 互換性も要確認) |
| Volta | sm_70 | V100, Titan V | ✗ | 動く可能性 (未検証) |
| Turing | sm_75 | RTX 2070 SUPER, GTX 16xx, T4 | ✗ | ✅ 確認済み |
| Ampere | sm_80 | A100, A30 | ✅ | n/a |
| Ampere | sm_86 | RTX 3080 Ti, RTX 30xx, A40, A10 | ✅ 確認済み (primary) | n/a |
| Ada | sm_89 | RTX 40xx | ✅ | n/a |
| Hopper | sm_90 | H100, H200 | ✅ | n/a |
| Blackwell | sm_100/120 | B100, B200, RTX 50xx | ✅ | n/a |

cuda-oxide の rev は本リポジトリの `Cargo.toml` に pin している (`gpu-runtime` /
GPU bin 群)。LLVM の組み合わせは LLVM 21 (sm_75) と LLVM 22 (sm_86) のいずれ
でも `cargo oxide doctor` / vecadd / atomics 全件 pass を確認済み。

## WSL2 ディスク注意

WSL2 環境では `/` (ext4) は **C: ドライブ上の sparse vhdx** が実体。`df -h /` の
表示は仮想容量で、物理は `df -h /mnt/c` の Avail に縛られる。数百 GB 級の
学習データ (PSV、checkpoint、ログ) と cuda-oxide の build artifact (`target/`)
は C: 圧迫を避けるため別ドライブに置くことを推奨する:

- 学習データは別ドライブの作業ディレクトリに置き、リポ内の `data/` から
  symlink を張る
- `CARGO_TARGET_DIR=<別ドライブのパス>` を設定

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

- [cuda-oxide adoption ADR](01-decisions/2026-05-09-cuda-oxide-adoption.md) —
  採用判断と Consequences
- [cuda-oxide upstream](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide-book installation requirements](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
- [cuda-oxide atomics example README (LLVM 22 syncscope の根拠)](https://github.com/NVlabs/cuda-oxide/blob/main/crates/rustc-codegen-cuda/examples/atomics/README.md)
