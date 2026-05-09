# 開発環境セットアップ

rshogi-nnue は **cuda-oxide** (NVIDIA Labs の Rust → PTX rustc backend) を中核
に据えるため、host (LLVM 21+) と GPU (sm_80+ 公式) の両方を整える。本リポは
**WSL2 Ubuntu 24.04 + RTX 2070 SUPER (sm_75 Turing, 8 GB)** で動作確認しており、
Turing GPU でも workaround 込みで Stage 0-1 の smoke test まで通る。

## システム要件

| 項目 | 要件 | 備考 |
|---|---|---|
| OS | Linux (Ubuntu 24.04 確認) | WSL2 含む |
| CUDA Toolkit | 12.x (12.9 で確認) | nvcc, libNVVM, nvJitLink |
| LLVM | **21+** (NVPTX backend 必須) | Ubuntu 24.04 noble は標準 apt が LLVM 20 まで → apt.llvm.org 追加 |
| Clang | **clang-21** + `libclang-common-21-dev` | `cuda-bindings` の bindgen に必要 |
| Rust | nightly-2026-04-03 (cuda-oxide pinned) | `rust-toolchain.toml` で固定 |
| GPU | **公式: Ampere+ (sm_80+)** | RTX 30/40/50 系, A100, H100, B200 等。Turing (sm_75) は workaround 必要 |

## システム install

```bash
# 基本ツール
sudo apt-get update
sudo apt-get install -y wget gnupg lsb-release

# LLVM 21 系一式 (apt.llvm.org)
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

## Smoke test (公式: sm_80+ GPU)

Ampere 以降の GPU を持っている場合は `cargo oxide run` で直接動く:

```bash
cd ~/git-repos/cuda-oxide
./target/release/cargo-oxide run vecadd
# → "✓ SUCCESS: All 1024 elements correct!"

./target/release/cargo-oxide run atomics
# → "=== SUCCESS: All 20 atomic tests passed! ==="
```

## sm_75 (Turing) workaround

### 制約

cuda-oxide の `rustc-codegen-cuda` は内部で **`llc --mcpu=sm_80` 固定** で PTX を
生成するため (`--arch=sm_75` を渡しても codegen 内部で sm_80 にクランプされる)、
Turing 系 GPU で `cargo oxide run` を直接実行すると `CUDA_ERROR_INVALID_PTX`
(driver error 218) になる。

### 仕組み (なぜ動かせるか)

cuda-oxide pipeline は

```
Rust → MIR → dialect-mir → mem2reg → dialect-llvm → LLVM IR → llc → PTX
```

の経路を取り、各 example dir に **`.ll` (LLVM IR)** と **`.ptx`** を出力する。
example binary 側は `ctx.load_module_from_file("<name>.ptx")` で **実行時に
PTX file を読む** ので、`.ll` を `llc-21 --mcpu=sm_75` で再 compile して
`<name>.ptx` を上書きしてから binary を直接実行すれば sm_75 GPU で動く
(LLVM IR に sm_80 専用 op が含まれない限り)。

### 自動化スクリプト

`scripts/build_for_sm75.sh` が build → IR → PTX 再生成 → 差し替え → 実行を
1 コマンドにまとめてある:

```bash
# rshogi-nnue リポ root から
scripts/build_for_sm75.sh vecadd
scripts/build_for_sm75.sh atomics

# build と PTX 差し替えのみ (実行しない)
scripts/build_for_sm75.sh vecadd --no-run

# cuda-oxide repo の場所を上書き
CUDA_OXIDE_DIR=/path/to/cuda-oxide scripts/build_for_sm75.sh vecadd

# 別 arch (例: sm_70 Volta) でも同じ方式で動く可能性
ARCH=sm_70 scripts/build_for_sm75.sh vecadd
```

### 適用範囲

LLVM IR に以下の sm_80+ 専用 op が **含まれていない** 単純 kernel に限る:

- `cp.async` — asynchronous global → shared copy (sm_80+)
- `wgmma` — warpgroup matrix-multiply-accumulate (sm_90+ Hopper)
- `tcgen05` — 5th-gen tensor cores (sm_100+ Blackwell)
- `tma.*` — Tensor Memory Accelerator (sm_90+)
- `cluster.*` — Thread Block Cluster (sm_90+)

スクリプトは IR を grep して該当 op があれば warning を出す。**Stage 1 KP-abs**
(forward / grad scatter / adam_step / eval) は適用範囲内見込み。Stage 2+ で
fused / async copy / cluster ops を使うと workaround 不能になる可能性があり、
その時は実機 sm_80+ GPU が必要 (sh11235 等)。

### 動作確認 (2026-05-09)

- 環境: WSL2 Ubuntu 24.04, RTX 2070 SUPER (sm_75), CUDA 12.9, LLVM 21.1.8,
  rustc nightly-2026-04-03
- cuda-oxide commit `6de0509` (NVlabs/cuda-oxide main, 2026-05-08)
- vecadd: `✓ SUCCESS: All 1024 elements correct!`
- atomics: `=== SUCCESS: All 20 atomic tests passed! ===` (F32/F64/U64
  atomicAdd 全て含む、Issue #1 受け入れ条件カバー)

## サポート GPU マトリクス

cuda-oxide 公式は **Ampere+ (sm_80+)**。本リポは WSL2 + sm_75 (Turing) を主環境
とし、sm_80+ 環境は smoke 後の本番学習用に分離している。

| 世代 | sm | 代表的な GPU | cargo oxide run | sm_75 workaround |
|---|---|---|---|---|
| Pascal | sm_60/61 | GTX 10xx, P100, P40 | ✗ | 未検証 (LLVM IR 互換性も未確認) |
| Volta | sm_70 | V100, Titan V | ✗ | `ARCH=sm_70` で動く可能性 (未検証) |
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
  **`CARGO_TARGET_DIR=/mnt/e/cuda-oxide-target`** を推奨 (毎回 export か
  `~/.cargo/config.toml` で永続化)

> **Caveat**: cuda-oxide の sub-workspace (`crates/rustc-codegen-cuda`) は
> in-tree target を要求するため、CARGO_TARGET_DIR を export したまま
> `cargo oxide doctor` を走らせると `.so` 探索が失敗する。症状が出たら
> sub-workspace の `librustc_codegen_cuda.so` を期待パスに symlink する:
>
> ```bash
> ln -sf $CARGO_TARGET_DIR/debug/librustc_codegen_cuda.so \
>        ~/git-repos/cuda-oxide/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
> ```

## 関連

- [ADR-0003 cuda-oxide adoption](01-decisions/0003-cuda-oxide-adoption.md) —
  採用判断と Consequences (本ファイルの上位)
- [cuda-oxide upstream](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide-book installation requirements](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
