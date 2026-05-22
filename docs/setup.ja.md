[English](setup.md) | **日本語**

# 開発環境セットアップ

tatara は **cuda-oxide** (NVIDIA Labs の Rust → PTX rustc backend) を中核
に据えるため、host (LLVM 21+, できれば LLVM 22) と GPU (sm_80+ 公式) の両方を
整える必要がある。Ampere (sm_86) を primary に、Turing (sm_75) も
`CUDA_OXIDE_TARGET=sm_75` 環境変数で動作する (制約は「sub-Ampere GPU」節と
末尾の GPU マトリクス参照)。

## 対応 OS

| OS | 位置づけ | 手順 |
|---|---|---|
| Linux | 一級サポート (Ubuntu 22.04 / 24.04 で確認) | 本ファイルの手順をそのまま実行 |
| Windows | WSL2 (Ubuntu) 経由でサポート。native Windows は cuda-oxide が公式に非サポート (CPU-only crate のみ native でビルド可) | 先に「Windows (WSL2) の準備」、以降は Linux と共通 |
| macOS | GPU ビルドは非対応 | リモートの Linux + NVIDIA GPU で作業 (下記) |

cuda-oxide と本リポの GPU crate は **NVIDIA GPU + CUDA Toolkit** を前提とする。
macOS は NVIDIA GPU を積めず CUDA Toolkit も提供されない (Apple Silicon /
Intel Mac いずれも) ため、macOS 単体では GPU crate (`gpu-runtime` / `bins/*`)
をビルドできない。macOS から開発する場合は NVIDIA GPU を持つリモート Linux
マシン (社内サーバや GPU クラウドインスタンス) 上でビルド・学習し、手元の
macOS は SSH / エディタとして使う。CPU-only crate (`shogi-format` /
`shogi-features` / `nnue-format` 等) 単体の編集と `cargo test -p <crate>` は
macOS でも可能だが、`cargo build` を workspace 全体に掛けると cuda-oxide 依存の
ビルドで失敗する。

## Windows (WSL2) の準備

native Windows での cuda-oxide ビルドは **upstream が公式に非サポート**。
cuda-oxide の installation ドキュメント (末尾「関連」のリンク) は "cuda-oxide
currently targets Linux only. Windows is not supported." と明記する (2026-05
時点、Windows 対応の issue / 作業も無い)。加えて cuda-oxide は rustc internal
ABI に直結する experimental backend で、本リポの `build.rs` も CUDA toolkit
root を Linux パス (`/usr/local/cuda` / `lib64/libcublas.so`) で解決する。
したがって GPU crate (`gpu-runtime` / `bins/*`) は Windows では
**WSL2 + Ubuntu** を使う。WSL2 からは NVIDIA GPU が CUDA 経由で見えるため、
WSL2 内では本ファイルの Linux 手順がそのまま通る (cuda-oxide が公式にテスト
しているのも Ubuntu 24.04)。

なお CPU-only crate (`shogi-format` / `shogi-features` / `gpu-kernels` /
`nnue-format` / `nnue-train`) は native Windows (MSVC toolchain) でも
`cargo test` がそのまま通る (2026-05 に Windows 11 + nightly-2026-04-03 で
242 tests green を確認)。GitHub Actions の CPU check と同じ範囲を手元の
Windows PowerShell で回せる:

```powershell
cargo test --workspace --exclude gpu-runtime --exclude progress-kpabs-train --exclude nnue-trainer
```

1. **Windows ホストに NVIDIA GPU ドライバを入れる。** WSL2 の CUDA は Windows
   側ドライバを使う仕組みで、WSL2 内に GPU ドライバを入れてはいけない。
2. **WSL2 と Ubuntu を入れる。** PowerShell (管理者) で:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

3. **WSL2 内で `nvidia-smi` が GPU を表示することを確認する。** 表示されない
   場合は Windows 側ドライバと WSL カーネル (`wsl --update`) を最新にする。
4. **WSL2 内に CUDA Toolkit を入れる。** WSL 用の toolkit パッケージを使い、
   ドライバ (`cuda-drivers`) は入れない (ドライバは Windows 側)。NVIDIA の
   "CUDA on WSL" 配布から `cuda-toolkit-12-x` のみを入れる。
5. これ以降は本ファイルの「システム install」からの手順を WSL2 のシェルで
   そのまま実行する。

WSL2 のディスク使用量には注意が要る (末尾「WSL2 ディスク注意」)。

## システム要件

| 項目 | 要件 | 備考 |
|---|---|---|
| OS | Linux / WSL2 (Windows) | 「対応 OS」参照 |
| CUDA Toolkit | 12.x (12.9 で確認) | nvcc, libNVVM, nvJitLink, **libcublas** |
| LLVM | **21+ (floor)、22 推奨** | apt.llvm.org が jammy / noble の両方に LLVM 20/21/22 を提供。`llc-22` が PATH にあれば cuda-oxide が優先する |
| Clang | **clang-21 or 22** + `libclang-common-{21,22}-dev` | `cuda-bindings` の bindgen に必要 (LLVM 22 にしても clang-21/22 のどちらかが要る) |
| Rust | nightly-2026-04-03 (cuda-oxide pinned) | `rust-toolchain.toml` で固定 |
| GPU | **公式: Ampere+ (sm_80+)**。Turing (sm_75) も `CUDA_OXIDE_TARGET=sm_75` で動作 | RTX 30/40/50, A100, H100, B200 等 |

## CUDA toolkit root の解決

`bins/nnue_train` は **libcublas** に dynamic link する (L1f weight backward を
`cublasSgemm_v2` で実行するため)。build.rs / runtime ともに以下の優先順で CUDA
toolkit root を探す:

1. `CUDA_TOOLKIT_PATH` env (build.rs 専用 legacy alias、最優先)
2. `CUDA_HOME` env (build / runtime 共通)
3. `CUDA_PATH` env (同上)
4. デフォルト path: `/usr/local/cuda` → `/usr/local/cuda-13.2` → `/usr/local/cuda-12.9` → `/opt/cuda`

`<root>/lib64/libcublas.so` が存在する最初の root を採用。標準パスに CUDA
Toolkit が入っていれば追加設定不要。非標準パスに置いている場合:

```bash
export CUDA_HOME=/path/to/cuda-12.9    # build.rs / runtime 両方が拾う
# または build.rs だけ向ける場合
export CUDA_TOOLKIT_PATH=/path/to/cuda-12.9
```

build.rs は `libcublas.so` が見つからなければ `cargo:warning` を出して
`/usr/local/cuda/lib64` を fallback で emit する (build 自体は失敗せず、最終的
に ld が `-lcublas` を解決できなければそこで止まる)。

> **LLVM 22 と atomics の syncscope**: cuda-oxide の `atomics` example README
> は「Atomic operations require llc-22 or newer for correct syncscope」と記載。
> LLVM 21 でも例題は完走するが、`memory_order` まわりの正確な PTX が必要な本番
> kernel では `llc-22` への昇格が望ましい。cuda-oxide の `cargo-oxide build`
> (Rust → `.ll`) は `llc-22` → `llc-21` の順で auto-discover する
> (`CUDA_OXIDE_LLC=/path` で固定可)。bin 起動時の `.ll`→`.ptx` 変換は別経路で、
> `LLC_BIN` 等の env を使う (「Smoke test」参照)。

## システム install

Linux / WSL2 共通。Ubuntu (apt) 前提:

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

Rust は rustup を使う。`rust-toolchain.toml` が nightly を pin しているので、
リポジトリ内で `cargo` を叩けば対応 toolchain (と `rust-src` / `rustc-dev`
component) が自動で入る。

## cuda-oxide のセットアップ

GPU kernel を PTX 化する cargo subcommand `cargo-oxide` を install する。
cuda-oxide repo を手動で clone する必要はない — upstream が `cargo install --git`
を公式サポートしており、`cargo-oxide` は初回実行時に codegen backend を自動で
取得・ビルドしてキャッシュする。

本リポジトリ用の wrapper スクリプトを使うのが簡単:

```bash
bash scripts/setup-cuda-oxide.sh
```

スクリプトは以下を行う:

- `Cargo.lock` が pin している cuda-oxide の rev を読む (library 側と codegen
  backend 側を同一 rev に揃えるため — rev ずれは backend ABI 不一致を招く)
- host 前提 (rustup / cargo / llc / clang / nvcc) の有無をチェックして報告
  (システムパッケージの install はしない)
- その rev で `cargo install --git ... cargo-oxide` を実行
- `cargo-oxide doctor` で環境を診断

スクリプトを使わず手動で入れる場合は、`Cargo.lock` の cuda-oxide rev に
合わせる:

```bash
rev=$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+' Cargo.lock | sed 's/.*rev=//')
cargo install --git https://github.com/NVlabs/cuda-oxide.git --rev "$rev" --force cargo-oxide
```

`~/.cargo/bin` を PATH に通しておくこと。スクリプトは毎回 pin rev で
`cargo-oxide` を入れ直すので、cuda-oxide の rev を bump したとき (library 側
`Cargo.toml` を更新したとき) も同じく `bash scripts/setup-cuda-oxide.sh` を
再実行すればよい。

## Smoke test

`cargo-oxide doctor` が全項目 ✓ なら host 側は OK。続いて実際の kernel を
ビルドする。リポジトリ root から:

```bash
bash scripts/build-kernels.sh
```

これは GPU の世代を `nvidia-smi` で判定し、kernel を持つ全 bin
(`nnue_train` / `progress_kpabs_train`) を `cargo-oxide build` でビルドする。
Ampere+ は既定 (sm_80 PTX、前方互換)、Turing (sm_75) は `CUDA_OXIDE_TARGET` を
自動設定するので、環境変数を手で打つ必要はない。

特定の bin だけビルドしたいときは手動でも可:

```bash
cd bins/nnue_train
cargo-oxide build
```

Ampere 以降 (sm_80+) はこれで OK。**Turing (sm_75) のみ**
`CUDA_OXIDE_TARGET=sm_75 cargo-oxide build` と前置する (下記「sub-Ampere GPU」)。

`cargo-oxide build` は `#[kernel]` を NVPTX IR (`.ll`) に compile する。bin は
起動時にこの `.ll` を libdevice と link して `.ptx` 化し CudaModule を load
する。`.ll`→`.ptx` には LLVM 21+ の `llvm-link` / `opt` / `llc` を使い、`-22`
→ `-21` を自動探索する (`LLVM_LINK_BIN` / `OPT_BIN` / `LLC_BIN` env で固定可)。
エラーなく学習が始まれば pipeline は通っている。

## sub-Ampere GPU (sm_75 Turing)

`cargo-oxide` 単独では auto-detect (`select_target()`) が kernel features から
target を選び、Basic フォールバックでは `sm_80` を選ぶ。`--arch=sm_75` を渡しても
auto-detect が override するため PTX header は `.target sm_80` のままになり、
Turing GPU では `CUDA_ERROR_INVALID_PTX` (driver error 218) で load が失敗する。

回避策は **`CUDA_OXIDE_TARGET=sm_75` 環境変数**。これは `select_target()` を
バイパスする一級 override で、`llc -mcpu=sm_75` までそのまま流れる:

```bash
cd bins/progress_kpabs_train
CUDA_OXIDE_TARGET=sm_75 cargo-oxide build
```

毎回打つのが面倒なら shell rc に export しておく。

### sm_75 の限界

`CUDA_OXIDE_TARGET=sm_75` で動くのは LLVM IR に sm_80+ 専用 op が含まれない
場合に限る:

- `cp.async` — asynchronous global → shared copy (sm_80+)
- `wgmma` — warpgroup matrix-multiply-accumulate (sm_90+ Hopper)
- `tcgen05` — 5th-gen tensor cores (sm_100+ Blackwell)
- `tma.*` — Tensor Memory Accelerator (sm_90+)
- `cluster.*` — Thread Block Cluster (sm_90+)

これらを含む IR を sm_75 PTX に compile すると `llc` か CUDA driver の JIT
load 段階で失敗する。KP-abs progress 系の単純な kernel (forward / grad scatter
/ adam_step / eval) は sm_75 の適用範囲内。fused optimizer step や async copy /
Hopper 専用 ops を使う kernel は sm_80+ GPU が要る。

`cargo-oxide build` が生成した `.ll` (build を実行した bin ディレクトリに出る)
を grep して sm_80+ op の混入を確認できる:

```bash
grep -E '(cp\.async|wgmma|tcgen05|tma\.|cluster\.)' \
  bins/progress_kpabs_train/progress_kpabs_train.ll
# (出力なし = OK)
```

## 実行のみのユーザー (kernel を改変しない場合)

bin は kernel module を `<name>.ll` → `<name>.cubin` → `<name>.ptx` の順で
探し、`.ll` があれば libdevice と link して `.ptx` に変換、無ければ既製の
`.cubin` / `.ptx` を直接 load する。探索先は bin の crate ディレクトリと
workspace root の両方。

このため、対象 GPU 向けの `.ptx` をあらかじめ持っていれば、それを bin の
crate ディレクトリか workspace root に置くだけで `cargo-oxide` も LLVM も
無しで kernel を load できる。`.ptx` は CUDA driver が JIT で SASS 化するので、
`sm_80` 向けに生成した `.ptx` は Ampere 以降 (sm_86/89/90/100…) で前方互換に
動く (Turing sm_75 は別途 sm_75 向け `.ptx` が要る)。

ただし `.ptx` は kernel ソースを変えると stale になり、`.gitignore` 済みで
git には含めない。現状この repo は pre-built `.ptx` を配布していないため、
kernel を改変しないユーザーも初回は `cargo-oxide build` が必要になる。

## サポート GPU マトリクス

| 世代 | sm | 代表的な GPU | 標準ビルドで動作 | `CUDA_OXIDE_TARGET=sm_XX` |
|---|---|---|---|---|
| Pascal | sm_60/61 | GTX 10xx, P100 | ✗ | 未検証 (LLVM IR 互換性も要確認) |
| Volta | sm_70 | V100, Titan V | ✗ | 動く可能性 (未検証) |
| Turing | sm_75 | RTX 2070 SUPER, GTX 16xx, T4 | ✗ | ✅ 確認済み |
| Ampere | sm_80 | A100, A30 | ✅ | n/a |
| Ampere | sm_86 | RTX 3080 Ti, RTX 30xx, A40, A10 | ✅ 確認済み (primary) | n/a |
| Ada | sm_89 | RTX 40xx | ✅ | n/a |
| Hopper | sm_90 | H100, H200 | ✅ | n/a |
| Blackwell | sm_100/120 | B100, B200, RTX 50xx | ✅ | n/a |

cuda-oxide の rev は本リポジトリの `Cargo.toml` (`[workspace.dependencies]`) に
pin し、`scripts/setup-cuda-oxide.sh` が `cargo-oxide` を同 rev に揃える。LLVM は
21 (sm_75) と 22 (sm_86) のいずれでも動作する。

## WSL2 ディスク注意

WSL2 環境では `/` (ext4) は **C: ドライブ上の sparse vhdx** が実体。`df -h /` の
表示は仮想容量で、物理は `df -h /mnt/c` の Avail に縛られる。数百 GB 級の
学習データ (PSV、checkpoint、ログ) と Rust の build artifact (`target/`) は
C: 圧迫を避けるため別ドライブに置くことを推奨する:

- 学習データは別ドライブの作業ディレクトリに置き、リポ内の `data/` から
  symlink を張る
- `CARGO_TARGET_DIR` を別ドライブのパスに向ける

## 関連

- [cuda-oxide adoption ADR](decisions/2026-05-09-cuda-oxide-adoption.md) —
  採用判断と Consequences
- [cuda-oxide upstream](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide-book installation requirements](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
- [cuda-oxide atomics example README (LLVM 22 syncscope の根拠)](https://github.com/NVlabs/cuda-oxide/blob/main/crates/rustc-codegen-cuda/examples/atomics/README.md)
