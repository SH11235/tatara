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
| Windows | WSL2 は既定 backend。CUDA C++ backend により native Windows も実験的に対応 | native は「Windows native (実験的)」、WSL2 は「Windows (WSL2) の準備」 |
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

## Windows native (実験的)

`native-cuda-host` feature は cuda-oxide を使わず、NVCC で build した CUDA C++
kernel を portable Rust CUDA Driver API runtime から
起動する。Windows 11、RTX 5090、driver 596.36、CUDA Toolkit 12.9.86、Visual
Studio 2022 (MSVC 19.44)、Rust nightly-2026-04-03 で build、GPU smoke、trainer の
1 step を確認済み。現時点では実験 backend であり、既定 backend は引き続き
Linux / WSL2 の cuda-oxide である。

### 前提の install

1. NVIDIA driver と **CUDA Toolkit 12.x** を Windows host に install する。driver
   だけでは `nvcc.exe`、`cuda.lib`、`cublas.lib` が無いため build できない。
2. Visual Studio 2022 または Build Tools 2022 で「C++ によるデスクトップ開発」を
   install する。通常の PowerShell で `cl.exe` が見えない場合は Developer
   PowerShell for VS 2022 を使う。
3. Rustup を install する。repository 内で最初に `cargo` を実行すると
   `rust-toolchain.toml` の pinned nightly が install される。
4. Smart App Control が有効な Windows 11 では、Cargo が生成する未署名の build
   script / test `.exe` が OS error 4551 で遮断される場合がある。継続的に native
   Rust 開発を行う開発機では Windows セキュリティの「アプリとブラウザー
   コントロール」→「Smart App Control」でオフにする。Microsoft Defender と
   メモリ整合性を無効にする必要はない。Smart App Control は一度オフにすると通常は
   Windows の reset / reinstall なしにオンへ戻せないため、WSL2 のみを使う場合は
   オンのままでよい。

新規 install 後は terminal を開き直す。開き直さず続行する場合は、Developer
PowerShell で `CUDA_PATH` と DLL 検索用 `PATH` を明示する:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
nvidia-smi
nvcc --version
cl
rustc -Vv
cargo -V
```

`CUDA_PATH\lib\x64\cuda.lib` と `cublas.lib` も存在することを確認する。
`CUDA_PATH` だけを設定して `PATH` に `CUDA_PATH\bin` を加えないと、build は通っても
trainer 起動時に cuBLAS DLL を解決できず `STATUS_DLL_NOT_FOUND (0xc0000135)` になる。

CUDA C++ kernel source は UTF-8 のコメントを含む。repository の build script は Windows
で NVCC の host compiler（MSVC）へ `/utf-8` を自動で渡す。古い commit や source を
NVCC で直接 compile するときに warning C4819 と、直後の定数について
`identifier ... is undefined` が同時に出る場合は、MSVC が source を CP932 として
誤解釈している。Developer PowerShell で `$env:CL = '/utf-8'` を設定して再実行する。

### build と smoke test

既定 feature を無効化し、必ず `native-cuda-host` だけを指定する:

```powershell
cargo tree -p nnue-trainer --no-default-features --features native-cuda-host |
  Select-String 'cuda-core|cuda-host|cuda-device'
# 出力が空であること

cargo build -p nnue-trainer --no-default-features --features native-cuda-host --release
cargo test -p cuda-native-runtime --features native-cuda --release -- --nocapture
cargo test -p nnue-trainer --no-default-features --features native-cuda-host --release
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple
```

最後のコマンドは教師データを使わず、nativeの対応範囲に限定したGPU smokeを実行する。
末尾に`[smoke/simple] PASSED`が出れば成功。

production CLI、dataloader、拡張 WRM、factorizer、全 FP16 経路、TF32、AdamW、
norm loss、2 種の checkpoint format を短い 1 run でまとめて確認する:

```powershell
$smokeOut = Join-Path ([System.IO.Path]::GetTempPath()) 'tatara-native-simple-cli'
cargo run -p nnue-trainer --no-default-features --features native-cuda-host --release -- simple `
  --data crates/shogi-format/tests/data/sample.psv --output $smokeOut --net-id native-simple-cli `
  --feature-set halfka-hm-merged --arch 8x2-8-8 --activation pairwise `
  --superbatches 1 --batches-per-superbatch 1 --batch-size 64 --threads 1 --save-rate 1 `
  --win-rate-model --scale 600 --wrm-nnue2score 600 `
  --loss-pow-exp 2.5 --loss-qp-asymmetry 0.2 `
  --loss-weight-boost-w1 1.5 --loss-weight-boost-w2 0.75 `
  --optimizer adamw --weight-decay 0.0001 `
  --norm-loss --norm-loss-factor 0.0001 --all-optim
Get-Item "$smokeOut/native-simple-cli-1.bin", "$smokeOut/native-simple-cli-1.ckpt"
```

正常終了し、両 file が空でないことを確認する。

同一の固定memory上fixtureでOS間throughputを比較する場合は次を実行する:

```powershell
cargo run -p nnue-trainer --release --no-default-features --features native-cuda-host -- `
  native-bench --architecture all --precision all
```

LayerStackとSimpleのFP32/all-optimを測り、各runと環境情報をJSONに保存する。固定profile、
WSL backend比較mode、統計、結果の扱いは[Native CUDA benchmark](native-cuda-benchmark.ja.md)
を参照。

現在の対応範囲は Simple (HalfKaHmMerged を含む) と LayerStack。Simpleは CReLU /
SCReLU / Pairwise と任意のhidden dimensionに対応する。LayerStackは可変層次元とbucket
mode、PSQT、feature factorizer、threat / effect featureに対応する。両architectureとも
FP32 / FP16 option/state (TF32 ON / OFF)、Sigmoid / WRM (拡張設定を含む)、norm loss、
Ranger / RAdam / AdamWを利用でき、各trainerが起動し得る全kernelを収録する。

## Windows (WSL2) の準備

native Windows での cuda-oxide ビルドは **upstream が公式に非サポート**。
cuda-oxide の installation ドキュメント (末尾「関連」のリンク) は "cuda-oxide
currently targets Linux only. Windows is not supported." と明記する。加えて
cuda-oxide は rustc internal
ABI に直結する experimental backend で、本リポの `build.rs` も CUDA toolkit
root を Linux パス (`/usr/local/cuda` / `lib64/libcublas.so`) で解決する。
したがって既定の cuda-oxide backend で GPU crate (`gpu-runtime` / `bins/*`) を
使う場合は **WSL2 + Ubuntu** を使う。WSL2 からは NVIDIA GPU が CUDA 経由で見えるため、
WSL2 内では本ファイルの Linux 手順がそのまま通る (cuda-oxide が公式にテスト
しているのも Ubuntu 24.04)。

なお CPU-only crate (`shogi-format` / `shogi-features` / `gpu-kernels` /
`nnue-format` / `nnue-train`) は native Windows (MSVC toolchain) でも
`cargo test` がそのまま通る。GitHub Actions の CPU check と同じ範囲を手元の
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

## システム要件

| 項目 | 要件 | 備考 |
|---|---|---|
| OS | Linux / WSL2、実験 backend は native Windows | 「対応 OS」参照 |
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

## Rust の install

ホストに `rustup` がまだ無ければ bootstrap する:

```bash
wget -qO- https://sh.rustup.rs | sh -s -- -y --default-toolchain none
. "$HOME/.cargo/env"   # 現シェルの PATH に ~/.cargo/bin を通す
```

`--default-toolchain none` はホスト既定 toolchain を選ばない指定。本リポの
`rust-toolchain.toml` が cuda-oxide 必須 nightly を pin しているので、
リポジトリ内で最初に `cargo` を叩いた時点で対応 toolchain と
`rust-src` / `rustc-dev` / `clippy` / `rust-analyzer` component が自動で入る:

```bash
cd /path/to/tatara
rustup show active-toolchain   # 初回はここで pin nightly の install が走る
```

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
- **codegen backend cache** (`~/.cargo/cuda-oxide/`) も同じ rev になっているか
  検証し、なっていなければ pin rev から作り直す (理由は後述の
  「トラブルシューティング」参照)
- `cargo-oxide doctor` で環境を診断

スクリプトを使わず手動で入れる場合は、`Cargo.lock` の cuda-oxide rev に
合わせる:

```bash
rev=$(grep -m1 -oE 'cuda-oxide\.git\?rev=[0-9a-f]+' Cargo.lock | sed 's/.*rev=//')
cargo install --git https://github.com/NVlabs/cuda-oxide.git --rev "$rev" --force cargo-oxide
```

`~/.cargo/bin` を PATH に通しておくこと。スクリプトは毎回 `cargo-oxide` と
backend cache の両方を pin rev に揃え直す (揃っていれば no-op) ので、
cuda-oxide の rev を bump したとき (library 側 `Cargo.toml` を更新したとき)
や、後述の backend cache の版ずれ問題に遭遇したときも、同じく
`bash scripts/setup-cuda-oxide.sh` を再実行すればよい。

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

## トラブルシューティング

### 他の人では再現しない device codegen エラー (cuda-oxide backend cache の版ずれ)

症状: `cargo-oxide build` が `rustc_codegen_cuda` 内で device codegen エラー
(例: "Unsupported construct" 系の翻訳失敗) を出して失敗するが、同じ source が
他の contributor や CI では問題なくビルドできる。OS / GPU / LLVM の構成も
特に他と変わらないように見える。

原因: `cargo-oxide` の codegen backend
(`~/.cargo/cuda-oxide/librustc_codegen_cuda.so`) は**一度きり、rev 指定なしで
fetch される**。ある機械で初めて何らかの `cargo-oxide` コマンドを実行した
瞬間の upstream cuda-oxide `main` HEAD を shallow clone してビルド・cache し、
以降は本リポジトリの `Cargo.lock` が pin する rev とは一切照合されない。
`cargo install --rev` で `cargo-oxide` の CLI 本体を pin rev に入れ直しても、
この backend cache には触れない — CLI バイナリと、それが駆動する backend
`.so` は別々の成果物だからだ。backend cache がたまたま pin rev から構造的に
乖離した `main` HEAD (cuda-oxide の MIR 翻訳層は頻繁に変わる) から作られて
いた場合、pin rev と正しく揃った環境では出ない翻訳エラーに遭遇しうる。

対応: `bash scripts/setup-cuda-oxide.sh` がこれを検知・修復するようになった。
pin rev + 手元の rustc nightly version を backend cache に stamp として
書き込み、次回実行時にこの stamp が食い違っていれば pin rev から cache を
作り直す (仕組みの詳細は
[該当 ADR](decisions/2026-07-04-cuda-oxide-backend-cache-pin.md) を参照)。
既に揃っていれば再実行しても no-op。それでも解消しない場合の最終手段として
`rm -rf ~/.cargo/cuda-oxide` してから再実行する。

### `cargo build` の ICE: "Missing SyntaxContext NN for crate alloc/core/std"

症状: `cargo build` が `rustc` 内で panic し、`Missing SyntaxContext NN for
crate "alloc"` (または `core` / `std`) が出る。query stack の末尾は
`Vec::from_parts` / `RawVecInner::with_capacity_in` のような stdlib symbol に
対する `type_of` / `associated_item`。同じ panic が `libloading` / `clap_lex`
等の素朴な依存にも再現する。原因は install 済 nightly toolchain の stdlib
`.rlib` / `.rmeta` 破損 — precompile された stdlib metadata 中の
`decode_syntax_context` 表のエントリが欠けており、metadata の他箇所がそこを
指したまま decode する。pin nightly を入れ直すと直る:

```bash
toolchain=$(grep -oP 'channel\s*=\s*"\K[^"]+' rust-toolchain.toml)
rustup toolchain uninstall "$toolchain"
rustup show active-toolchain   # rust-toolchain.toml を読んで pin nightly を再 install
cargo clean
cargo build
```

`rustup` は component 単位で content-addressed のため、reinstall は壊れた
`rust-std` / `rustc-dev` archive を取り直すだけで済む。`~/.cargo/registry` を
消したり toolchain を自前ビルドする必要は無い。

## 関連

- [cuda-oxide adoption ADR](decisions/2026-05-09-cuda-oxide-adoption.md) —
  採用判断と Consequences
- [cuda-oxide backend cache pin ADR](decisions/2026-07-04-cuda-oxide-backend-cache-pin.md) —
  `setup-cuda-oxide.sh` が codegen backend cache も検証する理由
- [cuda-oxide upstream](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide-book installation requirements](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
- [cuda-oxide atomics example README (LLVM 22 syncscope の根拠)](https://github.com/NVlabs/cuda-oxide/blob/main/crates/rustc-codegen-cuda/examples/atomics/README.md)
