[English](bench-pos.md) | **日本語**

# End-to-end学習benchmark

`nnue-train bench-pos`は、実PSV、production dataloader、progress bucket算出、選択したGPU
backendを使って学習throughputを測る。Rust runnerをLinux/WSLとnative Windowsで共用する。

`nnue-train native-bench`とは測定契約が異なる。`native-bench`は決定的なmemory上batchで
trainer/GPU throughputを分離して測る。`bench-pos`は実データを読み、dataloader、feature
抽出、bucket routing、学習を含むend-to-end変化の検出に使う。

## 設定

設定は2ファイルに分ける。

- [`bench-pos.toml`](../bench-pos.toml): git管理し、測定条件とcase matrixを固定する。
- `bench-pos.local.toml`: gitignore対象で、machine固有pathとhardware設定を置く。

初回にlocal設定を作る。

```sh
cp bench-pos.local.toml.example bench-pos.local.toml
```

PowerShellでも同じTOMLを使う。

```powershell
Copy-Item bench-pos.local.toml.example bench-pos.local.toml
```

`data`と`progress_coeff`を手元のfileへ向ける。相対pathはlocal config fileのdirectory
基準で解決する。`data_id`と`progress_id`はreportへ保存する安定した論理名で、絶対pathを
公開せずに複数machineが同じ入力を意図したか確認できる。reportにはbasenameとbyte sizeも
記録する。

Simple caseだけを選ぶ場合、`progress_coeff`と`progress_id`は省略できる。LayerStack caseを
選ぶ場合は必須。repositoryへsample `progress.bin`は添付せず、測定対象の学習設定で実際に
使う係数を指定する。

`threads`はdataloader worker数なのでmachine固有とする。`lock_gpu_clock = true`では
`nvidia-smi`を使ってsupported maximum graphics clockへの固定を試みる。権限不足などで
失敗した場合はwarningを表示し、clock lockなしで続行する。

## 固定standard profile

trackedな`standard-v1` profileはcaseごとに独立processを2回実行する。各processはbatch
size 65,536、200 batch × 5 superbatchで、file cacheとGPUのwarm-upにあたるsuperbatch 1を
除き、superbatch 2–5のmeanをrun値とする。熱・clock driftが常に同じcaseへ偏らないよう、
偶数runではcase順を反転する。

matrixは次の4 case。

- LayerStack factorized HalfKaHmMerged、FP32
- LayerStack factorized HalfKaHmMerged、all-optim
- Simple factorized HalfKP CReLU 256x2-32-32、FP32
- Simple factorized HalfKP CReLU 256x2-32-32、all-optim

standardの測定契約を変える場合はprofile名を更新する。machine固有path、thread数、clock
lock権限はtracked profileへ入れない。

## 実行

測定対象backendでbuildする。Linux/WSLのdefault buildはcuda-oxideを測る。

```sh
cargo build -p nnue-trainer --release
target/release/nnue-train bench-pos
```

native WindowsではCUDA C++ portable host backendをbuildする。

```powershell
cargo build -p nnue-trainer --release --no-default-features --features native-cuda-host
target\release\nnue-train.exe bench-pos
```

Linux/WSLでも同じportable host backendを測定できる。

```sh
cargo build -p nnue-trainer --release --no-default-features --features native-cuda-host
target/release/nnue-train bench-pos
```

`--case`は複数回指定でき、matrixの一部だけを実行できる。

```sh
target/release/nnue-train bench-pos --case layerstack-fp32
target/release/nnue-train bench-pos --case simple-halfkp-fp32 --case simple-halfkp-all-optim
```

非default fileは`--profile`と`--local-config`で指定する。dirty worktreeは既定で拒否し、
開発中だけ`--allow-dirty`で許可できる。parent CLIから継承されて表示される学習optionは
このsubcommandでは拒否する。測定・学習条件はtracked profileへ書き、host間で暗黙に
異ならないようにする。

## 結果

JSON reportとchild process logを`target/benchmark-results/bench-pos/`以下へ保存する。report
は[`bench-pos-v1.schema.json`](schemas/bench-pos-v1.schema.json)に従う。report
には全superbatch値、各runの測定mean、run間のmean/median/sample SD/min/max/CV、machine
pathをredactした完全な学習引数、入力identity、commit/dirty、backend feature、GPU、
driver、CUDA toolkit、rustc、OSを記録する。

build時間はbenchmark外。superbatchの`pos/s`はproduction training loopの値で、実データ
読み込みとfeature/bucket処理を含む。各child processの全wall timeは別項目として記録する。
