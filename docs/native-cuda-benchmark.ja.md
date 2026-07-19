# Native CUDA benchmark

`nnue-train native-bench`は、CUDA C++ backend用の再現可能な固定fixture throughput
benchmarkである。benchmark条件、計時、run順序、統計、環境情報収集、JSON出力をRustに
集約し、Linux/WSLとnative Windowsで同じ実装を使う。

`nnue-train bench-pos`は実データを使ったend-to-end学習を測る。一方`native-bench`は
決定的なmemory上batchでtrainer/GPU throughputを測り、trainer構築とbatch生成は計時に
含めない。

## 固定v1 profile

既定値はbatch 16,384、warm-up 3 step、計測100 step、独立3 run。fixture既定値を変更
するときは新しいprofile versionを追加する。

| Architecture | Fixture |
|---|---|
| LayerStack | `layerstack-halfka-hm-merged-factorized-v1`: factorized HalfKaHmMerged、FT 1536、L1 16、L2 32、Progress8KpAbs trainerと同じ9-bucket形状 |
| Simple | `simple-halfkp-factorized-v1`: factorized HalfKP、CReLU、FT 256、L1 32、L2 32 |

既定では各architectureについて次の2 precisionを測る。

- `fp32`: precision flagをすべてOFF。
- `all-optim`: TF32、FT weight/output FP16、optimizer state FP16をON。

precisionの実行順はrunごとに反転する。`compare` modeではcuda-oxideとCUDA C++の順序も
runごとに反転し、先に実行した側の偏りを一方のbackendへ固定しない。

LayerStack benchmarkは`progress.bin`を読まない。memory上batchのbucket indexを
round-robin（`row % 9`）で決定的に設定し、trainer/GPU throughputだけを測る。progress
算出と実データ読み込みまで含める場合は`nnue-train bench-pos`を使う。

## 実行方法

WindowsとLinux/WSLで使えるCUDA C++単体計測:

```sh
cargo run -p nnue-trainer --release --no-default-features --features native-cuda-host -- \
  native-bench --architecture all --precision all
```

PowerShellでは次のように実行する。

```powershell
cargo run -p nnue-trainer --release --no-default-features --features native-cuda-host -- `
  native-bench --architecture all --precision all
```

Linux/WSLで使えるcuda-oxide対CUDA C++比較:

```sh
bash scripts/build-kernels.sh
cargo run -p nnue-trainer --release --features native-cuda -- \
  native-bench --mode compare --architecture all --precision all
```

部分実行には`--architecture layerstack|simple`と`--precision fp32|all-optim`を使う。
`--batch-size`、`--warmup-steps`、`--steps`、`--runs`で条件を変更でき、その場合JSONの
`parameters.customized`は`true`になる。

WindowsでC++ build toolが`PATH`にない場合はVisual Studio Developer PowerShellから
実行する。`nvcc`を含むCUDA toolkitも必要。

Windowsでwarning C4819に続いて`identifier ... is undefined`が出る場合は、NVCCの
host compilerがUTF-8のkernel sourceを既定code pageとして誤解釈している。現行の
build scriptはMSVCへ`/utf-8`を自動で渡す。古いcommitを測る場合はDeveloper
PowerShellで`$env:CL = '/utf-8'`を設定してから同じbenchmark commandを実行する。

## 結果

既定の出力先は`target/benchmark-results/native-cuda/`で、
[`native-cuda-benchmark-v1.schema.json`](schemas/native-cuda-benchmark-v1.schema.json)に従う。
各run、mean/median/sample SD/min/max、backendのpaired delta、all-optim/FP32 speedup、
展開済みprecision flag、commit/dirty、platform（Windows/WSL/Linux）、GPU、driver、CUDA toolkit、rustc、Cargo
feature、完全なcommand lineを記録する。

通常のCPU-only CIではversioned schemaをcompileし、repository内の代表fixtureを検証する。
native feature testでは実際のRust report型もserializeし、同じschemaへ通す。

既定ではdirty worktreeを拒否する。開発中は`--allow-dirty`を使えるが、JSONには
`dirty: true`が残る。raw reportはgitignore済みの`target/`に置き、merge/release判断に
採用する代表結果だけをtatara commitと実行command付きで`rshogi-notes`へコピーまたは
要約する。
