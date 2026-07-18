# Native CUDA C++ backendを並行提供する

- **Status**: Accepted
- **Date**: 2026-07-18

## Context

GPU kernelはcuda-oxideで実装されている。Rustだけでhost/deviceを記述できる一方、
cuda-oxideはLinuxだけを対象とするため、Windows利用者はWSL2環境を用意する必要がある。
nightly Rust、LLVM、Clang、cuda-oxide codegen backendの組合せも導入時の障壁になる。

CUDA ToolkitのNVCC、NVRTC、CUDA Driver API、cuBLASはWindowsとLinuxの両方を対象とする。
また、NNUE trainerのarchitectureは既知のkernel集合で表現でき、runtimeに計算graphから
kernel sourceを生成する必要はない。

現在のtrainerはcuda-oxideのhost型とlaunch macroを直接利用しているため、device source
だけをCUDA C++へ翻訳してもWindowsでは動かない。buffer、stream、event、module load、
kernel launchをOS非依存のruntime境界へ分離する必要がある。

## Decision

CUDA C++で記述したkernelをNVCCでfat binaryへcompileし、RustのCUDA Driver API runtime
からロードするnative backendを追加する。既存cuda-oxide backendは数値・性能比較の
referenceとして並行提供する。

native backendのhost処理はRustに置く。C++共有libraryにtrainerやallocationの所有権を
渡さず、CUDA C++はdevice kernelだけに限定する。これによりOS間でC++ host ABIを持たず、
checkpoint、dataloader、schedule、trainer orchestrationを両backendで共有する。

fat binaryはrelease buildで生成して実行fileへ埋め込める構造にする。利用者環境での
runtime compileを必須にしない。source buildではNVCCを使用する。

実装順は、WSL上で既存host pipelineとcuBLASを維持したままcuda-oxide互換のkernel ABIへ
CUDA C++ fat binaryを差し込み、device側の数値・性能parityを先に確立する。その間に
Driver APIのportable runtimeを独立して整備し、kernel coverageの完成後にtrainerのhost型を
置き換える。最後に同じruntime境界をnative Windowsでbuild・実機検証する。

## Consequences

- Windows native trainerをcuda-oxideのWindows対応から独立して実装できる。
- Linux/WSL上で同一GPUを使い、compiler/backendだけを変えた数値・性能比較ができる。
- CUDA C++とRustの二言語を保守し、kernel ABIの一致をtestで固定する必要がある。
  「launch する全symbolがsource exportにある」検査はsource走査だけで動くが、
  「全source exportがfatbinから解決できる」最終検査はCUDA driverを要するため、
  GitHub-hosted runnerでは走らず`scripts/local-ci.sh` / `scripts/check-native-cuda-parity.sh`
  をローカルGPUで実行して初めて確認できる。
- backend parityの契約は各演算の許容誤差内での数値一致であり、異なるGPU世代を跨ぐ
  bit一致ではない。同一GPU上のhost runtime比較では追加の強い回帰検査としてbit fingerprintも使う。
- CUDA C++化だけでは高速化を保証しない。既存throughputを維持することを移植時の基準とし、
  NVIDIA固有intrinsicやlibraryによる最適化はparity確立後に個別計測する。
- native backendが全kernelを実装するまでは、対応architectureとprecision optionを明示して
  unsupported構成を起動前に拒否する。

## Backend feature構成

- 既定buildは`cuda-oxide` featureを使い、従来のdevice/host実装を維持する。
- `native-cuda`はcuda-oxide hostからCUDA C++ fat binaryを起動する比較用構成である。
- `native-cuda-host`はCUDA C++ fat binaryとportable Driver API host runtimeだけを使う。
  Windowsへ持ち込む対象はこの構成であり、次のようにcuda-oxide依存を無効化してbuildする。

```bash
cargo build -p nnue-trainer \
  --no-default-features --features native-cuda-host --release
```

`native-cuda-host`で現在対応するのは、HalfKaHmMergedを含むSimpleとLayerStackである。
SimpleはCReLU / SCReLU / Pairwiseと任意のhidden dimension、LayerStackは可変層次元と
bucket mode、PSQT、feature factorizer、threat / effect featureを扱う。両architectureで
FP32 / FP16 option/state（TF32有効または無効）、factorizer、Sigmoid / WRM（拡張設定を
含む）、norm loss、Ranger / RAdam / AdamWを利用でき、起動し得る全kernelを収録する。
cuBLASはCUDA ToolkitのC APIを直接呼び、stream
handleだけを共通runtimeから受け取るためcuda-oxideの型には依存しない。
