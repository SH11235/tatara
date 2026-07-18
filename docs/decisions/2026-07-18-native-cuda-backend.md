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
- CUDA C++化だけでは高速化を保証しない。既存throughputを維持することを移植時の基準とし、
  NVIDIA固有intrinsicやlibraryによる最適化はparity確立後に個別計測する。
- native backendが全kernelを実装するまでは、対応architectureとprecision optionを明示して
  unsupported構成を起動前に拒否する。
