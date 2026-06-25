#![feature(f16)]
//! `bins/nnue_train` binary entry point — NNUE trainer。
//!
//! 本 file は bin entry point (`fn main`) と module 宣言を持つ。`#[kernel]` device
//! 関数は [`kernels`] module、host 側コード (kernel loader / checkpoint format /
//! trainer / CLI / smoke test) と GPU↔CPU 同等性テストは各 sibling module に置く。

use clap::Parser;

// ===========================================================================
// module 宣言
//
// `#[kernel]` device 関数は `kernels` module に置く。cuda-oxide は bin crate 内に
// 置かれた `#[kernel]` のみ NVPTX 化するため別 crate には出せないが、bin crate 内の
// submodule なら問題ない。host 側コード (kernel loader / checkpoint format /
// trainer / CLI / smoke) と GPU↔CPU 同等性テストも sibling module に分割する。
// ===========================================================================

mod arch;
mod ckpt;
mod cli;
mod kernel_module;
mod kernels;
mod smoke;
mod threat_ablate;
mod trainer_common;
mod trainer_layerstack;
mod trainer_simple;
mod training;

#[cfg(test)]
mod tests;

use cli::Cli;
use smoke::smoke_test;
use training::run_training;
// `cuda_launch!` 呼出側 (trainer / smoke / tests) は `use crate::*;` で `#[kernel]`
// marker 型 (`__<name>_CudaKernel`) を解決する。`kernels` module の marker を crate
// root から見えるよう re-export する。
pub(crate) use kernels::*;

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = if cli.data.is_some() {
        run_training(&cli)
    } else {
        smoke_test(cli.arch.kind())
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::from(1)
        }
    }
}
