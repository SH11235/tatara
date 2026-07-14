#![cfg_attr(feature = "gpu", feature(f16))]
//! `bins/nnue_train` binary entry point — NNUE trainer。
//!
//! 本 file は bin entry point (`fn main`) と module 宣言を持つ。`#[kernel]` device
//! 関数は [`kernels`] module、host 側コード (kernel loader / checkpoint format /
//! trainer / CLI / smoke test) と GPU↔CPU 同等性テストは各 sibling module に置く。

#[cfg(feature = "gpu")]
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
#[cfg(any(feature = "gpu", test))]
mod ckpt;
mod cli;
#[cfg(feature = "gpu")]
mod ft_factorize_host;
#[cfg(feature = "gpu")]
mod kernel_module;
#[cfg(feature = "gpu")]
mod kernels;
#[cfg(feature = "gpu")]
mod smoke;
#[cfg(feature = "gpu")]
mod threat_ablate;
#[cfg(feature = "gpu")]
mod trainer_common;
#[cfg(feature = "gpu")]
mod trainer_layerstack;
#[cfg(feature = "gpu")]
mod trainer_simple;
mod training;

#[cfg(test)]
mod tests;

#[cfg(feature = "gpu")]
use cli::Cli;
#[cfg(feature = "gpu")]
use smoke::smoke_test;
#[cfg(feature = "gpu")]
use training::run_training;
// `cuda_launch!` 呼出側 (trainer / smoke / tests) は `use crate::*;` で `#[kernel]`
// marker 型 (`__<name>_CudaKernel`) を解決する。`kernels` module の marker を crate
// root から見えるよう re-export する。
#[cfg(feature = "gpu")]
pub(crate) use kernels::*;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    // 診断 flag (--eval-only / --threat-ablate / --threat-norm-dump) は学習データを
    // 読まない経路 (norm-dump / --test-data 評価) を持つため、--data 不在でも
    // run_training に dispatch する。--data の有無だけで分けると、これらを指定しても
    // smoke test に落ちて何もせず成功扱いになる。
    let result = if cli.data.is_some()
        || cli.eval_only
        || cli.threat_ablate.is_some()
        || cli.threat_norm_dump
    {
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

#[cfg(not(feature = "gpu"))]
fn main() -> std::process::ExitCode {
    eprintln!("nnue-train requires the default `gpu` feature");
    std::process::ExitCode::from(1)
}
