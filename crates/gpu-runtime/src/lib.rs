//! cuda-oxide host 側 API の薄い wrapper。
//!
//! GPU カーネルは cuda-oxide で書く (docs/decisions/ 参照)。`cuda-core` と
//! `cuda-host` の主要 type を再 export しつつ、`Error` で `DriverError` /
//! `LtoirError` を `thiserror` 経由でラップする。kernel artifact の探索と
//! `.ll`→`.ptx` 変換は [`kernel_loader`] に置く。
//!
//! ## 設計方針
//!
//! 「薄く」をモットーに、cuda-oxide の type-safe API を再発明せず素直に
//! 透過する。命名 alias (`DeviceAlloc`, `Stream`) は **type alias** で提供し、
//! cuda-oxide 側の名前 (`DeviceBuffer`, `CudaStream`) も並行して公開する。
//!
//! `KernelLauncher` 相当は **新規 struct を作らず**、cuda-oxide が提供する
//! `cuda_launch!` macro (本 crate からも再 export) をそのまま使う方針。
//! cuda-oxide 設計の中心が macro での type-safe launch にあり、これを上から
//! 薄く wrap してもメリットが出ないため。raw な launch が要る場合は
//! `cuda_core::launch_kernel_on_stream` (unsafe) を直接呼ぶ。
//!
//! ## 再 export しないもの
//!
//! - `cuda_host::cuda_launch_async` — マクロ展開先で `cuda_async::*` を要求
//!   するが、本 crate は `cuda-async` を dep にしていない。非同期 launch が
//!   必要になった段階で `cuda-async` 込みで再公開する。

pub mod kernel_loader;

pub use cuda_core::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError,
    LaunchConfig,
};
pub use cuda_host::{LtoirError, cuda_launch};
pub use kernel_loader::{BLOCK_DIM, grid_dim_1d, load_kernel_module_with_fallback};

/// `cuda_host::load_kernel_module` の再 export。
///
/// **NOTE**: cuda-oxide 内部実装は呼び出し元 crate の `CARGO_MANIFEST_DIR`
/// (run-time 解決) を起点に `<name>.cubin` / `.ptx` / `.ll` を順に探索する
/// ため、本 helper を呼んだ「呼び出し元 crate の dir」が起点になる
/// (`gpu-runtime` 自身ではない)。kernel artifact を自 crate に同梱するケース
/// ではそのまま使える。任意 path から PTX を読みたい場合は
/// `CudaContext::load_module_from_file(path)` を直接使うこと。
pub use cuda_host::load_kernel_module;

/// `DeviceBuffer<T>` の短縮名 alias。
///
/// `gpu_runtime::DeviceAlloc<T>` でも `gpu_runtime::DeviceBuffer<T>` でも同じ。
pub type DeviceAlloc<T> = cuda_core::DeviceBuffer<T>;

/// `CudaStream` の短縮名 alias。
pub type Stream = cuda_core::CudaStream;

/// gpu-runtime の error。
///
/// cuda-oxide 由来の `DriverError` (driver API), `LtoirError` (PTX/cubin
/// load) をそれぞれ `#[from]` で吸収する。`gpu_runtime::Result<T>` を返す
/// 関数の中で両方を `?` で扱える。
///
/// 将来、本 crate 固有の前提条件違反 (e.g. zero-sized allocation) や
/// kernel launch failure の独自分類はここに variant を増やす想定。
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] DriverError),
    #[error(transparent)]
    Ltoir(#[from] LtoirError),
    /// kernel artifact の探索 / `.ll`→`.ptx` 変換の失敗 (`kernel_loader` 参照)。
    #[error("{0}")]
    KernelArtifact(String),
}

/// `Result<T, gpu_runtime::Error>` の alias。
pub type Result<T> = std::result::Result<T, Error>;

/// `err` が CUDA out-of-memory (`CUDA_ERROR_OUT_OF_MEMORY`) を表すか判定する。
///
/// `DeviceBuffer` 確保失敗は `DriverError` として伝播する。本 crate の
/// [`Error::Cuda`] 経由でも、呼び出し側が `Box<dyn Error>` に直接 box した
/// `DriverError` でも検出できるよう `&dyn Error` を受ける。判定は driver の
/// `cuGetErrorName` ([`DriverError::error_name`]) が返す symbolic name で行い、
/// `cuda_bindings` の `CUresult` 内部表現に依存しない。OOM 以外では true を返さない。
pub fn is_out_of_memory(err: &(dyn std::error::Error + 'static)) -> bool {
    if let Some(e) = err.downcast_ref::<DriverError>() {
        return driver_error_is_out_of_memory(e);
    }
    if let Some(Error::Cuda(e)) = err.downcast_ref::<Error>() {
        return driver_error_is_out_of_memory(e);
    }
    false
}

fn driver_error_is_out_of_memory(e: &DriverError) -> bool {
    e.error_name()
        .map(|name| name.to_bytes() == b"CUDA_ERROR_OUT_OF_MEMORY")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_cuda_error_is_not_out_of_memory() {
        // CUDA 由来でない error は OOM 判定しない (false positive ゼロ)。
        let io_err = std::io::Error::other("disk full");
        assert!(!is_out_of_memory(&io_err));
        assert!(!is_out_of_memory(&Error::KernelArtifact(
            "missing .ptx".into()
        )));
    }
}
