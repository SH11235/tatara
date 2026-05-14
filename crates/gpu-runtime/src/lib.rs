//! cuda-oxide host 側 API の薄い wrapper。
//!
//! GPU カーネルは cuda-oxide で書く (docs/decisions/ 参照)。`cuda-core` と
//! `cuda-host` の主要 type を再 export しつつ、`Error` で `DriverError` /
//! `LtoirError` を `thiserror` 経由でラップする。
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

pub use cuda_core::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError,
    LaunchConfig,
};
pub use cuda_host::{LtoirError, cuda_launch};

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
}

/// `Result<T, gpu_runtime::Error>` の alias。
pub type Result<T> = std::result::Result<T, Error>;
