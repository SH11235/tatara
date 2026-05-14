//! cuda-oxide host 側 API の薄い wrapper。
//!
//! 本リポは GPU カーネルを cuda-oxide で書く (ADR-0003)。`cuda-core` と
//! `cuda-host` を git dep で取り込み (rev `6de0509`、Stage 0-1 で動作確認済)、
//! 主要 type を再 export しつつ、`Error` で `DriverError` / `LtoirError` を
//! `thiserror` 経由でラップする。
//!
//! ## 設計方針
//!
//! 「薄く」をモットーに、cuda-oxide の type-safe API を再発明せず素直に
//! 透過する。Issue #7 の命名 (`DeviceAlloc`, `Stream`) は **type alias** で
//! 提供し、cuda-oxide 側の名前 (`DeviceBuffer`, `CudaStream`) も並行して
//! 公開する。
//!
//! Issue #7 が言及している `KernelLauncher` 相当は **新規 struct を作らず**、
//! cuda-oxide が提供する `cuda_launch!` macro (本 crate からも再 export) を
//! そのまま使う方針。cuda-oxide 設計の中心が macro での type-safe launch
//! にあり、これを上から薄く wrap してもメリットが出ないため。raw な launch
//! が要る場合は `cuda_core::launch_kernel_on_stream` (unsafe) を直接呼ぶ。
//!
//! 将来 Stage 2+ で hand-fused kernel suite を作る段階で、複数 stream の
//! オーケストレーション、pinned memory、async copy 等を本 crate に集約する
//! 想定。今は最小。
//!
//! ## 再 export しないもの
//!
//! - `cuda_host::cuda_launch_async` — マクロ展開先で `cuda_async::*` を要求
//!   するが、本 crate は `cuda-async` を dep にしていない (今は最小方針)。
//!   非同期 launch が要る Stage 2+ で `cuda-async` 込みで再公開する。

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
/// (`gpu-runtime` 自身ではない)。kernel artifact を呼び出し元 crate に同梱する
/// 構成ではそのまま使える。任意 path から PTX を読みたい場合は
/// `CudaContext::load_module_from_file(path)` を直接使うこと。
pub use cuda_host::load_kernel_module;

/// `DeviceBuffer<T>` の Issue 命名 alias。
///
/// `gpu_runtime::DeviceAlloc<T>` でも `gpu_runtime::DeviceBuffer<T>` でも同じ。
pub type DeviceAlloc<T> = cuda_core::DeviceBuffer<T>;

/// `CudaStream` の Issue 命名 alias。
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
