//! GPU host runtime の薄い wrapper。backend は feature で排他選択する。
//! backend feature なしでは backend 依存 API を全て持たない空の stub として
//! コンパイルされる (workspace 一括の `--no-default-features` lint を通すため)。
//!
//! - `native`: CUDA C++ kernel を CUDA Driver API から launch する portable runtime
//! - `oxide`: cuda-oxide host API (`cuda-core` / `cuda-host`) を再 export し、
//!   `Error` で `DriverError` / `LtoirError` を `thiserror` 経由でラップする。
//!   kernel artifact の探索と `.ll`→`.ptx` 変換は [`kernel_loader`] に置く
//!
//! どちらの backend も同じ型名 (`CudaContext` / `DeviceBuffer` / `CudaStream` 等) と
//! `BLOCK_DIM` / `grid_dim_1d` を公開するため、上位 crate は backend を意識せずに書ける。
//!
//! ## 設計方針
//!
//! 「薄く」をモットーに、backend が提供する type-safe API を再発明せず素直に
//! 透過する。`oxide` では命名 alias (`DeviceAlloc`, `Stream`) を **type alias** で
//! 提供し、cuda-oxide 側の名前 (`DeviceBuffer`, `CudaStream`) も並行して公開する。
//!
//! `KernelLauncher` 相当は **新規 struct を作らず**、cuda-oxide が提供する
//! `cuda_launch!` macro を error context だけ付与する薄い macro で包む方針。
//! raw な launch が要る場合は
//! `cuda_core::launch_kernel_on_stream` (unsafe) を直接呼ぶ。
//!
//! ## 再 export しないもの
//!
//! - `cuda_host::cuda_launch_async` — マクロ展開先で `cuda_async::*` を要求
//!   するが、本 crate は `cuda-async` を dep にしていない。非同期 launch が
//!   必要になった段階で `cuda-async` 込みで再公開する。

#[cfg(all(feature = "oxide", feature = "native"))]
compile_error!(
    "gpu-runtime backends are mutually exclusive; inspect the dependency feature graph to find \
     what enables `oxide` and `native`"
);

#[cfg(feature = "oxide")]
pub mod kernel_loader;
#[cfg(feature = "native")]
mod native_backend;

#[cfg(feature = "oxide")]
pub use cuda_core::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, DeviceBuffer, DriverError,
    LaunchConfig,
};
#[cfg(feature = "oxide")]
pub use cuda_host::LtoirError;
#[cfg(feature = "oxide")]
pub use kernel_loader::{BLOCK_DIM, grid_dim_1d, load_kernel_module_with_fallback};
#[cfg(feature = "native")]
pub use native_backend::{
    CudaContext, CudaEvent, CudaModule, CudaStream, DeviceBuffer, KernelArgs, LaunchConfig,
};

#[cfg(feature = "native")]
pub const BLOCK_DIM: u32 = 256;

#[cfg(feature = "native")]
pub fn grid_dim_1d(n: usize, block: u32) -> (u32, u32, u32) {
    (((n as u32).max(1)).div_ceil(block), 1, 1)
}

#[doc(hidden)]
#[cfg(feature = "oxide")]
pub use cuda_host as __cuda_host;

/// CUDA kernel を起動し、失敗時に kernel 名を付与する。
///
/// 成功時は cuda-oxide の `cuda_launch!` が返す値をそのまま通す。kernel 名は
/// compile-time の `stringify!` で得るため、error が発生するまで allocation や
/// formatting は行わない。
/// 下層 macro は field 順不同だが、この wrapper の arm は `kernel:` が先頭の launch
/// のみ受理する。順序を変えた launch は分かりにくい macro error になるため先頭に書く。
#[macro_export]
#[cfg(feature = "oxide")]
macro_rules! cuda_launch {
    (kernel: $kernel:path, $($rest:tt)*) => {
        $crate::__cuda_host::cuda_launch! {
            kernel: $kernel,
            $($rest)*
        }
        .map_err(|source| $crate::Error::KernelLaunch {
            kernel: stringify!($kernel),
            source,
        })
    };
}

#[doc(hidden)]
#[macro_export]
#[cfg(feature = "native")]
macro_rules! __native_cuda_push_args {
    ($args:ident;) => {};
    ($args:ident; slice($buffer:expr) $(, $($rest:tt)*)?) => {{
        $args.push_slice(&$buffer);
        $crate::__native_cuda_push_args!($args; $($($rest)*)?);
    }};
    ($args:ident; slice_mut($buffer:expr) $(, $($rest:tt)*)?) => {{
        $args.push_slice(&$buffer);
        $crate::__native_cuda_push_args!($args; $($($rest)*)?);
    }};
    ($args:ident; $value:expr $(, $($rest:tt)*)?) => {{
        $args.push_scalar($value);
        $crate::__native_cuda_push_args!($args; $($($rest)*)?);
    }};
}

#[macro_export]
#[cfg(feature = "native")]
macro_rules! cuda_launch {
    (
        kernel: $kernel:path,
        stream: $stream:expr,
        module: $module:expr,
        config: $config:expr,
        args: [$($args:tt)*]
    ) => {{
        let mut __args = $crate::KernelArgs::new();
        $crate::__native_cuda_push_args!(__args; $($args)*);
        $module
            .launch(stringify!($kernel), &$stream, $config, &mut __args)
            .map_err(|source| $crate::Error::KernelArtifact(format!(
                "CUDA kernel launch `{}` failed: {source}", stringify!($kernel)
            )))
    }};
}

/// `cuda_host::load_kernel_module` の再 export。
///
/// **NOTE**: cuda-oxide 内部実装は呼び出し元 crate の `CARGO_MANIFEST_DIR`
/// (run-time 解決) を起点に `<name>.cubin` / `.ptx` / `.ll` を順に探索する
/// ため、本 helper を呼んだ「呼び出し元 crate の dir」が起点になる
/// (`gpu-runtime` 自身ではない)。kernel artifact を自 crate に同梱するケース
/// ではそのまま使える。任意 path から PTX を読みたい場合は
/// `CudaContext::load_module_from_file(path)` を直接使うこと。
#[cfg(feature = "oxide")]
pub use cuda_host::load_kernel_module;

/// `DeviceBuffer<T>` の短縮名 alias。
///
/// `gpu_runtime::DeviceAlloc<T>` でも `gpu_runtime::DeviceBuffer<T>` でも同じ。
#[cfg(feature = "oxide")]
pub type DeviceAlloc<T> = DeviceBuffer<T>;
#[cfg(feature = "native")]
pub type DeviceAlloc<T> = DeviceBuffer<T>;

/// `CudaStream` の短縮名 alias。
#[cfg(any(feature = "oxide", feature = "native"))]
pub type Stream = CudaStream;

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
    #[cfg(feature = "oxide")]
    #[error(transparent)]
    Cuda(#[from] DriverError),
    #[cfg(feature = "native")]
    #[error(transparent)]
    NativeCuda(#[from] cuda_native_runtime::NativeCudaError),
    #[cfg(feature = "oxide")]
    #[error(transparent)]
    Ltoir(#[from] LtoirError),
    /// CUDA kernel launch の失敗。
    #[error("CUDA kernel launch `{kernel}` failed: {source}")]
    #[cfg(feature = "oxide")]
    KernelLaunch {
        kernel: &'static str,
        #[source]
        source: DriverError,
    },
    /// kernel artifact の探索 / `.ll`→`.ptx` 変換の失敗 (`kernel_loader` 参照)。
    #[error("{0}")]
    KernelArtifact(String),
}

/// `Result<T, gpu_runtime::Error>` の alias。
pub type Result<T> = std::result::Result<T, Error>;

/// Creates the trainer's compute stream while preserving backend-specific ordering semantics.
/// cuda-oxide uses its context default stream; the native backend returns a fallible non-blocking
/// stream so allocation failures propagate through trainer construction instead of panicking.
#[cfg(any(feature = "oxide", feature = "native"))]
pub fn create_compute_stream(
    ctx: &std::sync::Arc<CudaContext>,
) -> Result<std::sync::Arc<CudaStream>> {
    #[cfg(feature = "oxide")]
    return Ok(ctx.default_stream());
    #[cfg(feature = "native")]
    ctx.new_stream()
}

/// `err` が CUDA out-of-memory (`CUDA_ERROR_OUT_OF_MEMORY`) を表すか判定する。
///
/// `DeviceBuffer` 確保失敗は `DriverError` として伝播する。本 crate の
/// [`Error::Cuda`] 経由でも、呼び出し側が `Box<dyn Error>` に直接 box した
/// `DriverError` でも検出できるよう `&dyn Error` を受ける。判定は driver の
/// `cuGetErrorName` ([`DriverError::error_name`]) が返す symbolic name で行い、
/// `cuda_bindings` の `CUresult` 内部表現に依存しない。OOM 以外では true を返さない。
#[cfg(feature = "oxide")]
pub fn is_out_of_memory(err: &(dyn std::error::Error + 'static)) -> bool {
    if let Some(e) = err.downcast_ref::<DriverError>() {
        return driver_error_is_out_of_memory(e);
    }
    if let Some(Error::Cuda(e)) = err.downcast_ref::<Error>() {
        return driver_error_is_out_of_memory(e);
    }
    if let Some(Error::KernelLaunch { source, .. }) = err.downcast_ref::<Error>() {
        return driver_error_is_out_of_memory(source);
    }
    false
}

#[cfg(feature = "oxide")]
fn driver_error_is_out_of_memory(e: &DriverError) -> bool {
    e.error_name()
        .map(|name| name.to_bytes() == b"CUDA_ERROR_OUT_OF_MEMORY")
        .unwrap_or(false)
}

#[cfg(feature = "native")]
pub use native_backend::is_out_of_memory;

/// Fills a device allocation byte-wise on `stream`.
#[cfg(any(feature = "oxide", feature = "native"))]
pub fn memset_d8_async<T: Copy>(
    buffer: &DeviceBuffer<T>,
    value: u8,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "oxide")]
    // SAFETY: buffer owns exactly num_bytes and stream/context compatibility is caller-owned.
    unsafe {
        cuda_core::memory::memset_d8_async(
            buffer.cu_deviceptr(),
            value,
            buffer.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    #[cfg(feature = "native")]
    buffer.fill_byte_async(value, stream)?;
    Ok(())
}

/// Enqueues an H2D copy.
///
/// # Safety
///
/// The caller must keep `values` alive and immutable until stream completion.
#[cfg(any(feature = "oxide", feature = "native"))]
pub unsafe fn memcpy_htod_async<T: Copy>(
    buffer: &DeviceBuffer<T>,
    values: &[T],
    stream: &CudaStream,
) -> Result<()> {
    assert!(values.len() <= buffer.len());
    #[cfg(feature = "oxide")]
    // SAFETY: capacity is checked and caller owns the source lifetime.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buffer.cu_deviceptr(),
            values.as_ptr(),
            std::mem::size_of_val(values),
            stream.cu_stream(),
        )?;
    }
    #[cfg(feature = "native")]
    // SAFETY: capacity is checked and caller owns the source lifetime.
    unsafe {
        buffer.copy_from_host_async(stream, values)?;
    }
    Ok(())
}

/// Enqueues a D2H copy.
///
/// # Safety
///
/// The caller must keep `values` alive and must not access it until stream completion.
#[cfg(any(feature = "oxide", feature = "native"))]
pub unsafe fn memcpy_dtoh_async<T: Copy>(
    values: &mut [T],
    buffer: &DeviceBuffer<T>,
    stream: &CudaStream,
) -> Result<()> {
    assert!(values.len() <= buffer.len());
    #[cfg(feature = "oxide")]
    // SAFETY: capacity is checked and caller owns exclusive destination access.
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            buffer.cu_deviceptr(),
            std::mem::size_of_val(values),
            stream.cu_stream(),
        )?;
    }
    #[cfg(feature = "native")]
    // SAFETY: capacity is checked and caller owns exclusive destination access.
    unsafe {
        buffer.copy_to_host_async(stream, values)?;
    }
    Ok(())
}

/// Allocates portable page-locked host memory.
///
/// # Safety
///
/// The returned pointer must be released exactly once with [`free_pinned_host`].
#[cfg(any(feature = "oxide", feature = "native"))]
pub unsafe fn alloc_pinned_host(bytes: usize) -> Result<*mut std::ffi::c_void> {
    #[cfg(feature = "oxide")]
    {
        use cuda_core::IntoResult as _;
        let mut raw = std::ptr::null_mut();
        // SAFETY: raw is valid output storage and ownership transfers to caller.
        unsafe {
            cuda_core::sys::cuMemHostAlloc(
                &mut raw,
                bytes,
                cuda_core::sys::CU_MEMHOSTALLOC_PORTABLE,
            )
            .result()?;
        }
        Ok(raw)
    }
    #[cfg(feature = "native")]
    // SAFETY: ownership transfers to caller.
    unsafe {
        native_backend::alloc_pinned_host(bytes)
    }
}

/// Releases memory returned by [`alloc_pinned_host`].
///
/// # Safety
///
/// `raw` must be a live allocation from [`alloc_pinned_host`] with no in-flight transfer.
#[cfg(any(feature = "oxide", feature = "native"))]
pub unsafe fn free_pinned_host(raw: *mut std::ffi::c_void) -> Result<()> {
    #[cfg(feature = "oxide")]
    {
        use cuda_core::IntoResult as _;
        // SAFETY: caller guarantees raw is live and no transfer is in flight.
        unsafe { cuda_core::sys::cuMemFreeHost(raw).result()? };
        Ok(())
    }
    #[cfg(feature = "native")]
    // SAFETY: caller guarantees raw is live and no transfer is in flight.
    unsafe {
        native_backend::free_pinned_host(raw)
    }
}

/// Returns `(multiprocessor_count, max_threads_per_multiprocessor)`.
#[cfg(any(feature = "oxide", feature = "native"))]
pub fn device_occupancy_attributes(ctx: &CudaContext) -> Result<(i32, i32)> {
    #[cfg(feature = "oxide")]
    {
        use cuda_core::IntoResult as _;
        ctx.bind_to_thread()?;
        let mut sm = std::mem::MaybeUninit::<i32>::uninit();
        let mut threads = std::mem::MaybeUninit::<i32>::uninit();
        // SAFETY: outputs are valid and cu_device belongs to the bound context.
        unsafe {
            cuda_core::sys::cuDeviceGetAttribute(
                sm.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                ctx.cu_device(),
            )
            .result()?;
            cuda_core::sys::cuDeviceGetAttribute(
                threads.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                ctx.cu_device(),
            )
            .result()?;
            Ok((sm.assume_init(), threads.assume_init()))
        }
    }
    #[cfg(feature = "native")]
    ctx.occupancy_attributes()
}

#[cfg(test)]
#[cfg(any(feature = "oxide", feature = "native"))]
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
