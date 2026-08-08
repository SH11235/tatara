#![cfg_attr(not(feature = "native-cuda"), allow(dead_code))]

// 実体は build script が `include!` して使う。lib からは呼ばないため、test を走らせる
// ためだけに compile する。
#[cfg(test)]
mod compute_cap;

#[cfg(feature = "native-cuda")]
mod runtime;

#[cfg(feature = "native-cuda")]
pub use runtime::{
    Context, DeviceBuffer, Event, Function, Module, NativeCudaError, PinnedBuffer, Result, Stream,
    alloc_pinned_host, free_pinned_host,
};

#[cfg(feature = "native-cuda")]
pub const NATIVE_KERNEL_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tatara_native.fatbin"));

#[cfg(feature = "native-cuda")]
pub const PROGRESS_KERNEL_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tatara_progress.fatbin"));
