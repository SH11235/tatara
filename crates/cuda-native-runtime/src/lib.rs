#![cfg_attr(not(feature = "native-cuda"), allow(dead_code))]

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
