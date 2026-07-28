use gpu_runtime::{CudaContext, CudaModule};

pub(crate) use gpu_runtime::{BLOCK_DIM, grid_dim_1d};

#[cfg(feature = "native-cuda")]
thread_local! {
    static NATIVE_BACKEND_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(feature = "native-cuda")]
pub(crate) fn with_native_backend<T>(native: bool, operation: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            NATIVE_BACKEND_OVERRIDE.set(self.0);
        }
    }

    let previous = NATIVE_BACKEND_OVERRIDE.replace(Some(native));
    let _restore = Restore(previous);
    operation()
}

#[cfg(all(test, feature = "native-cuda"))]
pub(crate) fn with_test_native_backend<T>(native: bool, operation: impl FnOnce() -> T) -> T {
    with_native_backend(native, operation)
}

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
pub(crate) fn native_backend_requested() -> bool {
    #[cfg(feature = "native-cuda-host")]
    return true;

    #[cfg(feature = "native-cuda")]
    {
        if let Some(native) = NATIVE_BACKEND_OVERRIDE.get() {
            return native;
        }
        std::env::var_os("TATARA_CUDA_BACKEND").as_deref() == Some(std::ffi::OsStr::new("native"))
    }
}

/// `gpu_runtime::load_kernel_module_with_fallback` の本 bin 向け wrapper。
/// `env!("CARGO_MANIFEST_DIR")` はコンパイル中の crate で評価されるため、
/// kernel artifact を持つ bin 側で固定して渡す。
pub(crate) fn load_kernel_module_with_fallback(
    ctx: &std::sync::Arc<CudaContext>,
    name: &str,
) -> gpu_runtime::Result<std::sync::Arc<CudaModule>> {
    #[cfg(not(any(feature = "native-cuda", feature = "native-cuda-host")))]
    if std::env::var_os("TATARA_CUDA_BACKEND").as_deref() == Some(std::ffi::OsStr::new("native")) {
        return Err(gpu_runtime::Error::KernelArtifact(
            "native CUDA comparison was requested; rebuild with \
             --no-default-features --features native-cuda"
                .into(),
        ));
    }

    #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
    if native_backend_requested() {
        #[cfg(feature = "native-cuda-host")]
        return ctx.load_module_from_image(cuda_native_runtime::NATIVE_KERNEL_FATBIN);
        #[cfg(feature = "native-cuda")]
        return ctx
            .load_module_from_image(cuda_native_runtime::NATIVE_KERNEL_FATBIN)
            .map_err(Into::into);
    }

    #[cfg(feature = "cuda-oxide")]
    {
        gpu_runtime::load_kernel_module_with_fallback(
            ctx,
            name,
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
        )
    }

    #[cfg(not(feature = "cuda-oxide"))]
    Err(gpu_runtime::Error::KernelArtifact(format!(
        "native CUDA module `{name}` was not selected"
    )))
}
