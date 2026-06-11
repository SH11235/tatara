use gpu_runtime::{CudaContext, CudaModule};

pub(crate) use gpu_runtime::{BLOCK_DIM, grid_dim_1d};

/// `gpu_runtime::load_kernel_module_with_fallback` の本 bin 向け wrapper。
/// `env!("CARGO_MANIFEST_DIR")` はコンパイル中の crate で評価されるため、
/// kernel artifact を持つ bin 側で固定して渡す。
pub(crate) fn load_kernel_module_with_fallback(
    ctx: &std::sync::Arc<CudaContext>,
    name: &str,
) -> gpu_runtime::Result<std::sync::Arc<CudaModule>> {
    gpu_runtime::load_kernel_module_with_fallback(
        ctx,
        name,
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}
