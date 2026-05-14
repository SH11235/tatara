//! `gpu-runtime` 経由で cuda-oxide の vecadd PTX を load し、kernel symbol を
//! resolve するまでの最小 demo。
//!
//! ## スコープ
//!
//! - PTX module 読み込み (`CudaContext::load_module_from_file`)
//! - kernel symbol resolve (`CudaModule::load_function("vecadd")`)
//! - `DeviceBuffer` host↔device round-trip
//!
//! 完全な kernel launch は本 demo に含めない。`#[kernel]` 定義 + `cuda_launch!`
//! macro を使う前提で、raw な `cuda_core::launch_kernel_on_stream` 経由の手書き
//! marshaling は意図的に避ける。本 demo は gpu-runtime crate の re-exports と
//! `Result` / `Error` 型が実機でつながることを示すための smoke 限定。
//!
//! ## 実行前提
//!
//! 1. cuda-oxide repo を clone 済みで、`CUDA_OXIDE_DIR` 環境変数で path を渡す
//!    (デフォルトは `~/git-repos/cuda-oxide`)。
//! 2. **vecadd の PTX が事前に当該 GPU の compute capability 向けに生成済み**
//!    である必要がある:
//!
//!    ```bash
//!    # Turing (sm_75)
//!    CUDA_OXIDE_TARGET=sm_75 \
//!        $CUDA_OXIDE_DIR/target/release/cargo-oxide run vecadd
//!    # Ampere+ なら CUDA_OXIDE_TARGET 不要
//!    ```
//!
//!    詳細は `docs/setup.md` の "sm_75 (Turing) workaround" 参照。
//!
//! ## 実行
//!
//! ```bash
//! cargo run -p gpu-runtime --example vecadd_demo
//! ```
//!
//! 成功時 (本マシン RTX 2070 SUPER):
//! ```text
//! gpu-runtime vecadd demo
//!   PTX:    .../cuda-oxide/.../vecadd.ptx
//! ✓ PTX module loaded
//! ✓ vecadd function symbol resolved (CudaFunction)
//! ✓ host→device→host round-trip OK (1024 elements)
//! ```

use std::env;
use std::path::PathBuf;

use gpu_runtime::{CudaContext, DeviceBuffer};

fn main() -> gpu_runtime::Result<()> {
    let cuda_oxide_dir = match env::var("CUDA_OXIDE_DIR") {
        Ok(p) => p,
        Err(_) => match env::var("HOME") {
            Ok(home) => format!("{home}/git-repos/cuda-oxide"),
            Err(_) => {
                eprintln!(
                    "error: CUDA_OXIDE_DIR is unset and HOME is also unset.\n\
                     Set CUDA_OXIDE_DIR=/path/to/cuda-oxide explicitly."
                );
                std::process::exit(2);
            }
        },
    };
    let ptx_path =
        PathBuf::from(&cuda_oxide_dir).join("crates/rustc-codegen-cuda/examples/vecadd/vecadd.ptx");

    if !ptx_path.exists() {
        eprintln!(
            "error: vecadd.ptx not found at {}\n\
             先に cuda-oxide repo で vecadd を build してください:\n\
                 cd {cuda_oxide_dir}\n\
                 CUDA_OXIDE_TARGET=sm_75 ./target/release/cargo-oxide run vecadd\n\
             (Ampere+ では CUDA_OXIDE_TARGET 不要。詳細は docs/setup.md)",
            ptx_path.display()
        );
        std::process::exit(2);
    }

    println!("gpu-runtime vecadd demo");
    println!("  PTX:    {}", ptx_path.display());

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module_from_file(ptx_path.to_str().expect("PTX path is not UTF-8"))?;
    println!("✓ PTX module loaded");

    let _func = module
        .load_function("vecadd")
        .expect("vecadd function symbol が PTX 内に見つからない");
    println!("✓ vecadd function symbol resolved (CudaFunction)");

    const N: usize = 1024;
    let host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let dev = DeviceBuffer::from_host(&stream, &host)?;
    let back = dev.to_host_vec(&stream)?;
    if back == host {
        println!("✓ host→device→host round-trip OK ({N} elements)");
        Ok(())
    } else {
        eprintln!("✗ round-trip mismatch");
        std::process::exit(1);
    }
}
