//! cuBLAS の dynamic link 設定。`dense_mm_bwd_weight_tiled` (L1f weight bwd) を
//! `cublasSgemm_v2` で置換するため。
//!
//! `CUDA_TOOLKIT_PATH` を尊重 (未設定なら `/usr/local/cuda`)。CI / nix 環境では
//! 環境変数経由で別 path に向ける。

fn main() {
    let cuda_path =
        std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={cuda_path}/lib64");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
}
