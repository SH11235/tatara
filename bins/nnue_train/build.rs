//! cuBLAS の dynamic link 設定。`dense_mm_bwd_weight_tiled` (L1f weight bwd) を
//! `cublasSgemm_v2` で置換するため。
//!
//! CUDA toolkit root 解決順 (gpu-runtime `kernel_loader` の `find_libdevice_bc`
//! (`CUDA_HOME` / `CUDA_PATH` + default 4 path) を踏襲しつつ、build script 専用の
//! legacy alias `CUDA_TOOLKIT_PATH` を最優先で追加。build script からは
//! gpu-runtime を参照できないため候補 list は重複定義):
//! 1. `CUDA_TOOLKIT_PATH` env (build.rs only)
//! 2. `CUDA_HOME` env (runtime と共通)
//! 3. `CUDA_PATH` env (runtime と共通)
//! 4. `/usr/local/cuda`、`/usr/local/cuda-13.2`、`/usr/local/cuda-12.9`、`/opt/cuda`
//!    (runtime と共通の default path)
//!
//! `<root>/lib64` が `libcublas.so` を持つ最初のパスを選ぶ。どれも該当しなければ
//! `/usr/local/cuda/lib64` を最終手段として emit (build 時に warning、link 時に
//! `-lcublas` が見つからなければ ld が報告)。

use std::path::{Path, PathBuf};

fn cuda_root_candidates() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Ok(p) = std::env::var(var) {
            roots.push(PathBuf::from(p));
        }
    }
    for default in [
        "/usr/local/cuda",
        "/usr/local/cuda-13.2",
        "/usr/local/cuda-12.9",
        "/opt/cuda",
    ] {
        roots.push(PathBuf::from(default));
    }
    roots
}

fn find_cuda_lib_dir(roots: &[PathBuf]) -> Option<PathBuf> {
    for root in roots {
        let lib = root.join("lib64");
        if lib.join("libcublas.so").exists() || lib.join("libcublas.so.12").exists() {
            return Some(lib);
        }
    }
    None
}

fn main() {
    let roots = cuda_root_candidates();
    let lib_dir = find_cuda_lib_dir(&roots).unwrap_or_else(|| {
        println!(
            "cargo:warning=build.rs: libcublas.so not found in CUDA_TOOLKIT_PATH / CUDA_HOME / \
             CUDA_PATH / /usr/local/cuda*; falling back to /usr/local/cuda/lib64 (link may fail)."
        );
        PathBuf::from("/usr/local/cuda/lib64")
    });
    println!(
        "cargo:rustc-link-search=native={}",
        Path::new(&lib_dir).display()
    );
    println!("cargo:rustc-link-lib=dylib=cublas");
    for var in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}
