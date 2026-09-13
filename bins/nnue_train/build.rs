//! cuBLAS の dynamic link 設定と、training / rescore 用の build 時 git commit
//! 埋め込み (`TATARA_BUILD_COMMIT`)。
//!
//! cuBLAS の dynamic link 設定。`dense_mm_bwd_weight_tiled` (L1 shared weight bwd) を
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
use std::process::Command;

mod build_identity;

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

fn find_cuda_lib_dir(roots: &[PathBuf], target_os: &str) -> Option<PathBuf> {
    for root in roots {
        let lib = if target_os == "windows" {
            root.join("lib").join("x64")
        } else {
            root.join("lib64")
        };
        let found = if target_os == "windows" {
            lib.join("cublas.lib").exists()
        } else {
            lib.join("libcublas.so").exists() || lib.join("libcublas.so.12").exists()
        };
        if found {
            return Some(lib);
        }
    }
    None
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let identity = build_identity::capture(root);
    let legacy = match (&identity.short_commit, identity.dirty) {
        (Some(commit), Some(false)) => commit.clone(),
        (Some(commit), Some(true)) => format!("{commit}-dirty"),
        _ => "unknown".to_owned(),
    };
    println!("cargo:rustc-env=TATARA_BUILD_COMMIT={legacy}");
    println!(
        "cargo:rustc-env=TATARA_BUILD_FULL_COMMIT={}",
        identity.commit.as_deref().unwrap_or("unknown")
    );
    println!(
        "cargo:rustc-env=TATARA_BUILD_DIRTY={}",
        match identity.dirty {
            Some(true) => "true",
            Some(false) => "false",
            None => "unknown",
        }
    );
    for path in build_identity::rerun_paths(root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_identity.rs");
    for key in ["TARGET", "PROFILE", "OPT_LEVEL", "DEBUG"] {
        println!(
            "cargo:rustc-env=TATARA_BUILD_{key}={}",
            std::env::var(key).expect("Cargo build setting")
        );
    }
    let rustc = Command::new(std::env::var_os("RUSTC").expect("Cargo rustc"))
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=TATARA_BUILD_RUSTC={}", rustc.trim());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPU");
    if std::env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("Cargo sets CARGO_CFG_TARGET_OS");
    let roots = cuda_root_candidates();
    let lib_dir = find_cuda_lib_dir(&roots, &target_os).unwrap_or_else(|| {
        let fallback = if target_os == "windows" {
            std::env::var_os("CUDA_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
                })
                .join("lib")
                .join("x64")
        } else {
            PathBuf::from("/usr/local/cuda/lib64")
        };
        println!(
            "cargo:warning=build.rs: cuBLAS import library not found in CUDA_TOOLKIT_PATH / \
             CUDA_HOME / CUDA_PATH / defaults; falling back to {} (link may fail).",
            fallback.display()
        );
        fallback
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
