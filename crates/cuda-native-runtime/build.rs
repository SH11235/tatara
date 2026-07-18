use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=TATARA_CUDA_COMPUTE");
    println!("cargo:rerun-if-changed=kernels/native_kernels.cu");

    if env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo sets CARGO_CFG_TARGET_OS");
    if target_os == "linux" && Path::new("/usr/lib/wsl/lib").exists() {
        println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");
    }
    if target_os == "windows"
        && let Some(root) = ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"]
            .iter()
            .find_map(env::var_os)
    {
        println!(
            "cargo:rustc-link-search=native={}",
            PathBuf::from(root).join("lib").join("x64").display()
        );
    }

    let nvcc = find_nvcc();
    let compute = env::var("TATARA_CUDA_COMPUTE").unwrap_or_else(|_| "75".into());
    assert!(
        !compute.is_empty() && compute.bytes().all(|byte| byte.is_ascii_digit()),
        "TATARA_CUDA_COMPUTE must be a numeric compute capability such as 75 or 120"
    );
    let codegen = format!("arch=compute_{compute},code=compute_{compute}");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("tatara_native.fatbin");
    let status = Command::new(&nvcc)
        .args([
            "--fatbin",
            "--std=c++17",
            "-O3",
            "--generate-code",
            &codegen,
            "kernels/native_kernels.cu",
            "-o",
        ])
        .arg(&output)
        .status()
        .unwrap_or_else(|e| panic!("failed to execute {}: {e}", nvcc.display()));
    assert!(status.success(), "NVCC failed with status {status}");
}

fn find_nvcc() -> PathBuf {
    if let Some(path) = env::var_os("NVCC") {
        return PathBuf::from(path);
    }

    let exe = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };
    for name in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(name) {
            let candidate = PathBuf::from(root).join("bin").join(exe);
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    if cfg!(target_os = "linux") {
        let conventional = Path::new("/usr/local/cuda").join("bin").join(exe);
        if conventional.is_file() {
            return conventional;
        }
        if let Ok(entries) = std::fs::read_dir("/usr/local") {
            let mut candidates = entries
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("cuda-"))
                .map(|entry| entry.path().join("bin").join(exe))
                .filter(|path| path.is_file())
                .collect::<Vec<_>>();
            candidates.sort();
            if let Some(candidate) = candidates.pop() {
                return candidate;
            }
        }
    }

    PathBuf::from(exe)
}
