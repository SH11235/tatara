//! cuBLAS の dynamic link 設定と、rescore fingerprint 用の build 時 git commit
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

/// build 時点の git commit (short) を返す。working tree が clean でなければ
/// `-dirty` を付ける。repo 外 build や git 不在では `None`。
///
/// 検出の限界: rerun 追跡は HEAD / index の変化のみで、`git add` されていない
/// 編集は次の index 変化まで反映されない (dirty 判定が古い binary が残り得る)。
/// 厳密な identity が要る運用は clean checkout でのビルドが前提で、rescore
/// driver 側も dirty / unknown ビルドでは fingerprint を一致不能にして完了
/// skip / resume を無効化する。
fn git_commit() -> Option<String> {
    let rev = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !rev.status.success() {
        return None;
    }
    let commit = String::from_utf8(rev.stdout).ok()?.trim().to_string();
    if commit.is_empty() {
        return None;
    }
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok();
    let is_dirty = dirty.is_some_and(|out| out.status.success() && !out.stdout.is_empty());
    Some(if is_dirty {
        format!("{commit}-dirty")
    } else {
        commit
    })
}

/// `.git` 実体 directory の絶対 path (worktree では `.git` が file のため
/// `rev-parse` で実体を引く)。
fn git_dir() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--absolute-git-dir"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let dir = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!dir.is_empty()).then_some(dir)
}

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
    // rescore fingerprint 用に build 時の commit id を埋め込む。runtime に実行時
    // CWD で git を呼ぶ方式は、実行場所によって unknown / 無関係 repo の commit に
    // なり binary の identity として成立しない。
    let commit = git_commit().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=TATARA_BUILD_COMMIT={commit}");
    if let Some(dir) = git_dir() {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
    }

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
