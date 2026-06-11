//! `cargo-oxide build` が出力した kernel artifact (`.ll` / `.cubin` / `.ptx`)
//! の探索・`.ll`→`.ptx` 変換・module load を行う host 側 helper。
//!
//! kernel 定義は cuda-oxide の bin-entry 制約で各 bin に置く必要があるが、
//! ここにあるのは host-only コードなので kernel を持つ bin 間で共有する。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{Error, Result};
use cuda_core::{CudaContext, CudaModule};

/// 1D launch の既定 block size。
pub const BLOCK_DIM: u32 = 256;

/// 1D launch の grid 数を計算する (= ceil(n / block)、n=0 は block=1 個 launch)。
pub fn grid_dim_1d(n: usize, block: u32) -> (u32, u32, u32) {
    let blocks = ((n as u32).max(1)).div_ceil(block);
    (blocks, 1, 1)
}

/// `cargo-oxide build` が出力した kernel `.ll` を見つけ、`.ptx` に変換した上で
/// CudaModule を load する。
///
/// `manifest_dir` は kernel artifact を持つ bin crate の `CARGO_MANIFEST_DIR`。
/// `env!("CARGO_MANIFEST_DIR")` はコンパイル中の crate で評価されるため、
/// 呼び出し側 bin が渡す (本 crate 内で評価すると gpu-runtime 自身の dir になる)。
///
/// ## 探索順序
///
/// `<name>.ll` → `<name>.cubin` → `<name>.ptx` の順で、`manifest_dir` とその
/// 2 階層上 (kernel bin は `bins/<name>` 配下にある前提で workspace root) の
/// 両方を見る。cargo-oxide は cwd 依存で `.ll` を後者に書くことがある。
///
/// `.ll` を最優先で probe する: kernel source を更新したら必ず `cargo-oxide
/// build` で `.ll` が再生成されるが、`.ptx` / `.cubin` は前回の build から
/// 残ったものが古いまま居座ることがある。`.ll` を見つけたら
/// `compile_ll_to_ptx_via_llc` 内の mtime キャッシュで `.ptx` 鮮度を判断
/// できる。`.ll` が無い場合のみ既製 `.cubin` / `.ptx` (例: 別ツールで事前生成
/// したもの) を fallback で受ける。
pub fn load_kernel_module_with_fallback(
    ctx: &Arc<CudaContext>,
    name: &str,
    manifest_dir: &Path,
) -> Result<Arc<CudaModule>> {
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.to_path_buf());

    let probe = |dir: &Path| {
        for ext in ["ll", "cubin", "ptx"] {
            let p = dir.join(format!("{name}.{ext}"));
            if p.exists() {
                return Some(p);
            }
        }
        None
    };

    let path = probe(manifest_dir)
        .or_else(|| probe(&workspace_root))
        .ok_or_else(|| {
            Error::KernelArtifact(format!(
                "kernel artifact `{name}.{{cubin,ptx,ll}}` not found in {} or {}.\n\
                 Run cargo-oxide build first:\n  \
                 cd {} && CUDA_OXIDE_TARGET=sm_75 cargo-oxide build",
                manifest_dir.display(),
                workspace_root.display(),
                manifest_dir.display(),
            ))
        })?;

    let to_load = if path.extension().and_then(|s| s.to_str()) == Some("ll") {
        compile_ll_to_ptx_via_llc(&path)?
    } else {
        path
    };

    let module =
        ctx.load_module_from_file(to_load.to_str().ok_or_else(|| {
            Error::KernelArtifact("kernel artifact path not valid UTF-8".into())
        })?)?;
    Ok(module)
}

/// `.ll` を libdevice と link、不要 symbol を internalize/dce、nvvm-reflect で
/// `__nvvm_reflect` を畳み込んで `.ptx` に変換して返す。
///
/// パイプライン (NVCC の `compileToCubin` と同等):
///
/// 1. **`llvm-link <ll> libdevice.10.bc → linked.bc`** で `__nv_sqrtf` 等の
///    libdevice intrinsic を IR レベルで取り込み
/// 2. **`opt --passes='nvvm-reflect,internalize,globaldce' linked.bc →
///    opt.bc`** で:
///    - kernel 以外の libdevice 関数を `internal` にして dead-code-elim 対象化
///    - `__nvvm_reflect()` 呼び出しを 0/1 const に置換 (libdevice 内
///      `__CUDA_FTZ` 等の query を解決)
///    - `globaldce` で未参照の libdevice 関数を削除
/// 3. **`llc --mtriple=nvptx64-nvidia-cuda --mcpu=<arch> -O2 opt.bc → .ptx`**
///    で NVPTX backend が PTX 生成
///
/// `#[kernel]` は cuda-oxide が `.ll` の `@llvm.used` に列挙するため
/// internalize されず、`--internalize-public-api-list` を渡さなくても PTX
/// entry として残る。結果の `.ptx` は `.extern .func` 宣言を含まず、`ptxas` /
/// driver JIT が完結する。
///
/// **なぜ `cuda-host::build_cubin_from_ll` (libNVVM 経由) を使わないか**:
/// cuda-oxide が出力する NVVM IR は opaque pointer 形式 (`define void @grad(ptr
/// ...)`) で、libNVVM は古い LLVM 版を内蔵していて opaque pointer を parse
/// できず `nvvmCompileProgram error 9: parse expected type` で reject される。
/// `llvm-link + opt + llc` (LLVM 21+) の組合せは opaque pointer をネイティブに
/// 扱うため成功する。
fn compile_ll_to_ptx_via_llc(ll_path: &Path) -> Result<PathBuf> {
    let stem = ll_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::KernelArtifact("ll path has no stem".into()))?;
    let dir = ll_path.parent().unwrap_or_else(|| Path::new("."));
    let linked_bc = dir.join(format!("{stem}.linked.bc"));
    let opt_bc = dir.join(format!("{stem}.opt.bc"));
    let ptx_path = dir.join(format!("{stem}.ptx"));

    // `.ll`→`.ptx` の中間/出力ファイル (linked.bc / opt.bc / .ptx) は stem 固定
    // パスのため、複数スレッドが同時に compile すると `llc` が書き込み途中の
    // `.bc` を読んで crash する。`cargo test` は 1 binary のテストを同一プロセスの
    // 複数スレッドで走らせるので、プロセス内 Mutex で直列化すれば足りる。最初の
    // スレッドが compile し、後続は lock 取得後に下の mtime cache で skip する。
    static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _compile_guard = COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // cache: skip rebuild if .ptx is newer than .ll
    if let (Ok(ll_meta), Ok(ptx_meta)) = (std::fs::metadata(ll_path), std::fs::metadata(&ptx_path))
        && let (Ok(ll_mtime), Ok(ptx_mtime)) = (ll_meta.modified(), ptx_meta.modified())
        && ptx_mtime > ll_mtime
    {
        return Ok(ptx_path);
    }

    let arch = std::env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_75".to_string());
    let llvm_link =
        std::env::var("LLVM_LINK_BIN").unwrap_or_else(|_| discover_llvm_tool("llvm-link"));
    let opt_bin = std::env::var("OPT_BIN").unwrap_or_else(|_| discover_llvm_tool("opt"));
    let llc_bin = std::env::var("LLC_BIN").unwrap_or_else(|_| discover_llvm_tool("llc"));
    let libdevice = find_libdevice_bc()?;

    // Step 1: llvm-link <ll> libdevice → linked.bc
    run_or_err(
        &llvm_link,
        &[
            ll_path.as_os_str(),
            libdevice.as_os_str(),
            "-o".as_ref(),
            linked_bc.as_os_str(),
        ],
    )?;

    // Step 2: opt --passes='nvvm-reflect,internalize,globaldce'
    // pass 順序は NVCC の compileToCubin 慣例 (reflect 先 → 定数畳み込み →
    // internalize → DCE) に合わせる。reflect が `__nvvm_reflect()` を 0/1 に
    // 畳み込んだ後で internalize/DCE を回すと、libdevice 内 dead branch が
    // 確実に削除される。
    run_or_err(
        &opt_bin,
        &[
            "--passes=nvvm-reflect,internalize,globaldce".as_ref(),
            linked_bc.as_os_str(),
            "-o".as_ref(),
            opt_bc.as_os_str(),
        ],
    )?;

    // Step 3: llc -mcpu=<arch> -O2 opt.bc → .ptx
    let mcpu = format!("--mcpu={arch}");
    run_or_err(
        &llc_bin,
        &[
            "--mtriple=nvptx64-nvidia-cuda".as_ref(),
            mcpu.as_ref(),
            "-O2".as_ref(),
            "-o".as_ref(),
            ptx_path.as_os_str(),
            opt_bc.as_os_str(),
        ],
    )?;

    Ok(ptx_path)
}

/// `.ll`→`.ptx` 変換に使う LLVM tool 名を解決する。`<tool>-22` → `<tool>-21`
/// → 無印 `<tool>` の順で `--version` が通る最初の名前を返す (cuda-oxide 本体の
/// `llc-22 → llc-21` 探索順に揃える)。どれも無ければ `<tool>-21` を返し、spawn
/// 失敗時に `run_or_err` が導入方法を案内する。`LLVM_LINK_BIN` / `OPT_BIN` /
/// `LLC_BIN` env が設定されていればそちらが優先される。
fn discover_llvm_tool(tool: &str) -> String {
    for suffix in ["-22", "-21", ""] {
        let name = format!("{tool}{suffix}");
        let ok = std::process::Command::new(&name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return name;
        }
    }
    format!("{tool}-21")
}

/// `Command::new` + `args` + `status` を 1 行にまとめる helper。
fn run_or_err(bin: &str, args: &[&std::ffi::OsStr]) -> Result<()> {
    let status = std::process::Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| {
            Error::KernelArtifact(format!(
                "failed to spawn {bin}: {e}. \
                 .ll→.ptx conversion uses the LLVM 21+ llvm-link / opt / llc tools \
                 (the llc path, because libNVVM cannot parse opaque-pointer IR). \
                 The -22 / -21 binaries are auto-discovered; if none are found, \
                 set them explicitly via the LLVM_LINK_BIN / OPT_BIN / LLC_BIN env vars."
            ))
        })?;
    if !status.success() {
        return Err(Error::KernelArtifact(format!(
            "{bin} failed with status {status}"
        )));
    }
    Ok(())
}

/// `libdevice.10.bc` を CUDA Toolkit から探す (cuda-oxide の `find_libdevice`
/// と同等)。CUDA root 候補は `bins/nnue_train/build.rs` の cuBLAS 探索と
/// 揃えてある (build script からは本 crate を参照できないため重複定義)。
fn find_libdevice_bc() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("CUDA_OXIDE_LIBDEVICE") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let mut tried = Vec::new();
    let roots: Vec<PathBuf> = std::env::var("CUDA_HOME")
        .ok()
        .into_iter()
        .chain(std::env::var("CUDA_PATH").ok())
        .map(PathBuf::from)
        .chain([
            PathBuf::from("/usr/local/cuda"),
            PathBuf::from("/usr/local/cuda-13.2"),
            PathBuf::from("/usr/local/cuda-12.9"),
            PathBuf::from("/opt/cuda"),
        ])
        .collect();
    for root in roots {
        let candidate = root.join("nvvm/libdevice/libdevice.10.bc");
        tried.push(candidate.display().to_string());
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(Error::KernelArtifact(format!(
        "libdevice.10.bc not found. Set CUDA_OXIDE_LIBDEVICE or CUDA_HOME, \
         or install the CUDA Toolkit. Tried:\n  {}",
        tried.join("\n  ")
    )))
}
