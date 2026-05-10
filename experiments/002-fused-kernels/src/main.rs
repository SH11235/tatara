//! experiments/002-fused-kernels binary entry point。
//!
//! Stage 2 (EPIC #16) の hand-fused kernel suite を build-time PTX 化するための
//! 受け皿。`#[kernel]` 定義は本 file に inline 配置する (cuda-oxide rustc-codegen-cuda
//! backend の "bin entry から到達可能な `#[kernel]` のみ NVPTX IR 化する" 制約、
//! Stage 1-5 で確立、`ATTRIBUTION.md` 参照)。
//!
//! ## 配置
//!
//! - **kernels** (`screlu_grad`, `loss_wdl`, `adamw_step`, `radam_step`,
//!   `ranger_step`, `sparse_ft_forward`, `sparse_ft_backward`) は
//!   Stage 2-1〜2-7 で各 issue が本 file に inline で追加する。Stage 2-1 (#37)
//!   までで `screlu_grad` のみ landed
//! - **reference CPU** は `gpu-kernels` crate の `pointwise/` / `sparse/`
//!   module に置く (Stage 1 の `progress/` と同列の慣行)
//! - **GPU↔CPU smoke test** は本 file の `#[cfg(test)] mod gpu_cpu_equivalence_tests`
//!   に置く。kernel symbol は bin にしか存在しないため `tests/*.rs` (= integration
//!   test) では呼び出せない (Stage 1-10 (#34) で確立した `bins/progress_kpabs_train`
//!   と同じ理由)
//!
//! ## 使い方 (Stage 2-1 以降)
//!
//! ```bash
//! cd experiments/002-fused-kernels && \
//! CUDA_OXIDE_TARGET=sm_75 \
//!     /mnt/e/cuda-oxide-target/release/cargo-oxide build
//!
//! # GPU↔CPU 等価性テスト (要 GPU、ローカル sm_75 box):
//! cargo test -p exp-002-fused-kernels --bin exp-002-fused-kernels --release \
//!     -- --test-threads=1
//! ```
//!
//! 出力 `.ll` は workspace root に `exp_002_fused_kernels.ll` として落ちる
//! (`bins/progress_kpabs_train` と同じ慣行、`KernelLoader` が両 path を probe)。
//!
//! ## CI
//!
//! 本 crate は `cuda-host` 経由で transitive に `cuda.h` を要求するため
//! GitHub-hosted runner では build できない。`.github/workflows/checks.yaml` の
//! `--exclude` リストに `exp-002-fused-kernels` を追加済 (Stage 1-9 で
//! `exp-001-cuda-oxide-kpabs` を exclude したのと同じ理由)。

use std::path::PathBuf;

use cuda_device::{DisjointSlice, kernel, thread};

#[allow(unused_imports)]
use cuda_host::cuda_launch;
#[allow(unused_imports)]
use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};

// ---------------------------------------------------------------------------
// GPU kernels (Stage 2-1 以降で inline 追加していく)
// ---------------------------------------------------------------------------

/// SCReLU activation gradient (fused) — Stage 2-1 (#37)。
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは cuda-oxide の rustc-codegen-cuda backend が
/// **bin entry から到達可能な kernel** のみ PTX 化する設計のため (Stage 1-5
/// で確立)。GPU launch は `#[cfg(test)] mod gpu_cpu_equivalence_tests` から
/// `cuda_launch!` macro 経由で行う。
///
/// アルゴリズムと bullet-shogi 上流 (`crates/compiler/src/tensor/operation/
/// autograd/dfo.rs::SCReLU`) との差分は reference CPU 実装
/// (`gpu_kernels::pointwise::screlu_grad::screlu_grad_cpu`) の docstring および
/// `ATTRIBUTION.md` の Stage 2-1 entry を参照。
///
/// 1 thread = 1 element、atomics 不要、in-place output (`dl_dx`)。
///
/// ## cuda-oxide 制限
///
/// - `f32::clamp` は内部で `f32::max` / `f32::min` を呼ぶ。`f32::max` は
///   Stage 1-7 で **lowering 失敗** (`Symbol std__intrinsics__maximum_number_nsz_f32
///   not found`) を確認しているので、本 kernel では `if-else` ladder で展開する。
///   CPU reference (`screlu_grad_cpu`) は host 実行で `f32::clamp` を使用。
#[kernel]
pub fn screlu_grad(x: &[f32], dl_dy: &[f32], mut dl_dx: DisjointSlice<f32>, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let xi = x[i.get()];
    // f32::clamp(0.0, 1.0) を if-else に展開 (cuda-oxide が f32::max を解決できないため、
    // Stage 1-7 で確認: `Symbol std__intrinsics__maximum_number_nsz_f32 not found`)。
    // CPU reference は host 実行で `f32::clamp` 使用。ここだけ clippy::manual_clamp を allow。
    #[allow(clippy::manual_clamp)]
    let a = if xi < 0.0_f32 {
        0.0_f32
    } else if xi > 1.0_f32 {
        1.0_f32
    } else {
        xi
    };
    let dydx = if a > 0.0_f32 && a < 1.0_f32 {
        2.0_f32 * a
    } else {
        0.0_f32
    };
    if let Some(out) = dl_dx.get_mut(i) {
        *out = dl_dy[i.get()] * dydx;
    }
}

// ---------------------------------------------------------------------------
// Host driver helpers (kernel module loader / launch utilities)
// ---------------------------------------------------------------------------

/// 1 D launch の grid 数を計算する (= ceil(n / block)、n=0 は block=1 個 launch)。
#[allow(dead_code)]
fn grid_dim_1d(n: usize, block: u32) -> (u32, u32, u32) {
    let blocks = ((n as u32).max(1)).div_ceil(block);
    (blocks, 1, 1)
}

#[allow(dead_code)]
const BLOCK_DIM: u32 = 256;

/// `cargo-oxide build` が出力した kernel `.ll` を見つけ、`.ptx` に変換した上で
/// CudaModule を load する。`bins/progress_kpabs_train` Stage 1-9 の同名関数と
/// 同等の loader pipeline。重複しているが、loader を crate 化する refactor は
/// 別 issue (Stage 2-8 wrap-up あたり) で扱う想定。
#[allow(dead_code)]
fn load_kernel_module_with_fallback(
    ctx: &std::sync::Arc<CudaContext>,
    name: &str,
) -> Result<std::sync::Arc<CudaModule>, Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.clone());

    let probe = |dir: &PathBuf| {
        for ext in ["ll", "cubin", "ptx"] {
            let p = dir.join(format!("{name}.{ext}"));
            if p.exists() {
                return Some(p);
            }
        }
        None
    };

    let path = probe(&manifest_dir)
        .or_else(|| probe(&workspace_root))
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!(
                "kernel artifact `{name}.{{cubin,ptx,ll}}` not found in {} or {}.\n\
                 先に cargo-oxide build を実行してください:\n  \
                 cd {} && CUDA_OXIDE_TARGET=sm_75 cargo-oxide build",
                manifest_dir.display(),
                workspace_root.display(),
                manifest_dir.display(),
            )
            .into()
        })?;

    let to_load = if path.extension().and_then(|s| s.to_str()) == Some("ll") {
        compile_ll_to_ptx_via_llc(&path)?
    } else {
        path
    };

    let module = ctx.load_module_from_file(
        to_load
            .to_str()
            .ok_or("kernel artifact path not valid UTF-8")?,
    )?;
    Ok(module)
}

/// `.ll` を libdevice と link、不要 symbol を internalize/dce、nvvm-reflect で
/// `__nvvm_reflect` を畳み込んで `.ptx` に変換して返す。
///
/// pipeline / 設計理由は Stage 1-9 (`bins/progress_kpabs_train/src/main.rs::
/// compile_ll_to_ptx_via_llc`) の docstring を参照 (内容は同一)。
#[allow(dead_code)]
fn compile_ll_to_ptx_via_llc(ll_path: &PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let stem = ll_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("ll path has no stem")?;
    let dir = ll_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let linked_bc = dir.join(format!("{stem}.linked.bc"));
    let opt_bc = dir.join(format!("{stem}.opt.bc"));
    let ptx_path = dir.join(format!("{stem}.ptx"));

    if let (Ok(ll_meta), Ok(ptx_meta)) = (std::fs::metadata(ll_path), std::fs::metadata(&ptx_path))
        && let (Ok(ll_mtime), Ok(ptx_mtime)) = (ll_meta.modified(), ptx_meta.modified())
        && ptx_mtime > ll_mtime
    {
        return Ok(ptx_path);
    }

    let arch = std::env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_75".to_string());
    let llvm_link = std::env::var("LLVM_LINK_BIN").unwrap_or_else(|_| "llvm-link-21".to_string());
    let opt_bin = std::env::var("OPT_BIN").unwrap_or_else(|_| "opt-21".to_string());
    let llc_bin = std::env::var("LLC_BIN").unwrap_or_else(|_| "llc-21".to_string());
    let libdevice = find_libdevice_bc()?;

    // 本 experiment crate の kernel 名 (Stage 2-1 以降で順次追加)。`@<name>` として
    // `.ll` 側に出ているものをそのまま渡す。順番は問わない。
    //
    // **Hazard**: Stage 2-2〜2-7 で kernel を追加するたび本 list に名前を 1 つ
    // 追記する必要がある。漏れると `opt-21 --internalize-public-api-list=...`
    // から外れて `globaldce` で削除され、`cuModuleGetFunction` が
    // `CUDA_ERROR_NOT_FOUND` を返す static failure になる (test では
    // `open_module` で気付ける)。kernel-list を build script から自動列挙する
    // refactor は Stage 2-8 wrap-up 候補。
    let kernel_names = "screlu_grad";

    run_or_err(
        &llvm_link,
        &[
            ll_path.as_os_str(),
            libdevice.as_os_str(),
            "-o".as_ref(),
            linked_bc.as_os_str(),
        ],
    )?;

    let api = format!("--internalize-public-api-list={kernel_names}");
    run_or_err(
        &opt_bin,
        &[
            "--passes=nvvm-reflect,internalize,globaldce".as_ref(),
            api.as_ref(),
            linked_bc.as_os_str(),
            "-o".as_ref(),
            opt_bc.as_os_str(),
        ],
    )?;

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

#[allow(dead_code)]
fn run_or_err(bin: &str, args: &[&std::ffi::OsStr]) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| {
            format!(
                "failed to spawn {bin}: {e}. \
                 Stage 2 は llvm-link-21 / opt-21 / llc-21 を要求します \
                 (libNVVM が opaque pointer IR を parse できないため)。\
                 LLVM_LINK_BIN / OPT_BIN / LLC_BIN env で別 binary を指定可。"
            )
        })?;
    if !status.success() {
        return Err(format!("{bin} failed with status {status}").into());
    }
    Ok(())
}

#[allow(dead_code)]
fn find_libdevice_bc() -> Result<PathBuf, Box<dyn std::error::Error>> {
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
    Err(format!(
        "libdevice.10.bc not found. CUDA_OXIDE_LIBDEVICE か CUDA_HOME を設定するか、\
         CUDA Toolkit を入れてください。Tried:\n  {}",
        tried.join("\n  ")
    )
    .into())
}

fn main() {
    println!(
        "exp-002-fused-kernels: Stage 2 fused kernel suite host driver \
         (Stage 2-1: screlu_grad landed)"
    );
}

// ---------------------------------------------------------------------------
// Stage 2-1 (#37): GPU ↔ CPU reference 数値同等性テスト
// ---------------------------------------------------------------------------
//
// 本 module は **GPU 必須**。CI ではないローカル sm_75 box でのみ走る想定で、
// `#[cfg(test)]` で main.rs 内に置くことで kernel symbol (screlu_grad) に
// 直接 path 解決できる (Stage 1-10 (#34) で確立した bins/progress_kpabs_train
// と同パターン、tests/*.rs では bin の `#[kernel]` に届かない)。
//
// 走らせる:
//
// ```bash
// cd experiments/002-fused-kernels
// CUDA_OXIDE_TARGET=sm_75 /mnt/e/cuda-oxide-target/release/cargo-oxide build
// cargo test -p exp-002-fused-kernels --bin exp-002-fused-kernels --release \
//     -- --test-threads=1
// ```
//
// CI からは workspace `--exclude` で本 crate ごと外れているので影響なし。
#[cfg(test)]
mod gpu_cpu_equivalence_tests {
    use super::*;
    use gpu_kernels::pointwise::screlu_grad::screlu_grad_cpu;

    /// f32 element-wise の screlu_grad は atomic 不要・1 thread = 1 element の
    /// 純粋 pointwise なので CPU reference と bit-equivalent 近い結果になる。
    /// f32 round-off の累積を見越して 1e-6 を使う (Stage 1-10 grad の 1e-5 より
    /// 厳しめでも余裕があるはず: scatter/atomic 経路が無いため)。
    const FLOAT_TOL: f32 = 1e-6;

    type CudaCtxModuleStream = (
        std::sync::Arc<CudaContext>,
        std::sync::Arc<CudaModule>,
        std::sync::Arc<CudaStream>,
    );

    fn open_module() -> Result<CudaCtxModuleStream, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(&ctx, "exp_002_fused_kernels")?;
        Ok((ctx, module, stream))
    }

    /// 決定論的な範囲 [-1, 2] にスパンする入力 + dl_dy。
    /// boundary (0, 1)、saturation (< 0, > 1)、interior (0,1) を全部踏む。
    fn build_fixed_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n);
        let mut dl_dy = Vec::with_capacity(n);
        for i in 0..n {
            let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
            let xi = -1.0_f32 + 3.0_f32 * (i as f32) / denom;
            x.push(xi);
            dl_dy.push(0.5_f32 + (i as f32) * 0.1_f32);
        }
        (x, dl_dy)
    }

    #[test]
    fn screlu_grad_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 1024_usize;
        let (x, dl_dy) = build_fixed_inputs(n);

        // CPU reference
        let mut dl_dx_cpu = vec![0.0_f32; n];
        screlu_grad_cpu(&x, &dl_dy, &mut dl_dx_cpu, n);

        // GPU
        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let dl_dy_dev = DeviceBuffer::from_host(&stream, &dl_dy)?;
        let mut dl_dx_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: screlu_grad,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(x_dev), slice(dl_dy_dev), slice_mut(dl_dx_dev), n_u32]
        }?;
        stream.synchronize()?;
        let dl_dx_gpu = dl_dx_dev.to_host_vec(&stream)?;

        assert_eq!(dl_dx_cpu.len(), dl_dx_gpu.len());
        for (i, (g, c)) in dl_dx_gpu.iter().zip(dl_dx_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff < FLOAT_TOL,
                "dl_dx[{i}]: gpu={g} cpu={c} diff={diff} > {FLOAT_TOL} (x={})",
                x[i],
            );
        }
        Ok(())
    }

    /// 端点の grad = 0 が GPU 側でも崩れないことの専用 ガード。`f32::clamp` の
    /// if-else 展開で `>` / `<` strict が正しく書けているかを確認する。
    #[test]
    fn screlu_grad_kernel_zeroes_grad_at_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let x = vec![-2.0_f32, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
        let dl_dy = vec![1.0_f32; x.len()];
        let n = x.len();

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let dl_dy_dev = DeviceBuffer::from_host(&stream, &dl_dy)?;
        let mut dl_dx_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: screlu_grad,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(x_dev), slice(dl_dy_dev), slice_mut(dl_dx_dev), n_u32]
        }?;
        stream.synchronize()?;
        let dl_dx_gpu = dl_dx_dev.to_host_vec(&stream)?;

        // [-2, -1, 0, 0.5, 1, 2, 3] → [0, 0, 0, 1.0, 0, 0, 0]
        // (a=0.5 で dydx = 2*0.5 = 1.0、dl_dx = 1.0)
        let expected = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        for (i, (g, e)) in dl_dx_gpu.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff < 1e-7,
                "boundary x={}: gpu={g} expected={e} diff={diff}",
                x[i],
            );
        }
        Ok(())
    }
}
