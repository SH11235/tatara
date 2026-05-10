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
//!   Stage 2-1〜2-7 で各 issue が本 file に inline で追加する。Stage 2-2 (#38)
//!   までで `screlu_grad` + `loss_wdl` の 2 件 landed
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

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
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

/// Sigmoid + WDL blend + scale loss kernel — Stage 2-2 (#38)。
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは Stage 1-5 で確立した cuda-oxide rustc-codegen-cuda
/// backend の bin-entry 制約のため。GPU launch は `#[cfg(test)] mod
/// gpu_cpu_equivalence_tests` から `cuda_launch!` macro 経由で行う。
///
/// アルゴリズムと bullet-shogi 上流 (`crates/bullet_lib/src/value/loader.rs::
/// 301-316` の WDL blend + `crates/compiler/src/tensor/operation/autograd/
/// dfo.rs::Sigmoid`) との対応 / divergence は reference CPU
/// (`gpu_kernels::pointwise::loss_wdl::loss_wdl_cpu`) の docstring および
/// `ATTRIBUTION.md` の Stage 2-2 entry を参照。
///
/// 1 thread = 1 position。`dl_dout` は 1 thread = 1 index で排他的更新 (atomics
/// 不要)、`loss_acc` は f64 単一 cell の Σ err^2 で `DeviceAtomicF64::fetch_add`
/// (Stage 1-6 grad / 1-8 eval 踏襲)。
///
/// 引数数 (9) は bullet 上流式の host invariant を漏れなく渡すため
/// `clippy::too_many_arguments` を allow (Stage 1 grad と同型)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn loss_wdl(
    out: &[f32],
    score: &[f32],
    wdl: &[f32],
    per_pos_norm: &[f32],
    mut dl_dout: DisjointSlice<f32>,
    loss_acc: &[f64],
    lambda: f32,
    scale: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let p = 1.0_f32 / (1.0_f32 + (-(out[i.get()] * scale)).exp());
    let ys = 1.0_f32 / (1.0_f32 + (-(score[i.get()] * scale)).exp());
    let y = lambda * wdl[i.get()] + (1.0_f32 - lambda) * ys;
    let err = p - y;
    let norm = per_pos_norm[i.get()];

    if let Some(g) = dl_dout.get_mut(i) {
        *g = 2.0_f32 * err * p * (1.0_f32 - p) * scale * norm;
    }

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell として確保済み
    // (Stage 1-6 grad / 1-8 eval と同型、`DeviceAtomicF64` への reinterpret cast)。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);
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
    let kernel_names = "screlu_grad,loss_wdl";

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
         (Stage 2-2: screlu_grad + loss_wdl landed)"
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

    /// loss_wdl: GPU と CPU reference の数値同等性。loss は f64 atomic、grad は
    /// 1 thread = 1 index で排他的なため atomic 不要。tolerance は loss / grad
    /// 共に 1e-6 (Stage 1-10 grad の f64 loss 1e-8 は 16 元の小規模で通った値、
    /// 本テストの 1024 元では atomic add reordering で ~1e-7 drift のため緩める。
    /// 詳細根拠は本 test 内 loss-assert の inline コメント参照)。
    #[test]
    fn loss_wdl_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        use gpu_kernels::pointwise::loss_wdl::loss_wdl_cpu;

        let (_ctx, module, stream) = open_module()?;
        let n = 1024_usize;

        // 決定論的入力: out / score を [-3, 3] にスパン (sigmoid の interior と
        // saturation の両方を踏む)、wdl は {0, 0.5, 1} を均等に振る、norm = 1/n。
        let mut out = Vec::with_capacity(n);
        let mut score = Vec::with_capacity(n);
        let mut wdl = Vec::with_capacity(n);
        let mut per_pos_norm = Vec::with_capacity(n);
        for i in 0..n {
            let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
            let t = (i as f32) / denom;
            out.push(-3.0_f32 + 6.0_f32 * t);
            score.push(-3.0_f32 + 6.0_f32 * t);
            wdl.push(match i % 3 {
                0 => 0.0_f32,
                1 => 0.5_f32,
                _ => 1.0_f32,
            });
            per_pos_norm.push(1.0_f32 / (n as f32));
        }
        let lambda = 0.5_f32;
        let scale = 1.0_f32;

        // CPU reference
        let mut dl_dout_cpu = vec![0.0_f32; n];
        let mut loss_cpu = 0.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &per_pos_norm,
            &mut dl_dout_cpu,
            &mut loss_cpu,
            lambda,
            scale,
            n,
        );

        // GPU
        let out_dev = DeviceBuffer::from_host(&stream, &out)?;
        let score_dev = DeviceBuffer::from_host(&stream, &score)?;
        let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
        let norm_dev = DeviceBuffer::from_host(&stream, &per_pos_norm)?;
        let mut dl_dout_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: loss_wdl,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice(out_dev),
                slice(score_dev),
                slice(wdl_dev),
                slice(norm_dev),
                slice_mut(dl_dout_dev),
                slice(loss_dev),
                lambda,
                scale,
                n_u32
            ]
        }?;
        stream.synchronize()?;
        let dl_dout_gpu = dl_dout_dev.to_host_vec(&stream)?;
        let loss_gpu = loss_dev.to_host_vec(&stream)?[0];

        // f64 atomic add (loss): 1024 元の Σerr^2 累積で順序依存の reordering が
        // ある。Stage 1-10 grad は 16 元で 1e-8 通ったが、1024 元では中間値の
        // magnitude 差で ~1e-7 の relative drift が出るため abs 1e-6 まで緩める
        // (relative 1.5e-8 程度、sum ~68 として)。
        let loss_diff = (loss_gpu - loss_cpu).abs();
        assert!(
            loss_diff < 1e-6,
            "loss: gpu={loss_gpu} cpu={loss_cpu} diff={loss_diff}"
        );

        // f32 grad は 1 thread = 1 index で atomic 不要、CPU と bit-equiv 近い
        for (i, (g, c)) in dl_dout_gpu.iter().zip(dl_dout_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff < 1e-6,
                "dl_dout[{i}]: gpu={g} cpu={c} diff={diff} (out={}, score={}, wdl={})",
                out[i],
                score[i],
                wdl[i],
            );
        }
        Ok(())
    }

    /// loss_wdl: `lambda = 1` で WDL 完全採用、`out = 0` (p=0.5) + `wdl = 0.5`
    /// (draw) → err = 0、loss = 0、dl_dout = 0 になる端点を GPU で確認。
    #[test]
    fn loss_wdl_kernel_zero_grad_at_draw_with_p_half() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 4_usize;
        let out = vec![0.0_f32; n];
        let score = vec![999.0_f32; n]; // lambda=1 で無視される
        let wdl = vec![0.5_f32; n];
        let per_pos_norm = vec![1.0_f32; n];
        let lambda = 1.0_f32;
        let scale = 1.0_f32;

        let out_dev = DeviceBuffer::from_host(&stream, &out)?;
        let score_dev = DeviceBuffer::from_host(&stream, &score)?;
        let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
        let norm_dev = DeviceBuffer::from_host(&stream, &per_pos_norm)?;
        let mut dl_dout_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: loss_wdl,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice(out_dev),
                slice(score_dev),
                slice(wdl_dev),
                slice(norm_dev),
                slice_mut(dl_dout_dev),
                slice(loss_dev),
                lambda,
                scale,
                n_u32
            ]
        }?;
        stream.synchronize()?;
        let dl_dout_gpu = dl_dout_dev.to_host_vec(&stream)?;
        let loss_gpu = loss_dev.to_host_vec(&stream)?[0];

        for (i, &g) in dl_dout_gpu.iter().enumerate() {
            assert_eq!(g, 0.0_f32, "dl_dout[{i}] = {g}, expected 0");
        }
        assert_eq!(loss_gpu, 0.0);
        Ok(())
    }
}
