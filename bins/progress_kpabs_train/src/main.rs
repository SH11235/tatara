//! `progress-kpabs-train` binary entry point。
//!
//! 本 file は kernel 定義 (`#[kernel]`) と host loop driver を統合する。host
//! loop は PSV file 群 → batch → forward → grad → adam_step を駆動し、最終
//! weight を `progress.bin` (f64 LE × N) に出力する。
//!
//! ## 設計
//!
//! - **kernels** (forward / grad / adam_step / eval) は本 file の inline
//!   `#[kernel]`。cuda-oxide の rustc-codegen-cuda backend は bin entry 経由で
//!   到達可能な `#[kernel]` のみ NVPTX IR 化するため、main.rs に置く必要がある。
//! - **GpuTrainer** は本 file。device buffer (weights / m / v / grad / loss_acc /
//!   hist + scratch) を所有し、`step` / `eval_forward` で 1 batch 分の
//!   forward → grad/eval → (training なら) adam_step を launch する。
//! - **host helper** (Batch builder / PSV reader / progress.bin I/O / CLI) は
//!   GPU 非依存なので `lib.rs` の `host` module に置く。host helper は
//!   `cargo test -p progress-kpabs-train` で GPU なしで単体テストできる。
//!
//! ## 使い方
//!
//! ```bash
//! # 1 epoch の動作確認 (smoke):
//! cargo run -p progress-kpabs-train -- \
//!     --data crates/shogi-format/tests/data/sample.psv \
//!     --output /tmp/progress.bin \
//!     --games-per-step 4 --max-games 8 --lr 1e-3
//!
//! # 実データで:
//! cargo run --release -p progress-kpabs-train -- \
//!     --data <path/to/training.bin> \
//!     --output progress.bin --epochs 1
//! ```
//!
//! 事前に `cargo-oxide build` で kernel `.ll` を生成しておく必要がある
//! (`cargo-oxide` の build 手順は `docs/setup.md` 参照):
//!
//! ```bash
//! cd bins/progress_kpabs_train && \
//!     CUDA_OXIDE_TARGET=sm_75 cargo-oxide build
//! ```
//!
//! 出力先 (workspace root の `progress_kpabs_train.ll`) は `KernelLoader`
//! が自動で probe する (CARGO_MANIFEST_DIR と workspace root の両方)。

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicU64};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_launch;
use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
use progress_kpabs_train::host::{
    ADAM_BETA1, ADAM_BETA2, ADAM_EPS, MAX_INDS_PER_POS,
    batch::Batch,
    cli::Args,
    games::{GameIterator, PackCursor},
    progress_bin::{read_progress_bin, write_progress_bin},
};
use shogi_features::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;

/// Forward kernel (KP-abs sigmoid prediction per position)。
///
/// `#[kernel]` を main.rs に inline 定義しているのは、cuda-oxide の
/// rustc-codegen-cuda backend が **bin entry から到達可能な kernel** のみを
/// PTX 化する設計のため (lib.rs 内 `#[kernel]` は `cargo oxide build` の出力
/// に現れない)。host 側 `GpuTrainer` は同じ file 内の `cuda_launch!` macro
/// 経由で呼び出す。
///
/// アルゴリズムと bullet-shogi 上流 (`KERNELS_SRC::k_forward`) との差分は
/// reference CPU 実装 (`gpu_kernels::progress::forward::forward_cpu`) の
/// docstring および `ATTRIBUTION.md` を参照。
#[kernel]
pub fn forward(
    indices: &[i32],
    weights: &[f32],
    mut preds: DisjointSlice<f32>,
    n_pos: u32,
    max_inds: u32,
) {
    let pos = thread::index_1d();
    if pos.get() >= n_pos as usize {
        return;
    }
    let mut z = 0.0f32;
    let base = pos.get() * (max_inds as usize);
    let mut j: u32 = 0;
    while j < max_inds {
        let idx = indices[base + (j as usize)];
        if idx >= 0 {
            z += weights[idx as usize];
        }
        j += 1;
    }
    if let Some(p) = preds.get_mut(pos) {
        *p = 1.0f32 / (1.0f32 + (-z).exp());
    }
}

/// Backward kernel (loss accumulation + gradient scatter + prediction histogram)。
///
/// `#[kernel]` を main.rs に inline 定義しているのは forward と同じ理由
/// (cuda-oxide backend は bin entry から到達可能な kernel しか PTX 化しない)。
///
/// アルゴリズムと bullet-shogi 上流 (`KERNELS_SRC::k_grad_loss_hist`) との
/// 差分は reference CPU 実装 (`gpu_kernels::progress::grad::grad_cpu`) の
/// docstring および `ATTRIBUTION.md` を参照。
///
/// Atomics:
/// - `grad[idx]` の f32 加算 → `DeviceAtomicF32::fetch_add` (`atomicrmw fadd`)
/// - `loss_acc` (single cell) の f64 加算 → `DeviceAtomicF64::fetch_add`
/// - `hist[bin]` の u64 加算 → `DeviceAtomicU64::fetch_add`
///
/// ordering は全 `Relaxed`。loss/grad/hist は最終 reduce 結果のみ問題で、
/// 順序保証は要らない (bullet 上流 C++ `atomicAdd` の暗黙 ordering と同等)。
///
/// 引数数 (9) は bullet 上流 `k_grad_loss_hist` と 1:1 対応のため
/// `too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn grad(
    indices: &[i32],
    preds: &[f32],
    targets: &[f32],
    per_pos_norm: &[f32],
    grad: &[f32],
    loss_acc: &[f64],
    hist: &[u64],
    n_pos: u32,
    max_inds: u32,
) {
    let pos = thread::index_1d();
    if pos.get() >= n_pos as usize {
        return;
    }

    let p = preds[pos.get()];
    let y = targets[pos.get()];
    let err = p - y;
    let norm = per_pos_norm[pos.get()];
    let gscale = 2.0f32 * err * p * (1.0f32 - p) * norm;

    let base = pos.get() * (max_inds as usize);
    let mut j: u32 = 0;
    while j < max_inds {
        let idx = indices[base + (j as usize)];
        if idx >= 0 {
            // SAFETY: `grad` は host 側で `DeviceBuffer<f32>` として確保され、
            // `idx` は `for_each_active_index` 経由で
            // `0..SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS` に収まることを保証
            // (shogi-features 側 invariant、host loop で全 `>= 0` index に対し
            // bounds 確認済の前提で kernel に渡る)。
            // alignment: `f32` の `align_of` (4) は `DeviceAtomicF32` と同一
            // (`#[repr(transparent)]` over `UnsafeCell<f32>`)。non-atomic 経路で
            // 同じ memory に書き込む code path は本 kernel / host loop には存在しない。
            let grad_atom =
                unsafe { &*(grad.as_ptr().add(idx as usize) as *const DeviceAtomicF32) };
            grad_atom.fetch_add(gscale, AtomicOrdering::Relaxed);
        }
        j += 1;
    }

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell として確保済み。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);

    // i32::clamp は内部で assert!(min <= max) → Debug::fmt panic path を含み、
    // cuda-oxide backend (rustc-codegen-cuda) が現状 lowering 未対応のため
    // bullet 上流 C++ の `if (b<0) b=0; if (b>7) b=7;` を verbatim 移植する。
    // ここだけ `clippy::manual_clamp` を allow。
    #[allow(clippy::manual_clamp)]
    let b = {
        let mut b = (p * 8.0f32) as i32;
        if b < 0 {
            b = 0;
        }
        if b > 7 {
            b = 7;
        }
        b
    };
    // SAFETY: `hist.len() == 8` を host 側 invariant とする。clamp [0,7] で範囲内。
    let hist_atom = unsafe { &*(hist.as_ptr().add(b as usize) as *const DeviceAtomicU64) };
    hist_atom.fetch_add(1u64, AtomicOrdering::Relaxed);
}

/// Adam optimizer step kernel (1 thread = 1 weight)。
///
/// `#[kernel]` を main.rs に inline 定義しているのは forward / grad と同じ
/// 理由 (cuda-oxide backend は bin entry から到達可能な kernel しか PTX 化
/// しない)。
///
/// アルゴリズムと bullet-shogi 上流 (`KERNELS_SRC::k_adam_step`) との差分は
/// reference CPU 実装 (`gpu_kernels::progress::adam_step::adam_step_cpu`) の
/// docstring および `ATTRIBUTION.md` を参照。
///
/// 引数数 (11) は bullet 上流 `k_adam_step` と 1:1 対応のため
/// `too_many_arguments` を allow する。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn adam_step(
    mut weights: DisjointSlice<f32>,
    mut m: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut grad: DisjointSlice<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }

    // 4 buffer すべて i 番目に対し 1 thread が排他的にアクセスするため atomics 不要。
    // get_mut が None になるのは host 側 invariant 違反 (len < n) のときのみで、
    // forward kernel の defensive pattern と同じく該当 thread は silent skip する。
    let g_opt = grad.get_mut(i);
    let m_opt = m.get_mut(i);
    let v_opt = v.get_mut(i);
    let w_opt = weights.get_mut(i);
    if let (Some(g_ref), Some(m_ref), Some(v_ref), Some(w_ref)) = (g_opt, m_opt, v_opt, w_opt) {
        let g = *g_ref;
        let mi = beta1 * *m_ref + (1.0f32 - beta1) * g;
        let vi = beta2 * *v_ref + (1.0f32 - beta2) * g * g;
        *m_ref = mi;
        *v_ref = vi;
        // f32::max は cuda-oxide が `std::intrinsics::maximum_number_nsz_f32` を解決できず
        // ('Symbol ... not found') lowering 失敗するため、bullet 上流 C++ `fmaxf(bc, 1e-30f)`
        // を if-else で verbatim 移植する。CPU reference (adam_step_cpu) は host 実行で
        // f32::max を使用する。
        let bc1_safe = if bc1 > 1e-30f32 { bc1 } else { 1e-30f32 };
        let bc2_safe = if bc2 > 1e-30f32 { bc2 } else { 1e-30f32 };
        let m_hat = mi / bc1_safe;
        let v_hat = vi / bc2_safe;
        *w_ref -= lr * m_hat / (v_hat.sqrt() + eps);
        *g_ref = 0.0f32;
    }
}

/// Evaluation kernel (loss + histogram only)。validation / test 時に gradient
/// 計算なしで loss と prediction 分布を取るために使う (`grad` から gradient
/// scatter 部分を取り除いたサブセット)。
///
/// `#[kernel]` を main.rs に inline 定義しているのは forward / grad / adam_step
/// と同じ理由。
///
/// アルゴリズムと bullet-shogi 上流 (`KERNELS_SRC::k_eval_loss_hist`) との
/// 差分は reference CPU 実装 (`gpu_kernels::progress::eval::eval_cpu`) の
/// docstring および `ATTRIBUTION.md` を参照。
///
/// Atomics:
/// - `loss_acc` (single f64 cell) → `DeviceAtomicF64::fetch_add` (`atomicrmw fadd double`)
/// - `hist[bin]` (u64) → `DeviceAtomicU64::fetch_add` (`atomicrmw add i64`)
///
/// ordering は `Relaxed` (collection 用途で順序保証不要)。
#[kernel]
pub fn eval(preds: &[f32], targets: &[f32], loss_acc: &[f64], hist: &[u64], n_pos: u32) {
    let pos = thread::index_1d();
    if pos.get() >= n_pos as usize {
        return;
    }

    let p = preds[pos.get()];
    let y = targets[pos.get()];
    let err = p - y;

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell として確保済み。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);

    // i32::clamp は cuda-oxide が現状 lowering 未対応 (`grad` kernel と同根の
    // panic 経路 `Debug::fmt` を含むため)。bullet 上流の if-else を verbatim
    // 移植。
    #[allow(clippy::manual_clamp)]
    let b = {
        let mut b = (p * 8.0f32) as i32;
        if b < 0 {
            b = 0;
        }
        if b > 7 {
            b = 7;
        }
        b
    };
    // SAFETY: `hist.len() == 8` を host 側 invariant とする。clamp [0,7] で範囲内。
    let hist_atom = unsafe { &*(hist.as_ptr().add(b as usize) as *const DeviceAtomicU64) };
    hist_atom.fetch_add(1u64, AtomicOrdering::Relaxed);
}

// ---------------------------------------------------------------------------
// Host driver (GpuTrainer + main)
// ---------------------------------------------------------------------------

/// 1 D launch の grid 数を計算する (= ceil(n / block)、n=0 は block=1 個 launch)。
fn grid_dim_1d(n: usize, block: u32) -> (u32, u32, u32) {
    let blocks = ((n as u32).max(1)).div_ceil(block);
    (blocks, 1, 1)
}

const BLOCK_DIM: u32 = 256;

/// `cargo-oxide build` が出力した kernel `.ll` を見つけ、`.ptx` に変換した上で
/// CudaModule を load する。
///
/// ## 探索順序
///
/// `<name>.cubin` → `<name>.ptx` → `<name>.ll` の順で、CARGO_MANIFEST_DIR と
/// workspace root の両方を見る。cargo-oxide rev `6de0509` は cwd 依存で `.ll` を
/// 後者に書くことがある (実機確認済)。
///
/// ## .ll → .ptx の変換
///
/// `cuda_host::ltoir::build_cubin_from_ll` (libNVVM 経由) は cuda-oxide が出力
/// する opaque pointer 形式の NVVM IR を parse できない (libNVVM 内蔵 LLVM が
/// 古く、`define void @grad(ptr ...)` 形式を `libnvvm error 9: parse expected
/// type` で reject する)。代わりに **`llc-21`** で素直に NVPTX backend を回す。
/// `__nv_sqrtf` 等 libdevice intrinsic は `.extern .func` 宣言として残るが、
/// CUDA driver の JIT linker が module load 時に libdevice と link する。
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

    // `.ll` を最優先で probe する: kernel source を更新したら必ず `cargo-oxide build`
    // で `.ll` が再生成されるが、`.ptx`/`.cubin` は前回の build から残ったものが
    // 古いまま居座ることがある。`.ll` を見つけたら `compile_ll_to_ptx_via_llc` 内の
    // mtime キャッシュで `.ptx` 鮮度を判断できる。`.ll` が無い場合のみ既製 `.ptx` /
    // `.cubin` (例: 別ツールで事前生成したもの) を fallback で受ける。
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
/// パイプライン (NVCC の `compileToCubin` と同等):
///
/// 1. **`llvm-link-21 <ll> libdevice.10.bc → linked.bc`** で `__nv_sqrtf` 等の
///    libdevice intrinsic を IR レベルで取り込み
/// 2. **`opt-21 --passes='nvvm-reflect,internalize,globaldce'
///    --internalize-public-api-list=<kernel symbols> linked.bc → opt.bc`** で:
///    - kernel 以外の libdevice 関数を `internal` にして dead-code-elim 対象化
///    - `__nvvm_reflect()` 呼び出しを 0/1 const に置換 (libdevice 内 `__CUDA_FTZ`
///      等の query を解決)
///    - `globaldce` で未参照の libdevice 関数を削除
/// 3. **`llc-21 --mtriple=nvptx64-nvidia-cuda --mcpu=<arch> -O2 opt.bc → .ptx`**
///    で NVPTX backend が PTX 生成
///
/// 結果の `.ptx` は `.extern .func` 宣言を含まず、`ptxas` / driver JIT が完結する。
///
/// **なぜ `cuda-host::build_cubin_from_ll` (libNVVM 経由) を使わないか**:
/// cuda-oxide が出力する NVVM IR は opaque pointer 形式 (`define void @grad(ptr
/// ...)`) で、libNVVM は古い LLVM 版を内蔵していて opaque pointer を parse
/// できず `nvvmCompileProgram error 9: parse expected type` で reject される。
/// `llvm-link-21 + opt-21 + llc-21` の組合せは LLVM 21 series 以降の opaque
/// pointer をネイティブに扱うため成功する。
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

    // cache: skip rebuild if .ptx is newer than .ll
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

    // 本 binary に inline 定義した kernel 名。`@<name>` として `.ll` 側に
    // 出ているものをそのまま渡す。順番は問わない。
    let kernel_names = "grad,forward,adam_step,eval";

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

/// `Command::new` + `args` + `status` を 1 行にまとめる helper。
fn run_or_err(bin: &str, args: &[&std::ffi::OsStr]) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| {
            format!(
                "failed to spawn {bin}: {e}. \
                 本 binary は llvm-link-21 / opt-21 / llc-21 を要求します \
                 (libNVVM が opaque pointer IR を parse できないため)。\
                 LLVM_LINK_BIN / OPT_BIN / LLC_BIN env で別 binary を指定可。"
            )
        })?;
    if !status.success() {
        return Err(format!("{bin} failed with status {status}").into());
    }
    Ok(())
}

/// `libdevice.10.bc` を CUDA Toolkit から探す (cuda-oxide の `find_libdevice` と同等)。
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

/// GPU 上で 4 kernel を順次起動する trainer。bullet-shogi 上流の `GpuTrainer`
/// 相当だが、cuda-oxide の `cuda_launch!` macro を使うため kernel シンボルを
/// 直接渡せる (NVRTC 経由ではない)。
///
/// device buffer は内部所有:
/// - `weights / m / v / grad`: `DeviceBuffer<f32>` (size = `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`)
/// - `loss_acc`: `DeviceBuffer<f64>` (size = 1)
/// - `hist`: `DeviceBuffer<u64>` (size = 8)
///
/// 入力 (`indices` / `targets` / `per_pos_norm` / `preds`) は `step` /
/// `eval_forward` 内で `DeviceBuffer::from_host` / `zeroed` する (scratch reuse
/// は cuda-oxide の `DeviceBuffer<T>::from_host` が新規 allocation のみ提供
/// するため未実装、`Stream::memcpy_h2d` を直接呼び出す形で将来再利用化する
/// 想定の TODO)。
struct GpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,

    weights: DeviceBuffer<f32>,
    m: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    grad: DeviceBuffer<f32>,
    loss_acc: DeviceBuffer<f64>,
    hist: DeviceBuffer<u64>,

    /// Adam の `beta^t` 累積値 (`bc1 = 1 - beta1_pow` を kernel に渡す)。
    beta1_pow: f32,
    beta2_pow: f32,
}

impl GpuTrainer {
    /// CUDA context を作成し、kernel module を load し、device buffer を確保する。
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        init_weights: Option<&[f32]>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "progress_kpabs_train")?;

        let n = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
        let weights = match init_weights {
            Some(init) => {
                if init.len() != n {
                    return Err(
                        format!("init_weights length {} != expected {}", init.len(), n).into(),
                    );
                }
                DeviceBuffer::from_host(&stream, init)?
            }
            None => DeviceBuffer::<f32>::zeroed(&stream, n)?,
        };
        let m = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let v = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let grad = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let loss_acc = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        let hist = DeviceBuffer::<u64>::zeroed(&stream, 8)?;

        Ok(Self {
            stream,
            module,
            weights,
            m,
            v,
            grad,
            loss_acc,
            hist,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        })
    }

    /// `loss_acc` / `hist` を 0 に reset する (epoch 開始時 / log 区間切り替え時)。
    fn zero_loss_hist(&mut self) -> gpu_runtime::Result<()> {
        // `DeviceBuffer<T>::zeroed` で作り直すのが一番素直 (memset より移植性が高い)。
        self.loss_acc = DeviceBuffer::<f64>::zeroed(&self.stream, 1)?;
        self.hist = DeviceBuffer::<u64>::zeroed(&self.stream, 8)?;
        Ok(())
    }

    /// 1 step (= 1 batch 分の forward → grad/loss/hist accumulate → adam_step) を実行する。
    fn step(&mut self, batch: &Batch, lr: f32) -> Result<(), Box<dyn std::error::Error>> {
        let n_pos = batch.n_positions;
        if n_pos == 0 {
            return Ok(());
        }

        // 入力 buffer は per-step に新規 alloc。perf 計測時に `Stream::memcpy_h2d`
        // を直接呼んで再利用化する想定 (TODO)。
        let indices_dev = DeviceBuffer::from_host(&self.stream, &batch.indices)?;
        let targets_dev = DeviceBuffer::from_host(&self.stream, &batch.targets)?;
        let norm_dev = DeviceBuffer::from_host(&self.stream, &batch.per_pos_norm)?;
        let mut preds_dev = DeviceBuffer::<f32>::zeroed(&self.stream, n_pos)?;

        let n_pos_u32 = n_pos as u32;
        let max_inds_u32 = MAX_INDS_PER_POS as u32;
        let grid_pos = grid_dim_1d(n_pos, BLOCK_DIM);

        let cfg_pos = LaunchConfig {
            grid_dim: grid_pos,
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };

        // Forward: preds[pos] = sigmoid(Σ weights[idx[base+j]])
        cuda_launch! {
            kernel: forward,
            stream: self.stream,
            module: self.module,
            config: cfg_pos,
            args: [
                slice(indices_dev),
                slice(self.weights),
                slice_mut(preds_dev),
                n_pos_u32,
                max_inds_u32
            ]
        }?;

        // Backward: gscale → grad[idx] (atomic) + loss_acc + hist
        cuda_launch! {
            kernel: grad,
            stream: self.stream,
            module: self.module,
            config: cfg_pos,
            args: [
                slice(indices_dev),
                slice(preds_dev),
                slice(targets_dev),
                slice(norm_dev),
                slice(self.grad),
                slice(self.loss_acc),
                slice(self.hist),
                n_pos_u32,
                max_inds_u32
            ]
        }?;

        // Adam step
        self.beta1_pow *= ADAM_BETA1;
        self.beta2_pow *= ADAM_BETA2;
        let bc1 = 1.0_f32 - self.beta1_pow;
        let bc2 = 1.0_f32 - self.beta2_pow;
        let beta1 = ADAM_BETA1;
        let beta2 = ADAM_BETA2;
        let eps = ADAM_EPS;
        let n_w = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
        let n_w_u32 = n_w as u32;
        let cfg_w = LaunchConfig {
            grid_dim: grid_dim_1d(n_w, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: adam_step,
            stream: self.stream,
            module: self.module,
            config: cfg_w,
            args: [
                slice_mut(self.weights),
                slice_mut(self.m),
                slice_mut(self.v),
                slice_mut(self.grad),
                lr,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                n_w_u32
            ]
        }?;

        self.stream.synchronize()?;
        Ok(())
    }

    /// 評価 path: forward → eval kernel (loss + histogram のみ、weight 不変)。
    #[allow(dead_code)]
    fn eval_forward(&mut self, batch: &Batch) -> Result<(), Box<dyn std::error::Error>> {
        let n_pos = batch.n_positions;
        if n_pos == 0 {
            return Ok(());
        }

        let indices_dev = DeviceBuffer::from_host(&self.stream, &batch.indices)?;
        let targets_dev = DeviceBuffer::from_host(&self.stream, &batch.targets)?;
        let mut preds_dev = DeviceBuffer::<f32>::zeroed(&self.stream, n_pos)?;

        let n_pos_u32 = n_pos as u32;
        let max_inds_u32 = MAX_INDS_PER_POS as u32;
        let cfg_pos = LaunchConfig {
            grid_dim: grid_dim_1d(n_pos, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };

        cuda_launch! {
            kernel: forward,
            stream: self.stream,
            module: self.module,
            config: cfg_pos,
            args: [
                slice(indices_dev),
                slice(self.weights),
                slice_mut(preds_dev),
                n_pos_u32,
                max_inds_u32
            ]
        }?;
        cuda_launch! {
            kernel: eval,
            stream: self.stream,
            module: self.module,
            config: cfg_pos,
            args: [
                slice(preds_dev),
                slice(targets_dev),
                slice(self.loss_acc),
                slice(self.hist),
                n_pos_u32
            ]
        }?;

        self.stream.synchronize()?;
        Ok(())
    }

    fn read_loss_hist(&self) -> gpu_runtime::Result<(f64, [u64; 8])> {
        let loss_vec = self.loss_acc.to_host_vec(&self.stream)?;
        let hist_vec = self.hist.to_host_vec(&self.stream)?;
        let mut hist_arr = [0_u64; 8];
        hist_arr.copy_from_slice(&hist_vec[..8]);
        Ok((loss_vec[0], hist_arr))
    }

    fn read_weights(&self) -> gpu_runtime::Result<Vec<f32>> {
        self.weights.to_host_vec(&self.stream).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Training driver
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct EpochStats {
    samples: usize,
    games: usize,
    steps: usize,
    mean_loss: f64,
    bucket_hist: [u64; 8],
}

/// 1 file 内の game を順次読んで batch を組み、`step` を呼ぶ。
fn train_one_epoch(
    trainer: &mut GpuTrainer,
    data_files: &[PathBuf],
    args: &Args,
    lr: f32,
    epoch: usize,
) -> Result<EpochStats, Box<dyn std::error::Error>> {
    trainer.zero_loss_hist()?;

    let max_games = if args.max_games > 0 {
        Some(args.max_games)
    } else {
        None
    };

    let mut batch = Batch::new();
    let mut scratch: Vec<usize> = Vec::with_capacity(96);
    let mut samples_total = 0_usize;
    let mut games_total = 0_usize;
    let mut steps = 0_usize;
    let start = Instant::now();

    'outer: for path in data_files {
        let cursor = PackCursor::open(path)?;
        let mut gi = GameIterator::new(cursor);
        while let Some(game) = gi.next_game()? {
            if game.is_empty() {
                continue;
            }
            if let Some(limit) = max_games
                && games_total + batch.n_games >= limit
            {
                break 'outer;
            }
            batch.push_game(&game, &mut scratch);
            if batch.n_games >= args.games_per_step {
                batch.finalize();
                games_total += batch.n_games;
                samples_total += batch.n_positions;
                trainer.step(&batch, lr)?;
                steps += 1;
                batch.clear();

                if args.log_interval_steps > 0 && steps.is_multiple_of(args.log_interval_steps) {
                    let (loss_sum, _) = trainer.read_loss_hist()?;
                    let avg = if samples_total > 0 {
                        loss_sum / samples_total as f64
                    } else {
                        0.0
                    };
                    let elapsed = start.elapsed().as_secs_f64();
                    let games_per_sec = games_total as f64 / elapsed.max(1e-9);
                    println!(
                        "epoch {} steps {} games {} samples {} avg_loss {:.6} games/s {:.0}",
                        epoch, steps, games_total, samples_total, avg, games_per_sec
                    );
                }
            }
        }
    }
    // 残り (n_games < games_per_step) も 1 step として処理。
    if batch.n_games > 0 {
        batch.finalize();
        games_total += batch.n_games;
        samples_total += batch.n_positions;
        trainer.step(&batch, lr)?;
        steps += 1;
    }

    let (loss_sum, hist) = trainer.read_loss_hist()?;
    let mean_loss = if samples_total > 0 {
        loss_sum / samples_total as f64
    } else {
        0.0
    };
    Ok(EpochStats {
        samples: samples_total,
        games: games_total,
        steps,
        mean_loss,
        bucket_hist: hist,
    })
}

fn run_training(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.epochs == 0 {
        return Err("--epochs must be >= 1".into());
    }
    if args.games_per_step == 0 {
        return Err("--games-per-step must be >= 1".into());
    }

    let data_paths = args.data_paths();
    if data_paths.is_empty() {
        return Err(
            "--data is required (comma-separated PSV files). 引数省略は scaffold 時のみ".into(),
        );
    }
    for p in &data_paths {
        if !p.exists() {
            return Err(format!("data file not found: {}", p.display()).into());
        }
    }

    let init_weights = args
        .init_from
        .as_deref()
        .map(read_progress_bin)
        .transpose()?;

    let ctx = CudaContext::new(args.device)?;
    println!(
        "CUDA device {} ready, kernel module loading...",
        args.device
    );
    let mut trainer = GpuTrainer::new(&ctx, init_weights.as_deref())?;
    println!(
        "GpuTrainer ready: {} weights, batch={} games, lr={} (effective={})",
        SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
        args.games_per_step,
        args.lr,
        args.effective_lr()
    );

    let lr = args.effective_lr();
    let mut last_stats: EpochStats = EpochStats::default();
    for epoch in 1..=args.epochs {
        let stats = train_one_epoch(&mut trainer, &data_paths, &args, lr, epoch)?;
        println!(
            "EPOCH {} DONE: games={} samples={} steps={} mean_loss={:.6} hist={:?}",
            epoch, stats.games, stats.samples, stats.steps, stats.mean_loss, stats.bucket_hist
        );
        last_stats = stats;
    }
    let _ = last_stats; // 後続で使う予定なら拡張、現状は EOI 用に保持

    let weights = trainer.read_weights()?;
    write_progress_bin(&args.output, &weights)?;
    println!(
        "wrote progress.bin: {} ({} weights, {} bytes)",
        args.output.display(),
        weights.len(),
        weights.len() * std::mem::size_of::<f64>()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run_training(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

// ---------------------------------------------------------------------------
// GPU ↔ CPU reference 数値同等性テスト + samples/sec ベンチ
// ---------------------------------------------------------------------------
//
// 本 module は **GPU 必須**。`#[cfg(test)]` で main.rs 内に置くことで kernel
// symbol (forward / grad / adam_step / eval) に直接 path 解決できる (lib.rs
// 経由では bin のみに存在する `#[kernel]` に届かないため `tests/*.rs` からは
// 呼び出せない)。
//
// 走らせる:
//
// ```bash
// cd bins/progress_kpabs_train
// CUDA_OXIDE_TARGET=sm_75 cargo-oxide build  # .ll 生成
// cargo test -p progress-kpabs-train --bin progress-kpabs-train --release \
//     -- --test-threads=1
// ```
//
// `--test-threads=1` は CudaContext を複数 test 間で共有しないため (各 test が
// 独自に context を作る前提)。`--release` 推奨だが debug build でも動く。
#[cfg(test)]
mod gpu_cpu_equivalence_tests {
    use super::*;
    use gpu_kernels::progress::adam_step::adam_step_cpu;
    use gpu_kernels::progress::eval::eval_cpu;
    use gpu_kernels::progress::forward::forward_cpu;
    use gpu_kernels::progress::grad::grad_cpu;

    /// f32 atomic 加算は順序非決定で、特に scatter 経路の `grad[idx]` に対する
    /// 多 thread 加算は CPU reference (single-thread, in-order) と完全一致しない。
    /// f32 add の round-off は ~1e-7 のオーダーで、本 test の小規模 batch
    /// (≤ 32 positions、≤ 80 indices) なら 1e-5 以内に収まる。
    const FLOAT_TOL: f32 = 1e-5;
    const F64_TOL: f64 = 1e-8;

    /// 決定論的な小規模 batch を組む。
    ///
    /// `n_pos` 個の position、`max_inds` 個の active index per position、
    /// `n_weights` 個の weight。`indices[pos*max_inds + j]` は `(pos+j) % n_weights`
    /// を入れ (一部 padding `-1` も混ぜる)、targets は `pos as f32 / n_pos as f32`、
    /// per_pos_norm は 1.0 で固定する。
    ///
    /// 戻り値は `(indices, weights, targets, per_pos_norm)`。
    fn build_fixed_inputs(
        n_pos: usize,
        max_inds: usize,
        n_weights: usize,
    ) -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut indices = Vec::with_capacity(n_pos * max_inds);
        for pos in 0..n_pos {
            for j in 0..max_inds {
                if j == max_inds - 1 && pos % 3 == 0 {
                    indices.push(-1); // 一部 padding
                } else {
                    indices.push(((pos + j) % n_weights) as i32);
                }
            }
        }
        let weights: Vec<f32> = (0..n_weights).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let targets: Vec<f32> = (0..n_pos)
            .map(|i| (i as f32) / (n_pos.max(1) as f32))
            .collect();
        let per_pos_norm: Vec<f32> = vec![1.0_f32; n_pos];
        (indices, weights, targets, per_pos_norm)
    }

    type CudaCtxModuleStream = (
        std::sync::Arc<CudaContext>,
        std::sync::Arc<CudaModule>,
        std::sync::Arc<CudaStream>,
    );

    /// `tests/host_smoke.rs` 同様、kernel module を loader 経由で読み込む。
    fn open_module() -> Result<CudaCtxModuleStream, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(&ctx, "progress_kpabs_train")?;
        Ok((ctx, module, stream))
    }

    #[test]
    fn forward_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n_pos = 16;
        let max_inds = 8;
        let n_weights = 64;
        let (indices, weights, _targets, _norm) = build_fixed_inputs(n_pos, max_inds, n_weights);

        // CPU reference
        let preds_cpu = forward_cpu(&indices, &weights, n_pos, max_inds);

        // GPU
        let indices_dev = DeviceBuffer::from_host(&stream, &indices)?;
        let weights_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut preds_dev = DeviceBuffer::<f32>::zeroed(&stream, n_pos)?;
        let n_pos_u32 = n_pos as u32;
        let max_inds_u32 = max_inds as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n_pos, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: forward,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(indices_dev), slice(weights_dev), slice_mut(preds_dev), n_pos_u32, max_inds_u32]
        }?;
        stream.synchronize()?;
        let preds_gpu = preds_dev.to_host_vec(&stream)?;

        assert_eq!(preds_cpu.len(), preds_gpu.len());
        for (i, (g, c)) in preds_gpu.iter().zip(preds_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff < FLOAT_TOL,
                "preds[{i}]: gpu={g} cpu={c} diff={diff} > {FLOAT_TOL}"
            );
        }
        Ok(())
    }

    #[test]
    fn grad_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n_pos = 16;
        let max_inds = 8;
        let n_weights = 64;
        let (indices, weights, targets, per_pos_norm) =
            build_fixed_inputs(n_pos, max_inds, n_weights);
        // GPU は forward を先に走らせて preds を作る (grad は preds を入力に取る)。
        let preds_cpu = forward_cpu(&indices, &weights, n_pos, max_inds);

        // CPU reference: grad/loss/hist
        let mut grad_cpu_out = vec![0.0_f32; n_weights];
        let mut loss_cpu = 0.0_f64;
        let mut hist_cpu = [0_u64; 8];
        grad_cpu(
            &indices,
            &preds_cpu,
            &targets,
            &per_pos_norm,
            &mut grad_cpu_out,
            &mut loss_cpu,
            &mut hist_cpu,
            n_pos,
            max_inds,
        );

        // GPU: 同じ preds を host から流し込む (kernel の forward path 経由でなく、
        // CPU と入力を厳密に一致させるため direct upload)
        let indices_dev = DeviceBuffer::from_host(&stream, &indices)?;
        let preds_dev = DeviceBuffer::from_host(&stream, &preds_cpu)?;
        let targets_dev = DeviceBuffer::from_host(&stream, &targets)?;
        let norm_dev = DeviceBuffer::from_host(&stream, &per_pos_norm)?;
        let grad_dev = DeviceBuffer::<f32>::zeroed(&stream, n_weights)?;
        let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        let hist_dev = DeviceBuffer::<u64>::zeroed(&stream, 8)?;
        let n_pos_u32 = n_pos as u32;
        let max_inds_u32 = max_inds as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n_pos, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: grad,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice(indices_dev), slice(preds_dev), slice(targets_dev),
                slice(norm_dev), slice(grad_dev), slice(loss_dev), slice(hist_dev),
                n_pos_u32, max_inds_u32
            ]
        }?;
        stream.synchronize()?;
        let grad_gpu = grad_dev.to_host_vec(&stream)?;
        let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
        let hist_gpu = hist_dev.to_host_vec(&stream)?;

        // grad: 多 thread atomic で順序非決定だが、本 test は max_inds=8、n_pos=16
        // で衝突が少なく f32 add の round-off は十分小さい。
        for (i, (g, c)) in grad_gpu.iter().zip(grad_cpu_out.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(diff < FLOAT_TOL, "grad[{i}]: gpu={g} cpu={c} diff={diff}");
        }
        // loss: f64 atomic で精度十分
        assert!(
            (loss_gpu - loss_cpu).abs() < F64_TOL,
            "loss: gpu={loss_gpu} cpu={loss_cpu} diff={}",
            (loss_gpu - loss_cpu).abs()
        );
        // hist: u64 atomic で完全一致
        assert_eq!(&hist_gpu[..], &hist_cpu[..], "hist mismatch");
        Ok(())
    }

    #[test]
    fn eval_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n_pos = 24;
        let preds_host: Vec<f32> = (0..n_pos).map(|i| (i as f32) / (n_pos as f32)).collect();
        let targets_host: Vec<f32> = (0..n_pos)
            .map(|i| ((i + 3) as f32) / (n_pos as f32))
            .collect();

        let mut loss_cpu = 0.0_f64;
        let mut hist_cpu = [0_u64; 8];
        eval_cpu(
            &preds_host,
            &targets_host,
            &mut loss_cpu,
            &mut hist_cpu,
            n_pos,
        );

        let preds_dev = DeviceBuffer::from_host(&stream, &preds_host)?;
        let targets_dev = DeviceBuffer::from_host(&stream, &targets_host)?;
        let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        let hist_dev = DeviceBuffer::<u64>::zeroed(&stream, 8)?;
        let n_pos_u32 = n_pos as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n_pos, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: eval,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(preds_dev), slice(targets_dev), slice(loss_dev), slice(hist_dev), n_pos_u32]
        }?;
        stream.synchronize()?;
        let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
        let hist_gpu = hist_dev.to_host_vec(&stream)?;

        assert!(
            (loss_gpu - loss_cpu).abs() < F64_TOL,
            "loss: gpu={loss_gpu} cpu={loss_cpu}"
        );
        assert_eq!(&hist_gpu[..], &hist_cpu[..], "hist mismatch");
        Ok(())
    }

    #[test]
    fn adam_step_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 32;
        let weights_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let m_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let v_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
        let grad_host: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        let lr = 0.001_f32;
        let beta1 = ADAM_BETA1;
        let beta2 = ADAM_BETA2;
        let eps = ADAM_EPS;
        let bc1 = 1.0_f32 - beta1; // step 1
        let bc2 = 1.0_f32 - beta2;

        // CPU reference
        let mut w_cpu = weights_host.clone();
        let mut m_cpu = m_host.clone();
        let mut v_cpu = v_host.clone();
        let mut g_cpu = grad_host.clone();
        adam_step_cpu(
            &mut w_cpu, &mut m_cpu, &mut v_cpu, &mut g_cpu, lr, beta1, beta2, eps, bc1, bc2, n,
        );

        // GPU
        let mut w_dev = DeviceBuffer::from_host(&stream, &weights_host)?;
        let mut m_dev = DeviceBuffer::from_host(&stream, &m_host)?;
        let mut v_dev = DeviceBuffer::from_host(&stream, &v_host)?;
        let mut g_dev = DeviceBuffer::from_host(&stream, &grad_host)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: adam_step,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice_mut(w_dev), slice_mut(m_dev), slice_mut(v_dev), slice_mut(g_dev),
                lr, beta1, beta2, eps, bc1, bc2, n_u32
            ]
        }?;
        stream.synchronize()?;
        let w_gpu = w_dev.to_host_vec(&stream)?;
        let m_gpu = m_dev.to_host_vec(&stream)?;
        let v_gpu = v_dev.to_host_vec(&stream)?;
        let g_gpu = g_dev.to_host_vec(&stream)?;

        for i in 0..n {
            let dw = (w_gpu[i] - w_cpu[i]).abs();
            let dm = (m_gpu[i] - m_cpu[i]).abs();
            let dv = (v_gpu[i] - v_cpu[i]).abs();
            let dg = (g_gpu[i] - g_cpu[i]).abs();
            assert!(
                dw < FLOAT_TOL,
                "w[{i}]: gpu={} cpu={} diff={dw}",
                w_gpu[i],
                w_cpu[i]
            );
            assert!(
                dm < FLOAT_TOL,
                "m[{i}]: gpu={} cpu={} diff={dm}",
                m_gpu[i],
                m_cpu[i]
            );
            assert!(
                dv < FLOAT_TOL,
                "v[{i}]: gpu={} cpu={} diff={dv}",
                v_gpu[i],
                v_cpu[i]
            );
            assert!(
                dg < FLOAT_TOL,
                "g[{i}]: gpu={} cpu={} diff={dg}",
                g_gpu[i],
                g_cpu[i]
            );
        }
        Ok(())
    }

    /// 最小ベンチ: sample.psv の先頭ゲームから 4 games × 8 pos = 32 pos の batch を
    /// 組み、warm-up 1 step + 計測 50 steps の `samples/sec` を `println!` で記録。
    /// 手動検証用の baseline で、`> 1.0` だけ assert する loose check により環境差
    /// を吸収する。安定した PR 間 perf tracking には criterion 等の別 bench workflow
    /// が必要。
    #[test]
    fn samples_per_sec_baseline_on_sample_psv() -> Result<(), Box<dyn std::error::Error>> {
        let (ctx, _module, _stream) = open_module()?;
        let mut trainer = GpuTrainer::new(&ctx, None)?;

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/shogi-format/tests/data/sample.psv");
        let cursor = PackCursor::open(&path)?;
        let mut gi = GameIterator::new(cursor);
        let game = gi.next_game()?.expect("at least 1 game");
        if game.len() < 8 {
            // sample.psv の game が短すぎる場合は skip
            return Ok(());
        }

        let mut batch = Batch::new();
        let mut scratch = Vec::with_capacity(96);
        // 同じ game を 4 回 push して 1 batch (4 games) にする
        for _ in 0..4 {
            batch.push_game(&game[..8], &mut scratch);
        }
        batch.finalize();

        // warm-up 1 step
        trainer.step(&batch, 1e-3)?;

        let n_steps: usize = 50;
        let n_pos_per_step = batch.n_positions;
        let start = std::time::Instant::now();
        for _ in 0..n_steps {
            trainer.step(&batch, 1e-3)?;
        }
        let elapsed = start.elapsed();
        let total_samples = n_pos_per_step * n_steps;
        let samples_per_sec = total_samples as f64 / elapsed.as_secs_f64();
        println!(
            "samples_per_sec baseline: {} positions/batch × {} steps in {:.3?} → {:.0} samples/sec",
            n_pos_per_step, n_steps, elapsed, samples_per_sec
        );
        // 環境依存なので閾値は緩く: 1 sample/sec 以上ならまず動いている (sm_75 級
        // GPU では 4 games × 8 positions = 32 / step が ~1ms 以下の見込み)。
        assert!(
            samples_per_sec > 1.0,
            "samples/sec too low: {samples_per_sec}"
        );
        Ok(())
    }
}
