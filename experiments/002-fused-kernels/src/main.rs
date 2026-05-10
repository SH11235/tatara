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
//!   `ranger_lookahead_lerp`, `sparse_ft_forward`, `sparse_ft_backward`) は
//!   Stage 2-1〜2-7 で各 issue が本 file に inline で追加する。Stage 2-6 (#42)
//!   までで pointwise 5 件 + `sparse_ft_forward` の合計 6 件 landed (sparse
//!   backward は Stage 2-7 で追加予定)
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

/// Fused AdamW optimizer step (decay + clip 込み) — Stage 2-3 (#39)。
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは Stage 1-5 で確立した cuda-oxide rustc-codegen-cuda
/// backend の bin-entry 制約のため。GPU launch は `#[cfg(test)] mod
/// gpu_cpu_equivalence_tests` から `cuda_launch!` macro 経由で行う。
///
/// アルゴリズムと bullet-shogi 上流 (`crates/trainer/src/optimiser/adam.rs::
/// AdamWParams::build`) との対応 / divergence は reference CPU
/// (`gpu_kernels::pointwise::adamw_step::adamw_step_cpu`) の docstring および
/// `ATTRIBUTION.md` の Stage 2-3 entry を参照。
///
/// 1 thread = 1 weight、atomics 不要 (Stage 1 `progress::adam_step` と同型)。
/// in-place output: `weights / m / v / grad`。`grad[i]` は次 batch の atomic
/// 累積に向けて 0 にリセット (Stage 1 慣行)。
///
/// 引数数 (12) は bullet 上流の AdamW 引数 + Stage 1 同 convention のため
/// `clippy::too_many_arguments` を allow。
///
/// ## cuda-oxide 制限
///
/// - `f32::clamp` / `f32::max` / `f32::min` は lowering 失敗 (Stage 1-7 で確認、
///   `Symbol std__intrinsics__maximum_number_nsz_f32 not found`)。本 kernel では
///   bullet 上流 `min(max(p, WMIN), WMAX)` を **`if-else` ladder** で展開する
/// - `v.sqrt()` は cuda-oxide が `__nv_sqrtf` (libdevice) に lowering する
///   (Stage 1-7 で動作確認済)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn adamw_step(
    mut weights: DisjointSlice<f32>,
    mut m: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut grad: DisjointSlice<f32>,
    lr: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_w: f32,
    max_w: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }

    // 4 buffer すべて i 番目に対し 1 thread が排他的にアクセスするため atomics 不要。
    // get_mut が None になるのは host 側 invariant 違反 (len < n) のときのみで、
    // Stage 1-7 adam_step 同型の defensive pattern (silent skip) を踏襲。
    let g_opt = grad.get_mut(i);
    let m_opt = m.get_mut(i);
    let v_opt = v.get_mut(i);
    let w_opt = weights.get_mut(i);
    if let (Some(g_ref), Some(m_ref), Some(v_ref), Some(w_ref)) = (g_opt, m_opt, v_opt, w_opt) {
        let g = *g_ref;
        let mut p = *w_ref;
        p *= 1.0_f32 - decay * lr;
        let mi = beta1 * *m_ref + (1.0_f32 - beta1) * g;
        let vi = beta2 * *v_ref + (1.0_f32 - beta2) * g * g;
        *m_ref = mi;
        *v_ref = vi;
        let val = mi / (vi.sqrt() + eps);
        p -= lr * val;
        // f32::clamp(min_w, max_w) を if-else に展開 (cuda-oxide が f32::max を
        // 解決できないため。Stage 1-7 adam_step / Stage 2-1 screlu_grad と同型)。
        let p_clamped = if p < min_w {
            min_w
        } else if p > max_w {
            max_w
        } else {
            p
        };
        *w_ref = p_clamped;
        *g_ref = 0.0_f32;
    }
}

/// Fused RAdam optimizer step (AdamW + bias correction + denom switch) — Stage
/// 2-4 (#40)。
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは Stage 1-5 で確立した cuda-oxide rustc-codegen-cuda
/// backend の bin-entry 制約のため。GPU launch は `#[cfg(test)] mod
/// gpu_cpu_equivalence_tests` から `cuda_launch!` macro 経由で行う。
///
/// アルゴリズム + bullet 上流 (`crates/trainer/src/optimiser/radam.rs::RAdam::
/// update` + `OP` template) との対応 / divergence は reference CPU
/// (`gpu_kernels::pointwise::radam_step::radam_step_cpu` +
/// `radam_compute_step_size_denom`) の docstring および `ATTRIBUTION.md` の
/// Stage 2-4 entry を参照。
///
/// `step_size` と `denom` は host (loader) 側で `radam_compute_step_size_denom`
/// により step number から事前計算した scalar を `f32` / `i32` で値渡しする
/// (Stage 2-3 `adamw_step` と同 convention、bullet 上流の 1-element device
/// buffer は本リポでは見送り。ATTRIBUTION 参照)。
///
/// 1 thread = 1 weight、atomics 不要 (Stage 1 / Stage 2-3 と同型)。
///
/// 引数数 (14) は AdamW + RAdam 拡張のため `clippy::too_many_arguments` を allow。
///
/// ## cuda-oxide 制限
///
/// - `f32::clamp` / `f32::max` / `f32::min` lowering 失敗 → `if-else` ladder で
///   展開 (Stage 1-7 / 2-1 / 2-3 と同 workaround)
/// - `f32::sqrt` は `__nv_sqrtf` (libdevice) に lowering される
/// - `denom != 0` 比較は cuda-oxide で問題なく compile される (Stage 1-6 grad の
///   bin clamp `b < 0` 比較と同型)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn radam_step(
    mut weights: DisjointSlice<f32>,
    mut m: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut grad: DisjointSlice<f32>,
    lr: f32,
    step_size: f32,
    denom: i32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_w: f32,
    max_w: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }

    let g_opt = grad.get_mut(i);
    let m_opt = m.get_mut(i);
    let v_opt = v.get_mut(i);
    let w_opt = weights.get_mut(i);
    if let (Some(g_ref), Some(m_ref), Some(v_ref), Some(w_ref)) = (g_opt, m_opt, v_opt, w_opt) {
        let g = *g_ref;
        let rate = lr * step_size;
        let mut p = *w_ref;
        p *= 1.0_f32 - decay * rate;
        let mi = beta1 * *m_ref + (1.0_f32 - beta1) * g;
        let vi = beta2 * *v_ref + (1.0_f32 - beta2) * g * g;
        *m_ref = mi;
        *v_ref = vi;
        let mut val = mi;
        if denom != 0 {
            val /= vi.sqrt() + eps;
        }
        p -= rate * val;
        // f32::clamp(min_w, max_w) を if-else に展開 (Stage 1-7 / 2-3 と同 workaround)。
        let p_clamped = if p < min_w {
            min_w
        } else if p > max_w {
            max_w
        } else {
            p
        };
        *w_ref = p_clamped;
        *g_ref = 0.0_f32;
    }
}

/// Ranger Lookahead lerp — Stage 2-5 (#41)。
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは Stage 1-5 で確立した cuda-oxide rustc-codegen-cuda
/// backend の bin-entry 制約のため。GPU launch は `#[cfg(test)] mod
/// gpu_cpu_equivalence_tests` から `cuda_launch!` macro 経由で行う。
///
/// Ranger は **RAdam (Stage 2-4 `radam_step`) + Lookahead lerp (本 kernel)** の
/// 2 kernel pair として host orchestration で組み立てる。本 kernel は
/// `step % k == 0` のときのみ呼ばれる lerp 部分のみで、bullet 上流
/// `optimiser/ranger.rs::build_ranger_op` (`:27-46`) の PointwiseIR を
/// hand-fused 形に書き直したもの。
///
/// アルゴリズム:
///
/// ```text
/// per element i:
///     new_w = alpha * weights[i] + (1 - alpha) * slow[i]
///     weights[i] = new_w
///     slow[i]    = new_w        # weights / slow が同期
/// ```
///
/// 1 thread = 1 weight、atomics 不要、`+` / `*` のみで cuda-oxide 制限に
/// 当たらない (Stage 1-5 forward の `+ z` と同等の素直な pointwise op)。
///
/// 詳細は reference CPU (`gpu_kernels::pointwise::ranger_step::
/// ranger_lookahead_lerp_cpu`) の docstring および `ATTRIBUTION.md` の
/// Stage 2-5 entry を参照。
#[kernel]
pub fn ranger_lookahead_lerp(
    mut weights: DisjointSlice<f32>,
    mut slow: DisjointSlice<f32>,
    alpha: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }

    let one_minus_alpha = 1.0_f32 - alpha;
    let w_opt = weights.get_mut(i);
    let s_opt = slow.get_mut(i);
    if let (Some(w_ref), Some(s_ref)) = (w_opt, s_opt) {
        let new_w = alpha * *w_ref + one_minus_alpha * *s_ref;
        *w_ref = new_w;
        *s_ref = new_w;
    }
}

/// Sparse feature transform forward (HalfKA_hm 用) — Stage 2-6 (#42)。
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは Stage 1-5 で確立した cuda-oxide rustc-codegen-cuda
/// backend の bin-entry 制約のため。GPU launch は `#[cfg(test)] mod
/// gpu_cpu_equivalence_tests` から `cuda_launch!` macro 経由で行う。
///
/// アルゴリズムと bullet-shogi 上流 (`crates/compiler/src/tensor/operation/
/// linear/sparse.rs::SparseMatmul::evaluate`) との対応 / divergence は
/// reference CPU (`gpu_kernels::sparse::sparse_ft_forward::sparse_ft_forward_cpu`)
/// の docstring および `ATTRIBUTION.md` の Stage 2-6 entry を参照。
///
/// 1 thread = 1 (batch_index, row_index) tuple、flat 1D で `tid = bi * rows + ri`、
/// `bi = tid / rows`、`ri = tid % rows`。`weight` は **column-major** (`weight[idx
/// * rows + ri]`)、atomics 不要 (各 thread は別 output cell に書く)。`-1`
/// padding と `idx >= cols` の異常入力は silent skip (Stage 1-6 grad / bullet
/// 上流 と同型 defensive pattern)。
///
/// 引数数 (7) は bullet 上流 evaluate と同型のため `clippy::too_many_arguments`
/// を allow。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn sparse_ft_forward(
    weight: &[f32],
    indices: &[i32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    rows: u32,
    cols: u32,
    nnz: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (rows as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (rows as usize);
    let ri = tid.get() % (rows as usize);

    let mut sum = 0.0_f32;
    let base = bi * (nnz as usize);
    let mut ni: u32 = 0;
    while ni < nnz {
        let idx = indices[base + (ni as usize)];
        if idx >= 0 && (idx as u32) < cols {
            sum += weight[(idx as usize) * (rows as usize) + ri];
        }
        ni += 1;
    }
    if let Some(o) = out.get_mut(tid) {
        *o = sum;
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
    let kernel_names =
        "screlu_grad,loss_wdl,adamw_step,radam_step,ranger_lookahead_lerp,sparse_ft_forward";

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
         (Stage 2-6: pointwise 5 + sparse_ft_forward landed)"
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

    /// adamw_step: 1 step の GPU と CPU reference 数値同等性。1 thread = 1 weight、
    /// atomics 不要なので tolerance は 1e-6 (Stage 2-1 screlu_grad / Stage 1
    /// progress::adam_step と同型)。
    #[test]
    fn adamw_step_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        use gpu_kernels::pointwise::adamw_step::adamw_step_cpu;

        let (_ctx, module, stream) = open_module()?;
        let n = 1024_usize;

        // 決定論的な weights / m / v / grad を作る。weights を [-1, 1] に分布、
        // m / v も小さい初期値、grad に nontrivial な勾配を入れる。
        let mut weights = Vec::with_capacity(n);
        let mut m = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut grad = Vec::with_capacity(n);
        for i in 0..n {
            let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
            let t = (i as f32) / denom;
            weights.push(-1.0_f32 + 2.0_f32 * t);
            m.push(0.001_f32 * (i as f32 - 512.0));
            v.push(0.0001_f32 * (i as f32) + 1e-6_f32);
            grad.push(0.01_f32 * ((i as f32) - 256.0).sin());
        }

        let lr = 1e-3_f32;
        let decay = 0.01_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let min_w = -2.0_f32;
        let max_w = 2.0_f32;

        // CPU reference (本体を mutate するので clone)
        let mut weights_cpu = weights.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        let mut grad_cpu = grad.clone();
        adamw_step_cpu(
            &mut weights_cpu,
            &mut m_cpu,
            &mut v_cpu,
            &mut grad_cpu,
            lr,
            decay,
            beta1,
            beta2,
            eps,
            min_w,
            max_w,
            n,
        );

        // GPU
        let mut weights_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut m_dev = DeviceBuffer::from_host(&stream, &m)?;
        let mut v_dev = DeviceBuffer::from_host(&stream, &v)?;
        let mut grad_dev = DeviceBuffer::from_host(&stream, &grad)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: adamw_step,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice_mut(weights_dev),
                slice_mut(m_dev),
                slice_mut(v_dev),
                slice_mut(grad_dev),
                lr,
                decay,
                beta1,
                beta2,
                eps,
                min_w,
                max_w,
                n_u32
            ]
        }?;
        stream.synchronize()?;
        let weights_gpu = weights_dev.to_host_vec(&stream)?;
        let m_gpu = m_dev.to_host_vec(&stream)?;
        let v_gpu = v_dev.to_host_vec(&stream)?;
        let grad_gpu = grad_dev.to_host_vec(&stream)?;

        let tol = 1e-6_f32;
        for i in 0..n {
            let dw = (weights_gpu[i] - weights_cpu[i]).abs();
            let dm = (m_gpu[i] - m_cpu[i]).abs();
            let dv = (v_gpu[i] - v_cpu[i]).abs();
            let dg = (grad_gpu[i] - grad_cpu[i]).abs();
            assert!(
                dw < tol,
                "weights[{i}]: gpu={} cpu={} diff={dw}",
                weights_gpu[i],
                weights_cpu[i]
            );
            assert!(
                dm < tol,
                "m[{i}]: gpu={} cpu={} diff={dm}",
                m_gpu[i],
                m_cpu[i]
            );
            assert!(
                dv < tol,
                "v[{i}]: gpu={} cpu={} diff={dv}",
                v_gpu[i],
                v_cpu[i]
            );
            assert!(
                dg < tol,
                "grad[{i}]: gpu={} cpu={} diff={dg}",
                grad_gpu[i],
                grad_cpu[i]
            );
        }
        // grad は全部 0 に reset されているはず
        for (i, &g) in grad_gpu.iter().enumerate() {
            assert_eq!(g, 0.0_f32, "grad[{i}] not reset: got {g}");
        }
        Ok(())
    }

    /// adamw_step: clip の if-else 展開が GPU でも崩れないこと。
    /// weights = [+100, -100, 0] で clip range [-1, 1]、grad = 0、lr = 0、decay = 0
    /// → weights = [+1, -1, 0]
    #[test]
    fn adamw_step_kernel_clamps_extreme_weights() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let weights = vec![100.0_f32, -100.0, 0.5];
        let m = vec![0.0_f32; 3];
        let v = vec![0.0_f32; 3];
        let grad = vec![0.0_f32; 3];
        let n = 3_usize;

        let mut weights_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut m_dev = DeviceBuffer::from_host(&stream, &m)?;
        let mut v_dev = DeviceBuffer::from_host(&stream, &v)?;
        let mut grad_dev = DeviceBuffer::from_host(&stream, &grad)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: adamw_step,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice_mut(weights_dev),
                slice_mut(m_dev),
                slice_mut(v_dev),
                slice_mut(grad_dev),
                0.0_f32,            // lr
                0.0_f32,            // decay
                0.9_f32,            // beta1
                0.999_f32,          // beta2
                1e-8_f32,           // eps
                -1.0_f32,           // min_w
                1.0_f32,            // max_w
                n_u32
            ]
        }?;
        stream.synchronize()?;
        let weights_gpu = weights_dev.to_host_vec(&stream)?;
        assert_eq!(weights_gpu, vec![1.0_f32, -1.0, 0.5]);
        Ok(())
    }

    /// radam_step: 1 step の GPU と CPU reference 数値同等性。AdamW (Stage 2-3)
    /// に host pre-compute された `step_size` / `denom` を渡す形に拡張。1024 元、
    /// step=1000 (`denom = 1` 領域) でテスト。tolerance 1e-6 (Stage 2-3 同型)。
    #[test]
    fn radam_step_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        use gpu_kernels::pointwise::radam_step::{radam_compute_step_size_denom, radam_step_cpu};

        let (_ctx, module, stream) = open_module()?;
        let n = 1024_usize;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let n_sma_threshold = 5.0_f32;
        let step = 1000_u64;
        let (step_size, denom) = radam_compute_step_size_denom(step, beta1, beta2, n_sma_threshold);
        // step=1000 で variance 補正 ON (denom = 1) になっていること
        assert_eq!(denom, 1, "expected denom=1 (variance-on) at step=1000");

        // 決定論的 weights / m / v / grad
        let mut weights = Vec::with_capacity(n);
        let mut m = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut grad = Vec::with_capacity(n);
        for i in 0..n {
            let denom_t = if n > 1 { (n - 1) as f32 } else { 1.0 };
            let t = (i as f32) / denom_t;
            weights.push(-1.0_f32 + 2.0_f32 * t);
            m.push(0.001_f32 * (i as f32 - 512.0));
            v.push(0.0001_f32 * (i as f32) + 1e-6_f32);
            grad.push(0.01_f32 * ((i as f32) - 256.0).sin());
        }

        let lr = 1e-3_f32;
        let decay = 0.01_f32;
        let eps = 1e-8_f32;
        let min_w = -2.0_f32;
        let max_w = 2.0_f32;

        // CPU reference
        let mut weights_cpu = weights.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        let mut grad_cpu = grad.clone();
        radam_step_cpu(
            &mut weights_cpu,
            &mut m_cpu,
            &mut v_cpu,
            &mut grad_cpu,
            lr,
            step_size,
            denom,
            decay,
            beta1,
            beta2,
            eps,
            min_w,
            max_w,
            n,
        );

        // GPU
        let mut weights_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut m_dev = DeviceBuffer::from_host(&stream, &m)?;
        let mut v_dev = DeviceBuffer::from_host(&stream, &v)?;
        let mut grad_dev = DeviceBuffer::from_host(&stream, &grad)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: radam_step,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice_mut(weights_dev),
                slice_mut(m_dev),
                slice_mut(v_dev),
                slice_mut(grad_dev),
                lr,
                step_size,
                denom,
                decay,
                beta1,
                beta2,
                eps,
                min_w,
                max_w,
                n_u32
            ]
        }?;
        stream.synchronize()?;
        let weights_gpu = weights_dev.to_host_vec(&stream)?;
        let m_gpu = m_dev.to_host_vec(&stream)?;
        let v_gpu = v_dev.to_host_vec(&stream)?;
        let grad_gpu = grad_dev.to_host_vec(&stream)?;

        let tol = 1e-6_f32;
        for i in 0..n {
            let dw = (weights_gpu[i] - weights_cpu[i]).abs();
            let dm = (m_gpu[i] - m_cpu[i]).abs();
            let dv = (v_gpu[i] - v_cpu[i]).abs();
            let dg = (grad_gpu[i] - grad_cpu[i]).abs();
            assert!(
                dw < tol,
                "weights[{i}]: gpu={} cpu={} diff={dw}",
                weights_gpu[i],
                weights_cpu[i]
            );
            assert!(
                dm < tol,
                "m[{i}]: gpu={} cpu={} diff={dm}",
                m_gpu[i],
                m_cpu[i]
            );
            assert!(
                dv < tol,
                "v[{i}]: gpu={} cpu={} diff={dv}",
                v_gpu[i],
                v_cpu[i]
            );
            assert!(
                dg < tol,
                "grad[{i}]: gpu={} cpu={} diff={dg}",
                grad_gpu[i],
                grad_cpu[i]
            );
        }
        Ok(())
    }

    /// radam_step: 学習初期 (`denom = 0`、variance off) 経路の GPU↔CPU 等価性。
    /// denom=1 経路と同水準 (1024 元、`weights / m / v / grad` 全 buffer を CPU
    /// reference と比較、tolerance 1e-6) のテスト。kernel 内 `if denom != 0`
    /// 分岐が GPU でも CPU と同じ動作になることをガード。
    #[test]
    fn radam_step_kernel_denom_zero_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>>
    {
        use gpu_kernels::pointwise::radam_step::radam_step_cpu;

        let (_ctx, module, stream) = open_module()?;
        let n = 1024_usize;

        // 決定論的入力 (denom=1 経路と同じ生成式、grad / m / v に non-trivial な値)
        let mut weights = Vec::with_capacity(n);
        let mut m = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut grad = Vec::with_capacity(n);
        for i in 0..n {
            let denom_t = if n > 1 { (n - 1) as f32 } else { 1.0 };
            let t = (i as f32) / denom_t;
            weights.push(-1.0_f32 + 2.0_f32 * t);
            m.push(0.001_f32 * (i as f32 - 512.0));
            v.push(0.0001_f32 * (i as f32) + 1e-6_f32);
            grad.push(0.01_f32 * ((i as f32) - 256.0).sin());
        }

        let lr = 1e-3_f32;
        // step=1 相当の host pre-compute 値を直接渡す (denom=0、`1/sqrt(v)` off 経路)
        let step_size = 10.0_f32;
        let denom: i32 = 0;
        let decay = 0.01_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let min_w = -2.0_f32;
        let max_w = 2.0_f32;

        // CPU reference
        let mut weights_cpu = weights.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        let mut grad_cpu = grad.clone();
        radam_step_cpu(
            &mut weights_cpu,
            &mut m_cpu,
            &mut v_cpu,
            &mut grad_cpu,
            lr,
            step_size,
            denom,
            decay,
            beta1,
            beta2,
            eps,
            min_w,
            max_w,
            n,
        );

        // GPU
        let mut weights_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut m_dev = DeviceBuffer::from_host(&stream, &m)?;
        let mut v_dev = DeviceBuffer::from_host(&stream, &v)?;
        let mut grad_dev = DeviceBuffer::from_host(&stream, &grad)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: radam_step,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice_mut(weights_dev),
                slice_mut(m_dev),
                slice_mut(v_dev),
                slice_mut(grad_dev),
                lr,
                step_size,
                denom,
                decay,
                beta1,
                beta2,
                eps,
                min_w,
                max_w,
                n_u32
            ]
        }?;
        stream.synchronize()?;
        let weights_gpu = weights_dev.to_host_vec(&stream)?;
        let m_gpu = m_dev.to_host_vec(&stream)?;
        let v_gpu = v_dev.to_host_vec(&stream)?;
        let grad_gpu = grad_dev.to_host_vec(&stream)?;

        let tol = 1e-6_f32;
        for i in 0..n {
            let dw = (weights_gpu[i] - weights_cpu[i]).abs();
            let dm = (m_gpu[i] - m_cpu[i]).abs();
            let dv = (v_gpu[i] - v_cpu[i]).abs();
            let dg = (grad_gpu[i] - grad_cpu[i]).abs();
            assert!(
                dw < tol,
                "weights[{i}]: gpu={} cpu={} diff={dw}",
                weights_gpu[i],
                weights_cpu[i]
            );
            assert!(
                dm < tol,
                "m[{i}]: gpu={} cpu={} diff={dm}",
                m_gpu[i],
                m_cpu[i]
            );
            assert!(
                dv < tol,
                "v[{i}]: gpu={} cpu={} diff={dv}",
                v_gpu[i],
                v_cpu[i]
            );
            assert!(
                dg < tol,
                "grad[{i}]: gpu={} cpu={} diff={dg}",
                grad_gpu[i],
                grad_cpu[i]
            );
        }
        Ok(())
    }

    /// ranger_lookahead_lerp: GPU と CPU reference の数値同等性。1 thread = 1
    /// weight、atomics 不要、`+` / `*` のみの単純 pointwise なので tolerance は
    /// 1e-7 (Stage 2-1 screlu_grad 同型、scatter/atomic 経路無し)。
    #[test]
    fn ranger_lookahead_lerp_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>>
    {
        use gpu_kernels::pointwise::ranger_step::ranger_lookahead_lerp_cpu;

        let (_ctx, module, stream) = open_module()?;
        let n = 1024_usize;
        let alpha = 0.5_f32;

        // 決定論的 weights / slow を [-2, 2] にスパン (interior + 端点を踏む)
        let mut weights = Vec::with_capacity(n);
        let mut slow = Vec::with_capacity(n);
        for i in 0..n {
            let denom_t = if n > 1 { (n - 1) as f32 } else { 1.0 };
            let t = (i as f32) / denom_t;
            weights.push(-2.0_f32 + 4.0_f32 * t);
            slow.push(2.0_f32 - 4.0_f32 * t);
        }

        // CPU reference
        let mut weights_cpu = weights.clone();
        let mut slow_cpu = slow.clone();
        ranger_lookahead_lerp_cpu(&mut weights_cpu, &mut slow_cpu, alpha, n);

        // GPU
        let mut weights_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut slow_dev = DeviceBuffer::from_host(&stream, &slow)?;
        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(n, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: ranger_lookahead_lerp,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice_mut(weights_dev), slice_mut(slow_dev), alpha, n_u32]
        }?;
        stream.synchronize()?;
        let weights_gpu = weights_dev.to_host_vec(&stream)?;
        let slow_gpu = slow_dev.to_host_vec(&stream)?;

        let tol = 1e-7_f32;
        for i in 0..n {
            let dw = (weights_gpu[i] - weights_cpu[i]).abs();
            let ds = (slow_gpu[i] - slow_cpu[i]).abs();
            assert!(
                dw < tol,
                "weights[{i}]: gpu={} cpu={} diff={dw}",
                weights_gpu[i],
                weights_cpu[i]
            );
            assert!(
                ds < tol,
                "slow[{i}]: gpu={} cpu={} diff={ds}",
                slow_gpu[i],
                slow_cpu[i]
            );
            // post-condition: weights == slow (bullet 上流 build_ranger_op の
            // pntwise.write(w/s, ..., new_w) と同型の同期)
            assert_eq!(
                weights_gpu[i], slow_gpu[i],
                "post-lerp sync broken at i={i}"
            );
        }
        Ok(())
    }

    /// ranger_lookahead_lerp: alpha の端点 (1.0 で weights 維持・slow 同期、
    /// 0.0 で weights を slow に引き戻し) が GPU でも崩れないこと。
    #[test]
    fn ranger_lookahead_lerp_kernel_alpha_endpoints() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 4_usize;

        // alpha = 1.0: weights 維持、slow を weights と同期
        {
            let mut weights_dev = DeviceBuffer::from_host(&stream, &[1.0_f32, 2.0, 3.0, 4.0])?;
            let mut slow_dev = DeviceBuffer::from_host(&stream, &[10.0_f32, 20.0, 30.0, 40.0])?;
            let cfg = LaunchConfig {
                grid_dim: grid_dim_1d(n, BLOCK_DIM),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: stream,
                module: module,
                config: cfg,
                args: [slice_mut(weights_dev), slice_mut(slow_dev), 1.0_f32, n as u32]
            }?;
            stream.synchronize()?;
            let w_gpu = weights_dev.to_host_vec(&stream)?;
            let s_gpu = slow_dev.to_host_vec(&stream)?;
            assert_eq!(w_gpu, vec![1.0_f32, 2.0, 3.0, 4.0]);
            assert_eq!(s_gpu, vec![1.0_f32, 2.0, 3.0, 4.0]);
        }

        // alpha = 0.0: weights を slow に引き戻し
        {
            let mut weights_dev = DeviceBuffer::from_host(&stream, &[1.0_f32, 2.0, 3.0, 4.0])?;
            let mut slow_dev = DeviceBuffer::from_host(&stream, &[10.0_f32, 20.0, 30.0, 40.0])?;
            let cfg = LaunchConfig {
                grid_dim: grid_dim_1d(n, BLOCK_DIM),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: stream,
                module: module,
                config: cfg,
                args: [slice_mut(weights_dev), slice_mut(slow_dev), 0.0_f32, n as u32]
            }?;
            stream.synchronize()?;
            let w_gpu = weights_dev.to_host_vec(&stream)?;
            let s_gpu = slow_dev.to_host_vec(&stream)?;
            assert_eq!(w_gpu, vec![10.0_f32, 20.0, 30.0, 40.0]);
            assert_eq!(s_gpu, vec![10.0_f32, 20.0, 30.0, 40.0]);
        }
        Ok(())
    }

    /// sparse_ft_forward: GPU と CPU reference の数値同等性。bullet 上流
    /// (`linear/sparse.rs::tests::evaluate`) と同 shape (batch=2, rows=2,
    /// cols=3, nnz=4) で 1 ケース完全一致を assert。1 thread = 1 (batch, row)
    /// tuple、atomics 不要なので tolerance は 1e-7 (Stage 2-5
    /// ranger_lookahead_lerp 同型、scatter/atomic 経路無し)。
    /// 大きい shape の網羅は別 test (`*_larger_shape_matches_cpu` /
    /// `*_multi_block_boundary_matches_cpu`) で行う。
    #[test]
    fn sparse_ft_forward_kernel_matches_cpu_reference() -> Result<(), Box<dyn std::error::Error>> {
        use gpu_kernels::sparse::sparse_ft_forward::sparse_ft_forward_cpu;

        let (_ctx, module, stream) = open_module()?;

        // bullet 上流テストと同 shape (batch=2, rows=2, cols=3, nnz=4)
        let weight = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0_i32, 1, -1, -1, 2, 2, 1, 0];
        let batch = 2_u32;
        let rows = 2_u32;
        let cols = 3_u32;
        let nnz = 4_u32;
        let total_out = (batch as usize) * (rows as usize);

        // CPU reference
        let mut out_cpu = vec![0.0_f32; total_out];
        sparse_ft_forward_cpu(
            &weight,
            &indices,
            &mut out_cpu,
            batch as usize,
            rows as usize,
            cols as usize,
            nnz as usize,
        );
        // bullet 上流 expected = [2, 4, 10, 14]
        assert_eq!(out_cpu, vec![2.0_f32, 4.0, 10.0, 14.0]);

        // GPU
        let weight_dev = DeviceBuffer::from_host(&stream, &weight)?;
        let indices_dev = DeviceBuffer::from_host(&stream, &indices)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, total_out)?;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(total_out, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: sparse_ft_forward,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice(weight_dev),
                slice(indices_dev),
                slice_mut(out_dev),
                batch,
                rows,
                cols,
                nnz
            ]
        }?;
        stream.synchronize()?;
        let out_gpu = out_dev.to_host_vec(&stream)?;

        for (i, (g, c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(diff < 1e-7, "out[{i}]: gpu={g} cpu={c} diff={diff}");
        }
        Ok(())
    }

    /// sparse_ft_forward: 大きい shape (batch=8, rows=16, cols=64, nnz=12) +
    /// 決定論的入力で GPU と CPU の bit-equivalent 近い一致を確認。padding と
    /// out-of-range index も混ぜて defensive path を踏ませる。
    #[test]
    fn sparse_ft_forward_kernel_larger_shape_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
    {
        use gpu_kernels::sparse::sparse_ft_forward::sparse_ft_forward_cpu;

        let (_ctx, module, stream) = open_module()?;
        let batch = 8_u32;
        let rows = 16_u32;
        let cols = 64_u32;
        let nnz = 12_u32;

        // 決定論的 weight (column-major、各 column 1 つ目を i*0.01、2 つ目以降は
        // sin で値ばらけさせる)
        let mut weight = Vec::with_capacity((rows * cols) as usize);
        for col in 0..cols {
            for row in 0..rows {
                let v = ((col as f32) * 0.1) + ((row as f32) * 0.01);
                weight.push(v);
            }
        }
        // 決定論的 indices (一部 -1 padding、一部 cols 超過の defensive 入力)
        let mut indices = Vec::with_capacity((batch * nnz) as usize);
        for bi in 0..batch {
            for ni in 0..nnz {
                let raw = ((bi as i32) * 7 + (ni as i32) * 13) % 70 - 5;
                // raw が -5..-1 なら -1 padding に丸める / 65 以上は超過 (cols=64)
                let idx = if raw < 0 { -1_i32 } else { raw };
                indices.push(idx);
            }
        }
        let total_out = (batch as usize) * (rows as usize);

        let mut out_cpu = vec![0.0_f32; total_out];
        sparse_ft_forward_cpu(
            &weight,
            &indices,
            &mut out_cpu,
            batch as usize,
            rows as usize,
            cols as usize,
            nnz as usize,
        );

        let weight_dev = DeviceBuffer::from_host(&stream, &weight)?;
        let indices_dev = DeviceBuffer::from_host(&stream, &indices)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, total_out)?;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(total_out, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: sparse_ft_forward,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice(weight_dev),
                slice(indices_dev),
                slice_mut(out_dev),
                batch,
                rows,
                cols,
                nnz
            ]
        }?;
        stream.synchronize()?;
        let out_gpu = out_dev.to_host_vec(&stream)?;

        let tol = 1e-6_f32; // 12 加算でも f32 の round-off は 1e-6 内
        for (i, (g, c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(diff < tol, "out[{i}]: gpu={g} cpu={c} diff={diff}");
        }
        Ok(())
    }

    /// sparse_ft_forward: **multi-block boundary** を踏ませる shape での GPU↔CPU
    /// 等価性。`batch * rows = 17 * 16 = 272 > BLOCK_DIM = 256` で 2 block 跨ぐ
    /// (single-block でしか動かない GPU bug を catch するガード)。
    #[test]
    fn sparse_ft_forward_kernel_multi_block_boundary_matches_cpu()
    -> Result<(), Box<dyn std::error::Error>> {
        use gpu_kernels::sparse::sparse_ft_forward::sparse_ft_forward_cpu;

        let (_ctx, module, stream) = open_module()?;
        // 17 * 16 = 272 > BLOCK_DIM (256)、必ず 2 block 跨ぐ
        let batch = 17_u32;
        let rows = 16_u32;
        let cols = 64_u32;
        let nnz = 8_u32;

        let mut weight = Vec::with_capacity((rows * cols) as usize);
        for col in 0..cols {
            for row in 0..rows {
                weight.push(((col as f32) * 0.1) + ((row as f32) * 0.01));
            }
        }
        let mut indices = Vec::with_capacity((batch * nnz) as usize);
        for bi in 0..batch {
            for ni in 0..nnz {
                let raw = ((bi as i32) * 5 + (ni as i32) * 11) % 70 - 5;
                let idx = if raw < 0 { -1_i32 } else { raw };
                indices.push(idx);
            }
        }
        let total_out = (batch as usize) * (rows as usize);
        assert!(total_out > 256, "test must cross BLOCK_DIM=256 boundary");

        let mut out_cpu = vec![0.0_f32; total_out];
        sparse_ft_forward_cpu(
            &weight,
            &indices,
            &mut out_cpu,
            batch as usize,
            rows as usize,
            cols as usize,
            nnz as usize,
        );

        let weight_dev = DeviceBuffer::from_host(&stream, &weight)?;
        let indices_dev = DeviceBuffer::from_host(&stream, &indices)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, total_out)?;
        let cfg = LaunchConfig {
            grid_dim: grid_dim_1d(total_out, BLOCK_DIM),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        cuda_launch! {
            kernel: sparse_ft_forward,
            stream: stream,
            module: module,
            config: cfg,
            args: [
                slice(weight_dev),
                slice(indices_dev),
                slice_mut(out_dev),
                batch,
                rows,
                cols,
                nnz
            ]
        }?;
        stream.synchronize()?;
        let out_gpu = out_dev.to_host_vec(&stream)?;

        let tol = 1e-6_f32;
        for (i, (g, c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(diff < tol, "out[{i}]: gpu={g} cpu={c} diff={diff}");
        }
        Ok(())
    }
}
