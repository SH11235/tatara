//! `bins/nnue_train` binary entry point — bullet-shogi v102 互換 NNUE trainer。
//!
//! Stage 3-7 (#63) で本 file は **v102 LayerStack arch** の `#[kernel]` 群 (27 個、
//! うち Stage 3 #84 で `loss_wrm` 追加) と host loop driver (GpuTrainer) を統合する。
//! Stage 3-8 (#65) で CLI + trainer
//! integrate、Stage 3-9 (#64) で自己対局検証 (rshogi-oss engine 2 個) で完結する
//! 予定。
//!
//! ## v102 アーキテクチャ (LayerStack 1536-16-32 + progress8kpabs 9 buckets)
//!
//! bullet `examples/shogi_layerstack.rs:2206-2289` の reference 実装を Rust +
//! cuda-oxide で再現。PSQT 無し、hand_count_dense 無し。
//!
//! - **L0 (FT)**: sparse_ft_forward — weight (73305 × 1536), bias (1536, 共有)
//! - **per-perspective post**: bias add → CReLU → pairwise_mul (1536→768) → ×127/128
//! - **combined**: stm.concat(nstm) → 1536
//! - **L1 (per-bucket)**: weight (9×16, 1536) + bias (9×16) → select(bucket) → 16
//! - **L1f (shared)**: weight (1536, 16) + bias (16) → 16
//! - **l1_out_t**: L1_select + L1f → 16; slice → l1_main (15) + l1_skip (1)
//! - **l1_sqr**: l1_main^2 * 127/128 → 15
//! - **l2_input**: CReLU(concat(l1_sqr, l1_main)) → 30
//! - **L2 (per-bucket)**: weight (9×32, 30) + bias (9×32) → select(bucket) → CReLU → 32
//! - **L3 (per-bucket)**: weight (9×32) + bias (9) → select(bucket) → 1
//! - **net_output**: l3_out + l1_skip → 1 scalar
//!
//! ## kernel 一覧 (27 個、bin entry reachability のため全て本 file に inline)
//!
//! ### STATED (Stage 2 #46-#52 で landed、本 bin に inline copy)
//! 1. `screlu_grad` — v102 では未使用、compile-reach のため preserve
//! 2. `loss_wdl` — sigmoid-MSE 損失 + dy_net_output 勾配 (`out ≈ cp` で収束)
//! 3. `adamw_step` — v102 では未使用、preserve
//! 4. `radam_step` — Ranger 1/2 (RAdam step)
//! 5. `ranger_lookahead_lerp` — Ranger 2/2 (Lookahead lerp)
//! 6. `sparse_ft_forward` — L0 forward
//! 7. `sparse_ft_backward` — L0 backward (atomic scatter)
//!
//! ### NEW (Stage 3-7 で v102 arch のため追加)
//! 8. `ft_post_perspective_fwd` (FUSED: bias+CReLU+pairwise+scale)
//! 9. `ft_post_perspective_grad` (FUSED: 上記 gradient + ft_bias grad)
//! 10. `dense_mm_fwd` — regular dense matmul + bias
//! 11. `dense_mm_bwd_input` — input grad
//! 12. `dense_mm_bwd_weight` — weight grad (1 thread = 1 cell)
//! 13. `bias_grad` — generic atomic accumulate
//! 14. `dense_mm_fwd_bucket` — per-bucket dense matmul + bias + select
//! 15. `dense_mm_bwd_input_bucket` — per-bucket input grad
//! 16. `dense_mm_bwd_weight_bucket` — per-bucket weight grad (1 thread = 1 (bucket,o,i) cell + batch loop、atomic 不要)
//! 17. `bias_grad_bucket` — per-bucket bias grad (atomic per (bucket, o))
//! 18. `crelu_fwd` — clip 0-1
//! 19. `crelu_grad` — 1 if 0<x<1 else 0
//! 20. `abs_pow2_scale_fwd` — y = x*x*scale (abs_pow(2) は |x|^2 = x^2 なので abs 不要)
//! 21. `abs_pow2_scale_grad` — dx = 2*x*scale*dy
//! 22. `concat_l1sqr_main_fwd` — concat 15+15 → 30
//! 23. `concat_l1sqr_main_grad` — split 30 → 15+15
//! 24. `elementwise_add` — a+b → c (forward + grad-copy 両用)
//! 25. `slice_extract_2d` — 2D row 範囲を切り出し (l1_main / l1_skip の slice_rows)
//! 26. `slice_scatter_2d` — 2D row 範囲へ書き戻し (l1_main / l1_skip slice の backward)
//!
//! ### NEW (Stage 3 #84 で bullet v102 厳密再現のため追加)
//! 27. `loss_wrm` — bullet win-rate-model 損失 + dy_net_output 勾配 (`out ≈ cp / nnue2score`
//!     で収束、`--win-rate-model` 指定時に `loss_wdl` の代わりに使う。CPU reference は
//!     `gpu_kernels::pointwise::loss_wrm::loss_wrm_cpu`)
//!
//! ## cuda-oxide 制限への対応 (Stage 1-5〜2-7 で確立)
//!
//! - `f32::clamp` / `f32::max` / `f32::min` lowering 失敗 → `if-else` ladder で展開
//! - `i32::clamp` も同様 (Debug::fmt panic 経路を含む)
//! - `f32::sqrt`, `f32::exp` は libdevice (`__nv_sqrtf`, `__nv_expf`) に lowering OK
//! - atomic add パターン: `unsafe { &*(slice.as_ptr().add(idx) as *const DeviceAtomicX) }
//!   .fetch_add(_, AtomicOrdering::Relaxed)`
//!
//! kernel_names list (`compile_ll_to_ptx_via_llc` に渡す、計 27 個):
//! `sparse_ft_forward,sparse_ft_backward,loss_wdl,loss_wrm,screlu_grad,adamw_step,radam_step,
//! ranger_lookahead_lerp,ft_post_perspective_fwd,ft_post_perspective_grad,dense_mm_fwd,
//! dense_mm_bwd_input,dense_mm_bwd_weight,bias_grad,dense_mm_fwd_bucket,
//! dense_mm_bwd_input_bucket,dense_mm_bwd_weight_bucket,bias_grad_bucket,crelu_fwd,
//! crelu_grad,abs_pow2_scale_fwd,abs_pow2_scale_grad,concat_l1sqr_main_fwd,
//! concat_l1sqr_main_grad,elementwise_add,slice_extract_2d,slice_scatter_2d`

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64};
use cuda_device::{DisjointSlice, kernel, thread};
#[allow(unused_imports)]
use cuda_host::cuda_launch;
#[allow(unused_imports)]
use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
#[allow(unused_imports)]
use nnue_format::V102Weights;
use nnue_train::dataloader::Batch;
#[allow(unused_imports)]
use nnue_train::optimizer::radam_compute_step_size_denom;
use nnue_train::schedule::{ConstantWDL, StepLR};
use nnue_train::trainer::{LossKind, TrainerBackend, TrainingConfig};
use shogi_features::progress_kpabs::ShogiProgressKPAbs;

// ===========================================================================
// STATED kernels — Stage 2 (PR #46-#52) で landed、本 bin に inline copy
// ===========================================================================

/// SCReLU activation gradient (fused) — Stage 2-1 (#37 / PR #46) に inline 配置。
///
/// 本 v102 path では **未使用** (CReLU + pairwise_mul を使うため)。compile-reach
/// 用に preserve (cuda-oxide の bin-entry constraint、Stage 1-5 で確立)。
///
/// 1 thread = 1 element、atomics 不要、in-place output (`dl_dx`)。
#[kernel]
pub fn screlu_grad(x: &[f32], dl_dy: &[f32], mut dl_dx: DisjointSlice<f32>, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let xi = x[i.get()];
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

/// Sigmoid + WDL blend + scale loss kernel — Stage 2-2 (#38 / PR #47) inline。
///
/// 1 thread = 1 position。`dl_dout` は 1 thread = 1 index で排他更新 (atomics 不要)、
/// `loss_acc` は f64 単一 cell の Σ err^2 で `DeviceAtomicF64::fetch_add`。
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

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell 確保済 (Stage 1-6 / Stage 2-2 と同型)。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);
}

/// bullet win-rate-model (WRM) loss kernel — Stage 3 (#84) inline。
///
/// bullet `examples/shogi_layerstack.rs:2177-2188` (`loss_fn_wrm`、`--win-rate-model`
/// + `--wrm-in-scaling` 指定時に選ばれる loss closure) + `crates/bullet_lib/src/value/
/// loader.rs:300-316` (data-layer の WRM target + WDL blend) を NNUE 専用に hand-fuse。
/// CPU reference は `gpu_kernels::pointwise::loss_wrm::loss_wrm_cpu`。
///
/// `loss_wdl` (`p = sigmoid(out * scale)` で `out ≈ cp` で収束) と違い、prediction /
/// target 双方に nodchip 流 WRM を適用するため net_output は `out ≈ cp / nnue2score`
/// (O(1)) で収束し、`crates/nnue-format` の量子化 (`QA=127 / QB=64 / FV_SCALE=28`、
/// bullet の `out ≈ cp/600` スケール前提) と整合する。bullet v102 を厳密再現するには
/// この kernel を使う。
///
/// - target: `pt = (score - 270)/380`、`pmt = (-score - 270)/380`、`target_wrm =
///   0.5*(1 + sigmoid(pt) - sigmoid(pmt))`、`target = lambda*wdl + (1-lambda)*target_wrm`。
///   in_scaling=380 と offset=270 は bullet ハードコード (`--wrm-in-scaling` ではない)。
/// - prediction: `scorenet = out * nnue2score`、`q = sigmoid((scorenet - 270)/in_scaling)`、
///   `qm = sigmoid((-scorenet - 270)/in_scaling)`、`qf = 0.5*(1 + q - qm)`。`in_scaling`
///   (= `--wrm-in-scaling` = 340) は **prediction 側のみ**、`nnue2score` (= `--wrm-nnue2score`
///   = 600)。
/// - `err = qf - target`、`loss_acc += err^2` (norm 無し、caller が position 数で割る)。
/// - chain rule: `dq/dout = q(1-q) * nnue2score/in_scaling`、`dqm/dout = -qm(1-qm) *
///   nnue2score/in_scaling`、`dqf/dout = 0.5 * (nnue2score/in_scaling) * (q(1-q) + qm(1-qm))`、
///   `dL/dout = 2*err * dqf/dout` → `2` と `0.5` が打ち消し合い `g = err *
///   (nnue2score/in_scaling) * (q(1-q) + qm(1-qm)) * per_pos_norm`。
///
/// 1 thread = 1 position。`dl_dout` は排他更新 (atomics 不要)、`loss_acc` は f64 単一
/// cell の `DeviceAtomicF64::fetch_add` (`loss_wdl` と同型)。`f32::exp` は libdevice
/// (`__nv_expf`) に lowering OK。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn loss_wrm(
    out: &[f32],
    score: &[f32],
    wdl: &[f32],
    per_pos_norm: &[f32],
    mut dl_dout: DisjointSlice<f32>,
    loss_acc: &[f64],
    lambda: f32,
    nnue2score: f32,
    in_scaling: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    // --- target (bullet loader WRM target、in_scaling=380 / offset=270 はハードコード) ---
    let s = score[i.get()];
    let sig_pt = 1.0_f32 / (1.0_f32 + (-((s - 270.0_f32) / 380.0_f32)).exp());
    let sig_pmt = 1.0_f32 / (1.0_f32 + (-((-s - 270.0_f32) / 380.0_f32)).exp());
    let target_wrm = 0.5_f32 * (1.0_f32 + sig_pt - sig_pmt);
    let target = lambda * wdl[i.get()] + (1.0_f32 - lambda) * target_wrm;

    // --- prediction (bullet loss_fn_wrm: WRM applied to net output) ---
    let scorenet = out[i.get()] * nnue2score;
    let q = 1.0_f32 / (1.0_f32 + (-((scorenet - 270.0_f32) / in_scaling)).exp());
    let qm = 1.0_f32 / (1.0_f32 + (-((-scorenet - 270.0_f32) / in_scaling)).exp());
    let qf = 0.5_f32 * (1.0_f32 + q - qm);

    let err = qf - target;
    let norm = per_pos_norm[i.get()];

    if let Some(g) = dl_dout.get_mut(i) {
        *g = err * (nnue2score / in_scaling) * (q * (1.0_f32 - q) + qm * (1.0_f32 - qm)) * norm;
    }

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell 確保済 (`loss_wdl` と同型)。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);
}

/// Fused AdamW optimizer step — Stage 2-3 (#39 / PR #48) inline。
///
/// 本 v102 path では **未使用** (Ranger 使用)。compile-reach 用に preserve。
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

/// Fused RAdam optimizer step — Stage 2-4 (#40 / PR #49) inline。
///
/// `step_size` / `denom` は host 側 (`gpu_kernels::pointwise::radam_step::
/// radam_compute_step_size_denom`) で step 番号から事前計算した scalar を値渡し。
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

/// Ranger Lookahead lerp — Stage 2-5 (#41 / PR #50) inline。
///
/// `weights[i] = alpha * weights[i] + (1 - alpha) * slow[i]`、`slow[i] = weights[i]`。
/// `step % k == 0` のときのみ host から呼ばれる lerp 部分。
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

/// Sparse feature transform forward (HalfKA_hm 用) — Stage 2-6 (#42 / PR #51) inline。
///
/// 1 thread = 1 (batch, row)、column-major weight (`weight[idx * rows + ri]`)、
/// atomics 不要 (各 thread は別 output cell に書く)。`-1` padding と `idx >= cols`
/// の異常入力は silent skip。
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

/// Sparse feature transform backward (atomic scatter) — Stage 2-7 (#43 / PR #52) inline。
///
/// 1 thread = 1 (batch, row)、column-major `grad_weight[idx * rows + ri]`、
/// **accumulate semantics** (host が呼出前に `grad_weight` を 0 で初期化)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn sparse_ft_backward(
    grad_out: &[f32],
    indices: &[i32],
    grad_weight: &[f32],
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

    let g = grad_out[tid.get()];
    let base = bi * (nnz as usize);
    let mut ni: u32 = 0;
    while ni < nnz {
        let idx = indices[base + (ni as usize)];
        if idx >= 0 && (idx as u32) < cols {
            // SAFETY: `grad_weight.len() == rows * cols` host invariant、`idx < cols` / `ri < rows`
            // で範囲内。`f32` (align 4) と `DeviceAtomicF32` (`#[repr(transparent)]` over UnsafeCell)
            // は同 alignment。non-atomic 経路で同 memory に書く path は本 kernel/host loop に無し。
            let cell = unsafe {
                &*(grad_weight
                    .as_ptr()
                    .add((idx as usize) * (rows as usize) + ri)
                    as *const DeviceAtomicF32)
            };
            cell.fetch_add(g, AtomicOrdering::Relaxed);
        }
        ni += 1;
    }
}

// ===========================================================================
// NEW kernels (Stage 3-7 で v102 arch のため追加)
// ===========================================================================
//
// 設計方針:
// - atomics は host が呼出前に gradient buffer を 0 初期化する accumulate semantics
//   (Stage 1-6 grad / Stage 2-7 sparse_ft_backward と同 convention)
// - DisjointSlice<f32> は 1 thread = 1 cell の排他書き込み、&[f32] + raw atomic は
//   多 thread → 1 cell の atomic accumulate
// - cuda-oxide 制限: `f32::clamp` / `f32::max` / `f32::min` は if-else 展開
// - 数値同等性テスト (GPU↔CPU reference) は Stage 3-7 skeleton 段階では skip
//   (Stage 3-8 trainer integrate / Stage 3-9 自己対局検証で本物の loss curve 比較)

/// Fused FT post-processing (forward) — bias add → CReLU → pairwise_mul → scale。
///
/// bullet `shogi_layerstack.rs:2241-2243` の `l0.forward(stm/nstm).crelu().
/// pairwise_mul() * (127.0/128.0)` + `stm.concat(nstm)` を 1 kernel に集約 (両
/// perspective まとめて combined 出力)。
///
/// 設計: 1 thread = combined buffer の 1 cell。`combined` の前半 (`[0, ft_dim/2)`) が
/// stm の pairwise_mul 出力、後半 (`[ft_dim/2, ft_dim)`) が nstm の pairwise_mul 出力。
/// 各 thread は自分が担当する combined cell の (batch, ri) と (is_stm, pair_idx) を
/// 判定して、対応する perspective ft_out を読みに行く。
///
/// `pairwise_mul` semantic (bullet `builder.rs:557-560`): `slice_rows(0, n/2) *
/// slice_rows(n/2, n)`、つまり前半 `[0, half)` と後半 `[half, n)` の **対応 index
/// 同士** の積 (隣接 pair でなく)。本 kernel も同じ。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn ft_post_perspective_fwd(
    stm_ft_out: &[f32],
    nstm_ft_out: &[f32],
    bias: &[f32],
    mut combined: DisjointSlice<f32>,
    batch: u32,
    ft_dim: u32, // = 1536 (per-perspective input dim)
    scale: f32,  // = 127.0/128.0
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (ft_dim as usize);
    let ri = tid.get() % (ft_dim as usize);
    let half = (ft_dim as usize) / 2;

    let ft_base = bi * (ft_dim as usize);
    let val = if ri < half {
        // stm side, pair_idx = ri in [0, half)
        let xa = stm_ft_out[ft_base + ri] + bias[ri];
        let xb = stm_ft_out[ft_base + half + ri] + bias[half + ri];
        let ya = if xa < 0.0_f32 {
            0.0_f32
        } else if xa > 1.0_f32 {
            1.0_f32
        } else {
            xa
        };
        let yb = if xb < 0.0_f32 {
            0.0_f32
        } else if xb > 1.0_f32 {
            1.0_f32
        } else {
            xb
        };
        ya * yb * scale
    } else {
        // nstm side, pair_idx = ri - half in [0, half)
        let pair_idx = ri - half;
        let xa = nstm_ft_out[ft_base + pair_idx] + bias[pair_idx];
        let xb = nstm_ft_out[ft_base + half + pair_idx] + bias[half + pair_idx];
        let ya = if xa < 0.0_f32 {
            0.0_f32
        } else if xa > 1.0_f32 {
            1.0_f32
        } else {
            xa
        };
        let yb = if xb < 0.0_f32 {
            0.0_f32
        } else if xb > 1.0_f32 {
            1.0_f32
        } else {
            xb
        };
        ya * yb * scale
    };

    if let Some(o) = combined.get_mut(tid) {
        *o = val;
    }
}

/// Fused FT post-processing (backward) — scale grad → pairwise_mul grad → CReLU grad
/// → bias grad。`ft_post_perspective_fwd` の per-perspective gradient。
///
/// **2 回呼ばれる** (stm と nstm 各 1 回)。`grad_bias` は両 call で **共有** (FT bias
/// は stm/nstm 共有のため、gradient は両方の和)。host は `grad_bias` を 1 回 zero 初期化、
/// 2 call で atomic accumulate される (Stage 2-7 sparse_ft_backward 同 convention)。
///
/// **stream synchronization**: 本 kernel は default stream で 2 connected launch
/// (stm 用 + nstm 用) として実行される。cuda-oxide の default stream は serialized
/// 実行 (各 launch は前の launch 完了後に開始) のため、`grad_bias` への atomic
/// accumulate は 2 call 間で race condition を起こさない。明示的な
/// `cudaStreamSynchronize` は host loop 末尾の `self.stream.synchronize()` で 1 回のみ。
///
/// 1 thread = 1 (batch, ft_dim_index) cell of this perspective's `grad_ft_out`。
/// tid in `[0, batch * ft_dim)`、tid IS the cell to write。
///
/// `d_combined_offset` で combined buffer 内の自 perspective の位置を指す
/// (stm: 0, nstm: ft_dim/2)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn ft_post_perspective_grad(
    d_combined: &[f32],                  // (batch × combined_dim)
    ft_out: &[f32],                      // perspective's sparse_ft_forward output (batch × ft_dim)
    bias: &[f32],                        // shared FT bias (ft_dim)
    mut grad_ft_out: DisjointSlice<f32>, // perspective's dft output (batch × ft_dim)
    grad_bias: &[f32],                   // shared, atomic accumulate (ft_dim)
    batch: u32,
    ft_dim: u32,
    d_combined_offset: u32, // 0 (stm) or ft_dim/2 (nstm)
    d_combined_stride: u32, // = combined_dim = ft_dim
    scale: f32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (ft_dim as usize);
    let ii = tid.get() % (ft_dim as usize);
    let half = (ft_dim as usize) / 2;

    // どちらの side か (前半 first or 後半 second)、ペア相手は同 pair_idx の反対側
    let (pair_idx, is_first) = if ii < half {
        (ii, true)
    } else {
        (ii - half, false)
    };

    // d_combined の対応 output cell を読む
    let dy =
        d_combined[bi * (d_combined_stride as usize) + (d_combined_offset as usize) + pair_idx];

    let ft_base = bi * (ft_dim as usize);
    let xa = ft_out[ft_base + pair_idx] + bias[pair_idx];
    let xb = ft_out[ft_base + half + pair_idx] + bias[half + pair_idx];

    let ya = if xa < 0.0_f32 {
        0.0_f32
    } else if xa > 1.0_f32 {
        1.0_f32
    } else {
        xa
    };
    let yb = if xb < 0.0_f32 {
        0.0_f32
    } else if xb > 1.0_f32 {
        1.0_f32
    } else {
        xb
    };

    let (my_pre, partner_post) = if is_first { (xa, yb) } else { (xb, ya) };

    // forward: out = ya * yb * scale → d(out)/d(my_post) = partner_post * scale
    let grad_my_post = dy * partner_post * scale;
    // CReLU grad: 1 if 0 < my_pre < 1 else 0
    let grad_my_pre = if my_pre > 0.0_f32 && my_pre < 1.0_f32 {
        grad_my_post
    } else {
        0.0_f32
    };

    // grad_ft_out[tid] = grad_my_pre (bias は加算なので grad 同値)
    if let Some(out) = grad_ft_out.get_mut(tid) {
        *out = grad_my_pre;
    }

    // grad_bias[ii] += grad_my_pre (atomic, 共有 bias)
    // SAFETY: grad_bias.len() == ft_dim、ii < ft_dim。
    let bias_cell = unsafe { &*(grad_bias.as_ptr().add(ii) as *const DeviceAtomicF32) };
    bias_cell.fetch_add(grad_my_pre, AtomicOrdering::Relaxed);
}

/// Regular dense matrix multiply forward + bias add。
///
/// `y[b][o] = bias[o] + sum_i x[b][i] * w[i][o]`。Layout: `x` row-major (batch × in_dim)、
/// `w` row-major (in_dim × out_dim)、`y` row-major (batch × out_dim)、`bias` (out_dim)。
///
/// 1 thread = 1 (batch, out_index) cell、atomics 不要。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_fwd(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    mut y: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (out_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (out_dim as usize);
    let oi = tid.get() % (out_dim as usize);
    let mut sum = bias[oi];
    let mut k: u32 = 0;
    while k < in_dim {
        sum += x[bi * (in_dim as usize) + (k as usize)] * w[(k as usize) * (out_dim as usize) + oi];
        k += 1;
    }
    if let Some(o) = y.get_mut(tid) {
        *o = sum;
    }
}

/// Regular dense matrix multiply backward (wrt input)。`dx[b][i] = sum_o dy[b][o] * w[i][o]`。
/// 1 thread = 1 (batch, in_index) cell、atomics 不要。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_input(
    dy: &[f32],
    w: &[f32],
    mut dx: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (in_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (in_dim as usize);
    let ii = tid.get() % (in_dim as usize);
    let mut sum = 0.0_f32;
    let mut o: u32 = 0;
    while o < out_dim {
        sum +=
            dy[bi * (out_dim as usize) + (o as usize)] * w[ii * (out_dim as usize) + (o as usize)];
        o += 1;
    }
    if let Some(d) = dx.get_mut(tid) {
        *d = sum;
    }
}

/// Regular dense matrix multiply backward (wrt weight)。`dw[i][o] = sum_b x[b][i] * dy[b][o]`。
/// 1 thread = 1 (in_index, out_index) weight cell、batch loop 内で sum、atomics 不要 (overwrite)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight(
    x: &[f32],
    dy: &[f32],
    mut grad_w: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (in_dim as usize) * (out_dim as usize);
    if tid.get() >= total {
        return;
    }
    let ii = tid.get() / (out_dim as usize);
    let oi = tid.get() % (out_dim as usize);
    let mut sum = 0.0_f32;
    let mut b: u32 = 0;
    while b < batch {
        sum +=
            x[(b as usize) * (in_dim as usize) + ii] * dy[(b as usize) * (out_dim as usize) + oi];
        b += 1;
    }
    if let Some(g) = grad_w.get_mut(tid) {
        *g = sum;
    }
}

/// Bias gradient (generic) — `grad_bias[o] += sum_b dy[b][o]` (atomic accumulate)。
///
/// 1 thread = 1 (batch, out) cell、各 oi が batch 数の atomic 寄与を受ける。
/// host が呼出前に `grad_bias` を 0 で初期化する責務 (accumulate semantics)。
#[kernel]
pub fn bias_grad(dy: &[f32], grad_bias: &[f32], batch: u32, out_dim: u32) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (out_dim as usize);
    if tid.get() >= total {
        return;
    }
    let oi = tid.get() % (out_dim as usize);
    let dyv = dy[tid.get()];
    // SAFETY: grad_bias[oi] within bounds (oi < out_dim、host が grad_bias.len() = out_dim 確保)。
    let cell = unsafe { &*(grad_bias.as_ptr().add(oi) as *const DeviceAtomicF32) };
    cell.fetch_add(dyv, AtomicOrdering::Relaxed);
}

/// Per-bucket dense matrix multiply forward + bias + select。
///
/// `y[b] (out_dim 次元) = bias[bucket_idx[b]] + sum_i x[b][i] * w[bucket_idx[b]][i]`。
/// Layout: `w` row-major (num_buckets * out_dim × in_dim) — bucket-major、その中で
/// out-major。`bias` (num_buckets * out_dim)、`y` (batch × out_dim)。
///
/// 1 thread = 1 (batch, out_index) cell、`bucket_idx[bi]` で per-position bucket 選択。
/// out-of-range bucket は silent skip (y は 0 のままになる)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_fwd_bucket(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    bucket_idx: &[i32],
    mut y: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (out_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (out_dim as usize);
    let oi = tid.get() % (out_dim as usize);
    let buc = bucket_idx[bi];
    if buc < 0 || (buc as u32) >= num_buckets {
        if let Some(o) = y.get_mut(tid) {
            *o = 0.0_f32;
        }
        return;
    }
    let buc_u = buc as usize;
    let w_row_base = buc_u * (out_dim as usize) * (in_dim as usize) + oi * (in_dim as usize);
    let bias_idx = buc_u * (out_dim as usize) + oi;
    let mut sum = bias[bias_idx];
    let mut k: u32 = 0;
    while k < in_dim {
        sum += x[bi * (in_dim as usize) + (k as usize)] * w[w_row_base + (k as usize)];
        k += 1;
    }
    if let Some(o) = y.get_mut(tid) {
        *o = sum;
    }
}

/// Per-bucket dense matmul backward (wrt input)。`dx[b][i] = sum_o dy[b][o] * w[bucket_idx[b]][o][i]`。
/// 1 thread = 1 (batch, in_index)、atomics 不要。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_input_bucket(
    dy: &[f32],
    w: &[f32],
    bucket_idx: &[i32],
    mut dx: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (in_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (in_dim as usize);
    let ii = tid.get() % (in_dim as usize);
    let buc = bucket_idx[bi];
    if buc < 0 || (buc as u32) >= num_buckets {
        if let Some(d) = dx.get_mut(tid) {
            *d = 0.0_f32;
        }
        return;
    }
    let buc_u = buc as usize;
    let mut sum = 0.0_f32;
    let mut o: u32 = 0;
    while o < out_dim {
        let w_idx =
            buc_u * (out_dim as usize) * (in_dim as usize) + (o as usize) * (in_dim as usize) + ii;
        sum += dy[bi * (out_dim as usize) + (o as usize)] * w[w_idx];
        o += 1;
    }
    if let Some(d) = dx.get_mut(tid) {
        *d = sum;
    }
}

/// Per-bucket dense matmul backward (wrt weight)。
/// `grad_w[bucket][o][i] = sum_{b: bucket_idx[b]==bucket} x[b][i] * dy[b][o]` (overwrite、atomics 不要)。
///
/// 1 thread = 1 (bucket, out_index, in_index) weight cell。batch を inner loop で回し、
/// `bucket_idx[b]` が自分の bucket の position だけ accumulate する。non-bucket 版
/// `dense_mm_bwd_weight` と同じ「1 cell = 1 thread + batch loop」形なので atomic scatter
/// は不要 (旧実装は 1 thread = 1 (batch, out, in) で同 weight cell へ多 thread atomic add
/// していた → bucket 偏りで contention 大、Stage 3-quality #77 で本形に変更)。
/// Layout: `grad_w` row-major (num_buckets * out_dim × in_dim) — bucket-major、その中 out-major
/// (= `dense_mm_fwd_bucket` の weight layout と一致、`tid == grad_w index`)。
/// out-of-range bucket (`bucket_idx[b] < 0` 等) の position はどの bucket cell にも match
/// しないので無視される (旧実装の silent skip と同じ)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight_bucket(
    x: &[f32],
    dy: &[f32],
    bucket_idx: &[i32],
    mut grad_w: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    let tid = thread::index_1d();
    let per_bucket = (out_dim as usize) * (in_dim as usize);
    let total = (num_buckets as usize) * per_bucket;
    if tid.get() >= total {
        return;
    }
    let buc_u = tid.get() / per_bucket;
    let rem = tid.get() % per_bucket;
    let oi = rem / (in_dim as usize);
    let ii = rem % (in_dim as usize);
    // num_buckets は小さい (= 9) ので buc_u as i32 は wrap しない。負の bucket_idx は match しない。
    let target_buc = buc_u as i32;
    let mut sum = 0.0_f32;
    let mut b: u32 = 0;
    while b < batch {
        let bb = b as usize;
        if bucket_idx[bb] == target_buc {
            sum += x[bb * (in_dim as usize) + ii] * dy[bb * (out_dim as usize) + oi];
        }
        b += 1;
    }
    if let Some(g) = grad_w.get_mut(tid) {
        *g = sum;
    }
}

/// Per-bucket bias gradient (atomic accumulate)。
/// `grad_bias[bucket][o] += sum_{b ∈ bucket} dy[b][o]`。1 thread = 1 (batch, out)、atomic。
#[kernel]
pub fn bias_grad_bucket(
    dy: &[f32],
    bucket_idx: &[i32],
    grad_bias: &[f32],
    batch: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (out_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (out_dim as usize);
    let oi = tid.get() % (out_dim as usize);
    let buc = bucket_idx[bi];
    if buc < 0 || (buc as u32) >= num_buckets {
        return;
    }
    let buc_u = buc as usize;
    let dyv = dy[tid.get()];
    let cell_idx = buc_u * (out_dim as usize) + oi;
    // SAFETY: cell_idx < num_buckets * out_dim、host が grad_bias.len() = same 確保。
    let cell = unsafe { &*(grad_bias.as_ptr().add(cell_idx) as *const DeviceAtomicF32) };
    cell.fetch_add(dyv, AtomicOrdering::Relaxed);
}

/// CReLU forward — `y[i] = clip(x[i], 0, 1)`。1 thread = 1 element。
#[kernel]
pub fn crelu_fwd(x: &[f32], mut y: DisjointSlice<f32>, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let xi = x[i.get()];
    #[allow(clippy::manual_clamp)]
    let yi = if xi < 0.0_f32 {
        0.0_f32
    } else if xi > 1.0_f32 {
        1.0_f32
    } else {
        xi
    };
    if let Some(out) = y.get_mut(i) {
        *out = yi;
    }
}

/// CReLU gradient — `dx[i] = dy[i] if 0 < x[i] < 1 else 0`。1 thread = 1 element。
#[kernel]
pub fn crelu_grad(x: &[f32], dy: &[f32], mut dx: DisjointSlice<f32>, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let xi = x[i.get()];
    let g = if xi > 0.0_f32 && xi < 1.0_f32 {
        dy[i.get()]
    } else {
        0.0_f32
    };
    if let Some(out) = dx.get_mut(i) {
        *out = g;
    }
}

/// abs_pow(2) * scale forward — `y[i] = x[i] * x[i] * scale`。
/// bullet `abs_pow(2.0)` は `|x|^2 = x^2` なので abs 不要。1 thread = 1 element。
#[kernel]
pub fn abs_pow2_scale_fwd(x: &[f32], mut y: DisjointSlice<f32>, scale: f32, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let xi = x[i.get()];
    if let Some(out) = y.get_mut(i) {
        *out = xi * xi * scale;
    }
}

/// abs_pow(2) * scale gradient — `dx[i] = 2 * x[i] * scale * dy[i]`。
#[kernel]
pub fn abs_pow2_scale_grad(x: &[f32], dy: &[f32], mut dx: DisjointSlice<f32>, scale: f32, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let xi = x[i.get()];
    let g = 2.0_f32 * xi * scale * dy[i.get()];
    if let Some(out) = dx.get_mut(i) {
        *out = g;
    }
}

/// Concat l1_sqr + l1_main forward — `out[b][..a_dim] = a[b]`, `out[b][a_dim..a_dim+b_dim] = b[b]`。
///
/// 1 thread = 1 (batch, output_index) cell。`out_dim = a_dim + b_dim`。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn concat_l1sqr_main_fwd(
    a: &[f32],
    b: &[f32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    a_dim: u32,
    b_dim: u32,
) {
    let tid = thread::index_1d();
    let out_dim = (a_dim as usize) + (b_dim as usize);
    let total = (batch as usize) * out_dim;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / out_dim;
    let oi = tid.get() % out_dim;
    let val = if oi < (a_dim as usize) {
        a[bi * (a_dim as usize) + oi]
    } else {
        b[bi * (b_dim as usize) + (oi - (a_dim as usize))]
    };
    if let Some(o) = out.get_mut(tid) {
        *o = val;
    }
}

/// Concat l1_sqr + l1_main backward — `da[b] = dout[b][..a_dim]`, `db[b] = dout[b][a_dim..]`。
///
/// **Precondition: `a_dim == b_dim`** (v102 では両方 `l1_effective` = 15)。tid は
/// `da[tid]` と `db[tid]` (両 slice の同 tid cell) に書き込む。
/// 1 thread = 1 (batch, dim_index) cell。
#[kernel]
pub fn concat_l1sqr_main_grad(
    dout: &[f32],
    mut da: DisjointSlice<f32>,
    mut db: DisjointSlice<f32>,
    batch: u32,
    dim: u32, // a_dim == b_dim assumed
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (dim as usize);
    let ii = tid.get() % (dim as usize);
    let out_dim = 2 * (dim as usize);

    let da_val = dout[bi * out_dim + ii];
    let db_val = dout[bi * out_dim + (dim as usize) + ii];

    if let Some(o) = da.get_mut(tid) {
        *o = da_val;
    }
    if let Some(o) = db.get_mut(tid) {
        *o = db_val;
    }
}

/// Elementwise add — `c[i] = a[i] + b[i]`。forward (l1+l1f, l3+l1_skip) と
/// gradient-copy (双方に同 grad 配る) 両用。1 thread = 1 element。
#[kernel]
pub fn elementwise_add(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    if let Some(out) = c.get_mut(i) {
        *out = a[i.get()] + b[i.get()];
    }
}

/// Extract a 2D slice — `dst[bi][oi] = src[bi*src_stride + src_offset + oi]`。
/// 1 thread = 1 dst cell。l1_total (B×16) → l1_main (B×15) / l1_skip (B×1) 抽出に使用。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn slice_extract_2d(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
    batch: u32,
    src_stride: u32,
    src_offset: u32,
    out_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (out_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (out_dim as usize);
    let oi = tid.get() % (out_dim as usize);
    if let Some(o) = dst.get_mut(tid) {
        *o = src[bi * (src_stride as usize) + (src_offset as usize) + oi];
    }
}

/// Scatter a 2D slice — `dst[bi*dst_stride + dst_offset + ii] = src[bi*in_dim + ii]`。
/// 1 thread = 1 src cell、`get_unchecked_mut` で任意 dst index に書き込む (escape hatch)。
/// host が dst を呼出前に 0 (or 適切値) で初期化する責務。
///
/// 用途: backward で dl1_main (B×15) + dl1_skip (B×1) を dl1_total (B×16) に書き戻す
/// (2 回 call、`dst_offset` で位置切替)。
///
/// SAFETY: 各 thread が unique (bi, ii) → unique dst_idx に書き込み。複数 call で
/// `dst_offset` を変えれば disjoint な dst 範囲を書く。`dst_idx < dst.len()` は host
/// invariant (`dst.len() == batch * dst_stride`、`dst_offset + in_dim <= dst_stride`)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn slice_scatter_2d(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    dst_stride: u32,
    dst_offset: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (in_dim as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (in_dim as usize);
    let ii = tid.get() % (in_dim as usize);
    let val = src[tid.get()];
    let dst_idx = bi * (dst_stride as usize) + (dst_offset as usize) + ii;
    // SAFETY: see docstring above. Each thread writes to a unique dst_idx, and host ensures bounds.
    unsafe {
        *dst.get_unchecked_mut(dst_idx) = val;
    }
}

// ===========================================================================
// Host driver helpers (kernel module loader / launch utilities)
//
// Stage 1-9 (`bins/progress_kpabs_train::main.rs:308-539`) / Stage 2 EPIC #16
// (`experiments/002-fused-kernels/src/main.rs:500-697`) の慣行を踏襲。
// `kernel_names` のみ 27 個に拡張 (Stage 3-7 で 26 + Stage 3 #84 で `loss_wrm`)。
// ===========================================================================

#[allow(dead_code)]
const BLOCK_DIM: u32 = 256;

/// 1 D launch の grid 数を計算 (= ceil(n / block)、n=0 は block=1 個 launch)。
#[allow(dead_code)]
fn grid_dim_1d(n: usize, block: u32) -> (u32, u32, u32) {
    let blocks = ((n as u32).max(1)).div_ceil(block);
    (blocks, 1, 1)
}

/// `cargo-oxide build` が出力した kernel `.ll` を見つけ、`.ptx` に変換した上で
/// CudaModule を load。Stage 1-9 / Stage 2 と同 fallback 順 (`.ll` → `.cubin` → `.ptx`)。
#[allow(dead_code)]
fn load_kernel_module_with_fallback(
    ctx: &std::sync::Arc<CudaContext>,
    name: &str,
) -> Result<std::sync::Arc<CudaModule>, Box<dyn std::error::Error>> {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| manifest_dir.clone());

    let probe = |dir: &std::path::PathBuf| {
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

/// `.ll` を libdevice と link → opt → llc で `.ptx` 生成。Stage 1-9 / Stage 2 と
/// 同 pipeline、`kernel_names` のみ全 27 kernel (Stage 3-7 の 26 + #84 の `loss_wrm`)
/// を内 internalize。
#[allow(dead_code)]
fn compile_ll_to_ptx_via_llc(
    ll_path: &std::path::PathBuf,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
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

    // Stage 3-7 + #84 で 全 27 kernel 名。`@<name>` として `.ll` に出ているものを漏れなく
    // internalize-public-api-list で残す (kernel-list hazard、Stage 2-2 で確立)。
    let kernel_names = "sparse_ft_forward,sparse_ft_backward,loss_wdl,loss_wrm,screlu_grad,\
                       adamw_step,radam_step,ranger_lookahead_lerp,\
                       ft_post_perspective_fwd,ft_post_perspective_grad,\
                       dense_mm_fwd,dense_mm_bwd_input,dense_mm_bwd_weight,bias_grad,\
                       dense_mm_fwd_bucket,dense_mm_bwd_input_bucket,\
                       dense_mm_bwd_weight_bucket,bias_grad_bucket,\
                       crelu_fwd,crelu_grad,abs_pow2_scale_fwd,abs_pow2_scale_grad,\
                       concat_l1sqr_main_fwd,concat_l1sqr_main_grad,elementwise_add,\
                       slice_extract_2d,slice_scatter_2d";

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
                 Stage 3-7 は llvm-link-21 / opt-21 / llc-21 を要求 \
                 (libNVVM が opaque pointer IR を parse できないため)。\
                 LLVM_LINK_BIN / OPT_BIN / LLC_BIN env で別 binary 指定可。"
            )
        })?;
    if !status.success() {
        return Err(format!("{bin} failed with status {status}").into());
    }
    Ok(())
}

/// `libdevice.10.bc` を CUDA Toolkit から探す (Stage 1-9 / Stage 2 と同探索順)。
fn find_libdevice_bc() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    if let Ok(p) = std::env::var("CUDA_OXIDE_LIBDEVICE") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let mut tried = Vec::new();
    let roots: Vec<std::path::PathBuf> = std::env::var("CUDA_HOME")
        .ok()
        .into_iter()
        .chain(std::env::var("CUDA_PATH").ok())
        .map(std::path::PathBuf::from)
        .chain([
            std::path::PathBuf::from("/usr/local/cuda"),
            std::path::PathBuf::from("/usr/local/cuda-13.2"),
            std::path::PathBuf::from("/usr/local/cuda-12.9"),
            std::path::PathBuf::from("/opt/cuda"),
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

// ===========================================================================
// v102 architecture constants (bullet `shogi_layerstack.rs:1831-1834, 2097-2101` 由来)
// ===========================================================================

const FT_IN: usize = 73_305; // `HALFKA_HM_DIMENSIONS` (shogi-features::halfka_hm)
const FT_OUT: usize = 1536; // per-perspective FT output dim
const MAX_ACTIVE: usize = 40; // `MAX_ACTIVE_FEATURES` (nnz per perspective per position)
const COMBINED_DIM: usize = FT_OUT; // pairwise (1536 → 768) × 2 perspectives concat = 1536
const L1_OUT: usize = 16;
const L1_EFFECTIVE: usize = L1_OUT - 1; // = 15 (skip 1 dim、bullet:1433)
const L1_SKIP: usize = L1_OUT - L1_EFFECTIVE; // = 1
const L2_IN: usize = L1_EFFECTIVE * 2; // = 30 (l1_sqr.concat(l1_main))、bullet:1434
const L2_OUT: usize = 32;
const NUM_BUCKETS: usize = 9; // progress8kpabs

// ===========================================================================
// raw checkpoint format (Issue #88、`--resume` 用)
// ===========================================================================

/// raw checkpoint format magic (`b"RNRC"` = "RShogi Nnue Resume Checkpoint")。
/// `crates/nnue-train::optimizer` の `b"RNGR"` (RangerHostState single-file format) とは
/// 別物 — こちらは weight group raw f32 + Ranger state + step + superbatch を 1 file に
/// まとめた self-contained format (`RNGR` は optimizer state だけ、weight は持たない)。
const RAW_CKPT_MAGIC: [u8; 4] = *b"RNRC";

/// raw checkpoint format version (本 PR は 1、後続変更で increment)。
const RAW_CKPT_VERSION: u32 = 1;

/// raw checkpoint 1 group 分の host buffer (`w`, `m`, `v`, `slow` の f32 Vec、`grad` は含めない)。
type RawCkptGroup = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// `io::ErrorKind::InvalidData` の `Box<dyn Error>` を作る短縮 helper (raw checkpoint
/// の magic/version/dim 検証で使う、`RangerHostState::load_from_reader` と同方針)。
fn invalid_data(msg: String) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
}

/// f32 slice を little-endian で `w` に書き出す (`bytemuck` 不使用、依存を増やさない)。
fn write_f32_slice<W: std::io::Write>(w: &mut W, data: &[f32]) -> std::io::Result<()> {
    // 4 byte ずつの write_all は遅いので、一旦 byte Vec に詰めてから 1 回で書く
    // (`raw_ckpt_groups` 最大 113M f32 = ~450MB、呼び出し側は BufWriter で wrap 済だが
    //  chunk write の方が更に system call が減る)。
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&bytes)
}

/// `r.read_exact(buf)` を呼び、`UnexpectedEof` (= file が途中で切れている、破損 / 部分書き)
/// を `InvalidData` + context message に正規化する。raw checkpoint の robustness contract
/// 「malformed input は全部 `InvalidData`、panic しない」を満たすため、`load_raw_checkpoint`
/// 内の全 `read_exact` はこの helper 経由で呼ぶ (`what` は読もうとしていた field の説明)。
fn read_exact_or_invalid<R: std::io::Read>(
    r: &mut R,
    buf: &mut [u8],
    what: &str,
) -> std::io::Result<()> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("raw checkpoint truncated while reading {what}"),
            )
        } else {
            e
        }
    })
}

/// little-endian f32 を `n` 個読む (`RangerHostState::load_from_reader` の `read_f32_vec`
/// と同型だが本 module 内ローカル版、`io::Result` を返す)。`what` は短読み (破損 file) 時の
/// context message に使う (`UnexpectedEof` → `InvalidData` に正規化、`read_exact_or_invalid` 経由)。
fn read_f32_vec_io<R: std::io::Read>(r: &mut R, n: usize, what: &str) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![
        0u8;
        n.checked_mul(4).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("f32 vec len {n} overflows byte count"),
            )
        })?
    ];
    read_exact_or_invalid(r, &mut bytes, what)?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// scale 定数 (bullet shogi_layerstack.rs:2241, 2260)
const FT_POST_SCALE: f32 = 127.0 / 128.0;
const L1_SQR_SCALE: f32 = 127.0 / 128.0;

// Ranger optimizer params。bullet `RangerParams::default()` 由来の値は
// `nnue_train::optimizer::RangerParams::DEFAULT` を single source of truth として参照する
// (Stage 3-quality #86: 旧 main.rs での const 二重定義を解消)。
const RANGER_DEFAULTS: nnue_train::optimizer::RangerParams =
    nnue_train::optimizer::RangerParams::DEFAULT;
const BETA1: f32 = RANGER_DEFAULTS.beta1;
const BETA2: f32 = RANGER_DEFAULTS.beta2;
const EPS: f32 = RANGER_DEFAULTS.eps;
const MIN_W: f32 = RANGER_DEFAULTS.min_weight;
const MAX_W: f32 = RANGER_DEFAULTS.max_weight;
const RANGER_ALPHA: f32 = RANGER_DEFAULTS.alpha;
const RANGER_K: u64 = RANGER_DEFAULTS.k as u64;
const N_SMA_THRESHOLD: f32 = RANGER_DEFAULTS.n_sma_threshold;
// v102 は weight-decay を 0.0 に override する (`RangerParams::DEFAULT.decay` = 0.01 は
// bullet の汎用 default、本 trainer は recipe どおり 0.0 を使う、memory project_v102_recipe.md)。
const DECAY: f32 = 0.0;

// smoke 用 loss params (v102 doc: scale=290, wdl=0.0、wrm in_scaling 340 / nnue2score 600)。
// trainer 経路では CLI から `LossKind` を組み立てるのでここは smoke 専用。
const WDL_LAMBDA: f32 = 0.0;
/// smoke で使う固定 batch position 数 (`GpuTrainer::new` の workspace 初期 batch にも使う)。
const SMOKE_BATCH: usize = 4;
const SMOKE_LOSS_SIGMOID: LossKind = LossKind::Sigmoid { scale: 1.0 / 290.0 };
const SMOKE_LOSS_WRM: LossKind = LossKind::Wrm {
    nnue2score: 600.0,
    in_scaling: 340.0,
};

// ===========================================================================
// GpuTrainer (v102 LayerStack 1536-16-32 + progress8kpabs 9 buckets)
//
// 10 weight groups × {w, m, v, slow, grad} = 50 device buffers + loss_acc + step_count。
// Forward は 15 kernel launch、backward は ~16 kernel launch、optimizer は 10×{radam+lerp}。
// Stage 3-7 段階では smoke 動作 (NaN check) のみ、Stage 3-8 trainer integrate で
// `crates/nnue-train::trainer` loop と統合する。
// ===========================================================================

#[allow(dead_code)] // 一部 field は Stage 3-8 で host state 直接更新時に使う
struct GpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,

    // FT (single, shared across perspectives)
    ft_w: DeviceBuffer<f32>,
    ft_w_m: DeviceBuffer<f32>,
    ft_w_v: DeviceBuffer<f32>,
    ft_w_slow: DeviceBuffer<f32>,
    ft_w_grad: DeviceBuffer<f32>,
    ft_b: DeviceBuffer<f32>,
    ft_b_m: DeviceBuffer<f32>,
    ft_b_v: DeviceBuffer<f32>,
    ft_b_slow: DeviceBuffer<f32>,
    ft_b_grad: DeviceBuffer<f32>,

    // L1 per-bucket delta
    l1_w: DeviceBuffer<f32>,
    l1_w_m: DeviceBuffer<f32>,
    l1_w_v: DeviceBuffer<f32>,
    l1_w_slow: DeviceBuffer<f32>,
    l1_w_grad: DeviceBuffer<f32>,
    l1_b: DeviceBuffer<f32>,
    l1_b_m: DeviceBuffer<f32>,
    l1_b_v: DeviceBuffer<f32>,
    l1_b_slow: DeviceBuffer<f32>,
    l1_b_grad: DeviceBuffer<f32>,

    // L1f shared factorized
    l1f_w: DeviceBuffer<f32>,
    l1f_w_m: DeviceBuffer<f32>,
    l1f_w_v: DeviceBuffer<f32>,
    l1f_w_slow: DeviceBuffer<f32>,
    l1f_w_grad: DeviceBuffer<f32>,
    l1f_b: DeviceBuffer<f32>,
    l1f_b_m: DeviceBuffer<f32>,
    l1f_b_v: DeviceBuffer<f32>,
    l1f_b_slow: DeviceBuffer<f32>,
    l1f_b_grad: DeviceBuffer<f32>,

    // L2 per-bucket
    l2_w: DeviceBuffer<f32>,
    l2_w_m: DeviceBuffer<f32>,
    l2_w_v: DeviceBuffer<f32>,
    l2_w_slow: DeviceBuffer<f32>,
    l2_w_grad: DeviceBuffer<f32>,
    l2_b: DeviceBuffer<f32>,
    l2_b_m: DeviceBuffer<f32>,
    l2_b_v: DeviceBuffer<f32>,
    l2_b_slow: DeviceBuffer<f32>,
    l2_b_grad: DeviceBuffer<f32>,

    // L3 per-bucket output
    l3_w: DeviceBuffer<f32>,
    l3_w_m: DeviceBuffer<f32>,
    l3_w_v: DeviceBuffer<f32>,
    l3_w_slow: DeviceBuffer<f32>,
    l3_w_grad: DeviceBuffer<f32>,
    l3_b: DeviceBuffer<f32>,
    l3_b_m: DeviceBuffer<f32>,
    l3_b_v: DeviceBuffer<f32>,
    l3_b_slow: DeviceBuffer<f32>,
    l3_b_grad: DeviceBuffer<f32>,

    // 中間 activation / activation-grad の永続 workspace (Issue #78、batch_size 固定前提
    // で `new` 時に確保。`step_impl` が requires より大きい batch を渡したら拡張)。
    ws: GpuWorkspace,

    // loss + step
    loss_acc: DeviceBuffer<f64>,
    step_count: u64,
}

/// `GpuTrainer::step_impl` の forward / backward で使う中間 activation と
/// activation-gradient buffer を **1 step ごとに再 alloc せず永続化** するための
/// workspace (Issue #78)。
///
/// 各 buffer は `len_batch` 個の position 分のサイズで確保され、`step_impl` が
/// より大きな batch を渡してきたら [`GpuWorkspace::ensure_batch`] で grow-only に
/// 再 alloc する (典型的には batch_size 固定 + 末尾の partial batch なので拡張は
/// 起きないか 1 回のみ)。`len_batch == 0` は「まだ未確保」を表す番兵 (実際には
/// `GpuTrainer::new` で `batch_size` 分を確保するので step 時には常に > 0)。
///
/// **メモリ覚書**: forward path は DAG で各 activation は読まれる前に kernel が
/// 全 cell を上書きするため memset 不要。`ws_batch` が現 batch `b` より大きい場合の
/// 末尾 `[b*dim .. ws_batch*dim)` は kernel が触らないが、後続 kernel も `b` で
/// bound するので read されない。例外は `dl1_total`: `slice_scatter_2d` の host
/// 契約 (「dst を 0 初期化」) を守るため `step_impl` で毎 step memset する。
/// grad buffer (`GpuTrainer::*_grad`) と `loss_acc` は atomic accumulate semantics
/// なので `step_impl` で毎 step memset する (元実装の `DeviceBuffer::zeroed` 再 alloc を
/// memset_async 0 に置換、`cudaMalloc`/`cudaFree` の stream stall を回避)。
struct GpuWorkspace {
    /// この workspace が確保している batch (= position) 数。0 = 未確保。
    len_batch: usize,

    // -- forward activations --
    ft_stm_out: DeviceBuffer<f32>,    // b × FT_OUT
    ft_nstm_out: DeviceBuffer<f32>,   // b × FT_OUT
    combined: DeviceBuffer<f32>,      // b × COMBINED_DIM
    l1_bucket: DeviceBuffer<f32>,     // b × L1_OUT
    l1f_out: DeviceBuffer<f32>,       // b × L1_OUT
    l1_total: DeviceBuffer<f32>,      // b × L1_OUT
    l1_main: DeviceBuffer<f32>,       // b × L1_EFFECTIVE
    l1_skip: DeviceBuffer<f32>,       // b × L1_SKIP
    l1_sqr: DeviceBuffer<f32>,        // b × L1_EFFECTIVE
    l2_pre: DeviceBuffer<f32>,        // b × L2_IN
    l2_input: DeviceBuffer<f32>,      // b × L2_IN
    l2_out: DeviceBuffer<f32>,        // b × L2_OUT
    l2_acted: DeviceBuffer<f32>,      // b × L2_OUT
    l3_out: DeviceBuffer<f32>,        // b
    net_output: DeviceBuffer<f32>,    // b
    dy_net_output: DeviceBuffer<f32>, // b (loss kernel が書き込む dnet)

    // -- backward activation-grads --
    dl2_acted: DeviceBuffer<f32>,            // b × L2_OUT
    dl2_out: DeviceBuffer<f32>,              // b × L2_OUT
    dl2_input: DeviceBuffer<f32>,            // b × L2_IN
    dl2_pre: DeviceBuffer<f32>,              // b × L2_IN
    dl1_sqr: DeviceBuffer<f32>,              // b × L1_EFFECTIVE
    dl1_main_from_concat: DeviceBuffer<f32>, // b × L1_EFFECTIVE
    dl1_main_from_sqr: DeviceBuffer<f32>,    // b × L1_EFFECTIVE
    dl1_main: DeviceBuffer<f32>,             // b × L1_EFFECTIVE
    dl1_total: DeviceBuffer<f32>,            // b × L1_OUT (毎 step memset、slice_scatter 契約)
    dcombined_from_l1f: DeviceBuffer<f32>,   // b × FT_OUT
    dcombined_from_l1: DeviceBuffer<f32>,    // b × FT_OUT
    dcombined: DeviceBuffer<f32>,            // b × FT_OUT
    dft_stm_out: DeviceBuffer<f32>,          // b × FT_OUT
    dft_nstm_out: DeviceBuffer<f32>,         // b × FT_OUT
}

impl GpuWorkspace {
    /// `batch` 個の position 分の全 buffer を確保する (`GpuTrainer::new` から呼ぶ)。
    fn new(stream: &CudaStream, batch: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(stream, n).map_err(Into::into)
        };
        Ok(Self {
            len_batch: batch,
            ft_stm_out: z(batch * FT_OUT)?,
            ft_nstm_out: z(batch * FT_OUT)?,
            combined: z(batch * COMBINED_DIM)?,
            l1_bucket: z(batch * L1_OUT)?,
            l1f_out: z(batch * L1_OUT)?,
            l1_total: z(batch * L1_OUT)?,
            l1_main: z(batch * L1_EFFECTIVE)?,
            l1_skip: z(batch * L1_SKIP)?,
            l1_sqr: z(batch * L1_EFFECTIVE)?,
            l2_pre: z(batch * L2_IN)?,
            l2_input: z(batch * L2_IN)?,
            l2_out: z(batch * L2_OUT)?,
            l2_acted: z(batch * L2_OUT)?,
            l3_out: z(batch)?,
            net_output: z(batch)?,
            dy_net_output: z(batch)?,
            dl2_acted: z(batch * L2_OUT)?,
            dl2_out: z(batch * L2_OUT)?,
            dl2_input: z(batch * L2_IN)?,
            dl2_pre: z(batch * L2_IN)?,
            dl1_sqr: z(batch * L1_EFFECTIVE)?,
            dl1_main_from_concat: z(batch * L1_EFFECTIVE)?,
            dl1_main_from_sqr: z(batch * L1_EFFECTIVE)?,
            dl1_main: z(batch * L1_EFFECTIVE)?,
            dl1_total: z(batch * L1_OUT)?,
            dcombined_from_l1f: z(batch * FT_OUT)?,
            dcombined_from_l1: z(batch * FT_OUT)?,
            dcombined: z(batch * FT_OUT)?,
            dft_stm_out: z(batch * FT_OUT)?,
            dft_nstm_out: z(batch * FT_OUT)?,
        })
    }

    /// `batch` 以上の容量を保証する (grow-only)。現 `len_batch` が足りていれば no-op、
    /// 足りなければ全 buffer を `batch` 分で再 alloc して `len_batch` を更新する
    /// (典型的には batch_size 固定なので一度も走らないか、起動時の確保で十分)。
    fn ensure_batch(
        &mut self,
        stream: &CudaStream,
        batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if batch > self.len_batch {
            *self = Self::new(stream, batch)?;
        }
        Ok(())
    }
}

/// Smoke / Stage 3-8 trainer 用の 1 batch 入力データ。
#[allow(dead_code)]
struct BatchData {
    n_pos: usize,
    stm_indices: Vec<i32>, // (n_pos × MAX_ACTIVE)、-1 padding 可
    nstm_indices: Vec<i32>,
    bucket_idx: Vec<i32>,   // (n_pos)、progress8kpabs の 0-8
    score: Vec<f32>,        // (n_pos)、target eval cp の元
    wdl: Vec<f32>,          // (n_pos)、0.0 (Loss) / 0.5 (Draw) / 1.0 (Win)
    per_pos_norm: Vec<f32>, // (n_pos)、bullet loss normalisation
}

impl BatchData {
    /// 決定論的な smoke 用 dummy batch。bucket_idx=0、small random sparse indices。
    fn smoke_dummy(n_pos: usize) -> Self {
        let mut stm_indices = vec![-1_i32; n_pos * MAX_ACTIVE];
        let mut nstm_indices = vec![-1_i32; n_pos * MAX_ACTIVE];
        // 各 position に MAX_ACTIVE 個 (実 HalfKA_hm の典型局面と同等) の deterministic indices
        // を入れる。range [0, FT_IN) で seed-based に分散。
        let mut s: u64 = 0xdead_beef;
        for b in 0..n_pos {
            for k in 0..MAX_ACTIVE {
                // xorshift
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let idx = (s as usize % FT_IN) as i32;
                stm_indices[b * MAX_ACTIVE + k] = idx;
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let idx2 = (s as usize % FT_IN) as i32;
                nstm_indices[b * MAX_ACTIVE + k] = idx2;
            }
        }
        Self {
            n_pos,
            stm_indices,
            nstm_indices,
            bucket_idx: vec![0_i32; n_pos],
            score: vec![0.0_f32; n_pos],
            wdl: vec![0.5_f32; n_pos],
            per_pos_norm: vec![1.0_f32; n_pos],
        }
    }

    /// `nnue-train` dataloader の `Batch` + per-position bucket から `BatchData`
    /// を作る (Stage 3-8 trainer integrate)。
    ///
    /// `Batch` は `batch_size * max_active` 容量を持つが有効件数は `n_positions`
    /// なので、sparse index は先頭 `n_positions * MAX_ACTIVE` 要素だけ切り出す。
    /// `Batch::max_active` は HalfKA_hm の `MAX_ACTIVE_FEATURES` (= `MAX_ACTIVE`
    /// = 40) を前提とする。
    ///
    /// `per_pos_norm` は **`1.0 / n_pos`** にする: `loss_wdl` kernel は per-position
    /// gradient に `per_pos_norm` を掛けて backward に流すため、これで weight
    /// gradient が batch 平均になり (atomic accumulate 後)、learning rate が
    /// batch size に依存しなくなる (bullet の `mean` reduction と同じ意味)。
    /// 報告 loss は `loss_acc` (= `Σ err²`) を position 数で割って平均にする
    /// (kernel 側は `loss_acc` を norm で割らずに raw `err²` を足す)。
    fn from_batch(batch: &Batch, bucket_idx: &[i32]) -> Self {
        let n_pos = batch.n_positions;
        assert_eq!(
            bucket_idx.len(),
            n_pos,
            "bucket_idx len ({}) must equal batch.n_positions ({})",
            bucket_idx.len(),
            n_pos
        );
        assert_eq!(
            batch.max_active, MAX_ACTIVE,
            "Batch::max_active ({}) must equal MAX_ACTIVE ({})",
            batch.max_active, MAX_ACTIVE
        );
        let span = n_pos * MAX_ACTIVE;
        let norm = if n_pos == 0 {
            0.0
        } else {
            1.0_f32 / n_pos as f32
        };
        Self {
            n_pos,
            stm_indices: batch.stm_indices[..span].to_vec(),
            nstm_indices: batch.nstm_indices[..span].to_vec(),
            bucket_idx: bucket_idx.to_vec(),
            score: batch.score[..n_pos].to_vec(),
            wdl: batch.wdl[..n_pos].to_vec(),
            per_pos_norm: vec![norm; n_pos],
        }
    }
}

/// `LaunchConfig` builder for 1D launch with `BLOCK_DIM` per block.
fn cfg_1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid_dim_1d(n, BLOCK_DIM),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Deterministic xorshift init for weights (small random in `[-scale, scale]`)。
fn xorshift_init(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // [0, 1) → [-1, 1) → × scale
        let u = (s >> 11) as f32 / ((1u64 << 53) as f32);
        let r = (u * 2.0 - 1.0) * scale;
        v.push(r);
    }
    v
}

/// `buf` の全 byte を 0 にする (stream 上、async)。`DeviceBuffer::zeroed` の
/// 再 alloc を伴わず既存 buffer を in-place で reset するため (Issue #78、grad /
/// `loss_acc` の毎 step reset で `cudaMalloc`/`cudaFree` の stream stall を回避)。
fn memset_zero<T>(
    stream: &CudaStream,
    buf: &DeviceBuffer<T>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = buf.num_bytes();
    if bytes > 0 {
        // SAFETY: `buf.cu_deviceptr()` は本 `DeviceBuffer` が確保した `bytes` byte の
        // 有効 device ptr、`stream` は同 context (`buf` も `stream` も `GpuTrainer` が
        // 同 context から作る)。`cuMemsetD8Async` は overlap を要求しない。0 fill は
        // f32/f64 ともに数値 0.0 を表すバイトパターン (全 0) なので型に依らず正しい。
        unsafe {
            cuda_core::memory::memset_d8_async(buf.cu_deviceptr(), 0, bytes, stream.cu_stream())?;
        }
    }
    Ok(())
}

impl GpuTrainer {
    /// CUDA context を作成し、kernel module を load、10 weight groups + Ranger state +
    /// 中間 activation workspace (`batch_size` 分、Issue #78) を確保。
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "nnue_train")?;

        // 各 weight group の element 数
        let ft_w_n = FT_IN * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;

        // Weight init: small random for non-degenerate forward (smoke 用、Stage 3-8 で
        // proper init: ft は bullet `init_with_effective_input_size(32)`、l1 は Zeroed 等)
        let init_scale = 0.01_f32;
        let ft_w_init = xorshift_init(0x100_u64, ft_w_n, init_scale);
        let l1_w_init = xorshift_init(0x101_u64, l1_w_n, init_scale);
        let l1f_w_init = xorshift_init(0x102_u64, l1f_w_n, init_scale);
        let l2_w_init = xorshift_init(0x103_u64, l2_w_n, init_scale);
        let l3_w_init = xorshift_init(0x104_u64, l3_w_n, init_scale);

        // Ranger Lookahead の slow weight は **0 初期化** (bullet `RangerLookahead::new`
        // = `vec![0.0; size]` と同じ、Issue #84/#L6)。初回 lerp (`step % k == 0`) で
        // `weights = alpha*weights + (1-alpha)*0 = alpha*weights` になる挙動も bullet と一致。
        Ok(Self {
            stream: stream.clone(),
            module,
            // FT
            ft_w: DeviceBuffer::from_host(&stream, &ft_w_init)?,
            ft_w_m: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_v: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_slow: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_grad: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_b: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_m: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_v: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_slow: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_grad: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            // L1
            l1_w: DeviceBuffer::from_host(&stream, &l1_w_init)?,
            l1_w_m: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_w_v: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_b: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_m: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_v: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            // L1f
            l1f_w: DeviceBuffer::from_host(&stream, &l1f_w_init)?,
            l1f_w_m: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_w_v: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_b: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_m: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_v: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            // L2
            l2_w: DeviceBuffer::from_host(&stream, &l2_w_init)?,
            l2_w_m: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_w_v: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_b: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_m: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_v: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            // L3
            l3_w: DeviceBuffer::from_host(&stream, &l3_w_init)?,
            l3_w_m: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_w_v: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_b: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_m: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_v: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            // 中間 activation workspace (Issue #78、`batch_size` 分。最低 1 で確保して
            // `len_batch == 0` (未確保) を作らない — smoke は batch=4 等を渡す)。
            ws: GpuWorkspace::new(&stream, batch_size.max(1))?,
            // loss + step
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            step_count: 0,
        })
    }

    /// `V102Weights` から weight buffer を device に upload (pretrained 注入、`--init-from`)。
    ///
    /// Optimizer state reset:
    /// - `m`, `v`: 0 (fresh start、Ranger 1st/2nd moment)
    /// - `slow`: **loaded weights と同値** (warm-start anchor。`GpuTrainer::new` (from-scratch)
    ///   は bullet `RangerLookahead::new` どおり `slow = 0` だが、`--init-from` は量子化済 NNUE
    ///   の continue-training/fine-tuning であって bullet checkpoint resume (`slow.bin` 付き)
    ///   ではない。`slow = 0` のままだと初回 lookahead lerp で `new_w = alpha*fast + (1-alpha)*0
    ///   = alpha*fast` となり読み込んだ重みが全て ~alpha 倍に縮む。`slow = w_loaded` にすると
    ///   初回 lerp は `new_w = alpha*fast + (1-alpha)*w_loaded` で、fine-tuning は lr が小さく
    ///   `fast ≈ w_loaded` なので **0 ではなく読み込んだ重みの方へ寄せる** anchor になる
    ///   (true な bullet resume なら `slow.bin` を読むべきだが、量子化 NNUE には optimizer
    ///   state が無いので next-best な default) — PR #92 review (Codex P2) 指摘)
    /// - `grad`: 0
    /// - `step_count`: 0 (1-indexed、次 step は 1)
    ///
    /// 注: `step_count = 0` 状態で `step()` を呼ぶと `self.step_count += 1` → 1 に
    /// なってから `radam_compute_step_size_denom(1, BETA1, BETA2, N_SMA_THRESHOLD)`
    /// を呼ぶ。bullet `radam_step.rs::radam_compute_step_size_denom` は step >= 1 で
    /// 安全動作 (step=0 では `beta^0 = 1` → `bc1 = 0` で `step_size = 1/0 = inf` に
    /// なる、本 helper も `step >= 1` 前提)。本実装は step=0 で呼ばないため OK。
    fn load_v102_weights(&mut self, w: &V102Weights) -> Result<(), Box<dyn std::error::Error>> {
        self.ft_w = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_b = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.l1_w = DeviceBuffer::from_host(&self.stream, &w.l1_w)?;
        self.l1_b = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1f_w = DeviceBuffer::from_host(&self.stream, &w.l1f_w)?;
        self.l1f_b = DeviceBuffer::from_host(&self.stream, &w.l1f_b)?;
        self.l2_w = DeviceBuffer::from_host(&self.stream, &w.l2_w)?;
        self.l2_b = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l3_w = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_b = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;
        // Optimizer state reset:
        // - m, v: 0 (fresh start)
        // - slow: loaded weights と同値 (warm-start anchor: 初回 lookahead lerp が
        //   0 でなく読み込んだ重みの方へ寄る。`slow = 0` だと alpha 倍に縮む — PR #92 review 指摘)
        // - grad: 0
        let zeros_f32 = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(&self.stream, n).map_err(Into::into)
        };
        let ft_w_n = FT_IN * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        self.ft_w_m = zeros_f32(ft_w_n)?;
        self.ft_w_v = zeros_f32(ft_w_n)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_w_grad = zeros_f32(ft_w_n)?;
        self.ft_b_m = zeros_f32(ft_b_n)?;
        self.ft_b_v = zeros_f32(ft_b_n)?;
        self.ft_b_slow = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.ft_b_grad = zeros_f32(ft_b_n)?;
        self.l1_w_m = zeros_f32(l1_w_n)?;
        self.l1_w_v = zeros_f32(l1_w_n)?;
        self.l1_w_slow = DeviceBuffer::from_host(&self.stream, &w.l1_w)?;
        self.l1_w_grad = zeros_f32(l1_w_n)?;
        self.l1_b_m = zeros_f32(l1_b_n)?;
        self.l1_b_v = zeros_f32(l1_b_n)?;
        self.l1_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1_b_grad = zeros_f32(l1_b_n)?;
        self.l1f_w_m = zeros_f32(l1f_w_n)?;
        self.l1f_w_v = zeros_f32(l1f_w_n)?;
        self.l1f_w_slow = DeviceBuffer::from_host(&self.stream, &w.l1f_w)?;
        self.l1f_w_grad = zeros_f32(l1f_w_n)?;
        self.l1f_b_m = zeros_f32(l1f_b_n)?;
        self.l1f_b_v = zeros_f32(l1f_b_n)?;
        self.l1f_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1f_b)?;
        self.l1f_b_grad = zeros_f32(l1f_b_n)?;
        self.l2_w_m = zeros_f32(l2_w_n)?;
        self.l2_w_v = zeros_f32(l2_w_n)?;
        self.l2_w_slow = DeviceBuffer::from_host(&self.stream, &w.l2_w)?;
        self.l2_w_grad = zeros_f32(l2_w_n)?;
        self.l2_b_m = zeros_f32(l2_b_n)?;
        self.l2_b_v = zeros_f32(l2_b_n)?;
        self.l2_b_slow = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l2_b_grad = zeros_f32(l2_b_n)?;
        self.l3_w_m = zeros_f32(l3_w_n)?;
        self.l3_w_v = zeros_f32(l3_w_n)?;
        self.l3_w_slow = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_w_grad = zeros_f32(l3_w_n)?;
        self.l3_b_m = zeros_f32(l3_b_n)?;
        self.l3_b_v = zeros_f32(l3_b_n)?;
        self.l3_b_slow = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;
        self.l3_b_grad = zeros_f32(l3_b_n)?;
        self.step_count = 0;
        Ok(())
    }

    /// device buffer を host に download し `V102Weights` を返す (save_quantised 前)。
    fn to_v102_weights(&self) -> Result<V102Weights, Box<dyn std::error::Error>> {
        Ok(V102Weights {
            ft_w: self.ft_w.to_host_vec(&self.stream)?,
            ft_b: self.ft_b.to_host_vec(&self.stream)?,
            l1_w: self.l1_w.to_host_vec(&self.stream)?,
            l1_b: self.l1_b.to_host_vec(&self.stream)?,
            l1f_w: self.l1f_w.to_host_vec(&self.stream)?,
            l1f_b: self.l1f_b.to_host_vec(&self.stream)?,
            l2_w: self.l2_w.to_host_vec(&self.stream)?,
            l2_b: self.l2_b.to_host_vec(&self.stream)?,
            l3_w: self.l3_w.to_host_vec(&self.stream)?,
            l3_b: self.l3_b.to_host_vec(&self.stream)?,
        })
    }

    /// 各 weight group の `(name, expected_len, &w, &m, &v, &slow)` を v102 固定順で返す
    /// (raw checkpoint の save/load で iterate するための immutable view)。`grad` は
    /// resume に不要なので含めない。順序 = ft_w, ft_b, l1_w, l1_b, l1f_w, l1f_b, l2_w,
    /// l2_b, l3_w, l3_b ([`V102Weights`] と同順、raw checkpoint format の group 順)。
    #[allow(clippy::type_complexity)]
    fn raw_ckpt_groups(
        &self,
    ) -> [(
        &'static str,
        usize,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
    ); 10] {
        let ft_w_n = FT_IN * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        [
            (
                "ft_w",
                ft_w_n,
                &self.ft_w,
                &self.ft_w_m,
                &self.ft_w_v,
                &self.ft_w_slow,
            ),
            (
                "ft_b",
                ft_b_n,
                &self.ft_b,
                &self.ft_b_m,
                &self.ft_b_v,
                &self.ft_b_slow,
            ),
            (
                "l1_w",
                l1_w_n,
                &self.l1_w,
                &self.l1_w_m,
                &self.l1_w_v,
                &self.l1_w_slow,
            ),
            (
                "l1_b",
                l1_b_n,
                &self.l1_b,
                &self.l1_b_m,
                &self.l1_b_v,
                &self.l1_b_slow,
            ),
            (
                "l1f_w",
                l1f_w_n,
                &self.l1f_w,
                &self.l1f_w_m,
                &self.l1f_w_v,
                &self.l1f_w_slow,
            ),
            (
                "l1f_b",
                l1f_b_n,
                &self.l1f_b,
                &self.l1f_b_m,
                &self.l1f_b_v,
                &self.l1f_b_slow,
            ),
            (
                "l2_w",
                l2_w_n,
                &self.l2_w,
                &self.l2_w_m,
                &self.l2_w_v,
                &self.l2_w_slow,
            ),
            (
                "l2_b",
                l2_b_n,
                &self.l2_b,
                &self.l2_b_m,
                &self.l2_b_v,
                &self.l2_b_slow,
            ),
            (
                "l3_w",
                l3_w_n,
                &self.l3_w,
                &self.l3_w_m,
                &self.l3_w_v,
                &self.l3_w_slow,
            ),
            (
                "l3_b",
                l3_b_n,
                &self.l3_b,
                &self.l3_b_m,
                &self.l3_b_v,
                &self.l3_b_slow,
            ),
        ]
    }

    /// `--resume` 用 **raw f32 checkpoint** を atomic に書き出す (Issue #88)。
    ///
    /// 量子化 `.bin` ([`GpuTrainer::save_checkpoint`]/`to_v102_weights` → `save_quantised`)
    /// は推論用 final artifact なので別途従来どおり保存される。本 method はそれとは別の
    /// `*.ckpt` file に、全 10 weight group の **raw f32** `{w, m, v, slow}` (Ranger の
    /// 1st/2nd moment + Lookahead slow weight、`grad` は resume に不要なので含めない) +
    /// `step_count` (Ranger lookahead step counter) + 完了 `superbatch` 番号を書き出す。
    ///
    /// layout (全 little-endian、[`RAW_CKPT_MAGIC`] / [`RAW_CKPT_VERSION`]):
    /// ```text
    /// 0..4     magic   b"RNRC"
    /// 4..8     version u32 (1)
    /// 8..16    superbatch u64  (この checkpoint が表す完了 superbatch、resume はこの +1 から)
    /// 16..24   step_count u64  (Ranger lookahead step counter)
    /// 24..32   num_groups u64  (= 10、固定だが将来検証用)
    /// then for each of 10 groups (順序 = `raw_ckpt_groups()` = ft_w, ft_b, l1_w, l1_b,
    ///   l1f_w, l1f_b, l2_w, l2_b, l3_w, l3_b):
    ///   len u64
    ///   w[f32 × len]
    ///   m[f32 × len]
    ///   v[f32 × len]
    ///   slow[f32 × len]
    /// ```
    ///
    /// device → host download (`DeviceBuffer::to_host_vec`) → `<path>.tmp` へ `BufWriter`
    /// で書く → `std::fs::rename(<path>.tmp, <path>)` で atomic に置換 (書き込み途中で
    /// crash しても `<path>` は前回の完全な checkpoint のまま)。
    fn save_raw_checkpoint(
        &self,
        path: &Path,
        superbatch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let tmp_path = {
            let mut p = path.as_os_str().to_os_string();
            p.push(".tmp");
            std::path::PathBuf::from(p)
        };

        // write+flush 本体を closure に括り、`fs::rename` 前の error path で
        // 中途半端な `<path>.tmp` を best-effort で消す (device→host download / write /
        // flush 失敗で残骸を残さない、Issue #88 review)。
        let write_tmp = || -> Result<(), Box<dyn std::error::Error>> {
            let groups = self.raw_ckpt_groups();
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);
            w.write_all(&RAW_CKPT_MAGIC)?;
            w.write_all(&RAW_CKPT_VERSION.to_le_bytes())?;
            w.write_all(&(superbatch as u64).to_le_bytes())?;
            w.write_all(&self.step_count.to_le_bytes())?;
            w.write_all(&(groups.len() as u64).to_le_bytes())?;

            for (name, expected_len, w_buf, m_buf, v_buf, slow_buf) in groups {
                // 念のため device buffer の要素数を arch 期待値と照合 (内部整合性)。
                let w_host = w_buf.to_host_vec(&self.stream)?;
                let m_host = m_buf.to_host_vec(&self.stream)?;
                let v_host = v_buf.to_host_vec(&self.stream)?;
                let slow_host = slow_buf.to_host_vec(&self.stream)?;
                for (label, got) in [
                    ("w", w_host.len()),
                    ("m", m_host.len()),
                    ("v", v_host.len()),
                    ("slow", slow_host.len()),
                ] {
                    if got != expected_len {
                        return Err(format!(
                            "raw checkpoint: group {name} {label} buffer len {got} != expected {expected_len}"
                        )
                        .into());
                    }
                }
                w.write_all(&(expected_len as u64).to_le_bytes())?;
                write_f32_slice(&mut w, &w_host)?;
                write_f32_slice(&mut w, &m_host)?;
                write_f32_slice(&mut w, &v_host)?;
                write_f32_slice(&mut w, &slow_host)?;
            }
            w.flush()?;
            Ok(())
        };
        if let Err(e) = write_tmp() {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        Ok(())
    }

    /// `--resume` で raw checkpoint を読み戻す (Issue #88)。返り値は checkpoint に
    /// 記録された **完了 superbatch 番号** (caller は通常その +1 から resume する)。
    ///
    /// magic / version 不一致、group 数 / 各 group の len が v102 arch と不一致、または
    /// `u64 → usize` overflow (32-bit / 破損 file) は `InvalidData` で reject
    /// (`crates/nnue-train::optimizer::RangerHostState::load_from_reader` と同方針、
    /// Codex convention #62)。読み込んだ raw f32 を host → device upload し、
    /// `self.step_count` を復元する。`grad` buffer は触らない (step ごとに memset される)。
    fn load_raw_checkpoint(&mut self, path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 4];
        read_exact_or_invalid(&mut r, &mut magic, "magic")?;
        if magic != RAW_CKPT_MAGIC {
            return Err(invalid_data(format!(
                "raw checkpoint magic mismatch: got {magic:?}, want {RAW_CKPT_MAGIC:?}"
            )));
        }
        let mut buf4 = [0u8; 4];
        read_exact_or_invalid(&mut r, &mut buf4, "version")?;
        let version = u32::from_le_bytes(buf4);
        if version != RAW_CKPT_VERSION {
            return Err(invalid_data(format!(
                "raw checkpoint version mismatch: got {version}, want {RAW_CKPT_VERSION}"
            )));
        }
        let mut buf8 = [0u8; 8];
        read_exact_or_invalid(&mut r, &mut buf8, "superbatch")?;
        let superbatch_u64 = u64::from_le_bytes(buf8);
        let superbatch: usize = superbatch_u64.try_into().map_err(|_| {
            invalid_data(format!(
                "raw checkpoint superbatch {superbatch_u64} exceeds usize::MAX"
            ))
        })?;
        read_exact_or_invalid(&mut r, &mut buf8, "step_count")?;
        let step_count = u64::from_le_bytes(buf8);
        read_exact_or_invalid(&mut r, &mut buf8, "num_groups")?;
        let num_groups_u64 = u64::from_le_bytes(buf8);

        let expected_groups: [(&'static str, usize); 10] = {
            let g = self.raw_ckpt_groups();
            [
                (g[0].0, g[0].1),
                (g[1].0, g[1].1),
                (g[2].0, g[2].1),
                (g[3].0, g[3].1),
                (g[4].0, g[4].1),
                (g[5].0, g[5].1),
                (g[6].0, g[6].1),
                (g[7].0, g[7].1),
                (g[8].0, g[8].1),
                (g[9].0, g[9].1),
            ]
        };
        if num_groups_u64 != expected_groups.len() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint num_groups {num_groups_u64} != expected {}",
                expected_groups.len()
            )));
        }

        // 各 group を読み出し → host Vec に保持 (全部読んでから upload する。途中で
        // upload して途中 fail だと中途半端な state になるため)。
        let mut loaded: Vec<RawCkptGroup> = Vec::with_capacity(10);
        for (name, expected_len) in expected_groups {
            read_exact_or_invalid(&mut r, &mut buf8, &format!("group {name} len"))?;
            let len_u64 = u64::from_le_bytes(buf8);
            let len: usize = len_u64.try_into().map_err(|_| {
                invalid_data(format!(
                    "raw checkpoint group {name} len {len_u64} exceeds usize::MAX"
                ))
            })?;
            if len != expected_len {
                return Err(invalid_data(format!(
                    "raw checkpoint group {name} len mismatch: got {len}, want {expected_len} (v102 arch)"
                )));
            }
            let w_host = read_f32_vec_io(&mut r, len, &format!("group {name} w"))?;
            let m_host = read_f32_vec_io(&mut r, len, &format!("group {name} m"))?;
            let v_host = read_f32_vec_io(&mut r, len, &format!("group {name} v"))?;
            let slow_host = read_f32_vec_io(&mut r, len, &format!("group {name} slow"))?;
            loaded.push((w_host, m_host, v_host, slow_host));
        }
        // EOF 確認 (trailing garbage は許容するが、足りないのは上で read_exact が弾く)。

        // host → device upload。order は `raw_ckpt_groups` (= ft_w, ft_b, ..., l3_b)。
        macro_rules! up {
            ($idx:expr, $w:ident, $m:ident, $v:ident, $slow:ident) => {{
                let (w, m, v, s) = &loaded[$idx];
                self.$w = DeviceBuffer::from_host(&self.stream, w)?;
                self.$m = DeviceBuffer::from_host(&self.stream, m)?;
                self.$v = DeviceBuffer::from_host(&self.stream, v)?;
                self.$slow = DeviceBuffer::from_host(&self.stream, s)?;
            }};
        }
        up!(0, ft_w, ft_w_m, ft_w_v, ft_w_slow);
        up!(1, ft_b, ft_b_m, ft_b_v, ft_b_slow);
        up!(2, l1_w, l1_w_m, l1_w_v, l1_w_slow);
        up!(3, l1_b, l1_b_m, l1_b_v, l1_b_slow);
        up!(4, l1f_w, l1f_w_m, l1f_w_v, l1f_w_slow);
        up!(5, l1f_b, l1f_b_m, l1f_b_v, l1f_b_slow);
        up!(6, l2_w, l2_w_m, l2_w_v, l2_w_slow);
        up!(7, l2_b, l2_b_m, l2_b_v, l2_b_slow);
        up!(8, l3_w, l3_w_m, l3_w_v, l3_w_slow);
        up!(9, l3_b, l3_b_m, l3_b_v, l3_b_slow);

        self.step_count = step_count;
        Ok(superbatch)
    }

    /// 全 weight buffer を host に読み出して NaN/Inf がないことを assert する smoke 用 helper。
    fn assert_all_weights_finite(&self) -> Result<(), Box<dyn std::error::Error>> {
        let groups: [(&DeviceBuffer<f32>, &str); 10] = [
            (&self.ft_w, "ft_w"),
            (&self.ft_b, "ft_b"),
            (&self.l1_w, "l1_w"),
            (&self.l1_b, "l1_b"),
            (&self.l1f_w, "l1f_w"),
            (&self.l1f_b, "l1f_b"),
            (&self.l2_w, "l2_w"),
            (&self.l2_b, "l2_b"),
            (&self.l3_w, "l3_w"),
            (&self.l3_b, "l3_b"),
        ];
        for (buf, name) in groups {
            let v = buf.to_host_vec(&self.stream)?;
            for (i, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(format!(
                        "{name}[{i}] = {x} is not finite (NaN or Inf)、smoke fail"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// 1 batch 分の forward → loss kernel → backward → Ranger step を実行。
    /// 戻り値: batch 全体の loss (f64、loss_acc から読み出し)。
    ///
    /// 実体は [`GpuTrainer::step_impl`]。本 method は `NNUE_TRAIN_STEP_PROFILE`
    /// プロファイル時の前後 sync と **teardown tick** だけを担う。`step_impl` が
    /// return すると per-step device buffer の `Drop` (= `cuMemFree`) がそこで走るので、
    /// 最後の `prof_tick!` を `step_impl` の **外** で打つことで free 時間も breakdown に
    /// 含める。Issue #78 で中間 activation / grad buffer を `GpuTrainer` 上の workspace に
    /// 永続化したため、`step_impl` で drop されるのは入力 H2D buffer (`stm_idx_dev` 等、
    /// position 数に比例した小さい buffer) だけになり、teardown tick は ~0 に落ちる (期待動作)。
    /// 入力 H2D の永続化は Issue #81 (P5、pinned + 2-stream) の範囲なので本 issue では未対応。
    fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // 環境変数 `NNUE_TRAIN_STEP_PROFILE` がセットされていれば各 phase の境界で
        // `synchronize()` + 経過時間を stderr に出す (粗い h2d / forward / backward /
        // optimizer / teardown breakdown 用、Issue #76)。未設定なら追加の sync ゼロ。
        // WSL2 では ncu の GPU perf counter が使えず nsys も GPU-side kernel trace を
        // 取れないため、この粗い event timing が代替手段。
        let profile_step = std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some();
        if profile_step {
            self.stream.synchronize()?;
        }
        let mut prof_t0 = std::time::Instant::now();
        let result = self.step_impl(batch, lr, wdl_lambda, loss, profile_step, &mut prof_t0)?;
        // step_impl の per-step device buffer はここまでに全部 drop 済 (cuMemFree)。
        if profile_step {
            self.stream.synchronize()?;
            eprintln!(
                "[step-profile] {:<10} {:8.3} ms",
                "teardown",
                prof_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(result)
    }

    /// `step` の実体。`loss` が [`LossKind::Sigmoid`] なら `loss_wdl` (plain sigmoid-MSE)、
    /// [`LossKind::Wrm`] なら `loss_wrm` (bullet win-rate-model、v102 厳密再現) を起動する。
    ///
    /// Forward path (15 step): bullet `shogi_layerstack.rs:2241-2289` の reference 実装を
    /// 本 file の `#[kernel]` 群で再現。中間 activation は `GpuTrainer` 上の永続 workspace
    /// (`self.ws.*`、Issue #78) を使い回す — forward の各 activation は読まれる前に kernel が
    /// 全 cell を上書きするので memset 不要。Backward path (~16 step): forward 逆順、`*_grad`
    /// buffer は本 method 冒頭で `memset_async(0)` で reset してから kernel が書き込む
    /// (per-bucket weight grad `dense_mm_bwd_weight_bucket` は 1 cell = 1 thread の overwrite、
    /// FT / L1f / bias の grad は atomic accumulate なので reset 必須。`dl1_total` も
    /// `slice_scatter_2d` の host 契約を守るため reset)。`loss_acc` も同様に毎 step memset。
    /// 入力 H2D buffer (`stm_idx_dev` 等) だけは per-step `DeviceBuffer::from_host` のまま
    /// (永続化は Issue #81 / P5 の範囲)。Optimizer: 10 weight groups × `radam_step`
    /// (+ 周期 `ranger_lookahead_lerp`)。
    ///
    /// `profile_step` / `prof_t0` は呼び出し元 ([`GpuTrainer::step`]) が管理し、本 method
    /// 内の `prof_tick!` が各 phase 境界で `*prof_t0` を更新する (戻った後に呼び出し元が
    /// teardown tick で読む)。
    #[allow(clippy::too_many_arguments)]
    fn step_impl(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
        profile_step: bool,
        prof_t0: &mut std::time::Instant,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        let b_u32 = b as u32;

        // 中間 activation workspace を batch `b` 以上に拡張 (grow-only、Issue #78)。
        // batch_size 固定なら起動時の `GpuWorkspace::new` で足りているので no-op。
        self.ws.ensure_batch(&self.stream, b)?;

        macro_rules! prof_tick {
            ($label:expr) => {
                if profile_step {
                    self.stream.synchronize()?;
                    let now = std::time::Instant::now();
                    eprintln!(
                        "[step-profile] {:<10} {:8.3} ms",
                        $label,
                        now.duration_since(*prof_t0).as_secs_f64() * 1000.0
                    );
                    *prof_t0 = now;
                }
            };
        }

        // 入力 buffer を host → device
        let stm_idx_dev = DeviceBuffer::from_host(&self.stream, &batch.stm_indices)?;
        let nstm_idx_dev = DeviceBuffer::from_host(&self.stream, &batch.nstm_indices)?;
        let bucket_idx_dev = DeviceBuffer::from_host(&self.stream, &batch.bucket_idx)?;
        let score_dev = DeviceBuffer::from_host(&self.stream, &batch.score)?;
        let wdl_dev = DeviceBuffer::from_host(&self.stream, &batch.wdl)?;
        let norm_dev = DeviceBuffer::from_host(&self.stream, &batch.per_pos_norm)?;

        // loss_acc reset (accumulate semantics、再 alloc せず memset、Issue #78)
        memset_zero(&self.stream, &self.loss_acc)?;
        prof_tick!("h2d+reset");

        // -- Forward step 1-2: sparse_ft_forward × 2 (stm, nstm) --
        // 中間 activation は workspace (`self.ws.*`) を使い回す (Issue #78、再 alloc 無し)。
        // forward の各 activation は読まれる前に kernel が全 cell を上書きするので memset 不要。
        cuda_launch! {
            kernel: sparse_ft_forward,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ft_w),
                slice(stm_idx_dev),
                slice_mut(self.ws.ft_stm_out),
                b_u32, FT_OUT as u32, FT_IN as u32, MAX_ACTIVE as u32
            ]
        }?;
        cuda_launch! {
            kernel: sparse_ft_forward,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ft_w),
                slice(nstm_idx_dev),
                slice_mut(self.ws.ft_nstm_out),
                b_u32, FT_OUT as u32, FT_IN as u32, MAX_ACTIVE as u32
            ]
        }?;

        // -- Forward step 3: ft_post_perspective_fwd → combined (B × FT_OUT) --
        cuda_launch! {
            kernel: ft_post_perspective_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * COMBINED_DIM),
            args: [
                slice(self.ws.ft_stm_out),
                slice(self.ws.ft_nstm_out),
                slice(self.ft_b),
                slice_mut(self.ws.combined),
                b_u32, FT_OUT as u32, FT_POST_SCALE
            ]
        }?;

        // -- Forward step 4: dense_mm_fwd_bucket L1 → l1_bucket (B × L1_OUT) --
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.combined),
                slice(self.l1_w),
                slice(self.l1_b),
                slice(bucket_idx_dev),
                slice_mut(self.ws.l1_bucket),
                b_u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 5: dense_mm_fwd L1f shared → l1f_out (B × L1_OUT) --
        cuda_launch! {
            kernel: dense_mm_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.combined),
                slice(self.l1f_w),
                slice(self.l1f_b),
                slice_mut(self.ws.l1f_out),
                b_u32, FT_OUT as u32, L1_OUT as u32
            ]
        }?;

        // -- Forward step 6: l1_total = l1_bucket + l1f_out (B × L1_OUT) --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.l1_bucket),
                slice(self.ws.l1f_out),
                slice_mut(self.ws.l1_total),
                (b * L1_OUT) as u32
            ]
        }?;

        // -- Forward step 7: slice l1_total → l1_main (B × 15) + l1_skip (B × 1) --
        cuda_launch! {
            kernel: slice_extract_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.l1_total),
                slice_mut(self.ws.l1_main),
                b_u32, L1_OUT as u32, 0_u32, L1_EFFECTIVE as u32
            ]
        }?;
        cuda_launch! {
            kernel: slice_extract_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_SKIP),
            args: [
                slice(self.ws.l1_total),
                slice_mut(self.ws.l1_skip),
                b_u32, L1_OUT as u32, L1_EFFECTIVE as u32, L1_SKIP as u32
            ]
        }?;

        // -- Forward step 8: l1_sqr = l1_main^2 * scale (B × 15) --
        cuda_launch! {
            kernel: abs_pow2_scale_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.l1_main),
                slice_mut(self.ws.l1_sqr),
                L1_SQR_SCALE,
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Forward step 9: l2_pre = concat(l1_sqr, l1_main) (B × 30) --
        cuda_launch! {
            kernel: concat_l1sqr_main_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.l1_sqr),
                slice(self.ws.l1_main),
                slice_mut(self.ws.l2_pre),
                b_u32, L1_EFFECTIVE as u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Forward step 10: l2_input = CReLU(l2_pre) (B × 30) --
        cuda_launch! {
            kernel: crelu_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.l2_pre),
                slice_mut(self.ws.l2_input),
                (b * L2_IN) as u32
            ]
        }?;

        // -- Forward step 11: L2 per-bucket dense → l2_out (B × 32) --
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.l2_input),
                slice(self.l2_w),
                slice(self.l2_b),
                slice(bucket_idx_dev),
                slice_mut(self.ws.l2_out),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 12: l2_acted = CReLU(l2_out) (B × 32) --
        cuda_launch! {
            kernel: crelu_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.l2_out),
                slice_mut(self.ws.l2_acted),
                (b * L2_OUT) as u32
            ]
        }?;

        // -- Forward step 13: L3 per-bucket dense → l3_out (B × 1) --
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.l2_acted),
                slice(self.l3_w),
                slice(self.l3_b),
                slice(bucket_idx_dev),
                slice_mut(self.ws.l3_out),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 14: net_output = l3_out + l1_skip (B × 1) --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.l3_out),
                slice(self.ws.l1_skip),
                slice_mut(self.ws.net_output),
                b_u32
            ]
        }?;

        // -- Forward step 15: loss kernel → dy_net_output + loss_acc --
        // `LossKind::Sigmoid` → `loss_wdl` (plain sigmoid-MSE)、`LossKind::Wrm` →
        // `loss_wrm` (bullet win-rate-model、v102 厳密再現)。
        match loss {
            LossKind::Sigmoid { scale } => {
                cuda_launch! {
                    kernel: loss_wdl,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b),
                    args: [
                        slice(self.ws.net_output),
                        slice(score_dev),
                        slice(wdl_dev),
                        slice(norm_dev),
                        slice_mut(self.ws.dy_net_output),
                        slice(self.loss_acc),
                        wdl_lambda, scale, b_u32
                    ]
                }?;
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
            } => {
                cuda_launch! {
                    kernel: loss_wrm,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b),
                    args: [
                        slice(self.ws.net_output),
                        slice(score_dev),
                        slice(wdl_dev),
                        slice(norm_dev),
                        slice_mut(self.ws.dy_net_output),
                        slice(self.loss_acc),
                        wdl_lambda, nnue2score, in_scaling, b_u32
                    ]
                }?;
            }
        }
        prof_tick!("forward");

        // ===== BACKWARD =====
        // 全 *_grad buffer を 0 で reset (atomic accumulate semantic に従う kernel が
        // 多い、また overwrite kernel も in-place 安全のため統一)。再 alloc せず
        // `memset_async(0)` で既存 buffer を reset (Issue #78、`ft_w_grad` だけで ~450MB
        // の `cudaMalloc`/`cudaFree` を毎 step 走らせていたのを撤廃)。
        // `dl1_total` も `slice_scatter_2d` の host 契約 (「dst を 0 初期化」) を守るため reset。
        let ft_w_n = FT_IN * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        memset_zero(&self.stream, &self.ft_w_grad)?;
        memset_zero(&self.stream, &self.ft_b_grad)?;
        memset_zero(&self.stream, &self.l1_w_grad)?;
        memset_zero(&self.stream, &self.l1_b_grad)?;
        memset_zero(&self.stream, &self.l1f_w_grad)?;
        memset_zero(&self.stream, &self.l1f_b_grad)?;
        memset_zero(&self.stream, &self.l2_w_grad)?;
        memset_zero(&self.stream, &self.l2_b_grad)?;
        memset_zero(&self.stream, &self.l3_w_grad)?;
        memset_zero(&self.stream, &self.l3_b_grad)?;
        memset_zero(&self.stream, &self.ws.dl1_total)?;

        // -- Backward 14 reverse: dy_net_output が dl3_out と dl1_skip 両方の grad --
        // (elementwise_add 逆: dl3_out = dy, dl1_skip = dy、両者同じ buffer を直接渡せばよい)

        // -- Backward 13 reverse: L3 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.dy_net_output),
                slice(self.l3_w),
                slice(bucket_idx_dev),
                slice_mut(self.ws.dl2_acted),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket,
            stream: self.stream,
            module: self.module,
            // L3: in_dim=L2_OUT, out_dim=1 → grid = num_buckets * out_dim(=1) * in_dim
            config: cfg_1d(NUM_BUCKETS * L2_OUT),
            args: [
                slice(self.ws.l2_acted),
                slice(self.ws.dy_net_output),
                slice(bucket_idx_dev),
                slice_mut(self.l3_w_grad),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.dy_net_output),
                slice(bucket_idx_dev),
                slice(self.l3_b_grad),
                b_u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Backward 12 reverse: crelu_grad on l2_out --
        cuda_launch! {
            kernel: crelu_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.l2_out),
                slice(self.ws.dl2_acted),
                slice_mut(self.ws.dl2_out),
                (b * L2_OUT) as u32
            ]
        }?;

        // -- Backward 11 reverse: L2 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.dl2_out),
                slice(self.l2_w),
                slice(bucket_idx_dev),
                slice_mut(self.ws.dl2_input),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket,
            stream: self.stream,
            module: self.module,
            // L2: in_dim=L2_IN, out_dim=L2_OUT → grid = num_buckets * out_dim * in_dim
            config: cfg_1d(NUM_BUCKETS * L2_OUT * L2_IN),
            args: [
                slice(self.ws.l2_input),
                slice(self.ws.dl2_out),
                slice(bucket_idx_dev),
                slice_mut(self.l2_w_grad),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.dl2_out),
                slice(bucket_idx_dev),
                slice(self.l2_b_grad),
                b_u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Backward 10 reverse: crelu_grad on l2_pre --
        cuda_launch! {
            kernel: crelu_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.l2_pre),
                slice(self.ws.dl2_input),
                slice_mut(self.ws.dl2_pre),
                (b * L2_IN) as u32
            ]
        }?;

        // -- Backward 9 reverse: split dl2_pre → dl1_sqr (15) + dl1_main_from_concat (15) --
        cuda_launch! {
            kernel: concat_l1sqr_main_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.dl2_pre),
                slice_mut(self.ws.dl1_sqr),
                slice_mut(self.ws.dl1_main_from_concat),
                b_u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Backward 8 reverse: abs_pow2_scale_grad (l1_sqr 経由の grad) --
        cuda_launch! {
            kernel: abs_pow2_scale_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.l1_main),
                slice(self.ws.dl1_sqr),
                slice_mut(self.ws.dl1_main_from_sqr),
                L1_SQR_SCALE,
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Combine dl1_main = dl1_main_from_concat + dl1_main_from_sqr --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.dl1_main_from_concat),
                slice(self.ws.dl1_main_from_sqr),
                slice_mut(self.ws.dl1_main),
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Backward 7 reverse: assemble dl1_total from dl1_main (offset 0) + dl1_skip=dy_net_output (offset 15) --
        cuda_launch! {
            kernel: slice_scatter_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.dl1_main),
                slice_mut(self.ws.dl1_total),
                b_u32, L1_EFFECTIVE as u32, L1_OUT as u32, 0_u32
            ]
        }?;
        cuda_launch! {
            kernel: slice_scatter_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_SKIP),
            args: [
                slice(self.ws.dy_net_output),
                slice_mut(self.ws.dl1_total),
                b_u32, L1_SKIP as u32, L1_OUT as u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Backward 6 reverse: dl1_total を l1_bucket と l1f_out 両方の grad に流す --
        // (elementwise_add 逆: dl1_bucket = dl1_total, dl1f = dl1_total)
        // 直接 dl1_total を両 dense_mm_bwd に渡す

        // -- Backward 5 reverse: L1f shared dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1f_w),
                slice_mut(self.ws.dcombined_from_l1f),
                b_u32, FT_OUT as u32, L1_OUT as u32
            ]
        }?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(FT_OUT * L1_OUT),
            args: [
                slice(self.ws.combined),
                slice(self.ws.dl1_total),
                slice_mut(self.l1f_w_grad),
                b_u32, FT_OUT as u32, L1_OUT as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1f_b_grad),
                b_u32, L1_OUT as u32
            ]
        }?;

        // -- Backward 4 reverse: L1 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1_w),
                slice(bucket_idx_dev),
                slice_mut(self.ws.dcombined_from_l1),
                b_u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket,
            stream: self.stream,
            module: self.module,
            // L1: in_dim=FT_OUT, out_dim=L1_OUT → grid = num_buckets * out_dim * in_dim
            config: cfg_1d(NUM_BUCKETS * L1_OUT * FT_OUT),
            args: [
                slice(self.ws.combined),
                slice(self.ws.dl1_total),
                slice(bucket_idx_dev),
                slice_mut(self.l1_w_grad),
                b_u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(bucket_idx_dev),
                slice(self.l1_b_grad),
                b_u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Combine dcombined = dcombined_from_l1 + dcombined_from_l1f --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dcombined_from_l1),
                slice(self.ws.dcombined_from_l1f),
                slice_mut(self.ws.dcombined),
                (b * FT_OUT) as u32
            ]
        }?;

        // -- Backward 3 reverse: ft_post_perspective_grad × 2 (stm, nstm) --
        // stm: d_combined_offset = 0
        cuda_launch! {
            kernel: ft_post_perspective_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dcombined),
                slice(self.ws.ft_stm_out),
                slice(self.ft_b),
                slice_mut(self.ws.dft_stm_out),
                slice(self.ft_b_grad),
                b_u32, FT_OUT as u32, 0_u32, COMBINED_DIM as u32, FT_POST_SCALE
            ]
        }?;
        // nstm: d_combined_offset = FT_OUT/2 = 768
        cuda_launch! {
            kernel: ft_post_perspective_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dcombined),
                slice(self.ws.ft_nstm_out),
                slice(self.ft_b),
                slice_mut(self.ws.dft_nstm_out),
                slice(self.ft_b_grad),
                b_u32, FT_OUT as u32, (FT_OUT / 2) as u32, COMBINED_DIM as u32, FT_POST_SCALE
            ]
        }?;

        // -- Backward 1+2 reverse: sparse_ft_backward × 2 (atomic accumulate ft_w_grad) --
        cuda_launch! {
            kernel: sparse_ft_backward,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dft_stm_out),
                slice(stm_idx_dev),
                slice(self.ft_w_grad),
                b_u32, FT_OUT as u32, FT_IN as u32, MAX_ACTIVE as u32
            ]
        }?;
        cuda_launch! {
            kernel: sparse_ft_backward,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dft_nstm_out),
                slice(nstm_idx_dev),
                slice(self.ft_w_grad),
                b_u32, FT_OUT as u32, FT_IN as u32, MAX_ACTIVE as u32
            ]
        }?;
        prof_tick!("backward");

        // ===== OPTIMIZER STEP (Ranger = RAdam + Lookahead) =====
        self.step_count += 1;
        let (step_size, denom) =
            radam_compute_step_size_denom(self.step_count, BETA1, BETA2, N_SMA_THRESHOLD);

        // 10 weight groups × radam_step
        // FT
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
            args: [slice_mut(self.ft_w), slice_mut(self.ft_w_m), slice_mut(self.ft_w_v),
                   slice_mut(self.ft_w_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, ft_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(ft_b_n),
            args: [slice_mut(self.ft_b), slice_mut(self.ft_b_m), slice_mut(self.ft_b_v),
                   slice_mut(self.ft_b_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, ft_b_n as u32]
        }?;
        // L1
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1_w_n),
            args: [slice_mut(self.l1_w), slice_mut(self.l1_w_m), slice_mut(self.l1_w_v),
                   slice_mut(self.l1_w_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1_b_n),
            args: [slice_mut(self.l1_b), slice_mut(self.l1_b_m), slice_mut(self.l1_b_v),
                   slice_mut(self.l1_b_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1_b_n as u32]
        }?;
        // L1f
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1f_w_n),
            args: [slice_mut(self.l1f_w), slice_mut(self.l1f_w_m), slice_mut(self.l1f_w_v),
                   slice_mut(self.l1f_w_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1f_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1f_b_n),
            args: [slice_mut(self.l1f_b), slice_mut(self.l1f_b_m), slice_mut(self.l1f_b_v),
                   slice_mut(self.l1f_b_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1f_b_n as u32]
        }?;
        // L2
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l2_w_n),
            args: [slice_mut(self.l2_w), slice_mut(self.l2_w_m), slice_mut(self.l2_w_v),
                   slice_mut(self.l2_w_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l2_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l2_b_n),
            args: [slice_mut(self.l2_b), slice_mut(self.l2_b_m), slice_mut(self.l2_b_v),
                   slice_mut(self.l2_b_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l2_b_n as u32]
        }?;
        // L3
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l3_w_n),
            args: [slice_mut(self.l3_w), slice_mut(self.l3_w_m), slice_mut(self.l3_w_v),
                   slice_mut(self.l3_w_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l3_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l3_b_n),
            args: [slice_mut(self.l3_b), slice_mut(self.l3_b_m), slice_mut(self.l3_b_v),
                   slice_mut(self.l3_b_grad), lr, step_size, denom, DECAY, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l3_b_n as u32]
        }?;

        // Lookahead lerp every K steps
        if self.step_count % RANGER_K == 0 {
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), RANGER_ALPHA, ft_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(ft_b_n),
                args: [slice_mut(self.ft_b), slice_mut(self.ft_b_slow), RANGER_ALPHA, ft_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1_w_n),
                args: [slice_mut(self.l1_w), slice_mut(self.l1_w_slow), RANGER_ALPHA, l1_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1_b_n),
                args: [slice_mut(self.l1_b), slice_mut(self.l1_b_slow), RANGER_ALPHA, l1_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1f_w_n),
                args: [slice_mut(self.l1f_w), slice_mut(self.l1f_w_slow), RANGER_ALPHA, l1f_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1f_b_n),
                args: [slice_mut(self.l1f_b), slice_mut(self.l1f_b_slow), RANGER_ALPHA, l1f_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l2_w_n),
                args: [slice_mut(self.l2_w), slice_mut(self.l2_w_slow), RANGER_ALPHA, l2_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l2_b_n),
                args: [slice_mut(self.l2_b), slice_mut(self.l2_b_slow), RANGER_ALPHA, l2_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l3_w_n),
                args: [slice_mut(self.l3_w), slice_mut(self.l3_w_slow), RANGER_ALPHA, l3_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l3_b_n),
                args: [slice_mut(self.l3_b), slice_mut(self.l3_b_slow), RANGER_ALPHA, l3_b_n as u32]
            }?;
        }
        prof_tick!("optimizer");

        self.stream.synchronize()?;

        // Read loss (`loss_acc` は 1-cell f64 buffer。空なら異常 → error にして無防備 index を避ける)。
        // この後 step_impl が return すると per-step buffer が drop され、呼び出し元 `step` が
        // teardown tick を打つ (`*prof_t0` は上の `prof_tick!("optimizer")` が最後に更新した値)。
        let loss_vec = self.loss_acc.to_host_vec(&self.stream)?;
        loss_vec
            .first()
            .copied()
            .ok_or_else(|| -> Box<dyn std::error::Error> { "loss_acc buffer is empty".into() })
    }
}

// step() / step_impl() 実装は別 impl block (file 分割回避のため同 file 内)。

// ===========================================================================
// TrainerBackend impl — `nnue-train::trainer::run` から 1 batch ずつ呼ばれる
// ===========================================================================

impl TrainerBackend for GpuTrainer {
    fn train_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<f64> {
        let data = BatchData::from_batch(batch, bucket_idx);
        self.step(&data, lr, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("GpuTrainer::step failed: {e}")))
    }

    fn save_checkpoint(&mut self, path: &Path) -> std::io::Result<()> {
        let weights = self.to_v102_weights().map_err(|e| {
            std::io::Error::other(format!("GpuTrainer::to_v102_weights failed: {e}"))
        })?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        weights.save_quantised(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    fn save_resume_checkpoint(&mut self, path: &Path, superbatch: usize) -> std::io::Result<()> {
        self.save_raw_checkpoint(path, superbatch).map_err(|e| {
            // 既に io::Error なら kind を保つ、それ以外は other で包む。
            match e.downcast::<std::io::Error>() {
                Ok(io_err) => *io_err,
                Err(other) => std::io::Error::other(format!(
                    "GpuTrainer::save_raw_checkpoint failed: {other}"
                )),
            }
        })
    }
}

// ===========================================================================
// CLI (clap) — Stage 3-8 (#65)、bullet `examples/shogi_layerstack.rs` の引数群を
// v102 recipe (memory project_v102_recipe.md) に合わせて受ける
// ===========================================================================

/// bullet-shogi v102 互換 HalfKA_hm 1536-16-32 LayerStack NNUE trainer。
///
/// `--data <PSV>` を指定すると training loop を回す。省略すると GPU smoke test
/// (`GpuTrainer` の forward/backward path 確認、Stage 3-7 由来) を実行する。
#[derive(Parser, Debug)]
#[command(name = "nnue-train", about = "rshogi NNUE trainer (v102 LayerStack)")]
struct Cli {
    /// 教師データ PSV ファイル (`PackedSfenValue` × N、各 40 bytes)。省略時は GPU smoke test。
    #[arg(long)]
    data: Option<PathBuf>,

    /// checkpoint 出力先 directory (`{net_id}-{superbatch}.bin` を書き出す)。
    #[arg(long, default_value = "checkpoints")]
    output: PathBuf,

    /// network id (checkpoint file 名に使う)。
    #[arg(long, default_value = "rshogi")]
    net_id: String,

    /// 学習する superbatch 数 (1..=superbatches を回す)。default 10 は smoke 用、
    /// v102 recipe は 400 (memory project_v102_recipe.md)。
    #[arg(long, default_value_t = 10)]
    superbatches: usize,

    /// 1 superbatch あたりの batch 数 (v102 recipe = 6104)。
    #[arg(long, default_value_t = 6104)]
    batches_per_superbatch: usize,

    /// 1 batch あたりの position 数。default 16384 は本リポ既定、v102 recipe は 65536。
    #[arg(long, default_value_t = 16384)]
    batch_size: usize,

    /// 初期 learning rate。
    #[arg(long, default_value_t = 8.75e-4)]
    lr: f32,

    /// LR gamma (`lr_step` superbatch ごとに gamma 倍)。
    #[arg(long, default_value_t = 0.995)]
    lr_gamma: f32,

    /// LR step (gamma 倍する superbatch 間隔)。
    #[arg(long, default_value_t = 1)]
    lr_step: usize,

    /// WDL blend lambda (constant)。
    #[arg(long, default_value_t = 0.0)]
    wdl: f32,

    /// sigmoid loss の score scale (`loss_scale = 1 / scale`)。`--win-rate-model` 指定時は
    /// 使わない (WRM loss は in_scaling=380/340 を使う)。
    #[arg(long, default_value_t = 290.0)]
    scale: f32,

    /// `save_rate` superbatch ごと (および末尾) に checkpoint を書き出す。
    #[arg(long, default_value_t = 20)]
    save_rate: usize,

    /// progress8kpabs 係数ファイル (YaneuraOu 互換 `progress.bin`、f64 LE × 81*FE_OLD_END)。
    /// 未指定なら全 position が bucket 4 (zero weights → `sigmoid(0) = 0.5`)。
    #[arg(long)]
    progress_coeff: Option<PathBuf>,

    /// `|score| >= score_drop_abs` の position を loss から除外する (bullet `--score-drop-abs`)。
    #[arg(long)]
    score_drop_abs: Option<i32>,

    /// 学習開始前に量子化 v102 NNUE binary から weight を注入する (pretrained start)。
    /// optimizer state (Ranger m/v/slow/step) は **reset** される — 真の resume には
    /// `--resume` を使うこと (`--init-from` と `--resume` は排他)。
    #[arg(long)]
    init_from: Option<PathBuf>,

    /// raw checkpoint (`{net_id}-{sb}.ckpt`) から weight + Ranger optimizer state
    /// (m/v/slow/step) を復元して学習を再開する (Issue #88、真の resume)。`--init-from`
    /// とは排他 (`--init-from` は weight のみ注入し optimizer を reset するため)。
    /// `--start-superbatch` 未指定なら checkpoint に記録された superbatch の +1 から再開。
    #[arg(long)]
    resume: Option<PathBuf>,

    /// 学習を開始する superbatch 番号 (1-indexed, inclusive)。未指定時:
    /// `--resume` あり → checkpoint の superbatch +1、なし → 1。`1 <= N <= --superbatches`
    /// の範囲外ならエラー (resume で過去 sb をやり直す目的で明示指定も可)。
    #[arg(long)]
    start_superbatch: Option<usize>,

    /// raw checkpoint (`*.ckpt`) を直近 N 個だけ残す (Issue #88、ディスク節約)。
    /// 未指定なら全保持 (raw state は ~1.8GB/個 なので save-rate × superbatches が
    /// 大きい長期ランでは指定推奨; 例 save-rate 20 / 400sb = 20 個 ≈ 36GB)。量子化
    /// `.bin` (~116MB) は本設定に関わらず常に全保持 (推論 artifact)。
    #[arg(long)]
    keep_checkpoints: Option<usize>,

    /// bullet win-rate-model loss を使う (v102 recipe)。指定時は `loss_wrm` kernel
    /// (prediction / target 双方に nodchip 流 WRM) を使い、未指定なら `loss_wdl`
    /// (plain sigmoid-MSE + `--scale`)。net_output のスケールが `out ≈ cp/--wrm-nnue2score`
    /// になり量子化 (`QA=127/QB=64/FV_SCALE=28`) と整合するので bullet v102 互換 net を
    /// 学習するには必須。
    #[arg(long)]
    win_rate_model: bool,
    /// WRM prediction 側の in-scaling (nodchip default 340)。target 側は bullet ハードコード
    /// の 380。`--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 340.0)]
    wrm_in_scaling: f32,
    /// WRM の nnue2score (`scorenet = net_output * --wrm-nnue2score`、nodchip default 600)。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 600.0)]
    wrm_nnue2score: f32,
    // --- 以下は v102 recipe との CLI 互換のために受けるが、本 stage では未配線 ---
    /// optimizer 名 ("ranger" のみ実装)。
    #[arg(long, default_value = "ranger")]
    optimizer: String,
    /// weight decay (kernel は 0.0 固定、非 0 指定で warning)。
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// dataloader prefetch worker 数 (Issue #89)。各 worker が PSV パース +
    /// HalfKA_hm sparse 抽出 + progress8kpabs bucket 計算を `decode()` 1 回で済ませて
    /// 先読み供給する。`1` で従来の決定論的逐次 read 相当、`>= 2` で並列パース
    /// (1 epoch 内の position 順序は非決定的; training では問題ない)。
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// bucket mode ("progress8kpabs" のみ実装)。
    #[arg(long, default_value = "progress8kpabs")]
    bucket_mode: String,
    /// (受けるが未実装) epoch ごとに file shuffle する。本実装は逐次 read + EOF wrap
    /// (worker 数 >= 2 では各 worker が排他的に chunk を読むため batch 境界は epoch ごと
    /// 不変ではないが、明示的 file-level shuffle は別 issue)。
    #[arg(long)]
    epoch_file_shuffle: bool,
    /// (受けるが未使用) file shuffle seed。
    #[arg(long, default_value_t = 0)]
    file_shuffle_seed: u64,
}

fn run_training(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let data = cli.data.as_ref().expect("run_training called with --data");

    // --- 未実装フラグの validation / warning ---
    if cli.bucket_mode != "progress8kpabs" {
        return Err(format!(
            "--bucket-mode '{}' is not implemented (only 'progress8kpabs')",
            cli.bucket_mode
        )
        .into());
    }
    if !cli.optimizer.eq_ignore_ascii_case("ranger") {
        return Err(format!(
            "--optimizer '{}' is not implemented (only 'ranger')",
            cli.optimizer
        )
        .into());
    }
    // NaN / 範囲外を kernel に流さない (TrainingConfig::validate は loss params のみ見る)。
    if !(cli.lr.is_finite() && cli.lr > 0.0) {
        return Err(format!("--lr must be finite and > 0 (got {})", cli.lr).into());
    }
    if !cli.lr_gamma.is_finite() || cli.lr_gamma <= 0.0 {
        return Err(format!("--lr-gamma must be finite and > 0 (got {})", cli.lr_gamma).into());
    }
    if !cli.wdl.is_finite() || !(0.0..=1.0).contains(&cli.wdl) {
        return Err(format!("--wdl must be finite and in [0.0, 1.0] (got {})", cli.wdl).into());
    }
    // loss kernel の選択: --win-rate-model → loss_wrm (bullet v102)、未指定 → loss_wdl。
    let loss = if cli.win_rate_model {
        if !(cli.wrm_in_scaling.is_finite() && cli.wrm_in_scaling > 0.0) {
            return Err(format!(
                "--wrm-in-scaling must be finite and > 0 (got {})",
                cli.wrm_in_scaling
            )
            .into());
        }
        if !(cli.wrm_nnue2score.is_finite() && cli.wrm_nnue2score > 0.0) {
            return Err(format!(
                "--wrm-nnue2score must be finite and > 0 (got {})",
                cli.wrm_nnue2score
            )
            .into());
        }
        LossKind::Wrm {
            nnue2score: cli.wrm_nnue2score,
            in_scaling: cli.wrm_in_scaling,
        }
    } else {
        if !(cli.scale.is_finite() && cli.scale > 0.0) {
            return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
        }
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };
    if cli.weight_decay != 0.0 {
        eprintln!(
            "[train] warning: --weight-decay {} ignored; the Ranger kernel uses weight-decay 0.0 (v102)",
            cli.weight_decay
        );
    }
    if cli.epoch_file_shuffle {
        eprintln!(
            "[train] warning: --epoch-file-shuffle is not implemented; reading {} sequentially and wrapping at EOF (--file-shuffle-seed {} ignored). With --threads >= 2 each worker reads a disjoint chunk per batch, so batch boundaries are not identical across epochs, but no explicit file-level shuffle is performed.",
            data.display(),
            cli.file_shuffle_seed
        );
    }
    if cli.threads == 0 {
        return Err("--threads must be >= 1".into());
    }
    if cli.init_from.is_some() && cli.resume.is_some() {
        return Err("--init-from and --resume are mutually exclusive (--init-from injects weights but resets the Ranger optimizer state; --resume preserves it)".into());
    }
    if cli.superbatches == 0 {
        return Err("--superbatches must be >= 1".into());
    }
    if let Some(0) = cli.keep_checkpoints {
        return Err(
            "--keep-checkpoints must be >= 1 when set (0 would delete every raw checkpoint)".into(),
        );
    }

    std::fs::create_dir_all(&cli.output)?;

    // progress8kpabs weights (process-global; 未指定なら zero → 全 bucket 4)
    let progress = match &cli.progress_coeff {
        Some(p) => {
            println!("[train] loading progress8kpabs coeff: {}", p.display());
            ShogiProgressKPAbs::load_from_bin(p).map_err(|e| -> Box<dyn std::error::Error> {
                format!("failed to load --progress-coeff {}: {e}", p.display()).into()
            })?
        }
        None => {
            eprintln!(
                "[train] note: --progress-coeff not given; all positions map to bucket 4 (sigmoid(0) = 0.5)"
            );
            ShogiProgressKPAbs
        }
    };

    let ctx = CudaContext::new(0)?;
    println!("[train] CUDA context ready, building GpuTrainer (v102 LayerStack)...");
    // workspace を batch_size 分で確保 (Issue #78、partial 末尾 batch は grow-only で対応)。
    let mut trainer = GpuTrainer::new(&ctx, cli.batch_size)?;
    // resume / init-from の処理 → resumed_superbatch を決める。
    let resumed_superbatch: Option<usize> = if let Some(init) = &cli.init_from {
        println!(
            "[train] injecting pretrained weights from {} (optimizer state reset)",
            init.display()
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
        let weights = V102Weights::load_quantised(&mut reader)?;
        trainer.load_v102_weights(&weights)?;
        None
    } else if let Some(ckpt) = &cli.resume {
        let sb = trainer.load_raw_checkpoint(ckpt)?;
        println!(
            "[train] resuming from {} at superbatch {}",
            ckpt.display(),
            sb + 1
        );
        Some(sb)
    } else {
        None
    };

    // start_superbatch の決定 + 範囲チェック (1 <= start <= --superbatches)。
    let start_superbatch = match cli.start_superbatch {
        Some(n) => n,
        None => match resumed_superbatch {
            Some(sb) => sb + 1,
            None => 1,
        },
    };
    if start_superbatch == 0 {
        return Err("--start-superbatch must be >= 1 (1-indexed)".into());
    }
    if start_superbatch > cli.superbatches {
        return Err(format!(
            "--start-superbatch {start_superbatch} > --superbatches {} (nothing to train); pass a larger --superbatches or a smaller start",
            cli.superbatches
        )
        .into());
    }
    if cli.resume.is_some() && cli.start_superbatch.is_some() {
        println!(
            "[train] (--start-superbatch {start_superbatch} overrides the resumed checkpoint's superbatch+1)"
        );
    }

    let lr_scheduler = StepLR {
        start: cli.lr,
        gamma: cli.lr_gamma,
        step: cli.lr_step.max(1),
    };
    let wdl_scheduler = ConstantWDL { value: cli.wdl };
    let cfg = TrainingConfig {
        net_id: cli.net_id.clone(),
        output_dir: cli.output.clone(),
        start_superbatch,
        end_superbatch: cli.superbatches,
        batches_per_superbatch: cli.batches_per_superbatch,
        batch_size: cli.batch_size,
        save_rate: cli.save_rate,
        keep_raw_checkpoints: cli.keep_checkpoints,
        loss,
        score_drop_abs: cli.score_drop_abs,
        threads: cli.threads,
    };

    nnue_train::trainer::run(
        &mut trainer,
        data,
        &progress,
        &lr_scheduler,
        &wdl_scheduler,
        &cfg,
    )?;
    Ok(())
}

fn smoke_test() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    println!("[smoke] CUDA context created, loading kernel module...");
    // workspace を smoke の固定 batch 分で確保 (Issue #78)。
    let mut trainer = GpuTrainer::new(&ctx, SMOKE_BATCH)?;
    println!(
        "[smoke] GpuTrainer ready: 10 weight groups, ~{:.1}M params total",
        (FT_IN * FT_OUT
            + FT_OUT
            + NUM_BUCKETS * L1_OUT * FT_OUT
            + NUM_BUCKETS * L1_OUT
            + FT_OUT * L1_OUT
            + L1_OUT
            + NUM_BUCKETS * L2_OUT * L2_IN
            + NUM_BUCKETS * L2_OUT
            + NUM_BUCKETS * L2_OUT
            + NUM_BUCKETS) as f64
            / 1.0e6
    );

    trainer.assert_all_weights_finite()?;
    println!("[smoke] step 0: init weights all finite ✓");

    // /tmp/v102_100_quantised.bin が利用可能なら、bullet v102-100 (sb=100 checkpoint) を
    // 注入して **golden forward 経路** (forward + backward + save) を検証する。
    // 不在時は random init smoke のみ。
    let v102_path = "/tmp/v102_100_quantised.bin";
    if std::path::Path::new(v102_path).exists() {
        println!("[smoke] loading bullet v102-100 reference from {v102_path} ...");
        let mut reader = std::io::BufReader::new(std::fs::File::open(v102_path)?);
        let weights = V102Weights::load_quantised(&mut reader)?;
        trainer.load_v102_weights(&weights)?;
        trainer.assert_all_weights_finite()?;
        println!("[smoke] v102-100 weights injected, all finite ✓");

        // forward + step 1 batch (sigmoid-MSE、golden forward/backward/save 経路)
        let batch = BatchData::smoke_dummy(SMOKE_BATCH);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch, lr, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke] step 1 (post-v102-100 init, sigmoid-MSE): loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

        // save back as our quantised.bin
        let out_path = "/tmp/our_quantised.bin";
        println!("[smoke] saving trained weights to {out_path} ...");
        let saved_weights = trainer.to_v102_weights()?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(out_path)?);
        saved_weights.save_quantised(&mut writer)?;
        drop(writer);
        let out_size = std::fs::metadata(out_path)?.len();
        println!("[smoke] wrote {out_path}: {out_size} bytes");
        println!(
            "[smoke] verify with:\n  /home/sh11235/git-repos/rshogi-oss/target/release/verify_nnue_accumulator \\\n    --nnue-file {out_path} \\\n    --ls-progress-coeff /mnt/e/rshogi-nnue/data/progress/progress_hao_full_cuda.e1.bin \\\n    --moves 10"
        );

        // 追加 step: WRM loss kernel (`loss_wrm`) を runtime でも exercise する (#84)。
        // 上で save 済なので weights が変わっても verify 対象 (`out_path`) には影響しない。
        let batch = BatchData::smoke_dummy(SMOKE_BATCH);
        let loss_wrm = trainer.step(&batch, 1e-3_f32, WDL_LAMBDA, SMOKE_LOSS_WRM)?;
        println!("[smoke] step 2 (win-rate-model): loss = {loss_wrm:.6e}");
        if !loss_wrm.is_finite() {
            return Err(format!("step 2 (wrm) loss = {loss_wrm} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 2: all weights finite ✓");
    } else {
        println!("[smoke] (no v102_100_quantised.bin available; running random-init smoke only)");
        let batch = BatchData::smoke_dummy(SMOKE_BATCH);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch, lr, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke] step 1 (sigmoid-MSE): loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

        // step 2: WRM loss kernel (`loss_wrm`) を runtime でも exercise する (#84)。
        let loss_wrm = trainer.step(&batch, lr, WDL_LAMBDA, SMOKE_LOSS_WRM)?;
        println!("[smoke] step 2 (win-rate-model): loss = {loss_wrm:.6e}");
        if !loss_wrm.is_finite() {
            return Err(format!("step 2 (wrm) loss = {loss_wrm} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 2: all weights finite ✓");

        // save random-init as quantised.bin for verify-nnue check
        let out_path = "/tmp/our_quantised_randinit.bin";
        let saved_weights = trainer.to_v102_weights()?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(out_path)?);
        saved_weights.save_quantised(&mut writer)?;
        drop(writer);
        let out_size = std::fs::metadata(out_path)?.len();
        println!("[smoke] wrote {out_path}: {out_size} bytes");
    }

    println!("[smoke] PASSED — Stage 3-7 GpuTrainer skeleton OK (v102 arch full path)");
    Ok(())
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = if cli.data.is_some() {
        run_training(&cli)
    } else {
        smoke_test()
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::from(1)
        }
    }
}

// ===========================================================================
// raw checkpoint format helper tests (Issue #88、GPU 不要)
// ===========================================================================
#[cfg(test)]
mod raw_ckpt_format_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn write_then_read_f32_slice_round_trips() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 3.5, f32::MIN_POSITIVE, -42.25, 1e9];
        let mut buf = Vec::new();
        write_f32_slice(&mut buf, &data).unwrap();
        assert_eq!(buf.len(), data.len() * 4);
        let back = read_f32_vec_io(&mut Cursor::new(&buf), data.len(), "test").unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn empty_f32_slice_round_trips() {
        let mut buf = Vec::new();
        write_f32_slice(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
        let back = read_f32_vec_io(&mut Cursor::new(&buf), 0, "test").unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn read_f32_vec_errors_on_short_input() {
        // 破損 / 部分書き checkpoint の短読みは `UnexpectedEof` ではなく context 付き
        // `InvalidData` に正規化される (robustness contract: malformed input は全部 InvalidData)。
        let buf = vec![0u8; 6]; // 1.5 f32 worth
        let err = read_f32_vec_io(&mut Cursor::new(&buf), 2, "group ft_w w")
            .expect_err("must error on short read");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("truncated") && err.to_string().contains("group ft_w w"),
            "error message should describe the truncated field: {err}"
        );
    }

    #[test]
    fn read_exact_or_invalid_maps_eof_to_invalid_data() {
        let buf = vec![0u8; 2];
        let mut out = [0u8; 8];
        let err = read_exact_or_invalid(&mut Cursor::new(&buf), &mut out, "superbatch")
            .expect_err("must error on short read");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("truncated"));
        assert!(err.to_string().contains("superbatch"));
    }

    #[test]
    fn raw_ckpt_constants_are_stable() {
        // format identity が変わると古い checkpoint を resume できなくなるので pin。
        assert_eq!(&RAW_CKPT_MAGIC, b"RNRC");
        assert_eq!(RAW_CKPT_VERSION, 1);
    }

    #[test]
    fn invalid_data_helper_makes_invalid_data_error() {
        let e = invalid_data("boom".to_string());
        let io_err = e.downcast::<std::io::Error>().expect("is io::Error");
        assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidData);
        assert!(io_err.to_string().contains("boom"));
    }
}

// ===========================================================================
// GPU ↔ CPU reference 数値同等性テスト (Issue #85)
//
// 本 module は **GPU 必須**。CI ではないローカル sm_75 box でのみ走る想定で、
// `#[cfg(test)]` を本 main.rs 内に置くことで kernel symbol (上の `#[kernel]` 群) に
// 直接 path 解決できる (Stage 1-10 / Stage 2 (`experiments/002-fused-kernels`) と
// 同パターン、tests/*.rs では bin の `#[kernel]` に届かない)。`nnue-trainer` は
// 既に workspace `--exclude` で CI から外れているので CI には影響しない (が typecheck
// は通る必要がある — `cargo test -p nnue-trainer --release --no-run`)。
//
// 走らせる:
//
// ```bash
// cd bins/nnue_train
// CUDA_OXIDE_TARGET=sm_75 /mnt/e/cuda-oxide-target/release/cargo-oxide build
// cd ../.. && cargo test -p nnue-trainer --release -- --test-threads=1
// ```
//
// 各テストは小規模 batch (b = 3〜4) で GPU kernel を launch → download → 上の
// `gpu_kernels::layerstack::*_cpu` reference と比較。`-1` padding (sparse index /
// bucket_idx)、全 9 bucket、CReLU 境界値 (ちょうど 0.0 / 1.0 / 負)、NaN 伝搬を含む。
// tolerance: forward / gradient 1e-5、整数/index 出力は完全一致 (Stage 1/2 基準)。
//
// kernel ↔ CPU ref 対応表は `gpu_kernels::layerstack` の module doc 参照。
#[cfg(test)]
mod gpu_cpu_equivalence_tests {
    use super::*;
    use gpu_kernels::layerstack::{
        abs_pow2_scale::{abs_pow2_scale_fwd_cpu, abs_pow2_scale_grad_cpu},
        concat_l1sqr_main::{concat_l1sqr_main_fwd_cpu, concat_l1sqr_main_grad_cpu},
        crelu::{crelu_fwd_cpu, crelu_grad_cpu},
        dense_mm::{
            bias_grad_cpu, dense_mm_bwd_input_cpu, dense_mm_bwd_weight_cpu, dense_mm_fwd_cpu,
        },
        dense_mm_bucket::{
            bias_grad_bucket_cpu, dense_mm_bwd_input_bucket_cpu, dense_mm_bwd_weight_bucket_cpu,
            dense_mm_fwd_bucket_cpu,
        },
        elementwise::elementwise_add_cpu,
        ft_post_perspective::{ft_post_perspective_fwd_cpu, ft_post_perspective_grad_cpu},
        slice2d::{slice_extract_2d_cpu, slice_scatter_2d_cpu},
    };

    /// forward / gradient の f32 tolerance (Stage 1/2 標準の 1e-5)。
    const TOL: f32 = 1e-5;

    type CudaCtxModuleStream = (
        std::sync::Arc<CudaContext>,
        std::sync::Arc<CudaModule>,
        std::sync::Arc<CudaStream>,
    );

    fn open_module() -> Result<CudaCtxModuleStream, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(&ctx, "nnue_train")?;
        Ok((ctx, module, stream))
    }

    /// 決定論的な「面白い」値列を作る (interior / CReLU 境界 0.0・1.0 / 負 / >1 を踏む)。
    fn deterministic_floats(n: usize, seed: f32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            // -1.5 .. 1.5 を span、加えて i % 5 == 0/1 でちょうど 0.0 / 1.0 を入れる
            let r = match i % 7 {
                0 => 0.0_f32,
                1 => 1.0_f32,
                2 => -0.5_f32,
                3 => 1.5_f32,
                _ => {
                    let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
                    -1.5_f32 + 3.0_f32 * (i as f32) / denom + 0.0137_f32 * seed
                }
            };
            v.push(r);
        }
        v
    }

    fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: len mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if c.is_nan() {
                assert!(g.is_nan(), "{label}[{i}]: cpu=NaN but gpu={g}");
            } else {
                let diff = (g - c).abs();
                assert!(
                    diff <= tol,
                    "{label}[{i}]: gpu={g} cpu={c} diff={diff} > {tol}"
                );
            }
        }
    }

    /// `assert_close` の relative-tolerance 版。atomic reduce (`fetch_add`) で複数
    /// thread が 1 cell に加算する出力は加算順序が GPU と CPU で異なり、和の大きさに
    /// 比例した f32 round-off drift が出る (Stage 2-2 で atomic loss を 1e-6→1e-5 に
    /// 緩めたのと同根)。`|gpu - cpu| <= tol * (1 + |cpu|)` で判定する。
    fn assert_close_rel(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: len mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if c.is_nan() {
                assert!(g.is_nan(), "{label}[{i}]: cpu=NaN but gpu={g}");
            } else {
                let diff = (g - c).abs();
                let bound = tol * (1.0_f32 + c.abs());
                assert!(
                    diff <= bound,
                    "{label}[{i}]: gpu={g} cpu={c} diff={diff} > {bound} (tol={tol})"
                );
            }
        }
    }

    // -- crelu --------------------------------------------------------------

    #[test]
    fn crelu_fwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 257_usize;
        let mut x = deterministic_floats(n, 1.0);
        x.push(f32::NAN); // NaN propagation: clip(NaN) → NaN (if-else passes through)
        let n = x.len();
        let mut y_cpu = vec![0.0_f32; n];
        crelu_fwd_cpu(&x, &mut y_cpu, n);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        cuda_launch! {
            kernel: crelu_fwd, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice_mut(y_dev), n as u32]
        }?;
        stream.synchronize()?;
        assert_close("crelu_fwd", &y_dev.to_host_vec(&stream)?, &y_cpu, 0.0);
        Ok(())
    }

    #[test]
    fn crelu_grad_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 257_usize;
        let mut x = deterministic_floats(n, 2.0);
        x.push(f32::NAN);
        let n = x.len();
        let dy: Vec<f32> = (0..n).map(|i| 0.3_f32 + 0.11_f32 * i as f32).collect();
        let mut dx_cpu = vec![0.0_f32; n];
        crelu_grad_cpu(&x, &dy, &mut dx_cpu, n);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        cuda_launch! {
            kernel: crelu_grad, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice(dy_dev), slice_mut(dx_dev), n as u32]
        }?;
        stream.synchronize()?;
        assert_close("crelu_grad", &dx_dev.to_host_vec(&stream)?, &dx_cpu, 0.0);
        Ok(())
    }

    // -- abs_pow2_scale -----------------------------------------------------

    #[test]
    fn abs_pow2_scale_fwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 256_usize;
        let x = deterministic_floats(n, 3.0);
        let scale = L1_SQR_SCALE;
        let mut y_cpu = vec![0.0_f32; n];
        abs_pow2_scale_fwd_cpu(&x, &mut y_cpu, scale, n);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        cuda_launch! {
            kernel: abs_pow2_scale_fwd, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice_mut(y_dev), scale, n as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "abs_pow2_scale_fwd",
            &y_dev.to_host_vec(&stream)?,
            &y_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn abs_pow2_scale_grad_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 256_usize;
        let x = deterministic_floats(n, 4.0);
        let dy: Vec<f32> = (0..n).map(|i| -0.7_f32 + 0.05_f32 * i as f32).collect();
        let scale = L1_SQR_SCALE;
        let mut dx_cpu = vec![0.0_f32; n];
        abs_pow2_scale_grad_cpu(&x, &dy, &mut dx_cpu, scale, n);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        cuda_launch! {
            kernel: abs_pow2_scale_grad, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice(dy_dev), slice_mut(dx_dev), scale, n as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "abs_pow2_scale_grad",
            &dx_dev.to_host_vec(&stream)?,
            &dx_cpu,
            TOL,
        );
        Ok(())
    }

    // -- elementwise_add ----------------------------------------------------

    #[test]
    fn elementwise_add_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 300_usize;
        let mut a = deterministic_floats(n, 5.0);
        let mut b: Vec<f32> = (0..n).map(|i| 0.13_f32 * i as f32 - 2.0).collect();
        a.push(f32::NAN);
        b.push(1.0);
        let n = a.len();
        let mut c_cpu = vec![0.0_f32; n];
        elementwise_add_cpu(&a, &b, &mut c_cpu, n);

        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        cuda_launch! {
            kernel: elementwise_add, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), n as u32]
        }?;
        stream.synchronize()?;
        assert_close("elementwise_add", &c_dev.to_host_vec(&stream)?, &c_cpu, 0.0);
        Ok(())
    }

    // -- slice_extract_2d / slice_scatter_2d (v102 l1_main / l1_skip shapes) -

    #[test]
    fn slice_extract_2d_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 4_usize;
        let src: Vec<f32> = (0..batch * L1_OUT).map(|i| i as f32 * 0.5 - 3.0).collect();
        let src_dev = DeviceBuffer::from_host(&stream, &src)?;

        // l1_main: offset 0, out_dim 15
        let mut main_cpu = vec![0.0_f32; batch * L1_EFFECTIVE];
        slice_extract_2d_cpu(&src, &mut main_cpu, batch, L1_OUT, 0, L1_EFFECTIVE);
        let mut main_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * L1_EFFECTIVE)?;
        cuda_launch! {
            kernel: slice_extract_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_EFFECTIVE),
            args: [slice(src_dev), slice_mut(main_dev),
                   batch as u32, L1_OUT as u32, 0_u32, L1_EFFECTIVE as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "slice_extract l1_main",
            &main_dev.to_host_vec(&stream)?,
            &main_cpu,
            0.0,
        );

        // l1_skip: offset 15, out_dim 1
        let mut skip_cpu = vec![0.0_f32; batch * L1_SKIP];
        slice_extract_2d_cpu(&src, &mut skip_cpu, batch, L1_OUT, L1_EFFECTIVE, L1_SKIP);
        let mut skip_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * L1_SKIP)?;
        cuda_launch! {
            kernel: slice_extract_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_SKIP),
            args: [slice(src_dev), slice_mut(skip_dev),
                   batch as u32, L1_OUT as u32, L1_EFFECTIVE as u32, L1_SKIP as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "slice_extract l1_skip",
            &skip_dev.to_host_vec(&stream)?,
            &skip_cpu,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn slice_scatter_2d_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 4_usize;
        let dl1_main: Vec<f32> = (0..batch * L1_EFFECTIVE).map(|i| i as f32 + 0.25).collect();
        let dl1_skip: Vec<f32> = (0..batch * L1_SKIP).map(|i| -(i as f32) - 1.0).collect();

        // host 契約: dst を 0 初期化してから 2 回 scatter (offset 0 と 15)
        let mut dl1_total_cpu = vec![0.0_f32; batch * L1_OUT];
        slice_scatter_2d_cpu(
            &dl1_main,
            &mut dl1_total_cpu,
            batch,
            L1_EFFECTIVE,
            L1_OUT,
            0,
        );
        slice_scatter_2d_cpu(
            &dl1_skip,
            &mut dl1_total_cpu,
            batch,
            L1_SKIP,
            L1_OUT,
            L1_EFFECTIVE,
        );

        let main_dev = DeviceBuffer::from_host(&stream, &dl1_main)?;
        let skip_dev = DeviceBuffer::from_host(&stream, &dl1_skip)?;
        let mut total_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * L1_OUT)?;
        cuda_launch! {
            kernel: slice_scatter_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_EFFECTIVE),
            args: [slice(main_dev), slice_mut(total_dev),
                   batch as u32, L1_EFFECTIVE as u32, L1_OUT as u32, 0_u32]
        }?;
        cuda_launch! {
            kernel: slice_scatter_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_SKIP),
            args: [slice(skip_dev), slice_mut(total_dev),
                   batch as u32, L1_SKIP as u32, L1_OUT as u32, L1_EFFECTIVE as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "slice_scatter",
            &total_dev.to_host_vec(&stream)?,
            &dl1_total_cpu,
            0.0,
        );
        Ok(())
    }

    // -- concat_l1sqr_main fwd / grad (v102 dim 15 + 15 → 30) ----------------

    #[test]
    fn concat_l1sqr_main_fwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let a: Vec<f32> = (0..batch * L1_EFFECTIVE).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..batch * L1_EFFECTIVE)
            .map(|i| -(i as f32) - 0.5)
            .collect();
        let mut out_cpu = vec![0.0_f32; batch * L2_IN];
        concat_l1sqr_main_fwd_cpu(&a, &b, &mut out_cpu, batch, L1_EFFECTIVE, L1_EFFECTIVE);

        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * L2_IN)?;
        cuda_launch! {
            kernel: concat_l1sqr_main_fwd, stream: stream, module: module,
            config: cfg_1d(batch * L2_IN),
            args: [slice(a_dev), slice(b_dev), slice_mut(out_dev),
                   batch as u32, L1_EFFECTIVE as u32, L1_EFFECTIVE as u32]
        }?;
        stream.synchronize()?;
        assert_close("concat_fwd", &out_dev.to_host_vec(&stream)?, &out_cpu, 0.0);
        Ok(())
    }

    #[test]
    fn concat_l1sqr_main_grad_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let dout: Vec<f32> = (0..batch * L2_IN).map(|i| i as f32 * 0.7 - 4.0).collect();
        let mut da_cpu = vec![0.0_f32; batch * L1_EFFECTIVE];
        let mut db_cpu = vec![0.0_f32; batch * L1_EFFECTIVE];
        concat_l1sqr_main_grad_cpu(&dout, &mut da_cpu, &mut db_cpu, batch, L1_EFFECTIVE);

        let dout_dev = DeviceBuffer::from_host(&stream, &dout)?;
        let mut da_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * L1_EFFECTIVE)?;
        let mut db_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * L1_EFFECTIVE)?;
        cuda_launch! {
            kernel: concat_l1sqr_main_grad, stream: stream, module: module,
            config: cfg_1d(batch * L1_EFFECTIVE),
            args: [slice(dout_dev), slice_mut(da_dev), slice_mut(db_dev),
                   batch as u32, L1_EFFECTIVE as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "concat_grad da",
            &da_dev.to_host_vec(&stream)?,
            &da_cpu,
            0.0,
        );
        assert_close(
            "concat_grad db",
            &db_dev.to_host_vec(&stream)?,
            &db_cpu,
            0.0,
        );
        Ok(())
    }

    // -- dense_mm (regular) fwd / bwd_input / bwd_weight / bias_grad ---------
    // v102 L1f shape: in_dim=FT_OUT(=1536) は重いので、ここは小さい shape で
    // layout 規約 (in-major weight、row-major x/y) を確認 (実 shape は equivalence で
    // 担保不要、layout が一致すれば良い)。1 つは L1f 実 shape の縮小版も入れる。

    #[test]
    fn dense_mm_fwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 4_usize;
        let in_dim = 30_usize;
        let out_dim = 16_usize;
        let x: Vec<f32> = (0..batch * in_dim).map(|i| i as f32 * 0.01 - 1.0).collect();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| i as f32 * 0.003 + 0.1)
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.5 - 2.0).collect();
        let mut y_cpu = vec![0.0_f32; batch * out_dim];
        dense_mm_fwd_cpu(&x, &w, &bias, &mut y_cpu, batch, in_dim, out_dim);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * out_dim)?;
        cuda_launch! {
            kernel: dense_mm_fwd, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice_mut(y_dev),
                   batch as u32, in_dim as u32, out_dim as u32]
        }?;
        stream.synchronize()?;
        assert_close("dense_mm_fwd", &y_dev.to_host_vec(&stream)?, &y_cpu, TOL);
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_input_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 4_usize;
        let in_dim = 30_usize;
        let out_dim = 16_usize;
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.02 - 0.5)
            .collect();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| i as f32 * 0.003 + 0.1)
            .collect();
        let mut dx_cpu = vec![0.0_f32; batch * in_dim];
        dense_mm_bwd_input_cpu(&dy, &w, &mut dx_cpu, batch, in_dim, out_dim);

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * in_dim)?;
        cuda_launch! {
            kernel: dense_mm_bwd_input, stream: stream, module: module, config: cfg_1d(batch * in_dim),
            args: [slice(dy_dev), slice(w_dev), slice_mut(dx_dev),
                   batch as u32, in_dim as u32, out_dim as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "dense_mm_bwd_input",
            &dx_dev.to_host_vec(&stream)?,
            &dx_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_weight_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 4_usize;
        let in_dim = 30_usize;
        let out_dim = 16_usize;
        let x: Vec<f32> = (0..batch * in_dim).map(|i| i as f32 * 0.01 - 1.0).collect();
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.02 - 0.5)
            .collect();
        let mut dw_cpu = vec![0.0_f32; in_dim * out_dim];
        dense_mm_bwd_weight_cpu(&x, &dy, &mut dw_cpu, batch, in_dim, out_dim);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let mut dw_dev = DeviceBuffer::<f32>::zeroed(&stream, in_dim * out_dim)?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight, stream: stream, module: module, config: cfg_1d(in_dim * out_dim),
            args: [slice(x_dev), slice(dy_dev), slice_mut(dw_dev),
                   batch as u32, in_dim as u32, out_dim as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "dense_mm_bwd_weight",
            &dw_dev.to_host_vec(&stream)?,
            &dw_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn bias_grad_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 5_usize;
        let out_dim = 16_usize;
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.07 - 1.2)
            .collect();
        // accumulate semantics: host が呼出前に 0 初期化 → CPU 側も 0 から
        let mut gb_cpu = vec![0.0_f32; out_dim];
        bias_grad_cpu(&dy, &mut gb_cpu, batch, out_dim);

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, out_dim)?;
        cuda_launch! {
            kernel: bias_grad, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(dy_dev), slice(gb_dev), batch as u32, out_dim as u32]
        }?;
        stream.synchronize()?;
        // atomic fetch_add で reduce されるため relative tol (grad_bias と同様)。
        assert_close_rel("bias_grad", &gb_dev.to_host_vec(&stream)?, &gb_cpu, TOL);
        Ok(())
    }

    // -- dense_mm_bucket fwd / bwd_input / bwd_weight / bias_grad (9 buckets, -1 padding) --

    /// batch を num_buckets(=9) より大きくして全 bucket を踏み、`-1` (out-of-range)
    /// と `>= num_buckets` の position も入れる。
    fn bucket_idx_with_padding(batch: usize, num_buckets: usize) -> Vec<i32> {
        (0..batch)
            .map(|i| match i % (num_buckets + 2) {
                k if k < num_buckets => k as i32,
                k if k == num_buckets => -1_i32,
                _ => (num_buckets + 3) as i32, // >= num_buckets
            })
            .collect()
    }

    #[test]
    fn dense_mm_fwd_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 13_usize; // > 9 + 2 → all buckets + both out-of-range kinds
        let in_dim = 30_usize;
        let out_dim = 32_usize;
        let nb = NUM_BUCKETS;
        let x: Vec<f32> = (0..batch * in_dim).map(|i| i as f32 * 0.01 - 1.0).collect();
        let w: Vec<f32> = (0..nb * out_dim * in_dim)
            .map(|i| i as f32 * 0.0007 + 0.05)
            .collect();
        let bias: Vec<f32> = (0..nb * out_dim).map(|i| i as f32 * 0.02 - 1.0).collect();
        let bucket_idx = bucket_idx_with_padding(batch, nb);
        let mut y_cpu = vec![0.0_f32; batch * out_dim];
        dense_mm_fwd_bucket_cpu(
            &x,
            &w,
            &bias,
            &bucket_idx,
            &mut y_cpu,
            batch,
            in_dim,
            out_dim,
            nb,
        );

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * out_dim)?;
        cuda_launch! {
            kernel: dense_mm_fwd_bucket, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice(bidx_dev), slice_mut(y_dev),
                   batch as u32, in_dim as u32, out_dim as u32, nb as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "dense_mm_fwd_bucket",
            &y_dev.to_host_vec(&stream)?,
            &y_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_input_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 13_usize;
        let in_dim = 30_usize;
        let out_dim = 32_usize;
        let nb = NUM_BUCKETS;
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.013 - 0.4)
            .collect();
        let w: Vec<f32> = (0..nb * out_dim * in_dim)
            .map(|i| i as f32 * 0.0007 + 0.05)
            .collect();
        let bucket_idx = bucket_idx_with_padding(batch, nb);
        let mut dx_cpu = vec![0.0_f32; batch * in_dim];
        dense_mm_bwd_input_bucket_cpu(
            &dy,
            &w,
            &bucket_idx,
            &mut dx_cpu,
            batch,
            in_dim,
            out_dim,
            nb,
        );

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * in_dim)?;
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket, stream: stream, module: module, config: cfg_1d(batch * in_dim),
            args: [slice(dy_dev), slice(w_dev), slice(bidx_dev), slice_mut(dx_dev),
                   batch as u32, in_dim as u32, out_dim as u32, nb as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "dense_mm_bwd_input_bucket",
            &dx_dev.to_host_vec(&stream)?,
            &dx_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_weight_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 13_usize;
        let in_dim = 30_usize;
        let out_dim = 32_usize;
        let nb = NUM_BUCKETS;
        let x: Vec<f32> = (0..batch * in_dim).map(|i| i as f32 * 0.01 - 1.0).collect();
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.013 - 0.4)
            .collect();
        let bucket_idx = bucket_idx_with_padding(batch, nb);
        let mut dw_cpu = vec![0.0_f32; nb * out_dim * in_dim];
        dense_mm_bwd_weight_bucket_cpu(
            &x,
            &dy,
            &bucket_idx,
            &mut dw_cpu,
            batch,
            in_dim,
            out_dim,
            nb,
        );

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
        let mut dw_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim * in_dim)?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket, stream: stream, module: module,
            config: cfg_1d(nb * out_dim * in_dim),
            args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice_mut(dw_dev),
                   batch as u32, in_dim as u32, out_dim as u32, nb as u32]
        }?;
        stream.synchronize()?;
        assert_close(
            "dense_mm_bwd_weight_bucket",
            &dw_dev.to_host_vec(&stream)?,
            &dw_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn bias_grad_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 13_usize;
        let out_dim = 32_usize;
        let nb = NUM_BUCKETS;
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.017 - 0.9)
            .collect();
        let bucket_idx = bucket_idx_with_padding(batch, nb);
        // accumulate semantics: 0 から
        let mut gb_cpu = vec![0.0_f32; nb * out_dim];
        bias_grad_bucket_cpu(&dy, &bucket_idx, &mut gb_cpu, batch, out_dim, nb);

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim)?;
        cuda_launch! {
            kernel: bias_grad_bucket, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(dy_dev), slice(bidx_dev), slice(gb_dev),
                   batch as u32, out_dim as u32, nb as u32]
        }?;
        stream.synchronize()?;
        // atomic fetch_add で reduce されるため relative tol (grad_bias と同様)。
        assert_close_rel(
            "bias_grad_bucket",
            &gb_dev.to_host_vec(&stream)?,
            &gb_cpu,
            TOL,
        );
        Ok(())
    }

    // -- ft_post_perspective fwd / grad (the trickiest: pairwise indexing + shared bias) --

    #[test]
    fn ft_post_perspective_fwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let ft_dim = FT_OUT; // 1536 (even, half = 768 = COMBINED_DIM/2)
        // ft_out + bias の和が CReLU 境界 (0, 1) を跨ぐように値を散らす。
        let stm: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
            .collect();
        let nstm: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.5_f32 + 3.0_f32 * ((i * 5) % 11) as f32 / 10.0)
            .collect();
        let bias: Vec<f32> = (0..ft_dim)
            .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
            .collect();
        let scale = FT_POST_SCALE;
        let mut combined_cpu = vec![0.0_f32; batch * ft_dim];
        ft_post_perspective_fwd_cpu(&stm, &nstm, &bias, &mut combined_cpu, batch, ft_dim, scale);

        let stm_dev = DeviceBuffer::from_host(&stream, &stm)?;
        let nstm_dev = DeviceBuffer::from_host(&stream, &nstm)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let mut combined_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
        cuda_launch! {
            kernel: ft_post_perspective_fwd, stream: stream, module: module, config: cfg_1d(batch * ft_dim),
            args: [slice(stm_dev), slice(nstm_dev), slice(bias_dev), slice_mut(combined_dev),
                   batch as u32, ft_dim as u32, scale]
        }?;
        stream.synchronize()?;
        assert_close(
            "ft_post_perspective_fwd",
            &combined_dev.to_host_vec(&stream)?,
            &combined_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn ft_post_perspective_grad_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let ft_dim = FT_OUT;
        let half = ft_dim / 2;
        let bias: Vec<f32> = (0..ft_dim)
            .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
            .collect();
        let scale = FT_POST_SCALE;
        // d_combined: batch × COMBINED_DIM(=ft_dim)。前半 stm pair grad、後半 nstm pair grad。
        let d_combined: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -2.0_f32 + 0.013_f32 * i as f32)
            .collect();
        let stm_ft: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
            .collect();
        let nstm_ft: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.5_f32 + 3.0_f32 * ((i * 5) % 11) as f32 / 10.0)
            .collect();

        // CPU reference: grad_bias は 2 call (stm offset 0, nstm offset half) で accumulate。
        let mut grad_bias_cpu = vec![0.0_f32; ft_dim];
        let mut dft_stm_cpu = vec![0.0_f32; batch * ft_dim];
        let mut dft_nstm_cpu = vec![0.0_f32; batch * ft_dim];
        ft_post_perspective_grad_cpu(
            &d_combined,
            &stm_ft,
            &bias,
            &mut dft_stm_cpu,
            &mut grad_bias_cpu,
            batch,
            ft_dim,
            0,
            ft_dim,
            scale,
        );
        ft_post_perspective_grad_cpu(
            &d_combined,
            &nstm_ft,
            &bias,
            &mut dft_nstm_cpu,
            &mut grad_bias_cpu,
            batch,
            ft_dim,
            half,
            ft_dim,
            scale,
        );

        // GPU: host loop と同じく grad_bias を 0 初期化 → stm call → nstm call (default stream serialized)。
        let dc_dev = DeviceBuffer::from_host(&stream, &d_combined)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let stm_ft_dev = DeviceBuffer::from_host(&stream, &stm_ft)?;
        let nstm_ft_dev = DeviceBuffer::from_host(&stream, &nstm_ft)?;
        let grad_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_dim)?;
        let mut dft_stm_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
        let mut dft_nstm_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
        cuda_launch! {
            kernel: ft_post_perspective_grad, stream: stream, module: module, config: cfg_1d(batch * ft_dim),
            args: [slice(dc_dev), slice(stm_ft_dev), slice(bias_dev), slice_mut(dft_stm_dev),
                   slice(grad_bias_dev), batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale]
        }?;
        cuda_launch! {
            kernel: ft_post_perspective_grad, stream: stream, module: module, config: cfg_1d(batch * ft_dim),
            args: [slice(dc_dev), slice(nstm_ft_dev), slice(bias_dev), slice_mut(dft_nstm_dev),
                   slice(grad_bias_dev), batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale]
        }?;
        stream.synchronize()?;
        assert_close(
            "ft_grad dft_stm",
            &dft_stm_dev.to_host_vec(&stream)?,
            &dft_stm_cpu,
            TOL,
        );
        assert_close(
            "ft_grad dft_nstm",
            &dft_nstm_dev.to_host_vec(&stream)?,
            &dft_nstm_cpu,
            TOL,
        );
        // grad_bias は batch*2 個の atomic fetch_add で 1 cell に reduce されるため
        // 和の大きさに比例した f32 round-off drift (相対 1e-6 級) が出る。relative tol。
        assert_close_rel(
            "ft_grad grad_bias",
            &grad_bias_dev.to_host_vec(&stream)?,
            &grad_bias_cpu,
            TOL,
        );
        Ok(())
    }
}
