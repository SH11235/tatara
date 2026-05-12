//! `bins/nnue_train` binary entry point — bullet-shogi v102 互換 NNUE trainer。
//!
//! Stage 3-7 (#63) で本 file は **v102 LayerStack arch** の `#[kernel]` 群 (26 個)
//! と host loop driver (GpuTrainer) を統合する。Stage 3-8 (#65) で CLI + trainer
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
//! ## kernel 一覧 (26 個、bin entry reachability のため全て本 file に inline)
//!
//! ### STATED (Stage 2 #46-#52 で landed、本 bin に inline copy)
//! 1. `screlu_grad` — v102 では未使用、compile-reach のため preserve
//! 2. `loss_wdl` — 損失 + dy_net_output 勾配
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
//! ## cuda-oxide 制限への対応 (Stage 1-5〜2-7 で確立)
//!
//! - `f32::clamp` / `f32::max` / `f32::min` lowering 失敗 → `if-else` ladder で展開
//! - `i32::clamp` も同様 (Debug::fmt panic 経路を含む)
//! - `f32::sqrt`, `f32::exp` は libdevice (`__nv_sqrtf`, `__nv_expf`) に lowering OK
//! - atomic add パターン: `unsafe { &*(slice.as_ptr().add(idx) as *const DeviceAtomicX) }
//!   .fetch_add(_, AtomicOrdering::Relaxed)`
//!
//! kernel_names list (`compile_ll_to_ptx_via_llc` に渡す、計 26 個):
//! `sparse_ft_forward,sparse_ft_backward,loss_wdl,screlu_grad,adamw_step,radam_step,
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
use nnue_train::trainer::{TrainerBackend, TrainingConfig};
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
// `kernel_names` のみ 26 個に拡張。
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
/// 同 pipeline、`kernel_names` のみ Stage 3-7 全 26 kernel を内 internalize。
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

    // Stage 3-7 全 26 kernel 名。`@<name>` として `.ll` に出ているものを漏れなく
    // internalize-public-api-list で残す (kernel-list hazard、Stage 2-2 で確立)。
    let kernel_names = "sparse_ft_forward,sparse_ft_backward,loss_wdl,screlu_grad,\
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

// loss_wdl params (v102 doc: scale=290, wdl=0.0)
const WDL_LAMBDA: f32 = 0.0;
const LOSS_SCALE: f32 = 1.0 / 290.0;

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

    // loss + step
    loss_acc: DeviceBuffer<f64>,
    step_count: u64,
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

impl GpuTrainer {
    /// CUDA context を作成し、kernel module を load、10 weight groups + Ranger state を確保。
    fn new(ctx: &std::sync::Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
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

        Ok(Self {
            stream: stream.clone(),
            module,
            // FT
            ft_w: DeviceBuffer::from_host(&stream, &ft_w_init)?,
            ft_w_m: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_v: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_slow: DeviceBuffer::from_host(&stream, &ft_w_init)?,
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
            l1_w_slow: DeviceBuffer::from_host(&stream, &l1_w_init)?,
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
            l1f_w_slow: DeviceBuffer::from_host(&stream, &l1f_w_init)?,
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
            l2_w_slow: DeviceBuffer::from_host(&stream, &l2_w_init)?,
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
            l3_w_slow: DeviceBuffer::from_host(&stream, &l3_w_init)?,
            l3_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_b: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_m: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_v: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            // loss + step
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            step_count: 0,
        })
    }

    /// `V102Weights` から weight buffer を device に upload (pretrained 注入)。
    ///
    /// Optimizer state reset:
    /// - `m`, `v`: 0 (fresh start、Ranger 1st/2nd moment)
    /// - `slow`: weight と同値 (Ranger Lookahead 初期 slow = weight、bullet 慣行)
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
        // Optimizer state (m, v, slow) は trained weight に合わせて reset:
        // - m, v: 0 (fresh start)
        // - slow: weight と同値で初期化 (Ranger Lookahead initial state)
        // - grad: 0
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.l1_w_slow = DeviceBuffer::from_host(&self.stream, &w.l1_w)?;
        self.l1f_w_slow = DeviceBuffer::from_host(&self.stream, &w.l1f_w)?;
        self.l2_w_slow = DeviceBuffer::from_host(&self.stream, &w.l2_w)?;
        self.l3_w_slow = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        // m / v / grad / b_slow は zero reset
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
        self.ft_w_grad = zeros_f32(ft_w_n)?;
        self.ft_b_m = zeros_f32(ft_b_n)?;
        self.ft_b_v = zeros_f32(ft_b_n)?;
        self.ft_b_slow = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.ft_b_grad = zeros_f32(ft_b_n)?;
        self.l1_w_m = zeros_f32(l1_w_n)?;
        self.l1_w_v = zeros_f32(l1_w_n)?;
        self.l1_w_grad = zeros_f32(l1_w_n)?;
        self.l1_b_m = zeros_f32(l1_b_n)?;
        self.l1_b_v = zeros_f32(l1_b_n)?;
        self.l1_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1_b_grad = zeros_f32(l1_b_n)?;
        self.l1f_w_m = zeros_f32(l1f_w_n)?;
        self.l1f_w_v = zeros_f32(l1f_w_n)?;
        self.l1f_w_grad = zeros_f32(l1f_w_n)?;
        self.l1f_b_m = zeros_f32(l1f_b_n)?;
        self.l1f_b_v = zeros_f32(l1f_b_n)?;
        self.l1f_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1f_b)?;
        self.l1f_b_grad = zeros_f32(l1f_b_n)?;
        self.l2_w_m = zeros_f32(l2_w_n)?;
        self.l2_w_v = zeros_f32(l2_w_n)?;
        self.l2_w_grad = zeros_f32(l2_w_n)?;
        self.l2_b_m = zeros_f32(l2_b_n)?;
        self.l2_b_v = zeros_f32(l2_b_n)?;
        self.l2_b_slow = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l2_b_grad = zeros_f32(l2_b_n)?;
        self.l3_w_m = zeros_f32(l3_w_n)?;
        self.l3_w_v = zeros_f32(l3_w_n)?;
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

    /// 1 batch 分の forward → loss_wdl → backward → Ranger step を実行。
    /// 戻り値: batch 全体の loss (f64、loss_acc から読み出し)。
    ///
    /// Forward path (15 step): bullet `shogi_layerstack.rs:2241-2289` の reference 実装を
    /// 本 file の `#[kernel]` 群で再現。Backward path (~16 step): forward 逆順、`*_grad`
    /// buffer は本 method 内で 0 init してから kernel が書き込む (per-bucket weight grad
    /// `dense_mm_bwd_weight_bucket` は 1 cell = 1 thread の overwrite、FT / L1f / bias の
    /// grad は atomic accumulate)。Optimizer: 10 weight groups × `radam_step`
    /// (+ 周期 `ranger_lookahead_lerp`)。
    fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss_scale: f32,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        let b_u32 = b as u32;

        // 入力 buffer を host → device
        let stm_idx_dev = DeviceBuffer::from_host(&self.stream, &batch.stm_indices)?;
        let nstm_idx_dev = DeviceBuffer::from_host(&self.stream, &batch.nstm_indices)?;
        let bucket_idx_dev = DeviceBuffer::from_host(&self.stream, &batch.bucket_idx)?;
        let score_dev = DeviceBuffer::from_host(&self.stream, &batch.score)?;
        let wdl_dev = DeviceBuffer::from_host(&self.stream, &batch.wdl)?;
        let norm_dev = DeviceBuffer::from_host(&self.stream, &batch.per_pos_norm)?;

        // loss_acc reset
        self.loss_acc = DeviceBuffer::<f64>::zeroed(&self.stream, 1)?;

        // -- Forward step 1-2: sparse_ft_forward × 2 (stm, nstm) --
        let mut ft_stm_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        let mut ft_nstm_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        cuda_launch! {
            kernel: sparse_ft_forward,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ft_w),
                slice(stm_idx_dev),
                slice_mut(ft_stm_out),
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
                slice_mut(ft_nstm_out),
                b_u32, FT_OUT as u32, FT_IN as u32, MAX_ACTIVE as u32
            ]
        }?;

        // -- Forward step 3: ft_post_perspective_fwd → combined (B × FT_OUT) --
        let mut combined = DeviceBuffer::<f32>::zeroed(&self.stream, b * COMBINED_DIM)?;
        cuda_launch! {
            kernel: ft_post_perspective_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * COMBINED_DIM),
            args: [
                slice(ft_stm_out),
                slice(ft_nstm_out),
                slice(self.ft_b),
                slice_mut(combined),
                b_u32, FT_OUT as u32, FT_POST_SCALE
            ]
        }?;

        // -- Forward step 4: dense_mm_fwd_bucket L1 → l1_bucket (B × L1_OUT) --
        let mut l1_bucket = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_OUT)?;
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(combined),
                slice(self.l1_w),
                slice(self.l1_b),
                slice(bucket_idx_dev),
                slice_mut(l1_bucket),
                b_u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 5: dense_mm_fwd L1f shared → l1f_out (B × L1_OUT) --
        let mut l1f_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_OUT)?;
        cuda_launch! {
            kernel: dense_mm_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(combined),
                slice(self.l1f_w),
                slice(self.l1f_b),
                slice_mut(l1f_out),
                b_u32, FT_OUT as u32, L1_OUT as u32
            ]
        }?;

        // -- Forward step 6: l1_total = l1_bucket + l1f_out (B × L1_OUT) --
        let mut l1_total = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_OUT)?;
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(l1_bucket),
                slice(l1f_out),
                slice_mut(l1_total),
                (b * L1_OUT) as u32
            ]
        }?;

        // -- Forward step 7: slice l1_total → l1_main (B × 15) + l1_skip (B × 1) --
        let mut l1_main = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_EFFECTIVE)?;
        let mut l1_skip = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_SKIP)?;
        cuda_launch! {
            kernel: slice_extract_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(l1_total),
                slice_mut(l1_main),
                b_u32, L1_OUT as u32, 0_u32, L1_EFFECTIVE as u32
            ]
        }?;
        cuda_launch! {
            kernel: slice_extract_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_SKIP),
            args: [
                slice(l1_total),
                slice_mut(l1_skip),
                b_u32, L1_OUT as u32, L1_EFFECTIVE as u32, L1_SKIP as u32
            ]
        }?;

        // -- Forward step 8: l1_sqr = l1_main^2 * scale (B × 15) --
        let mut l1_sqr = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_EFFECTIVE)?;
        cuda_launch! {
            kernel: abs_pow2_scale_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(l1_main),
                slice_mut(l1_sqr),
                L1_SQR_SCALE,
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Forward step 9: l2_pre = concat(l1_sqr, l1_main) (B × 30) --
        let mut l2_pre = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_IN)?;
        cuda_launch! {
            kernel: concat_l1sqr_main_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(l1_sqr),
                slice(l1_main),
                slice_mut(l2_pre),
                b_u32, L1_EFFECTIVE as u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Forward step 10: l2_input = CReLU(l2_pre) (B × 30) --
        let mut l2_input = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_IN)?;
        cuda_launch! {
            kernel: crelu_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(l2_pre),
                slice_mut(l2_input),
                (b * L2_IN) as u32
            ]
        }?;

        // -- Forward step 11: L2 per-bucket dense → l2_out (B × 32) --
        let mut l2_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_OUT)?;
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(l2_input),
                slice(self.l2_w),
                slice(self.l2_b),
                slice(bucket_idx_dev),
                slice_mut(l2_out),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 12: l2_acted = CReLU(l2_out) (B × 32) --
        let mut l2_acted = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_OUT)?;
        cuda_launch! {
            kernel: crelu_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(l2_out),
                slice_mut(l2_acted),
                (b * L2_OUT) as u32
            ]
        }?;

        // -- Forward step 13: L3 per-bucket dense → l3_out (B × 1) --
        let mut l3_out = DeviceBuffer::<f32>::zeroed(&self.stream, b)?;
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(l2_acted),
                slice(self.l3_w),
                slice(self.l3_b),
                slice(bucket_idx_dev),
                slice_mut(l3_out),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 14: net_output = l3_out + l1_skip (B × 1) --
        let mut net_output = DeviceBuffer::<f32>::zeroed(&self.stream, b)?;
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(l3_out),
                slice(l1_skip),
                slice_mut(net_output),
                b_u32
            ]
        }?;

        // -- Forward step 15: loss_wdl → dy_net_output + loss_acc --
        let mut dy_net_output = DeviceBuffer::<f32>::zeroed(&self.stream, b)?;
        cuda_launch! {
            kernel: loss_wdl,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(net_output),
                slice(score_dev),
                slice(wdl_dev),
                slice(norm_dev),
                slice_mut(dy_net_output),
                slice(self.loss_acc),
                wdl_lambda, loss_scale, b_u32
            ]
        }?;

        // ===== BACKWARD =====
        // 全 *_grad buffer を 0 で reset (atomic accumulate semantic に従う kernel が
        // 多い、また overwrite kernel も in-place 安全のため統一)
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
        self.ft_w_grad = DeviceBuffer::<f32>::zeroed(&self.stream, ft_w_n)?;
        self.ft_b_grad = DeviceBuffer::<f32>::zeroed(&self.stream, ft_b_n)?;
        self.l1_w_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l1_w_n)?;
        self.l1_b_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l1_b_n)?;
        self.l1f_w_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l1f_w_n)?;
        self.l1f_b_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l1f_b_n)?;
        self.l2_w_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l2_w_n)?;
        self.l2_b_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l2_b_n)?;
        self.l3_w_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l3_w_n)?;
        self.l3_b_grad = DeviceBuffer::<f32>::zeroed(&self.stream, l3_b_n)?;

        // -- Backward 14 reverse: dy_net_output が dl3_out と dl1_skip 両方の grad --
        // (elementwise_add 逆: dl3_out = dy, dl1_skip = dy、両者同じ buffer を直接渡せばよい)

        // -- Backward 13 reverse: L3 per-bucket dense grad --
        let mut dl2_acted = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_OUT)?;
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(dy_net_output),
                slice(self.l3_w),
                slice(bucket_idx_dev),
                slice_mut(dl2_acted),
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
                slice(l2_acted),
                slice(dy_net_output),
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
                slice(dy_net_output),
                slice(bucket_idx_dev),
                slice(self.l3_b_grad),
                b_u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Backward 12 reverse: crelu_grad on l2_out --
        let mut dl2_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_OUT)?;
        cuda_launch! {
            kernel: crelu_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(l2_out),
                slice(dl2_acted),
                slice_mut(dl2_out),
                (b * L2_OUT) as u32
            ]
        }?;

        // -- Backward 11 reverse: L2 per-bucket dense grad --
        let mut dl2_input = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_IN)?;
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(dl2_out),
                slice(self.l2_w),
                slice(bucket_idx_dev),
                slice_mut(dl2_input),
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
                slice(l2_input),
                slice(dl2_out),
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
                slice(dl2_out),
                slice(bucket_idx_dev),
                slice(self.l2_b_grad),
                b_u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Backward 10 reverse: crelu_grad on l2_pre --
        let mut dl2_pre = DeviceBuffer::<f32>::zeroed(&self.stream, b * L2_IN)?;
        cuda_launch! {
            kernel: crelu_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(l2_pre),
                slice(dl2_input),
                slice_mut(dl2_pre),
                (b * L2_IN) as u32
            ]
        }?;

        // -- Backward 9 reverse: split dl2_pre → dl1_sqr (15) + dl1_main_from_concat (15) --
        let mut dl1_sqr = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_EFFECTIVE)?;
        let mut dl1_main_from_concat = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_EFFECTIVE)?;
        cuda_launch! {
            kernel: concat_l1sqr_main_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(dl2_pre),
                slice_mut(dl1_sqr),
                slice_mut(dl1_main_from_concat),
                b_u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Backward 8 reverse: abs_pow2_scale_grad (l1_sqr 経由の grad) --
        let mut dl1_main_from_sqr = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_EFFECTIVE)?;
        cuda_launch! {
            kernel: abs_pow2_scale_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(l1_main),
                slice(dl1_sqr),
                slice_mut(dl1_main_from_sqr),
                L1_SQR_SCALE,
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Combine dl1_main = dl1_main_from_concat + dl1_main_from_sqr --
        let mut dl1_main = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_EFFECTIVE)?;
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(dl1_main_from_concat),
                slice(dl1_main_from_sqr),
                slice_mut(dl1_main),
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Backward 7 reverse: assemble dl1_total from dl1_main (offset 0) + dl1_skip=dy_net_output (offset 15) --
        let mut dl1_total = DeviceBuffer::<f32>::zeroed(&self.stream, b * L1_OUT)?;
        cuda_launch! {
            kernel: slice_scatter_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(dl1_main),
                slice_mut(dl1_total),
                b_u32, L1_EFFECTIVE as u32, L1_OUT as u32, 0_u32
            ]
        }?;
        cuda_launch! {
            kernel: slice_scatter_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_SKIP),
            args: [
                slice(dy_net_output),
                slice_mut(dl1_total),
                b_u32, L1_SKIP as u32, L1_OUT as u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Backward 6 reverse: dl1_total を l1_bucket と l1f_out 両方の grad に流す --
        // (elementwise_add 逆: dl1_bucket = dl1_total, dl1f = dl1_total)
        // 直接 dl1_total を両 dense_mm_bwd に渡す

        // -- Backward 5 reverse: L1f shared dense grad --
        let mut dcombined_from_l1f = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        cuda_launch! {
            kernel: dense_mm_bwd_input,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(dl1_total),
                slice(self.l1f_w),
                slice_mut(dcombined_from_l1f),
                b_u32, FT_OUT as u32, L1_OUT as u32
            ]
        }?;
        cuda_launch! {
            kernel: dense_mm_bwd_weight,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(FT_OUT * L1_OUT),
            args: [
                slice(combined),
                slice(dl1_total),
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
                slice(dl1_total),
                slice(self.l1f_b_grad),
                b_u32, L1_OUT as u32
            ]
        }?;

        // -- Backward 4 reverse: L1 per-bucket dense grad --
        let mut dcombined_from_l1 = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(dl1_total),
                slice(self.l1_w),
                slice(bucket_idx_dev),
                slice_mut(dcombined_from_l1),
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
                slice(combined),
                slice(dl1_total),
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
                slice(dl1_total),
                slice(bucket_idx_dev),
                slice(self.l1_b_grad),
                b_u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Combine dcombined = dcombined_from_l1 + dcombined_from_l1f --
        let mut dcombined = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(dcombined_from_l1),
                slice(dcombined_from_l1f),
                slice_mut(dcombined),
                (b * FT_OUT) as u32
            ]
        }?;

        // -- Backward 3 reverse: ft_post_perspective_grad × 2 (stm, nstm) --
        let mut dft_stm_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        let mut dft_nstm_out = DeviceBuffer::<f32>::zeroed(&self.stream, b * FT_OUT)?;
        // stm: d_combined_offset = 0
        cuda_launch! {
            kernel: ft_post_perspective_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(dcombined),
                slice(ft_stm_out),
                slice(self.ft_b),
                slice_mut(dft_stm_out),
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
                slice(dcombined),
                slice(ft_nstm_out),
                slice(self.ft_b),
                slice_mut(dft_nstm_out),
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
                slice(dft_stm_out),
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
                slice(dft_nstm_out),
                slice(nstm_idx_dev),
                slice(self.ft_w_grad),
                b_u32, FT_OUT as u32, FT_IN as u32, MAX_ACTIVE as u32
            ]
        }?;

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

        self.stream.synchronize()?;

        // Read loss (`loss_acc` は 1-cell f64 buffer。空なら異常 → error にして無防備 index を避ける)
        let loss_vec = self.loss_acc.to_host_vec(&self.stream)?;
        loss_vec
            .first()
            .copied()
            .ok_or_else(|| -> Box<dyn std::error::Error> { "loss_acc buffer is empty".into() })
    }
}

// step() 実装は別 impl block (file 分割回避のため同 file 内)。

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
        loss_scale: f32,
    ) -> std::io::Result<f64> {
        let data = BatchData::from_batch(batch, bucket_idx);
        self.step(&data, lr, wdl_lambda, loss_scale)
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

    /// sigmoid score scale (`loss_scale = 1 / scale`)。
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
    #[arg(long)]
    init_from: Option<PathBuf>,

    // --- 以下は v102 recipe との CLI 互換のために受けるが、本 stage では未配線 ---
    /// (受けるが未配線) win-rate-model loss。指定時 warning を出す (loss は sigmoid-MSE のまま)。
    #[arg(long)]
    win_rate_model: bool,
    /// (受けるが未配線) WRM in-scaling。
    #[arg(long, default_value_t = 340.0)]
    wrm_in_scaling: f32,
    /// (受けるが未配線) WRM nnue2score。
    #[arg(long, default_value_t = 600.0)]
    wrm_nnue2score: f32,
    /// optimizer 名 ("ranger" のみ実装)。
    #[arg(long, default_value = "ranger")]
    optimizer: String,
    /// weight decay (kernel は 0.0 固定、非 0 指定で warning)。
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// (受けるが未使用) dataloader thread 数。本実装は single-thread sequential read。
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// bucket mode ("progress8kpabs" のみ実装)。
    #[arg(long, default_value = "progress8kpabs")]
    bucket_mode: String,
    /// (受けるが未実装) epoch ごとに file shuffle する。本実装は逐次 read + EOF wrap。
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
    // NaN / 範囲外を kernel に流さない (TrainingConfig::validate は loss_scale しか見ない)。
    if !(cli.lr.is_finite() && cli.lr > 0.0) {
        return Err(format!("--lr must be finite and > 0 (got {})", cli.lr).into());
    }
    if !cli.lr_gamma.is_finite() || cli.lr_gamma <= 0.0 {
        return Err(format!("--lr-gamma must be finite and > 0 (got {})", cli.lr_gamma).into());
    }
    if !(cli.scale.is_finite() && cli.scale > 0.0) {
        return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
    }
    if !cli.wdl.is_finite() || !(0.0..=1.0).contains(&cli.wdl) {
        return Err(format!("--wdl must be finite and in [0.0, 1.0] (got {})", cli.wdl).into());
    }
    if cli.weight_decay != 0.0 {
        eprintln!(
            "[train] warning: --weight-decay {} ignored; the Ranger kernel uses weight-decay 0.0 (v102)",
            cli.weight_decay
        );
    }
    if cli.win_rate_model {
        eprintln!(
            "[train] warning: --win-rate-model is not yet wired into the loss kernel; \
             using sigmoid-MSE with --scale {} (--wrm-in-scaling {} / --wrm-nnue2score {} ignored)",
            cli.scale, cli.wrm_in_scaling, cli.wrm_nnue2score
        );
    }
    if cli.epoch_file_shuffle {
        eprintln!(
            "[train] warning: --epoch-file-shuffle is not implemented; reading {} sequentially and wrapping at EOF (--file-shuffle-seed {} ignored)",
            data.display(),
            cli.file_shuffle_seed
        );
    }
    let _ = cli.threads; // dataloader は single-thread sequential read

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
    let mut trainer = GpuTrainer::new(&ctx)?;
    if let Some(init) = &cli.init_from {
        println!(
            "[train] injecting pretrained weights from {}",
            init.display()
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
        let weights = V102Weights::load_quantised(&mut reader)?;
        trainer.load_v102_weights(&weights)?;
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
        start_superbatch: 1,
        end_superbatch: cli.superbatches,
        batches_per_superbatch: cli.batches_per_superbatch,
        batch_size: cli.batch_size,
        save_rate: cli.save_rate,
        loss_scale: 1.0 / cli.scale,
        score_drop_abs: cli.score_drop_abs,
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
    let mut trainer = GpuTrainer::new(&ctx)?;
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

        // forward + step 1 batch
        let batch = BatchData::smoke_dummy(4);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch, lr, WDL_LAMBDA, LOSS_SCALE)?;
        println!("[smoke] step 1 (post-v102-100 init): loss = {loss:.6e}");
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
    } else {
        println!("[smoke] (no v102_100_quantised.bin available; running random-init smoke only)");
        let batch = BatchData::smoke_dummy(4);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch, lr, WDL_LAMBDA, LOSS_SCALE)?;
        println!("[smoke] step 1: loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

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
