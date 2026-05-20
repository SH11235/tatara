#![feature(f16)]
//! `bins/nnue_train` binary entry point — LayerStack NNUE trainer。
//!
//! 本 file は **LayerStack arch** の `#[kernel]` 群と host loop driver
//! (`GpuTrainer`) を統合する。cuda-oxide の bin-entry reachability 制約により
//! 全 kernel を本 file に inline する必要がある (別 crate に置くと
//! `compile_ll_to_ptx_via_llc` の symbol resolution から外れる)。
//!
//! ## LayerStack アーキテクチャ (1536-16-32 + progress8kpabs 9 buckets)
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
//! ## kernel 一覧
//!
//! kernel の確定一覧は `compile_ll_to_ptx_via_llc` に渡す `kernel_names` 定数を
//! single source of truth とする (build 時の internalize-public-api list、ここから
//! 漏れた kernel は `opt` の globaldce で削除されるため常に最新)。各 kernel の役割は
//! 定義箇所の doc コメントを参照。アーキ上の繋がりは上記 LayerStack アーキテクチャ節を見る。
//!
//! ## cuda-oxide 制限への対応
//!
//! - `f32::clamp` / `f32::max` / `f32::min` lowering 失敗 → `if-else` ladder で展開
//! - `i32::clamp` も同様 (Debug::fmt panic 経路を含む)
//! - `f32::sqrt`, `f32::exp` は libdevice (`__nv_sqrtf`, `__nv_expf`) に lowering OK
//! - atomic add パターン: `unsafe { &*(slice.as_ptr().add(idx) as *const DeviceAtomicX) }
//!   .fetch_add(_, AtomicOrdering::Relaxed)`

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
#[allow(unused_imports)]
use cuda_core::IntoResult as _;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_launch;
#[allow(unused_imports)]
use gpu_runtime::{CudaContext, CudaEvent, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
#[allow(unused_imports)]
use nnue_format::LayerStackWeights;
use nnue_format::{ArchKind, SimpleActivation, SimpleId, SimpleWeights};
use nnue_train::dataloader::Batch;
use nnue_train::experiment::{DataInfo, ExperimentDoc, ExperimentLogger, Lineage, Params};
#[allow(unused_imports)]
use nnue_train::optimizer::radam_compute_step_size_denom;
use nnue_train::schedule::{ConstantWDL, StepLR};
use nnue_train::trainer::{LossKind, TrainerBackend, TrainingConfig, ValidationStepOutput};
use shogi_features::progress_kpabs::ShogiProgressKPAbs;
use shogi_features::{FeatureSet, FeatureSetSpec};

// ===========================================================================
// 共通 / 損失 / optimizer kernel (inline copy)
// ===========================================================================

/// SCReLU activation gradient (fused)。
///
/// LayerStack path では **未使用** (CReLU + pairwise_mul を使うため)。cuda-oxide の
/// bin-entry constraint に従い compile-reach のため preserve。
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

/// Sigmoid + WDL blend + scale loss kernel。
///
/// 1 thread = 1 position。`dl_dout` は 1 thread = 1 index で排他更新 (atomics 不要)、
/// `loss_acc` は f64 単一 cell の Σ err^2 で `DeviceAtomicF64::fetch_add`。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn loss_wdl(
    out: &[f32],
    score: &[f32],
    wdl: &[f32],
    per_pos_norm: f32, // scalar (= 1/n_pos)。元 `&[f32]` の broadcast を kernel arg 化
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
    let norm = per_pos_norm;

    if let Some(g) = dl_dout.get_mut(i) {
        *g = 2.0_f32 * err * p * (1.0_f32 - p) * scale * norm;
    }

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell 確保済。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);
}

/// Win-rate-model (WRM) loss kernel。
///
/// 教師 score (centipawn) と net 出力の双方を win-rate に変換し、その二乗誤差を loss と
/// する。`loss_wdl` (`p = sigmoid(out * scale)` で `out ≈ cp` で収束) と違い、prediction
/// / target 双方に WRM 変換を掛けるため net_output は `out ≈ cp / nnue2score` (O(1)) の
/// スケールで収束し、`crates/nnue-format` の量子化フォーマット (`QA=127 / QB=64 /
/// FV_SCALE=28`) が前提とする net 出力スケールと整合する。CPU reference は
/// `gpu_kernels::pointwise::loss_wrm::loss_wrm_cpu`。
///
/// - target: `pt = (score - target_offset)/target_scaling`、`pmt = (-score -
///   target_offset)/target_scaling`、`target_wrm = 0.5*(1 + sigmoid(pt) - sigmoid(pmt))`、
///   `target = lambda*wdl + (1-lambda)*target_wrm`。`target_offset` / `target_scaling` は
///   WRM target sigmoid の中心と入力スケールで、CLI `--wrm-target-offset` /
///   `--wrm-target-scaling` から渡る (既定 270 / 380、score 分布に応じて再調整可)。
/// - prediction: `scorenet = out * nnue2score`、`q = sigmoid((scorenet - 270)/in_scaling)`、
///   `qm = sigmoid((-scorenet - 270)/in_scaling)`、`qf = 0.5*(1 + q - qm)`。prediction 側の
///   offset 270 はハードコード (CLI 非公開、可変なのは target 側のみ)。`in_scaling`
///   (= `--wrm-in-scaling`、既定 340) は prediction 側のみ、`nnue2score`
///   (= `--wrm-nnue2score`、既定 600)。
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
    per_pos_norm: f32, // scalar
    mut dl_dout: DisjointSlice<f32>,
    loss_acc: &[f64],
    lambda: f32,
    nnue2score: f32,
    in_scaling: f32,
    target_offset: f32,
    target_scaling: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    // --- target (WRM applied to teacher score、offset/scaling は caller 指定) ---
    let s = score[i.get()];
    let sig_pt = 1.0_f32 / (1.0_f32 + (-((s - target_offset) / target_scaling)).exp());
    let sig_pmt = 1.0_f32 / (1.0_f32 + (-((-s - target_offset) / target_scaling)).exp());
    let target_wrm = 0.5_f32 * (1.0_f32 + sig_pt - sig_pmt);
    let target = lambda * wdl[i.get()] + (1.0_f32 - lambda) * target_wrm;

    // --- prediction (WRM applied to net output) ---
    let scorenet = out[i.get()] * nnue2score;
    let q = 1.0_f32 / (1.0_f32 + (-((scorenet - 270.0_f32) / in_scaling)).exp());
    let qm = 1.0_f32 / (1.0_f32 + (-((-scorenet - 270.0_f32) / in_scaling)).exp());
    let qf = 0.5_f32 * (1.0_f32 + q - qm);

    let err = qf - target;
    let norm = per_pos_norm;

    if let Some(g) = dl_dout.get_mut(i) {
        *g = err * (nnue2score / in_scaling) * (q * (1.0_f32 - q) + qm * (1.0_f32 - qm)) * norm;
    }

    // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell 確保済 (`loss_wdl` と同型)。
    let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
    loss_atom.fetch_add((err as f64) * (err as f64), AtomicOrdering::Relaxed);
}

/// Fused AdamW optimizer step。
///
/// LayerStack path では **未使用** (Ranger 使用)。cuda-oxide の bin-entry constraint に従い
/// compile-reach のため preserve。
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

/// Fused RAdam optimizer step。
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

/// `radam_step` の FP16 mirror 同時更新 variant (`--ft-fp16` の `ft_w` 専用)。
///
/// forward は `ft_w` の FP16 mirror (`ft_w_h`) を読む。mirror を別 `cast_f32_to_f16`
/// kernel で毎 step 作り直すと `ft_w` を丸ごと再 read する DRAM traffic が要るが、
/// optimizer が `ft_w` を更新するこの kernel なら FP32 master が既に register に
/// 載っているので、確定後の値をその場で `mirror[i]` へ half 精度で書けば再 read
/// 不要になる。`mirror` は `weights` と同要素数 (caller 保証)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn radam_step_fp16_mirror(
    mut weights: DisjointSlice<f32>,
    mut m: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut grad: DisjointSlice<f32>,
    mut mirror: DisjointSlice<f16>,
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
        let mirror_ptr = mirror.as_mut_ptr();
        unsafe {
            mirror_ptr.add(i.get()).write(p_clamped as f16);
        }
    }
}

/// `radam_step` の 1st/2nd moment (`m` / `v`) を `f16` で保持する variant
/// (`--fp16-opt-state` の `ft_w` 専用)。
///
/// Ranger の `m` / `v` を半精度で持つと、112.6M 要素の `ft_w` optimizer step の
/// `m` / `v` read+write DRAM traffic が半減する。`m` / `v` は batch 正規化された
/// 勾配由来で値域が極めて小さく (`|m|` 中央値 ~1e-9、`v` 中央値 ~1e-15) `f16` の
/// normal range (`>= 6.1e-5`) を大きく下回るため、格納時に `m_scale` / `v_scale`
/// (power-of-2、scale 自体は無誤差) を掛けて normal range へ持ち上げ、読み出し時に
/// 割り戻す。算術は全て `f32`。
///
/// scale 後でも `f16` 有限域 (`|x| <= 65504`) を超えうる外れ値は格納前に clamp する。
/// clamp された moment はその要素を高分散扱いにするだけだが、未 clamp の `+inf` は
/// 以降の step で `vi = beta2*inf + ... = inf` と伝播し、その weight を恒久的に
/// 更新不能にするため必ず潰す。`vi >= 0` なので `v` は上側のみ clamp する。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn radam_step_f16state(
    mut weights: DisjointSlice<f32>,
    mut m: DisjointSlice<f16>,
    mut v: DisjointSlice<f16>,
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
    m_scale: f32,
    v_scale: f32,
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
        // f16 格納値を真値へ割り戻す (scale は power-of-2 なので除算は無誤差)。
        let m_prev = (*m_ref as f32) / m_scale;
        let v_prev = (*v_ref as f32) / v_scale;
        let mi = beta1 * m_prev + (1.0_f32 - beta1) * g;
        let vi = beta2 * v_prev + (1.0_f32 - beta2) * g * g;
        // 格納: scale 後 f16 有限域に clamp してから半精度化。
        let ms = mi * m_scale;
        let ms_c = if ms > 65504.0_f32 {
            65504.0_f32
        } else if ms < -65504.0_f32 {
            -65504.0_f32
        } else {
            ms
        };
        *m_ref = ms_c as f16;
        let vs = vi * v_scale;
        let vs_c = if vs > 65504.0_f32 { 65504.0_f32 } else { vs };
        *v_ref = vs_c as f16;
        // val は本 step の真値 mi / vi で計算する (f16 丸めは次 step の read で 1 回だけ入る)。
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

/// [`radam_step_f16state`] に FP16 weight mirror 同時更新を足した variant
/// (`--fp16-opt-state` かつ `--ft-fp16` 時の `ft_w` 専用)。`m` / `v` が `f16`、
/// かつ forward 用 `ft_w` mirror (`mirror`) も更新する。mirror 同時更新の意図は
/// [`radam_step_fp16_mirror`] と同一。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn radam_step_f16state_mirror(
    mut weights: DisjointSlice<f32>,
    mut m: DisjointSlice<f16>,
    mut v: DisjointSlice<f16>,
    mut grad: DisjointSlice<f32>,
    mut mirror: DisjointSlice<f16>,
    lr: f32,
    step_size: f32,
    denom: i32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    min_w: f32,
    max_w: f32,
    m_scale: f32,
    v_scale: f32,
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
        let m_prev = (*m_ref as f32) / m_scale;
        let v_prev = (*v_ref as f32) / v_scale;
        let mi = beta1 * m_prev + (1.0_f32 - beta1) * g;
        let vi = beta2 * v_prev + (1.0_f32 - beta2) * g * g;
        let ms = mi * m_scale;
        let ms_c = if ms > 65504.0_f32 {
            65504.0_f32
        } else if ms < -65504.0_f32 {
            -65504.0_f32
        } else {
            ms
        };
        *m_ref = ms_c as f16;
        let vs = vi * v_scale;
        let vs_c = if vs > 65504.0_f32 { 65504.0_f32 } else { vs };
        *v_ref = vs_c as f16;
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
        // SAFETY: `mirror` は `weights` / `m` / `v` / `grad` と同要素数 `n` (caller が
        // `ft_w` の要素数 `ft_w_n` を渡す)。kernel 冒頭で `i < n` を確認済みなので
        // `mirror.add(i)` は in-bounds。各 thread は自分の `i` のみ書くため thread 間で
        // aliasing は無い。`mirror` は他 buffer と別 alloc (caller 保証)。
        let mirror_ptr = mirror.as_mut_ptr();
        unsafe {
            mirror_ptr.add(i.get()).write(p_clamped as f16);
        }
    }
}

/// Ranger Lookahead lerp。
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

/// `ranger_lookahead_lerp` の FP16 mirror 同時更新 variant (`--ft-fp16` の `ft_w` 専用)。
///
/// lerp step では `radam_step_fp16_mirror` の後に lerp が `ft_w` を再度書き換えるため、
/// forward が読む `ft_w_h` を lerp 後の最終値で同期し直す。`mirror` は `weights` と
/// 同要素数 (caller 保証)。
#[kernel]
pub fn ranger_lookahead_lerp_fp16_mirror(
    mut weights: DisjointSlice<f32>,
    mut slow: DisjointSlice<f32>,
    mut mirror: DisjointSlice<f16>,
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
        let mirror_ptr = mirror.as_mut_ptr();
        unsafe {
            mirror_ptr.add(i.get()).write(new_w as f16);
        }
    }
}

/// Sparse feature transform forward (HalfKA_hm 用)。
///
/// 1 thread = 4 連続 row (output cells)、column-major weight (`weight[idx * rows + ri]`)、
/// atomics 不要 (各 thread は別 4 output cell に書く)。`-1` padding と `idx >= cols`
/// の異常入力は silent skip。caller は `rows % 4 == 0` を保証する (`FT_OUT = 1536`
/// で arch 上 invariant)、grid は `cfg_1d(batch * rows / 4)`。
///
/// inner loop は 4 連続 scalar weight read + 4 scalar partial-sum 更新形 (LLVM/NVPTX
/// backend は `f32` pointer の 4-byte alignment 推論止まりで `ld.global.v4.f32` へ
/// 集約しない、`#[repr(C, align(16))]` struct cast 経由でも SROA が align を保持せず
/// scalar load + local-mem spill になる)。warp coalesce は 32 thread × 4 row = 128
/// 連続 row が同 idx の cache line をまたいで読まれる pattern で維持される。
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
    let rows_u = rows as usize;
    let rows_q = rows_u / 4;
    let total = (batch as usize) * rows_q;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / rows_q;
    let ri_q = tid.get() % rows_q;
    let ri_base = ri_q * 4;

    // raw pointer 版。unsafe 妥当性: indices.len() == batch * nnz (dataloader が `-1`
    // padding 含めて確保)、weight.len() == cols * rows (FT 重み、arch 固定、rows %
    // 4 == 0)、`if idx >= 0 && (idx as u32) < cols` のロジックチェックは値検査として保持。
    let indices_ptr = indices.as_ptr();
    let weight_ptr = weight.as_ptr();
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let base = bi * (nnz as usize);
    let mut ni: u32 = 0;
    while ni < nnz {
        let idx = unsafe { indices_ptr.add(base + (ni as usize)).read() };
        if idx >= 0 && (idx as u32) < cols {
            let off = (idx as usize) * rows_u + ri_base;
            let w0 = unsafe { weight_ptr.add(off).read() };
            let w1 = unsafe { weight_ptr.add(off + 1).read() };
            let w2 = unsafe { weight_ptr.add(off + 2).read() };
            let w3 = unsafe { weight_ptr.add(off + 3).read() };
            s0 += w0;
            s1 += w1;
            s2 += w2;
            s3 += w3;
        }
        ni += 1;
    }
    let out_ptr = out.as_mut_ptr();
    let out_base = bi * rows_u + ri_base;
    unsafe {
        out_ptr.add(out_base).write(s0);
        out_ptr.add(out_base + 1).write(s1);
        out_ptr.add(out_base + 2).write(s2);
        out_ptr.add(out_base + 3).write(s3);
    }
}

/// [`sparse_ft_forward`] の FP16 weight 版。`weight` を `f16` で読み、各値を `f32` に
/// 変換してから累算する。累算と出力 (`out`) は `f32` のまま。
///
/// `sparse_ft_forward` は DRAM 帯域律速 (RTX 3080 Ti 実測で peak DRAM BW の ~90%)
/// で、その traffic の大半は active feature 行の weight gather read。weight を半精度に
/// すると read byte 数が半減し、L2 にも 2 倍の行が載るため DRAM 律速が緩む。
/// caller は `weight` に `ft_w` の FP16 mirror を渡し、FP32 master とは別管理する
/// (optimizer は FP32 master を更新し、mirror は毎 step 変換し直す)。
///
/// `out` も `f16` にする版は [`sparse_ft_forward_fp16_out`] (`--ft-fp16-out`)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn sparse_ft_forward_fp16(
    weight: &[f16],
    indices: &[i32],
    mut out: DisjointSlice<f32>,
    batch: u32,
    rows: u32,
    cols: u32,
    nnz: u32,
) {
    let tid = thread::index_1d();
    let rows_u = rows as usize;
    let rows_q = rows_u / 4;
    let total = (batch as usize) * rows_q;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / rows_q;
    let ri_q = tid.get() % rows_q;
    let ri_base = ri_q * 4;

    // raw pointer 版。unsafe 妥当性は [`sparse_ft_forward`] と同一 (indices.len() ==
    // batch * nnz、weight.len() == cols * rows、out.len() == batch * rows、
    // rows % 4 == 0)。weight のみ要素型が `f16` で、4 連続 row の read は 8 byte
    // 境界に整列する (idx*rows は 4 の倍数 [rows = FT_OUT = 1536]、ri_base は 4 の倍数)。
    let indices_ptr = indices.as_ptr();
    let weight_ptr = weight.as_ptr();
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let base = bi * (nnz as usize);
    let mut ni: u32 = 0;
    while ni < nnz {
        let idx = unsafe { indices_ptr.add(base + (ni as usize)).read() };
        if idx >= 0 && (idx as u32) < cols {
            let off = (idx as usize) * rows_u + ri_base;
            let w0 = unsafe { weight_ptr.add(off).read() } as f32;
            let w1 = unsafe { weight_ptr.add(off + 1).read() } as f32;
            let w2 = unsafe { weight_ptr.add(off + 2).read() } as f32;
            let w3 = unsafe { weight_ptr.add(off + 3).read() } as f32;
            s0 += w0;
            s1 += w1;
            s2 += w2;
            s3 += w3;
        }
        ni += 1;
    }
    let out_ptr = out.as_mut_ptr();
    let out_base = bi * rows_u + ri_base;
    unsafe {
        out_ptr.add(out_base).write(s0);
        out_ptr.add(out_base + 1).write(s1);
        out_ptr.add(out_base + 2).write(s2);
        out_ptr.add(out_base + 3).write(s3);
    }
}

/// [`sparse_ft_forward_fp16`] の出力も `f16` にした版 (`--ft-fp16-out`)。`weight` を
/// `f16` で読み、累算は `f32`、累算結果を round-to-nearest で `f16` に変換して `out`
/// へ書く。
///
/// `out` (`ft_*_out`、b × FT_OUT) を `f16` にすると書き出し DRAM traffic が半減し、
/// 後続の [`ft_post_perspective_fwd_fp16`] / [`ft_post_perspective_grad_fused_fp16`]
/// の read も半精度になる。`ft_*_out` は CReLU 前の FT accumulator で値域は ~O(1〜数十)、
/// f16 の有限域に収まる (loss scaling 不要、underflow する dft とは異なる)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn sparse_ft_forward_fp16_out(
    weight: &[f16],
    indices: &[i32],
    mut out: DisjointSlice<f16>,
    batch: u32,
    rows: u32,
    cols: u32,
    nnz: u32,
) {
    let tid = thread::index_1d();
    let rows_u = rows as usize;
    let rows_q = rows_u / 4;
    let total = (batch as usize) * rows_q;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / rows_q;
    let ri_q = tid.get() % rows_q;
    let ri_base = ri_q * 4;

    // unsafe 妥当性は [`sparse_ft_forward_fp16`] と同一。`weight` / `out` とも `f16` で、
    // 4 連続 row の read / write は 8 byte 境界に整列する (idx*rows は 4 の倍数
    // [rows = FT_OUT = 1536]、ri_base は 4 の倍数)。
    let indices_ptr = indices.as_ptr();
    let weight_ptr = weight.as_ptr();
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let base = bi * (nnz as usize);
    let mut ni: u32 = 0;
    while ni < nnz {
        let idx = unsafe { indices_ptr.add(base + (ni as usize)).read() };
        if idx >= 0 && (idx as u32) < cols {
            let off = (idx as usize) * rows_u + ri_base;
            let w0 = unsafe { weight_ptr.add(off).read() } as f32;
            let w1 = unsafe { weight_ptr.add(off + 1).read() } as f32;
            let w2 = unsafe { weight_ptr.add(off + 2).read() } as f32;
            let w3 = unsafe { weight_ptr.add(off + 3).read() } as f32;
            s0 += w0;
            s1 += w1;
            s2 += w2;
            s3 += w3;
        }
        ni += 1;
    }
    let out_ptr = out.as_mut_ptr();
    let out_base = bi * rows_u + ri_base;
    unsafe {
        out_ptr.add(out_base).write(s0 as f16);
        out_ptr.add(out_base + 1).write(s1 as f16);
        out_ptr.add(out_base + 2).write(s2 as f16);
        out_ptr.add(out_base + 3).write(s3 as f16);
    }
}

/// `f32` buffer を `f16` buffer へ要素ごとに round-to-nearest 変換する。
/// FP32 master weight (`ft_w`) から forward 用 FP16 mirror を毎 step 生成するのに使う。
/// 1 thread = 1 要素、`dst` は thread ごとに disjoint な cell へ書く
/// ([`DisjointSlice`] で mutable な device 出力として受ける)。
#[kernel]
pub fn cast_f32_to_f16(src: &[f32], mut dst: DisjointSlice<f16>, n: u32) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    // caller が `src.len() == dst.len() == n` を保証 (`ft_w` と同要素数で確保)。
    let v = src[i.get()];
    let dst_ptr = dst.as_mut_ptr();
    unsafe {
        dst_ptr.add(i.get()).write(v as f16);
    }
}

/// Phase 1 of inverse-index sparse_ft_backward: per-feature 出現回数を histogram。
/// `counts[f]` に (b, slot) で `indices[b*nnz+slot] == f` の数を atomic accumulate。
/// host が呼出前に `counts` を 0 reset。
#[kernel]
pub fn build_feature_counts(indices: &[i32], counts: &[u32], batch: u32, nnz: u32, cols: u32) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (nnz as usize);
    if tid.get() >= total {
        return;
    }
    let idx = indices[tid.get()];
    if idx >= 0 && (idx as u32) < cols {
        let cell = unsafe { &*(counts.as_ptr().add(idx as usize) as *const DeviceAtomicU32) };
        cell.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

/// Phase 2 of inverse-index: exclusive prefix sum over `counts[0..n]` → `offsets[0..=n]`。
/// 73K elements、1 block × 1024 threads で **並列** Hillis-Steele scan:
/// 1. 各 thread が n/1024 個の chunk を直列和算 → shared PARTIALS[tid] (per-thread total)
/// 2. block 内で PARTIALS の exclusive scan (sync_threads × log2(1024) = 10 round)
/// 3. 各 thread が chunk_offset を起点に再走査して `offsets[j]` を書き出す
/// 4. tid=1023 が `offsets[n]` (= total) を書く
///
/// host: block_dim=(1024, 1, 1), grid_dim=(1, 1, 1)、shared_mem_bytes=0 (static)。
#[kernel]
pub fn exclusive_prefix_sum_small(counts: &[u32], offsets: &[u32], n: u32) {
    static mut PARTIALS: SharedArray<u32, 1024> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let n_u = n as usize;

    let chunk = n_u.div_ceil(block_dim_u);
    let start = tid * chunk;
    let end_candidate = start + chunk;
    let end = if end_candidate < n_u {
        end_candidate
    } else {
        n_u
    };

    // Phase 1: per-thread sum
    let mut local_sum: u32 = 0;
    let mut i = start;
    while i < end {
        local_sum += counts[i];
        i += 1;
    }
    unsafe {
        PARTIALS[tid] = local_sum;
    }
    thread::sync_threads();

    // Phase 2: Hillis-Steele inclusive scan
    let mut offset_step: usize = 1;
    while offset_step < block_dim_u {
        let val: u32 = if tid >= offset_step {
            unsafe { PARTIALS[tid - offset_step] }
        } else {
            0
        };
        thread::sync_threads();
        unsafe {
            PARTIALS[tid] += val;
        }
        thread::sync_threads();
        offset_step <<= 1;
    }

    // PARTIALS[tid] is now INCLUSIVE scan. exclusive offset of own chunk:
    let chunk_offset: u32 = if tid == 0 {
        0
    } else {
        unsafe { PARTIALS[tid - 1] }
    };
    thread::sync_threads();

    // Phase 3: per-thread output exclusive scan of chunk
    let out_ptr = offsets.as_ptr() as *mut u32;
    let mut acc = chunk_offset;
    let mut j = start;
    while j < end {
        unsafe {
            out_ptr.add(j).write(acc);
        }
        acc += counts[j];
        j += 1;
    }

    // 最終 thread (= 担当 chunk が n-1 を含む thread) が offsets[n] = total を書く。
    // 簡素化: tid=block_dim-1 が常に最後の chunk を持つ (chunk size ceil で配分なので)。
    if tid == block_dim_u - 1 {
        unsafe {
            out_ptr.add(n_u).write(acc);
        }
    }
}

/// Phase 3 of inverse-index: 各 (b, slot) を inverse 順 (feature 別) に配置。
/// `write_counters[f]` を atomic increment、`positions[offsets[f] + write_counters[f]] = bi`。
/// host が呼出前に `write_counters` を 0 reset。
#[kernel]
pub fn scatter_positions(
    indices: &[i32],
    offsets: &[u32],
    write_counters: &[u32],
    positions: &[u32],
    batch: u32,
    nnz: u32,
    cols: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (nnz as usize);
    if tid.get() >= total {
        return;
    }
    let bi = (tid.get() / (nnz as usize)) as u32;
    let idx = indices[tid.get()];
    if idx >= 0 && (idx as u32) < cols {
        let cell =
            unsafe { &*(write_counters.as_ptr().add(idx as usize) as *const DeviceAtomicU32) };
        let pos = cell.fetch_add(1, AtomicOrdering::Relaxed);
        let abs_pos = offsets[idx as usize] + pos;
        unsafe {
            let p = positions.as_ptr().add(abs_pos as usize) as *mut u32;
            p.write(bi);
        }
    }
}

/// Phase 4 of inverse-index: 各 feature について grad_out の対応 row を sum し、
/// `grad_w[feature][ri]` に書き出し (overwrite 版)。
///
/// block 構成: blockIdx_x = feature_id (`cols`)、blockIdx_y = ri tile (`ft_out / blockDim`)。
/// block_dim threads (各 1 ri cell、cell 境界は block 内で disjoint なため atomic 不要)。
/// 呼出 host は呼出前に grad_w を 0 reset (`memset_zero`)、書かなかった cell は 0 のまま。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gather_and_sum_per_feature_overwrite(
    grad_out: &[f32],
    positions: &[u32],
    offsets: &[u32],
    grad_w: &[f32],
    n_features: u32,
    ft_out: u32,
) {
    let feature = thread::blockIdx_x() as usize;
    let ri_block = thread::blockIdx_y() as usize;
    let tid_local = thread::threadIdx_x() as usize;
    let block_dim = thread::blockDim_x() as usize;
    let ri = ri_block * block_dim + tid_local;
    let ft_out_u = ft_out as usize;
    if ri >= ft_out_u || feature >= (n_features as usize) {
        return;
    }

    let off_start = offsets[feature] as usize;
    let off_end = offsets[feature + 1] as usize;

    // raw pointer 版 (PTX で `setp.ge.u64; @%p bra` の bounds check 3 箇所を除去)。
    // unsafe 妥当性: caller (`step_impl`) が `feature_positions.len() == batch * max_active` を保証、
    // `feat_offsets[feature]..feat_offsets[feature+1]` は phase B が正しく構築。
    // grad_out / grad_w の範囲は arch (ft_in × ft_out) で固定、launch config 上 ri < ft_out_u。
    let grad_out_ptr = grad_out.as_ptr();
    let positions_ptr = positions.as_ptr();
    // 4-way unroll: 1 thread あたり 4 outstanding load + 4 accumulator で fadd dep chain
    // を分割。1-load-1-fadd 版は per-thread に in-flight load 1 個しかなく、warp scheduler は
    // memory load 待ちの Long Scoreboard stall で大半 idle になる (occupancy は full でも eligible
    // warps が極小)。partial sum 加算順が変わるため f32 fadd 非結合則で sum bit-pattern は
    // 同値ではなくなる (`gpu_cpu_equivalence_tests` の release tolerance 範囲)。
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut i = off_start;
    let unroll_end = if off_end >= off_start + 3 {
        off_end - 3
    } else {
        off_start
    };
    while i < unroll_end {
        let bi0 = unsafe { positions_ptr.add(i).read() } as usize;
        let bi1 = unsafe { positions_ptr.add(i + 1).read() } as usize;
        let bi2 = unsafe { positions_ptr.add(i + 2).read() } as usize;
        let bi3 = unsafe { positions_ptr.add(i + 3).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi0 * ft_out_u + ri).read() };
        sum1 += unsafe { grad_out_ptr.add(bi1 * ft_out_u + ri).read() };
        sum2 += unsafe { grad_out_ptr.add(bi2 * ft_out_u + ri).read() };
        sum3 += unsafe { grad_out_ptr.add(bi3 * ft_out_u + ri).read() };
        i += 4;
    }
    while i < off_end {
        let bi = unsafe { positions_ptr.add(i).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi * ft_out_u + ri).read() };
        i += 1;
    }
    let sum = (sum0 + sum1) + (sum2 + sum3);

    // 範囲外 (n_f=0、つまり off_start == off_end) でも sum=0 を書く: stm/nstm 共通の host が
    // 呼出前 0-reset を委ねる代わりに本 kernel が常に書き切るほうが simpler。
    let out_ptr = grad_w.as_ptr() as *mut f32;
    unsafe {
        out_ptr.add(feature * ft_out_u + ri).write(sum);
    }
}

/// Phase 4 (add 版): nstm 第 2 回呼び出し用。stm の overwrite 結果に atomic 加算。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gather_and_sum_per_feature_add(
    grad_out: &[f32],
    positions: &[u32],
    offsets: &[u32],
    grad_w: &[f32],
    n_features: u32,
    ft_out: u32,
) {
    let feature = thread::blockIdx_x() as usize;
    let ri_block = thread::blockIdx_y() as usize;
    let tid_local = thread::threadIdx_x() as usize;
    let block_dim = thread::blockDim_x() as usize;
    let ri = ri_block * block_dim + tid_local;
    let ft_out_u = ft_out as usize;
    if ri >= ft_out_u || feature >= (n_features as usize) {
        return;
    }

    let off_start = offsets[feature] as usize;
    let off_end = offsets[feature + 1] as usize;

    // raw pointer 版 (overwrite と同じ理由、bounds check 3 箇所除去)。
    let grad_out_ptr = grad_out.as_ptr();
    let positions_ptr = positions.as_ptr();
    // 4-way unroll: overwrite kernel と同方針 (Long Scoreboard stall 分散)。
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut i = off_start;
    let unroll_end = if off_end >= off_start + 3 {
        off_end - 3
    } else {
        off_start
    };
    while i < unroll_end {
        let bi0 = unsafe { positions_ptr.add(i).read() } as usize;
        let bi1 = unsafe { positions_ptr.add(i + 1).read() } as usize;
        let bi2 = unsafe { positions_ptr.add(i + 2).read() } as usize;
        let bi3 = unsafe { positions_ptr.add(i + 3).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi0 * ft_out_u + ri).read() };
        sum1 += unsafe { grad_out_ptr.add(bi1 * ft_out_u + ri).read() };
        sum2 += unsafe { grad_out_ptr.add(bi2 * ft_out_u + ri).read() };
        sum3 += unsafe { grad_out_ptr.add(bi3 * ft_out_u + ri).read() };
        i += 4;
    }
    while i < off_end {
        let bi = unsafe { positions_ptr.add(i).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi * ft_out_u + ri).read() };
        i += 1;
    }
    let sum = (sum0 + sum1) + (sum2 + sum3);

    // atomicAdd で stm の結果に加算。
    if sum != 0.0_f32 {
        let cell =
            unsafe { &*(grad_w.as_ptr().add(feature * ft_out_u + ri) as *const DeviceAtomicF32) };
        cell.fetch_add(sum, AtomicOrdering::Relaxed);
    }
}

/// [`gather_and_sum_per_feature_overwrite`] の FP16 入力版。`grad_out` (dft) を `f16`
/// で読み、各値を `f32` に変換してから累算する。累算と `grad_w` への書き出しは `f32`。
///
/// `grad_out` は b × FT_OUT で、本 kernel は 1 feature の出現位置すべてに対して全 ri
/// 行を gather-read するため step 中で最も read DRAM traffic が大きい。`ft_post_
/// perspective_grad_fused_fp16` が dft を `f16` で書くようになったため、その read 側も
/// 半精度化して帯域を半減させる。
///
/// `grad_out` は `ft_post_perspective_grad_fused_fp16` 側で loss scaling 済 (値が
/// `dft_scale` 倍されている)。本 kernel は scale 済の値を累算し、`grad_w` へ書く直前に
/// `dft_inv_scale` (= 1 / dft_scale) を掛けて元の scale に戻す。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gather_and_sum_per_feature_overwrite_fp16(
    grad_out: &[f16],
    positions: &[u32],
    offsets: &[u32],
    grad_w: &[f32],
    n_features: u32,
    ft_out: u32,
    dft_inv_scale: f32, // = 1 / dft_scale、loss scaling を打ち消す
) {
    let feature = thread::blockIdx_x() as usize;
    let ri_block = thread::blockIdx_y() as usize;
    let tid_local = thread::threadIdx_x() as usize;
    let block_dim = thread::blockDim_x() as usize;
    let ri = ri_block * block_dim + tid_local;
    let ft_out_u = ft_out as usize;
    if ri >= ft_out_u || feature >= (n_features as usize) {
        return;
    }

    let off_start = offsets[feature] as usize;
    let off_end = offsets[feature + 1] as usize;

    // unsafe 妥当性は [`gather_and_sum_per_feature_overwrite`] と同一。`grad_out` のみ
    // 要素型が `f16`、read 時に `f32` へ変換する。
    let grad_out_ptr = grad_out.as_ptr();
    let positions_ptr = positions.as_ptr();
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut i = off_start;
    let unroll_end = if off_end >= off_start + 3 {
        off_end - 3
    } else {
        off_start
    };
    while i < unroll_end {
        let bi0 = unsafe { positions_ptr.add(i).read() } as usize;
        let bi1 = unsafe { positions_ptr.add(i + 1).read() } as usize;
        let bi2 = unsafe { positions_ptr.add(i + 2).read() } as usize;
        let bi3 = unsafe { positions_ptr.add(i + 3).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi0 * ft_out_u + ri).read() } as f32;
        sum1 += unsafe { grad_out_ptr.add(bi1 * ft_out_u + ri).read() } as f32;
        sum2 += unsafe { grad_out_ptr.add(bi2 * ft_out_u + ri).read() } as f32;
        sum3 += unsafe { grad_out_ptr.add(bi3 * ft_out_u + ri).read() } as f32;
        i += 4;
    }
    while i < off_end {
        let bi = unsafe { positions_ptr.add(i).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi * ft_out_u + ri).read() } as f32;
        i += 1;
    }
    let sum = (sum0 + sum1) + (sum2 + sum3);

    let out_ptr = grad_w.as_ptr() as *mut f32;
    unsafe {
        out_ptr
            .add(feature * ft_out_u + ri)
            .write(sum * dft_inv_scale);
    }
}

/// [`gather_and_sum_per_feature_add`] の FP16 入力版。`grad_out` (dft) を `f16` で読み、
/// `dft_inv_scale` で loss scaling を打ち消す以外は `gather_and_sum_per_feature_add` と
/// 同一 (nstm 第 2 回呼び出しで stm の overwrite 結果へ atomic 加算)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gather_and_sum_per_feature_add_fp16(
    grad_out: &[f16],
    positions: &[u32],
    offsets: &[u32],
    grad_w: &[f32],
    n_features: u32,
    ft_out: u32,
    dft_inv_scale: f32, // = 1 / dft_scale、loss scaling を打ち消す
) {
    let feature = thread::blockIdx_x() as usize;
    let ri_block = thread::blockIdx_y() as usize;
    let tid_local = thread::threadIdx_x() as usize;
    let block_dim = thread::blockDim_x() as usize;
    let ri = ri_block * block_dim + tid_local;
    let ft_out_u = ft_out as usize;
    if ri >= ft_out_u || feature >= (n_features as usize) {
        return;
    }

    let off_start = offsets[feature] as usize;
    let off_end = offsets[feature + 1] as usize;

    // unsafe 妥当性は [`gather_and_sum_per_feature_overwrite`] / その `_fp16` 版と同一:
    // caller が `positions.len() == batch * max_active` を保証、`off_start..off_end` は
    // phase B が構築した有効範囲、`grad_out` (`f16`) / `grad_w` (`f32`) の範囲は arch
    // (ft_in × ft_out) 固定で launch config 上 `ri < ft_out_u`。`grad_out` のみ要素型が
    // `f16` で read 時に `f32` へ変換する。`grad_w` への書き込みは atomic add: 末尾の
    // `&*(grad_w.as_ptr().add(..) as *const DeviceAtomicF32)` cast は、`DeviceAtomicF32`
    // が `f32` (align 4) と同レイアウト (`#[repr(transparent)]` over `UnsafeCell<f32>`)
    // で `grad_w` の backing allocation が要求 alignment を満たすため有効。同 cell へ
    // non-atomic に書く path は本 kernel / host loop に無い。
    let grad_out_ptr = grad_out.as_ptr();
    let positions_ptr = positions.as_ptr();
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut i = off_start;
    let unroll_end = if off_end >= off_start + 3 {
        off_end - 3
    } else {
        off_start
    };
    while i < unroll_end {
        let bi0 = unsafe { positions_ptr.add(i).read() } as usize;
        let bi1 = unsafe { positions_ptr.add(i + 1).read() } as usize;
        let bi2 = unsafe { positions_ptr.add(i + 2).read() } as usize;
        let bi3 = unsafe { positions_ptr.add(i + 3).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi0 * ft_out_u + ri).read() } as f32;
        sum1 += unsafe { grad_out_ptr.add(bi1 * ft_out_u + ri).read() } as f32;
        sum2 += unsafe { grad_out_ptr.add(bi2 * ft_out_u + ri).read() } as f32;
        sum3 += unsafe { grad_out_ptr.add(bi3 * ft_out_u + ri).read() } as f32;
        i += 4;
    }
    while i < off_end {
        let bi = unsafe { positions_ptr.add(i).read() } as usize;
        sum0 += unsafe { grad_out_ptr.add(bi * ft_out_u + ri).read() } as f32;
        i += 1;
    }
    let sum = (sum0 + sum1) + (sum2 + sum3);

    if sum != 0.0_f32 {
        let cell =
            unsafe { &*(grad_w.as_ptr().add(feature * ft_out_u + ri) as *const DeviceAtomicF32) };
        cell.fetch_add(sum * dft_inv_scale, AtomicOrdering::Relaxed);
    }
}

/// Sparse feature transform backward (atomic scatter)。
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

/// Fused stm+nstm sparse_ft_backward。2 回呼び出しを 1 kernel に統合し、kernel launch
/// オーバーヘッドと per-thread setup を削減 (`bi` / `ri` / 計算は thread 共有)。
/// per-thread の atomic add ops 数は変わらない (38 stm + 38 nstm = 76)。
/// host が呼出前に `grad_weight` を 0 で初期化。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn sparse_ft_backward_dual(
    grad_out_stm: &[f32],
    grad_out_nstm: &[f32],
    indices_stm: &[i32],
    indices_nstm: &[i32],
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
    let rows_u = rows as usize;
    let nnz_u = nnz as usize;
    let cols_u = cols as usize;

    let g_stm = grad_out_stm[tid.get()];
    let g_nstm = grad_out_nstm[tid.get()];
    let base = bi * nnz_u;

    let mut ni: u32 = 0;
    while ni < nnz {
        let idx_s = indices_stm[base + (ni as usize)];
        if idx_s >= 0 && (idx_s as usize) < cols_u {
            let cell = unsafe {
                &*(grad_weight.as_ptr().add((idx_s as usize) * rows_u + ri)
                    as *const DeviceAtomicF32)
            };
            cell.fetch_add(g_stm, AtomicOrdering::Relaxed);
        }
        let idx_n = indices_nstm[base + (ni as usize)];
        if idx_n >= 0 && (idx_n as usize) < cols_u {
            let cell = unsafe {
                &*(grad_weight.as_ptr().add((idx_n as usize) * rows_u + ri)
                    as *const DeviceAtomicF32)
            };
            cell.fetch_add(g_nstm, AtomicOrdering::Relaxed);
        }
        ni += 1;
    }
}

// ===========================================================================
// LayerStack 専用 kernel
// ===========================================================================
//
// 設計方針:
// - atomics は host が呼出前に gradient buffer を 0 初期化する accumulate semantics
// - DisjointSlice<f32> は 1 thread = 1 cell の排他書き込み、&[f32] + raw atomic は
//   多 thread → 1 cell の atomic accumulate
// - cuda-oxide 制限: `f32::clamp` / `f32::max` / `f32::min` は if-else 展開

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

/// [`ft_post_perspective_fwd`] の FP16 入力版。`stm_ft_out` / `nstm_ft_out` を `f16`
/// で読み、`f32` に変換してから bias add 以降を計算する。math と `combined` 出力は
/// `f32` のまま (`combined` は後続 dense L1 path が `f32` で読む)。
///
/// `sparse_ft_forward_fp16` が `ft_*_out` を `f16` で書くようになったため、その read
/// 側も半精度化して DRAM traffic を合わせる。`f16` → `f32` 変換は値域を保つ無損失
/// 変換なので、`combined` は FP32 版と同じ値域・同じ丸めで計算される (入力 `ft_*_out`
/// 自体が `sparse_ft_forward_fp16` 時点で既に半精度量子化されている点のみ FP32 path と
/// 異なる)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn ft_post_perspective_fwd_fp16(
    stm_ft_out: &[f16],
    nstm_ft_out: &[f16],
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
        let xa = stm_ft_out[ft_base + ri] as f32 + bias[ri];
        let xb = stm_ft_out[ft_base + half + ri] as f32 + bias[half + ri];
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
        let xa = nstm_ft_out[ft_base + pair_idx] as f32 + bias[pair_idx];
        let xb = nstm_ft_out[ft_base + half + pair_idx] as f32 + bias[half + pair_idx];
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
/// 2 call で atomic accumulate される。
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
    // 1 thread = 1 (bi, pair_idx) → 2 出力 (ii=pair_idx と ii=pair_idx+half) を per-thread に
    // 担当させて dy / xa / xb / bias を 1 回読みで共有する。caller の launch config は
    // `cfg_1d(batch * ft_dim / 2)` で、`ft_dim` 偶数性 (= `2 * half`、arch 上 invariant) が前提。
    // grad_ft_out の cell 数と grad_bias への atomic 回数は thread 数半減 + per-thread 出力倍で
    // 不変。同一 (bi, ii) cell に書く thread は 1 つのみ (cross-thread disjoint)。
    let tid = thread::index_1d();
    let half = (ft_dim as usize) / 2;
    let total_pairs = (batch as usize) * half;
    if tid.get() >= total_pairs {
        return;
    }
    let bi = tid.get() / half;
    let pair_idx = tid.get() % half;

    // d_combined の対応 output cell (pair_idx 共通)
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

    // First side (ii = pair_idx): my_pre = xa, partner_post = yb
    let grad_a_post = dy * yb * scale;
    let grad_a = if xa > 0.0_f32 && xa < 1.0_f32 {
        grad_a_post
    } else {
        0.0_f32
    };
    // Second side (ii = pair_idx + half): my_pre = xb, partner_post = ya
    let grad_b_post = dy * ya * scale;
    let grad_b = if xb > 0.0_f32 && xb < 1.0_f32 {
        grad_b_post
    } else {
        0.0_f32
    };

    // 1 thread が 2 cell (ft_base + pair_idx) と (ft_base + half + pair_idx) を書く。
    // DisjointSlice の `get_mut(ThreadIndex)` は 1 thread = 1 cell 安全契約を要求するので、
    // 2 cell 書きは sparse_ft_forward と同じく raw pointer 経由。
    // SAFETY: grad_ft_out.len() == batch * ft_dim (caller 契約)、`ft_dim = 2 * half` の偶数性で
    // pair_idx ∈ [0, half) → ii ∈ {pair_idx, pair_idx + half} ⊂ [0, ft_dim) に限る。tid 範囲
    // チェック (`tid >= total_pairs` で `bi < batch`) と合わせて `ft_base + half + pair_idx <
    // batch * ft_dim` が成立。同一 (bi, ii) cell を書く thread は他に存在しない (pair_idx
    // 単射、cross-thread disjoint)。
    let out_ptr = grad_ft_out.as_mut_ptr();
    unsafe {
        out_ptr.add(ft_base + pair_idx).write(grad_a);
        out_ptr.add(ft_base + half + pair_idx).write(grad_b);
    }

    // grad_bias[ii] += grad_my_pre (atomic, 共有 bias)。
    // SAFETY: grad_bias.len() == ft_dim、pair_idx < half、half + pair_idx < ft_dim。
    let bias_cell_a = unsafe { &*(grad_bias.as_ptr().add(pair_idx) as *const DeviceAtomicF32) };
    bias_cell_a.fetch_add(grad_a, AtomicOrdering::Relaxed);
    let bias_cell_b =
        unsafe { &*(grad_bias.as_ptr().add(half + pair_idx) as *const DeviceAtomicF32) };
    bias_cell_b.fetch_add(grad_b, AtomicOrdering::Relaxed);
}

/// Fused 版 [`ft_post_perspective_grad`]: `dy = dcombined_a[idx] + dcombined_b[idx]`
/// を in-register sum で計算し、materialized な合算 buffer 経由を避ける。math は
/// `ft_post_perspective_grad` と同等で、`dy` の読み出し元のみ単一 buffer → 2 source
/// の elementwise sum に置換。
///
/// 1 step あたり stm / nstm の 2 launch のみで完結 (合算 buffer を介す場合の合算
/// kernel + grad 2 launch = 3 launch / 384MB DRAM roundtrip と比較して 1 launch +
/// ~768MB DRAM 削減)。
///
/// `d_combined_stride` は両 source の row-stride (= COMBINED_DIM = FT_OUT)、
/// `d_combined_offset` は perspective 別 offset (stm: 0、nstm: ft_dim/2)、両 source
/// は同 stride・同 layout を caller が保証 (両者とも `b × COMBINED_DIM` workspace)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn ft_post_perspective_grad_fused(
    d_combined_a: &[f32],
    d_combined_b: &[f32],
    ft_out: &[f32],
    bias: &[f32],
    mut grad_ft_out: DisjointSlice<f32>,
    grad_bias: &[f32],
    batch: u32,
    ft_dim: u32,
    d_combined_offset: u32,
    d_combined_stride: u32,
    scale: f32,
) {
    let tid = thread::index_1d();
    let half = (ft_dim as usize) / 2;
    let total_pairs = (batch as usize) * half;
    if tid.get() >= total_pairs {
        return;
    }
    let bi = tid.get() / half;
    let pair_idx = tid.get() % half;

    let dy_idx = bi * (d_combined_stride as usize) + (d_combined_offset as usize) + pair_idx;
    let dy = d_combined_a[dy_idx] + d_combined_b[dy_idx];

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

    let grad_a_post = dy * yb * scale;
    let grad_a = if xa > 0.0_f32 && xa < 1.0_f32 {
        grad_a_post
    } else {
        0.0_f32
    };
    let grad_b_post = dy * ya * scale;
    let grad_b = if xb > 0.0_f32 && xb < 1.0_f32 {
        grad_b_post
    } else {
        0.0_f32
    };

    let out_ptr = grad_ft_out.as_mut_ptr();
    unsafe {
        out_ptr.add(ft_base + pair_idx).write(grad_a);
        out_ptr.add(ft_base + half + pair_idx).write(grad_b);
    }

    let bias_cell_a = unsafe { &*(grad_bias.as_ptr().add(pair_idx) as *const DeviceAtomicF32) };
    bias_cell_a.fetch_add(grad_a, AtomicOrdering::Relaxed);
    let bias_cell_b =
        unsafe { &*(grad_bias.as_ptr().add(half + pair_idx) as *const DeviceAtomicF32) };
    bias_cell_b.fetch_add(grad_b, AtomicOrdering::Relaxed);
}

/// [`ft_post_perspective_grad_fused`] の FP16 版。forward activation `ft_out` を `f16`
/// で読み、`grad_ft_out` (dft) を `f16` で書く。`d_combined_a` / `_b` と `bias` /
/// `grad_bias` は `f32` のまま (それぞれ dense L1 backward 出力と共有 FT bias で、
/// 半精度化はこの kernel の scope 外)。
///
/// math は `ft_post_perspective_grad_fused` と同等。`grad_bias` への atomic accumulate
/// は `f32` の `grad_a` / `grad_b` をそのまま使い (FP32 path と同じ精度)、`grad_ft_out`
/// へ書く分のみ round-to-nearest で `f16` に変換する。`grad_ft_out` を半精度にすると
/// 後続の inverse-index gather (`gather_and_sum_per_feature_*_fp16`) の read DRAM
/// traffic が半減する (dft は b × FT_OUT で step 中で最も read 量が多い buffer)。
///
/// **loss scaling**: dft の値は batch 正規化 (loss が 1/batch) のため `1/batch` に比例し、
/// そのまま f16 化すると全要素が subnormal 下限 (2^-24 ≈ 6e-8) を下回って 0 に潰れる。
/// これを防ぐため `grad_ft_out` へ書く値だけ caller 計算の `dft_scale`
/// ([`FT_DFT_FP16_BASE_SCALE`] × batch) を掛けて f16 normal range に持ち上げる。gather
/// 側 (`gather_and_sum_per_feature_*_fp16`) が逆数を掛けて元の scale に戻す。`grad_bias`
/// は scale しない (f32 のため不要)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn ft_post_perspective_grad_fused_fp16(
    d_combined_a: &[f32],
    d_combined_b: &[f32],
    ft_out: &[f16],
    bias: &[f32],
    mut grad_ft_out: DisjointSlice<f16>,
    grad_bias: &[f32],
    batch: u32,
    ft_dim: u32,
    d_combined_offset: u32,
    d_combined_stride: u32,
    scale: f32,
    dft_scale: f32, // grad_ft_out (f16) loss scaling 係数 (= FT_DFT_FP16_BASE_SCALE × batch)
) {
    let tid = thread::index_1d();
    let half = (ft_dim as usize) / 2;
    let total_pairs = (batch as usize) * half;
    if tid.get() >= total_pairs {
        return;
    }
    let bi = tid.get() / half;
    let pair_idx = tid.get() % half;

    let dy_idx = bi * (d_combined_stride as usize) + (d_combined_offset as usize) + pair_idx;
    let dy = d_combined_a[dy_idx] + d_combined_b[dy_idx];

    let ft_base = bi * (ft_dim as usize);
    let xa = ft_out[ft_base + pair_idx] as f32 + bias[pair_idx];
    let xb = ft_out[ft_base + half + pair_idx] as f32 + bias[half + pair_idx];

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

    let grad_a_post = dy * yb * scale;
    let grad_a = if xa > 0.0_f32 && xa < 1.0_f32 {
        grad_a_post
    } else {
        0.0_f32
    };
    let grad_b_post = dy * ya * scale;
    let grad_b = if xb > 0.0_f32 && xb < 1.0_f32 {
        grad_b_post
    } else {
        0.0_f32
    };

    // grad_ft_out は f16。1 thread が 2 cell を書く構造・disjoint 性は
    // `ft_post_perspective_grad_fused` と同一 (SAFETY 不変条件はそのまま、要素型のみ f16)。
    // dft_scale を掛けてから f16 化する (loss scaling、gather 側で逆数を掛けて戻す)。
    //
    // `grad * dft_scale` は f16 有限域 (`|x| <= 65504`) を超えうる。clamp せず `as f16`
    // すると天井を越えた値が `±inf` になり、gather で `ft_w_grad` に伝播 → optimizer
    // 経由で weight を NaN 化させ学習を発散させる。これを防ぐため格納前に clamp する。
    // clamp が当たるのは天井を越えた稀な外れ値のみで、その要素の勾配が cap される
    // (発散の代わりに有界な近似)。
    let da = grad_a * dft_scale;
    let da_c = if da > 65504.0_f32 {
        65504.0_f32
    } else if da < -65504.0_f32 {
        -65504.0_f32
    } else {
        da
    };
    let db = grad_b * dft_scale;
    let db_c = if db > 65504.0_f32 {
        65504.0_f32
    } else if db < -65504.0_f32 {
        -65504.0_f32
    } else {
        db
    };
    let out_ptr = grad_ft_out.as_mut_ptr();
    unsafe {
        out_ptr.add(ft_base + pair_idx).write(da_c as f16);
        out_ptr.add(ft_base + half + pair_idx).write(db_c as f16);
    }

    // grad_bias は f32 accumulate を維持 (f32 の grad_a / grad_b をそのまま atomic add)。
    let bias_cell_a = unsafe { &*(grad_bias.as_ptr().add(pair_idx) as *const DeviceAtomicF32) };
    bias_cell_a.fetch_add(grad_a, AtomicOrdering::Relaxed);
    let bias_cell_b =
        unsafe { &*(grad_bias.as_ptr().add(half + pair_idx) as *const DeviceAtomicF32) };
    bias_cell_b.fetch_add(grad_b, AtomicOrdering::Relaxed);
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

/// Tiled shared-memory variant of [`dense_mm_bwd_input`]. L1f 用 fixed shape
/// (`in_dim=1536`, `out_dim=16`)、`batch % 16 == 0`、`in_dim % 16 == 0` を host が保証。
///
/// 元 `dense_mm_bwd_input` は w[ii][o] (out-major) read で warp 内 ii=0..31 が stride 16 = 64B
/// = 32 cache lines load → uncoalesced。本 kernel は W_TILE / DY_TILE を shared に load
/// (coalesced)、各 thread が 1 (bi, ii) cell を 16 FMA で完成。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_input_tiled(
    dy: &[f32],
    w: &[f32],
    mut dx: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
) {
    static mut W_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // TILE_IN × 16
    static mut DY_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // TILE_B × 16

    let tid_local = thread::threadIdx_x() as usize;
    // 1D grid: block_idx encodes (b_block, ii_block). 全 cell の 1D 順序を保持し
    // `dx.get_mut(thread::index_1d())` で disjoint write を成立させる。
    // grid_dim = (in_dim/16) * (batch/16)、block index = b_block * (in_dim/16) + ii_block。
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let blocks_per_b_row = in_dim_u >> 4; // in_dim / 16
    let block_lin = thread::blockIdx_x() as usize;
    let block_b = block_lin / blocks_per_b_row;
    let block_ii = block_lin % blocks_per_b_row;
    let tid_b = tid_local >> 4;
    let tid_i = tid_local & 15;
    let b_start = block_b << 4;
    let ii_start = block_ii << 4;
    let global_bi = b_start + tid_b;
    let global_ii = ii_start + tid_i;

    let bi_ok = global_bi < batch_u;
    let ii_ok = global_ii < in_dim_u;

    // W_TILE [TILE_IN × out_dim=16]: 256 cells.
    // Cell layout: W_TILE[ii_local * 16 + o] = w[(ii_start + ii_local) * out_dim + o]
    // Map tid_local → (ii_local = tid/16, o = tid%16). For warp tid 0..31: ii_local in {0,1},
    // o in 0..15 → 16-thread sub-group reads 16 consecutive o (= 1 cache line). Coalesced ✓
    unsafe {
        let ii_local_load = tid_b;
        let o_load = tid_i;
        let ii_global_load = ii_start + ii_local_load;
        W_TILE[tid_local] = if ii_global_load < in_dim_u && o_load < out_dim_u {
            w[ii_global_load * out_dim_u + o_load]
        } else {
            0.0_f32
        };
        // DY_TILE [TILE_B × 16] = 256 cells.
        // Cell DY_TILE[b_local * 16 + o] = dy[(b_start + b_local) * out_dim + o]
        // Map tid_local → (b_local = tid/16, o = tid%16). Coalesced.
        let b_local_load = tid_b;
        let bb_global_load = b_start + b_local_load;
        DY_TILE[tid_local] = if bb_global_load < batch_u && o_load < out_dim_u {
            dy[bb_global_load * out_dim_u + o_load]
        } else {
            0.0_f32
        };
    }
    thread::sync_threads();

    if bi_ok && ii_ok {
        let mut acc = 0.0_f32;
        let mut o: usize = 0;
        while o < 16 {
            unsafe {
                acc += DY_TILE[(tid_b << 4) | o] * W_TILE[(tid_i << 4) | o];
            }
            o += 1;
        }
        // 2D tile grid → cell index は (b_block, ii_block) と (tid_b, tid_i) から合成。
        // thread::index_1d() (block_lin * 256 + tid_local) と cell_idx は order が異なるため
        // raw pointer 経由で write (各 thread は disjoint cell を担当、host が grid_dim 整合)。
        let cell_idx = global_bi * in_dim_u + global_ii;
        unsafe {
            *dx.as_mut_ptr().add(cell_idx) = acc;
        }
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

/// Tiled shared-memory variant of [`dense_mm_bwd_weight`]. L1f 用 (`in_dim=1536`,
/// `out_dim=16`, `batch=65536`) を想定した固定タイル形状 (TILE_K=16, TILE_IN=16,
/// TILE_OUT=16, block=256 threads)。`in_dim % 16 == 0 && out_dim == 16 && batch % 16 == 0`
/// が host 契約。非該当形状では結果未定義 (host 側で sizes チェックの上で本 kernel を選ぶ)。
///
/// 1 block = 1 (TILE_IN × TILE_OUT) W tile。block 内 256 threads が batch を TILE_K=16
/// chunk で cooperatively load し、shared memory 上で TILE_K 回 FMA。current "1 thread =
/// 1 cell、scan batch" 比 ~33x 少ない unique memory read (x 16x redundant → 1x、dy 1536x → 1x)。
///
/// SAFETY: `static mut TILE` への access は block-local barrier (`sync_threads`) で
/// race を防ぐ。各 thread の write index は disjoint なので per-thread access は安全。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight_tiled(
    x: &[f32],
    dy: &[f32],
    mut grad_w: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
) {
    // 256 element tiles → 1 KB / tile (= within 100 KB sm_86 shared mem budget)。
    static mut X_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // TILE_K × TILE_IN
    static mut DY_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // TILE_K × TILE_OUT

    let tid_local = thread::threadIdx_x() as usize;
    let block_x = thread::blockIdx_x() as usize;
    let tid_i = tid_local >> 4; // tid / 16
    let tid_o = tid_local & 15; // tid % 16
    let global_ii = block_x * 16 + tid_i;
    let global_oi = tid_o;

    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let in_ok = global_ii < in_dim_u;
    let out_ok = global_oi < out_dim_u;

    let mut acc: f32 = 0.0_f32;
    let n_k_tiles = batch_u >> 4; // batch / 16
    let mut k_tile: usize = 0;
    while k_tile < n_k_tiles {
        let b_start = k_tile << 4;
        // Cooperative load: 256 threads × 1 cell each.
        // X_TILE[k * TILE_IN + ii] = x[(b_start + k) * in_dim + (block_x * TILE_IN + ii)]
        //  Warp threads (tid 0..31) → k = tid/16 ∈ {0,1}, ii = tid%16 ∈ 0..15.
        //  Within k segment (tid 0..15 or 16..31), 16 consecutive ii → coalesced read of x row.
        unsafe {
            let bb = b_start + tid_i;
            let global_ii_load = (block_x << 4) | tid_o;
            // Use tid_i as k (0..15) and tid_o as ii within tile (0..15) for X load.
            let mapped = (tid_i << 4) | tid_o; // = tid_local
            if bb < batch_u && global_ii_load < in_dim_u {
                X_TILE[mapped] = x[bb * in_dim_u + global_ii_load];
            } else {
                X_TILE[mapped] = 0.0_f32;
            }
            // DY_TILE[k * TILE_OUT + oi] = dy[(b_start + k) * out_dim + oi]
            // Use tid_i as k and tid_o as oi.
            if bb < batch_u && tid_o < out_dim_u {
                DY_TILE[mapped] = dy[bb * out_dim_u + tid_o];
            } else {
                DY_TILE[mapped] = 0.0_f32;
            }
        }
        thread::sync_threads();

        // Compute: each thread computes 1 (global_ii, global_oi) cell using 16 K iterations.
        if in_ok && out_ok {
            let mut k: usize = 0;
            while k < 16 {
                unsafe {
                    acc += X_TILE[(k << 4) | tid_i] * DY_TILE[(k << 4) | tid_o];
                }
                k += 1;
            }
        }
        thread::sync_threads();
        k_tile += 1;
    }

    if in_ok && out_ok {
        // cell_idx == thread::index_1d() since tid_i = tid/16, tid_o = tid%16 and
        // global cell_idx = global_ii * out_dim + global_oi
        //                 = (block_x * 16 + tid_i) * 16 + tid_o
        //                 = block_x * 256 + tid_local = thread::index_1d().get()
        let global_tid = thread::index_1d();
        if let Some(g) = grad_w.get_mut(global_tid) {
            *g = acc;
        }
    }
}

/// Tiled per-bucket weight backward (L1 / fixed shape: `in_dim=1536`, `out_dim=16`,
/// `num_buckets=9`, `batch % 16 == 0`)。
///
/// 元の `dense_mm_bwd_weight_bucket` (1 thread = 1 (buc, oi, ii) cell、scan batch、
/// bucket filter を inner loop で 9 倍冗長に評価) を「block per W tile (16x16)、
/// 1 thread = 9 bucket × 1 cell の register accumulator、batch scan 1 回」に書き換え。
/// 副作用: `dy_tile`、`x_tile`、`buc_tile` を shared mem に coalesced load し、batch を
/// TILE_K=16 chunk で消化。bucket 分岐は uniform (同 k 内で warp 全 thread が同 buc) なので
/// divergence なし。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight_bucket_tiled_l1(
    x: &[f32],
    dy: &[f32],
    bucket_idx: &[i32],
    grad_w: &[f32],
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    static mut X_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut DY_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut BUC_TILE: SharedArray<i32, 16> = SharedArray::UNINIT;

    let tid_local = thread::threadIdx_x() as usize;
    let block_x = thread::blockIdx_x() as usize;
    let block_split = thread::blockIdx_y() as usize;
    let num_splits = thread::gridDim_y() as usize;
    let tid_i = tid_local >> 4;
    let tid_o = tid_local & 15;
    let global_ii = (block_x << 4) | tid_i;
    let global_oi = tid_o;

    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let num_buc_u = num_buckets as usize;
    let in_ok = global_ii < in_dim_u;
    let out_ok = global_oi < out_dim_u;

    // split-K: 各 block が batch slice を担当。num_splits=1 で 1 block が全 batch を scan。
    let positions_per_split = batch_u.div_ceil(num_splits);
    let split_b_start = block_split * positions_per_split;
    if split_b_start >= batch_u {
        return;
    }
    let split_b_end_candidate = split_b_start + positions_per_split;
    let split_b_end = if split_b_end_candidate < batch_u {
        split_b_end_candidate
    } else {
        batch_u
    };
    // TILE_K=16 単位で並ぶよう、batch slice は 16 の倍数を host が保証 (`debug_assert` 済)。
    // 端数 split は最後の block が短くなる (b_end が batch_u に丸まる)。

    // 9 個の bucket accumulator (fixed expansion で register に置く)。
    let mut a0 = 0.0_f32;
    let mut a1 = 0.0_f32;
    let mut a2 = 0.0_f32;
    let mut a3 = 0.0_f32;
    let mut a4 = 0.0_f32;
    let mut a5 = 0.0_f32;
    let mut a6 = 0.0_f32;
    let mut a7 = 0.0_f32;
    let mut a8 = 0.0_f32;

    let n_k_tiles = (split_b_end - split_b_start) >> 4;
    let mut k_tile: usize = 0;
    while k_tile < n_k_tiles {
        let b_start = split_b_start + (k_tile << 4);
        unsafe {
            let bb = b_start + tid_i;
            let global_ii_load = (block_x << 4) | tid_o;
            let mapped = (tid_i << 4) | tid_o;
            X_TILE[mapped] = if bb < batch_u && global_ii_load < in_dim_u {
                x[bb * in_dim_u + global_ii_load]
            } else {
                0.0_f32
            };
            DY_TILE[mapped] = if bb < batch_u && tid_o < out_dim_u {
                dy[bb * out_dim_u + tid_o]
            } else {
                0.0_f32
            };
            // BUC_TILE: 16 個 (= TILE_K)。先頭 16 thread (tid_local < 16) が load 担当。
            if tid_local < 16 {
                let bb2 = b_start + tid_local;
                BUC_TILE[tid_local] = if bb2 < batch_u {
                    bucket_idx[bb2]
                } else {
                    -1_i32
                };
            }
        }
        thread::sync_threads();

        if in_ok && out_ok {
            let mut k: usize = 0;
            while k < 16 {
                unsafe {
                    let buc = BUC_TILE[k];
                    let mul = X_TILE[(k << 4) | tid_i] * DY_TILE[(k << 4) | tid_o];
                    // num_buckets=9 を想定。負値・>=9 は無視 (silent skip、元 kernel と同じ)。
                    if buc == 0 {
                        a0 += mul;
                    } else if buc == 1 {
                        a1 += mul;
                    } else if buc == 2 {
                        a2 += mul;
                    } else if buc == 3 {
                        a3 += mul;
                    } else if buc == 4 {
                        a4 += mul;
                    } else if buc == 5 {
                        a5 += mul;
                    } else if buc == 6 {
                        a6 += mul;
                    } else if buc == 7 {
                        a7 += mul;
                    } else if buc == 8 {
                        a8 += mul;
                    }
                }
                k += 1;
            }
        }
        thread::sync_threads();
        k_tile += 1;
    }

    // Write: grad_w[buc * out_dim * in_dim + global_ii * out_dim + global_oi] かと思いきや、
    // 元 kernel の layout は `grad_w[buc][o][i]` row-major、つまり buc * out_dim * in_dim +
    // out_idx * in_dim + in_idx (out-major そして in-major) で、`tid_in_block` 全 thread が
    // bucket buc に対して書く 1 cell の index = buc * (out_dim * in_dim) + oi * in_dim + ii。
    if in_ok && out_ok {
        let per_bucket = out_dim_u * in_dim_u;
        let cell_in_bucket = global_oi * in_dim_u + global_ii;
        // split-K では num_splits >= 1 block が同 cell に partial sum を寄せるため atomicAdd。
        // num_splits=1 でも 1 回の atomicAdd になるだけで結果は同じ (grad_w は host が memset 0)。
        let raw = grad_w.as_ptr();
        if num_buc_u >= 1 {
            unsafe {
                let c = &*(raw.add(cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a0, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 2 {
            unsafe {
                let c = &*(raw.add(per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a1, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 3 {
            unsafe {
                let c = &*(raw.add(2 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a2, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 4 {
            unsafe {
                let c = &*(raw.add(3 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a3, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 5 {
            unsafe {
                let c = &*(raw.add(4 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a4, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 6 {
            unsafe {
                let c = &*(raw.add(5 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a5, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 7 {
            unsafe {
                let c = &*(raw.add(6 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a6, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 8 {
            unsafe {
                let c = &*(raw.add(7 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a7, AtomicOrdering::Relaxed);
            }
        }
        if num_buc_u >= 9 {
            unsafe {
                let c = &*(raw.add(8 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
                c.fetch_add(a8, AtomicOrdering::Relaxed);
            }
        }
    }
}

/// Sorted layout 版 [`dense_mm_bwd_weight_bucket_tiled_l1`]。caller が batch を bucket で
/// sort 済かつ各 bucket の sorted 開始 offset が `TILE_B = 16` 境界に align 済を保証する
/// (`exclusive_scan_aligned` 経由)。grid 構成:
/// - `blockIdx_x` = in_tile (`in_dim / 16` 個)
/// - `blockIdx_y` = bucket 内 split-K (`gridDim_y` 個の連続 TILE_K slice)
/// - `blockIdx_z` = bucket (`num_buckets` 個)
///
/// 各 block は uniform-by-construction で 1 bucket の slice のみ accumulate。9-way if-else
/// dispatch / 9 register accumulator / 9 atomic write はすべて 1 個ずつに集約され、
/// 終端で `grad_w[block_buc][oi][ii]` に 1 atomicAdd。
///
/// padding 行 (perm=-1 由来で `permute_rows_f32` が 0 fill) は x,dy=0 で sum=0 contribution、
/// bucket slice 末端の 16-alignment slack 行も同様に silent に 0 contribution。
///
/// 数値同等性: 加算順序が sort 済 batch 順 + split-K 集約順になるため fp32 associativity で
/// baseline と bit-exact ではないが、reduction tolerance (相対誤差 < `TOL`) 内で一致。
/// `in_dim % 16 == 0` / `out_dim == 16` / `num_buckets <= 9` / `padded_batch % 16 == 0` /
/// `bucket_offsets` が aligned exclusive scan 出力 は caller 契約。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight_bucket_tiled_l1_sorted(
    x: &[f32],
    dy: &[f32],
    bucket_offsets: &[u32],
    grad_w: &[f32],
    padded_batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    static mut X_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut DY_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid_local = thread::threadIdx_x() as usize;
    let block_x = thread::blockIdx_x() as usize;
    let block_split = thread::blockIdx_y() as usize;
    let num_splits = thread::gridDim_y() as usize;
    let block_buc = thread::blockIdx_z() as usize;
    let tid_i = tid_local >> 4;
    let tid_o = tid_local & 15;
    let global_ii = (block_x << 4) | tid_i;
    let global_oi = tid_o;

    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let padded_b_u = padded_batch as usize;
    let num_buc_u = num_buckets as usize;
    let in_ok = global_ii < in_dim_u;
    let out_ok = global_oi < out_dim_u;
    let buc_ok = block_buc < num_buc_u;

    let buc_start = bucket_offsets[block_buc] as usize;
    let buc_end_raw = bucket_offsets[block_buc + 1] as usize;
    let buc_end = if buc_end_raw < padded_b_u {
        buc_end_raw
    } else {
        padded_b_u
    };
    let buc_size = buc_end.saturating_sub(buc_start);
    let n_total_tiles = buc_size >> 4;

    let tiles_per_split = n_total_tiles.div_ceil(num_splits);
    let split_tile_start = block_split * tiles_per_split;
    let split_tile_end_cand = split_tile_start + tiles_per_split;
    let split_tile_end = if split_tile_end_cand < n_total_tiles {
        split_tile_end_cand
    } else {
        n_total_tiles
    };

    let mut acc: f32 = 0.0_f32;
    if buc_ok && split_tile_start < n_total_tiles {
        let mut k_tile = split_tile_start;
        while k_tile < split_tile_end {
            let b_start = buc_start + (k_tile << 4);
            unsafe {
                let bb = b_start + tid_i;
                let global_ii_load = (block_x << 4) | tid_o;
                let mapped = (tid_i << 4) | tid_o;
                X_TILE[mapped] = if bb < buc_end && global_ii_load < in_dim_u {
                    x[bb * in_dim_u + global_ii_load]
                } else {
                    0.0_f32
                };
                DY_TILE[mapped] = if bb < buc_end && tid_o < out_dim_u {
                    dy[bb * out_dim_u + tid_o]
                } else {
                    0.0_f32
                };
            }
            thread::sync_threads();

            if in_ok && out_ok {
                let mut k: usize = 0;
                while k < 16 {
                    unsafe {
                        acc += X_TILE[(k << 4) | tid_i] * DY_TILE[(k << 4) | tid_o];
                    }
                    k += 1;
                }
            }
            thread::sync_threads();
            k_tile += 1;
        }
    }

    if buc_ok && in_ok && out_ok {
        let per_bucket = out_dim_u * in_dim_u;
        let cell_in_bucket = global_oi * in_dim_u + global_ii;
        let raw = grad_w.as_ptr();
        unsafe {
            let c = &*(raw.add(block_buc * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(acc, AtomicOrdering::Relaxed);
        }
    }
}

/// Bias gradient (block-level shared-mem reduction) — L1f 用 (`out_dim=16`)。
///
/// 元 `bias_grad` は 1M threads × 1 atomic → 16 cells で contention 大。本 kernel は
/// 各 block (256 threads) が shared-mem 16-cell accumulator に集約 → 1 block × 16 atomic
/// add で global に flush。global atomic 数 = blocks × 16 (= ~64K) で contention 大幅減。
#[kernel]
pub fn bias_grad_shared_l1f(dy: &[f32], grad_bias: &[f32], batch: u32, out_dim: u32) {
    use core::ptr::addr_of_mut;
    static mut PARTIAL: SharedArray<f32, 16> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    let block_idx = thread::blockIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let total = batch_u * out_dim_u;

    let partial_ptr: *mut f32 = addr_of_mut!(PARTIAL) as *mut f32;

    // 初期化: 先頭 out_dim threads が PARTIAL を 0 reset。
    if tid < out_dim_u {
        unsafe {
            partial_ptr.add(tid).write(0.0_f32);
        }
    }
    thread::sync_threads();

    // accumulate: 各 thread = 1 (b, oi) cell の dy 値を shared atomic add (16 cells に contention)。
    let global_idx = block_idx * block_dim_u + tid;
    if global_idx < total {
        let oi = global_idx % out_dim_u;
        let dyv = dy[global_idx];
        let cell = unsafe { &*(partial_ptr.add(oi) as *const DeviceAtomicF32) };
        cell.fetch_add(dyv, AtomicOrdering::Relaxed);
    }
    thread::sync_threads();

    // flush: 先頭 out_dim threads が PARTIAL → grad_bias に atomic add。
    if tid < out_dim_u {
        let p = unsafe { partial_ptr.add(tid).read() };
        let cell = unsafe { &*(grad_bias.as_ptr().add(tid) as *const DeviceAtomicF32) };
        cell.fetch_add(p, AtomicOrdering::Relaxed);
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

/// Tiled non-bucket forward dense matmul (L1f 用 fixed shape: `in_dim=1536`, `out_dim=16`)。
/// 元 `dense_mm_fwd` は coalesced だが 1 thread = 1 (b, oi) で per-thread 1536 K iter で
/// 並列度限界。本 kernel は block tile (TILE_B=16 × TILE_OUT=16 = 256 cells)、K=16 chunk
/// で shared-mem cooperative load → 256 cells / block で並列度 4K blocks × 256 threads。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_fwd_tiled_l1f(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    mut y: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
) {
    static mut X_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut W_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid_local = thread::threadIdx_x() as usize;
    let block_b = thread::blockIdx_x() as usize;
    let tid_b = tid_local >> 4;
    let tid_o = tid_local & 15;
    let b_start = block_b << 4;
    let global_bi = b_start + tid_b;
    let global_oi = tid_o;
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let bi_ok = global_bi < batch_u;
    let oi_ok = global_oi < out_dim_u;

    let bias_init = if bi_ok && oi_ok {
        bias[global_oi]
    } else {
        0.0_f32
    };
    let mut acc: f32 = bias_init;

    let n_k_tiles = in_dim_u >> 4;
    let mut k_tile: usize = 0;
    while k_tile < n_k_tiles {
        let k_start = k_tile << 4;
        // X_TILE [TILE_B × TILE_K]: x[(b_start+tid_b)*in_dim + (k_start+tid_o)]
        unsafe {
            let bb = b_start + tid_b;
            let kk = k_start + tid_o;
            X_TILE[tid_local] = if bb < batch_u && kk < in_dim_u {
                x[bb * in_dim_u + kk]
            } else {
                0.0_f32
            };
            // W_TILE [TILE_OUT × TILE_K]: w[(k_start+k_local) * out_dim + tid_o_load]
            // w layout: in-major × out-major (`w[ii * out_dim + oi]`)、coalesced for `tid_o` varies.
            // Map tid_local → (k_local = tid/16, o_load = tid%16)
            let k_local = tid_b; // tid_local / 16
            let o_load = tid_o; // tid_local & 15
            let kk2 = k_start + k_local;
            W_TILE[tid_local] = if kk2 < in_dim_u && o_load < out_dim_u {
                w[kk2 * out_dim_u + o_load]
            } else {
                0.0_f32
            };
        }
        thread::sync_threads();

        if bi_ok && oi_ok {
            let mut k: usize = 0;
            while k < 16 {
                unsafe {
                    acc += X_TILE[(tid_b << 4) | k] * W_TILE[(k << 4) | tid_o];
                }
                k += 1;
            }
        }
        thread::sync_threads();
        k_tile += 1;
    }

    if bi_ok
        && oi_ok
        && let Some(o) = y.get_mut(thread::index_1d())
    {
        *o = acc;
    }
}

/// Tiled per-bucket forward dense matmul (L1 用 fixed shape: `in_dim=1536`, `out_dim=16`,
/// `num_buckets=9`)。
///
/// 元 `dense_mm_fwd_bucket` は `w[buc][oi][ii]` layout のため、warp 内 16-thread sub-group が
/// oi 軸を varying させると stride=in_dim=1536 で uncoalesced。本 kernel は 1 block = 1 batch
/// tile (TILE_B=16) × 全 oi (= TILE_OUT=16)、K (= in_dim) を TILE_K=16 chunk で消化し、shared
/// memory 上で `w_tile [NUM_BUCKETS × TILE_OUT × TILE_K]` を per-K-tile load (coalesced)。各
/// thread は自分の bucket の W 行を shared から読んで accumulate。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_fwd_bucket_tiled_l1(
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
    static mut X_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // 16 × 16
    static mut W_TILE: SharedArray<f32, 2304> = SharedArray::UNINIT; // 9 × 16 × 16
    static mut BUC_TILE: SharedArray<i32, 16> = SharedArray::UNINIT;

    let tid_local = thread::threadIdx_x() as usize;
    let block_b = thread::blockIdx_x() as usize;
    let tid_b = tid_local >> 4; // tid / 16
    let tid_o = tid_local & 15; // tid % 16
    let b_start = block_b << 4;
    let global_bi = b_start + tid_b;
    let global_oi = tid_o;

    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let num_buc_u = num_buckets as usize;
    let bi_ok = global_bi < batch_u;
    let oi_ok = global_oi < out_dim_u;

    // BUC_TILE load (1 回だけ、K loop の前)。
    unsafe {
        if tid_local < 16 {
            let bb = b_start + tid_local;
            BUC_TILE[tid_local] = if bb < batch_u { bucket_idx[bb] } else { -1_i32 };
        }
    }
    thread::sync_threads();

    // bucket 別 bias を初期値に。
    let my_buc = unsafe { BUC_TILE[tid_b] };
    let bias_init = if bi_ok && oi_ok && my_buc >= 0 && (my_buc as u32) < num_buckets {
        bias[(my_buc as usize) * out_dim_u + global_oi]
    } else {
        0.0_f32
    };
    let mut acc: f32 = bias_init;

    let n_k_tiles = in_dim_u >> 4; // in_dim / 16
    let mut k_tile: usize = 0;
    while k_tile < n_k_tiles {
        let k_start = k_tile << 4;
        // X_TILE [TILE_B × TILE_K]: 16x16 = 256 cells、tid → (tid_b, tid_o) → ((b_start+tid_b), (k_start+tid_o))
        unsafe {
            let bb = b_start + tid_b;
            let kk = k_start + tid_o;
            X_TILE[tid_local] = if bb < batch_u && kk < in_dim_u {
                x[bb * in_dim_u + kk]
            } else {
                0.0_f32
            };
        }
        // W_TILE [NUM_BUCKETS × TILE_OUT × TILE_K] = 2304 cells, 256 threads × 9 cells each
        // Cell layout: cell_idx = buc * 256 + oi_local * 16 + k_local
        // tid_local → (oi_local = tid/16, k_local = tid%16)
        // Per-bucket: read w[buc * out_dim * in_dim + oi_local * in_dim + (k_start + k_local)]
        unsafe {
            let oi_local = tid_b; // = tid_local / 16
            let k_local = tid_o; // = tid_local & 15
            let kk = k_start + k_local;
            let mut buc: usize = 0;
            while buc < num_buc_u {
                let val = if oi_local < out_dim_u && kk < in_dim_u {
                    w[buc * out_dim_u * in_dim_u + oi_local * in_dim_u + kk]
                } else {
                    0.0_f32
                };
                W_TILE[(buc << 8) | (oi_local << 4) | k_local] = val;
                buc += 1;
            }
        }
        thread::sync_threads();

        // Compute: each thread accumulates 1 cell (global_bi, global_oi) over TILE_K K iterations.
        if bi_ok && oi_ok && my_buc >= 0 && (my_buc as u32) < num_buckets {
            let buc_u = my_buc as usize;
            let mut k: usize = 0;
            while k < 16 {
                unsafe {
                    acc += X_TILE[(tid_b << 4) | k] * W_TILE[(buc_u << 8) | (tid_o << 4) | k];
                }
                k += 1;
            }
        }
        thread::sync_threads();
        k_tile += 1;
    }

    if bi_ok && oi_ok {
        if my_buc < 0 || (my_buc as u32) >= num_buckets {
            if let Some(o) = y.get_mut(thread::index_1d()) {
                *o = 0.0_f32;
            }
        } else if let Some(o) = y.get_mut(thread::index_1d()) {
            *o = acc;
        }
    }
}

/// Bucket histogram。`bucket_idx` の各 value (有効 range `[0, num_buckets)`) ごとに
/// thread が atomic add する。範囲外 (-1, >= num_buckets) は最後 slot `num_buckets`
/// に集約 (invalid bin、後段で値 0 を書き込ませる)。counts は `num_buckets + 1` 要素。
#[kernel]
pub fn count_buckets(bucket_idx: &[i32], counts: &[u32], batch: u32, num_buckets: u32) {
    let tid = thread::index_1d();
    if tid.get() >= batch as usize {
        return;
    }
    let b = bucket_idx[tid.get()];
    let bin = if b >= 0 && (b as u32) < num_buckets {
        b as u32
    } else {
        num_buckets
    };
    unsafe {
        let atom = &*(counts.as_ptr().add(bin as usize) as *const DeviceAtomicU32);
        atom.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

/// `counts[0..n]` の exclusive prefix sum を `offsets[0..n]` に書く。`align` (= 16) で
/// 各 bucket の sorted layout 開始 offset を round up し、bucket 境界を block size
/// (`TILE_B = 16`) に揃える。bucket 末端と次 bucket 開始の間は padding 行 (caller 側で
/// invalid bucket marker `-1` で埋める) になり、kernel は uniform block 前提で走れる。
/// n ≤ NUM_BUCKETS + 1 = 10 想定で 1 thread sequential。
#[kernel]
pub fn exclusive_scan_aligned(counts: &[u32], offsets: &[u32], n: u32, align: u32) {
    if thread::index_1d().get() != 0 {
        return;
    }
    let n_u = n as usize;
    let mut acc: u32 = 0;
    let mut i: usize = 0;
    while i < n_u {
        // acc を align 倍数に切り上げ (acc % align == 0 でなければ次の境界へ)
        let rem = acc % align;
        if rem != 0 {
            acc += align - rem;
        }
        unsafe {
            let dst = offsets.as_ptr().add(i) as *mut u32;
            *dst = acc;
        }
        acc += counts[i];
        i += 1;
    }
}

/// stable counting sort の scatter phase。各 thread が bucket_idx[i] = b を読み、
/// dst = offsets[b] + (b 内 in-order rank) に perm[dst] = i / sorted_bucket[dst] = b
/// を書き込む。in-order rank は `write_ctr[b]` を atomic_inc して取る (atomic 順
/// 依存で stable ではない、bit-exact が必要な kernel では bucket boundary 内
/// associativity 注意)。
#[kernel]
pub fn scatter_bucket_perm(
    bucket_idx: &[i32],
    offsets: &[u32],
    write_ctr: &[u32],
    perm: &[i32],
    sorted_bucket: &[i32],
    batch: u32,
    num_buckets: u32,
) {
    let tid = thread::index_1d();
    if tid.get() >= batch as usize {
        return;
    }
    let b = bucket_idx[tid.get()];
    let bin = if b >= 0 && (b as u32) < num_buckets {
        b as u32
    } else {
        num_buckets
    };
    let rank = unsafe {
        let atom = &*(write_ctr.as_ptr().add(bin as usize) as *const DeviceAtomicU32);
        atom.fetch_add(1, AtomicOrdering::Relaxed)
    };
    let dst = (offsets[bin as usize] + rank) as usize;
    unsafe {
        let perm_dst = perm.as_ptr().add(dst) as *mut i32;
        *perm_dst = tid.get() as i32;
        let sb_dst = sorted_bucket.as_ptr().add(dst) as *mut i32;
        *sb_dst = b;
    }
}

/// Row-permute (gather): `out[i, :] = in[perm[i], :]`。1 thread = 1 (row, col) cell、
/// 1D launch (`batch * dim`)。perm[i] が範囲外 (`< 0 || >= batch`) は host 契約違反。
#[kernel]
pub fn permute_rows_f32(
    input: &[f32],
    perm: &[i32],
    mut output: DisjointSlice<f32>,
    batch: u32,
    dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (dim as usize);
    if tid.get() >= total {
        return;
    }
    let row = tid.get() / (dim as usize);
    let col = tid.get() % (dim as usize);
    let src_row = perm[row];
    let val = if src_row >= 0 && (src_row as u32) < batch {
        input[(src_row as usize) * (dim as usize) + col]
    } else {
        0.0_f32
    };
    if let Some(o) = output.get_mut(tid) {
        *o = val;
    }
}

/// Row-inverse-permute (scatter): `out[perm[i], :] = in[i, :]`。perm は forward
/// gather index で、bijection 前提 (counting sort 出力)。1 thread = 1 (row, col) cell、
/// 各 thread の write は disjoint なので raw ptr write OK。
#[kernel]
pub fn inverse_permute_rows_f32(input: &[f32], perm: &[i32], output: &[f32], batch: u32, dim: u32) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (dim as usize);
    if tid.get() >= total {
        return;
    }
    let row = tid.get() / (dim as usize);
    let col = tid.get() % (dim as usize);
    let dst_row = perm[row];
    if dst_row < 0 || (dst_row as u32) >= batch {
        return;
    }
    let dst_idx = (dst_row as usize) * (dim as usize) + col;
    unsafe {
        let dst = output.as_ptr().add(dst_idx) as *mut f32;
        *dst = input[tid.get()];
    }
}

/// Sorted layout 版 [`dense_mm_fwd_bucket_tiled_l1`]。caller が batch を bucket で
/// sort 済かつ各 bucket の sorted 開始 offset が `TILE_B = 16` 境界に align 済
/// (`exclusive_scan_aligned` 経由) を保証する前提。block 内全 TILE_B = 16 row は同一 bucket
/// (uniform-by-construction、boundary block は存在しない)、per-K-tile の W_TILE shared-mem
/// は 1 bucket 分 (16 × 16 = 256 cell) のみ load する分岐なし実装。padding 行は
/// `bucket_idx = -1` で kernel が y=0 を書き、後段の inverse permute が perm=-1 sentinel で
/// skip して original 配列には戻らない。
///
/// 数値同等性: per-row independent (k=0..15 加算順保持) で baseline と bit-exact、
/// sort stability 不要。`in_dim % 16 == 0` / `out_dim == 16` / `batch % 16 == 0` /
/// `num_buckets <= 9` は caller 契約。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_fwd_bucket_tiled_l1_sorted(
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
    static mut X_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut W_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // 1 × 16 × 16

    let tid_local = thread::threadIdx_x() as usize;
    let block_b = thread::blockIdx_x() as usize;
    let tid_b = tid_local >> 4;
    let tid_o = tid_local & 15;
    let b_start = block_b << 4;
    let global_bi = b_start + tid_b;
    let global_oi = tid_o;

    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let bi_ok = global_bi < batch_u;
    let oi_ok = global_oi < out_dim_u;

    // aligned sorted layout 前提で block は uniform-by-construction。b_start の bucket を
    // 代表 = 全 row 共通 bucket。padding 行 / 終端 block は bucket = -1 で skip。
    let block_buc = if b_start < batch_u {
        bucket_idx[b_start]
    } else {
        -1_i32
    };
    let block_buc_ok = block_buc >= 0 && (block_buc as u32) < num_buckets;
    let block_buc_u = if block_buc_ok { block_buc as usize } else { 0 };

    let bias_init = if bi_ok && oi_ok && block_buc_ok {
        bias[block_buc_u * out_dim_u + global_oi]
    } else {
        0.0_f32
    };
    let mut acc: f32 = bias_init;

    let n_k_tiles = in_dim_u >> 4;
    let mut k_tile: usize = 0;
    while k_tile < n_k_tiles {
        let k_start = k_tile << 4;
        unsafe {
            let bb = b_start + tid_b;
            let kk = k_start + tid_o;
            X_TILE[tid_local] = if bb < batch_u && kk < in_dim_u {
                x[bb * in_dim_u + kk]
            } else {
                0.0_f32
            };
        }
        unsafe {
            let oi_local = tid_b;
            let k_local = tid_o;
            let kk = k_start + k_local;
            let val = if block_buc_ok && oi_local < out_dim_u && kk < in_dim_u {
                w[block_buc_u * out_dim_u * in_dim_u + oi_local * in_dim_u + kk]
            } else {
                0.0_f32
            };
            W_TILE[(oi_local << 4) | k_local] = val;
        }
        thread::sync_threads();

        if bi_ok && oi_ok && block_buc_ok {
            let mut k: usize = 0;
            while k < 16 {
                unsafe {
                    acc += X_TILE[(tid_b << 4) | k] * W_TILE[(tid_o << 4) | k];
                }
                k += 1;
            }
        }
        thread::sync_threads();
        k_tile += 1;
    }

    if bi_ok && oi_ok {
        if !block_buc_ok {
            if let Some(o) = y.get_mut(thread::index_1d()) {
                *o = 0.0_f32;
            }
        } else if let Some(o) = y.get_mut(thread::index_1d()) {
            *o = acc;
        }
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
/// は不要 (1 thread = 1 (batch, out, in) で同 weight cell へ多 thread atomic add する
/// 素直な形は bucket 偏りで contention が大きいので採用しない)。
/// Layout: `grad_w` row-major (num_buckets * out_dim × in_dim) — bucket-major、その中 out-major
/// (= `dense_mm_fwd_bucket` の weight layout と一致、`tid == grad_w index`)。
/// out-of-range bucket (`bucket_idx[b] < 0` 等) の position はどの bucket cell にも match
/// しないので silent skip される。
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

/// L3 weight backward (specialized: `in_dim=32`, `out_dim=1`, `num_buckets=9`)。
///
/// 元 `dense_mm_bwd_weight_bucket` は 288 cells × scan 65536 = 288 threads と並列度極小、
/// 9.2ms 占有。本 kernel は split-K + 9 bucket register accumulator で並列度を上げる:
/// - block dim = 32 (1 thread = 1 ii cell)
/// - grid = num_batch_splits (e.g., 64) → 64 blocks × 32 threads = 2048 threads ≈ 25 / SM (sm_86)
/// - 各 thread が 9 bucket × 1 ii の partial sum を batch_slice 内で集計
/// - 完了後、9 cell ぶん atomicAdd で global grad_w に flush
///
/// host 契約: grad_w は呼出前に 0 reset (accumulate semantics)。in_dim==32, out_dim==1,
/// num_buckets==9 を満たすこと。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight_bucket_tiled_l3(
    x: &[f32],
    dy: &[f32],
    bucket_idx: &[i32],
    grad_w: &[f32],
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    let tid_local = thread::threadIdx_x() as usize;
    let block_split = thread::blockIdx_x() as usize;
    let num_splits = thread::gridDim_x() as usize;
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let ii = tid_local;
    if ii >= in_dim_u {
        return;
    }

    // 各 block が均等な batch slice を担当 (端数は block 0 に寄せず ceil で配分し overflow check)。
    // ceil(batch / num_splits)、cuda-oxide は usize の `min()` / `div_ceil` で drop_in_place を
    // 出してしまうので素朴な式で書く。
    let positions_per_block = batch_u.div_ceil(num_splits);
    let b_start = block_split * positions_per_block;
    if b_start >= batch_u {
        return;
    }
    let b_end_candidate = b_start + positions_per_block;
    let b_end = if b_end_candidate < batch_u {
        b_end_candidate
    } else {
        batch_u
    };

    let mut a0 = 0.0_f32;
    let mut a1 = 0.0_f32;
    let mut a2 = 0.0_f32;
    let mut a3 = 0.0_f32;
    let mut a4 = 0.0_f32;
    let mut a5 = 0.0_f32;
    let mut a6 = 0.0_f32;
    let mut a7 = 0.0_f32;
    let mut a8 = 0.0_f32;

    let mut bb = b_start;
    while bb < b_end {
        let buc = bucket_idx[bb];
        let xv = x[bb * in_dim_u + ii];
        // out_dim=1 想定 (oi=0 のみ)。dy[bb][0] を読む。
        let dyv = dy[bb * out_dim_u];
        let mul = xv * dyv;
        if buc == 0 {
            a0 += mul;
        } else if buc == 1 {
            a1 += mul;
        } else if buc == 2 {
            a2 += mul;
        } else if buc == 3 {
            a3 += mul;
        } else if buc == 4 {
            a4 += mul;
        } else if buc == 5 {
            a5 += mul;
        } else if buc == 6 {
            a6 += mul;
        } else if buc == 7 {
            a7 += mul;
        } else if buc == 8 {
            a8 += mul;
        }
        bb += 1;
    }

    // 9 cell flush。layout は buc * (out_dim * in_dim) + oi * in_dim + ii、oi=0 なので buc * in_dim + ii。
    let num_buc_u = num_buckets as usize;
    let raw = grad_w.as_ptr();
    if num_buc_u >= 1 {
        unsafe {
            let c = &*(raw.add(ii) as *const DeviceAtomicF32);
            c.fetch_add(a0, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 2 {
        unsafe {
            let c = &*(raw.add(in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a1, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 3 {
        unsafe {
            let c = &*(raw.add(2 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a2, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 4 {
        unsafe {
            let c = &*(raw.add(3 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a3, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 5 {
        unsafe {
            let c = &*(raw.add(4 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a4, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 6 {
        unsafe {
            let c = &*(raw.add(5 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a5, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 7 {
        unsafe {
            let c = &*(raw.add(6 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a6, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 8 {
        unsafe {
            let c = &*(raw.add(7 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a7, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 9 {
        unsafe {
            let c = &*(raw.add(8 * in_dim_u + ii) as *const DeviceAtomicF32);
            c.fetch_add(a8, AtomicOrdering::Relaxed);
        }
    }
}

/// L2 weight backward (specialized: `in_dim=30`, `out_dim=32`, `num_buckets=9`)。
///
/// 元 `dense_mm_bwd_weight_bucket` は 8640 cells × scan batch、並列度 ~34 blocks で遅い。
/// 本 kernel は split-K + per-bucket register accumulator (1 thread = 1 (oi, ii) cell × 9 bucket
/// acc) で並列度を上げる。block_dim = 32 × 30 = 960 threads (sm_86 max 1024 以内)、
/// block grid = num_batch_splits。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_weight_bucket_tiled_l2(
    x: &[f32],
    dy: &[f32],
    bucket_idx: &[i32],
    grad_w: &[f32],
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    let tid_local = thread::threadIdx_x() as usize;
    let block_split = thread::blockIdx_x() as usize;
    let num_splits = thread::gridDim_x() as usize;
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    // tid → (oi, ii): oi = tid / in_dim, ii = tid % in_dim (block_dim = out_dim * in_dim)
    let oi = tid_local / in_dim_u;
    let ii = tid_local % in_dim_u;
    if oi >= out_dim_u {
        return;
    }

    let positions_per_block = batch_u.div_ceil(num_splits);
    let b_start = block_split * positions_per_block;
    if b_start >= batch_u {
        return;
    }
    let b_end_candidate = b_start + positions_per_block;
    let b_end = if b_end_candidate < batch_u {
        b_end_candidate
    } else {
        batch_u
    };

    let mut a0 = 0.0_f32;
    let mut a1 = 0.0_f32;
    let mut a2 = 0.0_f32;
    let mut a3 = 0.0_f32;
    let mut a4 = 0.0_f32;
    let mut a5 = 0.0_f32;
    let mut a6 = 0.0_f32;
    let mut a7 = 0.0_f32;
    let mut a8 = 0.0_f32;

    let mut bb = b_start;
    while bb < b_end {
        let buc = bucket_idx[bb];
        let xv = x[bb * in_dim_u + ii];
        let dyv = dy[bb * out_dim_u + oi];
        let mul = xv * dyv;
        if buc == 0 {
            a0 += mul;
        } else if buc == 1 {
            a1 += mul;
        } else if buc == 2 {
            a2 += mul;
        } else if buc == 3 {
            a3 += mul;
        } else if buc == 4 {
            a4 += mul;
        } else if buc == 5 {
            a5 += mul;
        } else if buc == 6 {
            a6 += mul;
        } else if buc == 7 {
            a7 += mul;
        } else if buc == 8 {
            a8 += mul;
        }
        bb += 1;
    }

    // grad_w layout: buc * (out_dim * in_dim) + oi * in_dim + ii。
    let per_bucket = out_dim_u * in_dim_u;
    let cell_in_bucket = oi * in_dim_u + ii;
    let num_buc_u = num_buckets as usize;
    let raw = grad_w.as_ptr();
    if num_buc_u >= 1 {
        unsafe {
            let c = &*(raw.add(cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a0, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 2 {
        unsafe {
            let c = &*(raw.add(per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a1, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 3 {
        unsafe {
            let c = &*(raw.add(2 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a2, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 4 {
        unsafe {
            let c = &*(raw.add(3 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a3, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 5 {
        unsafe {
            let c = &*(raw.add(4 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a4, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 6 {
        unsafe {
            let c = &*(raw.add(5 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a5, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 7 {
        unsafe {
            let c = &*(raw.add(6 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a6, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 8 {
        unsafe {
            let c = &*(raw.add(7 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a7, AtomicOrdering::Relaxed);
        }
    }
    if num_buc_u >= 9 {
        unsafe {
            let c = &*(raw.add(8 * per_bucket + cell_in_bucket) as *const DeviceAtomicF32);
            c.fetch_add(a8, AtomicOrdering::Relaxed);
        }
    }
}

/// Sorted layout 版 [`bias_grad_bucket`] (block-level shared-mem reduce)。caller が batch を
/// bucket で sort 済かつ各 bucket の sorted 開始 offset が `TILE_B = 16` 境界に align 済
/// (`exclusive_scan_aligned` 経由) を保証する前提。block は `padded_b * out_dim / 256` 個、
/// 1 block = 256 cells = `256 / out_dim` 行 × `out_dim` oi (L1 では 16×16、L2 では 8×32)。
/// `256 / out_dim ≤ 16` ⇒ 16-aligned sort 配下で 1 block の全 row は同一 bucket
/// (uniform-by-construction)、`bucket_idx_sorted[b_start]` で代表 bucket を取得し
/// PARTIAL[out_dim] shared-mem accumulator に集約 → 1 block × out_dim atomic add で
/// `grad_bias[block_buc][:]` に flush。global atomic 数 = blocks × out_dim
/// (L1: ~4106 × 16 = ~66K、L2: ~8213 × 32 = ~263K) で contention 大幅減。
///
/// padding 行 / 範囲外 bucket (block_buc = -1) は skip (PARTIAL flush しない)、
/// caller が `grad_bias` を 0 初期化済の前提 (accumulate semantics は元と同じ)。
///
/// 数値同等性: 加算順が sort 済 batch 順 + per-block reduce 順になるため fp32
/// associativity で baseline と bit-exact ではないが、reduction tolerance 内で一致。
/// `out_dim` は 16 / 32 を想定 (L1 bias / L2 bias)、いずれも `block_dim / out_dim ≤ 16`
/// なので 16-aligned sort 配下で 1 block の全 row は uniform-bucket。`block_dim == 256` /
/// `padded_batch % 16 == 0` / `num_buckets <= 9` / `out_dim <= 32` は caller 契約。
#[kernel]
pub fn bias_grad_bucket_shared_sorted(
    dy: &[f32],
    bucket_idx: &[i32],
    grad_bias: &[f32],
    padded_batch: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    use core::ptr::addr_of_mut;
    static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_idx = thread::blockIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let out_dim_u = out_dim as usize;
    let padded_b_u = padded_batch as usize;

    // 1 block = block_dim cells (= 16 sorted rows × out_dim oi)、b_start = block の先頭行。
    let b_start = (block_idx * block_dim_u) / out_dim_u;
    let block_buc = if b_start < padded_b_u {
        bucket_idx[b_start]
    } else {
        -1_i32
    };
    let block_buc_ok = block_buc >= 0 && (block_buc as u32) < num_buckets;
    let block_buc_u = if block_buc_ok { block_buc as usize } else { 0 };

    let partial_ptr: *mut f32 = addr_of_mut!(PARTIAL) as *mut f32;

    if tid < out_dim_u {
        unsafe {
            partial_ptr.add(tid).write(0.0_f32);
        }
    }
    thread::sync_threads();

    let global_idx = block_idx * block_dim_u + tid;
    let total = padded_b_u * out_dim_u;
    if block_buc_ok && global_idx < total {
        let oi = global_idx % out_dim_u;
        let dyv = dy[global_idx];
        let cell = unsafe { &*(partial_ptr.add(oi) as *const DeviceAtomicF32) };
        cell.fetch_add(dyv, AtomicOrdering::Relaxed);
    }
    thread::sync_threads();

    if block_buc_ok && tid < out_dim_u {
        let p = unsafe { partial_ptr.add(tid).read() };
        let cell_idx = block_buc_u * out_dim_u + tid;
        let cell = unsafe { &*(grad_bias.as_ptr().add(cell_idx) as *const DeviceAtomicF32) };
        cell.fetch_add(p, AtomicOrdering::Relaxed);
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

/// SCReLU forward — `y[i] = clip(x[i], 0, 1)²`。1 thread = 1 element。
///
/// `screlu_grad` と対の forward。host から未 launch だが cuda-oxide の bin-entry
/// constraint に従い `kernel_names` に残して compile-reach を確保する。
#[kernel]
pub fn screlu_fwd(x: &[f32], mut y: DisjointSlice<f32>, n: u32) {
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
    if let Some(out) = y.get_mut(i) {
        *out = a * a;
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
/// **Precondition: `a_dim == b_dim`** (LayerStack では両方 `l1_effective` = 15)。tid は
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

/// Broadcast bias add — `out[bi, ni] += bias[ni]` for all batch rows。
/// cuBLAS Sgemm (matmul のみ、bias 無し) の後に呼ぶ post-pass。1 thread = 1
/// (bi, ni) cell、bias は warp 内で同じ ni を共有するため L1 hit pattern が良好。
#[kernel]
pub fn bias_add_per_row(bias: &[f32], mut out: DisjointSlice<f32>, batch: u32, n: u32) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (n as usize);
    if tid.get() >= total {
        return;
    }
    let col = tid.get() % (n as usize);
    if let Some(o) = out.get_mut(tid) {
        *o += bias[col];
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

/// Simple FP16 FT activation forward (CReLU): f16 FT 出力 + f32 bias → f32 acted。
///
/// `--ft-fp16-out` 経路の融合 kernel。`sparse_ft_forward_fp16_out` の f16 出力
/// `ft_*_out_h` を直接 read (bias は別 buffer)、bias 加算と CReLU clamp を 1 pass で
/// 完了して f32 `ft_*_acted` を書く。FP32 path の `bias_add_per_row` + `crelu_fwd`
/// 2 launch を 1 launch に置き換え、`ft_*_out` (b × ft_dim) の DRAM read を f16 化
/// して帯域を半減する。
///
/// 1 thread = 1 (batch, row) cell、atomic 不要。`ft_acted` 出力は f32 のまま
/// (後続 `slice_scatter_2d` / cuBLAS Sgemm が f32 を要求)。bias は perspective 共有
/// で行内で同じ `ri` を warp 内で共有するため L1 hit pattern が良好。
#[kernel]
pub fn simple_bias_act_fwd_fp16_in_crelu(
    ft_out: &[f16],
    bias: &[f32],
    mut ft_acted: DisjointSlice<f32>,
    batch: u32,
    ft_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let ri = tid.get() % (ft_dim as usize);
    let x = ft_out[tid.get()] as f32 + bias[ri];
    #[allow(clippy::manual_clamp)]
    let y = if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    };
    if let Some(o) = ft_acted.get_mut(tid) {
        *o = y;
    }
}

/// Simple FP16 FT activation backward (CReLU) + loss scaling + ±65504 clamp + f16 cast。
///
/// `--ft-fp16-out` 経路の融合 kernel。`slice_extract_2d` が書いた `dft_*_acted`
/// (f32, b × ft_dim) を入力に、CReLU の indicator (`0 < x < 1`) を掛けて pre-activation
/// gradient を作る。pre-activation `x` は `ft_*_out_h` (f16) + `bias` (f32) から復元
/// (forward と同じく f16 read → f32 + bias)。
///
/// 結果は loss scaling 係数 `dft_scale` (= [`FT_DFT_FP16_BASE_SCALE`] × batch) を掛けて
/// f16 normal range へ持ち上げ、±65504 clamp してから f16 cast、`dft_*_out_h` へ書く。
/// 後続 [`simple_bias_grad_fp16`] / [`simple_sparse_ft_backward_fp16`] が `dft_inv_scale`
/// で打ち消す。
///
/// 1 thread = 1 (batch, row) cell、atomic 不要 (DisjointSlice f16 へ 1 cell 排他書き込み)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn simple_act_grad_to_fp16_crelu_with_scale(
    ft_out: &[f16],
    bias: &[f32],
    dft_acted: &[f32],
    mut dft_out: DisjointSlice<f16>,
    batch: u32,
    ft_dim: u32,
    dft_scale: f32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let ri = tid.get() % (ft_dim as usize);
    let x = ft_out[tid.get()] as f32 + bias[ri];
    let g = if x > 0.0_f32 && x < 1.0_f32 {
        dft_acted[tid.get()]
    } else {
        0.0_f32
    };
    let s = g * dft_scale;
    let s_c = if s > 65504.0_f32 {
        65504.0_f32
    } else if s < -65504.0_f32 {
        -65504.0_f32
    } else {
        s
    };
    if let Some(o) = dft_out.get_mut(tid) {
        *o = s_c as f16;
    }
}

/// Simple FP16 FT bias gradient: f16 dft + inv_scale → f32 grad_bias atomic add。
///
/// `--ft-fp16-out` 経路。`dft_*_out_h` (f16、loss scaling 済) を read、`dft_inv_scale`
/// で scaling を打ち消した f32 値を `grad_bias[ri]` へ atomic add。FT bias は stm / nstm
/// 共有なので 2 perspective 分の launch がそれぞれ `grad_bias` に accumulate する
/// (host は呼出前に 0 初期化)。
///
/// 1 thread = 1 (batch, row) cell。
#[kernel]
pub fn simple_bias_grad_fp16(
    dft_out: &[f16],
    grad_bias: &[f32],
    batch: u32,
    ft_dim: u32,
    dft_inv_scale: f32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let ri = tid.get() % (ft_dim as usize);
    let g = dft_out[tid.get()] as f32 * dft_inv_scale;
    // SAFETY: grad_bias[ri] は host invariant (`grad_bias.len() == ft_dim`、`ri < ft_dim`)。
    // `DeviceAtomicF32` は `f32` (align 4) と同 layout、non-atomic 経路で同 cell に書く
    // path は本 kernel / host loop に無い。
    let cell = unsafe { &*(grad_bias.as_ptr().add(ri) as *const DeviceAtomicF32) };
    cell.fetch_add(g, AtomicOrdering::Relaxed);
}

/// Simple FP16 sparse FT weight backward: f16 dft + inv_scale → f32 grad_weight atomic add。
///
/// [`sparse_ft_backward`] の f16 dft 入力版。`dft_*_out_h` (f16、loss scaling 済) を read、
/// `dft_inv_scale` で打ち消した f32 値を `grad_weight[idx*rows + ri]` へ atomic add する。
/// 既存 [`sparse_ft_backward`] と同じく 1 thread = 1 (batch, row)、column-major
/// `grad_weight`、accumulate semantics (host が呼出前に 0 初期化)。stm / nstm の 2 launch
/// で順に accumulate される。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn simple_sparse_ft_backward_fp16(
    grad_out: &[f16],
    indices: &[i32],
    grad_weight: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
    nnz: u32,
    dft_inv_scale: f32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (rows as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (rows as usize);
    let ri = tid.get() % (rows as usize);

    let g = grad_out[tid.get()] as f32 * dft_inv_scale;
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

/// Simple FT bias grad の dual variant: stm / nstm 両 perspective の dft (post-activation
/// gradient) を 1 launch で読み込み、`grad_bias[oi]` への atomic add を per-thread に 1 回
/// にまとめる kernel。1 thread = 1 (batch, ft_oi) cell、stm + nstm のローカル和を作って
/// から atomic add するため、ft_b_grad への atomic contention 数は B * ft_dim 回 (per-cell
/// 単発の bias_grad を 2 perspective 別 launch で 2 回打つ場合の半分)。
///
/// atomic add の演算は可換・結合的で、launch 順を入れ替えても per-FP32 cell の最終値は
/// 同等 (FP32 加算の非結合性で bit pattern は同一とは限らないが、CPU 参照との許容差
/// 範囲には収まる)。`grad_bias` は呼出前に host が 0 にリセット済 (`ws.ft_b_grad`)。
#[kernel]
pub fn simple_bias_grad_dual(
    dft_stm: &[f32],
    dft_nstm: &[f32],
    grad_bias: &[f32],
    batch: u32,
    ft_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let oi = tid.get() % (ft_dim as usize);
    let stm_val = dft_stm[tid.get()];
    let nstm_val = dft_nstm[tid.get()];
    let sum = stm_val + nstm_val;
    // SAFETY: `grad_bias.len() == ft_dim` を host が保証 (workspace の `ft_b_grad` は
    // ft_dim で固定)、`oi < ft_dim` は `tid % ft_dim` で保証。`f32` (align 4) と
    // `DeviceAtomicF32` (`#[repr(transparent)]` over UnsafeCell<f32>) は同 alignment。
    // 本 kernel 起動中に `grad_bias` を non-atomic 経路で書く path は無く (forward は
    // bias を READ のみ、本関数より先に走る同 step backward 段も `ft_b_grad` を書かない)、
    // atomic add 同士の競合は GPU が serialize する。
    let cell = unsafe { &*(grad_bias.as_ptr().add(oi) as *const DeviceAtomicF32) };
    cell.fetch_add(sum, AtomicOrdering::Relaxed);
}

/// Simple FT bias grad dual の FP16 入力版 (`--ft-fp16-out` 経路)。stm / nstm 両 dft
/// (`f16`、loss scaling 済) を読み、`dft_inv_scale` で打ち消した値を per-thread に 1 atomic
/// で `ft_b_grad[oi]` に accumulate。FP32 版と同じ atomic 半減効果がある。
#[kernel]
pub fn simple_bias_grad_dual_fp16(
    dft_stm: &[f16],
    dft_nstm: &[f16],
    grad_bias: &[f32],
    batch: u32,
    ft_dim: u32,
    dft_inv_scale: f32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let oi = tid.get() % (ft_dim as usize);
    let stm_val = dft_stm[tid.get()] as f32 * dft_inv_scale;
    let nstm_val = dft_nstm[tid.get()] as f32 * dft_inv_scale;
    let sum = stm_val + nstm_val;
    // SAFETY: FP32 版 `simple_bias_grad_dual` と同一の不変条件
    // (grad_bias.len() == ft_dim、oi < ft_dim、`DeviceAtomicF32` alignment 共有、
    // non-atomic 競合 path 無し、atomic add 同士のみ GPU serialize)。
    let cell = unsafe { &*(grad_bias.as_ptr().add(oi) as *const DeviceAtomicF32) };
    cell.fetch_add(sum, AtomicOrdering::Relaxed);
}

/// Simple fwd_ft_post の fused kernel (CReLU 版): `bias_add_per_row` + `crelu_fwd` +
/// `slice_scatter_2d` を 1 kernel に融合。`ft_out` に bias を in-place 加算してから (bwd
/// indicator のため post-bias 値を保持) CReLU 適用結果を直接 `combined` の per-perspective
/// slice (`dst_offset = 0` for stm / `ft_out_dim` for nstm) に書く。中間 `ft_*_acted`
/// buffer の DRAM write+read (b × ft_out × 4 byte × 2 traversal) と、`ft_*_out` の
/// bias_add → crelu 間の DRAM read+write (b × ft_out × 4 byte × 2 traversal) を消す。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn simple_ft_post_fused_crelu(
    mut ft_out: DisjointSlice<f32>,
    bias: &[f32],
    mut combined: DisjointSlice<f32>,
    batch: u32,
    ft_out_dim: u32,
    dst_offset: u32,
) {
    let tid = thread::index_1d();
    let ft_out_u = ft_out_dim as usize;
    let total = (batch as usize) * ft_out_u;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / ft_out_u;
    let oi = tid.get() % ft_out_u;
    // SAFETY: ft_out.len() == batch * ft_out_dim (caller workspace 規約)、tid.get() <
    // total で bounds、各 (bi, oi) cell は単独 writer (atomics 不要、disjoint)。
    let pre_val: f32 = unsafe {
        let cell = ft_out.get_unchecked_mut(tid.get());
        let v = *cell + bias[oi];
        *cell = v;
        v
    };
    #[allow(clippy::manual_clamp)]
    let acted = if pre_val <= 0.0_f32 {
        0.0_f32
    } else if pre_val >= 1.0_f32 {
        1.0_f32
    } else {
        pre_val
    };
    let combined_idx = bi * (2 * ft_out_u) + (dst_offset as usize) + oi;
    // SAFETY: combined.len() == batch * 2 * ft_out_dim、`dst_offset + oi < 2*ft_out_dim`
    // (caller が 0 or ft_out_dim を渡す)、bi < batch、disjoint write per (bi, oi)。
    unsafe {
        *combined.get_unchecked_mut(combined_idx) = acted;
    }
}

/// Simple fwd_ft_post の fused kernel (SCReLU 版): bias_add + SCReLU forward
/// (`y = clip(x, 0, 1) ^ 2`) + slice_scatter を融合。引数 / DRAM saving は
/// [`simple_ft_post_fused_crelu`] と同型。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn simple_ft_post_fused_screlu(
    mut ft_out: DisjointSlice<f32>,
    bias: &[f32],
    mut combined: DisjointSlice<f32>,
    batch: u32,
    ft_out_dim: u32,
    dst_offset: u32,
) {
    let tid = thread::index_1d();
    let ft_out_u = ft_out_dim as usize;
    let total = (batch as usize) * ft_out_u;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / ft_out_u;
    let oi = tid.get() % ft_out_u;
    // SAFETY: 同 [`simple_ft_post_fused_crelu`]。
    let pre_val: f32 = unsafe {
        let cell = ft_out.get_unchecked_mut(tid.get());
        let v = *cell + bias[oi];
        *cell = v;
        v
    };
    #[allow(clippy::manual_clamp)]
    let a = if pre_val < 0.0_f32 {
        0.0_f32
    } else if pre_val > 1.0_f32 {
        1.0_f32
    } else {
        pre_val
    };
    let acted = a * a;
    let combined_idx = bi * (2 * ft_out_u) + (dst_offset as usize) + oi;
    // SAFETY: 同 [`simple_ft_post_fused_crelu`]。
    unsafe {
        *combined.get_unchecked_mut(combined_idx) = acted;
    }
}

/// Simple bwd_ft_act の fused kernel (CReLU 版): `slice_extract_2d` で `dcombined`
/// の per-perspective 半分を切り出して読み取り、`ft_pre_act` (pre-activation FT 出力)
/// で CReLU 指示関数 `0 < x < 1` を作って `dft_out` に直接書く。元の
/// `slice_extract_2d` → `crelu_grad` の 2 kernel + 中間 `dft_*_acted` buffer の
/// DRAM round-trip (b × ft_out × 4 byte の write+read) を 1 kernel + write-only に縮める。
///
/// `src_offset` で stm (= 0) / nstm (= ft_out) を選択する。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn simple_bwd_ft_act_crelu_fused(
    dcombined: &[f32],
    ft_pre_act: &[f32],
    dft_out: &[f32],
    batch: u32,
    ft_out: u32,
    src_offset: u32,
) {
    let tid = thread::index_1d();
    let ft_out_u = ft_out as usize;
    let total = (batch as usize) * ft_out_u;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / ft_out_u;
    let oi = tid.get() % ft_out_u;
    let l1_in = 2 * ft_out_u;
    let dcomb_idx = bi * l1_in + (src_offset as usize) + oi;
    let dft_acted = dcombined[dcomb_idx];
    let xi = ft_pre_act[tid.get()];
    let g = if xi > 0.0_f32 && xi < 1.0_f32 {
        dft_acted
    } else {
        0.0_f32
    };
    // SAFETY: dft_out.len() == batch * ft_out (caller workspace 規約)、tid.get() < total
    // で bounds、各 tid は disjoint (bi, oi) cell に単独 writer、atomics 不要。
    unsafe {
        let p = dft_out.as_ptr().add(tid.get()) as *mut f32;
        p.write(g);
    }
}

/// Simple bwd_ft_act の fused kernel (SCReLU 版): `slice_extract_2d` + SCReLU grad
/// (`clip(x, 0, 1)` の derivative `2 * a` を `0 < a < 1` の indicator で gate) を融合。
/// 引数 / DRAM saving は [`simple_bwd_ft_act_crelu_fused`] と同型。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn simple_bwd_ft_act_screlu_fused(
    dcombined: &[f32],
    ft_pre_act: &[f32],
    dft_out: &[f32],
    batch: u32,
    ft_out: u32,
    src_offset: u32,
) {
    let tid = thread::index_1d();
    let ft_out_u = ft_out as usize;
    let total = (batch as usize) * ft_out_u;
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / ft_out_u;
    let oi = tid.get() % ft_out_u;
    let l1_in = 2 * ft_out_u;
    let dcomb_idx = bi * l1_in + (src_offset as usize) + oi;
    let dft_acted = dcombined[dcomb_idx];
    let xi = ft_pre_act[tid.get()];
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
    let g = dft_acted * dydx;
    // SAFETY: 同 [`simple_bwd_ft_act_crelu_fused`] と同一不変条件。
    unsafe {
        let p = dft_out.as_ptr().add(tid.get()) as *mut f32;
        p.write(g);
    }
}

// ===========================================================================
// Host driver helpers (kernel module loader / launch utilities)
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
/// CudaModule を load。fallback 順は `.ll` → `.cubin` → `.ptx`。
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

/// `.ll` を libdevice と link → opt → llc で `.ptx` 生成。`kernel_names` で全 27
/// kernel を internalize する。
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

    // module が launch する全 kernel 名。`@<name>` として `.ll` に出ているものを
    // 漏れなく internalize-public-api-list に残す (1 個でも漏れると opt の globaldce
    // で消える)。
    let kernel_names = "sparse_ft_forward,sparse_ft_backward,loss_wdl,loss_wrm,screlu_grad,\
                       adamw_step,radam_step,radam_step_fp16_mirror,\
                       radam_step_f16state,radam_step_f16state_mirror,\
                       ranger_lookahead_lerp,ranger_lookahead_lerp_fp16_mirror,\
                       ft_post_perspective_fwd,ft_post_perspective_grad,\
                       dense_mm_fwd,dense_mm_bwd_input,dense_mm_bwd_weight,bias_grad,\
                       dense_mm_fwd_bucket,dense_mm_bwd_input_bucket,\
                       dense_mm_bwd_weight_bucket,bias_grad_bucket,\
                       crelu_fwd,crelu_grad,screlu_fwd,abs_pow2_scale_fwd,abs_pow2_scale_grad,\
                       concat_l1sqr_main_fwd,concat_l1sqr_main_grad,elementwise_add,\
                       bias_add_per_row,\
                       slice_extract_2d,slice_scatter_2d,\
                       count_buckets,exclusive_scan_aligned,scatter_bucket_perm,\
                       permute_rows_f32,inverse_permute_rows_f32,\
                       dense_mm_fwd_bucket_tiled_l1_sorted,\
                       dense_mm_bwd_weight_bucket_tiled_l1_sorted,\
                       bias_grad_bucket_shared_sorted,\
                       ft_post_perspective_grad_fused,\
                       sparse_ft_forward_fp16,sparse_ft_forward_fp16_out,cast_f32_to_f16,\
                       ft_post_perspective_fwd_fp16,ft_post_perspective_grad_fused_fp16,\
                       build_feature_counts,exclusive_prefix_sum_small,scatter_positions,\
                       gather_and_sum_per_feature_overwrite,gather_and_sum_per_feature_add,\
                       gather_and_sum_per_feature_overwrite_fp16,\
                       gather_and_sum_per_feature_add_fp16,\
                       simple_bias_act_fwd_fp16_in_crelu,\
                       simple_act_grad_to_fp16_crelu_with_scale,\
                       simple_bias_grad_fp16,simple_sparse_ft_backward_fp16,\
                       simple_bias_grad_dual,simple_bias_grad_dual_fp16,\
                       simple_bwd_ft_act_crelu_fused,simple_bwd_ft_act_screlu_fused,\
                       simple_ft_post_fused_crelu,simple_ft_post_fused_screlu";

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
fn run_or_err(bin: &str, args: &[&std::ffi::OsStr]) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| {
            format!(
                "failed to spawn {bin}: {e}. \
                 .ll→.ptx 変換は LLVM 21+ の llvm-link / opt / llc を使う \
                 (libNVVM が opaque pointer IR を parse できないため llc 経路)。\
                 -22 / -21 を自動探索するが、見つからなければ \
                 LLVM_LINK_BIN / OPT_BIN / LLC_BIN env で明示指定する。"
            )
        })?;
    if !status.success() {
        return Err(format!("{bin} failed with status {status}").into());
    }
    Ok(())
}

/// `libdevice.10.bc` を CUDA Toolkit から探す。
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
// LayerStack architecture constants
// ===========================================================================
//
// FT input dim (`ft_in`) and active-feature count (`max_active`) are NOT
// constants: they depend on the input feature set chosen at startup (see
// `FeatureSetSpec`). They are carried as runtime fields on `GpuWorkspace` /
// `GpuTrainer`. The values below describe the LayerStack topology after the
// FT layer and are feature-set independent.

const FT_OUT: usize = 1536; // per-perspective FT output dim
const COMBINED_DIM: usize = FT_OUT; // pairwise (1536 → 768) × 2 perspectives concat = 1536
const L1_OUT: usize = 16;
const L1_EFFECTIVE: usize = L1_OUT - 1; // = 15 (skip 1 dim、bullet:1433)
const L1_SKIP: usize = L1_OUT - L1_EFFECTIVE; // = 1
const L2_IN: usize = L1_EFFECTIVE * 2; // = 30 (l1_sqr.concat(l1_main))、bullet:1434
const L2_OUT: usize = 32;
const NUM_BUCKETS: usize = 9; // progress8kpabs

// ===========================================================================
// raw checkpoint format (`--resume` 用)
// ===========================================================================

/// raw checkpoint format magic (`b"RNRC"` = "RShogi Nnue Resume Checkpoint")。
/// `crates/nnue-train::optimizer` の `b"RNGR"` (RangerHostState single-file format) とは
/// 別物 — こちらは weight group raw f32 + Ranger state + step + superbatch を 1 file に
/// まとめた self-contained format (`RNGR` は optimizer state だけ、weight は持たない)。
const RAW_CKPT_MAGIC: [u8; 4] = *b"RNRC";

/// raw checkpoint format version。
///
/// - `1`: no feature-set header; the weights are always `halfka-hm-merged`.
/// - `2`: a self-describing feature-set header (canonical name + `ft_in` +
///   `ft_out` + `max_active`) follows the magic + version fields.
/// - `3`: a producer run id (length-prefixed UTF-8 — the experiment.json `id`
///   of the run that wrote the checkpoint) follows the feature-set header.
///   `--resume` reads it to fill `lineage.parent_id`.
/// - `4`: an arch-kind name (length-prefixed UTF-8) and a topology header (a
///   count-prefixed list of `u64` layer dimensions) follow the producer run
///   id. They pin which architecture and layer shape the checkpoint belongs
///   to, so a checkpoint written by one architecture cannot be resumed by
///   another.
///
/// `load_raw_checkpoint` accepts versions 1..=4. Version 1 is interpreted as
/// `halfka-hm-merged`; versions 1..=3 predate the arch-kind header and are
/// interpreted as `layerstack`. Versions above 4 are rejected. The producer
/// run id is absent (`None`) for versions 1 and 2.
const RAW_CKPT_VERSION: u32 = 4;

/// `*.ckpt` の producer run id のバイト数上限。run id は `{net_id}-{時刻}-{pid}`
/// 程度で高々数十バイト。破損 file の巨大な length 値で過大確保しないための上限。
const MAX_RUN_ID_BYTES: usize = 256;

/// raw checkpoint 1 group 分の host buffer (`w`, `m`, `v`, `slow` の f32 Vec、`grad` は含めない)。
type RawCkptGroup = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// `SimpleGpuTrainer::raw_ckpt_groups` の 1 要素。weight name + element count + 各
/// `(weight, m, v, slow)` device buffer の借用 tuple。
type SimpleRawCkptGroupEntry<'a> = (
    &'static str,
    usize,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
);

/// LayerStack アーキの topology header (v4+): FT 出力次元・L1 出力次元・L2 出力次元・
/// bucket 数。`load_raw_checkpoint` がこの並びを checkpoint と照合する。
const LAYERSTACK_TOPOLOGY: [u64; 4] = [
    FT_OUT as u64,
    L1_OUT as u64,
    L2_OUT as u64,
    NUM_BUCKETS as u64,
];

/// raw checkpoint header の arch identity 部 (write / read 双方の引数)。feature
/// set・arch 種別・FT 出力次元・topology 次元列をまとめて持つ。
struct RawCkptArch<'a> {
    /// 入力 feature set (canonical 名 / `ft_in` / `max_active` の源)。
    feature_set: FeatureSetSpec,
    /// network アーキ種別。
    arch_kind: ArchKind,
    /// FT 出力次元 (feature set header の `ft_out` 欄に書く値)。
    ft_out: u64,
    /// arch 固有の層次元列 (v4 topology header)。
    topology: &'a [u64],
}

/// `read_raw_ckpt_header` が返す raw checkpoint header の解析結果。
#[derive(Debug)]
struct RawCkptHeader {
    /// この checkpoint が表す完了 superbatch 番号。
    superbatch: usize,
    /// Ranger lookahead step counter。
    step_count: u64,
    /// format 記載の weight group 数 (caller が arch 期待値と照合する)。
    num_groups: u64,
    /// producer run の experiment.json id (version 3+ かつ記録ありなら `Some`)。
    producer_run_id: Option<String>,
}

/// raw checkpoint の header (magic 〜 num_groups、group 本体の手前まで) を書く。
/// 常に最新 [`RAW_CKPT_VERSION`] で書き出す。
fn write_raw_ckpt_header<W: Write>(
    w: &mut W,
    arch: &RawCkptArch,
    run_id: &str,
    superbatch: u64,
    step_count: u64,
    num_groups: u64,
) -> std::io::Result<()> {
    w.write_all(&RAW_CKPT_MAGIC)?;
    w.write_all(&RAW_CKPT_VERSION.to_le_bytes())?;
    // feature set header (v2+): canonical 名 + 次元 3 値。
    let fs_name = arch.feature_set.canonical_name();
    w.write_all(&(fs_name.len() as u32).to_le_bytes())?;
    w.write_all(fs_name.as_bytes())?;
    w.write_all(&(arch.feature_set.ft_in() as u64).to_le_bytes())?;
    w.write_all(&arch.ft_out.to_le_bytes())?;
    w.write_all(&(arch.feature_set.max_active() as u64).to_le_bytes())?;
    // producer run id (v3+)。
    w.write_all(&(run_id.len() as u32).to_le_bytes())?;
    w.write_all(run_id.as_bytes())?;
    // arch_kind + topology header (v4+)。
    let arch_name = arch.arch_kind.canonical_name();
    w.write_all(&(arch_name.len() as u32).to_le_bytes())?;
    w.write_all(arch_name.as_bytes())?;
    w.write_all(&(arch.topology.len() as u64).to_le_bytes())?;
    for &dim in arch.topology {
        w.write_all(&dim.to_le_bytes())?;
    }
    w.write_all(&superbatch.to_le_bytes())?;
    w.write_all(&step_count.to_le_bytes())?;
    w.write_all(&num_groups.to_le_bytes())?;
    Ok(())
}

/// raw checkpoint の header を読み、`expected` の arch identity と照合する。
/// version 1..=4 を受理し、不一致 / 破損は `InvalidData` で reject する。
///
/// version 1..=3 は arch-kind header を持たず暗黙に `layerstack`。version 4 は
/// arch_kind 名と topology 次元列を `expected` と照合する。
fn read_raw_ckpt_header<R: std::io::Read>(
    r: &mut R,
    expected: &RawCkptArch,
) -> Result<RawCkptHeader, Box<dyn std::error::Error>> {
    let mut magic = [0u8; 4];
    read_exact_or_invalid(r, &mut magic, "magic")?;
    if magic != RAW_CKPT_MAGIC {
        return Err(invalid_data(format!(
            "raw checkpoint magic mismatch: got {magic:?}, want {RAW_CKPT_MAGIC:?}"
        )));
    }
    let mut buf4 = [0u8; 4];
    read_exact_or_invalid(r, &mut buf4, "version")?;
    let version = u32::from_le_bytes(buf4);
    if version == 0 || version > RAW_CKPT_VERSION {
        return Err(invalid_data(format!(
            "raw checkpoint version {version} is not supported \
             (this build reads 1..={RAW_CKPT_VERSION})"
        )));
    }
    let mut buf8 = [0u8; 8];
    let want_name = expected.feature_set.canonical_name();

    // feature set header は version 2+。version 1 は header 無しで halfka-hm-merged 固定。
    if version >= 2 {
        read_exact_or_invalid(r, &mut buf4, "feature set name length")?;
        let fs_name_len = u32::from_le_bytes(buf4) as usize;
        if fs_name_len > 256 {
            return Err(invalid_data(format!(
                "raw checkpoint feature set name length {fs_name_len} is implausible (max 256)"
            )));
        }
        let mut fs_name_bytes = vec![0u8; fs_name_len];
        read_exact_or_invalid(r, &mut fs_name_bytes, "feature set name")?;
        let fs_name = String::from_utf8(fs_name_bytes).map_err(|_| {
            invalid_data("raw checkpoint feature set name is not valid UTF-8".to_string())
        })?;
        read_exact_or_invalid(r, &mut buf8, "ft_in")?;
        let ckpt_ft_in = u64::from_le_bytes(buf8);
        read_exact_or_invalid(r, &mut buf8, "ft_out")?;
        let ckpt_ft_out = u64::from_le_bytes(buf8);
        read_exact_or_invalid(r, &mut buf8, "max_active")?;
        let ckpt_max_active = u64::from_le_bytes(buf8);

        let want = expected.feature_set;
        if fs_name != want_name {
            return Err(invalid_data(format!(
                "raw checkpoint feature set mismatch: checkpoint is '{fs_name}', \
                 requested '{want_name}' (feature set を跨いだ resume は不可)"
            )));
        }
        if ckpt_ft_in != want.ft_in() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint ft_in mismatch: got {ckpt_ft_in}, want {}",
                want.ft_in()
            )));
        }
        if ckpt_ft_out != expected.ft_out {
            return Err(invalid_data(format!(
                "raw checkpoint ft_out mismatch: got {ckpt_ft_out}, want {}",
                expected.ft_out
            )));
        }
        if ckpt_max_active != want.max_active() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint max_active mismatch: got {ckpt_max_active}, want {}",
                want.max_active()
            )));
        }
    } else if want_name != FeatureSet::HalfKaHmMerged.spec().canonical_name() {
        return Err(invalid_data(format!(
            "raw checkpoint version 1 is always 'halfka-hm-merged', \
             requested '{want_name}' (feature set を跨いだ resume は不可)"
        )));
    }

    // producer run id は version 3+。長さ 0 も「未記録」扱いで `None`。
    let producer_run_id: Option<String> = if version >= 3 {
        read_exact_or_invalid(r, &mut buf4, "producer run id length")?;
        let run_id_len = u32::from_le_bytes(buf4) as usize;
        if run_id_len > MAX_RUN_ID_BYTES {
            return Err(invalid_data(format!(
                "raw checkpoint producer run id length {run_id_len} is implausible \
                 (max {MAX_RUN_ID_BYTES})"
            )));
        }
        if run_id_len == 0 {
            None
        } else {
            let mut run_id_bytes = vec![0u8; run_id_len];
            read_exact_or_invalid(r, &mut run_id_bytes, "producer run id")?;
            Some(String::from_utf8(run_id_bytes).map_err(|_| {
                invalid_data("raw checkpoint producer run id is not valid UTF-8".to_string())
            })?)
        }
    } else {
        None
    };

    // arch_kind + topology header は version 4+。version 1..=3 は arch-kind header
    // を持たず、Simple アーキが存在しなかった時代の checkpoint なので暗黙に layerstack。
    if version >= 4 {
        read_exact_or_invalid(r, &mut buf4, "arch kind name length")?;
        let arch_name_len = u32::from_le_bytes(buf4) as usize;
        if arch_name_len > 256 {
            return Err(invalid_data(format!(
                "raw checkpoint arch kind name length {arch_name_len} is implausible (max 256)"
            )));
        }
        let mut arch_name_bytes = vec![0u8; arch_name_len];
        read_exact_or_invalid(r, &mut arch_name_bytes, "arch kind name")?;
        let arch_name = String::from_utf8(arch_name_bytes).map_err(|_| {
            invalid_data("raw checkpoint arch kind name is not valid UTF-8".to_string())
        })?;
        let ckpt_arch = ArchKind::from_canonical_name(&arch_name).ok_or_else(|| {
            invalid_data(format!(
                "raw checkpoint has unknown arch kind '{arch_name}'"
            ))
        })?;
        if ckpt_arch != expected.arch_kind {
            return Err(invalid_data(format!(
                "raw checkpoint arch kind mismatch: checkpoint is '{}', requested '{}' \
                 (アーキを跨いだ resume は不可)",
                ckpt_arch.canonical_name(),
                expected.arch_kind.canonical_name()
            )));
        }
        read_exact_or_invalid(r, &mut buf8, "topology dim count")?;
        let topo_count = u64::from_le_bytes(buf8);
        if topo_count != expected.topology.len() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint topology dim count {topo_count} != expected {}",
                expected.topology.len()
            )));
        }
        for (i, &want_dim) in expected.topology.iter().enumerate() {
            read_exact_or_invalid(r, &mut buf8, "topology dim")?;
            let got = u64::from_le_bytes(buf8);
            if got != want_dim {
                return Err(invalid_data(format!(
                    "raw checkpoint topology dim {i} mismatch: got {got}, want {want_dim} \
                     (network architecture mismatch)"
                )));
            }
        }
    } else if expected.arch_kind != ArchKind::LayerStack {
        return Err(invalid_data(format!(
            "raw checkpoint version {version} predates the arch-kind header and is \
             always 'layerstack', requested '{}' (アーキを跨いだ resume は不可)",
            expected.arch_kind.canonical_name()
        )));
    }

    read_exact_or_invalid(r, &mut buf8, "superbatch")?;
    let superbatch_u64 = u64::from_le_bytes(buf8);
    let superbatch: usize = superbatch_u64.try_into().map_err(|_| {
        invalid_data(format!(
            "raw checkpoint superbatch {superbatch_u64} exceeds usize::MAX"
        ))
    })?;
    read_exact_or_invalid(r, &mut buf8, "step_count")?;
    let step_count = u64::from_le_bytes(buf8);
    read_exact_or_invalid(r, &mut buf8, "num_groups")?;
    let num_groups = u64::from_le_bytes(buf8);

    Ok(RawCkptHeader {
        superbatch,
        step_count,
        num_groups,
        producer_run_id,
    })
}

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

/// `--ft-fp16-out` で dft (FT activation gradient) を `f16` 化するときの loss scaling
/// 基準係数。実際に使う scale は **`FT_DFT_FP16_BASE_SCALE * batch`** (caller が batch を
/// 掛ける)。
///
/// dft は batch 正規化 (loss が `1/batch`) のため値が `1/batch` に比例し、無 scale だと
/// 全要素が f16 subnormal 下限 (2^-24 ≈ 6e-8) を下回り 0 に潰れて勾配が消える。`f16` へ
/// 書く前に scale を掛けて normal range に持ち上げ、gather 側で逆数を掛けて戻す。
///
/// scale を `batch` 比例にするのは、dft ∝ `1/batch` なので `dft * (BASE * batch)` が
/// batch に依らず一定 (`dft * batch` の不変量 × BASE) になり、どの `--batch-size` でも
/// 同じ f16 域に載るため。固定 scale だと小 batch で dft が大きくなり f16 max (65504) を
/// 超えて inf 化し `ft_w_grad` を破壊する。
///
/// 2^14: `dft * batch` の不変量は学習初期で ~1.2e-3、`× BASE ≈ 19` と f16 normal
/// range に収まる。ただし dft は学習が進むと縮まず成長し、scale 後の値が学習中盤で
/// f16 上限 (65504) に達しうるため、`ft_post_perspective_grad_fused_fp16` は f16
/// 書き込み前に `±65504` へ clamp する (overflow → `±inf` → 学習発散を防ぐ)。
/// batch=65536 のとき実 scale は `2^14 * 2^16 = 2^30` で power-of-2 (scale 自体は無誤差)。
const FT_DFT_FP16_BASE_SCALE: f32 = (1_u32 << 14) as f32;

/// `--fp16-opt-state` で `ft_w` の Ranger 1st moment (`m`) を `f16` 格納するときの
/// scale。`m` を `f16` へ書く前に掛け、読み戻し時に割る ([`radam_step_f16state`])。
///
/// `m` は batch 正規化された勾配の EMA で、実測 (1000 step 時点の `ft_w` checkpoint)
/// で `|m|` は p5 ~3e-13・中央値 ~3e-9・最大 ~1e-5。無 scale だと大半が `f16` の
/// subnormal 下限 (2^-24 ≈ 6e-8) 以下に潰れる。`2^28` を掛けると中央値が `f16` の
/// normal range 内 (~0.8)、最大値も `1e-5 * 2^28 ≈ 2.7e3 ≪ 65504` で overflow せず
/// (学習初期の勾配増大に ~24× の余裕)。scale は power-of-2 で scale 自体は無誤差。
const FT_OPT_M_SCALE: f32 = (1_u32 << 28) as f32;

/// `--fp16-opt-state` で `ft_w` の Ranger 2nd moment (`v`) を `f16` 格納するときの
/// scale。`v` を `f16` へ書く前に掛け、読み戻し時に割る ([`radam_step_f16state`])。
///
/// `v` は勾配二乗の EMA で `m` よりさらに小さく、実測で中央値 ~2e-15・最大 ~2e-9。
/// `m` と別 scale なのは値域が約 `m^2` のオーダーで異なるため。`2^40` を掛けると
/// 中央値が `f16` normal range 内、最大値も `2e-9 * 2^40 ≈ 2.2e3 ≪ 65504` で
/// overflow せず (初期勾配増大に ~30× の余裕)。`v >= 0` なので格納時の clamp は
/// 上側のみ。scale は power-of-2。
const FT_OPT_V_SCALE: f32 = (1_u64 << 40) as f32;

// Ranger optimizer params。bullet `RangerParams::default()` 由来の値は
// `nnue_train::optimizer::RangerParams::DEFAULT` を single source of truth として参照する。
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

// smoke 用 loss params (scale=290, wdl=0.0、wrm in_scaling 340 / nnue2score 600 /
// target offset 270 / target scaling 380)。
// trainer 経路では CLI から `LossKind` を組み立てるのでここは smoke 専用。
const WDL_LAMBDA: f32 = 0.0;
/// smoke で使う固定 batch position 数 (`GpuTrainer::new` の workspace 初期 batch にも使う)。
const SMOKE_BATCH: usize = 4;
const SMOKE_LOSS_SIGMOID: LossKind = LossKind::Sigmoid { scale: 1.0 / 290.0 };
const SMOKE_LOSS_WRM: LossKind = LossKind::Wrm {
    nnue2score: 600.0,
    in_scaling: 340.0,
    target_offset: 270.0,
    target_scaling: 380.0,
};

/// `ft_w` の Ranger moment (`m` / `v`) buffer。既定は `f32`、`--fp16-opt-state` で
/// `f16` (格納時 scale 付き、[`radam_step_f16state`])。`ft_w` は 112.6M 要素で
/// optimizer phase の DRAM traffic を占めるため `f16` 化の効果がある一方、他 9 group
/// の moment は小さく `f16` 化の意味が無いので `f32` (`DeviceBuffer<f32>`) のまま。
enum MomentBuf {
    F32(DeviceBuffer<f32>),
    F16(DeviceBuffer<f16>),
}

impl MomentBuf {
    /// 要素数 `n` の 0 初期化 moment buffer。`fp16` で `f16` / `f32` を選ぶ。
    fn zeroed(
        stream: &CudaStream,
        n: usize,
        fp16: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if fp16 {
            Ok(MomentBuf::F16(DeviceBuffer::<f16>::zeroed(stream, n)?))
        } else {
            Ok(MomentBuf::F32(DeviceBuffer::<f32>::zeroed(stream, n)?))
        }
    }

    /// device → host download し **真値の `f32`** で返す (raw checkpoint 用)。`f16`
    /// variant は格納値が `scale` 倍されているので割り戻す (`f32` variant は scale
    /// 無関係でそのまま)。
    fn to_host_f32(
        &self,
        stream: &CudaStream,
        scale: f32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match self {
            MomentBuf::F32(b) => Ok(b.to_host_vec(stream)?),
            MomentBuf::F16(b) => {
                let inv = 1.0_f32 / scale;
                Ok(b.to_host_vec(stream)?
                    .into_iter()
                    .map(|x| (x as f32) * inv)
                    .collect())
            }
        }
    }

    /// **真値の `f32`** slice から moment buffer を作る (raw checkpoint resume 用)。
    /// `fp16` で variant を選ぶ。`f16` variant は `scale` を掛けてから半精度化し、
    /// `f16` 有限域 (`|x| <= 65504`) を超える値は clamp する (`f32` で書かれた
    /// checkpoint を `--fp16-opt-state` で resume したとき外れ値が `inf` 化して以降の
    /// step を壊すのを防ぐ。[`radam_step_f16state`] の格納時 clamp と同方針)。
    fn from_host_f32(
        stream: &CudaStream,
        data: &[f32],
        fp16: bool,
        scale: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if fp16 {
            let h: Vec<f16> = data
                .iter()
                .map(|&x| (x * scale).clamp(-65504.0, 65504.0) as f16)
                .collect();
            Ok(MomentBuf::F16(DeviceBuffer::from_host(stream, &h)?))
        } else {
            Ok(MomentBuf::F32(DeviceBuffer::from_host(stream, data)?))
        }
    }
}

// ===========================================================================
// GpuTrainer (LayerStack 1536-16-32 + progress8kpabs 9 buckets)
//
// 10 weight groups × {w, m, v, slow, grad} = 50 device buffers + loss_acc + step_count。
// Forward は 15 kernel launch、backward は ~16 kernel launch、optimizer は 10×{radam+lerp}。
// ===========================================================================

#[allow(dead_code)] // 一部 field は host state 直接更新時のみ使う
struct GpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,

    // FT (single, shared across perspectives)
    ft_w: DeviceBuffer<f32>,
    /// Ranger 1st/2nd moment。既定 `f32`、`--fp16-opt-state` で `f16` ([`MomentBuf`])。
    ft_w_m: MomentBuf,
    ft_w_v: MomentBuf,
    ft_w_slow: DeviceBuffer<f32>,
    ft_w_grad: DeviceBuffer<f32>,
    /// `ft_w` の FP16 mirror。`ft_fp16` が true のときだけ確保され、毎 step `ft_w`
    /// (FP32 master) から変換される。`sparse_ft_forward_fp16` の weight 入力。
    ft_w_h: Option<DeviceBuffer<f16>>,
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

    // 中間 activation / activation-grad の永続 workspace (batch_size 固定前提で `new`
    // 時に確保。`step_impl` が requires より大きい batch を渡したら拡張)。
    ws: GpuWorkspace,

    // loss + step
    loss_acc: DeviceBuffer<f64>,
    /// step() 末の `loss_acc` 同期読みを async + 1-step lag に置換する pinned host ring。
    /// host が `stream.synchronize` を待たずに次 batch の launch を発行できるようになる。
    loss_ring: AsyncLossRing,
    /// step 先頭の入力 H2D を専用 copy stream で直前 step の compute と overlap させる ring。
    input_ring: InputUploadRing,
    /// L1f weight backward の `dense_mm_bwd_weight_tiled` を `cublasSgemm_v2` に置換するための
    /// cuBLAS handle。stream は `self.stream` に bind 済 (cuBLAS の launch は same-stream で
    /// in-order に走る)。
    cublas: CublasHandle,
    /// true なら forward の `sparse_ft_forward` を FP16 weight 版に切替える
    /// (`--ft-fp16`)。false で従来の FP32 path と bit-identical。
    ft_fp16: bool,
    /// true なら FT activation (`ft_*_out` forward 出力 / `dft_*_out` backward 勾配) も
    /// FP16 で保持する (`--ft-fp16-out`)。`ft_fp16` が true のときのみ true になりうる。
    ft_fp16_out: bool,
    /// true なら `ft_w` の Ranger moment (`m` / `v`) を `f16` で保持する
    /// (`--fp16-opt-state`)。`ft_w_m` / `ft_w_v` が [`MomentBuf::F16`] になり、optimizer
    /// step は [`radam_step_f16state`] 系を使う。false で従来の `f32` path。
    fp16_opt_state: bool,
    /// 入力 feature set spec。FT 入力次元 (`ft_in`) / active feature 数
    /// (`max_active`) / artifact identity の単一の真実源。起動時に
    /// `--feature-set` から一度だけ決まり、以降不変。
    feature_set: FeatureSetSpec,
    /// Ranger optimizer の weight decay 係数。各 weight group の `radam_step`
    /// に一律 `decay` 引数として渡す。`--weight-decay` から起動時に決まり、
    /// 以降不変。既定 0.0 で decay 無し。
    weight_decay: f32,
    step_count: u64,
}

impl Drop for GpuTrainer {
    fn drop(&mut self) {
        // 残り queue 済 GPU 操作 (`loss_ring` の async D2H が `loss_acc` を read する、
        // `input_ring` の copy stream H2D が `ws` の input buffer を write する等) を
        // 両 stream で完了させてから field の Drop に進む。さもなければ struct field
        // 宣言順で device memory が先に `cuMemFree` され、in-flight な copy が解放済
        // メモリに触れる race になる。両 sync 後は後続 per-field cleanup が全部 safe。
        // 失敗は無視 (Drop 中の error 報告は実用上困難、stream 破棄で driver が
        // tracking を解除する debug-build 動作と等価)。
        let _ = self.stream.synchronize();
        let _ = self.input_ring.copy_stream.synchronize();
    }
}

/// `GpuTrainer::step_impl` の forward / backward で使う中間 activation と
/// activation-gradient buffer を **1 step ごとに再 alloc せず永続化** するための
/// workspace。
///
/// 各 buffer は `len_batch` 個の position 分のサイズで `GpuTrainer::new` 時に一度だけ
/// 確保する。固定 batch 前提で、`step_impl` は [`GpuWorkspace::check_batch_capacity`]
/// で batch が `len_batch` に収まることを検証する (実 dataloader は `batch_size` 以下の
/// batch しか出さない。step 中の再 alloc は in-flight な compute の device memory を
/// 解放する race になるため行わない)。`len_batch == 0` は「まだ未確保」を表す番兵
/// (実際には `GpuTrainer::new` で `batch_size` 分を確保するので step 時には常に > 0)。
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

    /// FT 入力次元 (feature set ごとに異なる)。inverse-index scratch
    /// (`feat_*`) と FT forward/backward kernel の launch arg に使う。
    ft_in: usize,
    /// 1 perspective あたりの active feature 数 (feature set ごとに異なる)。
    /// 入力 index buffer (`stm_idx_dev` 等) の容量と FT kernel の launch arg。
    max_active: usize,

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
    dft_stm_out: DeviceBuffer<f32>,          // b × FT_OUT
    dft_nstm_out: DeviceBuffer<f32>,         // b × FT_OUT

    // FT activation の FP16 版。`ft_fp16_out` (`--ft-fp16-out`) が true のときだけ
    // b × FT_OUT で確保され、`ft_*_out` / `dft_*_out` (f32) の代わりに使われる
    // (f32 版はそのとき placeholder size でしか確保しない)。false なら全て `None`。
    ft_stm_out_h: Option<DeviceBuffer<f16>>,   // b × FT_OUT
    ft_nstm_out_h: Option<DeviceBuffer<f16>>,  // b × FT_OUT
    dft_stm_out_h: Option<DeviceBuffer<f16>>,  // b × FT_OUT
    dft_nstm_out_h: Option<DeviceBuffer<f16>>, // b × FT_OUT

    // -- inverse-index sparse_ft_backward scratch (sized by feature set) --
    feat_counts: DeviceBuffer<u32>, // ft_in: per-feature histogram (atomic build)
    feat_offsets: DeviceBuffer<u32>, // ft_in + 1: exclusive prefix sum
    feat_write_ctr: DeviceBuffer<u32>, // ft_in: scatter atomic counter
    feat_positions: DeviceBuffer<u32>, // up to batch * max_active: sorted positions

    // -- pre-allocated input buffers (per-step `from_host` の cudaMalloc/Free を排除) --
    // `*_dev` が現 step の active、`*_dev_back` が double-buffer の back。`step_impl` が
    // 毎 step `mem::swap` し、直前 step が読んでいない back 側へ次 step 入力を copy
    // stream で先行 H2D する ([`InputUploadRing`])。
    stm_idx_dev: DeviceBuffer<i32>,         // batch * max_active
    nstm_idx_dev: DeviceBuffer<i32>,        // batch * max_active
    bucket_idx_dev: DeviceBuffer<i32>,      // batch
    score_dev: DeviceBuffer<f32>,           // batch
    wdl_dev: DeviceBuffer<f32>,             // batch
    stm_idx_dev_back: DeviceBuffer<i32>,    // batch * max_active
    nstm_idx_dev_back: DeviceBuffer<i32>,   // batch * max_active
    bucket_idx_dev_back: DeviceBuffer<i32>, // batch
    score_dev_back: DeviceBuffer<f32>,      // batch
    wdl_dev_back: DeviceBuffer<f32>,        // batch

    // -- bucket sort scratch (fwd_L1 用 sorted layout 切換) --
    bucket_counts_dev: DeviceBuffer<u32>, // NUM_BUCKETS + 1 (histogram + invalid bin)
    bucket_offsets_dev: DeviceBuffer<u32>, // NUM_BUCKETS + 1 (exclusive scan)
    bucket_write_ctr_dev: DeviceBuffer<u32>, // NUM_BUCKETS + 1 (scatter ranking counter)
    bucket_perm_dev: DeviceBuffer<i32>,   // batch (perm[i] = original row index)
    bucket_idx_sorted_dev: DeviceBuffer<i32>, // batch (sorted bucket values)
    combined_sorted: DeviceBuffer<f32>,   // batch × FT_OUT (combined を perm で gather)
    l1_bucket_sorted: DeviceBuffer<f32>,  // batch × L1_OUT (sorted fwd_L1 出力)
    dl1_total_sorted: DeviceBuffer<f32>,  // batch × L1_OUT (dl1_total を perm で gather)
    dl2_out_sorted: DeviceBuffer<f32>,    // batch × L2_OUT (dl2_out を perm で gather、L2 bias 用)
}

impl GpuWorkspace {
    /// `batch` 個の position 分の全 buffer を確保する (`GpuTrainer::new` から呼ぶ)。
    ///
    /// `ft_fp16_out` が true なら FT activation (`ft_*_out` / `dft_*_out`) を `f16` で
    /// 持つ。その場合 f32 版は使われないので placeholder size (FT_OUT 要素 = 1 行) で
    /// のみ確保し、`*_h` (f16) を b × FT_OUT で確保する。false なら f32 版を
    /// b × FT_OUT、`*_h` は `None`。
    fn new(
        stream: &CudaStream,
        batch: usize,
        ft_fp16_out: bool,
        feature_set: FeatureSetSpec,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ft_in = feature_set.ft_in();
        let max_active = feature_set.max_active();
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(stream, n).map_err(Into::into)
        };
        // FT activation の f32 buffer size。ft_fp16_out 時は f16 版を使うので f32 版は
        // placeholder (FT_OUT 要素) のみ。
        let ft_act_f32_n = if ft_fp16_out { FT_OUT } else { batch * FT_OUT };
        let alloc_h = |on: bool| -> Result<Option<DeviceBuffer<f16>>, Box<dyn std::error::Error>> {
            if on {
                Ok(Some(DeviceBuffer::<f16>::zeroed(stream, batch * FT_OUT)?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            len_batch: batch,
            ft_in,
            max_active,
            ft_stm_out: z(ft_act_f32_n)?,
            ft_nstm_out: z(ft_act_f32_n)?,
            ft_stm_out_h: alloc_h(ft_fp16_out)?,
            ft_nstm_out_h: alloc_h(ft_fp16_out)?,
            dft_stm_out_h: alloc_h(ft_fp16_out)?,
            dft_nstm_out_h: alloc_h(ft_fp16_out)?,
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
            dft_stm_out: z(ft_act_f32_n)?,
            dft_nstm_out: z(ft_act_f32_n)?,
            feat_counts: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_offsets: DeviceBuffer::<u32>::zeroed(stream, ft_in + 1)?,
            feat_write_ctr: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_positions: DeviceBuffer::<u32>::zeroed(stream, batch * max_active)?,
            stm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            bucket_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch)?,
            score_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            stm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            bucket_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch)?,
            score_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            bucket_counts_dev: DeviceBuffer::<u32>::zeroed(stream, NUM_BUCKETS + 1)?,
            bucket_offsets_dev: DeviceBuffer::<u32>::zeroed(stream, NUM_BUCKETS + 1)?,
            bucket_write_ctr_dev: DeviceBuffer::<u32>::zeroed(stream, NUM_BUCKETS + 1)?,
            bucket_perm_dev: DeviceBuffer::<i32>::zeroed(stream, padded_sort_batch(batch))?,
            bucket_idx_sorted_dev: DeviceBuffer::<i32>::zeroed(stream, padded_sort_batch(batch))?,
            combined_sorted: z(padded_sort_batch(batch) * FT_OUT)?,
            l1_bucket_sorted: z(padded_sort_batch(batch) * L1_OUT)?,
            dl1_total_sorted: z(padded_sort_batch(batch) * L1_OUT)?,
            dl2_out_sorted: z(padded_sort_batch(batch) * L2_OUT)?,
        })
    }

    /// `GpuTrainer::new` で確保した `len_batch` 容量に `batch` が収まることを検証する。
    /// 収まらなければ error を返す (caller が step を中断)。
    ///
    /// workspace は固定 batch 前提で `GpuTrainer::new` 時に一度だけ確保する。実 dataloader
    /// は `batch_size` 以下の batch しか出さない (末尾の partial batch は小さい) ので
    /// 通常この検証は通る。step 中は前 step の compute が in-flight でありうるため、
    /// ここで buffer を再 alloc すると使用中の device memory を解放する race になる。
    /// よって grow はせず、容量超過は error として扱う。
    fn check_batch_capacity(&self, batch: usize) -> Result<(), Box<dyn std::error::Error>> {
        if batch > self.len_batch {
            return Err(format!(
                "batch {batch} exceeds workspace capacity {} (workspace は GpuTrainer::new で\
                 一度だけ確保する。--batch-size を増やす場合は再起動が要る)",
                self.len_batch
            )
            .into());
        }
        Ok(())
    }
}

/// [`GpuTrainer::step_impl`] の出力。
///
/// `loss` は batch 全体の二乗誤差和 (`Σ err²`、position 数で割る前)。`net_output`
/// は held-out validation (`validate == true`) のときだけ position ごとの net
/// 出力スカラ (`n_pos` 個) で埋まり、通常の training step では空。
struct StepOutput {
    loss: f64,
    net_output: Vec<f32>,
}

/// Smoke / trainer 用の 1 batch 入力データ。
/// owned 版 (smoke path) と borrowed 版 (train_step path) を統一するため scalar の
/// `per_pos_norm` を持ち (= 1/n_pos)、ref 化された slice を直接 H2D 投入する。
#[allow(dead_code)]
struct BatchData<'a> {
    n_pos: usize,
    stm_indices: &'a [i32], // (n_pos × max_active)、-1 padding 可
    nstm_indices: &'a [i32],
    bucket_idx: &'a [i32], // (n_pos)、progress8kpabs の 0-8
    score: &'a [f32],      // (n_pos)、target eval cp の元
    wdl: &'a [f32],        // (n_pos)、0.0 (Loss) / 0.5 (Draw) / 1.0 (Win)
    per_pos_norm: f32,     // 1/n_pos scalar (loss kernel が `norm[bi]` を本値の broadcast で読む)
}

/// `BatchData` を owned 形で組み立てるための一時 buffer (smoke / test 用)。本体 train_step
/// path では `BatchData::from_batch_ref` を使う (slice 借用)。
struct BatchDataOwned {
    n_pos: usize,
    stm_indices: Vec<i32>,
    nstm_indices: Vec<i32>,
    bucket_idx: Vec<i32>,
    score: Vec<f32>,
    wdl: Vec<f32>,
}

impl BatchDataOwned {
    fn as_ref(&self) -> BatchData<'_> {
        let n = self.n_pos;
        BatchData {
            n_pos: n,
            stm_indices: &self.stm_indices,
            nstm_indices: &self.nstm_indices,
            bucket_idx: &self.bucket_idx,
            score: &self.score,
            wdl: &self.wdl,
            per_pos_norm: if n == 0 { 0.0 } else { 1.0_f32 / n as f32 },
        }
    }
}

impl BatchData<'_> {
    /// 決定論的な smoke 用 dummy batch。bucket_idx=0、small random sparse indices。
    /// `feature_set` で `max_active` (1 perspective あたり active feature 数) と
    /// index の範囲 `[0, ft_in)` が決まる。
    fn smoke_dummy(n_pos: usize, feature_set: FeatureSetSpec) -> BatchDataOwned {
        let ft_in = feature_set.ft_in();
        let max_active = feature_set.max_active();
        let mut stm_indices = vec![-1_i32; n_pos * max_active];
        let mut nstm_indices = vec![-1_i32; n_pos * max_active];
        // 各 position に max_active 個の deterministic indices を入れる。
        // range [0, ft_in) で seed-based に分散。
        let mut s: u64 = 0xdead_beef;
        for b in 0..n_pos {
            for k in 0..max_active {
                // xorshift
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let idx = (s as usize % ft_in) as i32;
                stm_indices[b * max_active + k] = idx;
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let idx2 = (s as usize % ft_in) as i32;
                nstm_indices[b * max_active + k] = idx2;
            }
        }
        BatchDataOwned {
            n_pos,
            stm_indices,
            nstm_indices,
            bucket_idx: vec![0_i32; n_pos],
            score: vec![0.0_f32; n_pos],
            wdl: vec![0.5_f32; n_pos],
        }
    }

    /// bucket-aware backend (LayerStack) 用: `nnue-train` dataloader の `Batch` +
    /// per-position bucket (= `n_pos` 個) から borrowed `BatchData` を作る (`.to_vec()` を
    /// 避けて 22 MB の CPU memcpy を削減)。`bucket_idx.len() == n_pos` を厳密 assert する
    /// ので、誤って空 slice を渡すと panic で検出される。
    fn from_batch_ref<'a>(batch: &'a Batch, bucket_idx: &'a [i32]) -> BatchData<'a> {
        let n_pos = batch.n_positions;
        assert_eq!(
            bucket_idx.len(),
            n_pos,
            "bucket_idx len ({}) must equal batch.n_positions ({})",
            bucket_idx.len(),
            n_pos
        );
        Self::from_batch_inner(batch, bucket_idx)
    }

    /// bucket-less backend (Simple) 用: bucket_idx は空 slice。`TrainingConfig::compute_bucket
    /// = false` で worker が bucket 計算を skip した経路で使う (`SimpleGpuTrainer::train_step`
    /// は元から bucket_idx を参照しない契約のため空 slice で安全)。LayerStack 経路で誤って
    /// 本 fn を呼ぶと bucket_idx 不在で backend kernel が読む先がなくなるが、それは
    /// host driver の責任で本 fn 内は検査しない (LayerStack ↔ Simple の host driver は
    /// 別 path で型混在しない)。
    fn from_batch_ref_bucketless<'a>(batch: &'a Batch) -> BatchData<'a> {
        Self::from_batch_inner(batch, &[])
    }

    fn from_batch_inner<'a>(batch: &'a Batch, bucket_idx: &'a [i32]) -> BatchData<'a> {
        let n_pos = batch.n_positions;
        let max_active = batch.feature_set.max_active();
        assert_eq!(
            batch.max_active,
            max_active,
            "Batch::max_active ({}) must equal feature set '{}' max_active ({})",
            batch.max_active,
            batch.feature_set.canonical_name(),
            max_active
        );
        let span = n_pos * max_active;
        let norm = if n_pos == 0 {
            0.0
        } else {
            1.0_f32 / n_pos as f32
        };
        BatchData {
            n_pos,
            stm_indices: &batch.stm_indices[..span],
            nstm_indices: &batch.nstm_indices[..span],
            bucket_idx,
            score: &batch.score[..n_pos],
            wdl: &batch.wdl[..n_pos],
            per_pos_norm: norm,
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
/// 再 alloc を伴わず既存 buffer を in-place で reset するため (grad / `loss_acc` の
/// 毎 step reset で `cudaMalloc`/`cudaFree` の stream stall を回避)。
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

/// `i32` buffer の全要素を `-1` (= 0xFFFFFFFF) に async fill。bucket sort padding 行
/// の bucket marker / perm sentinel を invalid に初期化する用途。`memset_d8(0xFF)` は
/// 二の補数で -1 を作るため i32 専用 (符号無し型に対しては UINT_MAX を意味する)。
fn memset_minus_one_i32(
    stream: &CudaStream,
    buf: &DeviceBuffer<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = buf.num_bytes();
    if bytes > 0 {
        unsafe {
            cuda_core::memory::memset_d8_async(
                buf.cu_deviceptr(),
                0xFF,
                bytes,
                stream.cu_stream(),
            )?;
        }
    }
    Ok(())
}

/// bucket sort 用の padded sorted layout 容量を計算する。各 bucket は次 16-row 境界に
/// align するため最大 `(NUM_BUCKETS + 1) * 15` 行の padding を要する。安全側で
/// `(NUM_BUCKETS + 1) * 16` を上乗せして 16 倍数に切り上げる。
fn padded_sort_batch(batch: usize) -> usize {
    let raw = batch + (NUM_BUCKETS + 1) * 16;
    raw.div_ceil(16) * 16
}

/// pre-allocated device buffer に host slice を async memcpy。`DeviceBuffer::from_host`
/// の毎-step cudaMalloc/Free を排除するため。Caller は `buf` と `src` の長さが一致
/// (バッチ毎 fixed shape) を保証。
fn copy_host_to_device_async_i32(
    stream: &CudaStream,
    buf: &DeviceBuffer<i32>,
    src: &[i32],
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        src.len() <= buf.len(),
        "src.len()={} exceeds buf.len()={}",
        src.len(),
        buf.len()
    );
    let bytes = std::mem::size_of_val(src);
    if bytes == 0 {
        return Ok(());
    }
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buf.cu_deviceptr(),
            src.as_ptr(),
            bytes,
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

fn copy_host_to_device_async_f32(
    stream: &CudaStream,
    buf: &DeviceBuffer<f32>,
    src: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        src.len() <= buf.len(),
        "src.len()={} exceeds buf.len()={}",
        src.len(),
        buf.len()
    );
    let bytes = std::mem::size_of_val(src);
    if bytes == 0 {
        return Ok(());
    }
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buf.cu_deviceptr(),
            src.as_ptr(),
            bytes,
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

// ===========================================================================
// cuBLAS FFI — `dense_mm_bwd_weight_tiled` (L1f weight bwd) を `cublasSgemm_v2`
// に置換。CUDA Toolkit 12.x の dynamic link で取得 (`build.rs` で
// `cargo:rustc-link-lib=dylib=cublas`)。
// ===========================================================================

#[repr(C)]
#[allow(non_camel_case_types)]
struct cublasContext {
    _opaque: [u8; 0],
}
#[allow(non_camel_case_types)]
type cublasHandle_t = *mut cublasContext;
#[allow(non_camel_case_types)]
type cublasStatus_t = std::os::raw::c_int;
#[allow(non_camel_case_types)]
type cublasOperation_t = std::os::raw::c_int;

const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
const CUBLAS_OP_N: cublasOperation_t = 0;
const CUBLAS_OP_T: cublasOperation_t = 1;

// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` を呼ぶと、以後の
// Sgemm は FP32 input → TF32 (8-bit exp + 10-bit mantissa) cast → TC mma →
// FP32 accum に lower される (Ampere+)。FP32 比 ~2x スループット、~10-bit
// mantissa の精度低下。
//
// `cublasMath_t` enum (`/usr/local/cuda-*/include/cublas_api.h`、CUDA 12.9 時点):
//   CUBLAS_DEFAULT_MATH                              = 0
//   CUBLAS_TENSOR_OP_MATH                            = 1  (deprecated alias、FP16 TC fallback)
//   CUBLAS_PEDANTIC_MATH                             = 2
//   CUBLAS_TF32_TENSOR_OP_MATH                       = 3
//   CUBLAS_FP32_EMULATED_BF16X9_MATH                 = 4  (Hopper+ BF16x9 emulation)
//   CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16 (bit mask)
#[allow(non_camel_case_types)]
type cublasMath_t = std::os::raw::c_uint;
const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
const CUBLAS_TF32_TENSOR_OP_MATH: cublasMath_t = 3;

#[link(name = "cublas", kind = "dylib")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    fn cublasSetStream_v2(
        handle: cublasHandle_t,
        stream_id: cuda_core::sys::CUstream,
    ) -> cublasStatus_t;
    fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;
    fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: std::os::raw::c_int,
        n: std::os::raw::c_int,
        k: std::os::raw::c_int,
        alpha: *const f32,
        a: *const f32,
        lda: std::os::raw::c_int,
        b: *const f32,
        ldb: std::os::raw::c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: std::os::raw::c_int,
    ) -> cublasStatus_t;
}

/// RAII wrapper for `cublasHandle_t`。Create 失敗 / Set stream 失敗 / Destroy 失敗を
/// `Result` で返す。CUDA stream に bind して以後の Sgemm を同 stream で in-order 実行。
struct CublasHandle {
    handle: cublasHandle_t,
}

// SAFETY: `cublasHandle_t` は CUDA driver が tracking する opaque handle。cuBLAS API は
// driver thread safety guarantees に従い handle を別 thread から呼び出してよい
// (`cublasSetStream_v2` が thread-affinity を切り替えるとき内部 lock を取る)。
unsafe impl Send for CublasHandle {}

impl CublasHandle {
    /// `enable_tf32 = true` で Ampere+ Tensor Core を TF32 mode で活用する
    /// (`cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`)。Sgemm の FP32
    /// input は内部で TF32 (8-bit exp + 10-bit mantissa) cast → TC mma → FP32
    /// accum に lower され、throughput と引き換えに仮数 ~3 桁の精度低下を受ける。
    /// `false` では `CUBLAS_DEFAULT_MATH` (純 FP32 path、TC 不使用) を使う。
    ///
    /// 本 handle は fwd (`sgemm_fwd_rowmajor`) / bwd (`sgemm_xt_y_rowmajor`)
    /// 双方で共有されるため、L1f forward と weight backward の両 Sgemm に同
    /// mode が効く。
    fn new(stream: &CudaStream, enable_tf32: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        // SAFETY: cublasCreate_v2 は &mut handle に新規 handle を書き、CUBLAS_STATUS_SUCCESS
        // 以外を返したら handle は invalid (read 禁止)。失敗時は早期 return。
        let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasCreate_v2 failed: status={status}").into());
        }
        // SAFETY: handle is valid (above), stream.cu_stream() returns the wrapped CUstream。
        let status =
            unsafe { cublasSetStream_v2(handle, stream.cu_stream() as cuda_core::sys::CUstream) };
        if status != CUBLAS_STATUS_SUCCESS {
            // SAFETY: handle is valid (cleanup before erroring).
            unsafe {
                cublasDestroy_v2(handle);
            }
            return Err(format!("cublasSetStream_v2 failed: status={status}").into());
        }
        let mode = if enable_tf32 {
            CUBLAS_TF32_TENSOR_OP_MATH
        } else {
            CUBLAS_DEFAULT_MATH
        };
        // SAFETY: handle is valid.
        let status = unsafe { cublasSetMathMode(handle, mode) };
        if status != CUBLAS_STATUS_SUCCESS {
            // SAFETY: handle is valid (cleanup before erroring).
            unsafe {
                cublasDestroy_v2(handle);
            }
            let label = if enable_tf32 { "TF32" } else { "FP32" };
            return Err(format!("cublasSetMathMode({label}) failed: status={status}").into());
        }
        Ok(Self { handle })
    }

    /// row-major C[M, N] = A[M, K] @ B[K, N]、`alpha=1`, `beta=0` (overwrite)。
    /// fwd_L1f 用: combined[B, FT_OUT] @ l1f_w[FT_OUT, L1_OUT] → l1f_out[B, L1_OUT]。
    ///
    /// col-major cuBLAS で row-major matmul を計算する転置 trick: 同 memory 表現
    /// を cublas は col-major と解釈するので、`A_rm[m, k]` は `A_cm[k, m]`、
    /// `B_rm[k, n]` は `B_cm[n, k]`、`C_rm[m, n]` は `C_cm[n, m]` と等価。
    ///   row-major C[m, n] = sum_k A[m, k] * B[k, n]
    ///   = col-major C[n, m] = sum_k B_cm[n, k] * A_cm[k, m]
    /// → cublas call: A_arg=B_dev, B_arg=A_dev, transA=N, transB=N, m=N, n=M, k=K,
    ///   lda=N, ldb=K, ldc=N。両 trans=N の単純形なので bwd 用 `sgemm_xt_y_rowmajor`
    ///   (X^T @ Y、transB=T) より素直。
    ///
    /// SAFETY:
    /// - 全 device pointer は `cudaMalloc` 由来、長さは仕様分 (a.len() >= m*k、
    ///   b.len() >= k*n、c.len() >= m*n)。
    /// - stream は `cublasSetStream_v2` で bind 済の同一 stream を再利用。
    /// - math mode は handle 作成時の `enable_tf32` 引数で固定 ([`CublasHandle::new`]):
    ///   `true` で `CUBLAS_TF32_TENSOR_OP_MATH` (Ampere+ TC 経由、仮数 10-bit)、
    ///   `false` で `CUBLAS_DEFAULT_MATH` (純 FP32 path)。本関数は mode 非依存で
    ///   呼び出し可能、numeric tolerance は CLI `--tf32` 指定有無で変動する。
    /// - `beta=0` overwrite なので `c_ptr` の事前内容は使われない (caller は
    ///   `c_ptr` への書き込みを同 stream 内 in-order で行うこと、別 stream からの
    ///   race 書き込みは未定義動作)。
    unsafe fn sgemm_fwd_rowmajor(
        &self,
        m: i32,
        n: i32,
        k: i32,
        a_ptr: *const f32, // row-major [m, k]
        b_ptr: *const f32, // row-major [k, n]
        c_ptr: *mut f32,   // row-major [m, n]
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, // cublas m = N (cols of C in col-major)
                m, // cublas n = M (rows of C in col-major)
                k,
                &alpha,
                b_ptr, // cublas A = B (row-major [k, n] = col-major [n, k])
                n,
                a_ptr, // cublas B = A (row-major [m, k] = col-major [k, m])
                k,
                &beta,
                c_ptr,
                n,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 (fwd) failed: status={status}").into());
        }
        Ok(())
    }

    /// row-major C[M, N] = X^T @ Y、X[K, M] row-major、Y[K, N] row-major (X^T Y の reduce
    /// 軸は K)。col-major cuBLAS で計算するため転置 trick を使う:
    /// cublas は C_cm[N, M] = Y_cm[N, K] @ (X_cm[M, K])^T を計算、行列要素は同 memory。
    /// 詳細は call 元コメント参照。`alpha=1`, `beta=0` (overwrite)。
    ///
    /// SAFETY: 全 device pointer は cudaMalloc 由来 + 各 buffer 長 == 仕様分、stream は
    /// `cublasSetStream_v2` で bind 済の同一 stream を再利用。caller が形状不変条件
    /// (X.len() >= k*m、Y.len() >= k*n、C.len() >= m*n) を保証。
    unsafe fn sgemm_xt_y_rowmajor(
        &self,
        m: i32,
        n: i32,
        k: i32,
        x_ptr: *const f32, // row-major [k, m]
        y_ptr: *const f32, // row-major [k, n]
        c_ptr: *mut f32,   // row-major [m, n]
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // col-major cuBLAS で row-major C_rm = X_rm^T @ Y_rm を出すには:
        //   cublas C_cm[N, M] = Y_cm[N, K] @ (X_cm[M, K])^T と計算 (Y は trans=N、X は trans=T)
        //   Y_cm[n, k] = Y_rm[k, n] (同 memory)、X_cm[m, k] = X_rm[k, m] (同 memory)。
        //   結果 C_cm[n, m] = sum_k Y_rm[k, n] * X_rm[k, m] = C_rm[m, n] (同 memory)。
        // 引数: A=Y, B=X, transA=N, transB=T, m=N, n=M, k=K, lda=N, ldb=M, ldc=N。
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n, // m for cublas = n (out_dim)
                m, // n for cublas = m (in_dim)
                k,
                &alpha,
                y_ptr,
                n,
                x_ptr,
                m,
                &beta,
                c_ptr,
                n,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 failed: status={status}").into());
        }
        Ok(())
    }

    /// row-major C[M, N] = X @ Y^T、X[M, K] row-major、Y[N, K] row-major。bwd_input 用:
    /// dx[B, in_dim] = dy[B, out_dim] @ w[in_dim, out_dim]^T、reduce 軸は out_dim。
    /// col-major cuBLAS で計算する転置 trick:
    ///   C_rm[m, n] = sum_k X_rm[m, k] * Y_rm[n, k]
    ///   = C_cm[n, m] = sum_k Y_cm[k, n] * X_cm[k, m]   (X_rm[m, k] と X_cm[k, m] は同 memory)
    ///   → cublas A=Y (transA=T、shape [k, n] → 効果 [n, k])、B=X (transB=N、shape [k, m])
    ///     m_cublas=n、n_cublas=m、k_cublas=k、lda=k、ldb=k、ldc=n。
    /// `alpha=1`, `beta=0` (overwrite)。
    ///
    /// SAFETY: 全 device pointer は cudaMalloc 由来 + 各 buffer 長 >= 仕様分、stream は
    /// `cublasSetStream_v2` で bind 済の同一 stream を再利用。caller が形状不変条件
    /// (X.len() >= m*k、Y.len() >= n*k、C.len() >= m*n) を保証。
    unsafe fn sgemm_x_yt_rowmajor(
        &self,
        m: i32,
        n: i32,
        k: i32,
        x_ptr: *const f32, // row-major [m, k]
        y_ptr: *const f32, // row-major [n, k]
        c_ptr: *mut f32,   // row-major [m, n]
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                y_ptr,
                k,
                x_ptr,
                k,
                &beta,
                c_ptr,
                n,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 (bwd_input) failed: status={status}").into());
        }
        Ok(())
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is valid (created in new()).
            unsafe {
                cublasDestroy_v2(self.handle);
            }
        }
    }
}

/// `step()` 末尾の `loss_acc.to_host_vec` (内部で `stream.synchronize`) を排除し
/// host が次 batch の launch を即発行できるようにする ring。
///
/// 2-slot ring + 1-step lag: step N で device の `loss_acc` を pinned cell[N%2] に
/// async D2H + event record。返り値は step N-1 の loss (event[(N-1)%2] sync 後に
/// pinned cell[(N-1)%2] を読む)。最初の 1 step は前 step が無いので 0.0 を返す。
///
/// pinned host (`cuMemHostAlloc`) なので driver は staging copy 無しで直接 DMA、
/// 8 byte D2H + event record は host work と完全並行。
///
/// 末尾 step の loss は [`AsyncLossRing::flush_pending_loss`] で drain する。
/// [`crate::TrainerBackend::flush_pending_loss`] 経由で本 ring の `flush_pending_loss`
/// が superbatch 末で 1 回呼ばれ、未報告分が `sb_loss` に加算される。これにより
/// pipeline 化しても per-sb loss 集計は正確 (`sum(L_0..L_{N-1})`、warmup placeholder
/// 0 は sum に影響なし)。
struct AsyncLossRing {
    pinned: [*mut f64; 2],
    events: [CudaEvent; 2],
    step: usize,
    primed: bool,
}

// SAFETY: `pinned` は `cuMemHostAlloc` で確保した page-locked memory、CUDA driver の
// 内部 tracking 経由でアクセスされる。pointer 自体は host メモリで `Send` 安全。
unsafe impl Send for AsyncLossRing {}

impl AsyncLossRing {
    fn new(ctx: &std::sync::Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pinned = [std::ptr::null_mut::<f64>(); 2];
        for slot in pinned.iter_mut() {
            let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
            // SAFETY: cuMemHostAlloc は page-locked host memory を 8 byte 確保、
            // failure 時は CUresult != SUCCESS を返す (.result()? で check)。
            unsafe {
                cuda_core::sys::cuMemHostAlloc(
                    &mut p as *mut _,
                    std::mem::size_of::<f64>(),
                    cuda_core::sys::CU_MEMHOSTALLOC_PORTABLE,
                )
                .result()?;
                // 初期値 0 (warmup で読まれないが defensive)
                std::ptr::write(p as *mut f64, 0.0);
            }
            *slot = p as *mut f64;
        }
        let events = [ctx.new_event(None)?, ctx.new_event(None)?];
        Ok(Self {
            pinned,
            events,
            step: 0,
            primed: false,
        })
    }

    /// `loss_acc` (device 1-cell f64) を async D2H で pinned[cur] へ copy、event 記録。
    /// 前 step (= step - 1) の event を sync して pinned[prev] を読み返り値とする。
    /// 最初の呼出 (warmup) は 0.0 を返す。
    fn read_and_queue_next(
        &mut self,
        stream: &CudaStream,
        loss_acc: &DeviceBuffer<f64>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let cur = self.step % 2;
        // SAFETY: pinned[cur] is page-locked host memory (cuMemHostAlloc 8 bytes),
        // loss_acc has len == 1 (= 8 bytes), stream 上 in-order なので async D2H は
        // 直前の memset/atomic 完了後に実行される。
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                self.pinned[cur],
                loss_acc.cu_deviceptr(),
                std::mem::size_of::<f64>(),
                stream.cu_stream(),
            )?;
        }
        self.events[cur].record(stream)?;

        let returned = if self.primed {
            let prev = (self.step + 1) % 2; // = (step - 1) % 2
            self.events[prev].synchronize()?;
            // SAFETY: event sync 完了 = D2H 完了、pinned[prev] に書き込まれた f64 を読む。
            unsafe { *self.pinned[prev] }
        } else {
            self.primed = true;
            0.0
        };

        self.step += 1;
        Ok(returned)
    }

    /// pipeline 末尾の drain: 最後に queue した step (= step - 1) の event を sync
    /// して pinned[(step - 1) % 2] の loss を返す。未呼出 (warmup 直後など primed
    /// = false) なら 0.0 を返す。
    ///
    /// `primed` を `false` に戻し `step` も 0 にリセットする。これにより次回 call
    /// は warmup として 0.0 を返し、その次の call から再び lag-1 の正常 pipeline が
    /// 始まる。caller (sb 末尾の trainer) は本 fn の返り値を sb_loss に加算する。
    fn flush_pending_loss(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        let returned = if self.primed {
            let last = (self.step + 1) % 2;
            self.events[last].synchronize()?;
            // SAFETY: event sync 完了 = D2H 完了、pinned[last] に書き込まれた f64 を読む。
            unsafe { *self.pinned[last] }
        } else {
            0.0
        };
        // 次回 sb は warmup から再開する (step も reset することで step % 2 計算が
        // 一貫し、pinned/event 再利用に矛盾無し)。
        self.primed = false;
        self.step = 0;
        Ok(returned)
    }
}

impl Drop for AsyncLossRing {
    fn drop(&mut self) {
        // pinned cell を free する前に未完了の async D2H を待つ。さもなければ in-flight な
        // memcpy_dtoh_async が解放後 host memory に書き戻して UB になる。primed = false の
        // 場合は record されていない event なので skip。失敗は無視 (Drop 中の error 報告は
        // 実用上困難、driver が in-flight copy を tracking する debug-build 動作と等価)。
        if self.primed {
            for event in self.events.iter() {
                let _ = event.synchronize();
            }
        }
        for slot in self.pinned.iter() {
            if !slot.is_null() {
                // SAFETY: cuMemHostAlloc で確保した pointer は cuMemFreeHost で解放する。
                // 上の event sync で in-flight D2H が完了済。
                unsafe {
                    cuda_core::sys::cuMemFreeHost(*slot as *mut std::os::raw::c_void);
                }
            }
        }
    }
}

/// step 先頭の入力 H2D (`stm/nstm idx` + `bucket/score/wdl` の 5 buffer) を専用
/// copy stream で発行し、直前 step の compute と overlap させる ring。
///
/// 入力は dataloader から pageable な `Vec` で来る。pageable のままだと
/// `cuMemcpyHtoDAsync` は driver の同期 staging copy になり copy engine の DMA を
/// 使えず、compute と並走しない。pinned host buffer を経由し compute stream とは
/// 別の copy stream で発行することで、H2D は直前 step の compute と並走する。
///
/// 2-slot pinned ring: step N は `pinned[N%2]` を使う。`pinned[N%2]` を最後に読んだ
/// H2D (= step N-2) の event を [`upload`](Self::upload) 冒頭で sync してから上書き
/// するので、in-flight な H2D が読んでいる pinned を host が書き換える race は起きない。
///
/// device 側の double-buffer (直前 step が読む buffer と次 step を H2D する buffer の
/// 物理分離) は caller (`step_impl`) が active / back buffer を `mem::swap` して担う。
/// 本 ring は H2D 先として「現在 active な」device buffer を受け取り、H2D 完了 event を
/// compute stream に待たせる。
struct InputUploadRing {
    copy_stream: std::sync::Arc<CudaStream>,
    // pinned host staging。stm/nstm は `batch * max_active`、bucket/score/wdl は `batch`。
    // bucket は LayerStack のみ持ち、Simple アーキは bucket-less 入力なので `None`。
    pinned_stm: [*mut i32; 2],
    pinned_nstm: [*mut i32; 2],
    pinned_bucket: Option<[*mut i32; 2]>,
    pinned_score: [*mut f32; 2],
    pinned_wdl: [*mut f32; 2],
    /// 各 slot の H2D 完了 event (copy stream に record)。compute stream が forward 前に待つ。
    h2d_done: [CudaEvent; 2],
    /// 各 slot を使った step の compute 完了 event (compute stream に record、
    /// [`mark_step_done`](Self::mark_step_done))。同じ物理 device buffer を次に使う
    /// step (= 2 step 後) の H2D 前に copy stream が待ち、in-flight な compute が
    /// 読んでいる buffer を H2D が上書きする race を防ぐ。
    step_done: [CudaEvent; 2],
    /// stm/nstm pinned の要素容量 (`batch * max_active`)。
    cap_idx: usize,
    /// bucket/score/wdl pinned の要素容量 (`batch`)。
    cap_scalar: usize,
    step: usize,
}

// SAFETY: 非 `Send` な field は raw pointer `pinned_*` のみ。これは `cuMemHostAlloc` で
// 確保した page-locked host memory への pointer で、`InputUploadRing` が単独 owner、
// 全 access は `&mut self` method (`upload` / `mark_step_done`) 経由で直列化される。
// raw pointer 経由の aliasing も内部からの concurrent access も無いので別 thread へ
// 移しても安全 (`AsyncLossRing` と同じ理由)。
unsafe impl Send for InputUploadRing {}

impl InputUploadRing {
    /// LayerStack 用: copy stream + 2-slot pinned buffer + event を確保する (bucket あり)。
    /// `batch` は最大 position 数、`max_active` は 1 perspective あたりの active feature 数
    /// (feature set 依存)。
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch: usize,
        max_active: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(ctx, batch, max_active, true)
    }

    /// Simple アーキ用: bucket buffer を確保しないバリアント。Simple は bucket-less 入力
    /// で kernel への bucket dispatch も無いため、bucket H2D 経路自体を持たない。
    fn new_simple(
        ctx: &std::sync::Arc<CudaContext>,
        batch: usize,
        max_active: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(ctx, batch, max_active, false)
    }

    fn new_inner(
        ctx: &std::sync::Arc<CudaContext>,
        batch: usize,
        max_active: usize,
        has_bucket: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let copy_stream = ctx.new_stream()?;
        let cap_idx = batch.max(1) * max_active;
        let cap_scalar = batch.max(1);
        let pinned_bucket = if has_bucket {
            Some(alloc_pinned_host::<i32>(cap_scalar)?)
        } else {
            None
        };
        Ok(Self {
            copy_stream,
            pinned_stm: alloc_pinned_host::<i32>(cap_idx)?,
            pinned_nstm: alloc_pinned_host::<i32>(cap_idx)?,
            pinned_bucket,
            pinned_score: alloc_pinned_host::<f32>(cap_scalar)?,
            pinned_wdl: alloc_pinned_host::<f32>(cap_scalar)?,
            h2d_done: [ctx.new_event(None)?, ctx.new_event(None)?],
            step_done: [ctx.new_event(None)?, ctx.new_event(None)?],
            cap_idx,
            cap_scalar,
            step: 0,
        })
    }

    /// `batch` の入力 5 slice を pinned 経由で `dev_*` (caller が swap で active 化した
    /// device buffer) へ copy stream で async H2D し、`compute_stream` に H2D 完了を
    /// 待たせる。
    ///
    /// caller (`step_impl`) は呼出前に active / back device buffer を `mem::swap` 済で、
    /// `dev_*` は直前 step が読んでいない側の物理 buffer であること。これにより H2D は
    /// 直前 step の compute と物理 buffer 競合なしに並走する。
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        compute_stream: &CudaStream,
        dev_stm: &DeviceBuffer<i32>,
        h_stm: &[i32],
        dev_nstm: &DeviceBuffer<i32>,
        h_nstm: &[i32],
        dev_bucket: &DeviceBuffer<i32>,
        h_bucket: &[i32],
        dev_score: &DeviceBuffer<f32>,
        h_score: &[f32],
        dev_wdl: &DeviceBuffer<f32>,
        h_wdl: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            h_stm.len() <= self.cap_idx && h_nstm.len() <= self.cap_idx,
            "input batch ({} idx) exceeds pinned capacity {}",
            h_stm.len().max(h_nstm.len()),
            self.cap_idx
        );
        assert!(
            h_bucket.len() <= self.cap_scalar
                && h_score.len() <= self.cap_scalar
                && h_wdl.len() <= self.cap_scalar,
            "input batch (scalar) exceeds pinned capacity {}",
            self.cap_scalar
        );
        let slot = self.step % 2;
        if self.step >= 2 {
            // この物理 device buffer を最後に使った step (= step-2) の compute 完了を
            // copy stream に待たせてから H2D する。host は loss_ring 経由で複数 step
            // 先行しうるため、待たないと step-2 の backward がまだ読んでいる input
            // buffer を H2D が上書きする race になる。step 0/1 は当該 slot 未 record。
            self.copy_stream.wait(&self.step_done[slot])?;
            // 同 slot の pinned を最後に read した H2D (= step-2) の完了を host が待ち、
            // in-flight な H2D の読み元 pinned を下の copy_nonoverlapping が壊さないよう
            // にする。
            self.h2d_done[slot].synchronize()?;
        }
        let pinned_bucket = self
            .pinned_bucket
            .as_ref()
            .expect("InputUploadRing::upload (LayerStack) requires bucket-enabled ring");
        // host: Vec → pinned[slot]。
        // SAFETY: pinned[slot] は cuMemHostAlloc で cap 要素確保した有効 host memory、
        // 上の assert で `src.len() <= cap` を保証。src (Vec) / dst (pinned) は別領域。
        // step >= 2 の slot は直上の h2d_done sync で前回 H2D 完了済 (in-flight でない)。
        unsafe {
            std::ptr::copy_nonoverlapping(h_stm.as_ptr(), self.pinned_stm[slot], h_stm.len());
            std::ptr::copy_nonoverlapping(h_nstm.as_ptr(), self.pinned_nstm[slot], h_nstm.len());
            std::ptr::copy_nonoverlapping(h_bucket.as_ptr(), pinned_bucket[slot], h_bucket.len());
            std::ptr::copy_nonoverlapping(h_score.as_ptr(), self.pinned_score[slot], h_score.len());
            std::ptr::copy_nonoverlapping(h_wdl.as_ptr(), self.pinned_wdl[slot], h_wdl.len());
        }
        // device: pinned[slot] → dev_* (copy stream で async H2D)。
        // SAFETY: 各 pinned[slot] は直上の copy_nonoverlapping で先頭 `h_*.len()` 要素を
        // 初期化済の page-locked host memory。`from_raw_parts` で同じ `h_*.len()` 長の
        // slice 化して既存 H2D helper に渡す (helper が `src.len() <= dev.len()` を assert)。
        let cs: &CudaStream = &self.copy_stream;
        unsafe {
            copy_host_to_device_async_i32(
                cs,
                dev_stm,
                std::slice::from_raw_parts(self.pinned_stm[slot], h_stm.len()),
            )?;
            copy_host_to_device_async_i32(
                cs,
                dev_nstm,
                std::slice::from_raw_parts(self.pinned_nstm[slot], h_nstm.len()),
            )?;
            copy_host_to_device_async_i32(
                cs,
                dev_bucket,
                std::slice::from_raw_parts(pinned_bucket[slot], h_bucket.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_score,
                std::slice::from_raw_parts(self.pinned_score[slot], h_score.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_wdl,
                std::slice::from_raw_parts(self.pinned_wdl[slot], h_wdl.len()),
            )?;
        }
        self.h2d_done[slot].record(cs)?;
        // compute stream は H2D 完了後に forward が input を読むよう待つ。
        compute_stream.wait(&self.h2d_done[slot])?;
        Ok(())
    }

    /// Simple アーキ用 upload: bucket buffer を持たない 4 buffer 版 (stm/nstm/score/wdl)。
    /// 動作セマンティクスは [`upload`](Self::upload) と同じ — caller が active/back を
    /// `mem::swap` 済の `dev_*` に対し pinned 経由 copy stream で先行 H2D し、compute
    /// stream に H2D 完了を待たせる。
    #[allow(clippy::too_many_arguments)]
    fn upload_simple(
        &mut self,
        compute_stream: &CudaStream,
        dev_stm: &DeviceBuffer<i32>,
        h_stm: &[i32],
        dev_nstm: &DeviceBuffer<i32>,
        h_nstm: &[i32],
        dev_score: &DeviceBuffer<f32>,
        h_score: &[f32],
        dev_wdl: &DeviceBuffer<f32>,
        h_wdl: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            self.pinned_bucket.is_none(),
            "InputUploadRing::upload_simple called on bucket-enabled ring (LayerStack); \
             use InputUploadRing::new_simple to construct the ring"
        );
        assert!(
            h_stm.len() <= self.cap_idx && h_nstm.len() <= self.cap_idx,
            "input batch ({} idx) exceeds pinned capacity {}",
            h_stm.len().max(h_nstm.len()),
            self.cap_idx
        );
        assert!(
            h_score.len() <= self.cap_scalar && h_wdl.len() <= self.cap_scalar,
            "input batch (scalar) exceeds pinned capacity {}",
            self.cap_scalar
        );
        let slot = self.step % 2;
        if self.step >= 2 {
            self.copy_stream.wait(&self.step_done[slot])?;
            self.h2d_done[slot].synchronize()?;
        }
        // SAFETY: pinned[slot] は cap 要素確保済 host memory、上 assert で `src.len() <= cap` を
        // 保証。step >= 2 の slot は h2d_done sync で前回 H2D 完了済。
        unsafe {
            std::ptr::copy_nonoverlapping(h_stm.as_ptr(), self.pinned_stm[slot], h_stm.len());
            std::ptr::copy_nonoverlapping(h_nstm.as_ptr(), self.pinned_nstm[slot], h_nstm.len());
            std::ptr::copy_nonoverlapping(h_score.as_ptr(), self.pinned_score[slot], h_score.len());
            std::ptr::copy_nonoverlapping(h_wdl.as_ptr(), self.pinned_wdl[slot], h_wdl.len());
        }
        let cs: &CudaStream = &self.copy_stream;
        // SAFETY: pinned[slot] は直上で先頭 `h_*.len()` 要素を初期化済、`from_raw_parts`
        // で `src.len() <= dev.len()` を満たすよう slice 化 (helper が assert)。
        unsafe {
            copy_host_to_device_async_i32(
                cs,
                dev_stm,
                std::slice::from_raw_parts(self.pinned_stm[slot], h_stm.len()),
            )?;
            copy_host_to_device_async_i32(
                cs,
                dev_nstm,
                std::slice::from_raw_parts(self.pinned_nstm[slot], h_nstm.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_score,
                std::slice::from_raw_parts(self.pinned_score[slot], h_score.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_wdl,
                std::slice::from_raw_parts(self.pinned_wdl[slot], h_wdl.len()),
            )?;
        }
        self.h2d_done[slot].record(cs)?;
        compute_stream.wait(&self.h2d_done[slot])?;
        Ok(())
    }

    /// step の compute が input buffer を読み終えた (= step 全体が完了した) ことを
    /// `compute_stream` 上の event に記録し、step counter を進める。`step_impl` 末尾で
    /// 呼ぶ。同じ物理 device buffer を使う次 step ([`upload`](Self::upload) の step+2)
    /// が H2D 前にこの event を待ち、buffer reuse race を防ぐ。
    fn mark_step_done(
        &mut self,
        compute_stream: &CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let slot = self.step % 2;
        self.step_done[slot].record(compute_stream)?;
        self.step += 1;
        Ok(())
    }
}

impl Drop for InputUploadRing {
    fn drop(&mut self) {
        // pinned を free する前に in-flight な H2D を完了させる。copy stream を sync
        // すれば全 H2D が完了する。失敗は無視 (Drop 中の error 報告は実用上困難)。
        let _ = self.copy_stream.synchronize();
        let bucket_slots: &[*mut i32] = match self.pinned_bucket.as_ref() {
            Some(slots) => slots.as_slice(),
            None => &[],
        };
        for slot in self
            .pinned_stm
            .iter()
            .chain(self.pinned_nstm.iter())
            .chain(bucket_slots.iter())
        {
            if !slot.is_null() {
                // SAFETY: cuMemHostAlloc で確保した pointer を cuMemFreeHost で解放。
                // 上の copy stream sync で in-flight H2D は完了済。
                unsafe {
                    cuda_core::sys::cuMemFreeHost(*slot as *mut std::os::raw::c_void);
                }
            }
        }
        for slot in self.pinned_score.iter().chain(self.pinned_wdl.iter()) {
            if !slot.is_null() {
                // SAFETY: 同上。
                unsafe {
                    cuda_core::sys::cuMemFreeHost(*slot as *mut std::os::raw::c_void);
                }
            }
        }
    }
}

/// `cuMemHostAlloc` で page-locked host memory を `n` 要素分 2 slot 確保する。
fn alloc_pinned_host<T>(n: usize) -> Result<[*mut T; 2], Box<dyn std::error::Error>> {
    let mut out = [std::ptr::null_mut::<T>(); 2];
    for slot in out.iter_mut() {
        let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc は page-locked host memory を `n * size_of::<T>()` byte
        // 確保、失敗時は CUresult != SUCCESS を返す (.result()? で check)。
        unsafe {
            cuda_core::sys::cuMemHostAlloc(
                &mut p as *mut _,
                n * std::mem::size_of::<T>(),
                cuda_core::sys::CU_MEMHOSTALLOC_PORTABLE,
            )
            .result()?;
        }
        *slot = p as *mut T;
    }
    Ok(out)
}

impl GpuTrainer {
    /// CUDA context を作成し、kernel module を load、10 weight groups + Ranger state +
    /// 中間 activation workspace (`batch_size` 分) を確保。
    ///
    /// `enable_tf32` は cuBLAS の `cublasSetMathMode` 引数を切替 ([`CublasHandle::new`])、
    /// `true` で Ampere+ TC TF32 mode、`false` で純 FP32。default は CLI 側で OFF。
    ///
    /// `ft_fp16` が true なら FP16 weight mirror (`ft_w_h`) を確保し、forward の
    /// `sparse_ft_forward` を FP16 版に切替える。false なら mirror は未確保で従来 path。
    /// `ft_fp16_out` が true なら FT activation も FP16 で持つ (`ft_fp16` を要求、
    /// caller が validation 済)。
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch_size: usize,
        enable_tf32: bool,
        ft_fp16: bool,
        ft_fp16_out: bool,
        fp16_opt_state: bool,
        feature_set: FeatureSetSpec,
        weight_decay: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // `ft_fp16_out` は weight FP16 path の拡張なので `ft_fp16` を含意する。CLI 検証
        // (`run_training`) で reject 済だが、forward 分岐の各 `.expect()` がこの不変条件を
        // 前提にするため constructor でも明示する。
        debug_assert!(!ft_fp16_out || ft_fp16, "ft_fp16_out requires ft_fp16");
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "nnue_train")?;

        // 各 weight group の element 数 (FT 入力次元は feature set 依存)
        let ft_in = feature_set.ft_in();
        let ft_w_n = ft_in * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;

        // Weight init: small random for non-degenerate forward (smoke 用、後段で
        // proper init を適用: ft は bullet `init_with_effective_input_size(32)`、l1 は Zeroed 等)
        let init_scale = 0.01_f32;
        let ft_w_init = xorshift_init(0x100_u64, ft_w_n, init_scale);
        let l1_w_init = xorshift_init(0x101_u64, l1_w_n, init_scale);
        let l1f_w_init = xorshift_init(0x102_u64, l1f_w_n, init_scale);
        let l2_w_init = xorshift_init(0x103_u64, l2_w_n, init_scale);
        let l3_w_init = xorshift_init(0x104_u64, l3_w_n, init_scale);

        // Ranger Lookahead の slow weight は **0 初期化** (bullet `RangerLookahead::new`
        // = `vec![0.0; size]` と同じ)。初回 lerp (`step % k == 0`) で
        // `weights = alpha*weights + (1-alpha)*0 = alpha*weights` になる挙動も bullet と一致。
        Ok(Self {
            stream: stream.clone(),
            module,
            // FT
            ft_w: DeviceBuffer::from_host(&stream, &ft_w_init)?,
            ft_w_m: MomentBuf::zeroed(&stream, ft_w_n, fp16_opt_state)?,
            ft_w_v: MomentBuf::zeroed(&stream, ft_w_n, fp16_opt_state)?,
            ft_w_slow: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_grad: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_h: if ft_fp16 {
                Some(DeviceBuffer::<f16>::zeroed(&stream, ft_w_n)?)
            } else {
                None
            },
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
            // 中間 activation workspace (`batch_size` 分。最低 1 で確保して
            // `len_batch == 0` (未確保) を作らない — smoke は batch=4 等を渡す)。
            // FT activation の f16 buffer 確保は `ft_fp16_out` で決まる。
            ws: GpuWorkspace::new(&stream, batch_size.max(1), ft_fp16_out, feature_set)?,
            // loss + step
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            loss_ring: AsyncLossRing::new(ctx)?,
            input_ring: InputUploadRing::new(ctx, batch_size.max(1), feature_set.max_active())?,
            cublas: CublasHandle::new(&stream, enable_tf32)?,
            ft_fp16,
            ft_fp16_out,
            fp16_opt_state,
            feature_set,
            weight_decay,
            step_count: 0,
        })
    }

    /// `LayerStackWeights` から weight buffer を device に upload (pretrained 注入、`--init-from`)。
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
    ///   state が無いので next-best な default)
    /// - `grad`: 0
    /// - `step_count`: 0 (1-indexed、次 step は 1)
    ///
    /// 注: `step_count = 0` 状態で `step()` を呼ぶと `self.step_count += 1` → 1 に
    /// なってから `radam_compute_step_size_denom(1, BETA1, BETA2, N_SMA_THRESHOLD)`
    /// を呼ぶ。bullet `radam_step.rs::radam_compute_step_size_denom` は step >= 1 で
    /// 安全動作 (step=0 では `beta^0 = 1` → `bc1 = 0` で `step_size = 1/0 = inf` に
    /// なる、本 helper も `step >= 1` 前提)。本実装は step=0 で呼ばないため OK。
    fn load_layerstack_weights(
        &mut self,
        w: &LayerStackWeights,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // optimizer companion buffer (`ft_w_m`/`v`/`grad`/`slow`) は trainer の
        // feature set で確保済。weight の feature set が異なると `ft_w` だけ別長に
        // なり optimizer step が out-of-bounds になるため、ここで弾く。
        if w.feature_set != self.feature_set {
            return Err(invalid_data(format!(
                "weight feature set '{}' does not match trainer feature set '{}'",
                w.feature_set.canonical_name(),
                self.feature_set.canonical_name()
            )));
        }
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
        //   0 でなく読み込んだ重みの方へ寄る。`slow = 0` だと alpha 倍に縮む)
        // - grad: 0
        let zeros_f32 = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(&self.stream, n).map_err(Into::into)
        };
        let ft_w_n = self.feature_set.ft_in() * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        self.ft_w_m = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        self.ft_w_v = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
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

    /// device buffer を host に download し `LayerStackWeights` を返す (save_quantised 前)。
    fn to_layerstack_weights(&self) -> Result<LayerStackWeights, Box<dyn std::error::Error>> {
        Ok(LayerStackWeights {
            feature_set: self.feature_set,
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

    /// `ft_w` を **除く** 9 weight group の `(name, expected_len, &w, &m, &v, &slow)` を
    /// 固定順で返す (raw checkpoint の save/load で iterate するための immutable view)。
    /// `grad` は resume に不要なので含めない。順序 = ft_b, l1_w, l1_b, l1f_w, l1f_b,
    /// l2_w, l2_b, l3_w, l3_b。
    ///
    /// `ft_w` は `m` / `v` が `--fp16-opt-state` で `f16` ([`MomentBuf`]) になり buffer
    /// 型が他 group と揃わないため本配列から外し、checkpoint format 上 1 番目の group
    /// として save/load 側で個別に処理する (format の group 順は ft_w が先頭で不変)。
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
    ); 9] {
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

    /// `--resume` 用 **raw f32 checkpoint** を atomic に書き出す。
    ///
    /// 量子化 `.bin` ([`GpuTrainer::save_checkpoint`]/`to_layerstack_weights` → `save_quantised`)
    /// は推論用 final artifact として別 method で保存される。本 method はそれとは別の
    /// `*.ckpt` file に、全 10 weight group の **raw f32** `{w, m, v, slow}` (Ranger の
    /// 1st/2nd moment + Lookahead slow weight、`grad` は resume に不要なので含めない) +
    /// `step_count` (Ranger lookahead step counter) + 完了 `superbatch` 番号を書き出す。
    ///
    /// header の write / read は [`write_raw_ckpt_header`] / [`read_raw_ckpt_header`]
    /// に切り出してある。layout (全 little-endian、現行 [`RAW_CKPT_VERSION`] = 4):
    /// ```text
    /// magic        b"RNRC"             (4 bytes)
    /// version      u32 (4)             (4 bytes)
    /// fs_name_len  u32                 (4 bytes、feature set canonical 名の長さ)
    /// fs_name      UTF-8 [fs_name_len]  (feature set canonical 名、例 "halfka-hm-merged")
    /// ft_in        u64                 (FT 入力次元、feature set 依存)
    /// ft_out       u64                 (FT 出力次元、= FT_OUT)
    /// max_active   u64                 (1 perspective あたり active feature 数)
    /// run_id_len   u32                 (4 bytes、producer run id の長さ、0 可)
    /// run_id       UTF-8 [run_id_len]   (この checkpoint を書いた run の experiment.json `id`)
    /// arch_len     u32                 (4 bytes、arch kind canonical 名の長さ)
    /// arch_kind    UTF-8 [arch_len]     (arch kind canonical 名、LayerStack は "layerstack")
    /// topo_count   u64                 (topology 次元の個数)
    /// topology     u64 [topo_count]     (層次元列、LayerStack は FT_OUT/L1_OUT/L2_OUT/NUM_BUCKETS)
    /// superbatch   u64  (この checkpoint が表す完了 superbatch、resume はこの +1 から)
    /// step_count   u64  (Ranger lookahead step counter)
    /// num_groups   u64  (= 10、固定だが将来検証用)
    /// then for each of 10 groups (順序 = `raw_ckpt_groups()` = ft_w, ft_b, l1_w, l1_b,
    ///   l1f_w, l1f_b, l2_w, l2_b, l3_w, l3_b):
    ///   len u64
    ///   w[f32 × len]
    ///   m[f32 × len]
    ///   v[f32 × len]
    ///   slow[f32 × len]
    /// ```
    ///
    /// version 1 file には feature set header も run id も arch header も無く、weights
    /// は常に `halfka-hm-merged` / `layerstack` として解釈される。version 2/3 file は
    /// arch header を持たず `layerstack` 扱い。writer は常に最新 version を書く。
    ///
    /// device → host download (`DeviceBuffer::to_host_vec`) → `<path>.tmp` へ `BufWriter`
    /// で書く → `std::fs::rename(<path>.tmp, <path>)` で atomic に置換 (書き込み途中で
    /// crash しても `<path>` は前回の完全な checkpoint のまま)。
    ///
    /// `run_id` はこの checkpoint を書き出す run の experiment.json `id`。空文字列、
    /// または `MAX_RUN_ID_BYTES` 超過 (warning を出して省略) のときは run id を持た
    /// ない checkpoint になり、resume 時の `lineage.parent_id` は解決されない。
    fn save_raw_checkpoint(
        &self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        // 過長な run id (`{net_id}-{時刻}-{pid}`、通常数十バイト) は lineage という
        // メタデータのために学習を中断させる価値がない。上限超過時は埋め込みを
        // 省略 (長さ 0) し、warning を出して checkpoint 保存は続行する。
        let run_id = if run_id.len() > MAX_RUN_ID_BYTES {
            eprintln!(
                "[train] warning: producer run id ({} bytes) exceeds {MAX_RUN_ID_BYTES}; \
                 omitting it from {} (resume lineage parent will be unresolved)",
                run_id.len(),
                path.display()
            );
            ""
        } else {
            run_id
        };

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = {
            let mut p = path.as_os_str().to_os_string();
            p.push(".tmp");
            std::path::PathBuf::from(p)
        };

        // write+flush 本体を closure に括り、`fs::rename` 前の error path で
        // 中途半端な `<path>.tmp` を best-effort で消す (device→host download / write /
        // flush 失敗で残骸を残さないため)。
        let write_tmp = || -> Result<(), Box<dyn std::error::Error>> {
            let groups = self.raw_ckpt_groups();
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);
            // header (magic 〜 num_groups)。format 上の group 数は ft_w (個別処理) +
            // `raw_ckpt_groups` の 9 = 10。
            write_raw_ckpt_header(
                &mut w,
                &RawCkptArch {
                    feature_set: self.feature_set,
                    arch_kind: ArchKind::LayerStack,
                    ft_out: FT_OUT as u64,
                    topology: &LAYERSTACK_TOPOLOGY,
                },
                run_id,
                superbatch as u64,
                self.step_count,
                (groups.len() + 1) as u64,
            )?;

            // group 0: ft_w。`m` / `v` は `--fp16-opt-state` で `f16` 格納だが、
            // checkpoint は常に真値 `f32` で書く (mode 非依存・format version 不変、
            // resume 時に当該 run の精度へ再 quantize される)。
            let ft_w_n = self.feature_set.ft_in() * FT_OUT;
            {
                let w_host = self.ft_w.to_host_vec(&self.stream)?;
                let m_host = self.ft_w_m.to_host_f32(&self.stream, FT_OPT_M_SCALE)?;
                let v_host = self.ft_w_v.to_host_f32(&self.stream, FT_OPT_V_SCALE)?;
                let slow_host = self.ft_w_slow.to_host_vec(&self.stream)?;
                for (label, got) in [
                    ("w", w_host.len()),
                    ("m", m_host.len()),
                    ("v", v_host.len()),
                    ("slow", slow_host.len()),
                ] {
                    if got != ft_w_n {
                        return Err(format!(
                            "raw checkpoint: group ft_w {label} buffer len {got} != expected {ft_w_n}"
                        )
                        .into());
                    }
                }
                w.write_all(&(ft_w_n as u64).to_le_bytes())?;
                write_f32_slice(&mut w, &w_host)?;
                write_f32_slice(&mut w, &m_host)?;
                write_f32_slice(&mut w, &v_host)?;
                write_f32_slice(&mut w, &slow_host)?;
            }

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

    /// raw checkpoint を読み戻す (`--resume` 用)。返り値は `(完了 superbatch 番号,
    /// producer run id)` — superbatch は caller が通常その +1 から resume する。
    /// producer run id は version 3+ の checkpoint なら `Some` (resume run の
    /// `lineage.parent_id` に使う)、version 1/2 や run id 未記録なら `None`。
    ///
    /// magic 不一致、`version > 4`、arch kind / topology が LayerStack と不一致、group 数
    /// や各 group の len が LayerStack arch と不一致、または `u64 → usize` overflow
    /// (32-bit / 破損 file) は `InvalidData` で reject
    /// (`crates/nnue-train::optimizer::RangerHostState::load_from_reader` と同方針)。
    ///
    /// header の解析 (feature set / arch kind / topology の照合) は
    /// [`read_raw_ckpt_header`] が担当する。version 1 file は feature set header を
    /// 持たず weights を `halfka-hm-merged` とみなす。version 1..=3 は arch header を
    /// 持たず `layerstack` とみなす。読み込んだ raw f32 を host → device upload し、
    /// `self.step_count` を復元する。`grad` buffer は触らない (step ごとに memset される)。
    fn load_raw_checkpoint(
        &mut self,
        path: &Path,
    ) -> Result<(usize, Option<String>), Box<dyn std::error::Error>> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);

        // header (magic 〜 num_groups) を読み、feature set / arch / topology を照合する。
        let header = read_raw_ckpt_header(
            &mut r,
            &RawCkptArch {
                feature_set: self.feature_set,
                arch_kind: ArchKind::LayerStack,
                ft_out: FT_OUT as u64,
                topology: &LAYERSTACK_TOPOLOGY,
            },
        )?;
        let superbatch = header.superbatch;
        let step_count = header.step_count;
        let producer_run_id = header.producer_run_id;

        // format 上の group 数は ft_w (個別処理) + `raw_ckpt_groups` の 9 = 10。
        let expected_groups: [(&'static str, usize); 9] = {
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
            ]
        };
        let total_groups = expected_groups.len() + 1;
        if header.num_groups != total_groups as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint num_groups {} != expected {total_groups}",
                header.num_groups
            )));
        }

        // 1 group 分 (len + w/m/v/slow の f32 × len) を読む helper。`expected_len` と
        // file 記載 len の不一致 / overflow は `InvalidData` で reject。
        let read_group = |r: &mut std::io::BufReader<std::fs::File>,
                          name: &str,
                          expected_len: usize|
         -> Result<RawCkptGroup, Box<dyn std::error::Error>> {
            let mut buf8 = [0u8; 8];
            read_exact_or_invalid(r, &mut buf8, &format!("group {name} len"))?;
            let len_u64 = u64::from_le_bytes(buf8);
            let len: usize = len_u64.try_into().map_err(|_| {
                invalid_data(format!(
                    "raw checkpoint group {name} len {len_u64} exceeds usize::MAX"
                ))
            })?;
            if len != expected_len {
                return Err(invalid_data(format!(
                    "raw checkpoint group {name} len mismatch: got {len}, want {expected_len} \
                     (network architecture mismatch)"
                )));
            }
            let w_host = read_f32_vec_io(r, len, &format!("group {name} w"))?;
            let m_host = read_f32_vec_io(r, len, &format!("group {name} m"))?;
            let v_host = read_f32_vec_io(r, len, &format!("group {name} v"))?;
            let slow_host = read_f32_vec_io(r, len, &format!("group {name} slow"))?;
            Ok((w_host, m_host, v_host, slow_host))
        };

        // 各 group を読み出し → host Vec に保持 (全部読んでから upload する。途中で
        // upload して途中 fail だと中途半端な state になるため)。group 0 は ft_w。
        let ft_w_loaded = read_group(&mut r, "ft_w", self.feature_set.ft_in() * FT_OUT)?;
        let mut loaded: Vec<RawCkptGroup> = Vec::with_capacity(expected_groups.len());
        for (name, expected_len) in expected_groups {
            loaded.push(read_group(&mut r, name, expected_len)?);
        }
        // EOF 確認 (trailing garbage は許容するが、足りないのは上で read_exact が弾く)。

        // host → device upload。ft_w の m / v は当該 run の精度 (`fp16_opt_state`) へ
        // 量子化して載せ直す (checkpoint は真値 f32、mode 非依存)。
        let (ftw_w, ftw_m, ftw_v, ftw_slow) = &ft_w_loaded;
        self.ft_w = DeviceBuffer::from_host(&self.stream, ftw_w)?;
        self.ft_w_m =
            MomentBuf::from_host_f32(&self.stream, ftw_m, self.fp16_opt_state, FT_OPT_M_SCALE)?;
        self.ft_w_v =
            MomentBuf::from_host_f32(&self.stream, ftw_v, self.fp16_opt_state, FT_OPT_V_SCALE)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, ftw_slow)?;

        // 残り 9 group (順序は `raw_ckpt_groups` = ft_b, l1_w, ..., l3_b)。
        macro_rules! up {
            ($idx:expr, $w:ident, $m:ident, $v:ident, $slow:ident) => {{
                let (w, m, v, s) = &loaded[$idx];
                self.$w = DeviceBuffer::from_host(&self.stream, w)?;
                self.$m = DeviceBuffer::from_host(&self.stream, m)?;
                self.$v = DeviceBuffer::from_host(&self.stream, v)?;
                self.$slow = DeviceBuffer::from_host(&self.stream, s)?;
            }};
        }
        up!(0, ft_b, ft_b_m, ft_b_v, ft_b_slow);
        up!(1, l1_w, l1_w_m, l1_w_v, l1_w_slow);
        up!(2, l1_b, l1_b_m, l1_b_v, l1_b_slow);
        up!(3, l1f_w, l1f_w_m, l1f_w_v, l1f_w_slow);
        up!(4, l1f_b, l1f_b_m, l1f_b_v, l1f_b_slow);
        up!(5, l2_w, l2_w_m, l2_w_v, l2_w_slow);
        up!(6, l2_b, l2_b_m, l2_b_v, l2_b_slow);
        up!(7, l3_w, l3_w_m, l3_w_v, l3_w_slow);
        up!(8, l3_b, l3_b_m, l3_b_v, l3_b_slow);

        self.step_count = step_count;
        Ok((superbatch, producer_run_id))
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

    /// `--ft-fp16` の FP16 weight mirror (`ft_w_h`) を現在の `ft_w` から再生成する。
    ///
    /// 学習中の mirror は optimizer (`radam_step_fp16_mirror` /
    /// `ranger_lookahead_lerp_fp16_mirror`) が `ft_w` 更新と同時に書く。ただし学習
    /// 開始時は optimizer 未実行で mirror が初期 0 のままなので、最初の forward の前に
    /// 一度だけ明示同期する。`ft_fp16` 無効時 (`ft_w_h` が `None`) は no-op。
    fn sync_ft_w_h_mirror(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
            let ft_w_n = self.feature_set.ft_in() * FT_OUT;
            cuda_launch! {
                kernel: cast_f32_to_f16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_w_n),
                args: [slice(self.ft_w), slice_mut(ft_w_h), ft_w_n as u32]
            }?;
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
    /// 含める。中間 activation / grad buffer は `GpuTrainer` 上の workspace に永続化
    /// しているので、`step_impl` で drop されるのは入力 H2D buffer (`stm_idx_dev` 等、
    /// position 数に比例した小さい buffer) だけになり、teardown tick は ~0 に落ちる。
    fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // 環境変数 `NNUE_TRAIN_STEP_PROFILE` がセットされていれば各 phase の境界で
        // `synchronize()` + 経過時間を stderr に出す (粗い h2d / forward / backward /
        // optimizer / teardown breakdown 用)。未設定なら追加の sync ゼロ。
        // WSL2 では ncu の GPU perf counter が使えず nsys も GPU-side kernel trace を
        // 取れないため、この粗い event timing が代替手段。
        let profile_step = std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some();
        if profile_step {
            self.stream.synchronize()?;
        }
        let mut prof_t0 = std::time::Instant::now();
        let result = self.step_impl(
            batch,
            lr,
            wdl_lambda,
            loss,
            false,
            profile_step,
            &mut prof_t0,
        )?;
        // step_impl の per-step device buffer はここまでに全部 drop 済 (cuMemFree)。
        if profile_step {
            self.stream.synchronize()?;
            eprintln!(
                "[step-profile] {:<10} {:8.3} ms",
                "teardown",
                prof_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(result.loss)
    }

    /// held-out validation の 1 batch を実行する。[`GpuTrainer::step_impl`] を
    /// `validate = true` で呼び、forward + loss kernel のみ走らせる (backward /
    /// optimizer step は無く、weight も optimizer state も一切更新しない)。
    ///
    /// 戻り値 [`StepOutput`] は batch 全体の `Σ err²` (`loss`) と position ごとの
    /// net 出力スカラ (`net_output`)。caller (`TrainerBackend::validate_step`) が
    /// 前者から平均 loss、後者から sign-agreement accuracy を出す。
    ///
    /// 冒頭で `stream.synchronize` し直前の training step (optimizer まで) の完了を
    /// 待ってから検証 forward を始める。検証は superbatch あたり 1 回・~1 batch 分
    /// なので同期コストは無視できる。
    fn validate(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        // 直前の training step の GPU work 完了を待つ。検証 forward の H2D / kernel が
        // in-flight な training compute と input buffer を取り合わないことを保証する。
        self.stream.synchronize()?;
        let profile_step = std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some();
        let mut prof_t0 = std::time::Instant::now();
        // lr は validate モードでは optimizer を呼ばないため未使用 (0.0 を渡す)。
        self.step_impl(
            batch,
            0.0,
            wdl_lambda,
            loss,
            true,
            profile_step,
            &mut prof_t0,
        )
    }

    /// `step` の実体。`loss` が [`LossKind::Sigmoid`] なら `loss_wdl` (plain sigmoid-MSE)、
    /// [`LossKind::Wrm`] なら `loss_wrm` (win-rate-model loss) を起動する。
    ///
    /// Forward path (15 step): bullet `shogi_layerstack.rs:2241-2289` の reference 実装を
    /// 本 file の `#[kernel]` 群で再現。中間 activation は `GpuTrainer` 上の永続 workspace
    /// (`self.ws.*`) を使い回す — forward の各 activation は読まれる前に kernel が
    /// 全 cell を上書きするので memset 不要。Backward path (~16 step): forward 逆順、`*_grad`
    /// buffer は本 method 冒頭で `memset_async(0)` で reset してから kernel が書き込む
    /// (per-bucket weight grad `dense_mm_bwd_weight_bucket` は 1 cell = 1 thread の overwrite、
    /// FT / L1f / bias の grad は atomic accumulate なので reset 必須。`dl1_total` も
    /// `slice_scatter_2d` の host 契約を守るため reset)。`loss_acc` も同様に毎 step memset。
    /// 入力 H2D buffer (`stm_idx_dev` 等) は workspace 上の pre-allocated buffer に
    /// async memcpy する。Optimizer: 10 weight groups × `radam_step` (+ 周期
    /// `ranger_lookahead_lerp`)。
    ///
    /// `profile_step` / `prof_t0` は呼び出し元 ([`GpuTrainer::step`]) が管理し、本 method
    /// 内の `prof_tick!` が各 phase 境界で `*prof_t0` を更新する (戻った後に呼び出し元が
    /// teardown tick で読む)。
    ///
    /// `validate == true` のときは **forward + loss kernel のみ**を実行し、loss kernel
    /// 直後に `loss_acc` と `net_output` を同期読み出しして early return する
    /// (backward / optimizer step は走らず weight は不変、held-out validation 用)。
    /// `validate == false` の通常 training path はこの分岐に入らないため、訓練の
    /// 数値挙動は本フラグ追加前と完全に同一。
    #[allow(clippy::too_many_arguments)]
    fn step_impl(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
        validate: bool,
        profile_step: bool,
        prof_t0: &mut std::time::Instant,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(StepOutput {
                loss: 0.0,
                net_output: Vec::new(),
            });
        }
        // defense-in-depth: tiled kernels (grid=b/16) は b % 16 == 0 を要求する。
        // CLI で `--batch-size` を 16 倍数に reject 済 (`run_training`)、`BucketedPrefetchedLoader`
        // も `n_positions == batch_size` を保証する (`dataloader.rs:572`) ため通常到達しない。
        // release で debug_assert! が消えるので、ここで `step_impl` 直入りされた場合の保険として
        // 明示的な runtime check を入れる。
        if !b.is_multiple_of(16) {
            return Err(format!(
                "batch.n_pos must be a multiple of 16 (got {}); tiled dense matmul kernels \
                 require b % 16 == 0 — partial last batch will silently truncate via grid=b/16",
                b
            )
            .into());
        }
        let b_u32 = b as u32;

        // batch `b` が workspace 容量に収まることを検証する (固定 batch 前提、
        // 起動時の `GpuWorkspace::new` で確保済)。
        self.ws.check_batch_capacity(b)?;

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

        // 入力 5 buffer を host → device。active / back buffer を `mem::swap` してから
        // back 側 (= 直前 step が読んでいない物理 buffer) へ専用 copy stream で先行 H2D
        // する。H2D は直前 step の compute と並走し、compute stream は H2D 完了 event を
        // 待ってから forward に進む ([`InputUploadRing`])。pageable な dataloader `Vec`
        // は ring 内の pinned host buffer 経由で copy engine の DMA に載る。
        std::mem::swap(&mut self.ws.stm_idx_dev, &mut self.ws.stm_idx_dev_back);
        std::mem::swap(&mut self.ws.nstm_idx_dev, &mut self.ws.nstm_idx_dev_back);
        std::mem::swap(
            &mut self.ws.bucket_idx_dev,
            &mut self.ws.bucket_idx_dev_back,
        );
        std::mem::swap(&mut self.ws.score_dev, &mut self.ws.score_dev_back);
        std::mem::swap(&mut self.ws.wdl_dev, &mut self.ws.wdl_dev_back);
        self.input_ring.upload(
            &self.stream,
            &self.ws.stm_idx_dev,
            batch.stm_indices,
            &self.ws.nstm_idx_dev,
            batch.nstm_indices,
            &self.ws.bucket_idx_dev,
            batch.bucket_idx,
            &self.ws.score_dev,
            batch.score,
            &self.ws.wdl_dev,
            batch.wdl,
        )?;
        // per_pos_norm は scalar (1/n_pos) として直接 kernel arg に渡す。

        // loss_acc reset (accumulate semantics、再 alloc せず memset)
        memset_zero(&self.stream, &self.loss_acc)?;
        prof_tick!("h2d+reset");

        // -- Forward step 1-2: sparse_ft_forward × 2 (stm, nstm) --
        // 中間 activation は workspace (`self.ws.*`) を使い回す (再 alloc 無し)。
        // forward の各 activation は読まれる前に kernel が全 cell を上書きするので memset 不要。
        // sparse_ft_forward は 1 thread = 4 row (output cell) なので grid は b * FT_OUT / 4。
        // FT_OUT = 1536 (4 の倍数) を arch 固定で保証。
        // forward kernel は 3 通り:
        //  - `ft_fp16_out`: `sparse_ft_forward_fp16_out` — f16 weight read + f16 出力
        //    (`ft_*_out_h`)。書き出し DRAM 帯域も半減。
        //  - `ft_fp16` のみ: `sparse_ft_forward_fp16` — f16 weight read + f32 出力。
        //  - どちらも無し: `sparse_ft_forward` — FP32 path、bit-identical。
        // いずれも累算は f32、1 thread = 4 row なので grid は b * FT_OUT / 4。
        debug_assert!(FT_OUT.is_multiple_of(4));
        if self.ft_fp16_out {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16 is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16_out,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out_h.as_mut()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled")),
                    b_u32, FT_OUT as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward_fp16_out,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out_h.as_mut()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    b_u32, FT_OUT as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        } else if self.ft_fp16 {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16 is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out),
                    b_u32, FT_OUT as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out),
                    b_u32, FT_OUT as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: sparse_ft_forward,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 4),
                args: [
                    slice(self.ft_w),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out),
                    b_u32, FT_OUT as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 4),
                args: [
                    slice(self.ft_w),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out),
                    b_u32, FT_OUT as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        }
        prof_tick!("fwd_ft");

        // -- Forward step 3: ft_post_perspective_fwd → combined (B × FT_OUT) --
        // `ft_fp16_out` 時は f16 入力版 (`ft_post_perspective_fwd_fp16`)。`combined` 出力は
        // 両 path とも f32 (後続 dense L1 path が f32 で読む)。
        if self.ft_fp16_out {
            cuda_launch! {
                kernel: ft_post_perspective_fwd_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * COMBINED_DIM),
                args: [
                    slice(self.ws.ft_stm_out_h.as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ws.ft_nstm_out_h.as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b),
                    slice_mut(self.ws.combined),
                    b_u32, FT_OUT as u32, FT_POST_SCALE
                ]
            }?;
        } else {
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
        }

        prof_tick!("fwd_ftpost");

        // Forward L1: bucket sort で row を bucket_idx 昇順に並べ替え、各 bucket の sorted
        // 開始 offset を TILE_B=16 境界に align してから `dense_mm_fwd_bucket_tiled_l1_sorted`
        // を 1-bucket-per-block で走らせる (per-K-tile の W_TILE shared-mem load は 1 bucket
        // 分のみ)。inverse permute で `l1_bucket` を original order に戻して後続に渡す。
        // 数値同等性: fwd_L1 は per-row independent (k=0..15 加算順保持) のため sort stability
        // に依らず baseline と bit-exact。
        debug_assert!(
            FT_OUT.is_multiple_of(16) && L1_OUT == 16 && NUM_BUCKETS == 9 && b.is_multiple_of(16)
        );

        // a) histogram + 16-aligned scan + scatter。aligned offset で各 bucket が 16-row
        // 境界に整列し、bucket 末端 / 次 bucket 開始間に padding 行ができる。padding 行は
        // bucket=-1 で initialise (sorted kernel 側で skip)、perm も -1 sentinel (inverse
        // permute が skip)。
        let padded_b = padded_sort_batch(b);
        memset_zero(&self.stream, &self.ws.bucket_counts_dev)?;
        memset_zero(&self.stream, &self.ws.bucket_write_ctr_dev)?;
        memset_minus_one_i32(&self.stream, &self.ws.bucket_perm_dev)?;
        memset_minus_one_i32(&self.stream, &self.ws.bucket_idx_sorted_dev)?;
        cuda_launch! {
            kernel: count_buckets,
            stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.bucket_idx_dev),
                slice(self.ws.bucket_counts_dev),
                b_u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: exclusive_scan_aligned,
            stream: self.stream, module: self.module,
            config: LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.bucket_counts_dev),
                slice(self.ws.bucket_offsets_dev),
                (NUM_BUCKETS + 1) as u32,
                16_u32
            ]
        }?;
        cuda_launch! {
            kernel: scatter_bucket_perm,
            stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.bucket_idx_dev),
                slice(self.ws.bucket_offsets_dev),
                slice(self.ws.bucket_write_ctr_dev),
                slice(self.ws.bucket_perm_dev),
                slice(self.ws.bucket_idx_sorted_dev),
                b_u32, NUM_BUCKETS as u32
            ]
        }?;

        // b) combined を perm で gather → combined_sorted。padding 行 (perm=-1) は
        // permute kernel が 0 fill (sorted kernel 側で bucket=-1 で skip するので値不問)。
        cuda_launch! {
            kernel: permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * FT_OUT),
            args: [
                slice(self.ws.combined),
                slice(self.ws.bucket_perm_dev),
                slice_mut(self.ws.combined_sorted),
                padded_b as u32, FT_OUT as u32
            ]
        }?;

        // c) sorted fwd_L1 → l1_bucket_sorted。padded_b/16 block、各 block uniform 保証。
        cuda_launch! {
            kernel: dense_mm_fwd_bucket_tiled_l1_sorted,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: ((padded_b / 16) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.combined_sorted),
                slice(self.l1_w),
                slice(self.l1_b),
                slice(self.ws.bucket_idx_sorted_dev),
                slice_mut(self.ws.l1_bucket_sorted),
                padded_b as u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // d) l1_bucket_sorted を perm で inverse-scatter → l1_bucket (original order)。
        // padding 行 (perm=-1) は inverse permute kernel が skip。
        cuda_launch! {
            kernel: inverse_permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L1_OUT),
            args: [
                slice(self.ws.l1_bucket_sorted),
                slice(self.ws.bucket_perm_dev),
                slice(self.ws.l1_bucket),
                padded_b as u32, L1_OUT as u32
            ]
        }?;

        prof_tick!("fwd_L1");

        // -- Forward step 5: L1f shared dense → l1f_out (B × L1_OUT) --
        // cuBLAS Sgemm (TF32 TC) で matmul、`bias_add_per_row` kernel で bias を別 pass。
        // shape: combined[B, FT_OUT] @ l1f_w[FT_OUT, L1_OUT] → l1f_out[B, L1_OUT]。
        //
        // SAFETY: combined / l1f_w / l1f_out は cudaMalloc 由来、長さは arch 上 invariant
        // (combined.len() == B*FT_OUT、l1f_w.len() == FT_OUT*L1_OUT、l1f_out.len() == B*L1_OUT)、
        // `self.cublas` は `self.stream` に bind 済で同 stream 内 in-order 実行 (先行 kernel
        // 完了後に Sgemm が走り、結果は後続 bias_add_per_row が観測)。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,  // m = batch
                L1_OUT as i32, // n = out_dim
                FT_OUT as i32, // k = in_dim
                self.ws.combined.cu_deviceptr() as *const f32,
                self.l1f_w.cu_deviceptr() as *const f32,
                self.ws.l1f_out.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_add_per_row,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.l1f_b),
                slice_mut(self.ws.l1f_out),
                b_u32, L1_OUT as u32
            ]
        }?;

        prof_tick!("fwd_L1f");

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

        prof_tick!("fwd_L1tail");

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
                slice(self.ws.bucket_idx_dev),
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

        prof_tick!("fwd_L2");

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
                slice(self.ws.bucket_idx_dev),
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
        // `loss_wrm` (win-rate-model loss)。
        match loss {
            LossKind::Sigmoid { scale } => {
                cuda_launch! {
                    kernel: loss_wdl,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b),
                    args: [
                        slice(self.ws.net_output),
                        slice(self.ws.score_dev),
                        slice(self.ws.wdl_dev),
                        batch.per_pos_norm,
                        slice_mut(self.ws.dy_net_output),
                        slice(self.loss_acc),
                        wdl_lambda, scale, b_u32
                    ]
                }?;
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
                target_offset,
                target_scaling,
            } => {
                cuda_launch! {
                    kernel: loss_wrm,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b),
                    args: [
                        slice(self.ws.net_output),
                        slice(self.ws.score_dev),
                        slice(self.ws.wdl_dev),
                        batch.per_pos_norm,
                        slice_mut(self.ws.dy_net_output),
                        slice(self.loss_acc),
                        wdl_lambda, nnue2score, in_scaling,
                        target_offset, target_scaling, b_u32
                    ]
                }?;
            }
        }
        prof_tick!("forward");

        // held-out validation: backward / optimizer をスキップし、loss kernel が
        // 書いた `loss_acc` (batch の Σ err²) と `net_output` (position ごとの net
        // 出力) を同期読み出しして early return する。weight も optimizer state も
        // 更新しない。`net_output` workspace は固定 batch 容量で確保されているため
        // 有効 position 数 `b` で truncate する。`to_host_vec` は内部で
        // `stream.synchronize` するので forward kernel 完了後の値が読める。
        if validate {
            let loss = self.loss_acc.to_host_vec(&self.stream)?[0];
            let mut net_output = self.ws.net_output.to_host_vec(&self.stream)?;
            net_output.truncate(b);
            prof_tick!("validate_io");
            return Ok(StepOutput { loss, net_output });
        }

        // ===== BACKWARD =====
        // 全 *_grad buffer を 0 で reset (atomic accumulate semantic に従う kernel が
        // 多い、また overwrite kernel も in-place 安全のため統一)。再 alloc せず
        // `memset_async(0)` で既存 buffer を reset (`ft_w_grad` だけで ~450MB の
        // `cudaMalloc`/`cudaFree` を毎 step 走らせるのを避けるため)。
        // `dl1_total` も `slice_scatter_2d` の host 契約 (「dst を 0 初期化」) を守るため reset。
        let ft_w_n = self.feature_set.ft_in() * FT_OUT;
        let ft_b_n = FT_OUT;
        let l1_w_n = NUM_BUCKETS * L1_OUT * FT_OUT;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = FT_OUT * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        // ft_w_grad の memset_zero は意図的に省略している: phase D iter 0 (stm) の
        // `gather_and_sum_per_feature_overwrite` が全 (feature, ri) cell を sum
        // (off_start==off_end の時も sum=0) で書き切るため、ここで 450MB を reset
        // するのは無意味 (毎 step の no-op を排除する論理整理)。
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
        prof_tick!("bwd_reset");

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
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.dl2_acted),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;
        // L3 weight bwd: in_dim=L2_OUT=32, out_dim=1, num_buckets=9。
        // 元 kernel は 288 cells × scan batch で並列度小。split-K + 9 bucket register
        // accumulator (`dense_mm_bwd_weight_bucket_tiled_l3`) に切替。
        // num_splits=64 → 64 blocks × 32 threads = 2048 threads ≈ 26 / SM (sm_86)。
        const _: () = assert!(L2_OUT == 32 && NUM_BUCKETS == 9);
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket_tiled_l3,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (64, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.l2_acted),
                slice(self.ws.dy_net_output),
                slice(self.ws.bucket_idx_dev),
                slice(self.l3_w_grad),
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
                slice(self.ws.bucket_idx_dev),
                slice(self.l3_b_grad),
                b_u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        prof_tick!("bwd_L3");

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
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.dl2_input),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        // L2 weight bwd: in_dim=L2_IN=30, out_dim=L2_OUT=32, num_buckets=9。
        // split-K + 9 bucket register accumulator (block_dim = 32 × 30 = 960、grid = 64 splits)。
        const _: () = assert!(L2_IN == 30 && L2_OUT == 32 && NUM_BUCKETS == 9);
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket_tiled_l2,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (64, 1, 1),
                block_dim: ((L2_OUT * L2_IN) as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.l2_input),
                slice(self.ws.dl2_out),
                slice(self.ws.bucket_idx_dev),
                slice(self.l2_w_grad),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        // L2 bias backward (sorted): dl2_out を bucket_perm_dev で gather → dl2_out_sorted、
        // per-block shared-mem reduce で global atomic 数を ~2M → ~131K (16× 削減)。
        // out_dim=32、block(256) = 8 sorted 行 × 32 oi cell、16-aligned sort で uniform-
        // bucket。fwd_L1 で構築済の bucket_perm_dev / bucket_idx_sorted_dev を再利用。
        cuda_launch! {
            kernel: permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L2_OUT),
            args: [
                slice(self.ws.dl2_out),
                slice(self.ws.bucket_perm_dev),
                slice_mut(self.ws.dl2_out_sorted),
                padded_b as u32, L2_OUT as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad_bucket_shared_sorted,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L2_OUT),
            args: [
                slice(self.ws.dl2_out_sorted),
                slice(self.ws.bucket_idx_sorted_dev),
                slice(self.l2_b_grad),
                padded_b as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        prof_tick!("bwd_L2");

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

        prof_tick!("bwd_L1eff");

        // -- Backward 5 reverse: L1f shared dense grad --
        // L1f input bwd: in_dim=FT_OUT=1536, out_dim=L1_OUT=16, batch=multiple of 16
        // → tiled (block=256 = 16 batch × 16 in_dim cells、grid=batch/16 × in_dim/16 = 4096*96).
        debug_assert!(b.is_multiple_of(16) && FT_OUT.is_multiple_of(16) && L1_OUT == 16);
        cuda_launch! {
            kernel: dense_mm_bwd_input_tiled,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (((b / 16) * (FT_OUT / 16)) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1f_w),
                slice_mut(self.ws.dcombined_from_l1f),
                b_u32, FT_OUT as u32, L1_OUT as u32
            ]
        }?;
        // L1f weight backward: row-major `grad_w[FT_OUT, L1_OUT] = combined^T @ dl1_total`。
        // combined[batch, FT_OUT] row-major、dl1_total[batch, L1_OUT] row-major、reduce 軸は
        // batch = 65536。M = 16 と細いが K が大きい reduce-bound shape は cuBLAS Sgemm の
        // split-K + tensor pipeline 最適化が効きやすい。
        //
        // SAFETY: combined / dl1_total / l1f_w_grad は cudaMalloc 由来、長さは arch 上
        // invariant (`combined.len() == b*FT_OUT`、`dl1_total.len() == b*L1_OUT`、
        // `l1f_w_grad.len() == FT_OUT*L1_OUT`)、`self.cublas` は `self.stream` に bind 済で
        // 同 stream 内 in-order 実行 (先行 kernel 完了後に Sgemm が走り、結果は後続 kernel
        // が観測する)。
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                FT_OUT as i32, // m = in_dim
                L1_OUT as i32, // n = out_dim
                b_u32 as i32,  // k = batch
                self.ws.combined.cu_deviceptr() as *const f32,
                self.ws.dl1_total.cu_deviceptr() as *const f32,
                self.l1f_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        // L1f bias backward: shared-mem reduce で global atomic を 1M → ~16K に削減。
        const _: () = assert!(L1_OUT == 16);
        cuda_launch! {
            kernel: bias_grad_shared_l1f,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1f_b_grad),
                b_u32, L1_OUT as u32
            ]
        }?;

        prof_tick!("bwd_L1f");

        // -- Backward 4 reverse: L1 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * FT_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1_w),
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.dcombined_from_l1),
                b_u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        prof_tick!("bwd_L1_inB");
        // L1 weight backward (sorted layout): combined_sorted は fwd_L1 で構築済、dl1_total を
        // 同 perm で gather → dl1_total_sorted。bucket_offsets_dev も fwd_L1 で構築済。各 block
        // は uniform-by-construction で 1 bucket の slice のみ accumulate (9-way if-else /
        // 9 register accumulator / 9 atomicAdd を 1 個ずつに集約)。
        debug_assert!(
            FT_OUT.is_multiple_of(16) && L1_OUT == 16 && NUM_BUCKETS == 9 && b.is_multiple_of(16)
        );
        cuda_launch! {
            kernel: permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L1_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.ws.bucket_perm_dev),
                slice_mut(self.ws.dl1_total_sorted),
                padded_b as u32, L1_OUT as u32
            ]
        }?;
        // split-K dim を grid_y に追加。num_splits=8 × NUM_BUCKETS=9 × in_tiles=96 = 6912 blocks。
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket_tiled_l1_sorted,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: ((FT_OUT / 16) as u32, 8, NUM_BUCKETS as u32),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.combined_sorted),
                slice(self.ws.dl1_total_sorted),
                slice(self.ws.bucket_offsets_dev),
                slice(self.l1_w_grad),
                padded_b as u32, FT_OUT as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        prof_tick!("bwd_L1_wB");
        // L1 bias: sorted layout で per-block shared-mem reduce、global atomic 数を
        // ~1M → ~66K に削減。dl1_total_sorted / bucket_idx_sorted_dev は同 step 内で
        // 構築済 (fwd_L1 + bwd_L1_wB 前 permute)。
        cuda_launch! {
            kernel: bias_grad_bucket_shared_sorted,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(padded_b * L1_OUT),
            args: [
                slice(self.ws.dl1_total_sorted),
                slice(self.ws.bucket_idx_sorted_dev),
                slice(self.l1_b_grad),
                padded_b as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        prof_tick!("bwd_L1");

        // dft (FT activation gradient) FP16 化の loss scaling 係数。dft ∝ 1/batch なので
        // batch 比例にして batch 非依存に f16 域へ載せる ([`FT_DFT_FP16_BASE_SCALE`])。
        // grad kernel が `* dft_scale` で書き、gather kernel が `* dft_inv_scale` で戻す。
        let dft_scale = FT_DFT_FP16_BASE_SCALE * (b as f32);
        let dft_inv_scale = 1.0_f32 / dft_scale;

        // -- Backward 3 reverse: ft_post_perspective_grad fused × 2 (stm, nstm) --
        // `dy = dcombined_from_l1 + dcombined_from_l1f` を fused kernel が in-register
        // で計算、合算済 buffer の materialize と read-back の DRAM roundtrip を避ける。
        // `ft_fp16_out` 時は forward activation `ft_*_out` を f16 で読み、dft 出力も f16
        // で書く版 (`ft_post_perspective_grad_fused_fp16`)。`d_combined_*` / `ft_b` /
        // `ft_b_grad` は両 path とも f32。stm: d_combined_offset = 0、nstm: = FT_OUT/2。
        if self.ft_fp16_out {
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_stm_out_h.as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_stm_out_h.as_mut()
                        .expect("dft_stm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b_grad),
                    b_u32, FT_OUT as u32, 0_u32, COMBINED_DIM as u32, FT_POST_SCALE,
                    dft_scale
                ]
            }?;
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_nstm_out_h.as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_nstm_out_h.as_mut()
                        .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b_grad),
                    b_u32, FT_OUT as u32, (FT_OUT / 2) as u32, COMBINED_DIM as u32, FT_POST_SCALE,
                    dft_scale
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_stm_out),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_stm_out),
                    slice(self.ft_b_grad),
                    b_u32, FT_OUT as u32, 0_u32, COMBINED_DIM as u32, FT_POST_SCALE
                ]
            }?;
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * FT_OUT / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_nstm_out),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_nstm_out),
                    slice(self.ft_b_grad),
                    b_u32, FT_OUT as u32, (FT_OUT / 2) as u32, COMBINED_DIM as u32, FT_POST_SCALE
                ]
            }?;
        }

        prof_tick!("bwd_ftpost");

        // -- Backward 1+2 reverse: sparse_ft_backward × 2 を inverse-index pipeline で実装。
        // 各 (b, ri) thread が直接 38 atomic add する素朴版は atomic contention で memory
        // bandwidth が飽和するため、phase A (count) → B (prefix sum) → C (scatter) →
        // D (per-feature gather+sum) の inverse-index 構成にして atomic を D だけに局所化する。
        // ft_w_grad は host が memset_zero 済、phase D は atomic add で stm/nstm を合算。
        // `gout` (dft) は phase D でのみ使うため loop は idx_dev のみで回し、phase D で
        // iter_idx に対応する dft buffer を選ぶ (`ft_fp16_out` 時は f16 版)。
        // feature set 依存の次元を loop 前に読み出す (per-iter の field 借用を避ける)。
        let ft_in = self.ws.ft_in;
        let max_active = self.ws.max_active;
        let total_pairs = (b * max_active) as u32;
        for (iter_idx, idx_dev) in [&self.ws.stm_idx_dev, &self.ws.nstm_idx_dev]
            .into_iter()
            .enumerate()
        {
            // A: feat_counts ← 0
            memset_zero(&self.stream, &self.ws.feat_counts)?;
            memset_zero(&self.stream, &self.ws.feat_write_ctr)?;
            prof_tick!("phA_reset");
            // A: build_feature_counts
            cuda_launch! {
                kernel: build_feature_counts,
                stream: self.stream, module: self.module,
                config: cfg_1d(b * max_active),
                args: [
                    slice(idx_dev),
                    slice(self.ws.feat_counts),
                    b_u32, max_active as u32, ft_in as u32
                ]
            }?;
            prof_tick!("phA_count");
            // B: exclusive_prefix_sum_small (1 block × 1024 threads, ft_in ≈ 73K)
            cuda_launch! {
                kernel: exclusive_prefix_sum_small,
                stream: self.stream, module: self.module,
                config: LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [
                    slice(self.ws.feat_counts),
                    slice(self.ws.feat_offsets),
                    ft_in as u32
                ]
            }?;
            prof_tick!("phB_psum");
            // C: scatter_positions
            cuda_launch! {
                kernel: scatter_positions,
                stream: self.stream, module: self.module,
                config: cfg_1d(b * max_active),
                args: [
                    slice(idx_dev),
                    slice(self.ws.feat_offsets),
                    slice(self.ws.feat_write_ctr),
                    slice(self.ws.feat_positions),
                    b_u32, max_active as u32, ft_in as u32
                ]
            }?;
            prof_tick!("phC_scat");
            // D: gather_and_sum_per_feature。block grid = (ft_in, FT_OUT/128), block_dim=128.
            // 1 回目 (stm) は overwrite、2 回目 (nstm) は atomic add で stm 結果に加算。
            // host は grad_w を memset_zero 済みだが、overwrite kernel は全 cell を書き切る。
            let d_config = LaunchConfig {
                grid_dim: (ft_in as u32, (FT_OUT / 128) as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // iter 0 = stm (dft_stm_out / overwrite)、iter 1 = nstm (dft_nstm_out / add)。
            // `ft_fp16_out` 時は dft が f16 なので f16 入力版の gather kernel を使う。
            if iter_idx == 0 {
                if self.ft_fp16_out {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_overwrite_fp16,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_stm_out_h.as_ref()
                                .expect("dft_stm_out_h is Some when ft_fp16_out is enabled")),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, FT_OUT as u32, dft_inv_scale
                        ]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_overwrite,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_stm_out),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, FT_OUT as u32
                        ]
                    }?;
                }
                // P-obs: phD iter 0 (stm overwrite) を独立計測する。`prof_tick!` は
                // stream.synchronize を打つので、これが無いと前 iter の compute が次
                // tick (phA_reset iter 1) に流れ込んで観測上 phA_reset が肥大化する。
                prof_tick!("phD_stm");
            } else {
                if self.ft_fp16_out {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_add_fp16,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_nstm_out_h.as_ref()
                                .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled")),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, FT_OUT as u32, dft_inv_scale
                        ]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_add,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_nstm_out),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, FT_OUT as u32
                        ]
                    }?;
                }
                prof_tick!("phD_nstm");
            }
            let _ = total_pairs; // unused yet
        }
        prof_tick!("bwd_ftbwd");

        // ===== OPTIMIZER STEP (Ranger = RAdam + Lookahead) =====
        self.step_count += 1;
        let (step_size, denom) =
            radam_compute_step_size_denom(self.step_count, BETA1, BETA2, N_SMA_THRESHOLD);

        // 10 weight groups × radam_step。FT weight (`ft_w`) の radam は 2 つの opt-in
        // flag で 4 通りに分岐する:
        //  - `--ft-fp16`: FP16 mirror (`ft_w_h`) 同時更新 variant を使い、forward 用
        //    mirror を別 cast kernel 無しで同期する (master FP32 が既に register 上に
        //    あるので half 書き出しのみ追加)。
        //  - `--fp16-opt-state`: m / v を `f16` で読み書きする `*_f16state` variant
        //    (DRAM traffic 半減、scale 付き格納)。
        // 他 9 group は moment が小さく `f16` 化の意味が無いので常に `radam_step`。
        // FT
        match (&mut self.ft_w_m, &mut self.ft_w_v) {
            (MomentBuf::F16(ft_w_m), MomentBuf::F16(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                    cuda_launch! {
                        kernel: radam_step_f16state_mirror,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W,
                               FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n as u32]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: radam_step_f16state,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W,
                               FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n as u32]
                    }?;
                }
            }
            (MomentBuf::F32(ft_w_m), MomentBuf::F32(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                    cuda_launch! {
                        kernel: radam_step_fp16_mirror,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W, ft_w_n as u32]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: radam_step,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2,
                               EPS, MIN_W, MAX_W, ft_w_n as u32]
                    }?;
                }
            }
            // m / v は同じ flag で `MomentBuf::zeroed` され、load/init でも同期するので
            // 精度が食い違うことはない。
            _ => unreachable!("ft_w m and v moment buffers always share precision"),
        }
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(ft_b_n),
            args: [slice_mut(self.ft_b), slice_mut(self.ft_b_m), slice_mut(self.ft_b_v),
                   slice_mut(self.ft_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, ft_b_n as u32]
        }?;
        // L1
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1_w_n),
            args: [slice_mut(self.l1_w), slice_mut(self.l1_w_m), slice_mut(self.l1_w_v),
                   slice_mut(self.l1_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1_b_n),
            args: [slice_mut(self.l1_b), slice_mut(self.l1_b_m), slice_mut(self.l1_b_v),
                   slice_mut(self.l1_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1_b_n as u32]
        }?;
        // L1f
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1f_w_n),
            args: [slice_mut(self.l1f_w), slice_mut(self.l1f_w_m), slice_mut(self.l1f_w_v),
                   slice_mut(self.l1f_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1f_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1f_b_n),
            args: [slice_mut(self.l1f_b), slice_mut(self.l1f_b_m), slice_mut(self.l1f_b_v),
                   slice_mut(self.l1f_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1f_b_n as u32]
        }?;
        // L2
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l2_w_n),
            args: [slice_mut(self.l2_w), slice_mut(self.l2_w_m), slice_mut(self.l2_w_v),
                   slice_mut(self.l2_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l2_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l2_b_n),
            args: [slice_mut(self.l2_b), slice_mut(self.l2_b_m), slice_mut(self.l2_b_v),
                   slice_mut(self.l2_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l2_b_n as u32]
        }?;
        // L3
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l3_w_n),
            args: [slice_mut(self.l3_w), slice_mut(self.l3_w_m), slice_mut(self.l3_w_v),
                   slice_mut(self.l3_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l3_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l3_b_n),
            args: [slice_mut(self.l3_b), slice_mut(self.l3_b_m), slice_mut(self.l3_b_v),
                   slice_mut(self.l3_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l3_b_n as u32]
        }?;

        // Lookahead lerp every K steps。lerp は radam の後に FT weight を再度書き換える
        // ので、`--ft-fp16` 時は FT weight の lerp も FP16 mirror 同時更新 variant を使い、
        // forward 用 `ft_w_h` を lerp 後の最終値で同期し直す。
        if self.step_count.is_multiple_of(RANGER_K) {
            if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                cuda_launch! {
                    kernel: ranger_lookahead_lerp_fp16_mirror,
                    stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                    args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), slice_mut(ft_w_h),
                           RANGER_ALPHA, ft_w_n as u32]
                }?;
            } else {
                cuda_launch! {
                    kernel: ranger_lookahead_lerp,
                    stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                    args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), RANGER_ALPHA, ft_w_n as u32]
                }?;
            }
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

        // 本 step の compute (input buffer の read を含む) 完了を copy stream 用の
        // event に記録する。同じ物理 input buffer を使う step+2 の H2D がこれを待ち、
        // in-flight compute が読む buffer を H2D が上書きする race を防ぐ。
        self.input_ring.mark_step_done(&self.stream)?;

        // `loss_acc` の host への取り出しを `AsyncLossRing` 経由で async 化。
        // pinned host cell に `memcpy_dtoh_async` + event record、前 step の event を
        // sync して 1 step lag で loss を返す (step 0 は warmup として 0.0、sb 末で
        // [`TrainerBackend::flush_pending_loss`] が最終 step 分を drain する)。host は
        // 次 batch の launch 発行で `stream.synchronize` 相当の block 待ちが消える。
        let loss = self
            .loss_ring
            .read_and_queue_next(&self.stream, &self.loss_acc)?;
        Ok(StepOutput {
            loss,
            net_output: Vec::new(),
        })
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
        // dataloader が出した batch の feature set が trainer 構築時に選んだ feature set
        // と一致することを確認する (buffer サイズ / kernel launch 次元が前者を前提に
        // 確保済のため、不一致は out-of-bounds になる)。
        if batch.feature_set != self.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.feature_set.canonical_name()
            )));
        }
        let data = BatchData::from_batch_ref(batch, bucket_idx);
        self.step(&data, lr, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("GpuTrainer::step failed: {e}")))
    }

    fn validate_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<ValidationStepOutput> {
        // train_step と同じく batch の feature set が trainer の feature set と
        // 一致することを確認する (GPU buffer / kernel 次元の前提)。
        if batch.feature_set != self.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.feature_set.canonical_name()
            )));
        }
        let data = BatchData::from_batch_ref(batch, bucket_idx);
        let out = self
            .validate(&data, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("GpuTrainer::validate failed: {e}")))?;
        Ok(ValidationStepOutput {
            sum_sq_err: out.loss,
            net_output: out.net_output,
        })
    }

    fn flush_pending_loss(&mut self) -> std::io::Result<f64> {
        self.loss_ring.flush_pending_loss().map_err(|e| {
            std::io::Error::other(format!(
                "GpuTrainer::loss_ring.flush_pending_loss failed: {e}"
            ))
        })
    }

    fn save_checkpoint(&mut self, path: &Path) -> std::io::Result<()> {
        let weights = self.to_layerstack_weights().map_err(|e| {
            std::io::Error::other(format!("GpuTrainer::to_layerstack_weights failed: {e}"))
        })?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        weights.save_quantised(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    fn save_resume_checkpoint(
        &mut self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
    ) -> std::io::Result<()> {
        self.save_raw_checkpoint(path, superbatch, run_id)
            .map_err(|e| {
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
// CLI (clap) — 引数群は bullet-shogi `examples/shogi_layerstack.rs` に対応
// ===========================================================================

/// rshogi NNUE trainer。
///
/// 学習する NNUE アーキを `layerstack` / `simple` サブコマンドで選ぶ。共有引数は
/// サブコマンドの前後どちらに置いてもよい global 引数。`--data <PSV>` を指定すると
/// training loop を回し、省略すると GPU smoke test (forward/backward path 確認) を
/// 実行する。
#[derive(Parser, Debug)]
#[command(name = "nnue-train", about = "rshogi NNUE trainer")]
struct Cli {
    /// 教師データ PSV ファイル (`PackedSfenValue` × N、各 40 bytes)。省略時は GPU smoke test。
    #[arg(long, global = true)]
    data: Option<PathBuf>,

    /// held-out validation 用の PSV ファイル。学習 `--data` とは別の、勾配更新に
    /// 一度も使わない局面を渡す。指定すると各 superbatch 末に forward-only 検証を
    /// 走らせ、test_loss (held-out 平均 loss) と test_accuracy (出力符号と対局結果の
    /// 一致率) を train ログと experiment.json に出す。発散・過学習の早期検出に使う。
    #[arg(long, global = true)]
    test_data: Option<PathBuf>,

    /// held-out validation 1 回あたりの検証局面数。test PSV の先頭からこの数だけ
    /// 取り、`--batch-size` 単位に切り上げて満タン batch を作る。`--test-data`
    /// 指定時のみ使う。
    #[arg(long, default_value_t = 10000, global = true)]
    test_positions: usize,

    /// checkpoint 出力先 directory (`{net_id}-{superbatch}.bin` を書き出す)。
    #[arg(long, default_value = "checkpoints", global = true)]
    output: PathBuf,

    /// network id (checkpoint file 名に使う)。
    #[arg(long, default_value = "rshogi", global = true)]
    net_id: String,

    /// 入力 feature set。次のいずれか: halfkp, halfka-split, halfka-merged,
    /// halfka-hm-split, halfka-hm-merged。FT 入力次元と active feature 数を決める。
    /// 既定の halfka-hm-merged は king-symmetric merged HalfKA。
    #[arg(long, default_value = "halfka-hm-merged", global = true)]
    feature_set: String,

    /// experiment.json の `name` (実験管理 UI での表示名)。未指定なら net_id、
    /// `--resume` 時は `{net_id} (resume @sb{開始 superbatch})`。
    #[arg(long, global = true)]
    experiment_name: Option<String>,

    /// 学習する superbatch 数 (1..=superbatches を回す)。default 10 は smoke 用、
    /// 本番は 400 程度。
    #[arg(long, default_value_t = 10, global = true)]
    superbatches: usize,

    /// 1 superbatch あたりの batch 数。
    #[arg(long, default_value_t = 6104, global = true)]
    batches_per_superbatch: usize,

    /// 1 batch あたりの position 数。default 16384 は smoke 用、本番は 65536 程度。
    #[arg(long, default_value_t = 16384, global = true)]
    batch_size: usize,

    /// 初期 learning rate。
    #[arg(long, default_value_t = 8.75e-4, global = true)]
    lr: f32,

    /// LR gamma (`lr_step` superbatch ごとに gamma 倍)。
    #[arg(long, default_value_t = 0.995, global = true)]
    lr_gamma: f32,

    /// LR step (gamma 倍する superbatch 間隔)。
    #[arg(long, default_value_t = 1, global = true)]
    lr_step: usize,

    /// WDL blend lambda (constant)。
    #[arg(long, default_value_t = 0.0, global = true)]
    wdl: f32,

    /// sigmoid loss の score scale (`loss_scale = 1 / scale`)。`--win-rate-model` 指定時は
    /// 使わない (WRM loss は `--wrm-*` 系の scaling を使う)。
    #[arg(long, default_value_t = 290.0, global = true)]
    scale: f32,

    /// `save_rate` superbatch ごと (および末尾) に checkpoint を書き出す。
    #[arg(long, default_value_t = 20, global = true)]
    save_rate: usize,

    /// `|score| >= score_drop_abs` の position を loss から除外する (bullet `--score-drop-abs`)。
    #[arg(long, global = true)]
    score_drop_abs: Option<i32>,

    /// 学習開始前に量子化 NNUE binary から weight を注入する (pretrained start)。
    /// optimizer state (Ranger m/v/slow/step) は **reset** される — 真の resume には
    /// `--resume` を使うこと (`--init-from` と `--resume` は排他)。
    #[arg(long, global = true)]
    init_from: Option<PathBuf>,

    /// raw checkpoint (`{net_id}-{sb}.ckpt`) から weight + Ranger optimizer state
    /// (m/v/slow/step) を復元して学習を再開する (真の resume)。`--init-from`
    /// とは排他 (`--init-from` は weight のみ注入し optimizer を reset するため)。
    /// `--start-superbatch` 未指定なら checkpoint に記録された superbatch の +1 から再開。
    #[arg(long, global = true)]
    resume: Option<PathBuf>,

    /// 学習を開始する superbatch 番号 (1-indexed, inclusive)。未指定時:
    /// `--resume` あり → checkpoint の superbatch +1、なし → 1。`1 <= N <= --superbatches`
    /// の範囲外ならエラー (resume で過去 sb をやり直す目的で明示指定も可)。
    #[arg(long, global = true)]
    start_superbatch: Option<usize>,

    /// raw checkpoint (`*.ckpt`) を直近 N 個だけ残す (ディスク節約)。
    /// 未指定なら全保持 (raw state は ~1.8GB/個 なので save-rate × superbatches が
    /// 大きい長期ランでは指定推奨; 例 save-rate 20 / 400sb = 20 個 ≈ 36GB)。量子化
    /// `.bin` (~116MB) は本設定に関わらず常に全保持 (推論 artifact)。
    #[arg(long, global = true)]
    keep_checkpoints: Option<usize>,

    /// win-rate-model loss を使う。指定時は `loss_wrm` kernel (prediction / target
    /// 双方に WRM を適用) を使い、未指定なら `loss_wdl` (plain sigmoid-MSE + `--scale`)。
    /// net_output のスケールが `out ≈ cp/--wrm-nnue2score` になり、量子化
    /// (`QA=127/QB=64/FV_SCALE=28`) が前提とするスケールと整合する。
    #[arg(long, global = true)]
    win_rate_model: bool,
    /// WRM prediction 側の in-scaling (既定 340)。target 側の scaling
    /// (`--wrm-target-scaling`) とは独立。`--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 340.0, global = true)]
    wrm_in_scaling: f32,
    /// WRM の nnue2score (`scorenet = net_output * --wrm-nnue2score`、既定 600)。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 600.0, global = true)]
    wrm_nnue2score: f32,
    /// WRM target sigmoid の中心オフセット (`target` が 0.5 になる score、既定 270)。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 270.0, global = true)]
    wrm_target_offset: f32,
    /// WRM target sigmoid の入力スケール (steepness の逆数、既定 380)。既定 270/380 は
    /// chess の評価値分布向けの値なので、score 分布が異なれば再調整する。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 380.0, global = true)]
    wrm_target_scaling: f32,
    /// optimizer 名 ("ranger" のみ実装)。
    #[arg(long, default_value = "ranger", global = true)]
    optimizer: String,
    /// Ranger optimizer の weight decay 係数 (AdamW 風の decoupled weight decay)。
    /// 既定 0.0 で decay 無し。非 0 で全 weight group の weight を毎 step わずかに
    /// 0 方向へ減衰させる。
    #[arg(long, default_value_t = 0.0, global = true)]
    weight_decay: f32,
    /// dataloader prefetch worker 数。各 worker が PSV パース + HalfKA_hm sparse 抽出 +
    /// progress8kpabs bucket 計算を `decode()` 1 回で済ませて先読み供給する。`1` で
    /// 決定論的逐次 read、`>= 2` で並列パース (1 epoch 内の position 順序は非決定的;
    /// training では問題ない)。
    #[arg(long, default_value_t = 16, global = true)]
    threads: usize,

    /// FT weight (`ft_w`) を FP16 mirror で forward する高速モード。default `false`
    /// では FP32 path と bit-identical。`true` で `sparse_ft_forward` の weight DRAM
    /// 帯域を半減する代わり、量子化誤差で棋力が変動しうる (簡易・高速学習向けの
    /// opt-in option、本番品質には SPRT で確認するまで default OFF)。
    ///
    /// FT weight は初期化・optimizer の MIN_W/MAX_W clamp (`|w| <= 1.98`)・量子化
    /// checkpoint いずれの経路でも小さく、FP16 の有限域 (`|x| <= 65504`) に十分
    /// 収まるため mirror 変換が ±inf へ overflow しない。
    #[arg(long, global = true)]
    ft_fp16: bool,

    /// 特徴変換器 (FT) の optimizer state を FP16 で保持する高速モード。default
    /// `false` では FP32 path と bit-identical。
    ///
    /// FT は本ネットで最も要素数の多い層で、その optimizer 更新は state の read/write
    /// がメモリ帯域律速。state を半精度化すると optimizer step のメモリ転送量が減って
    /// 学習スループットが上がる。state は値が極めて小さいため、固定係数を掛けて FP16
    /// の有効域に載せてから格納する。
    ///
    /// `--ft-fp16` / `--ft-fp16-out` とは独立した flag。量子化誤差で棋力が変動し
    /// うるため default OFF、本番品質は SPRT で確認するまで保証しない (動作確認や
    /// 簡易・高速な学習に使う opt-in option)。
    #[arg(long, global = true)]
    fp16_opt_state: bool,

    /// 学習する NNUE アーキを選ぶサブコマンド (`layerstack` / `simple`)。
    #[command(subcommand)]
    arch: ArchCommand,
}

/// 学習対象の NNUE アーキを選ぶサブコマンド。アーキ固有の引数を持つ。
#[derive(Subcommand, Debug)]
enum ArchCommand {
    /// progress8kpabs 9-bucket LayerStack アーキ (HalfKA_hm 1536-16-32)。
    #[command(name = "layerstack")]
    LayerStack(LayerstackArgs),
    /// bullet-shogi 由来の Simple 4 層アーキ。
    Simple(SimpleArgs),
}

impl ArchCommand {
    /// サブコマンドに対応する [`ArchKind`]。
    fn kind(&self) -> ArchKind {
        match self {
            ArchCommand::LayerStack(_) => ArchKind::LayerStack,
            ArchCommand::Simple(_) => ArchKind::Simple,
        }
    }
}

/// LayerStack アーキ固有の引数。
#[derive(Args, Debug)]
struct LayerstackArgs {
    /// progress8kpabs 係数ファイル (`progress.bin`、f64 LE × 81*FE_OLD_END)。
    /// 未指定なら全 position が bucket 4 (zero weights → `sigmoid(0) = 0.5`)。
    #[arg(long)]
    progress_coeff: Option<PathBuf>,

    /// bucket mode ("progress8kpabs" のみ実装)。
    #[arg(long, default_value = "progress8kpabs")]
    bucket_mode: String,

    /// Ampere+ Tensor Core を TF32 mode で使う opt-in flag。`true` で cuBLAS の
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` を呼び、Sgemm の
    /// 入力 FP32 を 10-bit mantissa の TF32 に丸めて TC mma → FP32 accum で走る
    /// (仮数精度 ~3 桁、指数範囲は FP32 同等)。default `false` では
    /// `CUBLAS_DEFAULT_MATH` (純 FP32 path、TC 不使用) で走る。
    ///
    /// 仮数 13 bit 切り捨てで `fwd_L1f` / `bwd_L1f` Sgemm の数値に影響するため、
    /// 品質 conservative に default OFF。
    #[arg(long)]
    tf32: bool,

    /// FT activation (`ft_*_out` の forward 出力と `dft_*_out` の backward 勾配) も
    /// FP16 で保持する。`--ft-fp16` を要求する (weight FP16 path の上に積む拡張)。
    ///
    /// `ft_*_out` は `sparse_ft_forward` の出力で、これを FP16 化すると後続 read +
    /// inverse-index gather (step 中で最も DRAM read が多い `phD`) の帯域が半減する。
    /// dft は batch 正規化で `1/batch` に比例する微小値のため、FP16 化時は loss scaling
    /// (batch に比例する係数) で normal range に持ち上げてから格納する。
    ///
    /// weight FP16 (`--ft-fp16`) とは別 flag に分けてあり、SPRT で
    /// FP32 → `--ft-fp16` → `--ft-fp16 --ft-fp16-out` の 2 段で棋力影響を切り分け
    /// られる。量子化誤差で棋力が変動しうるため default OFF、本番品質は SPRT 確認まで
    /// 保証しない。
    #[arg(long)]
    ft_fp16_out: bool,
}

/// Simple 4 層アーキ固有の引数。
#[derive(Args, Debug)]
struct SimpleArgs {
    /// 層次元 preset (`<l1>x2-<l2>-<l3>`)。l1 は accumulator (FT 出力) 次元、
    /// l2 / l3 は隠れ層次元。`--l1` / `--l2` / `--l3` で個別に上書きできる。
    #[arg(long, default_value = "256x2-32-32")]
    arch: String,

    /// accumulator (FT 出力) 次元。未指定なら `--arch` preset の値。
    #[arg(long)]
    l1: Option<usize>,

    /// 隠れ層 1 の次元。未指定なら `--arch` preset の値。
    #[arg(long)]
    l2: Option<usize>,

    /// 隠れ層 2 の次元。未指定なら `--arch` preset の値。
    #[arg(long)]
    l3: Option<usize>,

    /// 活性化関数 ("crelu" または "screlu")。
    #[arg(long, default_value = "crelu")]
    activation: String,

    /// FT activation (`ft_*_out` の forward 出力と `dft_*_out` の backward 勾配) も
    /// FP16 で保持する。global `--ft-fp16` を要求し、現状 `--activation crelu` 限定。
    ///
    /// `ft_*_out` は `sparse_ft_forward` の出力で、これを FP16 化すると後続 read +
    /// `sparse_ft_backward` の read 帯域が半減する。dft は batch 正規化で `1/batch`
    /// に比例する微小値のため、FP16 化時は loss scaling (batch 比例) で normal range
    /// に持ち上げてから格納する。
    ///
    /// 量子化誤差で棋力が変動しうるため default OFF、本番品質は SPRT で確認するまで
    /// 保証しない opt-in option。
    #[arg(long)]
    ft_fp16_out: bool,

    /// Ampere+ Tensor Core を TF32 mode で使う opt-in flag。`true` で cuBLAS の
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` を呼び、L1/L2/L3 dense
    /// Sgemm の FP32 入力を 10-bit mantissa の TF32 に丸めて TC mma → FP32 accum で走る
    /// (仮数精度 ~3 桁、指数範囲は FP32 同等)。default `false` では
    /// `CUBLAS_DEFAULT_MATH` (純 FP32 path、TC 不使用) で走る。
    ///
    /// 仮数 13 bit 切り捨てで dense Sgemm の数値に影響するため、品質 conservative に
    /// default OFF。LayerStack `--tf32` と同方針 (棋力 risk opt-in)。
    #[arg(long)]
    tf32: bool,
}

fn run_training(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // アーキ種別で host pipeline を分岐する。Simple は別 driver
    // ([`run_simple_training`]) で受け、LayerStack 側はそのまま既存の flow を継続する。
    let layerstack = match &cli.arch {
        ArchCommand::LayerStack(args) => args,
        ArchCommand::Simple(args) => return run_simple_training(cli, args),
    };

    let data = cli.data.as_ref().expect("run_training called with --data");

    // 入力 feature set を CLI から一度だけ決める (以降の buffer 確保 / kernel launch /
    // dataloader / checkpoint identity が参照する単一の真実源)。
    let feature_set = FeatureSet::from_canonical_name(&cli.feature_set)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            let names: Vec<&str> = FeatureSet::ALL
                .iter()
                .map(|fs| fs.canonical_name())
                .collect();
            format!(
                "--feature-set '{}' is not a known feature set (expected one of: {})",
                cli.feature_set,
                names.join(", ")
            )
            .into()
        })?
        .spec();

    // --- 未実装オプション値の reject ---
    if layerstack.bucket_mode != "progress8kpabs" {
        return Err(format!(
            "--bucket-mode '{}' is not implemented (only 'progress8kpabs')",
            layerstack.bucket_mode
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
    // --ft-fp16-out は weight FP16 path の上に積む拡張なので --ft-fp16 を要求する。
    if layerstack.ft_fp16_out && !cli.ft_fp16 {
        return Err(
            "--ft-fp16-out requires --ft-fp16 (FT activation FP16 は weight FP16 \
                    path の拡張)"
                .into(),
        );
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
    if !cli.weight_decay.is_finite() || cli.weight_decay < 0.0 {
        return Err(format!(
            "--weight-decay must be finite and >= 0 (got {})",
            cli.weight_decay
        )
        .into());
    }
    // tiled dense matmul kernels (`dense_mm_fwd_bucket_tiled_l1` / `dense_mm_fwd_tiled_l1f`
    // / `dense_mm_bwd_input_tiled` / `dense_mm_bwd_weight_*_tiled_*`) は grid 計算が
    // `b / 16` で partial tile を切り捨てる前提なので、`b % 16 != 0` だと末尾 (b mod 16)
    // position の forward / backward が 走らず loss / gradient が corrupt する。`debug_assert!`
    // は release で消えるので CLI で early reject する。
    if !cli.batch_size.is_multiple_of(16) {
        return Err(format!(
            "--batch-size must be a multiple of 16 (got {}); tiled dense matmul kernels \
             require b % 16 == 0 (block_dim=256 × grid_dim=b/16)",
            cli.batch_size
        )
        .into());
    }
    // loss kernel の選択: --win-rate-model → loss_wrm、未指定 → loss_wdl。
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
        if !cli.wrm_target_offset.is_finite() {
            return Err(format!(
                "--wrm-target-offset must be finite (got {})",
                cli.wrm_target_offset
            )
            .into());
        }
        if !(cli.wrm_target_scaling.is_finite() && cli.wrm_target_scaling > 0.0) {
            return Err(format!(
                "--wrm-target-scaling must be finite and > 0 (got {})",
                cli.wrm_target_scaling
            )
            .into());
        }
        LossKind::Wrm {
            nnue2score: cli.wrm_nnue2score,
            in_scaling: cli.wrm_in_scaling,
            target_offset: cli.wrm_target_offset,
            target_scaling: cli.wrm_target_scaling,
        }
    } else {
        if !(cli.scale.is_finite() && cli.scale > 0.0) {
            return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
        }
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };
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
    let progress = match &layerstack.progress_coeff {
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
    println!("[train] CUDA context ready, building GpuTrainer (LayerStack)...");
    // workspace を batch_size 分で確保 (partial 末尾 batch は grow-only で対応)。
    let mut trainer = GpuTrainer::new(
        &ctx,
        cli.batch_size,
        layerstack.tf32,
        cli.ft_fp16,
        layerstack.ft_fp16_out,
        cli.fp16_opt_state,
        feature_set,
        cli.weight_decay,
    )?;
    // resume / init-from の処理 → 開始 superbatch と (resume なら) 親 run id を決める。
    let (resumed_superbatch, resume_parent_id): (Option<usize>, Option<String>) =
        if let Some(init) = &cli.init_from {
            println!(
                "[train] injecting pretrained weights from {} (optimizer state reset)",
                init.display()
            );
            let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
            let weights = LayerStackWeights::load_quantised(&mut reader, feature_set)?;
            trainer.load_layerstack_weights(&weights)?;
            (None, None)
        } else if let Some(ckpt) = &cli.resume {
            let (sb, parent_id) = trainer.load_raw_checkpoint(ckpt)?;
            println!(
                "[train] resuming from {} at superbatch {}",
                ckpt.display(),
                sb + 1
            );
            if parent_id.is_none() {
                println!(
                    "[train] note: {} predates producer run id embedding; \
                     experiment.json lineage.parent_id will be omitted",
                    ckpt.display()
                );
            }
            (Some(sb), parent_id)
        } else {
            (None, None)
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
        feature_set,
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
        test_data: cli.test_data.clone(),
        test_positions: cli.test_positions,
        compute_bucket: true,
    };

    // `--ft-fp16` の FP16 weight mirror を学習開始時の `ft_w` (init / --init-from /
    // --resume いずれか) と一度同期する。以降は optimizer が step ごとに維持する。
    trainer.sync_ft_w_h_mirror()?;

    let mut experiment = build_experiment_logger(
        cli,
        layerstack,
        feature_set,
        start_superbatch,
        resumed_superbatch,
        resume_parent_id,
        data,
    );
    println!("[train] experiment log: {}", experiment.path().display());

    let result = nnue_train::trainer::run(
        &mut trainer,
        data,
        &progress,
        &lr_scheduler,
        &wdl_scheduler,
        &cfg,
        Some(&mut experiment),
    );
    if result.is_err() {
        // run が error 終了したことを experiment.json に残す (status は "running"
        // のまま、results.interrupted を立てる)。`run` は正常終了時のみ
        // status を "completed" にする。
        experiment.mark_interrupted();
        if let Err(e) = experiment.write() {
            eprintln!(
                "[train] warning: failed to write experiment log {}: {e}",
                experiment.path().display()
            );
        }
    }
    result?;
    Ok(())
}

/// PSV 教師データ 1 局面のバイト数 (`shogi_format::PackedSfenValue` = `[u8; 40]`)。
const PSV_RECORD_BYTES: u64 = 40;

/// LayerStack network の architecture 記述子 (FT 1536 → L1 16 → L2 32、
/// progress8kpabs 9 bucket)。experiment.json `params.architecture` に記録する。
const LAYERSTACK_ARCHITECTURE: &str = "LayerStack-1536-16-32-9bucket";

/// 非有限な f32 (NaN / inf) を `0.0` に丸める。experiment.json の数値フィールド
/// に使う。JSON は非有限値を表現できず、混入すると serialise が丸ごと失敗して
/// 構造化ログが 1 件も書けなくなる。`--scale` は `--win-rate-model` 指定時に
/// CLI 側の finite 検証を経ないため防御する。
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// `path` の basename を `String` で返す。file_name が取れなければ path 全体の
/// 表示文字列で代替する。
fn file_basename(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// rshogi-nnue の git revision を best-effort で取得する。git が見つからない、
/// または git repository 外で実行された場合は `None`。working tree に未 commit
/// の変更があれば `-dirty` を付ける。
fn git_commit() -> Option<String> {
    let rev = std::process::Command::new("git")
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
    let dirty = std::process::Command::new("git")
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

/// 学習 run の experiment.json ロガーを CLI 設定から組み立てる。書き込み先は
/// `{--output}/experiments/{id}.json`、`id` は `{net_id}-{UTC 開始時刻}`。
fn build_experiment_logger(
    cli: &Cli,
    layerstack: &LayerstackArgs,
    feature_set: FeatureSetSpec,
    start_superbatch: usize,
    resumed_superbatch: Option<usize>,
    resume_parent_id: Option<String>,
    data: &Path,
) -> ExperimentLogger {
    let start_secs = nnue_train::experiment::now_epoch_secs();
    // id 末尾に process id を付ける。同一 net_id / output で複数プロセスが同一
    // 秒に開始しても (sweep / retry script 等)、pid が異なるため experiment.json
    // の書き込み先 path が衝突せず、incremental write の上書き喪失が起きない。
    let id = format!(
        "{}-{}-{}",
        cli.net_id,
        nnue_train::experiment::format_utc_compact(start_secs),
        std::process::id()
    );
    let name = cli.experiment_name.clone().unwrap_or_else(|| {
        if cli.resume.is_some() {
            format!("{} (resume @sb{start_superbatch})", cli.net_id)
        } else {
            cli.net_id.clone()
        }
    });

    let lineage = cli.resume.as_ref().map(|ckpt| Lineage {
        // resume 元 `*.ckpt` (format version 3+) に埋め込まれた親 run の
        // experiment.json `id`。version 1/2 の `*.ckpt` には無く `None` になり、
        // その resume run の lineage は checkpoint 参照のみになる。
        parent_id: resume_parent_id.clone(),
        resumed_from_checkpoint: file_basename(ckpt),
        resumed_from_superbatch: resumed_superbatch.unwrap_or(start_superbatch.saturating_sub(1)),
    });

    let is_wrm = cli.win_rate_model;
    let params = Params {
        architecture: LAYERSTACK_ARCHITECTURE.to_string(),
        feature_set: feature_set.canonical_name().to_string(),
        ft_in: feature_set.ft_in(),
        l0: FT_OUT,
        l1: L1_OUT,
        l2: L2_OUT,
        num_buckets: Some(NUM_BUCKETS),
        optimizer: cli.optimizer.clone(),
        bucket_mode: Some(layerstack.bucket_mode.clone()),
        activation: None,
        progress_coeff: layerstack.progress_coeff.as_deref().map(file_basename),
        lr: finite_or_zero(cli.lr),
        lr_gamma: finite_or_zero(cli.lr_gamma),
        lr_step: cli.lr_step.max(1),
        batch_size: cli.batch_size,
        batches_per_superbatch: cli.batches_per_superbatch,
        superbatches: cli.superbatches,
        start_superbatch,
        wdl: finite_or_zero(cli.wdl),
        scale: finite_or_zero(cli.scale),
        weight_decay: finite_or_zero(cli.weight_decay),
        qa: nnue_format::layerstack_weights::QA,
        qb: nnue_format::layerstack_weights::QB,
        loss_kind: if is_wrm { "wrm" } else { "sigmoid" }.to_string(),
        wrm_in_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_in_scaling)),
        wrm_nnue2score: is_wrm.then(|| finite_or_zero(cli.wrm_nnue2score)),
        wrm_target_offset: is_wrm.then(|| finite_or_zero(cli.wrm_target_offset)),
        wrm_target_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_target_scaling)),
        score_drop_abs: cli.score_drop_abs,
        init_from: cli.init_from.as_deref().map(file_basename),
        // test_data / test_positions は `--test-data` 指定時のみ記録する
        // (未指定 run の experiment.json では両フィールドとも省略)。
        test_data: cli.test_data.as_deref().map(file_basename),
        test_positions: cli.test_data.as_ref().map(|_| cli.test_positions),
        tf32: layerstack.tf32,
        ft_fp16: cli.ft_fp16,
        ft_fp16_out: layerstack.ft_fp16_out,
        fp16_opt_state: cli.fp16_opt_state,
        threads: cli.threads,
    };

    let dataset_positions = std::fs::metadata(data)
        .map(|m| m.len() / PSV_RECORD_BYTES)
        .unwrap_or(0);
    let data_info = DataInfo {
        name: file_basename(data),
        positions: dataset_positions,
        total_positions: 0,
        dataset_passes: 0.0,
    };

    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let json_path = cli.output.join("experiments").join(format!("{id}.json"));
    let doc = ExperimentDoc::new(
        id,
        name,
        start_secs,
        git_commit(),
        command,
        lineage,
        params,
        data_info,
    );
    ExperimentLogger::new(json_path, doc)
}

/// Simple アーキ用の experiment.json ロガーを CLI 設定から組み立てる。
/// LayerStack 用 [`build_experiment_logger`] と並ぶ Simple 用 helper で、
/// `Params` の bucket / progress / TF32 / FT-FP16 系フィールドは Simple では
/// 概念が無い (`bucket_mode` / `num_buckets` / `progress_coeff` は `None`、
/// `tf32` / `ft_fp16` / `ft_fp16_out` / `fp16_opt_state` は `false`)。
/// 量子化 multiplier (`qa` / `qb`) は活性化と `simple_weights` の固定値から決める。
#[allow(clippy::too_many_arguments)]
fn build_experiment_logger_simple(
    cli: &Cli,
    id: SimpleId,
    start_superbatch: usize,
    resumed_superbatch: Option<usize>,
    resume_parent_id: Option<String>,
    data: &Path,
) -> ExperimentLogger {
    let start_secs = nnue_train::experiment::now_epoch_secs();
    let net_id_compact = format!(
        "{}-{}-{}",
        cli.net_id,
        nnue_train::experiment::format_utc_compact(start_secs),
        std::process::id()
    );
    let name = cli.experiment_name.clone().unwrap_or_else(|| {
        if cli.resume.is_some() {
            format!("{} (resume @sb{start_superbatch})", cli.net_id)
        } else {
            cli.net_id.clone()
        }
    });

    let lineage = cli.resume.as_ref().map(|ckpt| Lineage {
        parent_id: resume_parent_id.clone(),
        resumed_from_checkpoint: file_basename(ckpt),
        resumed_from_superbatch: resumed_superbatch.unwrap_or(start_superbatch.saturating_sub(1)),
    });

    let architecture = format!(
        "simple-{}-{}x2-{}-{}-{}",
        id.feature_set.canonical_name(),
        id.ft_out,
        id.l1_out,
        id.l2_out,
        id.activation.canonical_name(),
    );

    let is_wrm = cli.win_rate_model;
    let params = Params {
        architecture,
        feature_set: id.feature_set.canonical_name().to_string(),
        ft_in: id.ft_in(),
        l0: id.ft_out,
        l1: id.l1_out,
        l2: id.l2_out,
        num_buckets: None,
        optimizer: cli.optimizer.clone(),
        bucket_mode: None,
        activation: Some(id.activation.canonical_name().to_string()),
        progress_coeff: None,
        lr: finite_or_zero(cli.lr),
        lr_gamma: finite_or_zero(cli.lr_gamma),
        lr_step: cli.lr_step.max(1),
        batch_size: cli.batch_size,
        batches_per_superbatch: cli.batches_per_superbatch,
        superbatches: cli.superbatches,
        start_superbatch,
        wdl: finite_or_zero(cli.wdl),
        scale: finite_or_zero(cli.scale),
        weight_decay: finite_or_zero(cli.weight_decay),
        qa: id.activation.qa(),
        qb: nnue_format::simple_weights::QB,
        loss_kind: if is_wrm { "wrm" } else { "sigmoid" }.to_string(),
        wrm_in_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_in_scaling)),
        wrm_nnue2score: is_wrm.then(|| finite_or_zero(cli.wrm_nnue2score)),
        wrm_target_offset: is_wrm.then(|| finite_or_zero(cli.wrm_target_offset)),
        wrm_target_scaling: is_wrm.then(|| finite_or_zero(cli.wrm_target_scaling)),
        score_drop_abs: cli.score_drop_abs,
        init_from: cli.init_from.as_deref().map(file_basename),
        test_data: cli.test_data.as_deref().map(file_basename),
        test_positions: cli.test_data.as_ref().map(|_| cli.test_positions),
        tf32: false,
        ft_fp16: false,
        ft_fp16_out: false,
        fp16_opt_state: false,
        threads: cli.threads,
    };

    let dataset_positions = std::fs::metadata(data)
        .map(|m| m.len() / PSV_RECORD_BYTES)
        .unwrap_or(0);
    let data_info = DataInfo {
        name: file_basename(data),
        positions: dataset_positions,
        total_positions: 0,
        dataset_passes: 0.0,
    };

    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let json_path = cli
        .output
        .join("experiments")
        .join(format!("{net_id_compact}.json"));
    let doc = ExperimentDoc::new(
        net_id_compact,
        name,
        start_secs,
        git_commit(),
        command,
        lineage,
        params,
        data_info,
    );
    ExperimentLogger::new(json_path, doc)
}

/// Simple アーキの層次元 preset 文字列 (`"<ft_out>x2-<l1_out>-<l2_out>"`) を
/// `(ft_out, l1_out, l2_out)` にパースする。bullet-shogi 由来の表記。
///
/// 例: `"256x2-32-32"` → `(256, 32, 32)`、`"1024x2-128-64"` → `(1024, 128, 64)`。
/// 形式不一致や非整数は `--arch` の不正値として `InvalidInput` で返す。
fn parse_simple_preset(s: &str) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let (head, tail) = s
        .split_once('-')
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!(
                "--arch '{s}' must look like '<ft_out>x2-<l1_out>-<l2_out>' (e.g. '256x2-32-32')"
            )
            .into()
        })?;
    let ft_out_str = head
        .strip_suffix("x2")
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': leading FT block must end with 'x2' (e.g. '256x2-32-32')").into()
        })?;
    let ft_out: usize = ft_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{ft_out_str}' is not a non-negative integer FT dimension")
                .into()
        })?;
    let (l1_out_str, l2_out_str) =
        tail.split_once('-')
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!(
                    "--arch '{s}': trailing block must look like '<l1_out>-<l2_out>' (got '{tail}')"
                )
                .into()
            })?;
    let l1_out: usize = l1_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{l1_out_str}' is not a non-negative integer L1 dimension")
                .into()
        })?;
    let l2_out: usize = l2_out_str
        .parse()
        .map_err(|_| -> Box<dyn std::error::Error> {
            format!("--arch '{s}': '{l2_out_str}' is not a non-negative integer L2 dimension")
                .into()
        })?;
    Ok((ft_out, l1_out, l2_out))
}

/// Simple 4 層アーキの training driver。`run_training` から `ArchCommand::Simple`
/// 分岐で呼ばれる。LayerStack 側 (`run_training` 本体) と並ぶ単独 entrypoint で、
/// trainer 構築・init_from / resume・lr / wdl スケジューラ・superbatch loop は
/// 同じ `nnue_train::trainer::run` driver を使う。
fn run_simple_training(
    cli: &Cli,
    simple_args: &SimpleArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = cli
        .data
        .as_ref()
        .expect("run_simple_training called with --data");

    let feature_set = FeatureSet::from_canonical_name(&cli.feature_set)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            let names: Vec<&str> = FeatureSet::ALL
                .iter()
                .map(|fs| fs.canonical_name())
                .collect();
            format!(
                "--feature-set '{}' is not a known feature set (expected one of: {})",
                cli.feature_set,
                names.join(", ")
            )
            .into()
        })?
        .spec();

    if !cli.optimizer.eq_ignore_ascii_case("ranger") {
        return Err(format!(
            "--optimizer '{}' is not implemented (only 'ranger')",
            cli.optimizer
        )
        .into());
    }
    // `--ft-fp16-out` は FP16 weight mirror 経路の上に積む拡張で、`--ft-fp16` を要求する。
    if simple_args.ft_fp16_out && !cli.ft_fp16 {
        return Err(
            "--ft-fp16-out requires --ft-fp16 (FT activation FP16 は weight FP16 \
             path の拡張)"
                .into(),
        );
    }
    if !(cli.lr.is_finite() && cli.lr > 0.0) {
        return Err(format!("--lr must be finite and > 0 (got {})", cli.lr).into());
    }
    if !cli.lr_gamma.is_finite() || cli.lr_gamma <= 0.0 {
        return Err(format!("--lr-gamma must be finite and > 0 (got {})", cli.lr_gamma).into());
    }
    if !cli.wdl.is_finite() || !(0.0..=1.0).contains(&cli.wdl) {
        return Err(format!("--wdl must be finite and in [0.0, 1.0] (got {})", cli.wdl).into());
    }
    if !cli.weight_decay.is_finite() || cli.weight_decay < 0.0 {
        return Err(format!(
            "--weight-decay must be finite and >= 0 (got {})",
            cli.weight_decay
        )
        .into());
    }
    if !cli.batch_size.is_multiple_of(16) {
        return Err(format!(
            "--batch-size must be a multiple of 16 (got {})",
            cli.batch_size
        )
        .into());
    }
    if cli.threads == 0 {
        return Err("--threads must be >= 1".into());
    }
    if cli.init_from.is_some() && cli.resume.is_some() {
        return Err("--init-from and --resume are mutually exclusive".into());
    }
    if cli.superbatches == 0 {
        return Err("--superbatches must be >= 1".into());
    }
    if let Some(0) = cli.keep_checkpoints {
        return Err(
            "--keep-checkpoints must be >= 1 when set (0 would delete every raw checkpoint)".into(),
        );
    }

    // 層次元の決定: --arch preset + --l1 / --l2 / --l3 override。
    let (preset_ft_out, preset_l1_out, preset_l2_out) = parse_simple_preset(&simple_args.arch)?;
    let ft_out = simple_args.l1.unwrap_or(preset_ft_out);
    let l1_out = simple_args.l2.unwrap_or(preset_l1_out);
    let l2_out = simple_args.l3.unwrap_or(preset_l2_out);
    let activation = SimpleActivation::from_canonical_name(&simple_args.activation).ok_or_else(
        || -> Box<dyn std::error::Error> {
            format!(
                "--activation '{}' is not implemented (expected one of: crelu, screlu)",
                simple_args.activation
            )
            .into()
        },
    )?;
    let id = SimpleId {
        feature_set,
        activation,
        ft_out,
        l1_out,
        l2_out,
    };

    // `--ft-fp16-out` 経路は現在 CReLU 限定 (SCReLU 用 fused kernel は未提供)。
    // 必要になれば `simple_bias_act_fwd_fp16_in_screlu` 等を追加して許可する。
    if simple_args.ft_fp16_out && activation != SimpleActivation::CReLU {
        return Err(
            "--ft-fp16-out currently requires --activation crelu (Simple SCReLU FP16 path \
             is not implemented)"
                .into(),
        );
    }

    // Simple は loss kind に関わらず `cli.scale` を量子化 `fv_scale` の算出で参照
    // するため、WRM 経路でも finite / 正値を要求する (LayerStack は WRM 時に scale
    // を参照しないので sigmoid 経路でのみ検証していた)。
    if !(cli.scale.is_finite() && cli.scale > 0.0) {
        return Err(format!("--scale must be finite and > 0 (got {})", cli.scale).into());
    }
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
        if !cli.wrm_target_offset.is_finite() {
            return Err(format!(
                "--wrm-target-offset must be finite (got {})",
                cli.wrm_target_offset
            )
            .into());
        }
        if !(cli.wrm_target_scaling.is_finite() && cli.wrm_target_scaling > 0.0) {
            return Err(format!(
                "--wrm-target-scaling must be finite and > 0 (got {})",
                cli.wrm_target_scaling
            )
            .into());
        }
        LossKind::Wrm {
            nnue2score: cli.wrm_nnue2score,
            in_scaling: cli.wrm_in_scaling,
            target_offset: cli.wrm_target_offset,
            target_scaling: cli.wrm_target_scaling,
        }
    } else {
        LossKind::Sigmoid {
            scale: 1.0 / cli.scale,
        }
    };

    std::fs::create_dir_all(&cli.output)?;

    // Simple は bucket-aware progress を持たない: dataloader に渡す
    // `ShogiProgressKPAbs` は zero-weight default (全 position が bucket 4)、
    // TrainerBackend::train_step 内で bucket index は無視される。
    let progress = ShogiProgressKPAbs;

    let ctx = CudaContext::new(0)?;
    println!("[train] CUDA context ready, building SimpleGpuTrainer...");
    // 推論側 evaluation scale: round(QA * QB / 学習 scale)。`cli.scale` は前段で
    // 有限・正値を保証済。
    let fv_scale =
        ((id.activation.qa() * nnue_format::simple_weights::QB) as f32 / cli.scale).round() as i32;
    let mut trainer = SimpleGpuTrainer::new(
        &ctx,
        cli.batch_size,
        id,
        cli.weight_decay,
        fv_scale,
        cli.ft_fp16,
        simple_args.ft_fp16_out,
        cli.fp16_opt_state,
        simple_args.tf32,
    )?;

    let (resumed_superbatch, resume_parent_id): (Option<usize>, Option<String>) =
        if let Some(init) = &cli.init_from {
            println!(
                "[train] injecting pretrained weights from {} (optimizer state reset)",
                init.display()
            );
            let mut reader = std::io::BufReader::new(std::fs::File::open(init)?);
            let weights = SimpleWeights::load(&mut reader, id)?;
            trainer.load_simple_weights(&weights)?;
            (None, None)
        } else if let Some(ckpt) = &cli.resume {
            let (sb, parent_id) = trainer.load_raw_checkpoint(ckpt)?;
            println!(
                "[train] resuming from {} at superbatch {}",
                ckpt.display(),
                sb + 1
            );
            (Some(sb), parent_id)
        } else {
            (None, None)
        };

    // `--ft-fp16` の FP16 weight mirror を学習開始時の `ft_w` (init / --init-from /
    // --resume いずれか) と一度同期する。以降は optimizer が step ごとに維持する。
    // `--ft-fp16` 未指定なら no-op。
    trainer.sync_ft_w_h_mirror()?;

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
            "--start-superbatch {start_superbatch} > --superbatches {} (nothing to train)",
            cli.superbatches
        )
        .into());
    }

    let lr_scheduler = StepLR {
        start: cli.lr,
        gamma: cli.lr_gamma,
        step: cli.lr_step.max(1),
    };
    let wdl_scheduler = ConstantWDL { value: cli.wdl };
    let cfg = TrainingConfig {
        net_id: cli.net_id.clone(),
        feature_set,
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
        test_data: cli.test_data.clone(),
        test_positions: cli.test_positions,
        compute_bucket: false,
    };

    let mut experiment = build_experiment_logger_simple(
        cli,
        id,
        start_superbatch,
        resumed_superbatch,
        resume_parent_id,
        data,
    );
    println!("[train] experiment log: {}", experiment.path().display());

    let result = nnue_train::trainer::run(
        &mut trainer,
        data,
        &progress,
        &lr_scheduler,
        &wdl_scheduler,
        &cfg,
        Some(&mut experiment),
    );
    if result.is_err() {
        experiment.mark_interrupted();
        if let Err(e) = experiment.write() {
            eprintln!(
                "[train] warning: failed to write experiment log {}: {e}",
                experiment.path().display()
            );
        }
    }
    result?;
    Ok(())
}

// ===========================================================================
// SimpleGpuTrainer — bucket 無し 4 層 Simple アーキの GPU トレーナ
// ===========================================================================
//
// LayerStack 用 `GpuTrainer` (本 file 上方) と並ぶ、もう一方のアーキの host driver。
// `SimpleId` (feature set / 活性化 / ft_out / l1_out / l2_out) で形が決まる 8
// weight 群 (FT/L1/L2/L3 各 w・b) を device buffer で保持し、`forward` で 1 batch
// の FT → bias add → 活性化 → concat → L1/L2/L3 dense → loss を既存 kernel の
// 合成として走らせる。
//
// 本段階は **forward + smoke 専用** で、backward / optimizer / checkpoint /
// `TrainerBackend` 実装は持たない。weight は xorshift 初期化のみ。

/// Simple アーキ専用の forward 用 workspace。中間 activation buffer (全 f32) と
/// 入力 buffer (sparse idx + score/wdl) を固定 batch 容量で `new` 時に確保する。
struct SimpleGpuWorkspace {
    /// `new` 時の固定 batch 容量 (`forward` 実行時にこれ以下を要求)。
    len_batch: usize,
    /// FT 入力次元 (`id.feature_set.ft_in()`)、kernel launch 引数で使う。
    ft_in: usize,
    /// 1 perspective あたりの active feature 上限 (`id.feature_set.max_active()`)。
    max_active: usize,

    // -- forward 中間 activation (すべて f32) --
    /// sparse_ft_forward の stm 出力 (`b × ft_out`)。bias add は in-place でここに書き戻す。
    ft_stm_out: DeviceBuffer<f32>,
    /// 同 nstm。
    ft_nstm_out: DeviceBuffer<f32>,
    /// stm bias add + 活性化後 (`b × ft_out`)、concat 元の tmp。
    ft_stm_acted: DeviceBuffer<f32>,
    /// 同 nstm。
    ft_nstm_acted: DeviceBuffer<f32>,
    /// concat(stm_acted, nstm_acted) = `b × (2*ft_out)`。L1 dense の入力。
    combined: DeviceBuffer<f32>,
    /// L1 dense 出力 (pre-activation、`b × l1_out`)。
    l1_pre: DeviceBuffer<f32>,
    /// L1 活性化後 (`b × l1_out`)、L2 dense の入力。
    l1_acted: DeviceBuffer<f32>,
    /// L2 dense 出力 (`b × l2_out`)。
    l2_pre: DeviceBuffer<f32>,
    /// L2 活性化後 (`b × l2_out`)、L3 dense の入力。
    l2_acted: DeviceBuffer<f32>,
    /// L3 dense 出力 = ネットワーク 1 次元出力 (`b`)。
    net_output: DeviceBuffer<f32>,
    /// loss kernel が書く dnet (= dy/d net_output、`b`)。backward の起点。
    dy_net_output: DeviceBuffer<f32>,

    // -- backward gradient buffer (forward と対称配置) --
    /// L2 活性化後への grad (`b × l2_out`)、`dense_mm_bwd_input` で L3 から伝播。
    dl2_acted: DeviceBuffer<f32>,
    /// L2 dense 出力への grad (`b × l2_out`)、活性化逆伝播の出力。
    dl2_pre: DeviceBuffer<f32>,
    /// L1 活性化後への grad (`b × l1_out`)。
    dl1_acted: DeviceBuffer<f32>,
    /// L1 dense 出力への grad (`b × l1_out`)。
    dl1_pre: DeviceBuffer<f32>,
    /// concat 後 (`b × 2*ft_out`) への grad。L1 dense backward の入力 grad 先。
    dcombined: DeviceBuffer<f32>,
    /// stm 活性化後への grad (`b × ft_out`)、concat 逆 (`slice_extract_2d`) の出力。
    dft_stm_acted: DeviceBuffer<f32>,
    /// 同 nstm。
    dft_nstm_acted: DeviceBuffer<f32>,
    /// stm FT 出力 (= bias add 直後 / 活性化直前) への grad (`b × ft_out`)。
    /// `sparse_ft_backward` の入力 grad + `ft_b` bias grad の reduce 対象。
    dft_stm_out: DeviceBuffer<f32>,
    /// 同 nstm。
    dft_nstm_out: DeviceBuffer<f32>,

    // -- `--ft-fp16-out` 経路の f16 buffer (ft_fp16_out が true のときだけ Some) --
    /// `sparse_ft_forward_fp16_out` の出力 (`b × ft_out`、f16、bias 未加算)。
    /// 後続 [`simple_bias_act_fwd_fp16_in_crelu`] / [`simple_act_grad_to_fp16_crelu_with_scale`]
    /// が bias を別 buffer から read して加算する。
    ft_stm_out_h: Option<DeviceBuffer<f16>>,
    /// 同 nstm。
    ft_nstm_out_h: Option<DeviceBuffer<f16>>,
    /// FT pre-activation gradient (`b × ft_out`、f16、loss scaling 済)。
    /// [`simple_act_grad_to_fp16_crelu_with_scale`] が書き、[`simple_bias_grad_fp16`]
    /// と [`simple_sparse_ft_backward_fp16`] が `dft_inv_scale` で打ち消して accumulate。
    dft_stm_out_h: Option<DeviceBuffer<f16>>,
    /// 同 nstm。
    dft_nstm_out_h: Option<DeviceBuffer<f16>>,

    // -- inverse-index sparse_ft_backward scratch (`build_feature_counts` →
    //    `exclusive_prefix_sum_small` → `scatter_positions` → `gather_and_sum_per_feature_*`
    //    pipeline 用)。per-feature gather で `dft_*_out` の DRAM read を 1 perspective につき
    //    各 (feature, ft_out) cell ちょうど 1 回に抑え、global atomic 数も `b * ft_out *
    //    max_active` から `b * max_active` (histogram + scatter) まで圧縮する。サイズは
    //    feature set ごとに固定 (`ft_in` / `max_active` で決まる)。
    /// per-feature 出現回数 histogram (`ft_in`、`build_feature_counts` で atomic build)。
    feat_counts: DeviceBuffer<u32>,
    /// `feat_counts` の exclusive prefix sum (`ft_in + 1`、`exclusive_prefix_sum_small` で構築)。
    feat_offsets: DeviceBuffer<u32>,
    /// `scatter_positions` 中の per-feature 書き込み位置カウンタ (`ft_in`、atomic incremented)。
    feat_write_ctr: DeviceBuffer<u32>,
    /// 各 feature 出現位置の sorted ストレージ (`batch * max_active`、`scatter_positions` が書く)。
    feat_positions: DeviceBuffer<u32>,

    // -- 入力 buffer (active / back ペア。`InputUploadRing` の double-buffer 規約) --
    /// stm sparse index (`b × max_active`、無効 slot は -1)。active 側 = 現 step が forward
    /// で read する物理 buffer。`step` 冒頭で back と `mem::swap` してから ring が back
    /// (旧 active = 直前 step が読んでいない側) に async H2D する。
    stm_idx_dev: DeviceBuffer<i32>,
    /// 同 nstm。
    nstm_idx_dev: DeviceBuffer<i32>,
    /// target score (`b`、centipawn)。
    score_dev: DeviceBuffer<f32>,
    /// target wdl (`b`、0.0/0.5/1.0)。
    wdl_dev: DeviceBuffer<f32>,
    /// 同上、back 側物理 buffer。次 step の H2D 先 (`step` 冒頭で `mem::swap` で active
    /// へ昇格)。直前 step の compute が読んでいる active と物理分離されるため、H2D は
    /// 直前 step の compute と並走しても buffer 競合が起きない。
    stm_idx_dev_back: DeviceBuffer<i32>,
    nstm_idx_dev_back: DeviceBuffer<i32>,
    score_dev_back: DeviceBuffer<f32>,
    wdl_dev_back: DeviceBuffer<f32>,
}

impl SimpleGpuWorkspace {
    fn new(
        stream: &CudaStream,
        batch: usize,
        id: SimpleId,
        ft_fp16_out: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ft_in = id.ft_in();
        let max_active = id.feature_set.max_active();
        let ft_out = id.ft_out;
        let l1_out = id.l1_out;
        let l2_out = id.l2_out;
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(stream, n).map_err(Into::into)
        };
        let alloc_h = |on: bool| -> Result<Option<DeviceBuffer<f16>>, Box<dyn std::error::Error>> {
            if on {
                Ok(Some(DeviceBuffer::<f16>::zeroed(stream, batch * ft_out)?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            len_batch: batch,
            ft_in,
            max_active,
            ft_stm_out: z(batch * ft_out)?,
            ft_nstm_out: z(batch * ft_out)?,
            ft_stm_acted: z(batch * ft_out)?,
            ft_nstm_acted: z(batch * ft_out)?,
            combined: z(batch * 2 * ft_out)?,
            l1_pre: z(batch * l1_out)?,
            l1_acted: z(batch * l1_out)?,
            l2_pre: z(batch * l2_out)?,
            l2_acted: z(batch * l2_out)?,
            net_output: z(batch)?,
            dy_net_output: z(batch)?,
            dl2_acted: z(batch * l2_out)?,
            dl2_pre: z(batch * l2_out)?,
            dl1_acted: z(batch * l1_out)?,
            dl1_pre: z(batch * l1_out)?,
            dcombined: z(batch * 2 * ft_out)?,
            dft_stm_acted: z(batch * ft_out)?,
            dft_nstm_acted: z(batch * ft_out)?,
            dft_stm_out: z(batch * ft_out)?,
            dft_nstm_out: z(batch * ft_out)?,
            ft_stm_out_h: alloc_h(ft_fp16_out)?,
            ft_nstm_out_h: alloc_h(ft_fp16_out)?,
            dft_stm_out_h: alloc_h(ft_fp16_out)?,
            dft_nstm_out_h: alloc_h(ft_fp16_out)?,
            feat_counts: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_offsets: DeviceBuffer::<u32>::zeroed(stream, ft_in + 1)?,
            feat_write_ctr: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_positions: DeviceBuffer::<u32>::zeroed(stream, batch * max_active)?,
            stm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            score_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            stm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            score_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
        })
    }

    /// `new` 時の `len_batch` 容量に `batch` が収まることを検証する
    /// (`GpuWorkspace::check_batch_capacity` と同じ規約: 固定 batch 前提で
    /// 容量超過は error)。
    fn check_batch_capacity(&self, batch: usize) -> Result<(), Box<dyn std::error::Error>> {
        if batch > self.len_batch {
            return Err(format!(
                "SimpleGpuTrainer: batch {batch} exceeds workspace capacity {} \
                 (re-construct SimpleGpuTrainer with a larger batch_size)",
                self.len_batch
            )
            .into());
        }
        Ok(())
    }
}

/// Simple 4 層アーキ用 GPU トレーナ。LayerStack 用 `GpuTrainer` と並ぶもう一方の
/// アーキの host driver。1 batch の forward → loss → backward → Ranger optimizer step
/// を 8 weight group ({ft, l1, l2, l3} × {w, b}) について実行する。
///
/// `--ft-fp16` / `--ft-fp16-out` / `--fp16-opt-state` / `--tf32` の risky 精度系 flag は
/// LayerStack と同形で opt-in (default OFF で FP32 bit-identical、ON で risky 最適化)。
///
/// L1 dense (B × 2*ft_out × l1_out) は forward / bwd_input / bwd_weight 3 経路を cuBLAS
/// Sgemm に乗せる。default math mode は `CUBLAS_DEFAULT_MATH` (純 FP32、TC 不使用)、
/// `--tf32` 指定で `CUBLAS_TF32_TENSOR_OP_MATH` (Ampere+ TC、仮数 10-bit 丸め)。L2 / L3
/// は次元が小さく untiled `dense_mm_*` で残す。FT は専用 `sparse_ft_*` kernel、活性化と
/// loss は固有 kernel。
struct SimpleGpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,
    /// L1 dense (FP32) 用 cuBLAS handle (TF32 不使用、`CUBLAS_DEFAULT_MATH`)。
    /// stream に bind 済で同一 stream 内 in-order 実行。
    cublas: CublasHandle,
    /// `--ft-fp16` opt-in flag。`true` の間 forward は `sparse_ft_forward_fp16`
    /// (FP16 weight read) を使い、optimizer は `radam_step_fp16_mirror` /
    /// `ranger_lookahead_lerp_fp16_mirror` で `ft_w` 更新と同時に `ft_w_h` を書く。
    /// FP32 master `ft_w` / 量子化 checkpoint byte layout は不変。
    ft_fp16: bool,
    /// `ft_w` の FP16 mirror。`ft_fp16` が `true` のときだけ `Some`。`sparse_ft_forward_fp16`
    /// の weight 入力で、`radam_step_fp16_mirror` が optimizer step ごとに同期する。
    /// 学習開始時の初期同期は [`sync_ft_w_h_mirror`](Self::sync_ft_w_h_mirror) で。
    ft_w_h: Option<DeviceBuffer<f16>>,
    /// `--ft-fp16-out` opt-in flag。`true` の間 forward は `sparse_ft_forward_fp16_out`
    /// (FP16 weight + FP16 出力) と [`simple_bias_act_fwd_fp16_in_crelu`] を使い、
    /// backward は [`simple_act_grad_to_fp16_crelu_with_scale`] / [`simple_bias_grad_fp16`]
    /// / [`simple_sparse_ft_backward_fp16`] で dft も f16 で持つ。`ft_fp16` を要求し、
    /// CLI 検証段階で activation = CReLU を要求する。dft は loss scaling で f16 域に
    /// 持ち上げる ([`FT_DFT_FP16_BASE_SCALE`] × batch、±65504 clamp 付き)。
    ft_fp16_out: bool,
    /// `--fp16-opt-state` opt-in flag。`true` の間 `ft_w_m` / `ft_w_v` を [`MomentBuf::F16`]
    /// で確保し、optimizer step は [`radam_step_f16state`] / [`radam_step_f16state_mirror`]
    /// で動かす。FT 以外の moment は変更なし。raw checkpoint format は不変 (真値 f32 で
    /// 書き出し、resume 時に当該 run の精度へ再 quantize する)。
    fp16_opt_state: bool,

    // -- weight (FP32) --
    /// FT 重み (`ft_in × ft_out`、feature-major: `ft_w[feat*ft_out + out]`)。
    /// `sparse_ft_forward` の weight layout と一致する。
    ft_w: DeviceBuffer<f32>,
    /// FT bias (`ft_out`、stm/nstm 共有)。
    ft_b: DeviceBuffer<f32>,
    /// L1 dense 重み (`(2*ft_out) × l1_out`、in-major: `l1_w[in*l1_out + out]`)。
    /// `dense_mm_fwd` の weight layout (`w[k*out_dim+oi]`) と一致する。
    l1_w: DeviceBuffer<f32>,
    /// L1 dense bias (`l1_out`)。
    l1_b: DeviceBuffer<f32>,
    /// L2 dense 重み (`l1_out × l2_out`、in-major)。
    l2_w: DeviceBuffer<f32>,
    /// L2 dense bias (`l2_out`)。
    l2_b: DeviceBuffer<f32>,
    /// L3 dense 重み (`l2_out × 1`、in-major)。
    l3_w: DeviceBuffer<f32>,
    /// L3 dense bias (1 要素)。
    l3_b: DeviceBuffer<f32>,

    // -- gradient buffer (各 weight と同 shape、f32) --
    ft_w_grad: DeviceBuffer<f32>,
    ft_b_grad: DeviceBuffer<f32>,
    l1_w_grad: DeviceBuffer<f32>,
    l1_b_grad: DeviceBuffer<f32>,
    l2_w_grad: DeviceBuffer<f32>,
    l2_b_grad: DeviceBuffer<f32>,
    l3_w_grad: DeviceBuffer<f32>,
    l3_b_grad: DeviceBuffer<f32>,

    // -- Ranger optimizer state (RAdam 1st/2nd moment + Lookahead slow weight、各 weight と同 shape) --
    /// FT Ranger 1st/2nd moment。既定 `f32`、`--fp16-opt-state` で `f16` ([`MomentBuf`])。
    /// 他 7 group の moment buffer は小さく `f16` 化の意味が無いので `f32` のまま。
    ft_w_m: MomentBuf,
    ft_w_v: MomentBuf,
    ft_w_slow: DeviceBuffer<f32>,
    ft_b_m: DeviceBuffer<f32>,
    ft_b_v: DeviceBuffer<f32>,
    ft_b_slow: DeviceBuffer<f32>,
    l1_w_m: DeviceBuffer<f32>,
    l1_w_v: DeviceBuffer<f32>,
    l1_w_slow: DeviceBuffer<f32>,
    l1_b_m: DeviceBuffer<f32>,
    l1_b_v: DeviceBuffer<f32>,
    l1_b_slow: DeviceBuffer<f32>,
    l2_w_m: DeviceBuffer<f32>,
    l2_w_v: DeviceBuffer<f32>,
    l2_w_slow: DeviceBuffer<f32>,
    l2_b_m: DeviceBuffer<f32>,
    l2_b_v: DeviceBuffer<f32>,
    l2_b_slow: DeviceBuffer<f32>,
    l3_w_m: DeviceBuffer<f32>,
    l3_w_v: DeviceBuffer<f32>,
    l3_w_slow: DeviceBuffer<f32>,
    l3_b_m: DeviceBuffer<f32>,
    l3_b_v: DeviceBuffer<f32>,
    l3_b_slow: DeviceBuffer<f32>,

    ws: SimpleGpuWorkspace,
    /// loss kernel が atomic add する Σerr² (f64、1 要素)。
    loss_acc: DeviceBuffer<f64>,
    /// `step()` 末の `loss_acc` 同期読みを 1-step lag な async D2H に置換する pinned
    /// host ring。host が `stream.synchronize` 待ち無しで次 batch の launch を発行できる
    /// ようになる。sb 末で [`TrainerBackend::flush_pending_loss`] が drain する
    /// (default `0.0` を本 trainer は override する)。`forward` / `validate` の同期
    /// read 経路は ring を介さず loss_acc を直接読む。
    loss_ring: AsyncLossRing,
    /// `step()` 先頭の入力 H2D (`stm/nstm idx` + `score/wdl` の 4 buffer、Simple は
    /// bucket 無し) を専用 copy stream で直前 step の compute と overlap させる ring。
    /// [`AsyncLossRing`] による host run-ahead と組合せて compute と H2D を同時実行する。
    /// `forward` / `validate` の同期 read 経路は ring を介さず直接 H2D する (1-shot で
    /// 後続 backward / optimizer が無いため overlap 余地が薄い)。
    input_ring: InputUploadRing,
    /// このトレーナのアーキ identity (feature set / 活性化 / 層次元)。
    id: SimpleId,
    /// Ranger lookahead step counter。`RANGER_K` の倍数で lerp する。
    step_count: u64,
    /// Ranger optimizer の weight decay 係数 (`radam_step` 引数)。
    weight_decay: f32,
    /// 推論時の評価値スケール (`round(QA * QB / 学習 scale)`)。量子化 checkpoint
    /// 出力の arch 文字列に書く (`SimpleWeights::fv_scale`)。
    fv_scale: i32,
}

impl Drop for SimpleGpuTrainer {
    fn drop(&mut self) {
        // device buffer 解放前に compute / copy 両 stream の in-flight 操作を排出する
        // (GpuTrainer と同じ規約: field drop 順による race を回避)。`input_ring` の
        // copy stream H2D が `ws` の入力 buffer を write 中の解放を防ぐため。
        let _ = self.stream.synchronize();
        let _ = self.input_ring.copy_stream.synchronize();
    }
}

impl SimpleGpuTrainer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch_size: usize,
        id: SimpleId,
        weight_decay: f32,
        fv_scale: i32,
        ft_fp16: bool,
        ft_fp16_out: bool,
        fp16_opt_state: bool,
        tf32: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // `--ft-fp16-out` は weight FP16 path の拡張なので `--ft-fp16` を含意する。CLI
        // 検証で reject 済だが、`SimpleGpuTrainer::new` を直接呼ぶ smoke / test 経路でも
        // invariant を成立させる。
        debug_assert!(!ft_fp16_out || ft_fp16, "ft_fp16_out requires ft_fp16");
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "nnue_train")?;
        // `tf32` (CLI の `--tf32`) で cuBLAS math mode 切替。default OFF は
        // `CUBLAS_DEFAULT_MATH` (純 FP32 path、L1/L2/L3 dense Sgemm bit-identical)。
        // `true` で `CUBLAS_TF32_TENSOR_OP_MATH` を有効化し Ampere+ TC を使う (Sgemm 高速化、
        // 仮数 23-bit → 10-bit 丸めで数値挙動が変わる、棋力 risk opt-in)。LayerStack
        // `--tf32` と同方針。
        let cublas = CublasHandle::new(&stream, tf32)?;

        let ft_in = id.ft_in();
        let ft_out = id.ft_out;
        let l1_out = id.l1_out;
        let l2_out = id.l2_out;
        // sparse_ft_forward は 1 thread = 4 row なので ft_out が 4 の倍数必須。
        // Simple の preset (256/512/1024) は全部 4 の倍数だが、`--l1` override で
        // 4 の倍数でない値が来る可能性があるので early reject する。
        if !ft_out.is_multiple_of(4) {
            return Err(format!(
                "SimpleGpuTrainer: ft_out {ft_out} must be a multiple of 4 \
                 (sparse_ft_forward processes 4 rows per thread)"
            )
            .into());
        }

        // small random init。group ごとに seed を変えて weight が同一値で潰れない
        // ようにする (forward の合成 layer 構造を踏むため)。
        let ft_w_h = xorshift_init(0x5071_e001, ft_in * ft_out, 0.01);
        let ft_b_h = xorshift_init(0x5071_e002, ft_out, 0.01);
        let l1_w_h = xorshift_init(0x5071_e003, 2 * ft_out * l1_out, 0.01);
        let l1_b_h = xorshift_init(0x5071_e004, l1_out, 0.01);
        let l2_w_h = xorshift_init(0x5071_e005, l1_out * l2_out, 0.01);
        let l2_b_h = xorshift_init(0x5071_e006, l2_out, 0.01);
        let l3_w_h = xorshift_init(0x5071_e007, l2_out, 0.01);
        let l3_b_h = xorshift_init(0x5071_e008, 1, 0.01);

        let batch = batch_size.max(1);
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(&stream, n).map_err(Into::into)
        };
        let ft_w_n = ft_in * ft_out;
        let ft_b_n = ft_out;
        let l1_w_n = 2 * ft_out * l1_out;
        let l1_b_n = l1_out;
        let l2_w_n = l1_out * l2_out;
        let l2_b_n = l2_out;
        let l3_w_n = l2_out;
        let l3_b_n = 1;
        // Lookahead slow weight は学習開始時 weights と同値で初期化する (bullet
        // `Ranger` の初期 `slow_param ← param` 規約)。
        let ft_w = DeviceBuffer::from_host(&stream, &ft_w_h)?;
        let ft_b = DeviceBuffer::from_host(&stream, &ft_b_h)?;
        let l1_w = DeviceBuffer::from_host(&stream, &l1_w_h)?;
        let l1_b = DeviceBuffer::from_host(&stream, &l1_b_h)?;
        let l2_w = DeviceBuffer::from_host(&stream, &l2_w_h)?;
        let l2_b = DeviceBuffer::from_host(&stream, &l2_b_h)?;
        let l3_w = DeviceBuffer::from_host(&stream, &l3_w_h)?;
        let l3_b = DeviceBuffer::from_host(&stream, &l3_b_h)?;
        let ft_w_slow = DeviceBuffer::from_host(&stream, &ft_w_h)?;
        let ft_b_slow = DeviceBuffer::from_host(&stream, &ft_b_h)?;
        let l1_w_slow = DeviceBuffer::from_host(&stream, &l1_w_h)?;
        let l1_b_slow = DeviceBuffer::from_host(&stream, &l1_b_h)?;
        let l2_w_slow = DeviceBuffer::from_host(&stream, &l2_w_h)?;
        let l2_b_slow = DeviceBuffer::from_host(&stream, &l2_b_h)?;
        let l3_w_slow = DeviceBuffer::from_host(&stream, &l3_w_h)?;
        let l3_b_slow = DeviceBuffer::from_host(&stream, &l3_b_h)?;
        Ok(Self {
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
            ft_w_grad: z(ft_w_n)?,
            ft_b_grad: z(ft_b_n)?,
            l1_w_grad: z(l1_w_n)?,
            l1_b_grad: z(l1_b_n)?,
            l2_w_grad: z(l2_w_n)?,
            l2_b_grad: z(l2_b_n)?,
            l3_w_grad: z(l3_w_n)?,
            l3_b_grad: z(l3_b_n)?,
            ft_w_m: MomentBuf::zeroed(&stream, ft_w_n, fp16_opt_state)?,
            ft_w_v: MomentBuf::zeroed(&stream, ft_w_n, fp16_opt_state)?,
            ft_w_slow,
            ft_b_m: z(ft_b_n)?,
            ft_b_v: z(ft_b_n)?,
            ft_b_slow,
            l1_w_m: z(l1_w_n)?,
            l1_w_v: z(l1_w_n)?,
            l1_w_slow,
            l1_b_m: z(l1_b_n)?,
            l1_b_v: z(l1_b_n)?,
            l1_b_slow,
            l2_w_m: z(l2_w_n)?,
            l2_w_v: z(l2_w_n)?,
            l2_w_slow,
            l2_b_m: z(l2_b_n)?,
            l2_b_v: z(l2_b_n)?,
            l2_b_slow,
            l3_w_m: z(l3_w_n)?,
            l3_w_v: z(l3_w_n)?,
            l3_w_slow,
            l3_b_m: z(l3_b_n)?,
            l3_b_v: z(l3_b_n)?,
            l3_b_slow,
            ws: SimpleGpuWorkspace::new(&stream, batch, id, ft_fp16_out)?,
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            loss_ring: AsyncLossRing::new(ctx)?,
            input_ring: InputUploadRing::new_simple(ctx, batch, id.feature_set.max_active())?,
            ft_w_h: if ft_fp16 {
                Some(DeviceBuffer::<f16>::zeroed(&stream, ft_w_n)?)
            } else {
                None
            },
            stream,
            module,
            id,
            step_count: 0,
            weight_decay,
            fv_scale,
            cublas,
            ft_fp16,
            ft_fp16_out,
            fp16_opt_state,
        })
    }

    /// 学習開始時の `ft_w_h` 初期同期。`ft_w_h` は `new` で zeroed 確保、optimizer
    /// (`radam_step_fp16_mirror` / `ranger_lookahead_lerp_fp16_mirror`) が以後 step ごと
    /// に維持するが、最初の forward の前に一度 `ft_w` (FP32 master) から cast しないと
    /// mirror が全 0 で forward が trivial になる。`--init-from` / `--resume` で `ft_w`
    /// を読み込んだ後にも呼ぶ。`ft_fp16` が無効 (`ft_w_h` が `None`) なら no-op。
    fn sync_ft_w_h_mirror(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
            let ft_w_n = self.ws.ft_in * self.id.ft_out;
            cuda_launch! {
                kernel: cast_f32_to_f16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_w_n),
                args: [slice(self.ft_w), slice_mut(ft_w_h), ft_w_n as u32]
            }?;
        }
        Ok(())
    }

    /// 全 weight buffer を host に download し NaN/Inf が無いことを assert する。
    fn assert_all_weights_finite(&self) -> Result<(), Box<dyn std::error::Error>> {
        let groups: [(&DeviceBuffer<f32>, &str); 8] = [
            (&self.ft_w, "ft_w"),
            (&self.ft_b, "ft_b"),
            (&self.l1_w, "l1_w"),
            (&self.l1_b, "l1_b"),
            (&self.l2_w, "l2_w"),
            (&self.l2_b, "l2_b"),
            (&self.l3_w, "l3_w"),
            (&self.l3_b, "l3_b"),
        ];
        for (buf, name) in groups {
            let v = buf.to_host_vec(&self.stream)?;
            for (i, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(format!("{name}[{i}] = {x} is not finite (NaN or Inf)").into());
                }
            }
        }
        Ok(())
    }

    /// 1 batch の forward を走らせ、loss kernel が累積した Σerr² を返す。
    /// backward は走らせず、loss kernel が書く dnet (`dy_net_output`) は捨てる。
    fn forward(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        self.ws.check_batch_capacity(b)?;
        self.run_forward_kernels(batch, wdl_lambda, loss, false)?;
        let loss_host = self.loss_acc.to_host_vec(&self.stream)?;
        Ok(loss_host[0])
    }

    /// 1 batch の forward → backward → Ranger optimizer step を走らせ、loss kernel が
    /// 累積した Σerr² を返す。`bucket_idx` は受け取らない (Simple アーキは bucket 無し)。
    ///
    /// 環境変数 `NNUE_TRAIN_STEP_PROFILE` がセットされていれば各 phase の境界で
    /// `synchronize()` + 経過時間を stderr に出す (粗い forward / backward / optimizer /
    /// loss_readback breakdown)。LayerStack `GpuTrainer::step` と同 env var を共有。
    fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        self.ws.check_batch_capacity(b)?;

        let mut prof_t0 = if std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 入力 4 buffer の H2D を `InputUploadRing` 経由で発行する。active / back を
        // `mem::swap` してから back (= 直前 step が読んでいない側の物理 buffer) へ専用
        // copy stream で先行 H2D。pageable な `BatchData` slice は ring 内の pinned host
        // を経由して copy engine の DMA に載り、compute stream は H2D 完了 event を
        // 待ってから forward に進む ([`AsyncLossRing`] による host run-ahead と組合せて
        // h2d_reset を直前 step の compute と並走させる)。
        std::mem::swap(&mut self.ws.stm_idx_dev, &mut self.ws.stm_idx_dev_back);
        std::mem::swap(&mut self.ws.nstm_idx_dev, &mut self.ws.nstm_idx_dev_back);
        std::mem::swap(&mut self.ws.score_dev, &mut self.ws.score_dev_back);
        std::mem::swap(&mut self.ws.wdl_dev, &mut self.ws.wdl_dev_back);
        self.input_ring.upload_simple(
            &self.stream,
            &self.ws.stm_idx_dev,
            &batch.stm_indices[..b * self.ws.max_active],
            &self.ws.nstm_idx_dev,
            &batch.nstm_indices[..b * self.ws.max_active],
            &self.ws.score_dev,
            &batch.score[..b],
            &self.ws.wdl_dev,
            &batch.wdl[..b],
        )?;

        self.run_forward_kernels(batch, wdl_lambda, loss, true)?;
        if let Some(ref mut t0) = prof_t0 {
            self.stream.synchronize()?;
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "forward",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
            *t0 = now;
        }

        self.run_backward_kernels(b)?;
        if let Some(ref mut t0) = prof_t0 {
            self.stream.synchronize()?;
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "backward",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
            *t0 = now;
        }

        self.run_optimizer_step(lr)?;
        if let Some(ref mut t0) = prof_t0 {
            self.stream.synchronize()?;
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "optimizer",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
            *t0 = now;
        }

        // 本 step の compute (input buffer の read を含む) 完了を copy stream 用の
        // event に記録する。同じ物理 input buffer を使う step+2 の H2D がこれを待ち、
        // in-flight な compute が読む buffer を H2D が上書きする race を防ぐ。
        self.input_ring.mark_step_done(&self.stream)?;

        // `loss_acc` の host 読みを [`AsyncLossRing`] 経由で async + 1-step lag に
        // する。pinned host cell に `memcpy_dtoh_async` + event record、前 step の
        // event を sync して 1 step 遅れで loss を返す (step 0 は warmup として 0.0、
        // sb 末で [`TrainerBackend::flush_pending_loss`] が最終 step 分を drain する)。
        // host は次 batch の launch 発行で `stream.synchronize` 相当の block 待ちが消える。
        let loss = self
            .loss_ring
            .read_and_queue_next(&self.stream, &self.loss_acc)?;
        if let Some(ref t0) = prof_t0 {
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "loss_readback",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
        }

        Ok(loss)
    }

    /// forward kernel 列のみを走らせる (loss は host 同期 read しない)。`forward` /
    /// `step` 共通の前段で、終了時 `net_output` / `dy_net_output` / `loss_acc` が device
    /// 上に書かれている。caller が batch capacity を事前検査する責務。
    ///
    /// `inputs_uploaded_externally` が `true` のとき caller (= `step`) が直前に
    /// [`InputUploadRing`] で `ws.{stm,nstm}_idx_dev` / `ws.{score,wdl}_dev` への H2D
    /// を queue 済とみなし、本 method 内では H2D を発行しない (compute stream は ring
    /// の `h2d_done` event 経由で H2D 完了を既に待つ)。`false` のとき (`forward` /
    /// `validate` 経路) は同期 H2D を default stream 上で発行する。
    fn run_forward_kernels(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
        inputs_uploaded_externally: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut prof_t0 = if std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let tick = |label: &str,
                    stream: &CudaStream,
                    t0: &mut Option<std::time::Instant>|
         -> Result<(), Box<dyn std::error::Error>> {
            if let Some(t) = t0 {
                stream.synchronize()?;
                let now = std::time::Instant::now();
                eprintln!(
                    "[step-profile]   {:<12} {:8.3} ms",
                    label,
                    now.duration_since(*t).as_secs_f64() * 1000.0
                );
                *t = now;
            }
            Ok(())
        };
        let b = batch.n_pos;
        let b_u32 = b as u32;
        let ft_out_u32 = self.id.ft_out as u32;
        let l1_in_u32 = (2 * self.id.ft_out) as u32; // L1 入力 = stm/nstm concat 後
        let l1_out_u32 = self.id.l1_out as u32;
        let l2_out_u32 = self.id.l2_out as u32;
        let ft_n = b * self.id.ft_out;

        // -- H2D upload (default stream 上の async memcpy、launch 列に直列で並ぶ) --
        // ring 経路では caller (= `step`) が swap + `input_ring.upload_simple` で
        // copy stream に H2D を発行済で、`compute_stream.wait(h2d_done)` で本 stream に
        // 完了待ちが乗っているため、ここでの H2D は不要。
        if !inputs_uploaded_externally {
            copy_host_to_device_async_i32(
                &self.stream,
                &self.ws.stm_idx_dev,
                &batch.stm_indices[..b * self.ws.max_active],
            )?;
            copy_host_to_device_async_i32(
                &self.stream,
                &self.ws.nstm_idx_dev,
                &batch.nstm_indices[..b * self.ws.max_active],
            )?;
            copy_host_to_device_async_f32(&self.stream, &self.ws.score_dev, &batch.score[..b])?;
            copy_host_to_device_async_f32(&self.stream, &self.ws.wdl_dev, &batch.wdl[..b])?;
        }

        // -- loss_acc を 0 にリセット (再 alloc 無し) --
        memset_zero(&self.stream, &self.loss_acc)?;
        tick("h2d_reset", &self.stream, &mut prof_t0)?;

        // -- sparse_ft_forward × 2 (stm, nstm)。1 thread = 4 row。
        // 3 path 分岐:
        //  - `ft_fp16_out`: `sparse_ft_forward_fp16_out` (f16 weight + f16 出力)、ft_*_out
        //    は f16 buffer (`ft_*_out_h`) に書く (pre-bias、bias は後段 fused kernel で加算)
        //  - `ft_fp16`: `sparse_ft_forward_fp16` (f16 weight + f32 出力)、ft_*_out は f32
        //  - 既定: `sparse_ft_forward` (FP32 master、bit-identical)
        if self.ft_fp16_out {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16_out is enabled");
            let mut ft_stm_out_h = self
                .ws
                .ft_stm_out_h
                .as_mut()
                .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16_out,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(ft_stm_out_h),
                    b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            let mut ft_nstm_out_h = self
                .ws
                .ft_nstm_out_h
                .as_mut()
                .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16_out,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(ft_nstm_out_h),
                    b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        } else if self.ft_fp16 {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16 is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out),
                    b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out),
                    b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: sparse_ft_forward,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.ft_out / 4),
                args: [
                    slice(self.ft_w),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out),
                    b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.ft_out / 4),
                args: [
                    slice(self.ft_w),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out),
                    b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        }
        tick("fwd_ft", &self.stream, &mut prof_t0)?;

        // -- FT post = bias add + 活性化 + concat。
        // `ft_fp16_out` 時は `simple_bias_act_fwd_fp16_in_crelu` が bias 加算 + CReLU を
        // 融合し、f16 `ft_*_out_h` → f32 `ft_*_acted` に書く (CLI 検証で SCReLU は reject)。
        // 既定は LayerStack の ft_post_perspective_fwd と違い pairwise / scale を含まない
        // ため既存 4 launch (bias_add ×2 + 活性化 ×2) で合成する。
        if self.ft_fp16_out {
            let ft_stm_out_h = self
                .ws
                .ft_stm_out_h
                .as_ref()
                .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: simple_bias_act_fwd_fp16_in_crelu,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(ft_stm_out_h),
                    slice(self.ft_b),
                    slice_mut(self.ws.ft_stm_acted),
                    b_u32, ft_out_u32
                ]
            }?;
            let ft_nstm_out_h = self
                .ws
                .ft_nstm_out_h
                .as_ref()
                .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: simple_bias_act_fwd_fp16_in_crelu,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(ft_nstm_out_h),
                    slice(self.ft_b),
                    slice_mut(self.ws.ft_nstm_acted),
                    b_u32, ft_out_u32
                ]
            }?;
        } else {
            // DEFAULT (FP32) path: bias_add + activation + slice_scatter (per perspective × 2)
            // を 1 kernel に融合。`ft_*_out` に bias を in-place 加算 (bwd indicator が読む)
            // した後、活性化結果を直接 `combined` の per-perspective slice に書く。中間
            // `ft_*_acted` buffer の DRAM write+read と、bias_add → 活性化 間の `ft_*_out`
            // 再 read+write が消える (1 perspective につき ~536 MB DRAM、2 perspective で
            // ~1.07 GB)。`ft_fp16_out` 経路は scale / clamp / f16 cast 含むため融合外。
            match self.id.activation {
                SimpleActivation::CReLU => {
                    cuda_launch! {
                        kernel: simple_ft_post_fused_crelu,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [
                            slice_mut(self.ws.ft_stm_out),
                            slice(self.ft_b),
                            slice_mut(self.ws.combined),
                            b_u32, ft_out_u32, 0_u32
                        ]
                    }?;
                    cuda_launch! {
                        kernel: simple_ft_post_fused_crelu,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [
                            slice_mut(self.ws.ft_nstm_out),
                            slice(self.ft_b),
                            slice_mut(self.ws.combined),
                            b_u32, ft_out_u32, ft_out_u32
                        ]
                    }?;
                }
                SimpleActivation::SCReLU => {
                    cuda_launch! {
                        kernel: simple_ft_post_fused_screlu,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [
                            slice_mut(self.ws.ft_stm_out),
                            slice(self.ft_b),
                            slice_mut(self.ws.combined),
                            b_u32, ft_out_u32, 0_u32
                        ]
                    }?;
                    cuda_launch! {
                        kernel: simple_ft_post_fused_screlu,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [
                            slice_mut(self.ws.ft_nstm_out),
                            slice(self.ft_b),
                            slice_mut(self.ws.combined),
                            b_u32, ft_out_u32, ft_out_u32
                        ]
                    }?;
                }
            }
        }

        // `ft_fp16_out` 経路は融合 kernel が `ft_*_acted` までしか書かないため `combined`
        // へ slice_scatter する。DEFAULT 経路は `simple_ft_post_fused_*` が `combined` を
        // 直書きするためここを skip する。
        if self.ft_fp16_out {
            cuda_launch! {
                kernel: slice_scatter_2d,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(self.ws.ft_stm_acted),
                    slice_mut(self.ws.combined),
                    b_u32, ft_out_u32, l1_in_u32, 0_u32
                ]
            }?;
            cuda_launch! {
                kernel: slice_scatter_2d,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(self.ws.ft_nstm_acted),
                    slice_mut(self.ws.combined),
                    b_u32, ft_out_u32, l1_in_u32, ft_out_u32
                ]
            }?;
        }
        tick("fwd_ft_post", &self.stream, &mut prof_t0)?;

        // -- L1 dense (combined → l1_pre) cuBLAS Sgemm + bias_add_per_row --
        // shape: combined[B, 2*ft_out] @ l1_w[2*ft_out, l1_out] → l1_pre[B, l1_out]、
        // 続けて bias を別 kernel で row-add する (Sgemm 自身は bias 非対応)。
        //
        // SAFETY: combined / l1_w / l1_pre は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (workspace 系の `combined.len() >= b * (2*ft_out)` / `l1_pre.len() >= b * l1_out`
        // は事前の `ws.check_batch_capacity(b)` で保証、weight `l1_w.len() == (2*ft_out)
        // * l1_out` は固定 shape)、`self.cublas` は `self.stream` に bind 済で同 stream 内
        // in-order 実行 (先行 kernel 完了後に Sgemm が走り、後続 bias_add_per_row が観測)。
        // `cu_deviceptr() as *const/*mut f32` cast の妥当性: cuMemAlloc が返す device
        // pointer は 256 byte aligned (`f32` の 4 byte 要求を満たす)、`self` の借用が
        // unsafe block を超えて生存するので元 buffer も同 lifetime で valid。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,
                l1_out_u32 as i32,
                l1_in_u32 as i32,
                self.ws.combined.cu_deviceptr() as *const f32,
                self.l1_w.cu_deviceptr() as *const f32,
                self.ws.l1_pre.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_add_per_row,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * self.id.l1_out),
            args: [slice(self.l1_b), slice_mut(self.ws.l1_pre), b_u32, l1_out_u32]
        }?;
        let l1_n = b * self.id.l1_out;
        let l1_n_u32 = l1_n as u32;
        match self.id.activation {
            SimpleActivation::CReLU => cuda_launch! {
                kernel: crelu_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(l1_n),
                args: [slice(self.ws.l1_pre), slice_mut(self.ws.l1_acted), l1_n_u32]
            }?,
            SimpleActivation::SCReLU => cuda_launch! {
                kernel: screlu_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(l1_n),
                args: [slice(self.ws.l1_pre), slice_mut(self.ws.l1_acted), l1_n_u32]
            }?,
        }

        // -- L2 dense (l1_acted → l2_pre) cuBLAS Sgemm + bias_add_per_row --
        // shape: l1_acted[B, l1_out] @ l2_w[l1_out, l2_out] → l2_pre[B, l2_out]、
        // 続けて bias を別 kernel で row-add する (Sgemm 自身は bias 非対応)。
        // L1 fwd と同じ手で、`dense_mm_fwd` の thread 数が `B * out_dim` で
        // SM 占有率が低いのを cuBLAS で塗り替える。
        //
        // SAFETY: l1_acted / l2_w / l2_pre は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (`l1_acted.len() >= b * l1_out` / `l2_pre.len() >= b * l2_out` は事前の
        // `ws.check_batch_capacity(b)` で保証、weight `l2_w.len() == l1_out * l2_out`
        // は固定 shape)、`self.cublas` は `self.stream` に bind 済の同一 stream を再利用。
        // `alpha=1`, `beta=0` overwrite なので `l2_pre` の事前内容は使われない (後続の
        // `bias_add_per_row` が書き戻し)。`cu_deviceptr() as *const/*mut f32` cast の
        // 妥当性は L1 と同じ前提 (256 byte aligned cuMemAlloc 由来、`self` 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,
                l2_out_u32 as i32,
                l1_out_u32 as i32,
                self.ws.l1_acted.cu_deviceptr() as *const f32,
                self.l2_w.cu_deviceptr() as *const f32,
                self.ws.l2_pre.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_add_per_row,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * self.id.l2_out),
            args: [slice(self.l2_b), slice_mut(self.ws.l2_pre), b_u32, l2_out_u32]
        }?;
        let l2_n = b * self.id.l2_out;
        let l2_n_u32 = l2_n as u32;
        match self.id.activation {
            SimpleActivation::CReLU => cuda_launch! {
                kernel: crelu_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(l2_n),
                args: [slice(self.ws.l2_pre), slice_mut(self.ws.l2_acted), l2_n_u32]
            }?,
            SimpleActivation::SCReLU => cuda_launch! {
                kernel: screlu_fwd, stream: self.stream, module: self.module,
                config: cfg_1d(l2_n),
                args: [slice(self.ws.l2_pre), slice_mut(self.ws.l2_acted), l2_n_u32]
            }?,
        }

        // -- L3 dense (l2_acted → net_output)。out_dim = 1 (スカラ出力)、cuBLAS Sgemm + bias --
        // shape: l2_acted[B, l2_out] @ l3_w[l2_out, 1] → net_output[B, 1]。cuBLAS は
        // N=1 (= matrix-vector 相当) でも内部で適切な algorithm を選ぶ。
        //
        // SAFETY: l2_acted / l3_w / net_output は cudaMalloc 由来 + 長さは仕様分以上
        // (`l2_acted.len() >= b * l2_out` / `net_output.len() >= b` は事前の
        // `ws.check_batch_capacity(b)` で保証、weight `l3_w.len() == l2_out` は固定 shape)。
        // `alpha=1`, `beta=0` overwrite なので `net_output` の事前内容は使われない (後続の
        // `bias_add_per_row` が書き戻し)。残りの不変条件は L1 / L2 と同じ。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,
                1_i32,
                l2_out_u32 as i32,
                self.ws.l2_acted.cu_deviceptr() as *const f32,
                self.l3_w.cu_deviceptr() as *const f32,
                self.ws.net_output.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_add_per_row,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [slice(self.l3_b), slice_mut(self.ws.net_output), b_u32, 1_u32]
        }?;
        tick("fwd_dense", &self.stream, &mut prof_t0)?;

        // -- loss kernel (Σerr² を loss_acc に atomic accumulate)、`dy_net_output` に
        // L3 出力への grad を書く。`wdl_lambda` で score/wdl ターゲットを blend する。
        match loss {
            LossKind::Sigmoid { scale } => {
                cuda_launch! {
                    kernel: loss_wdl,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b),
                    args: [
                        slice(self.ws.net_output),
                        slice(self.ws.score_dev),
                        slice(self.ws.wdl_dev),
                        batch.per_pos_norm,
                        slice_mut(self.ws.dy_net_output),
                        slice(self.loss_acc),
                        wdl_lambda,
                        scale,
                        b_u32
                    ]
                }?;
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
                target_offset,
                target_scaling,
            } => {
                cuda_launch! {
                    kernel: loss_wrm,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b),
                    args: [
                        slice(self.ws.net_output),
                        slice(self.ws.score_dev),
                        slice(self.ws.wdl_dev),
                        batch.per_pos_norm,
                        slice_mut(self.ws.dy_net_output),
                        slice(self.loss_acc),
                        wdl_lambda,
                        nnue2score, in_scaling,
                        target_offset, target_scaling, b_u32
                    ]
                }?;
            }
        }
        tick("fwd_loss", &self.stream, &mut prof_t0)?;
        Ok(())
    }

    /// `run_forward_kernels` の直後に呼び、`dy_net_output` を起点として 8 weight group
    /// の gradient buffer を埋める。bias / FT weight は atomic accumulate のため host で
    /// 0 初期化する。dense weight (l1/l2/l3_w) は `dense_mm_bwd_weight` が overwrite
    /// 書きなので初期化不要。
    fn run_backward_kernels(&mut self, b: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut prof_t0 = if std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let tick = |label: &str,
                    stream: &CudaStream,
                    t0: &mut Option<std::time::Instant>|
         -> Result<(), Box<dyn std::error::Error>> {
            if let Some(t) = t0 {
                stream.synchronize()?;
                let now = std::time::Instant::now();
                eprintln!(
                    "[step-profile]   {:<12} {:8.3} ms",
                    label,
                    now.duration_since(*t).as_secs_f64() * 1000.0
                );
                *t = now;
            }
            Ok(())
        };
        let b_u32 = b as u32;
        let ft_out_u32 = self.id.ft_out as u32;
        let l1_in_u32 = (2 * self.id.ft_out) as u32;
        let l1_out_u32 = self.id.l1_out as u32;
        let l2_out_u32 = self.id.l2_out as u32;
        let ft_in_u32 = self.ws.ft_in as u32;
        let max_active_u32 = self.ws.max_active as u32;
        let ft_n = b * self.id.ft_out;
        let l1_n = b * self.id.l1_out;
        let l1_n_u32 = l1_n as u32;
        let l2_n = b * self.id.l2_out;
        let l2_n_u32 = l2_n as u32;

        // bias_grad は atomic add で累積するため host で 0 初期化必須。`ft_w_grad` は
        // 後段の `gather_and_sum_per_feature_overwrite` (本関数末尾の inverse-index pipeline、
        // iter 0 = stm) が全 `(feature, ri)` cell を書き切るため reset 不要。
        memset_zero(&self.stream, &self.ft_b_grad)?;
        memset_zero(&self.stream, &self.l1_b_grad)?;
        memset_zero(&self.stream, &self.l2_b_grad)?;
        memset_zero(&self.stream, &self.l3_b_grad)?;
        tick("bwd_memset", &self.stream, &mut prof_t0)?;

        // ---- L3: dy_net_output (b × 1) -> dl2_acted (b × l2_out), l3_w_grad, l3_b_grad ----
        // bwd_input: dl2_acted[B, l2_out] = dy_net_output[B, 1] @ l3_w[l2_out, 1]^T
        // bwd_weight: l3_w_grad[l2_out, 1] = l2_acted[B, l2_out]^T @ dy_net_output[B, 1]
        // out_dim=1 で weight grad は thread = l2_out (= 数十) しか起動できない matmul
        // shape のため、cuBLAS Sgemm に委譲して内部 Sgemv-相当 algorithm + B 軸並列で
        // SM 占有率を稼ぐ (untiled kernel は in_dim*out_dim thread 駆動で 1 warp 規模)。
        //
        // SAFETY: 全 device pointer は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (`dy_net_output.len() >= b` / `dl2_acted.len() >= b*l2_out` / `l2_acted.len()
        // >= b*l2_out` は `ws.check_batch_capacity(b)` で保証、weight grad `l3_w_grad.len()
        // == l2_out` は固定 shape)、`self.cublas` は `self.stream` に bind 済。
        // `alpha=1`, `beta=0` overwrite なので `dl2_acted` / `l3_w_grad` の事前内容は
        // 使われない。cast 妥当性は L1/L2 と同じ前提 (256 byte aligned cuMemAlloc 由来、
        // `self` 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_x_yt_rowmajor(
                b_u32 as i32,
                l2_out_u32 as i32,
                1_i32,
                self.ws.dy_net_output.cu_deviceptr() as *const f32,
                self.l3_w.cu_deviceptr() as *const f32,
                self.ws.dl2_acted.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                l2_out_u32 as i32,
                1_i32,
                b_u32 as i32,
                self.ws.l2_acted.cu_deviceptr() as *const f32,
                self.ws.dy_net_output.cu_deviceptr() as *const f32,
                self.l3_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_grad, stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [slice(self.ws.dy_net_output), slice(self.l3_b_grad), b_u32, 1_u32]
        }?;
        tick("L3_dense", &self.stream, &mut prof_t0)?;

        // ---- L2 activation grad: dl2_acted -> dl2_pre (kernel reads l2_pre) ----
        match self.id.activation {
            SimpleActivation::CReLU => cuda_launch! {
                kernel: crelu_grad, stream: self.stream, module: self.module,
                config: cfg_1d(l2_n),
                args: [slice(self.ws.l2_pre), slice(self.ws.dl2_acted),
                       slice_mut(self.ws.dl2_pre), l2_n_u32]
            }?,
            SimpleActivation::SCReLU => cuda_launch! {
                kernel: screlu_grad, stream: self.stream, module: self.module,
                config: cfg_1d(l2_n),
                args: [slice(self.ws.l2_pre), slice(self.ws.dl2_acted),
                       slice_mut(self.ws.dl2_pre), l2_n_u32]
            }?,
        }

        // ---- L2 dense backward: dl2_pre -> dl1_acted (cuBLAS), l2_w_grad (cuBLAS),
        //      l2_b_grad (kernel) ----
        // bwd_input: dl1_acted[B, l1_out] = dl2_pre[B, l2_out] @ l2_w[l1_out, l2_out]^T
        // bwd_weight: l2_w_grad[l1_out, l2_out] = l1_acted[B, l1_out]^T @ dl2_pre[B, l2_out]
        // weight grad は thread = l1_out*l2_out 駆動で `B` を内側 loop に置く構造、
        // 小 matmul shape (in_dim*out_dim が数百-千) で SM 占有率を稼げない。cuBLAS Sgemm
        // は `B` を block 並列に展開できる algorithm を選ぶ。
        //
        // SAFETY: 全 device pointer は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (`dl2_pre.len() >= b*l2_out` / `dl1_acted.len() >= b*l1_out` / `l1_acted.len()
        // >= b*l1_out` は `ws.check_batch_capacity(b)` で保証、weight grad `l2_w_grad.len()
        // == l1_out*l2_out` は固定 shape)、`self.cublas` は `self.stream` に bind 済。
        // `alpha=1`, `beta=0` overwrite なので `dl1_acted` / `l2_w_grad` の事前内容は
        // 使われない。cast 妥当性は L1 と同じ前提 (256 byte aligned cuMemAlloc、`self`
        // 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_x_yt_rowmajor(
                b_u32 as i32,
                l1_out_u32 as i32,
                l2_out_u32 as i32,
                self.ws.dl2_pre.cu_deviceptr() as *const f32,
                self.l2_w.cu_deviceptr() as *const f32,
                self.ws.dl1_acted.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                l1_out_u32 as i32,
                l2_out_u32 as i32,
                b_u32 as i32,
                self.ws.l1_acted.cu_deviceptr() as *const f32,
                self.ws.dl2_pre.cu_deviceptr() as *const f32,
                self.l2_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_grad, stream: self.stream, module: self.module,
            config: cfg_1d(l2_n),
            args: [slice(self.ws.dl2_pre), slice(self.l2_b_grad), b_u32, l2_out_u32]
        }?;
        tick("L2_dense", &self.stream, &mut prof_t0)?;

        // ---- L1 activation grad: dl1_acted -> dl1_pre (kernel reads l1_pre) ----
        match self.id.activation {
            SimpleActivation::CReLU => cuda_launch! {
                kernel: crelu_grad, stream: self.stream, module: self.module,
                config: cfg_1d(l1_n),
                args: [slice(self.ws.l1_pre), slice(self.ws.dl1_acted),
                       slice_mut(self.ws.dl1_pre), l1_n_u32]
            }?,
            SimpleActivation::SCReLU => cuda_launch! {
                kernel: screlu_grad, stream: self.stream, module: self.module,
                config: cfg_1d(l1_n),
                args: [slice(self.ws.l1_pre), slice(self.ws.dl1_acted),
                       slice_mut(self.ws.dl1_pre), l1_n_u32]
            }?,
        }

        // ---- L1 dense backward: dl1_pre -> dcombined (cuBLAS), l1_w_grad (cuBLAS), l1_b_grad (kernel) ----
        // bwd_input: dcombined[B, 2*ft_out] = dl1_pre[B, l1_out] @ l1_w[2*ft_out, l1_out]^T
        //   ( = dx[b][i] = sum_o dy[b][o] * w[i][o] )
        // SAFETY: dl1_pre / l1_w / dcombined は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (workspace 系の `dl1_pre.len() >= b*l1_out` / `dcombined.len() >= b*(2*ft_out)`
        // は `ws.check_batch_capacity(b)` で保証、weight `l1_w.len() == (2*ft_out)*l1_out`
        // は固定 shape)、`self.cublas` は `self.stream` に bind 済。`cu_deviceptr() as
        // *const/*mut f32` cast の妥当性: cuMemAlloc が返す device pointer は 256 byte
        // aligned (`f32` の 4 byte 要求を満たす)、`self` の借用が unsafe block を超えて
        // 生存するので元 buffer も同 lifetime で valid。
        unsafe {
            self.cublas.sgemm_x_yt_rowmajor(
                b_u32 as i32,
                l1_in_u32 as i32,
                l1_out_u32 as i32,
                self.ws.dl1_pre.cu_deviceptr() as *const f32,
                self.l1_w.cu_deviceptr() as *const f32,
                self.ws.dcombined.cu_deviceptr() as *mut f32,
            )?;
        }
        // bwd_weight: l1_w_grad[2*ft_out, l1_out] = combined[B, 2*ft_out]^T @ dl1_pre[B, l1_out]
        // SAFETY: combined / dl1_pre / l1_w_grad は cudaMalloc 由来 + 長さは Sgemm 仕様分
        // 以上 (workspace 系の `combined.len() >= b*(2*ft_out)` / `dl1_pre.len() >= b*
        // l1_out` は `ws.check_batch_capacity(b)` で保証、weight grad `l1_w_grad.len() ==
        // (2*ft_out)*l1_out` は固定 shape)、stream 共有 + cast 妥当性は bwd_input と同じ
        // 前提 (256 byte aligned cuMemAlloc 由来、`self` 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                l1_in_u32 as i32,
                l1_out_u32 as i32,
                b_u32 as i32,
                self.ws.combined.cu_deviceptr() as *const f32,
                self.ws.dl1_pre.cu_deviceptr() as *const f32,
                self.l1_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_grad, stream: self.stream, module: self.module,
            config: cfg_1d(l1_n),
            args: [slice(self.ws.dl1_pre), slice(self.l1_b_grad), b_u32, l1_out_u32]
        }?;
        tick("L1_dense", &self.stream, &mut prof_t0)?;

        // ---- Concat inverse + activation grad の融合 ----
        // dcombined (b × 2*ft_out) の per-perspective 半分を `src_offset` で切り出して読み、
        // pre-activation `ft_*_out` で gate した値を `dft_*_out` に直接書く融合 kernel。
        // 中間 `dft_*_acted` buffer の DRAM round-trip (b × ft_out × 4 byte の write+read) を
        // 消す。`ft_fp16_out` 経路は f16 dft buffer + loss scaling + clamp + f16 cast を含む
        // ため `slice_extract_2d` + `simple_act_grad_to_fp16_crelu_with_scale` の 2 kernel に
        // 留める (融合範囲外、scale や clamp の数値挙動を切替で揺らさないため)。
        if self.ft_fp16_out {
            cuda_launch! {
                kernel: slice_extract_2d, stream: self.stream, module: self.module,
                config: cfg_1d(ft_n),
                args: [slice(self.ws.dcombined), slice_mut(self.ws.dft_stm_acted),
                       b_u32, l1_in_u32, 0_u32, ft_out_u32]
            }?;
            cuda_launch! {
                kernel: slice_extract_2d, stream: self.stream, module: self.module,
                config: cfg_1d(ft_n),
                args: [slice(self.ws.dcombined), slice_mut(self.ws.dft_nstm_acted),
                       b_u32, l1_in_u32, ft_out_u32, ft_out_u32]
            }?;
            let dft_scale = FT_DFT_FP16_BASE_SCALE * (b as f32);
            let ft_stm_out_h = self
                .ws
                .ft_stm_out_h
                .as_ref()
                .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
            let mut dft_stm_out_h = self
                .ws
                .dft_stm_out_h
                .as_mut()
                .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: simple_act_grad_to_fp16_crelu_with_scale,
                stream: self.stream, module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(ft_stm_out_h),
                    slice(self.ft_b),
                    slice(self.ws.dft_stm_acted),
                    slice_mut(dft_stm_out_h),
                    b_u32, ft_out_u32, dft_scale
                ]
            }?;
            let ft_nstm_out_h = self
                .ws
                .ft_nstm_out_h
                .as_ref()
                .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
            let mut dft_nstm_out_h = self
                .ws
                .dft_nstm_out_h
                .as_mut()
                .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: simple_act_grad_to_fp16_crelu_with_scale,
                stream: self.stream, module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(ft_nstm_out_h),
                    slice(self.ft_b),
                    slice(self.ws.dft_nstm_acted),
                    slice_mut(dft_nstm_out_h),
                    b_u32, ft_out_u32, dft_scale
                ]
            }?;
        } else {
            match self.id.activation {
                SimpleActivation::CReLU => {
                    cuda_launch! {
                        kernel: simple_bwd_ft_act_crelu_fused,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [slice(self.ws.dcombined), slice(self.ws.ft_stm_out),
                               slice(self.ws.dft_stm_out), b_u32, ft_out_u32, 0_u32]
                    }?;
                    cuda_launch! {
                        kernel: simple_bwd_ft_act_crelu_fused,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [slice(self.ws.dcombined), slice(self.ws.ft_nstm_out),
                               slice(self.ws.dft_nstm_out), b_u32, ft_out_u32, ft_out_u32]
                    }?;
                }
                SimpleActivation::SCReLU => {
                    cuda_launch! {
                        kernel: simple_bwd_ft_act_screlu_fused,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [slice(self.ws.dcombined), slice(self.ws.ft_stm_out),
                               slice(self.ws.dft_stm_out), b_u32, ft_out_u32, 0_u32]
                    }?;
                    cuda_launch! {
                        kernel: simple_bwd_ft_act_screlu_fused,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                        args: [slice(self.ws.dcombined), slice(self.ws.ft_nstm_out),
                               slice(self.ws.dft_nstm_out), b_u32, ft_out_u32, ft_out_u32]
                    }?;
                }
            }
        }
        tick("bwd_ft_act", &self.stream, &mut prof_t0)?;

        // ---- FT bias grad + FT weight grad: stm/nstm の両 perspective が同じ ft_b / ft_w
        // を共有するため atomic accumulate (host が呼出前に 0 初期化)。
        // `ft_fp16_out` 時は f16 dft buffer を read、`dft_inv_scale` で loss scaling を打ち消す。
        // FT bias grad は両 perspective が同じ ft_b を共有するため atomic accumulate。
        // host が呼出前に `ft_b_grad` を 0 reset 済 (本関数冒頭の memset_zero ブロック)。
        let dft_inv_scale_fp16 = if self.ft_fp16_out {
            1.0_f32 / (FT_DFT_FP16_BASE_SCALE * (b as f32))
        } else {
            1.0_f32 // unused on FP32 path
        };
        // FT bias grad は両 perspective が同じ `ft_b` を共有するので、per-cell に stm + nstm
        // の ローカル和を作って atomic add 1 回で `ft_b_grad[oi]` に accumulate する
        // (`simple_bias_grad_dual`)。atomic contention は B * ft_dim 回で、stm / nstm を別
        // launch で 2 回 atomic 打つ構成より半減。
        if self.ft_fp16_out {
            let dft_stm_out_h = self
                .ws
                .dft_stm_out_h
                .as_ref()
                .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
            let dft_nstm_out_h = self
                .ws
                .dft_nstm_out_h
                .as_ref()
                .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
            cuda_launch! {
                kernel: simple_bias_grad_dual_fp16, stream: self.stream, module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(dft_stm_out_h),
                    slice(dft_nstm_out_h),
                    slice(self.ft_b_grad),
                    b_u32, ft_out_u32, dft_inv_scale_fp16
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: simple_bias_grad_dual, stream: self.stream, module: self.module,
                config: cfg_1d(ft_n),
                args: [
                    slice(self.ws.dft_stm_out),
                    slice(self.ws.dft_nstm_out),
                    slice(self.ft_b_grad),
                    b_u32, ft_out_u32
                ]
            }?;
        }

        // FT weight grad — **inverse-index pipeline** で per-feature gather に変換する経路。
        // 各 perspective につき (A) `build_feature_counts` で histogram、(B)
        // `exclusive_prefix_sum_small` で offset、(C) `scatter_positions` で sorted position 列を
        // 構築し、(D) `gather_and_sum_per_feature_overwrite` (1 回目 = stm) /
        // `gather_and_sum_per_feature_add` (2 回目 = nstm) が `(feature, ri)` cell ごとに
        // sum を書く。FP16 path は同 pipeline で `_fp16` 変種に dft_inv_scale を渡す。
        // stm の overwrite が `ft_w_grad` の全 `(feature, ri)` cell を unconditionally 書き切るため、
        // 本関数冒頭の memset_zero 群に `ft_w_grad` は含めない (LayerStack の同 pipeline と同規約)。
        let gather_config = LaunchConfig {
            grid_dim: (ft_in_u32, self.id.ft_out.div_ceil(128) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        for (iter_idx, idx_dev) in [&self.ws.stm_idx_dev, &self.ws.nstm_idx_dev]
            .into_iter()
            .enumerate()
        {
            // A1: feat_counts / feat_write_ctr を 0 にリセット (atomic build / scatter の前提)。
            memset_zero(&self.stream, &self.ws.feat_counts)?;
            memset_zero(&self.stream, &self.ws.feat_write_ctr)?;
            // A2: 各 (b, ni) sparse index について feat_counts[feature] を atomic increment。
            cuda_launch! {
                kernel: build_feature_counts,
                stream: self.stream, module: self.module,
                config: cfg_1d(b * self.ws.max_active),
                args: [
                    slice(idx_dev),
                    slice(self.ws.feat_counts),
                    b_u32, max_active_u32, ft_in_u32
                ]
            }?;
            // B: exclusive prefix sum (1 block × 1024 threads、`ft_in` ~73K-138K に対応)。
            cuda_launch! {
                kernel: exclusive_prefix_sum_small,
                stream: self.stream, module: self.module,
                config: LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [
                    slice(self.ws.feat_counts),
                    slice(self.ws.feat_offsets),
                    ft_in_u32
                ]
            }?;
            // C: 各 (b, ni) sparse index について feat_positions の per-feature slot に
            // batch position `b` を書き込む (`feat_write_ctr[feature]++` で位置決定)。
            cuda_launch! {
                kernel: scatter_positions,
                stream: self.stream, module: self.module,
                config: cfg_1d(b * self.ws.max_active),
                args: [
                    slice(idx_dev),
                    slice(self.ws.feat_offsets),
                    slice(self.ws.feat_write_ctr),
                    slice(self.ws.feat_positions),
                    b_u32, max_active_u32, ft_in_u32
                ]
            }?;
            // D: 各 (feature, ri) cell について feat_positions[feat_offsets[f]..feat_offsets[f+1]]
            // を順 read して accumulate。iter 0 = stm は overwrite (全 cell 書き切り)、iter 1 =
            // nstm は atomic add で stm 結果に重ねる。FP16 path は `dft_inv_scale` で loss scaling
            // を打ち消しながら read。
            if self.ft_fp16_out {
                let dft_stm_out_h = self
                    .ws
                    .dft_stm_out_h
                    .as_ref()
                    .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
                let dft_nstm_out_h = self
                    .ws
                    .dft_nstm_out_h
                    .as_ref()
                    .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
                let dft_h = if iter_idx == 0 {
                    dft_stm_out_h
                } else {
                    dft_nstm_out_h
                };
                if iter_idx == 0 {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_overwrite_fp16,
                        stream: self.stream, module: self.module, config: gather_config,
                        args: [
                            slice(dft_h),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in_u32, ft_out_u32, dft_inv_scale_fp16
                        ]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_add_fp16,
                        stream: self.stream, module: self.module, config: gather_config,
                        args: [
                            slice(dft_h),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in_u32, ft_out_u32, dft_inv_scale_fp16
                        ]
                    }?;
                }
            } else {
                let dft = if iter_idx == 0 {
                    &self.ws.dft_stm_out
                } else {
                    &self.ws.dft_nstm_out
                };
                if iter_idx == 0 {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_overwrite,
                        stream: self.stream, module: self.module, config: gather_config,
                        args: [
                            slice(dft),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in_u32, ft_out_u32
                        ]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_add,
                        stream: self.stream, module: self.module, config: gather_config,
                        args: [
                            slice(dft),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in_u32, ft_out_u32
                        ]
                    }?;
                }
            }
        }
        tick("bwd_ft_bw", &self.stream, &mut prof_t0)?;

        Ok(())
    }

    /// Ranger optimizer step (RAdam + Lookahead) を 8 weight group に走らせる。
    /// `RANGER_K` の倍数 step では lookahead lerp を続けて走らせ slow weight と
    /// master weight を補間する。`run_backward_kernels` の直後に呼ぶ。
    fn run_optimizer_step(&mut self, lr: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.step_count += 1;
        let (step_size, denom) =
            radam_compute_step_size_denom(self.step_count, BETA1, BETA2, N_SMA_THRESHOLD);

        let ft_in = self.ws.ft_in;
        let ft_out = self.id.ft_out;
        let l1_out = self.id.l1_out;
        let l2_out = self.id.l2_out;
        let ft_w_n = (ft_in * ft_out) as u32;
        let ft_b_n = ft_out as u32;
        let l1_w_n = (2 * ft_out * l1_out) as u32;
        let l1_b_n = l1_out as u32;
        let l2_w_n = (l1_out * l2_out) as u32;
        let l2_b_n = l2_out as u32;
        let l3_w_n = l2_out as u32;
        let l3_b_n = 1_u32;

        // ft_w optimizer: 2 つの opt-in flag で 4 通りに分岐する (LayerStack と同じパターン)。
        //  - `--ft-fp16`: FP16 mirror (`ft_w_h`) 同時更新版 (`*_mirror`) を使い、forward 用
        //    mirror を別 cast kernel 無しで同期する。
        //  - `--fp16-opt-state`: m / v を `f16` で読み書きする `*_f16state` 系を使う (DRAM
        //    traffic 半減、`FT_OPT_M_SCALE` / `FT_OPT_V_SCALE` で scale 付き格納)。
        // 他 7 group は moment が小さく `f16` 化の意味が無いので常に `radam_step`。
        match (&mut self.ft_w_m, &mut self.ft_w_v) {
            (MomentBuf::F16(ft_w_m), MomentBuf::F16(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                    cuda_launch! {
                        kernel: radam_step_f16state_mirror,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n as usize),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W,
                               FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: radam_step_f16state,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n as usize),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W,
                               FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n]
                    }?;
                }
            }
            (MomentBuf::F32(ft_w_m), MomentBuf::F32(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                    cuda_launch! {
                        kernel: radam_step_fp16_mirror, stream: self.stream, module: self.module,
                        config: cfg_1d(ft_w_n as usize),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W, ft_w_n]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: radam_step, stream: self.stream, module: self.module,
                        config: cfg_1d(ft_w_n as usize),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), lr, step_size, denom, self.weight_decay,
                               BETA1, BETA2, EPS, MIN_W, MAX_W, ft_w_n]
                    }?;
                }
            }
            _ => unreachable!("ft_w m and v moment buffers always share precision"),
        }

        // 残り 7 group (ft_b, l1_w/b, l2_w/b, l3_w/b) は moment buffer 縮小余地が無いので
        // 常に FP32 master + `radam_step`。FT 以外の `f16` 化は本フラグの範囲外。
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(ft_b_n as usize),
            args: [slice_mut(self.ft_b), slice_mut(self.ft_b_m), slice_mut(self.ft_b_v),
                   slice_mut(self.ft_b_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, ft_b_n]
        }?;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l1_w_n as usize),
            args: [slice_mut(self.l1_w), slice_mut(self.l1_w_m), slice_mut(self.l1_w_v),
                   slice_mut(self.l1_w_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, l1_w_n]
        }?;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l1_b_n as usize),
            args: [slice_mut(self.l1_b), slice_mut(self.l1_b_m), slice_mut(self.l1_b_v),
                   slice_mut(self.l1_b_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, l1_b_n]
        }?;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l2_w_n as usize),
            args: [slice_mut(self.l2_w), slice_mut(self.l2_w_m), slice_mut(self.l2_w_v),
                   slice_mut(self.l2_w_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, l2_w_n]
        }?;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l2_b_n as usize),
            args: [slice_mut(self.l2_b), slice_mut(self.l2_b_m), slice_mut(self.l2_b_v),
                   slice_mut(self.l2_b_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, l2_b_n]
        }?;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l3_w_n as usize),
            args: [slice_mut(self.l3_w), slice_mut(self.l3_w_m), slice_mut(self.l3_w_v),
                   slice_mut(self.l3_w_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, l3_w_n]
        }?;
        cuda_launch! {
            kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l3_b_n as usize),
            args: [slice_mut(self.l3_b), slice_mut(self.l3_b_m), slice_mut(self.l3_b_v),
                   slice_mut(self.l3_b_grad), lr, step_size, denom, self.weight_decay,
                   BETA1, BETA2, EPS, MIN_W, MAX_W, l3_b_n]
        }?;

        if self.step_count.is_multiple_of(RANGER_K) {
            // ft_w lookahead lerp: lerp は radam の後に ft_w を再度書き換えるので、
            // `ft_fp16` 時は mirror 同時更新版で `ft_w_h` を lerp 後の最終値に同期する。
            if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                cuda_launch! {
                    kernel: ranger_lookahead_lerp_fp16_mirror, stream: self.stream, module: self.module,
                    config: cfg_1d(ft_w_n as usize),
                    args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), slice_mut(ft_w_h),
                           RANGER_ALPHA, ft_w_n]
                }?;
            } else {
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(ft_w_n as usize),
                    args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), RANGER_ALPHA, ft_w_n]
                }?;
            }
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(ft_b_n as usize),
                args: [slice_mut(self.ft_b), slice_mut(self.ft_b_slow), RANGER_ALPHA, ft_b_n]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(l1_w_n as usize),
                args: [slice_mut(self.l1_w), slice_mut(self.l1_w_slow), RANGER_ALPHA, l1_w_n]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(l1_b_n as usize),
                args: [slice_mut(self.l1_b), slice_mut(self.l1_b_slow), RANGER_ALPHA, l1_b_n]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(l2_w_n as usize),
                args: [slice_mut(self.l2_w), slice_mut(self.l2_w_slow), RANGER_ALPHA, l2_w_n]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(l2_b_n as usize),
                args: [slice_mut(self.l2_b), slice_mut(self.l2_b_slow), RANGER_ALPHA, l2_b_n]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(l3_w_n as usize),
                args: [slice_mut(self.l3_w), slice_mut(self.l3_w_slow), RANGER_ALPHA, l3_w_n]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                config: cfg_1d(l3_b_n as usize),
                args: [slice_mut(self.l3_b), slice_mut(self.l3_b_slow), RANGER_ALPHA, l3_b_n]
            }?;
        }
        Ok(())
    }

    /// 現在の device 上の f32 weight を `SimpleWeights` (host 側 f32 row-major) に
    /// 書き出す。
    ///
    /// 重み layout の対応:
    /// - `ft_w` / `ft_b` / `l3_w` / `l3_b` : device と `SimpleWeights` で同 layout (転置不要)。
    /// - `l1_w` : device は `[in=2*ft_out, out=l1_out]` (`l1_w[in*l1_out + out]`、
    ///   `dense_mm_fwd` の weight pattern)、`SimpleWeights` は `[out=l1_out, in=2*ft_out]` 行優先
    ///   (`l1_w[out*(2*ft_out) + in]`、`save_quantised` の i8 量子化が前提とする out-major 並び)
    ///   → host 側で in-major → out-major に転置する。
    /// - `l2_w` : 同パターンの転置 (`[l1_out, l2_out]` → `[l2_out, l1_out]`)。
    fn to_simple_weights(&self) -> Result<SimpleWeights, Box<dyn std::error::Error>> {
        let id = self.id;
        let ft_out = id.ft_out;
        let l1_out = id.l1_out;
        let l2_out = id.l2_out;
        let l1_in = 2 * ft_out;

        let ft_w = self.ft_w.to_host_vec(&self.stream)?;
        let ft_b = self.ft_b.to_host_vec(&self.stream)?;
        let l1_w_in_major = self.l1_w.to_host_vec(&self.stream)?;
        let l1_b = self.l1_b.to_host_vec(&self.stream)?;
        let l2_w_in_major = self.l2_w.to_host_vec(&self.stream)?;
        let l2_b = self.l2_b.to_host_vec(&self.stream)?;
        let l3_w = self.l3_w.to_host_vec(&self.stream)?;
        let l3_b = self.l3_b.to_host_vec(&self.stream)?;

        let mut l1_w = vec![0.0_f32; l1_out * l1_in];
        for out in 0..l1_out {
            for inp in 0..l1_in {
                l1_w[out * l1_in + inp] = l1_w_in_major[inp * l1_out + out];
            }
        }
        let mut l2_w = vec![0.0_f32; l2_out * l1_out];
        for out in 0..l2_out {
            for inp in 0..l1_out {
                l2_w[out * l1_out + inp] = l2_w_in_major[inp * l2_out + out];
            }
        }

        Ok(SimpleWeights {
            id,
            fv_scale: self.fv_scale,
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
        })
    }

    /// `SimpleWeights` を device 上に upload して現在の重み・lookahead slow を置き換える。
    /// `m` / `v` / `grad` は 0 リセット、`step_count` を 0 に戻す (Ranger を最初から
    /// やり直すのと等価)。`load_simple_weights` 後の slow weight は upload した weight
    /// と同値 (lookahead が `w == slow` 状態から始まる、`new` と同じ規約)。
    ///
    /// `w.id` が現トレーナの `id` と一致しなければ early reject する (異 topology /
    /// feature set の weight は受け入れない)。`fv_scale` は受け入れた値で上書きする。
    ///
    /// device buffer の layout 変換は [`Self::to_simple_weights`] と逆向きで、L1/L2 weight
    /// は host 側で out-major → in-major に転置してから upload する。
    fn load_simple_weights(&mut self, w: &SimpleWeights) -> Result<(), Box<dyn std::error::Error>> {
        if w.id != self.id {
            return Err(format!(
                "SimpleGpuTrainer::load_simple_weights: id mismatch \
                 (trainer ft_in={}, ft_out={}, l1_out={}, l2_out={}, activation={}, \
                 feature_set={}; weights ft_in={}, ft_out={}, l1_out={}, l2_out={}, \
                 activation={}, feature_set={})",
                self.id.ft_in(),
                self.id.ft_out,
                self.id.l1_out,
                self.id.l2_out,
                self.id.activation.canonical_name(),
                self.id.feature_set.canonical_name(),
                w.id.ft_in(),
                w.id.ft_out,
                w.id.l1_out,
                w.id.l2_out,
                w.id.activation.canonical_name(),
                w.id.feature_set.canonical_name(),
            )
            .into());
        }
        let ft_out = self.id.ft_out;
        let l1_out = self.id.l1_out;
        let l2_out = self.id.l2_out;
        let l1_in = 2 * ft_out;

        let mut l1_w_in_major = vec![0.0_f32; l1_in * l1_out];
        for out in 0..l1_out {
            for inp in 0..l1_in {
                l1_w_in_major[inp * l1_out + out] = w.l1_w[out * l1_in + inp];
            }
        }
        let mut l2_w_in_major = vec![0.0_f32; l1_out * l2_out];
        for out in 0..l2_out {
            for inp in 0..l1_out {
                l2_w_in_major[inp * l2_out + out] = w.l2_w[out * l1_out + inp];
            }
        }

        // weight 本体 (master と slow の両方を同値で初期化)。
        self.ft_w = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_b = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.ft_b_slow = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.l1_w = DeviceBuffer::from_host(&self.stream, &l1_w_in_major)?;
        self.l1_w_slow = DeviceBuffer::from_host(&self.stream, &l1_w_in_major)?;
        self.l1_b = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l2_w = DeviceBuffer::from_host(&self.stream, &l2_w_in_major)?;
        self.l2_w_slow = DeviceBuffer::from_host(&self.stream, &l2_w_in_major)?;
        self.l2_b = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l2_b_slow = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l3_w = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_w_slow = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_b = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;
        self.l3_b_slow = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;

        // m / v / grad を 0 リセット、step_count を 0 に戻す (Ranger を最初から)。
        // ft_w の m / v は [`MomentBuf`] で `--fp16-opt-state` 精度を保つため `zeroed`
        // で作り直す (`memset_zero` が `MomentBuf` を取らないため)。
        let ft_w_n = self.id.ft_in() * self.id.ft_out;
        self.ft_w_m = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        self.ft_w_v = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        for buf in [
            &self.ft_w_grad,
            &self.ft_b_m,
            &self.ft_b_v,
            &self.ft_b_grad,
            &self.l1_w_m,
            &self.l1_w_v,
            &self.l1_w_grad,
            &self.l1_b_m,
            &self.l1_b_v,
            &self.l1_b_grad,
            &self.l2_w_m,
            &self.l2_w_v,
            &self.l2_w_grad,
            &self.l2_b_m,
            &self.l2_b_v,
            &self.l2_b_grad,
            &self.l3_w_m,
            &self.l3_w_v,
            &self.l3_w_grad,
            &self.l3_b_m,
            &self.l3_b_v,
            &self.l3_b_grad,
        ] {
            memset_zero(&self.stream, buf)?;
        }
        self.step_count = 0;
        self.fv_scale = w.fv_scale;
        Ok(())
    }

    /// resume 用 raw f32 checkpoint を `path` に atomic に書き出す (LayerStack の
    /// [`GpuTrainer::save_raw_checkpoint`] と同 format / 同方針)。
    ///
    /// 8 weight group の順序: `ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b`。
    /// 各 group は `len u64 + w[f32×len] + m[f32×len] + v[f32×len] + slow[f32×len]`。
    /// L1/L2 weight は device-native `[in, out]` 並びそのまま書く (resume 互換性は
    /// device 上の layout で完結する)。
    fn save_raw_checkpoint(
        &self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let run_id = if run_id.len() > MAX_RUN_ID_BYTES {
            eprintln!(
                "[train] warning: producer run id ({} bytes) exceeds {MAX_RUN_ID_BYTES}; \
                 omitting it from {} (resume lineage parent will be unresolved)",
                run_id.len(),
                path.display()
            );
            ""
        } else {
            run_id
        };

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = {
            let mut p = path.as_os_str().to_os_string();
            p.push(".tmp");
            std::path::PathBuf::from(p)
        };

        let groups = self.raw_ckpt_groups();
        let topology: [u64; 3] = [
            self.id.ft_out as u64,
            self.id.l1_out as u64,
            self.id.l2_out as u64,
        ];
        let ft_w_n = self.id.ft_in() * self.id.ft_out;

        let write_tmp = || -> Result<(), Box<dyn std::error::Error>> {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);
            // header (magic 〜 num_groups)。format 上の group 数は ft_w (個別処理) +
            // `raw_ckpt_groups` の 7 = 8 (Simple は 8 weight group)。
            write_raw_ckpt_header(
                &mut w,
                &RawCkptArch {
                    feature_set: self.id.feature_set,
                    arch_kind: ArchKind::Simple,
                    ft_out: self.id.ft_out as u64,
                    topology: &topology,
                },
                run_id,
                superbatch as u64,
                self.step_count,
                (groups.len() + 1) as u64,
            )?;

            // group 0: ft_w。`m` / `v` は `--fp16-opt-state` で `f16` 格納だが、
            // checkpoint は常に真値 `f32` で書く (mode 非依存・format version 不変、
            // resume 時に当該 run の精度へ再 quantize される)。LayerStack の同 file format
            // 規約と同じ。
            {
                let w_host = self.ft_w.to_host_vec(&self.stream)?;
                let m_host = self.ft_w_m.to_host_f32(&self.stream, FT_OPT_M_SCALE)?;
                let v_host = self.ft_w_v.to_host_f32(&self.stream, FT_OPT_V_SCALE)?;
                let slow_host = self.ft_w_slow.to_host_vec(&self.stream)?;
                for (label, got) in [
                    ("w", w_host.len()),
                    ("m", m_host.len()),
                    ("v", v_host.len()),
                    ("slow", slow_host.len()),
                ] {
                    if got != ft_w_n {
                        return Err(format!(
                            "raw checkpoint: group ft_w {label} buffer len {got} != expected {ft_w_n}"
                        )
                        .into());
                    }
                }
                w.write_all(&(ft_w_n as u64).to_le_bytes())?;
                write_f32_slice(&mut w, &w_host)?;
                write_f32_slice(&mut w, &m_host)?;
                write_f32_slice(&mut w, &v_host)?;
                write_f32_slice(&mut w, &slow_host)?;
            }

            for (name, expected_len, w_buf, m_buf, v_buf, slow_buf) in groups {
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

    /// `--resume` 用に raw f32 checkpoint を読み戻す。返り値は完了 `(superbatch,
    /// producer run id)` で、caller は通常 `superbatch + 1` から resume する。
    ///
    /// header (`arch_kind=Simple`, `topology=[ft_out, l1_out, l2_out]`, feature set)
    /// は [`read_raw_ckpt_header`] が照合する。8 group 各 `(w, m, v, slow)` を
    /// device へ upload し直し、`step_count` を復元する。`grad` は触らない
    /// (step ごとに memset される)。
    fn load_raw_checkpoint(
        &mut self,
        path: &Path,
    ) -> Result<(usize, Option<String>), Box<dyn std::error::Error>> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let topology: [u64; 3] = [
            self.id.ft_out as u64,
            self.id.l1_out as u64,
            self.id.l2_out as u64,
        ];
        let header = read_raw_ckpt_header(
            &mut r,
            &RawCkptArch {
                feature_set: self.id.feature_set,
                arch_kind: ArchKind::Simple,
                ft_out: self.id.ft_out as u64,
                topology: &topology,
            },
        )?;

        // format 上の group 数は ft_w (個別処理) + `raw_ckpt_groups` の 7 = 8。
        let expected_groups: [(&'static str, usize); 7] = {
            let g = self.raw_ckpt_groups();
            [
                (g[0].0, g[0].1),
                (g[1].0, g[1].1),
                (g[2].0, g[2].1),
                (g[3].0, g[3].1),
                (g[4].0, g[4].1),
                (g[5].0, g[5].1),
                (g[6].0, g[6].1),
            ]
        };
        let total_groups = expected_groups.len() + 1;
        if header.num_groups != total_groups as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint num_groups {} != expected {total_groups}",
                header.num_groups
            )));
        }

        let read_group = |r: &mut std::io::BufReader<std::fs::File>,
                          name: &str,
                          expected_len: usize|
         -> Result<RawCkptGroup, Box<dyn std::error::Error>> {
            let mut buf8 = [0u8; 8];
            read_exact_or_invalid(r, &mut buf8, &format!("group {name} len"))?;
            let len_u64 = u64::from_le_bytes(buf8);
            let len: usize = len_u64.try_into().map_err(|_| {
                invalid_data(format!(
                    "raw checkpoint group {name} len {len_u64} exceeds usize::MAX"
                ))
            })?;
            if len != expected_len {
                return Err(invalid_data(format!(
                    "raw checkpoint group {name} len mismatch: got {len}, want {expected_len} \
                     (network architecture mismatch)"
                )));
            }
            let w_host = read_f32_vec_io(r, len, &format!("group {name} w"))?;
            let m_host = read_f32_vec_io(r, len, &format!("group {name} m"))?;
            let v_host = read_f32_vec_io(r, len, &format!("group {name} v"))?;
            let slow_host = read_f32_vec_io(r, len, &format!("group {name} slow"))?;
            Ok((w_host, m_host, v_host, slow_host))
        };

        // group 0 は ft_w (個別処理、`m` / `v` は当該 run の精度へ再 quantize)。
        let ft_w_loaded = read_group(&mut r, "ft_w", self.id.ft_in() * self.id.ft_out)?;
        let mut loaded: Vec<RawCkptGroup> = Vec::with_capacity(expected_groups.len());
        for (name, expected_len) in expected_groups {
            loaded.push(read_group(&mut r, name, expected_len)?);
        }

        // host → device upload。ft_w の m / v は `--fp16-opt-state` の現在精度へ
        // 量子化して載せ直す (checkpoint は真値 f32、mode 非依存)。
        let (ftw_w, ftw_m, ftw_v, ftw_slow) = &ft_w_loaded;
        self.ft_w = DeviceBuffer::from_host(&self.stream, ftw_w)?;
        self.ft_w_m =
            MomentBuf::from_host_f32(&self.stream, ftw_m, self.fp16_opt_state, FT_OPT_M_SCALE)?;
        self.ft_w_v =
            MomentBuf::from_host_f32(&self.stream, ftw_v, self.fp16_opt_state, FT_OPT_V_SCALE)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, ftw_slow)?;

        macro_rules! up {
            ($idx:expr, $w:ident, $m:ident, $v:ident, $slow:ident) => {{
                let (w, m, v, s) = &loaded[$idx];
                self.$w = DeviceBuffer::from_host(&self.stream, w)?;
                self.$m = DeviceBuffer::from_host(&self.stream, m)?;
                self.$v = DeviceBuffer::from_host(&self.stream, v)?;
                self.$slow = DeviceBuffer::from_host(&self.stream, s)?;
            }};
        }
        up!(0, ft_b, ft_b_m, ft_b_v, ft_b_slow);
        up!(1, l1_w, l1_w_m, l1_w_v, l1_w_slow);
        up!(2, l1_b, l1_b_m, l1_b_v, l1_b_slow);
        up!(3, l2_w, l2_w_m, l2_w_v, l2_w_slow);
        up!(4, l2_b, l2_b_m, l2_b_v, l2_b_slow);
        up!(5, l3_w, l3_w_m, l3_w_v, l3_w_slow);
        up!(6, l3_b, l3_b_m, l3_b_v, l3_b_slow);

        self.step_count = header.step_count;
        Ok((header.superbatch, header.producer_run_id))
    }

    /// `save_raw_checkpoint` / `load_raw_checkpoint` で iterate する 7 weight group の
    /// `(name, len, w, m, v, slow)`。ft_w は `m` / `v` が [`MomentBuf`] (`f32`/`f16`) で
    /// 型が異なるため本 array から除外し、save/load 側で個別処理する (format の group
    /// 順は ft_w が先頭で不変)。
    fn raw_ckpt_groups(&self) -> [SimpleRawCkptGroupEntry<'_>; 7] {
        let ft_b_n = self.id.ft_out;
        let l1_w_n = 2 * self.id.ft_out * self.id.l1_out;
        let l1_b_n = self.id.l1_out;
        let l2_w_n = self.id.l1_out * self.id.l2_out;
        let l2_b_n = self.id.l2_out;
        let l3_w_n = self.id.l2_out;
        let l3_b_n = 1;
        [
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

    /// held-out validation 用 forward-only。weight は更新せず、batch 全体の `Σerr²`
    /// と position ごとの net 出力 (`b` 個) を返す。
    fn validate(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(StepOutput {
                loss: 0.0,
                net_output: Vec::new(),
            });
        }
        self.ws.check_batch_capacity(b)?;
        self.run_forward_kernels(batch, wdl_lambda, loss, false)?;
        let net_output = self.ws.net_output.to_host_vec(&self.stream)?[..b].to_vec();
        let loss_host = self.loss_acc.to_host_vec(&self.stream)?;
        Ok(StepOutput {
            loss: loss_host[0],
            net_output,
        })
    }
}

impl TrainerBackend for SimpleGpuTrainer {
    fn train_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<f64> {
        if batch.feature_set != self.id.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.id.feature_set.canonical_name(),
            )));
        }
        // Simple は bucket を持たないため `bucket_idx` を kernel に渡さない (caller の
        // `TrainerBackend` 契約上は受け取るが、`from_batch_ref_bucketless` で空 slice 化
        // して下流 backend が `bucket_idx` に触れない経路を強制する)。
        let _ = bucket_idx;
        let data = BatchData::from_batch_ref_bucketless(batch);
        self.step(&data, lr, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("SimpleGpuTrainer::step failed: {e}")))
    }

    fn validate_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<ValidationStepOutput> {
        if batch.feature_set != self.id.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.id.feature_set.canonical_name(),
            )));
        }
        let _ = bucket_idx;
        let data = BatchData::from_batch_ref_bucketless(batch);
        let out = self.validate(&data, wdl_lambda, loss).map_err(|e| {
            std::io::Error::other(format!("SimpleGpuTrainer::validate failed: {e}"))
        })?;
        Ok(ValidationStepOutput {
            sum_sq_err: out.loss,
            net_output: out.net_output,
        })
    }

    fn flush_pending_loss(&mut self) -> std::io::Result<f64> {
        self.loss_ring.flush_pending_loss().map_err(|e| {
            std::io::Error::other(format!(
                "SimpleGpuTrainer::loss_ring.flush_pending_loss failed: {e}"
            ))
        })
    }

    fn save_checkpoint(&mut self, path: &Path) -> std::io::Result<()> {
        let weights = self.to_simple_weights().map_err(|e| {
            std::io::Error::other(format!("SimpleGpuTrainer::to_simple_weights failed: {e}"))
        })?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        weights.save_quantised(&mut writer)?;
        use std::io::Write;
        writer.flush()?;
        Ok(())
    }

    fn save_resume_checkpoint(
        &mut self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
    ) -> std::io::Result<()> {
        self.save_raw_checkpoint(path, superbatch, run_id)
            .map_err(|e| match e.downcast::<std::io::Error>() {
                Ok(io_err) => *io_err,
                Err(other) => std::io::Error::other(format!(
                    "SimpleGpuTrainer::save_raw_checkpoint failed: {other}"
                )),
            })
    }
}

/// Simple アーキ用 smoke test。preset `256x2-32-32` (HalfKaHmMerged + CReLU) で
/// `SimpleGpuTrainer` を構築し、以下 4 段を踏む:
/// 1. forward sanity — CReLU + SCReLU 両活性化 + sigmoid / WRM 両 loss kernel を
///    1 step ずつ launch して loss が finite であること。
/// 2. step が gradient を正しく配線していることを 10 step の loss 推移で確認する
///    (loss が初期値より下がる)。
/// 3. `to_simple_weights` → `save_quantised` → `SimpleWeights::load` →
///    `load_simple_weights` の量子化 round-trip 後の forward が finite に走る。
/// 4. `save_raw_checkpoint` → 新 trainer での `load_raw_checkpoint` 後の forward が
///    元と完全一致 (raw f32 round-trip の exact preservation)。
fn simple_smoke_test() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    println!("[smoke/simple] CUDA context created, loading kernel module...");
    let id = SimpleId {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    let smoke_fv_scale = 16_i32;
    let smoke_weight_decay = 1e-7_f32;
    let mut trainer = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        smoke_weight_decay,
        smoke_fv_scale,
        false,
        false,
        false,
        false,
    )?;
    let params = id.ft_in() * id.ft_out
        + id.ft_out
        + 2 * id.ft_out * id.l1_out
        + id.l1_out
        + id.l1_out * id.l2_out
        + id.l2_out
        + id.l2_out
        + 1;
    println!(
        "[smoke/simple] SimpleGpuTrainer ready: 8 weight groups, ~{:.1}M params total \
         (ft_in={}, ft_out={}, l1_out={}, l2_out={}, activation={})",
        params as f64 / 1.0e6,
        id.ft_in(),
        id.ft_out,
        id.l1_out,
        id.l2_out,
        id.activation.canonical_name(),
    );
    trainer.assert_all_weights_finite()?;

    let batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    let loss = trainer.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!("[smoke/simple] forward 1 (sigmoid-MSE, crelu): loss = {loss:.6e}");
    if !loss.is_finite() {
        return Err(format!("forward 1 loss = {loss} is not finite").into());
    }
    trainer.assert_all_weights_finite()?;

    let id_screlu = SimpleId {
        activation: SimpleActivation::SCReLU,
        ..id
    };
    let mut trainer_screlu = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id_screlu,
        smoke_weight_decay,
        smoke_fv_scale,
        false,
        false,
        false,
        false,
    )?;
    let loss_screlu = trainer_screlu.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!("[smoke/simple] forward 2 (sigmoid-MSE, screlu): loss = {loss_screlu:.6e}");
    if !loss_screlu.is_finite() {
        return Err(format!("forward 2 loss = {loss_screlu} is not finite").into());
    }

    let loss_wrm_val = trainer.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
    println!("[smoke/simple] forward 3 (win-rate-model, crelu): loss = {loss_wrm_val:.6e}");
    if !loss_wrm_val.is_finite() {
        return Err(format!("forward 3 (wrm) loss = {loss_wrm_val} is not finite").into());
    }
    println!("[smoke/simple] forward sanity OK ✓");

    // 10-step gradient direction check。`smoke_dummy` は score=0 / wdl=0.5 で
    // sigmoid loss の最小点 (target=0.5、init weights が小さく net_output≈0、
    // p≈0.5) に近すぎるため、score を非ゼロにして target を 0.5 から動かし
    // 学習信号を作る。
    let mut training_batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    for s in training_batch.score.iter_mut() {
        *s = 200.0;
    }
    for w in training_batch.wdl.iter_mut() {
        *w = 0.8;
    }
    let lr = 1e-1_f32;
    let initial_loss = trainer.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    for step_idx in 0..10 {
        let step_loss = trainer.step(&training_batch.as_ref(), lr, 0.0, SMOKE_LOSS_SIGMOID)?;
        if !step_loss.is_finite() {
            return Err(format!("step {step_idx} loss = {step_loss} is not finite").into());
        }
    }
    trainer.assert_all_weights_finite()?;
    let final_loss = trainer.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!("[smoke/simple] 10-step training: loss {initial_loss:.6e} -> {final_loss:.6e}");
    // NaN は `>=` でも `<` でも false になるので、`is_finite` で別途弾く。
    if !final_loss.is_finite() || final_loss >= initial_loss {
        return Err(format!(
            "10-step training did not decrease loss: initial = {initial_loss}, final = {final_loss} \
             (backward / optimizer wiring likely broken)"
        )
        .into());
    }
    println!("[smoke/simple] gradient direction OK ✓");

    // 量子化 round-trip: `to_simple_weights` -> `save_quantised` -> `SimpleWeights::load`
    // -> `load_simple_weights`。量子化丸めで weight 値は変わるため loss は厳密一致しない;
    // ここでは round-trip が format 上完結し、再 upload した weight で forward が finite
    // に走ることを確認する。bit-identical な round-trip は次段の raw checkpoint で確認する。
    let weights = trainer.to_simple_weights()?;
    let mut quantised_bytes = Vec::new();
    weights.save_quantised(&mut quantised_bytes)?;
    let reloaded = SimpleWeights::load(&mut std::io::Cursor::new(&quantised_bytes), id)?;
    if reloaded.fv_scale != smoke_fv_scale {
        return Err(format!(
            "SimpleWeights round-trip: fv_scale mismatch (got {}, want {smoke_fv_scale})",
            reloaded.fv_scale
        )
        .into());
    }
    let mut trainer_q = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        smoke_weight_decay,
        smoke_fv_scale,
        false,
        false,
        false,
        false,
    )?;
    trainer_q.load_simple_weights(&reloaded)?;
    let loss_q = trainer_q.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!(
        "[smoke/simple] quantised round-trip: trained loss {final_loss:.6e} \
         -> reloaded loss {loss_q:.6e} ({} bytes)",
        quantised_bytes.len()
    );
    if !loss_q.is_finite() {
        return Err(format!(
            "quantised round-trip forward loss = {loss_q} is not finite \
             (to_simple_weights / load_simple_weights transpose direction or format mismatch)"
        )
        .into());
    }

    // raw f32 checkpoint round-trip: save -> 新 trainer で load -> 同 batch で forward。
    // raw checkpoint は f32 を bit-identical に保つので loss は完全一致するはず。
    let raw_path = std::env::temp_dir().join(format!("simple-smoke-{}.ckpt", std::process::id()));
    trainer.save_raw_checkpoint(&raw_path, 1, "smoke")?;
    let mut trainer_r = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        smoke_weight_decay,
        smoke_fv_scale,
        false,
        false,
        false,
        false,
    )?;
    let (sb, _producer) = trainer_r.load_raw_checkpoint(&raw_path)?;
    if sb != 1 {
        return Err(format!("raw round-trip superbatch mismatch: got {sb}, want 1").into());
    }
    let loss_r = trainer_r.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    let _ = std::fs::remove_file(&raw_path);
    let loss_r_rel = ((loss_r - final_loss).abs() / final_loss.abs().max(1e-12)) as f32;
    println!(
        "[smoke/simple] raw checkpoint round-trip: loss {final_loss:.6e} -> {loss_r:.6e} \
         (relative {loss_r_rel:.3e})"
    );
    if !loss_r.is_finite() || loss_r_rel > 1e-6 {
        return Err(format!(
            "raw round-trip loss mismatch: final = {final_loss}, reloaded = {loss_r} \
             (raw f32 should be bit-identical; group ordering / topology / step_count likely broken)"
        )
        .into());
    }

    println!(
        "[smoke/simple] PASSED — forward + gradient + quantised round-trip + raw round-trip OK"
    );
    Ok(())
}

fn smoke_test(arch_kind: ArchKind) -> Result<(), Box<dyn std::error::Error>> {
    // Simple アーキは別 host pipeline (SimpleGpuTrainer) を持つので smoke も別系統。
    if arch_kind == ArchKind::Simple {
        return simple_smoke_test();
    }
    let ctx = CudaContext::new(0)?;
    println!("[smoke] CUDA context created, loading kernel module...");
    // smoke は production feature set (`halfka-hm-merged`) で動作確認する。
    let feature_set = FeatureSet::HalfKaHmMerged.spec();
    // workspace を smoke の固定 batch 分で確保 (smoke は TF32 OFF 固定で動作確認、
    // training は CLI の `--tf32` を pass する)。
    let mut trainer = GpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        false,
        false,
        false,
        false,
        feature_set,
        0.0,
    )?;
    println!(
        "[smoke] GpuTrainer ready: 10 weight groups, ~{:.1}M params total",
        (feature_set.ft_in() * FT_OUT
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

    // `RSHOGI_NNUE_LAYERSTACK_REF_BIN` に既存の量子化 checkpoint (`.bin`) path を
    // 指定すると、その weight を注入して forward + backward + save を一通り走らせる。
    // 未設定なら random-init での smoke のみ。
    let layerstack_ref = std::env::var("RSHOGI_NNUE_LAYERSTACK_REF_BIN").ok();
    if let Some(ref_path) = layerstack_ref
        .as_deref()
        .filter(|p| std::path::Path::new(p).exists())
    {
        println!("[smoke] loading reference checkpoint from {ref_path} ...");
        let mut reader = std::io::BufReader::new(std::fs::File::open(ref_path)?);
        let weights = LayerStackWeights::load_quantised(&mut reader, feature_set)?;
        trainer.load_layerstack_weights(&weights)?;
        trainer.assert_all_weights_finite()?;
        println!("[smoke] reference weights injected, all finite ✓");

        // forward + step 1 batch (sigmoid-MSE、golden forward/backward/save 経路)
        let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch.as_ref(), lr, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke] step 1 (post reference-inject, sigmoid-MSE): loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

        // save back as our quantised.bin
        let out_path = std::env::temp_dir().join("our_quantised.bin");
        let out_path_str = out_path.display();
        println!("[smoke] saving trained weights to {out_path_str} ...");
        let saved_weights = trainer.to_layerstack_weights()?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        saved_weights.save_quantised(&mut writer)?;
        drop(writer);
        let out_size = std::fs::metadata(&out_path)?.len();
        println!("[smoke] wrote {out_path_str}: {out_size} bytes");

        // 追加 step: WRM loss kernel (`loss_wrm`) を runtime でも exercise する。
        // 上で save 済なので weights が変わっても verify 対象 (`out_path`) には影響しない。
        let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        let loss_wrm = trainer.step(&batch.as_ref(), 1e-3_f32, WDL_LAMBDA, SMOKE_LOSS_WRM)?;
        println!("[smoke] step 2 (win-rate-model): loss = {loss_wrm:.6e}");
        if !loss_wrm.is_finite() {
            return Err(format!("step 2 (wrm) loss = {loss_wrm} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 2: all weights finite ✓");
    } else {
        println!(
            "[smoke] (RSHOGI_NNUE_LAYERSTACK_REF_BIN not set or path missing; running random-init smoke only)"
        );
        let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch.as_ref(), lr, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke] step 1 (sigmoid-MSE): loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

        // step 2: WRM loss kernel (`loss_wrm`) を runtime でも exercise する。
        let loss_wrm = trainer.step(&batch.as_ref(), lr, WDL_LAMBDA, SMOKE_LOSS_WRM)?;
        println!("[smoke] step 2 (win-rate-model): loss = {loss_wrm:.6e}");
        if !loss_wrm.is_finite() {
            return Err(format!("step 2 (wrm) loss = {loss_wrm} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 2: all weights finite ✓");

        // save random-init as quantised.bin for verify-nnue check
        let out_path = std::env::temp_dir().join("our_quantised_randinit.bin");
        let out_path_str = out_path.display();
        let saved_weights = trainer.to_layerstack_weights()?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        saved_weights.save_quantised(&mut writer)?;
        drop(writer);
        let out_size = std::fs::metadata(&out_path)?.len();
        println!("[smoke] wrote {out_path_str}: {out_size} bytes");
    }

    println!("[smoke] PASSED — GpuTrainer skeleton OK (LayerStack arch full path)");
    Ok(())
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = if cli.data.is_some() {
        run_training(&cli)
    } else {
        smoke_test(cli.arch.kind())
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
// raw checkpoint format helper tests (GPU 不要)
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
        // magic は format identity。version は後方互換読み (version 1..=3 file の受理)
        // を維持しつつ前進するので、現行値を pin して意図しない変更を検出する。
        assert_eq!(&RAW_CKPT_MAGIC, b"RNRC");
        assert_eq!(RAW_CKPT_VERSION, 4);
    }

    #[test]
    fn invalid_data_helper_makes_invalid_data_error() {
        let e = invalid_data("boom".to_string());
        let io_err = e.downcast::<std::io::Error>().expect("is io::Error");
        assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidData);
        assert!(io_err.to_string().contains("boom"));
    }

    /// arch header を持たない legacy (version 1..=3) raw checkpoint header を組む。
    fn legacy_raw_ckpt_header(
        version: u32,
        fs: shogi_features::FeatureSetSpec,
        run_id: Option<&str>,
        superbatch: u64,
        step_count: u64,
        num_groups: u64,
    ) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&RAW_CKPT_MAGIC);
        b.extend_from_slice(&version.to_le_bytes());
        if version >= 2 {
            let name = fs.canonical_name();
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(fs.ft_in() as u64).to_le_bytes());
            b.extend_from_slice(&(FT_OUT as u64).to_le_bytes());
            b.extend_from_slice(&(fs.max_active() as u64).to_le_bytes());
        }
        if version >= 3 {
            let rid = run_id.unwrap_or("");
            b.extend_from_slice(&(rid.len() as u32).to_le_bytes());
            b.extend_from_slice(rid.as_bytes());
        }
        b.extend_from_slice(&superbatch.to_le_bytes());
        b.extend_from_slice(&step_count.to_le_bytes());
        b.extend_from_slice(&num_groups.to_le_bytes());
        b
    }

    fn layerstack_arch() -> RawCkptArch<'static> {
        RawCkptArch {
            feature_set: FeatureSet::HalfKaHmMerged.spec(),
            arch_kind: ArchKind::LayerStack,
            ft_out: FT_OUT as u64,
            topology: &LAYERSTACK_TOPOLOGY,
        }
    }

    #[test]
    fn raw_ckpt_header_v4_round_trips() {
        let arch = layerstack_arch();
        let mut buf = Vec::new();
        write_raw_ckpt_header(&mut buf, &arch, "net-20260520-1234", 7, 99, 10).unwrap();
        let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
        assert_eq!(h.superbatch, 7);
        assert_eq!(h.step_count, 99);
        assert_eq!(h.num_groups, 10);
        assert_eq!(h.producer_run_id.as_deref(), Some("net-20260520-1234"));
    }

    #[test]
    fn raw_ckpt_header_empty_run_id_round_trips_to_none() {
        let arch = layerstack_arch();
        let mut buf = Vec::new();
        write_raw_ckpt_header(&mut buf, &arch, "", 1, 0, 10).unwrap();
        let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
        assert_eq!(h.producer_run_id, None);
    }

    #[test]
    fn raw_ckpt_header_reads_legacy_v1_v2_v3() {
        let fs = FeatureSet::HalfKaHmMerged.spec();
        let arch = layerstack_arch();
        // v1: header 無し、halfka-hm-merged 固定、arch は暗黙 layerstack。
        let v1 = legacy_raw_ckpt_header(1, fs, None, 3, 30, 10);
        let h1 = read_raw_ckpt_header(&mut Cursor::new(&v1), &arch).unwrap();
        assert_eq!((h1.superbatch, h1.step_count, h1.num_groups), (3, 30, 10));
        assert_eq!(h1.producer_run_id, None);
        // v2: feature set header あり、run id 無し。
        let v2 = legacy_raw_ckpt_header(2, fs, None, 4, 40, 10);
        let h2 = read_raw_ckpt_header(&mut Cursor::new(&v2), &arch).unwrap();
        assert_eq!(h2.superbatch, 4);
        assert_eq!(h2.producer_run_id, None);
        // v3: producer run id あり。
        let v3 = legacy_raw_ckpt_header(3, fs, Some("legacy-run"), 5, 50, 10);
        let h3 = read_raw_ckpt_header(&mut Cursor::new(&v3), &arch).unwrap();
        assert_eq!(h3.superbatch, 5);
        assert_eq!(h3.producer_run_id.as_deref(), Some("legacy-run"));
    }

    #[test]
    fn raw_ckpt_header_rejects_wrong_arch_kind() {
        // fs header は一致するが arch_kind だけ異なる v4 header を read → reject。
        let fs = FeatureSet::HalfKaHmMerged.spec();
        let written = RawCkptArch {
            feature_set: fs,
            arch_kind: ArchKind::Simple,
            ft_out: FT_OUT as u64,
            topology: &LAYERSTACK_TOPOLOGY,
        };
        let mut buf = Vec::new();
        write_raw_ckpt_header(&mut buf, &written, "", 1, 0, 10).unwrap();
        let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch())
            .expect_err("arch kind mismatch must reject");
        assert!(err.to_string().contains("arch kind mismatch"));
    }

    #[test]
    fn raw_ckpt_header_rejects_wrong_topology() {
        let fs = FeatureSet::HalfKaHmMerged.spec();
        let wrong_topo = [1u64, 2, 3, 4];
        let written = RawCkptArch {
            feature_set: fs,
            arch_kind: ArchKind::LayerStack,
            ft_out: FT_OUT as u64,
            topology: &wrong_topo,
        };
        let mut buf = Vec::new();
        write_raw_ckpt_header(&mut buf, &written, "", 1, 0, 10).unwrap();
        let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch())
            .expect_err("topology mismatch must reject");
        assert!(err.to_string().contains("topology dim"));
    }

    #[test]
    fn raw_ckpt_header_rejects_legacy_non_layerstack_request() {
        // version 1..=3 は arch header を持たず暗黙 layerstack。Simple として読もう
        // とすると reject される。
        let fs = FeatureSet::HalfKaHmMerged.spec();
        let v3 = legacy_raw_ckpt_header(3, fs, None, 1, 0, 8);
        let simple_topo = [256u64, 32, 32];
        let simple_arch = RawCkptArch {
            feature_set: fs,
            arch_kind: ArchKind::Simple,
            ft_out: FT_OUT as u64,
            topology: &simple_topo,
        };
        let err = read_raw_ckpt_header(&mut Cursor::new(&v3), &simple_arch)
            .expect_err("legacy checkpoint cannot be read as a non-layerstack arch");
        assert!(err.to_string().contains("predates the arch-kind header"));
    }

    #[test]
    fn raw_ckpt_header_rejects_unsupported_version() {
        let fs = FeatureSet::HalfKaHmMerged.spec();
        let mut buf = legacy_raw_ckpt_header(3, fs, None, 1, 0, 10);
        // magic (4 bytes) 直後の version u32 を範囲外の値に書き換える。
        buf[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch()).is_err());
    }
}

// ===========================================================================
// CLI 構成テスト (clap、GPU 不要)
// ===========================================================================
#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        // clap derive の構成 (global 引数 + 必須サブコマンド) が破綻していないこと。
        Cli::command().debug_assert();
    }

    #[test]
    fn layerstack_subcommand_parses() {
        let cli = Cli::try_parse_from(["nnue-train", "layerstack"]).expect("layerstack subcommand");
        assert_eq!(cli.arch.kind(), ArchKind::LayerStack);
    }

    #[test]
    fn simple_subcommand_parses() {
        let cli = Cli::try_parse_from(["nnue-train", "simple"]).expect("simple subcommand");
        assert_eq!(cli.arch.kind(), ArchKind::Simple);
    }

    #[test]
    fn subcommand_is_required() {
        // サブコマンド未指定はエラー (clap サブコマンド必須化により CLI 文字列互換は破壊)。
        assert!(Cli::try_parse_from(["nnue-train"]).is_err());
    }

    #[test]
    fn shared_args_are_global_around_subcommand() {
        // 共有 (global) 引数は値付き / フラグ いずれもサブコマンドの後ろに置ける。
        let cli = Cli::try_parse_from([
            "nnue-train",
            "layerstack",
            "--ft-fp16",
            "--data",
            "x.psv",
            "--batch-size",
            "4096",
        ])
        .expect("global args after subcommand");
        assert!(cli.ft_fp16);
        assert_eq!(cli.data.as_deref(), Some(std::path::Path::new("x.psv")));
        assert_eq!(cli.batch_size, 4096);
    }

    #[test]
    fn simple_accepts_tf32_flag() {
        // `--tf32` は LayerStack / Simple 両 subcommand で受理される (両方 cuBLAS handle
        // に同 flag を渡す opt-in)。default OFF / 渡せば ON で TF32 TC 有効化。
        let cli = Cli::try_parse_from(["nnue-train", "simple", "--tf32"])
            .expect("simple should accept --tf32");
        match cli.arch {
            ArchCommand::Simple(args) => assert!(args.tf32),
            ArchCommand::LayerStack(_) => panic!("expected Simple subcommand"),
        }
    }

    #[test]
    fn layerstack_specific_arg_rejected_before_subcommand() {
        // layerstack 固有引数 (--progress-coeff) は global ではないので、
        // サブコマンドより前には置けずエラーになる。
        assert!(
            Cli::try_parse_from(["nnue-train", "--progress-coeff", "p.bin", "layerstack"]).is_err()
        );
    }
}

// ===========================================================================
// GPU ↔ CPU reference 数値同等性テスト
//
// 本 module は **GPU 必須**。`#[cfg(test)]` を main.rs 内に置くことで kernel
// symbol (上の `#[kernel]` 群) に直接 path 解決できる (tests/*.rs では bin の
// `#[kernel]` に届かない)。`nnue-trainer` は workspace `--exclude` で CI から
// 外しているので CI には影響しないが、typecheck は通す必要あり
// (`cargo test -p nnue-trainer --release --no-run`)。
//
// 走らせる:
//
// ```bash
// cd bins/nnue_train && CUDA_OXIDE_TARGET=sm_75 cargo-oxide build
// cd ../.. && cargo test -p nnue-trainer --release -- --test-threads=1
// ```
//
// 各テストは小規模 batch (b = 3〜4) で GPU kernel を launch → download → 上の
// `gpu_kernels::layerstack::*_cpu` reference と比較。`-1` padding (sparse index /
// bucket_idx)、全 9 bucket、CReLU 境界値 (ちょうど 0.0 / 1.0 / 負)、NaN 伝搬を含む。
// tolerance: forward / gradient 1e-5、整数/index 出力は完全一致。
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
    use gpu_kernels::pointwise::screlu_fwd::screlu_fwd_cpu;

    /// forward / gradient の f32 tolerance。
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
    /// 比例した f32 round-off drift が出る。`|gpu - cpu| <= tol * (1 + |cpu|)` で判定する。
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

    // -- screlu -------------------------------------------------------------

    #[test]
    fn screlu_fwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let n = 257_usize;
        let mut x = deterministic_floats(n, 1.0);
        x.push(f32::NAN); // NaN propagation: clip(NaN)² → NaN (if-else passes through)
        let n = x.len();
        let mut y_cpu = vec![0.0_f32; n];
        screlu_fwd_cpu(&x, &mut y_cpu, n);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        cuda_launch! {
            kernel: screlu_fwd, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice_mut(y_dev), n as u32]
        }?;
        stream.synchronize()?;
        assert_close("screlu_fwd", &y_dev.to_host_vec(&stream)?, &y_cpu, 0.0);
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

    // -- slice_extract_2d / slice_scatter_2d (LayerStack l1_main / l1_skip shapes) -

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

    // -- concat_l1sqr_main fwd / grad (LayerStack dim 15 + 15 → 30) ----------------

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
    // L1f 実 shape: in_dim=FT_OUT(=1536) は重いので、ここは小さい shape で
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
    fn dense_mm_bwd_input_tiled_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (64, 96)] {
            let out_dim = 16_usize;
            let dy: Vec<f32> = (0..batch * out_dim)
                .map(|i| (i as f32) * 0.013 - 0.4)
                .collect();
            let w: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| (i as f32) * 0.0017 + 0.03)
                .collect();
            let mut dx_cpu = vec![0.0_f32; batch * in_dim];
            dense_mm_bwd_input_cpu(&dy, &w, &mut dx_cpu, batch, in_dim, out_dim);

            let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
            let w_dev = DeviceBuffer::from_host(&stream, &w)?;
            let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * in_dim)?;
            let blocks = (batch / 16) * (in_dim / 16);
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: dense_mm_bwd_input_tiled, stream: stream, module: module,
                config: config,
                args: [slice(dy_dev), slice(w_dev), slice_mut(dx_dev),
                       batch as u32, in_dim as u32, out_dim as u32]
            }?;
            stream.synchronize()?;
            assert_close(
                &format!("dense_mm_bwd_input_tiled b={batch} in={in_dim}"),
                &dx_dev.to_host_vec(&stream)?,
                &dx_cpu,
                TOL,
            );
        }
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_weight_tiled_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        // tiled kernel は in_dim % 16 == 0 && out_dim == 16 && batch % 16 == 0 を要求
        let (_ctx, module, stream) = open_module()?;
        for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (64, 96)] {
            let out_dim = 16_usize;
            let x: Vec<f32> = (0..batch * in_dim)
                .map(|i| (i as f32) * 0.0031 - 0.7)
                .collect();
            let dy: Vec<f32> = (0..batch * out_dim)
                .map(|i| (i as f32) * 0.013 - 0.3)
                .collect();
            let mut dw_cpu = vec![0.0_f32; in_dim * out_dim];
            dense_mm_bwd_weight_cpu(&x, &dy, &mut dw_cpu, batch, in_dim, out_dim);

            let x_dev = DeviceBuffer::from_host(&stream, &x)?;
            let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
            let mut dw_dev = DeviceBuffer::<f32>::zeroed(&stream, in_dim * out_dim)?;
            // launch with block size 256, grid = in_dim/16 blocks
            let blocks = in_dim / 16;
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: dense_mm_bwd_weight_tiled, stream: stream, module: module, config: config,
                args: [slice(x_dev), slice(dy_dev), slice_mut(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32]
            }?;
            stream.synchronize()?;
            assert_close(
                &format!("dense_mm_bwd_weight_tiled b={batch} in={in_dim}"),
                &dw_dev.to_host_vec(&stream)?,
                &dw_cpu,
                TOL,
            );
        }
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
    fn dense_mm_fwd_bucket_tiled_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        // tiled (L1): in_dim % 16 == 0、out_dim == 16、batch % 16 == 0、num_buckets <= 9
        for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (48, 96), (64, 32)] {
            let out_dim = 16_usize;
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
            let blocks = batch / 16;
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: dense_mm_fwd_bucket_tiled_l1, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice(bidx_dev), slice_mut(y_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }?;
            stream.synchronize()?;
            assert_close(
                &format!("dense_mm_fwd_bucket_tiled_l1 b={batch} in={in_dim}"),
                &y_dev.to_host_vec(&stream)?,
                &y_cpu,
                TOL,
            );
        }
        Ok(())
    }

    /// 16-aligned bucket sort + sorted fwd_L1 + inverse permute の合成 pipeline が
    /// `dense_mm_fwd_bucket_cpu` と bit-exact 一致することを確認。fwd_L1 は per-row
    /// independent (k=0..15 加算順保持) のため sort stability に依らず tolerance=0 が成立。
    #[test]
    fn bucket_sort_fwd_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (48, 96), (64, 32)] {
            let out_dim = 16_usize;
            let nb = NUM_BUCKETS;
            let padded = padded_sort_batch(batch);
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

            let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let perm_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
            let bidx_sorted_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
            let mut x_sorted_dev = DeviceBuffer::<f32>::zeroed(&stream, padded * in_dim)?;
            let mut y_sorted_dev = DeviceBuffer::<f32>::zeroed(&stream, padded * out_dim)?;
            let y_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * out_dim)?;

            memset_minus_one_i32(&stream, &perm_dev)?;
            memset_minus_one_i32(&stream, &bidx_sorted_dev)?;

            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }?;
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * in_dim),
                args: [slice(x_dev), slice(perm_dev), slice_mut(x_sorted_dev),
                       padded as u32, in_dim as u32]
            }?;
            let blocks = padded / 16;
            cuda_launch! {
                kernel: dense_mm_fwd_bucket_tiled_l1_sorted, stream: stream, module: module,
                config: LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [slice(x_sorted_dev), slice(w_dev), slice(bias_dev), slice(bidx_sorted_dev),
                       slice_mut(y_sorted_dev), padded as u32, in_dim as u32, out_dim as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: inverse_permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(y_sorted_dev), slice(perm_dev), slice(y_dev),
                       padded as u32, out_dim as u32]
            }?;
            stream.synchronize()?;

            let y_gpu = y_dev.to_host_vec(&stream)?;
            for (i, (&g, &c)) in y_gpu.iter().zip(y_cpu.iter()).enumerate() {
                if g != c {
                    panic!(
                        "bucket_sort_fwd_l1 b={batch} in={in_dim} idx={i}: gpu={g} cpu={c} delta={}",
                        g - c
                    );
                }
            }
        }
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
    fn dense_mm_bwd_weight_bucket_tiled_l2_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        // tiled L2 (in_dim=30, out_dim=32, num_buckets=9)
        let (_ctx, module, stream) = open_module()?;
        for &batch in &[16_usize, 64, 256, 1024] {
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
            let dw_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim * in_dim)?;
            let num_splits = 8_usize;
            let config = LaunchConfig {
                grid_dim: (num_splits as u32, 1, 1),
                block_dim: (960, 1, 1), // 32 × 30
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l2, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }?;
            stream.synchronize()?;
            assert_close_rel(
                &format!("dense_mm_bwd_weight_bucket_tiled_l2 b={batch}"),
                &dw_dev.to_host_vec(&stream)?,
                &dw_cpu,
                TOL,
            );
        }
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_weight_bucket_tiled_l3_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        // tiled L3 (in_dim=32, out_dim=1, num_buckets=9)
        let (_ctx, module, stream) = open_module()?;
        for &batch in &[16_usize, 64, 256, 1024] {
            let in_dim = 32_usize;
            let out_dim = 1_usize;
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
            let dw_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim * in_dim)?;
            let num_splits = 8_usize;
            let config = LaunchConfig {
                grid_dim: (num_splits as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l3, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }?;
            stream.synchronize()?;
            assert_close_rel(
                &format!("dense_mm_bwd_weight_bucket_tiled_l3 b={batch}"),
                &dw_dev.to_host_vec(&stream)?,
                &dw_cpu,
                TOL,
            );
        }
        Ok(())
    }

    #[test]
    fn dense_mm_bwd_weight_bucket_tiled_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        // tiled (L1): in_dim % 16 == 0、out_dim == 16、batch % 16 == 0、num_buckets == 9 を要求
        for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (32, 96)] {
            let out_dim = 16_usize;
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
            let blocks = in_dim / 16;
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l1, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice_mut(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }?;
            stream.synchronize()?;
            assert_close(
                &format!("dense_mm_bwd_weight_bucket_tiled_l1 b={batch} in={in_dim}"),
                &dw_dev.to_host_vec(&stream)?,
                &dw_cpu,
                TOL,
            );
        }
        Ok(())
    }

    /// 16-aligned bucket sort + permute_rows (dl1_total) + sorted bwd_weight が
    /// `dense_mm_bwd_weight_bucket_cpu` と reduction tolerance 内で一致することを確認。
    /// per-cell の partial sum 順序が sort 済 batch + split-K 順になるため fp32 associativity
    /// で bit-exact ではないが、`assert_close_rel` で相対誤差判定する。
    #[test]
    fn bucket_sort_bwd_weight_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (48, 96), (64, 32)] {
            let out_dim = 16_usize;
            let nb = NUM_BUCKETS;
            let padded = padded_sort_batch(batch);
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

            let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let perm_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
            let bidx_sorted_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
            let mut x_sorted_dev = DeviceBuffer::<f32>::zeroed(&stream, padded * in_dim)?;
            let mut dy_sorted_dev = DeviceBuffer::<f32>::zeroed(&stream, padded * out_dim)?;
            let dw_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim * in_dim)?;

            memset_minus_one_i32(&stream, &perm_dev)?;
            memset_minus_one_i32(&stream, &bidx_sorted_dev)?;

            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }?;
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * in_dim),
                args: [slice(x_dev), slice(perm_dev), slice_mut(x_sorted_dev),
                       padded as u32, in_dim as u32]
            }?;
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(dy_dev), slice(perm_dev), slice_mut(dy_sorted_dev),
                       padded as u32, out_dim as u32]
            }?;
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l1_sorted, stream: stream, module: module,
                config: LaunchConfig {
                    grid_dim: ((in_dim / 16) as u32, 8, nb as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [slice(x_sorted_dev), slice(dy_sorted_dev), slice(offsets_dev),
                       slice(dw_dev), padded as u32, in_dim as u32, out_dim as u32, nb as u32]
            }?;
            stream.synchronize()?;
            assert_close_rel(
                &format!("dense_mm_bwd_weight_bucket_tiled_l1_sorted b={batch} in={in_dim}"),
                &dw_dev.to_host_vec(&stream)?,
                &dw_cpu,
                TOL,
            );
        }
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

    /// 16-aligned bucket sort + permute_rows (dy) + sorted block-shared bias_grad が
    /// `bias_grad_bucket_cpu` と reduction tolerance 内で一致することを確認。
    /// per-block shared atomic + per-block global atomic で加算順 ≠ baseline、
    /// `assert_close_rel` で判定。
    #[test]
    fn bias_grad_bucket_shared_sorted_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        for &(batch, out_dim) in &[
            (16_usize, 16_usize), // L1 bias 形状
            (32, 16),
            (48, 16),
            (64, 16),
            (16, 32), // L2 bias 形状
            (32, 32),
            (48, 32),
            (64, 32),
        ] {
            let nb = NUM_BUCKETS;
            let padded = padded_sort_batch(batch);
            let dy: Vec<f32> = (0..batch * out_dim)
                .map(|i| i as f32 * 0.017 - 0.9)
                .collect();
            let bucket_idx = bucket_idx_with_padding(batch, nb);
            let mut gb_cpu = vec![0.0_f32; nb * out_dim];
            bias_grad_bucket_cpu(&dy, &bucket_idx, &mut gb_cpu, batch, out_dim, nb);

            let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
            let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;

            let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
            let perm_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
            let bidx_sorted_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
            let mut dy_sorted_dev = DeviceBuffer::<f32>::zeroed(&stream, padded * out_dim)?;
            let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim)?;

            memset_minus_one_i32(&stream, &perm_dev)?;
            memset_minus_one_i32(&stream, &bidx_sorted_dev)?;

            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }?;
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }?;
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(dy_dev), slice(perm_dev), slice_mut(dy_sorted_dev),
                       padded as u32, out_dim as u32]
            }?;
            cuda_launch! {
                kernel: bias_grad_bucket_shared_sorted, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(dy_sorted_dev), slice(bidx_sorted_dev), slice(gb_dev),
                       padded as u32, out_dim as u32, nb as u32]
            }?;
            stream.synchronize()?;
            assert_close_rel(
                &format!("bias_grad_bucket_shared_sorted b={batch}"),
                &gb_dev.to_host_vec(&stream)?,
                &gb_cpu,
                TOL,
            );
        }
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
            kernel: ft_post_perspective_grad, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(dc_dev), slice(stm_ft_dev), slice(bias_dev), slice_mut(dft_stm_dev),
                   slice(grad_bias_dev), batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale]
        }?;
        cuda_launch! {
            kernel: ft_post_perspective_grad, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
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

    /// `ft_post_perspective_grad_fused` (d_combined = a+b の融合) が CPU reference
    /// (元 kernel と同じ math) と reduction tolerance 内一致することを確認。
    /// fused 版は `d_combined_a[idx] + d_combined_b[idx]` を in-register sum、それ以降
    /// は元 kernel と同じ。
    #[test]
    fn ft_post_perspective_grad_fused_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let ft_dim = FT_OUT;
        let half = ft_dim / 2;
        let bias: Vec<f32> = (0..ft_dim)
            .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
            .collect();
        let scale = FT_POST_SCALE;
        let d_a: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.2_f32 + 0.011_f32 * i as f32)
            .collect();
        let d_b: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -0.8_f32 + 0.007_f32 * i as f32)
            .collect();
        let stm_ft: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
            .collect();
        let nstm_ft: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.5_f32 + 3.0_f32 * ((i * 5) % 11) as f32 / 10.0)
            .collect();

        // CPU reference: a+b を summed buffer に組み立てて元 grad_cpu を回す。
        let d_combined: Vec<f32> = d_a.iter().zip(d_b.iter()).map(|(&x, &y)| x + y).collect();
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

        let da_dev = DeviceBuffer::from_host(&stream, &d_a)?;
        let db_dev = DeviceBuffer::from_host(&stream, &d_b)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let stm_ft_dev = DeviceBuffer::from_host(&stream, &stm_ft)?;
        let nstm_ft_dev = DeviceBuffer::from_host(&stream, &nstm_ft)?;
        let grad_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_dim)?;
        let mut dft_stm_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
        let mut dft_nstm_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(stm_ft_dev), slice(bias_dev),
                   slice_mut(dft_stm_dev), slice(grad_bias_dev),
                   batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale]
        }?;
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(nstm_ft_dev), slice(bias_dev),
                   slice_mut(dft_nstm_dev), slice(grad_bias_dev),
                   batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale]
        }?;
        stream.synchronize()?;
        // dft_*: 和の順序は CPU と同じ (per-thread, no reduction)、tolerance は relative。
        assert_close_rel(
            "ft_grad_fused dft_stm",
            &dft_stm_dev.to_host_vec(&stream)?,
            &dft_stm_cpu,
            TOL,
        );
        assert_close_rel(
            "ft_grad_fused dft_nstm",
            &dft_nstm_dev.to_host_vec(&stream)?,
            &dft_nstm_cpu,
            TOL,
        );
        assert_close_rel(
            "ft_grad_fused grad_bias",
            &grad_bias_dev.to_host_vec(&stream)?,
            &grad_bias_cpu,
            TOL,
        );
        Ok(())
    }

    // -- ft_post FP16 版 (--ft-fp16-out で FT activation を半精度化する経路) -------
    //
    // f16 入力は事前に round-to-nearest 量子化し、CPU reference にも同じ f16→f32 値を
    // 渡す。これで「kernel の f16 read / indexing が正しいか」を、量子化誤差と分離して
    // 検証できる (f16→f32 拡張は無損失なので GPU と CPU の演算入力は bit 一致)。

    /// `f32` 列を round-to-nearest で `f16` 量子化し、`(f16 列, f16→f32 に戻した列)` を返す。
    fn quantize_f16(v: &[f32]) -> (Vec<f16>, Vec<f32>) {
        let h: Vec<f16> = v.iter().map(|&x| x as f16).collect();
        let back: Vec<f32> = h.iter().map(|&x| x as f32).collect();
        (h, back)
    }

    #[test]
    fn ft_post_perspective_fwd_fp16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let ft_dim = FT_OUT;
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
        let (stm_h, stm_q) = quantize_f16(&stm);
        let (nstm_h, nstm_q) = quantize_f16(&nstm);

        // CPU reference は GPU が読むのと同じ f16→f32 値で計算する。
        let mut combined_cpu = vec![0.0_f32; batch * ft_dim];
        ft_post_perspective_fwd_cpu(
            &stm_q,
            &nstm_q,
            &bias,
            &mut combined_cpu,
            batch,
            ft_dim,
            scale,
        );

        let stm_dev = DeviceBuffer::from_host(&stream, &stm_h)?;
        let nstm_dev = DeviceBuffer::from_host(&stream, &nstm_h)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let mut combined_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
        cuda_launch! {
            kernel: ft_post_perspective_fwd_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim),
            args: [slice(stm_dev), slice(nstm_dev), slice(bias_dev), slice_mut(combined_dev),
                   batch as u32, ft_dim as u32, scale]
        }?;
        stream.synchronize()?;
        // 入力 f16 値・f32 演算とも GPU/CPU 一致のため tight tolerance。
        assert_close(
            "ft_post_perspective_fwd_fp16",
            &combined_dev.to_host_vec(&stream)?,
            &combined_cpu,
            TOL,
        );
        Ok(())
    }

    #[test]
    fn ft_post_perspective_grad_fused_fp16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let (_ctx, module, stream) = open_module()?;
        let batch = 3_usize;
        let ft_dim = FT_OUT;
        let half = ft_dim / 2;
        let bias: Vec<f32> = (0..ft_dim)
            .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
            .collect();
        let scale = FT_POST_SCALE;
        let d_a: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.2_f32 + 0.011_f32 * i as f32)
            .collect();
        let d_b: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -0.8_f32 + 0.007_f32 * i as f32)
            .collect();
        let stm_ft: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
            .collect();
        let nstm_ft: Vec<f32> = (0..batch * ft_dim)
            .map(|i| -1.5_f32 + 3.0_f32 * ((i * 5) % 11) as f32 / 10.0)
            .collect();
        let (stm_ft_h, stm_ft_q) = quantize_f16(&stm_ft);
        let (nstm_ft_h, nstm_ft_q) = quantize_f16(&nstm_ft);

        // CPU reference は f16→f32 に戻した ft_out で計算する。
        let d_combined: Vec<f32> = d_a.iter().zip(d_b.iter()).map(|(&x, &y)| x + y).collect();
        let mut grad_bias_cpu = vec![0.0_f32; ft_dim];
        let mut dft_stm_cpu = vec![0.0_f32; batch * ft_dim];
        let mut dft_nstm_cpu = vec![0.0_f32; batch * ft_dim];
        ft_post_perspective_grad_cpu(
            &d_combined,
            &stm_ft_q,
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
            &nstm_ft_q,
            &bias,
            &mut dft_nstm_cpu,
            &mut grad_bias_cpu,
            batch,
            ft_dim,
            half,
            ft_dim,
            scale,
        );

        let da_dev = DeviceBuffer::from_host(&stream, &d_a)?;
        let db_dev = DeviceBuffer::from_host(&stream, &d_b)?;
        let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
        let stm_ft_dev = DeviceBuffer::from_host(&stream, &stm_ft_h)?;
        let nstm_ft_dev = DeviceBuffer::from_host(&stream, &nstm_ft_h)?;
        let grad_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_dim)?;
        let mut dft_stm_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * ft_dim)?;
        let mut dft_nstm_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * ft_dim)?;
        // test 入力 dft は O(数十) なので、production の dft_scale (FT_DFT_FP16_BASE_SCALE
        // × batch) では overflow する。loss scaling round-trip 検証用の小さい値を使う。
        let dft_scale = 64.0_f32;
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(stm_ft_dev), slice(bias_dev),
                   slice_mut(dft_stm_dev), slice(grad_bias_dev),
                   batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale, dft_scale]
        }?;
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(nstm_ft_dev), slice(bias_dev),
                   slice_mut(dft_nstm_dev), slice(grad_bias_dev),
                   batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale, dft_scale]
        }?;
        stream.synchronize()?;
        // dft 出力は f16 かつ dft_scale 倍されているので、読み戻して逆数を掛ける。GPU と
        // CPU は同じ f32 演算結果を持つが、GPU 側のみ最後に f16 量子化されるため、f16
        // round-off (相対 ~5e-4) を許容する relative tolerance。
        let inv = 1.0_f32 / dft_scale;
        let dft_stm_gpu: Vec<f32> = dft_stm_dev
            .to_host_vec(&stream)?
            .iter()
            .map(|&x| x as f32 * inv)
            .collect();
        let dft_nstm_gpu: Vec<f32> = dft_nstm_dev
            .to_host_vec(&stream)?
            .iter()
            .map(|&x| x as f32 * inv)
            .collect();
        assert_close_rel(
            "ft_grad_fused_fp16 dft_stm",
            &dft_stm_gpu,
            &dft_stm_cpu,
            2e-3,
        );
        assert_close_rel(
            "ft_grad_fused_fp16 dft_nstm",
            &dft_nstm_gpu,
            &dft_nstm_cpu,
            2e-3,
        );
        // grad_bias は f32 accumulate (FP32 path と同じ)、atomic 順序由来の drift のみ。
        assert_close_rel(
            "ft_grad_fused_fp16 grad_bias",
            &grad_bias_dev.to_host_vec(&stream)?,
            &grad_bias_cpu,
            TOL,
        );
        Ok(())
    }
}
