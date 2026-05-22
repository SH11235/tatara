#![feature(f16)]
//! `bins/nnue_train` binary entry point — LayerStack NNUE trainer。
//!
//! 本 file は `#[kernel]` 群と bin entry point (`fn main`) を持つ。cuda-oxide の
//! bin-entry reachability 制約により全 kernel を本 file に inline する必要がある
//! (別 crate に置くと `compile_ll_to_ptx_via_llc` の symbol resolution から
//! 外れる)。host 側コード (kernel loader / checkpoint format / trainer / CLI /
//! smoke test) と GPU↔CPU 同等性テストは sibling module 群に分割している。
//!
//! ## LayerStack アーキテクチャ (FT → L1 16 → L2 32 + progress8kpabs 9 buckets)
//!
//! bullet `examples/shogi_layerstack.rs:2206-2289` の reference 実装を Rust +
//! cuda-oxide で再現。PSQT 無し、hand_count_dense 無し。FT 入力次元 `ft_in` は
//! feature set 依存、FT 出力次元 `ft_out` は `--ft-out` 依存の runtime 値。
//!
//! - **L0 (FT)**: sparse_ft_forward — weight (ft_in × ft_out), bias (ft_out, 共有)
//! - **per-perspective post**: bias add → CReLU → pairwise_mul (ft_out→ft_out/2) → ×127/128
//! - **combined**: stm.concat(nstm) → ft_out
//! - **L1 (per-bucket)**: weight (9×16, ft_out) + bias (9×16) → select(bucket) → 16
//! - **L1f (shared)**: weight (ft_out, 16) + bias (16) → 16
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

use clap::Parser;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

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
/// の異常入力は silent skip。caller は `rows % 4 == 0` を保証する (`rows` は FT 出力
/// 次元で `--ft-out` 検証により 128 の倍数)、grid は `cfg_1d(batch * rows / 4)`。
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
    // 境界に整列する (idx*rows は 4 の倍数 [rows は 128 の倍数]、ri_base は 4 の倍数)。
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
/// `out` (`ft_*_out`、b × ft_out) を `f16` にすると書き出し DRAM traffic が半減し、
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
    // [rows は 128 の倍数]、ri_base は 4 の倍数)。
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
/// `grad_out` は b × ft_out で、本 kernel は 1 feature の出現位置すべてに対して全 ri
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
    ft_dim: u32, // per-perspective の FT 出力次元 (runtime、--ft-out)
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
    ft_dim: u32, // per-perspective の FT 出力次元 (runtime、--ft-out)
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
/// `d_combined_stride` は両 source の row-stride (= FT 出力次元 ft_out)、
/// `d_combined_offset` は perspective 別 offset (stm: 0、nstm: ft_dim/2)、両 source
/// は同 stride・同 layout を caller が保証 (両者とも `b × ft_out` workspace)。
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
/// traffic が半減する (dft は b × ft_out で step 中で最も read 量が多い buffer)。
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

/// 非 fused FP16 版 [`ft_post_perspective_grad`]: forward activation `ft_out` を `f16`
/// で読み、`grad_ft_out` (dft) を loss scaling 付き `f16` で書く。`d_combined` は
/// 単一 source (`ft_post_perspective_grad` と同じく、`d_combined_offset` で perspective
/// の半分を切り出す)。`d_combined` / `bias` / `grad_bias` は `f32` のまま。
///
/// math は [`ft_post_perspective_grad_fused_fp16`] と同等で、`dy` の読み出し元のみ
/// 2 source の in-register sum → 単一 buffer read に置き換わる。`grad_ft_out` へ書く
/// 値は `dft_scale` ([`FT_DFT_FP16_BASE_SCALE`] × batch) を掛けて f16 normal range に
/// 持ち上げ ±65504 clamp してから cast し、後続 [`simple_sparse_ft_backward_fp16`] が
/// `dft_inv_scale` で打ち消す。`grad_bias` への atomic accumulate は scale しない
/// `f32` の `grad_a` / `grad_b` をそのまま使う (FP32 path と同じ精度)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn ft_post_perspective_grad_fp16(
    d_combined: &[f32],
    ft_out: &[f16],
    bias: &[f32],
    mut grad_ft_out: DisjointSlice<f16>,
    grad_bias: &[f32],
    batch: u32,
    ft_dim: u32,
    d_combined_offset: u32,
    d_combined_stride: u32,
    scale: f32,
    dft_scale: f32,
) {
    let tid = thread::index_1d();
    let half = (ft_dim as usize) / 2;
    let total_pairs = (batch as usize) * half;
    if tid.get() >= total_pairs {
        return;
    }
    let bi = tid.get() / half;
    let pair_idx = tid.get() % half;

    let dy =
        d_combined[bi * (d_combined_stride as usize) + (d_combined_offset as usize) + pair_idx];

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
    // `ft_post_perspective_grad_fused_fp16` と同一。dft_scale を掛けてから f16 域へ
    // clamp する (天井超過を ±inf にすると gather 経由で weight を NaN 化させるため)。
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
    // SAFETY: grad_ft_out.len() == batch * ft_dim (caller 契約)、`ft_dim = 2 * half` の
    // 偶数性で pair_idx ∈ [0, half) → {pair_idx, half + pair_idx} ⊂ [0, ft_dim)、tid 範囲
    // チェックで bi < batch。同一 (bi, ii) cell を書く thread は他に無い (pair_idx 単射)。
    let out_ptr = grad_ft_out.as_mut_ptr();
    unsafe {
        out_ptr.add(ft_base + pair_idx).write(da_c as f16);
        out_ptr.add(ft_base + half + pair_idx).write(db_c as f16);
    }

    // grad_bias は f32 accumulate を維持 (scale 無しの grad_a / grad_b を atomic add)。
    // SAFETY: grad_bias.len() == ft_dim、pair_idx < half、half + pair_idx < ft_dim。
    // `f32` (align 4) と `DeviceAtomicF32` は同 layout、non-atomic 書き込み path は無し。
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

/// Tiled shared-memory variant of [`dense_mm_bwd_input`]. L1f 用 (`in_dim=ft_out`,
/// `out_dim=16` 固定)、`batch % 16 == 0`、`in_dim % 16 == 0` を host が保証。
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

/// Tiled shared-memory variant of [`dense_mm_bwd_weight`]. L1f 用 (`in_dim=ft_out`,
/// `out_dim=16` 固定) を想定した固定タイル形状 (TILE_K=16, TILE_IN=16,
/// TILE_OUT=16, block=256 threads)。`in_dim % 16 == 0 && out_dim == 16 && batch % 16 == 0`
/// が host 契約。非該当形状では結果未定義 (host 側で sizes チェックの上で本 kernel を選ぶ)。
///
/// 1 block = 1 (TILE_IN × TILE_OUT) W tile。block 内 256 threads が batch を TILE_K=16
/// chunk で cooperatively load し、shared memory 上で TILE_K 回 FMA。current "1 thread =
/// 1 cell、scan batch" 比 ~33x 少ない unique memory read (x 16x redundant → 1x、dy ft_out x → 1x)。
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

/// Tiled per-bucket weight backward (L1 用: `in_dim=ft_out`、`out_dim=16` /
/// `num_buckets=9` は固定、`batch % 16 == 0`)。
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

/// Tiled non-bucket forward dense matmul (L1f 用: `in_dim=ft_out`、`out_dim=16` 固定)。
/// 元 `dense_mm_fwd` は coalesced だが 1 thread = 1 (b, oi) で per-thread ft_out K iter で
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

/// Tiled per-bucket forward dense matmul (L1 用: `in_dim=ft_out`、`out_dim=16` /
/// `num_buckets=9` は固定)。
///
/// 元 `dense_mm_fwd_bucket` は `w[buc][oi][ii]` layout のため、warp 内 16-thread sub-group が
/// oi 軸を varying させると stride=in_dim=ft_out で uncoalesced。本 kernel は 1 block = 1 batch
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

/// Simple FP16 FT activation forward (SCReLU): f16 FT 出力 + f32 bias → f32 acted。
///
/// [`simple_bias_act_fwd_fp16_in_crelu`] の SCReLU 版。活性化のみ `clamp(x, 0, 1)²`
/// に置き換わり、f16 read / bias 加算 / 出力 layout は同一。
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn simple_bias_act_fwd_fp16_in_screlu(
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
    let a = if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    };
    if let Some(o) = ft_acted.get_mut(tid) {
        *o = a * a;
    }
}

/// Simple FP16 FT activation backward (SCReLU) + loss scaling + ±65504 clamp + f16 cast。
///
/// [`simple_act_grad_to_fp16_crelu_with_scale`] の SCReLU 版。CReLU の指示関数
/// (`0 < x < 1` で 1) の代わりに SCReLU の局所微分 `d/dx clamp(x,0,1)² = 2·clamp(x,0,1)`
/// (`0 < clamp < 1` 範囲、外は 0) を掛ける。loss scaling / ±65504 clamp / f16 cast は同一。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)]
#[kernel]
pub fn simple_act_grad_to_fp16_screlu_with_scale(
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
    let a = if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    };
    let dydx = if a > 0.0_f32 && a < 1.0_f32 {
        2.0_f32 * a
    } else {
        0.0_f32
    };
    let g = dft_acted[tid.get()] * dydx;
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
// module 宣言
//
// host 側コード (kernel loader / checkpoint format / trainer / CLI / smoke) と
// GPU↔CPU 同等性テストは sibling module に分割。`#[kernel]` 群は cuda-oxide の
// bin-entry reachability 制約で本 crate-root file に残す。
// ===========================================================================

mod arch;
mod ckpt;
mod cli;
mod kernel_module;
mod smoke;
mod trainer_common;
mod trainer_layerstack;
mod trainer_simple;
mod training;

#[cfg(test)]
mod tests;

use cli::Cli;
use smoke::smoke_test;
use training::run_training;

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
