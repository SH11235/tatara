//! 共通 / 損失 / optimizer kernel。

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicU32};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

/// SCReLU activation gradient (fused)。
///
/// Simple アーキの `--activation screlu` backward で使う。LayerStack path は
/// CReLU + pairwise_mul を使うため未使用。
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
    per_pos_norm: f32, // scalar (= 1/n_pos)
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
/// - prediction: `scorenet = out * nnue2score`、`q = sigmoid((scorenet - in_offset)/in_scaling)`、
///   `qm = sigmoid((-scorenet - in_offset)/in_scaling)`、`qf = 0.5*(1 + q - qm)`。
///   `in_offset` (= `--wrm-in-offset`、既定 270、prediction sigmoid の中心) /
///   `in_scaling` (= `--wrm-in-scaling`、既定 340) は prediction 側のみ、`nnue2score`
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
    in_offset: f32,
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
    let q = 1.0_f32 / (1.0_f32 + (-((scorenet - in_offset) / in_scaling)).exp());
    let qm = 1.0_f32 / (1.0_f32 + (-((-scorenet - in_offset) / in_scaling)).exp());
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
/// `sparse_ft_forward` は DRAM 帯域律速で、その traffic の大半は active feature 行の
/// weight gather read。weight を半精度にすると read byte 数が半減し、L2 にも 2 倍の
/// 行が載るため DRAM 律速が緩む。
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
/// perspective_grad_fused_fp16` が dft を `f16` で書くのに合わせ、その read 側も
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
