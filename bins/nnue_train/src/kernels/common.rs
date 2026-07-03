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
    let j = i.get();
    if let Some(out) = dl_dx.get_mut(i) {
        *out = dl_dy[j] * dydx;
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
/// 教師 score (centipawn) と net 出力の双方を win-rate に変換し、その誤差を loss と
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
/// - `err = qf - target`。
///
/// ## 2 経路: default (二乗誤差) と extended
///
/// `extended == 0` のとき loss は plain な二乗誤差 `err^2`:
/// - `loss_acc += err^2` (norm 無し、caller が position 数で割る)。
/// - chain rule: `dqf/dout = 0.5 * (nnue2score/in_scaling) * (q(1-q) + qm(1-qm))`、
///   `dL/dout = 2*err * dqf/dout` → `2` と `0.5` が打ち消し `g = err *
///   (nnue2score/in_scaling) * (q(1-q) + qm(1-qm)) * per_pos_norm`。
///
/// `extended == 1` のとき nnue-pytorch `calculate_sf_loss`
/// (`model/lightning_module.py`) の一般化 loss を適用する:
/// - `pf = target_wrm` (WDL blend 前の score 由来 win-rate)。
/// - per-position weight `w = 1 + (2^w1 - 1) * ((pf-0.5)^2 * pf*(1-pf))^w2`。
/// - asymmetry `a = qf > target ? 1 + qp_asymmetry : 1`。
/// - per-position loss `L_i = a * |err|^pow_exp`、全体 loss は `Σ(L_i w_i) / Σ w_j`。
///   `Σ w_j` は呼出前に `wrm_weight_sum` kernel が `sum_w_acc` に reduce 済で、本
///   kernel は全 thread でそれを読む (同 stream 上の先行 launch なので可視)。
/// - grad: `dL_i/dqf = a * pow_exp * |err|^(pow_exp-1) * sign(err)` (weight `w_i` と
///   indicator `qf>target` は qf に対して定数扱い、nnue-pytorch の autograd と同じ)、
///   `dL/dout = (w_i / Σw) * (dL_i/dqf) * dqf/dout`。
/// - loss_acc には `L_i * w_i * n / Σw` を加える。caller が batch を跨いで和を取り
///   `Σ position 数` で割る集計のため、`n/Σw` を掛けて per-batch 寄与を
///   `n * (Σ(L w)/Σw)` (= position 数 × batch 平均 loss) に揃える。default の
///   `Σ err^2` (= n × 二乗誤差平均) と同じ単位。
///
/// default 経路を別 branch で残すのは bit-identical 要件のため: `|err|^2` は
/// `__nv_powf(|err|, 2)` 経由だと `err*err` と bit 一致せず、`1/Σw` も `per_pos_norm`
/// (= host が f32 で計算した `1/n`) と最終 bit が一致しない。extended の既定値
/// (`pow_exp=2 / qp_asymmetry=0 / w1=0`) では caller が `extended=0` を渡す。
///
/// 1 thread = 1 position。`dl_dout` (= 訓練に使う勾配) は per-position 排他更新 (atomics 不要)。
/// `loss_acc` (ログ表示用の loss 値) は block 内 partial を shared-mem tree reduction で集約し
/// block あたり 1 回だけ `DeviceAtomicF64::fetch_add` する (single-cell への atomic 競合回避)。
/// f64 reduction は加算順に依存し loss 値の最下位 bit (~1e-15 rel) が動くが、grad は `loss_acc` を
/// 読まない (per-position に別書き) ので影響を受けない。`f32::exp` / `f32::powf` / `f32::abs` は
/// libdevice (`__nv_expf` / `__nv_powf` / `__nv_fabsf`) に lowering OK。block_dim は BLOCK_DIM (256)
/// の 2 冪前提 (tree reduction が完全に畳める)。
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
    pow_exp: f32,
    qp_asymmetry: f32,
    weight_boost_w1: f32,
    weight_boost_w2: f32,
    sum_w_acc: &[f64], // extended のときのみ参照 (Σw、wrm_weight_sum が事前 reduce)
    extended: u32,     // 0 = 二乗誤差 (bit-identical)、1 = nnue-pytorch 一般化 loss
    n: u32,
) {
    use core::ptr::addr_of_mut;
    // block 内 partial loss を shared-mem tree reduction で集約し、`loss_acc` への atomic は
    // block あたり 1 回だけにする (block_dim == BLOCK_DIM = 256)。out-of-range thread は寄与 0 で
    // reduction に参加する (sync_threads を全 thread が一様に通すため early return しない)。
    static mut PARTIAL: SharedArray<f64, 256> = SharedArray::UNINIT;
    let i = thread::index_1d();
    let tid = thread::threadIdx_x() as usize;
    let valid = i.get() < n as usize;

    // 本 thread の loss 寄与 (out-of-range は 0)。grad `dl_dout` は valid thread のみ排他更新で
    // atomics 不要、loss reduction とは独立 (loss_acc を読まない)。
    let mut contrib = 0.0_f64;
    if valid {
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

        if extended == 0 {
            // 二乗誤差 grad は項の順序・grouping をそのまま評価する (f32 乗算は非結合則で、
            // くくり出すと最終 bit が変わり量子化出力の bit 再現に効くため)。
            let norm = per_pos_norm;
            if let Some(g) = dl_dout.get_mut(i) {
                *g = err
                    * (nnue2score / in_scaling)
                    * (q * (1.0_f32 - q) + qm * (1.0_f32 - qm))
                    * norm;
            }
            contrib = (err as f64) * (err as f64);
        } else {
            // --- extended: nnue-pytorch 一般化 loss (weight boost / pow_exp / asymmetry) ---
            let pf = target_wrm;
            let wb_base = (pf - 0.5_f32) * (pf - 0.5_f32) * pf * (1.0_f32 - pf);
            let weight =
                1.0_f32 + (2.0_f32.powf(weight_boost_w1) - 1.0_f32) * wb_base.powf(weight_boost_w2);
            let asym = if qf > target {
                1.0_f32 + qp_asymmetry
            } else {
                1.0_f32
            };
            let abs_err = err.abs();
            // sign(err) * |err|^(pow_exp-1): err=0 では pow_exp>1 で 0、符号は err の符号。
            // pow_exp=1 (L1) は原点で subgradient +1 を返す (実害は測度 0 の点のみ)。
            let pow_abs = abs_err.powf(pow_exp - 1.0_f32);
            let signed_pow = if err < 0.0_f32 { -pow_abs } else { pow_abs };

            // Σw を読む (先行 launch の wrm_weight_sum が書込済、本 kernel では read only)。
            // host validation が w1,w2 >= 0 を保証するので weight >= 1、Σw >= n > 0 (n は本
            // kernel が launch される batch サイズで >= 1) ゆえ除算は安全。
            let inv_sum_w = (1.0_f64 / sum_w_acc[0]) as f32;
            // dqf/dout = 0.5 * (nnue2score/in_scaling) * (q(1-q)+qm(1-qm))。
            let dqf_dout =
                0.5_f32 * (nnue2score / in_scaling) * (q * (1.0_f32 - q) + qm * (1.0_f32 - qm));

            if let Some(g) = dl_dout.get_mut(i) {
                *g = (weight * inv_sum_w) * (asym * pow_exp * signed_pow) * dqf_dout;
            }

            contrib = (asym * abs_err.powf(pow_exp) * weight * (n as f32) * inv_sum_w) as f64;
        }
    }

    // loss_acc は単一 cell ゆえ全 thread が直接 atomic add すると競合する。block 内で contrib を
    // tree reduction し、block 0 番 thread が block あたり 1 atomic だけ打って競合を block 数に抑える。
    let partial_ptr: *mut f64 = addr_of_mut!(PARTIAL) as *mut f64;
    unsafe {
        partial_ptr.add(tid).write(contrib);
    }
    thread::sync_threads();
    let mut stride = (thread::blockDim_x() as usize) / 2;
    while stride >= 1 {
        if tid < stride {
            let v = unsafe { partial_ptr.add(tid).read() + partial_ptr.add(tid + stride).read() };
            unsafe {
                partial_ptr.add(tid).write(v);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    if tid == 0 {
        // SAFETY: `loss_acc.len() == 1`、host 側で f64 単一 cell 確保済 (`loss_wdl` と同型)。
        let loss_atom = unsafe { &*(loss_acc.as_ptr() as *const DeviceAtomicF64) };
        let block_sum = unsafe { partial_ptr.add(0).read() };
        loss_atom.fetch_add(block_sum, AtomicOrdering::Relaxed);
    }
}

/// extended WRM loss の per-position weight `w = 1 + (2^w1 - 1) * ((pf-0.5)^2 *
/// pf*(1-pf))^w2` を全 position で Σ し、`sum_w_acc` (f64 単一 cell) に atomic
/// accumulate する。`pf` は target side の WRM 変換 (`loss_wrm` の `target_wrm` と同式、
/// score / target_offset / target_scaling のみに依存し net 出力には依らない)。
///
/// `loss_wrm` の extended 経路は grad / loss を `1/Σw` で正規化する (nnue-pytorch
/// `calculate_sf_loss` の `loss = (loss*weights).sum() / weights.sum()`)。Σw は全
/// position の reduction なので grad を書く `loss_wrm` の前に本 kernel で先に確定させ、
/// 同 stream 上で順序保証する。host は呼出前に `sum_w_acc` を 0 reset する。
///
/// 1 thread = 1 position。`f32::powf` は libdevice (`__nv_powf`) に lowering OK。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn wrm_weight_sum(
    score: &[f32],
    sum_w_acc: &[f64],
    weight_boost_w1: f32,
    weight_boost_w2: f32,
    target_offset: f32,
    target_scaling: f32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    let s = score[i.get()];
    let sig_pt = 1.0_f32 / (1.0_f32 + (-((s - target_offset) / target_scaling)).exp());
    let sig_pmt = 1.0_f32 / (1.0_f32 + (-((-s - target_offset) / target_scaling)).exp());
    let pf = 0.5_f32 * (1.0_f32 + sig_pt - sig_pmt);
    let wb_base = (pf - 0.5_f32) * (pf - 0.5_f32) * pf * (1.0_f32 - pf);
    let weight =
        1.0_f32 + (2.0_f32.powf(weight_boost_w1) - 1.0_f32) * wb_base.powf(weight_boost_w2);

    // SAFETY: `sum_w_acc.len() == 1`、host 側で f64 単一 cell 確保済 (`loss_acc` と同型)。
    let sum_atom = unsafe { &*(sum_w_acc.as_ptr() as *const DeviceAtomicF64) };
    sum_atom.fetch_add(weight as f64, AtomicOrdering::Relaxed);
}

macro_rules! radam_update_moments {
    (f32, $m_ref:ident, $v_ref:ident, $g:ident, $beta1:ident, $beta2:ident) => {{
        let mi = $beta1 * *$m_ref + (1.0_f32 - $beta1) * $g;
        let vi = $beta2 * *$v_ref + (1.0_f32 - $beta2) * $g * $g;
        *$m_ref = mi;
        *$v_ref = vi;
        (mi, vi)
    }};
    (
        f16,
        $m_ref:ident,
        $v_ref:ident,
        $g:ident,
        $beta1:ident,
        $beta2:ident,
        $m_scale:ident,
        $v_scale:ident
    ) => {{
        // f16 格納値を真値へ割り戻す (scale は power-of-2 なので除算は無誤差)。
        let m_prev = (*$m_ref as f32) / $m_scale;
        let v_prev = (*$v_ref as f32) / $v_scale;
        let mi = $beta1 * m_prev + (1.0_f32 - $beta1) * $g;
        let vi = $beta2 * v_prev + (1.0_f32 - $beta2) * $g * $g;
        // 格納: scale 後 f16 有限域に clamp してから半精度化。
        let ms = mi * $m_scale;
        let ms_c = if ms > 65504.0_f32 {
            65504.0_f32
        } else if ms < -65504.0_f32 {
            -65504.0_f32
        } else {
            ms
        };
        *$m_ref = ms_c as f16;
        let vs = vi * $v_scale;
        let vs_c = if vs > 65504.0_f32 {
            65504.0_f32
        } else {
            vs
        };
        *$v_ref = vs_c as f16;
        // val は本 step の真値 mi / vi で計算する (f16 丸めは次 step の read で 1 回だけ入る)。
        (mi, vi)
    }};
}

macro_rules! radam_finish {
    (
        $grad_mode:ident,
        $mirror_mode:ident,
        $i:ident,
        $g_ref:ident,
        $p_clamped:ident
        $(, $mirror:ident)?
    ) => {
        radam_finish!(@grad $grad_mode, $g_ref);
        radam_finish!(@mirror $mirror_mode, $i, $p_clamped $(, $mirror)?);
    };
    (@grad reset, $g_ref:ident) => {
        *$g_ref = 0.0_f32;
    };
    (@grad keep_grad, $g_ref:ident) => {
        // ft_w_grad は毎 step backward (overwrite-gather、factorizer 有効時は仮想行を
        // `ft_reduce_virtual_grad`) が全 cell を書き直すため、ここで grad を 0 に戻さない
        // (戻しても次 read 前に上書きされ DRAM 書き込みを浪費するだけ)。
        // 通常の radam_step は atomic 累積 group で次 step 前の zero-out が必須。
    };
    (@mirror no_mirror, $i:ident, $p_clamped:ident) => {
    };
    (@mirror mirror, $i:ident, $p_clamped:ident, $mirror:ident) => {
        // SAFETY: `mirror` は他の buffer と同要素数で別 alloc (caller 保証)。
        // kernel 冒頭で `i < n` を確認し、各 thread は自分の `i` のみ書く。
        let mirror_ptr = $mirror.as_mut_ptr();
        unsafe {
            mirror_ptr.add($i).write($p_clamped as f16);
        }
    };
}

// RAdam kernel の各変種で decay、moment 更新、denom 分岐、clamp のコアを共有する。
macro_rules! radam_step_body {
    (
        $weights:ident,
        $m:ident,
        $v:ident,
        $grad:ident,
        $lr:ident,
        $step_size:ident,
        $denom:ident,
        $decay:ident,
        $beta1:ident,
        $beta2:ident,
        $eps:ident,
        $min_w:ident,
        $max_w:ident,
        $n:ident,
        $i:ident;
        $state:ident $(, $m_scale:ident, $v_scale:ident)?;
        $grad_mode:ident, $mirror_mode:ident $(, $mirror:ident)?
    ) => {
        if $i.get() >= $n as usize {
            return;
        }
        let i = $i.get();
        let (g_opt, m_opt, v_opt, w_opt) =
            if i < $grad.len() && i < $m.len() && i < $v.len() && i < $weights.len() {
                // SAFETY: 上の bounds check を通過し、4 buffer は互いに別 allocation。
                // 1D launch の各 thread は一意な i の cell だけを更新する。
                let (g, m, v, w) = unsafe {
                    (
                        $grad.get_unchecked_mut(i),
                        $m.get_unchecked_mut(i),
                        $v.get_unchecked_mut(i),
                        $weights.get_unchecked_mut(i),
                    )
                };
                (Some(g), Some(m), Some(v), Some(w))
            } else {
                (None, None, None, None)
            };
        if let (Some(g_ref), Some(m_ref), Some(v_ref), Some(w_ref)) =
            (g_opt, m_opt, v_opt, w_opt)
        {
            let g = *g_ref;
            let rate = $lr * $step_size;
            let mut p = *w_ref;
            p *= 1.0_f32 - $decay * rate;
            let (mi, vi) = radam_update_moments!(
                $state, m_ref, v_ref, g, $beta1, $beta2 $(, $m_scale, $v_scale)?
            );
            let mut val = mi;
            if $denom != 0 {
                val /= vi.sqrt() + $eps;
            }
            p -= rate * val;
            let p_clamped = if p < $min_w {
                $min_w
            } else if p > $max_w {
                $max_w
            } else {
                p
            };
            *w_ref = p_clamped;
            radam_finish!(
                $grad_mode,
                $mirror_mode,
                i,
                g_ref,
                p_clamped
                $(, $mirror)?
            );
        }
    };
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
    radam_step_body!(
        weights, m, v, grad, lr, step_size, denom, decay, beta1, beta2, eps, min_w, max_w, n, i;
        f32;
        reset, no_mirror
    );
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
    radam_step_body!(
        weights, m, v, grad, lr, step_size, denom, decay, beta1, beta2, eps, min_w, max_w, n, i;
        f32;
        keep_grad, mirror, mirror
    );
}

/// Norm loss (Georgiou et al. 2021) の per-weight-group sumsq 計算 (reduce pass)。
///
/// 2D grid: x 軸 = group (隣接 thread = 隣接 group)、y 軸 = pos チャンク。group `g` の
/// element offset は `g*group_pitch + pos*elem_stride` (`pos < group_len`)。各 thread は
/// 担当 pos チャンク (`blockIdx_y` 始点、`gridDim_y` stride) の Σ w² を計算し `norms[g]`
/// へ atomicAdd する。呼び出し側は launch 前に `norms` を 0 fill し、後段
/// [`norm_loss_finalize`] が sqrt して L2 norm にする。strided column (FT/L1f,
/// `group_pitch=1`) では同一 pos の隣接 g が連続アドレスなので x 軸 thread で coalesce する。
///
/// `(group_pitch, elem_stride, group_len, n_groups)` の 3 レイアウト統一表現は
/// [`norm_loss_finalize`] / [`norm_loss_apply`] と共通: contiguous row (dense weight
/// `[n_groups, group_len]`、`pitch=group_len, stride=1`)、strided column (FT/L1f weight
/// `[group_len, n_groups]`、`pitch=1, stride=n_groups`)、per-tensor scalar (bias、
/// `n_groups=1, pitch=0, stride=1`)。
#[kernel]
pub fn norm_loss_reduce(
    weight: &[f32],
    norms: &[f32],
    n_groups: u32,
    group_pitch: u32,
    elem_stride: u32,
    group_len: u32,
) {
    let g = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
        + thread::threadIdx_x() as usize;
    if g >= n_groups as usize {
        return;
    }
    let base = g * group_pitch as usize;
    let stride = elem_stride as usize;
    let len = group_len as usize;
    // SAFETY: caller (`GpuTrainer::step_impl`) が対象 tensor のレイアウト
    // (n_groups/group_pitch/elem_stride/group_len) を `weight.len()` に整合する値で
    // 渡す。3 レイアウトいずれも `base + pos*stride` は weight 範囲内の単射。
    let wptr = weight.as_ptr();
    let mut sumsq = 0.0_f32;
    let mut pos = thread::blockIdx_y() as usize;
    let ystep = thread::gridDim_y() as usize;
    while pos < len {
        let w = unsafe { wptr.add(base + pos * stride).read() };
        sumsq += w * w;
        pos += ystep;
    }
    if sumsq != 0.0_f32 {
        // SAFETY: `norms.len() >= n_groups` (norm_scratch は対象 tensor 群の最大 group 数で
        // 確保され、小さい group では余剰あり)、`g < n_groups`。`DeviceAtomicF32` は f32 と
        // 同 layout。本 kernel 起動中 `norms` を non-atomic で書く path は無く (0 fill は
        // 先行 launch で完了)、atomicAdd 同士のみ GPU が serialize する。
        let cell = unsafe { &*(norms.as_ptr().add(g) as *const DeviceAtomicF32) };
        cell.fetch_add(sumsq, AtomicOrdering::Relaxed);
    }
}

/// Norm loss reduce の最終 pass。[`norm_loss_reduce`] が atomicAdd で貯めた Σ w² を
/// sqrt して L2 norm に変換する。1 thread = 1 group。
#[kernel]
pub fn norm_loss_finalize(mut norms: DisjointSlice<f32>, n_groups: u32) {
    let g = thread::index_1d();
    if g.get() >= n_groups as usize {
        return;
    }
    if let Some(o) = norms.get_mut(g) {
        *o = (*o).sqrt();
    }
}

/// Norm loss 補正の適用 (apply pass)。1 thread = 1 weight element。
///
/// thread `t` を、stride==1 の連続軸が最内になるよう `(g, pos)` へ分解して coalesce する:
/// strided column (FT/L1f、`group_pitch==1`) は `g` 最内 (`g=t%n_groups, pos=t/n_groups`)
/// で連続 thread が連続 g (同 pos) を触り、contiguous row / scalar は `pos` 最内
/// (`g=t/group_len, pos=t%group_len`)。どちらも offset は `g*group_pitch + pos*elem_stride`
/// で、weight に `*= 1 - lr*2*factor*(1 - 1/(norms[g]+eps))` を掛ける。各要素が受ける補正は
/// 分解方法に依らず同一 (norms[g] は同じ) なので結果は不変、メモリアクセスのみ coalesced 化。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn norm_loss_apply(
    mut weight: DisjointSlice<f32>,
    norms: &[f32],
    factor: f32,
    lr: f32,
    eps: f32,
    n_groups: u32,
    group_pitch: u32,
    elem_stride: u32,
    group_len: u32,
) {
    let t = thread::index_1d().get();
    let ng = n_groups as usize;
    let len = group_len as usize;
    if t >= ng * len {
        return;
    }
    // 連続 thread が連続メモリを触るよう、stride==1 の軸を最内にする。strided column
    // (group_pitch==1, FT/L1f) は g を、それ以外 (contiguous row / scalar) は pos を最内に。
    let (g, pos) = if group_pitch == 1 {
        (t % ng, t / ng)
    } else {
        (t / len, t % len)
    };
    let off = g * group_pitch as usize + pos * elem_stride as usize;
    let correction = 2.0_f32 * factor * (1.0_f32 - 1.0_f32 / (norms[g] + eps));
    let mult = 1.0_f32 - lr * correction;
    // SAFETY: `off` は `t` の単射 (各 thread 一意 offset)、caller がレイアウト整合を
    // 保証 (`norm_loss_reduce` と同じ契約)。
    let wptr = weight.as_mut_ptr();
    unsafe {
        let cur = wptr.add(off).read();
        wptr.add(off).write(cur * mult);
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
    radam_step_body!(
        weights, m, v, grad, lr, step_size, denom, decay, beta1, beta2, eps, min_w, max_w, n, i;
        f16, m_scale, v_scale;
        keep_grad, no_mirror
    );
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
    radam_step_body!(
        weights, m, v, grad, lr, step_size, denom, decay, beta1, beta2, eps, min_w, max_w, n, i;
        f16, m_scale, v_scale;
        keep_grad, mirror, mirror
    );
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
    let w_opt = weights.get_mut(thread::index_1d());
    let s_opt = slow.get_mut(thread::index_1d());
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
    let w_opt = weights.get_mut(thread::index_1d());
    let s_opt = slow.get_mut(thread::index_1d());
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

macro_rules! sparse_ft_weight_read {
    (f32, $weight_ptr:ident, $index:expr) => {
        unsafe { $weight_ptr.add($index).read() }
    };
    (f16, $weight_ptr:ident, $index:expr) => {
        unsafe { $weight_ptr.add($index).read() } as f32
    };
}

macro_rules! sparse_ft_out_write {
    (f32, $out_ptr:ident, $out_base:ident, $s0:ident, $s1:ident, $s2:ident, $s3:ident) => {
        unsafe {
            $out_ptr.add($out_base).write($s0);
            $out_ptr.add($out_base + 1).write($s1);
            $out_ptr.add($out_base + 2).write($s2);
            $out_ptr.add($out_base + 3).write($s3);
        }
    };
    (f16, $out_ptr:ident, $out_base:ident, $s0:ident, $s1:ident, $s2:ident, $s3:ident) => {
        unsafe {
            $out_ptr.add($out_base).write($s0 as f16);
            $out_ptr.add($out_base + 1).write($s1 as f16);
            $out_ptr.add($out_base + 2).write($s2 as f16);
            $out_ptr.add($out_base + 3).write($s3 as f16);
        }
    };
}

// sparse FT forward の各変種で index 解決、4-row 累算、出力のコアを共有する。
macro_rules! sparse_ft_forward_body {
    (
        $weight:ident,
        $indices:ident,
        $out:ident,
        $batch:ident,
        $rows:ident,
        $cols:ident,
        $nnz:ident,
        $tid:ident;
        $weight_type:ident,
        $out_type:ident
    ) => {
        let rows_u = $rows as usize;
        let rows_q = rows_u / 4;
        let total = ($batch as usize) * rows_q;
        if $tid.get() >= total {
            return;
        }
        let bi = $tid.get() / rows_q;
        let ri_q = $tid.get() % rows_q;
        let ri_base = ri_q * 4;

        // caller は indices.len() == batch * nnz、weight.len() == cols * rows、
        // out.len() == batch * rows、rows % 4 == 0 を保証する。idx の範囲検査は
        // padding と異常入力を除外する。idx*rows は 4 の倍数 (rows は 128 の倍数)、
        // ri_base も 4 の倍数なので、4 連続 row は最低 8-byte 境界に整列する。
        let indices_ptr = $indices.as_ptr();
        let weight_ptr = $weight.as_ptr();
        let mut s0: f32 = 0.0;
        let mut s1: f32 = 0.0;
        let mut s2: f32 = 0.0;
        let mut s3: f32 = 0.0;
        let base = bi * ($nnz as usize);
        let mut ni: u32 = 0;
        while ni < $nnz {
            let idx = unsafe { indices_ptr.add(base + (ni as usize)).read() };
            if idx >= 0 && (idx as u32) < $cols {
                let off = (idx as usize) * rows_u + ri_base;
                // f32 path は LLVM/NVPTX が pointer の 4-byte alignment しか推論せず
                // vector load 化しない。align(16) の struct cast も SROA 後に alignment
                // を保持せず scalar load + local-mem spill になる。scalar load のままでも
                // warp の 32 thread × 4 row が同じ feature の 128 連続 row を読み、
                // coalescing を維持する。
                let w0 = sparse_ft_weight_read!($weight_type, weight_ptr, off);
                let w1 = sparse_ft_weight_read!($weight_type, weight_ptr, off + 1);
                let w2 = sparse_ft_weight_read!($weight_type, weight_ptr, off + 2);
                let w3 = sparse_ft_weight_read!($weight_type, weight_ptr, off + 3);
                s0 += w0;
                s1 += w1;
                s2 += w2;
                s3 += w3;
            }
            ni += 1;
        }
        let out_ptr = $out.as_mut_ptr();
        let out_base = bi * rows_u + ri_base;
        sparse_ft_out_write!($out_type, out_ptr, out_base, s0, s1, s2, s3);
    };
}

/// Sparse feature transform forward (HalfKA_hm 用)。
///
/// 1 thread = 4 連続 row (output cells)、column-major weight
/// (`weight[idx * rows + ri]`)、atomics 不要 (各 thread は別 4 output cell に書く)。
/// `-1` padding と `idx >= cols` の異常入力は silent skip。caller は `rows % 4 == 0`
/// を保証する (`rows` は FT 出力次元で `--ft-out` 検証により 128 の倍数)、grid は
/// `cfg_1d(batch * rows / 4)`。
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
    sparse_ft_forward_body!(weight, indices, out, batch, rows, cols, nnz, tid; f32, f32);
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
    sparse_ft_forward_body!(weight, indices, out, batch, rows, cols, nnz, tid; f16, f32);
}

/// [`sparse_ft_forward_fp16`] の出力も `f16` にした版 (`--ft-fp16-out`)。`weight` を
/// `f16` で読み、累算は `f32`、累算結果を round-to-nearest で `f16` に変換して `out`
/// へ書く。
///
/// `out` (`ft_*_out`、b × ft_out) を `f16` にすると書き出し DRAM traffic が半減し、
/// 後続の [`ft_post_perspective_fwd_fp16`] / [`ft_post_perspective_grad_fused_fp16`]
/// の read も半精度になる。`ft_*_out` は CReLU 前の FT accumulator で値域は
/// ~O(1〜数十)、f16 の有限域に収まる (loss scaling 不要、underflow する dft とは異なる)。
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
    sparse_ft_forward_body!(weight, indices, out, batch, rows, cols, nnz, tid; f16, f16);
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

/// FT factorizer の forward 用畳み込み: base king-bucket セルの各要素へ同じ piece
/// plane の仮想行を加算し、forward が読む畳み込み済み weight `comb` (export 形状
/// `ft_in × ft_out` = base + threat) を作る。`w` は train 形状
/// (`(ft_in + piece_inputs) × ft_out`、column-major で `w[feature * ft_out + ri]`)。
/// 線形性により base 実行は `Σ_active (w_real + w_virt) = Σ_active comb`。
/// `base_ft_in` が仮想行を持つ base セル数、`ft_in` (= base + threat) が仮想 P plane
/// の手前。threat real 行 (`[base_ft_in, ft_in)`) は仮想行を持たないので `comb = w`
/// で素通しする。threat 無効時は `base_ft_in == ft_in` で全セルが畳まれ threat
/// 連結前と bit-identical。1 thread = 1 出力要素。仮想要素 offset は恒等式
/// `(ft_in + p)·ft_out + ri == ft_in·ft_out + (i mod pi·ft_out)` で mod 1 回。
#[kernel]
pub fn ft_fold_virtual(
    w: &[f32],
    mut comb: DisjointSlice<f32>,
    base_ft_in: u32,
    ft_in: u32,
    ft_out: u32,
    piece_inputs: u32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    // caller が `n == ft_in * ft_out`、`w.len() == (ft_in + piece_inputs) * ft_out`、
    // `comb.len() == n`、`base_ft_in <= ft_in` を保証。
    let feature = i.get() / (ft_out as usize);
    let v = if feature < base_ft_in as usize {
        let virt_block = (piece_inputs as usize) * (ft_out as usize);
        let virt = (ft_in as usize) * (ft_out as usize) + i.get() % virt_block;
        w[i.get()] + w[virt]
    } else {
        w[i.get()] // threat real 行: 仮想行を持たない
    };
    let comb_ptr = comb.as_mut_ptr();
    unsafe {
        comb_ptr.add(i.get()).write(v);
    }
}

/// [`ft_fold_virtual`] の f16 出力版 (`--ft-fp16` 系列の forward 用 mirror を畳み
/// 込みと同時に生成する)。f32 で加算してから 1 回だけ f16 へ丸める。和の f16
/// clamp はしない: ここで丸める和 (実行 + 仮想行) は量子化 export
/// (`save_quantised`) の coalesce が i16 飽和検査に掛ける値と同一で、その飽和
/// 閾値は f16 有限上限 (65504) より 2 桁以上小さい — overflow に至る重み成長は
/// 量子化 `.bin` 保存のたびに先に検出される。
#[kernel]
pub fn ft_fold_virtual_f16(
    w: &[f32],
    mut comb: DisjointSlice<f16>,
    base_ft_in: u32,
    ft_in: u32,
    ft_out: u32,
    piece_inputs: u32,
    n: u32,
) {
    let i = thread::index_1d();
    if i.get() >= n as usize {
        return;
    }
    // caller が `n == ft_in * ft_out`、`w.len() == (ft_in + piece_inputs) * ft_out`、
    // `comb.len() == n`、`base_ft_in <= ft_in` を保証。threat 行 (`feature >=
    // base_ft_in`) は仮想行を持たないので素通し。
    let feature = i.get() / (ft_out as usize);
    let v = if feature < base_ft_in as usize {
        let virt_block = (piece_inputs as usize) * (ft_out as usize);
        let virt = (ft_in as usize) * (ft_out as usize) + i.get() % virt_block;
        w[i.get()] + w[virt]
    } else {
        w[i.get()]
    };
    let comb_ptr = comb.as_mut_ptr();
    unsafe {
        comb_ptr.add(i.get()).write(v as f16);
    }
}

/// FT factorizer の backward 縮約: 仮想行 p の勾配を同じ piece plane を持つ
/// **base** 実行の勾配和で埋める (`grad[(ft_in + p) * ft_out + ri] =
/// Σ_{kb < base_ft_in/pi} grad[(kb * piece_inputs + p) * ft_out + ri]`)。各仮想
/// 特徴の出現列が「同 p を持つ base 実特徴の出現列の合併」である (base 実特徴 1 つ
/// につき仮想特徴ちょうど 1 つが対応) ことから、仮想 index を sparse backward に
/// 流す直接 gather と数学的に等価 (f32 加算順のみ異なる)。`base_ft_in` が縮約対象
/// の king-bucket セル数、`ft_in` (= base + threat) が仮想 P plane の手前。threat
/// real 行は仮想行に寄与しない。threat 無効時は `base_ft_in == ft_in` で threat
/// 連結前と bit-identical。実 block の gather (`gather_and_sum_per_feature_*`) が stm / nstm
/// 両方完了した後に launch する。1 thread = 1 仮想要素、仮想 block は overwrite。
#[kernel]
pub fn ft_reduce_virtual_grad(
    grad: &[f32],
    base_ft_in: u32,
    ft_in: u32,
    ft_out: u32,
    piece_inputs: u32,
) {
    let i = thread::index_1d();
    let ft_out_u = ft_out as usize;
    let pi_u = piece_inputs as usize;
    if i.get() >= pi_u * ft_out_u {
        return;
    }
    let p = i.get() / ft_out_u;
    let ri = i.get() - p * ft_out_u;
    let n_kb = (base_ft_in as usize) / pi_u;

    // raw pointer 版 (PTX の bounds check 除去)。unsafe 妥当性: caller が
    // `grad.len() == (ft_in + piece_inputs) * ft_out`、`base_ft_in <= ft_in`、
    // `base_ft_in % piece_inputs == 0` を保証し、読みは base 実 block
    // (`feature < base_ft_in`)、書きは thread ごとに disjoint な仮想 cell
    // (`ft_in + p` 行) に閉じる。
    // 4-way unroll: 1-load-1-fadd の依存 chain を 4 accumulator に分割して
    // in-flight load を確保する (`gather_and_sum_per_feature_*` と同じ理由)。
    // 加算順は kb 逐次和と異なるが f32 非結合則の丸め差のみ。
    let grad_ptr = grad.as_ptr();
    let row_stride = pi_u * ft_out_u;
    let base = p * ft_out_u + ri;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut kb = 0;
    let unroll_end = n_kb.saturating_sub(3);
    while kb < unroll_end {
        sum0 += unsafe { grad_ptr.add(base + kb * row_stride).read() };
        sum1 += unsafe { grad_ptr.add(base + (kb + 1) * row_stride).read() };
        sum2 += unsafe { grad_ptr.add(base + (kb + 2) * row_stride).read() };
        sum3 += unsafe { grad_ptr.add(base + (kb + 3) * row_stride).read() };
        kb += 4;
    }
    while kb < n_kb {
        sum0 += unsafe { grad_ptr.add(base + kb * row_stride).read() };
        kb += 1;
    }
    let sum = (sum0 + sum1) + (sum2 + sum3);
    let out_ptr = grad_ptr as *mut f32;
    unsafe {
        out_ptr.add((ft_in as usize + p) * ft_out_u + ri).write(sum);
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
        // SAFETY: `counts.len() == cols` (caller 契約) かつ `idx < cols`。`u32` と
        // `DeviceAtomicU32` は同 layout で、本 kernel 中の書き込みは atomic add のみ。
        let cell = unsafe { &*(counts.as_ptr().add(idx as usize) as *const DeviceAtomicU32) };
        cell.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

/// Exclusive prefix sum over `counts[0..n]` → `offsets[0..=n]`、単一 block × 1024
/// threads の **並列** Hillis-Steele scan (小 `n` 用)。inverse-index pipeline では
/// multi-block scan の level 2 = block 総和列 (`num_blocks` ≲ 135 要素) の scan に使う
/// (feature 数本体の分割は [`prefix_sum_block_local`] / [`prefix_sum_add_block_offset`])。
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
    // SAFETY: `offsets.len() == n + 1` (caller 契約)。各 thread の `[start, end)` は
    // 互いに重ならず、末尾 `offsets[n]` は最後の thread だけが書く。
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

/// Phase 2 of inverse-index, multi-block scan level 1: 各 block が `counts` の連続
/// `blockDim` 要素 (`blockIdx*blockDim ..`) を block 内 exclusive scan し、**block 内
/// local** な exclusive 値を `offsets` へ書く (block をまたぐ global offset は level 3
/// `prefix_sum_add_block_offset` で加算)。block の総和を `block_sums[blockIdx]` へ emit。
/// 単一 block scan ([`exclusive_prefix_sum_small`]) が 1 SM しか使えず大 `n` で律速に
/// なるため、feature 数を block 群へ分割して全 SM を使う。
///
/// host: block_dim=(1024,1,1), grid_dim=(num_blocks,1,1)、num_blocks=ceil(n/1024)。
#[kernel]
pub fn prefix_sum_block_local(counts: &[u32], offsets: &[u32], block_sums: &[u32], n: u32) {
    static mut PARTIALS: SharedArray<u32, 1024> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let blk = thread::blockIdx_x() as usize;
    let idx = blk * block_dim_u + tid;
    let n_u = n as usize;

    // 範囲外 thread は 0 を寄与 (末尾 block の partial 対応)。
    let val: u32 = if idx < n_u { counts[idx] } else { 0 };
    unsafe {
        PARTIALS[tid] = val;
    }
    thread::sync_threads();

    // Hillis-Steele inclusive scan
    let mut offset_step: usize = 1;
    while offset_step < block_dim_u {
        let add: u32 = if tid >= offset_step {
            unsafe { PARTIALS[tid - offset_step] }
        } else {
            0
        };
        thread::sync_threads();
        unsafe {
            PARTIALS[tid] += add;
        }
        thread::sync_threads();
        offset_step <<= 1;
    }

    // 自要素の block-local exclusive 値 (= 1 つ前の inclusive)。
    let excl: u32 = if tid == 0 {
        0
    } else {
        unsafe { PARTIALS[tid - 1] }
    };
    if idx < n_u {
        let out_ptr = offsets.as_ptr() as *mut u32;
        unsafe {
            out_ptr.add(idx).write(excl);
        }
    }
    // block 総和 = inclusive scan 最終値。tid=block_dim-1 が代表して書く。
    if tid == block_dim_u - 1 {
        let bs_ptr = block_sums.as_ptr() as *mut u32;
        unsafe {
            bs_ptr.add(blk).write(PARTIALS[block_dim_u - 1]);
        }
    }
}

/// Phase 2 of inverse-index, multi-block scan level 3: [`prefix_sum_block_local`] が
/// 書いた block-local exclusive offsets に、level 2 で求めた `block_offsets[blockIdx]`
/// (先行 block 群の総和) を in-place 加算し global exclusive prefix sum を確定する。
/// `offsets[n]` (= 全総和 = `block_offsets[num_blocks]`) も 1 thread が書く。
///
/// host: block_dim=(1024,1,1), grid_dim=(num_blocks,1,1)。`block_offsets` は
/// `block_sums` を [`exclusive_prefix_sum_small`] で scan した `num_blocks+1` 要素。
#[kernel]
pub fn prefix_sum_add_block_offset(
    offsets: &[u32],
    block_offsets: &[u32],
    n: u32,
    num_blocks: u32,
) {
    let tid = thread::threadIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let blk = thread::blockIdx_x() as usize;
    let idx = blk * block_dim_u + tid;
    let n_u = n as usize;

    let add = block_offsets[blk];
    if idx < n_u {
        let out_ptr = offsets.as_ptr() as *mut u32;
        unsafe {
            let p = out_ptr.add(idx);
            p.write(p.read() + add);
        }
    }
    // 全総和を offsets[n] へ。block 0 の tid 0 が代表。
    if blk == 0 && tid == 0 {
        let out_ptr = offsets.as_ptr() as *mut u32;
        unsafe {
            out_ptr.add(n_u).write(block_offsets[num_blocks as usize]);
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
        // SAFETY: `positions.len() == batch * nnz`、`offsets` は `counts` の prefix sum、
        // `write_counters[idx] < counts[idx]` なので `abs_pos` は範囲内。atomic increment
        // が同じ feature 内の rank を一意にするため、各 thread の書き込み先は重ならない。
        unsafe {
            let p = positions.as_ptr().add(abs_pos as usize) as *mut u32;
            p.write(bi);
        }
    }
}

macro_rules! gather_grad_read {
    (f32, $grad_out_ptr:ident, $index:expr) => {
        unsafe { $grad_out_ptr.add($index).read() }
    };
    (f16, $grad_out_ptr:ident, $index:expr) => {
        unsafe { $grad_out_ptr.add($index).read() } as f32
    };
}

macro_rules! gather_grad_scale {
    (noscale, $sum:ident) => {
        $sum
    };
    (scale, $sum:ident, $dft_inv_scale:ident) => {
        $sum * $dft_inv_scale
    };
}

macro_rules! gather_grad_flush {
    (overwrite, $grad_w:ident, $index:expr, $value:expr) => {{
        // 範囲外 (n_f=0、off_start == off_end) でも sum=0 を書き、host の事前 0-reset を不要にする。
        let out_ptr = $grad_w.as_ptr() as *mut f32;
        unsafe {
            out_ptr.add($index).write($value);
        }
    }};
    (add, $grad_w:ident, $index:expr, $sum:ident, $value:expr) => {
        if $sum != 0.0_f32 {
            // DeviceAtomicF32 は f32 と同じ layout / alignment で、同じ cell を
            // non-atomic に更新する path はこの kernel の実行中に存在しない。
            let cell = unsafe { &*($grad_w.as_ptr().add($index) as *const DeviceAtomicF32) };
            cell.fetch_add($value, AtomicOrdering::Relaxed);
        }
    };
}

macro_rules! gather_grad_finish {
    (
        overwrite,
        $scale:ident,
        $grad_w:ident,
        $index:expr,
        $sum:ident
        $(, $dft_inv_scale:ident)?
    ) => {
        gather_grad_flush!(
            overwrite,
            $grad_w,
            $index,
            gather_grad_scale!($scale, $sum $(, $dft_inv_scale)?)
        );
    };
    (
        add,
        $scale:ident,
        $grad_w:ident,
        $index:expr,
        $sum:ident
        $(, $dft_inv_scale:ident)?
    ) => {
        gather_grad_flush!(
            add,
            $grad_w,
            $index,
            $sum,
            gather_grad_scale!($scale, $sum $(, $dft_inv_scale)?)
        );
    };
}

// gather kernel の各変種で index 解決、4-way unroll、累算のコアを共有する。
macro_rules! gather_and_sum_per_feature_body {
    (
        $grad_out:ident,
        $positions:ident,
        $offsets:ident,
        $grad_w:ident,
        $n_features:ident,
        $ft_out:ident;
        $grad_type:ident,
        $scale:ident,
        $flush:ident
        $(, $dft_inv_scale:ident)?
    ) => {
        let feature = thread::blockIdx_x() as usize;
        let ri_block = thread::blockIdx_y() as usize;
        let tid_local = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x() as usize;
        let ri = ri_block * block_dim + tid_local;
        let ft_out_u = $ft_out as usize;
        if ri >= ft_out_u || feature >= ($n_features as usize) {
            return;
        }

        let off_start = $offsets[feature] as usize;
        let off_end = $offsets[feature + 1] as usize;

        // caller は positions の容量、offsets が示す有効範囲、grad_out / grad_w の
        // arch 固定長を保証する。launch config と上の検査により ri < ft_out_u。
        let grad_out_ptr = $grad_out.as_ptr();
        let positions_ptr = $positions.as_ptr();
        // 1-load-1-fadd では in-flight load が 1 個に限られ Long Scoreboard stall 中に
        // warp scheduler が idle になるため、4-way unroll で load 待ちを分散する。
        // 加算順は kernel の数値挙動の一部。
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
            sum0 += gather_grad_read!($grad_type, grad_out_ptr, bi0 * ft_out_u + ri);
            sum1 += gather_grad_read!($grad_type, grad_out_ptr, bi1 * ft_out_u + ri);
            sum2 += gather_grad_read!($grad_type, grad_out_ptr, bi2 * ft_out_u + ri);
            sum3 += gather_grad_read!($grad_type, grad_out_ptr, bi3 * ft_out_u + ri);
            i += 4;
        }
        while i < off_end {
            let bi = unsafe { positions_ptr.add(i).read() } as usize;
            sum0 += gather_grad_read!($grad_type, grad_out_ptr, bi * ft_out_u + ri);
            i += 1;
        }
        let sum = (sum0 + sum1) + (sum2 + sum3);

        gather_grad_finish!(
            $flush,
            $scale,
            $grad_w,
            feature * ft_out_u + ri,
            sum
            $(, $dft_inv_scale)?
        );
    };
}

/// Phase 4 of inverse-index: 各 feature について grad_out の対応 row を sum し、
/// `grad_w[feature][ri]` に書き出し (overwrite 版)。
///
/// block 構成: blockIdx_x = feature_id (`cols`)、blockIdx_y = ri tile (`ft_out / blockDim`)。
/// block_dim threads (各 1 ri cell、cell 境界は block 内で disjoint なため atomic 不要)。
/// launch grid の全 `(feature, ri)` cell を必ず書く (`off_start == off_end` の feature でも
/// sum=0 を書く) ため、caller は grad_w を事前 0-reset しなくてよい。
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
    gather_and_sum_per_feature_body!(
        grad_out, positions, offsets, grad_w, n_features, ft_out;
        f32, noscale, overwrite
    );
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
    gather_and_sum_per_feature_body!(
        grad_out, positions, offsets, grad_w, n_features, ft_out;
        f32, noscale, add
    );
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
    gather_and_sum_per_feature_body!(
        grad_out, positions, offsets, grad_w, n_features, ft_out;
        f16, scale, overwrite, dft_inv_scale
    );
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
    gather_and_sum_per_feature_body!(
        grad_out, positions, offsets, grad_w, n_features, ft_out;
        f16, scale, add, dft_inv_scale
    );
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
