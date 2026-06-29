//! Simple アーキ専用 kernel (`simple_*`)。

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicU64};
use cuda_device::{DisjointSlice, kernel, thread};

/// Simple FP16 FT activation forward (CReLU): f16 FT 出力 + f32 bias → f32 acted。
///
/// `--ft-fp16-out` 経路の融合 kernel。`sparse_ft_forward_fp16_out` の f16 出力
/// `ft_*_out_h` を直接 read (bias は別 buffer)、bias 加算と CReLU clamp を 1 pass で
/// 完了して f32 `combined` の per-perspective 列範囲 (`col_offset`) へ直接書く。FP32
/// path の `bias_add_per_row` + `crelu_fwd` 2 launch を 1 launch に置き換え、`ft_*_out`
/// (b × ft_dim) の DRAM read を f16 化して帯域を半減する。
///
/// 1 thread = 1 (batch, row) cell、atomic 不要。出力 `combined` は f32 のまま
/// (cuBLAS Sgemm が f32 を要求、中間 `ft_acted` + `slice_scatter_2d` 段は融合で除去)。bias は perspective 共有
/// で行内で同じ `ri` を warp 内で共有するため L1 hit pattern が良好。
#[kernel]
pub fn simple_bias_act_fwd_fp16_in_crelu(
    ft_out: &[f16],
    bias: &[f32],
    mut combined: DisjointSlice<f32>,
    combined_stride: u32,
    col_offset: u32,
    batch: u32,
    ft_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let ri = tid.get() % (ft_dim as usize);
    let bi = tid.get() / (ft_dim as usize);
    let x = ft_out[tid.get()] as f32 + bias[ri];
    #[allow(clippy::manual_clamp)]
    let y = if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    };
    // combined (batch × combined_stride) の per-perspective 列範囲へ直接 scatter
    // (中間 ft_acted + slice_scatter_2d の DRAM round-trip を融合で省く)。
    let idx = bi * (combined_stride as usize) + (col_offset as usize) + ri;
    // SAFETY: 各 thread が unique (bi, ri) → unique idx に書く。host が
    // `2*ft_out <= combined_dim` を call site の debug_assert で保証。
    unsafe {
        *combined.get_unchecked_mut(idx) = y;
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
    dcombined: &[f32],
    combined_stride: u32,
    col_offset: u32,
    mut dft_out: DisjointSlice<f16>,
    clamp_counter: &[u64], // len=1、clamp 発火数の cumulative atomic counter
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
    let bi = tid.get() / (ft_dim as usize);
    // dcombined (batch × combined_stride) の per-perspective 列範囲を直接読む
    // (slice_extract_2d で中間 buffer に取り出す DRAM round-trip を融合で省く)。
    let dft = dcombined[bi * (combined_stride as usize) + (col_offset as usize) + ri];
    let x = ft_out[tid.get()] as f32 + bias[ri];
    let g = if x > 0.0_f32 && x < 1.0_f32 {
        dft
    } else {
        0.0_f32
    };
    let s = g * dft_scale;
    let mut local_clamps: u64 = 0;
    let s_c = if s > 65504.0_f32 {
        local_clamps = 1;
        65504.0_f32
    } else if s < -65504.0_f32 {
        local_clamps = 1;
        -65504.0_f32
    } else {
        s
    };
    if let Some(o) = dft_out.get_mut(tid) {
        *o = s_c as f16;
    }
    if local_clamps > 0 {
        // SAFETY: see `ft_post_perspective_grad_fused_fp16` clamp_counter atomic add。
        let cell = unsafe { &*(clamp_counter.as_ptr() as *const DeviceAtomicU64) };
        cell.fetch_add(local_clamps, AtomicOrdering::Relaxed);
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
    mut combined: DisjointSlice<f32>,
    combined_stride: u32,
    col_offset: u32,
    batch: u32,
    ft_dim: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (ft_dim as usize);
    if tid.get() >= total {
        return;
    }
    let ri = tid.get() % (ft_dim as usize);
    let bi = tid.get() / (ft_dim as usize);
    let x = ft_out[tid.get()] as f32 + bias[ri];
    let a = if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    };
    // combined (batch × combined_stride) の per-perspective 列範囲へ直接 scatter
    // (中間 ft_acted + slice_scatter_2d の DRAM round-trip を融合で省く)。
    let idx = bi * (combined_stride as usize) + (col_offset as usize) + ri;
    // SAFETY: 各 thread が unique (bi, ri) → unique idx に書く。host が
    // `2*ft_out <= combined_dim` を call site の debug_assert で保証。
    unsafe {
        *combined.get_unchecked_mut(idx) = a * a;
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
    dcombined: &[f32],
    combined_stride: u32,
    col_offset: u32,
    mut dft_out: DisjointSlice<f16>,
    clamp_counter: &[u64], // len=1、clamp 発火数の cumulative atomic counter
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
    let bi = tid.get() / (ft_dim as usize);
    // dcombined (batch × combined_stride) の per-perspective 列範囲を直接読む
    // (slice_extract_2d で中間 buffer に取り出す DRAM round-trip を融合で省く)。
    let dft = dcombined[bi * (combined_stride as usize) + (col_offset as usize) + ri];
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
    let g = dft * dydx;
    let s = g * dft_scale;
    let mut local_clamps: u64 = 0;
    let s_c = if s > 65504.0_f32 {
        local_clamps = 1;
        65504.0_f32
    } else if s < -65504.0_f32 {
        local_clamps = 1;
        -65504.0_f32
    } else {
        s
    };
    if let Some(o) = dft_out.get_mut(tid) {
        *o = s_c as f16;
    }
    if local_clamps > 0 {
        // SAFETY: see `ft_post_perspective_grad_fused_fp16` clamp_counter atomic add。
        let cell = unsafe { &*(clamp_counter.as_ptr() as *const DeviceAtomicU64) };
        cell.fetch_add(local_clamps, AtomicOrdering::Relaxed);
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

/// Simple bwd_ft_act の fused kernel (CReLU 版): `slice_extract_2d` と `crelu_grad`
/// を 1 kernel に融合。`dcombined` の per-perspective 半分を切り出して読み取り、
/// `ft_pre_act` (pre-activation FT 出力) で CReLU 指示関数 `0 < x < 1` を作って
/// `dft_out` に直接書く。非融合の 2 kernel が要する中間 `dft_*_acted` buffer の
/// DRAM round-trip (b × ft_out × 4 byte の write+read) を省く。
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
