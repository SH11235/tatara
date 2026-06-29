//! LayerStack 専用 kernel。
//!
//! 設計方針:
//! - atomics は host が呼出前に gradient buffer を 0 初期化する accumulate semantics
//! - DisjointSlice<f32> は 1 thread = 1 cell の排他書き込み、&[f32] + raw atomic は
//!   多 thread → 1 cell の atomic accumulate
//! - cuda-oxide 制限: `f32::clamp` / `f32::max` / `f32::min` は if-else 展開

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicU32, DeviceAtomicU64};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

/// Fused FT post-processing (forward) — bias add → CReLU → pairwise_mul → scale。
///
/// `l0.forward(stm/nstm).crelu().pairwise_mul() * (127.0/128.0)` + `stm.concat(
/// nstm)` を 1 kernel に集約 (両 perspective まとめて combined 出力)。
///
/// 設計: 1 thread = combined buffer の 1 cell。`combined` の前半 (`[0, ft_dim/2)`) が
/// stm の pairwise_mul 出力、後半 (`[ft_dim/2, ft_dim)`) が nstm の pairwise_mul 出力。
/// 各 thread は自分が担当する combined cell の (batch, ri) と (is_stm, pair_idx) を
/// 判定して、対応する perspective ft_out を読みに行く。
///
/// `pairwise_mul` semantic: `slice_rows(0, n/2) * slice_rows(n/2, n)`、つまり
/// 前半 `[0, half)` と後半 `[half, n)` の **対応 index 同士** の積 (隣接 pair
/// でなく)。
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
/// `sparse_ft_forward_fp16` が `ft_*_out` を `f16` で書くのに合わせ、その read
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
    clamp_counter: &[u64], // len=1、clamp 発火数の cumulative atomic counter
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
    // (発散の代わりに有界な近似)。`clamp_counter` は cap が当たった要素数の cumulative
    // atomic counter で、host (`--monitor-fp16-clamps`) が sb 末に D2H read する。
    let da = grad_a * dft_scale;
    let mut local_clamps: u64 = 0;
    let da_c = if da > 65504.0_f32 {
        local_clamps += 1;
        65504.0_f32
    } else if da < -65504.0_f32 {
        local_clamps += 1;
        -65504.0_f32
    } else {
        da
    };
    let db = grad_b * dft_scale;
    let db_c = if db > 65504.0_f32 {
        local_clamps += 1;
        65504.0_f32
    } else if db < -65504.0_f32 {
        local_clamps += 1;
        -65504.0_f32
    } else {
        db
    };
    let out_ptr = grad_ft_out.as_mut_ptr();
    unsafe {
        out_ptr.add(ft_base + pair_idx).write(da_c as f16);
        out_ptr.add(ft_base + half + pair_idx).write(db_c as f16);
    }
    if local_clamps > 0 {
        // SAFETY: `clamp_counter.len() == 1` (host 契約)、`DeviceAtomicU64` は `u64`
        // (align 8) と同 layout (`#[repr(transparent)]` over `UnsafeCell<u64>`)。
        // 同 cell を更新するのは本 kernel および同 file の `ft_post_perspective_
        // grad_fp16`、`bins/nnue_train/src/kernels/simple.rs` の `simple_act_grad_
        // to_fp16_{crelu,screlu}_with_scale` の計 4 kernel 関数で、いずれも
        // `DeviceAtomicU64::fetch_add` 経由でのみ書く (non-atomic 経路無し)。
        // cumulative counter なので host も memset reset を出さない。
        let cell = unsafe { &*(clamp_counter.as_ptr() as *const DeviceAtomicU64) };
        cell.fetch_add(local_clamps, AtomicOrdering::Relaxed);
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
    clamp_counter: &[u64], // len=1、clamp 発火数の cumulative atomic counter
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
    // `clamp_counter` は cap が当たった要素数の cumulative atomic counter。
    let da = grad_a * dft_scale;
    let mut local_clamps: u64 = 0;
    let da_c = if da > 65504.0_f32 {
        local_clamps += 1;
        65504.0_f32
    } else if da < -65504.0_f32 {
        local_clamps += 1;
        -65504.0_f32
    } else {
        da
    };
    let db = grad_b * dft_scale;
    let db_c = if db > 65504.0_f32 {
        local_clamps += 1;
        65504.0_f32
    } else if db < -65504.0_f32 {
        local_clamps += 1;
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
    if local_clamps > 0 {
        // SAFETY: see `ft_post_perspective_grad_fused_fp16` clamp_counter atomic add。
        let cell = unsafe { &*(clamp_counter.as_ptr() as *const DeviceAtomicU64) };
        cell.fetch_add(local_clamps, AtomicOrdering::Relaxed);
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

/// Tiled shared-memory variant of [`dense_mm_bwd_input`]. L1f 用 (`in_dim=ft_out`)、
/// `batch % 16 == 0`、`in_dim % 16 == 0` を host が保証。`out_dim` は reduction 軸で、
/// 16 幅の out-tile に分割して loop で消化するため任意の値に対応する (`out_dim` が 16 の
/// 倍数でなければ末尾 out-tile は 0 padding)。
///
/// 非 tiled の [`dense_mm_bwd_input`] は w[ii][o] (out-major) read で warp 内 ii=0..31 が
/// stride out_dim の uncoalesced になる。本 kernel は W_TILE / DY_TILE を shared に
/// coalesced load し、各 thread が 1 (bi, ii) cell を out-tile ごと 16 FMA で完成させる。
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
    // W_TILE[ii_local][o] は row stride 17 で pad する。reduction の FMA read
    // `W_TILE[tid_i * 17 + o]` は tid_i (in-index) が warp 内で変化するため、stride 16 だと
    // 16 lane が 2 bank に集中して 8-way bank conflict になる。stride 17 で 16 bank に散る。
    static mut W_TILE: SharedArray<f32, 272> = SharedArray::UNINIT; // 16 ii × (16 o + 1 pad)
    static mut DY_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // TILE_B × TILE_OUT (16×16)

    let tid_local = thread::threadIdx_x() as usize;
    // 1D grid: block_idx encodes (b_block, ii_block). out 軸は kernel 内 loop で消化
    // するため grid には現れない。grid_dim = (batch/16) * (in_dim/16)、block index =
    // b_block * (in_dim/16) + ii_block。
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

    // dx[bi][ii] = sum_o dy[bi][o] * w[ii][o] は out 軸の reduction。out_dim を 16 幅の
    // out-tile に分割し、各 out-tile の W_TILE / DY_TILE を shared に coalesced load して
    // 単一 accumulator に加算する (accumulator は全 out-tile を貫いて 1 個)。
    let n_out_tiles = (out_dim_u + 15) >> 4;
    let mut acc = 0.0_f32;
    let mut ot: usize = 0;
    while ot < n_out_tiles {
        let o_load = (ot << 4) + tid_i;
        unsafe {
            // W_TILE[ii_local * 17 + o] = w[(ii_start + ii_local) * out_dim + (ot*16 + o)]。
            // tid 0..31 内で 16-thread sub-group が 16 連続 o を読む → coalesced。
            let ii_global_load = ii_start + tid_b;
            W_TILE[tid_b * 17 + tid_i] = if ii_global_load < in_dim_u && o_load < out_dim_u {
                w[ii_global_load * out_dim_u + o_load]
            } else {
                0.0_f32
            };
            // DY_TILE[b_local * 16 + o] = dy[(b_start + b_local) * out_dim + (ot*16 + o)]。
            let bb_global_load = b_start + tid_b;
            DY_TILE[tid_local] = if bb_global_load < batch_u && o_load < out_dim_u {
                dy[bb_global_load * out_dim_u + o_load]
            } else {
                0.0_f32
            };
        }
        thread::sync_threads();

        if bi_ok && ii_ok {
            let mut o: usize = 0;
            while o < 16 {
                unsafe {
                    acc += DY_TILE[(tid_b << 4) | o] * W_TILE[tid_i * 17 + o];
                }
                o += 1;
            }
        }
        thread::sync_threads();
        ot += 1;
    }

    if bi_ok && ii_ok {
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
/// chunk で cooperatively load し、shared memory 上で TILE_K 回 FMA。1 thread = 1 cell
/// で batch を scan する [`dense_mm_bwd_weight`] 比で unique memory read が ~33x 少ない
/// (x の 16x redundant read → 1x、dy も ft_out 回 → 1x)。
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
/// `num_buckets=9` は固定、`batch % 16 == 0`)。[`dense_mm_bwd_weight_bucket`] の
/// tiled variant。
///
/// 1 block = 1 W tile (16×16)。1 thread が 9 bucket 分の register accumulator を持ち、
/// `dy_tile` / `x_tile` / `buc_tile` を shared mem に coalesced load しながら batch を
/// TILE_K=16 chunk で 1 回 scan する。bucket 分岐は uniform (同 k 内で warp 全 thread が
/// 同 buc) なので divergence なし。
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
                    // num_buckets=9 を想定。負値・>=9 は無視 (silent skip)。
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

    // grad_w layout は `grad_w[buc][o][i]` row-major (bucket-major、その中 out-major)、
    // つまり cell index = buc * (out_dim * in_dim) + oi * in_dim + ii。
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
/// - `blockIdx_x` = `out_tile * (in_dim/16) + in_tile` (out-tile と in-tile を畳んだ 1 軸、
///   in_tile は `in_dim/16` 個、out_tile は `ceil(out_dim/16)` 個)
/// - `blockIdx_y` = bucket 内 split-K (`gridDim_y` 個の連続 TILE_K slice)
/// - `blockIdx_z` = bucket (`num_buckets` 個)
///
/// 各 block は uniform-by-construction で 1 bucket の slice のみ accumulate し、終端で
/// `grad_w[block_buc][oi][ii]` に 1 atomicAdd。`out_dim` は `TILE_OUT = 16` 幅の out-tile に
/// 分割、`out_dim` が 16 の倍数でないとき末尾 out-tile は `out_ok` guard で部分書き込み。
///
/// padding 行 (perm=-1 由来で `permute_rows_f32` が 0 fill) は x,dy=0 で sum=0 contribution、
/// bucket slice 末端の 16-alignment slack 行も同様に silent に 0 contribution。
///
/// 数値同等性: 加算順序が sort 済 batch 順 + split-K 集約順になるため fp32 associativity で
/// baseline と bit-exact ではないが、reduction tolerance (相対誤差 < `TOL`) 内で一致。
/// `in_dim % 16 == 0` / `num_buckets <= 9` / `padded_batch % 16 == 0` /
/// `bucket_offsets` が aligned exclusive scan 出力 / `blockIdx_x` 範囲は caller 契約。
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
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let padded_b_u = padded_batch as usize;
    let num_buc_u = num_buckets as usize;
    // blockIdx_x は (out_tile, in_tile) を畳んだ 1 軸。in_tile が下位。
    let in_tiles = in_dim_u >> 4;
    let block_x_raw = thread::blockIdx_x() as usize;
    let block_x = block_x_raw % in_tiles;
    let out_tile = block_x_raw / in_tiles;
    let block_split = thread::blockIdx_y() as usize;
    let num_splits = thread::gridDim_y() as usize;
    let block_buc = thread::blockIdx_z() as usize;
    let tid_i = tid_local >> 4;
    let tid_o = tid_local & 15;
    let global_ii = (block_x << 4) | tid_i;
    let global_oi = (out_tile << 4) + tid_o;

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
                let oi_load = (out_tile << 4) + tid_o;
                let mapped = (tid_i << 4) | tid_o;
                X_TILE[mapped] = if bb < buc_end && global_ii_load < in_dim_u {
                    x[bb * in_dim_u + global_ii_load]
                } else {
                    0.0_f32
                };
                DY_TILE[mapped] = if bb < buc_end && oi_load < out_dim_u {
                    dy[bb * out_dim_u + oi_load]
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

/// Bias gradient (block-level shared-mem reduction) — L1f 用。`out_dim` は `--l1` 依存で、
/// host が `<= 256` を保証する (`PARTIAL` の固定容量)。
///
/// 各 block (256 threads) が shared-mem の out_dim-cell accumulator に集約してから
/// 1 block × out_dim atomic add で global に flush する。全 thread が直接 global の
/// out_dim cells へ atomic add する [`bias_grad`] の contention を避ける。
#[kernel]
pub fn bias_grad_shared_l1f(dy: &[f32], grad_bias: &[f32], batch: u32, out_dim: u32) {
    use core::ptr::addr_of_mut;
    // out_dim (= l1_out) は host が <= 256 を保証。PARTIAL は固定上限で確保し先頭
    // out_dim cell のみ使う (1 KB、occupancy への影響は無視できる)。
    static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
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
/// [`dense_mm_fwd`] の tiled variant。
///
/// block tile (TILE_B=16 × TILE_OUT=16 = 256 cells) を K=16 chunk の shared-mem
/// cooperative load で計算する。1 thread = 1 (b, oi) の [`dense_mm_fwd`] は per-thread
/// で ft_out 回の K iteration を回すため並列度が限られるが、本 kernel は 256 cells /
/// block の tile 並列で 4K blocks × 256 threads まで広げる。
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
/// `num_buckets=9` は固定)。[`dense_mm_fwd_bucket`] の tiled variant。
///
/// 1 block = 1 batch tile (TILE_B=16) × 全 oi (= TILE_OUT=16)、K (= in_dim) を
/// TILE_K=16 chunk で消化し、shared memory 上に `w_tile [NUM_BUCKETS × TILE_OUT ×
/// TILE_K]` を per-K-tile coalesced load する。各 thread は自分の bucket の W 行を
/// shared から読んで accumulate。`w[buc][oi][ii]` を直接 read する
/// [`dense_mm_fwd_bucket`] は oi 軸 varying で stride=ft_out の uncoalesced read に
/// なるのを、shared への coalesced load で回避する。
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
/// は 1 bucket 分 (16 oi × 16 k、shared は bank conflict 回避で row stride 17) のみ load する
/// 分岐なし実装。sorted padding 行は
/// `bucket_idx = -1` で kernel が y=0 を書き、後段の inverse permute が perm=-1 sentinel で
/// skip して original 配列には戻らない。
///
/// `out_dim` は `TILE_OUT = 16` 幅の out-tile に分割し、各 out-tile を `blockIdx_y` で
/// 受ける (grid_dim_y = `ceil(out_dim / 16)`)。1 block = 1 (batch-tile, out-tile)。
/// `out_dim` が 16 の倍数でないとき末尾 out-tile は `oi_ok` guard で部分書き込み。
///
/// 数値同等性: per-row independent (k=0..15 加算順保持) で baseline と bit-exact、
/// sort stability 不要。`in_dim % 16 == 0` / `batch % 16 == 0` / `num_buckets <= 9` /
/// `grid_dim_y == ceil(out_dim/16)` は caller 契約。
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
    // W_TILE[oi_local][k] は row stride 17 で pad する。reduction の FMA read
    // `W_TILE[tid_o * 17 + k]` は tid_o (output-index) が warp 内で変化するため、stride 16 だと
    // 16 lane が 2 bank に集中して 8-way bank conflict になる。stride 17 で 16 bank に散る。
    static mut W_TILE: SharedArray<f32, 272> = SharedArray::UNINIT; // 1 bucket × 16 oi × (16 k + 1 pad)

    let tid_local = thread::threadIdx_x() as usize;
    let block_b = thread::blockIdx_x() as usize;
    let out_tile = thread::blockIdx_y() as usize;
    let tid_b = tid_local >> 4;
    let tid_o = tid_local & 15;
    let b_start = block_b << 4;
    let global_bi = b_start + tid_b;
    let global_oi = (out_tile << 4) + tid_o;

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
            let oi_global = (out_tile << 4) + oi_local;
            let val = if block_buc_ok && oi_global < out_dim_u && kk < in_dim_u {
                w[block_buc_u * out_dim_u * in_dim_u + oi_global * in_dim_u + kk]
            } else {
                0.0_f32
            };
            W_TILE[oi_local * 17 + k_local] = val;
        }
        thread::sync_threads();

        if bi_ok && oi_ok && block_buc_ok {
            let mut k: usize = 0;
            while k < 16 {
                unsafe {
                    acc += X_TILE[(tid_b << 4) | k] * W_TILE[tid_o * 17 + k];
                }
                k += 1;
            }
        }
        thread::sync_threads();
        k_tile += 1;
    }

    if bi_ok && oi_ok {
        // out-tile を grid に展開したため cell index は thread::index_1d() と一致しない。
        // 各 thread は disjoint な (global_bi, global_oi) cell を担当するので raw ptr write。
        // SAFETY: global_bi < batch、global_oi < out_dim ⇒ cell_idx < y.len() (= batch*out_dim)。
        let cell_idx = global_bi * out_dim_u + global_oi;
        let val = if block_buc_ok { acc } else { 0.0_f32 };
        unsafe {
            *y.as_mut_ptr().add(cell_idx) = val;
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

/// Sorted layout 版の per-bucket dense backward-input matmul (L1 用)。
/// `dx[b][i] = sum_o dy[b][o] * w[bucket_idx[b_start]][o][i]`。caller が batch を bucket で sort 済
/// かつ各 bucket の開始 offset が TILE_B = 16 境界に align 済を保証する前提 (block 内 16 row は
/// uniform bucket、boundary block 無し)。[`dense_mm_bwd_input_bucket`] の tiled variant:
/// dy_tile[16 b × 16 o] / w_tile[16 o × 16 i] を shared に coalesced load し、各 thread が 1 (b, i)
/// cell を out_dim 軸 (= 縮約 K) の reduction で完成する。非 tiled 版が同 bucket の row ごとに
/// w[bucket] を global から重複 read していたのを、16-row tile あたり 1 回の shared load に集約する。
///
/// w_tile は [o][i] の row stride 16。reduction read `w_tile[o*16 + tid_i]` は i (in-index) が
/// warp 内 stride 1 = fast index なので bank conflict 無し (pad 不要)。`in_dim % 16 == 0` /
/// `batch % 16 == 0` / `num_buckets <= 9` が caller 契約。out_dim は 16 幅の K-tile に分割するため
/// 任意 (末尾 tile は 0 padding)。padding 行 (bucket_idx=-1) は dx=0 を書き、後段の inverse permute
/// が perm=-1 で skip する。
///
/// 数値同等性: per (b,i) で o=0.. の加算順を保持するため非 tiled 版と bit-exact。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn dense_mm_bwd_input_bucket_tiled_sorted(
    dy: &[f32],
    w: &[f32],
    bucket_idx: &[i32],
    mut dx: DisjointSlice<f32>,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_buckets: u32,
) {
    static mut DY_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // 16 b × 16 o
    static mut W_TILE: SharedArray<f32, 256> = SharedArray::UNINIT; // 16 o × 16 i

    let tid_local = thread::threadIdx_x() as usize;
    let block_b = thread::blockIdx_y() as usize; // batch-tile
    let in_tile = thread::blockIdx_x() as usize; // in-tile
    let tid_b = tid_local >> 4; // 0..15 (batch within tile)
    let tid_i = tid_local & 15; // 0..15 (in within tile)
    let b_start = block_b << 4;
    let i_start = in_tile << 4;
    let global_bi = b_start + tid_b;
    let global_ii = i_start + tid_i;

    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    let bi_ok = global_bi < batch_u;
    let ii_ok = global_ii < in_dim_u;

    let block_buc = if b_start < batch_u {
        bucket_idx[b_start]
    } else {
        -1_i32
    };
    let block_buc_ok = block_buc >= 0 && (block_buc as u32) < num_buckets;
    let block_buc_u = if block_buc_ok { block_buc as usize } else { 0 };

    let mut acc: f32 = 0.0_f32;
    let n_o_tiles = (out_dim_u + 15) >> 4;
    let mut o_tile: usize = 0;
    while o_tile < n_o_tiles {
        let o_start = o_tile << 4;
        unsafe {
            // DY_TILE[b][o] = dy[(b_start+b)][o_start+o]、thread → (b=tid_b, o=tid_i)、coalesced。
            let bb = b_start + tid_b;
            let oo = o_start + tid_i;
            DY_TILE[tid_local] = if bb < batch_u && oo < out_dim_u {
                dy[bb * out_dim_u + oo]
            } else {
                0.0_f32
            };
            // W_TILE[o][i] = w[buc][o_start+o][i_start+i]、thread → (o=tid_b, i=tid_i)、coalesced。
            let oo2 = o_start + tid_b;
            let ii2 = i_start + tid_i;
            W_TILE[tid_local] = if block_buc_ok && oo2 < out_dim_u && ii2 < in_dim_u {
                w[block_buc_u * out_dim_u * in_dim_u + oo2 * in_dim_u + ii2]
            } else {
                0.0_f32
            };
        }
        thread::sync_threads();

        if bi_ok && ii_ok && block_buc_ok {
            let mut o: usize = 0;
            while o < 16 {
                unsafe {
                    acc += DY_TILE[(tid_b << 4) | o] * W_TILE[(o << 4) | tid_i];
                }
                o += 1;
            }
        }
        thread::sync_threads();
        o_tile += 1;
    }

    if bi_ok && ii_ok {
        // 2D tile grid のため cell index は thread::index_1d() と一致しない。各 thread は
        // disjoint な (global_bi, global_ii) cell を担当する。
        // SAFETY: global_bi < batch、global_ii < in_dim ⇒ cell_idx < dx.len() (= batch*in_dim)。
        let cell_idx = global_bi * in_dim_u + global_ii;
        let val = if block_buc_ok { acc } else { 0.0_f32 };
        unsafe {
            *dx.as_mut_ptr().add(cell_idx) = val;
        }
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

/// L3 weight backward (specialized: `out_dim=1`, `num_buckets=9`; `in_dim` は L2 の
/// 出力次元で runtime arg)。
///
/// split-K + 9 bucket register accumulator で並列度を確保する:
/// - block dim = in_dim (1 thread = 1 ii cell)。`ii >= in_dim` の thread は return
///   するため、caller は block_dim を in_dim に一致させる (小さいと末尾 cell が未計算)。
/// - grid = num_batch_splits (e.g., 64)
/// - 各 thread が 9 bucket × 1 ii の partial sum を batch_slice 内で集計
/// - 完了後、9 cell ぶん atomicAdd で global grad_w に flush
///
/// 汎用の [`dense_mm_bwd_weight_bucket`] は L3 形状では (in_dim * num_buckets) cells
/// 分の threads しか使えず並列度が極小になるため、本 specialized kernel を使う。
///
/// host 契約: grad_w は呼出前に 0 reset (accumulate semantics)。out_dim==1,
/// num_buckets==9、block_dim==in_dim を満たすこと。
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

/// L2 weight backward (`out_dim` は L2 出力次元 `l2_out` (`--l2` 依存)、`in_dim = l2_in`
/// は `--l1` 依存、`num_buckets <= 9`)。
///
/// split-K + per-bucket register accumulator (1 thread = 1 (oi, ii) cell × 9 bucket acc)
/// で並列度を確保する。weight cell 空間 (per-bucket `out_dim * in_dim`) を `blockIdx_x`、
/// batch split-K を `blockIdx_y` に分け、`block_dim` は cell 数と独立な固定値で launch
/// する (`block_dim = out_dim * in_dim` だと `l2_in` 次第で 1024 thread を超えるため)。
/// 汎用の [`dense_mm_bwd_weight_bucket`] は batch を bucket ごとに再 scan する分遅い。
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
    let block_cell = thread::blockIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let block_split = thread::blockIdx_y() as usize;
    let num_splits = thread::gridDim_y() as usize;
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let batch_u = batch as usize;
    // weight cell 空間 (per-bucket out_dim*in_dim) を blockIdx_x で分割し、1 thread =
    // 1 (oi, ii) cell。範囲外 thread は早期 return。
    let per_bucket = out_dim_u * in_dim_u;
    let cell_in_bucket = block_cell * block_dim_u + tid_local;
    if cell_in_bucket >= per_bucket {
        return;
    }
    let oi = cell_in_bucket / in_dim_u;
    let ii = cell_in_bucket % in_dim_u;

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

    // grad_w layout: buc * (out_dim * in_dim) + oi * in_dim + ii (= per_bucket + cell_in_bucket)。
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
/// (`exclusive_scan_aligned` 経由) を保証する前提。1 block = sorted batch の連続 16 行 ×
/// `out_dim` oi。16-aligned sort 配下で 16 行は同一 bucket (uniform-by-construction)、
/// `bucket_idx_sorted[b_start]` で代表 bucket を取得し PARTIAL[out_dim] shared-mem
/// accumulator に集約 → 1 block × out_dim atomic add で `grad_bias[block_buc][:]` に flush。
/// 全 thread が直接 global へ atomic add する [`bias_grad_bucket`] の contention を避ける。
///
/// padding 行 / 範囲外 bucket (block_buc = -1) は skip (PARTIAL flush しない)、
/// caller が `grad_bias` を 0 初期化済の前提 (accumulate semantics は元と同じ)。
///
/// 数値同等性: 加算順が sort 済 batch 順 + per-block reduce 順になるため fp32
/// associativity で baseline と bit-exact ではないが、reduction tolerance 内で一致。
/// `block_dim == 256` / `padded_batch % 16 == 0` / `num_buckets <= 9` / `out_dim <= 256`
/// (PARTIAL 固定容量) / `grid_dim_x == padded_batch / 16` は caller 契約。
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
    // out_dim (= l1_out / l2_out) は host が <= 256 を保証。先頭 out_dim cell のみ使う。
    static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_idx = thread::blockIdx_x() as usize;
    let block_dim_u = thread::blockDim_x() as usize;
    let out_dim_u = out_dim as usize;
    let padded_b_u = padded_batch as usize;

    // 1 block = sorted batch の連続 16 行。16-aligned sort なので 16 行は同一 bucket。
    let b_start = block_idx << 4;
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

    // 16 行 × out_dim cell を block_dim stride で走査し、各 cell の dy を PARTIAL[oi] に
    // shared atomic add する (block 内の contention は out_dim cell に限定)。
    if block_buc_ok {
        let n_cells = 16 * out_dim_u;
        let mut c = tid;
        while c < n_cells {
            let row = c / out_dim_u;
            let oi = c % out_dim_u;
            let bb = b_start + row;
            if bb < padded_b_u {
                let dyv = dy[bb * out_dim_u + oi];
                let cell = unsafe { &*(partial_ptr.add(oi) as *const DeviceAtomicF32) };
                cell.fetch_add(dyv, AtomicOrdering::Relaxed);
            }
            c += block_dim_u;
        }
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
/// `screlu_grad` と対の forward。Simple アーキの dense 層活性化
/// (`--activation screlu`、`SimpleGpuTrainer` の forward 経路) で launch される。
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
/// `|x|^2 = x^2` なので abs は不要。1 thread = 1 element。
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

// =============================================================================
// PSQT shortcut (Stockfish SFNNv10 系の per-feature × per-bucket スカラー prior)
// =============================================================================
//
// 形式:
//   psqt_w shape   = (rows, NUM_BUCKETS) row-major (`psqt_w[feat * NB + bucket]`、
//                    rows は base `ft_in`、factorizer 有効時は仮想 P 行込みの
//                    `train_ft_in`。本 kernel は `feat < ft_in` の実 block のみ touch)
//   forward 出力   = net_output[b] += 0.5 * (Σ_f∈stm_active psqt_w[f,bk]
//                                            − Σ_f∈nstm_active psqt_w[f,bk])
//   backward       = 各 (b, ni) で psqt_w_grad[stm_idx[b,ni], bk] += +0.5 * dnet[b]
//                                  psqt_w_grad[nstm_idx[b,ni], bk] += −0.5 * dnet[b]
//
// PSQT bias は対称差で勾配 0 のため kernel は持たない (.bin 上は 0 固定で書く)。

/// PSQT shortcut forward (in-place add to net_output)。
///
/// 1 thread = 1 position (batch index `b`)。`net_output` には事前に
/// `l3_out + l1_skip` が書かれている前提で、PSQT delta を加算する (in-place)。
/// 1 thread / 1 cell の排他更新なので atomic 不要。
///
/// `bucket_idx[b] < 0` または `>= num_buckets` の position は skip (bias 0 と等価)。
/// `idx >= 0 && (idx as u32) < ft_in` の通常の sparse FT 防御も同様。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn psqt_diff_sparse_fwd_inplace(
    psqt_w: &[f32],
    stm_indices: &[i32],
    nstm_indices: &[i32],
    bucket_idx: &[i32],
    mut net_output: DisjointSlice<f32>,
    batch: u32,
    nnz: u32,
    num_buckets: u32,
    ft_in: u32,
) {
    let b = thread::index_1d();
    if b.get() >= batch as usize {
        return;
    }
    let bucket = bucket_idx[b.get()];
    if bucket < 0 || (bucket as u32) >= num_buckets {
        return;
    }
    let bucket_u = bucket as usize;
    let nb_u = num_buckets as usize;
    let base = b.get() * (nnz as usize);
    let mut sum_stm: f32 = 0.0;
    let mut sum_nstm: f32 = 0.0;
    let mut ni: u32 = 0;
    while ni < nnz {
        let idx_s = stm_indices[base + (ni as usize)];
        if idx_s >= 0 && (idx_s as u32) < ft_in {
            sum_stm += psqt_w[(idx_s as usize) * nb_u + bucket_u];
        }
        let idx_n = nstm_indices[base + (ni as usize)];
        if idx_n >= 0 && (idx_n as u32) < ft_in {
            sum_nstm += psqt_w[(idx_n as usize) * nb_u + bucket_u];
        }
        ni += 1;
    }
    let delta = 0.5_f32 * (sum_stm - sum_nstm);
    if let Some(out) = net_output.get_mut(b) {
        *out += delta;
    }
}

/// PSQT shortcut backward (atomic scatter to psqt_w_grad)。
///
/// 1 thread = 1 (batch index, nnz position) pair。stm + nstm の両 sparse index を
/// 同 thread で処理し、`psqt_w_grad[idx * num_buckets + bucket]` に
/// `±0.5 * dnet[b]` を atomic-add する。**accumulate semantics** (host が呼出前に
/// `psqt_w_grad` を 0 初期化する責務)。
///
/// `bucket_idx[b] < 0` または `>= num_buckets` の position は skip。num_buckets
/// 上限 9 が小さく contention は限定的 (例: N=9, batch=65536 / nnz=40 / stm+nstm
/// 両 add で 5.2M atomic-add / 660K cells ≒ 平均 8 thread/cell)。
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn psqt_diff_sparse_bwd(
    dnet: &[f32],
    stm_indices: &[i32],
    nstm_indices: &[i32],
    bucket_idx: &[i32],
    psqt_w_grad: &[f32],
    batch: u32,
    nnz: u32,
    num_buckets: u32,
    ft_in: u32,
) {
    let tid = thread::index_1d();
    let total = (batch as usize) * (nnz as usize);
    if tid.get() >= total {
        return;
    }
    let bi = tid.get() / (nnz as usize);
    let ni = tid.get() % (nnz as usize);
    let bucket = bucket_idx[bi];
    if bucket < 0 || (bucket as u32) >= num_buckets {
        return;
    }
    let bucket_u = bucket as usize;
    let nb_u = num_buckets as usize;
    let half_g = 0.5_f32 * dnet[bi];
    let idx_s = stm_indices[bi * (nnz as usize) + ni];
    if idx_s >= 0 && (idx_s as u32) < ft_in {
        // SAFETY: `psqt_w_grad.len() >= ft_in * num_buckets` host invariant
        // (factorizer 有効時は仮想行込みで train 長だが本 kernel は実 block のみ書く)、
        // `idx_s < ft_in` / `bucket_u < num_buckets` で範囲内。`f32` (align 4) と
        // `DeviceAtomicF32` (`#[repr(transparent)]`) は同 alignment、非 atomic 経路で
        // 同 memory に書く path は本 kernel + radam_step 以外無し。
        let cell = unsafe {
            &*(psqt_w_grad.as_ptr().add((idx_s as usize) * nb_u + bucket_u)
                as *const DeviceAtomicF32)
        };
        cell.fetch_add(half_g, AtomicOrdering::Relaxed);
    }
    let idx_n = nstm_indices[bi * (nnz as usize) + ni];
    if idx_n >= 0 && (idx_n as u32) < ft_in {
        let cell = unsafe {
            &*(psqt_w_grad.as_ptr().add((idx_n as usize) * nb_u + bucket_u)
                as *const DeviceAtomicF32)
        };
        cell.fetch_add(-half_g, AtomicOrdering::Relaxed);
    }
}
