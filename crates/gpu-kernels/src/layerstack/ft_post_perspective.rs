//! Fused FT post-processing (forward / backward) kernel の reference CPU 実装。
//!
//! GPU 側 (`#[kernel] fn ft_post_perspective_fwd` / `_grad`) は `bins/nnue_train/
//! src/main.rs` に inline 定義 (cuda-oxide bin-entry 制約)。`l0.forward(stm/nstm)
//! .crelu().pairwise_mul() * (127.0/128.0)` + `stm.concat(nstm)` を 1 kernel に
//! 集約したもの。`pairwise_mul` の semantic は `slice_rows(0, n/2) * slice_rows(
//! n/2, n)`、すなわち前半と後半の **対応 index 同士** の積 (隣接 pair ではない)。
//!
//! ## Layout 規約 (kernel と完全一致させる — テストの核心)
//!
//! - `stm_ft_out` / `nstm_ft_out`: row-major `batch × ft_dim` (`= sparse_ft_forward`
//!   の output、ft_dim = 1536)。
//! - `bias`: `ft_dim` (`= FT bias`、**stm/nstm 共有**)。
//! - `combined`: row-major `batch × ft_dim` (= COMBINED_DIM)。前半 `[0, half)` が
//!   stm の pairwise 出力 (`half = ft_dim/2 = 768`)、後半 `[half, ft_dim)` が nstm の
//!   pairwise 出力。`combined[bi][ri]` の値:
//!   - `ri < half`: stm の pair `ri` (= `crelu(stm_ft[ri] + bias[ri]) * crelu(stm_ft[half+ri] + bias[half+ri]) * scale`)
//!   - `ri >= half`: nstm の pair `ri-half` (同上、nstm_ft + 同じ bias を使う)
//! - `scale = 127.0/128.0` (`= FT_POST_SCALE`、qa=127 由来)。
//! - CReLU 境界は `[0, 1]` strict (gradient は `0 < x < 1` のみ非零)。NaN は kernel
//!   の if-else 展開どおり透過 (`!(NaN<0) && !(NaN>1)` → x).
//!
//! ## backward の per-perspective 規約
//!
//! [`ft_post_perspective_grad_cpu`] は **1 perspective 分** を計算する (kernel と同じ)。
//! - `ft_out`: その perspective の `sparse_ft_forward` output (`batch × ft_dim`)。
//! - `grad_ft_out`: その perspective の dft 出力 (`batch × ft_dim`)、overwrite。
//! - `grad_bias`: 共有 FT bias の grad (`ft_dim`)、**accumulate** (host が呼出前に
//!   0 初期化、stm/nstm の 2 call で足し合わせ)。
//! - `d_combined`: 上流の combined への grad (`batch × d_combined_stride`)。
//! - `d_combined_offset`: combined 内の自 perspective の起点 (stm: 0、nstm: half)。
//! - `d_combined_stride`: `= COMBINED_DIM = ft_dim`。
//!
//! 1 cell `ii ∈ [0, ft_dim)`: pair `(pair_idx, is_first) = if ii<half {(ii,true)}
//! else {(ii-half,false)}`、partner は同 `pair_idx` の反対側 (first ↔ second)。
//! `dy = d_combined[bi*stride + offset + pair_idx]`、`grad_my_post = dy * partner_post
//! * scale`、`grad_my_pre = grad_my_post if 0 < my_pre < 1 else 0`、
//! `grad_ft_out[bi*ft_dim+ii] = grad_my_pre`、`grad_bias[ii] += grad_my_pre`。

#[inline]
#[allow(clippy::manual_clamp)] // kernel の if-else 展開と bit-完全一致させるため (f32::clamp は cuda-oxide で lowering 失敗)。
fn crelu01(x: f32) -> f32 {
    // kernel の if-else 展開と bit-完全一致 (f32::clamp は lowering 失敗)。
    if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    }
}

/// Fused FT post-processing forward reference (両 perspective まとめて `combined` を作る)。
///
/// `stm_ft_out.len() == nstm_ft_out.len() == batch * ft_dim`、`bias.len() == ft_dim`、
/// `combined.len() == batch * ft_dim` 前提。`ft_dim` は偶数 (half = ft_dim/2)。
pub fn ft_post_perspective_fwd_cpu(
    stm_ft_out: &[f32],
    nstm_ft_out: &[f32],
    bias: &[f32],
    combined: &mut [f32],
    batch: usize,
    ft_dim: usize,
    scale: f32,
) {
    let half = ft_dim / 2;
    for bi in 0..batch {
        let ft_base = bi * ft_dim;
        for ri in 0..ft_dim {
            let val = if ri < half {
                let xa = stm_ft_out[ft_base + ri] + bias[ri];
                let xb = stm_ft_out[ft_base + half + ri] + bias[half + ri];
                crelu01(xa) * crelu01(xb) * scale
            } else {
                let pair_idx = ri - half;
                let xa = nstm_ft_out[ft_base + pair_idx] + bias[pair_idx];
                let xb = nstm_ft_out[ft_base + half + pair_idx] + bias[half + pair_idx];
                crelu01(xa) * crelu01(xb) * scale
            };
            combined[bi * ft_dim + ri] = val;
        }
    }
}

/// Fused FT post-processing backward reference (**1 perspective 分**)。
///
/// `d_combined.len() == batch * d_combined_stride`、`ft_out.len() == batch * ft_dim`、
/// `bias.len() == ft_dim`、`grad_ft_out.len() == batch * ft_dim`、`grad_bias.len() == ft_dim`
/// 前提。`grad_bias` は **既存値に加算** (kernel と同じ accumulate semantics、host が
/// 呼出前に 0 初期化、stm/nstm の 2 call で和)。
#[allow(clippy::too_many_arguments)]
pub fn ft_post_perspective_grad_cpu(
    d_combined: &[f32],
    ft_out: &[f32],
    bias: &[f32],
    grad_ft_out: &mut [f32],
    grad_bias: &mut [f32],
    batch: usize,
    ft_dim: usize,
    d_combined_offset: usize,
    d_combined_stride: usize,
    scale: f32,
) {
    let half = ft_dim / 2;
    for bi in 0..batch {
        let ft_base = bi * ft_dim;
        for ii in 0..ft_dim {
            let (pair_idx, is_first) = if ii < half {
                (ii, true)
            } else {
                (ii - half, false)
            };
            let dy = d_combined[bi * d_combined_stride + d_combined_offset + pair_idx];
            let xa = ft_out[ft_base + pair_idx] + bias[pair_idx];
            let xb = ft_out[ft_base + half + pair_idx] + bias[half + pair_idx];
            let ya = crelu01(xa);
            let yb = crelu01(xb);
            let (my_pre, partner_post) = if is_first { (xa, yb) } else { (xb, ya) };
            let grad_my_post = dy * partner_post * scale;
            let grad_my_pre = if my_pre > 0.0_f32 && my_pre < 1.0_f32 {
                grad_my_post
            } else {
                0.0_f32
            };
            grad_ft_out[bi * ft_dim + ii] = grad_my_pre;
            grad_bias[ii] += grad_my_pre;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALE: f32 = 127.0_f32 / 128.0_f32;

    /// 手計算: ft_dim=4 (half=2), batch=1, bias=0, scale=1.
    /// stm_ft = [0.5, 0.25, 2.0, -1.0] → pair0: crelu(0.5)*crelu(2.0) = 0.5*1.0 = 0.5
    ///                                    pair1: crelu(0.25)*crelu(-1.0) = 0.25*0.0 = 0.0
    /// nstm_ft = [0.8, 0.1, 0.3, 0.9] → pair0: crelu(0.8)*crelu(0.3) = 0.8*0.3 = 0.24
    ///                                   pair1: crelu(0.1)*crelu(0.9) = 0.1*0.9 = 0.09
    /// combined = [0.5, 0.0, 0.24, 0.09]
    #[test]
    fn forward_hand_computed_scale_one() {
        let stm = vec![0.5_f32, 0.25, 2.0, -1.0];
        let nstm = vec![0.8_f32, 0.1, 0.3, 0.9];
        let bias = vec![0.0_f32; 4];
        let mut combined = vec![0.0_f32; 4];
        ft_post_perspective_fwd_cpu(&stm, &nstm, &bias, &mut combined, 1, 4, 1.0);
        // 0.8*0.3 and 0.1*0.9 may have tiny f32 round-off; compare with computed f32.
        let exp = [0.5_f32 * 1.0, 0.25_f32 * 0.0, 0.8_f32 * 0.3, 0.1_f32 * 0.9];
        for i in 0..4 {
            assert_eq!(combined[i], exp[i], "i={i}");
        }
    }

    /// bias add then CReLU then pairwise. ft_dim=2 (half=1), batch=1.
    /// stm = [10.0, -10.0], bias = [0.0, 0.0] → xa=10→crelu=1, xb=-10→crelu=0 → 0*1*scale=0
    /// stm = [0.5, 0.5], bias = [0.5, 0.5] → xa=1.0→crelu=1, xb=1.0→crelu=1 → 1*1*scale=scale
    #[test]
    fn forward_bias_and_clamp() {
        let bias = vec![0.5_f32, 0.5];
        let stm1 = vec![10.0_f32, -10.0];
        let nstm1 = vec![0.0_f32, 0.0];
        let mut c1 = vec![0.0_f32; 2];
        ft_post_perspective_fwd_cpu(&stm1, &nstm1, &bias, &mut c1, 1, 2, SCALE);
        // ri=0 (stm pair0): xa = 10+0.5 → crelu 1; xb = -10+0.5 → crelu 0 → 0
        assert_eq!(c1[0], 0.0_f32);
        // ri=1 (nstm pair0): xa = 0+0.5 → crelu 0.5; xb = 0+0.5 → crelu 0.5 → 0.25*scale
        assert_eq!(c1[1], 0.25_f32 * SCALE);

        let stm2 = vec![0.5_f32, 0.5];
        let nstm2 = vec![0.5_f32, 0.5];
        let mut c2 = vec![0.0_f32; 2];
        ft_post_perspective_fwd_cpu(&stm2, &nstm2, &bias, &mut c2, 1, 2, SCALE);
        // xa = 0.5+0.5 = 1.0 → crelu 1; xb = 1.0 → crelu 1 → 1*scale
        assert_eq!(c2[0], SCALE);
        assert_eq!(c2[1], SCALE);
    }

    /// gradient: ft_dim=2 (half=1), batch=1, bias=0, scale=1, interior values.
    /// Perspective with ft_out = [0.5, 0.25] (pair0): xa=0.5, xb=0.25, ya=0.5, yb=0.25.
    /// d_combined gives dy for pair0 = 3.0 at offset 0.
    /// ii=0 (is_first, my_pre=xa=0.5, partner_post=yb=0.25): grad_my_post = 3.0*0.25*1 = 0.75;
    ///   0<0.5<1 → grad_ft_out[0] = 0.75; grad_bias[0] += 0.75
    /// ii=1 (second, my_pre=xb=0.25, partner_post=ya=0.5): grad_my_post = 3.0*0.5*1 = 1.5;
    ///   0<0.25<1 → grad_ft_out[1] = 1.5; grad_bias[1] += 1.5
    #[test]
    fn grad_hand_computed_interior() {
        let ft_out = vec![0.5_f32, 0.25];
        let bias = vec![0.0_f32; 2];
        // d_combined stride = ft_dim = 2; only [bi*2 + 0 + pair_idx] read. pair_idx in {0}.
        let d_combined = vec![3.0_f32, 999.0]; // [pair0=3.0, (unused for offset 0)]
        let mut grad_ft_out = vec![0.0_f32; 2];
        let mut grad_bias = vec![0.0_f32; 2];
        ft_post_perspective_grad_cpu(
            &d_combined,
            &ft_out,
            &bias,
            &mut grad_ft_out,
            &mut grad_bias,
            1,
            2,
            0, // stm offset
            2, // stride
            1.0,
        );
        assert_eq!(grad_ft_out, vec![0.75_f32, 1.5]);
        assert_eq!(grad_bias, vec![0.75_f32, 1.5]);
    }

    /// gradient zeroed at CReLU saturation. ft_out = [-1.0 (clamps to 0), 5.0 (clamps to 1)].
    /// xa = -1 → my_pre=-1 (not in (0,1)) → grad_ft_out[0] = 0
    /// xb = 5 → my_pre=5 (not in (0,1)) → grad_ft_out[1] = 0
    #[test]
    fn grad_zeroed_at_saturation() {
        let ft_out = vec![-1.0_f32, 5.0];
        let bias = vec![0.0_f32; 2];
        let d_combined = vec![10.0_f32, 0.0];
        let mut grad_ft_out = vec![0.0_f32; 2];
        let mut grad_bias = vec![7.0_f32, 8.0]; // pre-existing accumulate
        ft_post_perspective_grad_cpu(
            &d_combined,
            &ft_out,
            &bias,
            &mut grad_ft_out,
            &mut grad_bias,
            1,
            2,
            0,
            2,
            1.0,
        );
        assert_eq!(grad_ft_out, vec![0.0_f32, 0.0]);
        // accumulate += 0 → unchanged
        assert_eq!(grad_bias, vec![7.0_f32, 8.0]);
    }

    /// nstm offset path: with stride = ft_dim, offset = half. ft_dim=4, half=2, batch=1.
    /// d_combined layout: [stm_pair0, stm_pair1, nstm_pair0, nstm_pair1]. nstm uses
    /// indices [bi*4 + 2 + pair_idx], pair_idx in {0,1}.
    #[test]
    fn grad_nstm_offset_reads_second_half_of_d_combined() {
        let ft_dim = 4;
        let half = 2;
        // perspective ft_out interior
        let ft_out = vec![0.5_f32, 0.5, 0.5, 0.5]; // pair0: xa=ft[0]=0.5, xb=ft[2]=0.5; pair1: xa=ft[1]=0.5, xb=ft[3]=0.5
        let bias = vec![0.0_f32; 4];
        // d_combined: stm pair0 = 1, stm pair1 = 2, nstm pair0 = 10, nstm pair1 = 20
        let d_combined = vec![1.0_f32, 2.0, 10.0, 20.0];
        let mut grad_ft_out = vec![0.0_f32; 4];
        let mut grad_bias = vec![0.0_f32; 4];
        ft_post_perspective_grad_cpu(
            &d_combined,
            &ft_out,
            &bias,
            &mut grad_ft_out,
            &mut grad_bias,
            1,
            ft_dim,
            half,   // nstm offset
            ft_dim, // stride
            1.0,
        );
        // ii=0 (pair0, is_first): dy = d_combined[0 + 2 + 0] = 10; my_pre=xa=ft[0]=0.5; partner_post=yb=crelu(ft[2]=0.5)=0.5
        //   grad = 10*0.5*1 = 5.0 → grad_ft_out[0] = 5.0
        // ii=1 (pair1, is_first): dy = d_combined[2 + 1] = 20; my_pre=xa=ft[1]=0.5; partner_post=yb=crelu(ft[3]=0.5)=0.5
        //   grad = 20*0.5 = 10.0 → grad_ft_out[1] = 10.0
        // ii=2 (pair0, second): dy = d_combined[2+0] = 10; my_pre=xb=ft[2]=0.5; partner_post=ya=crelu(ft[0]=0.5)=0.5
        //   grad = 10*0.5 = 5.0 → grad_ft_out[2] = 5.0
        // ii=3 (pair1, second): dy = d_combined[2+1] = 20; my_pre=xb=ft[3]=0.5; partner_post=ya=crelu(ft[1])=0.5
        //   grad = 20*0.5 = 10.0 → grad_ft_out[3] = 10.0
        assert_eq!(grad_ft_out, vec![5.0_f32, 10.0, 5.0, 10.0]);
        assert_eq!(grad_bias, vec![5.0_f32, 10.0, 5.0, 10.0]);
    }

    /// forward/backward chain-rule cross-check: for one batch, with all CReLU
    /// inputs strictly interior, the product f(xa,xb) = crelu(xa)*crelu(xb)*scale =
    /// xa*xb*scale, so d/dxa = xb*scale, d/dxb = xa*scale. The grad kernel's
    /// grad_ft_out (since bias add doesn't change derivative wrt ft_out) should equal
    /// d_combined_dy * partner * scale. We verify via the identity
    ///   sum_{ii} grad_ft_out[ii] * ft_out[ii]  (proportional, not exact equality)
    /// — instead just check the closed form directly against fwd's pairwise product.
    #[test]
    fn fwd_bwd_chain_rule_interior_closed_form() {
        let ft_dim = 6;
        let half = 3;
        let batch = 2;
        // keep ft + bias sums in (0,1)
        let ft_out: Vec<f32> = (0..batch * ft_dim)
            .map(|i| 0.1 + 0.05 * (i % 7) as f32)
            .collect();
        let bias: Vec<f32> = (0..ft_dim).map(|i| 0.01 * i as f32).collect();
        let d_combined: Vec<f32> = (0..batch * ft_dim).map(|i| 0.3 * i as f32 - 0.7).collect();
        let mut grad_ft_out = vec![0.0_f32; batch * ft_dim];
        let mut grad_bias = vec![0.0_f32; ft_dim];
        ft_post_perspective_grad_cpu(
            &d_combined,
            &ft_out,
            &bias,
            &mut grad_ft_out,
            &mut grad_bias,
            batch,
            ft_dim,
            0,
            ft_dim,
            SCALE,
        );
        // closed form: for ii in [0,half) (is_first, pair=ii): grad = dy_pair * crelu(xb) * scale,
        //   xb = ft_out[half+ii]+bias[half+ii]; since interior, crelu(xb)=xb.
        // for ii in [half,ft_dim) (second, pair=ii-half): grad = dy_pair * crelu(xa) * scale,
        //   xa = ft_out[pair]+bias[pair].
        for bi in 0..batch {
            for ii in 0..ft_dim {
                let (pair, is_first) = if ii < half {
                    (ii, true)
                } else {
                    (ii - half, false)
                };
                let dy = d_combined[bi * ft_dim + pair];
                let xa = ft_out[bi * ft_dim + pair] + bias[pair];
                let xb = ft_out[bi * ft_dim + half + pair] + bias[half + pair];
                let partner = if is_first { xb } else { xa };
                // all interior by construction
                let exp = dy * partner * SCALE;
                let got = grad_ft_out[bi * ft_dim + ii];
                assert!(
                    (got - exp).abs() < 1e-5,
                    "bi={bi} ii={ii} got={got} exp={exp}"
                );
            }
        }
    }

    #[test]
    fn forward_nan_propagates() {
        let stm = vec![f32::NAN, 0.5];
        let nstm = vec![0.5_f32, 0.5];
        let bias = vec![0.0_f32; 2];
        let mut c = vec![0.0_f32; 2];
        ft_post_perspective_fwd_cpu(&stm, &nstm, &bias, &mut c, 1, 2, 1.0);
        // ri=0 (stm pair0): xa = NaN → crelu(NaN) = NaN (if-else passes through) → NaN*... = NaN
        assert!(c[0].is_nan());
    }

    #[test]
    fn empty_batch_is_noop() {
        let mut c: Vec<f32> = vec![];
        ft_post_perspective_fwd_cpu(&[], &[], &[0.0; 4], &mut c, 0, 4, 1.0);
        assert!(c.is_empty());
        let mut g: Vec<f32> = vec![];
        let mut gb = vec![1.0_f32; 4];
        ft_post_perspective_grad_cpu(&[], &[], &[0.0; 4], &mut g, &mut gb, 0, 4, 0, 4, 1.0);
        assert!(g.is_empty());
        assert_eq!(gb, vec![1.0_f32; 4]);
    }
}
