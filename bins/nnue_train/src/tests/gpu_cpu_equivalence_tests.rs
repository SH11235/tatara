//! GPU ↔ CPU reference 数値同等性テスト。
//!
//! 本 module は **GPU 必須**。crate 内の `#[cfg(test)] mod` として置くことで crate
//! root (`main.rs`) の `#[kernel]` 群へ `crate::*` 経由で path 解決できる (package
//! root の `tests/*.rs` integration test crate からは bin の `#[kernel]` に届かない)。
//! `nnue-trainer` は workspace `--exclude` で CI から外しているので CI には影響
//! しないが、typecheck は通す必要あり (`cargo test -p nnue-trainer --release --no-run`)。
//!
//! 走らせる:
//!
//! ```bash
//! cd bins/nnue_train && CUDA_OXIDE_TARGET=sm_75 cargo-oxide build
//! cd ../.. && cargo test -p nnue-trainer --release -- --test-threads=1
//! ```
//!
//! 各テストは小規模 batch (b = 3〜4) で GPU kernel を launch → download → reference
//! `gpu_kernels::{layerstack, pointwise, sparse}::*_cpu` と比較。`-1` padding
//! (sparse index / bucket_idx)、全 9 bucket、CReLU 境界値 (ちょうど 0.0 / 1.0 / 負)、
//! NaN 伝搬を含む。tolerance: forward / gradient 1e-5、整数/index 出力は完全一致。
//!
//! kernel ↔ CPU ref 対応表は `gpu_kernels` 各 module の doc 参照。

use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, cuda_launch};

use crate::*;
use crate::{arch::*, kernel_module::*, trainer_common::*};

use gpu_kernels::layerstack::psqt::{psqt_diff_sparse_bwd_cpu, psqt_diff_sparse_fwd_inplace_cpu};
use gpu_kernels::layerstack::{
    abs_pow2_scale::{abs_pow2_scale_fwd_cpu, abs_pow2_scale_grad_cpu},
    concat_l1sqr_main::{concat_l1sqr_main_fwd_cpu, concat_l1sqr_main_grad_cpu},
    crelu::{crelu_fwd_cpu, crelu_grad_cpu},
    dense_mm::{bias_grad_cpu, dense_mm_bwd_input_cpu, dense_mm_bwd_weight_cpu, dense_mm_fwd_cpu},
    dense_mm_bucket::{
        bias_grad_bucket_cpu, dense_mm_bwd_input_bucket_cpu, dense_mm_bwd_weight_bucket_cpu,
        dense_mm_fwd_bucket_cpu,
    },
    elementwise::elementwise_add_cpu,
    ft_post_perspective::{ft_post_perspective_fwd_cpu, ft_post_perspective_grad_cpu},
    slice2d::{slice_extract_2d_cpu, slice_scatter_2d_cpu},
};
use gpu_kernels::pointwise::loss_wdl::loss_wdl_cpu;
use gpu_kernels::pointwise::loss_wrm::loss_wrm_cpu;
use gpu_kernels::pointwise::norm_loss::{norm_loss_apply_cpu, norm_loss_compute_norms_cpu};
use gpu_kernels::pointwise::radam_step::{radam_compute_step_size_denom, radam_step_cpu};
use gpu_kernels::pointwise::ranger_step::{ranger_lookahead_lerp_cpu, ranger_step_cpu};
use gpu_kernels::pointwise::screlu_fwd::screlu_fwd_cpu;
use gpu_kernels::sparse::ft_factorize::{
    FT_FACTORIZE_BASE, FT_FACTORIZE_PER_EFFECT_BUCKET, FT_FACTORIZE_POOL_EFFECT_BUCKETS,
    FtFactorizeLayout, ft_fold_virtual_cpu, ft_reduce_virtual_grad_cpu,
};
use gpu_kernels::sparse::sparse_ft_backward::sparse_ft_backward_cpu;
use gpu_kernels::sparse::sparse_ft_forward::sparse_ft_forward_cpu;
use nnue_train::optimizer::OptimizerKind;

/// forward / gradient の f32 tolerance。atomic reduce (`fetch_add`) で加算順序が
/// GPU↔CPU で異なる出力向け (`assert_close_rel`)。順序差は和の項数に比例し得るため
/// 広め (1e-5 ≈ 84 ULP) を許す。
const TOL: f32 = 1e-5;

/// per-row independent (加算順保持) な matmul kernel の GPU↔CPU 相対 tolerance。
/// この経路は加算順序が一致するので drift 源は世代間 FMA 丸めのみ (~1 ULP) で、
/// atomic-reduce の順序差より桁違いに小さい。回帰検出力を保つため `TOL` (≈84 ULP)
/// より厳しい 1e-6 (≈8 ULP、実測 drift の ~8 倍) を使う。
const TOL_FMA: f32 = 1e-6;

// L1 系次元は `--l1` で runtime 可変だが、本 module の kernel は出力幅を runtime arg
// で受けるため、ここでは既定次元 (`DEFAULT_L1_OUT`) に固定した test fixture を使う。
const L1_OUT: usize = DEFAULT_L1_OUT;
const L1_EFFECTIVE: usize = L1_OUT - L1_SKIP;
const L2_IN: usize = L1_EFFECTIVE * 2;

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

#[test]
fn launch_error_includes_kernel_name() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let x_dev = DeviceBuffer::from_host(&stream, &[0.0_f32])?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let error = unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: crelu_fwd,
            stream: stream,
            module: module,
            config: LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (2048, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [slice(x_dev), slice_mut(y_dev), 1_u32]
        }
    }
    .unwrap_err();
    assert!(
        error.to_string().contains("crelu_fwd"),
        "kernel name is missing from error: {error}"
    );
    Ok(())
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

/// `assert_close` の relative-tolerance 版。GPU↔CPU の f32 出力には和の大きさに
/// 比例した round-off drift が出るため `|gpu - cpu| <= tol * (1 + |cpu|)` で判定する。
/// drift の発生源は 2 つ: (1) atomic reduce (`fetch_add`) で複数 thread が 1 cell に
/// 加算する出力は加算順序が GPU と CPU で異なる、(2) per-row independent (加算順保持) な
/// kernel でも、同一 PTX を JIT した SASS の FMA 丸めが GPU 世代ごとに ~1 ULP 異なる
/// (bit-exact 一致は開発・検証した特定の GPU 世代でのみ成立する)。
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: crelu_fwd, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice_mut(y_dev), n as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: crelu_grad, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice(dy_dev), slice_mut(dx_dev), n as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: screlu_fwd, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice_mut(y_dev), n as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: abs_pow2_scale_fwd, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice_mut(y_dev), scale, n as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: abs_pow2_scale_grad, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(x_dev), slice(dy_dev), slice_mut(dx_dev), scale, n as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: elementwise_add, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), n as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: slice_extract_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_EFFECTIVE),
            args: [slice(src_dev), slice_mut(main_dev),
                   batch as u32, L1_OUT as u32, 0_u32, L1_EFFECTIVE as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: slice_extract_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_SKIP),
            args: [slice(src_dev), slice_mut(skip_dev),
                   batch as u32, L1_OUT as u32, L1_EFFECTIVE as u32, L1_SKIP as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: slice_scatter_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_EFFECTIVE),
            args: [slice(main_dev), slice_mut(total_dev),
                   batch as u32, L1_EFFECTIVE as u32, L1_OUT as u32, 0_u32]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: slice_scatter_2d, stream: stream, module: module,
            config: cfg_1d(batch * L1_SKIP),
            args: [slice(skip_dev), slice_mut(total_dev),
                   batch as u32, L1_SKIP as u32, L1_OUT as u32, L1_EFFECTIVE as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: concat_l1sqr_main_fwd, stream: stream, module: module,
            config: cfg_1d(batch * L2_IN),
            args: [slice(a_dev), slice(b_dev), slice_mut(out_dev),
                   batch as u32, L1_EFFECTIVE as u32, L1_EFFECTIVE as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: concat_l1sqr_main_grad, stream: stream, module: module,
            config: cfg_1d(batch * L1_EFFECTIVE),
            args: [slice(dout_dev), slice_mut(da_dev), slice_mut(db_dev),
                   batch as u32, L1_EFFECTIVE as u32]
        }
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
// L1f 実 shape: in_dim=ft_out は重いので、ここは小さい shape で
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: dense_mm_fwd, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice_mut(y_dev),
                   batch as u32, in_dim as u32, out_dim as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: dense_mm_bwd_input, stream: stream, module: module, config: cfg_1d(batch * in_dim),
            args: [slice(dy_dev), slice(w_dev), slice_mut(dx_dev),
                   batch as u32, in_dim as u32, out_dim as u32]
        }
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: dense_mm_bwd_weight, stream: stream, module: module, config: cfg_1d(in_dim * out_dim),
            args: [slice(x_dev), slice(dy_dev), slice_mut(dw_dev),
                   batch as u32, in_dim as u32, out_dim as u32]
        }
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
    // out_dim は kernel 内の 16 幅 out-tile loop で消化される reduction 軸。既定の 16 に
    // 加え、16 の倍数 (32) と非倍数 (24 / 8 / 17) を検証して out-tile 化を網羅する。
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 16_usize, 16_usize),
        (32, 64, 32),
        (64, 96, 24),
        (16, 32, 8),
        (32, 48, 17),
    ] {
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_input_tiled, stream: stream, module: module,
                config: config,
                args: [slice(dy_dev), slice(w_dev), slice_mut(dx_dev),
                       batch as u32, in_dim as u32, out_dim as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("dense_mm_bwd_input_tiled b={batch} in={in_dim}"),
            &dx_dev.to_host_vec(&stream)?,
            &dx_cpu,
            TOL_FMA,
        );
    }
    Ok(())
}

/// `CublasHandle::sgemm_x_yt_rowmajor` (row-major `C[m,n] = X[m,k] @ Y[n,k]^T`) が
/// `dense_mm_bwd_input_cpu` と一致する。この helper は tf32 経路の L1f input backward
/// (m=batch, n=ft_out, k=l1_out で reduce 軸 k=16 が細い) と per-bucket L1 forward
/// (n=l1_out=16 が細い、k=ft_out) の双方を計算するため、両 shape regime を張る。
/// FP32 handle (`CUBLAS_DEFAULT_MATH`) は cuBLAS の K 分割による FMA 順序差のみなので
/// tight、TF32 handle (`CUBLAS_TF32_TENSOR_OP_MATH`) は入力の仮数 10-bit cast 分だけ
/// 緩める。入力を正値に寄せて桁落ちを避け、TF32 の相対 tolerance を安定させる。
#[test]
fn cublas_sgemm_x_yt_rowmajor_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, _module, stream) = open_module()?;
    // (m = batch, n = in_dim, k = out_dim/reduce)
    for &(m, n, k) in &[
        (64_usize, 128_usize, 16_usize), // L1f input-bwd regime: k=16 が細い
        (128, 256, 16),
        (64, 16, 128), // per-bucket L1-fwd regime: n=16 が細い
        (128, 16, 512),
        (48, 64, 32),
    ] {
        let x: Vec<f32> = deterministic_floats(m * k, 0.3)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect();
        let y: Vec<f32> = deterministic_floats(n * k, 0.7)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect();
        let mut c_cpu = vec![0.0_f32; m * n];
        dense_mm_bwd_input_cpu(&x, &y, &mut c_cpu, m, n, k);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let y_dev = DeviceBuffer::from_host(&stream, &y)?;
        for &(enable_tf32, tol) in &[(false, 1e-4_f32), (true, 5e-3_f32)] {
            let handle = CublasHandle::new(&stream, enable_tf32)?;
            let c_dev = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
            // SAFETY: device buffer 長は仕様分 (x=m*k, y=n*k, c=m*n)、handle は stream に
            // bind 済で同 stream 内 in-order 実行、beta=0 overwrite。
            unsafe {
                handle.sgemm_x_yt_rowmajor(
                    m as i32,
                    n as i32,
                    k as i32,
                    x_dev.cu_deviceptr() as *const f32,
                    y_dev.cu_deviceptr() as *const f32,
                    c_dev.cu_deviceptr() as *mut f32,
                )?;
            }
            stream.synchronize()?;
            assert_close_rel(
                &format!("sgemm_x_yt tf32={enable_tf32} m={m} n={n} k={k}"),
                &c_dev.to_host_vec(&stream)?,
                &c_cpu,
                tol,
            );
        }
    }
    Ok(())
}

/// `CublasHandle::sgemm_xt_y_rowmajor` (row-major `C[m,n] = X^T @ Y`、X[k,m]/Y[k,n]/C[m,n]) が
/// `dense_mm_bwd_weight_cpu` (`dw[i][o] = Σ_b x[b][i] * dy[b][o]`) と一致する。tf32 経路の
/// L1 weight backward (m=l1_out, n=ft_out, k=bucket の row 数) と L1f weight backward
/// (m=ft_out, n=l1_out, k=batch) がこの helper を使う。tolerance 方針は
/// [`cublas_sgemm_x_yt_rowmajor_matches_cpu`] と同一。
#[test]
fn cublas_sgemm_xt_y_rowmajor_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, _module, stream) = open_module()?;
    // (m = in_dim, n = out_dim, k = batch/reduce)
    for &(m, n, k) in &[
        (16_usize, 128_usize, 256_usize), // L1 wgrad regime: m=l1_out=16
        (128, 16, 256),                   // L1f wgrad regime: n=l1_out=16
        (64, 96, 512),
        (32, 48, 128),
    ] {
        let x: Vec<f32> = deterministic_floats(k * m, 0.3)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect();
        let y: Vec<f32> = deterministic_floats(k * n, 0.7)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect();
        let mut c_cpu = vec![0.0_f32; m * n];
        dense_mm_bwd_weight_cpu(&x, &y, &mut c_cpu, k, m, n);

        let x_dev = DeviceBuffer::from_host(&stream, &x)?;
        let y_dev = DeviceBuffer::from_host(&stream, &y)?;
        for &(enable_tf32, tol) in &[(false, 1e-4_f32), (true, 5e-3_f32)] {
            let handle = CublasHandle::new(&stream, enable_tf32)?;
            let c_dev = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
            // SAFETY: device buffer 長は仕様分 (x=k*m, y=k*n, c=m*n)、handle は stream に
            // bind 済で同 stream 内 in-order 実行、beta=0 overwrite。
            unsafe {
                handle.sgemm_xt_y_rowmajor(
                    m as i32,
                    n as i32,
                    k as i32,
                    x_dev.cu_deviceptr() as *const f32,
                    y_dev.cu_deviceptr() as *const f32,
                    c_dev.cu_deviceptr() as *mut f32,
                )?;
            }
            stream.synchronize()?;
            assert_close_rel(
                &format!("sgemm_xt_y tf32={enable_tf32} m={m} n={n} k={k}"),
                &c_dev.to_host_vec(&stream)?,
                &c_cpu,
                tol,
            );
        }
    }
    Ok(())
}

/// `CublasHandle::sgemm_fwd_rowmajor` (row-major `C[m,n] = A @ B`、A[m,k]/B[k,n]/C[m,n]) が
/// `dense_mm_fwd_cpu` (bias 0、`y[b][o] = Σ_k x[b][k] * w[k][o]`) と一致する。tf32 経路の
/// L1 input backward (m=bucket の row 数, n=ft_out, k=l1_out) と L1f forward (m=batch,
/// n=l1_out, k=ft_out) がこの helper を使う。tolerance 方針は
/// [`cublas_sgemm_x_yt_rowmajor_matches_cpu`] と同一。
#[test]
fn cublas_sgemm_fwd_rowmajor_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, _module, stream) = open_module()?;
    // (m = batch, n = out_dim, k = in_dim/reduce)
    for &(m, n, k) in &[
        (64_usize, 128_usize, 16_usize), // L1 input-bwd regime: k=l1_out=16
        (128, 16, 256),                  // L1f forward regime: n=l1_out=16
        (64, 96, 48),
        (32, 64, 32),
    ] {
        let a: Vec<f32> = deterministic_floats(m * k, 0.3)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect();
        let bmat: Vec<f32> = deterministic_floats(k * n, 0.7)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect();
        let zero_bias = vec![0.0_f32; n];
        let mut c_cpu = vec![0.0_f32; m * n];
        dense_mm_fwd_cpu(&a, &bmat, &zero_bias, &mut c_cpu, m, k, n);

        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &bmat)?;
        for &(enable_tf32, tol) in &[(false, 1e-4_f32), (true, 5e-3_f32)] {
            let handle = CublasHandle::new(&stream, enable_tf32)?;
            let c_dev = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
            // SAFETY: device buffer 長は仕様分 (a=m*k, b=k*n, c=m*n)、handle は stream に
            // bind 済で同 stream 内 in-order 実行、beta=0 overwrite。
            unsafe {
                handle.sgemm_fwd_rowmajor(
                    m as i32,
                    n as i32,
                    k as i32,
                    a_dev.cu_deviceptr() as *const f32,
                    b_dev.cu_deviceptr() as *const f32,
                    c_dev.cu_deviceptr() as *mut f32,
                )?;
            }
            stream.synchronize()?;
            assert_close_rel(
                &format!("sgemm_fwd tf32={enable_tf32} m={m} n={n} k={k}"),
                &c_dev.to_host_vec(&stream)?,
                &c_cpu,
                tol,
            );
        }
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_weight_tiled, stream: stream, module: module, config: config,
                args: [slice(x_dev), slice(dy_dev), slice_mut(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("dense_mm_bwd_weight_tiled b={batch} in={in_dim}"),
            &dw_dev.to_host_vec(&stream)?,
            &dw_cpu,
            TOL_FMA,
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: bias_grad, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(dy_dev), slice(gb_dev), batch as u32, out_dim as u32]
        }
    }?;
    stream.synchronize()?;
    // atomic fetch_add で reduce されるため relative tol (grad_bias と同様)。
    assert_close_rel("bias_grad", &gb_dev.to_host_vec(&stream)?, &gb_cpu, TOL);
    Ok(())
}

/// `simple_bias_grad_dual` (2D-grid per-output tile reduction) が `Σ_b (stm + nstm)` の CPU
/// 参照と一致する。整数値 dft で reduction を exact 化し、items > batch (1 block) / batch 倍数 /
/// 末尾 partial block / ft_dim 16・32・256・512・1024 (block 上限境界)・1536 (block_dim=1024、
/// grid.y=2 で末尾 output tile が partial)・2048 (block_dim=1024、grid.y=2 で full output tile)
/// を網羅。launch は trainer と同じ `block_dim = min(ft, 1024)`・`grid.y = ceil(ft/block_dim)`。
#[test]
fn simple_bias_grad_dual_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for &(batch, ft, items) in &[
        (5usize, 16u32, 8u32),
        (130, 32, 64),
        (128, 256, 64),
        (300, 256, 64),
        (256, 512, 64),
        (1024, 1024, 256),
        (200, 1536, 64),
        (300, 2048, 64),
    ] {
        let n = batch * ft as usize;
        // 整数値 (f32/f16 で exact) なので和の順序に依らず一致する。
        let stm: Vec<f32> = (0..n).map(|i| ((i % 7) as i32 - 3) as f32).collect();
        let nstm: Vec<f32> = (0..n).map(|i| ((i % 5) as i32 - 2) as f32).collect();
        let mut gb_cpu = vec![0.0_f32; ft as usize];
        for b in 0..batch {
            let row = &stm[b * ft as usize..(b + 1) * ft as usize];
            let nrow = &nstm[b * ft as usize..(b + 1) * ft as usize];
            for (g, (s, n)) in gb_cpu.iter_mut().zip(row.iter().zip(nrow)) {
                *g += *s + *n;
            }
        }
        let stm_dev = DeviceBuffer::from_host(&stream, &stm)?;
        let nstm_dev = DeviceBuffer::from_host(&stream, &nstm)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, ft as usize)?;
        let blocks = (batch as u32).div_ceil(items);
        let block_dim = ft.min(1024);
        let out_tiles = ft.div_ceil(block_dim);
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: simple_bias_grad_dual, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (blocks, out_tiles, 1), block_dim: (block_dim, 1, 1), shared_mem_bytes: 0 },
                args: [slice(stm_dev), slice(nstm_dev), slice(gb_dev), batch as u32, ft, items]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("simple_bias_grad_dual b={batch} ft={ft} items={items}"),
            &gb_dev.to_host_vec(&stream)?,
            &gb_cpu,
            TOL,
        );
    }
    Ok(())
}

/// `simple_bias_grad_dual_fp16` (FP16 入力 + `dft_inv_scale`) が CPU 参照と一致する。
/// 整数値 dft × scale=0.5 (f16/f32 で exact) で reduction を exact 化。ft_dim 1536 (partial
/// output tile) / 2048 で 2D-grid (`block_dim = min(ft, 1024)`・`grid.y > 1`) 経路も網羅。
#[test]
fn simple_bias_grad_dual_fp16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let scale = 0.5_f32;
    for &(batch, ft, items) in &[
        (5usize, 16u32, 8u32),
        (130, 256, 64),
        (128, 256, 64),
        (256, 512, 64),
        (200, 1536, 64),
        (300, 2048, 64),
    ] {
        let n = batch * ft as usize;
        let stm_f: Vec<f32> = (0..n).map(|i| ((i % 7) as i32 - 3) as f32).collect();
        let nstm_f: Vec<f32> = (0..n).map(|i| ((i % 5) as i32 - 2) as f32).collect();
        let (stm_h, stm_rt) = quantize_f16(&stm_f);
        let (nstm_h, nstm_rt) = quantize_f16(&nstm_f);
        let mut gb_cpu = vec![0.0_f32; ft as usize];
        for b in 0..batch {
            let row = &stm_rt[b * ft as usize..(b + 1) * ft as usize];
            let nrow = &nstm_rt[b * ft as usize..(b + 1) * ft as usize];
            for (g, (s, n)) in gb_cpu.iter_mut().zip(row.iter().zip(nrow)) {
                *g += *s * scale + *n * scale;
            }
        }
        let stm_dev = DeviceBuffer::from_host(&stream, &stm_h)?;
        let nstm_dev = DeviceBuffer::from_host(&stream, &nstm_h)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, ft as usize)?;
        let blocks = (batch as u32).div_ceil(items);
        let block_dim = ft.min(1024);
        let out_tiles = ft.div_ceil(block_dim);
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: simple_bias_grad_dual_fp16, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (blocks, out_tiles, 1), block_dim: (block_dim, 1, 1), shared_mem_bytes: 0 },
                args: [slice(stm_dev), slice(nstm_dev), slice(gb_dev), batch as u32, ft, scale, items]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("simple_bias_grad_dual_fp16 b={batch} ft={ft} items={items}"),
            &gb_dev.to_host_vec(&stream)?,
            &gb_cpu,
            TOL,
        );
    }
    Ok(())
}

/// FT bias grad の 2D-grid 化で ft_out > 1024 の CReLU / SCReLU が `SimpleGpuTrainer::new` で
/// reject されず、trainer の backward 経路 (`simple_bias_grad_dual[_fp16]` の grid.y > 1 launch
/// を含む) が step 後に finite な weight を生成することを確認する。ft_out=1536 は
/// `block_dim = min(ft, 1024) = 1024`・`grid.y = ceil(1536/1024) = 2` で末尾 output tile が
/// partial (oi 1024..1535 valid、1536..2047 padding) になる boundary。CReLU/SCReLU ×
/// FP32/FP16-out 経路を網羅する。
#[test]
fn simple_trainer_ft_out_gt_1024_steps() -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let ft_out = 1536_usize;
    for activation in [SimpleActivation::CReLU, SimpleActivation::SCReLU] {
        for &(ft_fp16, ft_fp16_out) in &[(false, false), (true, true)] {
            let id = SimpleId {
                feature_set: FeatureSet::HalfKaHmMerged.spec(),
                activation,
                ft_out,
                l1_out: 32,
                l2_out: 32,
            };
            let mut trainer = SimpleGpuTrainer::new(
                &ctx,
                SMOKE_BATCH,
                id,
                OptimizerKind::Ranger,
                1e-7,
                None,
                16,
                PrecisionFlags {
                    ft_fp16,
                    ft_fp16_out,
                    fp16_opt_state: false,
                    tf32: false,
                },
                &init,
            )?;
            // smoke_dummy は target=0.5 近傍で学習信号が無いため score / wdl を動かす
            // (backward が走り finite な grad を出すことを見るので値は任意の非ゼロでよい)。
            let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
            for s in batch.score.iter_mut() {
                *s = 200.0;
            }
            for w in batch.wdl.iter_mut() {
                *w = 0.8;
            }
            // step の戻り値 loss は更新前 weight の forward 値なので、backward (2D bias grad
            // launch) が NaN/Inf を出してもこれ単体では捕捉できない。optimizer 適用後に全
            // weight (2D grad で更新される ft_b を含む) が finite であることで backward 出力の
            // 健全性を確認する。
            let loss = trainer.step(&batch.as_ref(), 1e-1, 0.0, SMOKE_LOSS_SIGMOID)?;
            assert!(
                loss.is_finite(),
                "ft_out={ft_out} {} ft_fp16_out={ft_fp16_out}: forward loss {loss} not finite",
                activation.canonical_name(),
            );
            trainer.assert_all_weights_finite().map_err(|e| {
                format!(
                    "ft_out={ft_out} {} ft_fp16_out={ft_fp16_out}: non-finite weight after step: {e}",
                    activation.canonical_name(),
                )
            })?;
        }
    }
    Ok(())
}

/// `dense_bias_grad_tiled` (grid-stride per-column register 累積 + shared-mem tree reduction)
/// が `Σ_b dy[b][oi]` の CPU 参照と一致する。dy を整数値かつ全部分和が 2^24 未満 (各
/// |dy| <= 6) に収め、f32 加算を exact・順序非依存にして **bit 一致** で検証する。
/// grid-stride の partial (grid*R が batch を割り切らない) / idle thread (grid*R > batch) /
/// 単一 block grid-stride / R=1 (reduction loop skip) ・2・4・8・16・256 / out_dim
/// 1・16・32・64・128・256 (block_dim 上限) を網羅。末尾は本番 helper `cfg_dense_bias_grad`
/// の config でも一致することを確認する。
#[test]
fn dense_bias_grad_tiled_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (ctx, module, stream) = open_module()?;
    let occ = DeviceOccupancy::query(&ctx)?;
    // (batch, out_dim, grid, R): block_dim = R * out_dim、R は 2 冪。
    for &(batch, out_dim, grid, r) in &[
        (5usize, 1u32, 2u32, 4u32), // idle threads (grid*R=8 > batch)、out=1
        (1000, 1, 1, 256),          // 単一 block grid-stride、out=1
        (1000, 1, 7, 256),          // idle threads、out=1 reduction
        (5, 16, 1, 8),              // 小 batch、out=16
        (300, 16, 4, 16),           // R=16、partial
        (130, 32, 3, 8),            // partial (130 % (grid*R=24) != 0)、out=32
        (256, 64, 5, 4),            // out=64、R=4
        (300, 128, 4, 2),           // out=128、R=2 (tree reduction 1 step)
        (513, 256, 3, 1),           // out=256、R=1 (reduction loop skip、block_dim 上限)
        (4096, 32, 240, 8),         // 多 block、out=32
        (65536, 1, 240, 256),       // 本番 L3 shape
        (65536, 32, 240, 8),        // 本番 L1/L2 shape
    ] {
        let n = batch * out_dim as usize;
        let dy: Vec<f32> = (0..n).map(|i| ((i % 13) as i32 - 6) as f32).collect();
        let mut gb_cpu = vec![0.0_f32; out_dim as usize];
        bias_grad_cpu(&dy, &mut gb_cpu, batch, out_dim as usize);

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, out_dim as usize)?;
        let block = r * out_dim;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_bias_grad_tiled, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 },
                args: [slice(dy_dev), slice(gb_dev), batch as u32, out_dim]
            }
        }?;
        stream.synchronize()?;
        assert_eq!(
            gb_dev.to_host_vec(&stream)?,
            gb_cpu,
            "dense_bias_grad_tiled b={batch} out={out_dim} grid={grid} r={r}"
        );
    }

    for &(batch, out_dim) in &[(65536u32, 1u32), (65536, 32), (4096, 32), (1024, 256)] {
        let n = batch as usize * out_dim as usize;
        let dy: Vec<f32> = (0..n).map(|i| ((i % 13) as i32 - 6) as f32).collect();
        let mut gb_cpu = vec![0.0_f32; out_dim as usize];
        bias_grad_cpu(&dy, &mut gb_cpu, batch as usize, out_dim as usize);
        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, out_dim as usize)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_bias_grad_tiled, stream: stream, module: module,
                config: cfg_dense_bias_grad(occ, batch, out_dim),
                args: [slice(dy_dev), slice(gb_dev), batch, out_dim]
            }
        }?;
        stream.synchronize()?;
        assert_eq!(
            gb_dev.to_host_vec(&stream)?,
            gb_cpu,
            "dense_bias_grad_tiled(helper) b={batch} out={out_dim}"
        );
    }
    Ok(())
}

/// `cfg_dense_bias_grad` の launch geometry を検証する。out_dim の全許容域 (1..=256) で
/// kernel の構造不変条件 (`block_dim <= DENSE_BIAS_GRAD_MAX_OUT`、`block_dim % out_dim == 0`、
/// `R = block_dim / out_dim` は 2 冪) を満たし、かつ grid が
/// `min(ceil(batch / R), sm_count * floor(max_threads_per_sm / block_dim))` と**厳密一致**
/// すること (固定値や定数 1 へ縮退する回帰を弾く) を、SM 数の異なる複数 device 形状で
/// 確認する。構造不変条件を破ると kernel が shared `PARTIAL` を OOB write し得る (caller は
/// out_dim > 256 で generic `bias_grad` に fall back する)。
#[test]
fn cfg_dense_bias_grad_invariants() {
    const BATCH: u32 = 65536;
    // (sm_count, max_threads_per_sm): RTX 3080 Ti / RTX 5090 (どちらも実測値) / 極小 SM。
    for &(sm_count, max_threads_per_sm) in &[(80u32, 1536u32), (170, 1536), (1, 1536)] {
        let occ = DeviceOccupancy::from_counts(sm_count, max_threads_per_sm);
        for out_dim in 1..=DENSE_BIAS_GRAD_MAX_OUT {
            let cfg = cfg_dense_bias_grad(occ, BATCH, out_dim);
            let block = cfg.block_dim.0;
            let grid = cfg.grid_dim.0;
            assert!(
                block <= DENSE_BIAS_GRAD_MAX_OUT,
                "out_dim={out_dim} block={block} exceeds shared PARTIAL capacity"
            );
            assert_eq!(
                block % out_dim,
                0,
                "out_dim={out_dim} block={block} not divisible"
            );
            let r = block / out_dim;
            assert!(
                r.is_power_of_two(),
                "out_dim={out_dim} R={r} not power of two"
            );
            // R は floor(MAX/out_dim) を超えない最大 2 冪であること (R=1 へ縮退して occupancy を
            // 落とす実装を弾く)。cap は out_dim in 1..=256 で >= 1。
            let cap = DENSE_BIAS_GRAD_MAX_OUT / out_dim;
            assert!(
                r <= cap && 2 * r > cap,
                "out_dim={out_dim} R={r} not largest pow2 <= {cap}"
            );
            // grid は formula 値と厳密一致する (定数や SM 非依存の固定値へ縮退しない
            // ことを保証)。SM 占有を埋める cap と batch 被覆 cap の小さい方。oracle は
            // 本番 `DeviceOccupancy::fill_blocks` と同じ境界条件 (block.max(1) / per_sm.max(1) /
            // saturating_mul / 全体 max(1)) で計算し、`clamp(1, fill)` が `min > max` で panic
            // しない・overflow しないことを保証する。
            let per_sm = (max_threads_per_sm / block.max(1)).max(1);
            let fill = sm_count.saturating_mul(per_sm).max(1);
            let expected = BATCH.div_ceil(r).clamp(1, fill);
            assert_eq!(
                grid, expected,
                "sm={sm_count} out_dim={out_dim} grid={grid} != expected {expected}"
            );
        }
    }

    // 文書化した代表 shape の grid を直接固定する (実機 launch geometry の回帰ガード)。
    let occ_3080ti = DeviceOccupancy::from_counts(80, 1536);
    let occ_5090 = DeviceOccupancy::from_counts(170, 1536);
    // L1/L2 (out_dim=32, block_dim=256): SM 占有 cap = sm * floor(1536/256) = sm * 6。
    // 3080 Ti = 480、5090 = 1020。
    assert_eq!(cfg_dense_bias_grad(occ_3080ti, BATCH, 32).grid_dim.0, 480);
    assert_eq!(cfg_dense_bias_grad(occ_5090, BATCH, 32).grid_dim.0, 1020);
    // L3 (out_dim=1, block_dim=256): R=256 で batch 被覆 cap = ceil(65536/256) = 256 が
    // SM cap より小さく、どちらの device も 256 (batch 律速で不変)。
    assert_eq!(cfg_dense_bias_grad(occ_3080ti, BATCH, 1).grid_dim.0, 256);
    assert_eq!(cfg_dense_bias_grad(occ_5090, BATCH, 1).grid_dim.0, 256);
    // SM 数が増えれば cap も比例して上がる (特定 GPU 固定値へ縮退しないことの明示確認)。
    assert!(
        cfg_dense_bias_grad(occ_5090, BATCH, 32).grid_dim.0
            > cfg_dense_bias_grad(occ_3080ti, BATCH, 32).grid_dim.0,
        "larger SM count must raise the grid cap"
    );
}

/// `bias_grad_shared_l1f` (block-shared reduce 版) が `bias_grad_cpu` と reduction
/// tolerance 内で一致することを確認。out_dim (= l1_out) を 16 / 16 倍数 / 非倍数 /
/// 上限 256 で網羅する。
#[test]
fn bias_grad_shared_l1f_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for &(batch, out_dim) in &[
        (5_usize, 16_usize),
        (37, 24),
        (128, 32),
        (64, 8),
        (51, 17),
        (40, 256),
    ] {
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.07 - 1.2)
            .collect();
        let mut gb_cpu = vec![0.0_f32; out_dim];
        bias_grad_cpu(&dy, &mut gb_cpu, batch, out_dim);

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, out_dim)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: bias_grad_shared_l1f, stream: stream, module: module,
                config: cfg_1d(batch * out_dim),
                args: [slice(dy_dev), slice(gb_dev), batch as u32, out_dim as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("bias_grad_shared_l1f b={batch} out={out_dim}"),
            &gb_dev.to_host_vec(&stream)?,
            &gb_cpu,
            TOL,
        );
    }
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
    let nb = DEFAULT_NUM_BUCKETS;
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: dense_mm_fwd_bucket, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice(bidx_dev), slice_mut(y_dev),
                   batch as u32, in_dim as u32, out_dim as u32, nb as u32]
        }
    }?;
    stream.synchronize()?;
    assert_close_rel(
        "dense_mm_fwd_bucket",
        &y_dev.to_host_vec(&stream)?,
        &y_cpu,
        TOL_FMA,
    );
    Ok(())
}

#[test]
fn dense_mm_fwd_bucket_tiled_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    // tiled (L1): in_dim % 16 == 0、out_dim == 16、batch % 16 == 0、num_buckets <= 9
    for &(batch, in_dim) in &[(16_usize, 16_usize), (32, 64), (48, 96), (64, 32)] {
        let out_dim = 16_usize;
        let nb = DEFAULT_NUM_BUCKETS;
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_fwd_bucket_tiled_l1, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice(bidx_dev), slice_mut(y_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("dense_mm_fwd_bucket_tiled_l1 b={batch} in={in_dim}"),
            &y_dev.to_host_vec(&stream)?,
            &y_cpu,
            TOL_FMA,
        );
    }
    Ok(())
}

/// 16-aligned bucket sort + sorted fwd_L1 + inverse permute の合成 pipeline が
/// `dense_mm_fwd_bucket_cpu` と一致することを確認。fwd_L1 は per-row independent
/// (k=0..15 加算順保持) で sort stability に依らない。一致は相対 tolerance で判定する:
/// 同一 PTX でも GPU 世代ごとに JIT 後 SASS の FMA 丸めが ~1 ULP 異なり、bit-exact 一致は
/// 開発・検証した特定の GPU 世代でのみ成立するため。
#[test]
fn bucket_sort_fwd_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    // out_dim は 16 幅 out-tile (grid_y) で消化する。16 / 16 倍数 / 非倍数を網羅。
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 16_usize, 16_usize),
        (32, 64, 32),
        (48, 96, 24),
        (64, 32, 8),
        (32, 48, 17),
    ] {
        let nb = DEFAULT_NUM_BUCKETS;
        let padded = padded_sort_batch(batch, nb);
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

        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * in_dim),
                args: [slice(x_dev), slice(perm_dev), slice_mut(x_sorted_dev),
                       padded as u32, in_dim as u32]
            }
        }?;
        let n_out_tiles = out_dim.div_ceil(16);
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_fwd_bucket_tiled_l1_sorted, stream: stream, module: module,
                config: LaunchConfig {
                    grid_dim: ((padded / 16) as u32, n_out_tiles as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [slice(x_sorted_dev), slice(w_dev), slice(bias_dev), slice(bidx_sorted_dev),
                       slice_mut(y_sorted_dev), padded as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: inverse_permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(y_sorted_dev), slice(perm_dev), slice(y_dev),
                       padded as u32, out_dim as u32]
            }
        }?;
        stream.synchronize()?;

        assert_close_rel(
            &format!("bucket_sort_fwd_l1 b={batch} in={in_dim}"),
            &y_dev.to_host_vec(&stream)?,
            &y_cpu,
            TOL_FMA,
        );
    }
    Ok(())
}

#[test]
fn bucket_sort_bwd_input_l1_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    // out_dim は 16 幅 K-tile で消化する (16 / 16 倍数 / 非倍数を網羅)。in_dim % 16 == 0 が契約。
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 16_usize, 16_usize),
        (32, 64, 32),
        (48, 96, 24),
        (64, 32, 8),
        (32, 48, 17),
    ] {
        let nb = DEFAULT_NUM_BUCKETS;
        let padded = padded_sort_batch(batch, nb);
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

        let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
        let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
        let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, nb + 1)?;
        let perm_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
        let bidx_sorted_dev = DeviceBuffer::<i32>::zeroed(&stream, padded)?;
        let mut dy_sorted_dev = DeviceBuffer::<f32>::zeroed(&stream, padded * out_dim)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * in_dim)?;

        memset_minus_one_i32(&stream, &perm_dev)?;
        memset_minus_one_i32(&stream, &bidx_sorted_dev)?;

        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(dy_dev), slice(perm_dev), slice_mut(dy_sorted_dev),
                       padded as u32, out_dim as u32]
            }
        }?;
        // sorted で計算した dx を perm で original order に直接 scatter して dx_dev へ書く。
        // 下の CPU 参照 / 非 tiled kernel との一致で scatter 込みの値が正しいことを確認する。
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_input_bucket_tiled_sorted_scatter, stream: stream, module: module,
                config: LaunchConfig {
                    grid_dim: ((in_dim / 16) as u32, (padded / 16) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [slice(dy_sorted_dev), slice(w_dev), slice(bidx_sorted_dev), slice(perm_dev),
                       slice_mut(dx_dev), padded as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;

        let tiled_host = dx_dev.to_host_vec(&stream)?;
        assert_close_rel(
            &format!("bucket_sort_bwd_input_l1 b={batch} in={in_dim} out={out_dim}"),
            &tiled_host,
            &dx_cpu,
            TOL_FMA,
        );
        // 非 tiled `dense_mm_bwd_input_bucket` と GPU 上で bit-exact (同 dy/w・同 o-order の FMA、
        // sort/inverse-permute は値を変えない gather/scatter)。tiled 化が数値経路を変えないことを
        // CPU tolerance ではなく完全一致で裏付ける。
        let mut dx_nontiled_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * in_dim)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_input_bucket, stream: stream, module: module,
                config: cfg_1d(batch * in_dim),
                args: [slice(dy_dev), slice(w_dev), slice(bidx_dev), slice_mut(dx_nontiled_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;
        assert_eq!(
            tiled_host,
            dx_nontiled_dev.to_host_vec(&stream)?,
            "tiled vs non-tiled bit-exact b={batch} in={in_dim} out={out_dim}"
        );
    }
    Ok(())
}

#[test]
fn dense_mm_bwd_input_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let batch = 13_usize;
    let in_dim = 30_usize;
    let out_dim = 32_usize;
    let nb = DEFAULT_NUM_BUCKETS;
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket, stream: stream, module: module, config: cfg_1d(batch * in_dim),
            args: [slice(dy_dev), slice(w_dev), slice(bidx_dev), slice_mut(dx_dev),
                   batch as u32, in_dim as u32, out_dim as u32, nb as u32]
        }
    }?;
    stream.synchronize()?;
    assert_close_rel(
        "dense_mm_bwd_input_bucket",
        &dx_dev.to_host_vec(&stream)?,
        &dx_cpu,
        TOL_FMA,
    );
    Ok(())
}

#[test]
fn dense_mm_bwd_weight_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let batch = 13_usize;
    let in_dim = 30_usize;
    let out_dim = 32_usize;
    let nb = DEFAULT_NUM_BUCKETS;
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket, stream: stream, module: module,
            config: cfg_1d(nb * out_dim * in_dim),
            args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice_mut(dw_dev),
                   batch as u32, in_dim as u32, out_dim as u32, nb as u32]
        }
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
    // L2 weight backward: out_dim は L2 出力次元 (`--l2`、可変)、in_dim = l2_in
    // (= 2*(l1_out-1)、`--l1` 依存)。既定 out_dim 32 と非既定 {16, 64, 256} を
    // 各種 in_dim・非 16 倍数 out_dim と組み合わせて検証する。
    let (_ctx, module, stream) = open_module()?;
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 30_usize, 32_usize), // 既定形状
        (64, 46, 16),
        (256, 62, 64),
        (1024, 14, 256),
        (32, 96, 30), // 非 16 倍数の out_dim
    ] {
        let nb = DEFAULT_NUM_BUCKETS;
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
        let cell_blocks = (out_dim * in_dim).div_ceil(256);
        let config = LaunchConfig {
            grid_dim: (cell_blocks as u32, num_splits as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l2, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
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
    // tiled L3 (out_dim=1, num_buckets=9; in_dim は L2 出力次元 l2_out、可変)。
    // host は block_dim を in_dim に一致させる。既定 in_dim 32 と非既定
    // {16, 64, 256}・非 16 倍数の 30 を検証する。
    let (_ctx, module, stream) = open_module()?;
    for &(batch, in_dim) in &[
        (16_usize, 32_usize), // 既定形状
        (64, 16),
        (256, 64),
        (1024, 256),
        (32, 30), // 非 16 倍数の in_dim
    ] {
        let out_dim = 1_usize;
        let nb = DEFAULT_NUM_BUCKETS;
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
        let num_splits = 8_usize; // 小 grid。kernel は grid-stride で全 batch を覆う。
        // 列あたり R lane (= block_dim / in_dim) で batch reduction を並列化 (production と同じ
        // 算出)。R = in_dim を掛けて 256 を超えない最大 2 冪、block_dim = R*in_dim。
        let mut lanes = 1_usize;
        while lanes * 2 * in_dim <= 256 {
            lanes *= 2;
        }
        let block_dim = (lanes * in_dim) as u32;
        let config = LaunchConfig {
            grid_dim: (num_splits as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l3, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
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
        let nb = DEFAULT_NUM_BUCKETS;
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l1, stream: stream, module: module,
                config: config,
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice_mut(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
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
    // out_dim は 16 幅 out-tile を grid_x (in-tile と畳む) で消化する。16 / 倍数 / 非倍数を網羅。
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 16_usize, 16_usize),
        (32, 64, 32),
        (48, 96, 24),
        (64, 32, 8),
        (32, 48, 17),
    ] {
        let nb = DEFAULT_NUM_BUCKETS;
        let padded = padded_sort_batch(batch, nb);
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

        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * in_dim),
                args: [slice(x_dev), slice(perm_dev), slice_mut(x_sorted_dev),
                       padded as u32, in_dim as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(dy_dev), slice(perm_dev), slice_mut(dy_sorted_dev),
                       padded as u32, out_dim as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket_tiled_l1_sorted, stream: stream, module: module,
                config: LaunchConfig {
                    grid_dim: (((in_dim / 16) * out_dim.div_ceil(16)) as u32, 8, nb as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [slice(x_sorted_dev), slice(dy_sorted_dev), slice(offsets_dev),
                       slice(dw_dev), padded as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
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
    let nb = DEFAULT_NUM_BUCKETS;
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: bias_grad_bucket, stream: stream, module: module, config: cfg_1d(batch * out_dim),
            args: [slice(dy_dev), slice(bidx_dev), slice(gb_dev),
                   batch as u32, out_dim as u32, nb as u32]
        }
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
    // out_dim は L1 bias (= l1_out) と L2 bias (= l2_out) の双方を網羅する (ともに可変)。
    for &(batch, out_dim) in &[
        (16_usize, 16_usize), // L1 bias 既定形状
        (32, 16),
        (64, 16),
        (16, 32), // L2 bias 形状
        (48, 32),
        (32, 24), // 非 16 倍数の l1_out
        (48, 8),  // l1_out < 16
        (64, 64), // l1_out > 32
    ] {
        let nb = DEFAULT_NUM_BUCKETS;
        let padded = padded_sort_batch(batch, nb);
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

        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: count_buckets, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(counts_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_scan_aligned, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), (nb + 1) as u32, 16_u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: scatter_bucket_perm, stream: stream, module: module,
                config: cfg_1d(batch),
                args: [slice(bidx_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(perm_dev), slice(bidx_sorted_dev), batch as u32, nb as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: permute_rows_f32, stream: stream, module: module,
                config: cfg_1d(padded * out_dim),
                args: [slice(dy_dev), slice(perm_dev), slice_mut(dy_sorted_dev),
                       padded as u32, out_dim as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: bias_grad_bucket_shared_sorted, stream: stream, module: module,
                config: LaunchConfig {
                    grid_dim: ((padded / 16) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [slice(dy_sorted_dev), slice(bidx_sorted_dev), slice(gb_dev),
                       padded as u32, out_dim as u32, nb as u32]
            }
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
    let ft_dim = DEFAULT_FT_OUT; // even、half は pairwise の per-perspective 入力幅
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_fwd, stream: stream, module: module, config: cfg_1d(batch * ft_dim),
            args: [slice(stm_dev), slice(nstm_dev), slice(bias_dev), slice_mut(combined_dev),
                   batch as u32, ft_dim as u32, scale]
        }
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
    let ft_dim = DEFAULT_FT_OUT;
    let half = ft_dim / 2;
    let bias: Vec<f32> = (0..ft_dim)
        .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
        .collect();
    let scale = FT_POST_SCALE;
    // d_combined: batch × ft_dim。前半 stm pair grad、後半 nstm pair grad。
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(dc_dev), slice(stm_ft_dev), slice(bias_dev), slice_mut(dft_stm_dev),
                   slice(grad_bias_dev), batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(dc_dev), slice(nstm_ft_dev), slice(bias_dev), slice_mut(dft_nstm_dev),
                   slice(grad_bias_dev), batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale]
        }
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
    let ft_dim = DEFAULT_FT_OUT;
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(stm_ft_dev), slice(bias_dev),
                   slice_mut(dft_stm_dev), slice(grad_bias_dev),
                   batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused, stream: stream, module: module, config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(nstm_ft_dev), slice(bias_dev),
                   slice_mut(dft_nstm_dev), slice(grad_bias_dev),
                   batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale]
        }
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
    let ft_dim = DEFAULT_FT_OUT;
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
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_fwd_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim),
            args: [slice(stm_dev), slice(nstm_dev), slice(bias_dev), slice_mut(combined_dev),
                   batch as u32, ft_dim as u32, scale]
        }
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
    let ft_dim = DEFAULT_FT_OUT;
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
    // FP16 clamp 計装の累積 counter (test では cap が当たらない小 dft_scale を使うので 0)。
    let clamp_counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    // test 入力 dft は O(数十) なので、production の dft_scale (FT_DFT_FP16_BASE_SCALE
    // × batch) では overflow する。loss scaling round-trip 検証用の小さい値を使う。
    let dft_scale = 64.0_f32;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(stm_ft_dev), slice(bias_dev),
                   slice_mut(dft_stm_dev), slice(grad_bias_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale, dft_scale]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad_fused_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim / 2),
            args: [slice(da_dev), slice(db_dev), slice(nstm_ft_dev), slice(bias_dev),
                   slice_mut(dft_nstm_dev), slice(grad_bias_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale, dft_scale]
        }
    }?;
    stream.synchronize()?;
    // 小 dft_scale 経路では clamp は発火しない (`f16` 有限域 65504 を超えない)。
    assert_eq!(
        clamp_counter_dev.to_host_vec(&stream)?[0],
        0,
        "no clamp expected at small dft_scale"
    );
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

#[test]
fn ft_post_perspective_grad_fp16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    // 非 fused FP16 版 (Simple pairwise + --ft-fp16-out 経路)。単一 d_combined を
    // offset 読みする以外は fused FP16 版と同 math、CPU reference は FP32 ref。
    let (_ctx, module, stream) = open_module()?;
    let batch = 3_usize;
    let ft_dim = DEFAULT_FT_OUT;
    let half = ft_dim / 2;
    let bias: Vec<f32> = (0..ft_dim)
        .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
        .collect();
    let scale = FT_POST_SCALE;
    let d_combined: Vec<f32> = (0..batch * ft_dim)
        .map(|i| -2.0_f32 + 0.013_f32 * i as f32)
        .collect();
    let stm_ft: Vec<f32> = (0..batch * ft_dim)
        .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
        .collect();
    let nstm_ft: Vec<f32> = (0..batch * ft_dim)
        .map(|i| -1.5_f32 + 3.0_f32 * ((i * 5) % 11) as f32 / 10.0)
        .collect();
    let (stm_ft_h, stm_ft_q) = quantize_f16(&stm_ft);
    let (nstm_ft_h, nstm_ft_q) = quantize_f16(&nstm_ft);

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

    let dc_dev = DeviceBuffer::from_host(&stream, &d_combined)?;
    let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
    let stm_ft_dev = DeviceBuffer::from_host(&stream, &stm_ft_h)?;
    let nstm_ft_dev = DeviceBuffer::from_host(&stream, &nstm_ft_h)?;
    let grad_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_dim)?;
    let mut dft_stm_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * ft_dim)?;
    let mut dft_nstm_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * ft_dim)?;
    let clamp_counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    // test 入力は O(数十) なので production の dft_scale は overflow する。小さい値。
    let dft_scale = 64.0_f32;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim / 2),
            args: [slice(dc_dev), slice(stm_ft_dev), slice(bias_dev),
                   slice_mut(dft_stm_dev), slice(grad_bias_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale, dft_scale]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_post_perspective_grad_fp16, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim / 2),
            args: [slice(dc_dev), slice(nstm_ft_dev), slice(bias_dev),
                   slice_mut(dft_nstm_dev), slice(grad_bias_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale, dft_scale]
        }
    }?;
    stream.synchronize()?;
    assert_eq!(
        clamp_counter_dev.to_host_vec(&stream)?[0],
        0,
        "no clamp expected at small dft_scale"
    );
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
    assert_close_rel("ft_grad_fp16 dft_stm", &dft_stm_gpu, &dft_stm_cpu, 2e-3);
    assert_close_rel("ft_grad_fp16 dft_nstm", &dft_nstm_gpu, &dft_nstm_cpu, 2e-3);
    assert_close_rel(
        "ft_grad_fp16 grad_bias",
        &grad_bias_dev.to_host_vec(&stream)?,
        &grad_bias_cpu,
        TOL,
    );
    Ok(())
}

#[test]
fn simple_bias_act_fwd_fp16_in_screlu_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    // SCReLU FP16 FT activation forward: f16 ft_out + f32 bias → SCReLU `clamp(x,0,1)²`。
    let (_ctx, module, stream) = open_module()?;
    let batch = 3_usize;
    let ft_dim = DEFAULT_FT_OUT;
    let ft_out: Vec<f32> = (0..batch * ft_dim)
        .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
        .collect();
    let bias: Vec<f32> = (0..ft_dim)
        .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
        .collect();
    let (ft_out_h, ft_out_q) = quantize_f16(&ft_out);

    let mut acted_cpu = vec![0.0_f32; batch * ft_dim];
    for bi in 0..batch {
        for (ri, &bias_r) in bias.iter().enumerate() {
            let x = ft_out_q[bi * ft_dim + ri] + bias_r;
            let a = x.clamp(0.0, 1.0);
            acted_cpu[bi * ft_dim + ri] = a * a;
        }
    }

    let ft_out_dev = DeviceBuffer::from_host(&stream, &ft_out_h)?;
    let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
    let mut acted_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * ft_dim)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: simple_bias_act_fwd_fp16_in_screlu, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim),
            args: [slice(ft_out_dev), slice(bias_dev), slice_mut(acted_dev),
                   ft_dim as u32, 0_u32,
                   batch as u32, ft_dim as u32]
        }
    }?;
    stream.synchronize()?;
    assert_close(
        "simple_bias_act_fwd_fp16_in_screlu",
        &acted_dev.to_host_vec(&stream)?,
        &acted_cpu,
        TOL,
    );
    Ok(())
}

#[test]
fn simple_act_grad_to_fp16_screlu_with_scale_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    // SCReLU FP16 FT activation backward: SCReLU 局所微分 `2·clamp(x,0,1)` を
    // `dft_acted` に掛け、loss scaling 倍 → f16。読み戻して逆数を掛け CPU と比較。
    let (_ctx, module, stream) = open_module()?;
    let batch = 3_usize;
    let ft_dim = DEFAULT_FT_OUT;
    let ft_out: Vec<f32> = (0..batch * ft_dim)
        .map(|i| -1.0_f32 + 2.0_f32 * ((i * 7) % 13) as f32 / 12.0)
        .collect();
    let bias: Vec<f32> = (0..ft_dim)
        .map(|i| -0.5_f32 + (i % 3) as f32 * 0.6)
        .collect();
    let dft_acted: Vec<f32> = (0..batch * ft_dim)
        .map(|i| -2.0_f32 + 0.013_f32 * i as f32)
        .collect();
    let (ft_out_h, ft_out_q) = quantize_f16(&ft_out);

    let mut g_cpu = vec![0.0_f32; batch * ft_dim];
    for bi in 0..batch {
        for (ri, &bias_r) in bias.iter().enumerate() {
            let idx = bi * ft_dim + ri;
            let x = ft_out_q[idx] + bias_r;
            let a = x.clamp(0.0, 1.0);
            let dydx = if a > 0.0 && a < 1.0 { 2.0 * a } else { 0.0 };
            g_cpu[idx] = dft_acted[idx] * dydx;
        }
    }

    let ft_out_dev = DeviceBuffer::from_host(&stream, &ft_out_h)?;
    let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
    let dft_acted_dev = DeviceBuffer::from_host(&stream, &dft_acted)?;
    let mut dft_out_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * ft_dim)?;
    let clamp_counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    // test 入力 dft は O(数十) なので production の dft_scale は overflow する。小さい値。
    let dft_scale = 64.0_f32;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: simple_act_grad_to_fp16_screlu_with_scale, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim),
            args: [slice(ft_out_dev), slice(bias_dev), slice(dft_acted_dev),
                   ft_dim as u32, 0_u32,
                   slice_mut(dft_out_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, dft_scale]
        }
    }?;
    stream.synchronize()?;
    assert_eq!(
        clamp_counter_dev.to_host_vec(&stream)?[0],
        0,
        "no clamp expected at small dft_scale"
    );
    let inv = 1.0_f32 / dft_scale;
    let g_gpu: Vec<f32> = dft_out_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32 * inv)
        .collect();
    assert_close_rel("simple_act_grad_to_fp16_screlu", &g_gpu, &g_cpu, 2e-3);
    Ok(())
}

#[test]
fn simple_act_grad_to_fp16_crelu_clamp_counter_counts_overflows()
-> Result<(), Box<dyn std::error::Error>> {
    // FP16 clamp counter (`--ft-fp16-out` 経路の dft cap 監視) の単体検証。
    // 過大 `dft_scale` を渡し、grad の絶対値が非零 (`0 < x < 1` の cell) で必ず
    // overflow するように入力を仕組み、device atomic counter が cap された要素数と
    // 一致 + 2 回 launch で累積 (cumulative) することを確認する。
    let (_ctx, module, stream) = open_module()?;
    let batch = 2_usize;
    let ft_dim = DEFAULT_FT_OUT;
    // pre-activation を全 cell `0.5` にすると CReLU 指示関数 (`0 < x < 1`) が
    // 全 cell で発火し、`g = dft_acted`、`dft_scale * dft_acted` の絶対値が
    // overflow するように `dft_acted = 2.0`、`dft_scale = 1e6` を組合せる
    // (`|2e6| >> 65504` で確実に clamp)。
    let ft_out = vec![0.5_f32; batch * ft_dim];
    let bias = vec![0.0_f32; ft_dim];
    let dft_acted = vec![2.0_f32; batch * ft_dim];
    let (ft_out_h, _) = quantize_f16(&ft_out);

    let ft_out_dev = DeviceBuffer::from_host(&stream, &ft_out_h)?;
    let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
    let dft_acted_dev = DeviceBuffer::from_host(&stream, &dft_acted)?;
    let mut dft_out_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * ft_dim)?;
    let clamp_counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    let dft_scale = 1.0e6_f32;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: simple_act_grad_to_fp16_crelu_with_scale, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim),
            args: [slice(ft_out_dev), slice(bias_dev), slice(dft_acted_dev),
                   ft_dim as u32, 0_u32,
                   slice_mut(dft_out_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, dft_scale]
        }
    }?;
    stream.synchronize()?;
    let count = clamp_counter_dev.to_host_vec(&stream)?[0];
    let expected = (batch * ft_dim) as u64;
    assert_eq!(
        count, expected,
        "clamp counter should equal element count when all cells overflow"
    );
    // 2 回目 launch で counter は cumulative (= 2 × expected) になる (累積 counter)。
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: simple_act_grad_to_fp16_crelu_with_scale, stream: stream, module: module,
            config: cfg_1d(batch * ft_dim),
            args: [slice(ft_out_dev), slice(bias_dev), slice(dft_acted_dev),
                   ft_dim as u32, 0_u32,
                   slice_mut(dft_out_dev), slice(clamp_counter_dev),
                   batch as u32, ft_dim as u32, dft_scale]
        }
    }?;
    stream.synchronize()?;
    let count2 = clamp_counter_dev.to_host_vec(&stream)?[0];
    assert_eq!(
        count2,
        expected * 2,
        "counter must be cumulative across launches"
    );
    Ok(())
}

// =============================================================================
// `--num-buckets` 可変 N (N ≤ MAX_SUPPORTED_NUM_BUCKETS = 9) を GPU/CPU で
// exercise する parametrised test。既存テストは既定 N=9 で動かしているが、
// kernel は `num_buckets` を runtime 引数で受けるため N ≤ 9 で同じ correctness
// 不変条件 (`bucket_idx >= num_buckets` を silent skip、CPU と bit-equal) が
// 成立する。本セクションでは `N ∈ {2, 4, 8, 9}` の主要 N で fwd / bwd_input /
// bwd_weight / bias_grad を回し、host plumbing 経由で kernel が runtime N を
// 正しく受け取れていることを確認する。
// =============================================================================

/// `--num-buckets` で実験的に使う想定の N 値。
const NUM_BUCKETS_PARAM_VALUES: [usize; 4] = [2, 4, 8, 9];

#[test]
fn dense_mm_fwd_bucket_matches_cpu_for_each_num_buckets() -> Result<(), Box<dyn std::error::Error>>
{
    let (_ctx, module, stream) = open_module()?;
    let batch = 13_usize;
    let in_dim = 30_usize;
    let out_dim = 32_usize;
    for &nb in &NUM_BUCKETS_PARAM_VALUES {
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_fwd_bucket, stream: stream, module: module,
                config: cfg_1d(batch * out_dim),
                args: [slice(x_dev), slice(w_dev), slice(bias_dev), slice(bidx_dev),
                       slice_mut(y_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("dense_mm_fwd_bucket nb={nb}"),
            &y_dev.to_host_vec(&stream)?,
            &y_cpu,
            TOL_FMA,
        );
    }
    Ok(())
}

#[test]
fn dense_mm_bwd_input_bucket_matches_cpu_for_each_num_buckets()
-> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let batch = 13_usize;
    let in_dim = 30_usize;
    let out_dim = 32_usize;
    for &nb in &NUM_BUCKETS_PARAM_VALUES {
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_input_bucket, stream: stream, module: module,
                config: cfg_1d(batch * in_dim),
                args: [slice(dy_dev), slice(w_dev), slice(bidx_dev), slice_mut(dx_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("dense_mm_bwd_input_bucket nb={nb}"),
            &dx_dev.to_host_vec(&stream)?,
            &dx_cpu,
            TOL_FMA,
        );
    }
    Ok(())
}

#[test]
fn dense_mm_bwd_weight_bucket_matches_cpu_for_each_num_buckets()
-> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let batch = 13_usize;
    let in_dim = 30_usize;
    let out_dim = 32_usize;
    for &nb in &NUM_BUCKETS_PARAM_VALUES {
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
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_mm_bwd_weight_bucket, stream: stream, module: module,
                config: cfg_1d(nb * out_dim * in_dim),
                args: [slice(x_dev), slice(dy_dev), slice(bidx_dev), slice_mut(dw_dev),
                       batch as u32, in_dim as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close(
            &format!("dense_mm_bwd_weight_bucket nb={nb}"),
            &dw_dev.to_host_vec(&stream)?,
            &dw_cpu,
            TOL,
        );
    }
    Ok(())
}

#[test]
fn bias_grad_bucket_matches_cpu_for_each_num_buckets() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let batch = 13_usize;
    let out_dim = 32_usize;
    for &nb in &NUM_BUCKETS_PARAM_VALUES {
        let dy: Vec<f32> = (0..batch * out_dim)
            .map(|i| i as f32 * 0.017 - 0.9)
            .collect();
        let bucket_idx = bucket_idx_with_padding(batch, nb);
        let mut gb_cpu = vec![0.0_f32; nb * out_dim];
        bias_grad_bucket_cpu(&dy, &bucket_idx, &mut gb_cpu, batch, out_dim, nb);

        let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
        let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
        let gb_dev = DeviceBuffer::<f32>::zeroed(&stream, nb * out_dim)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: bias_grad_bucket, stream: stream, module: module,
                config: cfg_1d(batch * out_dim),
                args: [slice(dy_dev), slice(bidx_dev), slice(gb_dev),
                       batch as u32, out_dim as u32, nb as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("bias_grad_bucket nb={nb}"),
            &gb_dev.to_host_vec(&stream)?,
            &gb_cpu,
            TOL,
        );
    }
    Ok(())
}

// -- loss_wrm (win-rate-model loss) -------------------------------------

/// WRM loss の固定パラメータ (`SMOKE_LOSS_WRM` と同じ sigmoid 定数)。
const WRM_NNUE2SCORE: f32 = 600.0;
const WRM_IN_SCALING: f32 = 340.0;
const WRM_IN_OFFSET: f32 = 270.0;
const WRM_TARGET_OFFSET: f32 = 270.0;
const WRM_TARGET_SCALING: f32 = 380.0;

/// `loss_wrm` の default 経路 (`extended=0`、二乗誤差) が CPU reference と一致する。
#[test]
fn loss_wrm_default_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let out = vec![0.3_f32, -0.8, 2.5, -0.05];
    let score = vec![150.0_f32, -1200.0, 30.0, 5000.0];
    let wdl = vec![1.0_f32, 0.0, 0.5, 1.0];
    let b = out.len();
    let per_pos_norm = 1.0_f32 / b as f32;
    let lambda = 0.0_f32;

    let mut dl_cpu = vec![0.0_f32; b];
    let mut loss_cpu = 0.0_f64;
    loss_wrm_cpu(
        &out,
        &score,
        &wdl,
        &vec![per_pos_norm; b],
        &mut dl_cpu,
        &mut loss_cpu,
        lambda,
        WRM_NNUE2SCORE,
        WRM_IN_SCALING,
        WRM_IN_OFFSET,
        WRM_TARGET_OFFSET,
        WRM_TARGET_SCALING,
        2.0,
        0.0,
        0.0,
        0.5,
        false,
        b,
    );

    let out_dev = DeviceBuffer::from_host(&stream, &out)?;
    let score_dev = DeviceBuffer::from_host(&stream, &score)?;
    let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
    let mut dl_dev = DeviceBuffer::<f32>::zeroed(&stream, b)?;
    let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let sum_w_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: loss_wrm, stream: stream, module: module, config: cfg_1d(b),
            args: [
                slice(out_dev), slice(score_dev), slice(wdl_dev), per_pos_norm,
                slice_mut(dl_dev), slice(loss_dev), lambda,
                WRM_NNUE2SCORE, WRM_IN_SCALING, WRM_IN_OFFSET, WRM_TARGET_OFFSET, WRM_TARGET_SCALING,
                2.0_f32, 0.0_f32, 0.0_f32, 0.5_f32, slice(sum_w_dev), 0_u32, b as u32
            ]
        }
    }?;
    stream.synchronize()?;
    // libdevice exp と std exp の差で grad は ~ulp レベルずれるため relative tolerance。
    assert_close_rel(
        "loss_wrm/default grad",
        &dl_dev.to_host_vec(&stream)?,
        &dl_cpu,
        1e-4,
    );
    let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
    let diff = (loss_gpu - loss_cpu).abs();
    assert!(
        diff <= 1e-4 * (1.0 + loss_cpu.abs()),
        "loss_wrm/default loss: gpu={loss_gpu} cpu={loss_cpu} diff={diff}"
    );
    Ok(())
}

/// `loss_wrm` の extended 経路 (`extended=1`、pow_exp / qp_asymmetry / weight boost +
/// Σw 正規化) が CPU reference と一致する。`wrm_weight_sum` で Σw を先に reduce してから
/// `loss_wrm` を launch する (host の trainer と同じ 2 段構成)。`f32::powf` の libdevice
/// lowering もここで実機検証する。
#[test]
fn loss_wrm_extended_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let out = vec![0.4_f32, -1.1, 0.05, 2.0, -0.3, 1.2];
    let score = vec![300.0_f32, -700.0, 50.0, -2500.0, 1800.0, -40.0];
    let wdl = vec![1.0_f32, 0.0, 0.5, 0.0, 1.0, 0.5];
    let b = out.len();
    let per_pos_norm = 1.0_f32 / b as f32;
    let lambda = 0.0_f32;
    let pow_exp = 2.5_f32;
    let qp = 0.3_f32;
    let w1 = 1.0_f32;
    let w2 = 0.5_f32;

    let mut dl_cpu = vec![0.0_f32; b];
    let mut loss_cpu = 0.0_f64;
    loss_wrm_cpu(
        &out,
        &score,
        &wdl,
        &vec![per_pos_norm; b],
        &mut dl_cpu,
        &mut loss_cpu,
        lambda,
        WRM_NNUE2SCORE,
        WRM_IN_SCALING,
        WRM_IN_OFFSET,
        WRM_TARGET_OFFSET,
        WRM_TARGET_SCALING,
        pow_exp,
        qp,
        w1,
        w2,
        true,
        b,
    );

    let out_dev = DeviceBuffer::from_host(&stream, &out)?;
    let score_dev = DeviceBuffer::from_host(&stream, &score)?;
    let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
    let mut dl_dev = DeviceBuffer::<f32>::zeroed(&stream, b)?;
    let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let sum_w_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: wrm_weight_sum, stream: stream, module: module, config: cfg_1d(b),
            args: [
                slice(score_dev), slice(sum_w_dev), w1, w2,
                WRM_TARGET_OFFSET, WRM_TARGET_SCALING, b as u32
            ]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: loss_wrm, stream: stream, module: module, config: cfg_1d(b),
            args: [
                slice(out_dev), slice(score_dev), slice(wdl_dev), per_pos_norm,
                slice_mut(dl_dev), slice(loss_dev), lambda,
                WRM_NNUE2SCORE, WRM_IN_SCALING, WRM_IN_OFFSET, WRM_TARGET_OFFSET, WRM_TARGET_SCALING,
                pow_exp, qp, w1, w2, slice(sum_w_dev), 1_u32, b as u32
            ]
        }
    }?;
    stream.synchronize()?;
    // exp / powf の libdevice 差が sigmoid → weight → Σw → 正規化と多段で乗るため
    // relative tolerance を default 経路より少し緩める。
    assert_close_rel(
        "loss_wrm/extended grad",
        &dl_dev.to_host_vec(&stream)?,
        &dl_cpu,
        2e-4,
    );
    let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
    let diff = (loss_gpu - loss_cpu).abs();
    assert!(
        diff <= 2e-4 * (1.0 + loss_cpu.abs()),
        "loss_wrm/extended loss: gpu={loss_gpu} cpu={loss_cpu} diff={diff}"
    );
    Ok(())
}

/// `loss_wrm` の loss_acc 集約が **複数 block** (grid > 1) を跨いで正しく総和されることを確認する。
/// 各 block は block 内 reduction の結果を 1 atomic で寄与するため、ある block の総和が欠けると
/// loss は `loss/num_blocks` 規模でずれる (本テストが検出)。b=1000 で grid=4 block (block 256)、
/// 末尾 block は partial (1000 % 256 != 0、out-of-range thread は寄与 0)。grad (per-position) も
/// CPU 参照と一致することを併せて検証する (extended=0)。
#[test]
fn loss_wrm_default_multiblock_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let b = 1000usize;
    let out: Vec<f32> = (0..b).map(|i| ((i % 41) as f32 - 20.0) * 0.05).collect();
    let score: Vec<f32> = (0..b).map(|i| ((i % 401) as f32 - 200.0) * 5.0).collect();
    let wdl: Vec<f32> = (0..b).map(|i| (i % 3) as f32 * 0.5).collect();
    let per_pos_norm = 1.0_f32 / b as f32;
    let lambda = 0.0_f32;

    let mut dl_cpu = vec![0.0_f32; b];
    let mut loss_cpu = 0.0_f64;
    loss_wrm_cpu(
        &out,
        &score,
        &wdl,
        &vec![per_pos_norm; b],
        &mut dl_cpu,
        &mut loss_cpu,
        lambda,
        WRM_NNUE2SCORE,
        WRM_IN_SCALING,
        WRM_IN_OFFSET,
        WRM_TARGET_OFFSET,
        WRM_TARGET_SCALING,
        2.0,
        0.0,
        0.0,
        0.5,
        false,
        b,
    );

    let out_dev = DeviceBuffer::from_host(&stream, &out)?;
    let score_dev = DeviceBuffer::from_host(&stream, &score)?;
    let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
    let mut dl_dev = DeviceBuffer::<f32>::zeroed(&stream, b)?;
    let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let sum_w_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: loss_wrm, stream: stream, module: module, config: cfg_1d(b),
            args: [
                slice(out_dev), slice(score_dev), slice(wdl_dev), per_pos_norm,
                slice_mut(dl_dev), slice(loss_dev), lambda,
                WRM_NNUE2SCORE, WRM_IN_SCALING, WRM_IN_OFFSET, WRM_TARGET_OFFSET, WRM_TARGET_SCALING,
                2.0_f32, 0.0_f32, 0.0_f32, 0.5_f32, slice(sum_w_dev), 0_u32, b as u32
            ]
        }
    }?;
    stream.synchronize()?;
    assert_close_rel(
        "loss_wrm/multiblock grad",
        &dl_dev.to_host_vec(&stream)?,
        &dl_cpu,
        1e-4,
    );
    let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
    let diff = (loss_gpu - loss_cpu).abs();
    assert!(
        diff <= 1e-4 * (1.0 + loss_cpu.abs()),
        "loss_wrm/multiblock loss: gpu={loss_gpu} cpu={loss_cpu} diff={diff}"
    );
    Ok(())
}

/// extended 経路 (`extended=1`) の loss_acc 集約を **複数 block** で検証する。`wrm_weight_sum`
/// で Σw を先に reduce し、`loss_wrm` が `L_i·w_i·n/Σw` を block 跨ぎで総和する経路を b=1000
/// (grid 4 block、末尾 partial) で CPU 参照と比較する。
#[test]
fn loss_wrm_extended_multiblock_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let b = 1000usize;
    let out: Vec<f32> = (0..b).map(|i| ((i % 37) as f32 - 18.0) * 0.06).collect();
    let score: Vec<f32> = (0..b).map(|i| ((i % 433) as f32 - 216.0) * 6.0).collect();
    let wdl: Vec<f32> = (0..b).map(|i| (i % 3) as f32 * 0.5).collect();
    let per_pos_norm = 1.0_f32 / b as f32;
    let lambda = 0.0_f32;
    let pow_exp = 2.5_f32;
    let qp = 0.3_f32;
    let w1 = 1.0_f32;
    let w2 = 0.5_f32;

    let mut dl_cpu = vec![0.0_f32; b];
    let mut loss_cpu = 0.0_f64;
    loss_wrm_cpu(
        &out,
        &score,
        &wdl,
        &vec![per_pos_norm; b],
        &mut dl_cpu,
        &mut loss_cpu,
        lambda,
        WRM_NNUE2SCORE,
        WRM_IN_SCALING,
        WRM_IN_OFFSET,
        WRM_TARGET_OFFSET,
        WRM_TARGET_SCALING,
        pow_exp,
        qp,
        w1,
        w2,
        true,
        b,
    );

    let out_dev = DeviceBuffer::from_host(&stream, &out)?;
    let score_dev = DeviceBuffer::from_host(&stream, &score)?;
    let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
    let mut dl_dev = DeviceBuffer::<f32>::zeroed(&stream, b)?;
    let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let sum_w_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: wrm_weight_sum, stream: stream, module: module, config: cfg_1d(b),
            args: [
                slice(score_dev), slice(sum_w_dev), w1, w2,
                WRM_TARGET_OFFSET, WRM_TARGET_SCALING, b as u32
            ]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: loss_wrm, stream: stream, module: module, config: cfg_1d(b),
            args: [
                slice(out_dev), slice(score_dev), slice(wdl_dev), per_pos_norm,
                slice_mut(dl_dev), slice(loss_dev), lambda,
                WRM_NNUE2SCORE, WRM_IN_SCALING, WRM_IN_OFFSET, WRM_TARGET_OFFSET, WRM_TARGET_SCALING,
                pow_exp, qp, w1, w2, slice(sum_w_dev), 1_u32, b as u32
            ]
        }
    }?;
    stream.synchronize()?;
    assert_close_rel(
        "loss_wrm/extended-multiblock grad",
        &dl_dev.to_host_vec(&stream)?,
        &dl_cpu,
        2e-4,
    );
    let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
    let diff = (loss_gpu - loss_cpu).abs();
    assert!(
        diff <= 2e-4 * (1.0 + loss_cpu.abs()),
        "loss_wrm/extended-multiblock loss: gpu={loss_gpu} cpu={loss_cpu} diff={diff}"
    );
    Ok(())
}

// -- norm_loss ----------------------------------------------------------

/// 3 つの weight レイアウト (contiguous row / strided column / per-tensor scalar)
/// を (n_groups, group_pitch, elem_stride, group_len) で表した norm loss テスト
/// fixture。`step_impl` の norm loss 配線が渡す indexing と同じ形を踏む。
struct NormLossLayout {
    label: &'static str,
    n_groups: usize,
    group_pitch: usize,
    elem_stride: usize,
    group_len: usize,
}

fn norm_loss_layouts() -> Vec<NormLossLayout> {
    vec![
        // contiguous row [n_groups, group_len]: dense L1/L2/L3 weight 相当。
        NormLossLayout {
            label: "row",
            n_groups: 5,
            group_pitch: 8,
            elem_stride: 1,
            group_len: 8,
        },
        // strided column [group_len, n_groups]: FT / L1f weight 相当。
        NormLossLayout {
            label: "column",
            n_groups: 6,
            group_pitch: 1,
            elem_stride: 6,
            group_len: 7,
        },
        // strided column (PSQT weight 相当): psqt_w[feat*num_buckets + bucket] を
        // bucket 列ごとに ft_in 要素で reduce する。n_groups=num_buckets。
        NormLossLayout {
            label: "psqt-column",
            n_groups: 9,
            group_pitch: 1,
            elem_stride: 9,
            group_len: 11,
        },
        // per-tensor scalar: bias 相当 (テンソル全体で 1 norm)。
        NormLossLayout {
            label: "scalar",
            n_groups: 1,
            group_pitch: 0,
            elem_stride: 1,
            group_len: 23,
        },
    ]
}

#[test]
fn norm_loss_reduce_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for lo in norm_loss_layouts() {
        let total = lo.n_groups * lo.group_len;
        let w = deterministic_floats(total, 2.0);
        let mut norms_cpu = vec![0.0_f32; lo.n_groups];
        norm_loss_compute_norms_cpu(
            &w,
            &mut norms_cpu,
            lo.n_groups,
            lo.group_pitch,
            lo.elem_stride,
            lo.group_len,
        );

        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        // norms_dev は 0 init: reduce が sumsq を atomicAdd、finalize で sqrt。
        let mut norms_dev = DeviceBuffer::<f32>::zeroed(&stream, lo.n_groups)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: norm_loss_reduce, stream: stream, module: module,
                config: cfg_norm_loss_reduce(lo.n_groups, lo.group_len),
                args: [slice(w_dev), slice(norms_dev),
                       lo.n_groups as u32, lo.group_pitch as u32,
                       lo.elem_stride as u32, lo.group_len as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: norm_loss_finalize, stream: stream, module: module,
                config: cfg_1d(lo.n_groups),
                args: [slice_mut(norms_dev), lo.n_groups as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("norm_loss_reduce/{}", lo.label),
            &norms_dev.to_host_vec(&stream)?,
            &norms_cpu,
            TOL,
        );
    }
    Ok(())
}

#[test]
fn norm_loss_apply_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let factor = 1e-4_f32;
    let lr = 8.75e-4_f32;
    let eps = EPS;
    for lo in norm_loss_layouts() {
        let total = lo.n_groups * lo.group_len;
        let w = deterministic_floats(total, 2.0);
        // CPU: norm を先に計算し、その norm で apply。
        let mut norms = vec![0.0_f32; lo.n_groups];
        norm_loss_compute_norms_cpu(
            &w,
            &mut norms,
            lo.n_groups,
            lo.group_pitch,
            lo.elem_stride,
            lo.group_len,
        );
        let mut w_cpu = w.clone();
        norm_loss_apply_cpu(
            &mut w_cpu,
            &norms,
            factor,
            lr,
            eps,
            lo.n_groups,
            lo.group_pitch,
            lo.elem_stride,
            lo.group_len,
        );

        // GPU: 同じ norm を device に渡して apply のみ実行 (reduce は別 test で検証済)。
        let mut w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let norms_dev = DeviceBuffer::from_host(&stream, &norms)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: norm_loss_apply, stream: stream, module: module,
                config: cfg_1d(total),
                args: [slice_mut(w_dev), slice(norms_dev), factor, lr, eps,
                       lo.n_groups as u32, lo.group_pitch as u32,
                       lo.elem_stride as u32, lo.group_len as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close_rel(
            &format!("norm_loss_apply/{}", lo.label),
            &w_dev.to_host_vec(&stream)?,
            &w_cpu,
            TOL,
        );
    }
    Ok(())
}

/// 全零 group (norm=0 → `1/(0+eps)` の巨大負補正) を含む degenerate ケースで、
/// 対象 weight が 0 のまま (NaN/Inf 汚染しない) かつ非零 group は通常補正される
/// ことを GPU↔CPU で確認する (CPU unit test `zero_group_stays_zero` の GPU 版)。
#[test]
fn norm_loss_zero_norm_group_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (n_groups, group_len) = (3_usize, 4_usize);
    // contiguous row layout、group 1 を全零にする。
    let mut w = deterministic_floats(n_groups * group_len, 1.0);
    for x in w.iter_mut().take(2 * group_len).skip(group_len) {
        *x = 0.0;
    }
    let factor = 1e-4_f32;
    let lr = 8.75e-4_f32;
    let eps = EPS;

    let mut w_cpu = w.clone();
    let mut norms_cpu = vec![0.0_f32; n_groups];
    norm_loss_compute_norms_cpu(&w_cpu, &mut norms_cpu, n_groups, group_len, 1, group_len);
    norm_loss_apply_cpu(
        &mut w_cpu, &norms_cpu, factor, lr, eps, n_groups, group_len, 1, group_len,
    );

    let mut w_dev = DeviceBuffer::from_host(&stream, &w)?;
    let mut norms_dev = DeviceBuffer::<f32>::zeroed(&stream, n_groups)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: norm_loss_reduce, stream: stream, module: module,
            config: cfg_norm_loss_reduce(n_groups, group_len),
            args: [slice(w_dev), slice(norms_dev),
                   n_groups as u32, group_len as u32, 1_u32, group_len as u32]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: norm_loss_finalize, stream: stream, module: module,
            config: cfg_1d(n_groups),
            args: [slice_mut(norms_dev), n_groups as u32]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: norm_loss_apply, stream: stream, module: module,
            config: cfg_1d(n_groups * group_len),
            args: [slice_mut(w_dev), slice(norms_dev), factor, lr, eps,
                   n_groups as u32, group_len as u32, 1_u32, group_len as u32]
        }
    }?;
    stream.synchronize()?;
    let w_gpu = w_dev.to_host_vec(&stream)?;
    // 零 group は 0 のまま (NaN/Inf でない)。
    for &x in w_gpu.iter().take(2 * group_len).skip(group_len) {
        assert_eq!(x, 0.0, "zero group must stay exactly 0 (no NaN/Inf)");
    }
    assert_close_rel("norm_loss/zero-norm", &w_gpu, &w_cpu, TOL);
    Ok(())
}

// -- sparse FT forward / backward ----------------------------------------

/// sparse index fixture: 有効 idx (position 内重複・position 間共有あり)、`-1`
/// padding、`>= cols` の defensive skip 対象を混ぜた決定論的な列。`seed` でパターン
/// をずらす (stm / nstm で別系列を作る)。`r % 6 == 2` の枝だけが feature 0 を出し
/// (他の有効枝は `1..=cols-2`)、出現を feature 0 へ集めて高頻度 feature を作る。
/// feature `cols-1` はどの枝からも出ず出現 0 になる。inverse-index gather では
/// 前者が 4-way unroll 本体から端数 tail への同一区間内継続を、後者が空区間の
/// sum=0 書き切り経路を踏む (発火を guard する assert は pipeline テスト側)。
fn sparse_indices_fixture(batch: usize, nnz: usize, cols: usize, seed: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(batch * nnz);
    for bi in 0..batch {
        for ni in 0..nnz {
            let r = bi * 7 + ni * 3 + seed;
            let idx = match r % 6 {
                0 => -1,
                1 => (cols + r) as i32,
                2 => 0,
                _ => (1 + r % (cols - 2)) as i32,
            };
            v.push(idx);
        }
    }
    v
}

/// position ごとに実長が異なる固定幅 sparse index。先頭 2 row で
/// `nnz=0` / `nnz=max_active` の境界を作り、各 row の tail は `-1` にする。
fn sparse_indices_fixture_ragged(
    batch: usize,
    max_active: usize,
    cols: usize,
    seed: usize,
) -> (Vec<i32>, Vec<i32>) {
    let mut indices = vec![-1; batch * max_active];
    let mut nnz = vec![0; batch];
    for bi in 0..batch {
        let row_nnz = match bi {
            0 => 0,
            1 => max_active,
            _ => (bi * 3 + 1) % (max_active + 1),
        };
        nnz[bi] = row_nnz as i32;
        for ni in 0..row_nnz {
            indices[bi * max_active + ni] = ((bi * 7 + ni * 3 + seed) % cols) as i32;
        }
    }
    (indices, nnz)
}

#[test]
fn sparse_ft_forward_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    // rows は kernel 規約 (1 thread = 4 連続 row) の最小単位 4 の倍数。
    let (batch, rows, cols, max_active) = (5_usize, 8_usize, 9_usize, 6_usize);
    let weight = deterministic_floats(rows * cols, 3.0);
    let (indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, cols, 0);
    let mut out_cpu = vec![0.0_f32; batch * rows];
    sparse_ft_forward_cpu(
        &weight,
        &indices,
        &nnz,
        &mut out_cpu,
        batch,
        rows,
        cols,
        max_active,
    );

    // tail=-1 の固定幅全 slot 走査は、実長打ち切りと同じ項を同じ順序で加算する。
    let mut out_full_scan = vec![0.0_f32; batch * rows];
    sparse_ft_forward_cpu(
        &weight,
        &indices,
        &vec![max_active as i32; batch],
        &mut out_full_scan,
        batch,
        rows,
        cols,
        max_active,
    );
    assert_eq!(out_cpu, out_full_scan);
    assert_eq!(nnz[0], 0);
    assert_eq!(nnz[1], max_active as i32);

    let w_dev = DeviceBuffer::from_host(&stream, &weight)?;
    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * rows)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: sparse_ft_forward, stream: stream, module: module,
            config: cfg_1d(batch * rows / 4),
            args: [slice(w_dev), slice(idx_dev), slice(nnz_dev), slice_mut(out_dev),
                   batch as u32, rows as u32, cols as u32, max_active as u32]
        }
    }?;
    stream.synchronize()?;
    // 各出力 cell は weight の純加算を ni 順に積むだけで GPU/CPU の演算列が一致する
    // (乗算が無く fma 縮約も起きない) ため bit-exact。
    assert_close(
        "sparse_ft_forward",
        &out_dev.to_host_vec(&stream)?,
        &out_cpu,
        0.0,
    );
    Ok(())
}

#[test]
fn sparse_ft_forward_ignores_valid_indices_past_row_nnz() -> Result<(), Box<dyn std::error::Error>>
{
    let (_ctx, module, stream) = open_module()?;
    let (batch, rows, cols, max_active) = (5_usize, 8_usize, 9_usize, 6_usize);
    let weight = deterministic_floats(rows * cols, 3.25);
    let (mut indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, cols, 3);
    let mut expected = vec![0.0_f32; batch * rows];
    sparse_ft_forward_cpu(
        &weight,
        &indices,
        &nnz,
        &mut expected,
        batch,
        rows,
        cols,
        max_active,
    );

    // nnz 以降を範囲内の index で汚す。kernel が固定幅全体を走査すると結果が変わる。
    for bi in 0..batch {
        for ni in nnz[bi] as usize..max_active {
            indices[bi * max_active + ni] = ((bi + ni + 1) % cols) as i32;
        }
    }

    let w_dev = DeviceBuffer::from_host(&stream, &weight)?;
    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * rows)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: sparse_ft_forward, stream: stream, module: module,
            config: cfg_1d(batch * rows / 4),
            args: [slice(w_dev), slice(idx_dev), slice(nnz_dev), slice_mut(out_dev),
                   batch as u32, rows as u32, cols as u32, max_active as u32]
        }
    }?;
    stream.synchronize()?;
    assert_close(
        "sparse_ft_forward ignores tail",
        &out_dev.to_host_vec(&stream)?,
        &expected,
        0.0,
    );
    Ok(())
}

#[test]
fn sparse_ft_forward_fp16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, rows, cols, max_active) = (5_usize, 8_usize, 9_usize, 6_usize);
    let weight = deterministic_floats(rows * cols, 3.5);
    let (weight_h, weight_q) = quantize_f16(&weight);
    let (indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, cols, 1);
    // CPU reference は GPU が読むのと同じ f16→f32 値で計算する (f16→f32 は無損失)。
    let mut out_cpu = vec![0.0_f32; batch * rows];
    sparse_ft_forward_cpu(
        &weight_q,
        &indices,
        &nnz,
        &mut out_cpu,
        batch,
        rows,
        cols,
        max_active,
    );

    let w_dev = DeviceBuffer::from_host(&stream, &weight_h)?;
    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, batch * rows)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: sparse_ft_forward_fp16, stream: stream, module: module,
            config: cfg_1d(batch * rows / 4),
            args: [slice(w_dev), slice(idx_dev), slice(nnz_dev), slice_mut(out_dev),
                   batch as u32, rows as u32, cols as u32, max_active as u32]
        }
    }?;
    stream.synchronize()?;
    assert_close(
        "sparse_ft_forward_fp16",
        &out_dev.to_host_vec(&stream)?,
        &out_cpu,
        0.0,
    );
    Ok(())
}

#[test]
fn sparse_ft_forward_fp16_out_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, rows, cols, max_active) = (5_usize, 8_usize, 9_usize, 6_usize);
    let weight = deterministic_floats(rows * cols, 4.0);
    let (weight_h, weight_q) = quantize_f16(&weight);
    let (indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, cols, 2);
    // f32 累算結果は GPU と bit 一致するため、出力の round-to-nearest f16 量子化も
    // 同値になる。CPU 側も f16 量子化して bit-exact 比較する。
    let mut out_cpu = vec![0.0_f32; batch * rows];
    sparse_ft_forward_cpu(
        &weight_q,
        &indices,
        &nnz,
        &mut out_cpu,
        batch,
        rows,
        cols,
        max_active,
    );
    let (_, out_cpu_q) = quantize_f16(&out_cpu);

    let w_dev = DeviceBuffer::from_host(&stream, &weight_h)?;
    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let mut out_dev = DeviceBuffer::<f16>::zeroed(&stream, batch * rows)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: sparse_ft_forward_fp16_out, stream: stream, module: module,
            config: cfg_1d(batch * rows / 4),
            args: [slice(w_dev), slice(idx_dev), slice(nnz_dev), slice_mut(out_dev),
                   batch as u32, rows as u32, cols as u32, max_active as u32]
        }
    }?;
    stream.synchronize()?;
    let out_gpu: Vec<f32> = out_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    assert_close("sparse_ft_forward_fp16_out", &out_gpu, &out_cpu_q, 0.0);
    Ok(())
}

#[test]
fn sparse_ft_backward_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, rows, cols, nnz) = (5_usize, 8_usize, 9_usize, 6_usize);
    let grad_out = deterministic_floats(batch * rows, 4.5);
    let indices = sparse_indices_fixture(batch, nnz, cols, 3);
    let mut grad_w_cpu = vec![0.0_f32; rows * cols];
    sparse_ft_backward_cpu(&grad_out, &indices, &mut grad_w_cpu, batch, rows, cols, nnz);

    let gout_dev = DeviceBuffer::from_host(&stream, &grad_out)?;
    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let grad_w_dev = DeviceBuffer::<f32>::zeroed(&stream, rows * cols)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: sparse_ft_backward, stream: stream, module: module,
            config: cfg_1d(batch * rows),
            args: [slice(gout_dev), slice(idx_dev), slice(grad_w_dev),
                   batch as u32, rows as u32, cols as u32, nnz as u32]
        }
    }?;
    stream.synchronize()?;
    // atomic scatter の加算順は非決定的なので relative tolerance。
    assert_close_rel(
        "sparse_ft_backward",
        &grad_w_dev.to_host_vec(&stream)?,
        &grad_w_cpu,
        TOL,
    );
    Ok(())
}

// -- inverse-index sparse FT backward pipeline ---------------------------

/// 中間 kernel (`build_feature_counts` / `exclusive_prefix_sum_small` /
/// `scatter_positions`) を単体照合する。counts / offsets は決定論的 (u32 完全一致)。
/// positions は feature 区間内の並びが atomic write counter の獲得順に依存して
/// 非決定的なため、区間ごとに sort して多重集合として比較する。
///
/// ragged fixture で各 row の実長を変え、`nnz_arr` early-out を exercise する。さらに
/// 実長超 slot を範囲内のゴミで汚しても counts / offsets / positions が不変であること
/// (per-slot kernel が `nnz` までしか読まない契約) を確認する。
#[test]
fn inverse_index_phase_kernels_match_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, max_active, ft_in) = (6_usize, 6_usize, 9_usize);
    let (mut indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, ft_in, 0);
    // 実長超 slot を範囲内 index で汚す (kernel が全幅走査すると counts が変わる)。
    for bi in 0..batch {
        for ni in nnz[bi] as usize..max_active {
            indices[bi * max_active + ni] = ((bi + ni + 1) % ft_in) as i32;
        }
    }

    // CPU reference は実長 (`nnz[bi]`) までのみ集計する。
    let mut counts_cpu = vec![0_u32; ft_in];
    let mut segments_cpu: Vec<Vec<u32>> = vec![Vec::new(); ft_in];
    for bi in 0..batch {
        for ni in 0..nnz[bi] as usize {
            let idx = indices[bi * max_active + ni];
            if idx >= 0 && (idx as usize) < ft_in {
                counts_cpu[idx as usize] += 1;
                segments_cpu[idx as usize].push(bi as u32);
            }
        }
    }
    let mut offsets_cpu = vec![0_u32; ft_in + 1];
    for f in 0..ft_in {
        offsets_cpu[f + 1] = offsets_cpu[f] + counts_cpu[f];
    }
    let total_valid = offsets_cpu[ft_in] as usize;
    for seg in &mut segments_cpu {
        seg.sort_unstable();
    }

    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in)?;
    let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in + 1)?;
    let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in)?;
    let positions_dev = DeviceBuffer::<u32>::zeroed(&stream, batch * max_active)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: build_feature_counts, stream: stream, module: module,
            config: cfg_1d(batch * max_active),
            args: [slice(idx_dev), slice(nnz_dev), slice(counts_dev),
                   batch as u32, max_active as u32, ft_in as u32]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: exclusive_prefix_sum_small, stream: stream, module: module,
            config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
            args: [slice(counts_dev), slice(offsets_dev), ft_in as u32]
        }
    }?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: scatter_positions, stream: stream, module: module,
            config: cfg_1d(batch * max_active),
            args: [slice(idx_dev), slice(nnz_dev), slice(offsets_dev), slice(write_ctr_dev),
                   slice(positions_dev), batch as u32, max_active as u32, ft_in as u32]
        }
    }?;
    stream.synchronize()?;

    assert_eq!(
        counts_dev.to_host_vec(&stream)?,
        counts_cpu,
        "build_feature_counts (実長超のゴミを無視)"
    );
    assert_eq!(
        offsets_dev.to_host_vec(&stream)?,
        offsets_cpu,
        "exclusive_prefix_sum_small"
    );
    let positions = positions_dev.to_host_vec(&stream)?;
    for f in 0..ft_in {
        let mut seg = positions[offsets_cpu[f] as usize..offsets_cpu[f + 1] as usize].to_vec();
        seg.sort_unstable();
        assert_eq!(seg, segments_cpu[f], "scatter_positions feature {f}");
    }
    // 有効 index の総数が offsets 末尾と一致する (実長 slot のみ scatter された)。
    assert_eq!(
        nnz.iter().map(|&n| n as usize).sum::<usize>(),
        total_valid,
        "scatter された position 数 = Σ nnz[bi]"
    );
    Ok(())
}

/// `exclusive_prefix_sum_small` の per-thread chunk 直列和経路 (n > 1024、本番
/// ft_in ≈ 73K 相当) と、余剰 thread が出る n < 1024 の両方を照合する。
#[test]
fn exclusive_prefix_sum_small_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for &n in &[300_usize, 5000] {
        // 0 を含む決定論的な count 列。
        let counts: Vec<u32> = (0..n).map(|i| ((i * 13 + 5) % 7) as u32).collect();
        let mut offsets_cpu = vec![0_u32; n + 1];
        for i in 0..n {
            offsets_cpu[i + 1] = offsets_cpu[i] + counts[i];
        }
        let counts_dev = DeviceBuffer::from_host(&stream, &counts)?;
        let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, n + 1)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_prefix_sum_small, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), n as u32]
            }
        }?;
        stream.synchronize()?;
        assert_eq!(
            offsets_dev.to_host_vec(&stream)?,
            offsets_cpu,
            "exclusive_prefix_sum_small n={n}"
        );
    }
    Ok(())
}

/// multi-block exclusive scan (`prefix_sum_block_local` → `exclusive_prefix_sum_small`
/// → `prefix_sum_add_block_offset`) が単一 block scan と同じ exclusive prefix sum を
/// 出す。1024 倍数でない n / block 跨ぎ / 末尾 partial block を含めて u32 完全一致を照合。
#[test]
fn prefix_sum_multiblock_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    // `small` は典型的な per-feature occurrence count、`large` は full-range u32 で
    // exclusive sum が 2^32 を跨ぐパターン (加算が順序非依存 = overflow 下でも分解と
    // 単一 block scan が一致することを確認)。n は block 境界 / 末尾 partial /
    // exact-multiple multi-block を網羅。
    let small: fn(usize) -> u32 = |i| ((i * 13 + 5) % 7) as u32;
    let large: fn(usize) -> u32 = |i| (i as u32).wrapping_mul(2_654_435_761);
    for &(n, gen_fn) in &[
        (1_usize, small),
        (1023, small),
        (1024, small),
        (1025, small),
        (2048, small),
        (4096, small),
        (73_305, small),
        (125_388, small),
        (125_388, large),
    ] {
        let counts: Vec<u32> = (0..n).map(gen_fn).collect();
        let mut offsets_cpu = vec![0_u32; n + 1];
        for i in 0..n {
            offsets_cpu[i + 1] = offsets_cpu[i].wrapping_add(counts[i]);
        }
        let num_blocks = n.div_ceil(1024);
        let counts_dev = DeviceBuffer::from_host(&stream, &counts)?;
        let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, n + 1)?;
        let block_sums_dev = DeviceBuffer::<u32>::zeroed(&stream, num_blocks)?;
        let block_offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, num_blocks + 1)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: prefix_sum_block_local, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (num_blocks as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), slice(block_sums_dev), n as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_prefix_sum_small, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(block_sums_dev), slice(block_offsets_dev), num_blocks as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: prefix_sum_add_block_offset, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (num_blocks as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(offsets_dev), slice(block_offsets_dev), n as u32, num_blocks as u32]
            }
        }?;
        stream.synchronize()?;
        assert_eq!(
            offsets_dev.to_host_vec(&stream)?,
            offsets_cpu,
            "prefix_sum_multiblock n={n}"
        );
    }
    Ok(())
}

/// inverse-index pipeline (count → prefix sum → scatter → gather) の合成が素朴
/// atomic scatter reference `sparse_ft_backward_cpu` と一致することを確認する。
/// trainer と同じく stm を overwrite 版・nstm を add 版で 2 周し ft_w_grad に合算
/// する。phase D の 4-way unroll partial sum で加算順が CPU と異なるため relative
/// tolerance。
#[test]
fn inverse_index_pipeline_matches_sparse_ft_backward_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    let (_ctx, module, stream) = open_module()?;
    let (batch, ft_out, ft_in, nnz) = (6_usize, 8_usize, 9_usize, 6_usize);
    let stm_indices = sparse_indices_fixture(batch, nnz, ft_in, 0);
    let nstm_indices = sparse_indices_fixture(batch, nnz, ft_in, 4);
    // fixture 前提の guard: feature 0 の区間は 4-way unroll 本体 (4 個) を通過した後
    // 端数 tail に継続する出現数 (> 4 かつ非 4 倍数)、feature ft_in-1 の区間は空
    // (overwrite kernel の sum=0 書き切り経路)。fixture を変えてこの前提が崩れたら
    // ここで気付く。
    for idx in [&stm_indices, &nstm_indices] {
        let count0 = idx.iter().filter(|&&i| i == 0).count();
        assert!(
            count0 > 4 && !count0.is_multiple_of(4),
            "fixture must hit unroll body + tail in one segment (feature 0 count = {count0})"
        );
        assert!(
            !idx.contains(&((ft_in - 1) as i32)),
            "fixture must leave feature ft_in-1 empty"
        );
    }
    let dft_stm = deterministic_floats(batch * ft_out, 5.0);
    let dft_nstm = deterministic_floats(batch * ft_out, 6.0);

    let mut grad_w_cpu = vec![0.0_f32; ft_in * ft_out];
    sparse_ft_backward_cpu(
        &dft_stm,
        &stm_indices,
        &mut grad_w_cpu,
        batch,
        ft_out,
        ft_in,
        nnz,
    );
    sparse_ft_backward_cpu(
        &dft_nstm,
        &nstm_indices,
        &mut grad_w_cpu,
        batch,
        ft_out,
        ft_in,
        nnz,
    );

    let grad_w_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_in * ft_out)?;
    let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in)?;
    let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in + 1)?;
    let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in)?;
    let positions_dev = DeviceBuffer::<u32>::zeroed(&stream, batch * nnz)?;
    // 全 row が全幅 nnz を使う fixture なので nnz_arr は一様 (early-out は no-op、
    // `sparse_ft_backward_cpu` の全幅走査と bit 一致)。
    let nnz_arr = vec![nnz as i32; batch];
    let nnz_arr_dev = DeviceBuffer::from_host(&stream, &nnz_arr)?;
    for (iter_idx, (idx_host, dft_host)) in [(&stm_indices, &dft_stm), (&nstm_indices, &dft_nstm)]
        .into_iter()
        .enumerate()
    {
        memset_zero(&stream, &counts_dev)?;
        memset_zero(&stream, &write_ctr_dev)?;
        let idx_dev = DeviceBuffer::from_host(&stream, idx_host)?;
        let dft_dev = DeviceBuffer::from_host(&stream, dft_host)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: build_feature_counts, stream: stream, module: module,
                config: cfg_1d(batch * nnz),
                args: [slice(idx_dev), slice(nnz_arr_dev), slice(counts_dev),
                       batch as u32, nnz as u32, ft_in as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_prefix_sum_small, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), ft_in as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: scatter_positions, stream: stream, module: module,
                config: cfg_1d(batch * nnz),
                args: [slice(idx_dev), slice(nnz_arr_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(positions_dev), batch as u32, nnz as u32, ft_in as u32]
            }
        }?;
        // production (block 128 × grid y = ft_out/128) と同じく ri 軸を blockIdx_y で
        // tile する。ft_out=8 を 4 幅 × 2 tile に割って y 軸も exercise する。
        if iter_idx == 0 {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: gather_and_sum_per_feature_overwrite, stream: stream, module: module,
                    config: LaunchConfig { grid_dim: (ft_in as u32, 2, 1), block_dim: (4, 1, 1), shared_mem_bytes: 0 },
                    args: [slice(dft_dev), slice(positions_dev), slice(offsets_dev), slice(grad_w_dev),
                           ft_in as u32, ft_out as u32]
                }
            }?;
        } else {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: gather_and_sum_per_feature_add, stream: stream, module: module,
                    config: LaunchConfig { grid_dim: (ft_in as u32, 2, 1), block_dim: (4, 1, 1), shared_mem_bytes: 0 },
                    args: [slice(dft_dev), slice(positions_dev), slice(offsets_dev), slice(grad_w_dev),
                           ft_in as u32, ft_out as u32]
                }
            }?;
        }
    }
    stream.synchronize()?;
    assert_close_rel(
        "inverse_index_pipeline grad_w",
        &grad_w_dev.to_host_vec(&stream)?,
        &grad_w_cpu,
        TOL,
    );
    Ok(())
}

/// inverse-index pipeline の FP16 dft 版 (`gather_and_sum_per_feature_{overwrite,add}
/// _fp16`)。dft は loss scaling 済の値を f16 量子化して渡し、gather 側の
/// `dft_inv_scale` が scale を打ち消す round-trip を検証する。CPU reference は
/// GPU が読む f16→f32 値で合算し、最後に inv_scale を掛ける。
#[test]
fn inverse_index_pipeline_fp16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, ft_out, ft_in, nnz) = (6_usize, 8_usize, 9_usize, 6_usize);
    let stm_indices = sparse_indices_fixture(batch, nnz, ft_in, 0);
    let nstm_indices = sparse_indices_fixture(batch, nnz, ft_in, 4);
    let dft_scale = 64.0_f32;
    let dft_inv_scale = 1.0_f32 / dft_scale;
    let dft_stm_scaled: Vec<f32> = deterministic_floats(batch * ft_out, 7.0)
        .iter()
        .map(|&x| x * dft_scale)
        .collect();
    let dft_nstm_scaled: Vec<f32> = deterministic_floats(batch * ft_out, 8.0)
        .iter()
        .map(|&x| x * dft_scale)
        .collect();
    let (dft_stm_h, dft_stm_q) = quantize_f16(&dft_stm_scaled);
    let (dft_nstm_h, dft_nstm_q) = quantize_f16(&dft_nstm_scaled);

    let mut grad_w_scaled_cpu = vec![0.0_f32; ft_in * ft_out];
    sparse_ft_backward_cpu(
        &dft_stm_q,
        &stm_indices,
        &mut grad_w_scaled_cpu,
        batch,
        ft_out,
        ft_in,
        nnz,
    );
    sparse_ft_backward_cpu(
        &dft_nstm_q,
        &nstm_indices,
        &mut grad_w_scaled_cpu,
        batch,
        ft_out,
        ft_in,
        nnz,
    );
    let grad_w_cpu: Vec<f32> = grad_w_scaled_cpu
        .iter()
        .map(|&x| x * dft_inv_scale)
        .collect();

    let grad_w_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_in * ft_out)?;
    let counts_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in)?;
    let offsets_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in + 1)?;
    let write_ctr_dev = DeviceBuffer::<u32>::zeroed(&stream, ft_in)?;
    let positions_dev = DeviceBuffer::<u32>::zeroed(&stream, batch * nnz)?;
    // 全 row が全幅 nnz を使う fixture なので nnz_arr は一様 (early-out は no-op)。
    let nnz_arr = vec![nnz as i32; batch];
    let nnz_arr_dev = DeviceBuffer::from_host(&stream, &nnz_arr)?;
    for (iter_idx, (idx_host, dft_host)) in
        [(&stm_indices, &dft_stm_h), (&nstm_indices, &dft_nstm_h)]
            .into_iter()
            .enumerate()
    {
        memset_zero(&stream, &counts_dev)?;
        memset_zero(&stream, &write_ctr_dev)?;
        let idx_dev = DeviceBuffer::from_host(&stream, idx_host)?;
        let dft_dev = DeviceBuffer::from_host(&stream, dft_host)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: build_feature_counts, stream: stream, module: module,
                config: cfg_1d(batch * nnz),
                args: [slice(idx_dev), slice(nnz_arr_dev), slice(counts_dev),
                       batch as u32, nnz as u32, ft_in as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: exclusive_prefix_sum_small, stream: stream, module: module,
                config: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                args: [slice(counts_dev), slice(offsets_dev), ft_in as u32]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: scatter_positions, stream: stream, module: module,
                config: cfg_1d(batch * nnz),
                args: [slice(idx_dev), slice(nnz_arr_dev), slice(offsets_dev), slice(write_ctr_dev),
                       slice(positions_dev), batch as u32, nnz as u32, ft_in as u32]
            }
        }?;
        if iter_idx == 0 {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: gather_and_sum_per_feature_overwrite_fp16, stream: stream, module: module,
                    config: LaunchConfig { grid_dim: (ft_in as u32, 2, 1), block_dim: (4, 1, 1), shared_mem_bytes: 0 },
                    args: [slice(dft_dev), slice(positions_dev), slice(offsets_dev), slice(grad_w_dev),
                           ft_in as u32, ft_out as u32, dft_inv_scale]
                }
            }?;
        } else {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: gather_and_sum_per_feature_add_fp16, stream: stream, module: module,
                    config: LaunchConfig { grid_dim: (ft_in as u32, 2, 1), block_dim: (4, 1, 1), shared_mem_bytes: 0 },
                    args: [slice(dft_dev), slice(positions_dev), slice(offsets_dev), slice(grad_w_dev),
                           ft_in as u32, ft_out as u32, dft_inv_scale]
                }
            }?;
        }
    }
    stream.synchronize()?;
    // GPU は iter ごとに inv_scale を掛けてから加算、CPU は合算後に 1 回掛けるため
    // 丸めが ~ulp 単位で異なる。atomic 加算順差と合わせ relative tolerance。
    assert_close_rel(
        "inverse_index_pipeline_fp16 grad_w",
        &grad_w_dev.to_host_vec(&stream)?,
        &grad_w_cpu,
        TOL,
    );
    Ok(())
}

// -- cast_f32_to_f16 ------------------------------------------------------

/// GPU の round-to-nearest f32→f16 変換が host の `as f16` cast と一致する。
/// f16 有限域外 (±inf 化)・subnormal 域・NaN を含む。
#[test]
fn cast_f32_to_f16_matches_host_cast() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let mut src = deterministic_floats(257, 9.0);
    src.extend_from_slice(&[70000.0, -70000.0, 65504.0, 1e-8, -1e-8, f32::NAN]);
    let n = src.len();
    let (_, expected) = quantize_f16(&src);

    let src_dev = DeviceBuffer::from_host(&stream, &src)?;
    let mut dst_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: cast_f32_to_f16, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(src_dev), slice_mut(dst_dev), n as u32]
        }
    }?;
    stream.synchronize()?;
    let got: Vec<f32> = dst_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    // ±inf を含むため差分比較でなく値同値で比較する (NaN は payload 非規定なので
    // is_nan のみ確認)。
    for (i, (&g, &c)) in got.iter().zip(expected.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "cast_f32_to_f16[{i}]: cpu=NaN but gpu={g}");
        } else {
            assert_eq!(g, c, "cast_f32_to_f16[{i}]");
        }
    }
    Ok(())
}

// -- ft_fold_virtual / ft_reduce_virtual_grad (FT factorizer) -------------

/// FT factorizer の fold / reduce テスト共通 fixture:
/// (ft_in, ft_out, piece_inputs, train 形状の決定論 weight)。kb = 7 × pi = 5 —
/// `ft_reduce_virtual_grad` の 4-way unroll 本体 (kb 0..4) と tail (kb 4..7) の
/// 両経路を踏む n_kb にする。
fn ft_factorize_fixture() -> (usize, usize, usize, Vec<f32>) {
    let ft_in = 35;
    let ft_out = 8;
    let pi = 5;
    let w = deterministic_floats((ft_in + pi) * ft_out, 3.0);
    (ft_in, ft_out, pi, w)
}

fn ft_factorize_layout(
    base_ft_in: usize,
    ft_in: usize,
    ft_out: usize,
    pi: usize,
    nb: usize,
    mode: u32,
) -> FtFactorizeLayout<'static> {
    FtFactorizeLayout {
        base_ft_in,
        ft_in,
        ft_out,
        piece_inputs: pi,
        nb,
        mode,
        threat_pair_starts: &[],
    }
}

fn ft_factorize_pack(nb: usize, mode: u32) -> u32 {
    (nb as u32) | (mode << 16)
}

fn ft_factorize_bounds(base_ft_in: usize, ft_in: usize) -> u64 {
    (base_ft_in as u64) | ((ft_in as u64) << 32)
}

fn no_threat_pair_starts(
    stream: &CudaStream,
) -> Result<DeviceBuffer<u32>, Box<dyn std::error::Error>> {
    DeviceBuffer::from_host(stream, &[0_u32]).map_err(Into::into)
}

fn threat_pair_starts_dev(
    stream: &CudaStream,
    starts: &[usize],
) -> Result<DeviceBuffer<u32>, Box<dyn std::error::Error>> {
    let starts_u32: Vec<u32> = starts.iter().map(|&x| x as u32).collect();
    DeviceBuffer::from_host(stream, &starts_u32).map_err(Into::into)
}

/// fold (f32 出力) が CPU reference と完全一致する (thread ごと 1 加算で
/// 加算順差が無い)。
#[test]
fn ft_fold_virtual_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (ft_in, ft_out, pi, w) = ft_factorize_fixture();
    let n = ft_in * ft_out;

    let mut comb_cpu = vec![0.0_f32; n];
    // threat 無し: base_ft_in == ft_in。
    ft_fold_virtual_cpu(
        &w,
        &mut comb_cpu,
        ft_factorize_layout(ft_in, ft_in, ft_out, pi, 1, FT_FACTORIZE_BASE),
    );

    let w_dev = DeviceBuffer::from_host(&stream, &w)?;
    let threat_starts_dev = no_threat_pair_starts(&stream)?;
    let mut comb_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_fold_virtual, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(w_dev), slice_mut(comb_dev), slice(threat_starts_dev),
                   ft_factorize_bounds(ft_in, ft_in), ft_out as u32, pi as u32,
                   ft_factorize_pack(1, FT_FACTORIZE_BASE)]
        }
    }?;
    stream.synchronize()?;
    assert_close(
        "ft_fold_virtual",
        &comb_dev.to_host_vec(&stream)?,
        &comb_cpu,
        0.0,
    );
    Ok(())
}

/// fold f16 出力版が「f32 で加算 → 1 回 f16 丸め」の host 計算と bit 一致する。
#[test]
fn ft_fold_virtual_f16_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (ft_in, ft_out, pi, w) = ft_factorize_fixture();
    let n = ft_in * ft_out;

    let mut comb_cpu = vec![0.0_f32; n];
    ft_fold_virtual_cpu(
        &w,
        &mut comb_cpu,
        ft_factorize_layout(ft_in, ft_in, ft_out, pi, 1, FT_FACTORIZE_BASE),
    );
    let (_, expected) = quantize_f16(&comb_cpu);

    let w_dev = DeviceBuffer::from_host(&stream, &w)?;
    let threat_starts_dev = no_threat_pair_starts(&stream)?;
    let mut comb_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_fold_virtual_f16, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(w_dev), slice_mut(comb_dev), slice(threat_starts_dev),
                   ft_factorize_bounds(ft_in, ft_in), ft_out as u32, pi as u32,
                   ft_factorize_pack(1, FT_FACTORIZE_BASE)]
        }
    }?;
    stream.synchronize()?;
    let got: Vec<f32> = comb_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    assert_close("ft_fold_virtual_f16", &got, &expected, 0.0);
    Ok(())
}

/// reduce が CPU reference と一致し、実 block を変更しない。仮想 block は
/// 4-way unroll の加算順差があるため relative tolerance。
#[test]
fn ft_reduce_virtual_grad_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (ft_in, ft_out, pi, grad_init) = ft_factorize_fixture();

    let mut grad_cpu = grad_init.clone();
    ft_reduce_virtual_grad_cpu(
        &mut grad_cpu,
        ft_factorize_layout(ft_in, ft_in, ft_out, pi, 1, FT_FACTORIZE_BASE),
    );

    let grad_dev = DeviceBuffer::from_host(&stream, &grad_init)?;
    let threat_starts_dev = no_threat_pair_starts(&stream)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_reduce_virtual_grad, stream: stream, module: module,
            config: cfg_1d(pi * ft_out),
            args: [slice(grad_dev), slice(threat_starts_dev),
                   ft_factorize_bounds(ft_in, ft_in), ft_out as u32, pi as u32,
                   ft_factorize_pack(1, FT_FACTORIZE_BASE)]
        }
    }?;
    stream.synchronize()?;
    let got = grad_dev.to_host_vec(&stream)?;
    assert_eq!(
        &got[..ft_in * ft_out],
        &grad_init[..ft_in * ft_out],
        "実 block は read-only"
    );
    assert_close_rel(
        "ft_reduce_virtual_grad virtual block",
        &got[ft_in * ft_out..],
        &grad_cpu[ft_in * ft_out..],
        TOL,
    );
    Ok(())
}

/// threat 同居 fixture: base 実行 (35 行) の後ろに threat real 行 (12)、その後ろに
/// piece-input 仮想行 (pi=5) と threat-pair 仮想行 (3)。pair 幅は 2/3/7。
fn ft_factorize_coexist_fixture() -> (usize, usize, usize, usize, Vec<f32>) {
    let base_ft_in = 35;
    let threat = 12;
    let ft_in = base_ft_in + threat;
    let ft_out = 8;
    let pi = 5;
    let threat_pairs = 3;
    let w = deterministic_floats((ft_in + pi + threat_pairs) * ft_out, 3.0);
    (base_ft_in, ft_in, ft_out, pi, w)
}

const COEXIST_THREAT_PAIR_STARTS: [usize; 4] = [0, 2, 5, 12];

/// 同居 fold: base セルは piece-input 仮想行を畳み、threat セルは pair 仮想行を畳む。
/// GPU が range-aware CPU reference と完全一致する (1 加算/cell で加算順差なし)。
#[test]
fn ft_fold_virtual_coexist_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (base_ft_in, ft_in, ft_out, pi, w) = ft_factorize_coexist_fixture();
    let n = ft_in * ft_out;

    let mut comb_cpu = vec![0.0_f32; n];
    ft_fold_virtual_cpu(
        &w,
        &mut comb_cpu,
        FtFactorizeLayout {
            threat_pair_starts: &COEXIST_THREAT_PAIR_STARTS,
            ..ft_factorize_layout(base_ft_in, ft_in, ft_out, pi, 1, FT_FACTORIZE_BASE)
        },
    );
    for pair in 0..COEXIST_THREAT_PAIR_STARTS.len() - 1 {
        let vrow = ft_in + pi + pair;
        for rel in COEXIST_THREAT_PAIR_STARTS[pair]..COEXIST_THREAT_PAIR_STARTS[pair + 1] {
            let feat = base_ft_in + rel;
            for ri in 0..ft_out {
                assert_eq!(
                    comb_cpu[feat * ft_out + ri],
                    w[feat * ft_out + ri] + w[vrow * ft_out + ri]
                );
            }
        }
    }
    for feat in 0..base_ft_in {
        let p = feat % pi;
        for ri in 0..ft_out {
            assert_eq!(
                comb_cpu[feat * ft_out + ri],
                w[feat * ft_out + ri] + w[(ft_in + p) * ft_out + ri]
            );
        }
    }

    let w_dev = DeviceBuffer::from_host(&stream, &w)?;
    let threat_starts_dev = threat_pair_starts_dev(&stream, &COEXIST_THREAT_PAIR_STARTS)?;
    let mut comb_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_fold_virtual, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(w_dev), slice_mut(comb_dev), slice(threat_starts_dev),
                   ft_factorize_bounds(base_ft_in, ft_in), ft_out as u32, pi as u32,
                   ft_factorize_pack(1, FT_FACTORIZE_BASE)]
        }
    }?;
    stream.synchronize()?;
    assert_close(
        "ft_fold_virtual coexist",
        &comb_dev.to_host_vec(&stream)?,
        &comb_cpu,
        0.0,
    );
    Ok(())
}

/// 同居 fold f16 出力版も threat pair table 付き CPU reference と一致する。
#[test]
fn ft_fold_virtual_f16_coexist_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (base_ft_in, ft_in, ft_out, pi, w) = ft_factorize_coexist_fixture();
    let n = ft_in * ft_out;

    let mut comb_cpu = vec![0.0_f32; n];
    ft_fold_virtual_cpu(
        &w,
        &mut comb_cpu,
        FtFactorizeLayout {
            threat_pair_starts: &COEXIST_THREAT_PAIR_STARTS,
            ..ft_factorize_layout(base_ft_in, ft_in, ft_out, pi, 1, FT_FACTORIZE_BASE)
        },
    );
    let (_, expected) = quantize_f16(&comb_cpu);

    let w_dev = DeviceBuffer::from_host(&stream, &w)?;
    let threat_starts_dev = threat_pair_starts_dev(&stream, &COEXIST_THREAT_PAIR_STARTS)?;
    let mut comb_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_fold_virtual_f16, stream: stream, module: module, config: cfg_1d(n),
            args: [slice(w_dev), slice_mut(comb_dev), slice(threat_starts_dev),
                   ft_factorize_bounds(base_ft_in, ft_in), ft_out as u32, pi as u32,
                   ft_factorize_pack(1, FT_FACTORIZE_BASE)]
        }
    }?;
    stream.synchronize()?;
    let got: Vec<f32> = comb_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    assert_close("ft_fold_virtual_f16 coexist", &got, &expected, 0.0);
    Ok(())
}

/// 同居 reduce: piece-input 仮想行は base 実行の和、threat-pair 仮想行は同じ
/// pair に属する threat 実行の和。base + threat 実 block は read-only。
#[test]
fn ft_reduce_virtual_grad_coexist_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (base_ft_in, ft_in, ft_out, pi, grad_init) = ft_factorize_coexist_fixture();

    let mut grad_cpu = grad_init.clone();
    ft_reduce_virtual_grad_cpu(
        &mut grad_cpu,
        FtFactorizeLayout {
            threat_pair_starts: &COEXIST_THREAT_PAIR_STARTS,
            ..ft_factorize_layout(base_ft_in, ft_in, ft_out, pi, 1, FT_FACTORIZE_BASE)
        },
    );

    let grad_dev = DeviceBuffer::from_host(&stream, &grad_init)?;
    let threat_starts_dev = threat_pair_starts_dev(&stream, &COEXIST_THREAT_PAIR_STARTS)?;
    let virtual_rows = pi + COEXIST_THREAT_PAIR_STARTS.len() - 1;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_reduce_virtual_grad, stream: stream, module: module,
            config: cfg_1d(virtual_rows * ft_out),
            args: [slice(grad_dev), slice(threat_starts_dev),
                   ft_factorize_bounds(base_ft_in, ft_in), ft_out as u32, pi as u32,
                   ft_factorize_pack(1, FT_FACTORIZE_BASE)]
        }
    }?;
    stream.synchronize()?;
    let got = grad_dev.to_host_vec(&stream)?;
    // base + threat 実 block は read-only。
    assert_eq!(
        &got[..ft_in * ft_out],
        &grad_init[..ft_in * ft_out],
        "実 block (base + threat) は read-only"
    );
    assert_close_rel(
        "ft_reduce_virtual_grad coexist virtual block",
        &got[ft_in * ft_out..],
        &grad_cpu[ft_in * ft_out..],
        TOL,
    );
    Ok(())
}

fn ft_factorize_effect_bucket_fixture(mode: u32) -> (usize, usize, usize, usize, Vec<f32>) {
    let kb = 7;
    let pi = 5;
    let nb = 4;
    let ft_in = kb * pi * nb;
    let ft_out = 8;
    let virtual_rows = if mode == FT_FACTORIZE_PER_EFFECT_BUCKET {
        pi * nb
    } else {
        pi
    };
    let w = deterministic_floats((ft_in + virtual_rows) * ft_out, 3.0);
    (ft_in, ft_out, pi, nb, w)
}

#[test]
fn ft_fold_virtual_effect_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for mode in [
        FT_FACTORIZE_POOL_EFFECT_BUCKETS,
        FT_FACTORIZE_PER_EFFECT_BUCKET,
    ] {
        let (ft_in, ft_out, pi, nb, w) = ft_factorize_effect_bucket_fixture(mode);
        let n = ft_in * ft_out;

        let mut comb_cpu = vec![0.0_f32; n];
        ft_fold_virtual_cpu(
            &w,
            &mut comb_cpu,
            ft_factorize_layout(ft_in, ft_in, ft_out, pi, nb, mode),
        );

        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let threat_starts_dev = no_threat_pair_starts(&stream)?;
        let mut comb_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: ft_fold_virtual, stream: stream, module: module, config: cfg_1d(n),
                args: [slice(w_dev), slice_mut(comb_dev), slice(threat_starts_dev),
                       ft_factorize_bounds(ft_in, ft_in), ft_out as u32, pi as u32,
                       ft_factorize_pack(nb, mode)]
            }
        }?;
        stream.synchronize()?;
        assert_close(
            "ft_fold_virtual effect_bucket",
            &comb_dev.to_host_vec(&stream)?,
            &comb_cpu,
            0.0,
        );
    }
    Ok(())
}

#[test]
fn ft_fold_virtual_f16_effect_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for mode in [
        FT_FACTORIZE_POOL_EFFECT_BUCKETS,
        FT_FACTORIZE_PER_EFFECT_BUCKET,
    ] {
        let (ft_in, ft_out, pi, nb, w) = ft_factorize_effect_bucket_fixture(mode);
        let n = ft_in * ft_out;

        let mut comb_cpu = vec![0.0_f32; n];
        ft_fold_virtual_cpu(
            &w,
            &mut comb_cpu,
            ft_factorize_layout(ft_in, ft_in, ft_out, pi, nb, mode),
        );
        let (_, expected) = quantize_f16(&comb_cpu);

        let w_dev = DeviceBuffer::from_host(&stream, &w)?;
        let threat_starts_dev = no_threat_pair_starts(&stream)?;
        let mut comb_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: ft_fold_virtual_f16, stream: stream, module: module, config: cfg_1d(n),
                args: [slice(w_dev), slice_mut(comb_dev), slice(threat_starts_dev),
                       ft_factorize_bounds(ft_in, ft_in), ft_out as u32, pi as u32,
                       ft_factorize_pack(nb, mode)]
            }
        }?;
        stream.synchronize()?;
        let got: Vec<f32> = comb_dev
            .to_host_vec(&stream)?
            .iter()
            .map(|&x| x as f32)
            .collect();
        assert_close("ft_fold_virtual_f16 effect_bucket", &got, &expected, 0.0);
    }
    Ok(())
}

#[test]
fn ft_reduce_virtual_grad_effect_bucket_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for mode in [
        FT_FACTORIZE_POOL_EFFECT_BUCKETS,
        FT_FACTORIZE_PER_EFFECT_BUCKET,
    ] {
        let (ft_in, ft_out, pi, nb, grad_init) = ft_factorize_effect_bucket_fixture(mode);
        let virtual_rows = if mode == FT_FACTORIZE_PER_EFFECT_BUCKET {
            pi * nb
        } else {
            pi
        };

        let mut grad_cpu = grad_init.clone();
        ft_reduce_virtual_grad_cpu(
            &mut grad_cpu,
            ft_factorize_layout(ft_in, ft_in, ft_out, pi, nb, mode),
        );

        let grad_dev = DeviceBuffer::from_host(&stream, &grad_init)?;
        let threat_starts_dev = no_threat_pair_starts(&stream)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: ft_reduce_virtual_grad, stream: stream, module: module,
                config: cfg_1d(virtual_rows * ft_out),
                args: [slice(grad_dev), slice(threat_starts_dev),
                       ft_factorize_bounds(ft_in, ft_in), ft_out as u32, pi as u32,
                       ft_factorize_pack(nb, mode)]
            }
        }?;
        stream.synchronize()?;
        let got = grad_dev.to_host_vec(&stream)?;
        assert_eq!(
            &got[..ft_in * ft_out],
            &grad_init[..ft_in * ft_out],
            "effect bucket 実 block は read-only"
        );
        assert_close_rel(
            "ft_reduce_virtual_grad effect_bucket virtual block",
            &got[ft_in * ft_out..],
            &grad_cpu[ft_in * ft_out..],
            TOL,
        );
    }
    Ok(())
}

// -- optimizer (RAdam step / Ranger lookahead lerp) -----------------------

/// optimizer テスト共通の決定論 fixture (weights / m / v / grad)。weights の先頭
/// 2 cell は clamp が効く外れ値、grad の中央 1 cell は NaN (伝搬確認)。`v` は勾配
/// 二乗の EMA なので非負。
fn radam_fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut weights = deterministic_floats(n, 11.0);
    weights[0] = 5.0;
    weights[1] = -5.0;
    let m: Vec<f32> = (0..n)
        .map(|i| -0.02_f32 + 0.0003_f32 * (i % 97) as f32)
        .collect();
    let v: Vec<f32> = (0..n)
        .map(|i| 1e-6_f32 + 1e-7_f32 * (i % 89) as f32)
        .collect();
    let mut grad = deterministic_floats(n, 13.0);
    grad[n / 2] = f32::NAN;
    (weights, m, v, grad)
}

/// `radam_step` が CPU reference と一致する。学習初期 (step=1、denom=0 で variance
/// 補正 off) と通常領域 (step=1000、denom=1) の両経路、weight decay、quantized
/// dense weight の clamp (`W_CLAMP_QUANT_*`) を踏む。grad は kernel 内で 0 reset。
#[test]
fn radam_step_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    for &step in &[1_u64, 1000] {
        let n = 259_usize;
        let (weights, m, v, grad) = radam_fixture(n);
        let (step_size, denom) = radam_compute_step_size_denom(step, BETA1, BETA2, N_SMA_THRESHOLD);
        let lr = 8.75e-4_f32;
        let decay = RANGER_DEFAULTS.decay;

        let mut w_cpu = weights.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        let mut g_cpu = grad.clone();
        radam_step_cpu(
            &mut w_cpu,
            &mut m_cpu,
            &mut v_cpu,
            &mut g_cpu,
            lr,
            step_size,
            denom,
            decay,
            BETA1,
            BETA2,
            EPS,
            W_CLAMP_QUANT_MIN,
            W_CLAMP_QUANT_MAX,
            n,
        );

        let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
        let mut m_dev = DeviceBuffer::from_host(&stream, &m)?;
        let mut v_dev = DeviceBuffer::from_host(&stream, &v)?;
        let mut g_dev = DeviceBuffer::from_host(&stream, &grad)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: stream, module: module, config: cfg_1d(n),
                args: [slice_mut(w_dev), slice_mut(m_dev), slice_mut(v_dev), slice_mut(g_dev),
                       lr, step_size, denom, decay, BETA1, BETA2, EPS,
                       W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, n as u32]
            }
        }?;
        stream.synchronize()?;
        assert_close(
            &format!("radam_step/step{step} weights"),
            &w_dev.to_host_vec(&stream)?,
            &w_cpu,
            TOL,
        );
        assert_close(
            &format!("radam_step/step{step} m"),
            &m_dev.to_host_vec(&stream)?,
            &m_cpu,
            TOL,
        );
        assert_close(
            &format!("radam_step/step{step} v"),
            &v_dev.to_host_vec(&stream)?,
            &v_cpu,
            TOL,
        );
        // grad は NaN cell も含め kernel 内で 0 reset。
        assert_close(
            &format!("radam_step/step{step} grad reset"),
            &g_dev.to_host_vec(&stream)?,
            &vec![0.0_f32; n],
            0.0,
        );
    }
    Ok(())
}

/// `radam_step_fp16_mirror` は `radam_step` と同 math + 確定 weight の f16 mirror
/// 同時書き込み。weights / m / v は FP32 reference と照合し、mirror は GPU weights
/// の round-to-nearest f16 量子化と同値 (NaN cell は NaN) であることを確認する。
#[test]
fn radam_step_fp16_mirror_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let n = 259_usize;
    let (weights, m, v, grad) = radam_fixture(n);
    let (step_size, denom) = radam_compute_step_size_denom(1000, BETA1, BETA2, N_SMA_THRESHOLD);
    let lr = 8.75e-4_f32;
    let decay = RANGER_DEFAULTS.decay;

    let mut w_cpu = weights.clone();
    let mut m_cpu = m.clone();
    let mut v_cpu = v.clone();
    let mut g_cpu = grad.clone();
    radam_step_cpu(
        &mut w_cpu,
        &mut m_cpu,
        &mut v_cpu,
        &mut g_cpu,
        lr,
        step_size,
        denom,
        decay,
        BETA1,
        BETA2,
        EPS,
        W_CLAMP_NONE_MIN,
        W_CLAMP_NONE_MAX,
        n,
    );

    let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
    let mut m_dev = DeviceBuffer::from_host(&stream, &m)?;
    let mut v_dev = DeviceBuffer::from_host(&stream, &v)?;
    let mut g_dev = DeviceBuffer::from_host(&stream, &grad)?;
    let mut mirror_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: radam_step_fp16_mirror, stream: stream, module: module, config: cfg_1d(n),
            args: [slice_mut(w_dev), slice_mut(m_dev), slice_mut(v_dev), slice_mut(g_dev),
                   slice_mut(mirror_dev),
                   lr, step_size, denom, decay, BETA1, BETA2, EPS,
                   W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, n as u32]
        }
    }?;
    stream.synchronize()?;
    let w_gpu = w_dev.to_host_vec(&stream)?;
    assert_close("radam_step_fp16_mirror weights", &w_gpu, &w_cpu, TOL);
    assert_close(
        "radam_step_fp16_mirror m",
        &m_dev.to_host_vec(&stream)?,
        &m_cpu,
        TOL,
    );
    assert_close(
        "radam_step_fp16_mirror v",
        &v_dev.to_host_vec(&stream)?,
        &v_cpu,
        TOL,
    );
    // mirror は GPU 確定 weight の f16 量子化と同値のはず (GPU weight から host で
    // 再量子化して比較、量子化器の差を排除)。
    let mirror: Vec<f32> = mirror_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let (_, w_gpu_q) = quantize_f16(&w_gpu);
    assert_close("radam_step_fp16_mirror mirror", &mirror, &w_gpu_q, 0.0);
    Ok(())
}

/// `radam_step_f16state` の CPU mimic。GPU kernel と同じ f16 round-trip
/// (scale → clamp ±65504 → f16 格納、読み出しで dequantize) を host で再現する。
/// 引数は kernel signature と同順 (`clippy::too_many_arguments` は他 reference と
/// 同じ理由で allow)。
#[allow(clippy::too_many_arguments)]
fn radam_step_f16state_mimic(
    weights: &mut [f32],
    m: &mut [f16],
    v: &mut [f16],
    grad: &mut [f32],
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
    n: usize,
) {
    let rate = lr * step_size;
    for i in 0..n {
        let g = grad[i];
        let mut p = weights[i];
        p *= 1.0_f32 - decay * rate;
        let m_prev = (m[i] as f32) / m_scale;
        let v_prev = (v[i] as f32) / v_scale;
        let mi = beta1 * m_prev + (1.0_f32 - beta1) * g;
        let vi = beta2 * v_prev + (1.0_f32 - beta2) * g * g;
        m[i] = (mi * m_scale).clamp(-65504.0, 65504.0) as f16;
        let vs = vi * v_scale;
        let vs_c = if vs > 65504.0_f32 { 65504.0_f32 } else { vs };
        v[i] = vs_c as f16;
        let mut val = mi;
        if denom != 0 {
            val /= vi.sqrt() + eps;
        }
        p -= rate * val;
        weights[i] = p.clamp(min_w, max_w);
        grad[i] = 0.0_f32;
    }
}

/// `radam_step_f16state` 用の production スケール fixture: `|m| ~ 1e-9` /
/// `v ~ 1e-15` / `|grad| ~ 1e-4` (FT weight の実測値域)。grad の先頭 2 cell は
/// scale 後に f16 有限域 (±65504) を超えて m / v の格納時 clamp を踏む外れ値。
#[allow(clippy::type_complexity)]
fn f16state_fixture(n: usize) -> (Vec<f32>, Vec<f16>, Vec<f16>, [Vec<f32>; 2]) {
    let weights = deterministic_floats(n, 17.0);
    let m_h: Vec<f16> = (0..n)
        .map(|i| (((i as f32) - 60.0) * 3e-11 * FT_OPT_M_SCALE) as f16)
        .collect();
    let v_h: Vec<f16> = (0..n)
        .map(|i| ((1e-15_f32 + (i % 13) as f32 * 2e-16) * FT_OPT_V_SCALE) as f16)
        .collect();
    let mut g0: Vec<f32> = (0..n)
        .map(|i| (((i * 7) % 23) as f32 - 11.0) * 1e-5)
        .collect();
    g0[0] = 0.05;
    g0[1] = -0.05;
    let g1: Vec<f32> = (0..n)
        .map(|i| (((i * 5) % 19) as f32 - 9.0) * 2e-5)
        .collect();
    (weights, m_h, v_h, [g0, g1])
}

/// `radam_step_f16state` が CPU mimic と一致する。denom=0 (step=1) → denom=1
/// (step=1000) の 2 step を同じ GPU state で回し、f16 格納値からの read-back 継続も
/// 検証する。f16 state は GPU/CPU で同一演算列だが fma 縮約差が量子化境界で f16
/// 1 ulp (相対 ~5e-4) の差になり得るため、dequantize 値を relative tolerance で比較。
#[test]
fn radam_step_f16state_matches_mimic() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let n = 131_usize;
    let (weights, m_h, v_h, grads) = f16state_fixture(n);
    let lr = 8.75e-4_f32;
    let decay = RANGER_DEFAULTS.decay;

    let mut w_cpu = weights.clone();
    let mut m_cpu = m_h.clone();
    let mut v_cpu = v_h.clone();
    let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
    let mut m_dev = DeviceBuffer::from_host(&stream, &m_h)?;
    let mut v_dev = DeviceBuffer::from_host(&stream, &v_h)?;
    for (g, step) in grads.iter().zip([1_u64, 1000]) {
        let (step_size, denom) = radam_compute_step_size_denom(step, BETA1, BETA2, N_SMA_THRESHOLD);
        let mut g_cpu = g.clone();
        radam_step_f16state_mimic(
            &mut w_cpu,
            &mut m_cpu,
            &mut v_cpu,
            &mut g_cpu,
            lr,
            step_size,
            denom,
            decay,
            BETA1,
            BETA2,
            EPS,
            W_CLAMP_NONE_MIN,
            W_CLAMP_NONE_MAX,
            FT_OPT_M_SCALE,
            FT_OPT_V_SCALE,
            n,
        );
        let mut g_dev = DeviceBuffer::from_host(&stream, g)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step_f16state, stream: stream, module: module, config: cfg_1d(n),
                args: [slice_mut(w_dev), slice_mut(m_dev), slice_mut(v_dev), slice_mut(g_dev),
                       lr, step_size, denom, decay, BETA1, BETA2, EPS,
                       W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX,
                       FT_OPT_M_SCALE, FT_OPT_V_SCALE, n as u32]
            }
        }?;
        stream.synchronize()?;
        if step == 1 {
            // 外れ値 grad cell の m / v は scale 後に f16 有限域を超え、step 1 の格納時
            // clamp で両側とも ±65504 ちょうどに飽和する (step 2 の EMA decay で
            // f16 max 未満へ戻るため、ここで確認する)。
            let m_gpu: Vec<f16> = m_dev.to_host_vec(&stream)?;
            let v_gpu: Vec<f16> = v_dev.to_host_vec(&stream)?;
            assert_eq!(m_gpu[0] as f32, 65504.0, "m[0] must saturate at f16 max");
            assert_eq!(m_gpu[1] as f32, -65504.0, "m[1] must saturate at f16 min");
            assert_eq!(v_gpu[0] as f32, 65504.0, "v[0] must saturate at f16 max");
        }
    }
    assert_close(
        "radam_step_f16state weights",
        &w_dev.to_host_vec(&stream)?,
        &w_cpu,
        TOL,
    );
    let m_gpu: Vec<f32> = m_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let m_exp: Vec<f32> = m_cpu.iter().map(|&x| x as f32).collect();
    assert_close_rel("radam_step_f16state m", &m_gpu, &m_exp, 2e-3);
    let v_gpu: Vec<f32> = v_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let v_exp: Vec<f32> = v_cpu.iter().map(|&x| x as f32).collect();
    assert_close_rel("radam_step_f16state v", &v_gpu, &v_exp, 2e-3);
    Ok(())
}

/// `radam_step_f16state_mirror` = f16 state + f16 mirror 同時書き込み。state 側は
/// `radam_step_f16state` と同 math なので 1 step に絞り、mirror == f16(GPU weights)
/// の同値性を中心に確認する。
#[test]
fn radam_step_f16state_mirror_matches_mimic() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let n = 131_usize;
    let (weights, m_h, v_h, grads) = f16state_fixture(n);
    let (step_size, denom) = radam_compute_step_size_denom(1000, BETA1, BETA2, N_SMA_THRESHOLD);
    let lr = 8.75e-4_f32;
    let decay = RANGER_DEFAULTS.decay;

    let mut w_cpu = weights.clone();
    let mut m_cpu = m_h.clone();
    let mut v_cpu = v_h.clone();
    let mut g_cpu = grads[0].clone();
    radam_step_f16state_mimic(
        &mut w_cpu,
        &mut m_cpu,
        &mut v_cpu,
        &mut g_cpu,
        lr,
        step_size,
        denom,
        decay,
        BETA1,
        BETA2,
        EPS,
        W_CLAMP_NONE_MIN,
        W_CLAMP_NONE_MAX,
        FT_OPT_M_SCALE,
        FT_OPT_V_SCALE,
        n,
    );

    let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
    let mut m_dev = DeviceBuffer::from_host(&stream, &m_h)?;
    let mut v_dev = DeviceBuffer::from_host(&stream, &v_h)?;
    let mut g_dev = DeviceBuffer::from_host(&stream, &grads[0])?;
    let mut mirror_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: radam_step_f16state_mirror, stream: stream, module: module, config: cfg_1d(n),
            args: [slice_mut(w_dev), slice_mut(m_dev), slice_mut(v_dev), slice_mut(g_dev),
                   slice_mut(mirror_dev),
                   lr, step_size, denom, decay, BETA1, BETA2, EPS,
                   W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX,
                   FT_OPT_M_SCALE, FT_OPT_V_SCALE, n as u32]
        }
    }?;
    stream.synchronize()?;
    let w_gpu = w_dev.to_host_vec(&stream)?;
    assert_close("radam_step_f16state_mirror weights", &w_gpu, &w_cpu, TOL);
    let m_gpu: Vec<f32> = m_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let m_exp: Vec<f32> = m_cpu.iter().map(|&x| x as f32).collect();
    assert_close_rel("radam_step_f16state_mirror m", &m_gpu, &m_exp, 2e-3);
    let v_gpu: Vec<f32> = v_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let v_exp: Vec<f32> = v_cpu.iter().map(|&x| x as f32).collect();
    assert_close_rel("radam_step_f16state_mirror v", &v_gpu, &v_exp, 2e-3);
    let mirror: Vec<f32> = mirror_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let (_, w_gpu_q) = quantize_f16(&w_gpu);
    assert_close("radam_step_f16state_mirror mirror", &mirror, &w_gpu_q, 0.0);
    Ok(())
}

#[test]
fn ranger_lookahead_lerp_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let n = 257_usize;
    let weights = deterministic_floats(n, 19.0);
    let slow = deterministic_floats(n, 23.0);
    let mut w_cpu = weights.clone();
    let mut s_cpu = slow.clone();
    ranger_lookahead_lerp_cpu(&mut w_cpu, &mut s_cpu, RANGER_ALPHA, n);

    let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
    let mut s_dev = DeviceBuffer::from_host(&stream, &slow)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ranger_lookahead_lerp, stream: stream, module: module, config: cfg_1d(n),
            args: [slice_mut(w_dev), slice_mut(s_dev), RANGER_ALPHA, n as u32]
        }
    }?;
    stream.synchronize()?;
    let w_gpu = w_dev.to_host_vec(&stream)?;
    let s_gpu = s_dev.to_host_vec(&stream)?;
    assert_close("ranger_lookahead_lerp weights", &w_gpu, &w_cpu, TOL);
    assert_close("ranger_lookahead_lerp slow", &s_gpu, &s_cpu, TOL);
    // lerp 後は weights == slow で完全同期 (同じ new_w を両 buffer へ書く)。
    assert_close("ranger_lookahead_lerp weights==slow", &w_gpu, &s_gpu, 0.0);
    Ok(())
}

#[test]
fn ranger_lookahead_lerp_fp16_mirror_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let n = 257_usize;
    let weights = deterministic_floats(n, 27.0);
    let slow = deterministic_floats(n, 29.0);
    let mut w_cpu = weights.clone();
    let mut s_cpu = slow.clone();
    ranger_lookahead_lerp_cpu(&mut w_cpu, &mut s_cpu, RANGER_ALPHA, n);

    let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
    let mut s_dev = DeviceBuffer::from_host(&stream, &slow)?;
    let mut mirror_dev = DeviceBuffer::<f16>::zeroed(&stream, n)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ranger_lookahead_lerp_fp16_mirror, stream: stream, module: module,
            config: cfg_1d(n),
            args: [slice_mut(w_dev), slice_mut(s_dev), slice_mut(mirror_dev),
                   RANGER_ALPHA, n as u32]
        }
    }?;
    stream.synchronize()?;
    let w_gpu = w_dev.to_host_vec(&stream)?;
    assert_close(
        "ranger_lookahead_lerp_fp16_mirror weights",
        &w_gpu,
        &w_cpu,
        TOL,
    );
    assert_close(
        "ranger_lookahead_lerp_fp16_mirror slow",
        &s_dev.to_host_vec(&stream)?,
        &s_cpu,
        TOL,
    );
    let mirror: Vec<f32> = mirror_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let (_, w_gpu_q) = quantize_f16(&w_gpu);
    assert_close(
        "ranger_lookahead_lerp_fp16_mirror mirror",
        &mirror,
        &w_gpu_q,
        0.0,
    );
    Ok(())
}

/// RAdam step + `step % k == 0` での lookahead lerp の 2-kernel 構成 (trainer と
/// 同じ host orchestration) が `ranger_step_cpu` と一致する。k=RANGER_K (6) で
/// 7 step 回し、lerp 発火 step (6) と非発火 step の双方を踏む。
#[test]
fn ranger_two_kernel_sequence_matches_ranger_step_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let n = 64_usize;
    let weights = deterministic_floats(n, 31.0);
    let lr = 8.75e-4_f32;
    let decay = RANGER_DEFAULTS.decay;

    let mut w_cpu = weights.clone();
    let mut m_cpu = vec![0.0_f32; n];
    let mut v_cpu = vec![0.0_f32; n];
    let mut slow_cpu = vec![0.0_f32; n];

    let mut w_dev = DeviceBuffer::from_host(&stream, &weights)?;
    let mut m_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut v_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut slow_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    for step in 1..=(RANGER_K + 1) {
        let grad: Vec<f32> = (0..n)
            .map(|i| 0.01_f32 * ((i + step as usize) % 7) as f32 - 0.02)
            .collect();
        let mut g_cpu = grad.clone();
        ranger_step_cpu(
            &mut w_cpu,
            &mut m_cpu,
            &mut v_cpu,
            &mut g_cpu,
            &mut slow_cpu,
            lr,
            decay,
            BETA1,
            BETA2,
            EPS,
            W_CLAMP_QUANT_MIN,
            W_CLAMP_QUANT_MAX,
            N_SMA_THRESHOLD,
            RANGER_ALPHA,
            step,
            RANGER_K,
            n,
        );

        let (step_size, denom) = radam_compute_step_size_denom(step, BETA1, BETA2, N_SMA_THRESHOLD);
        let mut g_dev = DeviceBuffer::from_host(&stream, &grad)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: stream, module: module, config: cfg_1d(n),
                args: [slice_mut(w_dev), slice_mut(m_dev), slice_mut(v_dev), slice_mut(g_dev),
                       lr, step_size, denom, decay, BETA1, BETA2, EPS,
                       W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, n as u32]
            }
        }?;
        if step.is_multiple_of(RANGER_K) {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: stream, module: module, config: cfg_1d(n),
                    args: [slice_mut(w_dev), slice_mut(slow_dev), RANGER_ALPHA, n as u32]
                }
            }?;
        }
    }
    stream.synchronize()?;
    assert_close(
        "ranger_sequence weights",
        &w_dev.to_host_vec(&stream)?,
        &w_cpu,
        TOL,
    );
    assert_close(
        "ranger_sequence m",
        &m_dev.to_host_vec(&stream)?,
        &m_cpu,
        TOL,
    );
    assert_close(
        "ranger_sequence v",
        &v_dev.to_host_vec(&stream)?,
        &v_cpu,
        TOL,
    );
    assert_close(
        "ranger_sequence slow",
        &slow_dev.to_host_vec(&stream)?,
        &slow_cpu,
        TOL,
    );
    Ok(())
}

// -- loss_wdl (sigmoid + WDL blend loss) ----------------------------------

/// `loss_wdl` が CPU reference と一致する。lambda の両端 (純 score sigmoid / 純 WDL)
/// と blend 中間値を踏む。GPU は `per_pos_norm` を scalar で受けるが CPU reference は
/// per-position slice なので uniform fill で合わせる。
#[test]
fn loss_wdl_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let out = vec![0.3_f32, -0.8, 2.5, -0.05, 120.0];
    let score = vec![150.0_f32, -1200.0, 30.0, 5000.0, -45.0];
    let wdl = vec![1.0_f32, 0.0, 0.5, 1.0, 0.0];
    let b = out.len();
    let per_pos_norm = 1.0_f32 / b as f32;
    let scale = 1.0_f32 / 290.0;

    let out_dev = DeviceBuffer::from_host(&stream, &out)?;
    let score_dev = DeviceBuffer::from_host(&stream, &score)?;
    let wdl_dev = DeviceBuffer::from_host(&stream, &wdl)?;
    for &lambda in &[0.0_f32, 0.7, 1.0] {
        let mut dl_cpu = vec![0.0_f32; b];
        let mut loss_cpu = 0.0_f64;
        loss_wdl_cpu(
            &out,
            &score,
            &wdl,
            &vec![per_pos_norm; b],
            &mut dl_cpu,
            &mut loss_cpu,
            lambda,
            scale,
            b,
        );

        let mut dl_dev = DeviceBuffer::<f32>::zeroed(&stream, b)?;
        let loss_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: loss_wdl, stream: stream, module: module, config: cfg_1d(b),
                args: [slice(out_dev), slice(score_dev), slice(wdl_dev), per_pos_norm,
                       slice_mut(dl_dev), slice(loss_dev), lambda, scale, b as u32]
            }
        }?;
        stream.synchronize()?;
        // libdevice exp と std exp の差で ~ulp レベルずれるため relative tolerance
        // (loss_wrm と同方針)。
        assert_close_rel(
            &format!("loss_wdl/lambda={lambda} grad"),
            &dl_dev.to_host_vec(&stream)?,
            &dl_cpu,
            1e-4,
        );
        let loss_gpu = loss_dev.to_host_vec(&stream)?[0];
        let diff = (loss_gpu - loss_cpu).abs();
        assert!(
            diff <= 1e-4 * (1.0 + loss_cpu.abs()),
            "loss_wdl/lambda={lambda} loss: gpu={loss_gpu} cpu={loss_cpu} diff={diff}"
        );
    }
    Ok(())
}

// -- PSQT shortcut (per-feature × per-bucket スカラー prior) ---------------

#[test]
fn psqt_diff_sparse_fwd_inplace_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, max_active, nb, ft_in) = (12_usize, 4_usize, DEFAULT_NUM_BUCKETS, 11_usize);
    let psqt_w = deterministic_floats(ft_in * nb, 33.0);
    let (stm_indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, ft_in, 0);
    let (nstm_indices, _) = sparse_indices_fixture_ragged(batch, max_active, ft_in, 4);
    // 全 9 bucket + `-1` / `>= nb` の invalid bucket (skip) を踏む。
    let bucket_idx = bucket_idx_with_padding(batch, nb);
    let net0 = deterministic_floats(batch, 37.0);

    let mut net_cpu = net0.clone();
    psqt_diff_sparse_fwd_inplace_cpu(
        &psqt_w,
        &stm_indices,
        &nstm_indices,
        &nnz,
        &bucket_idx,
        &mut net_cpu,
        batch,
        max_active,
        nb,
        ft_in,
    );

    let w_dev = DeviceBuffer::from_host(&stream, &psqt_w)?;
    let stm_dev = DeviceBuffer::from_host(&stream, &stm_indices)?;
    let nstm_dev = DeviceBuffer::from_host(&stream, &nstm_indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
    let mut net_dev = DeviceBuffer::from_host(&stream, &net0)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: psqt_diff_sparse_fwd_inplace, stream: stream, module: module,
            config: cfg_1d(batch),
            args: [slice(w_dev), slice(stm_dev), slice(nstm_dev), slice(nnz_dev),
                   slice(bidx_dev), slice_mut(net_dev), batch as u32, max_active as u32,
                   nb as u32, ft_in as u32]
        }
    }?;
    stream.synchronize()?;
    assert_close(
        "psqt_diff_sparse_fwd_inplace",
        &net_dev.to_host_vec(&stream)?,
        &net_cpu,
        TOL,
    );
    Ok(())
}

/// ragged fixture で `nnz_arr` early-out を exercise し、実長超 slot を範囲内のゴミで
/// 汚しても勾配が不変 (per-slot kernel が `nnz` までしか読まない) ことを確認する。
#[test]
fn psqt_diff_sparse_bwd_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let (_ctx, module, stream) = open_module()?;
    let (batch, max_active, nb, ft_in) = (12_usize, 4_usize, DEFAULT_NUM_BUCKETS, 11_usize);
    let dnet = deterministic_floats(batch, 41.0);
    let (mut stm_indices, nnz) = sparse_indices_fixture_ragged(batch, max_active, ft_in, 0);
    let (mut nstm_indices, _) = sparse_indices_fixture_ragged(batch, max_active, ft_in, 4);
    let bucket_idx = bucket_idx_with_padding(batch, nb);
    // 実長超 slot を範囲内 index で汚す (全幅走査すると勾配が変わる)。
    for bi in 0..batch {
        for ni in nnz[bi] as usize..max_active {
            stm_indices[bi * max_active + ni] = ((bi + ni + 1) % ft_in) as i32;
            nstm_indices[bi * max_active + ni] = ((bi + ni + 2) % ft_in) as i32;
        }
    }

    let mut grad_cpu = vec![0.0_f32; ft_in * nb];
    psqt_diff_sparse_bwd_cpu(
        &dnet,
        &stm_indices,
        &nstm_indices,
        &nnz,
        &bucket_idx,
        &mut grad_cpu,
        batch,
        max_active,
        nb,
        ft_in,
    );

    let dnet_dev = DeviceBuffer::from_host(&stream, &dnet)?;
    let stm_dev = DeviceBuffer::from_host(&stream, &stm_indices)?;
    let nstm_dev = DeviceBuffer::from_host(&stream, &nstm_indices)?;
    let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
    let bidx_dev = DeviceBuffer::from_host(&stream, &bucket_idx)?;
    let grad_dev = DeviceBuffer::<f32>::zeroed(&stream, ft_in * nb)?;
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: psqt_diff_sparse_bwd, stream: stream, module: module,
            config: cfg_1d(batch * max_active),
            args: [slice(dnet_dev), slice(stm_dev), slice(nstm_dev), slice(nnz_dev),
                   slice(bidx_dev), slice(grad_dev), batch as u32, max_active as u32,
                   nb as u32, ft_in as u32]
        }
    }?;
    stream.synchronize()?;
    // ±0.5*dnet の atomic scatter は加算順非決定 → relative tolerance。
    assert_close_rel(
        "psqt_diff_sparse_bwd (実長超のゴミを無視)",
        &grad_dev.to_host_vec(&stream)?,
        &grad_cpu,
        TOL,
    );
    Ok(())
}

// =========================================================================
// raw checkpoint save → load roundtrip (LayerStack、PSQT あり / なし)
// =========================================================================

/// LayerStack trainer を数 step 進めて optimizer state (m/v/slow) を非零にし、
/// raw checkpoint save → 別 instance load → 全 group bit 一致 + resume metadata
/// (superbatch / producer run id / lr_horizon) 復元を確認する。PSQT 有無で group
/// 数 (10/11) と topology 次元数 (4/5) が変わる分岐を両方踏む。
fn layerstack_raw_ckpt_roundtrip(with_psqt: bool) -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_layerstack::{GpuTrainer, OptimGroupConfig};
    use nnue_train::init::LayerStackInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // halfkp は公開 feature set 中最小の ft_in。ft_out は gather kernel の
    // grid y 軸単位である 128 の最小値で、buffer 容量を test 向けに抑える。
    let feature_set = FeatureSet::HalfKp.spec();
    let ft_out = 128;
    let psqt_init: Option<Vec<f32>> =
        with_psqt.then(|| deterministic_floats(feature_set.ft_in() * DEFAULT_NUM_BUCKETS, 7.0));
    let new_trainer = || -> Result<GpuTrainer, Box<dyn std::error::Error>> {
        GpuTrainer::new(
            &ctx,
            SMOKE_BATCH,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            nnue_train::dataloader::BucketMode::Progress8KpAbs,
            PrecisionFlags::default(),
            feature_set,
            OptimizerKind::Ranger,
            OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
            None,
            psqt_init.as_deref(),
            &LayerStackInit::default_uniform(),
        )
    };

    let mut saver = new_trainer()?;
    // Ranger lookahead の lerp (`step % k == 0`) を 1 回踏ませて slow weight も
    // 非零にする (k = RANGER_K)。
    let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
    for _ in 0..RANGER_K {
        let loss = saver.step(&batch.as_ref(), 1e-3, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        assert!(loss.is_finite(), "training step loss must be finite");
    }

    let suffix = if with_psqt { "psqt" } else { "nopsqt" };
    let path = std::env::temp_dir().join(format!(
        "layerstack-roundtrip-{suffix}-{}.ckpt",
        std::process::id()
    ));
    saver.save_raw_checkpoint(&path, 3, "roundtrip-test", Some(42))?;

    let mut loader = new_trainer()?;
    let (superbatch, producer, lr_horizon) = loader.load_raw_checkpoint(&path)?;
    assert_eq!(superbatch, 3);
    assert_eq!(producer.as_deref(), Some("roundtrip-test"));
    assert_eq!(lr_horizon, Some(42));

    // 全 group (PSQT 有効時は psqt_w 含む 11、無効時 10) の w/m/v/slow が bit 一致。
    let src_a = saver.raw_ckpt_group_sources();
    let src_b = loader.raw_ckpt_group_sources();
    assert_eq!(src_a.len(), if with_psqt { 11 } else { 10 });
    assert_eq!(src_b.len(), src_a.len());
    for (a, b) in src_a.iter().zip(src_b.iter()) {
        assert_eq!(a.name, b.name);
        let (aw, am, av, aslow) = a.to_host(&stream)?;
        let (bw, bm, bv, bslow) = b.to_host(&stream)?;
        for (label, x, y) in [
            ("w", &aw, &bw),
            ("m", &am, &bm),
            ("v", &av, &bv),
            ("slow", &aslow, &bslow),
        ] {
            assert_eq!(x.len(), y.len(), "group {} {label}: len mismatch", a.name);
            for (i, (xa, xb)) in x.iter().zip(y.iter()).enumerate() {
                assert!(
                    xa.to_bits() == xb.to_bits(),
                    "group {} {label}[{i}]: saver={xa:?} loader={xb:?} (bit mismatch)",
                    a.name
                );
            }
        }
    }

    // load 後に同条件で書き戻すと file が byte 一致する (`step_count` を含む header
    // と全 group の roundtrip が format 上も無損失である検証)。
    let path2 = std::env::temp_dir().join(format!(
        "layerstack-roundtrip-{suffix}-resave-{}.ckpt",
        std::process::id()
    ));
    loader.save_raw_checkpoint(&path2, 3, "roundtrip-test", Some(42))?;
    let bytes1 = std::fs::read(&path)?;
    let bytes2 = std::fs::read(&path2)?;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&path2);
    assert_eq!(bytes1, bytes2, "save → load → save must be byte-identical");
    Ok(())
}

#[test]
fn layerstack_raw_ckpt_roundtrips_without_psqt() -> Result<(), Box<dyn std::error::Error>> {
    layerstack_raw_ckpt_roundtrip(false)
}

#[test]
fn layerstack_raw_ckpt_roundtrips_with_psqt() -> Result<(), Box<dyn std::error::Error>> {
    layerstack_raw_ckpt_roundtrip(true)
}

/// Simple トレーナ + FT factorizer の end-to-end 配線を確認する。factorizer 有効時に
/// (1) FT master が train 形状 (実行 + piece-input 仮想行) で確保される、(2) backward の
/// reduce が仮想行 grad を埋め optimizer が仮想行を更新する (仮想行が非ゼロになる)、
/// (3) step / weight が finite、(4) 量子化 export は仮想行を実行へ畳み込んで base 形状に
/// なり id から factorizer modifier が外れ、非 factorize 網と同一 byte 長になる。
///
/// 3 活性化 (crelu / screlu / pairwise) × 精度 3 構成 (FP32 / `--ft-fp16` / `--all-optim`
/// = ft_fp16 + ft_fp16_out + fp16_opt_state + tf32) を網羅し、f16 comb 経路と f16
/// optimizer state 下でも仮想行の集約・更新・fold が成立することを確認する。
#[test]
fn simple_trainer_ft_factorize_end_to_end() -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId, SimpleWeights};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let base_spec = FeatureSet::HalfKaHmMerged.spec();
    let fac_spec = base_spec.with_ft_factorize();
    assert!(fac_spec.ft_factorize());
    assert!(
        fac_spec.train_ft_in() > fac_spec.ft_in(),
        "factorize train_ft_in must exceed base ft_in"
    );
    let ft_out = 256_usize;

    let precisions: [(&str, PrecisionFlags); 3] = [
        ("fp32", PrecisionFlags::default()),
        (
            "ft_fp16",
            PrecisionFlags {
                ft_fp16: true,
                ft_fp16_out: false,
                fp16_opt_state: false,
                tf32: false,
            },
        ),
        (
            "all-optim",
            PrecisionFlags {
                ft_fp16: true,
                ft_fp16_out: true,
                fp16_opt_state: true,
                tf32: true,
            },
        ),
    ];

    for activation in [
        SimpleActivation::CReLU,
        SimpleActivation::SCReLU,
        SimpleActivation::Pairwise,
    ] {
        let act = activation.canonical_name();
        let base_id = SimpleId {
            feature_set: base_spec,
            activation,
            ft_out,
            l1_out: 32,
            l2_out: 32,
        };
        // 非 factorize 網の量子化 byte 長 (export 同形の基準)。pairwise は combined_dim が
        // 半減し L1 形状が変わるので活性化ごとに基準を取り直す。
        let base_bytes_len = {
            let trainer = SimpleGpuTrainer::new(
                &ctx,
                SMOKE_BATCH,
                base_id,
                OptimizerKind::Ranger,
                1e-7,
                None,
                16,
                PrecisionFlags::default(),
                &init,
            )?;
            let mut bytes = Vec::new();
            trainer.to_simple_weights()?.save_quantised(&mut bytes)?;
            bytes.len()
        };

        for (label, precision) in precisions {
            let id = SimpleId {
                feature_set: fac_spec,
                ..base_id
            };
            let mut trainer = SimpleGpuTrainer::new(
                &ctx,
                SMOKE_BATCH,
                id,
                OptimizerKind::Ranger,
                1e-7,
                None,
                16,
                precision,
                &init,
            )?;
            trainer.sync_ft_forward_weights()?;

            // ft_w master は train 形状 (実行 + 仮想行)。
            assert_eq!(
                trainer.ft_w_to_host()?.len(),
                fac_spec.train_ft_in() * ft_out,
                "{act}/{label}: ft_w must be train-shaped"
            );

            let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
            for s in batch.score.iter_mut() {
                *s = 200.0;
            }
            for w in batch.wdl.iter_mut() {
                *w = 0.8;
            }
            for _ in 0..8 {
                let loss = trainer.step(&batch.as_ref(), 1e-1, 0.0, SMOKE_LOSS_SIGMOID)?;
                assert!(loss.is_finite(), "{act}/{label}: loss {loss} not finite");
            }
            trainer.assert_all_weights_finite()?;

            // reduce + optimizer が仮想行を更新している (train block 末尾が非ゼロ)。仮想行は
            // 0 初期化なので非ゼロ = reduce が grad を埋め optimizer が master を動かした証拠。
            // f16 optimizer state (`--fp16-opt-state`) では微小勾配が f16 で丸まり数 step では
            // 仮想行が動かないことがある (長時間学習では動く) ので、f32 moment state の構成
            // (FP32 / `--ft-fp16`) でのみ要求する。wiring の証明には十分。
            if !precision.fp16_opt_state {
                let ft_w_host = trainer.ft_w_to_host()?;
                let virtual_block = &ft_w_host[fac_spec.ft_in() * ft_out..];
                assert!(
                    virtual_block.iter().any(|&x| x != 0.0),
                    "{act}/{label}: virtual rows stayed zero (reduce/optimizer not wired)"
                );
            }

            // export: 仮想行 fold 後の base 形状 + factorizer modifier を外した id、
            // 量子化 byte 長は非 factorize 網と同形。
            let weights = trainer.to_simple_weights()?;
            assert!(
                !weights.id.feature_set.ft_factorize(),
                "{act}/{label}: exported id must drop the factorizer modifier"
            );
            assert_eq!(
                weights.ft_w.len(),
                fac_spec.ft_in() * ft_out,
                "{act}/{label}: exported ft_w must be base-shaped"
            );
            let mut bytes = Vec::new();
            weights.save_quantised(&mut bytes)?;
            assert_eq!(
                bytes.len(),
                base_bytes_len,
                "{act}/{label}: factorized export must be byte-identical in shape to a non-factorized net"
            );
            // base id で reload できる (推論側は factorizer を観測しない)。
            SimpleWeights::load(&mut std::io::Cursor::new(&bytes), base_id)?;
        }
    }
    Ok(())
}

/// raw checkpoint の resume は factorizer の on/off 不一致を reject する (header の
/// ft_factorize flag / train_ft_in が異なるため)。Simple でも [`ckpt`] の照合が効くことを
/// 両方向で確認する。
#[test]
fn simple_ft_factorize_resume_rejects_on_off_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let base_id = SimpleId {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    let fac_id = SimpleId {
        feature_set: base_id.feature_set.with_ft_factorize(),
        ..base_id
    };
    let mk = |id: SimpleId| -> Result<SimpleGpuTrainer, Box<dyn std::error::Error>> {
        SimpleGpuTrainer::new(
            &ctx,
            SMOKE_BATCH,
            id,
            OptimizerKind::Ranger,
            1e-7,
            None,
            16,
            PrecisionFlags::default(),
            &init,
        )
    };

    let dir = std::env::temp_dir();
    let fac_path = dir.join(format!("simple-fac-{}.ckpt", std::process::id()));
    let base_path = dir.join(format!("simple-base-{}.ckpt", std::process::id()));
    mk(fac_id)?.save_raw_checkpoint(&fac_path, 1, "t", None)?;
    mk(base_id)?.save_raw_checkpoint(&base_path, 1, "t", None)?;

    // factorize checkpoint を非 factorize trainer に load → reject。
    assert!(
        mk(base_id)?.load_raw_checkpoint(&fac_path).is_err(),
        "loading a factorized checkpoint into a non-factorized trainer must be rejected"
    );
    // 非 factorize checkpoint を factorize trainer に load → reject。
    assert!(
        mk(fac_id)?.load_raw_checkpoint(&base_path).is_err(),
        "loading a non-factorized checkpoint into a factorized trainer must be rejected"
    );

    let _ = std::fs::remove_file(&fac_path);
    let _ = std::fs::remove_file(&base_path);
    Ok(())
}

/// Simple トレーナの norm-loss が期待式どおり FT weight group を 1 へ緩めることを確認する。
/// weight_decay=0 かつ grad=0 (fresh trainer) で run_optimizer_step を 1 回呼ぶと radam は
/// no-op (更新量 ∝ grad=0、wd 項も 0) なので、ft_w への効果は norm-loss のみ。factorizer
/// 有効 (train 形状) にして FT group が train_ft_in 行 (piece-input 仮想行込み) をまたぐこと
/// もあわせて確認する。CPU 参照は norm_loss_compute_norms_cpu + norm_loss_apply_cpu。
#[test]
fn simple_norm_loss_step_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let fac_spec = FeatureSet::HalfKaHmMerged.spec().with_ft_factorize();
    let ft_out = 256_usize;
    let id = SimpleId {
        feature_set: fac_spec,
        activation: SimpleActivation::CReLU,
        ft_out,
        l1_out: 32,
        l2_out: 32,
    };
    let train_ft_in = fac_spec.train_ft_in();
    let lr = 0.1_f32;
    let factor = 0.05_f32;

    let mut trainer = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        OptimizerKind::Ranger,
        0.0,
        Some(factor),
        16,
        PrecisionFlags::default(),
        &init,
    )?;
    // CPU 参照: FT group は per-output-neuron (ft_out groups, pitch=1, stride=ft_out,
    // len=train_ft_in)。仮想行 (0 初期化) も同じ group に含まれる。
    let mut expected = trainer.ft_w_to_host()?;
    let mut norms = vec![0.0_f32; ft_out];
    norm_loss_compute_norms_cpu(&expected, &mut norms, ft_out, 1, ft_out, train_ft_in);
    norm_loss_apply_cpu(
        &mut expected,
        &norms,
        factor,
        lr,
        EPS,
        ft_out,
        1,
        ft_out,
        train_ft_in,
    );

    trainer.run_optimizer_step(lr)?;
    assert_close_rel(
        "simple norm-loss ft_w",
        &trainer.ft_w_to_host()?,
        &expected,
        TOL,
    );

    // factor=0 は no-op (norm-loss apply が ×1.0、radam も grad=0/wd=0 で no-op)。
    let mut trainer0 = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        OptimizerKind::Ranger,
        0.0,
        Some(0.0),
        16,
        PrecisionFlags::default(),
        &init,
    )?;
    let before = trainer0.ft_w_to_host()?;
    trainer0.run_optimizer_step(lr)?;
    assert_eq!(
        trainer0.ft_w_to_host()?,
        before,
        "factor=0 norm-loss must be a no-op"
    );
    Ok(())
}

/// Simple トレーナ実体を 3 種の optimizer で 6 step 駆動し、weight decay 経由で
/// per-step scalar (`step_size`) が種別どおり GPU kernel に届くことを確認する。
/// grad=0 (fresh trainer) では m/v が 0 のままなので更新量は decoupled decay
/// `w *= 1 - decay * lr * step_size_t` のみになり、CPU で厳密に再現できる。
/// step_size_t は ranger (beta1=0.99) / radam (beta1=0.9) / adamw (定数 1) で
/// 全て異なるため、種別の取り違え・lookahead gate の誤発火はここで検出される。
/// ranger は step 6 で lookahead lerp が 1 回入り、slow (= 初期 weight、Simple の
/// `slow_param ← param` 初期化規約) と補間される。
///
/// precision variant (`--ft-fp16` / `--fp16-opt-state`) もループし、
/// `radam_step_fp16_mirror` / `radam_step_f16state_mirror` の launch が同じ
/// per-step scalar を受けることを確認する (weight master は f32 のまま、moment の
/// `f16` 格納も 0 のままなので期待軌跡は variant 間で共通)。
#[test]
fn simple_optimizer_kinds_drive_expected_decay_trajectory() -> Result<(), Box<dyn std::error::Error>>
{
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let id = SimpleId {
        feature_set: FeatureSet::HalfKp.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    let lr = 0.1_f32;
    let decay = 0.01_f32;
    let steps = RANGER_K;
    assert_eq!(steps, 6, "expected exactly one lookahead lerp for ranger");

    let precisions = [
        ("f32", PrecisionFlags::default()),
        (
            "ft_fp16",
            PrecisionFlags {
                ft_fp16: true,
                ..PrecisionFlags::default()
            },
        ),
        (
            "fp16_opt_state",
            PrecisionFlags {
                fp16_opt_state: true,
                ..PrecisionFlags::default()
            },
        ),
        (
            "ft_fp16+fp16_opt_state",
            PrecisionFlags {
                ft_fp16: true,
                fp16_opt_state: true,
                ..PrecisionFlags::default()
            },
        ),
    ];

    for (label, precision) in precisions {
        for kind in [
            OptimizerKind::Ranger,
            OptimizerKind::RAdam,
            OptimizerKind::AdamW,
        ] {
            let mut trainer = SimpleGpuTrainer::new(
                &ctx,
                SMOKE_BATCH,
                id,
                kind,
                decay,
                None,
                16,
                precision,
                &init,
            )?;
            let w0 = trainer.ft_w_to_host()?;
            for _ in 0..steps {
                trainer.run_optimizer_step(lr)?;
            }
            let got = trainer.ft_w_to_host()?;

            // CPU 参照: kernel と同じ f32 演算順序で decay 係数を 1 step ずつ適用する。
            let mut factor = 1.0_f32;
            for step in 1..=steps {
                let (step_size, _denom) = kind.step_size_denom(step, BETA2, N_SMA_THRESHOLD);
                factor *= 1.0_f32 - decay * (lr * step_size);
            }
            for (i, (&g, &w)) in got.iter().zip(&w0).enumerate() {
                let mut expected = w * factor;
                if kind.uses_lookahead() {
                    expected = RANGER_ALPHA * expected + (1.0_f32 - RANGER_ALPHA) * w;
                }
                assert!(
                    (g - expected).abs() <= expected.abs() * 1e-5 + 1e-7,
                    "{label}/{kind:?} ft_w[{i}]: got {g} exp {expected}"
                );
            }
        }
    }
    Ok(())
}

/// Simple トレーナ実体を lr=0 で 1 step 駆動し、`beta1` が種別どおり kernel に
/// 届くことを 1st moment で確認する。weight 不変 + 決定的 init により grad `g` は
/// 種別間で (丸め順序を除き) 同一なので `m = (1 - beta1) * g` に閉じ、
/// `m_ranger * (1 - 0.9) ≈ m_radam * (1 - 0.99)` (elementwise) と
/// `m_radam ≈ m_adamw` が成り立つ。
///
/// FT の m を precision 4 variant で観測することで、`ft_w` の radam_step
/// 4 kernel variant (`radam_step` / `radam_step_fp16_mirror` /
/// `radam_step_f16state` / `radam_step_f16state_mirror`) 全ての beta1 配線を
/// runtime 検証する。`l1_w` (常に f32 `radam_step`) は f32 iteration でのみ確認。
/// FT grad は sparse backward の atomic 累積で丸め順序が run 間で変わり得るため、
/// 種別間比較も許容誤差付き (f16 格納 variant はさらに量子化誤差を上乗せ)。
#[test]
fn simple_beta1_follows_optimizer_kind() -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let id = SimpleId {
        feature_set: FeatureSet::HalfKp.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    // smoke_dummy の score=0 / wdl=0.5 は sigmoid loss の最小点に近く勾配が
    // 消えるので、score を動かして学習信号 (非ゼロ grad) を作る。
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    for s in batch.score.iter_mut() {
        *s = 200.0;
    }

    let precisions = [
        ("f32", PrecisionFlags::default(), 1e-4_f32),
        (
            "ft_fp16",
            PrecisionFlags {
                ft_fp16: true,
                ..PrecisionFlags::default()
            },
            1e-4,
        ),
        (
            "fp16_opt_state",
            PrecisionFlags {
                fp16_opt_state: true,
                ..PrecisionFlags::default()
            },
            1e-2,
        ),
        (
            "ft_fp16+fp16_opt_state",
            PrecisionFlags {
                ft_fp16: true,
                fp16_opt_state: true,
                ..PrecisionFlags::default()
            },
            1e-2,
        ),
    ];

    let c_ranger = 1.0_f32 - OptimizerKind::Ranger.beta1();
    let c_radam = 1.0_f32 - OptimizerKind::RAdam.beta1();

    for (label, precision, tol) in precisions {
        let mut ft_moments: Vec<(OptimizerKind, Vec<f32>)> = Vec::new();
        let mut l1_moments: Vec<(OptimizerKind, Vec<f32>)> = Vec::new();
        for kind in [
            OptimizerKind::Ranger,
            OptimizerKind::RAdam,
            OptimizerKind::AdamW,
        ] {
            let mut trainer = SimpleGpuTrainer::new(
                &ctx,
                SMOKE_BATCH,
                id,
                kind,
                0.0,
                None,
                16,
                precision,
                &init,
            )?;
            let loss = trainer.step(&batch.as_ref(), 0.0, 0.0, SMOKE_LOSS_SIGMOID)?;
            assert!(loss.is_finite(), "{label}/{kind:?}: loss must be finite");
            ft_moments.push((kind, trainer.ft_w_m_to_host()?));
            l1_moments.push((kind, trainer.l1_w_m_to_host()?));
        }

        let check = |moments: &[(OptimizerKind, Vec<f32>)], tensor: &str| {
            let m_of = |k: OptimizerKind| -> &Vec<f32> {
                &moments.iter().find(|(kind, _)| *kind == k).unwrap().1
            };
            let m_ranger = m_of(OptimizerKind::Ranger);
            let m_radam = m_of(OptimizerKind::RAdam);
            let m_adamw = m_of(OptimizerKind::AdamW);
            let max_abs = m_radam.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
            assert!(
                max_abs > 0.0,
                "{label}: {tensor} grads must be non-trivial for the m check"
            );
            for (i, (&ma, &md)) in m_adamw.iter().zip(m_radam).enumerate() {
                assert!(
                    (ma - md).abs() <= max_abs * tol,
                    "{label} {tensor} m[{i}]: adamw {ma} vs radam {md} (both beta1=0.9)"
                );
            }
            for (i, (&mr, &md)) in m_ranger.iter().zip(m_radam).enumerate() {
                assert!(
                    (mr * c_radam - md * c_ranger).abs() <= max_abs * c_radam * tol,
                    "{label} {tensor} m[{i}]: ranger {mr} vs radam {md} violate the \
                     (1 - beta1) ratio"
                );
            }
        };
        check(&ft_moments, "ft_w");
        if label == "f32" {
            check(&l1_moments, "l1_w");
        }
    }
    Ok(())
}

/// Simple トレーナ実体を lr>0 の 1 step で駆動し、`denom` が種別どおり kernel に
/// 届くことを weight 更新式で確認する。step 1 では radam (rectified schedule) は
/// n_sma が閾値未満で `denom=0` → `w -= rate * m`、adamw は `denom=1` →
/// `w -= lr * m / (sqrt(v) + eps)`。observed m から `g = m / (1 - beta1)`、
/// `v = (1 - beta2) * g^2` を復元して両式の期待値を組み立てる (decay=0)。
/// denom の取り違えは分母 `sqrt(v)` の有無で数百倍規模の差になり検出される。
#[test]
fn simple_denom_follows_optimizer_kind() -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_simple::SimpleGpuTrainer;
    use nnue_format::{SimpleActivation, SimpleId};
    use nnue_train::init::SimpleInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let init = SimpleInit::default_uniform();
    let id = SimpleId {
        feature_set: FeatureSet::HalfKp.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    for s in batch.score.iter_mut() {
        *s = 200.0;
    }
    let lr = 1e-3_f32;

    for kind in [OptimizerKind::RAdam, OptimizerKind::AdamW] {
        let mut trainer = SimpleGpuTrainer::new(
            &ctx,
            SMOKE_BATCH,
            id,
            kind,
            0.0,
            None,
            16,
            PrecisionFlags::default(),
            &init,
        )?;
        let w0 = trainer.l1_w_to_host()?;
        let loss = trainer.step(&batch.as_ref(), lr, 0.0, SMOKE_LOSS_SIGMOID)?;
        assert!(loss.is_finite(), "{kind:?}: loss must be finite");
        let w1 = trainer.l1_w_to_host()?;
        let m = trainer.l1_w_m_to_host()?;

        let (step_size, denom) = kind.step_size_denom(1, BETA2, N_SMA_THRESHOLD);
        let rate = lr * step_size;
        let max_abs_w = w0.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        let mut updated = 0usize;
        for i in 0..w0.len() {
            let mut val = m[i];
            if denom != 0 {
                let g = m[i] / (1.0_f32 - kind.beta1());
                let v = (1.0_f32 - BETA2) * g * g;
                val /= v.sqrt() + EPS;
            }
            let expected = w0[i] - rate * val;
            assert!(
                (w1[i] - expected).abs() <= max_abs_w * 1e-4 + 1e-7,
                "{kind:?} l1_w[{i}]: got {} exp {expected} (denom={denom})",
                w1[i]
            );
            if w1[i] != w0[i] {
                updated += 1;
            }
        }
        assert!(
            updated > 0,
            "{kind:?}: the step must move some l1_w weights"
        );
    }
    Ok(())
}

/// LayerStack トレーナ実体を lr=0 で RANGER_K step 駆動し、(1) lookahead lerp が
/// ranger でだけ発火すること、(2) `beta1` が種別どおり kernel に届くことを確認する。
///
/// lr=0 では radam_step の decay 項も更新項も 0 なので weight は不変のまま
/// m/v だけが実 batch の grad で更新される:
/// - weight: step 6 の lerp (LayerStack の slow は 0 初期化) だけが現れる。
///   ranger → `w = alpha * w0` (= 0.5 倍)、radam / adamw → w0 と値一致。
/// - 1st moment: weight 不変 + 決定的 init により全 step / 全種別で grad `g` が
///   同一なので、EMA は `m = (1 - beta1^6) * g` に閉じる。よって
///   `m_ranger * (1 - 0.9^6) == m_radam * (1 - 0.99^6)` (elementwise)、
///   `m_radam == m_adamw` (beta1 が同じ 0.9、m 更新は denom に依らない)。
///   beta1 の取り違え (例: 全種別 0.99) はこの比で検出される。
#[test]
fn layerstack_lookahead_and_beta1_follow_optimizer_kind() -> Result<(), Box<dyn std::error::Error>>
{
    use crate::trainer_layerstack::{GpuTrainer, OptimGroupConfig};
    use nnue_train::init::LayerStackInit;
    use shogi_features::FeatureSet;

    let ctx = CudaContext::new(0)?;
    let feature_set = FeatureSet::HalfKp.spec();
    // smoke_dummy の score=0 / wdl=0.5 は sigmoid loss の最小点に近く勾配が
    // 消えるので、score を動かして学習信号 (非ゼロ grad) を作る。
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
    for s in batch.score.iter_mut() {
        *s = 200.0;
    }

    let mut moments: Vec<(OptimizerKind, Vec<f32>)> = Vec::new();
    for kind in [
        OptimizerKind::Ranger,
        OptimizerKind::RAdam,
        OptimizerKind::AdamW,
    ] {
        // ft_out はフル次元 (1536) だと並列実行中の他 GPU テストと合わせて OOM に
        // なり得るため、checkpoint roundtrip テストと同じ縮小次元にする。
        let mut trainer = GpuTrainer::new(
            &ctx,
            SMOKE_BATCH,
            128,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            nnue_train::dataloader::BucketMode::Progress8KpAbs,
            PrecisionFlags::default(),
            feature_set,
            kind,
            OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
            None,
            None,
            &LayerStackInit::default_uniform(),
        )?;
        let w0 = trainer.to_layerstack_weights()?;
        for _ in 0..RANGER_K {
            let loss = trainer.step(&batch.as_ref(), 0.0, 0.0, SMOKE_LOSS_SIGMOID)?;
            assert!(loss.is_finite(), "{kind:?}: loss must be finite");
        }
        let w = trainer.to_layerstack_weights()?;

        if kind.uses_lookahead() {
            for (i, (&g, &e)) in w.l1_w.iter().zip(&w0.l1_w).enumerate() {
                let expected = RANGER_ALPHA * e;
                assert!(
                    (g - expected).abs() <= 1e-7,
                    "ranger l1_w[{i}]: got {g} exp {expected} (single lerp toward slow=0)"
                );
            }
        } else {
            assert_eq!(
                w.l1_w, w0.l1_w,
                "{kind:?}: lr=0 run must keep weights unchanged (no lookahead lerp)"
            );
        }
        moments.push((kind, trainer.l3_b_m_to_host()?));
    }

    let m_of =
        |k: OptimizerKind| -> &Vec<f32> { &moments.iter().find(|(kind, _)| *kind == k).unwrap().1 };
    let m_ranger = m_of(OptimizerKind::Ranger);
    let m_radam = m_of(OptimizerKind::RAdam);
    let m_adamw = m_of(OptimizerKind::AdamW);
    let max_abs = m_radam.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    assert!(
        max_abs > 0.0,
        "l3_b grads must be non-trivial for the m check"
    );
    // bias grad は atomic 累積で run 間の丸め順序が変わり得るため、bit 一致では
    // なく許容誤差で比較する (beta1 取り違えは 10 倍差なので検出力は十分)。
    for (i, (&ma, &md)) in m_adamw.iter().zip(m_radam).enumerate() {
        assert!(
            (ma - md).abs() <= max_abs * 1e-5,
            "l3_b m[{i}]: adamw {ma} vs radam {md} (both beta1=0.9)"
        );
    }
    let c_ranger = 1.0_f32 - OptimizerKind::Ranger.beta1().powi(RANGER_K as i32);
    let c_radam = 1.0_f32 - OptimizerKind::RAdam.beta1().powi(RANGER_K as i32);
    for (i, (&mr, &md)) in m_ranger.iter().zip(m_radam).enumerate() {
        let lhs = mr * c_radam;
        let rhs = md * c_ranger;
        assert!(
            (lhs - rhs).abs() <= max_abs * c_radam * 1e-5,
            "l3_b m[{i}]: ranger {mr} vs radam {md} violate the (1 - beta1^6) ratio"
        );
    }
    Ok(())
}
