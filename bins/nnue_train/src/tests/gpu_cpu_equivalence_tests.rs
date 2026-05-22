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
//! `gpu_kernels::layerstack::*_cpu` と比較。`-1` padding (sparse index / bucket_idx)、
//! 全 9 bucket、CReLU 境界値 (ちょうど 0.0 / 1.0 / 負)、NaN 伝搬を含む。tolerance:
//! forward / gradient 1e-5、整数/index 出力は完全一致。
//!
//! kernel ↔ CPU ref 対応表は `gpu_kernels::layerstack` の module doc 参照。

use cuda_host::cuda_launch;
use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};

use crate::*;
use crate::{arch::*, kernel_module::*, trainer_common::*};

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
use gpu_kernels::pointwise::screlu_fwd::screlu_fwd_cpu;

/// forward / gradient の f32 tolerance。
const TOL: f32 = 1e-5;

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
        cuda_launch! {
            kernel: bias_grad_shared_l1f, stream: stream, module: module,
            config: cfg_1d(batch * out_dim),
            args: [slice(dy_dev), slice(gb_dev), batch as u32, out_dim as u32]
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
    // out_dim は 16 幅 out-tile (grid_y) で消化する。16 / 16 倍数 / 非倍数を網羅。
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 16_usize, 16_usize),
        (32, 64, 32),
        (48, 96, 24),
        (64, 32, 8),
        (32, 48, 17),
    ] {
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
        let n_out_tiles = out_dim.div_ceil(16);
        cuda_launch! {
            kernel: dense_mm_fwd_bucket_tiled_l1_sorted, stream: stream, module: module,
            config: LaunchConfig {
                grid_dim: ((padded / 16) as u32, n_out_tiles as u32, 1),
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
        let cell_blocks = (out_dim * in_dim).div_ceil(256);
        let config = LaunchConfig {
            grid_dim: (cell_blocks as u32, num_splits as u32, 1),
            block_dim: (256, 1, 1),
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
        // block_dim は in_dim (= L2 出力次元) に一致させる (kernel は 1 thread = 1 ii cell)。
        let config = LaunchConfig {
            grid_dim: (num_splits as u32, 1, 1),
            block_dim: (in_dim as u32, 1, 1),
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
    // out_dim は 16 幅 out-tile を grid_x (in-tile と畳む) で消化する。16 / 倍数 / 非倍数を網羅。
    for &(batch, in_dim, out_dim) in &[
        (16_usize, 16_usize, 16_usize),
        (32, 64, 32),
        (48, 96, 24),
        (64, 32, 8),
        (32, 48, 17),
    ] {
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
                grid_dim: (((in_dim / 16) * out_dim.div_ceil(16)) as u32, 8, nb as u32),
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
            config: LaunchConfig {
                grid_dim: ((padded / 16) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
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
    // test 入力は O(数十) なので production の dft_scale は overflow する。小さい値。
    let dft_scale = 64.0_f32;
    cuda_launch! {
        kernel: ft_post_perspective_grad_fp16, stream: stream, module: module,
        config: cfg_1d(batch * ft_dim / 2),
        args: [slice(dc_dev), slice(stm_ft_dev), slice(bias_dev),
               slice_mut(dft_stm_dev), slice(grad_bias_dev),
               batch as u32, ft_dim as u32, 0_u32, ft_dim as u32, scale, dft_scale]
    }?;
    cuda_launch! {
        kernel: ft_post_perspective_grad_fp16, stream: stream, module: module,
        config: cfg_1d(batch * ft_dim / 2),
        args: [slice(dc_dev), slice(nstm_ft_dev), slice(bias_dev),
               slice_mut(dft_nstm_dev), slice(grad_bias_dev),
               batch as u32, ft_dim as u32, half as u32, ft_dim as u32, scale, dft_scale]
    }?;
    stream.synchronize()?;
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
    cuda_launch! {
        kernel: simple_bias_act_fwd_fp16_in_screlu, stream: stream, module: module,
        config: cfg_1d(batch * ft_dim),
        args: [slice(ft_out_dev), slice(bias_dev), slice_mut(acted_dev),
               batch as u32, ft_dim as u32]
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
    // test 入力 dft は O(数十) なので production の dft_scale は overflow する。小さい値。
    let dft_scale = 64.0_f32;
    cuda_launch! {
        kernel: simple_act_grad_to_fp16_screlu_with_scale, stream: stream, module: module,
        config: cfg_1d(batch * ft_dim),
        args: [slice(ft_out_dev), slice(bias_dev), slice(dft_acted_dev),
               slice_mut(dft_out_dev), batch as u32, ft_dim as u32, dft_scale]
    }?;
    stream.synchronize()?;
    let inv = 1.0_f32 / dft_scale;
    let g_gpu: Vec<f32> = dft_out_dev
        .to_host_vec(&stream)?
        .iter()
        .map(|&x| x as f32 * inv)
        .collect();
    assert_close_rel("simple_act_grad_to_fp16_screlu", &g_gpu, &g_cpu, 2e-3);
    Ok(())
}
