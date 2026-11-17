use super::gpu_test_context::test_cuda_context;
#[cfg(feature = "oxide")]
use crate::kernels::*;
use crate::{kernel_module::load_kernel_module_with_fallback, trainer_common::cfg_1d};
use gpu_runtime::{DeviceBuffer, cuda_launch};

#[test]
fn partial_threat_gpu_matches_cpu_forward_backward() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = test_cuda_context()?;
    let stream = gpu_runtime::create_compute_stream(&ctx)?;
    let module = load_kernel_module_with_fallback(&ctx, "nnue_train")?;
    let rows = 1536;
    let indices = [0_i32, 2, 4, -1, 1, 3, 2, 2];
    let nnz = [3_i32, 4];
    let pairs = [0_u32, 1, 3];
    for width in [512, 1536] {
        for factor in [false, true] {
            for half in [false, true] {
                let real_n = 2 * rows + 3 * width;
                let n = real_n + if factor { 2 * rows + 2 * width } else { 0 };
                let weight: Vec<f32> = (0..n)
                    .map(|i| ((i * 7 % 101) as f32 - 50.0) / 113.0)
                    .collect();
                let shape = [
                    rows as u32,
                    2,
                    3,
                    width as u32,
                    4,
                    2,
                    factor as u32,
                    half as u32,
                    if factor { 2 } else { 0 },
                ];
                let w = DeviceBuffer::from_host(&stream, &weight)?;
                let idx = DeviceBuffer::from_host(&stream, &indices)?;
                let nnz_dev = DeviceBuffer::from_host(&stream, &nnz)?;
                let pair_dev = DeviceBuffer::from_host(&stream, &pairs)?;
                let shape_dev = DeviceBuffer::from_host(&stream, &shape)?;
                let mut expected = vec![0.0_f32; 2 * rows];
                for b in 0..2 {
                    for &f in &indices[b * 4..b * 4 + nnz[b] as usize] {
                        let f = f as usize;
                        let row_width = if f < 2 { rows } else { width };
                        let start = if f < 2 {
                            f * rows
                        } else {
                            2 * rows + (f - 2) * width
                        };
                        let virtual_start = if f < 2 {
                            real_n + f * rows
                        } else {
                            real_n + 2 * rows + usize::from(f > 2) * width
                        };
                        for c in 0..row_width {
                            let col = if c < row_width / 2 {
                                c
                            } else {
                                rows / 2 + c - row_width / 2
                            };
                            let mut value = weight[start + c];
                            if factor {
                                value += weight[virtual_start + c];
                            }
                            if half {
                                value = (value as f16) as f32;
                            }
                            expected[b * rows + col] += value;
                        }
                    }
                }
                let out = DeviceBuffer::<f32>::zeroed(&stream, 2 * rows)?;
                // SAFETY: all allocations follow the synthetic shape, with one writer per output cell.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_forward, stream: stream, module: module, config: cfg_1d(2 * rows),
                        args: [slice(w), slice(idx), slice(nnz_dev), slice(pair_dev), slice(out), slice(shape_dev), 2_u32]
                    }
                }?;
                assert_eq!(out.to_host_vec(&stream)?, expected);
                if half {
                    let out_h = DeviceBuffer::<f16>::zeroed(&stream, 2 * rows)?;
                    // SAFETY: same shape as the FP32-output launch.
                    unsafe {
                        cuda_launch! {
                            kernel: partial_ft_forward_fp16_out, stream: stream, module: module, config: cfg_1d(2 * rows),
                            args: [slice(w), slice(idx), slice(nnz_dev), slice(pair_dev), slice(out_h), slice(shape_dev), 2_u32]
                        }
                    }?;
                    let expected_half: Vec<f16> = expected.iter().map(|&v| v as f16).collect();
                    assert_eq!(out_h.to_host_vec(&stream)?, expected_half);
                }
                let gradient: Vec<f32> = (0..2 * rows)
                    .map(|i| ((i % 47) as f32 - 23.0) / 64.0)
                    .collect();
                let gout = DeviceBuffer::from_host(&stream, &gradient)?;
                let gout_h = DeviceBuffer::from_host(
                    &stream,
                    &gradient.iter().map(|&v| v as f16).collect::<Vec<_>>(),
                )?;
                let mut positions = Vec::<u32>::new();
                let mut offsets = vec![0_u32];
                for f in 0..5 {
                    for b in 0..2 {
                        for &index in &indices[b * 4..b * 4 + nnz[b] as usize] {
                            if index == f {
                                positions.push(b as u32);
                            }
                        }
                    }
                    offsets.push(positions.len() as u32);
                }
                let positions_dev = DeviceBuffer::from_host(&stream, &positions)?;
                let offsets_dev = DeviceBuffer::from_host(&stream, &offsets)?;
                let result = DeviceBuffer::<f32>::zeroed(&stream, n)?;
                for add in 0..2_u32 {
                    if half {
                        // SAFETY: inverse-index lists contain valid batch rows and cover every real parameter.
                        unsafe {
                            cuda_launch! {
                                kernel: partial_ft_gather_fp16, stream: stream, module: module, config: cfg_1d(real_n),
                                args: [slice(gout_h), slice(positions_dev), slice(offsets_dev), slice(result), slice(shape_dev), add, 0.5_f32]
                            }
                        }?;
                    } else {
                        // SAFETY: same bounds as the half-gradient launch.
                        unsafe {
                            cuda_launch! {
                                kernel: partial_ft_gather, stream: stream, module: module, config: cfg_1d(real_n),
                                args: [slice(gout), slice(positions_dev), slice(offsets_dev), slice(result), slice(shape_dev), add, 0.5_f32]
                            }
                        }?;
                    }
                }
                let mut expected_grad = vec![0.0; n];
                for f in 0..5 {
                    let row_width = if f < 2 { rows } else { width };
                    let start = if f < 2 {
                        f * rows
                    } else {
                        2 * rows + (f - 2) * width
                    };
                    for c in 0..row_width {
                        let col = if c < row_width / 2 {
                            c
                        } else {
                            rows / 2 + c - row_width / 2
                        };
                        for p in offsets[f]..offsets[f + 1] {
                            expected_grad[start + c] +=
                                gradient[positions[p as usize] as usize * rows + col];
                        }
                    }
                }
                if factor {
                    // SAFETY: virtual writes and real reads occupy disjoint regions of result.
                    unsafe {
                        cuda_launch! {
                            kernel: partial_ft_reduce, stream: stream, module: module, config: cfg_1d(n - real_n),
                            args: [slice(result), slice(pair_dev), slice(shape_dev)]
                        }
                    }?;
                    let real_copy = expected_grad[..real_n].to_vec();
                    expected_grad[real_n..real_n + 2 * rows]
                        .copy_from_slice(&real_copy[..2 * rows]);
                    for f in 0..3 {
                        for c in 0..width {
                            expected_grad[real_n + 2 * rows + usize::from(f > 0) * width + c] +=
                                real_copy[2 * rows + f * width + c];
                        }
                    }
                }
                assert_eq!(result.to_host_vec(&stream)?, expected_grad);
                let mut groups = vec![Vec::<usize>::new(); rows];
                for f in 0..5 {
                    let size = if f < 2 { rows } else { width };
                    let start = if f < 2 {
                        f * rows
                    } else {
                        2 * rows + (f - 2) * width
                    };
                    for c in 0..size {
                        let col = if c < size / 2 {
                            c
                        } else {
                            rows / 2 + c - size / 2
                        };
                        groups[col].push(start + c);
                    }
                }
                if factor {
                    for p in 0..2 {
                        for (c, group) in groups.iter_mut().enumerate() {
                            group.push(real_n + p * rows + c);
                        }
                        for c in 0..width {
                            let col = if c < width / 2 {
                                c
                            } else {
                                rows / 2 + c - width / 2
                            };
                            groups[col].push(real_n + 2 * rows + p * width + c);
                        }
                    }
                }
                let norms = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
                // SAFETY: norms has one cell per output column and weight follows shape.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_norm, stream: stream, module: module, config: cfg_1d(rows),
                        args: [slice(w), slice(shape_dev), slice(norms)]
                    }
                }?;
                let actual_norms = norms.to_host_vec(&stream)?;
                let expected_norms: Vec<f32> = groups
                    .iter()
                    .map(|group| {
                        group
                            .iter()
                            .map(|&k| f64::from(weight[k]).powi(2))
                            .sum::<f64>()
                            .sqrt() as f32
                    })
                    .collect();
                for (&a, &b) in actual_norms.iter().zip(&expected_norms) {
                    assert!((a - b).abs() < 1.0e-6);
                }
                // SAFETY: one writer per parameter; each reads a valid FT-column norm.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_norm_apply, stream: stream, module: module, config: cfg_1d(n),
                        args: [slice(w), slice(shape_dev), slice(norms), 0.1_f32, 0.01_f32]
                    }
                }?;
                let updated = w.to_host_vec(&stream)?;
                for (col, group) in groups.iter().enumerate() {
                    let multiplier =
                        1.0 - 0.01 * (0.2 * (1.0 - 1.0 / (expected_norms[col] + 1.0e-8)));
                    for &k in group {
                        assert!((updated[k] - weight[k] * multiplier).abs() < 1.0e-6);
                    }
                }
            }
        }
    }
    Ok(())
}
