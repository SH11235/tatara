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
    for width in [2, 6, 12, 130, 512, 1536] {
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
                let comb = DeviceBuffer::<f32>::zeroed(&stream, real_n)?;
                // SAFETY: the packed cache covers only real rows; partners are in w's virtual block.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_fold, stream: stream, module: module, config: cfg_1d(real_n),
                        args: [slice(w), slice(pair_dev), slice(shape_dev), slice(comb)]
                    }
                }?;
                let out = DeviceBuffer::<f32>::zeroed(&stream, 2 * rows)?;
                let lanes = if cfg!(feature = "native") && width.is_multiple_of(8) {
                    4
                } else {
                    1
                };
                let order = DeviceBuffer::from_host(&stream, &[1_i32, 0])?;
                // SAFETY: all allocations follow the synthetic shape, with one writer per output cell.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_forward, stream: stream, module: module, config: cfg_1d(2 * rows / lanes),
                        args: [slice(comb), slice(idx), slice(nnz_dev), slice(out), slice(shape_dev), 2_u64, slice(order)]
                    }
                }?;
                assert_eq!(out.to_host_vec(&stream)?, expected);
                if half {
                    let out_from_fp32 =
                        DeviceBuffer::from_host(&stream, &vec![f16::NAN; 2 * rows])?;
                    // SAFETY: same shape as the FP32-output launch.
                    unsafe {
                        cuda_launch! {
                            kernel: partial_ft_forward_fp16_out, stream: stream, module: module, config: cfg_1d(2 * rows / lanes),
                            args: [slice(comb), slice(idx), slice(nnz_dev), slice(out_from_fp32), slice(shape_dev), 2_u64, slice(order)]
                        }
                    }?;
                    let expected_half: Vec<f16> = expected.iter().map(|&v| v as f16).collect();
                    assert_eq!(out_from_fp32.to_host_vec(&stream)?, expected_half);
                    #[cfg(feature = "native")]
                    {
                        let comb_h = DeviceBuffer::<f16>::zeroed(&stream, real_n)?;
                        // SAFETY: each real row owns its output and its partner lies in w.
                        unsafe {
                            cuda_launch! {
                                kernel: partial_ft_fold_to_fp16, stream: stream, module: module,
                                config: gpu_runtime::LaunchConfig {
                                    grid_dim: (5, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0,
                                },
                                args: [slice(w), slice(pair_dev), slice(shape_dev), slice(comb_h), 0_u32]
                            }
                        }?;
                        let folded_bits: Vec<u16> = comb
                            .to_host_vec(&stream)?
                            .into_iter()
                            .map(|value| (value as f16).to_bits())
                            .collect();
                        let half_bits: Vec<u16> = comb_h
                            .to_host_vec(&stream)?
                            .into_iter()
                            .map(f16::to_bits)
                            .collect();
                        assert_eq!(folded_bits, half_bits);
                        let out_h = DeviceBuffer::from_host(&stream, &vec![f16::NAN; 2 * rows])?;
                        let out_f = DeviceBuffer::from_host(&stream, &vec![f32::NAN; 2 * rows])?;
                        // SAFETY: both output arrays are sized for all ordered positions.
                        unsafe {
                            cuda_launch! {
                                kernel: partial_ft_forward_fp16_weights_out, stream: stream, module: module,
                                config: cfg_1d(2 * rows / lanes),
                                args: [slice(comb_h), slice(idx), slice(nnz_dev), slice(out_h), slice(shape_dev), 2_u64, slice(order)]
                            }?;
                            cuda_launch! {
                                kernel: partial_ft_forward_fp16_weights, stream: stream, module: module,
                                config: cfg_1d(2 * rows / lanes),
                                args: [slice(comb_h), slice(idx), slice(nnz_dev), slice(out_f), slice(shape_dev), 2_u64, slice(order)]
                            }?;
                        }
                        assert_eq!(out_h.to_host_vec(&stream)?, expected_half);
                        assert_eq!(out_f.to_host_vec(&stream)?, expected);
                    }
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
                let mut result = DeviceBuffer::<f32>::zeroed(&stream, n)?;
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
                #[cfg(feature = "native")]
                if half && width.is_multiple_of(4) {
                    let paired = DeviceBuffer::<f32>::zeroed(&stream, n)?;
                    let tiles = 2 * rows.div_ceil(128) + 3 * width.div_ceil(128);
                    for add in 0..2_u32 {
                        // SAFETY: each aligned pair stays inside a row half and tiles cover all real rows.
                        unsafe {
                            cuda_launch! {
                                kernel: partial_ft_gather_fp16_pairs, stream: stream, module: module,
                                config: gpu_runtime::LaunchConfig {
                                    grid_dim: (tiles as u32, 1, 1),
                                    block_dim: (64, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                args: [slice(gout_h), slice(positions_dev), slice(offsets_dev),
                                       slice(paired), slice(shape_dev), add, 0.5_f32]
                            }
                        }?;
                    }
                    let bits = |weights: &DeviceBuffer<f32>| {
                        weights
                            .to_host_vec(&stream)
                            .map(|values| values.into_iter().map(f32::to_bits).collect::<Vec<_>>())
                    };
                    assert_eq!(bits(&paired)?, bits(&result)?);
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
                            args: [slice_mut(result), slice(pair_dev), slice(shape_dev)]
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
                let sums = DeviceBuffer::<f64>::zeroed(&stream, rows * 256)?;
                // SAFETY: norms has one cell per output column and weight follows shape.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_norm, stream: stream, module: module, config: cfg_1d(rows * 256),
                        args: [slice(w), slice(shape_dev), slice(sums)]
                    }
                }?;
                // SAFETY: sums contains 256 partitions per column and norms has rows cells.
                unsafe {
                    cuda_launch! {
                        kernel: partial_ft_norm_finalize, stream: stream, module: module, config: cfg_1d(rows),
                        args: [slice(sums), slice(norms), rows as u32]
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

#[test]
fn partial_threat_partitioned_norm_matches_serial_f64() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = test_cuda_context()?;
    let stream = gpu_runtime::create_compute_stream(&ctx)?;
    let module = load_kernel_module_with_fallback(&ctx, "nnue_train")?;
    let rows = 32;
    let base = 513;
    let threats = 519;
    let piece = 3;
    for width in [2, 12, 32] {
        for factor in [false, true] {
            let virtual_piece = if factor { piece } else { 0 };
            let virtual_pairs = if factor { 2 } else { 0 };
            let real_n = base * rows + threats * width;
            let n = real_n + virtual_piece * rows + virtual_pairs * width;
            let weight: Vec<f32> = (0..n)
                .map(|i| ((i * 7 % 101) as f32 - 50.0) / 113.0)
                .collect();
            let shape = [
                rows as u32,
                base as u32,
                threats as u32,
                width as u32,
                4,
                piece as u32,
                factor as u32,
                0,
                virtual_pairs as u32,
            ];
            let w = DeviceBuffer::from_host(&stream, &weight)?;
            let shape_dev = DeviceBuffer::from_host(&stream, &shape)?;
            let sums = DeviceBuffer::<f64>::zeroed(&stream, rows * 256)?;
            let norms = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
            // SAFETY: all allocations match the packed shape and each output has a unique writer.
            unsafe {
                cuda_launch! {
                    kernel: partial_ft_norm, stream: stream, module: module,
                    config: cfg_1d(rows * 256),
                    args: [slice(w), slice(shape_dev), slice(sums)]
                }
            }?;
            // SAFETY: every norm reads exactly 256 initialized partial sums.
            unsafe {
                cuda_launch! {
                    kernel: partial_ft_norm_finalize, stream: stream, module: module,
                    config: cfg_1d(rows), args: [slice(sums), slice(norms), rows as u32]
                }
            }?;
            let actual = norms.to_host_vec(&stream)?;
            for (col, &norm) in actual.iter().enumerate() {
                let mut sum = 0.0_f64;
                for f in 0..base {
                    sum += f64::from(weight[f * rows + col]).powi(2);
                }
                for p in 0..virtual_piece {
                    sum += f64::from(weight[real_n + p * rows + col]).powi(2);
                }
                let c = col % (rows / 2);
                if c < width / 2 {
                    let pc = c + if col >= rows / 2 { width / 2 } else { 0 };
                    for f in 0..threats {
                        sum += f64::from(weight[base * rows + f * width + pc]).powi(2);
                    }
                    for p in 0..virtual_pairs {
                        sum += f64::from(weight[real_n + virtual_piece * rows + p * width + pc])
                            .powi(2);
                    }
                }
                let expected = sum.sqrt() as f32;
                assert!(
                    (norm - expected).abs() <= 1.0e-6 * expected,
                    "{width}/{factor}/{col}: {norm} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "native")]
#[test]
fn partial_threat_active_gather_and_reduce_preserve_gradient_bits()
-> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_layerstack::PartialGather;

    check_partial_threat_active_gather_bits(PartialGather::Row)?;
    check_partial_threat_active_gather_bits(PartialGather::Tile)
}

#[cfg(feature = "native")]
fn check_partial_threat_active_gather_bits(
    variant: crate::trainer_layerstack::PartialGather,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::trainer_layerstack::{
        PartialGather, partial_ft_gather_row_config, partial_ft_gather_tile_config,
    };

    let ctx = test_cuda_context()?;
    let stream = gpu_runtime::create_compute_stream(&ctx)?;
    let module = load_kernel_module_with_fallback(&ctx, "nnue_train")?;
    let batch = 11;
    let base = 4;
    let threats = 4;
    let piece = 2;
    let scale = 0.000091_827_f32;
    // Each section includes STM-only, NSTM-only, shared, and unused features.
    let active = [1_u32, 1, 0, 1, 1, 0, 1, 1];
    let mask = DeviceBuffer::from_host(&stream, &active)?;
    let pairs = DeviceBuffer::from_host(&stream, &[0_u32, 2, 4])?;
    let lists: [[&[u32]; 8]; 2] = [
        [
            &[3, 9, 0, 5, 3, 1, 2, 9, 10],
            &[],
            &[],
            &[8],
            &[10, 1, 7, 2, 4],
            &[],
            &[],
            &[2, 4, 5],
        ],
        [
            &[],
            &[3],
            &[],
            &[5, 5, 3, 0, 10, 7, 0],
            &[],
            &[],
            &[9, 1, 8, 6],
            &[4],
        ],
    ];
    let stale = f32::from_bits(0x7fc0_0001);
    for (rows, width) in [
        (320_usize, 192_usize),
        (320, 130),
        (321, 193),
        (1536, 512),
        (6144, 3072),
        (8192, 2816),
    ] {
        let paired = rows.is_multiple_of(4) && width.is_multiple_of(4);
        if !paired && variant == PartialGather::Tile {
            continue;
        }
        let shape = DeviceBuffer::from_host(
            &stream,
            &[
                rows as u32,
                base as u32,
                threats as u32,
                width as u32,
                0,
                piece as u32,
            ],
        )?;
        let real_len = base * rows + threats * width;
        let virtual_len = piece * rows + 2 * width;
        let reference = DeviceBuffer::from_host(&stream, &vec![stale; real_len + virtual_len])?;
        let actual = DeviceBuffer::from_host(&stream, &vec![stale; real_len + virtual_len])?;
        let scalar_config = cfg_1d(real_len);
        let pairs_config = partial_ft_gather_row_config(base, threats, rows, width)?;
        for (perspective, lists) in lists.iter().enumerate() {
            let mut offsets = vec![0_u32];
            let mut positions = Vec::new();
            for list in lists {
                positions.extend_from_slice(list);
                offsets.push(positions.len() as u32);
            }
            let offsets = DeviceBuffer::from_host(&stream, &offsets)?;
            let positions = DeviceBuffer::from_host(&stream, &positions)?;
            let hg: Vec<f16> = (0..batch * rows)
                .map(|i| {
                    let sign = ((i + perspective) % 2) as u16;
                    f16::from_bits((sign << 15) | ((i * 277 % 0x7c00) as u16))
                })
                .collect();
            let half = DeviceBuffer::from_host(&stream, &hg)?;
            let add = perspective as u32;
            // SAFETY: all positions are in the batch, the mask covers all features,
            // and both launches cover the same packed real block.
            unsafe {
                if paired {
                    cuda_launch! {
                        kernel: partial_ft_gather_fp16_pairs, stream: stream, module: module, config: pairs_config,
                        args: [slice(half), slice(positions), slice(offsets), slice(reference), slice(shape), add, scale]
                    }?;
                    if variant == PartialGather::Row {
                        cuda_launch! {
                            kernel: partial_ft_gather_fp16_pairs_active_row, stream: stream, module: module, config: pairs_config,
                            args: [slice(half), slice(positions), slice(offsets), slice(actual), slice(shape), add, scale, slice(mask)]
                        }?;
                    } else {
                        for (first, count, packed_width) in
                            [(0, base, rows), (base, threats, width)]
                        {
                            cuda_launch! {
                                kernel: partial_ft_gather_fp16_pairs_active_tile, stream: stream, module: module,
                                config: partial_ft_gather_tile_config(count, packed_width),
                                args: [slice(half), slice(positions), slice(offsets), slice(actual), slice(mask),
                                       first as u32, count as u32, rows as u32, packed_width as u32, add, scale]
                            }?;
                        }
                    }
                } else {
                    let float = DeviceBuffer::from_host(
                        &stream,
                        &hg.into_iter().map(|value| value as f32).collect::<Vec<_>>(),
                    )?;
                    cuda_launch! {
                        kernel: partial_ft_gather_fp16, stream: stream, module: module, config: scalar_config,
                        args: [slice(half), slice(positions), slice(offsets), slice(reference), slice(shape), add, scale]
                    }?;
                    cuda_launch! {
                        kernel: partial_ft_gather_active, stream: stream, module: module, config: scalar_config,
                        args: [slice(float), slice(positions), slice(offsets), slice(actual), slice(shape), add, scale, slice(mask)]
                    }?;
                    stream.synchronize()?;
                }
            }
            let reference = reference.to_host_vec(&stream)?;
            let actual = actual.to_host_vec(&stream)?;
            let mut start = 0;
            for (feature, live) in active.into_iter().enumerate() {
                let end = start + if feature < base { rows } else { width };
                for k in start..end {
                    let expected = if live == 0 { stale } else { reference[k] };
                    assert_eq!(
                        actual[k].to_bits(),
                        expected.to_bits(),
                        "{variant:?}, {rows}/{width}, pass {add}, feature {feature}, element {k}"
                    );
                }
                start = end;
            }
        }
        // SAFETY: both launches write only the virtual rows after the real block.
        unsafe {
            cuda_launch! {
                kernel: partial_ft_reduce, stream: stream, module: module, config: cfg_1d(virtual_len),
                args: [slice(reference), slice(pairs), slice(shape)]
            }?;
            cuda_launch! {
                kernel: partial_ft_reduce_active, stream: stream, module: module, config: cfg_1d(virtual_len),
                args: [slice(actual), slice(pairs), slice(shape), slice(mask)]
            }?;
        }
        let reference = reference.to_host_vec(&stream)?;
        let actual = actual.to_host_vec(&stream)?;
        for k in real_len..real_len + virtual_len {
            assert_eq!(
                actual[k].to_bits(),
                reference[k].to_bits(),
                "{variant:?}, {rows}/{width}, virtual element {k}"
            );
        }
    }
    Ok(())
}

#[cfg(feature = "native")]
#[test]
fn partial_threat_expand_indices_and_forward_piece_limits() -> Result<(), Box<dyn std::error::Error>>
{
    let ctx = test_cuda_context()?;
    let stream = gpu_runtime::create_compute_stream(&ctx)?;
    let module = load_kernel_module_with_fallback(&ctx, "nnue_train")?;
    let batch = 3_u32;
    let stride = 40_u32;
    let rows = 16_u32;
    let width = 8_u32;
    let base = 2_u32;
    let stm: Vec<i32> = (0..35).map(|i| if i == 34 { 1 } else { 3 }).collect();
    let nstm: Vec<i32> = (0..35).map(|i| if i == 17 { 0 } else { 2 }).collect();
    let pack = |head: &[i32; 3], active: &[i32], tail: &[i32]| {
        DeviceBuffer::from_host(
            &stream,
            &head
                .iter()
                .chain(active)
                .chain(tail)
                .copied()
                .collect::<Vec<_>>(),
        )
    };
    let packed_stm = pack(&[0, 35, 35], &stm, &[0, -1, 2])?;
    let packed_nstm = pack(&[0, 35, 35], &nstm, &[2, 1, 3])?;
    let mut dense_stm = DeviceBuffer::from_host(&stream, &[i32::MIN; 120])?;
    let mut dense_nstm = DeviceBuffer::from_host(&stream, &[i32::MIN; 120])?;
    let mut counts_and_limits = [i32::MIN; 9];
    counts_and_limits[..3].copy_from_slice(&[35, 0, 3]);
    let mut nnz = DeviceBuffer::from_host(&stream, &counts_and_limits)?;
    // SAFETY: each warp expands a row and writes its two piece limits after the count plane.
    unsafe {
        cuda_launch! {
            kernel: ft_expand_indices, stream: stream, module: module, config: cfg_1d((batch * 32) as usize),
            args: [slice(packed_stm), slice(packed_nstm), slice_mut(nnz),
                   slice_mut(dense_stm), slice_mut(dense_nstm), batch, stride, base]
        }
    }?;
    assert_eq!(nnz.to_host_vec(&stream)?, [35, 0, 3, 35, 0, 1, 18, 0, 2]);
    let dense_stm_host = dense_stm.to_host_vec(&stream)?;
    let dense_nstm_host = dense_nstm.to_host_vec(&stream)?;
    assert_eq!(&dense_stm_host[..35], stm);
    assert_eq!(&dense_nstm_host[..35], nstm);
    assert_eq!(&dense_stm_host[80..83], &[0, -1, 2]);
    assert_eq!(&dense_nstm_host[80..83], &[2, 1, 3]);
    for dense in [&dense_stm_host, &dense_nstm_host] {
        assert_eq!(&dense[35..80], &[i32::MIN; 45]);
        assert_eq!(&dense[83..], &[i32::MIN; 37]);
    }

    let mut counts_only = DeviceBuffer::from_host(&stream, &counts_and_limits)?;
    let mut stm_full = DeviceBuffer::from_host(&stream, &[i32::MIN; 120])?;
    let mut nstm_full = DeviceBuffer::from_host(&stream, &[i32::MIN; 120])?;
    // SAFETY: same packed rows, with piece-limit planes disabled.
    unsafe {
        cuda_launch! {
            kernel: ft_expand_indices, stream: stream, module: module, config: cfg_1d((batch * 32) as usize),
            args: [slice(packed_stm), slice(packed_nstm), slice_mut(counts_only),
                   slice_mut(stm_full), slice_mut(nstm_full), batch, stride, 0_u32]
        }
    }?;
    assert_eq!(counts_only.to_host_vec(&stream)?, counts_and_limits);
    assert_eq!(stm_full.to_host_vec(&stream)?, dense_stm_host);
    assert_eq!(nstm_full.to_host_vec(&stream)?, dense_nstm_host);

    let shape = DeviceBuffer::from_host(&stream, &[rows, base, 2, width, stride])?;
    let counts = DeviceBuffer::from_host(&stream, &[35_i32, 0, 3, 0, 0, 0, 0, 0, 0])?;
    let order = DeviceBuffer::from_host(&stream, &[2_i32, 0, 1, 4, 5, 3])?;
    let weights = DeviceBuffer::from_host(
        &stream,
        &(0..base * rows + 2 * width)
            .map(|v| (v as f32 / 3.0 - 5.0) as f16)
            .collect::<Vec<_>>(),
    )?;
    for (perspective, indices) in [&dense_stm, &dense_nstm].into_iter().enumerate() {
        let len = (rows * batch) as usize;
        let expected = DeviceBuffer::from_host(&stream, &vec![f16::INFINITY; len])?;
        let actual = DeviceBuffer::from_host(&stream, &vec![f16::NEG_INFINITY; len])?;
        let batch_order = u64::from(batch) | ((perspective as u64 * u64::from(batch)) << 32);
        let compact_batch_order = batch_order | crate::trainer_layerstack::PARTIAL_FT_PIECE_LIMITS;
        // SAFETY: both buffers hold the count plane, and the compact launch also has two valid limit planes.
        unsafe {
            cuda_launch! {
                kernel: partial_ft_forward_fp16_weights_out, stream: stream, module: module,
                config: cfg_1d((rows * batch / 4) as usize),
                args: [slice(weights), slice(indices), slice(counts), slice(expected),
                       slice(shape), batch_order, slice(order)]
            }?;
            cuda_launch! {
                kernel: partial_ft_forward_fp16_weights_out, stream: stream, module: module,
                config: cfg_1d((rows * batch / 4) as usize),
                args: [slice(weights), slice(indices), slice(nnz), slice(actual),
                       slice(shape), compact_batch_order, slice(order)]
            }?;
        }
        let bits = |out: &DeviceBuffer<f16>| {
            out.to_host_vec(&stream)
                .map(|values| values.into_iter().map(f16::to_bits).collect::<Vec<_>>())
        };
        let actual = bits(&actual)?;
        assert_eq!(actual, bits(&expected)?);
        assert_eq!(&actual[rows as usize..2 * rows as usize], &[0; 16]);
    }
    Ok(())
}
