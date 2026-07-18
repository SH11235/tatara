#![cfg(feature = "native-cuda")]

use std::{ffi::c_void, ptr};

use cuda_native_runtime::{Context, DeviceBuffer, NATIVE_KERNEL_FATBIN, PinnedBuffer};

fn arg<T>(value: &mut T) -> *mut c_void {
    ptr::from_mut(value).cast()
}

#[test]
fn vector_add_runs_from_embedded_fatbin() {
    let context = Context::new(0).unwrap();
    let stream = context.create_stream().unwrap();
    let module = context.load_module(NATIVE_KERNEL_FATBIN).unwrap();
    let function = module.function(c"native_vec_add").unwrap();

    let lhs = [1.0_f32, -2.0, 3.5, 8.0, -0.25];
    let rhs = [4.0_f32, 5.0, -1.5, 0.5, 0.75];
    let lhs_device = DeviceBuffer::from_slice(&context, &lhs).unwrap();
    let rhs_device = DeviceBuffer::from_slice(&context, &rhs).unwrap();
    let output_device = DeviceBuffer::<f32>::zeroed(&context, lhs.len()).unwrap();
    let mut lhs_ptr = lhs_device.device_ptr();
    let mut rhs_ptr = rhs_device.device_ptr();
    let mut output_ptr = output_device.device_ptr();
    let mut n = lhs.len() as u32;
    let mut args = [
        arg(&mut lhs_ptr),
        arg(&mut rhs_ptr),
        arg(&mut output_ptr),
        arg(&mut n),
    ];
    // SAFETY: args exactly match native_vec_add and all allocations contain n elements.
    unsafe {
        function
            .launch(&stream, (1, 1, 1), (256, 1, 1), 0, &mut args)
            .unwrap();
    }
    stream.synchronize().unwrap();

    let mut output = [0.0_f32; 5];
    output_device.copy_to(&mut output).unwrap();
    assert_eq!(output, [5.0, 3.0, 2.0, 8.5, 0.5]);
}

#[test]
fn pinned_async_copies_can_be_ordered_across_streams() {
    let context = Context::new(0).unwrap();
    let upload_stream = context.create_stream().unwrap();
    let download_stream = context.create_stream().unwrap();
    let uploaded = context.create_event().unwrap();
    let downloaded = context.create_event().unwrap();
    let input = PinnedBuffer::from_slice(&context, &[1_u32, 3, 5, 7, 9]).unwrap();
    let mut output = PinnedBuffer::<u32>::new(&context, input.len()).unwrap();
    let device = DeviceBuffer::<u32>::zeroed(&context, input.len()).unwrap();

    // SAFETY: input remains live and immutable until uploaded is synchronized by download_stream.
    unsafe {
        device
            .copy_from_pinned_async(&input, &upload_stream)
            .unwrap();
    }
    uploaded.record(&upload_stream).unwrap();
    download_stream.wait_event(&uploaded).unwrap();
    // SAFETY: output remains live and unobserved until downloaded is synchronized below.
    unsafe {
        device
            .copy_to_pinned_async(&mut output, &download_stream)
            .unwrap();
    }
    downloaded.record(&download_stream).unwrap();
    downloaded.synchronize().unwrap();

    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn wrm_default_matches_cpu_reference() {
    let output = [-0.9_f32, -0.1, 0.0, 0.2, 1.1, 2.0, -3.0];
    let score = [-1200.0_f32, -100.0, 0.0, 300.0, 900.0, 32000.0, -32000.0];
    let wdl = [0.0_f32, 0.0, 0.5, 1.0, 1.0, 1.0, 0.0];
    let n = output.len();
    let per_pos_norm = 1.0_f32 / n as f32;
    let mut expected_gradient = vec![0.0_f32; n];
    let mut expected_loss = 0.0_f64;
    gpu_kernels::pointwise::loss_wrm::loss_wrm_cpu(
        &output,
        &score,
        &wdl,
        &vec![per_pos_norm; n],
        &mut expected_gradient,
        &mut expected_loss,
        0.25,
        600.0,
        340.0,
        270.0,
        270.0,
        380.0,
        2.0,
        0.0,
        0.0,
        0.5,
        false,
        n,
    );

    let context = Context::new(0).unwrap();
    let stream = context.create_stream().unwrap();
    let module = context.load_module(NATIVE_KERNEL_FATBIN).unwrap();
    let function = module.function(c"native_loss_wrm_default").unwrap();
    let output_device = DeviceBuffer::from_slice(&context, &output).unwrap();
    let score_device = DeviceBuffer::from_slice(&context, &score).unwrap();
    let wdl_device = DeviceBuffer::from_slice(&context, &wdl).unwrap();
    let gradient_device = DeviceBuffer::<f32>::zeroed(&context, n).unwrap();
    let loss_device = DeviceBuffer::<f64>::zeroed(&context, 1).unwrap();

    let mut output_ptr = output_device.device_ptr();
    let mut score_ptr = score_device.device_ptr();
    let mut wdl_ptr = wdl_device.device_ptr();
    let mut gradient_ptr = gradient_device.device_ptr();
    let mut loss_ptr = loss_device.device_ptr();
    let mut norm = per_pos_norm;
    let mut lambda = 0.25_f32;
    let mut nnue2score = 600.0_f32;
    let mut input_scaling = 340.0_f32;
    let mut input_offset = 270.0_f32;
    let mut target_offset = 270.0_f32;
    let mut target_scaling = 380.0_f32;
    let mut count = n as u32;
    let mut args = [
        arg(&mut output_ptr),
        arg(&mut score_ptr),
        arg(&mut wdl_ptr),
        arg(&mut norm),
        arg(&mut gradient_ptr),
        arg(&mut loss_ptr),
        arg(&mut lambda),
        arg(&mut nnue2score),
        arg(&mut input_scaling),
        arg(&mut input_offset),
        arg(&mut target_offset),
        arg(&mut target_scaling),
        arg(&mut count),
    ];
    // SAFETY: args exactly match native_loss_wrm_default; block size matches shared storage.
    unsafe {
        function
            .launch(&stream, (1, 1, 1), (256, 1, 1), 0, &mut args)
            .unwrap();
    }
    stream.synchronize().unwrap();

    let mut actual_gradient = vec![0.0_f32; n];
    let mut actual_loss = [0.0_f64];
    gradient_device.copy_to(&mut actual_gradient).unwrap();
    loss_device.copy_to(&mut actual_loss).unwrap();
    for (actual, expected) in actual_gradient.iter().zip(&expected_gradient) {
        assert!(
            (actual - expected).abs() <= 2.0e-6,
            "actual={actual}, expected={expected}"
        );
    }
    assert!((actual_loss[0] - expected_loss).abs() <= 2.0e-6);
}

#[test]
fn radam_step_matches_cpu_reference() {
    let mut expected_weights = vec![0.8_f32, -0.5, 0.01, 1.9, -2.0];
    let mut expected_momentum = vec![0.02_f32, -0.04, 0.0, 0.3, -0.2];
    let mut expected_velocity = vec![0.03_f32, 0.01, 0.2, 0.4, 0.1];
    let mut expected_gradient = vec![0.4_f32, -0.7, 0.0, 1.2, -0.3];
    let n = expected_weights.len();
    let (step_size, denom) =
        gpu_kernels::pointwise::radam_step::radam_compute_step_size_denom(100, 0.99, 0.999, 5.0);
    gpu_kernels::pointwise::radam_step::radam_step_cpu(
        &mut expected_weights,
        &mut expected_momentum,
        &mut expected_velocity,
        &mut expected_gradient,
        0.001,
        step_size,
        denom,
        0.01,
        0.99,
        0.999,
        1.0e-8,
        -1.5,
        1.5,
        n,
    );

    let context = Context::new(0).unwrap();
    let stream = context.create_stream().unwrap();
    let module = context.load_module(NATIVE_KERNEL_FATBIN).unwrap();
    let function = module.function(c"native_radam_step").unwrap();
    let weights_device =
        DeviceBuffer::from_slice(&context, &[0.8_f32, -0.5, 0.01, 1.9, -2.0]).unwrap();
    let momentum_device =
        DeviceBuffer::from_slice(&context, &[0.02_f32, -0.04, 0.0, 0.3, -0.2]).unwrap();
    let velocity_device =
        DeviceBuffer::from_slice(&context, &[0.03_f32, 0.01, 0.2, 0.4, 0.1]).unwrap();
    let gradient_device =
        DeviceBuffer::from_slice(&context, &[0.4_f32, -0.7, 0.0, 1.2, -0.3]).unwrap();
    let mut weights_ptr = weights_device.device_ptr();
    let mut momentum_ptr = momentum_device.device_ptr();
    let mut velocity_ptr = velocity_device.device_ptr();
    let mut gradient_ptr = gradient_device.device_ptr();
    let mut learning_rate = 0.001_f32;
    let mut step_size_arg = step_size;
    let mut denom_arg = denom;
    let mut decay = 0.01_f32;
    let mut beta1 = 0.99_f32;
    let mut beta2 = 0.999_f32;
    let mut epsilon = 1.0e-8_f32;
    let mut min_weight = -1.5_f32;
    let mut max_weight = 1.5_f32;
    let mut count = n as u32;
    let mut args = [
        arg(&mut weights_ptr),
        arg(&mut momentum_ptr),
        arg(&mut velocity_ptr),
        arg(&mut gradient_ptr),
        arg(&mut learning_rate),
        arg(&mut step_size_arg),
        arg(&mut denom_arg),
        arg(&mut decay),
        arg(&mut beta1),
        arg(&mut beta2),
        arg(&mut epsilon),
        arg(&mut min_weight),
        arg(&mut max_weight),
        arg(&mut count),
    ];
    // SAFETY: args exactly match native_radam_step and all allocations contain n elements.
    unsafe {
        function
            .launch(&stream, (1, 1, 1), (256, 1, 1), 0, &mut args)
            .unwrap();
    }
    stream.synchronize().unwrap();

    let mut actual_weights = vec![0.0_f32; n];
    let mut actual_momentum = vec![0.0_f32; n];
    let mut actual_velocity = vec![0.0_f32; n];
    let mut actual_gradient = vec![1.0_f32; n];
    weights_device.copy_to(&mut actual_weights).unwrap();
    momentum_device.copy_to(&mut actual_momentum).unwrap();
    velocity_device.copy_to(&mut actual_velocity).unwrap();
    gradient_device.copy_to(&mut actual_gradient).unwrap();
    for (actual, expected) in actual_weights.iter().zip(&expected_weights) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }
    for (actual, expected) in actual_momentum.iter().zip(&expected_momentum) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }
    for (actual, expected) in actual_velocity.iter().zip(&expected_velocity) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }
    assert_eq!(actual_gradient, expected_gradient);
}

#[test]
fn sparse_ft_forward_matches_cpu_reference() {
    let batch = 2_usize;
    let rows = 8_usize;
    let columns = 4_usize;
    let max_active = 5_usize;
    let weight = (0..rows * columns)
        .map(|i| (i as f32 - 7.0) * 0.125)
        .collect::<Vec<_>>();
    let indices = vec![0_i32, 2, -1, 3, -1, 1, 3, 9, -1, -1];
    let nonzero_counts = vec![4_i32, 3];
    let mut expected = vec![0.0_f32; batch * rows];
    gpu_kernels::sparse::sparse_ft_forward::sparse_ft_forward_cpu(
        &weight,
        &indices,
        &nonzero_counts,
        &mut expected,
        batch,
        rows,
        columns,
        max_active,
    );

    let context = Context::new(0).unwrap();
    let stream = context.create_stream().unwrap();
    let module = context.load_module(NATIVE_KERNEL_FATBIN).unwrap();
    let function = module.function(c"native_sparse_ft_forward").unwrap();
    let weight_device = DeviceBuffer::from_slice(&context, &weight).unwrap();
    let indices_device = DeviceBuffer::from_slice(&context, &indices).unwrap();
    let counts_device = DeviceBuffer::from_slice(&context, &nonzero_counts).unwrap();
    let output_device = DeviceBuffer::<f32>::zeroed(&context, batch * rows).unwrap();
    let mut weight_ptr = weight_device.device_ptr();
    let mut indices_ptr = indices_device.device_ptr();
    let mut counts_ptr = counts_device.device_ptr();
    let mut output_ptr = output_device.device_ptr();
    let mut batch_arg = batch as u32;
    let mut rows_arg = rows as u32;
    let mut columns_arg = columns as u32;
    let mut max_active_arg = max_active as u32;
    let mut args = [
        arg(&mut weight_ptr),
        arg(&mut indices_ptr),
        arg(&mut counts_ptr),
        arg(&mut output_ptr),
        arg(&mut batch_arg),
        arg(&mut rows_arg),
        arg(&mut columns_arg),
        arg(&mut max_active_arg),
    ];
    let threads = batch * rows / 4;
    // SAFETY: args match native_sparse_ft_forward and every buffer follows its documented shape.
    unsafe {
        function
            .launch(
                &stream,
                (threads.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();

    let mut actual = vec![0.0_f32; batch * rows];
    output_device.copy_to(&mut actual).unwrap();
    assert_eq!(actual, expected);
}
