#![cfg(feature = "native-cuda")]

use std::{ffi::c_void, ptr};

use cuda_native_runtime::{Context, DeviceBuffer, PROGRESS_KERNEL_FATBIN};
use gpu_kernels::progress::{
    adam_step::adam_step_cpu, eval::eval_cpu, forward::forward_cpu, grad::grad_cpu,
};

fn arg<T>(value: &mut T) -> *mut c_void {
    ptr::from_mut(value).cast()
}

fn slice_args<T: Copy>(buffer: &DeviceBuffer<T>) -> (u64, u64) {
    (buffer.device_ptr(), buffer.len() as u64)
}

#[test]
fn every_progress_source_export_resolves_from_embedded_fatbin() {
    let context = Context::new(0).unwrap();
    let module = context.load_module(PROGRESS_KERNEL_FATBIN).unwrap();
    let source = include_str!("../kernels/progress_kernels.cu");
    let prefix = "extern \"C\" __global__ void ";
    let mut resolved = 0;

    for line in source.lines() {
        let Some(declaration) = line.strip_prefix(prefix) else {
            continue;
        };
        let name = declaration
            .split_once('(')
            .map(|(name, _)| name)
            .expect("CUDA export declaration must contain '('");
        let name = std::ffi::CString::new(name).expect("CUDA export name must not contain NUL");
        module.function(&name).unwrap_or_else(|error| {
            panic!(
                "progress fatbin is missing {}: {error}",
                name.to_string_lossy()
            )
        });
        resolved += 1;
    }

    assert_eq!(resolved, 4, "progress CUDA source export inventory changed");
}

#[test]
fn progress_kernels_match_cpu_references() {
    const FLOAT_TOLERANCE: f32 = 1.0e-5;
    const LOSS_TOLERANCE: f64 = 1.0e-8;
    const BLOCK_DIM: u32 = 256;

    let context = Context::new(0).unwrap();
    let stream = context.create_stream().unwrap();
    let module = context.load_module(PROGRESS_KERNEL_FATBIN).unwrap();
    let forward = module.function(c"progress_forward").unwrap();
    let grad_kernel = module.function(c"progress_grad").unwrap();
    let eval = module.function(c"progress_eval").unwrap();
    let adam = module.function(c"progress_adam_step").unwrap();

    let n_pos = 16_usize;
    let max_inds = 8_usize;
    let n_weights = 64_usize;
    let indices = (0..n_pos)
        .flat_map(|pos| {
            (0..max_inds).map(move |slot| {
                if slot == max_inds - 1 && pos % 3 == 0 {
                    -1
                } else {
                    ((pos + slot) % n_weights) as i32
                }
            })
        })
        .collect::<Vec<_>>();
    let weights = (0..n_weights)
        .map(|index| index as f32 * 0.01 - 0.5)
        .collect::<Vec<_>>();
    let targets = (0..n_pos)
        .map(|pos| pos as f32 / n_pos as f32)
        .collect::<Vec<_>>();
    let norms = vec![1.0_f32; n_pos];
    let expected_preds = forward_cpu(&indices, &weights, n_pos, max_inds);

    let indices_device = DeviceBuffer::from_slice(&context, &indices).unwrap();
    let weights_device = DeviceBuffer::from_slice(&context, &weights).unwrap();
    let preds_device = DeviceBuffer::<f32>::zeroed(&context, n_pos).unwrap();
    let (mut indices_ptr, mut indices_len) = slice_args(&indices_device);
    let (mut weights_ptr, mut weights_len) = slice_args(&weights_device);
    let (mut preds_ptr, mut preds_len) = slice_args(&preds_device);
    let mut n_pos_u32 = n_pos as u32;
    let mut max_inds_u32 = max_inds as u32;
    let mut forward_args = [
        arg(&mut indices_ptr),
        arg(&mut indices_len),
        arg(&mut weights_ptr),
        arg(&mut weights_len),
        arg(&mut preds_ptr),
        arg(&mut preds_len),
        arg(&mut n_pos_u32),
        arg(&mut max_inds_u32),
    ];
    // SAFETY: arguments match progress_forward and all logical lengths describe live buffers.
    unsafe {
        forward
            .launch(&stream, (1, 1, 1), (BLOCK_DIM, 1, 1), 0, &mut forward_args)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut actual_preds = vec![0.0_f32; n_pos];
    preds_device.copy_to(&mut actual_preds).unwrap();
    for (actual, expected) in actual_preds.iter().zip(&expected_preds) {
        assert!((actual - expected).abs() < FLOAT_TOLERANCE);
    }

    // grad / eval の検証対象は各 kernel 単体の等価性。GPU forward の preds は
    // `expf` と Rust `f32::exp` の ULP 差で CPU 参照とずれ、loss (err^2 の f64
    // 累積) は 1e-8 許容を超え得るため、両実装への入力を CPU preds に揃える。
    preds_device.copy_from(&expected_preds).unwrap();

    let targets_device = DeviceBuffer::from_slice(&context, &targets).unwrap();
    let norms_device = DeviceBuffer::from_slice(&context, &norms).unwrap();
    let grad_device = DeviceBuffer::<f32>::zeroed(&context, n_weights).unwrap();
    let loss_device = DeviceBuffer::<f64>::zeroed(&context, 1).unwrap();
    let hist_device = DeviceBuffer::<u64>::zeroed(&context, 8).unwrap();
    let mut expected_grad = vec![0.0_f32; n_weights];
    let mut expected_loss = 0.0_f64;
    let mut expected_hist = [0_u64; 8];
    grad_cpu(
        &indices,
        &expected_preds,
        &targets,
        &norms,
        &mut expected_grad,
        &mut expected_loss,
        &mut expected_hist,
        n_pos,
        max_inds,
    );

    let (mut targets_ptr, mut targets_len) = slice_args(&targets_device);
    let (mut norms_ptr, mut norms_len) = slice_args(&norms_device);
    let (mut grad_ptr, mut grad_len) = slice_args(&grad_device);
    let (mut loss_ptr, mut loss_len) = slice_args(&loss_device);
    let (mut hist_ptr, mut hist_len) = slice_args(&hist_device);
    let mut grad_args = [
        arg(&mut indices_ptr),
        arg(&mut indices_len),
        arg(&mut preds_ptr),
        arg(&mut preds_len),
        arg(&mut targets_ptr),
        arg(&mut targets_len),
        arg(&mut norms_ptr),
        arg(&mut norms_len),
        arg(&mut grad_ptr),
        arg(&mut grad_len),
        arg(&mut loss_ptr),
        arg(&mut loss_len),
        arg(&mut hist_ptr),
        arg(&mut hist_len),
        arg(&mut n_pos_u32),
        arg(&mut max_inds_u32),
    ];
    // SAFETY: arguments match progress_grad and all logical lengths describe live buffers.
    unsafe {
        grad_kernel
            .launch(&stream, (1, 1, 1), (BLOCK_DIM, 1, 1), 0, &mut grad_args)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut actual_grad = vec![0.0_f32; n_weights];
    let mut actual_loss = [0.0_f64; 1];
    let mut actual_hist = [0_u64; 8];
    grad_device.copy_to(&mut actual_grad).unwrap();
    loss_device.copy_to(&mut actual_loss).unwrap();
    hist_device.copy_to(&mut actual_hist).unwrap();
    for (actual, expected) in actual_grad.iter().zip(&expected_grad) {
        assert!((actual - expected).abs() < FLOAT_TOLERANCE);
    }
    assert!((actual_loss[0] - expected_loss).abs() < LOSS_TOLERANCE);
    assert_eq!(actual_hist, expected_hist);

    loss_device.zero_async(&stream).unwrap();
    hist_device.zero_async(&stream).unwrap();
    let mut eval_loss = 0.0_f64;
    let mut eval_hist = [0_u64; 8];
    eval_cpu(
        &expected_preds,
        &targets,
        &mut eval_loss,
        &mut eval_hist,
        n_pos,
    );
    let mut eval_args = [
        arg(&mut preds_ptr),
        arg(&mut preds_len),
        arg(&mut targets_ptr),
        arg(&mut targets_len),
        arg(&mut loss_ptr),
        arg(&mut loss_len),
        arg(&mut hist_ptr),
        arg(&mut hist_len),
        arg(&mut n_pos_u32),
    ];
    // SAFETY: arguments match progress_eval and all logical lengths describe live buffers.
    unsafe {
        eval.launch(&stream, (1, 1, 1), (BLOCK_DIM, 1, 1), 0, &mut eval_args)
            .unwrap();
    }
    stream.synchronize().unwrap();
    loss_device.copy_to(&mut actual_loss).unwrap();
    hist_device.copy_to(&mut actual_hist).unwrap();
    assert!((actual_loss[0] - eval_loss).abs() < LOSS_TOLERANCE);
    assert_eq!(actual_hist, eval_hist);

    let mut expected_weights = weights.clone();
    let mut expected_momentum = vec![0.0_f32; n_weights];
    let mut expected_velocity = vec![0.0_f32; n_weights];
    let mut expected_adam_grad = expected_grad.clone();
    let (mut lr, mut beta1, mut beta2, mut eps) = (0.001_f32, 0.9_f32, 0.999_f32, 1.0e-8_f32);
    let (mut bc1, mut bc2) = (1.0_f32 - beta1, 1.0_f32 - beta2);
    adam_step_cpu(
        &mut expected_weights,
        &mut expected_momentum,
        &mut expected_velocity,
        &mut expected_adam_grad,
        lr,
        beta1,
        beta2,
        eps,
        bc1,
        bc2,
        n_weights,
    );
    let momentum_device = DeviceBuffer::<f32>::zeroed(&context, n_weights).unwrap();
    let velocity_device = DeviceBuffer::<f32>::zeroed(&context, n_weights).unwrap();
    let adam_grad_device = DeviceBuffer::from_slice(&context, &expected_grad).unwrap();
    let (mut momentum_ptr, mut momentum_len) = slice_args(&momentum_device);
    let (mut velocity_ptr, mut velocity_len) = slice_args(&velocity_device);
    let (mut adam_grad_ptr, mut adam_grad_len) = slice_args(&adam_grad_device);
    let mut n_weights_u32 = n_weights as u32;
    let mut adam_args = [
        arg(&mut weights_ptr),
        arg(&mut weights_len),
        arg(&mut momentum_ptr),
        arg(&mut momentum_len),
        arg(&mut velocity_ptr),
        arg(&mut velocity_len),
        arg(&mut adam_grad_ptr),
        arg(&mut adam_grad_len),
        arg(&mut lr),
        arg(&mut beta1),
        arg(&mut beta2),
        arg(&mut eps),
        arg(&mut bc1),
        arg(&mut bc2),
        arg(&mut n_weights_u32),
    ];
    // SAFETY: arguments match progress_adam_step and all logical lengths describe live buffers.
    unsafe {
        adam.launch(&stream, (1, 1, 1), (BLOCK_DIM, 1, 1), 0, &mut adam_args)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut actual_weights = vec![0.0_f32; n_weights];
    let mut actual_momentum = vec![0.0_f32; n_weights];
    let mut actual_velocity = vec![0.0_f32; n_weights];
    let mut actual_adam_grad = vec![0.0_f32; n_weights];
    weights_device.copy_to(&mut actual_weights).unwrap();
    momentum_device.copy_to(&mut actual_momentum).unwrap();
    velocity_device.copy_to(&mut actual_velocity).unwrap();
    adam_grad_device.copy_to(&mut actual_adam_grad).unwrap();
    for (actual, expected) in actual_weights.iter().zip(&expected_weights) {
        assert!((actual - expected).abs() < FLOAT_TOLERANCE);
    }
    for (actual, expected) in actual_momentum.iter().zip(&expected_momentum) {
        assert!((actual - expected).abs() < FLOAT_TOLERANCE);
    }
    for (actual, expected) in actual_velocity.iter().zip(&expected_velocity) {
        assert!((actual - expected).abs() < FLOAT_TOLERANCE);
    }
    assert_eq!(actual_adam_grad, expected_adam_grad);
}
