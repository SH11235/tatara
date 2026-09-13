use super::*;
use crate::qat::QatMode;
use shogi_features::FeatureSet;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn qat_trains_with_fp16_tf32_and_factorizer() -> TestResult {
    let ctx = CudaContext::new(0)?;
    let features = FeatureSet::HalfKaHmMerged.spec().with_ft_factorize();
    let mut t = GpuTrainer::new(
        &ctx,
        16,
        128,
        16,
        32,
        2,
        BucketMode::ProgressKpAbs,
        PrecisionFlags {
            tf32: true,
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
        },
        features,
        OptimizerKind::RAdam,
        OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
        None,
        None,
        &LayerStackInit::default_uniform(),
    )?;
    t.configure_qat(Some(QatMode::Dense))?;
    let mut batch = BatchData::smoke_dummy(16, features);
    batch.score.fill(200.0);
    for _ in 0..2 {
        t.step(&batch.as_ref(), 0.0001, 0.0, SMOKE_LOSS_WRM)?;
    }
    assert!(
        t.validate(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?
            .loss
            .is_finite()
    );
    assert_eq!(t.step_count, 2);
    t.assert_all_weights_finite()?;
    Ok(())
}

#[test]
fn qat_activation_boundaries_and_main_ste() -> TestResult {
    let ctx = CudaContext::new(0)?;
    let mut t = trainer(&ctx, 128, false)?;
    let mut net = t.to_layerstack_weights()?;
    net.ft_w.fill(0.0);
    net.ft_b.fill(0.5);
    net.l1_w.fill(0.0);
    net.l1_shared_weight.fill(0.0);
    net.l1_shared_bias.fill(0.0);
    net.l2_w.fill(0.0);
    net.l2_b.fill(0.0);
    net.l3_w.fill(0.0);
    let values = [
        -8192, -4096, -725, -724, -64, 0, 63, 64, 724, 725, 4096, 8127, 8128, 8191, 8192, 0,
    ];
    for (i, b) in net.l1_b.iter_mut().enumerate() {
        *b = values[i % 16] as f32 / 8128.0;
    }
    for i in 0..15 {
        net.l2_w[i * 30 + i] = 1.007;
        net.l3_w[i] = 1.008;
    }
    net.l1_w[15 * 128] = 0.008;
    t.load_layerstack_weights(&net)?;
    t.sync_ft_forward_weights()?;
    t.configure_qat(Some(QatMode::Dense))?;
    let mut batch = BatchData::smoke_dummy(16, FeatureSet::HalfKaHmMerged.spec());
    batch.bucket_idx.fill(0);
    let data = batch.as_ref();
    let mut now = std::time::Instant::now();
    let mut context = StepContext::new(
        &t,
        &data,
        StepOptions {
            lr: 0.0,
            wdl_lambda: 0.0,
            loss: SMOKE_LOSS_WRM,
            validate: false,
            forward_output: false,
            profile_step: false,
            prof_t0: &mut now,
        },
    )?;
    t.qat_prepare()?;
    t.forward(&data, &mut context)?;
    let raw = t.qat_raw[0].to_host_vec(&t.stream)?;
    let act = t.ws.l2_input.to_host_vec(&t.stream)?;
    for i in 0..15 {
        assert_eq!(raw[i], values[i], "raw boundary {i}");
        let z = i64::from(values[i]);
        assert_eq!(act[i], ((z * z) >> 19).clamp(0, 127) as f32 / 127.0);
        assert_eq!(act[15 + i], (z >> 6).clamp(0, 127) as f32 / 127.0);
    }
    t.ws.dy_net_output = DeviceBuffer::from_host(&t.stream, &[1.0; 16])?;
    t.backward(&mut context)?;
    let grad = t.ws.dl1_total.to_host_vec(&t.stream)?;
    assert!(
        (grad[1] + 65.0 / 64.0).abs() < 1e-6,
        "negative square derivative {}",
        grad[1]
    );
    assert!(
        (grad[10] - 65.0 / 64.0).abs() < 1e-6,
        "positive square derivative {}",
        grad[10]
    );
    for i in [0, 5, 14] {
        assert_eq!(grad[i], 0.0, "clipped/zero derivative {i}");
    }
    assert_eq!(grad[15], 1.0);
    let input_grad = t.ws.dcombined_from_l1.to_host_vec(&t.stream)?;
    assert_eq!(
        input_grad[0],
        1.0 / 64.0,
        "input gradient uses quantized weight"
    );
    t.qat_swap();
    Ok(())
}

fn trainer(
    ctx: &std::sync::Arc<CudaContext>,
    width: usize,
    forward_only: bool,
) -> Result<GpuTrainer, Box<dyn std::error::Error>> {
    trainer_with_features(ctx, width, forward_only, FeatureSet::HalfKaHmMerged.spec())
}

fn trainer_with_features(
    ctx: &std::sync::Arc<CudaContext>,
    width: usize,
    forward_only: bool,
    features: FeatureSetSpec,
) -> Result<GpuTrainer, Box<dyn std::error::Error>> {
    let build = if forward_only {
        GpuTrainer::new_forward_only
    } else {
        GpuTrainer::new
    };
    build(
        ctx,
        16,
        width,
        16,
        32,
        2,
        BucketMode::ProgressKpAbs,
        PrecisionFlags {
            tf32: false,
            ft_fp16: false,
            ft_fp16_out: false,
            fp16_opt_state: false,
        },
        features,
        OptimizerKind::RAdam,
        OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
        None,
        None,
        &LayerStackInit::default_uniform(),
    )
}

fn quant_weight(w: f32) -> i64 {
    (f64::from(w) * 64.0).round().clamp(-128.0, 127.0) as i64
}

#[test]
fn qat_preserves_ft_factorizer_fold_order() -> TestResult {
    let ctx = CudaContext::new(0)?;
    let features = FeatureSet::HalfKaHmMerged.spec().with_ft_factorize();
    let mut t = trainer_with_features(&ctx, 128, true, features)?;
    let mut weights = vec![0.0; features.train_ft_in() * 128];
    weights[features.base_ft_in() * 128..].fill(0.25);
    t.ft_w = DeviceBuffer::from_host(&t.stream, &weights)?;
    t.sync_ft_forward_weights()?;
    let folded = t.to_layerstack_weights()?.ft_w;
    let batch = BatchData::smoke_dummy(16, features);
    t.validate(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
    let ordinary_ft = t.ws.ft_stm_out.to_host_vec(&t.stream)?;
    t.configure_qat(Some(QatMode::Dense))?;
    t.validate(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
    assert_eq!(t.ws.ft_stm_out.to_host_vec(&t.stream)?, ordinary_ft);
    assert_eq!(t.ft_w.to_host_vec(&t.stream)?, weights);
    assert_eq!(t.to_layerstack_weights()?.ft_w, folded);
    Ok(())
}
fn quant_bias(b: f32) -> i64 {
    (f64::from(b) * 8128.0).round() as i32 as i64
}

#[test]
fn qat_integer_forward_export_and_master_preservation() -> TestResult {
    let ctx = CudaContext::new(0)?;
    for width in [1536, 3072] {
        let mut t = trainer(&ctx, width, true)?;
        let mut net = t.to_layerstack_weights()?;
        net.ft_w.fill(0.0);
        net.ft_b.fill(0.5);
        net.l1_w.fill(0.49 / 64.0);
        net.l1_shared_weight.fill(0.49 / 64.0);
        let boundaries = [
            -8192, -8128, -724, -64, -1, 0, 1, 63, 64, 724, 725, 8127, 8128, 8191, 8192, -127,
        ];
        for (i, bias) in net.l1_b.iter_mut().enumerate() {
            *bias = boundaries[i % 16] as f32 / 8128.0;
        }
        net.l1_shared_bias.fill(0.5 / 8128.0);
        for (i, w) in net.l2_w.iter_mut().enumerate() {
            *w = ((i % 13) as f32 - 6.5) / 64.0;
        }
        net.l2_w[..8].copy_from_slice(&[
            -3.0,
            -128.5 / 64.0,
            -2.0,
            -0.5 / 64.0,
            0.5 / 64.0,
            127.0 / 64.0,
            127.5 / 64.0,
            3.0,
        ]);
        for (i, w) in net.l3_w.iter_mut().enumerate() {
            *w = ((i % 9) as f32 - 4.5) / 64.0;
        }
        net.l2_b.fill(-0.5 / 8128.0);
        net.l3_b.fill(-123.5 / 8128.0);
        t.load_layerstack_weights(&net)?;
        t.sync_ft_forward_weights()?;
        t.configure_qat(Some(QatMode::Dense))?;
        let batch = BatchData::smoke_dummy(16, FeatureSet::HalfKaHmMerged.spec());
        let actual = t.validate(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
        let x = t.ws.combined.to_host_vec(&t.stream)?;
        let raw1 = t.qat_raw[0].to_host_vec(&t.stream)?;
        let raw2 = t.qat_raw[1].to_host_vec(&t.stream)?;
        for row in 0..16 {
            let bucket = batch.bucket_idx[row] as usize;
            let mut a2 = [0_i64; 30];
            let mut skip = 0;
            for out in 0..16 {
                let mut z = quant_bias(net.l1_b[bucket * 16 + out] + net.l1_shared_bias[out]);
                for input in 0..width {
                    let w = net.l1_w[(bucket * 16 + out) * width + input]
                        + net.l1_shared_weight[input * 16 + out];
                    z += quant_weight(w)
                        * (f64::from(x[row * width + input]) * 127.0).round() as i64;
                }
                assert_eq!(raw1[row * 16 + out] as i64, z);
                if out == 15 {
                    skip = z;
                } else {
                    a2[out] = ((z * z) >> 19).clamp(0, 127);
                    a2[15 + out] = (z >> 6).clamp(0, 127);
                }
            }
            let mut final_raw = quant_bias(net.l3_b[bucket]) + skip;
            for out in 0..32 {
                let z = quant_bias(net.l2_b[bucket * 32 + out])
                    + a2.iter()
                        .enumerate()
                        .map(|(i, x)| x * quant_weight(net.l2_w[(bucket * 32 + out) * 30 + i]))
                        .sum::<i64>();
                assert_eq!(raw2[row * 32 + out] as i64, z);
                final_raw += (z >> 6).clamp(0, 127) * quant_weight(net.l3_w[bucket * 32 + out]);
            }
            assert_eq!(actual.net_output[row], (final_raw as f64 / 8128.0) as f32);
            assert_eq!((-127_i32) / 14, -9);
        }
        let after = t.to_layerstack_weights()?;
        assert_eq!(after.l1_w, net.l1_w);
        assert_eq!(after.l1_shared_weight, net.l1_shared_weight);
        assert_eq!(after.l2_w, net.l2_w);
        let mut before_export = Vec::new();
        let mut after_export = Vec::new();
        net.save_quantised(&mut before_export, Some(14))?;
        after.save_quantised(&mut after_export, Some(14))?;
        assert_eq!(before_export, after_export);
        let exported = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&before_export),
            FeatureSet::HalfKaHmMerged.spec(),
            width,
            16,
            32,
            2,
        )?;
        t.qat_prepare()?;
        assert_eq!(t.l1_w.to_host_vec(&t.stream)?, exported.l1_w);
        assert_eq!(t.l1_b.to_host_vec(&t.stream)?, exported.l1_b);
        assert_eq!(t.l2_w.to_host_vec(&t.stream)?, exported.l2_w);
        assert_eq!(t.l2_b.to_host_vec(&t.stream)?, exported.l2_b);
        t.qat_swap();
    }
    Ok(())
}

#[test]
fn qat_skip_ste_and_same_raw_branches() -> TestResult {
    let ctx = CudaContext::new(0)?;
    let mut t = trainer(&ctx, 128, false)?;
    let mut net = t.to_layerstack_weights()?;
    net.ft_w.fill(0.0);
    net.ft_b.fill(0.5);
    net.l3_w.fill(0.0);
    t.load_layerstack_weights(&net)?;
    t.sync_ft_forward_weights()?;
    let mut batch = BatchData::smoke_dummy(16, FeatureSet::HalfKaHmMerged.spec());
    batch.bucket_idx.fill(0);
    batch.score.fill(200.0);
    let path = std::env::temp_dir().join(format!("tatara-qat-{}.ckpt", std::process::id()));
    t.save_raw_checkpoint(&path, 1200, "qat-test", None)?;
    for mode in [QatMode::Off, QatMode::Dense] {
        t.load_raw_checkpoint(&path)?;
        t.configure_qat(Some(mode))?;
        if mode == QatMode::Dense {
            let mut now = std::time::Instant::now();
            let data = batch.as_ref();
            let mut context = StepContext::new(
                &t,
                &data,
                StepOptions {
                    lr: 0.0,
                    wdl_lambda: 0.0,
                    loss: SMOKE_LOSS_WRM,
                    validate: false,
                    forward_output: false,
                    profile_step: false,
                    prof_t0: &mut now,
                },
            )?;
            t.qat_prepare()?;
            assert!(t.forward(&data, &mut context)?.is_none());
            t.backward(&mut context)?;
            let dy = t.ws.dy_net_output.to_host_vec(&t.stream)?;
            let x = t.ws.combined.to_host_vec(&t.stream)?;
            let dw = t.l1_w_grad.to_host_vec(&t.stream)?;
            let shared = t.l1_shared_weight_grad.to_host_vec(&t.stream)?;
            for i in 0..128 {
                let expected: f32 = (0..16).map(|r| dy[r] * x[r * 128 + i]).sum();
                assert!(
                    (dw[15 * 128 + i] - expected).abs() < 1e-6,
                    "i={i}, dw={}, expected={expected}, dy={:?}",
                    dw[15 * 128 + i],
                    dy
                );
                assert!(
                    (shared[i * 16 + 15] - expected).abs() < 1e-6,
                    "shared={}, expected={expected}",
                    shared[i * 16 + 15]
                );
            }
            t.qat_swap();
            t.optimizer_step(&mut context)?;
        } else {
            t.step(&batch.as_ref(), 0.0, 0.0, SMOKE_LOSS_WRM)?;
        }
        assert_eq!(t.step_count, 1);
    }
    t.save_raw_checkpoint(&path, 1201, "qat-tail", None)?;
    t.configure_qat(Some(QatMode::Off))?;
    t.load_raw_checkpoint(&path)?;
    t.configure_qat(None)?;
    assert_eq!(t.qat_mode, QatMode::Dense);
    t.configure_qat(Some(QatMode::Off))?;
    assert_eq!(t.qat_mode, QatMode::Off);
    std::fs::remove_file(path)?;
    Ok(())
}
