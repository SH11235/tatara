use gpu_runtime::CudaContext;
use nnue_format::{SimpleActivation, SimpleId};
use nnue_train::{
    init::SimpleInit,
    optimizer::OptimizerKind,
    trainer::{LossKind, TrainerBackend},
};
use shogi_features::FeatureSet;

#[cfg(feature = "native-cuda")]
use crate::kernel_module::with_test_native_backend;
use crate::{
    arch::{SMOKE_BATCH, SMOKE_LOSS_WRM},
    trainer_common::{BatchData, PrecisionFlags},
    trainer_simple::{SimpleGpuTrainer, validate_native_simple_configuration},
};

fn standard_id() -> SimpleId {
    SimpleId {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    }
}

fn create_trainer(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    native: bool,
    batch_size: usize,
) -> Result<SimpleGpuTrainer, Box<dyn std::error::Error>> {
    create_trainer_with_options(
        context,
        id,
        native,
        batch_size,
        OptimizerKind::Ranger,
        None,
        PrecisionFlags::default(),
    )
}

fn create_trainer_with_options(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    native: bool,
    batch_size: usize,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
) -> Result<SimpleGpuTrainer, Box<dyn std::error::Error>> {
    #[cfg(feature = "native-cuda")]
    let result = with_test_native_backend(native, || {
        SimpleGpuTrainer::new(
            context,
            batch_size,
            id,
            optimizer,
            0.0,
            norm_loss_factor,
            16,
            precision,
            &SimpleInit::default_uniform(),
        )
    });
    #[cfg(feature = "native-cuda-host")]
    let result = {
        assert!(
            native,
            "native-host build cannot create a cuda-oxide trainer"
        );
        SimpleGpuTrainer::new(
            context,
            batch_size,
            id,
            optimizer,
            0.0,
            norm_loss_factor,
            16,
            precision,
            &SimpleInit::default_uniform(),
        )
    };
    result
}

#[test]
fn native_simple_configuration_accepts_ft_factorizer() {
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    assert!(validate_native_simple_configuration(id, None, PrecisionFlags::default(),).is_ok());
}

#[test]
fn native_simple_configuration_accepts_all_simple_activations() {
    for activation in [
        SimpleActivation::CReLU,
        SimpleActivation::SCReLU,
        SimpleActivation::Pairwise,
    ] {
        let mut id = standard_id();
        id.activation = activation;
        assert!(
            validate_native_simple_configuration(id, None, PrecisionFlags::default(),).is_ok(),
            "{activation:?}"
        );
    }
}

#[test]
fn standard_simple_crelu_runs_one_native_training_step() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let id = standard_id();
    let mut trainer = create_trainer(&context, id, true, SMOKE_BATCH)?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    let _lagged_loss = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let loss = trainer.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
    assert!(loss.is_finite(), "native Simple loss is not finite: {loss}");
    trainer.assert_all_weights_finite()?;
    let weights = trainer.to_simple_weights()?;
    let mut fingerprint = 0xcbf29ce484222325_u64;
    for value in weights
        .ft_w
        .iter()
        .chain(&weights.ft_b)
        .chain(&weights.l1_w)
        .chain(&weights.l1_b)
        .chain(&weights.l2_w)
        .chain(&weights.l2_b)
        .chain(&weights.l3_w)
        .chain(&weights.l3_b)
    {
        fingerprint ^= u64::from(value.to_bits());
        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    }
    eprintln!(
        "[native-host-parity] loss_bits={:016x}, weight_fingerprint={fingerprint:016x}",
        loss.to_bits()
    );
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
fn standard_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step(standard_id())
}

#[test]
#[cfg(feature = "native-cuda")]
fn factorized_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn screlu_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    let mut id = standard_id();
    id.activation = SimpleActivation::SCReLU;
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn pairwise_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.activation = SimpleActivation::Pairwise;
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn wide_hidden_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.l1_out = 257;
    id.l2_out = 257;
    assert_native_matches_cuda_oxide_after_one_step(id)
}

#[test]
#[cfg(feature = "native-cuda")]
fn radam_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::RAdam,
        PrecisionFlags::default(),
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn adamw_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::AdamW,
        PrecisionFlags::default(),
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn tf32_simple_native_matches_cuda_oxide_after_one_step() -> Result<(), Box<dyn std::error::Error>>
{
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            tf32: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn norm_loss_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        standard_id(),
        OptimizerKind::Ranger,
        Some(0.25),
        PrecisionFlags::default(),
        SMOKE_LOSS_WRM,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn sigmoid_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags::default(),
        LossKind::Sigmoid { scale: 1.0 / 600.0 },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn extended_wrm_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let extended = match SMOKE_LOSS_WRM {
        LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            ..
        } => LossKind::Wrm {
            nnue2score,
            in_scaling,
            in_offset,
            target_offset,
            target_scaling,
            pow_exp: 2.5,
            qp_asymmetry: 0.2,
            weight_boost_w1: 1.5,
            weight_boost_w2: 0.75,
        },
        LossKind::Sigmoid { .. } => unreachable!(),
    };
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags::default(),
        extended,
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_after_one_step(
    id: SimpleId,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags::default(),
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_after_one_step_with_options(
    id: SimpleId,
    optimizer: OptimizerKind,
    precision: PrecisionFlags,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_training_options(
        id,
        optimizer,
        None,
        precision,
        SMOKE_LOSS_WRM,
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_after_one_step_with_training_options(
    id: SimpleId,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
    loss: LossKind,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let mut oxide = create_trainer_with_options(
        &context,
        id,
        false,
        SMOKE_BATCH,
        optimizer,
        norm_loss_factor,
        precision,
    )?;
    let mut native = create_trainer_with_options(
        &context,
        id,
        true,
        SMOKE_BATCH,
        optimizer,
        norm_loss_factor,
        precision,
    )?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    let oxide_loss = oxide.forward(&batch.as_ref(), 0.0, loss)?;
    let native_loss = native.forward(&batch.as_ref(), 0.0, loss)?;
    if id.feature_set.ft_factorize() {
        let oxide_master = oxide.ft_w_to_host()?;
        let native_master = native.ft_w_to_host()?;
        assert_weight_group_close("ft_w_master", &oxide_master, &native_master);
    }
    let oxide_weights = oxide.to_simple_weights()?;
    let native_weights = native.to_simple_weights()?;

    let loss_difference = (oxide_loss - native_loss).abs();
    assert!(
        loss_difference <= 2.0e-6 * (1.0 + oxide_loss.abs()),
        "loss differs: oxide={oxide_loss}, native={native_loss}, diff={loss_difference}"
    );
    for (name, oxide_group, native_group) in [
        ("ft_w", &oxide_weights.ft_w, &native_weights.ft_w),
        ("ft_b", &oxide_weights.ft_b, &native_weights.ft_b),
        ("l1_w", &oxide_weights.l1_w, &native_weights.l1_w),
        ("l1_b", &oxide_weights.l1_b, &native_weights.l1_b),
        ("l2_w", &oxide_weights.l2_w, &native_weights.l2_w),
        ("l2_b", &oxide_weights.l2_b, &native_weights.l2_b),
        ("l3_w", &oxide_weights.l3_w, &native_weights.l3_w),
        ("l3_b", &oxide_weights.l3_b, &native_weights.l3_b),
    ] {
        assert_weight_group_close(name, oxide_group, native_group);
    }
    Ok(())
}

#[cfg(feature = "native-cuda")]
fn assert_weight_group_close(name: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    let mut maximum_difference = 0.0_f32;
    let mut maximum_bound = 0.0_f32;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let difference = (expected - actual).abs();
        let bound = 2.0e-6 * (1.0 + expected.abs());
        maximum_difference = maximum_difference.max(difference);
        maximum_bound = maximum_bound.max(bound);
        assert!(
            difference <= bound,
            "{name} differs: expected={expected}, actual={actual}, diff={difference}, bound={bound}"
        );
    }
    eprintln!(
        "[native-parity] {name}: max_abs_diff={maximum_difference:.3e}, max_bound={maximum_bound:.3e}"
    );
}

#[cfg(feature = "native-cuda")]
fn benchmark_backend(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    batch: &BatchData,
    native: bool,
    steps: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut trainer = create_trainer(context, id, native, batch.n_pos)?;
    for _ in 0..3 {
        let _ = trainer.step(batch, 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    }
    let _ = TrainerBackend::flush_pending_loss(&mut trainer)?;
    let start = std::time::Instant::now();
    for _ in 0..steps {
        let _ = trainer.step(batch, 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    }
    let _ = TrainerBackend::flush_pending_loss(&mut trainer)?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(batch.n_pos as f64 * steps as f64 / elapsed)
}

#[test]
#[cfg(feature = "native-cuda")]
#[ignore = "manual WSL performance comparison"]
fn benchmark_standard_simple_native_against_cuda_oxide() -> Result<(), Box<dyn std::error::Error>> {
    let parse = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let batch_size = parse("TATARA_NATIVE_BENCH_BATCH", 16_384);
    let steps = parse("TATARA_NATIVE_BENCH_STEPS", 20);
    let context = CudaContext::new(0)?;
    let id = standard_id();
    let mut owned = BatchData::smoke_dummy(batch_size, id.feature_set);
    owned.score.fill(200.0);
    owned.wdl.fill(0.8);
    let batch = owned.as_ref();

    let oxide = benchmark_backend(&context, id, &batch, false, steps)?;
    let native = benchmark_backend(&context, id, &batch, true, steps)?;
    eprintln!(
        "[native-bench] batch={batch_size}, steps={steps}, cuda-oxide={oxide:.0} pos/s, native={native:.0} pos/s, ratio={:.3}",
        native / oxide
    );
    Ok(())
}
