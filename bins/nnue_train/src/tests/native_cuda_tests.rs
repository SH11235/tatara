use gpu_runtime::CudaContext;
use nnue_format::{SimpleActivation, SimpleId};
use nnue_train::{init::SimpleInit, optimizer::OptimizerKind, trainer::TrainerBackend};
use shogi_features::FeatureSet;

#[cfg(feature = "native-cuda")]
use crate::kernel_module::with_test_native_backend;
use crate::{
    arch::{SMOKE_BATCH, SMOKE_LOSS_WRM},
    trainer_common::{BatchData, PrecisionFlags},
    trainer_simple::SimpleGpuTrainer,
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
    #[cfg(feature = "native-cuda")]
    let result = with_test_native_backend(native, || {
        SimpleGpuTrainer::new(
            context,
            batch_size,
            id,
            OptimizerKind::Ranger,
            0.0,
            None,
            16,
            PrecisionFlags::default(),
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
            OptimizerKind::Ranger,
            0.0,
            None,
            16,
            PrecisionFlags::default(),
            &SimpleInit::default_uniform(),
        )
    };
    result
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
    let context = CudaContext::new(0)?;
    let id = standard_id();
    let mut oxide = create_trainer(&context, id, false, SMOKE_BATCH)?;
    let mut native = create_trainer(&context, id, true, SMOKE_BATCH)?;
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
    let oxide_loss = oxide.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
    let native_loss = native.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
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
        assert_eq!(oxide_group.len(), native_group.len());
        let mut maximum_difference = 0.0_f32;
        let mut maximum_bound = 0.0_f32;
        for (&expected, &actual) in oxide_group.iter().zip(native_group) {
            let difference = (expected - actual).abs();
            let bound = 2.0e-6 * (1.0 + expected.abs());
            maximum_difference = maximum_difference.max(difference);
            maximum_bound = maximum_bound.max(bound);
            assert!(
                difference <= bound,
                "{name} differs: oxide={expected}, native={actual}, diff={difference}, bound={bound}"
            );
        }
        eprintln!(
            "[native-parity] {name}: max_abs_diff={maximum_difference:.3e}, max_bound={maximum_bound:.3e}"
        );
    }
    Ok(())
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
