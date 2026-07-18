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

#[test]
fn every_simple_native_kernel_is_exported() {
    let driver = include_str!("../trainer_simple.rs");
    let native = include_str!("../../../../crates/cuda-native-runtime/kernels/native_kernels.cu");
    let mut required = std::collections::BTreeSet::new();
    for line in driver.lines() {
        let Some((_, suffix)) = line.split_once("kernel:") else {
            continue;
        };
        let symbol = suffix
            .trim_start()
            .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .next()
            .unwrap_or_default();
        if !symbol.is_empty() {
            required.insert(symbol);
        }
    }
    let missing: Vec<_> = required
        .iter()
        .copied()
        .filter(|symbol| !native.contains(&format!("extern \"C\" __global__ void {symbol}(")))
        .collect();
    assert_eq!(required.len(), 46, "Simple kernel inventory changed");
    assert!(missing.is_empty(), "native CUDA is missing: {missing:?}");
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
fn complete_simple_native_configuration_matrix_runs_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let all_fp16 = PrecisionFlags {
        ft_fp16: true,
        ft_fp16_out: true,
        fp16_opt_state: true,
        ..PrecisionFlags::default()
    };
    let extended_wrm = match SMOKE_LOSS_WRM {
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

    for (activation, optimizer, norm_loss, loss) in [
        (
            SimpleActivation::CReLU,
            OptimizerKind::Ranger,
            None,
            SMOKE_LOSS_WRM,
        ),
        (
            SimpleActivation::SCReLU,
            OptimizerKind::RAdam,
            None,
            LossKind::Sigmoid { scale: 1.0 / 600.0 },
        ),
        (
            SimpleActivation::Pairwise,
            OptimizerKind::AdamW,
            Some(0.25),
            extended_wrm,
        ),
    ] {
        let mut id = standard_id();
        id.activation = activation;
        assert_native_configuration_runs(&context, id, optimizer, norm_loss, all_fp16, loss)?;
    }

    assert_native_configuration_runs(
        &context,
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags {
            ft_fp16: true,
            ..PrecisionFlags::default()
        },
        SMOKE_LOSS_WRM,
    )?;

    let mut factorized = standard_id();
    factorized.feature_set = factorized.feature_set.with_ft_factorize();
    assert_native_configuration_runs(
        &context,
        factorized,
        OptimizerKind::Ranger,
        None,
        all_fp16,
        SMOKE_LOSS_WRM,
    )?;

    let mut wide = standard_id();
    wide.l1_out = 257;
    wide.l2_out = 257;
    assert_native_configuration_runs(
        &context,
        wide,
        OptimizerKind::Ranger,
        None,
        PrecisionFlags {
            tf32: true,
            ..PrecisionFlags::default()
        },
        SMOKE_LOSS_WRM,
    )?;

    for feature_set in FeatureSet::ALL {
        let mut id = standard_id();
        id.feature_set = feature_set.spec();
        id.ft_out = 32;
        id.l1_out = 16;
        id.l2_out = 16;
        assert_native_configuration_runs(
            &context,
            id,
            OptimizerKind::Ranger,
            None,
            PrecisionFlags::default(),
            SMOKE_LOSS_WRM,
        )?;
    }
    Ok(())
}

fn assert_native_configuration_runs(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
    loss: LossKind,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = create_trainer_with_options(
        context,
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
    let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    let forward_loss = trainer.forward(&batch.as_ref(), 0.0, loss)?;
    assert!(forward_loss.is_finite());
    trainer.assert_all_weights_finite()?;
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
fn ft_fp16_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_out_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_out_screlu_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.activation = SimpleActivation::SCReLU;
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn ft_fp16_out_pairwise_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.activation = SimpleActivation::Pairwise;
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn fp16_optimizer_state_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_fp16_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn factorized_all_fp16_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        id,
        OptimizerKind::Ranger,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_fp16_adamw_simple_native_matches_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_after_one_step_with_options(
        standard_id(),
        OptimizerKind::AdamW,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_fp16_ranger_lookahead_simple_native_matches_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    assert_native_matches_cuda_oxide_with_training_options_and_steps(
        standard_id(),
        OptimizerKind::Ranger,
        None,
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        },
        SMOKE_LOSS_WRM,
        6,
    )
}

#[test]
#[cfg(feature = "native-cuda")]
fn all_feature_sets_simple_native_match_cuda_oxide_after_one_step()
-> Result<(), Box<dyn std::error::Error>> {
    for feature_set in FeatureSet::ALL {
        let mut id = standard_id();
        id.feature_set = feature_set.spec();
        id.ft_out = 32;
        id.l1_out = 16;
        id.l2_out = 16;
        assert_native_matches_cuda_oxide_after_one_step(id)?;
    }
    Ok(())
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
    assert_native_matches_cuda_oxide_with_training_options_and_steps(
        id,
        optimizer,
        norm_loss_factor,
        precision,
        loss,
        1,
    )
}

#[cfg(feature = "native-cuda")]
fn assert_native_matches_cuda_oxide_with_training_options_and_steps(
    id: SimpleId,
    optimizer: OptimizerKind,
    norm_loss_factor: Option<f32>,
    precision: PrecisionFlags,
    loss: LossKind,
    steps: usize,
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

    for _ in 0..steps {
        let _ = oxide.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
        let _ = native.step(&batch.as_ref(), 1.0e-3, 0.0, loss)?;
    }
    let oxide_loss = oxide.forward(&batch.as_ref(), 0.0, loss)?;
    let native_loss = native.forward(&batch.as_ref(), 0.0, loss)?;
    assert_trainers_close(id, oxide_loss, native_loss, &oxide, &native)
}

#[cfg(feature = "native-cuda")]
fn assert_trainers_close(
    id: SimpleId,
    oxide_loss: f64,
    native_loss: f64,
    oxide: &SimpleGpuTrainer,
    native: &SimpleGpuTrainer,
) -> Result<(), Box<dyn std::error::Error>> {
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

/// raw checkpoint が backend 非依存であることを、optimizer state と Ranger の
/// `step_count` まで含めて固定する。片方の backend で 5 step 進めて保存し、同じ
/// checkpoint を cuda-oxide / CUDA C++ の両方で読み、6 step 目 (lookahead 発火点)
/// を進めた結果を比較する。保存元も両 backend を試すため、双方向の resume を覆う。
#[test]
#[cfg(feature = "native-cuda")]
fn checkpoint_resume_simple_native_matches_cuda_oxide_in_both_directions()
-> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let id = SimpleId {
        feature_set: FeatureSet::HalfKaHmMerged.spec().with_ft_factorize(),
        activation: SimpleActivation::Pairwise,
        ft_out: 8,
        l1_out: 8,
        l2_out: 8,
    };
    let precision = PrecisionFlags {
        ft_fp16: true,
        ft_fp16_out: true,
        fp16_opt_state: true,
        ..PrecisionFlags::default()
    };
    let mut batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    batch.score.fill(200.0);
    batch.wdl.fill(0.8);

    for source_is_native in [false, true] {
        let source_name = if source_is_native { "native" } else { "oxide" };
        let path = std::env::temp_dir().join(format!(
            "tatara-simple-native-resume-{source_name}-{}.ckpt",
            std::process::id()
        ));
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let mut source = create_trainer_with_options(
                &context,
                id,
                source_is_native,
                SMOKE_BATCH,
                OptimizerKind::Ranger,
                None,
                precision,
            )?;
            for _ in 0..5 {
                let _ = source.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
            }
            source.save_raw_checkpoint(&path, 17, source_name, Some(42))?;
            drop(source);

            let mut oxide = create_trainer_with_options(
                &context,
                id,
                false,
                SMOKE_BATCH,
                OptimizerKind::Ranger,
                None,
                precision,
            )?;
            let mut native = create_trainer_with_options(
                &context,
                id,
                true,
                SMOKE_BATCH,
                OptimizerKind::Ranger,
                None,
                precision,
            )?;
            for trainer in [&mut oxide, &mut native] {
                let (superbatch, producer, horizon) = trainer.load_raw_checkpoint(&path)?;
                assert_eq!(superbatch, 17);
                assert_eq!(producer.as_deref(), Some(source_name));
                assert_eq!(horizon, Some(42));
                trainer.sync_ft_forward_weights()?;
                let _ = trainer.step(&batch.as_ref(), 1.0e-3, 0.0, SMOKE_LOSS_WRM)?;
            }
            let oxide_loss = oxide.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
            let native_loss = native.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
            assert_trainers_close(id, oxide_loss, native_loss, &oxide, &native)
        })();
        let _ = std::fs::remove_file(&path);
        result?;
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

#[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
fn benchmark_backend(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    batch: &BatchData,
    native: bool,
    steps: usize,
    precision: PrecisionFlags,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut trainer = create_trainer_with_options(
        context,
        id,
        native,
        batch.n_pos,
        OptimizerKind::Ranger,
        None,
        precision,
    )?;
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

#[cfg(feature = "native-cuda")]
fn benchmark_backends_alternating(
    context: &std::sync::Arc<CudaContext>,
    id: SimpleId,
    batch: &BatchData,
    steps: usize,
    precision: PrecisionFlags,
    runs: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let mut oxide_total = 0.0;
    let mut native_total = 0.0;
    for run in 0..runs {
        if run.is_multiple_of(2) {
            oxide_total += benchmark_backend(context, id, batch, false, steps, precision)?;
            native_total += benchmark_backend(context, id, batch, true, steps, precision)?;
        } else {
            native_total += benchmark_backend(context, id, batch, true, steps, precision)?;
            oxide_total += benchmark_backend(context, id, batch, false, steps, precision)?;
        }
    }
    Ok((oxide_total / runs as f64, native_total / runs as f64))
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
    let runs = parse("TATARA_NATIVE_BENCH_RUNS", 3).max(1);
    let context = CudaContext::new(0)?;
    let id = standard_id();
    let mut owned = BatchData::smoke_dummy(batch_size, id.feature_set);
    owned.score.fill(200.0);
    owned.wdl.fill(0.8);
    let batch = owned.as_ref();

    let (oxide, native) = benchmark_backends_alternating(
        &context,
        id,
        &batch,
        steps,
        PrecisionFlags::default(),
        runs,
    )?;
    eprintln!(
        "[native-bench] batch={batch_size}, steps={steps}, runs={runs}, cuda-oxide={oxide:.0} pos/s, native={native:.0} pos/s, ratio={:.3}",
        native / oxide
    );
    Ok(())
}

#[test]
#[cfg(feature = "native-cuda")]
#[ignore = "manual WSL performance comparison"]
fn benchmark_factorized_fp16_simple_native_against_cuda_oxide()
-> Result<(), Box<dyn std::error::Error>> {
    let parse = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let batch_size = parse("TATARA_NATIVE_BENCH_BATCH", 16_384);
    let steps = parse("TATARA_NATIVE_BENCH_STEPS", 20);
    let runs = parse("TATARA_NATIVE_BENCH_RUNS", 3).max(1);
    let context = CudaContext::new(0)?;
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    let precision = PrecisionFlags {
        ft_fp16: true,
        ft_fp16_out: true,
        fp16_opt_state: true,
        ..PrecisionFlags::default()
    };
    let mut owned = BatchData::smoke_dummy(batch_size, id.feature_set);
    owned.score.fill(200.0);
    owned.wdl.fill(0.8);
    let batch = owned.as_ref();

    let (oxide, native) =
        benchmark_backends_alternating(&context, id, &batch, steps, precision, runs)?;
    eprintln!(
        "[native-bench-fp16] batch={batch_size}, steps={steps}, runs={runs}, cuda-oxide={oxide:.0} pos/s, native={native:.0} pos/s, ratio={:.3}",
        native / oxide
    );
    Ok(())
}

/// cuda-oxideをcompileできないnative Windowsでも、WSLと同じdummy batch・precisionで
/// CUDA C++ backend単体のthroughputを測る。backend間比較は上のhybrid test、OS間の
/// portability確認は本testと役割を分ける。
#[test]
#[ignore = "manual portable native performance comparison"]
fn benchmark_factorized_fp16_simple_native_portable() -> Result<(), Box<dyn std::error::Error>> {
    let parse = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let batch_size = parse("TATARA_NATIVE_BENCH_BATCH", 16_384);
    let steps = parse("TATARA_NATIVE_BENCH_STEPS", 20);
    let runs = parse("TATARA_NATIVE_BENCH_RUNS", 3).max(1);
    let context = CudaContext::new(0)?;
    let mut id = standard_id();
    id.feature_set = id.feature_set.with_ft_factorize();
    let precision = PrecisionFlags {
        ft_fp16: true,
        ft_fp16_out: true,
        fp16_opt_state: true,
        ..PrecisionFlags::default()
    };
    let mut owned = BatchData::smoke_dummy(batch_size, id.feature_set);
    owned.score.fill(200.0);
    owned.wdl.fill(0.8);
    let batch = owned.as_ref();

    let mut total = 0.0;
    for _ in 0..runs {
        total += benchmark_backend(&context, id, &batch, true, steps, precision)?;
    }
    let native = total / runs as f64;
    eprintln!(
        "[native-bench-portable-fp16] batch={batch_size}, steps={steps}, runs={runs}, native={native:.0} pos/s"
    );
    Ok(())
}
