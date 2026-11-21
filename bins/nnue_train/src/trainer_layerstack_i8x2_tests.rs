use super::*;
use crate::tests::gpu_test_context::test_cuda_context_reserving_gib;
use nnue_format::layerstack_weights::{QA, coalesce_ft_factorized};
use shogi_features::{FeatureSet, ThreatProfile};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn forward_cache(trainer: &GpuTrainer) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let half = trainer.ft_w_h.as_ref();
    #[cfg(feature = "native")]
    let half = half.or(trainer.partial_ft_w_h.as_ref());
    if let Some(half) = half {
        Ok(half
            .to_host_vec(&trainer.stream)?
            .into_iter()
            .map(|w| w as f32)
            .collect())
    } else {
        Ok(trainer
            .ft_w_fold32
            .as_ref()
            .unwrap()
            .to_host_vec(&trainer.stream)?)
    }
}

fn bin_round_trip(weights: &LayerStackWeights) -> std::io::Result<LayerStackWeights> {
    let mut bytes = Vec::new();
    weights.save_quantised(&mut bytes, None)?;
    LayerStackWeights::load_quantised(
        &mut bytes.as_slice(),
        weights.feature_set,
        weights.ft_b.len(),
        weights.l1_shared_bias.len(),
        weights.l2_b.len() / weights.num_buckets,
        weights.num_buckets,
    )
}

fn projected_units(weight: f32) -> f32 {
    2.0 * (weight * (QA as f32 / 2.0)).round().clamp(-127.0, 127.0)
}

#[test]
fn piece_ft_qat_rounds_coalesced_forward_weights_after_optimizer_and_lookahead() -> TestResult {
    let ctx = test_cuda_context_reserving_gib(3)?;
    let piece = FeatureSet::HalfKaHmMerged.spec();
    let threat = piece.with_threat_profile(ThreatProfile::CrossSide);
    let spec: BucketMode = BucketMode::Spec("ownking:2/3".parse()?);
    let configurations = [
        (
            piece,
            128,
            BucketMode::ProgressKpAbs,
            2,
            true,
            OptimizerKind::Ranger,
        ),
        (
            piece.with_ft_factorize(),
            768,
            spec,
            3,
            true,
            OptimizerKind::Ranger,
        ),
        (
            piece.with_ft_factorize(),
            128,
            spec,
            3,
            false,
            OptimizerKind::Ranger,
        ),
        (
            threat.with_ft_factorize(),
            128,
            spec,
            3,
            true,
            OptimizerKind::RAdam,
        ),
        (
            threat.with_threat_width(32).with_ft_factorize(),
            128,
            spec,
            3,
            true,
            OptimizerKind::Sgd,
        ),
    ];

    let mut init = LayerStackInit::default_uniform();
    init.enable_piece_ft_i8x2_init();
    for (features, width, bucket_mode, buckets, _fused, optimizer) in configurations {
        let make_trainer = |ft_state_host| {
            GpuTrainer::new(
                &ctx,
                16,
                width,
                16,
                32,
                buckets,
                bucket_mode,
                PrecisionFlags {
                    tf32: true,
                    ft_fp16: true,
                    ft_fp16_out: true,
                    fp16_opt_state: true,
                    ft_state_host,
                },
                features,
                optimizer,
                OptimGroupConfig::resolve(0.0, Some(1.0), None, None, Some(32.0), None, None),
                None,
                None,
                &init,
            )
        };
        let mut trainer = make_trainer(false)?;
        #[cfg(feature = "native")]
        trainer.set_ft_fused_optimizer_enabled_for_test(_fused);

        let mut initial = trainer.ft_w.to_host_vec(&trainer.stream)?;
        let piece_end = trainer.piece_ft_rows() * width;
        // Real row 0 uses virtual piece-input row 0. Its negative activation in
        // column 3 leaves weight decay in control of the first step's update.
        let real = 3;
        let factor = features
            .ft_factorize()
            .then(|| features.ft_weight_len(width) + real);
        initial[..4].copy_from_slice(&[2.5, -2.5, 0.009, -0.5]);
        if let Some(factor) = factor {
            initial[factor - 3..=factor].copy_from_slice(&[0.2, -0.2, 0.01, -0.25]);
        }
        trainer.ft_w = DeviceBuffer::from_host(&trainer.stream, &initial)?;
        let mut bias = trainer.ft_b.to_host_vec(&trainer.stream)?;
        bias[real] = -1.0;
        trainer.ft_b = DeviceBuffer::from_host(&trainer.stream, &bias)?;
        if !features.ft_factorize() {
            let warm_start = bin_round_trip(&trainer.to_layerstack_weights()?)?;
            assert!(warm_start.validate_piece_ft_i8x2().is_err());
            trainer.load_layerstack_weights(&warm_start)?;
            initial = warm_start.ft_w;
        }
        trainer.set_ft_weight_i8x2(true)?;
        trainer.sync_ft_forward_weights()?;
        let mut batch = BatchData::smoke_dummy(16, features);
        for (row, bucket) in batch.bucket_idx.iter_mut().enumerate() {
            *bucket = (row % buckets) as i32;
        }
        batch.score.fill(180.0);
        batch.wdl.fill(0.8);
        for indices in [&mut batch.stm_indices, &mut batch.nstm_indices] {
            for row in indices.chunks_exact_mut(features.max_active()) {
                row[0] = 0;
            }
        }
        let full_nnz = batch.nnz[0];
        let probe = |master: &[f32]| master[real] + factor.map_or(0.0, |i| master[i]);
        let initial_units = projected_units(probe(&initial));

        for step in 0..=RANGER_K {
            if step > 0 {
                batch.nnz.fill(if step == 1 { 1 } else { full_nnz });
                assert!(
                    trainer
                        .step(&batch.as_ref(), 0.001, 0.5, SMOKE_LOSS_WRM)?
                        .is_finite()
                );
            }
            if step != 0 && step != 1 && step != RANGER_K {
                continue;
            }

            let master = trainer.ft_w.to_host_vec(&trainer.stream)?;
            let raw = coalesce_ft_factorized(&features, width, &master);
            let exported = trainer.to_layerstack_weights()?;
            exported.validate_piece_ft_i8x2()?;
            assert_ne!(&exported.ft_w[..piece_end], &raw[..piece_end]);
            assert_eq!(&exported.ft_w[piece_end..], &raw[piece_end..]);
            if step > 0 {
                assert_ne!(master, initial, "the optimizer must update the FP32 master");
            }

            let cache = forward_cache(&trainer)?;
            let units = projected_units(probe(&master));
            assert_eq!(cache[real], (units / QA as f32) as f16 as f32);
            if step == 1 {
                assert_ne!(master[real], initial[real]);
                if let Some(factor) = factor {
                    assert_ne!(master[factor], initial[factor]);
                }
                assert_ne!(
                    units, initial_units,
                    "the first update must cross a grid level"
                );
            }
            assert_eq!(cache.len(), exported.ft_w.len());
            for (index, (&cached, &weight)) in cache.iter().zip(&exported.ft_w).enumerate() {
                assert_eq!(
                    cached,
                    (weight as f16) as f32,
                    "step {step}, FT element {index}"
                );
            }

            if step == RANGER_K {
                let loaded = bin_round_trip(&exported)?;
                loaded.validate_piece_ft_i8x2()?;
                assert_eq!(loaded.bucket_spec, exported.bucket_spec);
                for (index, (&decoded, &cached)) in
                    loaded.ft_w[..piece_end].iter().zip(&cache).enumerate()
                {
                    let cached_units = (cached * QA as f32).round();
                    assert_eq!(
                        decoded,
                        cached_units / QA as f32,
                        "piece FT element {index}"
                    );
                }
                if !features.ft_factorize() {
                    trainer.load_layerstack_weights(&loaded)?;
                    trainer.set_freeze_ft(true);
                    trainer.sync_ft_forward_weights()?;
                    assert_eq!(forward_cache(&trainer)?, cache);
                    assert!(
                        trainer
                            .step(&batch.as_ref(), 0.001, 0.5, SMOKE_LOSS_WRM)?
                            .is_finite()
                    );
                    assert_eq!(trainer.ft_w.to_host_vec(&trainer.stream)?, loaded.ft_w);
                    assert_eq!(trainer.ft_b.to_host_vec(&trainer.stream)?, loaded.ft_b);
                    assert_eq!(forward_cache(&trainer)?, cache);
                }
                #[cfg(feature = "native")]
                if features.ft_factorize() && !_fused {
                    let mut resumed = make_trainer(true)?;
                    resumed.set_ft_fused_optimizer_enabled_for_test(false);
                    check_resume(&trainer, &mut resumed, &cache)?;
                    batch.nnz.fill(1);
                    for current in [&mut trainer, &mut resumed] {
                        assert!(
                            current
                                .step(&batch.as_ref(), 0.001, 0.5, SMOKE_LOSS_WRM)?
                                .is_finite()
                        );
                        assert_eq!(current.step_count, RANGER_K + 1);
                    }
                    let resumed_master = resumed.ft_w.to_host_vec(&resumed.stream)?;
                    let continued_master = trainer.ft_w.to_host_vec(&trainer.stream)?;
                    assert_eq!(probe(&resumed_master), probe(&continued_master));
                    assert_eq!(
                        forward_cache(&resumed)?[real],
                        forward_cache(&trainer)?[real]
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "native")]
fn check_resume(trainer: &GpuTrainer, resumed: &mut GpuTrainer, cache: &[f32]) -> TestResult {
    let path =
        std::env::temp_dir().join(format!("tatara-piece-ft-i8x2-{}.ckpt", std::process::id()));
    trainer.save_raw_checkpoint(
        &path,
        1,
        &CheckpointProducer::run_id_only("piece-ft-i8x2"),
        None,
    )?;
    resumed.set_ft_weight_i8x2(true)?;
    let result = resumed.load_raw_checkpoint(&path, false);
    std::fs::remove_file(path)?;
    assert_eq!(result?.0, 1);
    resumed.sync_ft_forward_weights()?;
    assert!(resumed.ft_state_mapped());
    assert_eq!(
        resumed.raw_checkpoint_state_to_host()?,
        trainer.raw_checkpoint_state_to_host()?
    );
    assert_eq!(forward_cache(resumed)?, cache);
    Ok(())
}
