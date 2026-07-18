use gpu_runtime::CudaContext;
use nnue_format::LayerStackWeights;
use nnue_format::{ArchKind, SimpleActivation, SimpleId, SimpleWeights};
use nnue_train::init::{LayerStackInit, SimpleInit};
use nnue_train::optimizer::OptimizerKind;
use shogi_features::FeatureSet;

use crate::{arch::*, trainer_common::*, trainer_layerstack::*, trainer_simple::*};

fn native_simple_smoke_scope() -> bool {
    #[cfg(any(feature = "native-cuda", feature = "native-cuda-host"))]
    {
        crate::kernel_module::native_backend_requested()
    }
    #[cfg(not(any(feature = "native-cuda", feature = "native-cuda-host")))]
    {
        false
    }
}

/// Simple アーキ用 smoke test。preset `256x2-32-32` (HalfKaHmMerged + CReLU) で
/// `SimpleGpuTrainer` を構築し、以下 4 段を踏む。native backendではfactorizerと
/// FP16 weight/output/optimizer stateを同時に有効化し、portable host pathも含めて検査する。
/// cuda-oxide backendでは追加の活性化/loss経路も検査する:
/// 1. forward sanity — CReLU + SCReLU 両活性化 + sigmoid / WRM 両 loss kernel を
///    1 step ずつ launch して loss が finite であること。
/// 2. step が gradient を正しく配線していることを 10 step の loss 推移で確認する
///    (loss が初期値より下がる)。
/// 3. `to_simple_weights` → `save_quantised` → `SimpleWeights::load` →
///    `load_simple_weights` の量子化 round-trip 後の forward が finite に走る。
/// 4. `save_raw_checkpoint` → 新 trainer での `load_raw_checkpoint` 後の forward が
///    元と完全一致 (raw f32 round-trip の exact preservation)。
pub(crate) fn simple_smoke_test() -> Result<(), Box<dyn std::error::Error>> {
    let native_scope = native_simple_smoke_scope();
    let primary_loss = if native_scope {
        SMOKE_LOSS_WRM
    } else {
        SMOKE_LOSS_SIGMOID
    };
    let primary_loss_name = if native_scope {
        "default WRM"
    } else {
        "sigmoid-MSE"
    };
    let ctx = CudaContext::new(0)?;
    println!("[smoke/simple] CUDA context created, loading kernel module...");
    if native_scope {
        println!(
            "[smoke/simple] native scope: CReLU, FT factorizer, FP16 weight/output/state, \
             Ranger, default WRM"
        );
    }
    let feature_set = if native_scope {
        FeatureSet::HalfKaHmMerged.spec().with_ft_factorize()
    } else {
        FeatureSet::HalfKaHmMerged.spec()
    };
    let id = SimpleId {
        feature_set,
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
    };
    let smoke_fv_scale = 16_i32;
    let smoke_weight_decay = 1e-7_f32;
    let smoke_precision = if native_scope {
        PrecisionFlags {
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: true,
            ..PrecisionFlags::default()
        }
    } else {
        PrecisionFlags::default()
    };
    let mut trainer = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        OptimizerKind::Ranger,
        smoke_weight_decay,
        None,
        smoke_fv_scale,
        smoke_precision,
        &SimpleInit::default_uniform(),
    )?;
    trainer.sync_ft_forward_weights()?;
    let params = id.ft_in() * id.ft_out
        + id.ft_out
        + id.combined_dim() * id.l1_out
        + id.l1_out
        + id.l1_out * id.l2_out
        + id.l2_out
        + id.l2_out
        + 1;
    println!(
        "[smoke/simple] SimpleGpuTrainer ready: 8 weight groups, ~{:.1}M params total \
         (ft_in={}, ft_out={}, l1_out={}, l2_out={}, activation={})",
        params as f64 / 1.0e6,
        id.ft_in(),
        id.ft_out,
        id.l1_out,
        id.l2_out,
        id.activation.canonical_name(),
    );
    trainer.assert_all_weights_finite()?;

    let batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    let loss = trainer.forward(&batch.as_ref(), 0.0, primary_loss)?;
    println!("[smoke/simple] forward 1 ({primary_loss_name}, crelu): loss = {loss:.6e}");
    if !loss.is_finite() {
        return Err(format!("forward 1 loss = {loss} is not finite").into());
    }
    trainer.assert_all_weights_finite()?;

    if !native_scope {
        let id_screlu = SimpleId {
            activation: SimpleActivation::SCReLU,
            ..id
        };
        let mut trainer_screlu = SimpleGpuTrainer::new(
            &ctx,
            SMOKE_BATCH,
            id_screlu,
            OptimizerKind::Ranger,
            smoke_weight_decay,
            None,
            smoke_fv_scale,
            PrecisionFlags::default(),
            &SimpleInit::default_uniform(),
        )?;
        let loss_screlu = trainer_screlu.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke/simple] forward 2 (sigmoid-MSE, screlu): loss = {loss_screlu:.6e}");
        if !loss_screlu.is_finite() {
            return Err(format!("forward 2 loss = {loss_screlu} is not finite").into());
        }

        let loss_wrm_val = trainer.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_WRM)?;
        println!("[smoke/simple] forward 3 (win-rate-model, crelu): loss = {loss_wrm_val:.6e}");
        if !loss_wrm_val.is_finite() {
            return Err(format!("forward 3 (wrm) loss = {loss_wrm_val} is not finite").into());
        }
    }
    println!("[smoke/simple] forward sanity OK ✓");

    // 10-step gradient direction check。`smoke_dummy` は score=0 / wdl=0.5 で
    // sigmoid loss の最小点 (target=0.5、init weights が小さく net_output≈0、
    // p≈0.5) に近すぎるため、score を非ゼロにして target を 0.5 から動かし
    // 学習信号を作る。
    let mut training_batch = BatchData::smoke_dummy(SMOKE_BATCH, id.feature_set);
    for s in training_batch.score.iter_mut() {
        *s = 200.0;
    }
    for w in training_batch.wdl.iter_mut() {
        *w = 0.8;
    }
    let lr = if native_scope { 1e-3_f32 } else { 1e-1_f32 };
    let initial_loss = trainer.forward(&training_batch.as_ref(), 0.0, primary_loss)?;
    for step_idx in 0..10 {
        let step_loss = trainer.step(&training_batch.as_ref(), lr, 0.0, primary_loss)?;
        if !step_loss.is_finite() {
            return Err(format!("step {step_idx} loss = {step_loss} is not finite").into());
        }
    }
    trainer.assert_all_weights_finite()?;
    let final_loss = trainer.forward(&training_batch.as_ref(), 0.0, primary_loss)?;
    println!("[smoke/simple] 10-step training: loss {initial_loss:.6e} -> {final_loss:.6e}");
    // NaN は `>=` でも `<` でも false になるので、`is_finite` で別途弾く。
    if !final_loss.is_finite() || final_loss >= initial_loss {
        return Err(format!(
            "10-step training did not decrease loss: initial = {initial_loss}, final = {final_loss} \
             (backward / optimizer wiring likely broken)"
        )
        .into());
    }
    println!("[smoke/simple] gradient direction OK ✓");

    // 量子化 round-trip: `to_simple_weights` -> `save_quantised` -> `SimpleWeights::load`
    // -> `load_simple_weights`。量子化丸めで weight 値は変わるため loss は厳密一致しない;
    // ここでは round-trip が format 上完結し、再 upload した weight で forward が finite
    // に走ることを確認する。bit-identical な round-trip は次段の raw checkpoint で確認する。
    let weights = trainer.to_simple_weights()?;
    let mut quantised_bytes = Vec::new();
    weights.save_quantised(&mut quantised_bytes)?;
    // Factorized training exports folded base-shape weights, so preserve the exported ID when
    // loading them instead of reattaching the trainer's virtual-row factorizer modifier.
    let reloaded = SimpleWeights::load(&mut std::io::Cursor::new(&quantised_bytes), weights.id)?;
    if reloaded.fv_scale != smoke_fv_scale {
        return Err(format!(
            "SimpleWeights round-trip: fv_scale mismatch (got {}, want {smoke_fv_scale})",
            reloaded.fv_scale
        )
        .into());
    }
    let mut trainer_q = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        reloaded.id,
        OptimizerKind::Ranger,
        smoke_weight_decay,
        None,
        smoke_fv_scale,
        smoke_precision,
        &SimpleInit::default_uniform(),
    )?;
    trainer_q.load_simple_weights(&reloaded)?;
    trainer_q.sync_ft_forward_weights()?;
    let loss_q = trainer_q.forward(&training_batch.as_ref(), 0.0, primary_loss)?;
    println!(
        "[smoke/simple] quantised round-trip: trained loss {final_loss:.6e} \
         -> reloaded loss {loss_q:.6e} ({} bytes)",
        quantised_bytes.len()
    );
    if !loss_q.is_finite() {
        return Err(format!(
            "quantised round-trip forward loss = {loss_q} is not finite \
             (to_simple_weights / load_simple_weights transpose direction or format mismatch)"
        )
        .into());
    }

    // raw f32 checkpoint round-trip: save -> 新 trainer で load -> 同 batch で forward。
    // raw checkpoint は f32 を bit-identical に保つので loss は完全一致するはず。
    let raw_path = std::env::temp_dir().join(format!("simple-smoke-{}.ckpt", std::process::id()));
    trainer.save_raw_checkpoint(&raw_path, 1, "smoke", None)?;
    let mut trainer_r = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id,
        OptimizerKind::Ranger,
        smoke_weight_decay,
        None,
        smoke_fv_scale,
        smoke_precision,
        &SimpleInit::default_uniform(),
    )?;
    let (sb, _producer, _lr_horizon) = trainer_r.load_raw_checkpoint(&raw_path)?;
    trainer_r.sync_ft_forward_weights()?;
    if sb != 1 {
        return Err(format!("raw round-trip superbatch mismatch: got {sb}, want 1").into());
    }
    let loss_r = trainer_r.forward(&training_batch.as_ref(), 0.0, primary_loss)?;
    let _ = std::fs::remove_file(&raw_path);
    let loss_r_rel = ((loss_r - final_loss).abs() / final_loss.abs().max(1e-12)) as f32;
    println!(
        "[smoke/simple] raw checkpoint round-trip: loss {final_loss:.6e} -> {loss_r:.6e} \
         (relative {loss_r_rel:.3e})"
    );
    if !loss_r.is_finite() || loss_r_rel > 1e-6 {
        return Err(format!(
            "raw round-trip loss mismatch: final = {final_loss}, reloaded = {loss_r} \
             (raw f32 should be bit-identical; group ordering / topology / step_count likely broken)"
        )
        .into());
    }

    if native_scope {
        println!(
            "[smoke/simple] PASSED — native factorized FP16 CReLU/default-WRM forward + \
             gradient + quantised round-trip + raw round-trip OK"
        );
        return Ok(());
    }

    // Pairwise 活性化: forward sanity + 10-step gradient + 量子化 round-trip。
    // Pairwise は L1 入力次元が半減する (combined_dim = ft_out) ため、workspace buffer /
    // cuBLAS Sgemm shape / l1_w layout / 量子化 format がその dim で一貫することを確認する。
    let id_pairwise = SimpleId {
        activation: SimpleActivation::Pairwise,
        ..id
    };
    let mut trainer_pw = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id_pairwise,
        OptimizerKind::Ranger,
        smoke_weight_decay,
        None,
        smoke_fv_scale,
        PrecisionFlags::default(),
        &SimpleInit::default_uniform(),
    )?;
    let loss_pw = trainer_pw.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!("[smoke/simple] forward 4 (sigmoid-MSE, pairwise): loss = {loss_pw:.6e}");
    if !loss_pw.is_finite() {
        return Err(format!("forward 4 (pairwise) loss = {loss_pw} is not finite").into());
    }
    let pw_initial_loss = trainer_pw.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    for step_idx in 0..10 {
        let step_loss = trainer_pw.step(&training_batch.as_ref(), lr, 0.0, SMOKE_LOSS_SIGMOID)?;
        if !step_loss.is_finite() {
            return Err(
                format!("pairwise step {step_idx} loss = {step_loss} is not finite").into(),
            );
        }
    }
    trainer_pw.assert_all_weights_finite()?;
    let pw_final_loss = trainer_pw.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!(
        "[smoke/simple] pairwise 10-step training: loss {pw_initial_loss:.6e} -> {pw_final_loss:.6e}"
    );
    if !pw_final_loss.is_finite() || pw_final_loss >= pw_initial_loss {
        return Err(format!(
            "pairwise 10-step training did not decrease loss: initial = {pw_initial_loss}, \
             final = {pw_final_loss} (pairwise backward / optimizer wiring likely broken)"
        )
        .into());
    }

    let pw_weights = trainer_pw.to_simple_weights()?;
    let mut pw_quantised = Vec::new();
    pw_weights.save_quantised(&mut pw_quantised)?;
    let pw_reloaded = SimpleWeights::load(&mut std::io::Cursor::new(&pw_quantised), id_pairwise)?;
    let mut trainer_pw_q = SimpleGpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        id_pairwise,
        OptimizerKind::Ranger,
        smoke_weight_decay,
        None,
        smoke_fv_scale,
        PrecisionFlags::default(),
        &SimpleInit::default_uniform(),
    )?;
    trainer_pw_q.load_simple_weights(&pw_reloaded)?;
    let pw_loss_q = trainer_pw_q.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
    println!(
        "[smoke/simple] pairwise quantised round-trip: trained loss {pw_final_loss:.6e} \
         -> reloaded loss {pw_loss_q:.6e} ({} bytes)",
        pw_quantised.len()
    );
    if !pw_loss_q.is_finite() {
        return Err(format!(
            "pairwise quantised round-trip forward loss = {pw_loss_q} is not finite \
             (combined_dim plumbing or l1_w transpose direction likely broken)"
        )
        .into());
    }
    println!("[smoke/simple] pairwise OK ✓");

    // `--ft-fp16-out` 経路 (FT activation を FP16 で保持) を 3 活性化すべてで確認する。
    // f16 FT-out kernel が活性化別に正しく分岐し、loss scaling 込みの backward が
    // finite に走って loss を下げることを見る。`ft_fp16` を要求するので両 flag ON、
    // `ft_w_h` mirror は `sync_ft_forward_weights` で初期同期する (`run_simple_training` と同じ)。
    for act in [
        SimpleActivation::CReLU,
        SimpleActivation::SCReLU,
        SimpleActivation::Pairwise,
    ] {
        let id_fp16 = SimpleId {
            activation: act,
            ..id
        };
        let mut trainer_fp16 = SimpleGpuTrainer::new(
            &ctx,
            SMOKE_BATCH,
            id_fp16,
            OptimizerKind::Ranger,
            smoke_weight_decay,
            None,
            smoke_fv_scale,
            PrecisionFlags {
                ft_fp16: true, // ft_fp16_out requires this
                ft_fp16_out: true,
                fp16_opt_state: false,
                tf32: false,
            },
            &SimpleInit::default_uniform(),
        )?;
        trainer_fp16.sync_ft_forward_weights()?;
        let fwd = trainer_fp16.forward(&batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
        if !fwd.is_finite() {
            return Err(format!("--ft-fp16-out {act:?} forward loss = {fwd} is not finite").into());
        }
        let init = trainer_fp16.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
        for step_idx in 0..10 {
            let l = trainer_fp16.step(&training_batch.as_ref(), lr, 0.0, SMOKE_LOSS_SIGMOID)?;
            if !l.is_finite() {
                return Err(format!(
                    "--ft-fp16-out {act:?} step {step_idx} loss = {l} is not finite"
                )
                .into());
            }
        }
        trainer_fp16.assert_all_weights_finite()?;
        let fin = trainer_fp16.forward(&training_batch.as_ref(), 0.0, SMOKE_LOSS_SIGMOID)?;
        println!(
            "[smoke/simple] --ft-fp16-out {act:?}: forward {fwd:.6e}, 10-step {init:.6e} -> {fin:.6e}"
        );
        if !fin.is_finite() || fin >= init {
            return Err(format!(
                "--ft-fp16-out {act:?} 10-step training did not decrease loss: \
                 initial = {init}, final = {fin} (FP16 FT activation wiring likely broken)"
            )
            .into());
        }
    }
    println!("[smoke/simple] --ft-fp16-out (crelu/screlu/pairwise) OK ✓");

    println!(
        "[smoke/simple] PASSED — forward + gradient + quantised round-trip + raw round-trip OK"
    );
    Ok(())
}

pub(crate) fn smoke_test(arch_kind: ArchKind) -> Result<(), Box<dyn std::error::Error>> {
    // Simple アーキは別 host pipeline (SimpleGpuTrainer) を持つので smoke も別系統。
    if arch_kind == ArchKind::Simple {
        return simple_smoke_test();
    }
    let ctx = CudaContext::new(0)?;
    println!("[smoke] CUDA context created, loading kernel module...");
    // smoke は production feature set (`halfka-hm-merged`) で動作確認する。
    let feature_set = FeatureSet::HalfKaHmMerged.spec();
    // workspace を smoke の固定 batch 分で確保 (smoke は TF32 OFF 固定で動作確認、
    // training は CLI の `--tf32` を pass する)。
    let mut trainer = GpuTrainer::new(
        &ctx,
        SMOKE_BATCH,
        DEFAULT_FT_OUT,
        DEFAULT_L1_OUT,
        DEFAULT_L2_OUT,
        DEFAULT_NUM_BUCKETS,
        nnue_train::dataloader::BucketMode::Progress8KpAbs,
        PrecisionFlags::default(),
        feature_set,
        OptimizerKind::Ranger,
        // smoke は per-group override 無し (全 group weight_decay=0 / lr_mult=1.0)。
        OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
        None,
        None,
        &LayerStackInit::default_uniform(),
    )?;
    // smoke は既定次元で走る。L2 入力次元は L1 出力から導出 (skip 1 dim を除いた ×2)。
    let l2_in = (DEFAULT_L1_OUT - L1_SKIP) * 2;
    let n_buckets = DEFAULT_NUM_BUCKETS;
    println!(
        "[smoke] GpuTrainer ready: 10 weight groups, ~{:.1}M params total",
        (feature_set.ft_in() * DEFAULT_FT_OUT
            + DEFAULT_FT_OUT
            + n_buckets * DEFAULT_L1_OUT * DEFAULT_FT_OUT
            + n_buckets * DEFAULT_L1_OUT
            + DEFAULT_FT_OUT * DEFAULT_L1_OUT
            + DEFAULT_L1_OUT
            + n_buckets * DEFAULT_L2_OUT * l2_in
            + n_buckets * DEFAULT_L2_OUT
            + n_buckets * DEFAULT_L2_OUT
            + n_buckets) as f64
            / 1.0e6
    );

    trainer.assert_all_weights_finite()?;
    println!("[smoke] step 0: init weights all finite ✓");

    // `RSHOGI_NNUE_LAYERSTACK_REF_BIN` に既存の量子化 checkpoint (`.bin`) path を
    // 指定すると、その weight を注入して forward + backward + save を一通り走らせる。
    // 未設定なら random-init での smoke のみ。
    let layerstack_ref = std::env::var("RSHOGI_NNUE_LAYERSTACK_REF_BIN").ok();
    if let Some(ref_path) = layerstack_ref
        .as_deref()
        .filter(|p| std::path::Path::new(p).exists())
    {
        println!("[smoke] loading reference checkpoint from {ref_path} ...");
        let mut reader = std::io::BufReader::new(std::fs::File::open(ref_path)?);
        let weights = LayerStackWeights::load_quantised(
            &mut reader,
            feature_set,
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )?;
        trainer.load_layerstack_weights(&weights)?;
        trainer.assert_all_weights_finite()?;
        println!("[smoke] reference weights injected, all finite ✓");

        // forward + step 1 batch (sigmoid-MSE、golden forward/backward/save 経路)
        let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch.as_ref(), lr, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke] step 1 (post reference-inject, sigmoid-MSE): loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

        // save back as our quantised.bin
        let out_path = std::env::temp_dir().join("our_quantised.bin");
        let out_path_str = out_path.display();
        println!("[smoke] saving trained weights to {out_path_str} ...");
        let saved_weights = trainer.to_layerstack_weights()?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        saved_weights
            .save_quantised(&mut writer, Some(nnue_format::layerstack_weights::FV_SCALE))?;
        drop(writer);
        let out_size = std::fs::metadata(&out_path)?.len();
        println!("[smoke] wrote {out_path_str}: {out_size} bytes");

        // 追加 step: WRM loss kernel (`loss_wrm`) を runtime でも exercise する。
        // 上で save 済なので weights が変わっても verify 対象 (`out_path`) には影響しない。
        let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        let loss_wrm = trainer.step(&batch.as_ref(), 1e-3_f32, WDL_LAMBDA, SMOKE_LOSS_WRM)?;
        println!("[smoke] step 2 (win-rate-model): loss = {loss_wrm:.6e}");
        if !loss_wrm.is_finite() {
            return Err(format!("step 2 (wrm) loss = {loss_wrm} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 2: all weights finite ✓");
    } else {
        println!(
            "[smoke] (RSHOGI_NNUE_LAYERSTACK_REF_BIN not set or path missing; running random-init smoke only)"
        );
        let batch = BatchData::smoke_dummy(SMOKE_BATCH, feature_set);
        let lr = 1e-3_f32;
        let loss = trainer.step(&batch.as_ref(), lr, WDL_LAMBDA, SMOKE_LOSS_SIGMOID)?;
        println!("[smoke] step 1 (sigmoid-MSE): loss = {loss:.6e}");
        if !loss.is_finite() {
            return Err(format!("step 1 loss = {loss} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 1: all weights finite ✓");

        // step 2: WRM loss kernel (`loss_wrm`) を runtime でも exercise する。
        let loss_wrm = trainer.step(&batch.as_ref(), lr, WDL_LAMBDA, SMOKE_LOSS_WRM)?;
        println!("[smoke] step 2 (win-rate-model): loss = {loss_wrm:.6e}");
        if !loss_wrm.is_finite() {
            return Err(format!("step 2 (wrm) loss = {loss_wrm} is not finite").into());
        }
        trainer.assert_all_weights_finite()?;
        println!("[smoke] step 2: all weights finite ✓");

        // save random-init as quantised.bin for verify-nnue check
        let out_path = std::env::temp_dir().join("our_quantised_randinit.bin");
        let out_path_str = out_path.display();
        let saved_weights = trainer.to_layerstack_weights()?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        saved_weights
            .save_quantised(&mut writer, Some(nnue_format::layerstack_weights::FV_SCALE))?;
        drop(writer);
        let out_size = std::fs::metadata(&out_path)?.len();
        println!("[smoke] wrote {out_path_str}: {out_size} bytes");
    }

    println!(
        "[smoke] PASSED — GpuTrainer forward / backward / save OK (LayerStack arch full path)"
    );
    Ok(())
}
