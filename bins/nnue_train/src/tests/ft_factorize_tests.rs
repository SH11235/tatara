//! FT factorizer の trainer 結合テスト (大半は GPU 必要)。
//!
//! 仮想行 zero-init の不変条件 (step-1 の forward が OFF 構成と一致する) と、
//! export coalesce (`to_layerstack_weights` が base 形状 + base spec を返す) を
//! 実 GPU trainer で検証する。dataloader `Batch` → `BatchData` の stride 整合
//! のみ CPU で検証する。

use gpu_kernels::sparse::ft_factorize::{
    FT_FACTORIZE_BASE, FT_FACTORIZE_PER_EFFECT_BUCKET, FT_FACTORIZE_POOL_EFFECT_BUCKETS,
    FtFactorizeLayout,
};
use gpu_runtime::CudaContext;
use nnue_train::dataloader::Batch;
use nnue_train::init::LayerStackInit;
use nnue_train::trainer::LossKind;
use shogi_features::{EffectBucketConfig, FeatureSet, FtFactorizeMode};

use crate::arch::*;
use crate::trainer_common::{BatchData, PrecisionFlags};
use crate::trainer_layerstack::{GpuTrainer, OptimGroupConfig};

#[test]
fn ft_factorize_batch_to_batchdata_uses_base_stride() {
    // 本番 dataloader 経路 (`Batch` → `BatchData::from_batch_ref`) の stride が
    // factorized spec でも base `max_active` であることを検証する (sparse index
    // 列は factorizer 非依存、GPU 不要)。GPU テスト群は
    // `BatchData::smoke_dummy` で `Batch` 型を bypass するため、この変換だけは
    // 単体で押さえる。
    let fact = FeatureSet::HalfKaHmMerged.spec().with_ft_factorize();
    let mut batch = Batch::with_capacity(4, fact);
    batch.n_positions = 2;
    let bucket_idx = [0_i32, 0];
    let data = BatchData::from_batch_ref(&batch, &bucket_idx);
    assert_eq!(data.n_pos, 2);
    assert_eq!(data.stm_indices.len(), 2 * fact.max_active());
    assert_eq!(data.nstm_indices.len(), 2 * fact.max_active());
}

const B: usize = 64;
// 重み buffer (w/m/v/slow/grad × ft_in) が VRAM を支配するため、テストは
// FT 出力次元を縮小して並列実行時の他 GPU テストとの競合を避ける。
const FT_OUT_TEST: usize = 256;
const LOSS: LossKind = LossKind::Sigmoid { scale: 290.0 };

fn make_trainer(
    ctx: &std::sync::Arc<CudaContext>,
    feature_set: shogi_features::FeatureSetSpec,
) -> Result<GpuTrainer, Box<dyn std::error::Error>> {
    GpuTrainer::new(
        ctx,
        B,
        FT_OUT_TEST,
        DEFAULT_L1_OUT,
        DEFAULT_L2_OUT,
        DEFAULT_NUM_BUCKETS,
        PrecisionFlags::default(),
        feature_set,
        OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
        None,
        None,
        &LayerStackInit::default_uniform(),
    )
}

#[test]
fn ft_factorize_init_export_is_bit_identical_to_off() -> Result<(), Box<dyn std::error::Error>> {
    // 仮想行 zero-init + 実 block の base 形状 sample により、学習前の export は
    // OFF 構成と bit-identical (zero の畳み込みは +0.0)。spec も base に落ちる。
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    // VRAM 節約のため trainer は逐次生成し同時保持しない。
    let w_off = {
        let t_off = make_trainer(&ctx, base)?;
        t_off.to_layerstack_weights()?
    };
    let w_on = {
        let t_on = make_trainer(&ctx, base.with_ft_factorize())?;
        t_on.to_layerstack_weights()?
    };
    assert_eq!(
        w_on.feature_set, base,
        "coalesce 後の spec は base に落ちる"
    );
    assert_eq!(w_on.ft_w.len(), base.ft_in() * FT_OUT_TEST);
    assert_eq!(
        w_on.ft_w, w_off.ft_w,
        "学習前の coalesced ft_w は OFF と一致"
    );
    assert_eq!(w_on.ft_b, w_off.ft_b);
    Ok(())
}

#[test]
fn ft_fold_virtual_cpu_matches_export_coalesce() {
    // 学習時 forward が読む comb (fold) と export の畳み込み (coalesce) は同じ
    // 「実行 + 同 p の仮想行」の和。数式は GPU kernel ×2 / CPU reference /
    // export の 4 実装に分散しているため、CPU reference と export 実装を直接
    // 照合して片側だけの仕様変更 (学習時 forward と export 重みが乖離する
    // silent drift) を能動検出する。GPU kernel 側は gpu_cpu_equivalence_tests
    // が CPU reference へ照合済みなので、この 1 本で 4 実装が鎖で繋がる。
    // どちらも 1 要素 1 加算で演算列が同一なので比較は bit 一致 (GPU 不要)。
    let spec = FeatureSet::HalfKaHmMerged.spec().with_ft_factorize();
    let ft_out = 4;
    let train_n = spec.train_ft_in() * ft_out;
    let w: Vec<f32> = (0..train_n)
        .map(|i| ((i * 31 % 197) as f32 - 98.0) * 0.011)
        .collect();

    let mut comb = vec![0.0_f32; spec.ft_in() * ft_out];
    gpu_kernels::sparse::ft_factorize::ft_fold_virtual_cpu(
        &w,
        &mut comb,
        FtFactorizeLayout {
            base_ft_in: spec.base_ft_in(),
            ft_in: spec.ft_in(),
            ft_out,
            piece_inputs: spec.piece_inputs(),
            nb: 1,
            mode: FT_FACTORIZE_BASE,
        },
    );
    let coalesced = nnue_format::layerstack_weights::coalesce_ft_factorized(&spec, ft_out, &w);
    assert_eq!(comb, coalesced);
}

#[test]
fn effect_bucket_ft_fold_virtual_cpu_matches_export_coalesce() {
    for (mode, kernel_mode) in [
        (
            FtFactorizeMode::PoolEffectBuckets,
            FT_FACTORIZE_POOL_EFFECT_BUCKETS,
        ),
        (
            FtFactorizeMode::PerEffectBucket,
            FT_FACTORIZE_PER_EFFECT_BUCKET,
        ),
    ] {
        let spec = FeatureSet::HalfKaHmMerged
            .spec()
            .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2)
            .with_ft_factorize_mode(mode);
        let ft_out = 4;
        let train_n = spec.train_ft_in() * ft_out;
        let w: Vec<f32> = (0..train_n)
            .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.007)
            .collect();

        let mut comb = vec![0.0_f32; spec.ft_in() * ft_out];
        gpu_kernels::sparse::ft_factorize::ft_fold_virtual_cpu(
            &w,
            &mut comb,
            FtFactorizeLayout {
                base_ft_in: spec.ft_in(),
                ft_in: spec.ft_in(),
                ft_out,
                piece_inputs: spec.piece_inputs(),
                nb: spec.effect_bucket_config().unwrap().nb,
                mode: kernel_mode,
            },
        );
        let coalesced = nnue_format::layerstack_weights::coalesce_ft_factorized(&spec, ft_out, &w);
        assert_eq!(comb, coalesced, "{mode:?}");
    }
}

#[test]
fn threat_ft_fold_virtual_cpu_keeps_threat_rows_and_matches_export_coalesce() {
    use shogi_features::ThreatProfile;
    let spec = FeatureSet::HalfKaHmMerged
        .spec()
        .with_threat_profile(ThreatProfile::CrossSide)
        .with_ft_factorize();
    let ft_out = 4;
    let train_n = spec.train_ft_in() * ft_out;
    let w: Vec<f32> = (0..train_n)
        .map(|i| ((i * 41 % 223) as f32 - 111.0) * 0.005)
        .collect();

    let mut comb = vec![0.0_f32; spec.ft_in() * ft_out];
    gpu_kernels::sparse::ft_factorize::ft_fold_virtual_cpu(
        &w,
        &mut comb,
        FtFactorizeLayout {
            base_ft_in: spec.base_ft_in(),
            ft_in: spec.ft_in(),
            ft_out,
            piece_inputs: spec.piece_inputs(),
            nb: 1,
            mode: FT_FACTORIZE_BASE,
        },
    );
    let coalesced = nnue_format::layerstack_weights::coalesce_ft_factorized(&spec, ft_out, &w);
    assert_eq!(comb, coalesced);
}

#[test]
fn ft_factorize_first_step_matches_off_and_virtual_rows_learn()
-> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    let fact = base.with_ft_factorize();
    // index 列は factorizer 非依存なので ON / OFF で同一 batch を共有する —
    // 「両者が同じ実特徴を見る」前提を fixture の同一性で構造的に保証する。
    // ON の仮想行は trainer の fold / reduce kernel だけが配線する。VRAM
    // 節約のため trainer は逐次生成し同時保持しない。
    let batch = BatchData::smoke_dummy(B, base);
    let (loss_off, w_off) = {
        let mut t_off = make_trainer(&ctx, base)?;
        let loss = t_off.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        (loss, t_off.to_layerstack_weights()?)
    };
    let (loss_on, w_on) = {
        let mut t_on = make_trainer(&ctx, fact)?;
        let loss = t_on.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        (loss, t_on.to_layerstack_weights()?)
    };

    // 仮想行は zero-init なので step-1 の forward / loss は一致する
    // (loss_acc の atomic 加算順による揺らぎのみ許容)。
    let tol = loss_off.abs() * 1e-6 + 1e-12;
    assert!(
        (loss_on - loss_off).abs() <= tol,
        "step-1 loss must match: on={loss_on:e} off={loss_off:e}"
    );

    // optimizer 1 step 後: 仮想行は勾配を受けて動くため、coalesce 済み export は
    // OFF と一致しなくなる (実 block の更新は同一勾配なので、差分 = 仮想行の学習)。
    assert_eq!(w_on.ft_w.len(), w_off.ft_w.len());
    assert!(
        w_on.ft_w != w_off.ft_w,
        "仮想行が学習されていれば coalesced ft_w は OFF と異なる"
    );
    // 「差分 = 仮想行の寄与」の裏付け: coalesce は加算で、実 block の更新勾配は
    // 仮想行の有無に依存しないため、ON−OFF の残差は同一 p を持つ feature 間で
    // 一致する (仮想行 1 本分)。先頭の数 p で確認する。
    let pi = base.piece_inputs();
    for p in 0..8 {
        let d0 = w_on.ft_w[p * FT_OUT_TEST] - w_off.ft_w[p * FT_OUT_TEST];
        let d1 = w_on.ft_w[(pi + p) * FT_OUT_TEST] - w_off.ft_w[(pi + p) * FT_OUT_TEST];
        assert!(
            (d0 - d1).abs() <= d0.abs().max(d1.abs()) * 1e-5 + 1e-7,
            "残差が仮想行由来なら plane 間で一致する: p={p} d0={d0:e} d1={d1:e}"
        );
    }
    Ok(())
}

#[test]
fn effect_bucket_ft_factorize_first_step_matches_off_and_virtual_rows_learn()
-> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged
        .spec()
        .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2);
    let fact = base.with_ft_factorize();
    let batch = BatchData::smoke_dummy(B, base);
    let (loss_off, w_off) = {
        let mut t_off = make_trainer(&ctx, base)?;
        let loss = t_off.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        (loss, t_off.to_layerstack_weights()?)
    };
    let (loss_on, w_on) = {
        let mut t_on = make_trainer(&ctx, fact)?;
        let loss = t_on.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        (loss, t_on.to_layerstack_weights()?)
    };

    let tol = loss_off.abs() * 1e-6 + 1e-12;
    assert!(
        (loss_on - loss_off).abs() <= tol,
        "effect bucket step-1 loss must match: on={loss_on:e} off={loss_off:e}"
    );
    assert_eq!(w_on.feature_set, base);
    assert_eq!(w_on.ft_w.len(), w_off.ft_w.len());
    assert!(w_on.ft_w != w_off.ft_w);

    let pi = base.piece_inputs();
    let nb = base.effect_bucket_config().unwrap().nb;
    for p in 0..8 {
        let feat0 = p * nb;
        let feat1 = (pi + p) * nb;
        let d0 = w_on.ft_w[feat0 * FT_OUT_TEST] - w_off.ft_w[feat0 * FT_OUT_TEST];
        let d1 = w_on.ft_w[feat1 * FT_OUT_TEST] - w_off.ft_w[feat1 * FT_OUT_TEST];
        assert!(
            (d0 - d1).abs() <= d0.abs().max(d1.abs()) * 1e-5 + 1e-7,
            "effect bucket 残差が同じ piece-input ordinal で一致する: p={p} d0={d0:e} d1={d1:e}"
        );
    }
    Ok(())
}

#[test]
fn ft_factorize_quantised_export_loads_as_base_net() -> Result<(), Box<dyn std::error::Error>> {
    // ON trainer の量子化 export が base feature set の .bin としてそのまま
    // load できる (= 推論エンジン側変更ゼロの根拠)。
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    let mut t_on = make_trainer(&ctx, base.with_ft_factorize())?;
    let b_on = BatchData::smoke_dummy(B, base.with_ft_factorize());
    let _ = t_on.step(&b_on.as_ref(), 1e-3, 0.5, LOSS)?;

    let w = t_on.to_layerstack_weights()?;
    let mut buf = Vec::new();
    w.save_quantised(&mut buf)?;
    let loaded = nnue_format::LayerStackWeights::load_quantised(
        &mut std::io::Cursor::new(&buf),
        base,
        FT_OUT_TEST,
        DEFAULT_L1_OUT,
        DEFAULT_L2_OUT,
        DEFAULT_NUM_BUCKETS,
    )?;
    assert_eq!(loaded.feature_set, base);
    assert_eq!(loaded.ft_w.len(), base.ft_in() * FT_OUT_TEST);
    Ok(())
}

#[test]
fn ft_factorize_threat_coexist_export_keeps_threat_rows() -> Result<(), Box<dyn std::error::Error>>
{
    use shogi_features::ThreatProfile;
    // factorizer × threat 同居: export は factorizer を畳んでも threat profile を
    // 保持し、ft_w は base + threat 行 (= ft_in()) を残す。最小 profile を使う。
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    let threat = base.with_threat_profile(ThreatProfile::CrossSide);
    let coexist = threat.with_ft_factorize();

    let w = {
        let t = make_trainer(&ctx, coexist)?;
        t.to_layerstack_weights()?
    };
    // export spec は factorizer を外しつつ threat を保持 (= threat-only spec)。
    assert_eq!(
        w.feature_set, threat,
        "export spec は factorizer を外し threat を保持"
    );
    assert!(!w.feature_set.ft_factorize());
    assert_eq!(
        w.feature_set.threat_profile(),
        Some(ThreatProfile::CrossSide)
    );
    // ft_w は base + threat 行 (threat を silent 切り落とししない)。
    assert_eq!(w.ft_w.len(), threat.ft_in() * FT_OUT_TEST);
    assert!(
        threat.ft_in() > base.ft_in(),
        "threat 連結で ft_in が伸びる"
    );
    Ok(())
}

#[test]
fn ft_factorize_threat_coexist_first_step_matches_threat_only()
-> Result<(), Box<dyn std::error::Error>> {
    use shogi_features::ThreatProfile;
    // 仮想行 zero-init なので、threat-active な step-1 の forward/loss は
    // threat-only (factorizer OFF) と一致する。同居の関数空間不変 (fold 後 weight の
    // forward == train 経路 forward) を実 GPU で押さえる。index 列は factorizer 非
    // 依存なので同一 batch を共有し「両者が同じ実特徴 (threat 含む) を見る」を保証。
    let ctx = CudaContext::new(0)?;
    let threat = FeatureSet::HalfKaHmMerged
        .spec()
        .with_threat_profile(ThreatProfile::CrossSide);
    let coexist = threat.with_ft_factorize();
    // batch は threat-on spec で生成 (index は threat 範囲も踏む)。
    let batch = BatchData::smoke_dummy(B, threat);

    let (loss_only, w_only) = {
        let mut t = make_trainer(&ctx, threat)?;
        let loss = t.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        (loss, t.to_layerstack_weights()?)
    };
    let (loss_coexist, w_coexist) = {
        let mut t = make_trainer(&ctx, coexist)?;
        let loss = t.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        (loss, t.to_layerstack_weights()?)
    };

    // zero-init 仮想行: step-1 forward/loss は一致 (atomic 加算順揺らぎのみ)。
    let tol = loss_only.abs() * 1e-6 + 1e-12;
    assert!(
        (loss_coexist - loss_only).abs() <= tol,
        "threat 同居 step-1 loss は threat-only と一致: coexist={loss_coexist:e} only={loss_only:e}"
    );
    // export 形状は両者一致 (base + threat)。1 step 後の差分 = 仮想行の学習。
    assert_eq!(w_coexist.ft_w.len(), w_only.ft_w.len());
    assert_eq!(w_coexist.ft_w.len(), threat.ft_in() * FT_OUT_TEST);
    assert!(
        w_coexist.ft_w != w_only.ft_w,
        "仮想行が学習されていれば coalesced ft_w は threat-only と異なる"
    );
    Ok(())
}

/// PSQT 有効の trainer を作る (psqt_init は base 形状)。factorize 有効 spec を
/// 渡すと PSQT block もpiece-input 仮想行を持ち、`psqt_init` の base 行を実 block に置いて
/// 仮想 block は zero append される。
fn make_trainer_psqt_init(
    ctx: &std::sync::Arc<CudaContext>,
    feature_set: shogi_features::FeatureSetSpec,
    psqt_init: &[f32],
) -> Result<GpuTrainer, Box<dyn std::error::Error>> {
    GpuTrainer::new(
        ctx,
        B,
        FT_OUT_TEST,
        DEFAULT_L1_OUT,
        DEFAULT_L2_OUT,
        DEFAULT_NUM_BUCKETS,
        PrecisionFlags::default(),
        feature_set,
        OptimGroupConfig::resolve(0.0, None, None, None, None, None, None),
        None,
        Some(psqt_init),
        &LayerStackInit::default_uniform(),
    )
}

/// `make_trainer_psqt_init` の zeros init 版。
fn make_trainer_psqt(
    ctx: &std::sync::Arc<CudaContext>,
    feature_set: shogi_features::FeatureSetSpec,
) -> Result<GpuTrainer, Box<dyn std::error::Error>> {
    let psqt_init = vec![0.0_f32; feature_set.ft_in() * DEFAULT_NUM_BUCKETS];
    make_trainer_psqt_init(ctx, feature_set, &psqt_init)
}

#[test]
fn ft_factorize_psqt_nonzero_init_virtual_block_is_zero() -> Result<(), Box<dyn std::error::Error>>
{
    // material 等の非 zero base init でも、factorize trainer の仮想 block は zero
    // append される。よって学習前の coalesced psqt_w (= 実 block + zero 仮想) は
    // base init そのもの・非 factorize psqt と一致する (step-0 forward 等価の根拠)。
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    let pn = base.ft_in() * DEFAULT_NUM_BUCKETS;
    // 決定論的な非 zero base init (material prior の代わり)。
    let init: Vec<f32> = (0..pn).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let w_off = make_trainer_psqt_init(&ctx, base, &init)?.to_layerstack_weights()?;
    let w_on =
        make_trainer_psqt_init(&ctx, base.with_ft_factorize(), &init)?.to_layerstack_weights()?;
    let pw_on = w_on.psqt_w.expect("psqt on");
    assert_eq!(pw_on.len(), pn, "coalesced psqt_w は base 形状");
    assert_eq!(pw_on, init, "仮想 block zero なので coalesced = base init");
    assert_eq!(
        pw_on,
        w_off.psqt_w.expect("psqt off"),
        "factorize 有無で init export 一致"
    );
    Ok(())
}

#[test]
fn ft_factorize_psqt_init_export_is_base_shape() -> Result<(), Box<dyn std::error::Error>> {
    // factorize + psqt の trainer は psqt_w を train 形状で持つが、export coalesce で
    // base 形状に畳む。psqt_init=zeros + 仮想行 zero なので、学習前の coalesced psqt_w
    // は非 factorize psqt と一致する (zero の畳み込みは +0.0)。
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    let w_off = make_trainer_psqt(&ctx, base)?.to_layerstack_weights()?;
    let w_on = make_trainer_psqt(&ctx, base.with_ft_factorize())?.to_layerstack_weights()?;
    let pn = base.ft_in() * DEFAULT_NUM_BUCKETS;
    assert_eq!(
        w_on.psqt_w.as_ref().map(Vec::len),
        Some(pn),
        "coalesced psqt_w は base 形状"
    );
    assert_eq!(
        w_on.psqt_w, w_off.psqt_w,
        "学習前の coalesced psqt_w は OFF と一致"
    );
    Ok(())
}

#[test]
fn ft_factorize_psqt_virtual_rows_learn() -> Result<(), Box<dyn std::error::Error>> {
    // 1 step 後: psqt 仮想行が grad を受けて動くため、coalesce 済み psqt_w は非
    // factorize psqt と一致しなくなる。残差は同一 p を持つ feature 間で一致する
    // (仮想行 1 本分の畳み込み、FT の同型テストと同じ裏取り)。psqt forward は
    // 初期 zero なので実 block の grad は両構成で同一 → 差分 = 仮想行の寄与。
    let ctx = CudaContext::new(0)?;
    let base = FeatureSet::HalfKaHmMerged.spec();
    let fact = base.with_ft_factorize();
    let batch = BatchData::smoke_dummy(B, base);
    let w_off = {
        let mut t = make_trainer_psqt(&ctx, base)?;
        t.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        t.to_layerstack_weights()?
    };
    let w_on = {
        let mut t = make_trainer_psqt(&ctx, fact)?;
        t.step(&batch.as_ref(), 1e-3, 0.5, LOSS)?;
        t.to_layerstack_weights()?
    };
    let pw_off = w_off.psqt_w.expect("psqt off");
    let pw_on = w_on.psqt_w.expect("psqt on");
    assert_eq!(pw_on.len(), pw_off.len());
    assert!(
        pw_on != pw_off,
        "psqt 仮想行が学習されていれば coalesced psqt_w は OFF と異なる"
    );
    let nb = DEFAULT_NUM_BUCKETS;
    let pi = base.piece_inputs();
    for p in 0..8 {
        let d0 = pw_on[p * nb] - pw_off[p * nb];
        let d1 = pw_on[(pi + p) * nb] - pw_off[(pi + p) * nb];
        assert!(
            (d0 - d1).abs() <= d0.abs().max(d1.abs()) * 1e-5 + 1e-7,
            "psqt 残差が仮想行由来なら plane 間で一致する: p={p} d0={d0:e} d1={d1:e}"
        );
    }
    Ok(())
}
