//! FT factorizer の trainer 結合テスト (大半は GPU 必要)。
//!
//! 仮想行 zero-init の不変条件 (step-1 の forward が OFF 構成と一致する) と、
//! export coalesce (`to_layerstack_weights` が base 形状 + base spec を返す) を
//! 実 GPU trainer で検証する。dataloader `Batch` → `BatchData` の stride 整合
//! のみ CPU で検証する。

use gpu_runtime::CudaContext;
use nnue_train::dataloader::Batch;
use nnue_train::init::LayerStackInit;
use nnue_train::trainer::LossKind;
use shogi_features::FeatureSet;

use crate::arch::*;
use crate::trainer_common::BatchData;
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
        false,
        false,
        false,
        false,
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
        spec.ft_in(),
        ft_out,
        spec.piece_inputs(),
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
