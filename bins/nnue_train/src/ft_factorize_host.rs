//! FT factorizer の fold / reduce kernel launch を LayerStack / Simple 両トレーナで
//! 共有する host helper。仮想行 (piece-input / threat-pair) と実行の対応、kernel 引数
//! の packing、mode ごとの割り切れ不変条件を単一実装に集約し、fold (forward comb の
//! 再生成) と reduce (backward の仮想行 gradient 充填) を配線する。
//!
//! 仮想行の意味論と kernel side の対応付けは [`gpu_kernels::sparse::ft_factorize`]
//! (CPU reference と同アルゴリズムの doc) を参照。PSQT shortcut の fold は LayerStack
//! 固有 (列 = num_buckets、threat 無し) なので本 module ではなく trainer 側に残す。

use gpu_kernels::sparse::ft_factorize::{
    FT_FACTORIZE_BASE, FT_FACTORIZE_PER_EFFECT_BUCKET, FT_FACTORIZE_POOL_EFFECT_BUCKETS,
};
use std::sync::Arc;

use gpu_runtime::{CudaModule, CudaStream, DeviceBuffer, cuda_launch};
use shogi_features::{FeatureSetSpec, FtFactorizeMode};

#[cfg(feature = "cuda-oxide")]
use crate::kernels::*;
use crate::trainer_common::cfg_1d;

/// feature set の共有 mode を fold/reduce kernel の mode 定数へ写す。
pub(crate) fn kernel_mode(feature_set: &FeatureSetSpec) -> u32 {
    match feature_set.ft_factorize_mode() {
        FtFactorizeMode::Base => FT_FACTORIZE_BASE,
        FtFactorizeMode::PoolEffectBuckets => FT_FACTORIZE_POOL_EFFECT_BUCKETS,
        FtFactorizeMode::PerEffectBucket => FT_FACTORIZE_PER_EFFECT_BUCKET,
    }
}

/// effect bucket 数 `nb` と mode を kernel の 1 引数へ pack する (下位 16bit=nb、上位=mode)。
pub(crate) fn kernel_pack(nb: usize, mode: u32) -> u32 {
    (nb as u32) | (mode << 16)
}

/// fold/reduce kernel の `ft_bounds` 引数 (下位 32bit=base_ft_in、上位=ft_in) を作る。
pub(crate) fn ft_bounds(base_ft_in: usize, ft_in: usize) -> u64 {
    (base_ft_in as u64) | ((ft_in as u64) << 32)
}

/// forward comb の格納先。`--ft-fp16` 系列は f16 comb、既定 FP32 経路は f32 comb。
pub(crate) enum FoldComb<'a> {
    F16(&'a mut DeviceBuffer<f16>),
    F32(&'a mut DeviceBuffer<f32>),
}

/// fold/reduce に共通する派生量 `(base_ft_in, ft_in, pi, nb, mode)`。
///
/// `base_ft_in` は piece-input 仮想行を持つ実行の行数 (effect bucket では base+bucket
/// 展開後の全行、それ以外は threat を除いた base 行)。`ft_in` は base+threat (= 仮想
/// 行の手前 / comb サイズ)。piece-input ordinal で割り切れない feature set は仮想行
/// 対応が崩れるため release でも弾く (全 feature set で成立する不変条件)。
fn params(feature_set: &FeatureSetSpec) -> (usize, usize, usize, usize, u32) {
    let ft_in = feature_set.ft_in();
    let base_ft_in = if feature_set.effect_bucket_config().is_some() {
        ft_in
    } else {
        feature_set.base_ft_in()
    };
    let pi = feature_set.piece_inputs();
    let nb = feature_set.effect_bucket_config().map_or(1, |cfg| cfg.nb);
    let mode = kernel_mode(feature_set);
    assert_eq!(
        base_ft_in % pi,
        0,
        "base_ft_in must be a multiple of piece_inputs for the factorizer"
    );
    if mode == FT_FACTORIZE_POOL_EFFECT_BUCKETS || mode == FT_FACTORIZE_PER_EFFECT_BUCKET {
        assert_eq!(
            base_ft_in % (pi * nb),
            0,
            "base_ft_in must be a multiple of piece_inputs * effect_buckets for EffectBucket factorizer modes"
        );
    }
    (base_ft_in, ft_in, pi, nb, mode)
}

/// FT weight の forward comb を `master` (train 形状 FP32) から再生成する。base 実行
/// には対応 piece-input 仮想行を、threat 実行には pair 仮想行を畳み込み base 形状
/// (`ft_in * ft_out`) の comb を上書きする。caller が factorizer 有効を保証する。
pub(crate) fn launch_ft_fold(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    feature_set: &FeatureSetSpec,
    ft_out: usize,
    master: &DeviceBuffer<f32>,
    comb: FoldComb<'_>,
    threat_pair_starts: &DeviceBuffer<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert!(feature_set.ft_factorize());
    let (stream, module) = (stream.clone(), module.clone());
    let (base_ft_in, ft_in, pi, nb, mode) = params(feature_set);
    let fold_mode = kernel_pack(nb, mode);
    let bounds = ft_bounds(base_ft_in, ft_in);
    let n = ft_in * ft_out;
    match comb {
        FoldComb::F16(mut comb) => unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: ft_fold_virtual_f16,
                stream: stream,
                module: module,
                config: cfg_1d(n),
                args: [slice(master), slice_mut(comb), slice(threat_pair_starts),
                       bounds, ft_out as u32, pi as u32, fold_mode]
            }
        }?,
        FoldComb::F32(mut comb) => unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: ft_fold_virtual,
                stream: stream,
                module: module,
                config: cfg_1d(n),
                args: [slice(master), slice_mut(comb), slice(threat_pair_starts),
                       bounds, ft_out as u32, pi as u32, fold_mode]
            }
        }?,
    }
    Ok(())
}

/// FT weight gradient (train 形状) の仮想 block を実行の勾配縮約で埋める。piece-input
/// 仮想行は対応 base 実行の grad 和、threat-pair 仮想行は同 pair の threat 実行の grad
/// 和。実行 block (`[0, ft_in)`) は読みのみ。caller が factorizer 有効を保証する。
pub(crate) fn launch_ft_reduce(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    feature_set: &FeatureSetSpec,
    ft_out: usize,
    grad: &DeviceBuffer<f32>,
    threat_pair_starts: &DeviceBuffer<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert!(feature_set.ft_factorize());
    let (stream, module) = (stream.clone(), module.clone());
    let (base_ft_in, ft_in, pi, nb, mode) = params(feature_set);
    let fold_mode = kernel_pack(nb, mode);
    let bounds = ft_bounds(base_ft_in, ft_in);
    let virtual_rows = feature_set.ft_factorize_virtual_rows();
    unsafe {
        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
        // stream の完了を待つ同期点まで生存する device allocation。
        cuda_launch! {
            kernel: ft_reduce_virtual_grad,
            stream: stream,
            module: module,
            config: cfg_1d(virtual_rows * ft_out),
            args: [slice(grad), slice(threat_pair_starts),
                   bounds, ft_out as u32, pi as u32, fold_mode]
        }
    }?;
    Ok(())
}
