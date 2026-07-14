use std::path::Path;

use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, cuda_launch};
use nnue_format::{ArchKind, SimpleActivation, SimpleId, SimpleWeights};
use nnue_train::init::{self, SimpleInit, WeightShape};
use nnue_train::optimizer::radam_compute_step_size_denom;
use nnue_train::trainer::LossKind;

use crate::ft_factorize_host::{self, FoldComb};
use crate::*;
use crate::{arch::*, ckpt::*, kernel_module::*, trainer_common::*};

// ===========================================================================
// SimpleGpuTrainer — bucket 無し 4 層 Simple アーキの GPU トレーナ
// ===========================================================================
//
// LayerStack 用 `GpuTrainer` (本 file 上方) と並ぶ、もう一方のアーキの host driver。
// `SimpleId` (feature set / 活性化 / ft_out / l1_out / l2_out) で形が決まる 8
// weight 群 (FT/L1/L2/L3 各 w・b) を device buffer で保持し、`forward` で 1 batch
// の FT → bias add → 活性化 → concat → L1/L2/L3 dense → loss を既存 kernel の
// 合成として走らせる。
//
// 本段階は **forward + smoke 専用** で、backward / optimizer / checkpoint /
// `TrainerBackend` 実装は持たない。weight は xorshift 初期化のみ。

/// Simple アーキ専用の forward 用 workspace。中間 activation buffer (全 f32) と
/// 入力 buffer (sparse idx + score/wdl) を固定 batch 容量で `new` 時に確保する。
pub(crate) struct SimpleGpuWorkspace {
    /// `new` 時の固定 batch 容量 (`forward` 実行時にこれ以下を要求)。
    len_batch: usize,
    /// FT 入力次元 (`id.feature_set.ft_in()`)、kernel launch 引数で使う。
    ft_in: usize,
    /// 1 perspective あたりの active feature 上限 (`id.feature_set.max_active()`)。
    max_active: usize,

    // -- forward 中間 activation (すべて f32) --
    /// sparse_ft_forward の stm 出力 (`b × ft_out`)。bias add は in-place でここに書き戻す。
    ft_stm_out: DeviceBuffer<f32>,
    /// 同 nstm。
    ft_nstm_out: DeviceBuffer<f32>,
    /// stm/nstm の FT post 出力を concat した L1 dense 入力 (`b × combined_dim`、
    /// CReLU/SCReLU は `2*ft_out`・Pairwise は `ft_out`)。
    combined: DeviceBuffer<f32>,
    /// L1 dense 出力 (pre-activation、`b × l1_out`)。
    l1_pre: DeviceBuffer<f32>,
    /// L1 活性化後 (`b × l1_out`)、L2 dense の入力。
    l1_acted: DeviceBuffer<f32>,
    /// L2 dense 出力 (`b × l2_out`)。
    l2_pre: DeviceBuffer<f32>,
    /// L2 活性化後 (`b × l2_out`)、L3 dense の入力。
    l2_acted: DeviceBuffer<f32>,
    /// L3 dense 出力 = ネットワーク 1 次元出力 (`b`)。
    net_output: DeviceBuffer<f32>,
    /// loss kernel が書く dnet (= dy/d net_output、`b`)。backward の起点。
    dy_net_output: DeviceBuffer<f32>,

    // -- backward gradient buffer (forward と対称配置) --
    /// L2 活性化後への grad (`b × l2_out`)、`dense_mm_bwd_input` で L3 から伝播。
    dl2_acted: DeviceBuffer<f32>,
    /// L2 dense 出力への grad (`b × l2_out`)、活性化逆伝播の出力。
    dl2_pre: DeviceBuffer<f32>,
    /// L1 活性化後への grad (`b × l1_out`)。
    dl1_acted: DeviceBuffer<f32>,
    /// L1 dense 出力への grad (`b × l1_out`)。
    dl1_pre: DeviceBuffer<f32>,
    /// concat 後 (`b × combined_dim`) への grad。L1 dense backward の入力 grad 先。
    dcombined: DeviceBuffer<f32>,
    /// stm FT 出力 (= bias add 直後 / 活性化直前) への grad (`b × ft_out`)。
    /// `sparse_ft_backward` の入力 grad + `ft_b` bias grad の reduce 対象。
    dft_stm_out: DeviceBuffer<f32>,
    /// 同 nstm。
    dft_nstm_out: DeviceBuffer<f32>,

    // -- `--ft-fp16-out` 経路の f16 buffer (ft_fp16_out が true のときだけ Some) --
    /// `sparse_ft_forward_fp16_out` の出力 (`b × ft_out`、f16、bias 未加算)。
    /// 後続 [`simple_bias_act_fwd_fp16_in_crelu`] / [`simple_act_grad_to_fp16_crelu_with_scale`]
    /// が bias を別 buffer から read して加算する。
    ft_stm_out_h: Option<DeviceBuffer<f16>>,
    /// 同 nstm。
    ft_nstm_out_h: Option<DeviceBuffer<f16>>,
    /// FT pre-activation gradient (`b × ft_out`、f16、loss scaling 済)。
    /// [`simple_act_grad_to_fp16_crelu_with_scale`] が書き、[`simple_bias_grad_fp16`]
    /// と [`simple_sparse_ft_backward_fp16`] が `dft_inv_scale` で打ち消して accumulate。
    dft_stm_out_h: Option<DeviceBuffer<f16>>,
    /// 同 nstm。
    dft_nstm_out_h: Option<DeviceBuffer<f16>>,

    // -- inverse-index sparse_ft_backward scratch (`build_feature_counts` → exclusive
    //    prefix sum (multi-block scan) → `scatter_positions` → `gather_and_sum_per_feature_*`
    //    pipeline 用)。per-feature gather で `dft_*_out` の DRAM read を 1 perspective につき
    //    各 (feature, ft_out) cell ちょうど 1 回に抑え、global atomic 数も `b * ft_out *
    //    max_active` から `b * max_active` (histogram + scatter) まで圧縮する。サイズは
    //    feature set ごとに固定 (`ft_in` / `max_active` で決まる)。
    /// per-feature 出現回数 histogram (`ft_in`、`build_feature_counts` で atomic build)。
    feat_counts: DeviceBuffer<u32>,
    /// `feat_counts` の exclusive prefix sum (`ft_in + 1`、multi-block scan で構築)。
    feat_offsets: DeviceBuffer<u32>,
    /// multi-block scan level 1 の per-block 総和 (`ceil(ft_in/1024)`、`prefix_sum_block_local` が書く)。
    feat_block_sums: DeviceBuffer<u32>,
    /// `feat_block_sums` の exclusive prefix sum (`ceil(ft_in/1024)+1`、level 2 = `exclusive_prefix_sum_small`)。
    feat_block_offsets: DeviceBuffer<u32>,
    /// `scatter_positions` 中の per-feature 書き込み位置カウンタ (`ft_in`、atomic incremented)。
    feat_write_ctr: DeviceBuffer<u32>,
    /// 各 feature 出現位置の sorted ストレージ (`batch * max_active`、`scatter_positions` が書く)。
    feat_positions: DeviceBuffer<u32>,

    // -- 入力 buffer (active / back ペア。`InputUploadRing` の double-buffer 規約) --
    /// stm sparse index (`b × max_active`、無効 slot は -1)。active 側 = 現 step が forward
    /// で read する物理 buffer。`step` 冒頭で back と `mem::swap` してから ring が back
    /// (旧 active = 直前 step が読んでいない側) に async H2D する。
    stm_idx_dev: DeviceBuffer<i32>,
    /// 同 nstm。
    nstm_idx_dev: DeviceBuffer<i32>,
    /// position ごとの実 active feature 数。
    nnz_dev: DeviceBuffer<i32>,
    /// target score (`b`、centipawn)。
    score_dev: DeviceBuffer<f32>,
    /// target wdl (`b`、0.0/0.5/1.0)。
    wdl_dev: DeviceBuffer<f32>,
    /// 同上、back 側物理 buffer。次 step の H2D 先 (`step` 冒頭で `mem::swap` で active
    /// へ昇格)。直前 step の compute が読んでいる active と物理分離されるため、H2D は
    /// 直前 step の compute と並走しても buffer 競合が起きない。
    stm_idx_dev_back: DeviceBuffer<i32>,
    nstm_idx_dev_back: DeviceBuffer<i32>,
    nnz_dev_back: DeviceBuffer<i32>,
    score_dev_back: DeviceBuffer<f32>,
    wdl_dev_back: DeviceBuffer<f32>,
}

impl SimpleGpuWorkspace {
    pub(crate) fn new(
        stream: &CudaStream,
        batch: usize,
        id: SimpleId,
        ft_fp16_out: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ft_in = id.ft_in();
        let max_active = id.feature_set.max_active();
        let ft_out = id.ft_out;
        let l1_out = id.l1_out;
        let l2_out = id.l2_out;
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(stream, n).map_err(Into::into)
        };
        let alloc_h = |on: bool| -> Result<Option<DeviceBuffer<f16>>, Box<dyn std::error::Error>> {
            if on {
                Ok(Some(DeviceBuffer::<f16>::zeroed(stream, batch * ft_out)?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            len_batch: batch,
            ft_in,
            max_active,
            ft_stm_out: z(batch * ft_out)?,
            ft_nstm_out: z(batch * ft_out)?,
            combined: z(batch * id.combined_dim())?,
            l1_pre: z(batch * l1_out)?,
            l1_acted: z(batch * l1_out)?,
            l2_pre: z(batch * l2_out)?,
            l2_acted: z(batch * l2_out)?,
            net_output: z(batch)?,
            dy_net_output: z(batch)?,
            dl2_acted: z(batch * l2_out)?,
            dl2_pre: z(batch * l2_out)?,
            dl1_acted: z(batch * l1_out)?,
            dl1_pre: z(batch * l1_out)?,
            dcombined: z(batch * id.combined_dim())?,
            dft_stm_out: z(batch * ft_out)?,
            dft_nstm_out: z(batch * ft_out)?,
            ft_stm_out_h: alloc_h(ft_fp16_out)?,
            ft_nstm_out_h: alloc_h(ft_fp16_out)?,
            dft_stm_out_h: alloc_h(ft_fp16_out)?,
            dft_nstm_out_h: alloc_h(ft_fp16_out)?,
            feat_counts: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_offsets: DeviceBuffer::<u32>::zeroed(stream, ft_in + 1)?,
            feat_block_sums: DeviceBuffer::<u32>::zeroed(stream, ft_in.div_ceil(1024))?,
            feat_block_offsets: DeviceBuffer::<u32>::zeroed(stream, ft_in.div_ceil(1024) + 1)?,
            feat_write_ctr: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_positions: DeviceBuffer::<u32>::zeroed(stream, batch * max_active)?,
            stm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nnz_dev: DeviceBuffer::<i32>::zeroed(stream, batch)?,
            score_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            stm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nnz_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch)?,
            score_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
        })
    }

    /// `new` 時の `len_batch` 容量に `batch` が収まることを検証する
    /// (`GpuWorkspace::check_batch_capacity` と同じ規約: 固定 batch 前提で
    /// 容量超過は error)。
    pub(crate) fn check_batch_capacity(
        &self,
        batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if batch > self.len_batch {
            return Err(format!(
                "SimpleGpuTrainer: batch {batch} exceeds workspace capacity {} \
                 (re-construct SimpleGpuTrainer with a larger batch_size)",
                self.len_batch
            )
            .into());
        }
        Ok(())
    }
}

/// Simple 4 層アーキ用 GPU トレーナ。LayerStack 用 `GpuTrainer` と並ぶもう一方の
/// アーキの host driver。1 batch の forward → loss → backward → Ranger optimizer step
/// を 8 weight group ({ft, l1, l2, l3} × {w, b}) について実行する。
///
/// `--ft-fp16` / `--ft-fp16-out` / `--fp16-opt-state` / `--tf32` の risky 精度系 flag は
/// LayerStack と同形で opt-in (default OFF で FP32 bit-identical、ON で risky 最適化)。
///
/// L1 dense (B × combined_dim × l1_out) は forward / bwd_input / bwd_weight 3 経路を
/// cuBLAS Sgemm に乗せる。default math mode は `CUBLAS_DEFAULT_MATH` (純 FP32、TC 不使用)、
/// `--tf32` 指定で `CUBLAS_TF32_TENSOR_OP_MATH` (Ampere+ TC、仮数 10-bit 丸め)。L2 / L3
/// は次元が小さく untiled `dense_mm_*` で残す。FT は専用 `sparse_ft_*` kernel、活性化と
/// loss は固有 kernel。
pub(crate) struct SimpleGpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,
    /// `dense_bias_grad_tiled` の grid 上限を実機 SM 数から導出するための occupancy
    /// パラメータ (`new` で 1 度問い合わせ、特定 GPU 固定値を避ける)。
    dense_bias_grad_occ: DeviceOccupancy,
    /// L1 dense (FP32) 用 cuBLAS handle (TF32 不使用、`CUBLAS_DEFAULT_MATH`)。
    /// stream に bind 済で同一 stream 内 in-order 実行。
    cublas: CublasHandle,
    /// `--ft-fp16` opt-in flag。`true` の間 forward は `sparse_ft_forward_fp16`
    /// (FP16 weight read) を使い、optimizer は `radam_step_fp16_mirror` /
    /// `ranger_lookahead_lerp_fp16_mirror` で `ft_w` 更新と同時に `ft_w_h` を書く。
    /// FP32 master `ft_w` / 量子化 checkpoint byte layout は不変。
    ft_fp16: bool,
    /// `ft_fp16` が `true` のときだけ `Some`。factorizer 無効時は `ft_w` (train 形状
    /// = base) の FP16 mirror で、`sparse_ft_forward_fp16` の weight 入力・
    /// `radam_step_fp16_mirror` が optimizer step ごとに同期する。factorizer 有効時は
    /// 仮想行を実行へ畳み込んだ base 形状の f16 comb で、毎 step 末の
    /// [`ft_factorize_host::launch_ft_fold`] が master (`ft_w`、train 形状) から再生成
    /// する (optimizer は master のみ更新)。初期同期は
    /// [`sync_ft_forward_weights`](Self::sync_ft_forward_weights)。
    ft_w_h: Option<DeviceBuffer<f16>>,
    /// factorizer 有効かつ `ft_fp16` 無効のときの forward 用畳み込み weight (comb、
    /// base 形状 `ft_in * ft_out` の f32)。`ft_w_h` の f16 comb と同役で、FP32 forward が
    /// `ft_w` (train 形状 master) の代わりにこれを読む。毎 step 末の fold が再生成する。
    ft_w_fold32: Option<DeviceBuffer<f32>>,
    /// fold/reduce kernel に渡す threat within-pair prefix table (Simple の base feature
    /// set は threat 行を持たないので実質 sentinel `[0]`、factorizer の virtual 対応を
    /// piece-input 行に限定する)。factorizer 無効でも 1 要素確保する。
    threat_pair_starts: DeviceBuffer<u32>,
    /// `--ft-fp16-out` opt-in flag。`true` の間 forward は `sparse_ft_forward_fp16_out`
    /// (FP16 weight + FP16 出力) で `ft_*_out` を f16 化し、FT post は活性化別の f16
    /// 入力 kernel (CReLU/SCReLU は `simple_bias_act_fwd_fp16_in_*`、Pairwise は
    /// `ft_post_perspective_fwd_fp16`) で combined を作る。backward も活性化別の f16
    /// 勾配 kernel (`simple_act_grad_to_fp16_*_with_scale` / `ft_post_perspective_grad_fp16`)
    /// で dft を f16 化し、`simple_bias_grad_fp16` / `simple_sparse_ft_backward_fp16` が
    /// 読み戻す。`ft_fp16` を要求する (3 活性化すべて対応)。dft は loss scaling で
    /// f16 域に持ち上げる ([`FT_DFT_FP16_BASE_SCALE`] × batch、±65504 clamp 付き)。
    ft_fp16_out: bool,
    /// `--fp16-opt-state` opt-in flag。`true` の間 `ft_w_m` / `ft_w_v` を [`MomentBuf::F16`]
    /// で確保し、optimizer step は [`radam_step_f16state`] / [`radam_step_f16state_mirror`]
    /// で動かす。FT 以外の moment は変更なし。raw checkpoint format は不変 (真値 f32 で
    /// 書き出し、resume 時に当該 run の精度へ再 quantize する)。
    fp16_opt_state: bool,

    // -- weight (FP32) --
    /// FT 重み (`ft_in × ft_out`、feature-major: `ft_w[feat*ft_out + out]`)。
    /// `sparse_ft_forward` の weight layout と一致する。
    ft_w: DeviceBuffer<f32>,
    /// FT bias (`ft_out`、stm/nstm 共有)。
    ft_b: DeviceBuffer<f32>,
    /// L1 dense 重み (`combined_dim × l1_out`、in-major: `l1_w[in*l1_out + out]`)。
    /// `dense_mm_fwd` の weight layout (`w[k*out_dim+oi]`) と一致する。
    l1_w: DeviceBuffer<f32>,
    /// L1 dense bias (`l1_out`)。
    l1_b: DeviceBuffer<f32>,
    /// L2 dense 重み (`l1_out × l2_out`、in-major)。
    l2_w: DeviceBuffer<f32>,
    /// L2 dense bias (`l2_out`)。
    l2_b: DeviceBuffer<f32>,
    /// L3 dense 重み (`l2_out × 1`、in-major)。
    l3_w: DeviceBuffer<f32>,
    /// L3 dense bias (1 要素)。
    l3_b: DeviceBuffer<f32>,

    // -- gradient buffer (各 weight と同 shape、f32) --
    ft_w_grad: DeviceBuffer<f32>,
    ft_b_grad: DeviceBuffer<f32>,
    l1_w_grad: DeviceBuffer<f32>,
    l1_b_grad: DeviceBuffer<f32>,
    l2_w_grad: DeviceBuffer<f32>,
    l2_b_grad: DeviceBuffer<f32>,
    l3_w_grad: DeviceBuffer<f32>,
    l3_b_grad: DeviceBuffer<f32>,

    // -- Ranger optimizer state (RAdam 1st/2nd moment + Lookahead slow weight、各 weight と同 shape) --
    /// FT Ranger 1st/2nd moment。既定 `f32`、`--fp16-opt-state` で `f16` ([`MomentBuf`])。
    /// 他 7 group の moment buffer は小さく `f16` 化の意味が無いので `f32` のまま。
    ft_w_m: MomentBuf,
    ft_w_v: MomentBuf,
    ft_w_slow: DeviceBuffer<f32>,
    ft_b_m: DeviceBuffer<f32>,
    ft_b_v: DeviceBuffer<f32>,
    ft_b_slow: DeviceBuffer<f32>,
    l1_w_m: DeviceBuffer<f32>,
    l1_w_v: DeviceBuffer<f32>,
    l1_w_slow: DeviceBuffer<f32>,
    l1_b_m: DeviceBuffer<f32>,
    l1_b_v: DeviceBuffer<f32>,
    l1_b_slow: DeviceBuffer<f32>,
    l2_w_m: DeviceBuffer<f32>,
    l2_w_v: DeviceBuffer<f32>,
    l2_w_slow: DeviceBuffer<f32>,
    l2_b_m: DeviceBuffer<f32>,
    l2_b_v: DeviceBuffer<f32>,
    l2_b_slow: DeviceBuffer<f32>,
    l3_w_m: DeviceBuffer<f32>,
    l3_w_v: DeviceBuffer<f32>,
    l3_w_slow: DeviceBuffer<f32>,
    l3_b_m: DeviceBuffer<f32>,
    l3_b_v: DeviceBuffer<f32>,
    l3_b_slow: DeviceBuffer<f32>,

    ws: SimpleGpuWorkspace,
    /// loss kernel が atomic add する Σerr² (f64、1 要素)。
    loss_acc: DeviceBuffer<f64>,
    /// extended WRM loss の per-position weight 和 Σw (f64、1 要素)。`wrm_weight_sum`
    /// kernel が atomic add し、`loss_wrm` の extended 経路が `1/Σw` 正規化に読む。
    /// 二乗誤差経路では未使用 (常に 0)。
    weight_sum_acc: DeviceBuffer<f64>,
    /// `--ft-fp16-out` 経路で `simple_act_grad_to_fp16_*_with_scale` が `dft_scale *
    /// grad` を `±65504` に cap した要素数の cumulative atomic counter (len 1)。
    /// `--monitor-fp16-clamps` 時に host が sb 末で D2H read、`[fp16-clamp]` line に
    /// 出す。`--ft-fp16-out` 無しでは対象 kernel が launch されないので常に 0。
    fp16_clamp_counter: DeviceBuffer<u64>,
    /// `--ft-fp16-out` 経路で dft FP16 書き込みを行った要素数の host-side cumulative
    /// counter。`[fp16-clamp]` ratio の分母。`--ft-fp16-out` 無しなら 0 のまま。
    fp16_clamp_elems_written: u64,
    /// `step()` 末の `loss_acc` 同期読みを 1-step lag な async D2H に置換する pinned
    /// host ring。host が `stream.synchronize` 待ち無しで次 batch の launch を発行できる
    /// ようになる。sb 末で [`TrainerBackend::flush_pending_loss`] が drain する
    /// (default `0.0` を本 trainer は override する)。`forward` / `validate` の同期
    /// read 経路は ring を介さず loss_acc を直接読む。
    loss_ring: AsyncLossRing,
    /// `step()` 先頭の入力 H2D (`stm/nstm idx` + `score/wdl` の 4 buffer、Simple は
    /// bucket 無し) を専用 copy stream で直前 step の compute と overlap させる ring。
    /// [`AsyncLossRing`] による host run-ahead と組合せて compute と H2D を同時実行する。
    /// `forward` / `validate` の同期 read 経路は ring を介さず直接 H2D する (1-shot で
    /// 後続 backward / optimizer が無いため overlap 余地が薄い)。
    input_ring: InputUploadRing,
    /// このトレーナのアーキ identity (feature set / 活性化 / 層次元)。
    id: SimpleId,
    /// Ranger lookahead step counter。`RANGER_K` の倍数で lerp する。
    step_count: u64,
    /// Ranger optimizer の weight decay 係数 (`radam_step` 引数)。
    weight_decay: f32,
    /// 推論時の評価値スケール (`round(QA * QB / 学習 scale)`)。量子化 checkpoint
    /// 出力の arch 文字列に書く (`SimpleWeights::fv_scale`)。
    fv_scale: i32,
}

impl Drop for SimpleGpuTrainer {
    fn drop(&mut self) {
        // device buffer 解放前に compute / copy 両 stream の in-flight 操作を排出する
        // (GpuTrainer と同じ規約: field drop 順による race を回避)。`input_ring` の
        // copy stream H2D が `ws` の入力 buffer を write 中の解放を防ぐため。
        let _ = self.stream.synchronize();
        let _ = self.input_ring.copy_stream.synchronize();
    }
}

impl SimpleGpuTrainer {
    /// 数値精度と optimizer state の形式は [`PrecisionFlags`] で指定する。
    pub(crate) fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch_size: usize,
        id: SimpleId,
        weight_decay: f32,
        fv_scale: i32,
        precision: PrecisionFlags,
        init_spec: &SimpleInit,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // `precision.ft_fp16_out` は `precision.ft_fp16` を必要とする。CLI validation は
        // 無効な組み合わせを拒否するが、smoke/test は constructor を直接呼べるため、ここでも検査する。
        debug_assert!(
            !precision.ft_fp16_out || precision.ft_fp16,
            "ft_fp16_out requires ft_fp16"
        );
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "nnue_train")?;
        let dense_bias_grad_occ = DeviceOccupancy::query(ctx)?;
        let cublas = CublasHandle::new(&stream, precision.tf32)?;

        let ft_in = id.ft_in();
        let ft_out = id.ft_out;
        let l1_out = id.l1_out;
        let l2_out = id.l2_out;
        // ft_out == 0 は FT 出力が無い退化アーキ。`is_multiple_of(4)` は 0 を通すため
        // 明示的に弾く。FT bias grad launch の grid.y = ceil(ft_out / min(ft_out, 1024)) が
        // 0 除算 panic になるのも防ぐ (CLI 経由でなく new を直接呼ぶ test / smoke 経路の保険)。
        if ft_out == 0 {
            return Err("SimpleGpuTrainer: ft_out must be greater than 0".into());
        }
        // sparse_ft_forward は 1 thread = 4 row なので ft_out が 4 の倍数必須。
        // Simple の preset (256/512/1024) は全部 4 の倍数だが、`--l1` override で
        // 4 の倍数でない値が来る可能性があるので early reject する。4 の倍数性は
        // pairwise が必要とする偶数性 (`half = ft_out / 2`) も内包する。
        if !ft_out.is_multiple_of(4) {
            return Err(format!(
                "SimpleGpuTrainer: ft_out {ft_out} must be a multiple of 4 \
                 (sparse_ft_forward processes 4 rows per thread)"
            )
            .into());
        }
        // pairwise の `half = ft_out / 2` 分割が割り切れることを明示確認する
        // (上の 4 の倍数チェックで保証済の不変条件、将来 4→2 緩和時の保険)。
        debug_assert!(
            ft_out.is_multiple_of(2),
            "pairwise requires even ft_out for the half-split"
        );

        // FT weight の行数。factorizer 有効時は base 実行 (`ft_in`) の後ろに piece-input
        // 仮想行が連結される (`train_ft_in`)。sparse index の範囲と active 数は factorizer
        // 非依存で base のまま — 仮想行は fold/reduce dense kernel でのみ読み書きされる。
        let ft_factorize = id.feature_set.ft_factorize();
        let train_ft_in = id.feature_set.train_ft_in();

        // weight / bias の初期値を `init_spec` から生成する。Simple は bucket / 共有
        // 因子層を持たないので全 group `flat`。fan_in は各層の入力次元 (FT=ft_in、
        // L1=combined_dim()、L2=l1_out、L3=l2_out)、bias は対応 weight と同じ fan_in。
        // FT weight は実 block を base 形状・base fan_in で sample し、factorizer の仮想
        // block は zero を append する (train 形状で一括 sample すると仮想行に noise が
        // 入り step-0 forward が OFF 構成とずれ、fan_in と RNG 消費数も変わって実 row の
        // 乱数列が OFF と不一致になる)。
        let mut ft_w_init = init::sample(WeightShape::flat(ft_in * ft_out, ft_in), &init_spec.ft_w);
        ft_w_init.resize(train_ft_in * ft_out, 0.0);
        let ft_b_h = init::sample(WeightShape::flat(ft_out, ft_in), &init_spec.ft_b);
        let l1_in = id.combined_dim();
        let l1_w_h = init::sample(WeightShape::flat(l1_in * l1_out, l1_in), &init_spec.l1_w);
        let l1_b_h = init::sample(WeightShape::flat(l1_out, l1_in), &init_spec.l1_b);
        let l2_w_h = init::sample(WeightShape::flat(l1_out * l2_out, l1_out), &init_spec.l2_w);
        let l2_b_h = init::sample(WeightShape::flat(l2_out, l1_out), &init_spec.l2_b);
        let l3_w_h = init::sample(WeightShape::flat(l2_out, l2_out), &init_spec.l3_w);
        let l3_b_h = init::sample(WeightShape::flat(1, l2_out), &init_spec.l3_b);

        let batch = batch_size.max(1);
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(&stream, n).map_err(Into::into)
        };
        let ft_w_n = train_ft_in * ft_out;
        let ft_b_n = ft_out;
        let l1_w_n = id.combined_dim() * l1_out;
        let l1_b_n = l1_out;
        let l2_w_n = l1_out * l2_out;
        let l2_b_n = l2_out;
        let l3_w_n = l2_out;
        let l3_b_n = 1;
        // Lookahead slow weight は学習開始時 weights と同値で初期化する
        // (Ranger の初期 `slow_param ← param` 規約)。
        let ft_w = DeviceBuffer::from_host(&stream, &ft_w_init)?;
        let ft_b = DeviceBuffer::from_host(&stream, &ft_b_h)?;
        let l1_w = DeviceBuffer::from_host(&stream, &l1_w_h)?;
        let l1_b = DeviceBuffer::from_host(&stream, &l1_b_h)?;
        let l2_w = DeviceBuffer::from_host(&stream, &l2_w_h)?;
        let l2_b = DeviceBuffer::from_host(&stream, &l2_b_h)?;
        let l3_w = DeviceBuffer::from_host(&stream, &l3_w_h)?;
        let l3_b = DeviceBuffer::from_host(&stream, &l3_b_h)?;
        let ft_w_slow = DeviceBuffer::from_host(&stream, &ft_w_init)?;
        let ft_b_slow = DeviceBuffer::from_host(&stream, &ft_b_h)?;
        let l1_w_slow = DeviceBuffer::from_host(&stream, &l1_w_h)?;
        let l1_b_slow = DeviceBuffer::from_host(&stream, &l1_b_h)?;
        let l2_w_slow = DeviceBuffer::from_host(&stream, &l2_w_h)?;
        let l2_b_slow = DeviceBuffer::from_host(&stream, &l2_b_h)?;
        let l3_w_slow = DeviceBuffer::from_host(&stream, &l3_w_h)?;
        let l3_b_slow = DeviceBuffer::from_host(&stream, &l3_b_h)?;
        let mut threat_pair_starts_host: Vec<u32> = id
            .feature_set
            .threat_factorize_pair_starts()
            .into_iter()
            .map(|x| x as u32)
            .collect();
        if threat_pair_starts_host.is_empty() {
            threat_pair_starts_host.push(0);
        }
        Ok(Self {
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
            ft_w_grad: z(ft_w_n)?,
            ft_b_grad: z(ft_b_n)?,
            l1_w_grad: z(l1_w_n)?,
            l1_b_grad: z(l1_b_n)?,
            l2_w_grad: z(l2_w_n)?,
            l2_b_grad: z(l2_b_n)?,
            l3_w_grad: z(l3_w_n)?,
            l3_b_grad: z(l3_b_n)?,
            ft_w_m: MomentBuf::zeroed(&stream, ft_w_n, precision.fp16_opt_state)?,
            ft_w_v: MomentBuf::zeroed(&stream, ft_w_n, precision.fp16_opt_state)?,
            ft_w_slow,
            ft_b_m: z(ft_b_n)?,
            ft_b_v: z(ft_b_n)?,
            ft_b_slow,
            l1_w_m: z(l1_w_n)?,
            l1_w_v: z(l1_w_n)?,
            l1_w_slow,
            l1_b_m: z(l1_b_n)?,
            l1_b_v: z(l1_b_n)?,
            l1_b_slow,
            l2_w_m: z(l2_w_n)?,
            l2_w_v: z(l2_w_n)?,
            l2_w_slow,
            l2_b_m: z(l2_b_n)?,
            l2_b_v: z(l2_b_n)?,
            l2_b_slow,
            l3_w_m: z(l3_w_n)?,
            l3_w_v: z(l3_w_n)?,
            l3_w_slow,
            l3_b_m: z(l3_b_n)?,
            l3_b_v: z(l3_b_n)?,
            l3_b_slow,
            ws: SimpleGpuWorkspace::new(&stream, batch, id, precision.ft_fp16_out)?,
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            weight_sum_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            fp16_clamp_counter: DeviceBuffer::<u64>::zeroed(&stream, 1)?,
            fp16_clamp_elems_written: 0,
            loss_ring: AsyncLossRing::new(ctx)?,
            input_ring: InputUploadRing::new_simple(ctx, batch, id.feature_set.max_active())?,
            // factorizer 有効時の `ft_w_h` / `ft_w_fold32` は forward 用 comb (base 形状)、
            // 無効時の `ft_w_h` は master の cast mirror (train 形状 = base と同値)。いずれも
            // `sync_ft_forward_weights` が初期同期する。
            ft_w_h: if precision.ft_fp16 {
                let n = if ft_factorize { ft_in * ft_out } else { ft_w_n };
                Some(DeviceBuffer::<f16>::zeroed(&stream, n)?)
            } else {
                None
            },
            ft_w_fold32: if ft_factorize && !precision.ft_fp16 {
                Some(DeviceBuffer::<f32>::zeroed(&stream, ft_in * ft_out)?)
            } else {
                None
            },
            threat_pair_starts: DeviceBuffer::from_host(&stream, &threat_pair_starts_host)?,
            stream,
            module,
            dense_bias_grad_occ,
            id,
            step_count: 0,
            weight_decay,
            fv_scale,
            cublas,
            ft_fp16: precision.ft_fp16,
            ft_fp16_out: precision.ft_fp16_out,
            fp16_opt_state: precision.fp16_opt_state,
        })
    }

    /// forward が読む FT weight buffer (`ft_w_h` mirror / factorizer の comb) を現在の
    /// `ft_w` (FP32 master) から再生成する。
    ///
    /// 構築直後の初期同期と、master を後から上書きする経路 (`--init-from` / `--resume`
    /// の load 後) の再同期が役目。学習中は factorizer 無効 + `--ft-fp16` の mirror を
    /// optimizer (`radam_step_fp16_mirror`) が step ごとに書き、factorizer 有効時の comb は
    /// 毎 step 末の [`launch_ft_fold`](Self::launch_ft_fold) が維持する。どちらにも該当
    /// しない構成 (FP32 + factorizer 無効) は forward が master を直接読むため no-op。
    pub(crate) fn sync_ft_forward_weights(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.id.feature_set.ft_factorize() {
            return self.launch_ft_fold();
        }
        if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
            let ft_w_n = self.ws.ft_in * self.id.ft_out;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: cast_f32_to_f16,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(ft_w_n),
                    args: [slice(self.ft_w), slice_mut(ft_w_h), ft_w_n as u32]
                }
            }?;
        }
        Ok(())
    }

    /// factorizer の forward 用畳み込み weight (comb = 実行 + 仮想行 broadcast、base 形状)
    /// を `ft_w` (train 形状の FP32 master) から再生成する。`--ft-fp16` 系列では f16 comb
    /// (`ft_w_h`)、FP32 では f32 comb (`ft_w_fold32`)。optimizer が master を書き換えた後、
    /// 次の forward が読む前に毎 step 1 回呼ぶ。caller が factorizer 有効を保証する
    /// (constructor が「factorize ⇒ comb buffer がちょうど 1 つ」を確立済み)。
    fn launch_ft_fold(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // 非 factorize で誤呼び出しされると ft_fp16 構成では train 形状 mirror へ base 想定
        // の fold が走り OOB read になるため、release でも入口で止める。
        assert!(self.id.feature_set.ft_factorize());
        let comb = if self.ft_fp16 {
            FoldComb::F16(
                self.ft_w_h
                    .as_mut()
                    .expect("ft_w_h (f16 comb) is Some when ft_factorize and ft_fp16 are enabled"),
            )
        } else {
            FoldComb::F32(
                self.ft_w_fold32
                    .as_mut()
                    .expect("ft_w_fold32 is Some when ft_factorize is enabled without ft_fp16"),
            )
        };
        ft_factorize_host::launch_ft_fold(
            &self.stream,
            &self.module,
            &self.id.feature_set,
            self.id.ft_out,
            &self.ft_w,
            comb,
            &self.threat_pair_starts,
        )
    }

    /// FT weight master (`ft_w`、factorizer 有効時は train 形状) を host に download する。
    /// factorizer の仮想行が更新されているかを test で観測するための accessor。
    #[cfg(test)]
    pub(crate) fn ft_w_to_host(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.ft_w.to_host_vec(&self.stream).map_err(Into::into)
    }

    /// 全 weight buffer を host に download し NaN/Inf が無いことを assert する。
    pub(crate) fn assert_all_weights_finite(&self) -> Result<(), Box<dyn std::error::Error>> {
        let groups: [(&DeviceBuffer<f32>, &str); 8] = [
            (&self.ft_w, "ft_w"),
            (&self.ft_b, "ft_b"),
            (&self.l1_w, "l1_w"),
            (&self.l1_b, "l1_b"),
            (&self.l2_w, "l2_w"),
            (&self.l2_b, "l2_b"),
            (&self.l3_w, "l3_w"),
            (&self.l3_b, "l3_b"),
        ];
        for (buf, name) in groups {
            let v = buf.to_host_vec(&self.stream)?;
            for (i, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(format!("{name}[{i}] = {x} is not finite (NaN or Inf)").into());
                }
            }
        }
        Ok(())
    }

    /// 1 batch の forward を走らせ、loss kernel が累積した Σerr² を返す。
    /// backward は走らせず、loss kernel が書く dnet (`dy_net_output`) は捨てる。
    pub(crate) fn forward(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        self.ws.check_batch_capacity(b)?;
        self.run_forward_kernels(batch, wdl_lambda, loss, false)?;
        let loss_host = self.loss_acc.to_host_vec(&self.stream)?;
        Ok(loss_host[0])
    }

    /// 1 batch の forward → backward → Ranger optimizer step を走らせ、loss kernel が
    /// 累積した Σerr² を返す。`bucket_idx` は受け取らない (Simple アーキは bucket 無し)。
    ///
    /// 環境変数 `NNUE_TRAIN_STEP_PROFILE` がセットされていれば各 phase の境界で
    /// `synchronize()` + 経過時間を stderr に出す (粗い forward / backward / optimizer /
    /// loss_readback breakdown)。LayerStack `GpuTrainer::step` と同 env var を共有。
    pub(crate) fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(0.0);
        }
        self.ws.check_batch_capacity(b)?;

        let mut prof_t0 = if std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 入力 4 buffer の H2D を `InputUploadRing` 経由で発行する。active / back を
        // `mem::swap` してから back (= 直前 step が読んでいない側の物理 buffer) へ専用
        // copy stream で先行 H2D。pageable な `BatchData` slice は ring 内の pinned host
        // を経由して copy engine の DMA に載り、compute stream は H2D 完了 event を
        // 待ってから forward に進む ([`AsyncLossRing`] による host run-ahead と組合せて
        // h2d_reset を直前 step の compute と並走させる)。
        std::mem::swap(&mut self.ws.stm_idx_dev, &mut self.ws.stm_idx_dev_back);
        std::mem::swap(&mut self.ws.nstm_idx_dev, &mut self.ws.nstm_idx_dev_back);
        std::mem::swap(&mut self.ws.nnz_dev, &mut self.ws.nnz_dev_back);
        std::mem::swap(&mut self.ws.score_dev, &mut self.ws.score_dev_back);
        std::mem::swap(&mut self.ws.wdl_dev, &mut self.ws.wdl_dev_back);
        self.input_ring.upload_simple(
            &self.stream,
            &self.ws.stm_idx_dev,
            &batch.stm_indices[..b * self.ws.max_active],
            &self.ws.nstm_idx_dev,
            &batch.nstm_indices[..b * self.ws.max_active],
            &self.ws.nnz_dev,
            &batch.nnz[..b],
            &self.ws.score_dev,
            &batch.score[..b],
            &self.ws.wdl_dev,
            &batch.wdl[..b],
        )?;

        self.run_forward_kernels(batch, wdl_lambda, loss, true)?;
        if let Some(ref mut t0) = prof_t0 {
            self.stream.synchronize()?;
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "forward",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
            *t0 = now;
        }

        self.run_backward_kernels(b)?;
        if let Some(ref mut t0) = prof_t0 {
            self.stream.synchronize()?;
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "backward",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
            *t0 = now;
        }

        self.run_optimizer_step(lr)?;
        if let Some(ref mut t0) = prof_t0 {
            self.stream.synchronize()?;
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "optimizer",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
            *t0 = now;
        }

        // 本 step の compute (input buffer の read を含む) 完了を copy stream 用の
        // event に記録する。同じ物理 input buffer を使う step+2 の H2D がこれを待ち、
        // in-flight な compute が読む buffer を H2D が上書きする race を防ぐ。
        self.input_ring.mark_step_done(&self.stream)?;

        // `loss_acc` の host 読みを [`AsyncLossRing`] 経由で async + 1-step lag に
        // する。pinned host cell に `memcpy_dtoh_async` + event record、前 step の
        // event を sync して 1 step 遅れで loss を返す (step 0 は warmup として 0.0、
        // sb 末で [`TrainerBackend::flush_pending_loss`] が最終 step 分を drain する)。
        // host は次 batch の launch 発行で `stream.synchronize` 相当の block 待ちが消える。
        let loss = self
            .loss_ring
            .read_and_queue_next(&self.stream, &self.loss_acc)?;
        if let Some(ref t0) = prof_t0 {
            let now = std::time::Instant::now();
            eprintln!(
                "[step-profile] {:<14} {:8.3} ms",
                "loss_readback",
                now.duration_since(*t0).as_secs_f64() * 1000.0
            );
        }

        Ok(loss)
    }

    /// forward kernel 列のみを走らせる (loss は host 同期 read しない)。`forward` /
    /// `step` 共通の前段で、終了時 `net_output` / `dy_net_output` / `loss_acc` が device
    /// 上に書かれている。caller が batch capacity を事前検査する責務。
    ///
    /// `inputs_uploaded_externally` が `true` のとき caller (= `step`) が直前に
    /// [`InputUploadRing`] で `ws.{stm,nstm}_idx_dev` / `ws.{score,wdl}_dev` への H2D
    /// を queue 済とみなし、本 method 内では H2D を発行しない (compute stream は ring
    /// の `h2d_done` event 経由で H2D 完了を既に待つ)。`false` のとき (`forward` /
    /// `validate` 経路) は同期 H2D を default stream 上で発行する。
    pub(crate) fn run_forward_kernels(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
        inputs_uploaded_externally: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut prof_t0 = if std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let tick = |label: &str,
                    stream: &CudaStream,
                    t0: &mut Option<std::time::Instant>|
         -> Result<(), Box<dyn std::error::Error>> {
            if let Some(t) = t0 {
                stream.synchronize()?;
                let now = std::time::Instant::now();
                eprintln!(
                    "[step-profile]   {:<12} {:8.3} ms",
                    label,
                    now.duration_since(*t).as_secs_f64() * 1000.0
                );
                *t = now;
            }
            Ok(())
        };
        let b = batch.n_pos;
        let b_u32 = b as u32;
        let ft_out_u32 = self.id.ft_out as u32;
        // L1 入力 = stm/nstm の FT post 出力 concat。CReLU/SCReLU は 2*ft_out、
        // Pairwise は pairwise 乗算で半減し ft_out。
        let l1_in_u32 = self.id.combined_dim() as u32;
        let l1_out_u32 = self.id.l1_out as u32;
        let l2_out_u32 = self.id.l2_out as u32;
        let ft_n = b * self.id.ft_out;

        // -- H2D upload (default stream 上の async memcpy、launch 列に直列で並ぶ) --
        // ring 経路では caller (= `step`) が swap + `input_ring.upload_simple` で
        // copy stream に H2D を発行済で、`compute_stream.wait(h2d_done)` で本 stream に
        // 完了待ちが乗っているため、ここでの H2D は不要。
        if !inputs_uploaded_externally {
            copy_host_to_device_async_i32(
                &self.stream,
                &self.ws.stm_idx_dev,
                &batch.stm_indices[..b * self.ws.max_active],
            )?;
            copy_host_to_device_async_i32(
                &self.stream,
                &self.ws.nstm_idx_dev,
                &batch.nstm_indices[..b * self.ws.max_active],
            )?;
            copy_host_to_device_async_i32(&self.stream, &self.ws.nnz_dev, &batch.nnz[..b])?;
            copy_host_to_device_async_f32(&self.stream, &self.ws.score_dev, &batch.score[..b])?;
            copy_host_to_device_async_f32(&self.stream, &self.ws.wdl_dev, &batch.wdl[..b])?;
        }

        // -- loss_acc / weight_sum_acc を 0 にリセット (再 alloc 無し) --
        memset_zero(&self.stream, &self.loss_acc)?;
        memset_zero(&self.stream, &self.weight_sum_acc)?;
        tick("h2d_reset", &self.stream, &mut prof_t0)?;

        // -- sparse_ft_forward × 2 (stm, nstm)。1 thread = 4 row。
        // 3 path 分岐:
        //  - `ft_fp16_out`: `sparse_ft_forward_fp16_out` (f16 weight + f16 出力)、ft_*_out
        //    は f16 buffer (`ft_*_out_h`) に書く (pre-bias、bias は後段 fused kernel で加算)
        //  - `ft_fp16`: `sparse_ft_forward_fp16` (f16 weight + f32 出力)、ft_*_out は f32
        //  - 既定: `sparse_ft_forward` (FP32 master、bit-identical)
        if self.ft_fp16_out {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16_out is enabled");
            let mut ft_stm_out_h = self
                .ws
                .ft_stm_out_h
                .as_mut()
                .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: sparse_ft_forward_fp16_out,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b * self.id.ft_out / 4),
                    args: [
                        slice(ft_w_h),
                        slice(self.ws.stm_idx_dev),
                        slice(self.ws.nnz_dev),
                        slice_mut(ft_stm_out_h),
                        b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                    ]
                }
            }?;
            let mut ft_nstm_out_h = self
                .ws
                .ft_nstm_out_h
                .as_mut()
                .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: sparse_ft_forward_fp16_out,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b * self.id.ft_out / 4),
                    args: [
                        slice(ft_w_h),
                        slice(self.ws.nstm_idx_dev),
                        slice(self.ws.nnz_dev),
                        slice_mut(ft_nstm_out_h),
                        b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                    ]
                }
            }?;
        } else if self.ft_fp16 {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16 is enabled");
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: sparse_ft_forward_fp16,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b * self.id.ft_out / 4),
                    args: [
                        slice(ft_w_h),
                        slice(self.ws.stm_idx_dev),
                        slice(self.ws.nnz_dev),
                        slice_mut(self.ws.ft_stm_out),
                        b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                    ]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: sparse_ft_forward_fp16,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b * self.id.ft_out / 4),
                    args: [
                        slice(ft_w_h),
                        slice(self.ws.nstm_idx_dev),
                        slice(self.ws.nnz_dev),
                        slice_mut(self.ws.ft_nstm_out),
                        b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                    ]
                }
            }?;
        } else {
            // factorizer 有効時は畳み込み済み comb (`ft_w_fold32`、base 形状) を読む
            // (sparse path を base 次元に保つ)。無効時は master を直接読む。
            let ft_w_fwd = self.ft_w_fold32.as_ref().unwrap_or(&self.ft_w);
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: sparse_ft_forward,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b * self.id.ft_out / 4),
                    args: [
                        slice(ft_w_fwd),
                        slice(self.ws.stm_idx_dev),
                        slice(self.ws.nnz_dev),
                        slice_mut(self.ws.ft_stm_out),
                        b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                    ]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: sparse_ft_forward,
                    stream: self.stream,
                    module: self.module,
                    config: cfg_1d(b * self.id.ft_out / 4),
                    args: [
                        slice(ft_w_fwd),
                        slice(self.ws.nstm_idx_dev),
                        slice(self.ws.nnz_dev),
                        slice_mut(self.ws.ft_nstm_out),
                        b_u32, ft_out_u32, self.ws.ft_in as u32, self.ws.max_active as u32
                    ]
                }
            }?;
        }
        tick("fwd_ft", &self.stream, &mut prof_t0)?;

        // -- FT post = bias add + 活性化 + concat。
        // `ft_fp16_out` 時は f16 `ft_*_out_h` を読む活性化別 kernel で `combined` を作る。
        // CReLU / SCReLU / Pairwise いずれも活性化 kernel が `combined` の per-perspective
        // 列範囲 (`col_offset`) へ直接書く (中間 `ft_*_acted` + `slice_scatter_2d` は融合で除去)。
        // 既定 (FP32) 経路は bias_add + 活性化 + slice_scatter を融合した kernel 群で合成する。
        if self.ft_fp16_out {
            // CReLU / SCReLU は nstm を `col_offset = ft_out` で combined に直書きするため
            // `2*ft_out <= combined_dim` (= row stride) が成立しないと OOB write になる。
            // CReLU/SCReLU は両 perspective 連結で combined_dim = 2*ft_out なので常に成立。
            // Pairwise は combined_dim = ft_out で別 kernel を使うため対象外 (matches! で除外)。
            debug_assert!(
                !matches!(
                    self.id.activation,
                    SimpleActivation::CReLU | SimpleActivation::SCReLU
                ) || 2 * ft_out_u32 <= l1_in_u32,
                "fp16-out CReLU/SCReLU fwd needs 2*ft_out <= combined_dim (got 2*{ft_out_u32} vs {l1_in_u32})"
            );
            match self.id.activation {
                SimpleActivation::CReLU => {
                    let ft_stm_out_h = self
                        .ws
                        .ft_stm_out_h
                        .as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bias_act_fwd_fp16_in_crelu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_stm_out_h), slice(self.ft_b),
                                slice_mut(self.ws.combined), l1_in_u32, 0_u32, b_u32, ft_out_u32
                            ]
                        }
                    }?;
                    let ft_nstm_out_h = self
                        .ws
                        .ft_nstm_out_h
                        .as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bias_act_fwd_fp16_in_crelu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_nstm_out_h), slice(self.ft_b),
                                slice_mut(self.ws.combined), l1_in_u32, ft_out_u32, b_u32, ft_out_u32
                            ]
                        }
                    }?;
                }
                SimpleActivation::SCReLU => {
                    let ft_stm_out_h = self
                        .ws
                        .ft_stm_out_h
                        .as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bias_act_fwd_fp16_in_screlu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_stm_out_h), slice(self.ft_b),
                                slice_mut(self.ws.combined), l1_in_u32, 0_u32, b_u32, ft_out_u32
                            ]
                        }
                    }?;
                    let ft_nstm_out_h = self
                        .ws
                        .ft_nstm_out_h
                        .as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bias_act_fwd_fp16_in_screlu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_nstm_out_h), slice(self.ft_b),
                                slice_mut(self.ws.combined), l1_in_u32, ft_out_u32, b_u32, ft_out_u32
                            ]
                        }
                    }?;
                }
                SimpleActivation::Pairwise => {
                    // f16 入力版 pairwise FT post。`combined` (b × ft_out) を直書きするため
                    // 後段の slice_scatter は不要 (FP32 pairwise path と同じ構造)。
                    let ft_stm_out_h = self
                        .ws
                        .ft_stm_out_h
                        .as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
                    let ft_nstm_out_h = self
                        .ws
                        .ft_nstm_out_h
                        .as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: ft_post_perspective_fwd_fp16,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_stm_out_h), slice(ft_nstm_out_h), slice(self.ft_b),
                                slice_mut(self.ws.combined), b_u32, ft_out_u32, FT_POST_SCALE
                            ]
                        }
                    }?;
                }
            }
        } else {
            // DEFAULT (FP32) path: bias_add + activation + slice_scatter (per perspective × 2)
            // を 1 kernel に融合。`ft_*_out` に bias を in-place 加算 (bwd indicator が読む)
            // した後、活性化結果を直接 `combined` の per-perspective slice に書く。中間
            // `ft_*_acted` buffer の DRAM write+read と、bias_add → 活性化 間の `ft_*_out`
            // 再 read+write が消える (1 perspective につき ~536 MB DRAM、2 perspective で
            // ~1.07 GB)。`ft_fp16_out` 経路は scale / clamp / f16 cast 含むため融合外。
            match self.id.activation {
                SimpleActivation::CReLU => {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_ft_post_fused_crelu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice_mut(self.ws.ft_stm_out),
                                slice(self.ft_b),
                                slice_mut(self.ws.combined),
                                b_u32, ft_out_u32, 0_u32
                            ]
                        }
                    }?;
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_ft_post_fused_crelu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice_mut(self.ws.ft_nstm_out),
                                slice(self.ft_b),
                                slice_mut(self.ws.combined),
                                b_u32, ft_out_u32, ft_out_u32
                            ]
                        }
                    }?;
                }
                SimpleActivation::SCReLU => {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_ft_post_fused_screlu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice_mut(self.ws.ft_stm_out),
                                slice(self.ft_b),
                                slice_mut(self.ws.combined),
                                b_u32, ft_out_u32, 0_u32
                            ]
                        }
                    }?;
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_ft_post_fused_screlu,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice_mut(self.ws.ft_nstm_out),
                                slice(self.ft_b),
                                slice_mut(self.ws.combined),
                                b_u32, ft_out_u32, ft_out_u32
                            ]
                        }
                    }?;
                }
                SimpleActivation::Pairwise => {
                    // Pairwise FT post: bias add + CReLU + pairwise_mul を 1 kernel で
                    // 両 perspective まとめて `combined` (b × ft_out) に直書きする
                    // (`ft_post_perspective_fwd` は stm を前半 `[0, ft_out/2)`、nstm を
                    // 後半 `[ft_out/2, ft_out)` に置く)。`ft_*_out` は bias 未加算のまま
                    // 保持し、backward の `ft_post_perspective_grad` が bias を再加算する。
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: ft_post_perspective_fwd,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(self.ws.ft_stm_out),
                                slice(self.ws.ft_nstm_out),
                                slice(self.ft_b),
                                slice_mut(self.ws.combined),
                                b_u32, ft_out_u32, FT_POST_SCALE
                            ]
                        }
                    }?;
                }
            }
        }

        tick("fwd_ft_post", &self.stream, &mut prof_t0)?;

        // -- L1 dense (combined → l1_pre) cuBLAS Sgemm + bias_add_per_row --
        // shape: combined[B, l1_in] @ l1_w[l1_in, l1_out] → l1_pre[B, l1_out] (l1_in =
        // combined_dim)、続けて bias を別 kernel で row-add する (Sgemm 自身は bias 非対応)。
        //
        // SAFETY: combined / l1_w / l1_pre は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (workspace 系の `combined.len() >= b * l1_in` / `l1_pre.len() >= b * l1_out`
        // は事前の `ws.check_batch_capacity(b)` で保証、weight `l1_w.len() == l1_in
        // * l1_out` は固定 shape)、`self.cublas` は `self.stream` に bind 済で同 stream 内
        // in-order 実行 (先行 kernel 完了後に Sgemm が走り、後続 bias_add_per_row が観測)。
        // `cu_deviceptr() as *const/*mut f32` cast の妥当性: cuMemAlloc が返す device
        // pointer は 256 byte aligned (`f32` の 4 byte 要求を満たす)、`self` の借用が
        // unsafe block を超えて生存するので元 buffer も同 lifetime で valid。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,
                l1_out_u32 as i32,
                l1_in_u32 as i32,
                self.ws.combined.cu_deviceptr() as *const f32,
                self.l1_w.cu_deviceptr() as *const f32,
                self.ws.l1_pre.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: bias_add_per_row,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.l1_out),
                args: [slice(self.l1_b), slice_mut(self.ws.l1_pre), b_u32, l1_out_u32]
            }
        }?;
        let l1_n = b * self.id.l1_out;
        let l1_n_u32 = l1_n as u32;
        // L1 / L2 の dense 出力は単一ベクトルで pairwise 乗算が適用できないため、
        // Pairwise でも CReLU で活性化する (pairwise 乗算は FT post 固有)。
        match self.id.activation {
            SimpleActivation::CReLU | SimpleActivation::Pairwise => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: crelu_fwd, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_n),
                    args: [slice(self.ws.l1_pre), slice_mut(self.ws.l1_acted), l1_n_u32]
                }
            }?,
            SimpleActivation::SCReLU => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: screlu_fwd, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_n),
                    args: [slice(self.ws.l1_pre), slice_mut(self.ws.l1_acted), l1_n_u32]
                }
            }?,
        }

        // -- L2 dense (l1_acted → l2_pre) cuBLAS Sgemm + bias_add_per_row --
        // shape: l1_acted[B, l1_out] @ l2_w[l1_out, l2_out] → l2_pre[B, l2_out]、
        // 続けて bias を別 kernel で row-add する (Sgemm 自身は bias 非対応)。
        // L1 fwd と同じ手で、`dense_mm_fwd` の thread 数が `B * out_dim` で
        // SM 占有率が低いのを cuBLAS で塗り替える。
        //
        // SAFETY: l1_acted / l2_w / l2_pre は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (`l1_acted.len() >= b * l1_out` / `l2_pre.len() >= b * l2_out` は事前の
        // `ws.check_batch_capacity(b)` で保証、weight `l2_w.len() == l1_out * l2_out`
        // は固定 shape)、`self.cublas` は `self.stream` に bind 済の同一 stream を再利用。
        // `alpha=1`, `beta=0` overwrite なので `l2_pre` の事前内容は使われない (後続の
        // `bias_add_per_row` が書き戻し)。`cu_deviceptr() as *const/*mut f32` cast の
        // 妥当性は L1 と同じ前提 (256 byte aligned cuMemAlloc 由来、`self` 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,
                l2_out_u32 as i32,
                l1_out_u32 as i32,
                self.ws.l1_acted.cu_deviceptr() as *const f32,
                self.l2_w.cu_deviceptr() as *const f32,
                self.ws.l2_pre.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: bias_add_per_row,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * self.id.l2_out),
                args: [slice(self.l2_b), slice_mut(self.ws.l2_pre), b_u32, l2_out_u32]
            }
        }?;
        let l2_n = b * self.id.l2_out;
        let l2_n_u32 = l2_n as u32;
        match self.id.activation {
            SimpleActivation::CReLU | SimpleActivation::Pairwise => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: crelu_fwd, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_n),
                    args: [slice(self.ws.l2_pre), slice_mut(self.ws.l2_acted), l2_n_u32]
                }
            }?,
            SimpleActivation::SCReLU => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: screlu_fwd, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_n),
                    args: [slice(self.ws.l2_pre), slice_mut(self.ws.l2_acted), l2_n_u32]
                }
            }?,
        }

        // -- L3 dense (l2_acted → net_output)。out_dim = 1 (スカラ出力)、cuBLAS Sgemm + bias --
        // shape: l2_acted[B, l2_out] @ l3_w[l2_out, 1] → net_output[B, 1]。cuBLAS は
        // N=1 (= matrix-vector 相当) でも内部で適切な algorithm を選ぶ。
        //
        // SAFETY: l2_acted / l3_w / net_output は cudaMalloc 由来 + 長さは仕様分以上
        // (`l2_acted.len() >= b * l2_out` / `net_output.len() >= b` は事前の
        // `ws.check_batch_capacity(b)` で保証、weight `l3_w.len() == l2_out` は固定 shape)。
        // `alpha=1`, `beta=0` overwrite なので `net_output` の事前内容は使われない (後続の
        // `bias_add_per_row` が書き戻し)。残りの不変条件は L1 / L2 と同じ。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,
                1_i32,
                l2_out_u32 as i32,
                self.ws.l2_acted.cu_deviceptr() as *const f32,
                self.l3_w.cu_deviceptr() as *const f32,
                self.ws.net_output.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: bias_add_per_row,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b),
                args: [slice(self.l3_b), slice_mut(self.ws.net_output), b_u32, 1_u32]
            }
        }?;
        tick("fwd_dense", &self.stream, &mut prof_t0)?;

        // -- loss kernel (Σerr² を loss_acc に atomic accumulate)、`dy_net_output` に
        // L3 出力への grad を書く。`wdl_lambda` で score/wdl ターゲットを blend する。
        match loss {
            LossKind::Sigmoid { scale } => {
                unsafe {
                    // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                    // stream の完了を待つ同期点まで生存する device allocation。
                    cuda_launch! {
                        kernel: loss_wdl,
                        stream: self.stream,
                        module: self.module,
                        config: cfg_1d(b),
                        args: [
                            slice(self.ws.net_output),
                            slice(self.ws.score_dev),
                            slice(self.ws.wdl_dev),
                            batch.per_pos_norm,
                            slice_mut(self.ws.dy_net_output),
                            slice(self.loss_acc),
                            wdl_lambda,
                            scale,
                            b_u32
                        ]
                    }
                }?;
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
                in_offset,
                target_offset,
                target_scaling,
                pow_exp,
                qp_asymmetry,
                weight_boost_w1,
                weight_boost_w2,
            } => {
                // extended (nnue-pytorch 一般化) loss は Σw 正規化を要するので、先に
                // wrm_weight_sum で Σw を確定させる。既定の拡張パラメータでは二乗誤差に
                // 帰着し weight_sum を launch せず bit-identical 経路を通す。
                let extended = loss.wrm_extended();
                if extended {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: wrm_weight_sum,
                            stream: self.stream,
                            module: self.module,
                            config: cfg_1d(b),
                            args: [
                                slice(self.ws.score_dev),
                                slice(self.weight_sum_acc),
                                weight_boost_w1, weight_boost_w2,
                                target_offset, target_scaling, b_u32
                            ]
                        }
                    }?;
                }
                unsafe {
                    // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                    // stream の完了を待つ同期点まで生存する device allocation。
                    cuda_launch! {
                        kernel: loss_wrm,
                        stream: self.stream,
                        module: self.module,
                        config: cfg_1d(b),
                        args: [
                            slice(self.ws.net_output),
                            slice(self.ws.score_dev),
                            slice(self.ws.wdl_dev),
                            batch.per_pos_norm,
                            slice_mut(self.ws.dy_net_output),
                            slice(self.loss_acc),
                            wdl_lambda,
                            nnue2score, in_scaling, in_offset,
                            target_offset, target_scaling,
                            pow_exp, qp_asymmetry, weight_boost_w1, weight_boost_w2,
                            slice(self.weight_sum_acc),
                            if extended { 1_u32 } else { 0_u32 },
                            b_u32
                        ]
                    }
                }?;
            }
        }
        tick("fwd_loss", &self.stream, &mut prof_t0)?;
        Ok(())
    }

    /// `run_forward_kernels` の直後に呼び、`dy_net_output` を起点として 8 weight group
    /// の gradient buffer を埋める。bias / FT weight は atomic accumulate のため host で
    /// 0 初期化する。dense weight (l1/l2/l3_w) は `dense_mm_bwd_weight` が overwrite
    /// 書きなので初期化不要。
    pub(crate) fn run_backward_kernels(
        &mut self,
        b: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut prof_t0 = if std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some() {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let tick = |label: &str,
                    stream: &CudaStream,
                    t0: &mut Option<std::time::Instant>|
         -> Result<(), Box<dyn std::error::Error>> {
            if let Some(t) = t0 {
                stream.synchronize()?;
                let now = std::time::Instant::now();
                eprintln!(
                    "[step-profile]   {:<12} {:8.3} ms",
                    label,
                    now.duration_since(*t).as_secs_f64() * 1000.0
                );
                *t = now;
            }
            Ok(())
        };
        let b_u32 = b as u32;
        let ft_out_u32 = self.id.ft_out as u32;
        let l1_in_u32 = self.id.combined_dim() as u32;
        let l1_out_u32 = self.id.l1_out as u32;
        let l2_out_u32 = self.id.l2_out as u32;
        let ft_in_u32 = self.ws.ft_in as u32;
        let max_active_u32 = self.ws.max_active as u32;
        let ft_n = b * self.id.ft_out;
        let l1_n = b * self.id.l1_out;
        let l1_n_u32 = l1_n as u32;
        let l2_n = b * self.id.l2_out;
        let l2_n_u32 = l2_n as u32;

        // bias grad kernel (`dense_bias_grad_tiled` / `simple_bias_grad_dual`) は atomic add で
        // 累積するため host で 0 初期化必須。`ft_w_grad` は後段の
        // `gather_and_sum_per_feature_overwrite` (本関数末尾の inverse-index pipeline、
        // iter 0 = stm) が全 `(feature, ri)` cell を書き切るため reset 不要。
        memset_zero(&self.stream, &self.ft_b_grad)?;
        memset_zero(&self.stream, &self.l1_b_grad)?;
        memset_zero(&self.stream, &self.l2_b_grad)?;
        memset_zero(&self.stream, &self.l3_b_grad)?;
        tick("bwd_memset", &self.stream, &mut prof_t0)?;

        // ---- L3: dy_net_output (b × 1) -> dl2_acted (b × l2_out), l3_w_grad, l3_b_grad ----
        // bwd_input: dl2_acted[B, l2_out] = dy_net_output[B, 1] @ l3_w[l2_out, 1]^T
        // bwd_weight: l3_w_grad[l2_out, 1] = l2_acted[B, l2_out]^T @ dy_net_output[B, 1]
        // out_dim=1 で weight grad は thread = l2_out (= 数十) しか起動できない matmul
        // shape のため、cuBLAS Sgemm に委譲して内部 Sgemv-相当 algorithm + B 軸並列で
        // SM 占有率を稼ぐ (untiled kernel は in_dim*out_dim thread 駆動で 1 warp 規模)。
        //
        // SAFETY: 全 device pointer は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (`dy_net_output.len() >= b` / `dl2_acted.len() >= b*l2_out` / `l2_acted.len()
        // >= b*l2_out` は `ws.check_batch_capacity(b)` で保証、weight grad `l3_w_grad.len()
        // == l2_out` は固定 shape)、`self.cublas` は `self.stream` に bind 済。
        // `alpha=1`, `beta=0` overwrite なので `dl2_acted` / `l3_w_grad` の事前内容は
        // 使われない。cast 妥当性は L1/L2 と同じ前提 (256 byte aligned cuMemAlloc 由来、
        // `self` 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_x_yt_rowmajor(
                b_u32 as i32,
                l2_out_u32 as i32,
                1_i32,
                self.ws.dy_net_output.cu_deviceptr() as *const f32,
                self.l3_w.cu_deviceptr() as *const f32,
                self.ws.dl2_acted.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                l2_out_u32 as i32,
                1_i32,
                b_u32 as i32,
                self.ws.l2_acted.cu_deviceptr() as *const f32,
                self.ws.dy_net_output.cu_deviceptr() as *const f32,
                self.l3_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: dense_bias_grad_tiled, stream: self.stream, module: self.module,
                config: cfg_dense_bias_grad(self.dense_bias_grad_occ, b_u32, 1),
                args: [slice(self.ws.dy_net_output), slice(self.l3_b_grad), b_u32, 1_u32]
            }
        }?;
        tick("L3_dense", &self.stream, &mut prof_t0)?;

        // ---- L2 activation grad: dl2_acted -> dl2_pre (kernel reads l2_pre) ----
        // L1 / L2 dense は Pairwise でも CReLU 活性化 (forward と対)。
        match self.id.activation {
            SimpleActivation::CReLU | SimpleActivation::Pairwise => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: crelu_grad, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_n),
                    args: [slice(self.ws.l2_pre), slice(self.ws.dl2_acted),
                           slice_mut(self.ws.dl2_pre), l2_n_u32]
                }
            }?,
            SimpleActivation::SCReLU => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: screlu_grad, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_n),
                    args: [slice(self.ws.l2_pre), slice(self.ws.dl2_acted),
                           slice_mut(self.ws.dl2_pre), l2_n_u32]
                }
            }?,
        }

        // ---- L2 dense backward: dl2_pre -> dl1_acted (cuBLAS), l2_w_grad (cuBLAS),
        //      l2_b_grad (kernel) ----
        // bwd_input: dl1_acted[B, l1_out] = dl2_pre[B, l2_out] @ l2_w[l1_out, l2_out]^T
        // bwd_weight: l2_w_grad[l1_out, l2_out] = l1_acted[B, l1_out]^T @ dl2_pre[B, l2_out]
        // weight grad は thread = l1_out*l2_out 駆動で `B` を内側 loop に置く構造、
        // 小 matmul shape (in_dim*out_dim が数百-千) で SM 占有率を稼げない。cuBLAS Sgemm
        // は `B` を block 並列に展開できる algorithm を選ぶ。
        //
        // SAFETY: 全 device pointer は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (`dl2_pre.len() >= b*l2_out` / `dl1_acted.len() >= b*l1_out` / `l1_acted.len()
        // >= b*l1_out` は `ws.check_batch_capacity(b)` で保証、weight grad `l2_w_grad.len()
        // == l1_out*l2_out` は固定 shape)、`self.cublas` は `self.stream` に bind 済。
        // `alpha=1`, `beta=0` overwrite なので `dl1_acted` / `l2_w_grad` の事前内容は
        // 使われない。cast 妥当性は L1 と同じ前提 (256 byte aligned cuMemAlloc、`self`
        // 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_x_yt_rowmajor(
                b_u32 as i32,
                l1_out_u32 as i32,
                l2_out_u32 as i32,
                self.ws.dl2_pre.cu_deviceptr() as *const f32,
                self.l2_w.cu_deviceptr() as *const f32,
                self.ws.dl1_acted.cu_deviceptr() as *mut f32,
            )?;
        }
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                l1_out_u32 as i32,
                l2_out_u32 as i32,
                b_u32 as i32,
                self.ws.l1_acted.cu_deviceptr() as *const f32,
                self.ws.dl2_pre.cu_deviceptr() as *const f32,
                self.l2_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        // `dense_bias_grad_tiled` の shared PARTIAL 容量 (block_dim <= 256) を超える out_dim
        // (`--arch`/`--l3` で l2_out > 256 を指定した層) では任意幅で動く generic `bias_grad`
        // に fall back する。常用 arch は l2_out <= 256 で tiled path を通る。
        if l2_out_u32 <= DENSE_BIAS_GRAD_MAX_OUT {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: dense_bias_grad_tiled, stream: self.stream, module: self.module,
                    config: cfg_dense_bias_grad(self.dense_bias_grad_occ, b_u32, l2_out_u32),
                    args: [slice(self.ws.dl2_pre), slice(self.l2_b_grad), b_u32, l2_out_u32]
                }
            }?;
        } else {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: bias_grad, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_n),
                    args: [slice(self.ws.dl2_pre), slice(self.l2_b_grad), b_u32, l2_out_u32]
                }
            }?;
        }
        tick("L2_dense", &self.stream, &mut prof_t0)?;

        // ---- L1 activation grad: dl1_acted -> dl1_pre (kernel reads l1_pre) ----
        match self.id.activation {
            SimpleActivation::CReLU | SimpleActivation::Pairwise => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: crelu_grad, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_n),
                    args: [slice(self.ws.l1_pre), slice(self.ws.dl1_acted),
                           slice_mut(self.ws.dl1_pre), l1_n_u32]
                }
            }?,
            SimpleActivation::SCReLU => unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: screlu_grad, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_n),
                    args: [slice(self.ws.l1_pre), slice(self.ws.dl1_acted),
                           slice_mut(self.ws.dl1_pre), l1_n_u32]
                }
            }?,
        }

        // ---- L1 dense backward: dl1_pre -> dcombined (cuBLAS), l1_w_grad (cuBLAS), l1_b_grad (kernel) ----
        // bwd_input: dcombined[B, l1_in] = dl1_pre[B, l1_out] @ l1_w[l1_in, l1_out]^T
        //   ( = dx[b][i] = sum_o dy[b][o] * w[i][o]、l1_in = combined_dim )
        // SAFETY: dl1_pre / l1_w / dcombined は cudaMalloc 由来 + 長さは Sgemm 仕様分以上
        // (workspace 系の `dl1_pre.len() >= b*l1_out` / `dcombined.len() >= b*l1_in`
        // は `ws.check_batch_capacity(b)` で保証、weight `l1_w.len() == l1_in*l1_out`
        // は固定 shape)、`self.cublas` は `self.stream` に bind 済。`cu_deviceptr() as
        // *const/*mut f32` cast の妥当性: cuMemAlloc が返す device pointer は 256 byte
        // aligned (`f32` の 4 byte 要求を満たす)、`self` の借用が unsafe block を超えて
        // 生存するので元 buffer も同 lifetime で valid。
        unsafe {
            self.cublas.sgemm_x_yt_rowmajor(
                b_u32 as i32,
                l1_in_u32 as i32,
                l1_out_u32 as i32,
                self.ws.dl1_pre.cu_deviceptr() as *const f32,
                self.l1_w.cu_deviceptr() as *const f32,
                self.ws.dcombined.cu_deviceptr() as *mut f32,
            )?;
        }
        // bwd_weight: l1_w_grad[l1_in, l1_out] = combined[B, l1_in]^T @ dl1_pre[B, l1_out]
        // SAFETY: combined / dl1_pre / l1_w_grad は cudaMalloc 由来 + 長さは Sgemm 仕様分
        // 以上 (workspace 系の `combined.len() >= b*l1_in` / `dl1_pre.len() >= b*
        // l1_out` は `ws.check_batch_capacity(b)` で保証、weight grad `l1_w_grad.len() ==
        // l1_in*l1_out` は固定 shape)、stream 共有 + cast 妥当性は bwd_input と同じ
        // 前提 (256 byte aligned cuMemAlloc 由来、`self` 借用と同 lifetime)。
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                l1_in_u32 as i32,
                l1_out_u32 as i32,
                b_u32 as i32,
                self.ws.combined.cu_deviceptr() as *const f32,
                self.ws.dl1_pre.cu_deviceptr() as *const f32,
                self.l1_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        // l2_out と同じく out_dim > 256 では generic `bias_grad` に fall back する。
        if l1_out_u32 <= DENSE_BIAS_GRAD_MAX_OUT {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: dense_bias_grad_tiled, stream: self.stream, module: self.module,
                    config: cfg_dense_bias_grad(self.dense_bias_grad_occ, b_u32, l1_out_u32),
                    args: [slice(self.ws.dl1_pre), slice(self.l1_b_grad), b_u32, l1_out_u32]
                }
            }?;
        } else {
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: bias_grad, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_n),
                    args: [slice(self.ws.dl1_pre), slice(self.l1_b_grad), b_u32, l1_out_u32]
                }
            }?;
        }
        tick("L1_dense", &self.stream, &mut prof_t0)?;

        // ---- Concat inverse + activation grad の融合 ----
        // dcombined (b × combined_dim) の per-perspective 半分を `col_offset` で直接 offset
        // 読みし、pre-activation `ft_*_out` で gate した値を `dft_*_out` に書く融合 kernel。
        // 中間 buffer への取り出し (`slice_extract_2d`) を介さず `dcombined` を直接 offset
        // 読みすることで、b × ft_out × 4 byte の DRAM round-trip を消す。`ft_fp16_out`
        // 経路は f16 dft buffer + loss scaling + clamp + f16 cast を含む。CReLU / SCReLU は
        // `simple_act_grad_to_fp16_*_with_scale`、Pairwise は `ft_post_perspective_grad_fp16`
        // がいずれも `dcombined` を `combined_stride` / `col_offset` で直接読む。
        if self.ft_fp16_out {
            let dft_scale = FT_DFT_FP16_BASE_SCALE * (b as f32);
            // CReLU / SCReLU は nstm を `col_offset = ft_out` で dcombined から直接読むため
            // `2*ft_out <= combined_dim` (= dcombined row stride) が成立しないと OOB read に
            // なる。CReLU/SCReLU は両 perspective 連結で combined_dim = 2*ft_out なので常に成立。
            // Pairwise は combined_dim = ft_out で別 kernel を使うため対象外 (matches! で除外)。
            debug_assert!(
                !matches!(
                    self.id.activation,
                    SimpleActivation::CReLU | SimpleActivation::SCReLU
                ) || 2 * ft_out_u32 <= l1_in_u32,
                "fp16-out CReLU/SCReLU grad needs 2*ft_out <= combined_dim (got 2*{ft_out_u32} vs {l1_in_u32})"
            );
            match self.id.activation {
                SimpleActivation::CReLU => {
                    let ft_stm_out_h = self
                        .ws
                        .ft_stm_out_h
                        .as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
                    let mut dft_stm_out_h = self
                        .ws
                        .dft_stm_out_h
                        .as_mut()
                        .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_act_grad_to_fp16_crelu_with_scale,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_stm_out_h), slice(self.ft_b),
                                slice(self.ws.dcombined), l1_in_u32, 0_u32, slice_mut(dft_stm_out_h),
                                slice(self.fp16_clamp_counter),
                                b_u32, ft_out_u32, dft_scale
                            ]
                        }
                    }?;
                    let ft_nstm_out_h = self
                        .ws
                        .ft_nstm_out_h
                        .as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
                    let mut dft_nstm_out_h = self
                        .ws
                        .dft_nstm_out_h
                        .as_mut()
                        .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_act_grad_to_fp16_crelu_with_scale,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_nstm_out_h), slice(self.ft_b),
                                slice(self.ws.dcombined), l1_in_u32, ft_out_u32, slice_mut(dft_nstm_out_h),
                                slice(self.fp16_clamp_counter),
                                b_u32, ft_out_u32, dft_scale
                            ]
                        }
                    }?;
                    // stm + nstm の 2 launch × ft_n thread × 1 elem/thread = 2 * ft_n。
                    // `ft_n = b * ft_out_u32` (= b * ft_dim) なので 2 * b * ft_dim 要素/step。
                    self.fp16_clamp_elems_written = self
                        .fp16_clamp_elems_written
                        .saturating_add(2_u64 * ft_n as u64);
                }
                SimpleActivation::SCReLU => {
                    let ft_stm_out_h = self
                        .ws
                        .ft_stm_out_h
                        .as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
                    let mut dft_stm_out_h = self
                        .ws
                        .dft_stm_out_h
                        .as_mut()
                        .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_act_grad_to_fp16_screlu_with_scale,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_stm_out_h), slice(self.ft_b),
                                slice(self.ws.dcombined), l1_in_u32, 0_u32, slice_mut(dft_stm_out_h),
                                slice(self.fp16_clamp_counter),
                                b_u32, ft_out_u32, dft_scale
                            ]
                        }
                    }?;
                    let ft_nstm_out_h = self
                        .ws
                        .ft_nstm_out_h
                        .as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
                    let mut dft_nstm_out_h = self
                        .ws
                        .dft_nstm_out_h
                        .as_mut()
                        .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_act_grad_to_fp16_screlu_with_scale,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [
                                slice(ft_nstm_out_h), slice(self.ft_b),
                                slice(self.ws.dcombined), l1_in_u32, ft_out_u32, slice_mut(dft_nstm_out_h),
                                slice(self.fp16_clamp_counter),
                                b_u32, ft_out_u32, dft_scale
                            ]
                        }
                    }?;
                    // see CReLU branch: stm+nstm 2 launch × ft_n elems = 2 * ft_n。
                    self.fp16_clamp_elems_written = self
                        .fp16_clamp_elems_written
                        .saturating_add(2_u64 * ft_n as u64);
                }
                SimpleActivation::Pairwise => {
                    // f16 入力版 pairwise FT post backward。`ft_post_perspective_grad` の
                    // f16 版で、bias 未加算の f16 `ft_*_out_h` を読み、loss scaling 済
                    // f16 `dft_*_out_h` を書く。FP32 pairwise と同じく `dft_*_out` の
                    // 書き込みと同 pass で `ft_b_grad` (f32) を accumulate するため、
                    // 後段の `simple_bias_grad_dual_fp16` は Pairwise では呼ばない。
                    let ft_stm_out_h = self
                        .ws
                        .ft_stm_out_h
                        .as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled");
                    let mut dft_stm_out_h = self
                        .ws
                        .dft_stm_out_h
                        .as_mut()
                        .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: ft_post_perspective_grad_fp16,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n / 2),
                            args: [slice(self.ws.dcombined), slice(ft_stm_out_h),
                                   slice(self.ft_b), slice_mut(dft_stm_out_h),
                                   slice(self.ft_b_grad), slice(self.fp16_clamp_counter),
                                   b_u32, ft_out_u32,
                                   0_u32, l1_in_u32, FT_POST_SCALE, dft_scale]
                        }
                    }?;
                    let ft_nstm_out_h = self
                        .ws
                        .ft_nstm_out_h
                        .as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled");
                    let mut dft_nstm_out_h = self
                        .ws
                        .dft_nstm_out_h
                        .as_mut()
                        .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: ft_post_perspective_grad_fp16,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n / 2),
                            args: [slice(self.ws.dcombined), slice(ft_nstm_out_h),
                                   slice(self.ft_b), slice_mut(dft_nstm_out_h),
                                   slice(self.ft_b_grad), slice(self.fp16_clamp_counter),
                                   b_u32, ft_out_u32,
                                   ft_out_u32 / 2, l1_in_u32, FT_POST_SCALE, dft_scale]
                        }
                    }?;
                    // Pairwise: stm+nstm 2 launch × (ft_n/2) thread × 2 elem/thread = 2 * ft_n。
                    self.fp16_clamp_elems_written = self
                        .fp16_clamp_elems_written
                        .saturating_add(2_u64 * ft_n as u64);
                }
            }
        } else {
            match self.id.activation {
                SimpleActivation::CReLU => {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bwd_ft_act_crelu_fused,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [slice(self.ws.dcombined), slice(self.ws.ft_stm_out),
                                   slice(self.ws.dft_stm_out), b_u32, ft_out_u32, 0_u32]
                        }
                    }?;
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bwd_ft_act_crelu_fused,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [slice(self.ws.dcombined), slice(self.ws.ft_nstm_out),
                                   slice(self.ws.dft_nstm_out), b_u32, ft_out_u32, ft_out_u32]
                        }
                    }?;
                }
                SimpleActivation::SCReLU => {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bwd_ft_act_screlu_fused,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [slice(self.ws.dcombined), slice(self.ws.ft_stm_out),
                                   slice(self.ws.dft_stm_out), b_u32, ft_out_u32, 0_u32]
                        }
                    }?;
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bwd_ft_act_screlu_fused,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n),
                            args: [slice(self.ws.dcombined), slice(self.ws.ft_nstm_out),
                                   slice(self.ws.dft_nstm_out), b_u32, ft_out_u32, ft_out_u32]
                        }
                    }?;
                }
                SimpleActivation::Pairwise => {
                    // Pairwise FT post backward: `ft_post_perspective_grad` を stm /
                    // nstm 各 1 回 launch する。1 thread = 1 pair (= dft の 2 cell を
                    // 書く)、`dcombined` の per-perspective 半分を `d_combined_offset`
                    // で切り出し、bias 未加算の `ft_*_out` に bias を再加算して CReLU
                    // 指示関数 + pairwise 乗算の勾配を作る。この kernel は `dft_*_out`
                    // を書くと同時に `ft_b_grad` へ bias 勾配を atomic accumulate する
                    // ため、後段の `simple_bias_grad_dual` は Pairwise では呼ばない。
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: ft_post_perspective_grad,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n / 2),
                            args: [slice(self.ws.dcombined), slice(self.ws.ft_stm_out),
                                   slice(self.ft_b), slice_mut(self.ws.dft_stm_out),
                                   slice(self.ft_b_grad), b_u32, ft_out_u32,
                                   0_u32, l1_in_u32, FT_POST_SCALE]
                        }
                    }?;
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: ft_post_perspective_grad,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_n / 2),
                            args: [slice(self.ws.dcombined), slice(self.ws.ft_nstm_out),
                                   slice(self.ft_b), slice_mut(self.ws.dft_nstm_out),
                                   slice(self.ft_b_grad), b_u32, ft_out_u32,
                                   ft_out_u32 / 2, l1_in_u32, FT_POST_SCALE]
                        }
                    }?;
                }
            }
        }
        tick("bwd_ft_act", &self.stream, &mut prof_t0)?;

        // ---- FT bias grad + FT weight grad: stm/nstm の両 perspective が同じ ft_b / ft_w
        // を共有するため atomic accumulate (host が呼出前に 0 初期化)。
        // `ft_fp16_out` 時は f16 dft buffer を read、`dft_inv_scale` で loss scaling を打ち消す。
        // FT bias grad は両 perspective が同じ ft_b を共有するため atomic accumulate。
        // host が呼出前に `ft_b_grad` を 0 reset 済 (本関数冒頭の memset_zero ブロック)。
        let dft_inv_scale_fp16 = if self.ft_fp16_out {
            1.0_f32 / (FT_DFT_FP16_BASE_SCALE * (b as f32))
        } else {
            1.0_f32 // unused on FP32 path
        };
        // FT bias grad: per-output tile reduction。thread `oi` が出力 oi を専有し、自 block の
        // `items` positions を register 累積してから `ft_b_grad[oi]` へ atomic 1 回
        // (`simple_bias_grad_dual[_fp16]`)。global atomic contention は ceil(B/items) * ft_dim で、
        // 1 thread 1 cell が直接 atomic add する素朴版 (B * ft_dim) より少ない。Pairwise は
        // `ft_post_perspective_grad[_fp16]` が
        // `dft_*_out` 書き込みと同 pass で同値を `ft_b_grad` へ accumulate 済 (FT bias grad =
        // pre-activation 勾配 dft の batch 和) のため、別 launch しない。
        // `block_dim = min(ft_out, 1024)`・`grid.y = ceil(ft_out / block_dim)` の 2D grid で
        // thread→output を対応付ける (grid.x = position tile、grid.y = output tile)。output を y
        // タイルに割るので ft_out が CUDA の block 上限 1024 を超えても起動できる。
        let bias_grad_items = 64_u32;
        let bias_grad_blocks = b_u32.div_ceil(bias_grad_items);
        let bias_grad_block_dim = ft_out_u32.min(1024);
        let bias_grad_out_tiles = ft_out_u32.div_ceil(bias_grad_block_dim);
        match self.id.activation {
            SimpleActivation::Pairwise => {}
            SimpleActivation::CReLU | SimpleActivation::SCReLU => {
                if self.ft_fp16_out {
                    let dft_stm_out_h = self
                        .ws
                        .dft_stm_out_h
                        .as_ref()
                        .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
                    let dft_nstm_out_h = self
                        .ws
                        .dft_nstm_out_h
                        .as_ref()
                        .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bias_grad_dual_fp16, stream: self.stream, module: self.module,
                            config: LaunchConfig {
                                grid_dim: (bias_grad_blocks, bias_grad_out_tiles, 1),
                                block_dim: (bias_grad_block_dim, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            args: [
                                slice(dft_stm_out_h),
                                slice(dft_nstm_out_h),
                                slice(self.ft_b_grad),
                                b_u32, ft_out_u32, dft_inv_scale_fp16, bias_grad_items
                            ]
                        }
                    }?;
                } else {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: simple_bias_grad_dual, stream: self.stream, module: self.module,
                            config: LaunchConfig {
                                grid_dim: (bias_grad_blocks, bias_grad_out_tiles, 1),
                                block_dim: (bias_grad_block_dim, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            args: [
                                slice(self.ws.dft_stm_out),
                                slice(self.ws.dft_nstm_out),
                                slice(self.ft_b_grad),
                                b_u32, ft_out_u32, bias_grad_items
                            ]
                        }
                    }?;
                }
            }
        }

        // FT weight grad — **inverse-index pipeline** で per-feature gather に変換する経路。
        // 各 perspective につき (A) `build_feature_counts` で histogram、(B) multi-block
        // exclusive prefix sum で offset、(C) `scatter_positions` で sorted position 列を
        // 構築し、(D) `gather_and_sum_per_feature_overwrite` (1 回目 = stm) /
        // `gather_and_sum_per_feature_add` (2 回目 = nstm) が `(feature, ri)` cell ごとに
        // sum を書く。FP16 path は同 pipeline で `_fp16` 変種に dft_inv_scale を渡す。
        // stm の overwrite が `ft_w_grad` の全 `(feature, ri)` cell を unconditionally 書き切るため、
        // 本関数冒頭の memset_zero 群に `ft_w_grad` は含めない (LayerStack の同 pipeline と同規約)。
        let gather_config = LaunchConfig {
            grid_dim: (ft_in_u32, self.id.ft_out.div_ceil(128) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        for (iter_idx, idx_dev) in [&self.ws.stm_idx_dev, &self.ws.nstm_idx_dev]
            .into_iter()
            .enumerate()
        {
            // A1: feat_counts / feat_write_ctr を 0 にリセット (atomic build / scatter の前提)。
            memset_zero(&self.stream, &self.ws.feat_counts)?;
            memset_zero(&self.stream, &self.ws.feat_write_ctr)?;
            // A2: 各 (b, ni) sparse index について feat_counts[feature] を atomic increment。
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: build_feature_counts,
                    stream: self.stream, module: self.module,
                    config: cfg_1d(b * self.ws.max_active),
                    args: [
                        slice(idx_dev),
                        slice(self.ws.nnz_dev),
                        slice(self.ws.feat_counts),
                        b_u32, max_active_u32, ft_in_u32
                    ]
                }
            }?;
            // B: feat_counts の exclusive prefix sum (multi-block scan で全 SM を使う)。
            // 単一 block scan は 1 SM 律速 (`ft_in` ~73K-138K で大半 idle) のため 3 段に分割。
            let prefix_blocks = ft_in_u32.div_ceil(1024);
            // level 1: 各 block が連続 1024 要素を block-local scan、block 総和を emit。
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: prefix_sum_block_local,
                    stream: self.stream, module: self.module,
                    config: LaunchConfig {
                        grid_dim: (prefix_blocks, 1, 1),
                        block_dim: (1024, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    args: [
                        slice(self.ws.feat_counts),
                        slice(self.ws.feat_offsets),
                        slice(self.ws.feat_block_sums),
                        ft_in_u32
                    ]
                }
            }?;
            // level 2: block 総和列 (prefix_blocks ≲ 135 要素) を単一 block で scan。
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: exclusive_prefix_sum_small,
                    stream: self.stream, module: self.module,
                    config: LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1024, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    args: [
                        slice(self.ws.feat_block_sums),
                        slice(self.ws.feat_block_offsets),
                        prefix_blocks
                    ]
                }
            }?;
            // level 3: block-local offsets へ block offset を加算 + offsets[n]=total。
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: prefix_sum_add_block_offset,
                    stream: self.stream, module: self.module,
                    config: LaunchConfig {
                        grid_dim: (prefix_blocks, 1, 1),
                        block_dim: (1024, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    args: [
                        slice(self.ws.feat_offsets),
                        slice(self.ws.feat_block_offsets),
                        ft_in_u32,
                        prefix_blocks
                    ]
                }
            }?;
            // C: 各 (b, ni) sparse index について feat_positions の per-feature slot に
            // batch position `b` を書き込む (`feat_write_ctr[feature]++` で位置決定)。
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: scatter_positions,
                    stream: self.stream, module: self.module,
                    config: cfg_1d(b * self.ws.max_active),
                    args: [
                        slice(idx_dev),
                        slice(self.ws.nnz_dev),
                        slice(self.ws.feat_offsets),
                        slice(self.ws.feat_write_ctr),
                        slice(self.ws.feat_positions),
                        b_u32, max_active_u32, ft_in_u32
                    ]
                }
            }?;
            // D: 各 (feature, ri) cell について feat_positions[feat_offsets[f]..feat_offsets[f+1]]
            // を順 read して accumulate。iter 0 = stm は overwrite (全 cell 書き切り)、iter 1 =
            // nstm は atomic add で stm 結果に重ねる。FP16 path は `dft_inv_scale` で loss scaling
            // を打ち消しながら read。
            if self.ft_fp16_out {
                let dft_stm_out_h = self
                    .ws
                    .dft_stm_out_h
                    .as_ref()
                    .expect("dft_stm_out_h is Some when ft_fp16_out is enabled");
                let dft_nstm_out_h = self
                    .ws
                    .dft_nstm_out_h
                    .as_ref()
                    .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled");
                let dft_h = if iter_idx == 0 {
                    dft_stm_out_h
                } else {
                    dft_nstm_out_h
                };
                if iter_idx == 0 {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: gather_and_sum_per_feature_overwrite_fp16,
                            stream: self.stream, module: self.module, config: gather_config,
                            args: [
                                slice(dft_h),
                                slice(self.ws.feat_positions),
                                slice(self.ws.feat_offsets),
                                slice(self.ft_w_grad),
                                ft_in_u32, ft_out_u32, dft_inv_scale_fp16
                            ]
                        }
                    }?;
                } else {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: gather_and_sum_per_feature_add_fp16,
                            stream: self.stream, module: self.module, config: gather_config,
                            args: [
                                slice(dft_h),
                                slice(self.ws.feat_positions),
                                slice(self.ws.feat_offsets),
                                slice(self.ft_w_grad),
                                ft_in_u32, ft_out_u32, dft_inv_scale_fp16
                            ]
                        }
                    }?;
                }
            } else {
                let dft = if iter_idx == 0 {
                    &self.ws.dft_stm_out
                } else {
                    &self.ws.dft_nstm_out
                };
                if iter_idx == 0 {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: gather_and_sum_per_feature_overwrite,
                            stream: self.stream, module: self.module, config: gather_config,
                            args: [
                                slice(dft),
                                slice(self.ws.feat_positions),
                                slice(self.ws.feat_offsets),
                                slice(self.ft_w_grad),
                                ft_in_u32, ft_out_u32
                            ]
                        }
                    }?;
                } else {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: gather_and_sum_per_feature_add,
                            stream: self.stream, module: self.module, config: gather_config,
                            args: [
                                slice(dft),
                                slice(self.ws.feat_positions),
                                slice(self.ws.feat_offsets),
                                slice(self.ft_w_grad),
                                ft_in_u32, ft_out_u32
                            ]
                        }
                    }?;
                }
            }
        }
        // factorizer: piece-input 仮想行の grad を対応 base 実行の grad 和で埋める。
        // 上の gather は実 block (`[0, ft_in)`) のみ書くので、仮想 block は reduce が
        // overwrite で書き切る。optimizer は train 形状 master の全行を更新する。
        if self.id.feature_set.ft_factorize() {
            ft_factorize_host::launch_ft_reduce(
                &self.stream,
                &self.module,
                &self.id.feature_set,
                self.id.ft_out,
                &self.ft_w_grad,
                &self.threat_pair_starts,
            )?;
        }
        tick("bwd_ft_bw", &self.stream, &mut prof_t0)?;

        Ok(())
    }

    /// Ranger optimizer step (RAdam + Lookahead) を 8 weight group に走らせる。
    /// `RANGER_K` の倍数 step では lookahead lerp を続けて走らせ slow weight と
    /// master weight を補間する。`run_backward_kernels` の直後に呼ぶ。
    pub(crate) fn run_optimizer_step(&mut self, lr: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.step_count += 1;
        let (step_size, denom) =
            radam_compute_step_size_denom(self.step_count, BETA1, BETA2, N_SMA_THRESHOLD);

        let ft_out = self.id.ft_out;
        let l1_out = self.id.l1_out;
        let l2_out = self.id.l2_out;
        // factorizer 有効時は FT master が train 形状 (実行 + piece-input 仮想行)。仮想行も
        // radam で更新し、forward comb は step 末の fold が master から再生成する。
        let ft_factorize = self.id.feature_set.ft_factorize();
        let ft_w_n = (self.id.feature_set.train_ft_in() * ft_out) as u32;
        let ft_b_n = ft_out as u32;
        let l1_w_n = (self.id.combined_dim() * l1_out) as u32;
        let l1_b_n = l1_out as u32;
        let l2_w_n = (l1_out * l2_out) as u32;
        let l2_b_n = l2_out as u32;
        let l3_w_n = l2_out as u32;
        let l3_b_n = 1_u32;

        // ft_w optimizer: 2 つの opt-in flag で 4 通りに分岐する (LayerStack と同じパターン)。
        //  - `--ft-fp16`: FP16 mirror (`ft_w_h`) 同時更新版 (`*_mirror`) を使い、forward 用
        //    mirror を別 cast kernel 無しで同期する。
        //  - `--fp16-opt-state`: m / v を `f16` で読み書きする `*_f16state` 系を使う (DRAM
        //    traffic 半減、`FT_OPT_M_SCALE` / `FT_OPT_V_SCALE` で scale 付き格納)。
        // 他 7 group は moment が小さく `f16` 化の意味が無いので常に `radam_step`。
        match (&mut self.ft_w_m, &mut self.ft_w_v) {
            (MomentBuf::F16(ft_w_m), MomentBuf::F16(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut().filter(|_| !ft_factorize) {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: radam_step_f16state_mirror,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_w_n as usize),
                            args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                                   slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                                   self.weight_decay, BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX,
                                   FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n]
                        }
                    }?;
                } else {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: radam_step_f16state,
                            stream: self.stream, module: self.module, config: cfg_1d(ft_w_n as usize),
                            args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                                   slice_mut(self.ft_w_grad), lr, step_size, denom,
                                   self.weight_decay, BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX,
                                   FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n]
                        }
                    }?;
                }
            }
            (MomentBuf::F32(ft_w_m), MomentBuf::F32(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut().filter(|_| !ft_factorize) {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: radam_step_fp16_mirror, stream: self.stream, module: self.module,
                            config: cfg_1d(ft_w_n as usize),
                            args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                                   slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                                   self.weight_decay, BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, ft_w_n]
                        }
                    }?;
                } else {
                    unsafe {
                        // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                        // stream の完了を待つ同期点まで生存する device allocation。
                        cuda_launch! {
                            kernel: radam_step, stream: self.stream, module: self.module,
                            config: cfg_1d(ft_w_n as usize),
                            args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                                   slice_mut(self.ft_w_grad), lr, step_size, denom, self.weight_decay,
                                   BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, ft_w_n]
                        }
                    }?;
                }
            }
            _ => unreachable!("ft_w m and v moment buffers always share precision"),
        }

        // 残り 7 group (ft_b, l1_w/b, l2_w/b, l3_w/b) は moment buffer 縮小余地が無いので
        // 常に FP32 master + `radam_step`。FT 以外の `f16` 化は本フラグの範囲外。
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(ft_b_n as usize),
                args: [slice_mut(self.ft_b), slice_mut(self.ft_b_m), slice_mut(self.ft_b_v),
                       slice_mut(self.ft_b_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, ft_b_n]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l1_w_n as usize),
                args: [slice_mut(self.l1_w), slice_mut(self.l1_w_m), slice_mut(self.l1_w_v),
                       slice_mut(self.l1_w_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, l1_w_n]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l1_b_n as usize),
                args: [slice_mut(self.l1_b), slice_mut(self.l1_b_m), slice_mut(self.l1_b_v),
                       slice_mut(self.l1_b_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, l1_b_n]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l2_w_n as usize),
                args: [slice_mut(self.l2_w), slice_mut(self.l2_w_m), slice_mut(self.l2_w_v),
                       slice_mut(self.l2_w_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, l2_w_n]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l2_b_n as usize),
                args: [slice_mut(self.l2_b), slice_mut(self.l2_b_m), slice_mut(self.l2_b_v),
                       slice_mut(self.l2_b_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, l2_b_n]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l3_w_n as usize),
                args: [slice_mut(self.l3_w), slice_mut(self.l3_w_m), slice_mut(self.l3_w_v),
                       slice_mut(self.l3_w_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_QUANT_MIN, W_CLAMP_QUANT_MAX, l3_w_n]
            }
        }?;
        unsafe {
            // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
            // stream の完了を待つ同期点まで生存する device allocation。
            cuda_launch! {
                kernel: radam_step, stream: self.stream, module: self.module, config: cfg_1d(l3_b_n as usize),
                args: [slice_mut(self.l3_b), slice_mut(self.l3_b_m), slice_mut(self.l3_b_v),
                       slice_mut(self.l3_b_grad), lr, step_size, denom, self.weight_decay,
                       BETA1, BETA2, EPS, W_CLAMP_NONE_MIN, W_CLAMP_NONE_MAX, l3_b_n]
            }
        }?;

        if self.step_count.is_multiple_of(RANGER_K) {
            // ft_w lookahead lerp: lerp は radam の後に ft_w を再度書き換えるので、
            // `ft_fp16` 時は mirror 同時更新版で `ft_w_h` を lerp 後の最終値に同期する。
            // factorizer 有効時は mirror variant を使わず (comb は base 形状で train 形状の
            // lerp からは書けない)、master のみ lerp し step 末の fold が comb を再生成する。
            if let Some(mut ft_w_h) = self.ft_w_h.as_mut().filter(|_| !ft_factorize) {
                unsafe {
                    // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                    // stream の完了を待つ同期点まで生存する device allocation。
                    cuda_launch! {
                        kernel: ranger_lookahead_lerp_fp16_mirror, stream: self.stream, module: self.module,
                        config: cfg_1d(ft_w_n as usize),
                        args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), slice_mut(ft_w_h),
                               RANGER_ALPHA, ft_w_n]
                    }
                }?;
            } else {
                unsafe {
                    // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                    // stream の完了を待つ同期点まで生存する device allocation。
                    cuda_launch! {
                        kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                        config: cfg_1d(ft_w_n as usize),
                        args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), RANGER_ALPHA, ft_w_n]
                    }
                }?;
            }
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(ft_b_n as usize),
                    args: [slice_mut(self.ft_b), slice_mut(self.ft_b_slow), RANGER_ALPHA, ft_b_n]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_w_n as usize),
                    args: [slice_mut(self.l1_w), slice_mut(self.l1_w_slow), RANGER_ALPHA, l1_w_n]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(l1_b_n as usize),
                    args: [slice_mut(self.l1_b), slice_mut(self.l1_b_slow), RANGER_ALPHA, l1_b_n]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_w_n as usize),
                    args: [slice_mut(self.l2_w), slice_mut(self.l2_w_slow), RANGER_ALPHA, l2_w_n]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(l2_b_n as usize),
                    args: [slice_mut(self.l2_b), slice_mut(self.l2_b_slow), RANGER_ALPHA, l2_b_n]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(l3_w_n as usize),
                    args: [slice_mut(self.l3_w), slice_mut(self.l3_w_slow), RANGER_ALPHA, l3_w_n]
                }
            }?;
            unsafe {
                // SAFETY: kernel signature と args の個数・順序・型は一致し、渡す buffer は
                // stream の完了を待つ同期点まで生存する device allocation。
                cuda_launch! {
                    kernel: ranger_lookahead_lerp, stream: self.stream, module: self.module,
                    config: cfg_1d(l3_b_n as usize),
                    args: [slice_mut(self.l3_b), slice_mut(self.l3_b_slow), RANGER_ALPHA, l3_b_n]
                }
            }?;
        }
        // factorizer: master (`ft_w`) の本 step の全更新 (radam / lookahead) が確定した後、
        // 次 step の forward が読む comb を再生成する。
        if ft_factorize {
            self.launch_ft_fold()?;
        }
        Ok(())
    }

    /// 現在の device 上の f32 weight を `SimpleWeights` (host 側 f32 row-major) に
    /// 書き出す。
    ///
    /// 重み layout の対応:
    /// - `ft_w` / `ft_b` / `l3_w` / `l3_b` : device と `SimpleWeights` で同 layout (転置不要)。
    /// - `l1_w` : device は `[in=combined_dim, out=l1_out]` (`l1_w[in*l1_out + out]`、
    ///   `dense_mm_fwd` の weight pattern)、`SimpleWeights` は `[out=l1_out, in=combined_dim]`
    ///   行優先 (`l1_w[out*combined_dim + in]`、`save_quantised` の i8 量子化が前提とする
    ///   out-major 並び) → host 側で in-major → out-major に転置する。
    /// - `l2_w` : 同パターンの転置 (`[l1_out, l2_out]` → `[l2_out, l1_out]`)。
    pub(crate) fn to_simple_weights(&self) -> Result<SimpleWeights, Box<dyn std::error::Error>> {
        let id = self.id;
        let l1_out = id.l1_out;
        let l2_out = id.l2_out;
        let l1_in = id.combined_dim();

        let ft_w = self.ft_w.to_host_vec(&self.stream)?;
        // factorizer 有効時は piece-input 仮想行を実行へ畳み込み base 形状で返す
        // (量子化・飽和検査は畳み込み後の値に掛かる)。export id は factorizer modifier を
        // 外した base spec にする (量子化 .bin は仮想行を持たず shape は非 factorize と同形)。
        let (export_id, ft_w) = if self.id.feature_set.ft_factorize() {
            let folded = nnue_format::layerstack_weights::coalesce_ft_factorized(
                &self.id.feature_set,
                self.id.ft_out,
                &ft_w,
            );
            let base_id = SimpleId {
                feature_set: self.id.feature_set.feature_set().spec(),
                ..self.id
            };
            (base_id, folded)
        } else {
            (id, ft_w)
        };
        let ft_b = self.ft_b.to_host_vec(&self.stream)?;
        let l1_w_in_major = self.l1_w.to_host_vec(&self.stream)?;
        let l1_b = self.l1_b.to_host_vec(&self.stream)?;
        let l2_w_in_major = self.l2_w.to_host_vec(&self.stream)?;
        let l2_b = self.l2_b.to_host_vec(&self.stream)?;
        let l3_w = self.l3_w.to_host_vec(&self.stream)?;
        let l3_b = self.l3_b.to_host_vec(&self.stream)?;

        let mut l1_w = vec![0.0_f32; l1_out * l1_in];
        for out in 0..l1_out {
            for inp in 0..l1_in {
                l1_w[out * l1_in + inp] = l1_w_in_major[inp * l1_out + out];
            }
        }
        let mut l2_w = vec![0.0_f32; l2_out * l1_out];
        for out in 0..l2_out {
            for inp in 0..l1_out {
                l2_w[out * l1_out + inp] = l2_w_in_major[inp * l2_out + out];
            }
        }

        Ok(SimpleWeights {
            id: export_id,
            fv_scale: self.fv_scale,
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
        })
    }

    /// `SimpleWeights` を device 上に upload して現在の重み・lookahead slow を置き換える。
    /// `m` / `v` / `grad` は 0 リセット、`step_count` を 0 に戻す (Ranger を最初から
    /// やり直すのと等価)。`load_simple_weights` 後の slow weight は upload した weight
    /// と同値 (lookahead が `w == slow` 状態から始まる、`new` と同じ規約)。
    ///
    /// `w.id` が現トレーナの `id` と一致しなければ early reject する (異 topology /
    /// feature set の weight は受け入れない)。`fv_scale` は受け入れた値で上書きする。
    ///
    /// device buffer の layout 変換は [`Self::to_simple_weights`] と逆向きで、L1/L2 weight
    /// は host 側で out-major → in-major に転置してから upload する。
    pub(crate) fn load_simple_weights(
        &mut self,
        w: &SimpleWeights,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if w.id != self.id {
            return Err(format!(
                "SimpleGpuTrainer::load_simple_weights: id mismatch \
                 (trainer ft_in={}, ft_out={}, l1_out={}, l2_out={}, activation={}, \
                 feature_set={}; weights ft_in={}, ft_out={}, l1_out={}, l2_out={}, \
                 activation={}, feature_set={})",
                self.id.ft_in(),
                self.id.ft_out,
                self.id.l1_out,
                self.id.l2_out,
                self.id.activation.canonical_name(),
                self.id.feature_set.canonical_name(),
                w.id.ft_in(),
                w.id.ft_out,
                w.id.l1_out,
                w.id.l2_out,
                w.id.activation.canonical_name(),
                w.id.feature_set.canonical_name(),
            )
            .into());
        }
        let l1_out = self.id.l1_out;
        let l2_out = self.id.l2_out;
        let l1_in = self.id.combined_dim();

        let mut l1_w_in_major = vec![0.0_f32; l1_in * l1_out];
        for out in 0..l1_out {
            for inp in 0..l1_in {
                l1_w_in_major[inp * l1_out + out] = w.l1_w[out * l1_in + inp];
            }
        }
        let mut l2_w_in_major = vec![0.0_f32; l1_out * l2_out];
        for out in 0..l2_out {
            for inp in 0..l1_out {
                l2_w_in_major[inp * l2_out + out] = w.l2_w[out * l1_out + inp];
            }
        }

        // weight 本体 (master と slow の両方を同値で初期化)。
        self.ft_w = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_b = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.ft_b_slow = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.l1_w = DeviceBuffer::from_host(&self.stream, &l1_w_in_major)?;
        self.l1_w_slow = DeviceBuffer::from_host(&self.stream, &l1_w_in_major)?;
        self.l1_b = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l2_w = DeviceBuffer::from_host(&self.stream, &l2_w_in_major)?;
        self.l2_w_slow = DeviceBuffer::from_host(&self.stream, &l2_w_in_major)?;
        self.l2_b = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l2_b_slow = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l3_w = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_w_slow = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_b = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;
        self.l3_b_slow = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;

        // m / v / grad を 0 リセット、step_count を 0 に戻す (Ranger を最初から)。
        // ft_w の m / v は [`MomentBuf`] で `--fp16-opt-state` 精度を保つため `zeroed`
        // で作り直す (`memset_zero` が `MomentBuf` を取らないため)。長さは `ft_w` と同じ
        // train 形状 (factorizer 有効時は仮想行込み。`--init-from` では factorizer が
        // auto-suppress されるので通常 base と一致する)。
        let ft_w_n = self.id.feature_set.train_ft_in() * self.id.ft_out;
        self.ft_w_m = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        self.ft_w_v = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        for buf in [
            &self.ft_w_grad,
            &self.ft_b_m,
            &self.ft_b_v,
            &self.ft_b_grad,
            &self.l1_w_m,
            &self.l1_w_v,
            &self.l1_w_grad,
            &self.l1_b_m,
            &self.l1_b_v,
            &self.l1_b_grad,
            &self.l2_w_m,
            &self.l2_w_v,
            &self.l2_w_grad,
            &self.l2_b_m,
            &self.l2_b_v,
            &self.l2_b_grad,
            &self.l3_w_m,
            &self.l3_w_v,
            &self.l3_w_grad,
            &self.l3_b_m,
            &self.l3_b_v,
            &self.l3_b_grad,
        ] {
            memset_zero(&self.stream, buf)?;
        }
        self.step_count = 0;
        self.fv_scale = w.fv_scale;
        Ok(())
    }

    /// resume 用 raw f32 checkpoint を `path` に atomic に書き出す (LayerStack の
    /// [`GpuTrainer::save_raw_checkpoint`] と同 format / 同方針)。
    ///
    /// file layout と atomic 書き出しは [`save_raw_checkpoint_file`] が担い、本 method
    /// は arch identity (`topology = [ft_out, l1_out, l2_out]`) と group 列
    /// ([`Self::raw_ckpt_group_sources`]、8 group) を組んで渡すだけ。L1/L2 weight は
    /// device-native `[in, out]` 並びそのまま書く (resume 互換性は device 上の layout
    /// で完結する)。
    pub(crate) fn save_raw_checkpoint(
        &self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
        lr_horizon: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let topology: [u64; 3] = [
            self.id.ft_out as u64,
            self.id.l1_out as u64,
            self.id.l2_out as u64,
        ];
        save_raw_checkpoint_file(
            path,
            &self.stream,
            &RawCkptArch {
                feature_set: self.id.feature_set,
                arch_kind: ArchKind::Simple,
                ft_out: self.id.ft_out as u64,
                topology: &topology,
            },
            &RawCkptMeta {
                run_id,
                superbatch,
                step_count: self.step_count,
                lr_horizon,
            },
            &self.raw_ckpt_group_sources(),
        )
    }

    /// `--resume` 用に raw f32 checkpoint を読み戻す。返り値は完了 `(superbatch,
    /// producer run id, LR-schedule horizon)` で、caller は通常 `superbatch + 1`
    /// から resume する。horizon は version 5+ で保存されていれば `Some`。
    ///
    /// header (`arch_kind=Simple`, `topology=[ft_out, l1_out, l2_out]`, feature set)
    /// と group 本体の読み出し・照合は [`load_raw_checkpoint_file`] が担当する。
    /// 本 method は 8 group 各 `(w, m, v, slow)` を device へ upload し直し、
    /// `step_count` を復元する。`grad` は触らない (step ごとに memset される)。
    pub(crate) fn load_raw_checkpoint(
        &mut self,
        path: &Path,
    ) -> Result<RawCkptResumeState, Box<dyn std::error::Error>> {
        let topology: [u64; 3] = [
            self.id.ft_out as u64,
            self.id.l1_out as u64,
            self.id.l2_out as u64,
        ];
        let expected_groups: Vec<(&'static str, usize)> = self
            .raw_ckpt_group_sources()
            .iter()
            .map(|g| (g.name, g.len))
            .collect();
        let (header, loaded) = load_raw_checkpoint_file(
            path,
            &RawCkptArch {
                feature_set: self.id.feature_set,
                arch_kind: ArchKind::Simple,
                ft_out: self.id.ft_out as u64,
                topology: &topology,
            },
            &expected_groups,
        )?;

        // host → device upload (`loaded` の順序は `raw_ckpt_group_sources` = format の
        // group 順)。ft_w の m / v は `--fp16-opt-state` の現在精度へ量子化して
        // 載せ直す (checkpoint は真値 f32、mode 非依存)。
        let (ftw_w, ftw_m, ftw_v, ftw_slow) = &loaded[0];
        self.ft_w = DeviceBuffer::from_host(&self.stream, ftw_w)?;
        self.ft_w_m =
            MomentBuf::from_host_f32(&self.stream, ftw_m, self.fp16_opt_state, FT_OPT_M_SCALE)?;
        self.ft_w_v =
            MomentBuf::from_host_f32(&self.stream, ftw_v, self.fp16_opt_state, FT_OPT_V_SCALE)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, ftw_slow)?;

        macro_rules! up {
            ($idx:expr, $w:ident, $m:ident, $v:ident, $slow:ident) => {{
                let (w, m, v, s) = &loaded[$idx];
                self.$w = DeviceBuffer::from_host(&self.stream, w)?;
                self.$m = DeviceBuffer::from_host(&self.stream, m)?;
                self.$v = DeviceBuffer::from_host(&self.stream, v)?;
                self.$slow = DeviceBuffer::from_host(&self.stream, s)?;
            }};
        }
        up!(1, ft_b, ft_b_m, ft_b_v, ft_b_slow);
        up!(2, l1_w, l1_w_m, l1_w_v, l1_w_slow);
        up!(3, l1_b, l1_b_m, l1_b_v, l1_b_slow);
        up!(4, l2_w, l2_w_m, l2_w_v, l2_w_slow);
        up!(5, l2_b, l2_b_m, l2_b_v, l2_b_slow);
        up!(6, l3_w, l3_w_m, l3_w_v, l3_w_slow);
        up!(7, l3_b, l3_b_m, l3_b_v, l3_b_slow);

        self.step_count = header.step_count;
        Ok((header.superbatch, header.producer_run_id, header.lr_horizon))
    }

    /// raw checkpoint format の全 weight group を format の group 順 (= ft_w, ft_b,
    /// l1_w, l1_b, l2_w, l2_b, l3_w, l3_b) で返す (save / load で iterate する
    /// immutable view)。`grad` は resume に不要なので含めない。ft_w は `m` / `v` が
    /// [`MomentBuf`] (`f32`/`f16`) で型が他 group と揃わないため
    /// [`RawCkptGroupBufs::FtMoment`]、他は全 buffer `f32` の
    /// [`RawCkptGroupBufs::Uniform`]。
    pub(crate) fn raw_ckpt_group_sources(&self) -> Vec<RawCkptGroupSource<'_>> {
        macro_rules! uniform {
            ($name:literal, $len:expr, $w:ident, $m:ident, $v:ident, $slow:ident) => {
                RawCkptGroupSource {
                    name: $name,
                    len: $len,
                    bufs: RawCkptGroupBufs::Uniform {
                        w: &self.$w,
                        m: &self.$m,
                        v: &self.$v,
                        slow: &self.$slow,
                    },
                }
            };
        }
        // ft_w は train 形状 (factorizer 有効時は piece-input 仮想行込み)。header にも
        // train_ft_in と ft_factorize flag が書かれ、resume 時の on/off 不一致は
        // [`load_raw_checkpoint_file`] が reject する。
        let ft_w_n = self.id.feature_set.train_ft_in() * self.id.ft_out;
        let ft_b_n = self.id.ft_out;
        let l1_w_n = self.id.combined_dim() * self.id.l1_out;
        let l1_b_n = self.id.l1_out;
        let l2_w_n = self.id.l1_out * self.id.l2_out;
        let l2_b_n = self.id.l2_out;
        let l3_w_n = self.id.l2_out;
        let l3_b_n = 1;
        vec![
            RawCkptGroupSource {
                name: "ft_w",
                len: ft_w_n,
                bufs: RawCkptGroupBufs::FtMoment {
                    w: &self.ft_w,
                    m: &self.ft_w_m,
                    v: &self.ft_w_v,
                    slow: &self.ft_w_slow,
                },
            },
            uniform!("ft_b", ft_b_n, ft_b, ft_b_m, ft_b_v, ft_b_slow),
            uniform!("l1_w", l1_w_n, l1_w, l1_w_m, l1_w_v, l1_w_slow),
            uniform!("l1_b", l1_b_n, l1_b, l1_b_m, l1_b_v, l1_b_slow),
            uniform!("l2_w", l2_w_n, l2_w, l2_w_m, l2_w_v, l2_w_slow),
            uniform!("l2_b", l2_b_n, l2_b, l2_b_m, l2_b_v, l2_b_slow),
            uniform!("l3_w", l3_w_n, l3_w, l3_w_m, l3_w_v, l3_w_slow),
            uniform!("l3_b", l3_b_n, l3_b, l3_b_m, l3_b_v, l3_b_slow),
        ]
    }

    /// held-out validation 用 forward-only。weight は更新せず、batch 全体の `Σerr²`
    /// と position ごとの net 出力 (`b` 個) を返す。
    pub(crate) fn validate(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(StepOutput {
                loss: 0.0,
                net_output: Vec::new(),
            });
        }
        self.ws.check_batch_capacity(b)?;
        self.run_forward_kernels(batch, wdl_lambda, loss, false)?;
        let net_output = self.ws.net_output.to_host_vec(&self.stream)?[..b].to_vec();
        let loss_host = self.loss_acc.to_host_vec(&self.stream)?;
        Ok(StepOutput {
            loss: loss_host[0],
            net_output,
        })
    }
}

trainer_backend_impl! {
    trainer: SimpleGpuTrainer,
    feature_set: id.feature_set,
    batch: bucketless,
    weights: to_simple_weights,
    step_error: "SimpleGpuTrainer::step failed: {}",
    validate_error: "SimpleGpuTrainer::validate failed: {}",
    flush_error: "SimpleGpuTrainer::loss_ring.flush_pending_loss failed: {}",
    weights_error: "SimpleGpuTrainer::to_simple_weights failed: {}",
    resume_error: "SimpleGpuTrainer::save_raw_checkpoint failed: {}",
}
