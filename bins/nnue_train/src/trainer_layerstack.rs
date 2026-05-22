use std::io::Write;
use std::path::Path;

use cuda_host::cuda_launch;
use gpu_runtime::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
use nnue_format::ArchKind;
use nnue_format::LayerStackWeights;
use nnue_train::dataloader::Batch;
use nnue_train::optimizer::radam_compute_step_size_denom;
use nnue_train::trainer::{LossKind, TrainerBackend, ValidationStepOutput};
use shogi_features::FeatureSetSpec;

use crate::*;
use crate::{arch::*, ckpt::*, kernel_module::*, trainer_common::*};

// ===========================================================================
// GpuTrainer (LayerStack: FT ft_out → L1 16 → L2 32 + progress8kpabs 9 buckets)
//
// 10 weight groups × {w, m, v, slow, grad} = 50 device buffers + loss_acc + step_count。
// Forward は 15 kernel launch、backward は ~16 kernel launch、optimizer は 10×{radam+lerp}。
// ===========================================================================

#[allow(dead_code)] // 一部 field は host state 直接更新時のみ使う
pub(crate) struct GpuTrainer {
    stream: std::sync::Arc<CudaStream>,
    module: std::sync::Arc<CudaModule>,

    // FT (single, shared across perspectives)
    ft_w: DeviceBuffer<f32>,
    /// Ranger 1st/2nd moment。既定 `f32`、`--fp16-opt-state` で `f16` ([`MomentBuf`])。
    ft_w_m: MomentBuf,
    ft_w_v: MomentBuf,
    ft_w_slow: DeviceBuffer<f32>,
    ft_w_grad: DeviceBuffer<f32>,
    /// `ft_w` の FP16 mirror。`ft_fp16` が true のときだけ確保され、毎 step `ft_w`
    /// (FP32 master) から変換される。`sparse_ft_forward_fp16` の weight 入力。
    ft_w_h: Option<DeviceBuffer<f16>>,
    ft_b: DeviceBuffer<f32>,
    ft_b_m: DeviceBuffer<f32>,
    ft_b_v: DeviceBuffer<f32>,
    ft_b_slow: DeviceBuffer<f32>,
    ft_b_grad: DeviceBuffer<f32>,

    // L1 per-bucket delta
    l1_w: DeviceBuffer<f32>,
    l1_w_m: DeviceBuffer<f32>,
    l1_w_v: DeviceBuffer<f32>,
    l1_w_slow: DeviceBuffer<f32>,
    l1_w_grad: DeviceBuffer<f32>,
    l1_b: DeviceBuffer<f32>,
    l1_b_m: DeviceBuffer<f32>,
    l1_b_v: DeviceBuffer<f32>,
    l1_b_slow: DeviceBuffer<f32>,
    l1_b_grad: DeviceBuffer<f32>,

    // L1f shared factorized
    l1f_w: DeviceBuffer<f32>,
    l1f_w_m: DeviceBuffer<f32>,
    l1f_w_v: DeviceBuffer<f32>,
    l1f_w_slow: DeviceBuffer<f32>,
    l1f_w_grad: DeviceBuffer<f32>,
    l1f_b: DeviceBuffer<f32>,
    l1f_b_m: DeviceBuffer<f32>,
    l1f_b_v: DeviceBuffer<f32>,
    l1f_b_slow: DeviceBuffer<f32>,
    l1f_b_grad: DeviceBuffer<f32>,

    // L2 per-bucket
    l2_w: DeviceBuffer<f32>,
    l2_w_m: DeviceBuffer<f32>,
    l2_w_v: DeviceBuffer<f32>,
    l2_w_slow: DeviceBuffer<f32>,
    l2_w_grad: DeviceBuffer<f32>,
    l2_b: DeviceBuffer<f32>,
    l2_b_m: DeviceBuffer<f32>,
    l2_b_v: DeviceBuffer<f32>,
    l2_b_slow: DeviceBuffer<f32>,
    l2_b_grad: DeviceBuffer<f32>,

    // L3 per-bucket output
    l3_w: DeviceBuffer<f32>,
    l3_w_m: DeviceBuffer<f32>,
    l3_w_v: DeviceBuffer<f32>,
    l3_w_slow: DeviceBuffer<f32>,
    l3_w_grad: DeviceBuffer<f32>,
    l3_b: DeviceBuffer<f32>,
    l3_b_m: DeviceBuffer<f32>,
    l3_b_v: DeviceBuffer<f32>,
    l3_b_slow: DeviceBuffer<f32>,
    l3_b_grad: DeviceBuffer<f32>,

    // 中間 activation / activation-grad の永続 workspace (batch_size 固定前提で `new`
    // 時に確保。`step_impl` が requires より大きい batch を渡したら拡張)。
    ws: GpuWorkspace,

    // loss + step
    loss_acc: DeviceBuffer<f64>,
    /// step() 末の `loss_acc` 同期読みを async + 1-step lag に置換する pinned host ring。
    /// host が `stream.synchronize` を待たずに次 batch の launch を発行できるようになる。
    loss_ring: AsyncLossRing,
    /// step 先頭の入力 H2D を専用 copy stream で直前 step の compute と overlap させる ring。
    input_ring: InputUploadRing,
    /// L1f weight backward の `dense_mm_bwd_weight_tiled` を `cublasSgemm_v2` に置換するための
    /// cuBLAS handle。stream は `self.stream` に bind 済 (cuBLAS の launch は same-stream で
    /// in-order に走る)。
    cublas: CublasHandle,
    /// true なら forward の `sparse_ft_forward` を FP16 weight 版に切替える
    /// (`--ft-fp16`)。false で従来の FP32 path と bit-identical。
    ft_fp16: bool,
    /// true なら FT activation (`ft_*_out` forward 出力 / `dft_*_out` backward 勾配) も
    /// FP16 で保持する (`--ft-fp16-out`)。`ft_fp16` が true のときのみ true になりうる。
    ft_fp16_out: bool,
    /// true なら `ft_w` の Ranger moment (`m` / `v`) を `f16` で保持する
    /// (`--fp16-opt-state`)。`ft_w_m` / `ft_w_v` が [`MomentBuf::F16`] になり、optimizer
    /// step は [`radam_step_f16state`] 系を使う。false で従来の `f32` path。
    fp16_opt_state: bool,
    /// 入力 feature set spec。FT 入力次元 (`ft_in`) / active feature 数
    /// (`max_active`) / artifact identity の単一の真実源。起動時に
    /// `--feature-set` から一度だけ決まり、以降不変。
    feature_set: FeatureSetSpec,
    /// Ranger optimizer の weight decay 係数。各 weight group の `radam_step`
    /// に一律 `decay` 引数として渡す。`--weight-decay` から起動時に決まり、
    /// 以降不変。既定 0.0 で decay 無し。
    weight_decay: f32,
    step_count: u64,
}

impl Drop for GpuTrainer {
    fn drop(&mut self) {
        // 残り queue 済 GPU 操作 (`loss_ring` の async D2H が `loss_acc` を read する、
        // `input_ring` の copy stream H2D が `ws` の input buffer を write する等) を
        // 両 stream で完了させてから field の Drop に進む。さもなければ struct field
        // 宣言順で device memory が先に `cuMemFree` され、in-flight な copy が解放済
        // メモリに触れる race になる。両 sync 後は後続 per-field cleanup が全部 safe。
        // 失敗は無視 (Drop 中の error 報告は実用上困難、stream 破棄で driver が
        // tracking を解除する debug-build 動作と等価)。
        let _ = self.stream.synchronize();
        let _ = self.input_ring.copy_stream.synchronize();
    }
}

/// `GpuTrainer::step_impl` の forward / backward で使う中間 activation と
/// activation-gradient buffer を **1 step ごとに再 alloc せず永続化** するための
/// workspace。
///
/// 各 buffer は `len_batch` 個の position 分のサイズで `GpuTrainer::new` 時に一度だけ
/// 確保する。固定 batch 前提で、`step_impl` は [`GpuWorkspace::check_batch_capacity`]
/// で batch が `len_batch` に収まることを検証する (実 dataloader は `batch_size` 以下の
/// batch しか出さない。step 中の再 alloc は in-flight な compute の device memory を
/// 解放する race になるため行わない)。`len_batch == 0` は「まだ未確保」を表す番兵
/// (実際には `GpuTrainer::new` で `batch_size` 分を確保するので step 時には常に > 0)。
///
/// **メモリ覚書**: forward path は DAG で各 activation は読まれる前に kernel が
/// 全 cell を上書きするため memset 不要。`ws_batch` が現 batch `b` より大きい場合の
/// 末尾 `[b*dim .. ws_batch*dim)` は kernel が触らないが、後続 kernel も `b` で
/// bound するので read されない。例外は `dl1_total`: `slice_scatter_2d` の host
/// 契約 (「dst を 0 初期化」) を守るため `step_impl` で毎 step memset する。
/// grad buffer (`GpuTrainer::*_grad`) と `loss_acc` は atomic accumulate semantics
/// なので `step_impl` で毎 step memset する (元実装の `DeviceBuffer::zeroed` 再 alloc を
/// memset_async 0 に置換、`cudaMalloc`/`cudaFree` の stream stall を回避)。
pub(crate) struct GpuWorkspace {
    /// この workspace が確保している batch (= position) 数。0 = 未確保。
    len_batch: usize,

    /// FT 入力次元 (feature set ごとに異なる)。inverse-index scratch
    /// (`feat_*`) と FT forward/backward kernel の launch arg に使う。
    ft_in: usize,
    /// 1 perspective あたりの active feature 数 (feature set ごとに異なる)。
    /// 入力 index buffer (`stm_idx_dev` 等) の容量と FT kernel の launch arg。
    max_active: usize,
    /// FT 出力次元 (1 perspective あたり)。`--ft-out` から起動時に決まる。
    /// post-FT activation buffer の幅と FT forward/backward kernel の launch arg。
    /// post-FT の `combined` buffer もこの幅 (pairwise で半減後に 2 perspective 連結)。
    ft_out: usize,

    // -- forward activations --
    ft_stm_out: DeviceBuffer<f32>,    // b × ft_out
    ft_nstm_out: DeviceBuffer<f32>,   // b × ft_out
    combined: DeviceBuffer<f32>,      // b × ft_out (post-FT、pairwise 後 2 perspective 連結)
    l1_bucket: DeviceBuffer<f32>,     // b × L1_OUT
    l1f_out: DeviceBuffer<f32>,       // b × L1_OUT
    l1_total: DeviceBuffer<f32>,      // b × L1_OUT
    l1_main: DeviceBuffer<f32>,       // b × L1_EFFECTIVE
    l1_skip: DeviceBuffer<f32>,       // b × L1_SKIP
    l1_sqr: DeviceBuffer<f32>,        // b × L1_EFFECTIVE
    l2_pre: DeviceBuffer<f32>,        // b × L2_IN
    l2_input: DeviceBuffer<f32>,      // b × L2_IN
    l2_out: DeviceBuffer<f32>,        // b × L2_OUT
    l2_acted: DeviceBuffer<f32>,      // b × L2_OUT
    l3_out: DeviceBuffer<f32>,        // b
    net_output: DeviceBuffer<f32>,    // b
    dy_net_output: DeviceBuffer<f32>, // b (loss kernel が書き込む dnet)

    // -- backward activation-grads --
    dl2_acted: DeviceBuffer<f32>,            // b × L2_OUT
    dl2_out: DeviceBuffer<f32>,              // b × L2_OUT
    dl2_input: DeviceBuffer<f32>,            // b × L2_IN
    dl2_pre: DeviceBuffer<f32>,              // b × L2_IN
    dl1_sqr: DeviceBuffer<f32>,              // b × L1_EFFECTIVE
    dl1_main_from_concat: DeviceBuffer<f32>, // b × L1_EFFECTIVE
    dl1_main_from_sqr: DeviceBuffer<f32>,    // b × L1_EFFECTIVE
    dl1_main: DeviceBuffer<f32>,             // b × L1_EFFECTIVE
    dl1_total: DeviceBuffer<f32>,            // b × L1_OUT (毎 step memset、slice_scatter 契約)
    dcombined_from_l1f: DeviceBuffer<f32>,   // b × ft_out
    dcombined_from_l1: DeviceBuffer<f32>,    // b × ft_out
    dft_stm_out: DeviceBuffer<f32>,          // b × ft_out
    dft_nstm_out: DeviceBuffer<f32>,         // b × ft_out

    // FT activation の FP16 版。`ft_fp16_out` (`--ft-fp16-out`) が true のときだけ
    // b × ft_out で確保され、`ft_*_out` / `dft_*_out` (f32) の代わりに使われる
    // (f32 版はそのとき placeholder size でしか確保しない)。false なら全て `None`。
    ft_stm_out_h: Option<DeviceBuffer<f16>>,   // b × ft_out
    ft_nstm_out_h: Option<DeviceBuffer<f16>>,  // b × ft_out
    dft_stm_out_h: Option<DeviceBuffer<f16>>,  // b × ft_out
    dft_nstm_out_h: Option<DeviceBuffer<f16>>, // b × ft_out

    // -- inverse-index sparse_ft_backward scratch (sized by feature set) --
    feat_counts: DeviceBuffer<u32>, // ft_in: per-feature histogram (atomic build)
    feat_offsets: DeviceBuffer<u32>, // ft_in + 1: exclusive prefix sum
    feat_write_ctr: DeviceBuffer<u32>, // ft_in: scatter atomic counter
    feat_positions: DeviceBuffer<u32>, // up to batch * max_active: sorted positions

    // -- pre-allocated input buffers (per-step `from_host` の cudaMalloc/Free を排除) --
    // `*_dev` が現 step の active、`*_dev_back` が double-buffer の back。`step_impl` が
    // 毎 step `mem::swap` し、直前 step が読んでいない back 側へ次 step 入力を copy
    // stream で先行 H2D する ([`InputUploadRing`])。
    stm_idx_dev: DeviceBuffer<i32>,         // batch * max_active
    nstm_idx_dev: DeviceBuffer<i32>,        // batch * max_active
    bucket_idx_dev: DeviceBuffer<i32>,      // batch
    score_dev: DeviceBuffer<f32>,           // batch
    wdl_dev: DeviceBuffer<f32>,             // batch
    stm_idx_dev_back: DeviceBuffer<i32>,    // batch * max_active
    nstm_idx_dev_back: DeviceBuffer<i32>,   // batch * max_active
    bucket_idx_dev_back: DeviceBuffer<i32>, // batch
    score_dev_back: DeviceBuffer<f32>,      // batch
    wdl_dev_back: DeviceBuffer<f32>,        // batch

    // -- bucket sort scratch (fwd_L1 用 sorted layout 切換) --
    bucket_counts_dev: DeviceBuffer<u32>, // NUM_BUCKETS + 1 (histogram + invalid bin)
    bucket_offsets_dev: DeviceBuffer<u32>, // NUM_BUCKETS + 1 (exclusive scan)
    bucket_write_ctr_dev: DeviceBuffer<u32>, // NUM_BUCKETS + 1 (scatter ranking counter)
    bucket_perm_dev: DeviceBuffer<i32>,   // batch (perm[i] = original row index)
    bucket_idx_sorted_dev: DeviceBuffer<i32>, // batch (sorted bucket values)
    combined_sorted: DeviceBuffer<f32>,   // batch × ft_out (combined を perm で gather)
    l1_bucket_sorted: DeviceBuffer<f32>,  // batch × L1_OUT (sorted fwd_L1 出力)
    dl1_total_sorted: DeviceBuffer<f32>,  // batch × L1_OUT (dl1_total を perm で gather)
    dl2_out_sorted: DeviceBuffer<f32>,    // batch × L2_OUT (dl2_out を perm で gather、L2 bias 用)
}

impl GpuWorkspace {
    /// `batch` 個の position 分の全 buffer を確保する (`GpuTrainer::new` から呼ぶ)。
    /// `ft_out` は FT 出力次元 (1 perspective あたり、`--ft-out`)。
    ///
    /// `ft_fp16_out` が true なら FT activation (`ft_*_out` / `dft_*_out`) を `f16` で
    /// 持つ。その場合 f32 版は使われないので placeholder size (`ft_out` 要素 = 1 行) で
    /// のみ確保し、`*_h` (f16) を `batch * ft_out` で確保する。false なら f32 版を
    /// `batch * ft_out`、`*_h` は `None`。
    pub(crate) fn new(
        stream: &CudaStream,
        batch: usize,
        ft_out: usize,
        ft_fp16_out: bool,
        feature_set: FeatureSetSpec,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ft_in = feature_set.ft_in();
        let max_active = feature_set.max_active();
        let z = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(stream, n).map_err(Into::into)
        };
        // FT activation の f32 buffer size。ft_fp16_out 時は f16 版を使うので f32 版は
        // placeholder (`ft_out` 要素) のみ。
        let ft_act_f32_n = if ft_fp16_out { ft_out } else { batch * ft_out };
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
            ft_out,
            ft_stm_out: z(ft_act_f32_n)?,
            ft_nstm_out: z(ft_act_f32_n)?,
            ft_stm_out_h: alloc_h(ft_fp16_out)?,
            ft_nstm_out_h: alloc_h(ft_fp16_out)?,
            dft_stm_out_h: alloc_h(ft_fp16_out)?,
            dft_nstm_out_h: alloc_h(ft_fp16_out)?,
            combined: z(batch * ft_out)?,
            l1_bucket: z(batch * L1_OUT)?,
            l1f_out: z(batch * L1_OUT)?,
            l1_total: z(batch * L1_OUT)?,
            l1_main: z(batch * L1_EFFECTIVE)?,
            l1_skip: z(batch * L1_SKIP)?,
            l1_sqr: z(batch * L1_EFFECTIVE)?,
            l2_pre: z(batch * L2_IN)?,
            l2_input: z(batch * L2_IN)?,
            l2_out: z(batch * L2_OUT)?,
            l2_acted: z(batch * L2_OUT)?,
            l3_out: z(batch)?,
            net_output: z(batch)?,
            dy_net_output: z(batch)?,
            dl2_acted: z(batch * L2_OUT)?,
            dl2_out: z(batch * L2_OUT)?,
            dl2_input: z(batch * L2_IN)?,
            dl2_pre: z(batch * L2_IN)?,
            dl1_sqr: z(batch * L1_EFFECTIVE)?,
            dl1_main_from_concat: z(batch * L1_EFFECTIVE)?,
            dl1_main_from_sqr: z(batch * L1_EFFECTIVE)?,
            dl1_main: z(batch * L1_EFFECTIVE)?,
            dl1_total: z(batch * L1_OUT)?,
            dcombined_from_l1f: z(batch * ft_out)?,
            dcombined_from_l1: z(batch * ft_out)?,
            dft_stm_out: z(ft_act_f32_n)?,
            dft_nstm_out: z(ft_act_f32_n)?,
            feat_counts: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_offsets: DeviceBuffer::<u32>::zeroed(stream, ft_in + 1)?,
            feat_write_ctr: DeviceBuffer::<u32>::zeroed(stream, ft_in)?,
            feat_positions: DeviceBuffer::<u32>::zeroed(stream, batch * max_active)?,
            stm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            bucket_idx_dev: DeviceBuffer::<i32>::zeroed(stream, batch)?,
            score_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            stm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            nstm_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch * max_active)?,
            bucket_idx_dev_back: DeviceBuffer::<i32>::zeroed(stream, batch)?,
            score_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            wdl_dev_back: DeviceBuffer::<f32>::zeroed(stream, batch)?,
            bucket_counts_dev: DeviceBuffer::<u32>::zeroed(stream, NUM_BUCKETS + 1)?,
            bucket_offsets_dev: DeviceBuffer::<u32>::zeroed(stream, NUM_BUCKETS + 1)?,
            bucket_write_ctr_dev: DeviceBuffer::<u32>::zeroed(stream, NUM_BUCKETS + 1)?,
            bucket_perm_dev: DeviceBuffer::<i32>::zeroed(stream, padded_sort_batch(batch))?,
            bucket_idx_sorted_dev: DeviceBuffer::<i32>::zeroed(stream, padded_sort_batch(batch))?,
            combined_sorted: z(padded_sort_batch(batch) * ft_out)?,
            l1_bucket_sorted: z(padded_sort_batch(batch) * L1_OUT)?,
            dl1_total_sorted: z(padded_sort_batch(batch) * L1_OUT)?,
            dl2_out_sorted: z(padded_sort_batch(batch) * L2_OUT)?,
        })
    }

    /// `GpuTrainer::new` で確保した `len_batch` 容量に `batch` が収まることを検証する。
    /// 収まらなければ error を返す (caller が step を中断)。
    ///
    /// workspace は固定 batch 前提で `GpuTrainer::new` 時に一度だけ確保する。実 dataloader
    /// は `batch_size` 以下の batch しか出さない (末尾の partial batch は小さい) ので
    /// 通常この検証は通る。step 中は前 step の compute が in-flight でありうるため、
    /// ここで buffer を再 alloc すると使用中の device memory を解放する race になる。
    /// よって grow はせず、容量超過は error として扱う。
    pub(crate) fn check_batch_capacity(
        &self,
        batch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if batch > self.len_batch {
            return Err(format!(
                "batch {batch} exceeds workspace capacity {} (workspace は GpuTrainer::new で\
                 一度だけ確保する。--batch-size を増やす場合は再起動が要る)",
                self.len_batch
            )
            .into());
        }
        Ok(())
    }
}

impl GpuTrainer {
    /// CUDA context を作成し、kernel module を load、10 weight groups + Ranger state +
    /// 中間 activation workspace (`batch_size` 分) を確保。
    ///
    /// `enable_tf32` は cuBLAS の `cublasSetMathMode` 引数を切替 ([`CublasHandle::new`])、
    /// `true` で Ampere+ TC TF32 mode、`false` で純 FP32。default は CLI 側で OFF。
    ///
    /// `ft_fp16` が true なら FP16 weight mirror (`ft_w_h`) を確保し、forward の
    /// `sparse_ft_forward` を FP16 版に切替える。false なら mirror は未確保で従来 path。
    /// `ft_fp16_out` が true なら FT activation も FP16 で持つ (`ft_fp16` を要求、
    /// caller が validation 済)。
    ///
    /// `ft_out` は FT 出力次元 (1 perspective あたり、`--ft-out`)。weight group の
    /// 要素数と activation workspace の幅を決める。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch_size: usize,
        ft_out: usize,
        enable_tf32: bool,
        ft_fp16: bool,
        ft_fp16_out: bool,
        fp16_opt_state: bool,
        feature_set: FeatureSetSpec,
        weight_decay: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // `ft_fp16_out` は weight FP16 path の拡張なので `ft_fp16` を含意する。CLI 検証
        // (`run_training`) で reject 済だが、forward 分岐の各 `.expect()` がこの不変条件を
        // 前提にするため constructor でも明示する。
        debug_assert!(!ft_fp16_out || ft_fp16, "ft_fp16_out requires ft_fp16");
        let stream = ctx.default_stream();
        let module = load_kernel_module_with_fallback(ctx, "nnue_train")?;

        // 各 weight group の element 数 (FT 入力次元は feature set 依存、FT 出力次元は
        // `--ft-out` 依存)
        let ft_in = feature_set.ft_in();
        let ft_w_n = ft_in * ft_out;
        let ft_b_n = ft_out;
        let l1_w_n = NUM_BUCKETS * L1_OUT * ft_out;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = ft_out * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;

        // Weight init: small random for non-degenerate forward (smoke 用、後段で
        // proper init を適用: ft は bullet `init_with_effective_input_size(32)`、l1 は Zeroed 等)
        let init_scale = 0.01_f32;
        let ft_w_init = xorshift_init(0x100_u64, ft_w_n, init_scale);
        let l1_w_init = xorshift_init(0x101_u64, l1_w_n, init_scale);
        let l1f_w_init = xorshift_init(0x102_u64, l1f_w_n, init_scale);
        let l2_w_init = xorshift_init(0x103_u64, l2_w_n, init_scale);
        let l3_w_init = xorshift_init(0x104_u64, l3_w_n, init_scale);

        // Ranger Lookahead の slow weight は **0 初期化** (bullet `RangerLookahead::new`
        // = `vec![0.0; size]` と同じ)。初回 lerp (`step % k == 0`) で
        // `weights = alpha*weights + (1-alpha)*0 = alpha*weights` になる挙動も bullet と一致。
        Ok(Self {
            stream: stream.clone(),
            module,
            // FT
            ft_w: DeviceBuffer::from_host(&stream, &ft_w_init)?,
            ft_w_m: MomentBuf::zeroed(&stream, ft_w_n, fp16_opt_state)?,
            ft_w_v: MomentBuf::zeroed(&stream, ft_w_n, fp16_opt_state)?,
            ft_w_slow: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_grad: DeviceBuffer::<f32>::zeroed(&stream, ft_w_n)?,
            ft_w_h: if ft_fp16 {
                Some(DeviceBuffer::<f16>::zeroed(&stream, ft_w_n)?)
            } else {
                None
            },
            ft_b: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_m: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_v: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_slow: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            ft_b_grad: DeviceBuffer::<f32>::zeroed(&stream, ft_b_n)?,
            // L1
            l1_w: DeviceBuffer::from_host(&stream, &l1_w_init)?,
            l1_w_m: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_w_v: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l1_w_n)?,
            l1_b: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_m: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_v: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            l1_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l1_b_n)?,
            // L1f
            l1f_w: DeviceBuffer::from_host(&stream, &l1f_w_init)?,
            l1f_w_m: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_w_v: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l1f_w_n)?,
            l1f_b: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_m: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_v: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            l1f_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l1f_b_n)?,
            // L2
            l2_w: DeviceBuffer::from_host(&stream, &l2_w_init)?,
            l2_w_m: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_w_v: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l2_w_n)?,
            l2_b: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_m: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_v: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            l2_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l2_b_n)?,
            // L3
            l3_w: DeviceBuffer::from_host(&stream, &l3_w_init)?,
            l3_w_m: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_w_v: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_w_slow: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_w_grad: DeviceBuffer::<f32>::zeroed(&stream, l3_w_n)?,
            l3_b: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_m: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_v: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_slow: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            l3_b_grad: DeviceBuffer::<f32>::zeroed(&stream, l3_b_n)?,
            // 中間 activation workspace (`batch_size` 分。最低 1 で確保して
            // `len_batch == 0` (未確保) を作らない — smoke は小さい固定 batch を渡す)。
            // FT activation の f16 buffer 確保は `ft_fp16_out` で決まる。
            ws: GpuWorkspace::new(&stream, batch_size.max(1), ft_out, ft_fp16_out, feature_set)?,
            // loss + step
            loss_acc: DeviceBuffer::<f64>::zeroed(&stream, 1)?,
            loss_ring: AsyncLossRing::new(ctx)?,
            input_ring: InputUploadRing::new(ctx, batch_size.max(1), feature_set.max_active())?,
            cublas: CublasHandle::new(&stream, enable_tf32)?,
            ft_fp16,
            ft_fp16_out,
            fp16_opt_state,
            feature_set,
            weight_decay,
            step_count: 0,
        })
    }

    /// `LayerStackWeights` から weight buffer を device に upload (pretrained 注入、`--init-from`)。
    ///
    /// Optimizer state reset:
    /// - `m`, `v`: 0 (fresh start、Ranger 1st/2nd moment)
    /// - `slow`: **loaded weights と同値** (warm-start anchor。`GpuTrainer::new` (from-scratch)
    ///   は bullet `RangerLookahead::new` どおり `slow = 0` だが、`--init-from` は量子化済 NNUE
    ///   の continue-training/fine-tuning であって bullet checkpoint resume (`slow.bin` 付き)
    ///   ではない。`slow = 0` のままだと初回 lookahead lerp で `new_w = alpha*fast + (1-alpha)*0
    ///   = alpha*fast` となり読み込んだ重みが全て ~alpha 倍に縮む。`slow = w_loaded` にすると
    ///   初回 lerp は `new_w = alpha*fast + (1-alpha)*w_loaded` で、fine-tuning は lr が小さく
    ///   `fast ≈ w_loaded` なので **0 ではなく読み込んだ重みの方へ寄せる** anchor になる
    ///   (true な bullet resume なら `slow.bin` を読むべきだが、量子化 NNUE には optimizer
    ///   state が無いので next-best な default)
    /// - `grad`: 0
    /// - `step_count`: 0 (1-indexed、次 step は 1)
    ///
    /// 注: `step_count = 0` 状態で `step()` を呼ぶと `self.step_count += 1` → 1 に
    /// なってから `radam_compute_step_size_denom(1, BETA1, BETA2, N_SMA_THRESHOLD)`
    /// を呼ぶ。bullet `radam_step.rs::radam_compute_step_size_denom` は step >= 1 で
    /// 安全動作 (step=0 では `beta^0 = 1` → `bc1 = 0` で `step_size = 1/0 = inf` に
    /// なる、本 helper も `step >= 1` 前提)。本実装は step=0 で呼ばないため OK。
    pub(crate) fn load_layerstack_weights(
        &mut self,
        w: &LayerStackWeights,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // optimizer companion buffer (`ft_w_m`/`v`/`grad`/`slow`) は trainer の
        // feature set で確保済。weight の feature set が異なると `ft_w` だけ別長に
        // なり optimizer step が out-of-bounds になるため、ここで弾く。
        if w.feature_set != self.feature_set {
            return Err(invalid_data(format!(
                "weight feature set '{}' does not match trainer feature set '{}'",
                w.feature_set.canonical_name(),
                self.feature_set.canonical_name()
            )));
        }
        self.ft_w = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_b = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.l1_w = DeviceBuffer::from_host(&self.stream, &w.l1_w)?;
        self.l1_b = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1f_w = DeviceBuffer::from_host(&self.stream, &w.l1f_w)?;
        self.l1f_b = DeviceBuffer::from_host(&self.stream, &w.l1f_b)?;
        self.l2_w = DeviceBuffer::from_host(&self.stream, &w.l2_w)?;
        self.l2_b = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l3_w = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_b = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;
        // Optimizer state reset:
        // - m, v: 0 (fresh start)
        // - slow: loaded weights と同値 (warm-start anchor: 初回 lookahead lerp が
        //   0 でなく読み込んだ重みの方へ寄る。`slow = 0` だと alpha 倍に縮む)
        // - grad: 0
        let zeros_f32 = |n: usize| -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
            DeviceBuffer::<f32>::zeroed(&self.stream, n).map_err(Into::into)
        };
        let ft_out = self.ws.ft_out;
        let ft_w_n = self.feature_set.ft_in() * ft_out;
        let ft_b_n = ft_out;
        let l1_w_n = NUM_BUCKETS * L1_OUT * ft_out;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = ft_out * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        self.ft_w_m = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        self.ft_w_v = MomentBuf::zeroed(&self.stream, ft_w_n, self.fp16_opt_state)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, &w.ft_w)?;
        self.ft_w_grad = zeros_f32(ft_w_n)?;
        self.ft_b_m = zeros_f32(ft_b_n)?;
        self.ft_b_v = zeros_f32(ft_b_n)?;
        self.ft_b_slow = DeviceBuffer::from_host(&self.stream, &w.ft_b)?;
        self.ft_b_grad = zeros_f32(ft_b_n)?;
        self.l1_w_m = zeros_f32(l1_w_n)?;
        self.l1_w_v = zeros_f32(l1_w_n)?;
        self.l1_w_slow = DeviceBuffer::from_host(&self.stream, &w.l1_w)?;
        self.l1_w_grad = zeros_f32(l1_w_n)?;
        self.l1_b_m = zeros_f32(l1_b_n)?;
        self.l1_b_v = zeros_f32(l1_b_n)?;
        self.l1_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1_b)?;
        self.l1_b_grad = zeros_f32(l1_b_n)?;
        self.l1f_w_m = zeros_f32(l1f_w_n)?;
        self.l1f_w_v = zeros_f32(l1f_w_n)?;
        self.l1f_w_slow = DeviceBuffer::from_host(&self.stream, &w.l1f_w)?;
        self.l1f_w_grad = zeros_f32(l1f_w_n)?;
        self.l1f_b_m = zeros_f32(l1f_b_n)?;
        self.l1f_b_v = zeros_f32(l1f_b_n)?;
        self.l1f_b_slow = DeviceBuffer::from_host(&self.stream, &w.l1f_b)?;
        self.l1f_b_grad = zeros_f32(l1f_b_n)?;
        self.l2_w_m = zeros_f32(l2_w_n)?;
        self.l2_w_v = zeros_f32(l2_w_n)?;
        self.l2_w_slow = DeviceBuffer::from_host(&self.stream, &w.l2_w)?;
        self.l2_w_grad = zeros_f32(l2_w_n)?;
        self.l2_b_m = zeros_f32(l2_b_n)?;
        self.l2_b_v = zeros_f32(l2_b_n)?;
        self.l2_b_slow = DeviceBuffer::from_host(&self.stream, &w.l2_b)?;
        self.l2_b_grad = zeros_f32(l2_b_n)?;
        self.l3_w_m = zeros_f32(l3_w_n)?;
        self.l3_w_v = zeros_f32(l3_w_n)?;
        self.l3_w_slow = DeviceBuffer::from_host(&self.stream, &w.l3_w)?;
        self.l3_w_grad = zeros_f32(l3_w_n)?;
        self.l3_b_m = zeros_f32(l3_b_n)?;
        self.l3_b_v = zeros_f32(l3_b_n)?;
        self.l3_b_slow = DeviceBuffer::from_host(&self.stream, &w.l3_b)?;
        self.l3_b_grad = zeros_f32(l3_b_n)?;
        self.step_count = 0;
        Ok(())
    }

    /// device buffer を host に download し `LayerStackWeights` を返す (save_quantised 前)。
    pub(crate) fn to_layerstack_weights(
        &self,
    ) -> Result<LayerStackWeights, Box<dyn std::error::Error>> {
        Ok(LayerStackWeights {
            feature_set: self.feature_set,
            ft_w: self.ft_w.to_host_vec(&self.stream)?,
            ft_b: self.ft_b.to_host_vec(&self.stream)?,
            l1_w: self.l1_w.to_host_vec(&self.stream)?,
            l1_b: self.l1_b.to_host_vec(&self.stream)?,
            l1f_w: self.l1f_w.to_host_vec(&self.stream)?,
            l1f_b: self.l1f_b.to_host_vec(&self.stream)?,
            l2_w: self.l2_w.to_host_vec(&self.stream)?,
            l2_b: self.l2_b.to_host_vec(&self.stream)?,
            l3_w: self.l3_w.to_host_vec(&self.stream)?,
            l3_b: self.l3_b.to_host_vec(&self.stream)?,
        })
    }

    /// `ft_w` を **除く** 9 weight group の `(name, expected_len, &w, &m, &v, &slow)` を
    /// 固定順で返す (raw checkpoint の save/load で iterate するための immutable view)。
    /// `grad` は resume に不要なので含めない。順序 = ft_b, l1_w, l1_b, l1f_w, l1f_b,
    /// l2_w, l2_b, l3_w, l3_b。
    ///
    /// `ft_w` は `m` / `v` が `--fp16-opt-state` で `f16` ([`MomentBuf`]) になり buffer
    /// 型が他 group と揃わないため本配列から外し、checkpoint format 上 1 番目の group
    /// として save/load 側で個別に処理する (format の group 順は ft_w が先頭で不変)。
    #[allow(clippy::type_complexity)]
    pub(crate) fn raw_ckpt_groups(
        &self,
    ) -> [(
        &'static str,
        usize,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
    ); 9] {
        let ft_out = self.ws.ft_out;
        let ft_b_n = ft_out;
        let l1_w_n = NUM_BUCKETS * L1_OUT * ft_out;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = ft_out * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        [
            (
                "ft_b",
                ft_b_n,
                &self.ft_b,
                &self.ft_b_m,
                &self.ft_b_v,
                &self.ft_b_slow,
            ),
            (
                "l1_w",
                l1_w_n,
                &self.l1_w,
                &self.l1_w_m,
                &self.l1_w_v,
                &self.l1_w_slow,
            ),
            (
                "l1_b",
                l1_b_n,
                &self.l1_b,
                &self.l1_b_m,
                &self.l1_b_v,
                &self.l1_b_slow,
            ),
            (
                "l1f_w",
                l1f_w_n,
                &self.l1f_w,
                &self.l1f_w_m,
                &self.l1f_w_v,
                &self.l1f_w_slow,
            ),
            (
                "l1f_b",
                l1f_b_n,
                &self.l1f_b,
                &self.l1f_b_m,
                &self.l1f_b_v,
                &self.l1f_b_slow,
            ),
            (
                "l2_w",
                l2_w_n,
                &self.l2_w,
                &self.l2_w_m,
                &self.l2_w_v,
                &self.l2_w_slow,
            ),
            (
                "l2_b",
                l2_b_n,
                &self.l2_b,
                &self.l2_b_m,
                &self.l2_b_v,
                &self.l2_b_slow,
            ),
            (
                "l3_w",
                l3_w_n,
                &self.l3_w,
                &self.l3_w_m,
                &self.l3_w_v,
                &self.l3_w_slow,
            ),
            (
                "l3_b",
                l3_b_n,
                &self.l3_b,
                &self.l3_b_m,
                &self.l3_b_v,
                &self.l3_b_slow,
            ),
        ]
    }

    /// `--resume` 用 **raw f32 checkpoint** を atomic に書き出す。
    ///
    /// 量子化 `.bin` ([`GpuTrainer::save_checkpoint`]/`to_layerstack_weights` → `save_quantised`)
    /// は推論用 final artifact として別 method で保存される。本 method はそれとは別の
    /// `*.ckpt` file に、全 10 weight group の **raw f32** `{w, m, v, slow}` (Ranger の
    /// 1st/2nd moment + Lookahead slow weight、`grad` は resume に不要なので含めない) +
    /// `step_count` (Ranger lookahead step counter) + 完了 `superbatch` 番号を書き出す。
    ///
    /// header の write / read は [`write_raw_ckpt_header`] / [`read_raw_ckpt_header`]
    /// に切り出してある。layout (全 little-endian、現行 [`RAW_CKPT_VERSION`] = 4):
    /// ```text
    /// magic        b"RNRC"             (4 bytes)
    /// version      u32 (4)             (4 bytes)
    /// fs_name_len  u32                 (4 bytes、feature set canonical 名の長さ)
    /// fs_name      UTF-8 [fs_name_len]  (feature set canonical 名、例 "halfka-hm-merged")
    /// ft_in        u64                 (FT 入力次元、feature set 依存)
    /// ft_out       u64                 (FT 出力次元、`--ft-out`)
    /// max_active   u64                 (1 perspective あたり active feature 数)
    /// run_id_len   u32                 (4 bytes、producer run id の長さ、0 可)
    /// run_id       UTF-8 [run_id_len]   (この checkpoint を書いた run の experiment.json `id`)
    /// arch_len     u32                 (4 bytes、arch kind canonical 名の長さ)
    /// arch_kind    UTF-8 [arch_len]     (arch kind canonical 名、LayerStack は "layerstack")
    /// topo_count   u64                 (topology 次元の個数)
    /// topology     u64 [topo_count]     (層次元列、LayerStack は ft_out/L1_OUT/L2_OUT/NUM_BUCKETS)
    /// superbatch   u64  (この checkpoint が表す完了 superbatch、resume はこの +1 から)
    /// step_count   u64  (Ranger lookahead step counter)
    /// num_groups   u64  (= 10、固定だが将来検証用)
    /// then for each of 10 groups (順序 = `raw_ckpt_groups()` = ft_w, ft_b, l1_w, l1_b,
    ///   l1f_w, l1f_b, l2_w, l2_b, l3_w, l3_b):
    ///   len u64
    ///   w[f32 × len]
    ///   m[f32 × len]
    ///   v[f32 × len]
    ///   slow[f32 × len]
    /// ```
    ///
    /// version 1 file には feature set header も run id も arch header も無く、weights
    /// は常に `halfka-hm-merged` / `layerstack` として解釈される。version 2/3 file は
    /// arch header を持たず `layerstack` 扱い。writer は常に最新 version を書く。
    ///
    /// device → host download (`DeviceBuffer::to_host_vec`) → `<path>.tmp` へ `BufWriter`
    /// で書く → `std::fs::rename(<path>.tmp, <path>)` で atomic に置換 (書き込み途中で
    /// crash しても `<path>` は前回の完全な checkpoint のまま)。
    ///
    /// `run_id` はこの checkpoint を書き出す run の experiment.json `id`。空文字列、
    /// または `MAX_RUN_ID_BYTES` 超過 (warning を出して省略) のときは run id を持た
    /// ない checkpoint になり、resume 時の `lineage.parent_id` は解決されない。
    pub(crate) fn save_raw_checkpoint(
        &self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        // 過長な run id (`{net_id}-{時刻}-{pid}`、通常数十バイト) は lineage という
        // メタデータのために学習を中断させる価値がない。上限超過時は埋め込みを
        // 省略 (長さ 0) し、warning を出して checkpoint 保存は続行する。
        let run_id = if run_id.len() > MAX_RUN_ID_BYTES {
            eprintln!(
                "[train] warning: producer run id ({} bytes) exceeds {MAX_RUN_ID_BYTES}; \
                 omitting it from {} (resume lineage parent will be unresolved)",
                run_id.len(),
                path.display()
            );
            ""
        } else {
            run_id
        };

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = {
            let mut p = path.as_os_str().to_os_string();
            p.push(".tmp");
            std::path::PathBuf::from(p)
        };

        // write+flush 本体を closure に括り、`fs::rename` 前の error path で
        // 中途半端な `<path>.tmp` を best-effort で消す (device→host download / write /
        // flush 失敗で残骸を残さないため)。
        let write_tmp = || -> Result<(), Box<dyn std::error::Error>> {
            let groups = self.raw_ckpt_groups();
            let ft_out = self.ws.ft_out;
            let topology = layerstack_topology(ft_out);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);
            // header (magic 〜 num_groups)。format 上の group 数は ft_w (個別処理) +
            // `raw_ckpt_groups` の 9 = 10。
            write_raw_ckpt_header(
                &mut w,
                &RawCkptArch {
                    feature_set: self.feature_set,
                    arch_kind: ArchKind::LayerStack,
                    ft_out: ft_out as u64,
                    topology: &topology,
                },
                run_id,
                superbatch as u64,
                self.step_count,
                (groups.len() + 1) as u64,
            )?;

            // group 0: ft_w。`m` / `v` は `--fp16-opt-state` で `f16` 格納だが、
            // checkpoint は常に真値 `f32` で書く (mode 非依存・format version 不変、
            // resume 時に当該 run の精度へ再 quantize される)。
            let ft_w_n = self.feature_set.ft_in() * ft_out;
            {
                let w_host = self.ft_w.to_host_vec(&self.stream)?;
                let m_host = self.ft_w_m.to_host_f32(&self.stream, FT_OPT_M_SCALE)?;
                let v_host = self.ft_w_v.to_host_f32(&self.stream, FT_OPT_V_SCALE)?;
                let slow_host = self.ft_w_slow.to_host_vec(&self.stream)?;
                for (label, got) in [
                    ("w", w_host.len()),
                    ("m", m_host.len()),
                    ("v", v_host.len()),
                    ("slow", slow_host.len()),
                ] {
                    if got != ft_w_n {
                        return Err(format!(
                            "raw checkpoint: group ft_w {label} buffer len {got} != expected {ft_w_n}"
                        )
                        .into());
                    }
                }
                w.write_all(&(ft_w_n as u64).to_le_bytes())?;
                write_f32_slice(&mut w, &w_host)?;
                write_f32_slice(&mut w, &m_host)?;
                write_f32_slice(&mut w, &v_host)?;
                write_f32_slice(&mut w, &slow_host)?;
            }

            for (name, expected_len, w_buf, m_buf, v_buf, slow_buf) in groups {
                // 念のため device buffer の要素数を arch 期待値と照合 (内部整合性)。
                let w_host = w_buf.to_host_vec(&self.stream)?;
                let m_host = m_buf.to_host_vec(&self.stream)?;
                let v_host = v_buf.to_host_vec(&self.stream)?;
                let slow_host = slow_buf.to_host_vec(&self.stream)?;
                for (label, got) in [
                    ("w", w_host.len()),
                    ("m", m_host.len()),
                    ("v", v_host.len()),
                    ("slow", slow_host.len()),
                ] {
                    if got != expected_len {
                        return Err(format!(
                            "raw checkpoint: group {name} {label} buffer len {got} != expected {expected_len}"
                        )
                        .into());
                    }
                }
                w.write_all(&(expected_len as u64).to_le_bytes())?;
                write_f32_slice(&mut w, &w_host)?;
                write_f32_slice(&mut w, &m_host)?;
                write_f32_slice(&mut w, &v_host)?;
                write_f32_slice(&mut w, &slow_host)?;
            }
            w.flush()?;
            Ok(())
        };
        if let Err(e) = write_tmp() {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        Ok(())
    }

    /// raw checkpoint を読み戻す (`--resume` 用)。返り値は `(完了 superbatch 番号,
    /// producer run id)` — superbatch は caller が通常その +1 から resume する。
    /// producer run id は version 3+ の checkpoint なら `Some` (resume run の
    /// `lineage.parent_id` に使う)、version 1/2 や run id 未記録なら `None`。
    ///
    /// magic 不一致、`version > 4`、arch kind / topology が LayerStack と不一致、group 数
    /// や各 group の len が LayerStack arch と不一致、または `u64 → usize` overflow
    /// (32-bit / 破損 file) は `InvalidData` で reject
    /// (`crates/nnue-train::optimizer::RangerHostState::load_from_reader` と同方針)。
    ///
    /// header の解析 (feature set / arch kind / topology の照合) は
    /// [`read_raw_ckpt_header`] が担当する。version 1 file は feature set header を
    /// 持たず weights を `halfka-hm-merged` とみなす。version 1..=3 は arch header を
    /// 持たず `layerstack` とみなす。読み込んだ raw f32 を host → device upload し、
    /// `self.step_count` を復元する。`grad` buffer は触らない (step ごとに memset される)。
    pub(crate) fn load_raw_checkpoint(
        &mut self,
        path: &Path,
    ) -> Result<(usize, Option<String>), Box<dyn std::error::Error>> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let ft_out = self.ws.ft_out;
        let topology = layerstack_topology(ft_out);

        // header (magic 〜 num_groups) を読み、feature set / arch / topology を照合する。
        let header = read_raw_ckpt_header(
            &mut r,
            &RawCkptArch {
                feature_set: self.feature_set,
                arch_kind: ArchKind::LayerStack,
                ft_out: ft_out as u64,
                topology: &topology,
            },
        )?;
        let superbatch = header.superbatch;
        let step_count = header.step_count;
        let producer_run_id = header.producer_run_id;

        // format 上の group 数は ft_w (個別処理) + `raw_ckpt_groups` の 9 = 10。
        let expected_groups: [(&'static str, usize); 9] = {
            let g = self.raw_ckpt_groups();
            [
                (g[0].0, g[0].1),
                (g[1].0, g[1].1),
                (g[2].0, g[2].1),
                (g[3].0, g[3].1),
                (g[4].0, g[4].1),
                (g[5].0, g[5].1),
                (g[6].0, g[6].1),
                (g[7].0, g[7].1),
                (g[8].0, g[8].1),
            ]
        };
        let total_groups = expected_groups.len() + 1;
        if header.num_groups != total_groups as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint num_groups {} != expected {total_groups}",
                header.num_groups
            )));
        }

        // 1 group 分 (len + w/m/v/slow の f32 × len) を読む helper。`expected_len` と
        // file 記載 len の不一致 / overflow は `InvalidData` で reject。
        let read_group = |r: &mut std::io::BufReader<std::fs::File>,
                          name: &str,
                          expected_len: usize|
         -> Result<RawCkptGroup, Box<dyn std::error::Error>> {
            let mut buf8 = [0u8; 8];
            read_exact_or_invalid(r, &mut buf8, &format!("group {name} len"))?;
            let len_u64 = u64::from_le_bytes(buf8);
            let len: usize = len_u64.try_into().map_err(|_| {
                invalid_data(format!(
                    "raw checkpoint group {name} len {len_u64} exceeds usize::MAX"
                ))
            })?;
            if len != expected_len {
                return Err(invalid_data(format!(
                    "raw checkpoint group {name} len mismatch: got {len}, want {expected_len} \
                     (network architecture mismatch)"
                )));
            }
            let w_host = read_f32_vec_io(r, len, &format!("group {name} w"))?;
            let m_host = read_f32_vec_io(r, len, &format!("group {name} m"))?;
            let v_host = read_f32_vec_io(r, len, &format!("group {name} v"))?;
            let slow_host = read_f32_vec_io(r, len, &format!("group {name} slow"))?;
            Ok((w_host, m_host, v_host, slow_host))
        };

        // 各 group を読み出し → host Vec に保持 (全部読んでから upload する。途中で
        // upload して途中 fail だと中途半端な state になるため)。group 0 は ft_w。
        let ft_w_loaded = read_group(&mut r, "ft_w", self.feature_set.ft_in() * ft_out)?;
        let mut loaded: Vec<RawCkptGroup> = Vec::with_capacity(expected_groups.len());
        for (name, expected_len) in expected_groups {
            loaded.push(read_group(&mut r, name, expected_len)?);
        }
        // EOF 確認 (trailing garbage は許容するが、足りないのは上で read_exact が弾く)。

        // host → device upload。ft_w の m / v は当該 run の精度 (`fp16_opt_state`) へ
        // 量子化して載せ直す (checkpoint は真値 f32、mode 非依存)。
        let (ftw_w, ftw_m, ftw_v, ftw_slow) = &ft_w_loaded;
        self.ft_w = DeviceBuffer::from_host(&self.stream, ftw_w)?;
        self.ft_w_m =
            MomentBuf::from_host_f32(&self.stream, ftw_m, self.fp16_opt_state, FT_OPT_M_SCALE)?;
        self.ft_w_v =
            MomentBuf::from_host_f32(&self.stream, ftw_v, self.fp16_opt_state, FT_OPT_V_SCALE)?;
        self.ft_w_slow = DeviceBuffer::from_host(&self.stream, ftw_slow)?;

        // 残り 9 group (順序は `raw_ckpt_groups` = ft_b, l1_w, ..., l3_b)。
        macro_rules! up {
            ($idx:expr, $w:ident, $m:ident, $v:ident, $slow:ident) => {{
                let (w, m, v, s) = &loaded[$idx];
                self.$w = DeviceBuffer::from_host(&self.stream, w)?;
                self.$m = DeviceBuffer::from_host(&self.stream, m)?;
                self.$v = DeviceBuffer::from_host(&self.stream, v)?;
                self.$slow = DeviceBuffer::from_host(&self.stream, s)?;
            }};
        }
        up!(0, ft_b, ft_b_m, ft_b_v, ft_b_slow);
        up!(1, l1_w, l1_w_m, l1_w_v, l1_w_slow);
        up!(2, l1_b, l1_b_m, l1_b_v, l1_b_slow);
        up!(3, l1f_w, l1f_w_m, l1f_w_v, l1f_w_slow);
        up!(4, l1f_b, l1f_b_m, l1f_b_v, l1f_b_slow);
        up!(5, l2_w, l2_w_m, l2_w_v, l2_w_slow);
        up!(6, l2_b, l2_b_m, l2_b_v, l2_b_slow);
        up!(7, l3_w, l3_w_m, l3_w_v, l3_w_slow);
        up!(8, l3_b, l3_b_m, l3_b_v, l3_b_slow);

        self.step_count = step_count;
        Ok((superbatch, producer_run_id))
    }

    /// 全 weight buffer を host に読み出して NaN/Inf がないことを assert する smoke 用 helper。
    pub(crate) fn assert_all_weights_finite(&self) -> Result<(), Box<dyn std::error::Error>> {
        let groups: [(&DeviceBuffer<f32>, &str); 10] = [
            (&self.ft_w, "ft_w"),
            (&self.ft_b, "ft_b"),
            (&self.l1_w, "l1_w"),
            (&self.l1_b, "l1_b"),
            (&self.l1f_w, "l1f_w"),
            (&self.l1f_b, "l1f_b"),
            (&self.l2_w, "l2_w"),
            (&self.l2_b, "l2_b"),
            (&self.l3_w, "l3_w"),
            (&self.l3_b, "l3_b"),
        ];
        for (buf, name) in groups {
            let v = buf.to_host_vec(&self.stream)?;
            for (i, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(format!(
                        "{name}[{i}] = {x} is not finite (NaN or Inf)、smoke fail"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// `--ft-fp16` の FP16 weight mirror (`ft_w_h`) を現在の `ft_w` から再生成する。
    ///
    /// 学習中の mirror は optimizer (`radam_step_fp16_mirror` /
    /// `ranger_lookahead_lerp_fp16_mirror`) が `ft_w` 更新と同時に書く。ただし学習
    /// 開始時は optimizer 未実行で mirror が初期 0 のままなので、最初の forward の前に
    /// 一度だけ明示同期する。`ft_fp16` 無効時 (`ft_w_h` が `None`) は no-op。
    pub(crate) fn sync_ft_w_h_mirror(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
            let ft_w_n = self.feature_set.ft_in() * self.ws.ft_out;
            cuda_launch! {
                kernel: cast_f32_to_f16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(ft_w_n),
                args: [slice(self.ft_w), slice_mut(ft_w_h), ft_w_n as u32]
            }?;
        }
        Ok(())
    }

    /// 1 batch 分の forward → loss kernel → backward → Ranger step を実行。
    /// 戻り値: batch 全体の loss (f64、loss_acc から読み出し)。
    ///
    /// 実体は [`GpuTrainer::step_impl`]。本 method は `NNUE_TRAIN_STEP_PROFILE`
    /// プロファイル時の前後 sync と **teardown tick** だけを担う。`step_impl` が
    /// return すると per-step device buffer の `Drop` (= `cuMemFree`) がそこで走るので、
    /// 最後の `prof_tick!` を `step_impl` の **外** で打つことで free 時間も breakdown に
    /// 含める。中間 activation / grad buffer は `GpuTrainer` 上の workspace に永続化
    /// しているので、`step_impl` で drop されるのは入力 H2D buffer (`stm_idx_dev` 等、
    /// position 数に比例した小さい buffer) だけになり、teardown tick は ~0 に落ちる。
    pub(crate) fn step(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // 環境変数 `NNUE_TRAIN_STEP_PROFILE` がセットされていれば各 phase の境界で
        // `synchronize()` + 経過時間を stderr に出す (粗い h2d / forward / backward /
        // optimizer / teardown breakdown 用)。未設定なら追加の sync ゼロ。
        // WSL2 では ncu の GPU perf counter が使えず nsys も GPU-side kernel trace を
        // 取れないため、この粗い event timing が代替手段。
        let profile_step = std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some();
        if profile_step {
            self.stream.synchronize()?;
        }
        let mut prof_t0 = std::time::Instant::now();
        let result = self.step_impl(
            batch,
            lr,
            wdl_lambda,
            loss,
            false,
            profile_step,
            &mut prof_t0,
        )?;
        // step_impl の per-step device buffer はここまでに全部 drop 済 (cuMemFree)。
        if profile_step {
            self.stream.synchronize()?;
            eprintln!(
                "[step-profile] {:<10} {:8.3} ms",
                "teardown",
                prof_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(result.loss)
    }

    /// held-out validation の 1 batch を実行する。[`GpuTrainer::step_impl`] を
    /// `validate = true` で呼び、forward + loss kernel のみ走らせる (backward /
    /// optimizer step は無く、weight も optimizer state も一切更新しない)。
    ///
    /// 戻り値 [`StepOutput`] は batch 全体の `Σ err²` (`loss`) と position ごとの
    /// net 出力スカラ (`net_output`)。caller (`TrainerBackend::validate_step`) が
    /// 前者から平均 loss、後者から sign-agreement accuracy を出す。
    ///
    /// 冒頭で `stream.synchronize` し直前の training step (optimizer まで) の完了を
    /// 待ってから検証 forward を始める。検証は superbatch あたり 1 回・~1 batch 分
    /// なので同期コストは無視できる。
    pub(crate) fn validate(
        &mut self,
        batch: &BatchData,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        // 直前の training step の GPU work 完了を待つ。検証 forward の H2D / kernel が
        // in-flight な training compute と input buffer を取り合わないことを保証する。
        self.stream.synchronize()?;
        let profile_step = std::env::var_os("NNUE_TRAIN_STEP_PROFILE").is_some();
        let mut prof_t0 = std::time::Instant::now();
        // lr は validate モードでは optimizer を呼ばないため未使用 (0.0 を渡す)。
        self.step_impl(
            batch,
            0.0,
            wdl_lambda,
            loss,
            true,
            profile_step,
            &mut prof_t0,
        )
    }

    /// `step` の実体。`loss` が [`LossKind::Sigmoid`] なら `loss_wdl` (plain sigmoid-MSE)、
    /// [`LossKind::Wrm`] なら `loss_wrm` (win-rate-model loss) を起動する。
    ///
    /// Forward path (15 step): bullet `shogi_layerstack.rs:2241-2289` の reference 実装を
    /// 本 file の `#[kernel]` 群で再現。中間 activation は `GpuTrainer` 上の永続 workspace
    /// (`self.ws.*`) を使い回す — forward の各 activation は読まれる前に kernel が
    /// 全 cell を上書きするので memset 不要。Backward path (~16 step): forward 逆順、`*_grad`
    /// buffer は本 method 冒頭で `memset_async(0)` で reset してから kernel が書き込む
    /// (per-bucket weight grad `dense_mm_bwd_weight_bucket` は 1 cell = 1 thread の overwrite、
    /// FT / L1f / bias の grad は atomic accumulate なので reset 必須。`dl1_total` も
    /// `slice_scatter_2d` の host 契約を守るため reset)。`loss_acc` も同様に毎 step memset。
    /// 入力 H2D buffer (`stm_idx_dev` 等) は workspace 上の pre-allocated buffer に
    /// async memcpy する。Optimizer: 10 weight groups × `radam_step` (+ 周期
    /// `ranger_lookahead_lerp`)。
    ///
    /// `profile_step` / `prof_t0` は呼び出し元 ([`GpuTrainer::step`]) が管理し、本 method
    /// 内の `prof_tick!` が各 phase 境界で `*prof_t0` を更新する (戻った後に呼び出し元が
    /// teardown tick で読む)。
    ///
    /// `validate == true` のときは **forward + loss kernel のみ**を実行し、loss kernel
    /// 直後に `loss_acc` と `net_output` を同期読み出しして early return する
    /// (backward / optimizer step は走らず weight は不変、held-out validation 用)。
    /// `validate == false` の通常 training path はこの分岐に入らないため、訓練の
    /// 数値挙動は本フラグ追加前と完全に同一。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_impl(
        &mut self,
        batch: &BatchData,
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
        validate: bool,
        profile_step: bool,
        prof_t0: &mut std::time::Instant,
    ) -> Result<StepOutput, Box<dyn std::error::Error>> {
        let b = batch.n_pos;
        if b == 0 {
            return Ok(StepOutput {
                loss: 0.0,
                net_output: Vec::new(),
            });
        }
        // defense-in-depth: tiled kernels (grid=b/16) は b % 16 == 0 を要求する。
        // CLI で `--batch-size` を 16 倍数に reject 済 (`run_training`)、`BucketedPrefetchedLoader`
        // も `n_positions == batch_size` を保証する (`dataloader.rs:572`) ため通常到達しない。
        // release で debug_assert! が消えるので、ここで `step_impl` 直入りされた場合の保険として
        // 明示的な runtime check を入れる。
        if !b.is_multiple_of(16) {
            return Err(format!(
                "batch.n_pos must be a multiple of 16 (got {}); tiled dense matmul kernels \
                 require b % 16 == 0 — partial last batch will silently truncate via grid=b/16",
                b
            )
            .into());
        }
        let b_u32 = b as u32;
        // FT 出力次元 (`--ft-out`)。post-FT buffer の幅と FT kernel の launch arg。
        let ft_out = self.ws.ft_out;

        // batch `b` が workspace 容量に収まることを検証する (固定 batch 前提、
        // 起動時の `GpuWorkspace::new` で確保済)。
        self.ws.check_batch_capacity(b)?;

        macro_rules! prof_tick {
            ($label:expr) => {
                if profile_step {
                    self.stream.synchronize()?;
                    let now = std::time::Instant::now();
                    eprintln!(
                        "[step-profile] {:<10} {:8.3} ms",
                        $label,
                        now.duration_since(*prof_t0).as_secs_f64() * 1000.0
                    );
                    *prof_t0 = now;
                }
            };
        }

        // 入力 5 buffer を host → device。active / back buffer を `mem::swap` してから
        // back 側 (= 直前 step が読んでいない物理 buffer) へ専用 copy stream で先行 H2D
        // する。H2D は直前 step の compute と並走し、compute stream は H2D 完了 event を
        // 待ってから forward に進む ([`InputUploadRing`])。pageable な dataloader `Vec`
        // は ring 内の pinned host buffer 経由で copy engine の DMA に載る。
        std::mem::swap(&mut self.ws.stm_idx_dev, &mut self.ws.stm_idx_dev_back);
        std::mem::swap(&mut self.ws.nstm_idx_dev, &mut self.ws.nstm_idx_dev_back);
        std::mem::swap(
            &mut self.ws.bucket_idx_dev,
            &mut self.ws.bucket_idx_dev_back,
        );
        std::mem::swap(&mut self.ws.score_dev, &mut self.ws.score_dev_back);
        std::mem::swap(&mut self.ws.wdl_dev, &mut self.ws.wdl_dev_back);
        self.input_ring.upload(
            &self.stream,
            &self.ws.stm_idx_dev,
            batch.stm_indices,
            &self.ws.nstm_idx_dev,
            batch.nstm_indices,
            &self.ws.bucket_idx_dev,
            batch.bucket_idx,
            &self.ws.score_dev,
            batch.score,
            &self.ws.wdl_dev,
            batch.wdl,
        )?;
        // per_pos_norm は scalar (1/n_pos) として直接 kernel arg に渡す。

        // loss_acc reset (accumulate semantics、再 alloc せず memset)
        memset_zero(&self.stream, &self.loss_acc)?;
        prof_tick!("h2d+reset");

        // -- Forward step 1-2: sparse_ft_forward × 2 (stm, nstm) --
        // 中間 activation は workspace (`self.ws.*`) を使い回す (再 alloc 無し)。
        // forward の各 activation は読まれる前に kernel が全 cell を上書きするので memset 不要。
        // sparse_ft_forward は 1 thread = 4 row (output cell) なので grid は b * ft_out / 4。
        // ft_out は `--ft-out` 検証で 128 の倍数 (= 4 の倍数) を保証済。
        // forward kernel は 3 通り:
        //  - `ft_fp16_out`: `sparse_ft_forward_fp16_out` — f16 weight read + f16 出力
        //    (`ft_*_out_h`)。書き出し DRAM 帯域も半減。
        //  - `ft_fp16` のみ: `sparse_ft_forward_fp16` — f16 weight read + f32 出力。
        //  - どちらも無し: `sparse_ft_forward` — FP32 path、bit-identical。
        // いずれも累算は f32、1 thread = 4 row なので grid は b * ft_out / 4。
        debug_assert!(ft_out.is_multiple_of(4));
        if self.ft_fp16_out {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16 is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16_out,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out_h.as_mut()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled")),
                    b_u32, ft_out as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward_fp16_out,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out_h.as_mut()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    b_u32, ft_out as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        } else if self.ft_fp16 {
            let ft_w_h = self
                .ft_w_h
                .as_ref()
                .expect("ft_w_h is Some when ft_fp16 is enabled");
            cuda_launch! {
                kernel: sparse_ft_forward_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out),
                    b_u32, ft_out as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 4),
                args: [
                    slice(ft_w_h),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out),
                    b_u32, ft_out as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: sparse_ft_forward,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 4),
                args: [
                    slice(self.ft_w),
                    slice(self.ws.stm_idx_dev),
                    slice_mut(self.ws.ft_stm_out),
                    b_u32, ft_out as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
            cuda_launch! {
                kernel: sparse_ft_forward,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 4),
                args: [
                    slice(self.ft_w),
                    slice(self.ws.nstm_idx_dev),
                    slice_mut(self.ws.ft_nstm_out),
                    b_u32, ft_out as u32, self.ws.ft_in as u32, self.ws.max_active as u32
                ]
            }?;
        }
        prof_tick!("fwd_ft");

        // -- Forward step 3: ft_post_perspective_fwd → combined (B × ft_out) --
        // `ft_fp16_out` 時は f16 入力版 (`ft_post_perspective_fwd_fp16`)。`combined` 出力は
        // 両 path とも f32 (後続 dense L1 path が f32 で読む)。
        if self.ft_fp16_out {
            cuda_launch! {
                kernel: ft_post_perspective_fwd_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out),
                args: [
                    slice(self.ws.ft_stm_out_h.as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ws.ft_nstm_out_h.as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b),
                    slice_mut(self.ws.combined),
                    b_u32, ft_out as u32, FT_POST_SCALE
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: ft_post_perspective_fwd,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out),
                args: [
                    slice(self.ws.ft_stm_out),
                    slice(self.ws.ft_nstm_out),
                    slice(self.ft_b),
                    slice_mut(self.ws.combined),
                    b_u32, ft_out as u32, FT_POST_SCALE
                ]
            }?;
        }

        prof_tick!("fwd_ftpost");

        // Forward L1: bucket sort で row を bucket_idx 昇順に並べ替え、各 bucket の sorted
        // 開始 offset を TILE_B=16 境界に align してから `dense_mm_fwd_bucket_tiled_l1_sorted`
        // を 1-bucket-per-block で走らせる (per-K-tile の W_TILE shared-mem load は 1 bucket
        // 分のみ)。inverse permute で `l1_bucket` を original order に戻して後続に渡す。
        // 数値同等性: fwd_L1 は per-row independent (k=0..15 加算順保持) のため sort stability
        // に依らず baseline と bit-exact。
        debug_assert!(
            ft_out.is_multiple_of(16) && L1_OUT == 16 && NUM_BUCKETS == 9 && b.is_multiple_of(16)
        );

        // a) histogram + 16-aligned scan + scatter。aligned offset で各 bucket が 16-row
        // 境界に整列し、bucket 末端 / 次 bucket 開始間に padding 行ができる。padding 行は
        // bucket=-1 で initialise (sorted kernel 側で skip)、perm も -1 sentinel (inverse
        // permute が skip)。
        let padded_b = padded_sort_batch(b);
        memset_zero(&self.stream, &self.ws.bucket_counts_dev)?;
        memset_zero(&self.stream, &self.ws.bucket_write_ctr_dev)?;
        memset_minus_one_i32(&self.stream, &self.ws.bucket_perm_dev)?;
        memset_minus_one_i32(&self.stream, &self.ws.bucket_idx_sorted_dev)?;
        cuda_launch! {
            kernel: count_buckets,
            stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.bucket_idx_dev),
                slice(self.ws.bucket_counts_dev),
                b_u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: exclusive_scan_aligned,
            stream: self.stream, module: self.module,
            config: LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.bucket_counts_dev),
                slice(self.ws.bucket_offsets_dev),
                (NUM_BUCKETS + 1) as u32,
                16_u32
            ]
        }?;
        cuda_launch! {
            kernel: scatter_bucket_perm,
            stream: self.stream, module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.bucket_idx_dev),
                slice(self.ws.bucket_offsets_dev),
                slice(self.ws.bucket_write_ctr_dev),
                slice(self.ws.bucket_perm_dev),
                slice(self.ws.bucket_idx_sorted_dev),
                b_u32, NUM_BUCKETS as u32
            ]
        }?;

        // b) combined を perm で gather → combined_sorted。padding 行 (perm=-1) は
        // permute kernel が 0 fill (sorted kernel 側で bucket=-1 で skip するので値不問)。
        cuda_launch! {
            kernel: permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * ft_out),
            args: [
                slice(self.ws.combined),
                slice(self.ws.bucket_perm_dev),
                slice_mut(self.ws.combined_sorted),
                padded_b as u32, ft_out as u32
            ]
        }?;

        // c) sorted fwd_L1 → l1_bucket_sorted。padded_b/16 block、各 block uniform 保証。
        cuda_launch! {
            kernel: dense_mm_fwd_bucket_tiled_l1_sorted,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: ((padded_b / 16) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.combined_sorted),
                slice(self.l1_w),
                slice(self.l1_b),
                slice(self.ws.bucket_idx_sorted_dev),
                slice_mut(self.ws.l1_bucket_sorted),
                padded_b as u32, ft_out as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // d) l1_bucket_sorted を perm で inverse-scatter → l1_bucket (original order)。
        // padding 行 (perm=-1) は inverse permute kernel が skip。
        cuda_launch! {
            kernel: inverse_permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L1_OUT),
            args: [
                slice(self.ws.l1_bucket_sorted),
                slice(self.ws.bucket_perm_dev),
                slice(self.ws.l1_bucket),
                padded_b as u32, L1_OUT as u32
            ]
        }?;

        prof_tick!("fwd_L1");

        // -- Forward step 5: L1f shared dense → l1f_out (B × L1_OUT) --
        // cuBLAS Sgemm (TF32 TC) で matmul、`bias_add_per_row` kernel で bias を別 pass。
        // shape: combined[B, ft_out] @ l1f_w[ft_out, L1_OUT] → l1f_out[B, L1_OUT]。
        //
        // SAFETY: combined / l1f_w / l1f_out は cudaMalloc 由来、長さは arch 上 invariant
        // (combined.len() == B*ft_out、l1f_w.len() == ft_out*L1_OUT、l1f_out.len() == B*L1_OUT)、
        // `self.cublas` は `self.stream` に bind 済で同 stream 内 in-order 実行 (先行 kernel
        // 完了後に Sgemm が走り、結果は後続 bias_add_per_row が観測)。
        unsafe {
            self.cublas.sgemm_fwd_rowmajor(
                b_u32 as i32,  // m = batch
                L1_OUT as i32, // n = out_dim
                ft_out as i32, // k = in_dim
                self.ws.combined.cu_deviceptr() as *const f32,
                self.l1f_w.cu_deviceptr() as *const f32,
                self.ws.l1f_out.cu_deviceptr() as *mut f32,
            )?;
        }
        cuda_launch! {
            kernel: bias_add_per_row,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.l1f_b),
                slice_mut(self.ws.l1f_out),
                b_u32, L1_OUT as u32
            ]
        }?;

        prof_tick!("fwd_L1f");

        // -- Forward step 6: l1_total = l1_bucket + l1f_out (B × L1_OUT) --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.l1_bucket),
                slice(self.ws.l1f_out),
                slice_mut(self.ws.l1_total),
                (b * L1_OUT) as u32
            ]
        }?;

        // -- Forward step 7: slice l1_total → l1_main (B × 15) + l1_skip (B × 1) --
        cuda_launch! {
            kernel: slice_extract_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.l1_total),
                slice_mut(self.ws.l1_main),
                b_u32, L1_OUT as u32, 0_u32, L1_EFFECTIVE as u32
            ]
        }?;
        cuda_launch! {
            kernel: slice_extract_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_SKIP),
            args: [
                slice(self.ws.l1_total),
                slice_mut(self.ws.l1_skip),
                b_u32, L1_OUT as u32, L1_EFFECTIVE as u32, L1_SKIP as u32
            ]
        }?;

        // -- Forward step 8: l1_sqr = l1_main^2 * scale (B × 15) --
        cuda_launch! {
            kernel: abs_pow2_scale_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.l1_main),
                slice_mut(self.ws.l1_sqr),
                L1_SQR_SCALE,
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Forward step 9: l2_pre = concat(l1_sqr, l1_main) (B × 30) --
        cuda_launch! {
            kernel: concat_l1sqr_main_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.l1_sqr),
                slice(self.ws.l1_main),
                slice_mut(self.ws.l2_pre),
                b_u32, L1_EFFECTIVE as u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Forward step 10: l2_input = CReLU(l2_pre) (B × 30) --
        cuda_launch! {
            kernel: crelu_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.l2_pre),
                slice_mut(self.ws.l2_input),
                (b * L2_IN) as u32
            ]
        }?;

        prof_tick!("fwd_L1tail");

        // -- Forward step 11: L2 per-bucket dense → l2_out (B × 32) --
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.l2_input),
                slice(self.l2_w),
                slice(self.l2_b),
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.l2_out),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 12: l2_acted = CReLU(l2_out) (B × 32) --
        cuda_launch! {
            kernel: crelu_fwd,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.l2_out),
                slice_mut(self.ws.l2_acted),
                (b * L2_OUT) as u32
            ]
        }?;

        prof_tick!("fwd_L2");

        // -- Forward step 13: L3 per-bucket dense → l3_out (B × 1) --
        cuda_launch! {
            kernel: dense_mm_fwd_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.l2_acted),
                slice(self.l3_w),
                slice(self.l3_b),
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.l3_out),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        // -- Forward step 14: net_output = l3_out + l1_skip (B × 1) --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.l3_out),
                slice(self.ws.l1_skip),
                slice_mut(self.ws.net_output),
                b_u32
            ]
        }?;

        // -- Forward step 15: loss kernel → dy_net_output + loss_acc --
        // `LossKind::Sigmoid` → `loss_wdl` (plain sigmoid-MSE)、`LossKind::Wrm` →
        // `loss_wrm` (win-rate-model loss)。
        match loss {
            LossKind::Sigmoid { scale } => {
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
                        wdl_lambda, scale, b_u32
                    ]
                }?;
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
                target_offset,
                target_scaling,
            } => {
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
                        wdl_lambda, nnue2score, in_scaling,
                        target_offset, target_scaling, b_u32
                    ]
                }?;
            }
        }
        prof_tick!("forward");

        // held-out validation: backward / optimizer をスキップし、loss kernel が
        // 書いた `loss_acc` (batch の Σ err²) と `net_output` (position ごとの net
        // 出力) を同期読み出しして early return する。weight も optimizer state も
        // 更新しない。`net_output` workspace は固定 batch 容量で確保されているため
        // 有効 position 数 `b` で truncate する。`to_host_vec` は内部で
        // `stream.synchronize` するので forward kernel 完了後の値が読める。
        if validate {
            let loss = self.loss_acc.to_host_vec(&self.stream)?[0];
            let mut net_output = self.ws.net_output.to_host_vec(&self.stream)?;
            net_output.truncate(b);
            prof_tick!("validate_io");
            return Ok(StepOutput { loss, net_output });
        }

        // ===== BACKWARD =====
        // 全 *_grad buffer を 0 で reset (atomic accumulate semantic に従う kernel が
        // 多い、また overwrite kernel も in-place 安全のため統一)。再 alloc せず
        // `memset_async(0)` で既存 buffer を reset (`ft_w_grad` だけで ~450MB の
        // `cudaMalloc`/`cudaFree` を毎 step 走らせるのを避けるため)。
        // `dl1_total` も `slice_scatter_2d` の host 契約 (「dst を 0 初期化」) を守るため reset。
        let ft_w_n = self.feature_set.ft_in() * ft_out;
        let ft_b_n = ft_out;
        let l1_w_n = NUM_BUCKETS * L1_OUT * ft_out;
        let l1_b_n = NUM_BUCKETS * L1_OUT;
        let l1f_w_n = ft_out * L1_OUT;
        let l1f_b_n = L1_OUT;
        let l2_w_n = NUM_BUCKETS * L2_OUT * L2_IN;
        let l2_b_n = NUM_BUCKETS * L2_OUT;
        let l3_w_n = NUM_BUCKETS * L2_OUT;
        let l3_b_n = NUM_BUCKETS;
        // ft_w_grad の memset_zero は意図的に省略している: phase D iter 0 (stm) の
        // `gather_and_sum_per_feature_overwrite` が全 (feature, ri) cell を sum
        // (off_start==off_end の時も sum=0) で書き切るため、ここで 450MB を reset
        // するのは無意味 (毎 step の no-op を排除する論理整理)。
        memset_zero(&self.stream, &self.ft_b_grad)?;
        memset_zero(&self.stream, &self.l1_w_grad)?;
        memset_zero(&self.stream, &self.l1_b_grad)?;
        memset_zero(&self.stream, &self.l1f_w_grad)?;
        memset_zero(&self.stream, &self.l1f_b_grad)?;
        memset_zero(&self.stream, &self.l2_w_grad)?;
        memset_zero(&self.stream, &self.l2_b_grad)?;
        memset_zero(&self.stream, &self.l3_w_grad)?;
        memset_zero(&self.stream, &self.l3_b_grad)?;
        memset_zero(&self.stream, &self.ws.dl1_total)?;
        prof_tick!("bwd_reset");

        // -- Backward 14 reverse: dy_net_output が dl3_out と dl1_skip 両方の grad --
        // (elementwise_add 逆: dl3_out = dy, dl1_skip = dy、両者同じ buffer を直接渡せばよい)

        // -- Backward 13 reverse: L3 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.dy_net_output),
                slice(self.l3_w),
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.dl2_acted),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;
        // L3 weight bwd: in_dim=L2_OUT=32, out_dim=1, num_buckets=9。
        // 元 kernel は 288 cells × scan batch で並列度小。split-K + 9 bucket register
        // accumulator (`dense_mm_bwd_weight_bucket_tiled_l3`) に切替。
        // num_splits=64 → 64 blocks × 32 threads = 2048 threads ≈ 26 / SM (sm_86)。
        const _: () = assert!(L2_OUT == 32 && NUM_BUCKETS == 9);
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket_tiled_l3,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (64, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.l2_acted),
                slice(self.ws.dy_net_output),
                slice(self.ws.bucket_idx_dev),
                slice(self.l3_w_grad),
                b_u32, L2_OUT as u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b),
            args: [
                slice(self.ws.dy_net_output),
                slice(self.ws.bucket_idx_dev),
                slice(self.l3_b_grad),
                b_u32, 1_u32, NUM_BUCKETS as u32
            ]
        }?;

        prof_tick!("bwd_L3");

        // -- Backward 12 reverse: crelu_grad on l2_out --
        cuda_launch! {
            kernel: crelu_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_OUT),
            args: [
                slice(self.ws.l2_out),
                slice(self.ws.dl2_acted),
                slice_mut(self.ws.dl2_out),
                (b * L2_OUT) as u32
            ]
        }?;

        // -- Backward 11 reverse: L2 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.dl2_out),
                slice(self.l2_w),
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.dl2_input),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        // L2 weight bwd: in_dim=L2_IN=30, out_dim=L2_OUT=32, num_buckets=9。
        // split-K + 9 bucket register accumulator (block_dim = 32 × 30 = 960、grid = 64 splits)。
        const _: () = assert!(L2_IN == 30 && L2_OUT == 32 && NUM_BUCKETS == 9);
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket_tiled_l2,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (64, 1, 1),
                block_dim: ((L2_OUT * L2_IN) as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.l2_input),
                slice(self.ws.dl2_out),
                slice(self.ws.bucket_idx_dev),
                slice(self.l2_w_grad),
                b_u32, L2_IN as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        // L2 bias backward (sorted): dl2_out を bucket_perm_dev で gather → dl2_out_sorted、
        // per-block shared-mem reduce で global atomic 数を ~2M → ~131K (16× 削減)。
        // out_dim=32、block(256) = 8 sorted 行 × 32 oi cell、16-aligned sort で uniform-
        // bucket。fwd_L1 で構築済の bucket_perm_dev / bucket_idx_sorted_dev を再利用。
        cuda_launch! {
            kernel: permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L2_OUT),
            args: [
                slice(self.ws.dl2_out),
                slice(self.ws.bucket_perm_dev),
                slice_mut(self.ws.dl2_out_sorted),
                padded_b as u32, L2_OUT as u32
            ]
        }?;
        cuda_launch! {
            kernel: bias_grad_bucket_shared_sorted,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L2_OUT),
            args: [
                slice(self.ws.dl2_out_sorted),
                slice(self.ws.bucket_idx_sorted_dev),
                slice(self.l2_b_grad),
                padded_b as u32, L2_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        prof_tick!("bwd_L2");

        // -- Backward 10 reverse: crelu_grad on l2_pre --
        cuda_launch! {
            kernel: crelu_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L2_IN),
            args: [
                slice(self.ws.l2_pre),
                slice(self.ws.dl2_input),
                slice_mut(self.ws.dl2_pre),
                (b * L2_IN) as u32
            ]
        }?;

        // -- Backward 9 reverse: split dl2_pre → dl1_sqr (15) + dl1_main_from_concat (15) --
        cuda_launch! {
            kernel: concat_l1sqr_main_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.dl2_pre),
                slice_mut(self.ws.dl1_sqr),
                slice_mut(self.ws.dl1_main_from_concat),
                b_u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Backward 8 reverse: abs_pow2_scale_grad (l1_sqr 経由の grad) --
        cuda_launch! {
            kernel: abs_pow2_scale_grad,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.l1_main),
                slice(self.ws.dl1_sqr),
                slice_mut(self.ws.dl1_main_from_sqr),
                L1_SQR_SCALE,
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Combine dl1_main = dl1_main_from_concat + dl1_main_from_sqr --
        cuda_launch! {
            kernel: elementwise_add,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.dl1_main_from_concat),
                slice(self.ws.dl1_main_from_sqr),
                slice_mut(self.ws.dl1_main),
                (b * L1_EFFECTIVE) as u32
            ]
        }?;

        // -- Backward 7 reverse: assemble dl1_total from dl1_main (offset 0) + dl1_skip=dy_net_output (offset 15) --
        cuda_launch! {
            kernel: slice_scatter_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_EFFECTIVE),
            args: [
                slice(self.ws.dl1_main),
                slice_mut(self.ws.dl1_total),
                b_u32, L1_EFFECTIVE as u32, L1_OUT as u32, 0_u32
            ]
        }?;
        cuda_launch! {
            kernel: slice_scatter_2d,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_SKIP),
            args: [
                slice(self.ws.dy_net_output),
                slice_mut(self.ws.dl1_total),
                b_u32, L1_SKIP as u32, L1_OUT as u32, L1_EFFECTIVE as u32
            ]
        }?;

        // -- Backward 6 reverse: dl1_total を l1_bucket と l1f_out 両方の grad に流す --
        // (elementwise_add 逆: dl1_bucket = dl1_total, dl1f = dl1_total)
        // 直接 dl1_total を両 dense_mm_bwd に渡す

        prof_tick!("bwd_L1eff");

        // -- Backward 5 reverse: L1f shared dense grad --
        // L1f input bwd: in_dim=ft_out, out_dim=L1_OUT=16, batch=multiple of 16
        // → tiled (block=256 = 16 batch × 16 in_dim cells、grid=batch/16 × in_dim/16).
        debug_assert!(b.is_multiple_of(16) && ft_out.is_multiple_of(16) && L1_OUT == 16);
        cuda_launch! {
            kernel: dense_mm_bwd_input_tiled,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (((b / 16) * (ft_out / 16)) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1f_w),
                slice_mut(self.ws.dcombined_from_l1f),
                b_u32, ft_out as u32, L1_OUT as u32
            ]
        }?;
        // L1f weight backward: row-major `grad_w[ft_out, L1_OUT] = combined^T @ dl1_total`。
        // combined[batch, ft_out] row-major、dl1_total[batch, L1_OUT] row-major、reduce 軸は
        // batch。M = L1_OUT と細いが K が大きい reduce-bound shape は cuBLAS Sgemm の
        // split-K + tensor pipeline 最適化が効きやすい。
        //
        // SAFETY: combined / dl1_total / l1f_w_grad は cudaMalloc 由来、長さは arch 上
        // invariant (`combined.len() == b*ft_out`、`dl1_total.len() == b*L1_OUT`、
        // `l1f_w_grad.len() == ft_out*L1_OUT`)、`self.cublas` は `self.stream` に bind 済で
        // 同 stream 内 in-order 実行 (先行 kernel 完了後に Sgemm が走り、結果は後続 kernel
        // が観測する)。
        unsafe {
            self.cublas.sgemm_xt_y_rowmajor(
                ft_out as i32, // m = in_dim
                L1_OUT as i32, // n = out_dim
                b_u32 as i32,  // k = batch
                self.ws.combined.cu_deviceptr() as *const f32,
                self.ws.dl1_total.cu_deviceptr() as *const f32,
                self.l1f_w_grad.cu_deviceptr() as *mut f32,
            )?;
        }
        // L1f bias backward: shared-mem reduce で global atomic を 1M → ~16K に削減。
        const _: () = assert!(L1_OUT == 16);
        cuda_launch! {
            kernel: bias_grad_shared_l1f,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * L1_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1f_b_grad),
                b_u32, L1_OUT as u32
            ]
        }?;

        prof_tick!("bwd_L1f");

        // -- Backward 4 reverse: L1 per-bucket dense grad --
        cuda_launch! {
            kernel: dense_mm_bwd_input_bucket,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(b * ft_out),
            args: [
                slice(self.ws.dl1_total),
                slice(self.l1_w),
                slice(self.ws.bucket_idx_dev),
                slice_mut(self.ws.dcombined_from_l1),
                b_u32, ft_out as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        prof_tick!("bwd_L1_inB");
        // L1 weight backward (sorted layout): combined_sorted は fwd_L1 で構築済、dl1_total を
        // 同 perm で gather → dl1_total_sorted。bucket_offsets_dev も fwd_L1 で構築済。各 block
        // は uniform-by-construction で 1 bucket の slice のみ accumulate (9-way if-else /
        // 9 register accumulator / 9 atomicAdd を 1 個ずつに集約)。
        debug_assert!(
            ft_out.is_multiple_of(16) && L1_OUT == 16 && NUM_BUCKETS == 9 && b.is_multiple_of(16)
        );
        cuda_launch! {
            kernel: permute_rows_f32,
            stream: self.stream, module: self.module,
            config: cfg_1d(padded_b * L1_OUT),
            args: [
                slice(self.ws.dl1_total),
                slice(self.ws.bucket_perm_dev),
                slice_mut(self.ws.dl1_total_sorted),
                padded_b as u32, L1_OUT as u32
            ]
        }?;
        // split-K dim を grid_y に追加。num_splits=8 × NUM_BUCKETS=9 × in_tiles=96 = 6912 blocks。
        cuda_launch! {
            kernel: dense_mm_bwd_weight_bucket_tiled_l1_sorted,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: ((ft_out / 16) as u32, 8, NUM_BUCKETS as u32),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                slice(self.ws.combined_sorted),
                slice(self.ws.dl1_total_sorted),
                slice(self.ws.bucket_offsets_dev),
                slice(self.l1_w_grad),
                padded_b as u32, ft_out as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;
        prof_tick!("bwd_L1_wB");
        // L1 bias: sorted layout で per-block shared-mem reduce、global atomic 数を
        // ~1M → ~66K に削減。dl1_total_sorted / bucket_idx_sorted_dev は同 step 内で
        // 構築済 (fwd_L1 + bwd_L1_wB 前 permute)。
        cuda_launch! {
            kernel: bias_grad_bucket_shared_sorted,
            stream: self.stream,
            module: self.module,
            config: cfg_1d(padded_b * L1_OUT),
            args: [
                slice(self.ws.dl1_total_sorted),
                slice(self.ws.bucket_idx_sorted_dev),
                slice(self.l1_b_grad),
                padded_b as u32, L1_OUT as u32, NUM_BUCKETS as u32
            ]
        }?;

        prof_tick!("bwd_L1");

        // dft (FT activation gradient) FP16 化の loss scaling 係数。dft ∝ 1/batch なので
        // batch 比例にして batch 非依存に f16 域へ載せる ([`FT_DFT_FP16_BASE_SCALE`])。
        // grad kernel が `* dft_scale` で書き、gather kernel が `* dft_inv_scale` で戻す。
        let dft_scale = FT_DFT_FP16_BASE_SCALE * (b as f32);
        let dft_inv_scale = 1.0_f32 / dft_scale;

        // -- Backward 3 reverse: ft_post_perspective_grad fused × 2 (stm, nstm) --
        // `dy = dcombined_from_l1 + dcombined_from_l1f` を fused kernel が in-register
        // で計算、合算済 buffer の materialize と read-back の DRAM roundtrip を避ける。
        // `ft_fp16_out` 時は forward activation `ft_*_out` を f16 で読み、dft 出力も f16
        // で書く版 (`ft_post_perspective_grad_fused_fp16`)。`d_combined_*` / `ft_b` /
        // `ft_b_grad` は両 path とも f32。stm: d_combined_offset = 0、nstm: = ft_out/2。
        if self.ft_fp16_out {
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_stm_out_h.as_ref()
                        .expect("ft_stm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_stm_out_h.as_mut()
                        .expect("dft_stm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b_grad),
                    b_u32, ft_out as u32, 0_u32, ft_out as u32, FT_POST_SCALE,
                    dft_scale
                ]
            }?;
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused_fp16,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_nstm_out_h.as_ref()
                        .expect("ft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_nstm_out_h.as_mut()
                        .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled")),
                    slice(self.ft_b_grad),
                    b_u32, ft_out as u32, (ft_out / 2) as u32, ft_out as u32, FT_POST_SCALE,
                    dft_scale
                ]
            }?;
        } else {
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_stm_out),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_stm_out),
                    slice(self.ft_b_grad),
                    b_u32, ft_out as u32, 0_u32, ft_out as u32, FT_POST_SCALE
                ]
            }?;
            cuda_launch! {
                kernel: ft_post_perspective_grad_fused,
                stream: self.stream,
                module: self.module,
                config: cfg_1d(b * ft_out / 2),
                args: [
                    slice(self.ws.dcombined_from_l1),
                    slice(self.ws.dcombined_from_l1f),
                    slice(self.ws.ft_nstm_out),
                    slice(self.ft_b),
                    slice_mut(self.ws.dft_nstm_out),
                    slice(self.ft_b_grad),
                    b_u32, ft_out as u32, (ft_out / 2) as u32, ft_out as u32, FT_POST_SCALE
                ]
            }?;
        }

        prof_tick!("bwd_ftpost");

        // -- Backward 1+2 reverse: sparse_ft_backward × 2 を inverse-index pipeline で実装。
        // 各 (b, ri) thread が直接 38 atomic add する素朴版は atomic contention で memory
        // bandwidth が飽和するため、phase A (count) → B (prefix sum) → C (scatter) →
        // D (per-feature gather+sum) の inverse-index 構成にして atomic を D だけに局所化する。
        // ft_w_grad は host が memset_zero 済、phase D は atomic add で stm/nstm を合算。
        // `gout` (dft) は phase D でのみ使うため loop は idx_dev のみで回し、phase D で
        // iter_idx に対応する dft buffer を選ぶ (`ft_fp16_out` 時は f16 版)。
        // feature set 依存の次元を loop 前に読み出す (per-iter の field 借用を避ける)。
        let ft_in = self.ws.ft_in;
        let max_active = self.ws.max_active;
        let total_pairs = (b * max_active) as u32;
        for (iter_idx, idx_dev) in [&self.ws.stm_idx_dev, &self.ws.nstm_idx_dev]
            .into_iter()
            .enumerate()
        {
            // A: feat_counts ← 0
            memset_zero(&self.stream, &self.ws.feat_counts)?;
            memset_zero(&self.stream, &self.ws.feat_write_ctr)?;
            prof_tick!("phA_reset");
            // A: build_feature_counts
            cuda_launch! {
                kernel: build_feature_counts,
                stream: self.stream, module: self.module,
                config: cfg_1d(b * max_active),
                args: [
                    slice(idx_dev),
                    slice(self.ws.feat_counts),
                    b_u32, max_active as u32, ft_in as u32
                ]
            }?;
            prof_tick!("phA_count");
            // B: exclusive_prefix_sum_small (1 block × 1024 threads, ft_in ≈ 73K)
            cuda_launch! {
                kernel: exclusive_prefix_sum_small,
                stream: self.stream, module: self.module,
                config: LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                },
                args: [
                    slice(self.ws.feat_counts),
                    slice(self.ws.feat_offsets),
                    ft_in as u32
                ]
            }?;
            prof_tick!("phB_psum");
            // C: scatter_positions
            cuda_launch! {
                kernel: scatter_positions,
                stream: self.stream, module: self.module,
                config: cfg_1d(b * max_active),
                args: [
                    slice(idx_dev),
                    slice(self.ws.feat_offsets),
                    slice(self.ws.feat_write_ctr),
                    slice(self.ws.feat_positions),
                    b_u32, max_active as u32, ft_in as u32
                ]
            }?;
            prof_tick!("phC_scat");
            // D: gather_and_sum_per_feature。block grid = (ft_in, ft_out/128), block_dim=128.
            // 1 回目 (stm) は overwrite、2 回目 (nstm) は atomic add で stm 結果に加算。
            // host は grad_w を memset_zero 済みだが、overwrite kernel は全 cell を書き切る。
            let d_config = LaunchConfig {
                grid_dim: (ft_in as u32, (ft_out / 128) as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // iter 0 = stm (dft_stm_out / overwrite)、iter 1 = nstm (dft_nstm_out / add)。
            // `ft_fp16_out` 時は dft が f16 なので f16 入力版の gather kernel を使う。
            if iter_idx == 0 {
                if self.ft_fp16_out {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_overwrite_fp16,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_stm_out_h.as_ref()
                                .expect("dft_stm_out_h is Some when ft_fp16_out is enabled")),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, ft_out as u32, dft_inv_scale
                        ]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_overwrite,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_stm_out),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, ft_out as u32
                        ]
                    }?;
                }
                // P-obs: phD iter 0 (stm overwrite) を独立計測する。`prof_tick!` は
                // stream.synchronize を打つので、これが無いと前 iter の compute が次
                // tick (phA_reset iter 1) に流れ込んで観測上 phA_reset が肥大化する。
                prof_tick!("phD_stm");
            } else {
                if self.ft_fp16_out {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_add_fp16,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_nstm_out_h.as_ref()
                                .expect("dft_nstm_out_h is Some when ft_fp16_out is enabled")),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, ft_out as u32, dft_inv_scale
                        ]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: gather_and_sum_per_feature_add,
                        stream: self.stream, module: self.module,
                        config: d_config,
                        args: [
                            slice(self.ws.dft_nstm_out),
                            slice(self.ws.feat_positions),
                            slice(self.ws.feat_offsets),
                            slice(self.ft_w_grad),
                            ft_in as u32, ft_out as u32
                        ]
                    }?;
                }
                prof_tick!("phD_nstm");
            }
            let _ = total_pairs; // unused yet
        }
        prof_tick!("bwd_ftbwd");

        // ===== OPTIMIZER STEP (Ranger = RAdam + Lookahead) =====
        self.step_count += 1;
        let (step_size, denom) =
            radam_compute_step_size_denom(self.step_count, BETA1, BETA2, N_SMA_THRESHOLD);

        // 10 weight groups × radam_step。FT weight (`ft_w`) の radam は 2 つの opt-in
        // flag で 4 通りに分岐する:
        //  - `--ft-fp16`: FP16 mirror (`ft_w_h`) 同時更新 variant を使い、forward 用
        //    mirror を別 cast kernel 無しで同期する (master FP32 が既に register 上に
        //    あるので half 書き出しのみ追加)。
        //  - `--fp16-opt-state`: m / v を `f16` で読み書きする `*_f16state` variant
        //    (DRAM traffic 半減、scale 付き格納)。
        // 他 9 group は moment が小さく `f16` 化の意味が無いので常に `radam_step`。
        // FT
        match (&mut self.ft_w_m, &mut self.ft_w_v) {
            (MomentBuf::F16(ft_w_m), MomentBuf::F16(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                    cuda_launch! {
                        kernel: radam_step_f16state_mirror,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W,
                               FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n as u32]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: radam_step_f16state,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W,
                               FT_OPT_M_SCALE, FT_OPT_V_SCALE, ft_w_n as u32]
                    }?;
                }
            }
            (MomentBuf::F32(ft_w_m), MomentBuf::F32(ft_w_v)) => {
                let (mut ft_w_m, mut ft_w_v) = (ft_w_m, ft_w_v);
                if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                    cuda_launch! {
                        kernel: radam_step_fp16_mirror,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), slice_mut(ft_w_h), lr, step_size, denom,
                               self.weight_decay, BETA1, BETA2, EPS, MIN_W, MAX_W, ft_w_n as u32]
                    }?;
                } else {
                    cuda_launch! {
                        kernel: radam_step,
                        stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                        args: [slice_mut(self.ft_w), slice_mut(ft_w_m), slice_mut(ft_w_v),
                               slice_mut(self.ft_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2,
                               EPS, MIN_W, MAX_W, ft_w_n as u32]
                    }?;
                }
            }
            // m / v は同じ flag で `MomentBuf::zeroed` され、load/init でも同期するので
            // 精度が食い違うことはない。
            _ => unreachable!("ft_w m and v moment buffers always share precision"),
        }
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(ft_b_n),
            args: [slice_mut(self.ft_b), slice_mut(self.ft_b_m), slice_mut(self.ft_b_v),
                   slice_mut(self.ft_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, ft_b_n as u32]
        }?;
        // L1
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1_w_n),
            args: [slice_mut(self.l1_w), slice_mut(self.l1_w_m), slice_mut(self.l1_w_v),
                   slice_mut(self.l1_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1_b_n),
            args: [slice_mut(self.l1_b), slice_mut(self.l1_b_m), slice_mut(self.l1_b_v),
                   slice_mut(self.l1_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1_b_n as u32]
        }?;
        // L1f
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1f_w_n),
            args: [slice_mut(self.l1f_w), slice_mut(self.l1f_w_m), slice_mut(self.l1f_w_v),
                   slice_mut(self.l1f_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1f_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l1f_b_n),
            args: [slice_mut(self.l1f_b), slice_mut(self.l1f_b_m), slice_mut(self.l1f_b_v),
                   slice_mut(self.l1f_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l1f_b_n as u32]
        }?;
        // L2
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l2_w_n),
            args: [slice_mut(self.l2_w), slice_mut(self.l2_w_m), slice_mut(self.l2_w_v),
                   slice_mut(self.l2_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l2_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l2_b_n),
            args: [slice_mut(self.l2_b), slice_mut(self.l2_b_m), slice_mut(self.l2_b_v),
                   slice_mut(self.l2_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l2_b_n as u32]
        }?;
        // L3
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l3_w_n),
            args: [slice_mut(self.l3_w), slice_mut(self.l3_w_m), slice_mut(self.l3_w_v),
                   slice_mut(self.l3_w_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l3_w_n as u32]
        }?;
        cuda_launch! {
            kernel: radam_step,
            stream: self.stream, module: self.module, config: cfg_1d(l3_b_n),
            args: [slice_mut(self.l3_b), slice_mut(self.l3_b_m), slice_mut(self.l3_b_v),
                   slice_mut(self.l3_b_grad), lr, step_size, denom, self.weight_decay, BETA1, BETA2, EPS,
                   MIN_W, MAX_W, l3_b_n as u32]
        }?;

        // Lookahead lerp every K steps。lerp は radam の後に FT weight を再度書き換える
        // ので、`--ft-fp16` 時は FT weight の lerp も FP16 mirror 同時更新 variant を使い、
        // forward 用 `ft_w_h` を lerp 後の最終値で同期し直す。
        if self.step_count.is_multiple_of(RANGER_K) {
            if let Some(mut ft_w_h) = self.ft_w_h.as_mut() {
                cuda_launch! {
                    kernel: ranger_lookahead_lerp_fp16_mirror,
                    stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                    args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), slice_mut(ft_w_h),
                           RANGER_ALPHA, ft_w_n as u32]
                }?;
            } else {
                cuda_launch! {
                    kernel: ranger_lookahead_lerp,
                    stream: self.stream, module: self.module, config: cfg_1d(ft_w_n),
                    args: [slice_mut(self.ft_w), slice_mut(self.ft_w_slow), RANGER_ALPHA, ft_w_n as u32]
                }?;
            }
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(ft_b_n),
                args: [slice_mut(self.ft_b), slice_mut(self.ft_b_slow), RANGER_ALPHA, ft_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1_w_n),
                args: [slice_mut(self.l1_w), slice_mut(self.l1_w_slow), RANGER_ALPHA, l1_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1_b_n),
                args: [slice_mut(self.l1_b), slice_mut(self.l1_b_slow), RANGER_ALPHA, l1_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1f_w_n),
                args: [slice_mut(self.l1f_w), slice_mut(self.l1f_w_slow), RANGER_ALPHA, l1f_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l1f_b_n),
                args: [slice_mut(self.l1f_b), slice_mut(self.l1f_b_slow), RANGER_ALPHA, l1f_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l2_w_n),
                args: [slice_mut(self.l2_w), slice_mut(self.l2_w_slow), RANGER_ALPHA, l2_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l2_b_n),
                args: [slice_mut(self.l2_b), slice_mut(self.l2_b_slow), RANGER_ALPHA, l2_b_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l3_w_n),
                args: [slice_mut(self.l3_w), slice_mut(self.l3_w_slow), RANGER_ALPHA, l3_w_n as u32]
            }?;
            cuda_launch! {
                kernel: ranger_lookahead_lerp,
                stream: self.stream, module: self.module, config: cfg_1d(l3_b_n),
                args: [slice_mut(self.l3_b), slice_mut(self.l3_b_slow), RANGER_ALPHA, l3_b_n as u32]
            }?;
        }
        prof_tick!("optimizer");

        // 本 step の compute (input buffer の read を含む) 完了を copy stream 用の
        // event に記録する。同じ物理 input buffer を使う step+2 の H2D がこれを待ち、
        // in-flight compute が読む buffer を H2D が上書きする race を防ぐ。
        self.input_ring.mark_step_done(&self.stream)?;

        // `loss_acc` の host への取り出しを `AsyncLossRing` 経由で async 化。
        // pinned host cell に `memcpy_dtoh_async` + event record、前 step の event を
        // sync して 1 step lag で loss を返す (step 0 は warmup として 0.0、sb 末で
        // [`TrainerBackend::flush_pending_loss`] が最終 step 分を drain する)。host は
        // 次 batch の launch 発行で `stream.synchronize` 相当の block 待ちが消える。
        let loss = self
            .loss_ring
            .read_and_queue_next(&self.stream, &self.loss_acc)?;
        Ok(StepOutput {
            loss,
            net_output: Vec::new(),
        })
    }
}

// step() / step_impl() 実装は別 impl block (file 分割回避のため同 file 内)。

// ===========================================================================
// TrainerBackend impl — `nnue-train::trainer::run` から 1 batch ずつ呼ばれる
// ===========================================================================

impl TrainerBackend for GpuTrainer {
    fn train_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<f64> {
        // dataloader が出した batch の feature set が trainer 構築時に選んだ feature set
        // と一致することを確認する (buffer サイズ / kernel launch 次元が前者を前提に
        // 確保済のため、不一致は out-of-bounds になる)。
        if batch.feature_set != self.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.feature_set.canonical_name()
            )));
        }
        let data = BatchData::from_batch_ref(batch, bucket_idx);
        self.step(&data, lr, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("GpuTrainer::step failed: {e}")))
    }

    fn validate_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        wdl_lambda: f32,
        loss: LossKind,
    ) -> std::io::Result<ValidationStepOutput> {
        // train_step と同じく batch の feature set が trainer の feature set と
        // 一致することを確認する (GPU buffer / kernel 次元の前提)。
        if batch.feature_set != self.feature_set {
            return Err(std::io::Error::other(format!(
                "batch feature set '{}' does not match trainer feature set '{}'",
                batch.feature_set.canonical_name(),
                self.feature_set.canonical_name()
            )));
        }
        let data = BatchData::from_batch_ref(batch, bucket_idx);
        let out = self
            .validate(&data, wdl_lambda, loss)
            .map_err(|e| std::io::Error::other(format!("GpuTrainer::validate failed: {e}")))?;
        Ok(ValidationStepOutput {
            sum_sq_err: out.loss,
            net_output: out.net_output,
        })
    }

    fn flush_pending_loss(&mut self) -> std::io::Result<f64> {
        self.loss_ring.flush_pending_loss().map_err(|e| {
            std::io::Error::other(format!(
                "GpuTrainer::loss_ring.flush_pending_loss failed: {e}"
            ))
        })
    }

    fn save_checkpoint(&mut self, path: &Path) -> std::io::Result<()> {
        let weights = self.to_layerstack_weights().map_err(|e| {
            std::io::Error::other(format!("GpuTrainer::to_layerstack_weights failed: {e}"))
        })?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        weights.save_quantised(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    fn save_resume_checkpoint(
        &mut self,
        path: &Path,
        superbatch: usize,
        run_id: &str,
    ) -> std::io::Result<()> {
        self.save_raw_checkpoint(path, superbatch, run_id)
            .map_err(|e| {
                // 既に io::Error なら kind を保つ、それ以外は other で包む。
                match e.downcast::<std::io::Error>() {
                    Ok(io_err) => *io_err,
                    Err(other) => std::io::Error::other(format!(
                        "GpuTrainer::save_raw_checkpoint failed: {other}"
                    )),
                }
            })
    }
}
