use cuda_core::IntoResult as _;
use gpu_runtime::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig};
use nnue_train::dataloader::Batch;
use shogi_features::FeatureSetSpec;

use crate::kernel_module::*;

/// `ft_w` の Ranger moment (`m` / `v`) buffer。既定は `f32`、`--fp16-opt-state` で
/// `f16` (格納時 scale 付き、[`radam_step_f16state`])。`ft_w` は 112.6M 要素で
/// optimizer phase の DRAM traffic を占めるため `f16` 化の効果がある一方、他 9 group
/// の moment は小さく `f16` 化の意味が無いので `f32` (`DeviceBuffer<f32>`) のまま。
pub(crate) enum MomentBuf {
    F32(DeviceBuffer<f32>),
    F16(DeviceBuffer<f16>),
}

impl MomentBuf {
    /// 要素数 `n` の 0 初期化 moment buffer。`fp16` で `f16` / `f32` を選ぶ。
    pub(crate) fn zeroed(
        stream: &CudaStream,
        n: usize,
        fp16: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if fp16 {
            Ok(MomentBuf::F16(DeviceBuffer::<f16>::zeroed(stream, n)?))
        } else {
            Ok(MomentBuf::F32(DeviceBuffer::<f32>::zeroed(stream, n)?))
        }
    }

    /// device → host download し **真値の `f32`** で返す (raw checkpoint 用)。`f16`
    /// variant は格納値が `scale` 倍されているので割り戻す (`f32` variant は scale
    /// 無関係でそのまま)。
    pub(crate) fn to_host_f32(
        &self,
        stream: &CudaStream,
        scale: f32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match self {
            MomentBuf::F32(b) => Ok(b.to_host_vec(stream)?),
            MomentBuf::F16(b) => {
                let inv = 1.0_f32 / scale;
                Ok(b.to_host_vec(stream)?
                    .into_iter()
                    .map(|x| (x as f32) * inv)
                    .collect())
            }
        }
    }

    /// **真値の `f32`** slice から moment buffer を作る (raw checkpoint resume 用)。
    /// `fp16` で variant を選ぶ。`f16` variant は `scale` を掛けてから半精度化し、
    /// `f16` 有限域 (`|x| <= 65504`) を超える値は clamp する (`f32` で書かれた
    /// checkpoint を `--fp16-opt-state` で resume したとき外れ値が `inf` 化して以降の
    /// step を壊すのを防ぐ。[`radam_step_f16state`] の格納時 clamp と同方針)。
    pub(crate) fn from_host_f32(
        stream: &CudaStream,
        data: &[f32],
        fp16: bool,
        scale: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if fp16 {
            let h: Vec<f16> = data
                .iter()
                .map(|&x| (x * scale).clamp(-65504.0, 65504.0) as f16)
                .collect();
            Ok(MomentBuf::F16(DeviceBuffer::from_host(stream, &h)?))
        } else {
            Ok(MomentBuf::F32(DeviceBuffer::from_host(stream, data)?))
        }
    }
}

/// [`GpuTrainer::step_impl`] の出力。
///
/// `loss` は batch 全体の二乗誤差和 (`Σ err²`、position 数で割る前)。`net_output`
/// は held-out validation (`validate == true`) のときだけ position ごとの net
/// 出力スカラ (`n_pos` 個) で埋まり、通常の training step では空。
pub(crate) struct StepOutput {
    pub(crate) loss: f64,
    pub(crate) net_output: Vec<f32>,
}

/// Smoke / trainer 用の 1 batch 入力データ。
/// owned 版 (smoke path) と borrowed 版 (train_step path) を統一するため scalar の
/// `per_pos_norm` を持ち (= 1/n_pos)、ref 化された slice を直接 H2D 投入する。
pub(crate) struct BatchData<'a> {
    pub(crate) n_pos: usize,
    pub(crate) stm_indices: &'a [i32], // (n_pos × max_active)、-1 padding 可
    pub(crate) nstm_indices: &'a [i32],
    pub(crate) bucket_idx: &'a [i32], // (n_pos)、progress-kpabs が emit する 0..num_buckets-1
    pub(crate) score: &'a [f32],      // (n_pos)、target eval cp の元
    pub(crate) wdl: &'a [f32],        // (n_pos)、0.0 (Loss) / 0.5 (Draw) / 1.0 (Win)
    pub(crate) per_pos_norm: f32, // 1/n_pos scalar (loss kernel が `norm[bi]` を本値の broadcast で読む)
}

/// `BatchData` を owned 形で組み立てるための一時 buffer (smoke / test 用)。本体 train_step
/// path では `BatchData::from_batch_ref` を使う (slice 借用)。
pub(crate) struct BatchDataOwned {
    pub(crate) n_pos: usize,
    pub(crate) stm_indices: Vec<i32>,
    pub(crate) nstm_indices: Vec<i32>,
    pub(crate) bucket_idx: Vec<i32>,
    pub(crate) score: Vec<f32>,
    pub(crate) wdl: Vec<f32>,
}

impl BatchDataOwned {
    pub(crate) fn as_ref(&self) -> BatchData<'_> {
        let n = self.n_pos;
        BatchData {
            n_pos: n,
            stm_indices: &self.stm_indices,
            nstm_indices: &self.nstm_indices,
            bucket_idx: &self.bucket_idx,
            score: &self.score,
            wdl: &self.wdl,
            per_pos_norm: if n == 0 { 0.0 } else { 1.0_f32 / n as f32 },
        }
    }
}

impl BatchData<'_> {
    /// 決定論的な smoke 用 dummy batch。bucket_idx=0、small random sparse indices。
    /// `feature_set` で `max_active` (1 perspective あたり active feature 数) と
    /// index の範囲 `[0, ft_in)` が決まる。
    pub(crate) fn smoke_dummy(n_pos: usize, feature_set: FeatureSetSpec) -> BatchDataOwned {
        let ft_in = feature_set.ft_in();
        let max_active = feature_set.max_active();
        let mut stm_indices = vec![-1_i32; n_pos * max_active];
        let mut nstm_indices = vec![-1_i32; n_pos * max_active];
        // 各 position に max_active 個の deterministic 実 index を入れる。
        // range [0, ft_in) で seed-based に分散。index 列は factorizer 非依存
        // (仮想行は trainer の fold / reduce kernel が配線する)。
        let mut s: u64 = 0xdead_beef;
        for b in 0..n_pos {
            for k in 0..max_active {
                // xorshift
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let idx = (s as usize % ft_in) as i32;
                stm_indices[b * max_active + k] = idx;
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let idx2 = (s as usize % ft_in) as i32;
                nstm_indices[b * max_active + k] = idx2;
            }
        }
        BatchDataOwned {
            n_pos,
            stm_indices,
            nstm_indices,
            bucket_idx: vec![0_i32; n_pos],
            score: vec![0.0_f32; n_pos],
            wdl: vec![0.5_f32; n_pos],
        }
    }

    /// bucket-aware backend (LayerStack) 用: `nnue-train` dataloader の `Batch` +
    /// per-position bucket (= `n_pos` 個) から borrowed `BatchData` を作る (`.to_vec()` を
    /// 避けて 22 MB の CPU memcpy を削減)。`bucket_idx.len() == n_pos` を厳密 assert する
    /// ので、誤って空 slice を渡すと panic で検出される。
    pub(crate) fn from_batch_ref<'a>(batch: &'a Batch, bucket_idx: &'a [i32]) -> BatchData<'a> {
        let n_pos = batch.n_positions;
        assert_eq!(
            bucket_idx.len(),
            n_pos,
            "bucket_idx len ({}) must equal batch.n_positions ({})",
            bucket_idx.len(),
            n_pos
        );
        Self::from_batch_inner(batch, bucket_idx)
    }

    /// bucket-less backend (Simple) 用: bucket_idx は空 slice。`TrainingConfig::compute_bucket
    /// = false` で worker が bucket 計算を skip した経路で使う (`SimpleGpuTrainer::train_step`
    /// は元から bucket_idx を参照しない契約のため空 slice で安全)。LayerStack 経路で誤って
    /// 本 fn を呼ぶと bucket_idx 不在で backend kernel が読む先がなくなるが、それは
    /// host driver の責任で本 fn 内は検査しない (LayerStack ↔ Simple の host driver は
    /// 別 path で型混在しない)。
    pub(crate) fn from_batch_ref_bucketless<'a>(batch: &'a Batch) -> BatchData<'a> {
        Self::from_batch_inner(batch, &[])
    }

    pub(crate) fn from_batch_inner<'a>(batch: &'a Batch, bucket_idx: &'a [i32]) -> BatchData<'a> {
        let n_pos = batch.n_positions;
        let max_active = batch.feature_set.max_active();
        assert_eq!(
            batch.max_active,
            max_active,
            "Batch::max_active ({}) must equal feature set '{}' max_active ({})",
            batch.max_active,
            batch.feature_set.canonical_name(),
            max_active
        );
        let span = n_pos * max_active;
        let norm = if n_pos == 0 {
            0.0
        } else {
            1.0_f32 / n_pos as f32
        };
        BatchData {
            n_pos,
            stm_indices: &batch.stm_indices[..span],
            nstm_indices: &batch.nstm_indices[..span],
            bucket_idx,
            score: &batch.score[..n_pos],
            wdl: &batch.wdl[..n_pos],
            per_pos_norm: norm,
        }
    }
}

/// `LaunchConfig` builder for 1D launch with `BLOCK_DIM` per block.
pub(crate) fn cfg_1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid_dim_1d(n, BLOCK_DIM),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `dense_bias_grad_tiled` が扱える out_dim の上限。kernel の shared `PARTIAL` 容量
/// (= block_dim 上限) と一致させる。`block_dim = R * out_dim <= 256` を保証するため、
/// caller は out_dim がこれを超える層では generic `bias_grad` に fall back する。
pub(crate) const DENSE_BIAS_GRAD_MAX_OUT: u32 = 256;

/// grid-stride bias-grad reduction (`dense_bias_grad_tiled`) の grid 上限。grid を
/// 増やすと per-cell の global atomic contention (= gridDim) が増えるため、SM を埋める
/// 範囲でクランプする。RTX 3080 Ti (80 SM) で block 256 thread を 6 block/SM
/// (= 1536 thread/SM、full occupancy) 載せる固定値。SM 数の多い GPU では占有率を取り
/// きれず under-occupy し得る (grid-stride は全 row を覆うので正しさには無関係、
/// atomic contention / occupancy の trade-off だけが動く)。
const DENSE_BIAS_GRAD_BLOCKS: u32 = 480;

/// `dense_bias_grad_tiled` の launch config。`block_dim = R * out_dim` で `R` は
/// `floor(DENSE_BIAS_GRAD_MAX_OUT / out_dim)` を超えない最大 2 冪 (kernel の tree
/// reduction が `R` を完全に畳める前提、`block_dim <= DENSE_BIAS_GRAD_MAX_OUT` も満たす)。
/// grid は batch を `R` 行/thread で覆う上限 `ceil(batch / R)` と `DENSE_BIAS_GRAD_BLOCKS`
/// の小さい方。caller は `1 <= out_dim <= DENSE_BIAS_GRAD_MAX_OUT` を保証する。
pub(crate) fn cfg_dense_bias_grad(batch: u32, out_dim: u32) -> LaunchConfig {
    debug_assert!(
        (1..=DENSE_BIAS_GRAD_MAX_OUT).contains(&out_dim),
        "cfg_dense_bias_grad requires 1 <= out_dim <= {DENSE_BIAS_GRAD_MAX_OUT}, got {out_dim}"
    );
    let cap = (DENSE_BIAS_GRAD_MAX_OUT / out_dim).max(1);
    let mut r = 1_u32;
    while r * 2 <= cap {
        r *= 2;
    }
    let block = r * out_dim;
    let grid = batch.div_ceil(r).clamp(1, DENSE_BIAS_GRAD_BLOCKS);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `norm_loss_reduce` 用 2D launch。x = group (`BLOCK_DIM` 単位)、y = pos チャンク
/// (`group_len` を ~1024 要素/thread 目安で最大 64 分割し atomic 部分和を並列化)。FT の
/// ように group_len が巨大な層で occupancy を稼ぐのが狙いで、小さい group は y=1。
pub(crate) fn cfg_norm_loss_reduce(n_groups: usize, group_len: usize) -> LaunchConfig {
    let grid_x = (n_groups as u32).div_ceil(BLOCK_DIM);
    let chunks = group_len.div_ceil(1024).clamp(1, 64) as u32;
    LaunchConfig {
        grid_dim: (grid_x, chunks, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `buf` の全 byte を 0 にする (stream 上、async)。`DeviceBuffer::zeroed` の
/// 再 alloc を伴わず既存 buffer を in-place で reset するため (grad / `loss_acc` の
/// 毎 step reset で `cudaMalloc`/`cudaFree` の stream stall を回避)。
pub(crate) fn memset_zero<T>(
    stream: &CudaStream,
    buf: &DeviceBuffer<T>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = buf.num_bytes();
    if bytes > 0 {
        // SAFETY: `buf.cu_deviceptr()` は本 `DeviceBuffer` が確保した `bytes` byte の
        // 有効 device ptr、`stream` は同 context (`buf` も `stream` も `GpuTrainer` が
        // 同 context から作る)。`cuMemsetD8Async` は overlap を要求しない。0 fill は
        // f32/f64 ともに数値 0.0 を表すバイトパターン (全 0) なので型に依らず正しい。
        unsafe {
            cuda_core::memory::memset_d8_async(buf.cu_deviceptr(), 0, bytes, stream.cu_stream())?;
        }
    }
    Ok(())
}

/// `i32` buffer の全要素を `-1` (= 0xFFFFFFFF) に async fill。bucket sort padding 行
/// の bucket marker / perm sentinel を invalid に初期化する用途。`memset_d8(0xFF)` は
/// 二の補数で -1 を作るため i32 専用 (符号無し型に対しては UINT_MAX を意味する)。
pub(crate) fn memset_minus_one_i32(
    stream: &CudaStream,
    buf: &DeviceBuffer<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = buf.num_bytes();
    if bytes > 0 {
        // SAFETY: [`memset_zero`] と同じ前提 — `buf.cu_deviceptr()` は本
        // `DeviceBuffer` が確保した `bytes` byte の有効 device ptr、`stream` は
        // 同 context。0xFF fill が i32 の -1 になる根拠は関数 doc を参照。
        unsafe {
            cuda_core::memory::memset_d8_async(
                buf.cu_deviceptr(),
                0xFF,
                bytes,
                stream.cu_stream(),
            )?;
        }
    }
    Ok(())
}

/// bucket sort 用の padded sorted layout 容量を計算する。各 bucket は次 16-row 境界に
/// align するため最大 `(num_buckets + 1) * 15` 行の padding を要する。安全側で
/// `(num_buckets + 1) * 16` を上乗せして 16 倍数に切り上げる。`+ 1` は invalid bin
/// (bucket idx `< 0` / `>= num_buckets`) の slot 分。
pub(crate) fn padded_sort_batch(batch: usize, num_buckets: usize) -> usize {
    let raw = batch + (num_buckets + 1) * 16;
    raw.div_ceil(16) * 16
}

/// pre-allocated device buffer に host slice を async memcpy。`DeviceBuffer::from_host`
/// の毎-step cudaMalloc/Free を排除するため。Caller は `buf` と `src` の長さが一致
/// (バッチ毎 fixed shape) を保証。
pub(crate) fn copy_host_to_device_async_i32(
    stream: &CudaStream,
    buf: &DeviceBuffer<i32>,
    src: &[i32],
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        src.len() <= buf.len(),
        "src.len()={} exceeds buf.len()={}",
        src.len(),
        buf.len()
    );
    let bytes = std::mem::size_of_val(src);
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: `buf.cu_deviceptr()` は確保済みの有効 device ptr で、直上の assert
    // により `bytes` は容量内。`stream` は同 context。copy は async のため `src`
    // の生存は caller が保証する: pinned 経路 (`InputUploadRing`) は h2d 完了
    // event で slot 再利用を gate し、pageable 経路 (`BatchData` の slice) は
    // 「pageable src は staging へ copy してから return する」という CUDA Driver
    // API の同期挙動 (API synchronization behavior) に依る。
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buf.cu_deviceptr(),
            src.as_ptr(),
            bytes,
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

pub(crate) fn copy_host_to_device_async_f32(
    stream: &CudaStream,
    buf: &DeviceBuffer<f32>,
    src: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        src.len() <= buf.len(),
        "src.len()={} exceeds buf.len()={}",
        src.len(),
        buf.len()
    );
    let bytes = std::mem::size_of_val(src);
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: [`copy_host_to_device_async_i32`] と同じ前提 (assert 済み容量 +
    // 同 context + `src` 生存は caller 保証)。
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buf.cu_deviceptr(),
            src.as_ptr(),
            bytes,
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

// ===========================================================================
// cuBLAS FFI — `dense_mm_bwd_weight_tiled` (L1f weight bwd) を `cublasSgemm_v2`
// に置換。CUDA Toolkit 12.x の dynamic link で取得 (`build.rs` で
// `cargo:rustc-link-lib=dylib=cublas`)。
// ===========================================================================

#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct cublasContext {
    _opaque: [u8; 0],
}
#[allow(non_camel_case_types)]
pub(crate) type cublasHandle_t = *mut cublasContext;
#[allow(non_camel_case_types)]
pub(crate) type cublasStatus_t = std::os::raw::c_int;
#[allow(non_camel_case_types)]
pub(crate) type cublasOperation_t = std::os::raw::c_int;

pub(crate) const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub(crate) const CUBLAS_OP_N: cublasOperation_t = 0;
pub(crate) const CUBLAS_OP_T: cublasOperation_t = 1;

// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` を呼ぶと、以後の
// Sgemm は FP32 input → TF32 (8-bit exp + 10-bit mantissa) cast → TC mma →
// FP32 accum に lower される (Ampere+)。FP32 比 ~2x スループット、~10-bit
// mantissa の精度低下。
//
// `cublasMath_t` enum (`/usr/local/cuda-*/include/cublas_api.h`、CUDA 12.9 時点):
//   CUBLAS_DEFAULT_MATH                              = 0
//   CUBLAS_TENSOR_OP_MATH                            = 1  (deprecated alias、FP16 TC fallback)
//   CUBLAS_PEDANTIC_MATH                             = 2
//   CUBLAS_TF32_TENSOR_OP_MATH                       = 3
//   CUBLAS_FP32_EMULATED_BF16X9_MATH                 = 4  (Hopper+ BF16x9 emulation)
//   CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16 (bit mask)
#[allow(non_camel_case_types)]
pub(crate) type cublasMath_t = std::os::raw::c_uint;
pub(crate) const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
pub(crate) const CUBLAS_TF32_TENSOR_OP_MATH: cublasMath_t = 3;

#[link(name = "cublas", kind = "dylib")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    fn cublasSetStream_v2(
        handle: cublasHandle_t,
        stream_id: cuda_core::sys::CUstream,
    ) -> cublasStatus_t;
    fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;
    fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: std::os::raw::c_int,
        n: std::os::raw::c_int,
        k: std::os::raw::c_int,
        alpha: *const f32,
        a: *const f32,
        lda: std::os::raw::c_int,
        b: *const f32,
        ldb: std::os::raw::c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: std::os::raw::c_int,
    ) -> cublasStatus_t;
}

/// RAII wrapper for `cublasHandle_t`。Create 失敗 / Set stream 失敗 / Destroy 失敗を
/// `Result` で返す。CUDA stream に bind して以後の Sgemm を同 stream で in-order 実行。
pub(crate) struct CublasHandle {
    handle: cublasHandle_t,
}

// SAFETY: `cublasHandle_t` は CUDA driver が tracking する opaque handle。cuBLAS API は
// driver thread safety guarantees に従い handle を別 thread から呼び出してよい
// (`cublasSetStream_v2` が thread-affinity を切り替えるとき内部 lock を取る)。
unsafe impl Send for CublasHandle {}

impl CublasHandle {
    /// `enable_tf32 = true` で Ampere+ Tensor Core を TF32 mode で活用する
    /// (`cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`)。Sgemm の FP32
    /// input は内部で TF32 (8-bit exp + 10-bit mantissa) cast → TC mma → FP32
    /// accum に lower され、throughput と引き換えに仮数 ~3 桁の精度低下を受ける。
    /// `false` では `CUBLAS_DEFAULT_MATH` (純 FP32 path、TC 不使用) を使う。
    ///
    /// 本 handle は fwd (`sgemm_fwd_rowmajor`) / bwd (`sgemm_xt_y_rowmajor`)
    /// 双方で共有されるため、L1f forward と weight backward の両 Sgemm に同
    /// mode が効く。
    pub(crate) fn new(
        stream: &CudaStream,
        enable_tf32: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        // SAFETY: cublasCreate_v2 は &mut handle に新規 handle を書き、CUBLAS_STATUS_SUCCESS
        // 以外を返したら handle は invalid (read 禁止)。失敗時は早期 return。
        let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasCreate_v2 failed: status={status}").into());
        }
        // SAFETY: handle is valid (above), stream.cu_stream() returns the wrapped CUstream。
        let status =
            unsafe { cublasSetStream_v2(handle, stream.cu_stream() as cuda_core::sys::CUstream) };
        if status != CUBLAS_STATUS_SUCCESS {
            // SAFETY: handle is valid (cleanup before erroring).
            unsafe {
                cublasDestroy_v2(handle);
            }
            return Err(format!("cublasSetStream_v2 failed: status={status}").into());
        }
        let mode = if enable_tf32 {
            CUBLAS_TF32_TENSOR_OP_MATH
        } else {
            CUBLAS_DEFAULT_MATH
        };
        // SAFETY: handle is valid.
        let status = unsafe { cublasSetMathMode(handle, mode) };
        if status != CUBLAS_STATUS_SUCCESS {
            // SAFETY: handle is valid (cleanup before erroring).
            unsafe {
                cublasDestroy_v2(handle);
            }
            let label = if enable_tf32 { "TF32" } else { "FP32" };
            return Err(format!("cublasSetMathMode({label}) failed: status={status}").into());
        }
        Ok(Self { handle })
    }

    /// row-major C[M, N] = A[M, K] @ B[K, N]、`alpha=1`, `beta=0` (overwrite)。
    /// fwd_L1f 用: combined[B, ft_out] @ l1f_w[ft_out, l1_out] → l1f_out[B, l1_out]。
    ///
    /// col-major cuBLAS で row-major matmul を計算する転置 trick: 同 memory 表現
    /// を cublas は col-major と解釈するので、`A_rm[m, k]` は `A_cm[k, m]`、
    /// `B_rm[k, n]` は `B_cm[n, k]`、`C_rm[m, n]` は `C_cm[n, m]` と等価。
    ///   row-major C[m, n] = sum_k A[m, k] * B[k, n]
    ///   = col-major C[n, m] = sum_k B_cm[n, k] * A_cm[k, m]
    /// → cublas call: A_arg=B_dev, B_arg=A_dev, transA=N, transB=N, m=N, n=M, k=K,
    ///   lda=N, ldb=K, ldc=N。両 trans=N の単純形なので bwd 用 `sgemm_xt_y_rowmajor`
    ///   (X^T @ Y、transB=T) より素直。
    ///
    /// SAFETY:
    /// - 全 device pointer は `cudaMalloc` 由来、長さは仕様分 (a.len() >= m*k、
    ///   b.len() >= k*n、c.len() >= m*n)。
    /// - stream は `cublasSetStream_v2` で bind 済の同一 stream を再利用。
    /// - math mode は handle 作成時の `enable_tf32` 引数で固定 ([`CublasHandle::new`]):
    ///   `true` で `CUBLAS_TF32_TENSOR_OP_MATH` (Ampere+ TC 経由、仮数 10-bit)、
    ///   `false` で `CUBLAS_DEFAULT_MATH` (純 FP32 path)。本関数は mode 非依存で
    ///   呼び出し可能、numeric tolerance は CLI `--tf32` 指定有無で変動する。
    /// - `beta=0` overwrite なので `c_ptr` の事前内容は使われない (caller は
    ///   `c_ptr` への書き込みを同 stream 内 in-order で行うこと、別 stream からの
    ///   race 書き込みは未定義動作)。
    pub(crate) unsafe fn sgemm_fwd_rowmajor(
        &self,
        m: i32,
        n: i32,
        k: i32,
        a_ptr: *const f32, // row-major [m, k]
        b_ptr: *const f32, // row-major [k, n]
        c_ptr: *mut f32,   // row-major [m, n]
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, // cublas m = N (cols of C in col-major)
                m, // cublas n = M (rows of C in col-major)
                k,
                &alpha,
                b_ptr, // cublas A = B (row-major [k, n] = col-major [n, k])
                n,
                a_ptr, // cublas B = A (row-major [m, k] = col-major [k, m])
                k,
                &beta,
                c_ptr,
                n,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 (fwd) failed: status={status}").into());
        }
        Ok(())
    }

    /// row-major C[M, N] = X^T @ Y、X[K, M] row-major、Y[K, N] row-major (X^T Y の reduce
    /// 軸は K)。col-major cuBLAS で計算するため転置 trick を使う:
    /// cublas は C_cm[N, M] = Y_cm[N, K] @ (X_cm[M, K])^T を計算、行列要素は同 memory。
    /// 詳細は call 元コメント参照。`alpha=1`, `beta=0` (overwrite)。
    ///
    /// SAFETY: 全 device pointer は cudaMalloc 由来 + 各 buffer 長 == 仕様分、stream は
    /// `cublasSetStream_v2` で bind 済の同一 stream を再利用。caller が形状不変条件
    /// (X.len() >= k*m、Y.len() >= k*n、C.len() >= m*n) を保証。
    pub(crate) unsafe fn sgemm_xt_y_rowmajor(
        &self,
        m: i32,
        n: i32,
        k: i32,
        x_ptr: *const f32, // row-major [k, m]
        y_ptr: *const f32, // row-major [k, n]
        c_ptr: *mut f32,   // row-major [m, n]
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // col-major cuBLAS で row-major C_rm = X_rm^T @ Y_rm を出すには:
        //   cublas C_cm[N, M] = Y_cm[N, K] @ (X_cm[M, K])^T と計算 (Y は trans=N、X は trans=T)
        //   Y_cm[n, k] = Y_rm[k, n] (同 memory)、X_cm[m, k] = X_rm[k, m] (同 memory)。
        //   結果 C_cm[n, m] = sum_k Y_rm[k, n] * X_rm[k, m] = C_rm[m, n] (同 memory)。
        // 引数: A=Y, B=X, transA=N, transB=T, m=N, n=M, k=K, lda=N, ldb=M, ldc=N。
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n, // m for cublas = n (out_dim)
                m, // n for cublas = m (in_dim)
                k,
                &alpha,
                y_ptr,
                n,
                x_ptr,
                m,
                &beta,
                c_ptr,
                n,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 failed: status={status}").into());
        }
        Ok(())
    }

    /// row-major C[M, N] = X @ Y^T、X[M, K] row-major、Y[N, K] row-major。bwd_input 用:
    /// dx[B, in_dim] = dy[B, out_dim] @ w[in_dim, out_dim]^T、reduce 軸は out_dim。
    /// col-major cuBLAS で計算する転置 trick:
    ///   C_rm[m, n] = sum_k X_rm[m, k] * Y_rm[n, k]
    ///   = C_cm[n, m] = sum_k Y_cm[k, n] * X_cm[k, m]   (X_rm[m, k] と X_cm[k, m] は同 memory)
    ///   → cublas A=Y (transA=T、shape [k, n] → 効果 [n, k])、B=X (transB=N、shape [k, m])
    ///     m_cublas=n、n_cublas=m、k_cublas=k、lda=k、ldb=k、ldc=n。
    /// `alpha=1`, `beta=0` (overwrite)。
    ///
    /// SAFETY: 全 device pointer は cudaMalloc 由来 + 各 buffer 長 >= 仕様分、stream は
    /// `cublasSetStream_v2` で bind 済の同一 stream を再利用。caller が形状不変条件
    /// (X.len() >= m*k、Y.len() >= n*k、C.len() >= m*n) を保証。
    pub(crate) unsafe fn sgemm_x_yt_rowmajor(
        &self,
        m: i32,
        n: i32,
        k: i32,
        x_ptr: *const f32, // row-major [m, k]
        y_ptr: *const f32, // row-major [n, k]
        c_ptr: *mut f32,   // row-major [m, n]
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                y_ptr,
                k,
                x_ptr,
                k,
                &beta,
                c_ptr,
                n,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 (bwd_input) failed: status={status}").into());
        }
        Ok(())
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is valid (created in new()).
            unsafe {
                cublasDestroy_v2(self.handle);
            }
        }
    }
}

/// `step()` 末尾の `loss_acc.to_host_vec` (内部で `stream.synchronize`) を排除し
/// host が次 batch の launch を即発行できるようにする ring。
///
/// 2-slot ring + 1-step lag: step N で device の `loss_acc` を pinned cell[N%2] に
/// async D2H + event record。返り値は step N-1 の loss (event[(N-1)%2] sync 後に
/// pinned cell[(N-1)%2] を読む)。最初の 1 step は前 step が無いので 0.0 を返す。
///
/// pinned host (`cuMemHostAlloc`) なので driver は staging copy 無しで直接 DMA、
/// 8 byte D2H + event record は host work と完全並行。
///
/// 末尾 step の loss は [`AsyncLossRing::flush_pending_loss`] で drain する。
/// [`crate::TrainerBackend::flush_pending_loss`] 経由で本 ring の `flush_pending_loss`
/// が superbatch 末で 1 回呼ばれ、未報告分が `sb_loss` に加算される。これにより
/// pipeline 化しても per-sb loss 集計は正確 (`sum(L_0..L_{N-1})`、warmup placeholder
/// 0 は sum に影響なし)。
pub(crate) struct AsyncLossRing {
    pinned: [*mut f64; 2],
    events: [CudaEvent; 2],
    step: usize,
    primed: bool,
}

// SAFETY: `pinned` は `cuMemHostAlloc` で確保した page-locked memory、CUDA driver の
// 内部 tracking 経由でアクセスされる。pointer 自体は host メモリで `Send` 安全。
unsafe impl Send for AsyncLossRing {}

impl AsyncLossRing {
    pub(crate) fn new(
        ctx: &std::sync::Arc<CudaContext>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pinned = [std::ptr::null_mut::<f64>(); 2];
        for slot in pinned.iter_mut() {
            let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
            // SAFETY: cuMemHostAlloc は page-locked host memory を 8 byte 確保、
            // failure 時は CUresult != SUCCESS を返す (.result()? で check)。
            unsafe {
                cuda_core::sys::cuMemHostAlloc(
                    &mut p as *mut _,
                    std::mem::size_of::<f64>(),
                    cuda_core::sys::CU_MEMHOSTALLOC_PORTABLE,
                )
                .result()?;
                // 初期値 0 (warmup で読まれないが defensive)
                std::ptr::write(p as *mut f64, 0.0);
            }
            *slot = p as *mut f64;
        }
        let events = [ctx.new_event(None)?, ctx.new_event(None)?];
        Ok(Self {
            pinned,
            events,
            step: 0,
            primed: false,
        })
    }

    /// `loss_acc` (device 1-cell f64) を async D2H で pinned[cur] へ copy、event 記録。
    /// 前 step (= step - 1) の event を sync して pinned[prev] を読み返り値とする。
    /// 最初の呼出 (warmup) は 0.0 を返す。
    pub(crate) fn read_and_queue_next(
        &mut self,
        stream: &CudaStream,
        loss_acc: &DeviceBuffer<f64>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let cur = self.step % 2;
        // SAFETY: pinned[cur] is page-locked host memory (cuMemHostAlloc 8 bytes),
        // loss_acc has len == 1 (= 8 bytes), stream 上 in-order なので async D2H は
        // 直前の memset/atomic 完了後に実行される。
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                self.pinned[cur],
                loss_acc.cu_deviceptr(),
                std::mem::size_of::<f64>(),
                stream.cu_stream(),
            )?;
        }
        self.events[cur].record(stream)?;

        let returned = if self.primed {
            let prev = (self.step + 1) % 2; // = (step - 1) % 2
            self.events[prev].synchronize()?;
            // SAFETY: event sync 完了 = D2H 完了、pinned[prev] に書き込まれた f64 を読む。
            unsafe { *self.pinned[prev] }
        } else {
            self.primed = true;
            0.0
        };

        self.step += 1;
        Ok(returned)
    }

    /// pipeline 末尾の drain: 最後に queue した step (= step - 1) の event を sync
    /// して pinned[(step - 1) % 2] の loss を返す。未呼出 (warmup 直後など primed
    /// = false) なら 0.0 を返す。
    ///
    /// `primed` を `false` に戻し `step` も 0 にリセットする。これにより次回 call
    /// は warmup として 0.0 を返し、その次の call から再び lag-1 の正常 pipeline が
    /// 始まる。caller (sb 末尾の trainer) は本 fn の返り値を sb_loss に加算する。
    pub(crate) fn flush_pending_loss(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        let returned = if self.primed {
            let last = (self.step + 1) % 2;
            self.events[last].synchronize()?;
            // SAFETY: event sync 完了 = D2H 完了、pinned[last] に書き込まれた f64 を読む。
            unsafe { *self.pinned[last] }
        } else {
            0.0
        };
        // 次回 sb は warmup から再開する (step も reset することで step % 2 計算が
        // 一貫し、pinned/event 再利用に矛盾無し)。
        self.primed = false;
        self.step = 0;
        Ok(returned)
    }
}

impl Drop for AsyncLossRing {
    fn drop(&mut self) {
        // pinned cell を free する前に未完了の async D2H を待つ。さもなければ in-flight な
        // memcpy_dtoh_async が解放後 host memory に書き戻して UB になる。primed = false の
        // 場合は record されていない event なので skip。失敗は無視 (Drop 中の error 報告は
        // 実用上困難、driver が in-flight copy を tracking する debug-build 動作と等価)。
        if self.primed {
            for event in self.events.iter() {
                let _ = event.synchronize();
            }
        }
        for slot in self.pinned.iter() {
            if !slot.is_null() {
                // SAFETY: cuMemHostAlloc で確保した pointer は cuMemFreeHost で解放する。
                // 上の event sync で in-flight D2H が完了済。
                unsafe {
                    cuda_core::sys::cuMemFreeHost(*slot as *mut std::os::raw::c_void);
                }
            }
        }
    }
}

/// step 先頭の入力 H2D (`stm/nstm idx` + `bucket/score/wdl` の 5 buffer) を専用
/// copy stream で発行し、直前 step の compute と overlap させる ring。
///
/// 入力は dataloader から pageable な `Vec` で来る。pageable のままだと
/// `cuMemcpyHtoDAsync` は driver の同期 staging copy になり copy engine の DMA を
/// 使えず、compute と並走しない。pinned host buffer を経由し compute stream とは
/// 別の copy stream で発行することで、H2D は直前 step の compute と並走する。
///
/// 2-slot pinned ring: step N は `pinned[N%2]` を使う。`pinned[N%2]` を最後に読んだ
/// H2D (= step N-2) の event を [`upload`](Self::upload) 冒頭で sync してから上書き
/// するので、in-flight な H2D が読んでいる pinned を host が書き換える race は起きない。
///
/// device 側の double-buffer (直前 step が読む buffer と次 step を H2D する buffer の
/// 物理分離) は caller (`step_impl`) が active / back buffer を `mem::swap` して担う。
/// 本 ring は H2D 先として「現在 active な」device buffer を受け取り、H2D 完了 event を
/// compute stream に待たせる。
pub(crate) struct InputUploadRing {
    pub(crate) copy_stream: std::sync::Arc<CudaStream>,
    // pinned host staging。stm/nstm は `batch * max_active`、bucket/score/wdl は `batch`。
    // bucket は LayerStack のみ持ち、Simple アーキは bucket-less 入力なので `None`。
    pinned_stm: [*mut i32; 2],
    pinned_nstm: [*mut i32; 2],
    pinned_bucket: Option<[*mut i32; 2]>,
    pinned_score: [*mut f32; 2],
    pinned_wdl: [*mut f32; 2],
    /// 各 slot の H2D 完了 event (copy stream に record)。compute stream が forward 前に待つ。
    h2d_done: [CudaEvent; 2],
    /// 各 slot を使った step の compute 完了 event (compute stream に record、
    /// [`mark_step_done`](Self::mark_step_done))。同じ物理 device buffer を次に使う
    /// step (= 2 step 後) の H2D 前に copy stream が待ち、in-flight な compute が
    /// 読んでいる buffer を H2D が上書きする race を防ぐ。
    step_done: [CudaEvent; 2],
    /// stm/nstm pinned の要素容量 (`batch * max_active`)。
    cap_idx: usize,
    /// bucket/score/wdl pinned の要素容量 (`batch`)。
    cap_scalar: usize,
    step: usize,
}

// SAFETY: 非 `Send` な field は raw pointer `pinned_*` のみ。これは `cuMemHostAlloc` で
// 確保した page-locked host memory への pointer で、`InputUploadRing` が単独 owner、
// 全 access は `&mut self` method (`upload` / `mark_step_done`) 経由で直列化される。
// raw pointer 経由の aliasing も内部からの concurrent access も無いので別 thread へ
// 移しても安全 (`AsyncLossRing` と同じ理由)。
unsafe impl Send for InputUploadRing {}

impl InputUploadRing {
    /// LayerStack 用: copy stream + 2-slot pinned buffer + event を確保する (bucket あり)。
    /// `batch` は最大 position 数、`max_active` は 1 perspective あたりの active feature 数
    /// (feature set 依存)。
    pub(crate) fn new(
        ctx: &std::sync::Arc<CudaContext>,
        batch: usize,
        max_active: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(ctx, batch, max_active, true)
    }

    /// Simple アーキ用: bucket buffer を確保しないバリアント。Simple は bucket-less 入力
    /// で kernel への bucket dispatch も無いため、bucket H2D 経路自体を持たない。
    pub(crate) fn new_simple(
        ctx: &std::sync::Arc<CudaContext>,
        batch: usize,
        max_active: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(ctx, batch, max_active, false)
    }

    pub(crate) fn new_inner(
        ctx: &std::sync::Arc<CudaContext>,
        batch: usize,
        max_active: usize,
        has_bucket: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let copy_stream = ctx.new_stream()?;
        let cap_idx = batch.max(1) * max_active;
        let cap_scalar = batch.max(1);
        let pinned_bucket = if has_bucket {
            Some(alloc_pinned_host::<i32>(cap_scalar)?)
        } else {
            None
        };
        Ok(Self {
            copy_stream,
            pinned_stm: alloc_pinned_host::<i32>(cap_idx)?,
            pinned_nstm: alloc_pinned_host::<i32>(cap_idx)?,
            pinned_bucket,
            pinned_score: alloc_pinned_host::<f32>(cap_scalar)?,
            pinned_wdl: alloc_pinned_host::<f32>(cap_scalar)?,
            h2d_done: [ctx.new_event(None)?, ctx.new_event(None)?],
            step_done: [ctx.new_event(None)?, ctx.new_event(None)?],
            cap_idx,
            cap_scalar,
            step: 0,
        })
    }

    /// `batch` の入力 5 slice を pinned 経由で `dev_*` (caller が swap で active 化した
    /// device buffer) へ copy stream で async H2D し、`compute_stream` に H2D 完了を
    /// 待たせる。
    ///
    /// caller (`step_impl`) は呼出前に active / back device buffer を `mem::swap` 済で、
    /// `dev_*` は直前 step が読んでいない側の物理 buffer であること。これにより H2D は
    /// 直前 step の compute と物理 buffer 競合なしに並走する。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload(
        &mut self,
        compute_stream: &CudaStream,
        dev_stm: &DeviceBuffer<i32>,
        h_stm: &[i32],
        dev_nstm: &DeviceBuffer<i32>,
        h_nstm: &[i32],
        dev_bucket: &DeviceBuffer<i32>,
        h_bucket: &[i32],
        dev_score: &DeviceBuffer<f32>,
        h_score: &[f32],
        dev_wdl: &DeviceBuffer<f32>,
        h_wdl: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            h_stm.len() <= self.cap_idx && h_nstm.len() <= self.cap_idx,
            "input batch ({} idx) exceeds pinned capacity {}",
            h_stm.len().max(h_nstm.len()),
            self.cap_idx
        );
        assert!(
            h_bucket.len() <= self.cap_scalar
                && h_score.len() <= self.cap_scalar
                && h_wdl.len() <= self.cap_scalar,
            "input batch (scalar) exceeds pinned capacity {}",
            self.cap_scalar
        );
        let slot = self.step % 2;
        if self.step >= 2 {
            // この物理 device buffer を最後に使った step (= step-2) の compute 完了を
            // copy stream に待たせてから H2D する。host は loss_ring 経由で複数 step
            // 先行しうるため、待たないと step-2 の backward がまだ読んでいる input
            // buffer を H2D が上書きする race になる。step 0/1 は当該 slot 未 record。
            self.copy_stream.wait(&self.step_done[slot])?;
            // 同 slot の pinned を最後に read した H2D (= step-2) の完了を host が待ち、
            // in-flight な H2D の読み元 pinned を下の copy_nonoverlapping が壊さないよう
            // にする。
            self.h2d_done[slot].synchronize()?;
        }
        let pinned_bucket = self
            .pinned_bucket
            .as_ref()
            .expect("InputUploadRing::upload (LayerStack) requires bucket-enabled ring");
        // host: Vec → pinned[slot]。
        // SAFETY: pinned[slot] は cuMemHostAlloc で cap 要素確保した有効 host memory、
        // 上の assert で `src.len() <= cap` を保証。src (Vec) / dst (pinned) は別領域。
        // step >= 2 の slot は直上の h2d_done sync で前回 H2D 完了済 (in-flight でない)。
        unsafe {
            std::ptr::copy_nonoverlapping(h_stm.as_ptr(), self.pinned_stm[slot], h_stm.len());
            std::ptr::copy_nonoverlapping(h_nstm.as_ptr(), self.pinned_nstm[slot], h_nstm.len());
            std::ptr::copy_nonoverlapping(h_bucket.as_ptr(), pinned_bucket[slot], h_bucket.len());
            std::ptr::copy_nonoverlapping(h_score.as_ptr(), self.pinned_score[slot], h_score.len());
            std::ptr::copy_nonoverlapping(h_wdl.as_ptr(), self.pinned_wdl[slot], h_wdl.len());
        }
        // device: pinned[slot] → dev_* (copy stream で async H2D)。
        // SAFETY: 各 pinned[slot] は直上の copy_nonoverlapping で先頭 `h_*.len()` 要素を
        // 初期化済の page-locked host memory。`from_raw_parts` で同じ `h_*.len()` 長の
        // slice 化して既存 H2D helper に渡す (helper が `src.len() <= dev.len()` を assert)。
        let cs: &CudaStream = &self.copy_stream;
        unsafe {
            copy_host_to_device_async_i32(
                cs,
                dev_stm,
                std::slice::from_raw_parts(self.pinned_stm[slot], h_stm.len()),
            )?;
            copy_host_to_device_async_i32(
                cs,
                dev_nstm,
                std::slice::from_raw_parts(self.pinned_nstm[slot], h_nstm.len()),
            )?;
            copy_host_to_device_async_i32(
                cs,
                dev_bucket,
                std::slice::from_raw_parts(pinned_bucket[slot], h_bucket.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_score,
                std::slice::from_raw_parts(self.pinned_score[slot], h_score.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_wdl,
                std::slice::from_raw_parts(self.pinned_wdl[slot], h_wdl.len()),
            )?;
        }
        self.h2d_done[slot].record(cs)?;
        // compute stream は H2D 完了後に forward が input を読むよう待つ。
        compute_stream.wait(&self.h2d_done[slot])?;
        Ok(())
    }

    /// Simple アーキ用 upload: bucket buffer を持たない 4 buffer 版 (stm/nstm/score/wdl)。
    /// 動作セマンティクスは [`upload`](Self::upload) と同じ — caller が active/back を
    /// `mem::swap` 済の `dev_*` に対し pinned 経由 copy stream で先行 H2D し、compute
    /// stream に H2D 完了を待たせる。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload_simple(
        &mut self,
        compute_stream: &CudaStream,
        dev_stm: &DeviceBuffer<i32>,
        h_stm: &[i32],
        dev_nstm: &DeviceBuffer<i32>,
        h_nstm: &[i32],
        dev_score: &DeviceBuffer<f32>,
        h_score: &[f32],
        dev_wdl: &DeviceBuffer<f32>,
        h_wdl: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            self.pinned_bucket.is_none(),
            "InputUploadRing::upload_simple called on bucket-enabled ring (LayerStack); \
             use InputUploadRing::new_simple to construct the ring"
        );
        assert!(
            h_stm.len() <= self.cap_idx && h_nstm.len() <= self.cap_idx,
            "input batch ({} idx) exceeds pinned capacity {}",
            h_stm.len().max(h_nstm.len()),
            self.cap_idx
        );
        assert!(
            h_score.len() <= self.cap_scalar && h_wdl.len() <= self.cap_scalar,
            "input batch (scalar) exceeds pinned capacity {}",
            self.cap_scalar
        );
        let slot = self.step % 2;
        if self.step >= 2 {
            self.copy_stream.wait(&self.step_done[slot])?;
            self.h2d_done[slot].synchronize()?;
        }
        // SAFETY: pinned[slot] は cap 要素確保済 host memory、上 assert で `src.len() <= cap` を
        // 保証。step >= 2 の slot は h2d_done sync で前回 H2D 完了済。
        unsafe {
            std::ptr::copy_nonoverlapping(h_stm.as_ptr(), self.pinned_stm[slot], h_stm.len());
            std::ptr::copy_nonoverlapping(h_nstm.as_ptr(), self.pinned_nstm[slot], h_nstm.len());
            std::ptr::copy_nonoverlapping(h_score.as_ptr(), self.pinned_score[slot], h_score.len());
            std::ptr::copy_nonoverlapping(h_wdl.as_ptr(), self.pinned_wdl[slot], h_wdl.len());
        }
        let cs: &CudaStream = &self.copy_stream;
        // SAFETY: pinned[slot] は直上で先頭 `h_*.len()` 要素を初期化済、`from_raw_parts`
        // で `src.len() <= dev.len()` を満たすよう slice 化 (helper が assert)。
        unsafe {
            copy_host_to_device_async_i32(
                cs,
                dev_stm,
                std::slice::from_raw_parts(self.pinned_stm[slot], h_stm.len()),
            )?;
            copy_host_to_device_async_i32(
                cs,
                dev_nstm,
                std::slice::from_raw_parts(self.pinned_nstm[slot], h_nstm.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_score,
                std::slice::from_raw_parts(self.pinned_score[slot], h_score.len()),
            )?;
            copy_host_to_device_async_f32(
                cs,
                dev_wdl,
                std::slice::from_raw_parts(self.pinned_wdl[slot], h_wdl.len()),
            )?;
        }
        self.h2d_done[slot].record(cs)?;
        compute_stream.wait(&self.h2d_done[slot])?;
        Ok(())
    }

    /// step の compute が input buffer を読み終えた (= step 全体が完了した) ことを
    /// `compute_stream` 上の event に記録し、step counter を進める。`step_impl` 末尾で
    /// 呼ぶ。同じ物理 device buffer を使う次 step ([`upload`](Self::upload) の step+2)
    /// が H2D 前にこの event を待ち、buffer reuse race を防ぐ。
    pub(crate) fn mark_step_done(
        &mut self,
        compute_stream: &CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let slot = self.step % 2;
        self.step_done[slot].record(compute_stream)?;
        self.step += 1;
        Ok(())
    }
}

impl Drop for InputUploadRing {
    fn drop(&mut self) {
        // pinned を free する前に in-flight な H2D を完了させる。copy stream を sync
        // すれば全 H2D が完了する。失敗は無視 (Drop 中の error 報告は実用上困難)。
        let _ = self.copy_stream.synchronize();
        let bucket_slots: &[*mut i32] = match self.pinned_bucket.as_ref() {
            Some(slots) => slots.as_slice(),
            None => &[],
        };
        for slot in self
            .pinned_stm
            .iter()
            .chain(self.pinned_nstm.iter())
            .chain(bucket_slots.iter())
        {
            if !slot.is_null() {
                // SAFETY: cuMemHostAlloc で確保した pointer を cuMemFreeHost で解放。
                // 上の copy stream sync で in-flight H2D は完了済。
                unsafe {
                    cuda_core::sys::cuMemFreeHost(*slot as *mut std::os::raw::c_void);
                }
            }
        }
        for slot in self.pinned_score.iter().chain(self.pinned_wdl.iter()) {
            if !slot.is_null() {
                // SAFETY: 同上。
                unsafe {
                    cuda_core::sys::cuMemFreeHost(*slot as *mut std::os::raw::c_void);
                }
            }
        }
    }
}

/// `cuMemHostAlloc` で page-locked host memory を `n` 要素分 2 slot 確保する。
pub(crate) fn alloc_pinned_host<T>(n: usize) -> Result<[*mut T; 2], Box<dyn std::error::Error>> {
    let mut out = [std::ptr::null_mut::<T>(); 2];
    for slot in out.iter_mut() {
        let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc は page-locked host memory を `n * size_of::<T>()` byte
        // 確保、失敗時は CUresult != SUCCESS を返す (.result()? で check)。
        unsafe {
            cuda_core::sys::cuMemHostAlloc(
                &mut p as *mut _,
                n * std::mem::size_of::<T>(),
                cuda_core::sys::CU_MEMHOSTALLOC_PORTABLE,
            )
            .result()?;
        }
        *slot = p as *mut T;
    }
    Ok(out)
}
