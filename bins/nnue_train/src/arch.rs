use nnue_train::trainer::LossKind;

// ===========================================================================
// LayerStack architecture constants
// ===========================================================================
//
// FT input dim (`ft_in`) and active-feature count (`max_active`) depend on the
// input feature set chosen at startup (see `FeatureSetSpec`). The FT output dim
// is chosen from `--ft-out`, the L1 output dim from `--l1`, the L2 output dim
// from `--l2`, and the per-bucket dimension from `--num-buckets`. Those
// runtime dims, plus `ft_in` / `max_active`, are carried as fields on
// `GpuWorkspace`. The constants below are the defaults for the configurable
// dims.

/// Default FT output dim (per perspective), used when `--ft-out` is not given.
/// `--ft-out` accepts any positive multiple of 128: `gather_and_sum_per_feature`
/// launches its grid y-axis as `ft_out / 128`. The post-FT `combined` buffer has
/// the same width — pairwise halves each perspective, then the two perspectives
/// are concatenated back to the FT output width.
pub(crate) const DEFAULT_FT_OUT: usize = 1536;

/// Default L1 output dim, used when `--l1` is not given.
///
/// The L1 dense layer outputs `l1_out` values per position; one of them is the
/// skip-connection dim ([`L1_SKIP`]) and the rest ([`GpuWorkspace::l1_effective`])
/// feed the squared + concatenated L2 input. The per-bucket sorted/tiled dense
/// matmul kernels split the output into 16-wide out-tiles, so every `l1_out` runs
/// on the same tiled path — `DEFAULT_L1_OUT == 16` is simply one out-tile.
pub(crate) const DEFAULT_L1_OUT: usize = 16;

/// Skip-connection dim carved out of the L1 output: `l1_total` is sliced into
/// `l1_effective = l1_out - L1_SKIP` main dims plus this single skip dim, which
/// is added straight onto the network output.
pub(crate) const L1_SKIP: usize = 1;

/// Default L2 output dim, used when `--l2` is not given.
///
/// The L2 per-bucket dense layer outputs `l2_out` values per position; they feed
/// a CReLU activation and then the L3 output layer. `--l2` accepts any value in
/// [2, 256]; the upper bound is the fixed shared-memory accumulator capacity of
/// the per-bucket bias-gradient kernel.
pub(crate) const DEFAULT_L2_OUT: usize = 32;

/// Default LayerStack output bucket count, used when `--num-buckets` is not
/// given. progress-kpabs assigns each position to `floor(p * num_buckets)`,
/// so the default 9 keeps the binning + weight-buffer shape identical to the
/// historical layout and resume-compatible with existing checkpoints. The
/// trainer accepts `[2, MAX_SUPPORTED_NUM_BUCKETS]`.
pub(crate) const DEFAULT_NUM_BUCKETS: usize = 9;

/// Maximum supported bucket count without changing the per-bucket weight
/// backward kernels (`dense_mm_bwd_weight_bucket_tiled_{l2,l3}`). The kernels
/// hold a fixed 9-register accumulator (`a0..a8`); values up to 9 are silent
/// skipped via the runtime `num_buckets` arg, but larger N would need a kernel
/// restructure (register fan-out → `blockIdx.z` grid axis).
pub(crate) const MAX_SUPPORTED_NUM_BUCKETS: usize = 9;

// FT post-activation と l1_sqr の固定スケール (qa=127 量子化由来、`127.0/128.0`)。
pub(crate) const FT_POST_SCALE: f32 = 127.0 / 128.0;
pub(crate) const L1_SQR_SCALE: f32 = 127.0 / 128.0;

/// `--ft-fp16-out` で dft (FT activation gradient) を `f16` 化するときの loss scaling
/// 基準係数。実際に使う scale は **`FT_DFT_FP16_BASE_SCALE * batch`** (caller が batch を
/// 掛ける)。
///
/// dft は batch 正規化 (loss が `1/batch`) のため値が `1/batch` に比例し、無 scale だと
/// 全要素が f16 subnormal 下限 (2^-24 ≈ 6e-8) を下回り 0 に潰れて勾配が消える。`f16` へ
/// 書く前に scale を掛けて normal range に持ち上げ、gather 側で逆数を掛けて戻す。
///
/// scale を `batch` 比例にするのは、dft ∝ `1/batch` なので `dft * (BASE * batch)` が
/// batch に依らず一定 (`dft * batch` の不変量 × BASE) になり、どの `--batch-size` でも
/// 同じ f16 域に載るため。固定 scale だと小 batch で dft が大きくなり f16 max (65504) を
/// 超えて inf 化し `ft_w_grad` を破壊する。
///
/// 2^14: `dft * batch` の不変量は学習初期で ~1.2e-3、`× BASE ≈ 19` と f16 normal
/// range に収まる。ただし dft は学習が進むと縮まず成長し、scale 後の値が学習中盤で
/// f16 上限 (65504) に達しうるため、`ft_post_perspective_grad_fused_fp16` は f16
/// 書き込み前に `±65504` へ clamp する (overflow → `±inf` → 学習発散を防ぐ)。
/// batch=65536 のとき実 scale は `2^14 * 2^16 = 2^30` で power-of-2 (scale 自体は無誤差)。
pub(crate) const FT_DFT_FP16_BASE_SCALE: f32 = (1_u32 << 14) as f32;

/// `--fp16-opt-state` で `ft_w` の Ranger 1st moment (`m`) を `f16` 格納するときの
/// scale。`m` を `f16` へ書く前に掛け、読み戻し時に割る ([`radam_step_f16state`])。
///
/// `m` は batch 正規化された勾配の EMA で、実測 (1000 step 時点の `ft_w` checkpoint)
/// で `|m|` は p5 ~3e-13・中央値 ~3e-9・最大 ~1e-5。無 scale だと大半が `f16` の
/// subnormal 下限 (2^-24 ≈ 6e-8) 以下に潰れる。`2^28` を掛けると中央値が `f16` の
/// normal range 内 (~0.8)、最大値も `1e-5 * 2^28 ≈ 2.7e3 ≪ 65504` で overflow せず
/// (学習初期の勾配増大に ~24× の余裕)。scale は power-of-2 で scale 自体は無誤差。
pub(crate) const FT_OPT_M_SCALE: f32 = (1_u32 << 28) as f32;

/// `--fp16-opt-state` で `ft_w` の Ranger 2nd moment (`v`) を `f16` 格納するときの
/// scale。`v` を `f16` へ書く前に掛け、読み戻し時に割る ([`radam_step_f16state`])。
///
/// `v` は勾配二乗の EMA で `m` よりさらに小さく、実測で中央値 ~2e-15・最大 ~2e-9。
/// `m` と別 scale なのは値域が約 `m^2` のオーダーで異なるため。`2^40` を掛けると
/// 中央値が `f16` normal range 内、最大値も `2e-9 * 2^40 ≈ 2.2e3 ≪ 65504` で
/// overflow せず (初期勾配増大に ~30× の余裕)。`v >= 0` なので格納時の clamp は
/// 上側のみ。scale は power-of-2。
pub(crate) const FT_OPT_V_SCALE: f32 = (1_u64 << 40) as f32;

// Ranger optimizer params。値は `nnue_train::optimizer::RangerParams::DEFAULT`
// を single source of truth として参照する。
pub(crate) const RANGER_DEFAULTS: nnue_train::optimizer::RangerParams =
    nnue_train::optimizer::RangerParams::DEFAULT;
pub(crate) const BETA1: f32 = RANGER_DEFAULTS.beta1;
pub(crate) const BETA2: f32 = RANGER_DEFAULTS.beta2;
pub(crate) const EPS: f32 = RANGER_DEFAULTS.eps;
pub(crate) const RANGER_ALPHA: f32 = RANGER_DEFAULTS.alpha;
pub(crate) const RANGER_K: u64 = RANGER_DEFAULTS.k as u64;
pub(crate) const N_SMA_THRESHOLD: f32 = RANGER_DEFAULTS.n_sma_threshold;

// Per-layer training-time weight clamp。clamp 範囲は対象テンソルの量子化 dtype で
// 決まる:
//   - i8 dense weight (scale QB): `round(w·QB)` を [-128, 127] に量子化するため
//     正側端点 127 で飽和 → 対称 clamp ±i8::MAX/QB (`W_CLAMP_QUANT_*`)。負側は
//     厳密には -128/QB まで取れるが、nnue 系では対称 clamp が慣例。
//   - i16 FT weight/bias (scale QA、CReLU/Pairwise=127 · SCReLU=255): 飽和は
//     ±32767/QA で ~±128〜258、i32 bias / PSQT (scale QA·QB): ±2^31/8128 ≈ ±264k
//     と事実上 unbounded → clamp 無し (`W_CLAMP_NONE_*`)
// L3 (output) weight も i8@QB 量子化 (他 dense weight と同じ) なので clamp は
// ±127/QB で loss 非依存。nnue2score は出力 weight scale ではなく推論側 fv_scale
// (`round(QA·QB/scale)`) に畳まれるため、出力 weight を loss 別に締める必要はない。
// kernel launch ごとに対象テンソルの値を渡す。QB=64 は LayerStack と Simple 両
// format で共通 (`nnue_format::{layerstack_weights,simple_weights}`)。

/// i8 dense weight (L1 / L1f / L2 / L3 weight) と、挙動 neutral 維持のため同じ範囲に
/// 据える L1 / L1f / L2 bias に渡す対称 clamp ±i8::MAX/QB (= ±127/64)。i8 量子化
/// `round(w·QB)` の正側端点 (127) に対応する。
pub(crate) const W_CLAMP_QUANT_MIN: f32 =
    -(i8::MAX as f32) / nnue_format::layerstack_weights::QB as f32;
pub(crate) const W_CLAMP_QUANT_MAX: f32 =
    i8::MAX as f32 / nnue_format::layerstack_weights::QB as f32;

/// clamp 無しのテンソル (FT weight/bias、L3 bias、PSQT) に渡す sentinel。
/// `radam_step` の clamp 分岐 `p < min` / `p > max` は有限 weight に対し常に false
/// になり no-op (kernel signature を変えずに「clamp skip」を表現する)。
pub(crate) const W_CLAMP_NONE_MIN: f32 = f32::MIN;
pub(crate) const W_CLAMP_NONE_MAX: f32 = f32::MAX;

// smoke 用 loss params (scale=290, wdl=0.0、wrm in_scaling 340 / in offset 270 /
// nnue2score 600 / target offset 270 / target scaling 380)。
// trainer 経路では CLI から `LossKind` を組み立てるのでここは smoke 専用。
pub(crate) const WDL_LAMBDA: f32 = 0.0;
/// smoke で使う固定 batch position 数 (`GpuTrainer::new` の workspace 初期 batch
/// にも使う)。LayerStack の tiled dense matmul kernel は grid を `b / 16` で張るため
/// `b % 16 == 0` を要求する (`GpuTrainer::step_impl` の runtime check)。smoke は
/// `train_step` を介さず `step` を直叩きするので、ここで 16 の倍数にしておく。
pub(crate) const SMOKE_BATCH: usize = 16;
pub(crate) const SMOKE_LOSS_SIGMOID: LossKind = LossKind::Sigmoid { scale: 1.0 / 290.0 };
pub(crate) const SMOKE_LOSS_WRM: LossKind = LossKind::Wrm {
    nnue2score: 600.0,
    in_scaling: 340.0,
    in_offset: 270.0,
    target_offset: 270.0,
    target_scaling: 380.0,
    pow_exp: 2.0,
    qp_asymmetry: 0.0,
    weight_boost_w1: 0.0,
    weight_boost_w2: 0.5,
};
