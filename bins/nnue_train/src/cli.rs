use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use nnue_format::ArchKind;

use crate::arch::*;

// ===========================================================================
// CLI (clap) — 引数群は bullet-shogi `examples/shogi_layerstack.rs` に対応
// ===========================================================================

/// rshogi NNUE trainer。
///
/// 学習する NNUE アーキを `layerstack` / `simple` サブコマンドで選ぶ。共有引数は
/// サブコマンドの前後どちらに置いてもよい global 引数。`--data <PSV>` を指定すると
/// training loop を回し、省略すると GPU smoke test (forward/backward path 確認) を
/// 実行する。
#[derive(Parser, Debug)]
#[command(name = "nnue-train", about = "rshogi NNUE trainer")]
pub(crate) struct Cli {
    /// 教師データ PSV ファイル (`PackedSfenValue` × N、各 40 bytes)。省略時は GPU smoke test。
    #[arg(long, global = true)]
    pub(crate) data: Option<PathBuf>,

    /// held-out validation 用の PSV ファイル。学習 `--data` とは別の、勾配更新に
    /// 一度も使わない局面を渡す。指定すると各 superbatch 末に forward-only 検証を
    /// 走らせ、test_loss (held-out 平均 loss) と test_accuracy (出力符号と対局結果の
    /// 一致率) を train ログと experiment.json に出す。発散・過学習の早期検出に使う。
    #[arg(long, global = true)]
    pub(crate) test_data: Option<PathBuf>,

    /// held-out validation 1 回あたりの検証局面数。test PSV の先頭からこの数だけ
    /// 取り、`--batch-size` 単位に切り上げて満タン batch を作る。`--test-data`
    /// 指定時のみ使う。
    #[arg(long, default_value_t = 10000, global = true)]
    pub(crate) test_positions: usize,

    /// checkpoint 出力先 directory (`{net_id}-{superbatch}.bin` を書き出す)。
    #[arg(long, default_value = "checkpoints", global = true)]
    pub(crate) output: PathBuf,

    /// network id (checkpoint file 名に使う)。
    #[arg(long, default_value = "rshogi", global = true)]
    pub(crate) net_id: String,

    /// 入力 feature set。次のいずれか: halfkp, halfka-split, halfka-merged,
    /// halfka-hm-split, halfka-hm-merged。FT 入力次元と active feature 数を決める。
    /// 既定の halfka-hm-merged は king-symmetric merged HalfKA。
    #[arg(long, default_value = "halfka-hm-merged", global = true)]
    pub(crate) feature_set: String,

    /// experiment.json の `name` (実験管理 UI での表示名)。未指定なら net_id、
    /// `--resume` 時は `{net_id} (resume @sb{開始 superbatch})`。
    #[arg(long, global = true)]
    pub(crate) experiment_name: Option<String>,

    /// 学習する superbatch 数 (1..=superbatches を回す)。default 10 は smoke 用、
    /// 本番は 400 程度。
    #[arg(long, default_value_t = 10, global = true)]
    pub(crate) superbatches: usize,

    /// 1 superbatch あたりの batch 数。
    #[arg(long, default_value_t = 6104, global = true)]
    pub(crate) batches_per_superbatch: usize,

    /// 1 batch あたりの position 数。default 16384 は smoke 用、本番は 65536 程度。
    #[arg(long, default_value_t = 16384, global = true)]
    pub(crate) batch_size: usize,

    /// 初期 learning rate。
    #[arg(long, default_value_t = 8.75e-4, global = true)]
    pub(crate) lr: f32,

    /// LR gamma (`lr_step` superbatch ごとに gamma 倍)。
    #[arg(long, default_value_t = 0.995, global = true)]
    pub(crate) lr_gamma: f32,

    /// LR step (gamma 倍する superbatch 間隔)。
    #[arg(long, default_value_t = 1, global = true)]
    pub(crate) lr_step: usize,

    /// WDL blend lambda (constant)。
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) wdl: f32,

    /// sigmoid loss の score scale (`loss_scale = 1 / scale`)。`--win-rate-model` 指定時は
    /// 使わない (WRM loss は `--wrm-*` 系の scaling を使う)。
    #[arg(long, default_value_t = 290.0, global = true)]
    pub(crate) scale: f32,

    /// `save_rate` superbatch ごと (および末尾) に checkpoint を書き出す。
    #[arg(long, default_value_t = 20, global = true)]
    pub(crate) save_rate: usize,

    /// `|score| >= score_drop_abs` の position を loss から除外する (bullet `--score-drop-abs`)。
    #[arg(long, global = true)]
    pub(crate) score_drop_abs: Option<i32>,

    /// 学習開始前に量子化 NNUE binary から weight を注入する (pretrained start)。
    /// optimizer state (Ranger m/v/slow/step) は **reset** される — 真の resume には
    /// `--resume` を使うこと (`--init-from` と `--resume` は排他)。
    #[arg(long, global = true)]
    pub(crate) init_from: Option<PathBuf>,

    /// raw checkpoint (`{net_id}-{sb}.ckpt`) から weight + Ranger optimizer state
    /// (m/v/slow/step) を復元して学習を再開する (真の resume)。`--init-from`
    /// とは排他 (`--init-from` は weight のみ注入し optimizer を reset するため)。
    /// `--start-superbatch` 未指定なら checkpoint に記録された superbatch の +1 から再開。
    #[arg(long, global = true)]
    pub(crate) resume: Option<PathBuf>,

    /// 学習を開始する superbatch 番号 (1-indexed, inclusive)。未指定時:
    /// `--resume` あり → checkpoint の superbatch +1、なし → 1。`1 <= N <= --superbatches`
    /// の範囲外ならエラー (resume で過去 sb をやり直す目的で明示指定も可)。
    #[arg(long, global = true)]
    pub(crate) start_superbatch: Option<usize>,

    /// raw checkpoint (`*.ckpt`) を直近 N 個だけ残す (ディスク節約)。
    /// 未指定なら全保持 (raw state は ~1.8GB/個 なので save-rate × superbatches が
    /// 大きい長期ランでは指定推奨; 例 save-rate 20 / 400sb = 20 個 ≈ 36GB)。量子化
    /// `.bin` (~116MB) は本設定に関わらず常に全保持 (推論 artifact)。
    #[arg(long, global = true)]
    pub(crate) keep_checkpoints: Option<usize>,

    /// win-rate-model loss を使う。指定時は `loss_wrm` kernel (prediction / target
    /// 双方に WRM を適用) を使い、未指定なら `loss_wdl` (plain sigmoid-MSE + `--scale`)。
    /// net_output のスケールが `out ≈ cp/--wrm-nnue2score` になり、量子化
    /// (`QA=127/QB=64/FV_SCALE=28`) が前提とするスケールと整合する。
    #[arg(long, global = true)]
    pub(crate) win_rate_model: bool,
    /// WRM prediction 側の in-scaling (既定 340)。target 側の scaling
    /// (`--wrm-target-scaling`) とは独立。`--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 340.0, global = true)]
    pub(crate) wrm_in_scaling: f32,
    /// WRM の nnue2score (`scorenet = net_output * --wrm-nnue2score`、既定 600)。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 600.0, global = true)]
    pub(crate) wrm_nnue2score: f32,
    /// WRM target sigmoid の中心オフセット (`target` が 0.5 になる score、既定 270)。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 270.0, global = true)]
    pub(crate) wrm_target_offset: f32,
    /// WRM target sigmoid の入力スケール (steepness の逆数、既定 380)。既定 270/380 は
    /// chess の評価値分布向けの値なので、score 分布が異なれば再調整する。
    /// `--win-rate-model` 指定時のみ使う。
    #[arg(long, default_value_t = 380.0, global = true)]
    pub(crate) wrm_target_scaling: f32,
    /// optimizer 名 ("ranger" のみ実装)。
    #[arg(long, default_value = "ranger", global = true)]
    pub(crate) optimizer: String,
    /// Ranger optimizer の weight decay 係数 (AdamW 風の decoupled weight decay)。
    /// 既定 0.0 で decay 無し。非 0 で全 weight group の weight を毎 step わずかに
    /// 0 方向へ減衰させる。
    #[arg(long, default_value_t = 0.0, global = true)]
    pub(crate) weight_decay: f32,
    /// dataloader prefetch worker 数。各 worker が PSV パース + HalfKA_hm sparse 抽出 +
    /// progress8kpabs bucket 計算を `decode()` 1 回で済ませて先読み供給する。`1` で
    /// 決定論的逐次 read、`>= 2` で並列パース (1 epoch 内の position 順序は非決定的;
    /// training では問題ない)。
    #[arg(long, default_value_t = 16, global = true)]
    pub(crate) threads: usize,

    /// FT weight (`ft_w`) を FP16 mirror で forward する高速モード。default `false`
    /// では FP32 path と bit-identical。`true` で `sparse_ft_forward` の weight DRAM
    /// 帯域を半減する代わり、量子化誤差で棋力が変動しうる (簡易・高速学習向けの
    /// opt-in option、本番品質には SPRT で確認するまで default OFF)。
    ///
    /// FT weight は初期化・optimizer の MIN_W/MAX_W clamp (`|w| <= 1.98`)・量子化
    /// checkpoint いずれの経路でも小さく、FP16 の有限域 (`|x| <= 65504`) に十分
    /// 収まるため mirror 変換が ±inf へ overflow しない。
    #[arg(long, global = true)]
    pub(crate) ft_fp16: bool,

    /// 特徴変換器 (FT) の optimizer state を FP16 で保持する高速モード。default
    /// `false` では FP32 path と bit-identical。
    ///
    /// FT は本ネットで最も要素数の多い層で、その optimizer 更新は state の read/write
    /// がメモリ帯域律速。state を半精度化すると optimizer step のメモリ転送量が減って
    /// 学習スループットが上がる。state は値が極めて小さいため、固定係数を掛けて FP16
    /// の有効域に載せてから格納する。
    ///
    /// `--ft-fp16` / `--ft-fp16-out` とは独立した flag。量子化誤差で棋力が変動し
    /// うるため default OFF、本番品質は SPRT で確認するまで保証しない (動作確認や
    /// 簡易・高速な学習に使う opt-in option)。
    #[arg(long, global = true)]
    pub(crate) fp16_opt_state: bool,

    /// risky 速度 flag を 4 つまとめて opt-in する shortcut。`--ft-fp16` /
    /// `--fp16-opt-state` / (subcommand 側) `--ft-fp16-out` / `--tf32` を一括 ON
    /// 相当にする (個別 flag と OR 結合、両 subcommand 対応)。
    ///
    /// default OFF (全 flag OFF で純 FP32 path、bit-identical)。指定時の実効値は
    /// 起動時 log に展開出力 (`[train] --all-optim → ft_fp16=true ft_fp16_out=true
    /// fp16_opt_state=true tf32=true`) して experiment.json の reproducibility を
    /// 保つ。量子化 / TF32 誤差で棋力が変動しうるため default OFF。
    ///
    /// fine-grained 制御 (一部だけ ON) は本 flag を使わず個別 4 flag を列挙する。
    #[arg(long, global = true)]
    pub(crate) all_optim: bool,

    /// 学習する NNUE アーキを選ぶサブコマンド (`layerstack` / `simple`)。
    #[command(subcommand)]
    pub(crate) arch: ArchCommand,
}

/// `--ft-fp16-out` が `--ft-fp16` を要求する制約を **実効値** (`--all-optim` の含意込み)
/// で検証する。`true` を返したら制約違反 = error (FT activation FP16 が ON だが
/// FT weight FP16 が OFF)。
///
/// `--all-optim` は `--ft-fp16` / `--ft-fp16-out` の双方を ON 相当にするため、`--all-optim`
/// が指定されていれば制約は常に満たされる (両 flag が実効 ON)。よって制約違反は
/// 「`--ft-fp16-out` が raw 指定されていて、`--all-optim` も無く、`--ft-fp16` も raw 指定
/// されていない」ときのみ。これにより `--all-optim --ft-fp16-out` (冗長指定) を
/// false-positive reject しない。
pub(crate) fn ft_fp16_out_missing_ft_fp16(
    ft_fp16_out_raw: bool,
    ft_fp16_raw: bool,
    all_optim: bool,
) -> bool {
    ft_fp16_out_raw && !all_optim && !ft_fp16_raw
}

/// 学習対象の NNUE アーキを選ぶサブコマンド。アーキ固有の引数を持つ。
#[derive(Subcommand, Debug)]
pub(crate) enum ArchCommand {
    /// progress8kpabs 9-bucket LayerStack アーキ (FT → L1 16 → L2 32、FT 次元は --ft-out)。
    #[command(name = "layerstack")]
    LayerStack(LayerstackArgs),
    /// bullet-shogi 由来の Simple 4 層アーキ。
    Simple(SimpleArgs),
}

impl ArchCommand {
    /// サブコマンドに対応する [`ArchKind`]。
    pub(crate) fn kind(&self) -> ArchKind {
        match self {
            ArchCommand::LayerStack(_) => ArchKind::LayerStack,
            ArchCommand::Simple(_) => ArchKind::Simple,
        }
    }
}

/// LayerStack アーキ固有の引数。
#[derive(Args, Debug)]
pub(crate) struct LayerstackArgs {
    /// progress8kpabs 係数ファイル (`progress.bin`、f64 LE × 81*FE_OLD_END)。
    /// 未指定なら全 position が bucket 4 (zero weights → `sigmoid(0) = 0.5`)。
    #[arg(long)]
    pub(crate) progress_coeff: Option<PathBuf>,

    /// bucket mode ("progress8kpabs" のみ実装)。
    #[arg(long, default_value = "progress8kpabs")]
    pub(crate) bucket_mode: String,

    /// FT (feature transformer) の 1 perspective あたり出力次元。正の 128 の倍数を
    /// 指定する。既定値では従来構成と bit-identical で、既存 checkpoint と resume
    /// 互換を保つ。
    #[arg(long, default_value_t = DEFAULT_FT_OUT)]
    pub(crate) ft_out: usize,

    /// Ampere+ Tensor Core を TF32 mode で使う opt-in flag。`true` で cuBLAS の
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` を呼び、Sgemm の
    /// 入力 FP32 を 10-bit mantissa の TF32 に丸めて TC mma → FP32 accum で走る
    /// (仮数精度 ~3 桁、指数範囲は FP32 同等)。default `false` では
    /// `CUBLAS_DEFAULT_MATH` (純 FP32 path、TC 不使用) で走る。
    ///
    /// 仮数 13 bit 切り捨てで `fwd_L1f` / `bwd_L1f` Sgemm の数値に影響するため、
    /// 品質 conservative に default OFF。
    #[arg(long)]
    pub(crate) tf32: bool,

    /// FT activation (`ft_*_out` の forward 出力と `dft_*_out` の backward 勾配) も
    /// FP16 で保持する。`--ft-fp16` を要求する (weight FP16 path の上に積む拡張)。
    ///
    /// `ft_*_out` は `sparse_ft_forward` の出力で、これを FP16 化すると後続 read +
    /// inverse-index gather (step 中で最も DRAM read が多い `phD`) の帯域が半減する。
    /// dft は batch 正規化で `1/batch` に比例する微小値のため、FP16 化時は loss scaling
    /// (batch に比例する係数) で normal range に持ち上げてから格納する。
    ///
    /// weight FP16 (`--ft-fp16`) とは別 flag に分けてあり、SPRT で
    /// FP32 → `--ft-fp16` → `--ft-fp16 --ft-fp16-out` の 2 段で棋力影響を切り分け
    /// られる。量子化誤差で棋力が変動しうるため default OFF、本番品質は SPRT 確認まで
    /// 保証しない。
    #[arg(long)]
    pub(crate) ft_fp16_out: bool,
}

/// Simple 4 層アーキ固有の引数。
#[derive(Args, Debug)]
pub(crate) struct SimpleArgs {
    /// 層次元 preset (`<l1>x2-<l2>-<l3>`)。l1 は accumulator (FT 出力) 次元、
    /// l2 / l3 は隠れ層次元。`--l1` / `--l2` / `--l3` で個別に上書きできる。
    #[arg(long, default_value = "256x2-32-32")]
    pub(crate) arch: String,

    /// accumulator (FT 出力) 次元。未指定なら `--arch` preset の値。
    #[arg(long)]
    pub(crate) l1: Option<usize>,

    /// 隠れ層 1 の次元。未指定なら `--arch` preset の値。
    #[arg(long)]
    pub(crate) l2: Option<usize>,

    /// 隠れ層 2 の次元。未指定なら `--arch` preset の値。
    #[arg(long)]
    pub(crate) l3: Option<usize>,

    /// FT post 活性化関数 ("crelu" / "screlu" / "pairwise")。"pairwise" は前半×後半の
    /// 対応 index の積を取り L1 入力次元を半減する (L1 / L2 dense は CReLU 活性化)。
    #[arg(long, default_value = "crelu")]
    pub(crate) activation: String,

    /// FT activation (`ft_*_out` の forward 出力と `dft_*_out` の backward 勾配) も
    /// FP16 で保持する。global `--ft-fp16` を要求する (crelu / screlu / pairwise 対応)。
    ///
    /// `ft_*_out` は `sparse_ft_forward` の出力で、これを FP16 化すると後続 read +
    /// `sparse_ft_backward` の read 帯域が半減する。dft は batch 正規化で `1/batch`
    /// に比例する微小値のため、FP16 化時は loss scaling (batch 比例) で normal range
    /// に持ち上げてから格納する。
    ///
    /// 量子化誤差で棋力が変動しうるため default OFF、本番品質は SPRT で確認するまで
    /// 保証しない opt-in option。
    #[arg(long)]
    pub(crate) ft_fp16_out: bool,

    /// Ampere+ Tensor Core を TF32 mode で使う opt-in flag。`true` で cuBLAS の
    /// `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` を呼び、L1/L2/L3 dense
    /// Sgemm の FP32 入力を 10-bit mantissa の TF32 に丸めて TC mma → FP32 accum で走る
    /// (仮数精度 ~3 桁、指数範囲は FP32 同等)。default `false` では
    /// `CUBLAS_DEFAULT_MATH` (純 FP32 path、TC 不使用) で走る。
    ///
    /// 仮数 13 bit 切り捨てで dense Sgemm の数値に影響するため、品質 conservative に
    /// default OFF。LayerStack `--tf32` と同方針 (棋力 risk opt-in)。
    #[arg(long)]
    pub(crate) tf32: bool,
}
