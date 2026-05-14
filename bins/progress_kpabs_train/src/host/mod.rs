//! Host-side helpers for the KP-abs progress trainer。
//!
//! GPU 非依存の純粋ロジックを集める:
//!
//! - [`batch`]: position 群 → flat `indices` / `targets` / `per_pos_norm` 構造体
//! - [`games`]: PSV ファイルを順次読み出し、`game_ply` の減少で「ゲーム境界」
//!   を切る iterator (bullet-shogi の `GameIterator` と同等)
//! - [`progress_bin`]: YaneuraOu 互換の `progress.bin` (f64 LE × N_WEIGHTS)
//!   を読み書き
//! - [`cli`]: `clap` ベースの CLI 引数定義 (`--data` `--output` `--lr` 等)
//!
//! GPU 周り (`GpuTrainer`、4 kernel の launch、device buffer 管理) は kernels
//! が `src/main.rs` の `#[kernel]` を直接参照する都合上 main.rs に置く。本
//! module は GPU を持たない環境でも build / test できる。

pub mod batch;
pub mod cli;
pub mod games;
pub mod progress_bin;

/// Adam の β1 (bullet-shogi 上流と同値のデフォルト)。
pub const ADAM_BETA1: f32 = 0.9;
/// Adam の β2。
pub const ADAM_BETA2: f32 = 0.999;
/// Adam の epsilon (ゼロ割防止項)。
pub const ADAM_EPS: f32 = 1e-8;

/// 1 position あたりの最大 active KP-absolute 特徴 index 数。
///
/// 将棋の盤上 38 + 持ち駒の最大組合せでも、1 駒あたり 2 index (sq_bk / sq_wk
/// 由来) を考慮した上限は 76 程度。bullet-shogi の慣例に倣い 80 まで pad、
/// 余りは `-1` sentinel で埋める。kernel 側 `max_inds` パラメータと一致。
pub const MAX_INDS_PER_POS: usize = 80;
