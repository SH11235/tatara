//! `nnue-train` crate — HalfKA_hm 1536-16-32 NNUE training pipeline (host 側)。
//!
//! CPU-only training pipeline library。GPU `#[kernel]` 本体は持たず、host 側の
//! schedule / dataloader / optimizer state / superbatch loop driver のみを提供
//! する。GPU kernel 定義は `bins/nnue_train` 側に inline で置く: cuda-oxide
//! rustc-codegen-cuda backend が bin entry 経由でしか `#[kernel]` を NVPTX IR
//! 化しない制約のため。
//!
//! 本 crate は CI workflow で `--exclude` せず test を通す方針。host 側ロジック
//! のみで構成し、`gpu-runtime` / `cuda-host` 等 GPU build chain には依存しない。
//!
//! ## 提供 module
//!
//! - `schedule`: learning-rate / wdl scheduler (LrScheduler / WdlScheduler trait
//!   と `Constant` / `Step` / `LinearDecay` / `CosineDecay` / `Warmup` / `Sequence`
//!   等の実装)
//! - `dataloader`: PSV file → HalfKA_hm sparse batch + prefetch
//!   (`Batch { stm_indices, nstm_indices, nnz, score, wdl, per_pos_norm, n_positions }`、
//!   `PsvFileLoader` / `PrefetchedLoader` / `BucketedPrefetchedLoader`)
//! - `optimizer`: `OptimizerKind` (ranger / radam / adamw の選択と per-step
//!   scalar の解決)、`RangerParams` (ハイパーパラメータ)、
//!   `radam_compute_step_size_denom` (GPU `radam_step` kernel に渡す値の host 側
//!   事前計算 helper)
//! - `trainer`: superbatch training loop driver (`TrainerBackend` trait +
//!   `TrainingConfig` + `run`)。1 batch 分の GPU step は `bins/nnue_train::
//!   GpuTrainer` (= `TrainerBackend` impl) が担う
//! - `validation`: held-out 検証データでの per-superbatch loss / accuracy 計測
//!   (`HeldoutSet` + sign-agreement accuracy)
//! - `experiment`: 学習 run ごとの構造化ログ (experiment.json) を組み立て
//!   incremental に書き出す (`ExperimentLogger` / `ExperimentDoc`)
//! - `init`: 重み初期化の汎用記述 (`Dist` / `Scale` / `LayerInit`) と決定論的
//!   サンプラ (`sample`) + 各アーキの既定値 (`LayerStackInit::default_uniform` /
//!   `SimpleInit::default_uniform`)。bin 側 trainer 構築子が初期重みを生成するのに使う

pub mod dataloader;
pub mod experiment;
pub mod init;
pub mod optimizer;
pub mod schedule;
pub mod trainer;
pub mod validation;
