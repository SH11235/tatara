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
//!   (`Batch { stm_indices, nstm_indices, score, wdl, per_pos_norm, n_positions }`、
//!   `PsvFileLoader` / `PrefetchedLoader` / `BucketedPrefetchedLoader`)
//! - `optimizer`: Ranger (RAdam + Lookahead) の host-side state
//!   (`RAdamHostState` / `RangerHostState`) + パラメータ + checkpoint
//!   serialise。GPU `#[kernel]` 本体は bin 側 (`bins/nnue_train`) inline、本
//!   module は host state + `radam_compute_step_size_denom` host helper のみ
//! - `trainer`: superbatch training loop driver (`TrainerBackend` trait +
//!   `TrainingConfig` + `run`)。1 batch 分の GPU step は `bins/nnue_train::
//!   GpuTrainer` (= `TrainerBackend` impl) が担う

pub mod dataloader;
pub mod optimizer;
pub mod schedule;
pub mod trainer;
