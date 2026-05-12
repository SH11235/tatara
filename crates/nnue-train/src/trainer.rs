//! Training-loop driver — host-side superbatch loop for the v102 NNUE trainer。
//!
//! bullet-shogi `crates/bullet_lib/src/value.rs::ValueTrainerInner` 相当の loop
//! を、GPU 非依存の trait (`TrainerBackend`) 越しに駆動する。`bins/nnue_train::
//! GpuTrainer` が `TrainerBackend` を impl し、本 module の [`run`] がそれを
//! 呼ぶ (Stage 3-0 規約: `nnue-train` crate は `gpu-runtime` に依存せず、kernel
//! launch は bin 側に置く、Stage 1-9 pattern)。
//!
//! ## ループ構造 (bullet `ValueTrainerInner::train_custom` の簡易版)
//!
//! ```text
//! for sb in start_superbatch..=end_superbatch:
//!     for batch_idx in 0..batches_per_superbatch:
//!         lr  = lr_scheduler.lr(batch_idx, sb)
//!         wdl = wdl_scheduler.blend(batch_idx, sb, end_superbatch)
//!         fill Batch + per-position bucket from the PSV stream
//!           (EOF → 同 file を開き直す = 次 epoch)
//!         loss += backend.train_step(batch, buckets, lr, wdl, loss_kind)
//!     report(sb, loss / positions, pos/s, ETA)
//!     if sb % save_rate == 0 || sb == end_superbatch:
//!         backend.save_checkpoint("{output_dir}/{net_id}-{sb}.bin")          # 量子化 (推論用)
//!         backend.save_resume_checkpoint("{output_dir}/{net_id}-{sb}.ckpt", sb)  # raw f32 + Ranger state (resume 用、Issue #88)
//!         if keep_raw_checkpoints == Some(n): 直近 n 個より古い *.ckpt を削除
//! ```
//!
//! `start_superbatch != 1` で呼ぶと resume になる: lr/wdl scheduler は superbatch
//! index 駆動 (`StepLR` は `(sb-1)/step`) なので `start_superbatch` を渡せば lr が
//! 自動で正しい値に戻る (weight + optimizer state の復元自体は backend 側、
//! `bins/nnue_train --resume` が `GpuTrainer::load_raw_checkpoint` 経由で行う)。
//!
//! per-position の output bucket は progress8kpabs (`ShogiProgressKPAbs::bucket`、
//! YaneuraOu 互換 `progress.bin` の重み付き和 → sigmoid → `floor(p * 8)` を
//! `0..=7` に clamp) で求める。bullet v102 の network は 9 bucket を持つが
//! progress8kpabs は bucket 8 を使わない (bullet `ShogiLayerStackBucket9`
//! と同じ「9bucket 互換、bucket8 未使用」挙動)。`progress.bin` 未指定時は
//! 重みが全 0 で `p = sigmoid(0) = 0.5` → 全 position が bucket 4 になる。
//!
//! ## bullet 上流からの差分
//!
//! - bullet `ValueTrainerInner` (`bullet_core::Trainer` + `DataLoader` trait +
//!   `LoggingConfig` 等) は使わず、Stage 1 `bins/progress_kpabs_train::
//!   train_one_epoch` 流儀の直書き loop に簡素化 (Stage 1-1 / 3-1 / 3-4 と同じ
//!   bullet trait 削除ポリシー)。
//! - PSV stream は [`crate::dataloader::BucketedPrefetchedLoader`] (Issue #89)
//!   経由で読む: `--threads` 本の worker が PSV パース + HalfKA_hm sparse 抽出 +
//!   progress8kpabs bucket 計算を `PackedSfenValue::decode()` 1 回で済ませて
//!   `(Batch, per-position bucket)` を先読み供給する。EOF で同 file を開き直して
//!   次 epoch とする / `--score-drop-abs` skip / 空 file の `MAX_BARREN_PASSES`
//!   ガードは loader 内の `PsvEpochReader` が担う (旧 `EpochStream` を移設)。
//!   bullet の `--epoch-file-shuffle` (epoch ごと file shuffle) は本 stage では
//!   未実装 — CLI フラグは受けても no-op。**worker 数 ≥ 2 では 1 epoch 内の
//!   position の順序が非決定的になる** (`BucketedPrefetchedLoader` doc 参照;
//!   training では問題ない)。
//! - bullet `--score-drop-abs` (`5c4871c`: `|score| >= t` の position の
//!   per-position loss weight を 0 にする) は本実装では **batch に push しない
//!   (skip)** で近似する。loss/gradient へ寄与しない点は同じだが、batch の
//!   構成 (slot 割当・順序) は厳密一致しない。

use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shogi_features::progress_kpabs::ShogiProgressKPAbs;

use crate::dataloader::{Batch, BucketedPrefetchedLoader};
use crate::schedule::{LrScheduler, WdlScheduler};

// =============================================================================
// LossKind — どの loss kernel で 1 step を回すか
// =============================================================================

/// training step で使う loss の種別と固定パラメータ。
///
/// バックエンド ([`TrainerBackend::train_step`]) はこの enum で分岐して対応する
/// loss kernel (`bins/nnue_train` の `loss_wdl` / `loss_wrm`) を起動する。GPU には
/// 触らない (CPU-only crate に置ける、Stage 3-0 規約)。
///
/// - [`LossKind::Sigmoid`] — 旧来の plain sigmoid-MSE (`p = sigmoid(out * scale)`,
///   target = `lambda*wdl + (1-lambda)*sigmoid(score * scale)`)。net_output が
///   cp 単位 (`out ≈ cp`) で収束する。`bins/nnue_train` の `loss_wdl` kernel に対応。
/// - [`LossKind::Wrm`] — bullet win-rate-model loss (nodchip 流、v102 recipe
///   `--win-rate-model --wrm-in-scaling 340 --wrm-nnue2score 600`)。prediction /
///   target 双方に WRM を適用するため net_output が `out ≈ cp / nnue2score` (O(1)) で
///   収束し、`crates/nnue-format` の量子化 (`QA=127 / QB=64 / FV_SCALE=28`) と整合する。
///   `bins/nnue_train` の `loss_wrm` kernel に対応 (CPU reference は
///   `gpu_kernels::pointwise::loss_wrm::loss_wrm_cpu`)。target 側 in_scaling (380) と
///   offset (270) は bullet ハードコード、prediction 側 in_scaling は `--wrm-in-scaling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossKind {
    /// plain sigmoid-MSE。`scale = 1.0 / --scale` (v102 は `1/290`)。
    Sigmoid { scale: f32 },
    /// bullet win-rate-model loss。`nnue2score = --wrm-nnue2score` (600)、
    /// `in_scaling = --wrm-in-scaling` (340、prediction 側のみ)。
    Wrm { nnue2score: f32, in_scaling: f32 },
}

impl LossKind {
    /// CLI / config から渡されたパラメータが loss kernel に流せる値か検証する。
    fn validate(&self) -> io::Result<()> {
        match *self {
            LossKind::Sigmoid { scale } => {
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(io::Error::other(format!(
                        "loss scale must be finite and > 0 (got {scale})"
                    )));
                }
            }
            LossKind::Wrm {
                nnue2score,
                in_scaling,
            } => {
                if !nnue2score.is_finite() || nnue2score <= 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm nnue2score must be finite and > 0 (got {nnue2score})"
                    )));
                }
                if !in_scaling.is_finite() || in_scaling <= 0.0 {
                    return Err(io::Error::other(format!(
                        "wrm in_scaling must be finite and > 0 (got {in_scaling})"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for LossKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            LossKind::Sigmoid { scale } => write!(f, "sigmoid-MSE(scale={scale:.6})"),
            LossKind::Wrm {
                nnue2score,
                in_scaling,
            } => write!(f, "wrm(nnue2score={nnue2score}, in_scaling={in_scaling})"),
        }
    }
}

// =============================================================================
// TrainerBackend — 1 batch 分の forward → loss → backward → optimizer step
// =============================================================================

/// 1 batch 分の training step を実行する backend。
///
/// `bins/nnue_train::GpuTrainer` が impl する。本 trait を介すことで loop driver
/// を GPU 非依存に保ち (CPU-only crate に置ける)、mock backend で単体テストできる。
pub trait TrainerBackend {
    /// 1 batch 分 (forward → loss kernel → backward → Ranger step) を実行し、
    /// batch 全体で累積した二乗誤差 (`Σ err²`、まだ position 数で割っていない値)
    /// を返す。caller が報告時に position 数で割って平均 loss にする。
    ///
    /// - `batch`: HalfKA_hm sparse + score/wdl/norm (`batch.n_positions` が有効件数)
    /// - `bucket_idx`: `batch.n_positions` 個の output bucket index (`0..=8`)
    /// - `lr`: learning rate (`LrScheduler` 由来)
    /// - `wdl_lambda`: WDL blend lambda (`WdlScheduler` 由来、loss kernel の `lambda`)
    /// - `loss`: どの loss kernel を起動するか (sigmoid-MSE / WRM) + 固定パラメータ
    fn train_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> io::Result<f64>;

    /// 現在の weight を量子化 NNUE binary として `path` に書き出す
    /// (Stage 3-3 `nnue-format` の `save_quantised` 相当を backend 側で実行)。
    fn save_checkpoint(&mut self, path: &Path) -> io::Result<()>;

    /// resume 用 **raw f32 checkpoint** を `path` に書き出す (Issue #88)。
    ///
    /// 量子化 `.bin` ([`TrainerBackend::save_checkpoint`]) と違い、全 weight group の
    /// raw f32 値に加えて optimizer state (Ranger の `m` / `v` / `slow`) と step counter、
    /// および現在の `superbatch` 番号を保存する。これを `--resume` で読み戻すと
    /// optimizer state ごと学習を再開できる (`--init-from` は weight だけ注入し
    /// optimizer state を reset するため真の resume にはならない)。
    ///
    /// backend 側 (`bins/nnue_train::GpuTrainer`) は device → host download → file 書き出し
    /// (`.tmp` へ書いてから `rename` で atomic に置換) を行う。`crates/nnue-train` は
    /// GPU 非依存なので、本 trait は **path / superbatch 番号だけ** を受け取り device IO は
    /// backend 任せ (Stage 3-0 規約)。
    fn save_resume_checkpoint(&mut self, path: &Path, superbatch: usize) -> io::Result<()>;
}

// =============================================================================
// TrainingConfig
// =============================================================================

/// 1 回の [`run`] に渡す training hyper-parameter 一式。
///
/// bullet `TrainingSchedule` + `examples/shogi_layerstack.rs` CLI 引数のうち
/// v102 recipe (memory `project_v102_recipe.md`) の再現に必要な subset を持つ。
/// learning rate / WDL schedule は別に `LrScheduler` / `WdlScheduler` を渡す。
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// network id — checkpoint file 名にのみ使う (`{net_id}-{sb}.bin`)。
    pub net_id: String,
    /// checkpoint の出力先 directory (呼び出し側で作成しておくこと)。
    pub output_dir: PathBuf,
    /// 開始 superbatch (1-indexed, inclusive)。
    pub start_superbatch: usize,
    /// 終了 superbatch (inclusive)。
    pub end_superbatch: usize,
    /// 1 superbatch あたりの batch 数。
    pub batches_per_superbatch: usize,
    /// 1 batch あたりの position 数。
    pub batch_size: usize,
    /// `save_rate` superbatch ごと (および末尾) に checkpoint を書き出す。
    pub save_rate: usize,
    /// `Some(n)` のとき、新しい raw checkpoint (`{net_id}-{sb}.ckpt`) を書いた後、
    /// 直近 `n` 個より古い raw checkpoint を削除する (Issue #88、`--keep-checkpoints`)。
    /// `None` は全 raw checkpoint を保持 (default; raw state は ~1.8GB/個 なので
    /// save-rate × superbatches が大きい長期ランでは明示指定推奨)。量子化 `.bin`
    /// (~116MB) は本設定に関わらず常に全保持する (推論 artifact なので保守的に)。
    pub keep_raw_checkpoints: Option<usize>,
    /// どの loss kernel で学習するか (sigmoid-MSE / bullet WRM) + 固定パラメータ。
    pub loss: LossKind,
    /// `Some(t)` のとき `|score| >= t` の position を skip する (bullet `--score-drop-abs`)。
    pub score_drop_abs: Option<i32>,
    /// dataloader の prefetch worker 数 (`--threads`、Issue #89)。`0` は `1` 扱い。
    /// `1` で従来の決定論的逐次 read 相当、`>= 2` で並列パース (1 epoch 内の
    /// position 順序は非決定的になる; [`BucketedPrefetchedLoader`] doc 参照)。
    pub threads: usize,
}

impl TrainingConfig {
    fn validate(&self) -> io::Result<()> {
        if self.start_superbatch == 0 {
            return Err(io::Error::other(
                "start_superbatch must be >= 1 (1-indexed)",
            ));
        }
        if self.end_superbatch < self.start_superbatch {
            return Err(io::Error::other(format!(
                "end_superbatch ({}) < start_superbatch ({})",
                self.end_superbatch, self.start_superbatch
            )));
        }
        if self.batches_per_superbatch == 0 {
            return Err(io::Error::other("batches_per_superbatch must be >= 1"));
        }
        if self.batch_size == 0 {
            return Err(io::Error::other("batch_size must be >= 1"));
        }
        if self.save_rate == 0 {
            return Err(io::Error::other("save_rate must be >= 1"));
        }
        if let Some(0) = self.keep_raw_checkpoints {
            return Err(io::Error::other(
                "keep_raw_checkpoints must be >= 1 when set (0 would delete every raw checkpoint)",
            ));
        }
        self.loss.validate()?;
        if let Some(t) = self.score_drop_abs {
            if t < 1 {
                return Err(io::Error::other(format!(
                    "score_drop_abs must be >= 1 (got {t}); a non-positive threshold would drop every position"
                )));
            }
        }
        Ok(())
    }
}

// =============================================================================
// run — superbatch loop
// =============================================================================

/// superbatch training loop を実行し、`cfg.output_dir` 配下に checkpoint を書き出す。
///
/// - `backend`: GPU step を実行する backend (`bins/nnue_train::GpuTrainer`)
/// - `data_path`: PSV file (`PackedSfenValue` × N、40 bytes 固定)
/// - `progress`: progress8kpabs 重み (`--progress-coeff` 未指定なら zero-weight default → 全 bucket 4)。
///   重みは process-global `OnceLock` なので呼び出し前に `load_from_bin` 済であること
/// - `lr_scheduler` / `wdl_scheduler`: superbatch / batch index から lr / wdl lambda を返す
/// - `cfg`: hyper-parameter (superbatch 範囲、batch 構成、save 間隔、loss scale、score-drop-abs、`threads`)
///
/// PSV stream は [`BucketedPrefetchedLoader`] (Issue #89) で `cfg.threads` 本の
/// worker から `decode()` 1 回 / position の bucket-aware 先読み + ring-buffer
/// 再利用される。worker 数 ≥ 2 では 1 epoch 内の position 順序が非決定的になる
/// 点に注意 (training では問題ない)。
pub fn run<B, L, W>(
    backend: &mut B,
    data_path: &Path,
    progress: &ShogiProgressKPAbs,
    lr_scheduler: &L,
    wdl_scheduler: &W,
    cfg: &TrainingConfig,
) -> io::Result<()>
where
    B: TrainerBackend,
    L: LrScheduler,
    W: WdlScheduler,
{
    cfg.validate()?;

    let mut loader = BucketedPrefetchedLoader::spawn(
        data_path,
        cfg.batch_size,
        cfg.score_drop_abs,
        cfg.threads,
        *progress,
    )?;

    println!(
        "[train] data={} | net_id={} | superbatches {}..={} | {} batches/sb x bs {} \
         | lr-sched: {lr_scheduler} | wdl-sched: {wdl_scheduler} | loss: {} | score-drop-abs {:?} | dataloader threads {}",
        data_path.display(),
        cfg.net_id,
        cfg.start_superbatch,
        cfg.end_superbatch,
        cfg.batches_per_superbatch,
        cfg.batch_size,
        cfg.loss,
        cfg.score_drop_abs,
        cfg.threads.max(1),
    );

    let positions_per_sb =
        (cfg.batches_per_superbatch as u64).saturating_mul(cfg.batch_size as u64);
    let run_start = Instant::now();

    for sb in cfg.start_superbatch..=cfg.end_superbatch {
        let sb_start = Instant::now();
        let mut sb_loss: f64 = 0.0;
        let mut sb_positions: u64 = 0;

        for batch_idx in 0..cfg.batches_per_superbatch {
            let lr = lr_scheduler.lr(batch_idx, sb);
            let wdl = wdl_scheduler.blend(batch_idx, sb, cfg.end_superbatch);

            let (batch, buckets) = loader.next_batch()?.ok_or_else(|| {
                io::Error::other(
                    "dataloader stopped supplying batches unexpectedly (workers exited without an error)",
                )
            })?;
            let n_pos = batch.n_positions;

            let loss = backend.train_step(&batch, &buckets, lr, wdl, cfg.loss)?;
            loader.recycle((batch, buckets));
            sb_loss += loss;
            sb_positions += n_pos as u64;
        }

        let sb_secs = sb_start.elapsed().as_secs_f64().max(1e-9);
        let mean_loss = if sb_positions == 0 {
            f64::NAN
        } else {
            sb_loss / sb_positions as f64
        };
        let pos_per_sec = sb_positions as f64 / sb_secs;
        let remaining_positions = positions_per_sb.saturating_mul((cfg.end_superbatch - sb) as u64);
        let eta_secs = if pos_per_sec > 0.0 {
            remaining_positions as f64 / pos_per_sec
        } else {
            f64::NAN
        };
        let lr_now = lr_scheduler.lr(0, sb);
        let wdl_now = wdl_scheduler.blend(0, sb, cfg.end_superbatch);

        println!(
            "[train] superbatch {}/{} | loss {:.6} | {:.0} pos/s | lr {:.4e} | wdl {:.3} | sb {:.1}s | ETA {}",
            sb,
            cfg.end_superbatch,
            mean_loss,
            pos_per_sec,
            lr_now,
            wdl_now,
            sb_secs,
            format_hms(eta_secs),
        );

        // MSRV 1.85: `usize::is_multiple_of` は 1.87 stable なので使わない (memory 既知の罠)。
        if sb % cfg.save_rate == 0 || sb == cfg.end_superbatch {
            let path = cfg.output_dir.join(format!("{}-{}.bin", cfg.net_id, sb));
            backend.save_checkpoint(&path)?;
            println!("[train] checkpoint saved: {}", path.display());

            // resume 用 raw checkpoint (Issue #88): weight raw f32 + Ranger state + step + sb。
            let raw_path = cfg.output_dir.join(format!("{}-{}.ckpt", cfg.net_id, sb));
            backend.save_resume_checkpoint(&raw_path, sb)?;
            println!("[train] resume checkpoint saved: {}", raw_path.display());

            if let Some(keep) = cfg.keep_raw_checkpoints {
                prune_old_raw_checkpoints(&cfg.output_dir, &cfg.net_id, keep);
            }
        }
    }

    println!(
        "[train] done in {} ({} superbatches)",
        format_hms(run_start.elapsed().as_secs_f64()),
        cfg.end_superbatch + 1 - cfg.start_superbatch,
    );
    Ok(())
}

/// `{net_id}-{sb}.ckpt` 形式の raw checkpoint のうち、superbatch 番号 (`sb`) の
/// 大きい順に `keep` 個だけ残し、それより古いものを削除する (Issue #88、
/// `--keep-checkpoints`)。量子化 `.bin` には触らない (推論 artifact なので全保持)。
///
/// 削除失敗 (権限・他プロセス) は警告のみで `run` を止めない (training 続行優先)。
/// `keep == 0` は呼ばれない想定 (`TrainingConfig::validate` で reject 済) だが、
/// 万一渡されても全削除はしない (no-op で警告)。
fn prune_old_raw_checkpoints(output_dir: &Path, net_id: &str, keep: usize) {
    if keep == 0 {
        eprintln!(
            "[train] warning: keep_raw_checkpoints=0 ignored (would delete all raw checkpoints)"
        );
        return;
    }
    let prefix = format!("{net_id}-");
    let entries = match std::fs::read_dir(output_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "[train] warning: cannot read {} to prune raw checkpoints: {e}",
                output_dir.display()
            );
            return;
        }
    };
    // (superbatch 番号, パス) を収集。`{net_id}-<digits>.ckpt` だけ対象。
    let mut found: Vec<(usize, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(num_str) = rest.strip_suffix(".ckpt") else {
            continue;
        };
        if let Ok(sb) = num_str.parse::<usize>() {
            found.push((sb, path));
        }
    }
    if found.len() <= keep {
        return;
    }
    // superbatch 降順 → 先頭 `keep` 個を残し、残りを削除。
    found.sort_by_key(|(sb, _)| std::cmp::Reverse(*sb));
    for (sb, path) in found.into_iter().skip(keep) {
        match std::fs::remove_file(&path) {
            Ok(()) => println!(
                "[train] pruned old raw checkpoint: {} (sb {sb})",
                path.display()
            ),
            Err(e) => eprintln!(
                "[train] warning: failed to prune {} (sb {sb}): {e}",
                path.display()
            ),
        }
    }
}

/// 秒数を `1h23m45s` / `12m05s` / `42s` 形式に整形する (`??` if not finite)。
fn format_hms(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "??".to_string();
    }
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{ConstantWDL, StepLR};

    fn sample_psv_path() -> PathBuf {
        // crates/nnue-train/Cargo.toml から相対で shogi-format/tests/data/sample.psv。
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/nnue-train has a parent dir")
            .join("shogi-format/tests/data/sample.psv")
    }

    /// loop driver の挙動 (step 回数 / checkpoint path / bucket 受け渡し) を検証する
    /// 最小 backend。GPU には触らず、loss は単調減少する dummy 値を返す。
    struct MockBackend {
        steps: usize,
        saves: Vec<PathBuf>,
        /// raw resume checkpoint の保存呼び出し (path, superbatch)。
        resume_saves: Vec<(PathBuf, usize)>,
        last_buckets: Vec<i32>,
        max_batch_positions: usize,
        seen_lr: Vec<f32>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                steps: 0,
                saves: Vec::new(),
                resume_saves: Vec::new(),
                last_buckets: Vec::new(),
                max_batch_positions: 0,
                seen_lr: Vec::new(),
            }
        }
    }

    impl TrainerBackend for MockBackend {
        fn train_step(
            &mut self,
            batch: &Batch,
            bucket_idx: &[i32],
            lr: f32,
            wdl_lambda: f32,
            loss: LossKind,
        ) -> io::Result<f64> {
            assert_eq!(
                bucket_idx.len(),
                batch.n_positions,
                "one bucket per position"
            );
            assert!(batch.n_positions <= batch.batch_size);
            assert!(lr > 0.0, "lr should be positive");
            assert!(
                loss.validate().is_ok(),
                "loss params should be valid: {loss}"
            );
            assert!(wdl_lambda.is_finite());
            assert!(
                bucket_idx.iter().all(|&b| (0..9).contains(&b)),
                "bucket in 0..9: {bucket_idx:?}"
            );
            self.steps += 1;
            self.last_buckets = bucket_idx.to_vec();
            self.max_batch_positions = self.max_batch_positions.max(batch.n_positions);
            self.seen_lr.push(lr);
            // 単調減少する dummy loss (issue の「loss 推移 monotonic decreasing 観察」相当)。
            Ok(1.0 / self.steps as f64)
        }

        fn save_checkpoint(&mut self, path: &Path) -> io::Result<()> {
            self.saves.push(path.to_path_buf());
            Ok(())
        }

        fn save_resume_checkpoint(&mut self, path: &Path, superbatch: usize) -> io::Result<()> {
            self.resume_saves.push((path.to_path_buf(), superbatch));
            Ok(())
        }
    }

    fn base_cfg() -> TrainingConfig {
        TrainingConfig {
            net_id: "test".to_string(),
            output_dir: PathBuf::from("/tmp/nnue-train-trainer-test-unused"),
            start_superbatch: 1,
            end_superbatch: 3,
            batches_per_superbatch: 2,
            batch_size: 8,
            save_rate: 2,
            keep_raw_checkpoints: None,
            loss: LossKind::Sigmoid { scale: 1.0 / 290.0 },
            score_drop_abs: None,
            threads: 2,
        }
    }

    fn run_drives_superbatches_with_threads(threads: usize) {
        let progress = ShogiProgressKPAbs; // zero weights → p = sigmoid(0) = 0.5 → bucket 4
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            threads,
            ..base_cfg()
        };
        let mut backend = MockBackend::new();

        run(&mut backend, &sample_psv_path(), &progress, &lr, &wdl, &cfg).expect("run ok");

        // 3 superbatch × 2 batch = 6 train_step。
        assert_eq!(backend.steps, 6);
        assert_eq!(
            backend.max_batch_positions, cfg.batch_size,
            "every batch fully filled (file wraps)"
        );
        // save_rate=2 → sb 2 (2 % 2 == 0) と sb 3 (== end_superbatch) で save。
        assert_eq!(
            backend.saves,
            vec![
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-2.bin"),
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-3.bin"),
            ]
        );
        // raw resume checkpoint は同 superbatch で `{net_id}-{sb}.ckpt` に保存される。
        assert_eq!(
            backend.resume_saves,
            vec![
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-2.ckpt"),
                    2
                ),
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-3.ckpt"),
                    3
                ),
            ]
        );
        // 各 superbatch で lr が gamma 倍 (StepLR step=1, gamma=0.9)。batch 内は一定。
        // (lr は train_step 呼び出し順 = run の loop 順で決まり、dataloader の worker
        // 順序には依らない。)
        assert!((backend.seen_lr[0] - 1.0e-3).abs() < 1e-9);
        assert!((backend.seen_lr[2] - 1.0e-3 * 0.9).abs() < 1e-9); // 2nd superbatch, 1st batch
        // zero-weight progress → 全 position が bucket 4。
        assert!(!backend.last_buckets.is_empty());
        assert!(
            backend.last_buckets.iter().all(|&b| b == 4),
            "got {:?}",
            backend.last_buckets
        );
    }

    #[test]
    fn run_drives_superbatches_and_writes_checkpoints_single_worker() {
        // threads=1: 決定論的逐次 read 相当のパス。
        run_drives_superbatches_with_threads(1);
    }

    #[test]
    fn run_drives_superbatches_and_writes_checkpoints_multi_worker() {
        // threads>=2: 並列パース。順序は非決定的でも step 回数 / checkpoint / bucket /
        // lr schedule は不変。
        run_drives_superbatches_with_threads(4);
    }

    #[test]
    fn run_with_start_superbatch_offset_resumes_loop_and_lr_schedule() {
        // Issue #88: `start_superbatch != 1` (resume) で回したとき:
        //  - 正しい step 回数 (start..=end の superbatch 数 × batches/sb)
        //  - checkpoint / resume-checkpoint が start..=end の番号で命名される
        //  - lr schedule が offset を反映する (StepLR sb=3 = start * gamma^2)
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            start_superbatch: 3,
            end_superbatch: 5,
            save_rate: 2, // sb 4 (4 % 2 == 0) と sb 5 (== end) で save
            threads: 1,
            ..base_cfg()
        };
        let mut backend = MockBackend::new();
        run(&mut backend, &sample_psv_path(), &progress, &lr, &wdl, &cfg).expect("run ok");

        // 3 superbatch (3,4,5) × 2 batch = 6 step。
        assert_eq!(backend.steps, 6);
        // save_rate=2 → sb 4, sb 5。番号は start_superbatch offset を反映。
        assert_eq!(
            backend.saves,
            vec![
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-4.bin"),
                PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-5.bin"),
            ]
        );
        assert_eq!(
            backend.resume_saves,
            vec![
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-4.ckpt"),
                    4
                ),
                (
                    PathBuf::from("/tmp/nnue-train-trainer-test-unused/test-5.ckpt"),
                    5
                ),
            ]
        );
        // lr schedule: sb 3 (1st batch) = start * gamma^((3-1)/1) = start * gamma^2。
        // StepLR は `start * gamma^((sb-1)/step)` (resume 時は sb を渡せば自動で正しい lr)。
        let expected_sb3 = 1.0e-3 * 0.9_f32 * 0.9_f32;
        assert!(
            (backend.seen_lr[0] - expected_sb3).abs() < 1e-9,
            "sb3 lr = {} expected {expected_sb3}",
            backend.seen_lr[0]
        );
        // sb 5 (5th step = 1st batch of sb 5) = start * gamma^4。
        let expected_sb5 = 1.0e-3 * 0.9_f32.powi(4);
        assert!(
            (backend.seen_lr[4] - expected_sb5).abs() < 1e-9,
            "sb5 lr = {} expected {expected_sb5}",
            backend.seen_lr[4]
        );
    }

    #[test]
    fn keep_raw_checkpoints_prunes_oldest() {
        // Issue #88: `--keep-checkpoints N` 相当。end_superbatch=6, save_rate=1 で
        // 6 個の .ckpt が書かれるが keep=2 なら直近 2 個 (sb 5, 6) だけ残る。
        // (MockBackend は実 file を書かないので、テスト用に空 file を実 dir に置いて
        //  prune ロジックを exercise する。)
        let dir = std::env::temp_dir().join(format!(
            "nnue-train-trainer-prune-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");

        // 既存 .ckpt と .bin を散らかしておく (`.bin` は prune 対象外であることを確認)。
        for sb in 1..=4usize {
            std::fs::write(dir.join(format!("net-{sb}.ckpt")), b"x").unwrap();
            std::fs::write(dir.join(format!("net-{sb}.bin")), b"x").unwrap();
        }
        // 別 net_id の .ckpt は触られないこと。
        std::fs::write(dir.join("other-1.ckpt"), b"x").unwrap();
        // 数値でない名前は無視されること。
        std::fs::write(dir.join("net-foo.ckpt"), b"x").unwrap();

        prune_old_raw_checkpoints(&dir, "net", 2);

        // sb 3, 4 だけ残る (sb 1, 2 削除)。
        assert!(
            !dir.join("net-1.ckpt").exists(),
            "net-1.ckpt should be pruned"
        );
        assert!(
            !dir.join("net-2.ckpt").exists(),
            "net-2.ckpt should be pruned"
        );
        assert!(dir.join("net-3.ckpt").exists(), "net-3.ckpt should be kept");
        assert!(dir.join("net-4.ckpt").exists(), "net-4.ckpt should be kept");
        // .bin は全部残る。
        for sb in 1..=4usize {
            assert!(
                dir.join(format!("net-{sb}.bin")).exists(),
                "net-{sb}.bin kept"
            );
        }
        // 別 net_id / 非数値名は無傷。
        assert!(dir.join("other-1.ckpt").exists());
        assert!(dir.join("net-foo.ckpt").exists());

        // keep >= 個数 のときは何も消さない。
        prune_old_raw_checkpoints(&dir, "net", 10);
        assert!(dir.join("net-3.ckpt").exists());
        assert!(dir.join("net-4.ckpt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_raw_checkpoints_sorts_numerically_not_lexically() {
        // Issue #88 regression: superbatch 番号は parse 済 `usize` で降順 sort される。
        // 9, 10, 11 を keep=2 で prune したとき、数値 sort なら最古の 9 が消え 10/11 が残る。
        // lexical (string) sort に regress すると "10" < "11" < "9" となり 11 を誤って消す。
        let dir = std::env::temp_dir().join(format!(
            "nnue-train-trainer-prune-numeric-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");

        for sb in [9usize, 10, 11] {
            std::fs::write(dir.join(format!("net-{sb}.ckpt")), b"x").unwrap();
        }

        prune_old_raw_checkpoints(&dir, "net", 2);

        assert!(
            !dir.join("net-9.ckpt").exists(),
            "net-9.ckpt should be pruned (smallest superbatch by numeric sort)"
        );
        assert!(
            dir.join("net-10.ckpt").exists(),
            "net-10.ckpt should be kept (lexical sort would wrongly prune it)"
        );
        assert!(
            dir.join("net-11.ckpt").exists(),
            "net-11.ckpt should be kept (newest)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_data_file_errors_instead_of_looping_forever() {
        // 空 file は即 EOF → epoch wrap が無限ループする危険があるが、dataloader 内の
        // `PsvEpochReader` の `MAX_BARREN_PASSES` ガードで error にして抜け、worker が
        // error slot 経由で run に伝える。
        let progress = ShogiProgressKPAbs;
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 1.0,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = TrainingConfig {
            end_superbatch: 1,
            batches_per_superbatch: 1,
            threads: 1,
            ..base_cfg()
        };

        let tmp = std::env::temp_dir().join(format!(
            "nnue-train-trainer-empty-{}.psv",
            std::process::id()
        ));
        std::fs::write(&tmp, b"").expect("write empty psv");

        let mut backend = MockBackend::new();
        let result = run(&mut backend, &tmp, &progress, &lr, &wdl, &cfg);
        let _ = std::fs::remove_file(&tmp);

        let err = result.expect_err("empty data file should error, not hang");
        assert!(
            err.to_string().contains("no usable positions"),
            "got: {err}"
        );
        assert_eq!(backend.steps, 0, "no step should run on an empty data file");
    }

    #[test]
    fn config_validate_rejects_bad_ranges() {
        assert!(base_cfg().validate().is_ok());
        assert!(
            TrainingConfig {
                start_superbatch: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                start_superbatch: 5,
                end_superbatch: 4,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                batch_size: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                batches_per_superbatch: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                save_rate: 0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                keep_raw_checkpoints: Some(0),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                keep_raw_checkpoints: Some(3),
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Sigmoid { scale: 0.0 },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Sigmoid { scale: f32::NAN },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: 340.0
                },
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 0.0,
                    in_scaling: 340.0
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss: LossKind::Wrm {
                    nnue2score: 600.0,
                    in_scaling: -1.0
                },
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        // score-drop-abs は >= 1。0 や負値は「全 position を drop」になるので reject。
        assert!(
            TrainingConfig {
                score_drop_abs: Some(0),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                score_drop_abs: Some(-1),
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                score_drop_abs: Some(32000),
                ..base_cfg()
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn format_hms_renders_expected_buckets() {
        assert_eq!(format_hms(0.0), "0s");
        assert_eq!(format_hms(42.0), "42s");
        assert_eq!(format_hms(125.0), "2m05s");
        assert_eq!(format_hms(3661.0), "1h01m01s");
        assert_eq!(format_hms(f32::NAN as f64), "??");
        assert_eq!(format_hms(-1.0), "??");
    }
}
