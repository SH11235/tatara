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
//!         loss += backend.train_step(batch, buckets, lr, wdl, loss_scale)
//!     report(sb, loss / positions, pos/s, ETA)
//!     if sb % save_rate == 0 || sb == end_superbatch:
//!         backend.save_checkpoint("{output_dir}/{net_id}-{sb}.bin")
//! ```
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
//! - PSV stream は `PsvFileLoader` を逐次読み、EOF で同 file を開き直して次
//!   epoch とする。bullet の `--epoch-file-shuffle` (epoch ごと file shuffle) は
//!   本 stage では未実装 — CLI フラグは受けても no-op。
//! - bullet `--score-drop-abs` (`5c4871c`: `|score| >= t` の position の
//!   per-position loss weight を 0 にする) は本実装では **batch に push しない
//!   (skip)** で近似する。loss/gradient へ寄与しない点は同じだが、batch の
//!   構成 (slot 割当・順序) は厳密一致しない。

use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shogi_features::halfka_hm::MAX_ACTIVE_FEATURES;
use shogi_features::progress_kpabs::ShogiProgressKPAbs;
use shogi_format::PackedSfenValue;

use crate::dataloader::{Batch, PsvFileLoader};
use crate::schedule::{LrScheduler, WdlScheduler};

// =============================================================================
// TrainerBackend — 1 batch 分の forward → loss → backward → optimizer step
// =============================================================================

/// 1 batch 分の training step を実行する backend。
///
/// `bins/nnue_train::GpuTrainer` が impl する。本 trait を介すことで loop driver
/// を GPU 非依存に保ち (CPU-only crate に置ける)、mock backend で単体テストできる。
pub trait TrainerBackend {
    /// 1 batch 分 (forward → `loss_wdl` → backward → Ranger step) を実行し、
    /// batch 全体で累積した二乗誤差 (`Σ err²`、まだ position 数で割っていない値)
    /// を返す。caller が報告時に position 数で割って平均 loss にする。
    ///
    /// - `batch`: HalfKA_hm sparse + score/wdl/norm (`batch.n_positions` が有効件数)
    /// - `bucket_idx`: `batch.n_positions` 個の output bucket index (`0..=8`)
    /// - `lr`: learning rate (`LrScheduler` 由来)
    /// - `wdl_lambda`: WDL blend lambda (`WdlScheduler` 由来、`loss_wdl` kernel の `lambda`)
    /// - `loss_scale`: sigmoid scale (`1.0 / --scale`、`loss_wdl` kernel の `scale`)
    fn train_step(
        &mut self,
        batch: &Batch,
        bucket_idx: &[i32],
        lr: f32,
        wdl_lambda: f32,
        loss_scale: f32,
    ) -> io::Result<f64>;

    /// 現在の weight を量子化 NNUE binary として `path` に書き出す
    /// (Stage 3-3 `nnue-format` の `save_quantised` 相当を backend 側で実行)。
    fn save_checkpoint(&mut self, path: &Path) -> io::Result<()>;
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
    /// sigmoid loss scale (`1.0 / --scale`、v102 は `1/290`)。
    pub loss_scale: f32,
    /// `Some(t)` のとき `|score| >= t` の position を skip する (bullet `--score-drop-abs`)。
    pub score_drop_abs: Option<i32>,
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
        if !self.loss_scale.is_finite() || self.loss_scale <= 0.0 {
            return Err(io::Error::other(format!(
                "loss_scale must be finite and > 0 (got {})",
                self.loss_scale
            )));
        }
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
// EpochStream — PSV を逐次読み、EOF で開き直して次 epoch にする stream
// =============================================================================

/// `PsvFileLoader` を逐次読み、EOF に当たったら同 file を開き直す (= 次 epoch)。
/// `score-drop-abs` の skip と per-position bucket 計算もここで行う。
struct EpochStream<'a> {
    path: &'a Path,
    loader: PsvFileLoader,
    progress: &'a ShogiProgressKPAbs,
    score_drop_abs: Option<i32>,
    /// 直近の reopen 以降に実際に push した (= drop されなかった) position 数。
    pushed_this_epoch: u64,
    /// 1 epoch 丸ごと 0 push だった (= file を 1 周しても 1 件も使えなかった)
    /// 連続回数。空 file / 全 drop の無限ループ検出用。
    barren_passes: u32,
}

/// 連続 `barren_passes` がこれに達したら「使える position が無い」と判断して
/// 無限ループせず error を返す。
const MAX_BARREN_PASSES: u32 = 5;

impl<'a> EpochStream<'a> {
    fn new(
        path: &'a Path,
        progress: &'a ShogiProgressKPAbs,
        score_drop_abs: Option<i32>,
    ) -> io::Result<Self> {
        Ok(Self {
            path,
            loader: PsvFileLoader::new(path)?,
            progress,
            score_drop_abs,
            pushed_this_epoch: 0,
            barren_passes: 0,
        })
    }

    /// 次の使える PSV (+ その output bucket) を返す。EOF なら file を開き直す。
    fn next(&mut self) -> io::Result<(PackedSfenValue, i32)> {
        loop {
            match self.loader.next_psv()? {
                Some(psv) => {
                    // bullet `--score-drop-abs` の近似 (詳細は module doc)。
                    // i64 cast で i16::MIN の abs overflow を避ける。
                    if let Some(t) = self.score_drop_abs {
                        if i64::from(psv.score()).abs() >= i64::from(t) {
                            continue;
                        }
                    }
                    self.pushed_this_epoch += 1;
                    let bucket = i32::from(self.progress.bucket(&psv));
                    return Ok((psv, bucket));
                }
                None => {
                    if self.pushed_this_epoch == 0 {
                        self.barren_passes += 1;
                        if self.barren_passes >= MAX_BARREN_PASSES {
                            return Err(io::Error::other(format!(
                                "data file {} yielded no usable positions over {} full passes \
                                 (empty file, or all positions filtered out by score-drop-abs)",
                                self.path.display(),
                                self.barren_passes
                            )));
                        }
                    } else {
                        self.barren_passes = 0;
                    }
                    self.pushed_this_epoch = 0;
                    self.loader = PsvFileLoader::new(self.path)?;
                }
            }
        }
    }
}

// =============================================================================
// run — superbatch loop
// =============================================================================

/// superbatch training loop を実行し、`cfg.output_dir` 配下に checkpoint を書き出す。
///
/// - `backend`: GPU step を実行する backend (`bins/nnue_train::GpuTrainer`)
/// - `data_path`: PSV file (`PackedSfenValue` × N、40 bytes 固定)
/// - `progress`: progress8kpabs 重み (`--progress-coeff` 未指定なら zero-weight default → 全 bucket 4)
/// - `lr_scheduler` / `wdl_scheduler`: superbatch / batch index から lr / wdl lambda を返す
/// - `cfg`: hyper-parameter (superbatch 範囲、batch 構成、save 間隔、loss scale、score-drop-abs)
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

    let mut stream = EpochStream::new(data_path, progress, cfg.score_drop_abs)?;
    let mut batch = Batch::with_capacity(cfg.batch_size, MAX_ACTIVE_FEATURES);
    let mut buckets: Vec<i32> = Vec::with_capacity(cfg.batch_size);

    println!(
        "[train] data={} | net_id={} | superbatches {}..={} | {} batches/sb x bs {} \
         | lr-sched: {lr_scheduler} | wdl-sched: {wdl_scheduler} | loss-scale {:.6} | score-drop-abs {:?}",
        data_path.display(),
        cfg.net_id,
        cfg.start_superbatch,
        cfg.end_superbatch,
        cfg.batches_per_superbatch,
        cfg.batch_size,
        cfg.loss_scale,
        cfg.score_drop_abs,
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

            batch.reset();
            buckets.clear();
            while batch.n_positions < cfg.batch_size {
                let (psv, bucket) = stream.next()?;
                let pushed = batch.push(&psv);
                debug_assert!(pushed, "Batch::push refused below batch_size");
                buckets.push(bucket);
            }

            sb_loss += backend.train_step(&batch, &buckets, lr, wdl, cfg.loss_scale)?;
            sb_positions += batch.n_positions as u64;
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
        }
    }

    println!(
        "[train] done in {} ({} superbatches)",
        format_hms(run_start.elapsed().as_secs_f64()),
        cfg.end_superbatch + 1 - cfg.start_superbatch,
    );
    Ok(())
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
        last_buckets: Vec<i32>,
        max_batch_positions: usize,
        seen_lr: Vec<f32>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                steps: 0,
                saves: Vec::new(),
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
            loss_scale: f32,
        ) -> io::Result<f64> {
            assert_eq!(
                bucket_idx.len(),
                batch.n_positions,
                "one bucket per position"
            );
            assert!(batch.n_positions <= batch.batch_size);
            assert!(lr > 0.0, "lr should be positive");
            assert!(loss_scale > 0.0, "loss_scale should be positive");
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
            loss_scale: 1.0 / 290.0,
            score_drop_abs: None,
        }
    }

    #[test]
    fn run_drives_superbatches_and_writes_checkpoints() {
        let progress = ShogiProgressKPAbs; // zero weights → p = sigmoid(0) = 0.5 → bucket 4
        let lr = StepLR {
            start: 1.0e-3,
            gamma: 0.9,
            step: 1,
        };
        let wdl = ConstantWDL { value: 0.0 };
        let cfg = base_cfg();
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
        // 各 superbatch で lr が gamma 倍 (StepLR step=1, gamma=0.9)。batch 内は一定。
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
    fn empty_data_file_errors_instead_of_looping_forever() {
        // 空 file は `next_psv` が即 EOF → epoch wrap が無限ループする危険があるが、
        // `MAX_BARREN_PASSES` ガードで error にして抜ける。
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
                loss_scale: 0.0,
                ..base_cfg()
            }
            .validate()
            .is_err()
        );
        assert!(
            TrainingConfig {
                loss_scale: f32::NAN,
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
