//! 構造化実験ログ — 学習 run ごとに 1 件の experiment.json を書き出す。
//!
//! 学習 run の loss 軌跡・パラメータ・throughput を、人間可読の stdout と別に
//! 機械可読な JSON として残す。出力 1 ファイルはそのまま実験管理 Web アプリ
//! `nnue-lab` の取り込み口 (`ExperimentJsonV1` zod schema) に投入できる。
//!
//! format の設計根拠は `docs/decisions/2026-05-17-experiment-json.md` を参照。
//! 要点:
//!
//! - `nnue-lab` ExperimentJsonV1 の互換 superset。必須フィールド (`id` / `name`
//!   / `date` / `params.{lr,batch_size,superbatches}` / `history[]`) は常に
//!   schema どおりの型で出力する。本リポ固有の値は `params` / `data` /
//!   `results` (いずれも `nnue-lab` 側で passthrough) に置く。例外は schema v2 の
//!   held-out validation メトリクスで、`history[]` 要素 (passthrough 非対象) にも
//!   `test_loss` / `test_accuracy` を足す。これは `nnue-lab` の history schema
//!   拡張を要する coupled change で、未拡張の `nnue-lab` では両キーが取り込み時に
//!   削除される (ADR `docs/decisions/2026-05-17-experiment-json.md` 参照)。
//! - 1 run = 1 ファイル。crash 耐性は incremental write で得る:
//!   superbatch ごとに temp file + rename で全体を atomic に書き直すため、
//!   中断時も最後の書き込みが `status: "running"` の妥当な JSON として残る。
//!
//! [`ExperimentLogger`] が書き込み先 path と incremental な集約状態を保持し、
//! serialise 対象の本体 ([`ExperimentDoc`]) を更新しながら書き出す。

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// `nnue-lab` ExperimentJsonV1 と整合する schema 契約 version。producer (本
/// トレーナー) 自身の version は [`Generator::version`] が別に持つ。
///
/// version 2 は held-out validation メトリクスを optional フィールドとして含む:
/// `HistoryEntry::test_loss` / `test_accuracy`、`Results::best_test_loss`
/// (+ superbatch)、`Params::test_data` / `test_positions`。いずれも `--test-data`
/// 未指定の run では出力されない。`history[]` 要素への追加分は `nnue-lab` 側の
/// history schema 拡張が要る (ADR `2026-05-17-experiment-json.md` 参照)。
pub const SCHEMA_VERSION: u32 = 2;

/// LayerStack の量子化 `fv_scale` (`nnue_format::layerstack_weights::FV_SCALE` と
/// 同値)。`results.fv_scale` に記録する。`nnue-train` crate は `nnue-format` に
/// 依存しないため定数を持ち直す。
const FV_SCALE: i32 = 28;

// =============================================================================
// serialise 対象の本体
// =============================================================================

/// experiment.json を書き出した producer の識別。
#[derive(Debug, Clone, Serialize)]
pub struct Generator {
    pub name: String,
    pub version: String,
}

impl Generator {
    /// 本トレーナーを表す `Generator` (`name = "tatara"`、version は crate
    /// version)。
    pub fn tatara() -> Self {
        Self {
            name: "tatara".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// resume した run が持つ、親 run / resume 元 checkpoint への参照。
#[derive(Debug, Clone, Serialize)]
pub struct Lineage {
    /// 親 run の experiment.json `id`。解決できない場合は省略する。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// `--resume` で指定された raw checkpoint のファイル名。
    pub resumed_from_checkpoint: String,
    /// resume 元 checkpoint が表す完了 superbatch。
    pub resumed_from_superbatch: usize,
}

/// 学習パラメータ。`nnue-lab` が index する key (`architecture` / `lr` 等) と、
/// 本リポ固有の knob (`loss_kind` / `ft_fp16` 等) を flat に並べる。flat 配置は
/// `nnue-lab` のパラメータ差分表が run 間で比較できるようにするため。
#[derive(Debug, Clone, Serialize)]
pub struct Params {
    pub architecture: String,
    /// 入力 feature set の canonical 名 (`halfka-hm-merged` 等)。
    pub feature_set: String,
    /// 入力 feature 総次元 `ft_in` (feature set ごとに異なる)。
    pub ft_in: usize,
    /// FT 出力次元 (per-perspective)。入力次元ではない点に注意。
    pub l0: usize,
    pub l1: usize,
    pub l2: usize,
    /// output bucket 数。bucket を持たないアーキ (Simple 等) では未指定。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_buckets: Option<usize>,
    pub optimizer: String,
    /// bucket mode の canonical 名 (`progress8kpabs` 等)。bucket 無しアーキでは未指定。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket_mode: Option<String>,
    /// per-perspective 活性化の canonical 名 (`crelu` / `screlu`)。LayerStack のように
    /// 活性化を struct に持たないアーキでは未指定。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<String>,
    /// progress 係数ファイルの basename。未指定なら省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_coeff: Option<String>,
    pub lr: f32,
    pub lr_gamma: f32,
    pub lr_step: usize,
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub superbatches: usize,
    pub start_superbatch: usize,
    pub wdl: f32,
    pub scale: f32,
    pub weight_decay: f32,
    pub qa: i32,
    pub qb: i32,
    /// `"sigmoid"` または `"wrm"`。
    pub loss_kind: String,
    /// WRM loss の 5 パラメータ。`loss_kind == "wrm"` のときのみ `Some`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrm_in_scaling: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrm_in_offset: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrm_nnue2score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrm_target_offset: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrm_target_scaling: Option<f32>,
    /// `|score| >= score_drop_abs` の局面を loss から除外する閾値。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_drop_abs: Option<i32>,
    /// `--init-from` の入力ファイル basename (pretrained start)。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init_from: Option<String>,
    /// 重み初期化方式の要約 (`--init-preset` + seed + override)。preset が legacy で
    /// override も無い既定の run では省略する。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init_preset: Option<String>,
    /// held-out validation 用 PSV ファイルの basename (`--test-data`)。未指定なら省略。
    /// `--test-tail-positions` 経路では同 file (= training PSV) の末尾を holdout
    /// にするためここではなく `test_tail_positions` 側に N が入る。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_data: Option<String>,
    /// held-out validation の検証局面数 (`--test-positions` の要求値)。実際の
    /// 検証集合は `batch_size` 単位に切り上げた満タン batch 数になる。
    /// `--test-data` または `--test-tail-positions` 指定時のみ `Some`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_positions: Option<usize>,
    /// `--test-tail-positions` の値 (training PSV 末尾を holdout に分離する局面数)。
    /// 同 file 内 holdout 経路でのみ `Some`、外部 file holdout (`--test-data`) と
    /// holdout 無しでは省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_tail_positions: Option<u64>,
    pub tf32: bool,
    pub ft_fp16: bool,
    pub ft_fp16_out: bool,
    pub fp16_opt_state: bool,
    pub threads: usize,
}

/// 教師データの情報。`positions` は **training に使える局面数**
/// (`--test-tail-positions N` 指定時は file 中の生 record 数から N を引いた値)。
/// `total_positions` は実際に学習に流した総局面数 (training 範囲を pass 回数倍)。
#[derive(Debug, Clone, Serialize)]
pub struct DataInfo {
    /// 教師データファイルの basename。
    pub name: String,
    /// 教師データのうち training に使える局面数。`--test-tail-positions N`
    /// で末尾 N を holdout に切り出す場合、ここは raw 局面数 − N (= training
    /// range の長さ)。`dataset_passes` の分母として使われる。
    pub positions: u64,
    /// 学習で消費した局面数 (superbatch ごとに加算)。
    pub total_positions: u64,
    /// `total_positions / positions` (= データセット通過回数)。
    pub dataset_passes: f64,
}

/// run 全体の集約結果。
#[derive(Debug, Clone, Serialize)]
pub struct Results {
    pub training_time_seconds: u64,
    pub fv_scale: i32,
    /// 最小 loss と、それを記録した superbatch。1 点も記録されていなければ省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_loss_superbatch: Option<usize>,
    /// 最小の held-out `test_loss` と、それを記録した superbatch。held-out
    /// validation が一度も走らなかった (`--test-data` 未指定) 場合は省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_test_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_test_loss_superbatch: Option<usize>,
    /// run 全体の平均 throughput (consumed positions / wall time)。
    pub mean_pos_per_sec: u64,
    /// run が error で中断されたか (`mark_interrupted` 済か)。
    pub interrupted: bool,
}

/// superbatch 1 点分の training loss と、有効時の held-out validation メトリクス。
#[derive(Debug, Clone, Serialize)]
pub struct HistoryEntry {
    pub superbatch: usize,
    pub loss: f64,
    /// held-out 検証データ上の平均 loss。`--test-data` 指定時のみ `Some`。
    /// 非有限値 (発散) は JSON 数値に表せないため `None` として記録する。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_loss: Option<f64>,
    /// held-out 検証データ上の sign-agreement accuracy (`[0, 1]`、引き分け除外)。
    /// `--test-data` 指定時のみ `Some`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_accuracy: Option<f64>,
}

/// experiment.json として serialise される本体。フィールドは `nnue-lab`
/// ExperimentJsonV1 の key 名と一致させる。
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentDoc {
    pub schema_version: u32,
    pub generator: Generator,
    pub id: String,
    pub name: String,
    /// run 開始時刻 (ISO 8601 UTC)。
    pub date: String,
    /// `"running"` または `"completed"`。
    pub status: String,
    /// 直近の書き込み時刻 (ISO 8601 UTC)。
    pub last_updated_at: String,
    /// tatara の source revision。取得できなければ省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    pub command: String,
    /// resume した run のみ持つ。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage: Option<Lineage>,
    pub params: Params,
    pub data: DataInfo,
    pub results: Results,
    pub history: Vec<HistoryEntry>,
    /// run が書き出した checkpoint ファイル名の生成記録 (informational)。
    pub checkpoints: Vec<String>,
}

/// status 文字列。`nnue-lab` の enum が `running` / `completed` のみ受け付ける
/// ため、この 2 値だけ使う (error 中断は `status` は `running` のまま
/// [`Results::interrupted`] で表す)。
const STATUS_RUNNING: &str = "running";
const STATUS_COMPLETED: &str = "completed";

impl ExperimentDoc {
    /// run 開始時点の `ExperimentDoc` を組み立てる (`status = "running"`、
    /// `history` 空、`results` は 0 埋め)。`date` / `last_updated_at` は
    /// `start_epoch_secs` から導出する。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        name: String,
        start_epoch_secs: u64,
        commit: Option<String>,
        command: String,
        lineage: Option<Lineage>,
        params: Params,
        data: DataInfo,
    ) -> Self {
        let date = format_utc_iso(start_epoch_secs);
        Self {
            schema_version: SCHEMA_VERSION,
            generator: Generator::tatara(),
            id,
            name,
            last_updated_at: date.clone(),
            date,
            status: STATUS_RUNNING.to_string(),
            commit,
            command,
            lineage,
            params,
            data,
            results: Results {
                training_time_seconds: 0,
                fv_scale: FV_SCALE,
                best_loss: None,
                best_loss_superbatch: None,
                best_test_loss: None,
                best_test_loss_superbatch: None,
                mean_pos_per_sec: 0,
                interrupted: false,
            },
            history: Vec::new(),
            checkpoints: Vec::new(),
        }
    }
}

// =============================================================================
// ExperimentLogger — incremental write を駆動する
// =============================================================================

/// experiment.json の書き込み先 path と incremental な集約状態を保持し、
/// superbatch ごとに [`ExperimentDoc`] を更新して atomic に書き出す。
pub struct ExperimentLogger {
    doc: ExperimentDoc,
    path: PathBuf,
    /// これまでに消費した局面数 (`record_superbatch` で加算)。
    positions_trained: u64,
}

impl ExperimentLogger {
    /// `path` に書き出すロガーを作る。`path` 末尾は run 一意なファイル名。
    pub fn new(path: PathBuf, doc: ExperimentDoc) -> Self {
        Self {
            doc,
            path,
            positions_trained: 0,
        }
    }

    /// run の `id` (= experiment.json `id`)。
    pub fn id(&self) -> &str {
        &self.doc.id
    }

    /// 書き込み先 path。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 1 superbatch 完了を記録する。`history` に 1 点追加し、`results` 集約値
    /// (`best_loss` / `best_test_loss` / `mean_pos_per_sec` /
    /// `training_time_seconds`) と `data.total_positions` / `data.dataset_passes`
    /// を更新する。
    ///
    /// `elapsed_seconds` は run 開始からの経過秒 (wall time)。`test_loss` /
    /// `test_accuracy` は held-out validation を走らせた場合のメトリクス
    /// (`--test-data` 未指定なら `None`)。非有限な `test_loss` / `test_accuracy`
    /// は JSON 数値に表せないため `None` に落として記録する (発散シグナルは
    /// 呼び出し側が別途警告する)。
    pub fn record_superbatch(
        &mut self,
        superbatch: usize,
        mean_loss: f64,
        sb_positions: u64,
        elapsed_seconds: f64,
        test_loss: Option<f64>,
        test_accuracy: Option<f64>,
    ) {
        self.positions_trained = self.positions_trained.saturating_add(sb_positions);

        // JSON 数値に NaN / Inf は表現できない。非有限 loss (sb_positions == 0 の
        // 防御分岐等) は 0.0 として記録し warning を出す。best_loss には混ぜない。
        let loss = if mean_loss.is_finite() {
            mean_loss
        } else {
            eprintln!(
                "[train] warning: superbatch {superbatch} loss is non-finite ({mean_loss}); \
                 recording 0.0 in experiment.json"
            );
            0.0
        };
        // held-out メトリクスは非有限値を JSON に出せないため None に落とす
        // (0.0 だと best_test_loss を誤って更新するので 0.0 化はしない)。
        let test_loss = test_loss.filter(|v| v.is_finite());
        let test_accuracy = test_accuracy.filter(|v| v.is_finite());
        self.doc.history.push(HistoryEntry {
            superbatch,
            loss,
            test_loss,
            test_accuracy,
        });

        if mean_loss.is_finite()
            && self
                .doc
                .results
                .best_loss
                .is_none_or(|best| mean_loss < best)
        {
            self.doc.results.best_loss = Some(mean_loss);
            self.doc.results.best_loss_superbatch = Some(superbatch);
        }

        if let Some(tl) = test_loss
            && self.doc.results.best_test_loss.is_none_or(|best| tl < best)
        {
            self.doc.results.best_test_loss = Some(tl);
            self.doc.results.best_test_loss_superbatch = Some(superbatch);
        }

        let elapsed = elapsed_seconds.max(0.0);
        self.doc.results.training_time_seconds = elapsed.round() as u64;
        self.doc.results.mean_pos_per_sec = if elapsed > 0.0 {
            (self.positions_trained as f64 / elapsed).round() as u64
        } else {
            0
        };
        self.doc.data.total_positions = self.positions_trained;
        self.doc.data.dataset_passes = if self.doc.data.positions > 0 {
            self.positions_trained as f64 / self.doc.data.positions as f64
        } else {
            0.0
        };
        self.touch();
    }

    /// checkpoint ファイル名を `checkpoints` の生成記録に追加する。
    pub fn note_checkpoint(&mut self, file_name: impl Into<String>) {
        self.doc.checkpoints.push(file_name.into());
    }

    /// run の正常終了を記録する (`status = "completed"`)。
    pub fn mark_finished(&mut self, elapsed_seconds: f64) {
        self.doc.status = STATUS_COMPLETED.to_string();
        self.doc.results.training_time_seconds = elapsed_seconds.max(0.0).round() as u64;
        self.touch();
    }

    /// run が error で中断されたことを記録する。`status` は `running` のまま
    /// 残し (`nnue-lab` enum 制約)、[`Results::interrupted`] を立てる。
    pub fn mark_interrupted(&mut self) {
        self.doc.results.interrupted = true;
        self.touch();
    }

    /// 現在の [`ExperimentDoc`] を experiment.json に atomic に書き出す。
    ///
    /// `<path>.tmp` に書いてから同一ディレクトリ内で `rename` する。書き込み
    /// 途中で crash しても `<path>` は前回の完全な JSON のまま残る。
    pub fn write(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = {
            let mut p = self.path.as_os_str().to_os_string();
            p.push(".tmp");
            PathBuf::from(p)
        };
        let json = serde_json::to_string_pretty(&self.doc).map_err(io::Error::other)?;
        let write_tmp = || -> io::Result<()> {
            let mut w = io::BufWriter::new(std::fs::File::create(&tmp_path)?);
            w.write_all(json.as_bytes())?;
            w.write_all(b"\n")?;
            w.flush()?;
            Ok(())
        };
        if let Err(e) = write_tmp() {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp_path, &self.path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
        Ok(())
    }

    /// `last_updated_at` を現在時刻に更新する。
    fn touch(&mut self) {
        self.doc.last_updated_at = format_utc_iso(now_epoch_secs());
    }
}

// =============================================================================
// UTC 時刻フォーマット (chrono 非依存、host-only)
// =============================================================================

/// 現在の UNIX 時刻 (秒)。システム時刻が UNIX epoch より前なら 0。
pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// UNIX 時刻 (秒) を `YYYY-MM-DDTHH:MM:SSZ` (ISO 8601 UTC) に整形する。
/// `nnue-lab` zod の `.datetime()` が要求する形式。
pub fn format_utc_iso(epoch_secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil_from_epoch(epoch_secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// UNIX 時刻 (秒) を `YYYYMMDDtHHMMSSz` の compact 表記に整形する。run `id` の
/// 時刻成分に使う。
pub fn format_utc_compact(epoch_secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil_from_epoch(epoch_secs);
    format!("{y:04}{mo:02}{d:02}t{h:02}{mi:02}{s:02}z")
}

/// UNIX 時刻 (秒) を UTC の `(year, month, day, hour, minute, second)` に変換
/// する。日付部分は Howard Hinnant の `civil_from_days` アルゴリズム。
fn civil_from_epoch(epoch_secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (epoch_secs / 86_400) as i64;
    let rem = epoch_secs % 86_400;
    let hour = (rem / 3_600) as u32;
    let minute = ((rem % 3_600) / 60) as u32;
    let second = (rem % 60) as u32;

    // civil_from_days: 1970-01-01 を 0 日目とする日番号 → (year, month, day)。
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params() -> Params {
        Params {
            architecture: "LayerStack-1536-16-32-9bucket".to_string(),
            feature_set: "halfka-hm-merged".to_string(),
            ft_in: 73_305,
            l0: 1536,
            l1: 16,
            l2: 32,
            num_buckets: Some(9),
            optimizer: "ranger".to_string(),
            bucket_mode: Some("progress8kpabs".to_string()),
            activation: None,
            progress_coeff: Some("progress.bin".to_string()),
            lr: 8.75e-4,
            lr_gamma: 0.995,
            lr_step: 1,
            batch_size: 65_536,
            batches_per_superbatch: 6_104,
            superbatches: 400,
            start_superbatch: 1,
            wdl: 0.0,
            scale: 290.0,
            weight_decay: 0.0,
            qa: 127,
            qb: 64,
            loss_kind: "wrm".to_string(),
            wrm_in_scaling: Some(340.0),
            wrm_in_offset: Some(270.0),
            wrm_nnue2score: Some(600.0),
            wrm_target_offset: Some(270.0),
            wrm_target_scaling: Some(380.0),
            score_drop_abs: None,
            init_from: None,
            init_preset: None,
            test_data: None,
            test_positions: None,
            test_tail_positions: None,
            tf32: false,
            ft_fp16: true,
            ft_fp16_out: true,
            fp16_opt_state: false,
            threads: 16,
        }
    }

    fn sample_data() -> DataInfo {
        DataInfo {
            name: "teacher.psv".to_string(),
            positions: 800_000_000,
            total_positions: 0,
            dataset_passes: 0.0,
        }
    }

    fn sample_doc() -> ExperimentDoc {
        ExperimentDoc::new(
            "rshogi-20260517t041530z".to_string(),
            "rshogi".to_string(),
            1_747_000_000,
            Some("7beb263".to_string()),
            "nnue-train --data teacher.psv".to_string(),
            None,
            sample_params(),
            sample_data(),
        )
    }

    #[test]
    fn civil_from_epoch_known_values() {
        // epoch 0 = 1970-01-01T00:00:00Z
        assert_eq!(civil_from_epoch(0), (1970, 1, 1, 0, 0, 0));
        // epoch 1_700_000_000 = 2023-11-14T22:13:20Z
        assert_eq!(civil_from_epoch(1_700_000_000), (2023, 11, 14, 22, 13, 20));
    }

    #[test]
    fn format_utc_renders_iso_and_compact() {
        assert_eq!(format_utc_iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_utc_iso(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(format_utc_compact(1_700_000_000), "20231114t221320z");
    }

    #[test]
    fn new_doc_is_running_with_empty_history() {
        let doc = sample_doc();
        assert_eq!(doc.status, "running");
        assert_eq!(doc.schema_version, 2);
        assert!(doc.history.is_empty());
        // date は開始 epoch から ISO 8601 UTC で導出され、初期 last_updated_at と一致。
        assert_eq!(doc.date, format_utc_iso(1_747_000_000));
        assert_eq!(doc.date, doc.last_updated_at);
    }

    #[test]
    fn record_superbatch_updates_history_and_aggregates() {
        let mut logger = ExperimentLogger::new(PathBuf::from("/tmp/unused.json"), sample_doc());
        logger.record_superbatch(1, 0.04, 1_000_000, 10.0, None, None);
        logger.record_superbatch(2, 0.03, 1_000_000, 20.0, None, None);
        logger.record_superbatch(3, 0.035, 1_000_000, 30.0, None, None);

        assert_eq!(logger.doc.history.len(), 3);
        assert_eq!(logger.doc.results.best_loss, Some(0.03));
        assert_eq!(logger.doc.results.best_loss_superbatch, Some(2));
        assert_eq!(logger.doc.data.total_positions, 3_000_000);
        // 3_000_000 positions / 30 s = 100_000 pos/s
        assert_eq!(logger.doc.results.mean_pos_per_sec, 100_000);
        // 3_000_000 consumed / 800_000_000 dataset
        assert!((logger.doc.data.dataset_passes - 0.00375).abs() < 1e-9);
    }

    #[test]
    fn non_finite_loss_is_recorded_as_zero() {
        let mut logger = ExperimentLogger::new(PathBuf::from("/tmp/unused.json"), sample_doc());
        logger.record_superbatch(1, f64::NAN, 1_000, 1.0, None, None);
        assert_eq!(logger.doc.history[0].loss, 0.0);
        // best_loss は非有限 loss を採らない。
        assert_eq!(logger.doc.results.best_loss, None);
    }

    #[test]
    fn record_superbatch_tracks_validation_metrics() {
        let mut logger = ExperimentLogger::new(PathBuf::from("/tmp/unused.json"), sample_doc());
        logger.record_superbatch(1, 0.04, 1_000, 1.0, Some(0.05), Some(0.81));
        logger.record_superbatch(2, 0.03, 1_000, 2.0, Some(0.042), Some(0.84));
        // 非有限な test_loss は None に落ち、history にも best_test_loss にも入らない。
        logger.record_superbatch(3, 0.02, 1_000, 3.0, Some(f64::INFINITY), Some(0.9));

        assert_eq!(logger.doc.history[0].test_loss, Some(0.05));
        assert_eq!(logger.doc.history[0].test_accuracy, Some(0.81));
        assert_eq!(logger.doc.history[2].test_loss, None);
        // best_test_loss は最小の有限 test_loss (sb2) を採る。
        assert_eq!(logger.doc.results.best_test_loss, Some(0.042));
        assert_eq!(logger.doc.results.best_test_loss_superbatch, Some(2));
    }

    #[test]
    fn validation_metrics_omitted_when_never_recorded() {
        // `--test-data` 無し相当: test_loss / test_accuracy を渡さない run。
        let mut logger = ExperimentLogger::new(PathBuf::from("/tmp/unused.json"), sample_doc());
        logger.record_superbatch(1, 0.04, 1_000, 1.0, None, None);
        let v = serde_json::to_value(&logger.doc).expect("serialise");
        assert!(v["history"][0].get("test_loss").is_none());
        assert!(v["history"][0].get("test_accuracy").is_none());
        assert!(v["results"].get("best_test_loss").is_none());
    }

    #[test]
    fn mark_interrupted_keeps_running_status() {
        let mut logger = ExperimentLogger::new(PathBuf::from("/tmp/unused.json"), sample_doc());
        logger.record_superbatch(1, 0.04, 1_000, 1.0, None, None);
        logger.mark_interrupted();
        // error 中断は status を "running" のまま、interrupted フラグで表す。
        assert_eq!(logger.doc.status, "running");
        assert!(logger.doc.results.interrupted);
    }

    #[test]
    fn write_produces_valid_json_and_is_idempotent_path() {
        let dir = std::env::temp_dir().join(format!("nnue-exp-test-{}", std::process::id()));
        let path = dir.join("experiments").join("rshogi-test.json");
        let mut logger = ExperimentLogger::new(path.clone(), sample_doc());
        logger.record_superbatch(1, 0.04, 1_000_000, 10.0, Some(0.05), Some(0.8));
        logger.note_checkpoint("rshogi-20.bin");
        logger.note_checkpoint("rshogi-20.ckpt");
        logger.mark_finished(12.0);
        logger.write().expect("write ok");

        let raw = std::fs::read_to_string(&path).expect("file exists");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(v["schema_version"], 2);
        assert_eq!(v["status"], "completed");
        // held-out メトリクスを渡した superbatch は history に出る。
        assert_eq!(v["history"][0]["test_loss"], 0.05);
        assert_eq!(v["history"][0]["test_accuracy"], 0.8);
        assert_eq!(v["generator"]["name"], "tatara");
        assert_eq!(v["params"]["lr"], 0.000875);
        assert_eq!(v["history"][0]["superbatch"], 1);
        assert_eq!(v["checkpoints"][0], "rshogi-20.bin");
        assert_eq!(v["results"]["interrupted"], false);
        // 上書きでも有効な JSON のまま (incremental write を模す)。
        logger.write().expect("rewrite ok");
        let raw2 = std::fs::read_to_string(&path).expect("file still exists");
        serde_json::from_str::<serde_json::Value>(&raw2).expect("still valid json");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn optional_fields_are_omitted_when_none() {
        // commit / lineage / wrm_* なし、loss_kind sigmoid の doc。
        let mut params = sample_params();
        params.loss_kind = "sigmoid".to_string();
        params.wrm_in_scaling = None;
        params.wrm_in_offset = None;
        params.wrm_nnue2score = None;
        params.wrm_target_offset = None;
        params.wrm_target_scaling = None;
        params.progress_coeff = None;
        let doc = ExperimentDoc::new(
            "net-20260517t041530z".to_string(),
            "net".to_string(),
            1_747_000_000,
            None,
            "nnue-train".to_string(),
            None,
            params,
            sample_data(),
        );
        let v = serde_json::to_value(&doc).expect("serialise");
        assert!(v.get("commit").is_none());
        assert!(v.get("lineage").is_none());
        assert!(v["params"].get("wrm_in_scaling").is_none());
        assert!(v["params"].get("progress_coeff").is_none());
    }
}
