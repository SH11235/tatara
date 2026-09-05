//! GPU リスコア driver — PSV pool の全 record を読み、ロード済み LayerStack net の
//! 1-node 静的評価 (forward のみ) で little-endian i16 score sidecar を出力する。
//!
//! ```text
//! OrderedPsvLoader (順序保存・並列 decode、無フィルタ)
//!    → GpuTrainer::forward_step (loss なし forward、forward-only trainer)
//!    → cp = clamp(round(net_output × score_scale), ±score_clip)
//!    → ScoreSidecarWriter (fingerprint marker / 件数 resume / fail-closed)
//! ```
//!
//! 不変条件は「sidecar 行 `i` = 入力 record `i`」。fingerprint には出力を変える
//! 全条件 (net の sha256、routing、progress 係数の sha256、score 変換、batch
//! size) を書き、条件が 1 つでも変われば既存 sidecar は再生成される。完了時は
//! `.done` marker (fingerprint text) に加えて機械可読な `.meta.json` を書く。

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};

use nnue_train::dataloader::{BucketMode, PSV_RECORD_BYTES};
use nnue_train::rescore::{
    OrderedPsvLoader, ScoreSidecarWriter, SidecarOpen, default_decode_workers,
};
use shogi_features::FeatureSetSpec;

use crate::trainer_layerstack::GpuTrainer;

/// GPU tiled dense kernel の `b % 16 == 0` 制約に合わせる chunk padding 単位。
const PAD_MULTIPLE: usize = 16;

/// [`run_rescore`] に渡す設定 (CLI から解決済みの値)。
pub(crate) struct RescoreConfig<'a> {
    /// 入力 PSV (全 record を原順序で relabel する)。
    pub(crate) input: &'a Path,
    /// sidecar の出力 directory (`<入力名>.scores.i16` + marker + `.meta.json`)。
    pub(crate) output_dir: &'a Path,
    /// `cp = net_output * score_scale` の変換係数 (この net 世代の nnue2score)。
    pub(crate) score_scale: f32,
    /// |cp| の飽和値。
    pub(crate) score_clip: i16,
    /// forward の batch サイズ = loader の chunk サイズ。
    pub(crate) batch_size: usize,
    pub(crate) feature_set: FeatureSetSpec,
    pub(crate) bucket_mode: BucketMode,
    pub(crate) num_buckets: usize,
    /// ロードした weights ファイル (`--init-from` の .bin または `--resume` の .ckpt)。
    pub(crate) net_path: &'a Path,
    /// weights の種別: `"init-from-bin"` (量子化 → 逆量子化 fp32) /
    /// `"resume-ckpt"` (fp32 master)。
    pub(crate) weights_source: &'a str,
    /// arch 識別子 ([`crate::training::layerstack_architecture`] の値)。
    pub(crate) arch: String,
    /// progresskpabs の係数ファイル (kingrank9 では `None`)。
    pub(crate) progress_coeff: Option<&'a Path>,
}

/// file 全体の sha256 (hex) を streaming で計算する (net は数百 MB あるため
/// 一括読みしない)。
fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// file の `(len, modified_unix_ns)`。
fn file_size_mtime_ns(path: &Path) -> std::io::Result<(u64, u128)> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()?
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_err(|_| std::io::Error::other("file mtime is before the UNIX epoch"))?;
    Ok((meta.len(), mtime.as_nanos()))
}

/// fingerprint の key=value 対。text marker と `.meta.json` の両方をこの単一の
/// リストから作る (二重管理で項目がずれると「marker は一致するのに meta は別条件」
/// という取り違えの温床になる)。
struct Fingerprint {
    pairs: Vec<(&'static str, String)>,
}

impl Fingerprint {
    fn build(
        cfg: &RescoreConfig<'_>,
        input_records: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let input_canonical = cfg
            .input
            .canonicalize()
            .map_err(|e| format!("failed to canonicalize input {}: {e}", cfg.input.display()))?;
        let (input_size, input_mtime_ns) = file_size_mtime_ns(&input_canonical)?;
        let net_canonical = cfg
            .net_path
            .canonicalize()
            .map_err(|e| format!("failed to canonicalize net {}: {e}", cfg.net_path.display()))?;
        let (net_size, _) = file_size_mtime_ns(&net_canonical)?;
        let net_sha256 = sha256_file(&net_canonical)?;

        let mut pairs: Vec<(&'static str, String)> = vec![
            ("version", "1".to_string()),
            ("mode", "gpu-nnue-fp32".to_string()),
            ("tool_version", env!("CARGO_PKG_VERSION").to_string()),
            ("input_path", input_canonical.display().to_string()),
            ("input_size", input_size.to_string()),
            ("input_mtime_ns", input_mtime_ns.to_string()),
            ("input_records", input_records.to_string()),
            ("net_path", net_canonical.display().to_string()),
            ("net_size", net_size.to_string()),
            ("net_sha256", net_sha256),
            ("weights_source", cfg.weights_source.to_string()),
            ("arch", cfg.arch.clone()),
            ("feature_set", cfg.feature_set.canonical_name().to_string()),
            ("bucket_mode", cfg.bucket_mode.canonical_name().to_string()),
            ("num_buckets", cfg.num_buckets.to_string()),
        ];
        if let Some(coeff) = cfg.progress_coeff {
            let coeff_canonical = coeff.canonicalize().map_err(|e| {
                format!(
                    "failed to canonicalize progress coeff {}: {e}",
                    coeff.display()
                )
            })?;
            pairs.push(("progress_coeff_path", coeff_canonical.display().to_string()));
            pairs.push(("progress_coeff_sha256", sha256_file(&coeff_canonical)?));
        }
        pairs.push((
            "score_scale_bits",
            format!("0x{:08x}", cfg.score_scale.to_bits()),
        ));
        pairs.push(("score_scale", format!("{}", cfg.score_scale)));
        pairs.push(("score_clip", cfg.score_clip.to_string()));
        pairs.push(("batch_size", cfg.batch_size.to_string()));
        Ok(Self { pairs })
    }

    /// marker に書く text (key=value 行、[`ScoreSidecarWriter`] は byte 等値比較)。
    fn text(&self) -> String {
        let mut out = String::new();
        for (key, value) in &self.pairs {
            out.push_str(key);
            out.push('=');
            out.push_str(value);
            out.push('\n');
        }
        out
    }

    /// `.meta.json` の内容。fingerprint 全項目 + ラベル種別と変換式の注記。
    fn meta_json(&self, cfg: &RescoreConfig<'_>) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        object.insert(
            "format".to_string(),
            serde_json::Value::from("i16-le-score-sidecar"),
        );
        object.insert(
            "label_kind".to_string(),
            serde_json::Value::from(match cfg.weights_source {
                "resume-ckpt" => "fp32_master",
                _ => "fp32_dequantised",
            }),
        );
        object.insert(
            "score_formula".to_string(),
            serde_json::Value::from(
                "score[i] = clamp(round(net_output[i] * score_scale), -score_clip, score_clip)",
            ),
        );
        for (key, value) in &self.pairs {
            object.insert((*key).to_string(), serde_json::Value::from(value.clone()));
        }
        serde_json::Value::Object(object)
    }
}

/// `<sidecar>.meta.json` の path。
fn meta_json_path(sidecar: &Path) -> PathBuf {
    let mut s = sidecar.as_os_str().to_owned();
    s.push(".meta.json");
    PathBuf::from(s)
}

/// ロード済み forward-only trainer で `cfg.input` の全 record を relabel し、
/// `<出力 dir>/<入力名>.scores.i16` に書く。完了済み (`.done` + fingerprint 一致)
/// なら何もせず skip する。中断分は in-progress marker から件数ベースで resume する。
pub(crate) fn run_rescore(
    trainer: &mut GpuTrainer,
    cfg: &RescoreConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(cfg.output_dir)?;
    let file_name = cfg
        .input
        .file_name()
        .ok_or_else(|| format!("--rescore-input {} has no file name", cfg.input.display()))?;
    let mut sidecar_name = file_name.to_os_string();
    sidecar_name.push(".scores.i16");
    let sidecar = cfg.output_dir.join(sidecar_name);

    let input_size = std::fs::metadata(cfg.input)?.len();
    if !input_size.is_multiple_of(PSV_RECORD_BYTES) {
        return Err(format!(
            "--rescore-input {} size {input_size} is not a multiple of the PSV record size \
             ({PSV_RECORD_BYTES} bytes); refusing to rescore a torn file",
            cfg.input.display()
        )
        .into());
    }
    let total_records = input_size / PSV_RECORD_BYTES;
    if total_records == 0 {
        return Err(format!("--rescore-input {} is empty", cfg.input.display()).into());
    }

    let fingerprint = Fingerprint::build(cfg, total_records)?;
    let fingerprint_text = fingerprint.text();

    let (mut writer, resume_records) =
        match ScoreSidecarWriter::open(&sidecar, total_records, &fingerprint_text)? {
            SidecarOpen::Complete => {
                // 完了済み。`.meta.json` は昇格と別 file なので、欠けていれば補完する。
                write_meta_json(&sidecar, &fingerprint, cfg)?;
                println!(
                    "[rescore] {} is already complete ({total_records} records); skipping",
                    sidecar.display()
                );
                return Ok(());
            }
            SidecarOpen::Writer {
                writer,
                resume_records,
            } => (writer, resume_records),
        };
    if resume_records > 0 {
        println!("[rescore] resuming at record {resume_records}/{total_records}");
    }

    let mut loader = OrderedPsvLoader::spawn(
        cfg.input,
        cfg.batch_size,
        PAD_MULTIPLE,
        default_decode_workers(),
        cfg.bucket_mode,
        cfg.num_buckets,
        cfg.feature_set,
        resume_records,
    )?;

    let clip = f32::from(cfg.score_clip);
    let started = Instant::now();
    let mut last_log = Instant::now();
    let mut written = resume_records;
    let mut scores: Vec<i16> = Vec::with_capacity(cfg.batch_size);
    while let Some(chunk) = loader.next_chunk()? {
        let net_output = trainer.forward_step(&chunk.batch, &chunk.buckets)?;
        scores.clear();
        for &out in &net_output[..chunk.n_real] {
            let cp = out * cfg.score_scale;
            if !cp.is_finite() {
                // NaN / Inf のラベルを i16 化すると無警告の汚染になるため fail-closed。
                return Err(format!(
                    "non-finite net output {out} around record {written}; refusing to \
                     write a score for it (is the net or the score scale wrong?)"
                )
                .into());
            }
            scores.push(cp.round().clamp(-clip, clip) as i16);
        }
        writer.write_scores(&scores)?;
        written += chunk.n_real as u64;
        loader.recycle(chunk);

        if last_log.elapsed().as_secs() >= 10 {
            let done_this_run = written - resume_records;
            let pos_per_sec = done_this_run as f64 / started.elapsed().as_secs_f64();
            println!("[rescore] {written}/{total_records} records ({pos_per_sec:.0} pos/s)");
            last_log = Instant::now();
        }
    }
    writer.finish()?;
    write_meta_json(&sidecar, &fingerprint, cfg)?;

    let done_this_run = written - resume_records;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "[rescore] wrote {done_this_run} records (total {written}/{total_records}) in {elapsed:.1}s \
         ({:.0} pos/s) -> {}",
        done_this_run as f64 / elapsed.max(1e-9),
        sidecar.display()
    );
    Ok(())
}

fn write_meta_json(
    sidecar: &Path,
    fingerprint: &Fingerprint,
    cfg: &RescoreConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = meta_json_path(sidecar);
    let json = serde_json::to_string_pretty(&fingerprint.meta_json(cfg))?;
    std::fs::write(&path, json)?;
    Ok(())
}
