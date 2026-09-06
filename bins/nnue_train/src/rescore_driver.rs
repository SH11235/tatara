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
//! 全条件 — net / progress 係数の **ロードに使った byte 列そのものの sha256**
//! ([`LoadedArtifact`])、routing と arch 構成の全 provenance、score 変換、
//! ビルド識別 (crate version + git commit) — を書き、条件が 1 つでも変われば
//! 既存 sidecar は再生成される。`.done` 昇格前には入力 / net / 係数の現物が
//! ロード時から差し替わっていないことを stat で再検証する。完了時は `.done`
//! marker (fingerprint text) に加えて機械可読な `.meta.json` を書く。
//!
//! ## identity 検証の範囲 (残余リスク)
//!
//! - **内容 hash で守られるのは、一括読みした byte 列から identity を作る
//!   `--init-from` の .bin と progress 係数のみ。**
//! - `--resume` の .ckpt と入力 PSV の変更検出は stat (size + mtime) 等値で、
//!   同サイズかつ mtime を書き戻した置換は検出できない。
//! - build 識別 (`git_commit`) が `unknown` (repo 外 build) または `-dirty`
//!   (未 commit 変更込み build) の場合、同じ識別で別内容の binary があり得る
//!   ため、fingerprint に起動ごとの nonce を混ぜて **完了 skip と resume を
//!   無効化する** (常に最初から再生成)。campaign 実行は clean checkout で
//!   build した binary が前提。

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

/// build 時に埋め込んだ git commit (short、dirty なら `-dirty` 付き、repo 外
/// build は `unknown`)。埋め込みは `build.rs` (`TATARA_BUILD_COMMIT`)。
const BUILD_COMMIT: &str = env!("TATARA_BUILD_COMMIT");

/// build 識別が binary の内容を一意に指すか。`unknown` / `-dirty` は同じ識別で
/// 別内容の binary があり得るため false (fingerprint は nonce 入りになり、完了
/// skip と resume が無効化される)。
fn build_is_reproducible() -> bool {
    BUILD_COMMIT != "unknown" && !BUILD_COMMIT.ends_with("-dirty")
}

/// ロード済み artifact (net / progress 係数) の識別情報。
///
/// sha256 は **ロードに使った byte 列そのもの** (または stream hash 直後にロード
/// して stat 等値を確認したもの) から計算する。ロード後に path を開き直して
/// hash すると、その間の差し替えで「実際に評価した重みと違う識別情報」が完成
/// sidecar に付く。
pub(crate) struct LoadedArtifact {
    pub(crate) canonical: PathBuf,
    pub(crate) size: u64,
    pub(crate) mtime_ns: u128,
    pub(crate) sha256: String,
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

impl LoadedArtifact {
    /// ロードに使った byte 列そのものから identity を作る (.bin / progress 係数用)。
    /// stat の size が byte 列と食い違う場合は既に差し替えられているので error。
    ///
    /// `canonical` は **呼び出し側が起動時に一度だけ canonicalize 済みの path** を
    /// 渡す契約 (ここで解決し直すと、2 回の解決の間の symlink 差し替えで
    /// 「ロード・hash・検証が同一実体を見る」不変条件が崩れる)。
    pub(crate) fn from_loaded_bytes(canonical: &Path, bytes: &[u8]) -> std::io::Result<Self> {
        let canonical = canonical.to_path_buf();
        let (size, mtime_ns) = file_size_mtime_ns(&canonical)?;
        if size != bytes.len() as u64 {
            return Err(std::io::Error::other(format!(
                "{} changed while loading (read {} bytes, file is now {size} bytes)",
                canonical.display(),
                bytes.len()
            )));
        }
        Ok(Self {
            canonical,
            size,
            mtime_ns,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    /// file を streaming で hash して identity を作る (.ckpt のように一括読みが
    /// 重い artifact 用)。hash とロードの間の差し替えは、ロード直後に
    /// [`Self::verify_unchanged`] を呼んで stat 等値で検出する契約。
    /// `canonical` の契約は [`Self::from_loaded_bytes`] と同じ (解決済み path を
    /// そのまま使い、ここでは解決し直さない)。
    pub(crate) fn hash_file(canonical: &Path) -> std::io::Result<Self> {
        let canonical = canonical.to_path_buf();
        let (size, mtime_ns) = file_size_mtime_ns(&canonical)?;
        let mut file = std::fs::File::open(&canonical)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0_u8; 1 << 20];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(Self {
            canonical,
            size,
            mtime_ns,
            sha256: format!("{:x}", hasher.finalize()),
        })
    }

    /// 現物が identity 記録時から差し替わっていないかを **stat (size + mtime)
    /// 等値のみ** で検証する。同サイズかつ mtime を書き戻した置換は検出できない
    /// (module doc の残余リスク。内容 hash で守られるのは byte 列から identity を
    /// 作った経路のみ)。
    pub(crate) fn verify_unchanged(&self, what: &str) -> std::io::Result<()> {
        let (size, mtime_ns) = file_size_mtime_ns(&self.canonical)?;
        if (size, mtime_ns) != (self.size, self.mtime_ns) {
            return Err(std::io::Error::other(format!(
                "{what} {} changed since it was loaded (size {} -> {size}); the loaded \
                 data no longer matches the file on disk — restart the run",
                self.canonical.display(),
                self.size
            )));
        }
        Ok(())
    }
}

/// [`run_rescore`] に渡す設定 (CLI から解決済みの値 + ロード時に固定した identity)。
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
    /// ロードした weights の identity (`--init-from` の .bin は読んだ byte 列から、
    /// `--resume` の .ckpt は stream hash + ロード後 stat 検証で固定済み)。
    pub(crate) net: LoadedArtifact,
    /// weights の種別: `"init-from-bin"` (量子化 → 逆量子化 fp32) /
    /// `"resume-ckpt"` (fp32 master)。
    pub(crate) weights_source: &'a str,
    /// arch 識別子 ([`crate::training::layerstack_architecture`] の値)。
    pub(crate) arch: String,
    /// threat profile の CLI 名 (`off` 含む)。
    pub(crate) threat_profile: String,
    /// effect bucket 構成の CLI 名 (`off` 含む)。
    pub(crate) effect_bucket: String,
    /// FT factorizer mode (`off` / `base` / `pool-effect-buckets` /
    /// `per-effect-bucket`)。
    pub(crate) ft_factorize: &'static str,
    /// PSQT shortcut の有無。
    pub(crate) psqt: bool,
    /// stack-shared-delta の有無。
    pub(crate) stack_shared_delta: bool,
    /// progresskpabs の係数 identity (kingrank9 では `None`)。ロードに使った
    /// byte 列から固定済み。
    pub(crate) progress_coeff: Option<LoadedArtifact>,
}

/// fingerprint の key=value 対。text marker と `.meta.json` の両方をこの単一の
/// リストから作る (二重管理で項目がずれると「marker は一致するのに meta は別条件」
/// という取り違えの温床になる)。入力の identity は `.done` 昇格前の再検証用に
/// 別 field で保持する。
struct Fingerprint {
    pairs: Vec<(&'static str, String)>,
    input_canonical: PathBuf,
    input_size: u64,
    input_mtime_ns: u128,
}

impl Fingerprint {
    fn build(
        cfg: &RescoreConfig<'_>,
        input_records: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // `cfg.input` は呼び出し側 (training.rs) が起動時に一度だけ canonicalize
        // 済み。loader の worker open と同じ path をそのまま identity に使う
        // (ここで解決し直すと worker が読む実体と乖離し得る)。
        let input_canonical = cfg.input.to_path_buf();
        let (input_size, input_mtime_ns) = file_size_mtime_ns(&input_canonical)?;

        // ビルド識別。crate version は forward 実装が変わっても動かないことが
        // あるため、build 時に埋め込んだ git commit (dirty 印付き、`build.rs`) を
        // 併記して「同 version の別実装」で旧 sidecar を無言 skip しないように
        // する。
        let git_commit = BUILD_COMMIT.to_string();

        let mut pairs: Vec<(&'static str, String)> = vec![
            ("version", "1".to_string()),
            ("mode", "gpu-nnue-fp32".to_string()),
            ("tool_version", env!("CARGO_PKG_VERSION").to_string()),
            ("git_commit", git_commit),
            ("input_path", input_canonical.display().to_string()),
            ("input_size", input_size.to_string()),
            ("input_mtime_ns", input_mtime_ns.to_string()),
            ("input_records", input_records.to_string()),
            ("net_path", cfg.net.canonical.display().to_string()),
            ("net_size", cfg.net.size.to_string()),
            ("net_sha256", cfg.net.sha256.clone()),
            ("weights_source", cfg.weights_source.to_string()),
            ("arch", cfg.arch.clone()),
            ("feature_set", cfg.feature_set.canonical_name().to_string()),
            ("threat_profile", cfg.threat_profile.clone()),
            ("effect_bucket", cfg.effect_bucket.clone()),
            ("ft_factorize", cfg.ft_factorize.to_string()),
            ("psqt", cfg.psqt.to_string()),
            ("stack_shared_delta", cfg.stack_shared_delta.to_string()),
            ("bucket_mode", cfg.bucket_mode.canonical_name().to_string()),
            ("num_buckets", cfg.num_buckets.to_string()),
        ];
        if let Some(coeff) = &cfg.progress_coeff {
            pairs.push(("progress_coeff_path", coeff.canonical.display().to_string()));
            pairs.push(("progress_coeff_sha256", coeff.sha256.clone()));
        }
        pairs.push((
            "score_scale_bits",
            format!("0x{:08x}", cfg.score_scale.to_bits()),
        ));
        pairs.push(("score_scale", format!("{}", cfg.score_scale)));
        pairs.push(("score_clip", cfg.score_clip.to_string()));
        pairs.push(("batch_size", cfg.batch_size.to_string()));
        if !build_is_reproducible() {
            // unknown / dirty build は「同じ識別で別内容の binary」があり得るため
            // 既存 marker と決して一致しない nonce を混ぜる (module doc 参照)。
            // 完了 skip も resume も効かず、毎回最初から再生成になる。
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            pairs.push(("build_nonce", nonce.to_string()));
        }
        Ok(Self {
            pairs,
            input_canonical,
            input_size,
            input_mtime_ns,
        })
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

    /// 入力 / net / progress 係数の現物が fingerprint 構築時 (= ロード時) から
    /// 差し替わっていないことを stat で検証する。`.done` 昇格前に呼び、上書き・
    /// 差し替えを完成 sidecar に紛れ込ませない。
    fn verify_sources_unchanged(
        &self,
        cfg: &RescoreConfig<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (size, mtime_ns) = file_size_mtime_ns(&self.input_canonical)?;
        if (size, mtime_ns) != (self.input_size, self.input_mtime_ns) {
            return Err(format!(
                "input {} changed while rescoring (size {} -> {size}); refusing to promote \
                 the sidecar — restart the run",
                self.input_canonical.display(),
                self.input_size
            )
            .into());
        }
        cfg.net.verify_unchanged("net")?;
        if let Some(coeff) = &cfg.progress_coeff {
            coeff.verify_unchanged("--progress-coeff")?;
        }
        Ok(())
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

    if !build_is_reproducible() {
        eprintln!(
            "[rescore] warning: this binary was built from a {BUILD_COMMIT} tree; the \
             fingerprint includes a per-run nonce, so completion skip and resume are \
             disabled and the sidecar is always regenerated from scratch. Build from a \
             clean checkout for campaign runs."
        );
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
    // 昇格前に入力 / net / 係数の差し替えを検出する (差し替え後のロード済み weight
    // でも評価自体は正しいが、marker が指す実体と一致しない sidecar を完成扱いに
    // しない)。
    fingerprint.verify_sources_unchanged(cfg)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::FeatureSet;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tatara-rescore-fp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_config<'a>(
        input: &'a Path,
        output_dir: &'a Path,
        net: LoadedArtifact,
        coeff: Option<LoadedArtifact>,
    ) -> RescoreConfig<'a> {
        RescoreConfig {
            input,
            output_dir,
            score_scale: 1200.0,
            score_clip: 10_000,
            batch_size: 16,
            feature_set: FeatureSet::HalfKp.spec(),
            bucket_mode: BucketMode::ProgressKpAbs,
            num_buckets: 9,
            net,
            weights_source: "init-from-bin",
            arch: "LayerStack-128-16-32-9bucket".to_string(),
            threat_profile: "off".to_string(),
            effect_bucket: "off".to_string(),
            ft_factorize: "off",
            psqt: false,
            stack_shared_delta: false,
            progress_coeff: coeff,
        }
    }

    #[test]
    fn loaded_artifact_hashes_loaded_bytes_and_detects_swap() {
        let dir = temp_dir("artifact");
        let path = dir.join("net.bin");
        let bytes = b"net weights v1".to_vec();
        std::fs::write(&path, &bytes).unwrap();

        let artifact = LoadedArtifact::from_loaded_bytes(&path, &bytes).unwrap();
        assert_eq!(artifact.sha256, format!("{:x}", Sha256::digest(&bytes)));
        artifact
            .verify_unchanged("net")
            .expect("unchanged file must verify");

        // stream hash 版も同じ identity を作る。
        let streamed = LoadedArtifact::hash_file(&path).unwrap();
        assert_eq!(streamed.sha256, artifact.sha256);
        assert_eq!(streamed.size, artifact.size);

        // 差し替え (サイズ変更) は stat 検証で検出。
        std::fs::write(&path, b"swapped to different content").unwrap();
        let err = artifact
            .verify_unchanged("net")
            .expect_err("swapped file must be detected");
        assert!(
            err.to_string().contains("changed since it was loaded"),
            "{err}"
        );

        // 読んだ byte 列と現物サイズの不一致 (読み中の差し替え) も検出。
        let err = LoadedArtifact::from_loaded_bytes(&path, &bytes)
            .map(|_| ())
            .expect_err("size mismatch must be detected");
        assert!(err.to_string().contains("changed while loading"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fingerprint_records_build_identity_and_arch_provenance() {
        let dir = temp_dir("provenance");
        let input = dir.join("in.psv");
        std::fs::write(&input, vec![0_u8; 40 * 2]).unwrap();
        let net_path = dir.join("net.bin");
        let net_bytes = b"net bytes".to_vec();
        std::fs::write(&net_path, &net_bytes).unwrap();
        let coeff_path = dir.join("progress.bin");
        let coeff_bytes = b"coeff bytes".to_vec();
        std::fs::write(&coeff_path, &coeff_bytes).unwrap();

        let net = LoadedArtifact::from_loaded_bytes(&net_path, &net_bytes).unwrap();
        let coeff = LoadedArtifact::from_loaded_bytes(&coeff_path, &coeff_bytes).unwrap();
        let coeff_sha = coeff.sha256.clone();
        let out_dir = dir.join("out");
        let cfg = test_config(&input, &out_dir, net, Some(coeff));
        let fingerprint = Fingerprint::build(&cfg, 2).unwrap();
        let text = fingerprint.text();

        // ビルド識別: version だけでなく git commit も入る (同 version の別実装で
        // 旧 sidecar を無言 skip しない)。
        assert!(text.contains("tool_version="), "{text}");
        assert!(text.contains("git_commit="), "{text}");
        assert!(
            !text.contains("git_commit=\n"),
            "commit value must not be empty: {text}"
        );

        // ロード内容の sha が素通しで入る。
        assert!(
            text.contains(&format!("net_sha256={:x}", Sha256::digest(&net_bytes))),
            "{text}"
        );
        assert!(
            text.contains(&format!("progress_coeff_sha256={coeff_sha}")),
            "{text}"
        );

        // arch provenance: 生成条件を復元できる全構成 flag。
        for key in [
            "arch=LayerStack-128-16-32-9bucket",
            "feature_set=",
            "threat_profile=off",
            "effect_bucket=off",
            "ft_factorize=off",
            "psqt=false",
            "stack_shared_delta=false",
            "bucket_mode=progresskpabs",
            "num_buckets=9",
            "score_scale_bits=0x",
            "batch_size=16",
        ] {
            assert!(text.contains(key), "missing {key} in: {text}");
        }

        // meta.json も同じリストから生成される。
        let meta = fingerprint.meta_json(&cfg);
        assert_eq!(meta["label_kind"], serde_json::json!("fp32_dequantised"));
        assert_eq!(meta["threat_profile"], serde_json::json!("off"));
        assert_eq!(meta["ft_factorize"], serde_json::json!("off"));
        assert_eq!(meta["psqt"], serde_json::json!("false"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 既知の制約の固定 (module doc の残余リスク): stat (size + mtime) 検証は、
    /// 同サイズ + mtime を書き戻した置換を**検出できない**。内容 hash (sha256) は
    /// 当然一致しなくなる — この乖離が .ckpt / 入力 PSV 側の残余リスクの正体で、
    /// .bin / 係数が「ロードした byte 列から hash」で守られるのと対照的。
    #[test]
    fn stat_verification_cannot_detect_mtime_preserving_swap() {
        let dir = temp_dir("mtime-swap");
        let path = dir.join("artifact.bin");
        std::fs::write(&path, b"AAAA").unwrap();
        let artifact = LoadedArtifact::hash_file(&path).unwrap();
        let original_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        // 同サイズの別内容へ置換し、mtime を書き戻す。
        std::fs::write(&path, b"BBBB").unwrap();
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(original_mtime))
            .unwrap();
        drop(file);

        artifact
            .verify_unchanged("artifact")
            .expect("stat-only verification accepts an mtime-preserving swap (known limit)");
        let swapped = LoadedArtifact::hash_file(&path).unwrap();
        assert_ne!(
            swapped.sha256, artifact.sha256,
            "the content hash does diverge — only the stat check misses the swap"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fingerprint_verify_detects_source_swaps_before_promotion() {
        let dir = temp_dir("verify");
        let input = dir.join("in.psv");
        std::fs::write(&input, vec![0_u8; 40]).unwrap();
        let net_path = dir.join("net.bin");
        let net_bytes = b"net".to_vec();
        std::fs::write(&net_path, &net_bytes).unwrap();
        let net = LoadedArtifact::from_loaded_bytes(&net_path, &net_bytes).unwrap();
        let out_dir = dir.join("out");
        let cfg = test_config(&input, &out_dir, net, None);
        let fingerprint = Fingerprint::build(&cfg, 1).unwrap();
        fingerprint
            .verify_sources_unchanged(&cfg)
            .expect("unchanged sources must verify");

        // 入力の上書き (サイズ変更) を昇格前検証が検出する。
        std::fs::write(&input, vec![0_u8; 40 * 2]).unwrap();
        let err = fingerprint
            .verify_sources_unchanged(&cfg)
            .expect_err("input swap must be detected");
        assert!(err.to_string().contains("changed while rescoring"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
