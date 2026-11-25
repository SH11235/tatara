use std::path::Path;

#[cfg(feature = "gpu")]
use gpu_runtime::{CudaStream, DeviceBuffer};
use nnue_format::ArchKind;
use nnue_train::dataloader::BucketMode;
use nnue_train::optimizer::OptimizerKind;
#[cfg(any(feature = "gpu", test))]
use nnue_train::raw_checkpoint::read_raw_ckpt_group_len;
#[cfg(test)]
pub(crate) use nnue_train::raw_checkpoint::write_f32_slice;
#[cfg(any(feature = "gpu", test))]
pub(crate) use nnue_train::raw_checkpoint::write_raw_ckpt_header;
use nnue_train::raw_checkpoint::{
    F32_IO_CHUNK_BYTES, PARTIAL_THREAT_CKPT_VERSION, PARTIAL_THREAT_RADAM_CKPT_VERSION,
    PARTIAL_THREAT_SGD_CKPT_VERSION, RawCkptReadMode, RawCkptReader, SGD_CHECKPOINT_VERSION,
    read_raw_ckpt_exact,
};
#[cfg(feature = "gpu")]
use nnue_train::raw_checkpoint::{MAX_RUN_ID_BYTES, write_raw_ckpt_group};
#[cfg(test)]
pub(crate) use nnue_train::raw_checkpoint::{RAW_CKPT_MAGIC, RAW_CKPT_VERSION};
pub(crate) use nnue_train::raw_checkpoint::{RawCkptArch, RawCkptMeta};
use shogi_features::FeatureSet;

#[cfg(feature = "gpu")]
use crate::arch::{FT_OPT_M_SCALE, FT_OPT_V_SCALE};
#[cfg(feature = "gpu")]
use crate::trainer_common::MomentBuf;

// ===========================================================================
// raw checkpoint format (`--resume` 用)
//
// layout (全 little-endian、現行 RAW_CKPT_VERSION = 15):
//
//   magic        b"RNRC"             (4 bytes)
//   version      u32 (11..=15)             (4 bytes)
//   fs_name_len  u32                 (4 bytes、feature set canonical 名の長さ)
//   fs_name      UTF-8 [fs_name_len]  (feature set canonical 名、例 "halfka-hm-merged")
//   ft_in        u64                 (学習側 FT 入力次元、feature set 依存)
//   ft_out       u64                 (FT 出力次元、`--ft-out`)
//   max_active   u64                 (学習側の 1 perspective あたり active feature 数)
//   ft_factorize u8                  (v6+、FT factorizer 有効 flag)
//   feature_hash u32                 (v7+、feature mapping identity)
//   threat_width u64               (v13+, packed Threat row width)
//   run_id_len   u32                 (4 bytes、producer run id の長さ、0 可)
//   run_id       UTF-8 [run_id_len]   (この checkpoint を書いた run の experiment.json `id`)
//   arch_len     u32                 (4 bytes、arch kind canonical 名の長さ)
//   arch_kind    UTF-8 [arch_len]     (arch kind canonical 名、例 "layerstack")
//   bucket_mode_len u32              (v8+、bucket mode 名の長さ、bucket 無しは 0)
//   bucket_mode  UTF-8 [bucket_mode_len] (例 "progresskpabs" / "progresskpabsq16" / "kingrank9"。
//                reader は非推奨 alias "progress8kpabs" も progresskpabs として受理)
//   topo_count   u64                 (topology 次元の個数)
//   topology     u64 [topo_count]     (arch 固有の層次元列)
//   superbatch   u64  (この checkpoint が表す完了 superbatch、resume はこの +1 から)
//   step_count   u64  (optimizer step counter)
//   lr_horizon   u64  (v5+、LR schedule の終端 superbatch。0 = horizon 無し)
//   fv_scale     i32  (v9+、Simple の実効 evaluation scale。0 = 未記録)
//   qat_flags    u32  (v10+, bit 0 = dense, bit 1 = piece FT i8x2)
//   net_id_len   u32  (v11+、producer net id の長さ、0 = 未記録)
//   net_id       UTF-8 [net_id_len] (この checkpoint を書いた run の net id)
//   recipe_fp    u64  (v11+、producer のレシピの指紋。0 = 未記録)
//   num_groups   u64
//   then for each of num_groups groups (group 順と各 group の名前 / 要素数は arch
//   固有 — 各 trainer の `raw_ckpt_group_sources` を参照):
//     len u64
//     w[f32 × len]
//     m[f32 × len]
//     v[f32 × len]
//     slow[f32 × len]
//
// header 部の write / read は write_raw_ckpt_header / read_raw_ckpt_header、
// group 本体込みの file 全体は save_raw_checkpoint_file / load_raw_checkpoint_file。
// version 互換規則 (1..=15 の受理と各 version の差分) は RAW_CKPT_VERSION の doc を参照。
// ===========================================================================

/// resume 元 checkpoint を書き出した run の識別情報 (header に記録が無い項目は `None`)。
/// `lineage.parent_id` と、resume した run が続きか派生かの判定に使う。
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ResumeProducer {
    pub(crate) run_id: Option<String>,
    pub(crate) net_id: Option<String>,
    pub(crate) recipe_fingerprint: Option<u64>,
}

impl RawCkptHeader {
    pub(crate) fn validate_ft_weight_i8x2(
        &self,
        requested: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.ft_weight_i8x2 == requested {
            return Ok(());
        }
        let setting = if self.ft_weight_i8x2 {
            "pass --ft-weight-i8x2"
        } else {
            "omit --ft-weight-i8x2"
        };
        Err(invalid_data(format!(
            "raw checkpoint piece FT quantization differs from the requested mode; {setting} when resuming"
        )))
    }

    pub(crate) fn prepare_optimizer_state(
        &mut self,
        path: &Path,
        groups: &mut [RawCkptGroup],
        optimizer: OptimizerKind,
        transfer_radam_momentum: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if transfer_radam_momentum {
            if self.sgd || optimizer != OptimizerKind::Sgd {
                return Err(invalid_data(
                    "--transfer-radam-momentum requires a RAdam checkpoint and --optimizer sgd"
                        .to_string(),
                ));
            }
            if self.version != PARTIAL_THREAT_RADAM_CKPT_VERSION {
                self.check_radam_producer(path)?;
            }
            // Both optimizers store the same gradient EMA with beta1 = 0.9.
            for (_, _, variance, slow) in groups {
                variance.fill(0.0);
                slow.fill(0.0);
            }
            self.step_count = 0;
            self.lr_horizon = None;
        } else if self.version == PARTIAL_THREAT_RADAM_CKPT_VERSION
            && optimizer != OptimizerKind::RAdam
        {
            return Err(invalid_data(
                "Version 15 compact checkpoints require --optimizer radam for resume; to start SGD, use --optimizer sgd with --transfer-radam-momentum"
                    .to_string(),
            ));
        } else if self.sgd != (optimizer == OptimizerKind::Sgd) {
            return Err(invalid_data(
                "SGD checkpoints require --optimizer sgd; to start SGD from a RAdam checkpoint, use --transfer-radam-momentum"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn check_radam_producer(&self, checkpoint: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let missing_metadata = || {
            if self.version == PARTIAL_THREAT_CKPT_VERSION {
                Err(invalid_data(
                    "--transfer-radam-momentum requires producer metadata identifying RAdam for a version 13 compact checkpoint"
                        .to_string(),
                ))
            } else {
                Ok(())
            }
        };
        let Some(run_id) = self.producer_run_id.as_deref() else {
            return missing_metadata();
        };
        let run_path = Path::new(run_id);
        if !run_path
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
        {
            return Err(invalid_data(
                "invalid checkpoint producer run id".to_string(),
            ));
        }
        let checkpoint = checkpoint.canonicalize()?;
        let Some(directory) = checkpoint.ancestors().nth(run_path.components().count()) else {
            return missing_metadata();
        };
        let metadata = directory.join("experiments").join(format!("{run_id}.json"));
        let file = match std::fs::File::open(&metadata) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return missing_metadata();
            }
            Err(error) => return Err(error.into()),
        };
        let experiment: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(file))?;
        if experiment["id"].as_str() != Some(run_id)
            || experiment["params"]["optimizer"].as_str() != Some("radam")
        {
            return Err(invalid_data(format!(
                "RAdam source checkpoint optimizer could not be verified from {}",
                metadata.display()
            )));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn producer(&self) -> ResumeProducer {
        ResumeProducer {
            run_id: self.producer_run_id.clone(),
            net_id: self.producer_net_id.clone(),
            recipe_fingerprint: self.producer_recipe_fingerprint,
        }
    }
}

/// raw checkpoint 1 group 分の host buffer (`w`, `m`, `v`, `slow` の f32 Vec、`grad` は含めない)。
pub(crate) type RawCkptGroup = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// LayerStack の `load_raw_checkpoint` の戻り値: `(完了 superbatch, producer の識別情報,
/// LR-schedule horizon)`。caller は superbatch+1 から resume し、horizon を
/// `build_lr_scheduler` に渡す。
#[cfg(feature = "gpu")]
pub(crate) type RawCkptResumeState = (usize, ResumeProducer, Option<usize>);

/// Simple raw checkpoint の resume metadata。末尾は保存済み `fv_scale`。
#[cfg(feature = "gpu")]
pub(crate) type SimpleRawCkptResumeState = (usize, ResumeProducer, Option<usize>, Option<i32>);

/// raw checkpoint 1 group 分の device-side 参照。weight name + 要素数 +
/// `(w, m, v, slow)` device buffer の借用。trainer が format の group 順に並べた
/// 列を [`save_raw_checkpoint_file`] / [`load_raw_checkpoint_file`] へ渡す。
#[cfg(feature = "gpu")]
pub(crate) struct RawCkptGroupSource<'a> {
    pub(crate) name: &'static str,
    pub(crate) len: usize,
    pub(crate) bufs: RawCkptGroupBufs<'a>,
}

/// [`RawCkptGroupSource`] の buffer 借用部。`Uniform` は w/m/v/slow 全部
/// `DeviceBuffer<f32>` の group、`FtMoment` は `m` / `v` が [`MomentBuf`]
/// (`--fp16-opt-state` で `f16` 格納) の ft_w group。
#[cfg(feature = "gpu")]
pub(crate) enum RawCkptGroupBufs<'a> {
    Uniform {
        w: &'a DeviceBuffer<f32>,
        m: &'a DeviceBuffer<f32>,
        v: &'a DeviceBuffer<f32>,
        slow: &'a DeviceBuffer<f32>,
    },
    FtMoment {
        w: &'a DeviceBuffer<f32>,
        m: &'a MomentBuf,
        v: &'a MomentBuf,
        slow: &'a DeviceBuffer<f32>,
    },
}

#[cfg(feature = "gpu")]
impl RawCkptGroupSource<'_> {
    /// device → host download。`FtMoment` の `m` / `v` は格納精度 (`f32`/`f16`) に
    /// 依らず**真値 `f32`** に戻す — checkpoint format は mode 非依存で、resume 時に
    /// 当該 run の精度へ再 quantize される。
    pub(crate) fn to_host(
        &self,
        stream: &CudaStream,
    ) -> Result<RawCkptGroup, Box<dyn std::error::Error>> {
        Ok(match self.bufs {
            RawCkptGroupBufs::Uniform { w, m, v, slow } => (
                w.to_host_vec(stream)?,
                m.to_host_vec(stream)?,
                v.to_host_vec(stream)?,
                slow.to_host_vec(stream)?,
            ),
            RawCkptGroupBufs::FtMoment { w, m, v, slow } => (
                w.to_host_vec(stream)?,
                m.to_host_f32(stream, FT_OPT_M_SCALE)?,
                v.to_host_f32(stream, FT_OPT_V_SCALE)?,
                slow.to_host_vec(stream)?,
            ),
        })
    }
}

/// LayerStack アーキの topology header (v4+、PSQT 無し): FT 出力次元・L1 出力次元・
/// L2 出力次元・bucket 数。`load_raw_checkpoint` がこの並びを checkpoint と照合する。
/// FT 出力次元は `--ft-out`、L1 出力次元は `--l1`、L2 出力次元は `--l2`、bucket
/// 数は `--num-buckets` で可変 (resume 時に topology dim 列がそのまま照合され、
/// 不一致は load 時に reject される)。
pub(crate) const fn layerstack_topology(
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
) -> [u64; 4] {
    [
        ft_out as u64,
        l1_out as u64,
        l2_out as u64,
        num_buckets as u64,
    ]
}

/// PSQT 有効時の LayerStack topology header: 末尾に PSQT bucket 数 (= `num_buckets`)
/// を追加し、PSQT 無し ckpt (`[..., num_buckets]`, 4 dims) と PSQT 有り ckpt
/// (`[..., num_buckets, num_buckets]`, 5 dims) を `topo_count` で弁別可能にする。
/// `--resume` で PSQT 有無を跨ぐ load は dim 数不一致で reject される。PSQT bucket
/// は LayerStack bucket と必ず一致するため同 `num_buckets` を 2 回書く。
#[cfg(feature = "gpu")]
pub(crate) const fn layerstack_topology_with_psqt(
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
) -> [u64; 5] {
    [
        ft_out as u64,
        l1_out as u64,
        l2_out as u64,
        num_buckets as u64,
        num_buckets as u64,
    ]
}

/// `read_raw_ckpt_header` が返す raw checkpoint header の解析結果。
#[derive(Debug)]
pub(crate) struct RawCkptHeader {
    version: u32,
    pub(crate) sgd: bool,
    pub(crate) qat_dense: bool,
    pub(crate) ft_weight_i8x2: bool,
    /// この checkpoint が表す完了 superbatch 番号。
    pub(crate) superbatch: usize,
    /// optimizer step counter (ranger では lookahead lerp の周期判定にも使う)。
    pub(crate) step_count: u64,
    /// format 記載の weight group 数 (caller が arch 期待値と照合する)。
    pub(crate) num_groups: u64,
    /// producer run の experiment.json id (version 3+ かつ記録ありなら `Some`)。
    pub(crate) producer_run_id: Option<String>,
    /// LR-schedule horizon (version 5+ かつ horizon を持つ schedule なら `Some`)。
    /// version 1..=4 や horizon を持たない schedule では `None`。
    pub(crate) lr_horizon: Option<usize>,
    /// Simple の実効 evaluation scale。version 1..=8 または未記録なら `None`。
    pub(crate) fv_scale: Option<i32>,
    /// producer run の net id (version 11+ かつ記録ありなら `Some`)。
    pub(crate) producer_net_id: Option<String>,
    /// producer run のレシピの指紋 (version 11+ かつ記録ありなら `Some`)。
    pub(crate) producer_recipe_fingerprint: Option<u64>,
}

/// raw checkpoint の header を読み、`expected` の arch identity と照合する。
/// version 1..=15 を受理し、不一致 / 破損は `InvalidData` で reject する。
///
/// version 1..=3 は arch-kind header を持たず暗黙に `layerstack`。version 4 は
/// arch_kind 名と topology 次元列を `expected` と照合する。version 5 は
/// `step_count` の後に LR-schedule horizon の `u64` を持つ (`0` = horizon 無し)。
/// version 7 は feature-set header に feature hash、version 8 は arch kind の直後に
/// bucket mode 名、version 9 は LR-schedule horizon の直後に `fv_scale` を持つ。
/// version 7 以前の LayerStack は `progresskpabs` と解釈する。
pub(crate) fn read_raw_ckpt_header<R: std::io::Read>(
    r: &mut R,
    expected: &RawCkptArch,
) -> Result<RawCkptHeader, Box<dyn std::error::Error>> {
    let mut reader = RawCkptReader::new(r, RawCkptReadMode::Resume)?;
    let version = reader.version();
    let want_name = expected.feature_set.canonical_name();

    if let Some(features) = reader.features()? {
        let fs_name = features.name;
        let ckpt_ft_in = features.train_ft_in;
        let ckpt_ft_out = features.ft_out;
        let ckpt_max_active = features.max_active;
        let ckpt_ft_factorize = features.ft_factorize;
        let ckpt_feature_hash = features.feature_hash;
        let width = features.threat_width.unwrap_or(expected.ft_out);
        if width != expected.feature_set.threat_width(expected.ft_out as usize) as u64 {
            return Err(invalid_data("raw checkpoint Threat width mismatch".into()));
        }
        let want = expected.feature_set;
        if fs_name != want_name {
            return Err(invalid_data(format!(
                "raw checkpoint feature set mismatch: checkpoint is '{fs_name}', \
                 requested '{want_name}' (cannot resume across feature sets)"
            )));
        }
        // 次元照合より先に factorize 状態を見る (on/off 跨ぎは ft_in も必ず
        // ずれるが、原因が読めるエラーを先に出す)。
        if ckpt_ft_factorize != want.ft_factorize() {
            return Err(invalid_data(format!(
                "raw checkpoint ft-factorize mismatch: checkpoint {ckpt_ft_factorize}, \
                 requested {} (cannot resume across --ft-factorize on/off)",
                want.ft_factorize()
            )));
        }
        if let Some(ckpt_feature_hash) = ckpt_feature_hash {
            let want_feature_hash = want.feature_hash();
            if ckpt_feature_hash != want_feature_hash {
                return Err(invalid_data(format!(
                    "raw checkpoint feature hash mismatch: got {ckpt_feature_hash:#010x}, \
                     want {want_feature_hash:#010x} (effect bucket config / threat profile mismatch)"
                )));
            }
        } else if want.effect_bucket_config().is_some() {
            return Err(invalid_data(
                "raw checkpoint for this EffectBucket spec requires a checkpoint with feature hash"
                    .to_string(),
            ));
        }
        if ckpt_ft_in != want.train_ft_in() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint ft_in mismatch: got {ckpt_ft_in}, want {}",
                want.train_ft_in()
            )));
        }
        if ckpt_ft_out != expected.ft_out {
            return Err(invalid_data(format!(
                "raw checkpoint ft_out mismatch: got {ckpt_ft_out}, want {}",
                expected.ft_out
            )));
        }
        // factorize 有効時は base に加え 2×base も受理する: v6 の factorized
        // file には、仮想特徴を sparse index 列に流す実装が書いた 2×base 値の
        // 個体が存在する (RAW_CKPT_VERSION doc 参照)。tensor payload (group 長
        // は train_ft_in 由来) は max_active 値に依らず同一で、この field の差
        // だけで読める checkpoint を弾く理由がない。version 番号では弁別でき
        // ないため値で受理する。
        let legacy_factorized_max_active =
            want.ft_factorize() && ckpt_max_active == 2 * want.max_active() as u64;
        // Threat's sparse input capacity can grow without changing any checkpoint tensors.
        let legacy_threat_max_active = want.threat_profile().is_some()
            && ckpt_max_active == (want.feature_set().spec().max_active() + 128) as u64;
        if ckpt_max_active != want.max_active() as u64
            && !legacy_factorized_max_active
            && !legacy_threat_max_active
        {
            return Err(invalid_data(format!(
                "raw checkpoint max_active mismatch: got {ckpt_max_active}, want {}",
                want.max_active()
            )));
        }
    } else if expected.feature_set.effect_bucket_config().is_some() {
        return Err(invalid_data(
            "raw checkpoint for this EffectBucket spec requires a checkpoint with feature hash"
                .to_string(),
        ));
    } else if want_name != FeatureSet::HalfKaHmMerged.spec().canonical_name() {
        return Err(invalid_data(format!(
            "raw checkpoint version 1 is always 'halfka-hm-merged', \
             requested '{want_name}' (cannot resume across feature sets)"
        )));
    }

    let producer_run_id = reader.producer_run_id()?;

    if let Some(arch_name) = reader.arch_name()? {
        let ckpt_arch = ArchKind::from_canonical_name(&arch_name).ok_or_else(|| {
            invalid_data(format!(
                "raw checkpoint has unknown arch kind '{arch_name}'"
            ))
        })?;
        if ckpt_arch != expected.arch_kind {
            return Err(invalid_data(format!(
                "raw checkpoint arch kind mismatch: checkpoint is '{}', requested '{}' \
                 (cannot resume across architectures)",
                ckpt_arch.canonical_name(),
                expected.arch_kind.canonical_name()
            )));
        }
        let ckpt_bucket_mode = if let Some(bucket_mode) = reader.bucket_mode()? {
            // 旧名 alias で書かれた checkpoint も canonical 名の checkpoint と同じ
            // mode として照合する。未知の mode 名は正規化せずそのまま比較に回し、
            // 下の不一致エラーで surface させる。
            (!bucket_mode.is_empty()).then(|| {
                BucketMode::parse(&bucket_mode).map_or(bucket_mode, |(mode, _)| mode.identity())
            })
        } else if ckpt_arch == ArchKind::LayerStack {
            // bucket-mode header を持たない version の LayerStack は、当時唯一の
            // mode だった progresskpabs として照合する。
            Some(BucketMode::ProgressKpAbs.canonical_name().to_string())
        } else {
            None
        };
        if ckpt_bucket_mode.as_deref() != expected.bucket_mode {
            if let (Some(got), Some(want)) = (ckpt_bucket_mode.as_deref(), expected.bucket_mode) {
                return Err(invalid_data(format!(
                    "checkpoint bucket mode '{got}' does not match --bucket-mode '{want}'"
                )));
            }
            return Err(invalid_data(format!(
                "raw checkpoint bucket mode {:?} does not match architecture '{}'",
                ckpt_bucket_mode.as_deref().unwrap_or(""),
                expected.arch_kind.canonical_name()
            )));
        }
        let topo_count = reader.topology_count()?;
        if topo_count != expected.topology.len() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint topology dim count {topo_count} != expected {}",
                expected.topology.len()
            )));
        }
        for (i, &want_dim) in expected.topology.iter().enumerate() {
            let got = reader.topology_dim()?;
            if got != want_dim {
                return Err(invalid_data(format!(
                    "raw checkpoint topology dim {i} mismatch: got {got}, want {want_dim} \
                     (network architecture mismatch)"
                )));
            }
        }
    } else if expected.arch_kind != ArchKind::LayerStack {
        return Err(invalid_data(format!(
            "raw checkpoint version {version} predates the arch-kind header and is \
             always 'layerstack', requested '{}' (cannot resume across architectures)",
            expected.arch_kind.canonical_name()
        )));
    } else if expected.bucket_mode != Some(BucketMode::ProgressKpAbs.canonical_name()) {
        return Err(invalid_data(format!(
            "checkpoint bucket mode '{}' does not match --bucket-mode '{}'",
            BucketMode::ProgressKpAbs.canonical_name(),
            expected.bucket_mode.unwrap_or("")
        )));
    }

    let superbatch_u64 = reader.superbatch()?;
    let superbatch: usize = superbatch_u64.try_into().map_err(|_| {
        invalid_data(format!(
            "raw checkpoint superbatch {superbatch_u64} exceeds usize::MAX"
        ))
    })?;
    let step_count = reader.step_count()?;
    let lr_horizon = reader
        .lr_horizon()?
        .map(|horizon| {
            usize::try_from(horizon).map_err(|_| {
                invalid_data(format!(
                    "raw checkpoint lr_horizon {horizon} exceeds usize::MAX"
                ))
            })
        })
        .transpose()?;

    let fv_scale = reader.fv_scale()?;
    if let Some(value) = fv_scale.filter(|&value| value < 0) {
        return Err(invalid_data(format!(
            "raw checkpoint fv_scale {value} is invalid (must be non-negative)"
        )));
    }

    let flags = reader.qat_flags()?;
    let (qat_dense, ft_weight_i8x2) = nnue_train::raw_checkpoint::decode_qat_flags(flags)
        .filter(|_| flags == 0 || expected.arch_kind == ArchKind::LayerStack)
        .ok_or_else(|| invalid_data(format!("unsupported QAT flags {flags}")))?;

    let (producer_net_id, producer_recipe_fingerprint) = reader.producer_recipe()?;
    let num_groups = reader.group_count()?;

    Ok(RawCkptHeader {
        version,
        sgd: matches!(
            version,
            SGD_CHECKPOINT_VERSION | PARTIAL_THREAT_SGD_CKPT_VERSION
        ),
        qat_dense,
        ft_weight_i8x2,
        superbatch,
        step_count,
        num_groups,
        producer_run_id,
        lr_horizon,
        fv_scale,
        producer_net_id,
        producer_recipe_fingerprint,
    })
}

/// raw checkpoint を `path` に atomic に書き出す。header (arch identity + counters +
/// `num_groups` = `groups.len()`) に続けて `groups` を列挙順に書く。group 本体の
/// layout は各 group につき `len u64 + w[f32×len] + m[f32×len] + v[f32×len] +
/// slow[f32×len]` (全 little-endian)。device → host download は group 単位で write と
/// interleave するので、host 側のピークは最大 group (ft_w、~113M f32 = ~450MB) 1 個分。
///
/// 排他的な一時ファイルへ書き込み、同期してから atomic に置換する。
/// 書き込み途中で停止しても `path` は前回の完全な checkpoint のまま。
///
/// `meta.run_id` が空文字列、または [`MAX_RUN_ID_BYTES`] 超過 (warning を出して
/// 省略) のときは run id を持たない checkpoint になり、resume 時の
/// `lineage.parent_id` は解決されない。
#[cfg(feature = "gpu")]
pub(crate) fn save_raw_checkpoint_file(
    path: &Path,
    stream: &CudaStream,
    arch: &RawCkptArch,
    meta: &RawCkptMeta,
    groups: &[RawCkptGroupSource<'_>],
) -> Result<(), Box<dyn std::error::Error>> {
    // 過長な run id (`{net_id}-{時刻}-{pid}`、通常数十バイト) は lineage という
    // メタデータのために学習を中断させる価値がない。上限超過時は埋め込みを
    // 省略 (長さ 0) し、warning を出して checkpoint 保存は続行する。
    let run_id = if meta.run_id.len() > MAX_RUN_ID_BYTES {
        eprintln!(
            "[train] warning: producer run id ({} bytes) exceeds {MAX_RUN_ID_BYTES}; \
             omitting it from {} (resume lineage parent will be unresolved)",
            meta.run_id.len(),
            path.display()
        );
        ""
    } else {
        meta.run_id
    };

    nnue_train::artifact::write_atomic(path, |w| -> Result<(), Box<dyn std::error::Error>> {
        let header_meta = RawCkptMeta { run_id, ..*meta };
        write_raw_ckpt_header(w, arch, &header_meta, groups.len() as u64)?;
        for g in groups {
            let (w_host, m_host, v_host, slow_host) = g.to_host(stream)?;
            // 念のため device buffer の要素数を arch 期待値と照合 (内部整合性)。
            for (label, got) in [
                ("w", w_host.len()),
                ("m", m_host.len()),
                ("v", v_host.len()),
                ("slow", slow_host.len()),
            ] {
                if got != g.len {
                    return Err(format!(
                        "raw checkpoint: group {} {label} buffer len {got} != expected {}",
                        g.name, g.len
                    )
                    .into());
                }
            }
            write_raw_ckpt_group(w, [&w_host, &m_host, &v_host, &slow_host])?;
        }
        Ok(())
    })
}

/// raw checkpoint を読み、header 照合 + 全 group の host `Vec` を返す (`--resume` 用)。
/// `expected_groups` は format の group 順の `(name, 要素数)` で、header の
/// `num_groups` / 各 group の `len` をこれと照合し、不一致は `InvalidData` で
/// reject する。全 group を読み切ってから返すので、caller は読み途中の失敗で
/// device 側が中途半端な state になる心配なく upload できる (trailing garbage は
/// 許容、足りないのは `read_exact` が弾く)。
pub(crate) fn load_raw_checkpoint_file(
    path: &Path,
    expected_arch: &RawCkptArch,
    expected_groups: &[(&'static str, usize)],
) -> Result<(RawCkptHeader, Vec<RawCkptGroup>), Box<dyn std::error::Error>> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    let header = read_raw_ckpt_header(&mut r, expected_arch)?;
    if header.num_groups != expected_groups.len() as u64 {
        return Err(invalid_data(format!(
            "raw checkpoint num_groups {} != expected {}; for LayerStack, check whether \
             --stack-shared-delta matches the checkpoint",
            header.num_groups,
            expected_groups.len()
        )));
    }
    let mut groups = Vec::with_capacity(expected_groups.len());
    for &(name, expected_len) in expected_groups {
        groups.push(read_raw_ckpt_group(&mut r, name, expected_len)?);
    }
    Ok((header, groups))
}

/// 1 group 分 (len + w/m/v/slow の f32 × len) を読む。file 記載 `len` と
/// `expected_len` の不一致 / `u64 → usize` overflow は `InvalidData` で reject。
fn read_raw_ckpt_group<R: std::io::Read>(
    r: &mut R,
    name: &str,
    expected_len: usize,
) -> Result<RawCkptGroup, Box<dyn std::error::Error>> {
    let len_u64 =
        read_raw_ckpt_group_len(r, RawCkptReadMode::Resume, &format!("group {name} len"))?;
    let len: usize = len_u64.try_into().map_err(|_| {
        invalid_data(format!(
            "raw checkpoint group {name} len {len_u64} exceeds usize::MAX"
        ))
    })?;
    if len != expected_len {
        return Err(invalid_data(format!(
            "raw checkpoint group {name} len mismatch: got {len}, want {expected_len} \
             (network architecture mismatch)"
        )));
    }
    let w_host = read_f32_vec_io(r, len, &format!("group {name} w"))?;
    let m_host = read_f32_vec_io(r, len, &format!("group {name} m"))?;
    let v_host = read_f32_vec_io(r, len, &format!("group {name} v"))?;
    let slow_host = read_f32_vec_io(r, len, &format!("group {name} slow"))?;
    Ok((w_host, m_host, v_host, slow_host))
}

/// raw checkpoint の magic/version/dim 検証で使う
/// `io::ErrorKind::InvalidData` の `Box<dyn Error>` を作る短縮 helper。
pub(crate) fn invalid_data(msg: String) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
}

/// `r.read_exact(buf)` を呼び、`UnexpectedEof` (= file が途中で切れている、破損 / 部分書き)
/// を `InvalidData` + context message に正規化する。raw checkpoint の robustness contract
/// 「malformed input は全部 `InvalidData`、panic しない」を満たすため、`load_raw_checkpoint`
/// 内の全 `read_exact` はこの helper 経由で呼ぶ (`what` は読もうとしていた field の説明)。
pub(crate) fn read_exact_or_invalid<R: std::io::Read>(
    r: &mut R,
    buf: &mut [u8],
    what: &str,
) -> std::io::Result<()> {
    read_raw_ckpt_exact(r, buf, RawCkptReadMode::Resume, what)
}

/// little-endian f32 を `n` 個読み、`io::Result` を返す。`what` は短読み
/// (破損 file) 時の context message に使う (`UnexpectedEof` → `InvalidData` に
/// 正規化、`read_exact_or_invalid` 経由)。
pub(crate) fn read_f32_vec_io<R: std::io::Read>(
    r: &mut R,
    n: usize,
    what: &str,
) -> std::io::Result<Vec<f32>> {
    n.checked_mul(4)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("f32 vec len {n} exceeds allocation size limit"),
            )
        })?;
    let mut out = Vec::with_capacity(n);
    let mut bytes = vec![0_u8; n.min(F32_IO_CHUNK_BYTES / 4) * 4];
    while out.len() < n {
        let chunk_bytes = ((n - out.len()).min(F32_IO_CHUNK_BYTES / 4)) * 4;
        read_exact_or_invalid(r, &mut bytes[..chunk_bytes], what)?;
        for chunk in bytes[..chunk_bytes].chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }
    Ok(out)
}
