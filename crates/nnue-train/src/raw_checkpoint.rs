//! Host serialization for raw NNUE training checkpoints.

use std::io::{self, Read, Write};

use nnue_format::{ArchKind, LayerStackHead};
use shogi_features::FeatureSetSpec;

use crate::optimizer::OptimizerKind;

/// raw checkpoint format magic (`b"RNRC"` = "RShogi Nnue Resume Checkpoint")。
/// weight group raw f32 + optimizer state + step + superbatch を 1 file に
/// まとめた self-contained format。
pub const RAW_CKPT_MAGIC: [u8; 4] = *b"RNRC";

/// raw checkpoint format version。
///
/// - `1`: no feature-set header; the weights are always `halfka-hm-merged`.
/// - `2`: a self-describing feature-set header (canonical name + `ft_in` +
///   `ft_out` + `max_active`) follows the magic + version fields.
/// - `3`: a producer run id (length-prefixed UTF-8 — the experiment.json `id`
///   of the run that wrote the checkpoint) follows the feature-set header.
///   `--resume` reads it to fill `lineage.parent_id`.
/// - `4`: an arch-kind name (length-prefixed UTF-8) and a topology header (a
///   count-prefixed list of `u64` layer dimensions) follow the producer run
///   id. They pin which architecture and layer shape the checkpoint belongs
///   to, so a checkpoint written by one architecture cannot be resumed by
///   another.
/// - `5`: a `u64` LR-schedule horizon (the superbatch at which the LR curve
///   reaches its terminal value — the decay `final_superbatch` or one-cycle
///   `total_superbatch`) follows `step_count`. `0` means "no horizon recorded"
///   (the producing schedule had none, e.g. step / constant / drop). `--resume`
///   prefers it over the `--superbatches`-derived default so the curve is
///   reproduced independently of `--superbatches`.
///
/// - `6`: a FT-factorizer flag byte follows `max_active` in the feature-set
///   header, and the `ft_in` field holds the **training-side row count**
///   (`train_ft_in`; equal to the base `ft_in` whenever the factorizer is
///   off, so v6 files written without the factorizer keep the v2 field
///   semantics). `max_active` stays the base per-position active count — the
///   sparse index stream is factorizer-independent (virtual rows are wired
///   through dense fold / reduce kernels, not through sparse indices). The
///   flag pins whether the checkpoint's FT weight rows include the
///   training-time virtual factorizer block; resuming across
///   `--ft-factorize` on/off is rejected. Some factorized v6 files record
///   `2 × max_active` in this field (written by an implementation that
///   routed virtual features through the sparse index stream); the reader
///   accepts that value too, because the tensor payload is identical either
///   way.
///
/// - `7`: a `u32` feature hash follows the FT-factorizer flag in the
///   feature-set header. It distinguishes feature mappings that share the
///   same canonical feature-set name and dimensions.
///
/// - `8`: a bucket-mode name (length-prefixed UTF-8, empty when the architecture
///   has no bucket concept) follows the arch-kind name. Cross-bucket-mode resumes
///   are rejected. The reader also accepts the deprecated alias spelling
///   "progress8kpabs" and treats it as "progresskpabs". "progresskpabsq16" is a
///   distinct mode, so resuming across it and "progresskpabs" is rejected too. For v<=7 layerstack
///   checkpoints the mode is absent and is interpreted as "progresskpabs" (the
///   only mode that existed when they were written).
///
/// - `9`: an `i32` `fv_scale` follows the LR-schedule horizon. `0` means "not
///   recorded"; positive values preserve the Simple trainer's effective export
///   scale across resume. Negative values are rejected as corrupt.
///
/// - `10`: a u32 QAT mode follows fv_scale. 0 is off and 1 enables dense QAT.
///   Earlier versions imply QAT is off.
///
/// - `11`: the producer net id (length-prefixed UTF-8, empty = not recorded) and
///   a `u64` recipe fingerprint (`0` = not recorded) follow the QAT mode.
///   `--resume` compares them with the resuming run to record whether it
///   continues the producer's training or forks a new experiment from it.
///
/// - `12`: the version 11 layout with SGD momentum state.
/// - `13`: compact Threat with the version 11 optimizer state. A u64 Threat
///   row width follows the feature hash.
/// - `14`: the compact Threat layout with SGD momentum state.
/// - `15`: the compact Threat layout with RAdam optimizer state.
///
/// - `16`: optimizer-step schedule envelope: u32 layout version (11..=15),
///   then u64 LR total steps, followed by that layout's header and tensors.
///
/// The trainer resume format accepts versions 1..=16. Version 1 is interpreted as
/// `halfka-hm-merged`; versions 1..=3 predate the arch-kind header and are
/// interpreted as `layerstack`. Versions above 16 are rejected. The producer
/// run id is absent (`None`) for versions 1 and 2; the LR horizon is absent
/// (`None`) for versions 1..=4; the factorizer flag is absent (false) for
/// versions 1..=5; the feature hash is absent for versions 1..=6; `fv_scale`
/// is absent (`None`) for versions 1..=8; the producer net id and recipe
/// fingerprint are absent (`None`) for versions 1..=10.
pub const RAW_CKPT_VERSION: u32 = 16;
const STEP_SCHEDULE_CKPT_VERSION: u32 = 16;

/// LayerStack topology extension: PSQT bucket count (zero if absent), then this head tag.
pub const DENSE_CONCAT_TOPOLOGY_TAG: u64 = 1;

pub fn layerstack_topology_head(topology: &[u64]) -> io::Result<(LayerStackHead, bool)> {
    match topology {
        [_, _, _, _] => Ok((LayerStackHead::SplitSkip, false)),
        [_, _, _, buckets, psqt] if psqt == buckets => Ok((LayerStackHead::SplitSkip, true)),
        [_, _, _, buckets, psqt, DENSE_CONCAT_TOPOLOGY_TAG] if *psqt == 0 || psqt == buckets => {
            Ok((LayerStackHead::DenseConcat, *psqt != 0))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid LayerStack head, PSQT topology or bucket count",
        )),
    }
}

pub fn read_layerstack_checkpoint_head(r: impl Read) -> io::Result<LayerStackHead> {
    let mut reader = RawCkptReader::new(r, RawCkptReadMode::Resume)?;
    reader.features()?;
    reader.producer_run_id()?;
    let Some(arch_name) = reader.arch_name()? else {
        return Ok(LayerStackHead::SplitSkip);
    };
    if arch_name != ArchKind::LayerStack.canonical_name() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected a LayerStack checkpoint, got '{arch_name}'"),
        ));
    }
    reader.bucket_mode()?;
    let count = reader.topology_count()?;
    if !(4..=6).contains(&count) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid LayerStack topology dimension count",
        ));
    }
    let mut topology = [0; 6];
    let topology = &mut topology[..count as usize];
    for dim in &mut *topology {
        *dim = reader.topology_dim()?;
    }
    layerstack_topology_head(topology).map(|(head, _)| head)
}

pub const RAW_CKPT_ARCH_HEADER_VERSION: u32 = 4;
const PRODUCER_RECIPE_VERSION: u32 = 11;
const ADAM_CHECKPOINT_VERSION: u32 = PRODUCER_RECIPE_VERSION;
const MAX_NAME_BYTES: usize = 256;
pub const SGD_CHECKPOINT_VERSION: u32 = 12;
pub const PARTIAL_THREAT_CKPT_VERSION: u32 = 13;
pub const PARTIAL_THREAT_SGD_CKPT_VERSION: u32 = 14;
pub const PARTIAL_THREAT_RADAM_CKPT_VERSION: u32 = 15;

/// `*.ckpt` の producer run id のバイト数上限。run id は `{net_id}-{時刻}-{pid}`
/// 程度で高々数十バイト。破損 file の巨大な length 値で過大確保しないための上限。
pub const MAX_RUN_ID_BYTES: usize = 256;

/// raw checkpoint header の arch identity 部 (write / read 双方の引数)。feature
/// set・arch 種別・FT 出力次元・topology 次元列をまとめて持つ。
pub struct RawCkptArch<'a> {
    /// 入力 feature set (canonical 名 / `ft_in` / `max_active` の源)。
    pub feature_set: FeatureSetSpec,
    /// network アーキ種別。
    pub arch_kind: ArchKind,
    /// LayerStack の bucket 算出方式。bucket 概念を持たないアーキでは `None`。
    pub bucket_mode: Option<&'a str>,
    /// FT 出力次元 (feature set header の `ft_out` 欄に書く値)。
    pub ft_out: u64,
    /// arch 固有の層次元列 (v4 topology header)。
    pub topology: &'a [u64],
}

/// raw checkpoint header の arch identity 以外の metadata。
#[derive(Clone, Copy)]
pub struct RawCkptMeta<'a> {
    pub optimizer: OptimizerKind,
    pub qat_dense: bool,
    /// この checkpoint を書き出す run の experiment.json `id` (resume 時の
    /// `lineage.parent_id` に使う)。空文字列は「未記録」。
    pub run_id: &'a str,
    /// この checkpoint が表す完了 superbatch 番号 (resume はこの +1 から)。
    pub superbatch: usize,
    /// optimizer step counter (ranger では lookahead lerp の周期判定にも使う)。
    pub step_count: u64,
    /// LR-schedule horizon (horizon を持たない schedule では `None`)。
    pub lr_horizon: Option<usize>,
    pub lr_total_steps: Option<u64>,
    /// Simple の export に使う実効 evaluation scale。LayerStack は `None`。
    pub fv_scale: Option<i32>,
    /// この checkpoint を書き出す run の net id。空文字列は「未記録」。
    pub net_id: &'a str,
    /// この checkpoint を書き出す run のレシピの指紋。実験ログが無ければ `None`。
    pub recipe_fingerprint: Option<u64>,
}

#[derive(Clone, Copy)]
pub enum RawCkptReadMode {
    Resume,
    Average,
}

impl RawCkptReadMode {
    fn field(self, resume: &'static str, average: &'static str) -> &'static str {
        match self {
            Self::Resume => resume,
            Self::Average => average,
        }
    }
}

pub struct RawCkptFeatures {
    pub name: String,
    pub train_ft_in: u64,
    pub ft_out: u64,
    pub max_active: u64,
    pub ft_factorize: bool,
    pub feature_hash: Option<u32>,
    pub threat_width: Option<u64>,
}

/// Reads header sections in wire order so callers can validate each section before reading on.
pub struct RawCkptReader<R> {
    r: R,
    mode: RawCkptReadMode,
    version: u32,
    lr_total_steps: Option<u64>,
}

impl<R: Read> RawCkptReader<R> {
    pub fn new(r: R, mode: RawCkptReadMode) -> io::Result<Self> {
        let mut reader = Self {
            r,
            mode,
            version: 0,
            lr_total_steps: None,
        };
        let magic = reader.array::<4>("magic")?;
        if magic != RAW_CKPT_MAGIC {
            return Err(invalid(match mode {
                RawCkptReadMode::Resume => {
                    format!("raw checkpoint magic mismatch: got {magic:?}, want {RAW_CKPT_MAGIC:?}")
                }
                RawCkptReadMode::Average => "not a raw training checkpoint (magic mismatch)".into(),
            }));
        }
        let version = reader.u32("version")?;
        let min = match mode {
            RawCkptReadMode::Resume => 1,
            RawCkptReadMode::Average => RAW_CKPT_ARCH_HEADER_VERSION,
        };
        if !(min..=RAW_CKPT_VERSION).contains(&version) {
            let supported = mode.field("this build reads", "supported:");
            let prefix = mode.field("raw checkpoint", "checkpoint");
            return Err(invalid(format!(
                "{prefix} version {version} is not supported ({supported} {min}..={RAW_CKPT_VERSION})"
            )));
        }
        reader.version = if version == STEP_SCHEDULE_CKPT_VERSION {
            let layout = reader.u32("layout version")?;
            if !(11..=15).contains(&layout) {
                return Err(invalid(
                    "unsupported optimizer-step checkpoint layout".into(),
                ));
            }
            let total = reader.u64("LR total steps")?;
            if total == 0 {
                return Err(invalid("LR total steps must be positive".into()));
            }
            reader.lr_total_steps = Some(total);
            layout
        } else {
            version
        };
        Ok(reader)
    }

    pub fn lr_total_steps(&self) -> Option<u64> {
        self.lr_total_steps
    }

    /// Tensor/header layout version, excluding the optional schedule envelope.
    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn features(&mut self) -> io::Result<Option<RawCkptFeatures>> {
        if self.version == 1 {
            return Ok(None);
        }
        Ok(Some(RawCkptFeatures {
            name: self.name("feature set name", MAX_NAME_BYTES)?,
            train_ft_in: self.u64("ft_in")?,
            ft_out: self.u64("ft_out")?,
            max_active: self.u64("max_active")?,
            ft_factorize: self.version >= 6 && self.array::<1>("ft_factorize flag")?[0] != 0,
            feature_hash: (self.version >= 7)
                .then(|| self.u32("feature hash"))
                .transpose()?,
            threat_width: self.optional_u64(PARTIAL_THREAT_CKPT_VERSION, "Threat width")?,
        }))
    }

    pub fn producer_run_id(&mut self) -> io::Result<Option<String>> {
        if self.version < 3 {
            return Ok(None);
        }
        let id = self.name("producer run id", MAX_RUN_ID_BYTES)?;
        Ok((!id.is_empty()).then_some(id))
    }

    pub fn arch_name(&mut self) -> io::Result<Option<String>> {
        let field = self.mode.field("arch kind name", "arch kind");
        (self.version >= RAW_CKPT_ARCH_HEADER_VERSION)
            .then(|| self.name(field, MAX_NAME_BYTES))
            .transpose()
    }

    pub fn bucket_mode(&mut self) -> io::Result<Option<String>> {
        let field = self.mode.field("bucket mode name", "bucket mode");
        (self.version >= 8)
            .then(|| self.name(field, MAX_NAME_BYTES))
            .transpose()
    }

    pub fn topology_count(&mut self) -> io::Result<u64> {
        self.u64("topology dim count")
    }

    pub fn topology_dim(&mut self) -> io::Result<u64> {
        self.u64("topology dim")
    }

    pub fn superbatch(&mut self) -> io::Result<u64> {
        self.u64("superbatch")
    }

    pub fn step_count(&mut self) -> io::Result<u64> {
        self.u64(self.mode.field("step_count", "step count"))
    }

    pub fn lr_horizon(&mut self) -> io::Result<Option<u64>> {
        self.optional_u64(5, self.mode.field("lr_horizon", "LR horizon"))
            .map(|horizon| horizon.filter(|&value| value != 0))
    }

    pub fn fv_scale(&mut self) -> io::Result<Option<i32>> {
        if self.version < 9 {
            return Ok(None);
        }
        let value = i32::from_le_bytes(self.array("fv_scale")?);
        Ok((value != 0).then_some(value))
    }

    pub fn qat_flags(&mut self) -> io::Result<u32> {
        if self.version < 10 {
            return Ok(0);
        }
        self.u32("QAT flags")
    }

    pub fn producer_recipe(&mut self) -> io::Result<(Option<String>, Option<u64>)> {
        if self.version < PRODUCER_RECIPE_VERSION {
            return Ok((None, None));
        }
        let id = self.name("producer net id", MAX_RUN_ID_BYTES)?;
        let fingerprint = self.u64("recipe fingerprint")?;
        Ok((
            (!id.is_empty()).then_some(id),
            (fingerprint != 0).then_some(fingerprint),
        ))
    }

    pub fn group_count(&mut self) -> io::Result<u64> {
        self.u64(self.mode.field("num_groups", "group count"))
    }

    pub fn into_inner(self) -> R {
        self.r
    }

    fn array<const N: usize>(&mut self, field: &str) -> io::Result<[u8; N]> {
        let mut bytes = [0; N];
        read_raw_ckpt_exact(&mut self.r, &mut bytes, self.mode, field)?;
        Ok(bytes)
    }

    fn u32(&mut self, field: &str) -> io::Result<u32> {
        self.array(field).map(u32::from_le_bytes)
    }

    fn u64(&mut self, field: &str) -> io::Result<u64> {
        self.array(field).map(u64::from_le_bytes)
    }

    fn optional_u64(&mut self, min_version: u32, field: &str) -> io::Result<Option<u64>> {
        (self.version >= min_version)
            .then(|| self.u64(field))
            .transpose()
    }

    fn name(&mut self, field: &str, max: usize) -> io::Result<String> {
        let len = match self.mode {
            RawCkptReadMode::Resume => self.u32(&format!("{field} length"))?,
            RawCkptReadMode::Average => self.u32(field)?,
        } as usize;
        let prefix = self.mode.field("raw checkpoint ", "");
        if len > max {
            let message = format!("{prefix}{field} length {len} is implausible");
            return Err(invalid(match self.mode {
                RawCkptReadMode::Resume => format!("{message} (max {max})"),
                RawCkptReadMode::Average => message,
            }));
        }
        let mut bytes = vec![0; len];
        match self.mode {
            RawCkptReadMode::Resume => {
                read_raw_ckpt_exact(&mut self.r, &mut bytes, self.mode, field)?
            }
            RawCkptReadMode::Average => self
                .r
                .read_exact(&mut bytes)
                .map_err(|_| truncated(self.mode, field))?,
        }
        String::from_utf8(bytes).map_err(|_| invalid(format!("{prefix}{field} is not valid UTF-8")))
    }
}

pub fn read_raw_ckpt_exact(
    r: &mut impl Read,
    bytes: &mut [u8],
    mode: RawCkptReadMode,
    field: &str,
) -> io::Result<()> {
    r.read_exact(bytes).map_err(|err| {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            truncated(mode, field)
        } else {
            err
        }
    })
}

pub fn read_raw_ckpt_group_len(
    r: &mut impl Read,
    mode: RawCkptReadMode,
    field: &str,
) -> io::Result<u64> {
    let mut bytes = [0; 8];
    read_raw_ckpt_exact(r, &mut bytes, mode, field)?;
    Ok(u64::from_le_bytes(bytes))
}

fn truncated(mode: RawCkptReadMode, field: &str) -> io::Error {
    let prefix = mode.field("raw checkpoint", "checkpoint");
    invalid(format!("{prefix} truncated while reading {field}"))
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Decodes the dense QAT mode.
pub fn decode_qat_flags(flags: u32) -> Option<bool> {
    (flags <= 1).then_some(flags == 1)
}

/// raw checkpoint の header (magic 〜 num_groups、group 本体の手前まで) を書く。
pub fn write_raw_ckpt_header<W: Write>(
    w: &mut W,
    arch: &RawCkptArch,
    meta: &RawCkptMeta,
    num_groups: u64,
) -> std::io::Result<()> {
    w.write_all(&RAW_CKPT_MAGIC)?;
    let threat_width = arch.feature_set.threat_width(arch.ft_out as usize) as u64;
    let partial = threat_width != arch.ft_out;
    let version = match (partial, meta.optimizer) {
        (false, OptimizerKind::Sgd) => SGD_CHECKPOINT_VERSION,
        (false, _) => ADAM_CHECKPOINT_VERSION,
        (true, OptimizerKind::Sgd) => PARTIAL_THREAT_SGD_CKPT_VERSION,
        (true, OptimizerKind::RAdam) => PARTIAL_THREAT_RADAM_CKPT_VERSION,
        (true, _) => PARTIAL_THREAT_CKPT_VERSION,
    };
    if let Some(total) = meta.lr_total_steps {
        if total == 0 {
            return Err(invalid("LR total steps must be positive".into()));
        }
        w.write_all(&STEP_SCHEDULE_CKPT_VERSION.to_le_bytes())?;
        w.write_all(&version.to_le_bytes())?;
        w.write_all(&total.to_le_bytes())?;
    } else {
        w.write_all(&version.to_le_bytes())?;
    }
    // feature set header (v2+): canonical 名 + 次元 3 値。
    let fs_name = arch.feature_set.canonical_name();
    w.write_all(&(fs_name.len() as u32).to_le_bytes())?;
    w.write_all(fs_name.as_bytes())?;
    w.write_all(&(arch.feature_set.train_ft_in() as u64).to_le_bytes())?;
    w.write_all(&arch.ft_out.to_le_bytes())?;
    w.write_all(&(arch.feature_set.max_active() as u64).to_le_bytes())?;
    // FT factorizer flag (v6+)。
    w.write_all(&[arch.feature_set.ft_factorize() as u8])?;
    // feature mapping hash (v7+)。
    w.write_all(&arch.feature_set.feature_hash().to_le_bytes())?;
    if partial {
        w.write_all(&threat_width.to_le_bytes())?;
    }
    // producer run id (v3+)。
    w.write_all(&(meta.run_id.len() as u32).to_le_bytes())?;
    w.write_all(meta.run_id.as_bytes())?;
    // arch_kind + topology header (v4+)。
    let arch_name = arch.arch_kind.canonical_name();
    w.write_all(&(arch_name.len() as u32).to_le_bytes())?;
    w.write_all(arch_name.as_bytes())?;
    // bucket mode (v8+)。bucket 概念を持たないアーキは空文字列。
    let bucket_mode = arch.bucket_mode.unwrap_or("");
    w.write_all(&(bucket_mode.len() as u32).to_le_bytes())?;
    w.write_all(bucket_mode.as_bytes())?;
    w.write_all(&(arch.topology.len() as u64).to_le_bytes())?;
    for &dim in arch.topology {
        w.write_all(&dim.to_le_bytes())?;
    }
    w.write_all(&(meta.superbatch as u64).to_le_bytes())?;
    w.write_all(&meta.step_count.to_le_bytes())?;
    // LR-schedule horizon (v5+)。0 = horizon 無し (step / constant / drop)。
    w.write_all(&(meta.lr_horizon.unwrap_or(0) as u64).to_le_bytes())?;
    // Simple evaluation scale (v9+)。0 = 未記録。
    w.write_all(&meta.fv_scale.unwrap_or(0).to_le_bytes())?;
    let qat_flags = u32::from(meta.qat_dense);
    w.write_all(&qat_flags.to_le_bytes())?;
    // producer net id + recipe fingerprint (v11+)。過長な net id は lineage の判定材料に
    // すぎないので、埋め込みを省略して保存は続ける。
    let net_id = if meta.net_id.len() > MAX_RUN_ID_BYTES {
        ""
    } else {
        meta.net_id
    };
    w.write_all(&(net_id.len() as u32).to_le_bytes())?;
    w.write_all(net_id.as_bytes())?;
    w.write_all(&meta.recipe_fingerprint.unwrap_or(0).to_le_bytes())?;
    w.write_all(&num_groups.to_le_bytes())?;
    Ok(())
}

pub const TENSORS_PER_GROUP: usize = 4;

/// Write one master weight group followed by its first moment, second moment and slow weights.
pub fn write_raw_ckpt_group<W: Write>(
    w: &mut W,
    tensors: [&[f32]; TENSORS_PER_GROUP],
) -> io::Result<()> {
    let len = tensors[0].len();
    if tensors.iter().any(|tensor| tensor.len() != len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "raw checkpoint weight and optimizer tensor lengths differ",
        ));
    }
    w.write_all(&(len as u64).to_le_bytes())?;
    for tensor in tensors {
        write_f32_slice(w, tensor)?;
    }
    Ok(())
}

/// f32 slice を little-endian で `w` に書き出す (`bytemuck` 不使用、依存を増やさない)。
pub fn write_f32_slice<W: std::io::Write>(w: &mut W, data: &[f32]) -> std::io::Result<()> {
    let mut bytes = vec![0_u8; data.len().min(F32_IO_CHUNK_BYTES / 4) * 4];
    for chunk in data.chunks(F32_IO_CHUNK_BYTES / 4) {
        for (&value, dest) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            dest.copy_from_slice(&value.to_le_bytes());
        }
        w.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

pub const F32_IO_CHUNK_BYTES: usize = 4 * 1024 * 1024;
