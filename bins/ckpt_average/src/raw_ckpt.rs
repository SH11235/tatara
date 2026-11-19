//! LayerStack raw checkpoint (`*.ckpt`, magic `RNRC`) の読み出しと、複数 checkpoint の
//! master weight (`w`) の加重平均。
//!
//! file layout の正本は trainer (`nnue-train`) の raw checkpoint writer。本 module は
//! header を読み、各 group の `w` tensor の file offset だけを控えて `m` / `v` / `slow`
//! (optimizer state) は読み飛ばす。

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::coalesce_ft_factorized;
use nnue_train::dataloader::BucketMode;
use nnue_train::raw_checkpoint::{
    PARTIAL_THREAT_CKPT_VERSION, PARTIAL_THREAT_RADAM_CKPT_VERSION, RAW_CKPT_MAGIC,
};
use shogi_features::{
    BucketSpec, EffectBucketConfig, FeatureSet, FeatureSetSpec, FtFactorizeMode, ThreatProfile,
};
/// arch kind と topology を header に持つ最初の version。
const MIN_VERSION: u32 = 4;
const MAX_VERSION: u32 = PARTIAL_THREAT_RADAM_CKPT_VERSION;
const MAX_NAME_BYTES: usize = 256;
/// 各 group の本体は `len u64` に続く `w`, `m`, `v`, `slow` の 4 本の f32 tensor。
const TENSORS_PER_GROUP: u64 = 4;
const READ_CHUNK_F32: usize = 1 << 20;

/// 平均を取る checkpoint 同士で一致していなければならない header 項目。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CkptIdentity {
    pub(crate) feature_set_name: String,
    pub(crate) train_ft_in: u64,
    pub(crate) ft_out: u64,
    pub(crate) max_active: u64,
    pub(crate) ft_factorize: bool,
    pub(crate) feature_hash: Option<u32>,
    pub(crate) threat_width: Option<u64>,
    pub(crate) arch_kind: String,
    pub(crate) bucket_mode: String,
    pub(crate) topology: Vec<u64>,
    pub(crate) qat_dense: bool,
    pub(crate) ft_weight_i8x2: bool,
    pub(crate) group_lens: Vec<u64>,
}

/// The weight layout and routing must agree across checkpoints, including
/// checkpoints that use different versions or optimizer states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompatKey {
    train_spec: FeatureSetSpec,
    bucket_mode: String,
    topology: Vec<u64>,
    qat_dense: bool,
    ft_weight_i8x2: bool,
    shared_delta: bool,
}

impl CkptIdentity {
    pub(crate) fn compat_key(&self, layout: &LayerStackLayout) -> CompatKey {
        CompatKey {
            train_spec: layout.train_spec,
            bucket_mode: layout.bucket_mode.identity(),
            topology: self.topology.clone(),
            qat_dense: self.qat_dense,
            ft_weight_i8x2: self.ft_weight_i8x2,
            shared_delta: layout.shared_delta,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CkptInfo {
    pub(crate) version: u32,
    pub(crate) identity: CkptIdentity,
    pub(crate) superbatch: u64,
    pub(crate) step_count: u64,
    /// 各 group の `w` tensor 先頭の file offset。
    w_offsets: Vec<u64>,
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_array<const N: usize>(r: &mut impl Read, what: &str) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            invalid(format!("checkpoint truncated while reading {what}"))
        } else {
            e
        }
    })?;
    Ok(buf)
}

fn read_u32(r: &mut impl Read, what: &str) -> io::Result<u32> {
    read_array::<4>(r, what).map(u32::from_le_bytes)
}

fn read_u64(r: &mut impl Read, what: &str) -> io::Result<u64> {
    read_array::<8>(r, what).map(u64::from_le_bytes)
}

fn read_name(r: &mut impl Read, what: &str) -> io::Result<String> {
    let len = read_u32(r, what)? as usize;
    if len > MAX_NAME_BYTES {
        return Err(invalid(format!("{what} length {len} is implausible")));
    }
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)
        .map_err(|_| invalid(format!("checkpoint truncated while reading {what}")))?;
    String::from_utf8(bytes).map_err(|_| invalid(format!("{what} is not valid UTF-8")))
}

/// header と group 表を読む。tensor 本体は読まない。
pub(crate) fn read_ckpt_info(path: &Path) -> io::Result<CkptInfo> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::new(file);
    if read_array::<4>(&mut r, "magic")? != RAW_CKPT_MAGIC {
        return Err(invalid("not a raw training checkpoint (magic mismatch)"));
    }
    let version = read_u32(&mut r, "version")?;
    if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(invalid(format!(
            "checkpoint version {version} is not supported (supported: {MIN_VERSION}..={MAX_VERSION})"
        )));
    }
    let feature_set_name = read_name(&mut r, "feature set name")?;
    let train_ft_in = read_u64(&mut r, "ft_in")?;
    let ft_out = read_u64(&mut r, "ft_out")?;
    let max_active = read_u64(&mut r, "max_active")?;
    let ft_factorize = version >= 6 && read_array::<1>(&mut r, "ft_factorize flag")?[0] != 0;
    let feature_hash = if version >= 7 {
        Some(read_u32(&mut r, "feature hash")?)
    } else {
        None
    };
    let threat_width = if version >= PARTIAL_THREAT_CKPT_VERSION {
        Some(read_u64(&mut r, "Threat width")?)
    } else {
        None
    };
    read_name(&mut r, "producer run id")?;
    let arch_kind = read_name(&mut r, "arch kind")?;
    let bucket_mode = if version >= 8 {
        read_name(&mut r, "bucket mode")?
    } else {
        String::new()
    };
    let topo_count = read_u64(&mut r, "topology dim count")?;
    if topo_count > 16 {
        return Err(invalid(format!(
            "topology dim count {topo_count} is implausible"
        )));
    }
    let topology = (0..topo_count)
        .map(|_| read_u64(&mut r, "topology dim"))
        .collect::<io::Result<Vec<_>>>()?;
    let superbatch = read_u64(&mut r, "superbatch")?;
    let step_count = read_u64(&mut r, "step count")?;
    if version >= 5 {
        read_u64(&mut r, "LR horizon")?;
    }
    if version >= 9 {
        read_u32(&mut r, "fv_scale")?;
    }
    let (qat_dense, ft_weight_i8x2) = if version >= 10 {
        let flags = read_u32(&mut r, "QAT flags")?;
        nnue_train::raw_checkpoint::decode_qat_flags(flags)
            .ok_or_else(|| invalid(format!("unknown QAT flags {flags}")))?
    } else {
        (false, false)
    };
    if version >= 11 {
        read_name(&mut r, "producer net id")?;
        read_u64(&mut r, "recipe fingerprint")?;
    }
    let num_groups = read_u64(&mut r, "group count")?;
    if num_groups > 64 {
        return Err(invalid(format!("group count {num_groups} is implausible")));
    }
    let mut group_lens = Vec::with_capacity(num_groups as usize);
    let mut w_offsets = Vec::with_capacity(num_groups as usize);
    for g in 0..num_groups {
        let len = read_u64(&mut r, "group length")?;
        let offset = r.stream_position()?;
        let body = len
            .checked_mul(4 * TENSORS_PER_GROUP)
            .and_then(|b| b.checked_add(offset))
            .filter(|&end| end <= file_len)
            .ok_or_else(|| invalid(format!("checkpoint truncated inside group {g}")))?;
        group_lens.push(len);
        w_offsets.push(offset);
        r.seek(SeekFrom::Start(body))?;
    }
    Ok(CkptInfo {
        version,
        identity: CkptIdentity {
            feature_set_name,
            train_ft_in,
            ft_out,
            max_active,
            ft_factorize,
            feature_hash,
            threat_width,
            arch_kind,
            bucket_mode,
            topology,
            qat_dense,
            ft_weight_i8x2,
            group_lens,
        },
        superbatch,
        step_count,
        w_offsets,
    })
}

/// LayerStack checkpoint の層次元と、学習時 (factorizer 込み) の feature set spec。
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerStackLayout {
    pub(crate) train_spec: FeatureSetSpec,
    pub(crate) ft_out: usize,
    pub(crate) l1_out: usize,
    pub(crate) l2_out: usize,
    pub(crate) num_buckets: usize,
    bucket_mode: BucketMode,
    pub(crate) with_psqt: bool,
    shared_delta: bool,
}

impl LayerStackLayout {
    fn bucket_spec(&self) -> Option<BucketSpec> {
        match self.bucket_mode {
            BucketMode::Spec(spec) => Some(spec),
            _ => None,
        }
    }

    fn group_lens(&self) -> io::Result<Vec<(&'static str, usize)>> {
        let (ft_out, l1, l2, nb) = (self.ft_out, self.l1_out, self.l2_out, self.num_buckets);
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("checkpoint tensor dimensions overflow"))
        };
        let l2_in = product(l1 - 1, 2)?;
        let l1_w = product(ft_out, l1)?;
        let l2_w = product(l2_in, l2)?;
        // Full-width rows bound the packed FT size, including virtual rows.
        let train_ft_in = self.train_spec.train_ft_in();
        product(train_ft_in, ft_out)?;
        let mut groups = vec![
            ("ft_w", self.train_spec.train_ft_weight_len(ft_out)),
            ("ft_b", ft_out),
            ("l1_w", product(nb, l1_w)?),
            ("l1_b", product(nb, l1)?),
            ("l1_shared_weight", l1_w),
            ("l1_shared_bias", l1),
            ("l2_w", product(nb, l2_w)?),
            ("l2_b", product(nb, l2)?),
            ("l3_w", product(nb, l2)?),
            ("l3_b", nb),
        ];
        if self.shared_delta {
            groups.extend([
                ("l2_shared_weight", l2_w),
                ("l2_shared_bias", l2),
                ("l3_shared_weight", l2),
                ("l3_shared_bias", 1),
            ]);
        }
        if self.with_psqt {
            groups.push(("psqt_w", product(train_ft_in, nb)?));
        }
        Ok(groups)
    }

    /// export される `.bin` の feature set (factorizer の仮想行は畳み込まれて消える)。
    pub(crate) fn export_spec(&self) -> FeatureSetSpec {
        self.train_spec.without_ft_factorize()
    }
}

fn training_feature_spec(id: &CkptIdentity, ft_out: usize) -> io::Result<FeatureSetSpec> {
    let base = FeatureSet::from_canonical_name(&id.feature_set_name)
        .ok_or_else(|| invalid(format!("unknown feature set '{}'", id.feature_set_name)))?
        .spec();
    let width = if let Some(width) = id.threat_width {
        if width == 0 || !width.is_multiple_of(2) || width >= ft_out as u64 {
            return Err(invalid(format!(
                "compact Threat width {width} must be positive, even and less than FT width {ft_out}"
            )));
        }
        Some(width as usize)
    } else {
        None
    };
    let threats = ThreatProfile::ALL.map(|profile| {
        let spec = base.with_threat_profile(profile);
        width.map_or(spec, |width| spec.with_threat_width(width))
    });
    let effect_buckets = [
        EffectBucketConfig::KINGFIXED_2X2,
        EffectBucketConfig::KINGBUCKETED_2X2,
        EffectBucketConfig::KINGFIXED_3X3,
        EffectBucketConfig::KINGBUCKETED_3X3,
    ]
    .map(|config| base.with_effect_bucket_config(config));
    std::iter::once(base)
        .chain(threats)
        .chain(effect_buckets)
        .filter(|spec| match id.feature_hash {
            Some(hash) => spec.feature_hash() == hash,
            None => *spec == base,
        })
        .filter(|spec| width.is_none() || spec.threat_profile().is_some())
        .find_map(|spec| {
            let shared = if id.ft_factorize {
                spec.with_ft_factorize()
            } else {
                spec
            };
            let modes = if id.ft_factorize && spec.effect_bucket_config().is_some() {
                // The training row count distinguishes the two effect-bucket factorizer modes.
                [
                    Some(shared),
                    Some(spec.with_ft_factorize_mode(FtFactorizeMode::PerEffectBucket)),
                ]
            } else {
                [Some(shared), None]
            };
            modes.into_iter().flatten().find(|spec| {
                let max_active = spec.max_active() as u64;
                id.train_ft_in == spec.train_ft_in() as u64
                    && (id.max_active == max_active
                        || (id.ft_factorize && id.max_active == 2 * max_active))
            })
        })
        .ok_or_else(|| {
            invalid(format!(
                "feature hash or training dimensions do not match a '{}' feature layout",
                id.feature_set_name
            ))
        })
}

/// Restore the LayerStack layout from the header and validate the weight group lengths.
pub(crate) fn layerstack_layout(id: &CkptIdentity) -> io::Result<LayerStackLayout> {
    if id.arch_kind != "layerstack" {
        return Err(invalid(format!(
            "only LayerStack checkpoints are supported (got '{}')",
            id.arch_kind
        )));
    }
    let &[ft_out, l1_out, l2_out, num_buckets, ref psqt_dims @ ..] = id.topology.as_slice() else {
        return Err(invalid("invalid LayerStack topology"));
    };
    let with_psqt = match psqt_dims {
        [] => false,
        &[psqt_buckets] if psqt_buckets == num_buckets => true,
        _ => return Err(invalid("invalid LayerStack PSQT topology or bucket count")),
    };
    if ft_out == 0
        || !ft_out.is_multiple_of(2)
        || l1_out < 2
        || l2_out == 0
        || num_buckets == 0
        || ft_out != id.ft_out
    {
        return Err(invalid("checkpoint topology is inconsistent"));
    }
    let dimension = |value| {
        usize::try_from(value).map_err(|_| invalid("checkpoint dimension exceeds host limits"))
    };
    let bucket_mode = if id.bucket_mode.is_empty() {
        BucketMode::ProgressKpAbs
    } else {
        BucketMode::parse(&id.bucket_mode)
            .ok_or_else(|| invalid(format!("unknown bucket mode '{}'", id.bucket_mode)))?
            .0
    };
    let ft_out = dimension(ft_out)?;
    let train_spec = training_feature_spec(id, ft_out)?;
    if with_psqt && train_spec.threat_profile().is_some() {
        return Err(invalid(
            "LayerStack checkpoints with PSQT and Threat features are unsupported",
        ));
    }
    // Shared-delta has no topology field. Its four groups precede the optional PSQT group.
    let shared_delta = match id.group_lens.len().checked_sub(usize::from(with_psqt)) {
        Some(10) => false,
        Some(14) => true,
        _ => {
            return Err(invalid(format!(
                "found {} weight groups; expected {} or {} for LayerStack with this topology",
                id.group_lens.len(),
                10 + usize::from(with_psqt),
                14 + usize::from(with_psqt)
            )));
        }
    };
    let layout = LayerStackLayout {
        train_spec,
        ft_out,
        l1_out: dimension(l1_out)?,
        l2_out: dimension(l2_out)?,
        num_buckets: dimension(num_buckets)?,
        bucket_mode,
        with_psqt,
        shared_delta,
    };
    if layout
        .bucket_spec()
        .is_some_and(|s| s.num_buckets() != layout.num_buckets)
    {
        return Err(invalid("bucket spec does not match checkpoint topology"));
    }
    for ((name, want), &got) in layout.group_lens()?.into_iter().zip(&id.group_lens) {
        if want as u64 != got {
            return Err(invalid(format!(
                "group {name} has {got} elements, expected {want}"
            )));
        }
    }
    Ok(layout)
}

/// 一様平均の係数。
pub(crate) fn uniform_coefficients(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

/// 古い順に並んだ `n` 個の snapshot に `ema ← decay·ema + (1−decay)·x` を適用した結果を
/// 加重和で表した係数 (初期値は最古の snapshot)。係数の和は 1。
pub(crate) fn ema_coefficients(n: usize, decay: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let age = (n - 1 - k) as i32;
            if k == 0 {
                decay.powi(age)
            } else {
                (1.0 - decay) * decay.powi(age)
            }
        })
        .collect()
}

/// 各 checkpoint の `w` を `coefficients` で加重和する。group ごとに f64 で累積する。
pub(crate) fn weighted_average(
    layout: &LayerStackLayout,
    paths: &[&Path],
    infos: &[CkptInfo],
    coefficients: &[f64],
) -> io::Result<Vec<Vec<f32>>> {
    assert_eq!(paths.len(), infos.len());
    assert_eq!(paths.len(), coefficients.len());
    let mut files = paths
        .iter()
        .map(|p| File::open(p).map(|f| BufReader::with_capacity(8 << 20, f)))
        .collect::<io::Result<Vec<_>>>()?;
    let group_lens = layout.group_lens()?;
    let mut out = Vec::with_capacity(group_lens.len());
    let mut bytes = vec![0u8; READ_CHUNK_F32 * 4];
    for (g, (name, len)) in group_lens.into_iter().enumerate() {
        let mut acc = vec![0f64; len];
        for ((file, info), &c) in files.iter_mut().zip(infos).zip(coefficients) {
            file.seek(SeekFrom::Start(info.w_offsets[g]))?;
            let mut done = 0;
            while done < len {
                let n = (len - done).min(READ_CHUNK_F32);
                file.read_exact(&mut bytes[..n * 4])?;
                for (a, b) in acc[done..done + n].iter_mut().zip(bytes.chunks_exact(4)) {
                    let x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    if !x.is_finite() {
                        return Err(invalid(format!(
                            "non-finite weight in group {name} at superbatch {}",
                            info.superbatch
                        )));
                    }
                    *a += c * f64::from(x);
                }
                done += n;
            }
        }
        out.push(acc.into_iter().map(|a| a as f32).collect());
    }
    Ok(out)
}

/// 平均した group 列から export 用の weight を組む。FT factorizer の仮想行は trainer の
/// export と同じく実行へ畳み込む (畳み込みは線形なので平均との順序は結果に影響しない)。
pub(crate) fn to_layerstack_weights(
    layout: &LayerStackLayout,
    groups: Vec<Vec<f32>>,
) -> LayerStackWeights {
    let mut groups = groups.into_iter();
    let [
        ft_w,
        ft_b,
        l1_w,
        l1_b,
        l1_shared_weight,
        l1_shared_bias,
        mut l2_w,
        mut l2_b,
        mut l3_w,
        mut l3_b,
    ] = std::array::from_fn(|_| groups.next().expect("validated LayerStack weight group"));
    if layout.shared_delta {
        for bucketed in [&mut l2_w, &mut l2_b, &mut l3_w, &mut l3_b] {
            let shared = groups.next().expect("validated shared-delta weight group");
            for (value, shared_value) in bucketed.iter_mut().zip(shared.iter().cycle()) {
                *value += shared_value;
            }
        }
    }
    let psqt_w = groups
        .next()
        .map(|psqt| coalesce_ft_factorized(&layout.train_spec, layout.num_buckets, &psqt));
    LayerStackWeights {
        feature_set: layout.export_spec(),
        num_buckets: layout.num_buckets,
        bucket_spec: layout.bucket_spec(),
        ft_w: coalesce_ft_factorized(&layout.train_spec, layout.ft_out, &ft_w),
        ft_b,
        l1_w,
        l1_b,
        l1_shared_weight,
        l1_shared_bias,
        l2_w,
        l2_b,
        l3_w,
        l3_b,
        psqt_w,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    use nnue_format::ArchKind;
    use nnue_train::{
        optimizer::OptimizerKind,
        raw_checkpoint::{RawCkptArch, RawCkptMeta, write_raw_ckpt_group, write_raw_ckpt_header},
    };

    pub(crate) const FT_OUT: usize = 4;
    pub(crate) const L1: usize = 4;
    pub(crate) const L2: usize = 4;
    pub(crate) const NB: usize = 2;

    pub(crate) fn train_spec() -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged.spec().with_ft_factorize()
    }

    pub(crate) fn layout() -> LayerStackLayout {
        LayerStackLayout {
            train_spec: train_spec(),
            ft_out: FT_OUT,
            l1_out: L1,
            l2_out: L2,
            num_buckets: NB,
            bucket_mode: BucketMode::ProgressKpAbs,
            with_psqt: false,
            shared_delta: false,
        }
    }

    pub(crate) fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ckpt-average-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn put_name(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u32).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// v11 の factorized LayerStack checkpoint を書く。
    pub(crate) fn write_ckpt(
        path: &Path,
        superbatch: u64,
        topology: [u64; 4],
        weight: impl Fn(usize, usize) -> f32,
    ) {
        write_ckpt_versioned(
            path,
            11,
            train_spec(),
            superbatch,
            topology,
            "progresskpabs",
            weight,
        );
    }

    /// trainer の raw checkpoint writer が `version` で書く header の LayerStack checkpoint を
    /// 書く。`w` は `weight(group, index)`、optimizer state は平均に混入すれば結果が壊れる値
    /// (NaN / 巨大値) で埋める。
    pub(crate) fn write_ckpt_versioned(
        path: &Path,
        version: u32,
        spec: FeatureSetSpec,
        superbatch: u64,
        topology: [u64; 4],
        bucket_mode: &str,
        weight: impl Fn(usize, usize) -> f32,
    ) {
        let lay = LayerStackLayout {
            train_spec: spec,
            ft_out: topology[0] as usize,
            l1_out: topology[1] as usize,
            l2_out: topology[2] as usize,
            num_buckets: topology[3] as usize,
            bucket_mode: BucketMode::parse(bucket_mode).unwrap().0,
            with_psqt: false,
            shared_delta: false,
        };
        let mut out = Vec::new();
        out.extend(RAW_CKPT_MAGIC);
        out.extend(version.to_le_bytes());
        put_name(&mut out, spec.canonical_name());
        out.extend((spec.train_ft_in() as u64).to_le_bytes());
        out.extend(topology[0].to_le_bytes());
        out.extend((spec.max_active() as u64).to_le_bytes());
        if version >= 6 {
            out.push(u8::from(spec.ft_factorize()));
        }
        if version >= 7 {
            out.extend(spec.feature_hash().to_le_bytes());
        }
        put_name(&mut out, "test-run");
        put_name(&mut out, "layerstack");
        if version >= 8 {
            put_name(&mut out, bucket_mode);
        }
        out.extend(4u64.to_le_bytes());
        for d in topology {
            out.extend(d.to_le_bytes());
        }
        out.extend(superbatch.to_le_bytes());
        out.extend((superbatch * 10).to_le_bytes());
        if version >= 5 {
            out.extend(0u64.to_le_bytes());
        }
        if version >= 9 {
            out.extend(0i32.to_le_bytes());
        }
        if version >= 10 {
            out.extend(1u32.to_le_bytes());
        }
        if version >= 11 {
            put_name(&mut out, "test-net");
            out.extend(0u64.to_le_bytes());
        }
        out.extend(10u64.to_le_bytes());
        for (g, (_, len)) in lay.group_lens().unwrap().into_iter().enumerate() {
            out.extend((len as u64).to_le_bytes());
            for i in 0..len {
                out.extend(weight(g, i).to_le_bytes());
            }
            for poison in [f32::NAN, 1e30, f32::NAN] {
                for _ in 0..len {
                    out.extend(poison.to_le_bytes());
                }
            }
        }
        std::fs::File::create(path)
            .unwrap()
            .write_all(&out)
            .unwrap();
    }

    pub(crate) fn pattern(seed: f32) -> impl Fn(usize, usize) -> f32 {
        move |g, i| seed * (((g * 7919 + i) % 97) as f32 - 48.0) / 512.0
    }

    #[test]
    fn header_round_trips_identity_and_counters() {
        let path = temp_path("header.ckpt");
        write_ckpt(
            &path,
            40,
            [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64],
            pattern(1.0),
        );
        let info = read_ckpt_info(&path).unwrap();
        assert_eq!(info.version, 11);
        assert_eq!(info.superbatch, 40);
        assert_eq!(info.step_count, 400);
        assert!(info.identity.qat_dense);
        assert!(info.identity.ft_factorize);
        assert_eq!(info.identity.bucket_mode, "progresskpabs");
        let lay = layerstack_layout(&info.identity).unwrap();
        assert_eq!(lay.num_buckets, NB);
        assert_eq!(lay.export_spec(), FeatureSet::HalfKaHmMerged.spec());
    }

    #[test]
    fn average_reads_master_weights_only() {
        let a = temp_path("avg-a.ckpt");
        let b = temp_path("avg-b.ckpt");
        let topo = [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64];
        write_ckpt(&a, 20, topo, pattern(1.0));
        write_ckpt(&b, 40, topo, pattern(3.0));
        let infos = [read_ckpt_info(&a).unwrap(), read_ckpt_info(&b).unwrap()];
        let lay = layerstack_layout(&infos[0].identity).unwrap();
        let groups =
            weighted_average(&lay, &[a.as_path(), b.as_path()], &infos, &[0.25, 0.75]).unwrap();
        let expected = pattern(2.5);
        for (g, values) in groups.iter().enumerate() {
            assert_eq!(values.len() as u64, infos[0].identity.group_lens[g]);
            for (i, &v) in values.iter().enumerate() {
                assert!(
                    (v - expected(g, i)).abs() <= 1e-6,
                    "group {g} index {i}: {v}"
                );
            }
        }
    }

    #[test]
    fn rejects_truncated_checkpoint() {
        let path = temp_path("truncated.ckpt");
        write_ckpt(
            &path,
            20,
            [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64],
            pattern(1.0),
        );
        let len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 4)
            .unwrap();
        let err = read_ckpt_info(&path).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn rejects_unknown_feature_hash() {
        let path = temp_path("hash.ckpt");
        write_ckpt(
            &path,
            20,
            [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64],
            pattern(1.0),
        );
        let mut id = read_ckpt_info(&path).unwrap().identity;
        id.feature_hash = id.feature_hash.map(|h| h ^ 1);
        assert!(layerstack_layout(&id).is_err());
    }

    fn compat_key(id: &CkptIdentity) -> CompatKey {
        id.compat_key(&layerstack_layout(id).unwrap())
    }

    #[test]
    fn older_versions_share_the_compat_key_of_current_checkpoints() {
        let topo = [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64];
        let current = temp_path("compat-v11.ckpt");
        let older = temp_path("compat-v8.ckpt");
        let no_mode = temp_path("compat-v7.ckpt");
        write_ckpt(&current, 40, topo, pattern(1.0));
        write_ckpt_versioned(
            &older,
            8,
            train_spec(),
            20,
            topo,
            "progresskpabs",
            pattern(1.0),
        );
        write_ckpt_versioned(
            &no_mode,
            7,
            train_spec(),
            20,
            topo,
            "progresskpabs",
            pattern(1.0),
        );
        let v11 = read_ckpt_info(&current).unwrap().identity;
        let v8 = read_ckpt_info(&older).unwrap().identity;
        let v7 = read_ckpt_info(&no_mode).unwrap().identity;
        assert!(layerstack_layout(&v7).is_ok());
        assert_eq!(compat_key(&v7), compat_key(&v8));
        // QAT mode is recorded from v10 on, so a v8 checkpoint reads as QAT off.
        let mut v11_without_qat = v11.clone();
        v11_without_qat.qat_dense = false;
        assert_eq!(compat_key(&v8), compat_key(&v11_without_qat));
        let mut alias = v11.clone();
        alias.bucket_mode = "progress8kpabs".to_string();
        assert_eq!(compat_key(&alias), compat_key(&v11));
        let mut other_ft_grid = v11.clone();
        other_ft_grid.ft_weight_i8x2 = true;
        assert_ne!(compat_key(&other_ft_grid), compat_key(&v11));
        let mut other_mode = v11.clone();
        other_mode.bucket_mode = "progresskpabsq16".to_string();
        assert_ne!(compat_key(&other_mode), compat_key(&v11));
    }

    #[test]
    fn rejects_unsupported_versions() {
        let topo = [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64];
        for version in [MIN_VERSION - 1, MAX_VERSION + 1] {
            let path = temp_path(&format!("version-{version}.ckpt"));
            write_ckpt_versioned(
                &path,
                version,
                train_spec(),
                20,
                topo,
                "progresskpabs",
                pattern(1.0),
            );
            let err = read_ckpt_info(&path).unwrap_err();
            assert!(err.to_string().contains("not supported"), "{err}");
        }
    }

    #[test]
    fn non_factorized_checkpoint_exports_master_weights_unchanged() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let topo = [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64];
        let path = temp_path("plain.ckpt");
        write_ckpt_versioned(&path, 11, spec, 20, topo, "progresskpabs", pattern(1.0));
        let info = read_ckpt_info(&path).unwrap();
        assert!(!info.identity.ft_factorize);
        let lay = layerstack_layout(&info.identity).unwrap();
        assert_eq!(lay.export_spec(), spec);
        let groups = weighted_average(&lay, &[path.as_path()], &[info], &[1.0]).unwrap();
        let weights = to_layerstack_weights(&lay, groups);
        assert_eq!(weights.ft_w.len(), spec.train_ft_in() * FT_OUT);
        for (i, &v) in weights.ft_w.iter().enumerate() {
            assert_eq!(v, pattern(1.0)(0, i));
        }
    }

    #[test]
    fn ema_coefficients_match_recursive_update() {
        let xs = [1.0, 5.0, -2.0, 7.0, 3.0];
        let decay = 0.85;
        let mut ema = xs[0];
        for &x in &xs[1..] {
            ema = decay * ema + (1.0 - decay) * x;
        }
        let c = ema_coefficients(xs.len(), decay);
        let weighted: f64 = c.iter().zip(xs).map(|(c, x)| c * x).sum();
        assert!((weighted - ema).abs() < 1e-12);
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((uniform_coefficients(3).iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    fn write_training_checkpoint(
        path: &Path,
        spec: FeatureSetSpec,
        optimizer: OptimizerKind,
        scale: f32,
        shared_delta: bool,
        psqt: bool,
    ) {
        use nnue_format::layerstack_weights::{QA, QB};

        let dims = [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64, NB as u64];
        let topology = &dims[..if psqt { 5 } else { 4 }];
        let net = LayerStackWeights::zeroed(spec.without_ft_factorize(), FT_OUT, L1, L2, NB);
        let mut lengths = vec![
            spec.train_ft_weight_len(FT_OUT),
            net.ft_b.len(),
            net.l1_w.len(),
            net.l1_b.len(),
            net.l1_shared_weight.len(),
            net.l1_shared_bias.len(),
            net.l2_w.len(),
            net.l2_b.len(),
            net.l3_w.len(),
            net.l3_b.len(),
        ];
        if shared_delta {
            lengths.extend([
                net.l2_w.len() / NB,
                net.l2_b.len() / NB,
                net.l3_w.len() / NB,
                1,
            ]);
        }
        if psqt {
            lengths.push(spec.train_ft_in() * NB);
        }
        let arch = RawCkptArch {
            feature_set: spec,
            arch_kind: ArchKind::LayerStack,
            bucket_mode: Some("spec:ownking:2"),
            ft_out: topology[0],
            topology,
        };
        let meta = RawCkptMeta {
            optimizer,
            qat_dense: false,
            ft_weight_i8x2: false,
            run_id: "run",
            superbatch: scale as usize,
            step_count: 100,
            lr_horizon: None,
            fv_scale: None,
            net_id: "net",
            recipe_fingerprint: None,
        };
        let mut file = std::io::BufWriter::new(File::create(path).unwrap());
        write_raw_ckpt_header(&mut file, &arch, &meta, lengths.len() as u64).unwrap();
        for (group, &len) in lengths.iter().enumerate() {
            let mut weights = vec![0.0; len];
            if psqt && group == lengths.len() - 1 {
                weights[NB - 1] = scale / (QA * QB) as f32;
                if spec.ft_factorize() {
                    weights[spec.ft_in() * NB + NB - 1] = scale / (QA * QB) as f32;
                }
            } else {
                match group {
                    0 => {
                        weights[spec.ft_weight_len(FT_OUT) - 1] = scale / QA as f32;
                        if spec.ft_factorize() {
                            weights[spec.ft_weight_len(FT_OUT)] = scale / QA as f32;
                        }
                    }
                    1 => weights[0] = scale / QA as f32,
                    6 | 8 | 10 | 12 => weights[0] = (scale + group as f32) / QB as f32,
                    7 | 9 | 11 | 13 => weights[0] = (scale + group as f32) / (QA * QB) as f32,
                    _ => {}
                }
            }
            let state = vec![f32::NAN; len];
            write_raw_ckpt_group(&mut file, [&weights, &state, &state, &state]).unwrap();
        }
        file.flush().unwrap();
    }

    #[test]
    fn trainer_checkpoints_average_and_reload() {
        use OptimizerKind::{RAdam, Ranger, Sgd};
        use nnue_format::layerstack_weights::{QA, QB};

        let plain = FeatureSet::HalfKaHmMerged.spec();
        let threat = FeatureSet::HalfKp
            .spec()
            .with_threat_profile(ThreatProfile::CrossSide);
        let compact = threat.with_threat_width(2).with_ft_factorize();
        let effect = plain.with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2);
        for (name, spec, version, optimizer, shared_delta, psqt) in [
            ("plain", plain, 11, Ranger, false, false),
            (
                "threat-ranger",
                threat.with_ft_factorize(),
                11,
                Ranger,
                false,
                false,
            ),
            ("threat-sgd", threat, 12, Sgd, false, false),
            ("compact-ranger", compact, 13, Ranger, false, false),
            ("compact-sgd", compact, 14, Sgd, false, false),
            ("compact-radam", compact, 15, RAdam, false, false),
            ("compact-shared", compact, 15, RAdam, true, false),
            ("psqt", plain.with_ft_factorize(), 11, Ranger, false, true),
            (
                "psqt-shared",
                plain.with_ft_factorize(),
                11,
                Ranger,
                true,
                true,
            ),
            (
                "effect-pooled",
                effect.with_ft_factorize(),
                11,
                Ranger,
                false,
                false,
            ),
            (
                "effect-per-bucket",
                effect.with_ft_factorize_mode(FtFactorizeMode::PerEffectBucket),
                11,
                Ranger,
                false,
                false,
            ),
        ] {
            let paths = [1, 3].map(|scale| {
                let path = temp_path(&format!("{name}-{scale}.ckpt"));
                write_training_checkpoint(&path, spec, optimizer, scale as f32, shared_delta, psqt);
                let info = read_ckpt_info(&path).unwrap();
                assert_eq!(info.version, version, "{name}");
                assert_eq!(
                    info.identity.group_lens.len(),
                    10 + usize::from(psqt) + 4 * usize::from(shared_delta),
                    "{name}"
                );
                assert_eq!(
                    layerstack_layout(&info.identity).unwrap().train_spec,
                    spec,
                    "{name}"
                );
                path
            });

            let output = temp_path(&format!("{name}-average.bin"));
            let args = crate::Args {
                checkpoints: paths.to_vec(),
                output: output.clone(),
                mode: crate::Mode::Uniform,
                ema_decay: None,
                fv_scale: None,
                reference: None,
            };
            crate::run(&args).unwrap();
            let weights = LayerStackWeights::load_quantised_with_psqt(
                &mut BufReader::new(File::open(&output).unwrap()),
                spec.without_ft_factorize(),
                FT_OUT,
                L1,
                L2,
                NB,
                psqt,
            )
            .unwrap();
            assert_eq!(weights.bucket_spec.unwrap().to_string(), "ownking:2");
            let expected = 2.0 / QA as f32;
            assert_eq!(weights.ft_w.last(), Some(&expected), "{name}");
            assert_eq!(weights.ft_b[0], expected, "{name}");
            assert_eq!(
                weights.ft_w[0],
                if spec.ft_factorize() { expected } else { 0.0 },
                "{name}"
            );
            for (group, tensor, width, divisor) in [
                (6, &weights.l2_w, (L1 - 1) * 2 * L2, QB),
                (7, &weights.l2_b, L2, QA * QB),
                (8, &weights.l3_w, L2, QB),
                (9, &weights.l3_b, 1, QA * QB),
            ] {
                let shared = if shared_delta { 2 + group + 4 } else { 0 };
                assert_eq!(
                    tensor[0],
                    (2 + group + shared) as f32 / divisor as f32,
                    "{name} group {group}"
                );
                assert_eq!(
                    tensor[width],
                    shared as f32 / divisor as f32,
                    "{name} group {group}"
                );
            }
            if psqt {
                let psqt = weights.psqt_w.unwrap();
                assert_eq!(psqt[0], 0.0, "{name}");
                assert_eq!(psqt[NB - 1], 4.0 / (QA * QB) as f32, "{name}");
            }
            for path in paths.into_iter().chain([output]) {
                std::fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn rejects_invalid_layouts_and_distinguishes_bucket_routing() {
        let compact = FeatureSet::HalfKp
            .spec()
            .with_threat_profile(ThreatProfile::CrossSide)
            .with_threat_width(2);
        let path = temp_path("layout-validation.ckpt");
        write_training_checkpoint(&path, compact, OptimizerKind::RAdam, 1.0, true, false);
        let id = read_ckpt_info(&path).unwrap().identity;
        let mut other = id.clone();
        other.bucket_mode = "spec:oppking:2".into();
        assert_ne!(compat_key(&other), compat_key(&id));
        other.bucket_mode = "spec:handmajor3".into();
        assert!(
            layerstack_layout(&other)
                .unwrap_err()
                .to_string()
                .contains("bucket spec")
        );
        other = id.clone();
        other.threat_width = Some(FT_OUT as u64);
        assert!(
            layerstack_layout(&other)
                .unwrap_err()
                .to_string()
                .contains("compact Threat width")
        );
        other = id.clone();
        other.group_lens[10] += 1;
        assert!(
            layerstack_layout(&other)
                .unwrap_err()
                .to_string()
                .contains("group l2_shared_weight")
        );
        other.group_lens.pop();
        assert!(
            layerstack_layout(&other)
                .unwrap_err()
                .to_string()
                .contains("expected 10 or 14")
        );
        write_training_checkpoint(&path, compact, OptimizerKind::RAdam, 1.0, false, true);
        let invalid = read_ckpt_info(&path).unwrap().identity;
        assert!(
            layerstack_layout(&invalid)
                .unwrap_err()
                .to_string()
                .contains("with PSQT and Threat")
        );
        std::fs::remove_file(path).unwrap();
    }
}
