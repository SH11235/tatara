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
use shogi_features::{FeatureSet, FeatureSetSpec};

const MAGIC: [u8; 4] = *b"RNRC";
/// arch kind と topology を header に持つ最初の version。
const MIN_VERSION: u32 = 4;
const MAX_VERSION: u32 = 11;
const MAX_NAME_BYTES: usize = 256;
/// 各 group の本体は `len u64` に続く `w`, `m`, `v`, `slow` の 4 本の f32 tensor。
const TENSORS_PER_GROUP: u64 = 4;
const LAYERSTACK_GROUP_NAMES: [&str; 10] = [
    "ft_w",
    "ft_b",
    "l1_w",
    "l1_b",
    "l1_shared_weight",
    "l1_shared_bias",
    "l2_w",
    "l2_b",
    "l3_w",
    "l3_b",
];
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
    pub(crate) arch_kind: String,
    pub(crate) bucket_mode: String,
    pub(crate) topology: Vec<u64>,
    pub(crate) qat_dense: bool,
    pub(crate) group_lens: Vec<u64>,
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
    if read_array::<4>(&mut r, "magic")? != MAGIC {
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
    let qat_dense = if version >= 10 {
        match read_u32(&mut r, "QAT mode")? {
            0 => false,
            1 => true,
            other => return Err(invalid(format!("unknown QAT mode {other}"))),
        }
    } else {
        false
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
            arch_kind,
            bucket_mode,
            topology,
            qat_dense,
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
}

impl LayerStackLayout {
    fn group_lens(&self) -> [usize; 10] {
        let (ft_out, l1, l2, nb) = (self.ft_out, self.l1_out, self.l2_out, self.num_buckets);
        let l2_in = (l1 - 1) * 2;
        [
            self.train_spec.train_ft_in() * ft_out,
            ft_out,
            nb * l1 * ft_out,
            nb * l1,
            ft_out * l1,
            l1,
            nb * l2 * l2_in,
            nb * l2,
            nb * l2,
            nb,
        ]
    }

    /// export される `.bin` の feature set (factorizer の仮想行は畳み込まれて消える)。
    pub(crate) fn export_spec(&self) -> FeatureSetSpec {
        if self.train_spec.ft_factorize() {
            self.train_spec.feature_set().spec()
        } else {
            self.train_spec
        }
    }
}

/// header から LayerStack の層構成を復元し、group 長を照合する。threat / effect bucket /
/// PSQT / L2-L3 shared-delta の checkpoint は header だけでは spec を復元できないため拒否する。
pub(crate) fn layerstack_layout(id: &CkptIdentity) -> io::Result<LayerStackLayout> {
    if id.arch_kind != "layerstack" {
        return Err(invalid(format!(
            "only LayerStack checkpoints are supported (got '{}')",
            id.arch_kind
        )));
    }
    let &[ft_out, l1_out, l2_out, num_buckets] = id.topology.as_slice() else {
        return Err(invalid(
            "only LayerStack checkpoints without PSQT are supported (topology must have 4 dims)",
        ));
    };
    if id.group_lens.len() != LAYERSTACK_GROUP_NAMES.len() {
        return Err(invalid(format!(
            "expected {} weight groups, found {} (PSQT and L2/L3 shared-delta are not supported)",
            LAYERSTACK_GROUP_NAMES.len(),
            id.group_lens.len()
        )));
    }
    if l1_out < 2 || num_buckets == 0 || ft_out != id.ft_out {
        return Err(invalid("checkpoint topology is inconsistent"));
    }
    let base = FeatureSet::from_canonical_name(&id.feature_set_name)
        .ok_or_else(|| invalid(format!("unknown feature set '{}'", id.feature_set_name)))?
        .spec();
    let train_spec = if id.ft_factorize {
        base.with_ft_factorize()
    } else {
        base
    };
    let max_active_ok = id.max_active == train_spec.max_active() as u64
        || (id.ft_factorize && id.max_active == 2 * train_spec.max_active() as u64);
    if id.train_ft_in != train_spec.train_ft_in() as u64
        || id
            .feature_hash
            .is_some_and(|h| h != train_spec.feature_hash())
        || !max_active_ok
    {
        return Err(invalid(format!(
            "feature set '{}' with threat or effect-bucket features is not supported",
            id.feature_set_name
        )));
    }
    let layout = LayerStackLayout {
        train_spec,
        ft_out: ft_out as usize,
        l1_out: l1_out as usize,
        l2_out: l2_out as usize,
        num_buckets: num_buckets as usize,
    };
    for ((name, want), &got) in LAYERSTACK_GROUP_NAMES
        .iter()
        .zip(layout.group_lens())
        .zip(&id.group_lens)
    {
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
    let group_lens = &infos[0].identity.group_lens;
    let mut out = Vec::with_capacity(group_lens.len());
    let mut bytes = vec![0u8; READ_CHUNK_F32 * 4];
    for (g, &len) in group_lens.iter().enumerate() {
        let len = len as usize;
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
                            "non-finite weight in group {} at superbatch {}",
                            LAYERSTACK_GROUP_NAMES.get(g).unwrap_or(&"?"),
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
    let [
        ft_w,
        ft_b,
        l1_w,
        l1_b,
        l1_shared_weight,
        l1_shared_bias,
        l2_w,
        l2_b,
        l3_w,
        l3_b,
    ]: [Vec<f32>; 10] = groups
        .try_into()
        .expect("LayerStack checkpoints have 10 weight groups");
    LayerStackWeights {
        feature_set: layout.export_spec(),
        num_buckets: layout.num_buckets,
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
        psqt_w: None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

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

    /// v11 の LayerStack checkpoint を書く。`w` は `weight(group, index)`、optimizer state は
    /// 平均に混入すれば結果が壊れる値 (NaN / 巨大値) で埋める。
    pub(crate) fn write_ckpt(
        path: &Path,
        superbatch: u64,
        topology: [u64; 4],
        weight: impl Fn(usize, usize) -> f32,
    ) {
        let spec = train_spec();
        let lay = LayerStackLayout {
            train_spec: spec,
            ft_out: topology[0] as usize,
            l1_out: topology[1] as usize,
            l2_out: topology[2] as usize,
            num_buckets: topology[3] as usize,
        };
        let mut out = Vec::new();
        out.extend(MAGIC);
        out.extend(11u32.to_le_bytes());
        put_name(&mut out, spec.canonical_name());
        out.extend((spec.train_ft_in() as u64).to_le_bytes());
        out.extend(topology[0].to_le_bytes());
        out.extend((spec.max_active() as u64).to_le_bytes());
        out.push(1);
        out.extend(spec.feature_hash().to_le_bytes());
        put_name(&mut out, "test-run");
        put_name(&mut out, "layerstack");
        put_name(&mut out, "progresskpabs");
        out.extend(4u64.to_le_bytes());
        for d in topology {
            out.extend(d.to_le_bytes());
        }
        out.extend(superbatch.to_le_bytes());
        out.extend((superbatch * 10).to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend(0i32.to_le_bytes());
        out.extend(1u32.to_le_bytes());
        put_name(&mut out, "test-net");
        out.extend(0u64.to_le_bytes());
        out.extend(10u64.to_le_bytes());
        for (g, len) in lay.group_lens().into_iter().enumerate() {
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
        let groups = weighted_average(&[a.as_path(), b.as_path()], &infos, &[0.25, 0.75]).unwrap();
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
}
