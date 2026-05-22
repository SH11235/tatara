use std::io::Write;

use gpu_runtime::DeviceBuffer;
use nnue_format::ArchKind;
use shogi_features::{FeatureSet, FeatureSetSpec};

use crate::arch::*;

// ===========================================================================
// raw checkpoint format (`--resume` 用)
// ===========================================================================

/// raw checkpoint format magic (`b"RNRC"` = "RShogi Nnue Resume Checkpoint")。
/// `crates/nnue-train::optimizer` の `b"RNGR"` (RangerHostState single-file format) とは
/// 別物 — こちらは weight group raw f32 + Ranger state + step + superbatch を 1 file に
/// まとめた self-contained format (`RNGR` は optimizer state だけ、weight は持たない)。
pub(crate) const RAW_CKPT_MAGIC: [u8; 4] = *b"RNRC";

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
///
/// `load_raw_checkpoint` accepts versions 1..=4. Version 1 is interpreted as
/// `halfka-hm-merged`; versions 1..=3 predate the arch-kind header and are
/// interpreted as `layerstack`. Versions above 4 are rejected. The producer
/// run id is absent (`None`) for versions 1 and 2.
pub(crate) const RAW_CKPT_VERSION: u32 = 4;

/// `*.ckpt` の producer run id のバイト数上限。run id は `{net_id}-{時刻}-{pid}`
/// 程度で高々数十バイト。破損 file の巨大な length 値で過大確保しないための上限。
pub(crate) const MAX_RUN_ID_BYTES: usize = 256;

/// raw checkpoint 1 group 分の host buffer (`w`, `m`, `v`, `slow` の f32 Vec、`grad` は含めない)。
pub(crate) type RawCkptGroup = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// `SimpleGpuTrainer::raw_ckpt_groups` の 1 要素。weight name + element count + 各
/// `(weight, m, v, slow)` device buffer の借用 tuple。
pub(crate) type SimpleRawCkptGroupEntry<'a> = (
    &'static str,
    usize,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
);

/// LayerStack アーキの topology header (v4+): FT 出力次元・L1 出力次元・L2 出力次元・
/// bucket 数。`load_raw_checkpoint` がこの並びを checkpoint と照合する。FT 出力次元は
/// `--ft-out`、L1 出力次元は `--l1`、L2 出力次元は `--l2` で可変、bucket 数は固定。
pub(crate) const fn layerstack_topology(ft_out: usize, l1_out: usize, l2_out: usize) -> [u64; 4] {
    [
        ft_out as u64,
        l1_out as u64,
        l2_out as u64,
        NUM_BUCKETS as u64,
    ]
}

/// raw checkpoint header の arch identity 部 (write / read 双方の引数)。feature
/// set・arch 種別・FT 出力次元・topology 次元列をまとめて持つ。
pub(crate) struct RawCkptArch<'a> {
    /// 入力 feature set (canonical 名 / `ft_in` / `max_active` の源)。
    pub(crate) feature_set: FeatureSetSpec,
    /// network アーキ種別。
    pub(crate) arch_kind: ArchKind,
    /// FT 出力次元 (feature set header の `ft_out` 欄に書く値)。
    pub(crate) ft_out: u64,
    /// arch 固有の層次元列 (v4 topology header)。
    pub(crate) topology: &'a [u64],
}

/// `read_raw_ckpt_header` が返す raw checkpoint header の解析結果。
#[derive(Debug)]
pub(crate) struct RawCkptHeader {
    /// この checkpoint が表す完了 superbatch 番号。
    pub(crate) superbatch: usize,
    /// Ranger lookahead step counter。
    pub(crate) step_count: u64,
    /// format 記載の weight group 数 (caller が arch 期待値と照合する)。
    pub(crate) num_groups: u64,
    /// producer run の experiment.json id (version 3+ かつ記録ありなら `Some`)。
    pub(crate) producer_run_id: Option<String>,
}

/// raw checkpoint の header (magic 〜 num_groups、group 本体の手前まで) を書く。
/// 常に最新 [`RAW_CKPT_VERSION`] で書き出す。
pub(crate) fn write_raw_ckpt_header<W: Write>(
    w: &mut W,
    arch: &RawCkptArch,
    run_id: &str,
    superbatch: u64,
    step_count: u64,
    num_groups: u64,
) -> std::io::Result<()> {
    w.write_all(&RAW_CKPT_MAGIC)?;
    w.write_all(&RAW_CKPT_VERSION.to_le_bytes())?;
    // feature set header (v2+): canonical 名 + 次元 3 値。
    let fs_name = arch.feature_set.canonical_name();
    w.write_all(&(fs_name.len() as u32).to_le_bytes())?;
    w.write_all(fs_name.as_bytes())?;
    w.write_all(&(arch.feature_set.ft_in() as u64).to_le_bytes())?;
    w.write_all(&arch.ft_out.to_le_bytes())?;
    w.write_all(&(arch.feature_set.max_active() as u64).to_le_bytes())?;
    // producer run id (v3+)。
    w.write_all(&(run_id.len() as u32).to_le_bytes())?;
    w.write_all(run_id.as_bytes())?;
    // arch_kind + topology header (v4+)。
    let arch_name = arch.arch_kind.canonical_name();
    w.write_all(&(arch_name.len() as u32).to_le_bytes())?;
    w.write_all(arch_name.as_bytes())?;
    w.write_all(&(arch.topology.len() as u64).to_le_bytes())?;
    for &dim in arch.topology {
        w.write_all(&dim.to_le_bytes())?;
    }
    w.write_all(&superbatch.to_le_bytes())?;
    w.write_all(&step_count.to_le_bytes())?;
    w.write_all(&num_groups.to_le_bytes())?;
    Ok(())
}

/// raw checkpoint の header を読み、`expected` の arch identity と照合する。
/// version 1..=4 を受理し、不一致 / 破損は `InvalidData` で reject する。
///
/// version 1..=3 は arch-kind header を持たず暗黙に `layerstack`。version 4 は
/// arch_kind 名と topology 次元列を `expected` と照合する。
pub(crate) fn read_raw_ckpt_header<R: std::io::Read>(
    r: &mut R,
    expected: &RawCkptArch,
) -> Result<RawCkptHeader, Box<dyn std::error::Error>> {
    let mut magic = [0u8; 4];
    read_exact_or_invalid(r, &mut magic, "magic")?;
    if magic != RAW_CKPT_MAGIC {
        return Err(invalid_data(format!(
            "raw checkpoint magic mismatch: got {magic:?}, want {RAW_CKPT_MAGIC:?}"
        )));
    }
    let mut buf4 = [0u8; 4];
    read_exact_or_invalid(r, &mut buf4, "version")?;
    let version = u32::from_le_bytes(buf4);
    if version == 0 || version > RAW_CKPT_VERSION {
        return Err(invalid_data(format!(
            "raw checkpoint version {version} is not supported \
             (this build reads 1..={RAW_CKPT_VERSION})"
        )));
    }
    let mut buf8 = [0u8; 8];
    let want_name = expected.feature_set.canonical_name();

    // feature set header は version 2+。version 1 は header 無しで halfka-hm-merged 固定。
    if version >= 2 {
        read_exact_or_invalid(r, &mut buf4, "feature set name length")?;
        let fs_name_len = u32::from_le_bytes(buf4) as usize;
        if fs_name_len > 256 {
            return Err(invalid_data(format!(
                "raw checkpoint feature set name length {fs_name_len} is implausible (max 256)"
            )));
        }
        let mut fs_name_bytes = vec![0u8; fs_name_len];
        read_exact_or_invalid(r, &mut fs_name_bytes, "feature set name")?;
        let fs_name = String::from_utf8(fs_name_bytes).map_err(|_| {
            invalid_data("raw checkpoint feature set name is not valid UTF-8".to_string())
        })?;
        read_exact_or_invalid(r, &mut buf8, "ft_in")?;
        let ckpt_ft_in = u64::from_le_bytes(buf8);
        read_exact_or_invalid(r, &mut buf8, "ft_out")?;
        let ckpt_ft_out = u64::from_le_bytes(buf8);
        read_exact_or_invalid(r, &mut buf8, "max_active")?;
        let ckpt_max_active = u64::from_le_bytes(buf8);

        let want = expected.feature_set;
        if fs_name != want_name {
            return Err(invalid_data(format!(
                "raw checkpoint feature set mismatch: checkpoint is '{fs_name}', \
                 requested '{want_name}' (cannot resume across feature sets)"
            )));
        }
        if ckpt_ft_in != want.ft_in() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint ft_in mismatch: got {ckpt_ft_in}, want {}",
                want.ft_in()
            )));
        }
        if ckpt_ft_out != expected.ft_out {
            return Err(invalid_data(format!(
                "raw checkpoint ft_out mismatch: got {ckpt_ft_out}, want {}",
                expected.ft_out
            )));
        }
        if ckpt_max_active != want.max_active() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint max_active mismatch: got {ckpt_max_active}, want {}",
                want.max_active()
            )));
        }
    } else if want_name != FeatureSet::HalfKaHmMerged.spec().canonical_name() {
        return Err(invalid_data(format!(
            "raw checkpoint version 1 is always 'halfka-hm-merged', \
             requested '{want_name}' (cannot resume across feature sets)"
        )));
    }

    // producer run id は version 3+。長さ 0 も「未記録」扱いで `None`。
    let producer_run_id: Option<String> = if version >= 3 {
        read_exact_or_invalid(r, &mut buf4, "producer run id length")?;
        let run_id_len = u32::from_le_bytes(buf4) as usize;
        if run_id_len > MAX_RUN_ID_BYTES {
            return Err(invalid_data(format!(
                "raw checkpoint producer run id length {run_id_len} is implausible \
                 (max {MAX_RUN_ID_BYTES})"
            )));
        }
        if run_id_len == 0 {
            None
        } else {
            let mut run_id_bytes = vec![0u8; run_id_len];
            read_exact_or_invalid(r, &mut run_id_bytes, "producer run id")?;
            Some(String::from_utf8(run_id_bytes).map_err(|_| {
                invalid_data("raw checkpoint producer run id is not valid UTF-8".to_string())
            })?)
        }
    } else {
        None
    };

    // arch_kind + topology header は version 4+。version 1..=3 は arch-kind header
    // を持たず、Simple アーキが存在しなかった時代の checkpoint なので暗黙に layerstack。
    if version >= 4 {
        read_exact_or_invalid(r, &mut buf4, "arch kind name length")?;
        let arch_name_len = u32::from_le_bytes(buf4) as usize;
        if arch_name_len > 256 {
            return Err(invalid_data(format!(
                "raw checkpoint arch kind name length {arch_name_len} is implausible (max 256)"
            )));
        }
        let mut arch_name_bytes = vec![0u8; arch_name_len];
        read_exact_or_invalid(r, &mut arch_name_bytes, "arch kind name")?;
        let arch_name = String::from_utf8(arch_name_bytes).map_err(|_| {
            invalid_data("raw checkpoint arch kind name is not valid UTF-8".to_string())
        })?;
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
        read_exact_or_invalid(r, &mut buf8, "topology dim count")?;
        let topo_count = u64::from_le_bytes(buf8);
        if topo_count != expected.topology.len() as u64 {
            return Err(invalid_data(format!(
                "raw checkpoint topology dim count {topo_count} != expected {}",
                expected.topology.len()
            )));
        }
        for (i, &want_dim) in expected.topology.iter().enumerate() {
            read_exact_or_invalid(r, &mut buf8, "topology dim")?;
            let got = u64::from_le_bytes(buf8);
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
    }

    read_exact_or_invalid(r, &mut buf8, "superbatch")?;
    let superbatch_u64 = u64::from_le_bytes(buf8);
    let superbatch: usize = superbatch_u64.try_into().map_err(|_| {
        invalid_data(format!(
            "raw checkpoint superbatch {superbatch_u64} exceeds usize::MAX"
        ))
    })?;
    read_exact_or_invalid(r, &mut buf8, "step_count")?;
    let step_count = u64::from_le_bytes(buf8);
    read_exact_or_invalid(r, &mut buf8, "num_groups")?;
    let num_groups = u64::from_le_bytes(buf8);

    Ok(RawCkptHeader {
        superbatch,
        step_count,
        num_groups,
        producer_run_id,
    })
}

/// `io::ErrorKind::InvalidData` の `Box<dyn Error>` を作る短縮 helper (raw checkpoint
/// の magic/version/dim 検証で使う、`RangerHostState::load_from_reader` と同方針)。
pub(crate) fn invalid_data(msg: String) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
}

/// f32 slice を little-endian で `w` に書き出す (`bytemuck` 不使用、依存を増やさない)。
pub(crate) fn write_f32_slice<W: std::io::Write>(w: &mut W, data: &[f32]) -> std::io::Result<()> {
    // 4 byte ずつの write_all は遅いので、一旦 byte Vec に詰めてから 1 回で書く
    // (`raw_ckpt_groups` 最大 113M f32 = ~450MB、呼び出し側は BufWriter で wrap 済だが
    //  chunk write の方が更に system call が減る)。
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&bytes)
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
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("raw checkpoint truncated while reading {what}"),
            )
        } else {
            e
        }
    })
}

/// little-endian f32 を `n` 個読む (`RangerHostState::load_from_reader` の `read_f32_vec`
/// と同型だが本 module 内ローカル版、`io::Result` を返す)。`what` は短読み (破損 file) 時の
/// context message に使う (`UnexpectedEof` → `InvalidData` に正規化、`read_exact_or_invalid` 経由)。
pub(crate) fn read_f32_vec_io<R: std::io::Read>(
    r: &mut R,
    n: usize,
    what: &str,
) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![
        0u8;
        n.checked_mul(4).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("f32 vec len {n} overflows byte count"),
            )
        })?
    ];
    read_exact_or_invalid(r, &mut bytes, what)?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
