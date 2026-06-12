use std::io::Write;
use std::path::Path;

use gpu_runtime::{CudaStream, DeviceBuffer};
use nnue_format::ArchKind;
use shogi_features::{FeatureSet, FeatureSetSpec};

use crate::arch::{FT_OPT_M_SCALE, FT_OPT_V_SCALE};
use crate::trainer_common::MomentBuf;

// ===========================================================================
// raw checkpoint format (`--resume` 用)
//
// layout (全 little-endian、現行 RAW_CKPT_VERSION = 6):
//
//   magic        b"RNRC"             (4 bytes)
//   version      u32 (6)             (4 bytes)
//   fs_name_len  u32                 (4 bytes、feature set canonical 名の長さ)
//   fs_name      UTF-8 [fs_name_len]  (feature set canonical 名、例 "halfka-hm-merged")
//   ft_in        u64                 (学習側 FT 入力次元、feature set 依存)
//   ft_out       u64                 (FT 出力次元、`--ft-out`)
//   max_active   u64                 (学習側の 1 perspective あたり active feature 数)
//   ft_factorize u8                  (v6+、FT factorizer 有効 flag)
//   run_id_len   u32                 (4 bytes、producer run id の長さ、0 可)
//   run_id       UTF-8 [run_id_len]   (この checkpoint を書いた run の experiment.json `id`)
//   arch_len     u32                 (4 bytes、arch kind canonical 名の長さ)
//   arch_kind    UTF-8 [arch_len]     (arch kind canonical 名、例 "layerstack")
//   topo_count   u64                 (topology 次元の個数)
//   topology     u64 [topo_count]     (arch 固有の層次元列)
//   superbatch   u64  (この checkpoint が表す完了 superbatch、resume はこの +1 から)
//   step_count   u64  (Ranger lookahead step counter)
//   lr_horizon   u64  (v5+、LR schedule の終端 superbatch。0 = horizon 無し)
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
// version 互換規則 (1..=6 の受理と各 version の差分) は RAW_CKPT_VERSION の doc を参照。
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
///   `--ft-factorize` on/off is rejected.
///
/// `load_raw_checkpoint` accepts versions 1..=6. Version 1 is interpreted as
/// `halfka-hm-merged`; versions 1..=3 predate the arch-kind header and are
/// interpreted as `layerstack`. Versions above 6 are rejected. The producer
/// run id is absent (`None`) for versions 1 and 2; the LR horizon is absent
/// (`None`) for versions 1..=4; the factorizer flag is absent (false) for
/// versions 1..=5.
pub(crate) const RAW_CKPT_VERSION: u32 = 6;

/// `*.ckpt` の producer run id のバイト数上限。run id は `{net_id}-{時刻}-{pid}`
/// 程度で高々数十バイト。破損 file の巨大な length 値で過大確保しないための上限。
pub(crate) const MAX_RUN_ID_BYTES: usize = 256;

/// raw checkpoint 1 group 分の host buffer (`w`, `m`, `v`, `slow` の f32 Vec、`grad` は含めない)。
pub(crate) type RawCkptGroup = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// `load_raw_checkpoint` の戻り値: `(完了 superbatch, producer run id,
/// LR-schedule horizon)`。caller は superbatch+1 から resume し、horizon を
/// `build_lr_scheduler` に渡す。
pub(crate) type RawCkptResumeState = (usize, Option<String>, Option<usize>);

/// raw checkpoint 1 group 分の device-side 参照。weight name + 要素数 +
/// `(w, m, v, slow)` device buffer の借用。trainer が format の group 順に並べた
/// 列を [`save_raw_checkpoint_file`] / [`load_raw_checkpoint_file`] へ渡す。
pub(crate) struct RawCkptGroupSource<'a> {
    pub(crate) name: &'static str,
    pub(crate) len: usize,
    pub(crate) bufs: RawCkptGroupBufs<'a>,
}

/// [`RawCkptGroupSource`] の buffer 借用部。`Uniform` は w/m/v/slow 全部
/// `DeviceBuffer<f32>` の group、`FtMoment` は `m` / `v` が [`MomentBuf`]
/// (`--fp16-opt-state` で `f16` 格納) の ft_w group。
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

/// raw checkpoint header の counter / lineage 部 (arch identity 以外の可変 field)。
/// [`save_raw_checkpoint_file`] が [`RawCkptArch`] と並べて受け取る。
pub(crate) struct RawCkptMeta<'a> {
    /// この checkpoint を書き出す run の experiment.json `id` (resume 時の
    /// `lineage.parent_id` に使う)。空文字列は「未記録」。
    pub(crate) run_id: &'a str,
    /// この checkpoint が表す完了 superbatch 番号 (resume はこの +1 から)。
    pub(crate) superbatch: usize,
    /// Ranger lookahead step counter。
    pub(crate) step_count: u64,
    /// LR-schedule horizon (horizon を持たない schedule では `None`)。
    pub(crate) lr_horizon: Option<usize>,
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
    /// LR-schedule horizon (version 5+ かつ horizon を持つ schedule なら `Some`)。
    /// version 1..=4 や horizon を持たない schedule では `None`。
    pub(crate) lr_horizon: Option<usize>,
}

/// raw checkpoint の header (magic 〜 num_groups、group 本体の手前まで) を書く。
/// 常に最新 [`RAW_CKPT_VERSION`] で書き出す。
pub(crate) fn write_raw_ckpt_header<W: Write>(
    w: &mut W,
    arch: &RawCkptArch,
    run_id: &str,
    superbatch: u64,
    step_count: u64,
    lr_horizon: Option<usize>,
    num_groups: u64,
) -> std::io::Result<()> {
    w.write_all(&RAW_CKPT_MAGIC)?;
    w.write_all(&RAW_CKPT_VERSION.to_le_bytes())?;
    // feature set header (v2+): canonical 名 + 次元 3 値。
    let fs_name = arch.feature_set.canonical_name();
    w.write_all(&(fs_name.len() as u32).to_le_bytes())?;
    w.write_all(fs_name.as_bytes())?;
    w.write_all(&(arch.feature_set.train_ft_in() as u64).to_le_bytes())?;
    w.write_all(&arch.ft_out.to_le_bytes())?;
    w.write_all(&(arch.feature_set.max_active() as u64).to_le_bytes())?;
    // FT factorizer flag (v6+)。
    w.write_all(&[arch.feature_set.ft_factorize() as u8])?;
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
    // LR-schedule horizon (v5+)。0 = horizon 無し (step / constant / drop)。
    w.write_all(&(lr_horizon.unwrap_or(0) as u64).to_le_bytes())?;
    w.write_all(&num_groups.to_le_bytes())?;
    Ok(())
}

/// raw checkpoint の header を読み、`expected` の arch identity と照合する。
/// version 1..=6 を受理し、不一致 / 破損は `InvalidData` で reject する。
///
/// version 1..=3 は arch-kind header を持たず暗黙に `layerstack`。version 4 は
/// arch_kind 名と topology 次元列を `expected` と照合する。version 5 は
/// `step_count` の後に LR-schedule horizon の `u64` を持つ (`0` = horizon 無し)。
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
        // FT factorizer flag は version 6+。旧版は factorizer 以前の checkpoint
        // なので false 扱い (factorize 有効側との不一致は下の照合で reject)。
        let ckpt_ft_factorize = if version >= 6 {
            let mut buf1 = [0u8; 1];
            read_exact_or_invalid(r, &mut buf1, "ft_factorize flag")?;
            buf1[0] != 0
        } else {
            false
        };

        let want = expected.feature_set;
        if fs_name != want_name {
            return Err(invalid_data(format!(
                "raw checkpoint feature set mismatch: checkpoint is '{fs_name}', \
                 requested '{want_name}' (cannot resume across feature sets)"
            )));
        }
        // 次元照合より先に factorize 状態を見る (on/off 跨ぎは ft_in /
        // max_active も必ずずれるが、原因が読めるエラーを先に出す)。
        if ckpt_ft_factorize != want.ft_factorize() {
            return Err(invalid_data(format!(
                "raw checkpoint ft-factorize mismatch: checkpoint {ckpt_ft_factorize}, \
                 requested {} (cannot resume across --ft-factorize on/off)",
                want.ft_factorize()
            )));
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

    // LR-schedule horizon は version 5+。0 は「horizon 未記録」扱いで `None`。
    let lr_horizon: Option<usize> = if version >= 5 {
        read_exact_or_invalid(r, &mut buf8, "lr_horizon")?;
        let horizon_u64 = u64::from_le_bytes(buf8);
        if horizon_u64 == 0 {
            None
        } else {
            Some(horizon_u64.try_into().map_err(|_| {
                invalid_data(format!(
                    "raw checkpoint lr_horizon {horizon_u64} exceeds usize::MAX"
                ))
            })?)
        }
    } else {
        None
    };

    read_exact_or_invalid(r, &mut buf8, "num_groups")?;
    let num_groups = u64::from_le_bytes(buf8);

    Ok(RawCkptHeader {
        superbatch,
        step_count,
        num_groups,
        producer_run_id,
        lr_horizon,
    })
}

/// raw checkpoint を `path` に atomic に書き出す。header (arch identity + counters +
/// `num_groups` = `groups.len()`) に続けて `groups` を列挙順に書く。group 本体の
/// layout は各 group につき `len u64 + w[f32×len] + m[f32×len] + v[f32×len] +
/// slow[f32×len]` (全 little-endian)。device → host download は group 単位で write と
/// interleave するので、host 側のピークは最大 group (ft_w、~113M f32 = ~450MB) 1 個分。
///
/// `<path>.tmp` へ `BufWriter` で書いてから `std::fs::rename` で atomic に置換する
/// (書き込み途中で crash しても `<path>` は前回の完全な checkpoint のまま)。
///
/// `meta.run_id` が空文字列、または [`MAX_RUN_ID_BYTES`] 超過 (warning を出して
/// 省略) のときは run id を持たない checkpoint になり、resume 時の
/// `lineage.parent_id` は解決されない。
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

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = {
        let mut p = path.as_os_str().to_os_string();
        p.push(".tmp");
        std::path::PathBuf::from(p)
    };

    // write+flush 本体を closure に括り、`fs::rename` 前の error path で
    // 中途半端な `<path>.tmp` を best-effort で消す (device→host download / write /
    // flush 失敗で残骸を残さないため)。
    let write_tmp = || -> Result<(), Box<dyn std::error::Error>> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);
        write_raw_ckpt_header(
            &mut w,
            arch,
            run_id,
            meta.superbatch as u64,
            meta.step_count,
            meta.lr_horizon,
            groups.len() as u64,
        )?;
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
            w.write_all(&(g.len as u64).to_le_bytes())?;
            write_f32_slice(&mut w, &w_host)?;
            write_f32_slice(&mut w, &m_host)?;
            write_f32_slice(&mut w, &v_host)?;
            write_f32_slice(&mut w, &slow_host)?;
        }
        w.flush()?;
        Ok(())
    };
    if let Err(e) = write_tmp() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e.into());
    }
    Ok(())
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
            "raw checkpoint num_groups {} != expected {}",
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
    let mut buf8 = [0u8; 8];
    read_exact_or_invalid(r, &mut buf8, &format!("group {name} len"))?;
    let len_u64 = u64::from_le_bytes(buf8);
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

/// `io::ErrorKind::InvalidData` の `Box<dyn Error>` を作る短縮 helper (raw checkpoint
/// の magic/version/dim 検証で使う、`RangerHostState::load_from_reader` と同方針)。
pub(crate) fn invalid_data(msg: String) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
}

/// f32 slice を little-endian で `w` に書き出す (`bytemuck` 不使用、依存を増やさない)。
pub(crate) fn write_f32_slice<W: std::io::Write>(w: &mut W, data: &[f32]) -> std::io::Result<()> {
    // 4 byte ずつの write_all は遅いので、一旦 byte Vec に詰めてから 1 回で書く
    // (group は最大 113M f32 = ~450MB (ft_w)、呼び出し側は BufWriter で wrap 済だが
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
