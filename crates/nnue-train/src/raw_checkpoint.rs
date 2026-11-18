//! Host serialization for raw NNUE training checkpoints.

use std::io::{self, Write};

use nnue_format::ArchKind;
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
/// - `10`: a u32 QAT mode follows fv_scale: 0 = off, 1 = dense LayerStack.
///   Earlier versions imply off. Unknown modes and dense Simple files are rejected.
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
/// `load_raw_checkpoint` accepts versions 1..=15. Version 1 is interpreted as
/// `halfka-hm-merged`; versions 1..=3 predate the arch-kind header and are
/// interpreted as `layerstack`. Versions above 15 are rejected. The producer
/// run id is absent (`None`) for versions 1 and 2; the LR horizon is absent
/// (`None`) for versions 1..=4; the factorizer flag is absent (false) for
/// versions 1..=5; the feature hash is absent for versions 1..=6; `fv_scale`
/// is absent (`None`) for versions 1..=8; the producer net id and recipe
/// fingerprint are absent (`None`) for versions 1..=10.
pub const RAW_CKPT_VERSION: u32 = 15;
const ADAM_CHECKPOINT_VERSION: u32 = 11;
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
    /// Simple の export に使う実効 evaluation scale。LayerStack は `None`。
    pub fv_scale: Option<i32>,
    /// この checkpoint を書き出す run の net id。空文字列は「未記録」。
    pub net_id: &'a str,
    /// この checkpoint を書き出す run のレシピの指紋。実験ログが無ければ `None`。
    pub recipe_fingerprint: Option<u64>,
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
    w.write_all(&version.to_le_bytes())?;
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
    w.write_all(&u32::from(meta.qat_dense).to_le_bytes())?;
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

/// Write one master weight group followed by its first moment, second moment and slow weights.
pub fn write_raw_ckpt_group<W: Write>(w: &mut W, tensors: [&[f32]; 4]) -> io::Result<()> {
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
