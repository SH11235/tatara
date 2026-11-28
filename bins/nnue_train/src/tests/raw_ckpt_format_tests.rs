//! raw checkpoint format helper tests (GPU 不要)。

use nnue_format::ArchKind;
use nnue_train::dataloader::BucketMode;
use shogi_features::{EffectBucketConfig, FeatureSet};

use crate::{arch::*, ckpt::*};

use std::io::Cursor;
use std::path::Path;

struct LimitedIo<T> {
    inner: T,
    remaining: usize,
    max_request: usize,
}

impl<T: std::io::Read> std::io::Read for LimitedIo<T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.max_request = self.max_request.max(buf.len());
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            return Err(std::io::ErrorKind::PermissionDenied.into());
        }
        let len = buf.len().min(self.remaining).min(997);
        let n = self.inner.read(&mut buf[..len])?;
        self.remaining -= n;
        Ok(n)
    }
}

impl<T: std::io::Write> std::io::Write for LimitedIo<T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.max_request = self.max_request.max(buf.len());
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            return Err(std::io::ErrorKind::PermissionDenied.into());
        }
        let n = self
            .inner
            .write(&buf[..buf.len().min(self.remaining).min(997)])?;
        self.remaining -= n;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[test]
fn f32_io_is_byte_exact_across_chunks_and_short_operations() {
    let bits = [
        0,
        0x8000_0000,
        1,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0xffff_ffff,
    ];
    let data: Vec<f32> = bits
        .into_iter()
        .cycle()
        .take(2_097_155)
        .map(f32::from_bits)
        .collect();
    let expected: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    let mut writer = LimitedIo {
        inner: Vec::new(),
        remaining: usize::MAX,
        max_request: 0,
    };
    write_f32_slice(&mut writer, &data).unwrap();
    assert_eq!(writer.inner, expected);
    assert!(writer.max_request <= 4 * 1024 * 1024);
    let mut reader = LimitedIo {
        inner: Cursor::new(&expected),
        remaining: usize::MAX,
        max_request: 0,
    };
    let output = read_f32_vec_io(&mut reader, data.len(), "weights").unwrap();
    assert!(reader.max_request <= 4 * 1024 * 1024);
    assert_eq!(
        output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(reader.inner.position(), expected.len() as u64);
}

#[test]
fn f32_io_propagates_mid_tensor_errors() {
    let data = vec![1.0; 2_097_155];
    let mut writer = LimitedIo {
        inner: Vec::new(),
        remaining: 4_194_307,
        max_request: 0,
    };
    assert_eq!(
        write_f32_slice(&mut writer, &data).unwrap_err().kind(),
        std::io::ErrorKind::PermissionDenied
    );
    let mut reader = LimitedIo {
        inner: Cursor::new(vec![0; data.len() * 4]),
        remaining: 4_194_307,
        max_request: 0,
    };
    assert_eq!(
        read_f32_vec_io(&mut reader, data.len(), "weights")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
    let error =
        read_f32_vec_io(&mut Cursor::new(vec![0; 4_194_307]), data.len(), "weights").unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("weights"));
}

#[test]
fn f32_read_rejects_unrepresentable_allocation_sizes() {
    for n in [usize::MAX, isize::MAX as usize / 4 + 1] {
        assert_eq!(
            read_f32_vec_io(&mut Cursor::new([]), n, "weights")
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn failed_chunked_checkpoint_write_keeps_existing_artifact() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "tatara-chunked-checkpoint-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("checkpoint.ckpt");
    std::fs::write(&path, b"previous checkpoint").unwrap();
    let result = nnue_train::artifact::write_atomic(&path, |writer| {
        let mut failing = LimitedIo {
            inner: writer,
            remaining: 4_194_307,
            max_request: 0,
        };
        write_f32_slice(&mut failing, &vec![1.0; 2_097_155])
    });
    assert_eq!(
        result.unwrap_err().kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"previous checkpoint");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn write_then_read_f32_slice_round_trips() {
    let data: Vec<f32> = vec![0.0, 1.0, -1.0, 3.5, f32::MIN_POSITIVE, -42.25, 1e9];
    let mut buf = Vec::new();
    write_f32_slice(&mut buf, &data).unwrap();
    assert_eq!(buf.len(), data.len() * 4);
    let back = read_f32_vec_io(&mut Cursor::new(&buf), data.len(), "test").unwrap();
    assert_eq!(back, data);
}

#[test]
fn empty_f32_slice_round_trips() {
    let mut buf = Vec::new();
    write_f32_slice(&mut buf, &[]).unwrap();
    assert!(buf.is_empty());
    let back = read_f32_vec_io(&mut Cursor::new(&buf), 0, "test").unwrap();
    assert!(back.is_empty());
}

#[test]
fn read_f32_vec_errors_on_short_input() {
    // 破損 / 部分書き checkpoint の短読みは `UnexpectedEof` ではなく context 付き
    // `InvalidData` に正規化される (robustness contract: malformed input は全部 InvalidData)。
    let buf = vec![0u8; 6]; // 1.5 f32 worth
    let err = read_f32_vec_io(&mut Cursor::new(&buf), 2, "group ft_w w")
        .expect_err("must error on short read");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("truncated") && err.to_string().contains("group ft_w w"),
        "error message should describe the truncated field: {err}"
    );
}

#[test]
fn read_exact_or_invalid_maps_eof_to_invalid_data() {
    let buf = vec![0u8; 2];
    let mut out = [0u8; 8];
    let err = read_exact_or_invalid(&mut Cursor::new(&buf), &mut out, "superbatch")
        .expect_err("must error on short read");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("truncated"));
    assert!(err.to_string().contains("superbatch"));
}

#[test]
fn raw_ckpt_constants_are_stable() {
    // magic は format identity。version は後方互換読み (version 1..=9 file の受理)
    // を維持しつつ前進するので、現行値を pin して意図しない変更を検出する。
    assert_eq!(&RAW_CKPT_MAGIC, b"RNRC");
    assert_eq!(RAW_CKPT_VERSION, 18);
}

#[test]
fn raw_ckpt_marks_sgd_without_changing_adam_layout() {
    use nnue_train::optimizer::OptimizerKind;

    let arch = layerstack_arch();
    let meta = RawCkptMeta {
        optimizer: OptimizerKind::RAdam,
        qat_dense: false,
        run_id: "",
        superbatch: 1,
        step_count: 1,
        lr_horizon: None,
        qat_warmup_steps: None,
        lr_total_steps: None,
        adam_bias_correction: false,
        fv_scale: None,
        net_id: "",
        recipe_fingerprint: None,
    };
    let write = |optimizer| {
        let mut bytes = Vec::new();
        crate::ckpt::write_raw_ckpt_header(
            &mut bytes,
            &arch,
            &RawCkptMeta { optimizer, ..meta },
            10,
        )
        .unwrap();
        bytes
    };
    let adam = write(OptimizerKind::RAdam);
    let sgd = write(OptimizerKind::Sgd);
    assert_eq!(adam[4..8], 11_u32.to_le_bytes());
    assert_eq!(sgd[4..8], 12_u32.to_le_bytes());
    assert_eq!(adam[8..], sgd[8..]);
    assert!(
        !read_raw_ckpt_header(&mut Cursor::new(adam), &arch)
            .unwrap()
            .sgd
    );
    assert!(
        read_raw_ckpt_header(&mut Cursor::new(sgd), &arch)
            .unwrap()
            .sgd
    );
}

#[test]
fn qat_mode_header_roundtrip_and_legacy_default() {
    let arch = layerstack_arch();
    let mut bytes = Vec::new();
    crate::ckpt::write_raw_ckpt_header(
        &mut bytes,
        &arch,
        &RawCkptMeta {
            optimizer: nnue_train::optimizer::OptimizerKind::Ranger,
            qat_dense: true,
            run_id: "qat",
            superbatch: 1201,
            step_count: 10,
            lr_horizon: None,
            qat_warmup_steps: None,
            lr_total_steps: None,
            adam_bias_correction: false,
            fv_scale: None,
            net_id: "",
            recipe_fingerprint: None,
        },
        10,
    )
    .unwrap();
    downgrade_v11_to_v10(&mut bytes);
    assert!(
        read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch)
            .unwrap()
            .qat_dense
    );
    let offset = bytes.len() - 12;
    let mut invalid = bytes.clone();
    invalid[offset..offset + 4].copy_from_slice(&2_u32.to_le_bytes());
    assert!(read_raw_ckpt_header(&mut Cursor::new(&invalid), &arch).is_err());
    bytes.drain(offset..offset + 4);
    bytes[4..8].copy_from_slice(&9_u32.to_le_bytes());
    assert!(
        !read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch)
            .unwrap()
            .qat_dense
    );
    let simple = simple_arch();
    bytes.clear();
    crate::ckpt::write_raw_ckpt_header(
        &mut bytes,
        &simple,
        &RawCkptMeta {
            optimizer: nnue_train::optimizer::OptimizerKind::Ranger,
            qat_dense: true,
            run_id: "invalid-simple-qat",
            superbatch: 1,
            step_count: 1,
            lr_horizon: None,
            qat_warmup_steps: None,
            lr_total_steps: None,
            adam_bias_correction: false,
            fv_scale: Some(14),
            net_id: "",
            recipe_fingerprint: None,
        },
        8,
    )
    .unwrap();
    let error = read_raw_ckpt_header(&mut Cursor::new(&bytes), &simple).unwrap_err();
    assert!(error.to_string().contains("unsupported QAT mode 1"));
}

#[test]
fn invalid_data_helper_makes_invalid_data_error() {
    let e = invalid_data("boom".to_string());
    let io_err = e.downcast::<std::io::Error>().expect("is io::Error");
    assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidData);
    assert!(io_err.to_string().contains("boom"));
}

/// arch header を持たない legacy (version 1..=3) raw checkpoint header を組む。
fn legacy_raw_ckpt_header(
    version: u32,
    fs: shogi_features::FeatureSetSpec,
    run_id: Option<&str>,
    superbatch: u64,
    step_count: u64,
    num_groups: u64,
) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&RAW_CKPT_MAGIC);
    b.extend_from_slice(&version.to_le_bytes());
    if version >= 2 {
        let name = fs.canonical_name();
        b.extend_from_slice(&(name.len() as u32).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(fs.ft_in() as u64).to_le_bytes());
        b.extend_from_slice(&(DEFAULT_FT_OUT as u64).to_le_bytes());
        b.extend_from_slice(&(fs.max_active() as u64).to_le_bytes());
    }
    if version >= 3 {
        let rid = run_id.unwrap_or("");
        b.extend_from_slice(&(rid.len() as u32).to_le_bytes());
        b.extend_from_slice(rid.as_bytes());
    }
    b.extend_from_slice(&superbatch.to_le_bytes());
    b.extend_from_slice(&step_count.to_le_bytes());
    b.extend_from_slice(&num_groups.to_le_bytes());
    b
}

/// LR-horizon header を持たない version 4 の LayerStack raw checkpoint header を
/// 組む (arch-kind + topology header あり、horizon なし)。
fn v4_layerstack_header(superbatch: u64, step_count: u64, num_groups: u64) -> Vec<u8> {
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let mut b = Vec::new();
    b.extend_from_slice(&RAW_CKPT_MAGIC);
    b.extend_from_slice(&4u32.to_le_bytes());
    // feature set header (v2+)。
    let name = fs.canonical_name();
    b.extend_from_slice(&(name.len() as u32).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(fs.ft_in() as u64).to_le_bytes());
    b.extend_from_slice(&(DEFAULT_FT_OUT as u64).to_le_bytes());
    b.extend_from_slice(&(fs.max_active() as u64).to_le_bytes());
    // producer run id (v3+、空)。
    b.extend_from_slice(&0u32.to_le_bytes());
    // arch-kind + topology header (v4+)。
    let arch_name = ArchKind::LayerStack.canonical_name();
    b.extend_from_slice(&(arch_name.len() as u32).to_le_bytes());
    b.extend_from_slice(arch_name.as_bytes());
    b.extend_from_slice(&(DEFAULT_LAYERSTACK_TOPOLOGY.len() as u64).to_le_bytes());
    for &dim in &DEFAULT_LAYERSTACK_TOPOLOGY {
        b.extend_from_slice(&dim.to_le_bytes());
    }
    b.extend_from_slice(&superbatch.to_le_bytes());
    b.extend_from_slice(&step_count.to_le_bytes());
    b.extend_from_slice(&num_groups.to_le_bytes());
    b
}

/// 既定 FT / L1 / L2 出力次元 + 9 bucket の LayerStack topology (test helper、
/// `'static` 借用用)。9 は kingrank9 を含む全 bucket mode で有効な bucket 数なので、
/// bucket mode だけが異なる checkpoint を同じ topology で比較できる。
const DEFAULT_LAYERSTACK_TOPOLOGY: [u64; 4] = layerstack_topology(
    DEFAULT_FT_OUT,
    DEFAULT_L1_OUT,
    DEFAULT_L2_OUT,
    shogi_features::KINGRANK9_NUM_BUCKETS,
);

fn layerstack_arch() -> RawCkptArch<'static> {
    layerstack_arch_with_mode("progresskpabs")
}

fn layerstack_arch_with_mode(bucket_mode: &'static str) -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        arch_kind: ArchKind::LayerStack,
        bucket_mode: Some(bucket_mode),
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

fn simple_arch() -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        arch_kind: ArchKind::Simple,
        bucket_mode: None,
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &[
            DEFAULT_FT_OUT as u64,
            DEFAULT_L1_OUT as u64,
            DEFAULT_L2_OUT as u64,
        ],
    }
}

fn write_raw_ckpt_header<W: std::io::Write>(
    w: &mut W,
    arch: &RawCkptArch,
    run_id: &str,
    superbatch: u64,
    step_count: u64,
    lr_horizon: Option<usize>,
    num_groups: u64,
) -> std::io::Result<()> {
    crate::ckpt::write_raw_ckpt_header(
        w,
        arch,
        &RawCkptMeta {
            optimizer: nnue_train::optimizer::OptimizerKind::Ranger,
            qat_dense: false,
            run_id,
            superbatch: superbatch.try_into().expect("test superbatch fits usize"),
            step_count,
            lr_horizon,
            qat_warmup_steps: None,
            lr_total_steps: None,
            adam_bias_correction: false,
            fv_scale: None,
            net_id: "",
            recipe_fingerprint: None,
        },
        num_groups,
    )
}

fn write_raw_ckpt_header_with_fv_scale(
    buf: &mut Vec<u8>,
    arch: &RawCkptArch,
    fv_scale: Option<i32>,
) {
    crate::ckpt::write_raw_ckpt_header(
        buf,
        arch,
        &RawCkptMeta {
            optimizer: nnue_train::optimizer::OptimizerKind::Ranger,
            qat_dense: false,
            run_id: "fv-scale-test",
            superbatch: 9,
            step_count: 90,
            lr_horizon: Some(120),
            qat_warmup_steps: None,
            lr_total_steps: None,
            adam_bias_correction: false,
            fv_scale,
            net_id: "",
            recipe_fingerprint: None,
        },
        8,
    )
    .unwrap();
}

/// `layerstack_arch` の FT factorizer 有効版。
fn layerstack_arch_factorized() -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged.spec().with_ft_factorize(),
        arch_kind: ArchKind::LayerStack,
        bucket_mode: Some("progresskpabs"),
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

fn layerstack_arch_effect_bucket(config: EffectBucketConfig) -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged
            .spec()
            .with_effect_bucket_config(config),
        arch_kind: ArchKind::LayerStack,
        bucket_mode: Some("progresskpabs"),
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

fn bucket_mode_field_offset(buf: &[u8], arch: &RawCkptArch) -> usize {
    let mut off = 4 + 4 + 4 + arch.feature_set.canonical_name().len() + 8 + 8 + 8 + 1 + 4;
    let run_id_len = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    off += 4 + run_id_len;
    let arch_name_len = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    off + 4 + arch_name_len
}

/// v11 の producer net id (空) と recipe fingerprint を外して v10 の layout にする。
fn downgrade_v11_to_v10(buf: &mut Vec<u8>) {
    let tail_off = buf.len() - 8 - 8 - 4;
    assert_eq!(
        u32::from_le_bytes(buf[tail_off..tail_off + 4].try_into().unwrap()),
        0
    );
    buf.drain(tail_off..tail_off + 4 + 8);
    buf[4..8].copy_from_slice(&10u32.to_le_bytes());
}

fn downgrade_v9_to_v8(buf: &mut Vec<u8>, written_fv_scale: Option<i32>) {
    downgrade_v11_to_v10(buf);
    let qat_off = buf.len() - 4 - 8;
    buf.drain(qat_off..qat_off + 4);
    let fv_scale_off = buf.len() - 4 - 8;
    let stored_fv_scale =
        i32::from_le_bytes(buf[fv_scale_off..fv_scale_off + 4].try_into().unwrap());
    assert_eq!(stored_fv_scale, written_fv_scale.unwrap_or(0));
    buf.drain(fv_scale_off..fv_scale_off + 4);
    buf[4..8].copy_from_slice(&8u32.to_le_bytes());
}

fn downgrade_v8_to_v7(buf: &mut Vec<u8>, arch: &RawCkptArch) {
    downgrade_v9_to_v8(buf, None);
    let off = bucket_mode_field_offset(buf, arch);
    let len = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    buf.drain(off..off + 4 + len);
    buf[4..8].copy_from_slice(&7u32.to_le_bytes());
}

#[test]
fn raw_ckpt_v9_fv_scale_round_trips() {
    let arch = simple_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header_with_fv_scale(&mut buf, &arch, Some(14));

    let header = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(header.fv_scale, Some(14));
}

#[test]
fn raw_ckpt_v9_zero_fv_scale_reads_as_none() {
    let arch = simple_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header_with_fv_scale(&mut buf, &arch, Some(0));

    let header = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(header.fv_scale, None);
}

#[test]
fn raw_ckpt_v9_rejects_negative_fv_scale() {
    let arch = simple_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header_with_fv_scale(&mut buf, &arch, Some(-14));

    let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch)
        .expect_err("negative fv_scale must be rejected");
    let io_err = err.downcast::<std::io::Error>().expect("is io::Error");
    assert_eq!(io_err.kind(), std::io::ErrorKind::InvalidData);
    assert!(io_err.to_string().contains("fv_scale -14"));
}

#[test]
fn raw_ckpt_v8_reads_fv_scale_as_none_and_preserves_other_fields() {
    let arch = simple_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header_with_fv_scale(&mut buf, &arch, Some(14));
    downgrade_v9_to_v8(&mut buf, Some(14));
    assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 8);

    let header = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(header.fv_scale, None);
    assert_eq!(header.superbatch, 9);
    assert_eq!(header.step_count, 90);
    assert_eq!(header.lr_horizon, Some(120));
    assert_eq!(header.producer_run_id.as_deref(), Some("fv-scale-test"));
    assert_eq!(header.num_groups, 8);
}

fn remove_v7_feature_hash(buf: &mut Vec<u8>, arch: &RawCkptArch) {
    downgrade_v8_to_v7(buf, arch);
    buf[4..8].copy_from_slice(&6u32.to_le_bytes());
    let max_active_off = 4 + 4 + 4 + arch.feature_set.canonical_name().len() + 8 + 8;
    let feature_hash_off = max_active_off + 8 + 1;
    buf.drain(feature_hash_off..feature_hash_off + 4);
}

fn downgrade_v6_to_v5(buf: &mut Vec<u8>, arch: &RawCkptArch) {
    remove_v7_feature_hash(buf, arch);
    let factorizer_off = 4 + 4 + 4 + arch.feature_set.canonical_name().len() + 8 + 8 + 8;
    assert_eq!(buf[factorizer_off], 0);
    buf.drain(factorizer_off..factorizer_off + 1);
    buf[4..8].copy_from_slice(&5u32.to_le_bytes());
}

#[test]
fn raw_ckpt_v8_bucket_modes_round_trip() {
    for arch in [
        layerstack_arch_with_mode("progresskpabs"),
        layerstack_arch_with_mode("progresskpabsq16"),
        layerstack_arch_with_mode("kingrank9"),
        simple_arch(),
    ] {
        let mut buf = Vec::new();
        write_raw_ckpt_header(&mut buf, &arch, "bucket-mode-roundtrip", 2, 20, None, 10).unwrap();
        downgrade_v9_to_v8(&mut buf, None);
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 8);
        let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
        assert_eq!((h.superbatch, h.step_count, h.num_groups), (2, 20, 10));
        assert_eq!(h.producer_run_id.as_deref(), Some("bucket-mode-roundtrip"));
        assert_eq!(h.lr_horizon, None);
        assert_eq!(h.fv_scale, None);
    }
}

#[test]
fn raw_ckpt_v8_accepts_legacy_progresskpabs_alias() {
    // 旧綴り "progress8kpabs" を格納した checkpoint は canonical 名
    // "progresskpabs" を expected とする読み出しで受理される。
    let legacy = layerstack_arch_with_mode(BucketMode::LEGACY_PROGRESS_KPABS_NAME);
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &legacy, "legacy-alias", 3, 30, None, 10).unwrap();
    downgrade_v9_to_v8(&mut buf, None);

    let canonical = layerstack_arch_with_mode(BucketMode::ProgressKpAbs.canonical_name());
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &canonical).unwrap();
    assert_eq!((h.superbatch, h.step_count, h.num_groups), (3, 30, 10));

    // 旧綴り格納でも kingrank9 との cross-mode resume は canonical 名で拒否される。
    let err = read_raw_ckpt_header(
        &mut Cursor::new(&buf),
        &layerstack_arch_with_mode("kingrank9"),
    )
    .expect_err("legacy-alias checkpoint must not resume as kingrank9");
    assert_eq!(
        err.to_string(),
        "checkpoint bucket mode 'progresskpabs' does not match --bucket-mode 'kingrank9'"
    );
}

#[test]
fn raw_ckpt_v8_rejects_cross_bucket_mode_resume_in_both_directions() {
    for (written, requested) in [
        ("progresskpabs", "kingrank9"),
        ("kingrank9", "progresskpabs"),
        ("progresskpabs", "progresskpabsq16"),
        ("progresskpabsq16", "progresskpabs"),
    ] {
        let mut buf = Vec::new();
        write_raw_ckpt_header(
            &mut buf,
            &layerstack_arch_with_mode(written),
            "",
            1,
            0,
            None,
            10,
        )
        .unwrap();
        downgrade_v9_to_v8(&mut buf, None);
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 8);
        let err = read_raw_ckpt_header(
            &mut Cursor::new(&buf),
            &layerstack_arch_with_mode(requested),
        )
        .expect_err("cross-bucket-mode resume must be rejected");
        assert_eq!(
            err.to_string(),
            format!(
                "checkpoint bucket mode '{written}' does not match --bucket-mode '{requested}'"
            )
        );
    }
}

#[test]
fn raw_ckpt_v7_layerstack_defaults_to_progresskpabs() {
    let progress = layerstack_arch_with_mode("progresskpabs");
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &progress, "legacy-v7", 7, 70, Some(100), 10).unwrap();
    downgrade_v8_to_v7(&mut buf, &progress);

    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &progress).unwrap();
    assert_eq!((h.superbatch, h.step_count, h.num_groups), (7, 70, 10));

    let err = read_raw_ckpt_header(
        &mut Cursor::new(&buf),
        &layerstack_arch_with_mode("kingrank9"),
    )
    .expect_err("v7 layerstack checkpoint must not resume as kingrank9");
    assert_eq!(
        err.to_string(),
        "checkpoint bucket mode 'progresskpabs' does not match --bucket-mode 'kingrank9'"
    );
}

#[test]
fn raw_ckpt_header_ft_factorize_round_trips() {
    let arch = layerstack_arch_factorized();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "run-1", 2, 20, None, 10).unwrap();
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!((h.superbatch, h.step_count, h.num_groups), (2, 20, 10));
}

#[test]
fn raw_ckpt_header_rejects_effect_bucket_config_hash_mismatch() {
    let fixed = layerstack_arch_effect_bucket(EffectBucketConfig::KINGFIXED_2X2);
    let bucketed = layerstack_arch_effect_bucket(EffectBucketConfig::KINGBUCKETED_2X2);
    assert_eq!(
        fixed.feature_set.canonical_name(),
        bucketed.feature_set.canonical_name()
    );
    assert_eq!(
        fixed.feature_set.train_ft_in(),
        bucketed.feature_set.train_ft_in()
    );
    assert_eq!(
        fixed.feature_set.max_active(),
        bucketed.feature_set.max_active()
    );
    assert_ne!(
        fixed.feature_set.feature_hash(),
        bucketed.feature_set.feature_hash()
    );

    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &fixed, "effect_bucket-fixed", 5, 50, None, 10).unwrap();
    let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &bucketed)
        .expect_err("effect bucket config mismatch must be rejected");
    assert!(
        err.to_string().contains("feature hash mismatch"),
        "error should mention feature hash mismatch: {err}"
    );
}

#[test]
fn raw_ckpt_header_rejects_hashless_effect_bucket_checkpoint() {
    let arch = layerstack_arch_effect_bucket(EffectBucketConfig::KINGFIXED_2X2);
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "hashless-effect-bucket", 5, 50, None, 10).unwrap();
    remove_v7_feature_hash(&mut buf, &arch);

    let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch)
        .expect_err("hashless effect bucket checkpoint must be rejected");
    assert!(
        err.to_string()
            .contains("requires a checkpoint with feature hash"),
        "error should mention feature hash requirement: {err}"
    );
}

#[test]
fn raw_ckpt_header_accepts_legacy_factorized_max_active() {
    // v6 の factorized file には max_active = 2×base の個体が存在する (仮想
    // 特徴を sparse index 列に流す実装が書いたもの、RAW_CKPT_VERSION doc 参照)。
    // tensor payload は max_active 値に依らず同一のため reader は両値を受理
    // する (base 値側は round-trip テストが担保)。
    let arch = layerstack_arch_factorized();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "run-legacy", 3, 30, None, 10).unwrap();
    remove_v7_feature_hash(&mut buf, &arch);
    let base = arch.feature_set.max_active() as u64;
    // max_active field の offset = magic(4) + version(4) + name_len(4) +
    // name + ft_in(8) + ft_out(8)。書き換え前に base 値が居ることを確認して
    // fixture が field を正しく指していることを固定する。
    let off = 4 + 4 + 4 + arch.feature_set.canonical_name().len() + 8 + 8;
    assert_eq!(
        u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()),
        base,
        "fixture must point at the max_active field"
    );
    buf[off..off + 8].copy_from_slice(&(2 * base).to_le_bytes());
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!((h.superbatch, h.step_count, h.num_groups), (3, 30, 10));

    // 互換受理は factorize 限定: 非 factorize の 2×base は従来どおり reject。
    let off_arch = layerstack_arch();
    let mut buf_off = Vec::new();
    write_raw_ckpt_header(&mut buf_off, &off_arch, "", 1, 0, None, 10).unwrap();
    remove_v7_feature_hash(&mut buf_off, &off_arch);
    buf_off[off..off + 8].copy_from_slice(&(2 * base).to_le_bytes());
    let err = read_raw_ckpt_header(&mut Cursor::new(&buf_off), &off_arch)
        .expect_err("non-factorize header with 2x max_active must be rejected");
    assert!(
        err.to_string().contains("max_active"),
        "error should mention max_active: {err}"
    );
}

#[test]
fn raw_ckpt_header_rejects_ft_factorize_mismatch() {
    // on で書いた header を off expected で読む / その逆 — どちらも次元照合より
    // 先に ft-factorize mismatch として reject される (原因が読めるエラー)。
    let on = layerstack_arch_factorized();
    let off = layerstack_arch();

    let mut buf_on = Vec::new();
    write_raw_ckpt_header(&mut buf_on, &on, "", 1, 0, None, 10).unwrap();
    let err = read_raw_ckpt_header(&mut Cursor::new(&buf_on), &off)
        .expect_err("on ckpt + off expected must be rejected");
    assert!(
        err.to_string().contains("ft-factorize"),
        "error should mention ft-factorize: {err}"
    );

    let mut buf_off = Vec::new();
    write_raw_ckpt_header(&mut buf_off, &off, "", 1, 0, None, 10).unwrap();
    let err = read_raw_ckpt_header(&mut Cursor::new(&buf_off), &on)
        .expect_err("off ckpt + on expected must be rejected");
    assert!(
        err.to_string().contains("ft-factorize"),
        "error should mention ft-factorize: {err}"
    );
}

/// `layerstack_arch` の threat (cross-side) + factorizer 同居版。
fn layerstack_arch_threat_factorized() -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(shogi_features::ThreatProfile::CrossSide)
            .with_ft_factorize(),
        arch_kind: ArchKind::LayerStack,
        bucket_mode: Some("progresskpabs"),
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

#[test]
fn raw_ckpt_header_threat_factorize_coexist_round_trips() {
    // 同居 (threat + factorizer) の header が round-trip する。train_ft_in /
    // max_active が threat 連結分を含むことを identity に反映。
    let arch = layerstack_arch_threat_factorized();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "coexist-run", 4, 40, None, 10).unwrap();
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!((h.superbatch, h.step_count, h.num_groups), (4, 40, 10));
}

#[test]
fn raw_ckpt_header_rejects_threat_factorize_combo_mismatch() {
    use shogi_features::ThreatProfile;
    // 同居 ckpt を「threat-off + factorizer」「threat-only (factorizer-off)」「別
    // profile + factorizer」で読むと、いずれも identity 不一致で reject される
    // (ft-factorize flag か train_ft_in/feature-set 名のいずれかで弾く)。
    let coexist = layerstack_arch_threat_factorized();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &coexist, "", 1, 0, None, 10).unwrap();

    // threat-only (factorizer OFF): ft-factorize mismatch。
    let threat_only = RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(ThreatProfile::CrossSide),
        ..coexist
    };
    read_raw_ckpt_header(&mut Cursor::new(&buf), &threat_only)
        .expect_err("coexist ckpt read as threat-only must reject");

    // 別 profile + factorizer: train_ft_in (threat dims) 不一致で reject。
    let other_profile = RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(ThreatProfile::SameClass)
            .with_ft_factorize(),
        ..coexist
    };
    read_raw_ckpt_header(&mut Cursor::new(&buf), &other_profile)
        .expect_err("coexist ckpt read as different threat profile must reject");

    // threat-off + factorizer: train_ft_in 不一致で reject。
    read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch_factorized())
        .expect_err("coexist ckpt read as threat-off factorizer must reject");
}

#[test]
fn raw_ckpt_header_v5_round_trips_with_lr_horizon() {
    let arch = layerstack_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "net-20260520-1234", 7, 99, Some(123), 10).unwrap();
    downgrade_v6_to_v5(&mut buf, &arch);
    assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 5);
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(h.superbatch, 7);
    assert_eq!(h.step_count, 99);
    assert_eq!(h.num_groups, 10);
    assert_eq!(h.producer_run_id.as_deref(), Some("net-20260520-1234"));
    assert_eq!(h.lr_horizon, Some(123));
    assert_eq!(h.fv_scale, None);
}

#[test]
fn raw_ckpt_header_absent_lr_horizon_round_trips_to_none() {
    // horizon を持たない schedule (step / constant / drop) は None で書かれ、
    // read 側も None に戻る (sentinel 0)。
    let arch = layerstack_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "", 1, 0, None, 10).unwrap();
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(h.lr_horizon, None);
}

#[test]
fn raw_ckpt_header_empty_run_id_round_trips_to_none() {
    let arch = layerstack_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "", 1, 0, None, 10).unwrap();
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(h.producer_run_id, None);
}

#[test]
fn raw_ckpt_header_reads_legacy_v1_v2_v3() {
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let arch = layerstack_arch();
    // v1: header 無し、halfka-hm-merged 固定、arch は暗黙 layerstack。
    let v1 = legacy_raw_ckpt_header(1, fs, None, 3, 30, 10);
    let h1 = read_raw_ckpt_header(&mut Cursor::new(&v1), &arch).unwrap();
    assert_eq!((h1.superbatch, h1.step_count, h1.num_groups), (3, 30, 10));
    assert_eq!(h1.producer_run_id, None);
    // v2: feature set header あり、run id 無し。
    let v2 = legacy_raw_ckpt_header(2, fs, None, 4, 40, 10);
    let h2 = read_raw_ckpt_header(&mut Cursor::new(&v2), &arch).unwrap();
    assert_eq!(h2.superbatch, 4);
    assert_eq!(h2.producer_run_id, None);
    // v3: producer run id あり。
    let v3 = legacy_raw_ckpt_header(3, fs, Some("legacy-run"), 5, 50, 10);
    let h3 = read_raw_ckpt_header(&mut Cursor::new(&v3), &arch).unwrap();
    assert_eq!(h3.superbatch, 5);
    assert_eq!(h3.producer_run_id.as_deref(), Some("legacy-run"));
    // v1..=3 は LR horizon header を持たないので None に解釈される (後方互換)。
    assert_eq!(h1.lr_horizon, None);
    assert_eq!(h2.lr_horizon, None);
    assert_eq!(h3.lr_horizon, None);
}

#[test]
fn raw_ckpt_header_v4_reads_lr_horizon_as_none() {
    // version 4 は LR horizon header を持たない。read 側は schedule 情報無しと
    // みなし None を返す (caller は CLI 値から再構築する、現行挙動を維持)。
    let arch = layerstack_arch();
    let v4 = v4_layerstack_header(6, 60, 10);
    let h = read_raw_ckpt_header(&mut Cursor::new(&v4), &arch).unwrap();
    assert_eq!((h.superbatch, h.step_count, h.num_groups), (6, 60, 10));
    assert_eq!(h.lr_horizon, None);
}

#[test]
fn raw_ckpt_header_rejects_wrong_arch_kind() {
    // fs header は一致するが arch_kind だけ異なる v4 header を read → reject。
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let written = RawCkptArch {
        feature_set: fs,
        arch_kind: ArchKind::Simple,
        bucket_mode: None,
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    };
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &written, "", 1, 0, None, 10).unwrap();
    let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch())
        .expect_err("arch kind mismatch must reject");
    assert!(err.to_string().contains("arch kind mismatch"));
}

#[test]
fn raw_ckpt_header_rejects_wrong_topology() {
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let wrong_topo = [1u64, 2, 3, 4];
    let written = RawCkptArch {
        feature_set: fs,
        arch_kind: ArchKind::LayerStack,
        bucket_mode: Some("progresskpabs"),
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &wrong_topo,
    };
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &written, "", 1, 0, None, 10).unwrap();
    let err = read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch())
        .expect_err("topology mismatch must reject");
    assert!(err.to_string().contains("topology dim"));
}

#[test]
fn raw_ckpt_header_rejects_legacy_non_layerstack_request() {
    // version 1..=3 は arch header を持たず暗黙 layerstack。Simple として読もう
    // とすると reject される。
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let v3 = legacy_raw_ckpt_header(3, fs, None, 1, 0, 8);
    let simple_topo = [256u64, 32, 32];
    let simple_arch = RawCkptArch {
        feature_set: fs,
        arch_kind: ArchKind::Simple,
        bucket_mode: None,
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &simple_topo,
    };
    let err = read_raw_ckpt_header(&mut Cursor::new(&v3), &simple_arch)
        .expect_err("legacy checkpoint cannot be read as a non-layerstack arch");
    assert!(err.to_string().contains("predates the arch-kind header"));
}

#[test]
fn raw_ckpt_header_rejects_unsupported_version() {
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let mut buf = legacy_raw_ckpt_header(3, fs, None, 1, 0, 10);
    // magic (4 bytes) 直後の version u32 を範囲外の値に書き換える。
    buf[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(read_raw_ckpt_header(&mut Cursor::new(&buf), &layerstack_arch()).is_err());
}

#[test]
fn raw_ckpt_v11_producer_identity_round_trips() {
    let arch = layerstack_arch();
    let write = |net_id: &str, recipe_fingerprint: Option<u64>| {
        let mut bytes = Vec::new();
        crate::ckpt::write_raw_ckpt_header(
            &mut bytes,
            &arch,
            &RawCkptMeta {
                optimizer: nnue_train::optimizer::OptimizerKind::Ranger,
                qat_dense: false,
                run_id: "run-a",
                superbatch: 420,
                step_count: 1,
                lr_horizon: None,
                qat_warmup_steps: None,
                lr_total_steps: None,
                adam_bias_correction: false,
                fv_scale: None,
                net_id,
                recipe_fingerprint,
            },
            10,
        )
        .unwrap();
        bytes
    };

    let header =
        read_raw_ckpt_header(&mut Cursor::new(&write("net-a", Some(0xfeed))), &arch).unwrap();
    assert_eq!(header.producer_net_id.as_deref(), Some("net-a"));
    assert_eq!(header.producer_recipe_fingerprint, Some(0xfeed));
    assert_eq!(header.producer_run_id.as_deref(), Some("run-a"));
    assert_eq!(header.num_groups, 10);

    let unrecorded = read_raw_ckpt_header(&mut Cursor::new(&write("", None)), &arch).unwrap();
    assert_eq!(unrecorded.producer_net_id, None);
    assert_eq!(unrecorded.producer_recipe_fingerprint, None);

    let mut legacy = write("", None);
    downgrade_v11_to_v10(&mut legacy);
    let legacy = read_raw_ckpt_header(&mut Cursor::new(&legacy), &arch).unwrap();
    assert_eq!(legacy.producer_net_id, None);
    assert_eq!(legacy.producer_recipe_fingerprint, None);
    assert_eq!(legacy.superbatch, 420);
}

#[test]
fn partial_threat_checkpoint_identity_and_full_width_bytes() {
    let full = layerstack_arch_threat_factorized();
    let explicit = RawCkptArch {
        feature_set: full.feature_set.with_threat_width(DEFAULT_FT_OUT),
        ..full
    };
    let partial = RawCkptArch {
        feature_set: full.feature_set.with_threat_width(512),
        ..full
    };
    let with_spec = RawCkptArch {
        bucket_mode: Some("spec:ownking:2/3+oppking:2/3"),
        ..partial
    };
    use nnue_train::optimizer::OptimizerKind;

    let meta = RawCkptMeta {
        optimizer: OptimizerKind::Ranger,
        qat_dense: false,
        run_id: "run",
        superbatch: 1,
        step_count: 5,
        lr_horizon: None,
        qat_warmup_steps: None,
        lr_total_steps: None,
        adam_bias_correction: false,
        fv_scale: None,
        net_id: "",
        recipe_fingerprint: None,
    };
    let write = |arch, optimizer| {
        let mut bytes = Vec::new();
        crate::ckpt::write_raw_ckpt_header(
            &mut bytes,
            arch,
            &RawCkptMeta { optimizer, ..meta },
            10,
        )
        .unwrap();
        bytes
    };
    for (optimizer, version) in [
        (OptimizerKind::Ranger, 13_u32),
        (OptimizerKind::RAdam, 15),
        (OptimizerKind::AdamW, 13),
        (OptimizerKind::Sgd, 14),
    ] {
        let full_bytes = write(&full, optimizer);
        assert_eq!(full_bytes, write(&explicit, optimizer));
        let packed_bytes = write(&partial, optimizer);
        assert_eq!(packed_bytes[4..8], version.to_le_bytes());
        let spec_bytes = write(&with_spec, optimizer);
        assert_eq!(spec_bytes[4..8], version.to_le_bytes());
        read_raw_ckpt_header(&mut Cursor::new(&spec_bytes), &with_spec).unwrap();
        assert!(read_raw_ckpt_header(&mut Cursor::new(&spec_bytes), &partial).is_err());
        let mut header = read_raw_ckpt_header(&mut Cursor::new(&packed_bytes), &partial).unwrap();
        assert_eq!(header.step_count, 5);
        assert_eq!(header.sgd, optimizer == OptimizerKind::Sgd);
        for requested in [
            OptimizerKind::Ranger,
            OptimizerKind::RAdam,
            OptimizerKind::AdamW,
            OptimizerKind::Sgd,
        ] {
            let result =
                header.prepare_optimizer_state(Path::new("unused.ckpt"), &mut [], requested, false);
            let required_flag = match version {
                13 if requested == OptimizerKind::Sgd => Some("--transfer-radam-momentum"),
                14 if requested != OptimizerKind::Sgd => Some("--optimizer sgd"),
                15 if requested != OptimizerKind::RAdam => Some("--optimizer radam"),
                _ => None,
            };
            if let Some(flag) = required_flag {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains(flag),
                    "v{version} with {requested:?}: {error}"
                );
            } else {
                assert!(result.is_ok(), "v{version} with {requested:?}: {result:?}");
            }
            assert_eq!(header.step_count, 5);
        }
        assert!(read_raw_ckpt_header(&mut Cursor::new(&packed_bytes), &full).is_err());
        assert!(read_raw_ckpt_header(&mut Cursor::new(&full_bytes), &partial).is_err());
    }
}

#[test]
fn compact_v13_transfer_checks_producer_metadata() -> Result<(), Box<dyn std::error::Error>> {
    use nnue_train::optimizer::OptimizerKind;

    let full = layerstack_arch_threat_factorized();
    let arch = RawCkptArch {
        feature_set: full.feature_set.with_threat_width(512),
        ..full
    };
    let mut bytes = Vec::new();
    write_raw_ckpt_header(&mut bytes, &arch, "run", 6, 12, Some(20), 0)?;
    assert_eq!(bytes[4..8], 13_u32.to_le_bytes());

    let root = std::env::temp_dir().join(format!("tatara-compact-v13-{}", std::process::id()));
    let path = root.join("source.ckpt");
    std::fs::create_dir_all(root.join("experiments"))?;
    std::fs::write(&path, &bytes)?;
    let (mut header, mut groups) = load_raw_checkpoint_file(&path, &arch, &[])?;
    let transfer = |header: &mut RawCkptHeader, groups: &mut [RawCkptGroup]| {
        header.prepare_optimizer_state(&path, groups, OptimizerKind::Sgd, true)
    };
    let missing = transfer(&mut header, &mut groups).unwrap_err();
    assert!(missing.to_string().contains("requires producer metadata"));
    let metadata = root.join("experiments/run.json");
    std::fs::write(&metadata, r#"{"id":"run","params":{"optimizer":"ranger"}}"#)?;
    assert!(transfer(&mut header, &mut groups).is_err());
    assert_eq!((header.step_count, header.lr_horizon), (12, Some(20)));
    std::fs::write(&metadata, r#"{"id":"run","params":{"optimizer":"radam"}}"#)?;
    transfer(&mut header, &mut groups)?;
    assert_eq!((header.step_count, header.lr_horizon), (0, None));
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn raw_ckpt_accepts_legacy_threat_input_capacity() {
    let arch = layerstack_arch_threat_factorized();
    let mut bytes = Vec::new();
    write_raw_ckpt_header(&mut bytes, &arch, "run", 1, 5, None, 10).unwrap();
    let max_active_offset = 12 + arch.feature_set.canonical_name().len() + 16;
    let legacy_max_active = (arch.feature_set.feature_set().spec().max_active() + 128) as u64;
    let field = max_active_offset..max_active_offset + 8;
    bytes[field.clone()].copy_from_slice(&legacy_max_active.to_le_bytes());
    read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch).unwrap();
    bytes[field].copy_from_slice(&(legacy_max_active - 1).to_le_bytes());
    assert!(read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch).is_err());
}

#[test]
fn packed_ft_optimizer_payload_roundtrips_and_transfers_momentum()
-> Result<(), Box<dyn std::error::Error>> {
    use nnue_train::optimizer::OptimizerKind;
    use std::io::Write;

    let topology = layerstack_topology(8, 16, 32, 2);
    let full = layerstack_arch_threat_factorized();
    let arch = RawCkptArch {
        feature_set: full.feature_set.with_threat_width(4),
        ft_out: 8,
        topology: &topology,
        ..full
    };
    let len = arch.feature_set.train_ft_weight_len(8);
    let path = std::env::temp_dir().join(format!("tatara-packed-ft-{}.ckpt", std::process::id()));
    let mut groups = vec![(
        vec![0.25; len],
        vec![0.5; len],
        vec![0.0; len],
        vec![0.0; len],
    )];
    for optimizer in [
        OptimizerKind::Sgd,
        OptimizerKind::Ranger,
        OptimizerKind::AdamW,
        OptimizerKind::RAdam,
    ] {
        if optimizer != OptimizerKind::Sgd {
            groups[0].2.fill(0.125);
        }
        if optimizer == OptimizerKind::Ranger {
            groups[0].3.fill(0.75);
        } else {
            groups[0].3.fill(0.0);
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&path)?);
        crate::ckpt::write_raw_ckpt_header(
            &mut writer,
            &arch,
            &RawCkptMeta {
                optimizer,
                qat_dense: false,
                run_id: "",
                superbatch: 6,
                step_count: 12,
                lr_horizon: Some(20),
                qat_warmup_steps: None,
                lr_total_steps: None,
                adam_bias_correction: false,
                fv_scale: None,
                net_id: "",
                recipe_fingerprint: None,
            },
            1,
        )?;
        writer.write_all(&(len as u64).to_le_bytes())?;
        let (w, m, v, slow) = &groups[0];
        for tensor in [w, m, v, slow] {
            write_f32_slice(&mut writer, tensor)?;
        }
        writer.flush()?;

        let (mut header, mut loaded) = load_raw_checkpoint_file(&path, &arch, &[("ft_w", len)])?;
        header.prepare_optimizer_state(&path, &mut loaded, optimizer, false)?;
        assert_eq!(loaded, groups, "{optimizer:?}");
        assert_eq!(
            (header.superbatch, header.step_count, header.lr_horizon),
            (6, 12, Some(20))
        );
        if optimizer == OptimizerKind::RAdam {
            assert!(
                header
                    .prepare_optimizer_state(&path, &mut loaded, OptimizerKind::Sgd, false)
                    .is_err()
            );
            header.prepare_optimizer_state(&path, &mut loaded, OptimizerKind::Sgd, true)?;
            assert_eq!(
                (header.superbatch, header.step_count, header.lr_horizon),
                (6, 0, None)
            );
            let (loaded_w, loaded_m, loaded_v, loaded_slow) = &loaded[0];
            assert_eq!(loaded_w, w);
            assert_eq!(loaded_m, m);
            assert!(loaded_v.iter().chain(loaded_slow).all(|&x| x == 0.0));
        } else {
            let error = header
                .prepare_optimizer_state(&path, &mut loaded, OptimizerKind::Sgd, true)
                .unwrap_err();
            if optimizer != OptimizerKind::Sgd {
                assert!(error.to_string().contains("requires producer metadata"));
            }
            assert_eq!(loaded, groups);
            assert_eq!((header.step_count, header.lr_horizon), (12, Some(20)));
        }
    }
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn dense_concat_checkpoint_header_matches_only_dense_concat() {
    let split_skip = layerstack_arch();
    let mut topology = split_skip.topology.to_vec();
    topology.extend([0, nnue_train::raw_checkpoint::DENSE_CONCAT_TOPOLOGY_TAG]);
    let dense = RawCkptArch {
        topology: &topology,
        ..layerstack_arch()
    };
    for (written, other) in [(&split_skip, &dense), (&dense, &split_skip)] {
        let mut bytes = Vec::new();
        write_raw_ckpt_header(&mut bytes, written, "run", 1, 5, None, 10).unwrap();
        read_raw_ckpt_header(&mut Cursor::new(&bytes), written).unwrap();
        let error = read_raw_ckpt_header(&mut Cursor::new(&bytes), other).unwrap_err();
        assert!(error.to_string().contains("LayerStack head mismatch"));
    }
}

#[test]
fn layerstack_cli_inherits_checkpoint_head_and_rejects_mismatch() {
    use clap::Parser;
    use nnue_format::LayerStackHead::{DenseConcat, SplitSkip};
    use nnue_train::raw_checkpoint::DENSE_CONCAT_TOPOLOGY_TAG;

    let path = std::env::temp_dir().join(format!("tatara-cli-head-{}.ckpt", std::process::id()));
    let parse = |eval_only, head| {
        let mut args = vec![
            "nnue-train",
            "--resume",
            path.to_str().unwrap(),
            "layerstack",
        ];
        if eval_only {
            args.push("--eval-only");
        }
        if let Some(head) = head {
            args.extend(["--layerstack-head", head]);
        }
        crate::Cli::try_parse_from(args).unwrap()
    };

    let mut dense_topology = DEFAULT_LAYERSTACK_TOPOLOGY.to_vec();
    dense_topology.extend([0, DENSE_CONCAT_TOPOLOGY_TAG]);
    let dense_arch = RawCkptArch {
        topology: &dense_topology,
        ..layerstack_arch()
    };

    let mut dense = Vec::new();
    write_raw_ckpt_header(&mut dense, &dense_arch, "run", 1, 5, None, 10).unwrap();
    for (bytes, head, cli_value, conflicting) in [
        (dense, DenseConcat, "dense-concat", "split-skip"),
        (
            v4_layerstack_header(1, 5, 10),
            SplitSkip,
            "split-skip",
            "dense-concat",
        ),
        (
            legacy_raw_ckpt_header(1, FeatureSet::HalfKaHmMerged.spec(), None, 1, 5, 10),
            SplitSkip,
            "split-skip",
            "dense-concat",
        ),
    ] {
        std::fs::write(&path, bytes).unwrap();
        for eval_only in [false, true] {
            for explicit in [None, Some(cli_value)] {
                assert_eq!(
                    parse(eval_only, explicit)
                        .resolve_layerstack_head()
                        .unwrap(),
                    Some(head),
                );
            }
            assert_eq!(
                parse(eval_only, Some(conflicting))
                    .resolve_layerstack_head()
                    .unwrap_err(),
                format!(
                    "checkpoint head '{cli_value}' does not match --layerstack-head '{conflicting}'"
                ),
            );
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn raw_ckpt_spec_identity_roundtrip_and_mismatch() {
    for (routing, other, count) in [
        (
            "spec:handmajor3+ownking:2/3",
            "spec:handmajor3+oppking:2/3",
            9,
        ),
        (
            "spec:handmajor4+ownking:2/3/7+oppking:2/3/7",
            "spec:handmajor4+oppking:2/3/7+ownking:2/3/7",
            64,
        ),
    ] {
        let topology = layerstack_topology(DEFAULT_FT_OUT, DEFAULT_L1_OUT, DEFAULT_L2_OUT, count);
        let arch = RawCkptArch {
            topology: &topology,
            ..layerstack_arch_with_mode(routing)
        };
        let mut buf = Vec::new();
        write_raw_ckpt_header(&mut buf, &arch, "bucket-spec-roundtrip", 2, 20, None, 10).unwrap();
        read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
        let different_routing = RawCkptArch {
            topology: &topology,
            ..layerstack_arch_with_mode(other)
        };
        assert!(read_raw_ckpt_header(&mut Cursor::new(&buf), &different_routing).is_err());
    }
}

#[test]
fn forward_only_resume_skips_optimizer_step_schedule_check() {
    use crate::cli::LrScheduleArg;
    use crate::training::build_layerstack_lr_scheduler;
    use clap::Parser;

    let arch = layerstack_arch();
    let meta = RawCkptMeta {
        optimizer: nnue_train::optimizer::OptimizerKind::RAdam,
        qat_dense: false,
        run_id: "",
        superbatch: 7,
        step_count: 123,
        lr_horizon: None,
        qat_warmup_steps: None,
        lr_total_steps: Some(500),
        adam_bias_correction: false,
        fv_scale: None,
        net_id: "",
        recipe_fingerprint: None,
    };
    let mut bytes = Vec::new();
    crate::ckpt::write_raw_ckpt_header(&mut bytes, &arch, &meta, 0).unwrap();
    let header = read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch).unwrap();
    assert_eq!(header.lr_total_steps, Some(500));

    let parse = |mode: &[&str]| {
        let cli = crate::Cli::try_parse_from(
            ["nnue-train", "--resume", "model.ckpt", "layerstack"]
                .into_iter()
                .chain(mode.iter().copied()),
        )
        .unwrap();
        assert_eq!(cli.lr_schedule, LrScheduleArg::Step);
        cli
    };
    for mode in [
        ["--eval-only", "--test-data", "test.psv"].as_slice(),
        &[
            "--rescore-input",
            "input.psv",
            "--rescore-output",
            "scores",
            "--rescore-score-scale",
            "600",
        ],
    ] {
        let cli = parse(mode);
        assert_eq!(
            crate::qat::resolve_warmup(
                Some(99),
                Some(2),
                crate::qat::QatMode::Off,
                cli.resume.is_some(),
                crate::training::is_forward_only(&cli),
            )
            .unwrap(),
            Some(2),
        );
        assert!(
            build_layerstack_lr_scheduler(&cli, header.lr_horizon, header.lr_total_steps)
                .unwrap()
                .is_none()
        );
    }
    let training = parse(&["--data", "train.psv"]);
    let error = build_layerstack_lr_scheduler(&training, header.lr_horizon, header.lr_total_steps)
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("resumed optimizer-step one-cycle")
    );
}

#[test]
fn optimizer_step_schedule_checkpoint_envelope_preserves_layout_and_counter() {
    use nnue_train::raw_checkpoint::{RawCkptReadMode, RawCkptReader};
    for optimizer in [
        nnue_train::optimizer::OptimizerKind::RAdam,
        nnue_train::optimizer::OptimizerKind::Sgd,
    ] {
        let arch = layerstack_arch();
        let meta = RawCkptMeta {
            optimizer,
            qat_dense: false,
            run_id: "",
            superbatch: 7,
            step_count: 123,
            lr_horizon: None,
            qat_warmup_steps: None,
            lr_total_steps: Some(500),
            adam_bias_correction: false,
            fv_scale: None,
            net_id: "",
            recipe_fingerprint: None,
        };
        let mut bytes = Vec::new();
        crate::ckpt::write_raw_ckpt_header(&mut bytes, &arch, &meta, 0).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 16);
        let header = read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch).unwrap();
        assert_eq!(header.lr_total_steps, Some(500));
        assert_eq!(header.step_count, 123);
        assert_eq!(
            header.sgd,
            optimizer == nnue_train::optimizer::OptimizerKind::Sgd
        );
        let mut reader = RawCkptReader::new(bytes.as_slice(), RawCkptReadMode::Resume).unwrap();
        assert_eq!(reader.lr_total_steps(), Some(500));
        reader.features().unwrap();
        reader.producer_run_id().unwrap();
        reader.arch_name().unwrap();
        reader.bucket_mode().unwrap();
        for _ in 0..reader.topology_count().unwrap() {
            reader.topology_dim().unwrap();
        }
        assert_eq!(reader.superbatch().unwrap(), 7);
        assert_eq!(reader.step_count().unwrap(), 123);
        assert_eq!(reader.lr_horizon().unwrap(), None);
        reader.fv_scale().unwrap();
        reader.qat_flags().unwrap();
        reader.producer_recipe().unwrap();
        assert_eq!(reader.group_count().unwrap(), 0);
        assert!(reader.into_inner().is_empty());
    }
}

#[test]
fn adam_bias_checkpoint_roundtrip_and_resume_reject_mismatches() {
    use nnue_train::raw_checkpoint::{RawCkptReadMode, RawCkptReader};
    for corrected in [false, true] {
        for total in [None, Some(500)] {
            let arch = layerstack_arch();
            let meta = RawCkptMeta {
                optimizer: nnue_train::optimizer::OptimizerKind::AdamW,
                qat_dense: false,
                run_id: "",
                superbatch: 7,
                step_count: 123,
                lr_horizon: None,
                qat_warmup_steps: None,
                lr_total_steps: total,
                adam_bias_correction: corrected,
                fv_scale: None,
                net_id: "",
                recipe_fingerprint: None,
            };
            let mut bytes = Vec::new();
            crate::ckpt::write_raw_ckpt_header(&mut bytes, &arch, &meta, 0).unwrap();
            let wire_version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            assert_eq!(
                wire_version,
                if corrected {
                    17
                } else if total.is_some() {
                    16
                } else {
                    11
                }
            );
            let header = read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch).unwrap();
            assert_eq!(header.lr_total_steps, total);
            assert_eq!(header.step_count, 123);
            assert_eq!(header.adam_bias_correction, corrected);
            header.validate_adam_bias_correction(corrected).unwrap();
            assert!(
                header
                    .validate_adam_bias_correction(!corrected)
                    .unwrap_err()
                    .to_string()
                    .contains("--adam-bias-correction mismatch")
            );
            let reader = RawCkptReader::new(bytes.as_slice(), RawCkptReadMode::Average).unwrap();
            assert_eq!(reader.adam_bias_correction(), corrected);
            assert_eq!(reader.lr_total_steps(), total);
            if corrected {
                let mut invalid = bytes.clone();
                invalid[20] = 2;
                assert!(RawCkptReader::new(invalid.as_slice(), RawCkptReadMode::Resume).is_err());
                assert!(RawCkptReader::new(&bytes[..20], RawCkptReadMode::Resume).is_err());
            }
        }
    }
}

#[test]
fn qat_warmup_header_roundtrip_and_legacy_default() {
    for warmup in [None, Some(0), Some(2), Some(u64::MAX)] {
        for adam_bias_correction in [false, true] {
            for lr_total_steps in [None, Some(100)] {
                let arch = layerstack_arch();
                let meta = RawCkptMeta {
                    optimizer: nnue_train::optimizer::OptimizerKind::RAdam,
                    qat_dense: true,
                    run_id: "",
                    superbatch: 1,
                    step_count: 1,
                    lr_horizon: None,
                    lr_total_steps,
                    qat_warmup_steps: warmup,
                    adam_bias_correction,
                    fv_scale: None,
                    net_id: "",
                    recipe_fingerprint: None,
                };
                let mut bytes = Vec::new();
                nnue_train::raw_checkpoint::write_raw_ckpt_header(&mut bytes, &arch, &meta, 10)
                    .unwrap();
                let header = read_raw_ckpt_header(&mut Cursor::new(&bytes), &arch).unwrap();
                assert_eq!(header.qat_warmup_steps, warmup);
                assert_eq!(header.step_count, 1);
                assert_eq!(header.lr_total_steps, lr_total_steps);
                assert_eq!(header.adam_bias_correction, adam_bias_correction);
                let reader = nnue_train::raw_checkpoint::RawCkptReader::new(
                    Cursor::new(&bytes),
                    nnue_train::raw_checkpoint::RawCkptReadMode::Average,
                )
                .unwrap();
                assert_eq!(reader.qat_warmup_steps(), warmup);
            }
        }
    }
}
