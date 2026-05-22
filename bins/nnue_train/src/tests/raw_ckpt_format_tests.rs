//! raw checkpoint format helper tests (GPU 不要)。

use nnue_format::ArchKind;
use shogi_features::FeatureSet;

use crate::{arch::*, ckpt::*};

use std::io::Cursor;

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
    // magic は format identity。version は後方互換読み (version 1..=3 file の受理)
    // を維持しつつ前進するので、現行値を pin して意図しない変更を検出する。
    assert_eq!(&RAW_CKPT_MAGIC, b"RNRC");
    assert_eq!(RAW_CKPT_VERSION, 4);
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

/// 既定 FT / L1 / L2 出力次元の LayerStack topology (test helper、`'static` 借用用)。
const DEFAULT_LAYERSTACK_TOPOLOGY: [u64; 4] =
    layerstack_topology(DEFAULT_FT_OUT, DEFAULT_L1_OUT, DEFAULT_L2_OUT);

fn layerstack_arch() -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        arch_kind: ArchKind::LayerStack,
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

#[test]
fn raw_ckpt_header_v4_round_trips() {
    let arch = layerstack_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "net-20260520-1234", 7, 99, 10).unwrap();
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(h.superbatch, 7);
    assert_eq!(h.step_count, 99);
    assert_eq!(h.num_groups, 10);
    assert_eq!(h.producer_run_id.as_deref(), Some("net-20260520-1234"));
}

#[test]
fn raw_ckpt_header_empty_run_id_round_trips_to_none() {
    let arch = layerstack_arch();
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &arch, "", 1, 0, 10).unwrap();
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
}

#[test]
fn raw_ckpt_header_rejects_wrong_arch_kind() {
    // fs header は一致するが arch_kind だけ異なる v4 header を read → reject。
    let fs = FeatureSet::HalfKaHmMerged.spec();
    let written = RawCkptArch {
        feature_set: fs,
        arch_kind: ArchKind::Simple,
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    };
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &written, "", 1, 0, 10).unwrap();
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
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &wrong_topo,
    };
    let mut buf = Vec::new();
    write_raw_ckpt_header(&mut buf, &written, "", 1, 0, 10).unwrap();
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
