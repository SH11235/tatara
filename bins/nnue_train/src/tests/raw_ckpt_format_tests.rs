//! raw checkpoint format helper tests (GPU 不要)。

use nnue_format::ArchKind;
use shogi_features::{EffectBucketConfig, FeatureSet};

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
    // magic は format identity。version は後方互換読み (version 1..=6 file の受理)
    // を維持しつつ前進するので、現行値を pin して意図しない変更を検出する。
    assert_eq!(&RAW_CKPT_MAGIC, b"RNRC");
    assert_eq!(RAW_CKPT_VERSION, 7);
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

/// 既定 FT / L1 / L2 出力次元 + 既定 bucket 数の LayerStack topology
/// (test helper、`'static` 借用用)。
const DEFAULT_LAYERSTACK_TOPOLOGY: [u64; 4] = layerstack_topology(
    DEFAULT_FT_OUT,
    DEFAULT_L1_OUT,
    DEFAULT_L2_OUT,
    DEFAULT_NUM_BUCKETS,
);

fn layerstack_arch() -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged.spec(),
        arch_kind: ArchKind::LayerStack,
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

/// `layerstack_arch` の FT factorizer 有効版。
fn layerstack_arch_factorized() -> RawCkptArch<'static> {
    RawCkptArch {
        feature_set: FeatureSet::HalfKaHmMerged.spec().with_ft_factorize(),
        arch_kind: ArchKind::LayerStack,
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
        ft_out: DEFAULT_FT_OUT as u64,
        topology: &DEFAULT_LAYERSTACK_TOPOLOGY,
    }
}

fn remove_v7_feature_hash(buf: &mut Vec<u8>, arch: &RawCkptArch) {
    buf[4..8].copy_from_slice(&6u32.to_le_bytes());
    let max_active_off = 4 + 4 + 4 + arch.feature_set.canonical_name().len() + 8 + 8;
    let feature_hash_off = max_active_off + 8 + 1;
    buf.drain(feature_hash_off..feature_hash_off + 4);
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
    let h = read_raw_ckpt_header(&mut Cursor::new(&buf), &arch).unwrap();
    assert_eq!(h.superbatch, 7);
    assert_eq!(h.step_count, 99);
    assert_eq!(h.num_groups, 10);
    assert_eq!(h.producer_run_id.as_deref(), Some("net-20260520-1234"));
    assert_eq!(h.lr_horizon, Some(123));
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
