//! HalfKA_hm + PSQT NNUE binary の `save_quantised` / `load`。
//!
//! `NnueHeader` (22 bytes) に続いて、FT (sparse `73_305 → 1536`) + L1 (出力
//! 直前 linear) + PSQT (feature ごとの cp) を量子化 (i8 / i16 / i32 LE) して
//! 書き出す minimal layout。本 module は single-bucket PSQT を扱う最小形で、
//! 9-bucket LayerStack 形式は `layerstack_weights` を参照。
//!
//! ## binary layout (header 22 bytes 後)
//!
//! | order | field         | type     | count                       | LE bytes |
//! |-------|---------------|----------|------------------------------|----------|
//! | 1     | `ft_weights`  | `i16`    | `NUM_FEATURES * FT_OUT_DIM` | 2 each   |
//! | 2     | `ft_bias`     | `i16`    | `FT_OUT_DIM`                | 2 each   |
//! | 3     | `l1_weights`  | `i8`     | `FT_OUT_DIM*2 * L1_OUT_DIM` | 1 each   |
//! | 4     | `l1_bias`     | `i32`    | `L1_OUT_DIM`                | 4 each   |
//! | 5     | `psqt`        | `i32`    | `NUM_FEATURES` (single bucket) | 4 each |
//!
//! 量子化は header の `qa` / `qb` を multiplier として使う:
//! - `ft_weights` / `ft_bias`: i16 量子化、multiplier = `qa`
//! - `l1_weights`: i8 量子化、multiplier = `qb`
//! - `l1_bias`: i32 量子化、multiplier = `qa * qb`
//! - `psqt`: i32 量子化、multiplier = `qa * qb` (bullet LayerStack 互換)
//!
//! ## 出典
//!
//! i8/i16/i32 量子化ロジック (`QuantTarget`) は bullet-shogi
//! `crates/trainer/src/model/save.rs::QuantTarget::quantise` を移植
//! (詳細は `ATTRIBUTION.md`)。

use std::io::{self, Read, Write};

use crate::header::NnueHeader;

/// FT (sparse) の出力次元 (NNUE 1536-16-32 の FT 部分)。
pub const FT_OUT_DIM: usize = 1536;

/// L1 (出力直前 linear) の出力次元 (single-bucket minimum: 1)。
pub const L1_OUT_DIM: usize = 1;

/// HalfKA_hm 入力次元 (`shogi_features::halfka_hm::HALFKA_HM_DIMENSIONS` と
/// 同値、本 crate を CPU-only minimal dependency に保つため独立に宣言)。
pub const NUM_FEATURES: usize = 73_305;

// =============================================================================
// QuantTarget
// =============================================================================

/// 量子化目標型 (i8/i16/i32 + multiplier)。
///
/// `quantise(round, &[f32])` で f32 → 量子化後の byte 列を返す。
/// アルゴリズムは bullet-shogi `model/save.rs::QuantTarget` から移植
/// (詳細は `ATTRIBUTION.md`)。
#[derive(Clone, Copy, Debug)]
pub enum QuantTarget {
    /// `i16` 量子化、multiplier は通常 `qa`。
    I16(i16),
    /// `i8` 量子化、multiplier は通常 `qb`。
    I8(i16),
    /// `i32` 量子化、multiplier は通常 `qa * qb` (bias) or `qa` (psqt)。
    I32(i32),
}

impl QuantTarget {
    /// `buf` を量子化して LE bytes として書き出す。`round` true なら nearest 丸め、
    /// false なら truncate。量子化結果が target 型範囲を超えた場合は
    /// `InvalidData` を返す (silent overflow 防止)。
    pub fn quantise(self, round: bool, buf: &[f32]) -> io::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(buf.len() * self.elem_bytes());

        for &float in buf {
            let elem_err = || {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed quantisation from f32 to {self:?} for {float}"),
                )
            };

            match self {
                Self::I16(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i16;
                    if qf != f64::from(x) {
                        return Err(elem_err());
                    }
                    out.extend_from_slice(&x.to_le_bytes());
                }
                Self::I8(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i8;
                    if qf != f64::from(x) {
                        return Err(elem_err());
                    }
                    out.extend_from_slice(&x.to_le_bytes());
                }
                Self::I32(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i32;
                    if qf != f64::from(x) {
                        return Err(elem_err());
                    }
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }

        Ok(out)
    }

    /// `bytes` を量子化前の f32 列にデコード。LE 読み出し → 多項倍除算。
    pub fn dequantise(self, bytes: &[u8]) -> io::Result<Vec<f32>> {
        let elem = self.elem_bytes();
        if !bytes.len().is_multiple_of(elem) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "dequantise: bytes len {} not multiple of {}",
                    bytes.len(),
                    elem
                ),
            ));
        }

        let n = bytes.len() / elem;
        let mut out = Vec::with_capacity(n);

        for i in 0..n {
            let off = i * elem;
            match self {
                Self::I16(q) => {
                    let x = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    out.push(f32::from(x) / f32::from(q));
                }
                Self::I8(q) => {
                    let x = i8::from_le_bytes([bytes[off]]);
                    out.push(f32::from(x) / f32::from(q));
                }
                Self::I32(q) => {
                    let x = i32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                    out.push(x as f32 / q as f32);
                }
            }
        }

        Ok(out)
    }

    /// 1 要素あたりの byte 数 (i8=1, i16=2, i32=4)。
    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::I8(_) => 1,
            Self::I16(_) => 2,
            Self::I32(_) => 4,
        }
    }
}

#[inline]
fn round_or_trunc(x: f64, round: bool) -> f64 {
    if round { x.round() } else { x.trunc() }
}

// =============================================================================
// HalfKAPsqtNet — minimum scope (FT + L1 + PSQT)
// =============================================================================

/// HalfKA_hm + PSQT NNUE 構造体 (single-bucket minimum scope)。
///
/// dimensional info は runtime field として `ft_out_dim` / `l1_out_dim` /
/// `num_features` で保持し、Vec の長さがこれと整合することを `validate()` で
/// 確認する。
#[derive(Clone, Debug, PartialEq)]
pub struct HalfKAPsqtNet {
    /// 入力特徴次元 (typical `NUM_FEATURES = 73_305`)。
    pub num_features: usize,
    /// FT 出力次元 (typical `FT_OUT_DIM = 1536`)。
    pub ft_out_dim: usize,
    /// L1 出力次元 (single-bucket minimum: `L1_OUT_DIM = 1`)。
    pub l1_out_dim: usize,
    /// FT weights、shape `[num_features, ft_out_dim]` を row-major で flatten。
    pub ft_weights: Vec<f32>,
    /// FT bias、長さ `ft_out_dim`。
    pub ft_bias: Vec<f32>,
    /// L1 weights、shape `[ft_out_dim * 2, l1_out_dim]` を row-major flatten
    /// (stm/nstm concat 後の linear)。
    pub l1_weights: Vec<f32>,
    /// L1 bias、長さ `l1_out_dim`。
    pub l1_bias: Vec<f32>,
    /// PSQT、長さ `num_features` (feature ごとの cp 値)。
    pub psqt: Vec<f32>,
}

impl HalfKAPsqtNet {
    /// 与えた dimension で全 weight/bias = 0 の Net を構築 (test / init 用)。
    pub fn zeros(num_features: usize, ft_out_dim: usize, l1_out_dim: usize) -> Self {
        Self {
            num_features,
            ft_out_dim,
            l1_out_dim,
            ft_weights: vec![0.0; num_features * ft_out_dim],
            ft_bias: vec![0.0; ft_out_dim],
            l1_weights: vec![0.0; ft_out_dim * 2 * l1_out_dim],
            l1_bias: vec![0.0; l1_out_dim],
            psqt: vec![0.0; num_features],
        }
    }

    /// 全 Vec の長さが宣言 dim と整合するか検証。
    pub fn validate(&self) -> io::Result<()> {
        let expected = [
            (
                "ft_weights",
                self.ft_weights.len(),
                self.num_features * self.ft_out_dim,
            ),
            ("ft_bias", self.ft_bias.len(), self.ft_out_dim),
            (
                "l1_weights",
                self.l1_weights.len(),
                self.ft_out_dim * 2 * self.l1_out_dim,
            ),
            ("l1_bias", self.l1_bias.len(), self.l1_out_dim),
            ("psqt", self.psqt.len(), self.num_features),
        ];
        for (name, got, want) in expected {
            if got != want {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{name}: expected {want} elements, got {got}"),
                ));
            }
        }
        Ok(())
    }

    /// `NnueHeader` (22 bytes) + 量子化 weight を `w` に書き込む。
    ///
    /// 量子化 multiplier は `header.qa` / `header.qb` を使う。`round` true で
    /// nearest 丸め。
    pub fn save_quantised<W: Write>(
        &self,
        w: &mut W,
        header: &NnueHeader,
        round: bool,
    ) -> io::Result<()> {
        self.validate()?;
        header.write_to(w)?;

        let qa = header.qa;
        let qb = header.qb;
        let qa_qb = i32::from(qa) * i32::from(qb);

        let bytes = QuantTarget::I16(qa).quantise(round, &self.ft_weights)?;
        w.write_all(&bytes)?;
        let bytes = QuantTarget::I16(qa).quantise(round, &self.ft_bias)?;
        w.write_all(&bytes)?;
        let bytes = QuantTarget::I8(qb).quantise(round, &self.l1_weights)?;
        w.write_all(&bytes)?;
        let bytes = QuantTarget::I32(qa_qb).quantise(round, &self.l1_bias)?;
        w.write_all(&bytes)?;
        let bytes = QuantTarget::I32(qa_qb).quantise(round, &self.psqt)?;
        w.write_all(&bytes)?;

        Ok(())
    }

    /// header → quantised weight の順で `r` から読み、dequantise して
    /// `HalfKAPsqtNet` を再構築。`num_features` / `ft_out_dim` / `l1_out_dim` は
    /// 呼び出し側が指定 (header に含まれないため)。
    pub fn load<R: Read>(
        r: &mut R,
        num_features: usize,
        ft_out_dim: usize,
        l1_out_dim: usize,
    ) -> io::Result<(NnueHeader, Self)> {
        let header = NnueHeader::read_from(r)?;
        let qa = header.qa;
        let qb = header.qb;
        let qa_qb = i32::from(qa) * i32::from(qb);

        let ft_weights = read_dequant_block(r, num_features * ft_out_dim, QuantTarget::I16(qa))?;
        let ft_bias = read_dequant_block(r, ft_out_dim, QuantTarget::I16(qa))?;
        let l1_weights = read_dequant_block(r, ft_out_dim * 2 * l1_out_dim, QuantTarget::I8(qb))?;
        let l1_bias = read_dequant_block(r, l1_out_dim, QuantTarget::I32(qa_qb))?;
        // PSQT multiplier `qa * qb` (save_quantised 側と対称)。
        let psqt = read_dequant_block(r, num_features, QuantTarget::I32(qa_qb))?;

        Ok((
            header,
            Self {
                num_features,
                ft_out_dim,
                l1_out_dim,
                ft_weights,
                ft_bias,
                l1_weights,
                l1_bias,
                psqt,
            },
        ))
    }
}

fn read_dequant_block<R: Read>(
    r: &mut R,
    n_elements: usize,
    target: QuantTarget,
) -> io::Result<Vec<f32>> {
    let n_bytes = n_elements * target.elem_bytes();
    let mut bytes = vec![0u8; n_bytes];
    r.read_exact(&mut bytes)?;
    target.dequantise(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::HEADER_BYTES;
    use std::io::Cursor;

    /// 単純な round-trip 検証用 mini net (production 73_305 * 1536 は test に
    /// 重すぎるため、num_features=4 / ft_out_dim=2 / l1_out_dim=1 の小型版)。
    fn build_mini_net() -> HalfKAPsqtNet {
        HalfKAPsqtNet {
            num_features: 4,
            ft_out_dim: 2,
            l1_out_dim: 1,
            ft_weights: vec![0.5, -0.25, 0.125, 0.0, -0.5, 0.25, 0.75, -0.75], // 4*2 = 8
            ft_bias: vec![0.5, -0.5],                                          // 2
            l1_weights: vec![0.0625, -0.0625, 0.125, -0.125],                  // 2*2*1 = 4
            l1_bias: vec![0.0],                                                // 1
            psqt: vec![1.0, -1.0, 0.5, -0.5],                                  // 4
        }
    }

    #[test]
    fn validate_passes_for_consistent_dims() {
        build_mini_net().validate().unwrap();
    }

    #[test]
    fn validate_rejects_dim_mismatch() {
        let mut net = build_mini_net();
        net.ft_weights.push(0.0);
        let err = net.validate().expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("ft_weights"));
    }

    #[test]
    fn quant_target_i16_round_trip_exact_within_range() {
        // QA=64 で float = 0.5 → qf=32 → i16(32) → dequant 32/64=0.5、完全一致
        let buf = vec![0.5_f32, -0.5, 0.125, 0.0];
        let bytes = QuantTarget::I16(64).quantise(true, &buf).unwrap();
        assert_eq!(bytes.len(), 4 * 2);
        let back = QuantTarget::I16(64).dequantise(&bytes).unwrap();
        assert_eq!(back, buf);
    }

    #[test]
    fn quant_target_i16_rejects_out_of_range() {
        // QA=64、float=10000 → qf=640000、i16 range [-32768, 32767] 超
        let err = QuantTarget::I16(64)
            .quantise(true, &[10000.0])
            .expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn quant_target_i8_round_trip() {
        let buf = vec![0.25_f32, -0.5, 0.125];
        let bytes = QuantTarget::I8(64).quantise(true, &buf).unwrap();
        assert_eq!(bytes.len(), 3);
        let back = QuantTarget::I8(64).dequantise(&bytes).unwrap();
        assert_eq!(back, buf);
    }

    #[test]
    fn quant_target_i32_round_trip() {
        let buf = vec![1.0_f32, -1.0, 0.5, -0.5];
        let bytes = QuantTarget::I32(1024).quantise(true, &buf).unwrap();
        assert_eq!(bytes.len(), 4 * 4);
        let back = QuantTarget::I32(1024).dequantise(&bytes).unwrap();
        assert_eq!(back, buf);
    }

    #[test]
    fn save_quantised_then_load_round_trips_mini_net() {
        let net = build_mini_net();
        let header = NnueHeader {
            net_id: "mini".to_string(),
            fv_scale: 16,
            qa: 64,
            qb: 64,
        };

        let mut buf = Vec::new();
        net.save_quantised(&mut buf, &header, true).unwrap();

        // header (22) + ft_w 8*2 + ft_b 2*2 + l1_w 4 + l1_b 4 + psqt 4*4
        //  = 22 + 16 + 4 + 4 + 4 + 16 = 66 bytes
        assert_eq!(buf.len(), HEADER_BYTES + 8 * 2 + 2 * 2 + 4 + 4 + 4 * 4);

        let (rh, rn) = HalfKAPsqtNet::load(
            &mut Cursor::new(&buf),
            net.num_features,
            net.ft_out_dim,
            net.l1_out_dim,
        )
        .unwrap();

        assert_eq!(rh, header);
        // 量子化値は元の mini_net 値 (0.5, 0.125 等、QA=64 で完全表現可) と完全一致。
        assert_eq!(rn, net);
    }

    #[test]
    fn save_quantised_with_non_default_qa_qb_round_trips() {
        let net = HalfKAPsqtNet {
            num_features: 2,
            ft_out_dim: 2,
            l1_out_dim: 1,
            ft_weights: vec![0.5, -0.5, 0.25, 0.0], // 量子化 lossless
            ft_bias: vec![0.0, 0.0],
            l1_weights: vec![0.0, 0.0, 0.0, 0.0],
            l1_bias: vec![0.0],
            psqt: vec![0.0, 0.0],
        };
        let header = NnueHeader {
            net_id: "qa_qb_test".to_string(),
            fv_scale: 16,
            qa: 16,
            qb: 32,
        };

        let mut buf = Vec::new();
        net.save_quantised(&mut buf, &header, true).unwrap();

        let (rh, rn) = HalfKAPsqtNet::load(
            &mut Cursor::new(&buf),
            net.num_features,
            net.ft_out_dim,
            net.l1_out_dim,
        )
        .unwrap();
        assert_eq!(rh.qa, 16);
        assert_eq!(rh.qb, 32);
        assert_eq!(rn, net);
    }

    #[test]
    fn save_quantised_rejects_dim_mismatch() {
        let mut net = build_mini_net();
        net.psqt.pop(); // 3 elements instead of 4
        let header = NnueHeader::default();
        let err = net
            .save_quantised(&mut Vec::new(), &header, true)
            .expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("psqt"));
    }

    #[test]
    fn num_features_const_matches_halfka_hm_dimensions() {
        // shogi-features::HALFKA_HM_DIMENSIONS = 73_305 と整合確認
        // (本 crate は shogi-features に depend しないが、定数値の同期は重要)。
        assert_eq!(NUM_FEATURES, 73_305);
    }

    #[test]
    fn dequantise_rejects_unaligned_len() {
        // I32 (4 bytes/elem) なのに 3 bytes 渡すと InvalidData。
        let err = QuantTarget::I32(64)
            .dequantise(&[0, 0, 0])
            .expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let err = QuantTarget::I16(64)
            .dequantise(&[0, 0, 0])
            .expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_truncated_input() {
        // header の直後で buf を切ると `UnexpectedEof` で読み損ねを検知する。
        let net = build_mini_net();
        let header = NnueHeader::default();

        let mut buf = Vec::new();
        net.save_quantised(&mut buf, &header, true).unwrap();

        // header 22 bytes + 数 bytes だけ残して残り切り捨て。
        let truncated = &buf[..HEADER_BYTES + 4];
        let err = HalfKAPsqtNet::load(
            &mut Cursor::new(truncated),
            net.num_features,
            net.ft_out_dim,
            net.l1_out_dim,
        )
        .expect_err("must fail on truncated buffer");
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}
