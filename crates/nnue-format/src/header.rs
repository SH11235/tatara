//! NNUE binary header の (de)serialise。
//!
//! output binary 先頭に置く固定長 22 bytes metadata。`halfka_psqt` の
//! save/load で weight 本体の前に書かれ、将棋エンジン側 loader が読む。
//!
//! ## binary layout (固定長 22 bytes、little-endian)
//!
//! | offset | size | field      | encoding                                 |
//! |--------|------|------------|------------------------------------------|
//! | 0      | 16   | `net_id`   | UTF-8 bytes、`\0` padding、null-terminated |
//! | 16     | 2    | `fv_scale` | `i16` LE                                 |
//! | 18     | 2    | `qa`       | `i16` LE                                 |
//! | 20     | 2    | `qb`       | `i16` LE                                 |
//!
//! `net_id` は **最長 15 bytes UTF-8 + 末尾 `\0`** (= 計 16 bytes)。`String`
//! 中の UTF-8 bytes が 15 を超える場合は `write_to` が `InvalidInput` を返す
//! (silent truncate しない、weight 損失防止のため)。
//!
//! ## 既定値
//!
//! - `net_id`: 空文字列
//! - `fv_scale = 16` — centipawn 単位への scale
//! - `qa = 64`、`qb = 64` — input / output quantisation multiplier (典型値、
//!   actual quant value は trainer 側で上書きする)

use std::io::{self, Read, Write};

/// Header の固定 byte 長 (16 + 2 + 2 + 2 = 22)。
pub const HEADER_BYTES: usize = 22;

/// `net_id` 領域の長さ (UTF-8 bytes、`\0` padding 込み)。
///
/// 実 string は最大 `NET_ID_LEN - 1 = 15` bytes、末尾 `\0` 必須。
pub const NET_ID_LEN: usize = 16;

/// `fv_scale` 既定値 (centipawn スケール、典型値)。
pub const DEFAULT_FV_SCALE: i16 = 16;

/// `qa` 既定値 (input 量子化、trainer 側で実値に上書き)。
pub const DEFAULT_QA: i16 = 64;

/// `qb` 既定値 (output 量子化、trainer 側で実値に上書き)。
pub const DEFAULT_QB: i16 = 64;

/// NNUE binary 先頭 22 bytes の固定長 header。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NnueHeader {
    /// network identifier (最大 15 bytes UTF-8 + 末尾 `\0` の 16 bytes 領域に格納)。
    pub net_id: String,
    /// centipawn スケール (典型 16)。
    pub fv_scale: i16,
    /// input quantisation multiplier (trainer 側で実値に更新)。
    pub qa: i16,
    /// output quantisation multiplier (trainer 側で実値に更新)。
    pub qb: i16,
}

impl Default for NnueHeader {
    fn default() -> Self {
        Self {
            net_id: String::new(),
            fv_scale: DEFAULT_FV_SCALE,
            qa: DEFAULT_QA,
            qb: DEFAULT_QB,
        }
    }
}

impl NnueHeader {
    /// 22 bytes を `w` に書き込む。`net_id` が 15 bytes (UTF-8) を超える場合は
    /// `io::ErrorKind::InvalidInput` を返す (silent truncate しない)。
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let id_bytes = self.net_id.as_bytes();
        if id_bytes.len() >= NET_ID_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "net_id is {} bytes; must be <= {} bytes (15 UTF-8 + 1 NUL terminator)",
                    id_bytes.len(),
                    NET_ID_LEN - 1
                ),
            ));
        }

        let mut buf = [0u8; HEADER_BYTES];
        buf[..id_bytes.len()].copy_from_slice(id_bytes);
        // [id_bytes.len()..NET_ID_LEN] は zero padding (含む末尾 \0)。
        buf[NET_ID_LEN..NET_ID_LEN + 2].copy_from_slice(&self.fv_scale.to_le_bytes());
        buf[NET_ID_LEN + 2..NET_ID_LEN + 4].copy_from_slice(&self.qa.to_le_bytes());
        buf[NET_ID_LEN + 4..NET_ID_LEN + 6].copy_from_slice(&self.qb.to_le_bytes());

        w.write_all(&buf)
    }

    /// 22 bytes を `r` から読んで `NnueHeader` を返す。
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_BYTES];
        r.read_exact(&mut buf)?;

        // net_id: \0 終端まで (含まず) を UTF-8 として decode。
        let id_end = buf[..NET_ID_LEN]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(NET_ID_LEN);
        let net_id = std::str::from_utf8(&buf[..id_end])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .to_string();

        let fv_scale = i16::from_le_bytes([buf[NET_ID_LEN], buf[NET_ID_LEN + 1]]);
        let qa = i16::from_le_bytes([buf[NET_ID_LEN + 2], buf[NET_ID_LEN + 3]]);
        let qb = i16::from_le_bytes([buf[NET_ID_LEN + 4], buf[NET_ID_LEN + 5]]);

        Ok(Self {
            net_id,
            fv_scale,
            qa,
            qb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn default_values_are_typical_constants() {
        let h = NnueHeader::default();
        assert_eq!(h.net_id, "");
        assert_eq!(h.fv_scale, 16);
        assert_eq!(h.qa, 64);
        assert_eq!(h.qb, 64);
    }

    #[test]
    fn round_trip_default() {
        let h = NnueHeader::default();
        let mut buf = Vec::new();
        h.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_BYTES);
        let r = NnueHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(r, h);
    }

    #[test]
    fn round_trip_with_non_empty_id() {
        let h = NnueHeader {
            net_id: "halfka_hm_v2".to_string(),
            fv_scale: 16,
            qa: 255,
            qb: 64,
        };
        let mut buf = Vec::new();
        h.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_BYTES);

        let r = NnueHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(r, h);
    }

    #[test]
    fn round_trip_max_length_id() {
        // 15 bytes ちょうど (NET_ID_LEN - 1) は OK、末尾 \0 で 16 bytes に揃う。
        let id = "a".repeat(NET_ID_LEN - 1);
        let h = NnueHeader {
            net_id: id.clone(),
            ..NnueHeader::default()
        };
        let mut buf = Vec::new();
        h.write_to(&mut buf).unwrap();
        let r = NnueHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(r.net_id, id);
    }

    #[test]
    fn write_rejects_too_long_id() {
        // 16 bytes (= NET_ID_LEN) は NUL 余地なしで reject。
        let id = "a".repeat(NET_ID_LEN);
        let h = NnueHeader {
            net_id: id,
            ..NnueHeader::default()
        };
        let mut buf = Vec::new();
        let err = h.write_to(&mut buf).expect_err("must reject 16-byte id");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn write_layout_is_22_bytes_little_endian() {
        // 明示 byte-level 検証。
        let h = NnueHeader {
            net_id: "abc".to_string(),
            fv_scale: 0x0102,
            qa: 0x0304,
            qb: 0x0506,
        };
        let mut buf = Vec::new();
        h.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_BYTES);

        // net_id "abc" + 13 bytes of NUL padding
        assert_eq!(&buf[..3], b"abc");
        assert!(buf[3..NET_ID_LEN].iter().all(|&b| b == 0));

        // i16 LE: 0x0102 -> 0x02 0x01
        assert_eq!(&buf[NET_ID_LEN..NET_ID_LEN + 2], &[0x02, 0x01]);
        assert_eq!(&buf[NET_ID_LEN + 2..NET_ID_LEN + 4], &[0x04, 0x03]);
        assert_eq!(&buf[NET_ID_LEN + 4..NET_ID_LEN + 6], &[0x06, 0x05]);
    }

    #[test]
    fn read_handles_short_id_with_nul_padding() {
        // 手動で構築した buffer (id="x" + 15 NUL + i16 LE × 3) からの read 検証。
        let mut buf = [0u8; HEADER_BYTES];
        buf[0] = b'x';
        buf[NET_ID_LEN..NET_ID_LEN + 2].copy_from_slice(&16i16.to_le_bytes());
        buf[NET_ID_LEN + 2..NET_ID_LEN + 4].copy_from_slice(&255i16.to_le_bytes());
        buf[NET_ID_LEN + 4..NET_ID_LEN + 6].copy_from_slice(&64i16.to_le_bytes());

        let h = NnueHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h.net_id, "x");
        assert_eq!(h.fv_scale, 16);
        assert_eq!(h.qa, 255);
        assert_eq!(h.qb, 64);
    }

    #[test]
    fn read_handles_id_without_nul_terminator() {
        // 16 bytes 全てが非-NUL の場合 (find が None) は領域全体を id として扱う。
        // write_to が reject する経路だが、read 側は defensive に解釈する。
        let mut buf = [0u8; HEADER_BYTES];
        buf[..NET_ID_LEN].copy_from_slice(&[b'y'; NET_ID_LEN]);
        // i16 LE × 3 は default 値 (16/64/64) に置く。
        buf[NET_ID_LEN..NET_ID_LEN + 2].copy_from_slice(&16i16.to_le_bytes());
        buf[NET_ID_LEN + 2..NET_ID_LEN + 4].copy_from_slice(&64i16.to_le_bytes());
        buf[NET_ID_LEN + 4..NET_ID_LEN + 6].copy_from_slice(&64i16.to_le_bytes());

        let h = NnueHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(h.net_id, "y".repeat(NET_ID_LEN));
    }
}
