//! bullet-shogi v102 互換 LayerStack NNUE binary の save / load。
//!
//! `bins/nnue_train` が出力する binary 形式。bullet-shogi v102 と byte 互換に
//! することで、rshogi-oss 等の推論エンジンが `EvalFile=` で直接読み込めるように
//! 設計されている (出典・参照実装は `ATTRIBUTION.md`)。
//!
//! ## v102 architecture (PSQT 無し、Threat 無し、HandCountDense 無し)
//!
//! - L0 (FT): `73305 → 1536`、weight + bias 共有 stm/nstm
//! - per-perspective post: bias add → CReLU → pairwise_mul (1536 → 768) → ×127/128
//! - combined = stm.concat(nstm) = 1536
//! - L1 (per-bucket delta + shared l1f factorized): 9 × 16 + (1536, 16)
//! - l1_out_t = L1.select(bucket) + L1f、main + skip に slice
//! - L2 (per-bucket): 9 × 32 with input l2_in = (l1_effective)*2 = 30
//! - L3 (per-bucket output): 9 × 1
//!
//! ## file layout (top-level)
//!
//! 1. header: nnue_version (4 LE u32) + network_hash (4 LE u32) + arch_len (4 LE u32) + arch_str
//! 2. ft_hash (4 LE u32)
//! 3. ft_biases LEB128 (magic `COMPRESSED_LEB128` + size + signed LEB128 i16 列)
//! 4. ft_weights LEB128 (同上、piece 部分 = `halfka_dim * ft_out`、threat 無し)
//! 5. layerstacks: 9 bucket × {fc_hash (4 LE u32), L1 (bias + weight), L2 (同), L3 (同)}
//!
//! ## save 時の L1 / L1f coalesce
//!
//! per-bucket l1 と shared l1f を **save 時に merge** して per-bucket の単一
//! weight として書き出す:
//!
//! - `l1_bias_merged[bucket][out] = l1_b[bucket][out] + l1f_b[out]` (bias broadcast)
//! - `l1_weight_merged[bucket][out][in] = l1_w[bucket][out][in] + l1f_w[in][out]` (in/out 軸入替注意)
//!
//! rshogi-oss は `Factorizer` を含む arch を reject する (coalesced only を要求)
//! ため、save 時に必ず merge する不変条件。
//!
//! ## 量子化 scale
//!
//! | layer | bias scale | weight scale |
//! |---|---|---|
//! | FT | QA = 127 (i16) | QA = 127 (i16, LEB128 圧縮) |
//! | L1 (merged) | QA * QB = 8128 (i32) | QB = 64 (i8) |
//! | L2 | 127 * QB = 8128 (i32) | QB = 64 (i8) |
//! | L3 (output) | 127 * QB = 8128 (i32) | QB = 64 (i8) |
//!
//! L2 / L3 bias scale は形式上 `127 * QB` (input が CReLU 後の 127-scale 量子化
//! 値として扱われるため)。`QA == 127` 前提では結果値は `QA * QB` と同じ。
//!
//! ## pad32
//!
//! 各 layer の input dim は **32 の倍数に pad** されて i8 weight が書き出される
//! (SIMD load の align 要求のため)。pad32(1536) = 1536, pad32(30) = 32, pad32(32) = 32。
//! padding byte は 0 で埋める。
//!
//! ## 重み layout の差分 (bullet 内部 vs file)
//!
//! bullet 内部はすべて **column-major** (`w[in * rows + out]`)。file は **row-major**
//! per bucket (`for out in 0..out_dim: for in in 0..padded_in: write byte`)。
//! 本 crate のトレーナー側 weight は row-major (`l1_w[bucket * out_dim * in_dim
//! + out_idx * in_dim + in_idx]`) なので、そのまま file row-major に書ける (転置不要)。

use std::io::{self, Read, Write};

// =============================================================================
// constants (v102 architecture)
// =============================================================================

pub const NNUE_VERSION: u32 = 0x7AF32F20;
pub const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

pub const FT_IN: usize = 73_305; // HalfKA_hm dimensions
pub const FT_OUT: usize = 1536;
pub const L1_OUT: usize = 16;
pub const L1_EFFECTIVE: usize = L1_OUT - 1; // = 15
pub const L2_IN: usize = L1_EFFECTIVE * 2; // = 30
pub const L2_OUT: usize = 32;
pub const NUM_BUCKETS: usize = 9;

pub const QA: i32 = 127;
pub const QB: i32 = 64;
pub const FV_SCALE: i32 = 28;

/// `(127.0 / 290.0) * 28 == 12.262...` の denominator。
/// 推論エンジン側は arch_str から `fv_scale=28` を読み、本 SCALE と組み合わせる。
pub const SCALE: u32 = 290;

/// pad to multiple of 32 (SIMD alignment)。
#[inline]
pub fn pad32(x: usize) -> usize {
    x.div_ceil(32) * 32
}

// =============================================================================
// LEB128 encode (signed)
// =============================================================================

/// 符号付き LEB128 で `val` を `out` に append。
/// 推論エンジン側 (rshogi-oss `nnue/leb128.rs::read_signed_leb128`) で
/// 逆方向 decode できる形式。
pub fn encode_signed_leb128(val: i64, out: &mut Vec<u8>) {
    let mut value = val;
    loop {
        let byte = (value & 0x7f) as u8;
        // 算術右シフト (i64 は arithmetic shift で sign extension)
        value >>= 7;
        let sign_bit = (byte & 0x40) != 0;
        // more bytes needed?
        let more = (value != 0 || sign_bit) && (value != -1 || !sign_bit);
        if more {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            return;
        }
    }
}

/// i16 tensor を LEB128 magic + size + 圧縮データ形式で `out` に書く。
/// 推論エンジン側 (rshogi-oss `read_compressed_tensor_i16_all`) で読み戻せる。
pub fn write_leb128_tensor_i16<W: Write>(out: &mut W, values: &[i16]) -> io::Result<()> {
    let mut compressed = Vec::with_capacity(values.len() * 2);
    for &v in values {
        encode_signed_leb128(v as i64, &mut compressed);
    }
    out.write_all(LEB128_MAGIC)?;
    out.write_all(&(compressed.len() as u32).to_le_bytes())?;
    out.write_all(&compressed)?;
    Ok(())
}

/// LEB128 圧縮ブロックから i16 列を読み戻す。`expected_count` で sanity check
/// (なければ skip)。
pub fn read_leb128_tensor_i16<R: Read>(
    reader: &mut R,
    expected_count: Option<usize>,
) -> io::Result<Vec<i16>> {
    let mut magic = [0u8; 17];
    reader.read_exact(&mut magic)?;
    if magic != LEB128_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected COMPRESSED_LEB128 magic",
        ));
    }
    let mut size_buf = [0u8; 4];
    reader.read_exact(&mut size_buf)?;
    let compressed_size = u32::from_le_bytes(size_buf) as usize;
    if compressed_size == 0 || compressed_size > 256 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid LEB128 block size: {compressed_size}"),
        ));
    }
    let mut compressed = vec![0u8; compressed_size];
    reader.read_exact(&mut compressed)?;

    let mut result = Vec::new();
    let mut pos = 0;
    while pos < compressed.len() {
        let (val, consumed) = decode_single_leb128(&compressed[pos..])?;
        result.push(val as i16);
        pos += consumed;
    }
    if let Some(n) = expected_count
        && result.len() != n
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "LEB128 tensor count mismatch: expected {n}, got {}",
                result.len()
            ),
        ));
    }
    Ok(result)
}

fn decode_single_leb128(data: &[u8]) -> io::Result<(i64, usize)> {
    let mut result: i64 = 0;
    let mut shift = 0;
    let mut pos = 0;
    loop {
        if pos >= data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of LEB128 data",
            ));
        }
        let b = data[pos];
        pos += 1;
        result |= ((b & 0x7f) as i64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            if shift < 64 && (b & 0x40) != 0 {
                result |= !0i64 << shift;
            }
            return Ok((result, pos));
        }
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LEB128 overflow",
            ));
        }
    }
}

// =============================================================================
// arch_str + hash
// =============================================================================

/// v102 arch description string を生成。PSQT 無し / Threat 無し /
/// HandCountDense 無しの最小形 (bullet v102 由来、`ATTRIBUTION.md` 参照)。
///
/// 形式 (実際は改行無しの 1 行):
///
/// `Features=HalfKA_hm(Friend)[<input_size>-><ft_out>x2],
///  Network=AffineTransform[1<-<l2_out>](
///  ClippedReLU[<l2_out>](
///  AffineTransform[<l2_out>-<l2_in>](
///  SqrClippedReLU[<l2_in>](
///  AffineTransform[<l1_out>-<ft_out_x2>](
///  InputSlice[<ft_out_x2>(0:<ft_out_x2>)]))))),
///  fv_scale=<fv_scale>`
///
/// (実際は 1 行連結、ここでは可読性のため改行)
pub fn build_arch_str(
    input_size: usize,
    ft_out: usize,
    l1_out: usize,
    l2_in: usize,
    l2_out: usize,
    fv_scale: i32,
) -> String {
    format!(
        "Features=HalfKA_hm(Friend)[{}->{}x2],\
         Network=AffineTransform[1<-{}](\
         ClippedReLU[{}](\
         AffineTransform[{}<-{}](\
         SqrClippedReLU[{}](\
         AffineTransform[{}<-{}](\
         InputSlice[{}(0:{})]))))),\
         fv_scale={}",
        input_size,
        ft_out,
        l2_out,     // Output input
        l2_out,     // L2 output / L3 input
        l2_out,     // L2 output
        l2_in,      // L2 input
        l2_in,      // dual activation output
        l1_out,     // L1 output
        ft_out * 2, // L1 input (dual perspective)
        ft_out * 2,
        ft_out * 2,
        fv_scale,
    )
}

/// nnue-pytorch 互換 fc hash 計算。
///
/// **注**: bullet-shogi 由来の関数名 (`compute_layerstack_fc_hash(l1_out, l2_in,
/// l2_out)`) は misleading で、第 1 引数は実際には `FT_OUT` (= 1536)。本実装は
/// 引数名を `ft_out` に統一して命名を揃える。
///
/// 推論エンジン側 (rshogi-oss `network_layer_stacks.rs`) は本 hash を skip
/// するが、bullet 出力との byte 完全互換のために computed value を使う。
pub const fn compute_fc_hash(ft_out: usize, _l2_in: usize, l2_out: usize) -> u32 {
    // InputSlice hash (FT output × 2 dual perspective を XOR)
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= (ft_out * 2) as u32;

    // layer_sizes: 第 1 要素 = ft_out (has_relu=true)、
    // 第 2 要素 = l2_out (has_relu=true)、第 3 要素 = 1 (has_relu=false、出力)。
    // const fn なので `for` イテレータは使えず index ベースの `while` で回す。
    let layer_sizes = [(ft_out, true), (l2_out, true), (1_usize, false)];
    let mut i = 0;
    while i < layer_sizes.len() {
        let (out_features, has_relu) = layer_sizes[i];
        let mut layer_hash: u32 = 0xCC03DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;
        if has_relu {
            layer_hash = layer_hash.wrapping_add(0x538D24C7);
        }
        prev_hash = layer_hash;
        i += 1;
    }
    prev_hash
}

/// FT hash: `FEATURE_HASH_HM_V2 ^ (ft_out * 2)`。
pub const FT_HASH: u32 = 0x7f134cb8 ^ (FT_OUT as u32 * 2);

/// per-bucket fc_hash。v102 の (ft_out=1536, l2_in=30, l2_out=32) 固定値を
/// `compute_fc_hash` (const fn) で評価。
pub const FC_HASH: u32 = compute_fc_hash(FT_OUT, L2_IN, L2_OUT);

pub const NETWORK_HASH: u32 = FC_HASH ^ FT_HASH;

// =============================================================================
// V102Weights — トレーナー側 weight 表現 (f32、kernel と同 layout)
// =============================================================================

/// v102 LayerStack の全 weight (f32、host 側保持)。
///
/// Layout は本 crate trainer の kernel 内部 layout と一致:
/// - `ft_w`: `(FT_IN, FT_OUT)` row-major、`ft_w[feat * FT_OUT + out]`
/// - `ft_b`: `(FT_OUT)` (stm/nstm 共有)
/// - `l1_w`: `(NUM_BUCKETS, L1_OUT, FT_OUT)` row-major、`l1_w[buc * L1_OUT * FT_OUT + out * FT_OUT + in]`
/// - `l1_b`: `(NUM_BUCKETS, L1_OUT)` row-major
/// - `l1f_w`: `(FT_OUT, L1_OUT)` row-major、`l1f_w[in * L1_OUT + out]`
/// - `l1f_b`: `(L1_OUT)`
/// - `l2_w`: `(NUM_BUCKETS, L2_OUT, L2_IN)` row-major
/// - `l2_b`: `(NUM_BUCKETS, L2_OUT)`
/// - `l3_w`: `(NUM_BUCKETS, L2_OUT)` (out_dim=1 なので out 軸省略)
/// - `l3_b`: `(NUM_BUCKETS)`
#[derive(Debug, Clone)]
pub struct V102Weights {
    pub ft_w: Vec<f32>,
    pub ft_b: Vec<f32>,
    pub l1_w: Vec<f32>,
    pub l1_b: Vec<f32>,
    pub l1f_w: Vec<f32>,
    pub l1f_b: Vec<f32>,
    pub l2_w: Vec<f32>,
    pub l2_b: Vec<f32>,
    pub l3_w: Vec<f32>,
    pub l3_b: Vec<f32>,
}

impl V102Weights {
    /// 全 buffer を 0 で初期化した新規 instance。
    pub fn zeroed() -> Self {
        Self {
            ft_w: vec![0.0; FT_IN * FT_OUT],
            ft_b: vec![0.0; FT_OUT],
            l1_w: vec![0.0; NUM_BUCKETS * L1_OUT * FT_OUT],
            l1_b: vec![0.0; NUM_BUCKETS * L1_OUT],
            l1f_w: vec![0.0; FT_OUT * L1_OUT],
            l1f_b: vec![0.0; L1_OUT],
            l2_w: vec![0.0; NUM_BUCKETS * L2_OUT * L2_IN],
            l2_b: vec![0.0; NUM_BUCKETS * L2_OUT],
            l3_w: vec![0.0; NUM_BUCKETS * L2_OUT],
            l3_b: vec![0.0; NUM_BUCKETS],
        }
    }

    /// bullet v102 互換 quantised.bin を `writer` に書き出す。
    /// 推論エンジン側 `NetworkLayerStacks::read` で parse できる byte layout。
    pub fn save_quantised<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // ---- header ----
        writer.write_all(&NNUE_VERSION.to_le_bytes())?;
        writer.write_all(&NETWORK_HASH.to_le_bytes())?;
        let arch_str = build_arch_str(FT_IN, FT_OUT, L1_OUT, L2_IN, L2_OUT, FV_SCALE);
        let arch_bytes = arch_str.as_bytes();
        writer.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(arch_bytes)?;

        // ---- FT hash ----
        writer.write_all(&FT_HASH.to_le_bytes())?;

        // ---- FT biases LEB128 (i16, scale=QA) ----
        let qa_f = QA as f64;
        let ft_b_i16: Vec<i16> = self
            .ft_b
            .iter()
            .map(|&v| {
                (qa_f * v as f64)
                    .round()
                    .clamp(i16::MIN as f64, i16::MAX as f64) as i16
            })
            .collect();
        write_leb128_tensor_i16(writer, &ft_b_i16)?;

        // ---- FT weights LEB128 (i16, scale=QA) ----
        // piece 部分 = ft_in * ft_out (threat 無し)。本 trainer の ft_w は (FT_IN, FT_OUT)
        // row-major (ft_w[feat * FT_OUT + out])、これは bullet 内部の column-major と
        // 等価の access pattern (転置不要)。そのまま i16 quantize して書く。
        let ft_w_i16: Vec<i16> = self
            .ft_w
            .iter()
            .map(|&v| {
                (qa_f * v as f64)
                    .round()
                    .clamp(i16::MIN as f64, i16::MAX as f64) as i16
            })
            .collect();
        write_leb128_tensor_i16(writer, &ft_w_i16)?;

        // ---- LayerStacks (9 buckets × {fc_hash, L1, L2, L3}) ----
        let qb_f = QB as f64;
        let l1_bias_scale = (QA * QB) as f64; // = 8128
        let l2_bias_scale = 127.0 * qb_f; // = 8128 (QA == 127 前提)
        let l3_bias_scale = 127.0 * qb_f; // = 8128

        for buc in 0..NUM_BUCKETS {
            // fc_hash per bucket
            writer.write_all(&FC_HASH.to_le_bytes())?;

            // --- L1 (merged delta + shared) ---
            // Biases: l1_out i32 scale = QA*QB = 8128
            for out in 0..L1_OUT {
                let merged = self.l1_b[buc * L1_OUT + out] + self.l1f_b[out];
                let val = (l1_bias_scale * merged as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
            // Weights: l1_out × pad32(ft_out) i8 scale = QB
            // For each (buc, out, in) in [0, L1_OUT) × [0, pad32(FT_OUT))
            // - in < FT_OUT: merged = l1_w[buc][out][in] + l1f_w[in][out]
            // - else: padding 0
            let l1_padded_in = pad32(FT_OUT);
            for out in 0..L1_OUT {
                for in_idx in 0..l1_padded_in {
                    let q: i8 = if in_idx < FT_OUT {
                        let buc_w = self.l1_w[buc * L1_OUT * FT_OUT + out * FT_OUT + in_idx];
                        let shared_w = self.l1f_w[in_idx * L1_OUT + out];
                        let merged = buc_w + shared_w;
                        clamp_i8((qb_f * merged as f64).round())
                    } else {
                        0_i8
                    };
                    writer.write_all(&[q as u8])?;
                }
            }

            // --- L2 ---
            // Biases: l2_out i32 scale = 127 * QB = 8128
            for out in 0..L2_OUT {
                let val = (l2_bias_scale * self.l2_b[buc * L2_OUT + out] as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
            // Weights: l2_out × pad32(l2_in) i8 scale = QB
            let l2_padded_in = pad32(L2_IN);
            for out in 0..L2_OUT {
                for in_idx in 0..l2_padded_in {
                    let q: i8 = if in_idx < L2_IN {
                        let w = self.l2_w[buc * L2_OUT * L2_IN + out * L2_IN + in_idx];
                        clamp_i8((qb_f * w as f64).round())
                    } else {
                        0_i8
                    };
                    writer.write_all(&[q as u8])?;
                }
            }

            // --- L3 (output, 1 dim per bucket) ---
            // Bias: i32 scale = 127 * QB
            let val = (l3_bias_scale * self.l3_b[buc] as f64).round() as i32;
            writer.write_all(&val.to_le_bytes())?;
            // Weights: pad32(l2_out) i8 scale = QB
            let l3_padded_in = pad32(L2_OUT);
            for in_idx in 0..l3_padded_in {
                let q: i8 = if in_idx < L2_OUT {
                    let w = self.l3_w[buc * L2_OUT + in_idx];
                    clamp_i8((qb_f * w as f64).round())
                } else {
                    0_i8
                };
                writer.write_all(&[q as u8])?;
            }
        }

        Ok(())
    }

    /// bullet v102 quantised.bin を parse し `V102Weights` を返す。
    ///
    /// 注: save 時に per-bucket l1 と shared l1f は merge されて書き出されるため、
    /// load 時には分離不能。本実装は **l1_w に merged 値をそのまま入れ、l1f_w /
    /// l1f_b は 0 にする** 方針 (forward 計算は等価)。
    ///
    /// **継続学習時の注意**: forward は等価でも、l1f が「shared factorized 部」と
    /// しての意味は失われる (全て l1_w に畳まれた状態)。bullet 流に per-bucket l1
    /// と shared l1f を別々に学習し続ける場合と勾配の流れ方が変わるため、本 method
    /// で得た `V102Weights` から continue-training すると bullet の v102 学習軌跡
    /// とは厳密一致しない。「pretrained 注入 → 1 step → save が byte 互換か」を
    /// 見る用途、あるいは l1f を再び factorize し直す前提なら問題ない。
    pub fn load_quantised<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];

        // version
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != NNUE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown NNUE version: {version:#x} (expected {NNUE_VERSION:#x})"),
            ));
        }

        // network_hash (skip)
        reader.read_exact(&mut buf4)?;

        // arch_str
        reader.read_exact(&mut buf4)?;
        let arch_len = u32::from_le_bytes(buf4) as usize;
        if arch_len == 0 || arch_len > 16384 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid arch_len: {arch_len}"),
            ));
        }
        let mut arch_bytes = vec![0u8; arch_len];
        reader.read_exact(&mut arch_bytes)?;
        let arch_str = String::from_utf8_lossy(&arch_bytes);
        // PSQT / Threat / HandCount を含む arch は本実装でサポートしない (v102 標準形のみ)
        if arch_str.contains("PSQT=")
            || arch_str.contains("Threat=")
            || arch_str.contains("HandCount")
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported arch (only plain LayerStack supported): {arch_str}"),
            ));
        }

        // ft_hash (skip)
        reader.read_exact(&mut buf4)?;

        // FT biases (LEB128 i16, FT_OUT 個)
        let ft_b_i16 = read_leb128_tensor_i16(reader, Some(FT_OUT))?;
        let qa_f = QA as f32;
        let ft_b: Vec<f32> = ft_b_i16.iter().map(|&v| v as f32 / qa_f).collect();

        // FT weights (LEB128 i16, FT_IN * FT_OUT 個)
        let ft_w_i16 = read_leb128_tensor_i16(reader, Some(FT_IN * FT_OUT))?;
        let ft_w: Vec<f32> = ft_w_i16.iter().map(|&v| v as f32 / qa_f).collect();

        // LayerStacks (9 buckets × {fc_hash, L1, L2, L3})
        let qb_f = QB as f32;
        let l1_bias_scale = (QA * QB) as f32;
        let l2_bias_scale = 127.0 * qb_f;
        let l3_bias_scale = 127.0 * qb_f;

        let mut l1_w = vec![0.0_f32; NUM_BUCKETS * L1_OUT * FT_OUT];
        let mut l1_b = vec![0.0_f32; NUM_BUCKETS * L1_OUT];
        let mut l2_w = vec![0.0_f32; NUM_BUCKETS * L2_OUT * L2_IN];
        let mut l2_b = vec![0.0_f32; NUM_BUCKETS * L2_OUT];
        let mut l3_w = vec![0.0_f32; NUM_BUCKETS * L2_OUT];
        let mut l3_b = vec![0.0_f32; NUM_BUCKETS];

        let l1_padded_in = pad32(FT_OUT);
        let l2_padded_in = pad32(L2_IN);
        let l3_padded_in = pad32(L2_OUT);

        for buc in 0..NUM_BUCKETS {
            // fc_hash (skip)
            reader.read_exact(&mut buf4)?;

            // L1 biases (i32 × L1_OUT)
            for out in 0..L1_OUT {
                reader.read_exact(&mut buf4)?;
                let v = i32::from_le_bytes(buf4);
                l1_b[buc * L1_OUT + out] = v as f32 / l1_bias_scale;
            }
            // L1 weights (i8 × L1_OUT × l1_padded_in)、保存値は merged
            // → l1_w に直接書き込み、l1f_w は 0 のまま (forward 等価)
            for out in 0..L1_OUT {
                for in_idx in 0..l1_padded_in {
                    let mut buf1 = [0u8; 1];
                    reader.read_exact(&mut buf1)?;
                    if in_idx < FT_OUT {
                        let q = buf1[0] as i8;
                        l1_w[buc * L1_OUT * FT_OUT + out * FT_OUT + in_idx] = q as f32 / qb_f;
                    }
                    // padding 部分は破棄
                }
            }

            // L2 biases
            for out in 0..L2_OUT {
                reader.read_exact(&mut buf4)?;
                let v = i32::from_le_bytes(buf4);
                l2_b[buc * L2_OUT + out] = v as f32 / l2_bias_scale;
            }
            // L2 weights
            for out in 0..L2_OUT {
                for in_idx in 0..l2_padded_in {
                    let mut buf1 = [0u8; 1];
                    reader.read_exact(&mut buf1)?;
                    if in_idx < L2_IN {
                        let q = buf1[0] as i8;
                        l2_w[buc * L2_OUT * L2_IN + out * L2_IN + in_idx] = q as f32 / qb_f;
                    }
                }
            }

            // L3 bias (1 i32)
            reader.read_exact(&mut buf4)?;
            l3_b[buc] = i32::from_le_bytes(buf4) as f32 / l3_bias_scale;
            // L3 weights (l3_padded_in i8)
            for in_idx in 0..l3_padded_in {
                let mut buf1 = [0u8; 1];
                reader.read_exact(&mut buf1)?;
                if in_idx < L2_OUT {
                    let q = buf1[0] as i8;
                    l3_w[buc * L2_OUT + in_idx] = q as f32 / qb_f;
                }
            }
        }

        Ok(Self {
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l1f_w: vec![0.0; FT_OUT * L1_OUT], // save 時に l1_w に merge 済 → load 側は 0
            l1f_b: vec![0.0; L1_OUT],
            l2_w,
            l2_b,
            l3_w,
            l3_b,
        })
    }
}

fn clamp_i8(v: f64) -> i8 {
    if v < -128.0 {
        -128
    } else if v > 127.0 {
        127
    } else {
        v as i8
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb128_roundtrip_zero() {
        let mut buf = Vec::new();
        encode_signed_leb128(0, &mut buf);
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn leb128_roundtrip_small() {
        let cases = [-65_i64, -64, -1, 0, 1, 63, 64, 127, 128, 32767, -32768];
        for &v in &cases {
            let mut buf = Vec::new();
            encode_signed_leb128(v, &mut buf);
            let (decoded, consumed) = decode_single_leb128(&buf).unwrap();
            assert_eq!(decoded, v, "encode/decode mismatch for {v}");
            assert_eq!(consumed, buf.len());
        }
    }

    #[test]
    fn leb128_tensor_roundtrip() {
        let values: Vec<i16> = vec![0, 1, -1, 127, -128, 32767, -32768, 100, -100];
        let mut buf = Vec::new();
        write_leb128_tensor_i16(&mut buf, &values).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let decoded = read_leb128_tensor_i16(&mut cursor, Some(values.len())).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn pad32_correct() {
        assert_eq!(pad32(0), 0);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(30), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(33), 64);
        assert_eq!(pad32(FT_OUT), FT_OUT); // 1536 は 32 の倍数
        assert_eq!(pad32(L2_IN), 32);
        assert_eq!(pad32(L2_OUT), 32);
    }

    #[test]
    fn weights_zeroed_save_load_roundtrip() {
        // 0 weight で save → load → 同じ 0 weight が復元できる
        let original = V102Weights::zeroed();
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = V102Weights::load_quantised(&mut cursor).unwrap();

        assert_eq!(loaded.ft_w.len(), original.ft_w.len());
        assert_eq!(loaded.ft_b.len(), original.ft_b.len());
        assert_eq!(loaded.l1_w.len(), original.l1_w.len());
        assert_eq!(loaded.l1_b.len(), original.l1_b.len());
        assert_eq!(loaded.l2_w.len(), original.l2_w.len());
        assert_eq!(loaded.l2_b.len(), original.l2_b.len());
        assert_eq!(loaded.l3_w.len(), original.l3_w.len());
        assert_eq!(loaded.l3_b.len(), original.l3_b.len());
        // 値も全 0
        assert!(loaded.ft_w.iter().all(|&x| x == 0.0));
        assert!(loaded.l1_w.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn load_v102_100_reference_if_available() {
        // `/tmp/v102_100_quantised.bin` (bullet で生成した参照 checkpoint) が
        // 存在するときのみ load + sanity check を回すローカル動作確認。CI では
        // ファイルが無いので skip。
        let path = "/tmp/v102_100_quantised.bin";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping load_v102_100_reference (file not found at {path})");
            return;
        }
        let mut file = std::fs::File::open(path).expect("open quantised.bin");
        let weights = V102Weights::load_quantised(&mut file).expect("parse quantised.bin");

        // Sanity: 各 weight buffer の長さが期待値
        assert_eq!(weights.ft_w.len(), FT_IN * FT_OUT);
        assert_eq!(weights.ft_b.len(), FT_OUT);
        assert_eq!(weights.l1_w.len(), NUM_BUCKETS * L1_OUT * FT_OUT);
        assert_eq!(weights.l1_b.len(), NUM_BUCKETS * L1_OUT);
        assert_eq!(weights.l2_w.len(), NUM_BUCKETS * L2_OUT * L2_IN);
        assert_eq!(weights.l2_b.len(), NUM_BUCKETS * L2_OUT);
        assert_eq!(weights.l3_w.len(), NUM_BUCKETS * L2_OUT);
        assert_eq!(weights.l3_b.len(), NUM_BUCKETS);

        // 非自明 weight (training 済みなので非 0 が多い)
        let nz_ft = weights.ft_w.iter().filter(|&&x| x != 0.0).count();
        let pct_ft = nz_ft as f64 / weights.ft_w.len() as f64 * 100.0;
        let max_ft = weights.ft_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let max_l1 = weights.l1_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let max_l2 = weights.l2_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let max_l3 = weights.l3_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        eprintln!(
            "[v102-100 load] FT nonzero: {nz_ft}/{} ({pct_ft:.2}%)",
            weights.ft_w.len()
        );
        eprintln!("[v102-100 load] FT weight max abs: {max_ft:.6}");
        eprintln!("[v102-100 load] L1 weight max abs: {max_l1:.6}");
        eprintln!("[v102-100 load] L2 weight max abs: {max_l2:.6}");
        eprintln!("[v102-100 load] L3 weight max abs: {max_l3:.6}");

        // 量子化範囲チェック: i8 weight は |w| <= 127/QB ≈ 1.984
        assert!(max_ft <= 2.0, "FT max abs {max_ft} > 2.0");
        assert!(max_l1 <= 2.0, "L1 max abs {max_l1} > 2.0");
        assert!(max_l2 <= 2.0, "L2 max abs {max_l2} > 2.0");
        assert!(max_l3 <= 2.0, "L3 max abs {max_l3} > 2.0");

        // 全 0 でないこと (trained model)
        assert!(
            max_ft > 0.001,
            "FT weights are all near 0 — likely format mismatch"
        );
        assert!(
            max_l1 > 0.001,
            "L1 weights are all near 0 — likely format mismatch"
        );
    }

    #[test]
    fn save_v102_100_resaved_if_available() {
        // `/tmp/v102_100_quantised.bin` を load → save し直して、size diff と
        // byte diff count を確認するローカル regression check (CI では skip)。
        // 別途 `verify_nnue_accumulator` 等で同等性を手動確認する想定。
        let in_path = "/tmp/v102_100_quantised.bin";
        let out_path = "/tmp/v102_100_resaved.bin";
        if !std::path::Path::new(in_path).exists() {
            eprintln!("skipping save_v102_100_resaved (file not found at {in_path})");
            return;
        }
        let mut reader = std::io::BufReader::new(std::fs::File::open(in_path).unwrap());
        let weights = V102Weights::load_quantised(&mut reader).unwrap();
        let mut writer = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
        weights.save_quantised(&mut writer).unwrap();
        drop(writer);

        let in_size = std::fs::metadata(in_path).unwrap().len();
        let out_size = std::fs::metadata(out_path).unwrap().len();
        let diff = (out_size as i64) - (in_size as i64);
        eprintln!("[resave] in_size={in_size}, out_size={out_size}, diff={diff}");
        // Byte size は完全一致を期待 (layout regression detect)。
        // 値の byte 差は network_hash + 9 fc_hash + rounding boundary で最大 ~50 bytes、
        // size diff は 0 のはず。layout に regression があれば size が大きく変わる。
        assert_eq!(diff, 0, "size diff {diff} != 0 — layout regression?");

        // 参照との byte 差は最大 100 bytes 程度を許容範囲とする (rounding boundary
        // の本数は実 weight 分布次第だが、典型的に 0-5 bytes、安全 margin で 100 まで OK)
        let in_bytes = std::fs::read(in_path).unwrap();
        let out_bytes = std::fs::read(out_path).unwrap();
        let byte_diff_count = in_bytes
            .iter()
            .zip(out_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("[resave] byte_diff_count={byte_diff_count}");
        assert!(
            byte_diff_count < 100,
            "byte diff count {byte_diff_count} >= 100 — format regression suspected"
        );
    }

    #[test]
    fn fc_hash_matches_bullet_formula() {
        // FC_HASH const と compute_fc_hash 関数の結果が一致 (const 展開の sanity)。
        assert_eq!(FC_HASH, compute_fc_hash(FT_OUT, L2_IN, L2_OUT));
        // v102 (ft_out=1536, l2_in=30, l2_out=32) で computed value が
        // 0 (placeholder) でないことを確認。
        assert_ne!(FC_HASH, 0, "FC_HASH should be computed, not placeholder");
    }

    #[test]
    fn arch_str_format() {
        let s = build_arch_str(FT_IN, FT_OUT, L1_OUT, L2_IN, L2_OUT, FV_SCALE);
        assert!(s.contains("HalfKA_hm"));
        assert!(s.contains("73305->1536x2"));
        assert!(s.contains("AffineTransform[1<-32]"));
        assert!(s.contains("ClippedReLU[32]"));
        assert!(s.contains("AffineTransform[32<-30]"));
        assert!(s.contains("SqrClippedReLU[30]"));
        assert!(s.contains("AffineTransform[16<-3072]"));
        assert!(s.contains("InputSlice[3072(0:3072)]"));
        assert!(s.contains("fv_scale=28"));
        assert!(!s.contains("PSQT="));
        assert!(!s.contains("Threat="));
    }
}
