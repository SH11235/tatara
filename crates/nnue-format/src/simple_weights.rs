//! Simple 4 層 NNUE quantised binary の save / load。
//!
//! `ArchKind::Simple` (bullet-shogi 由来の bucket 無し 4 層 dense アーキ) の
//! 量子化 weight を 1 本の binary に (de)serialise する **GPU 非依存・pure CPU**
//! module。LayerStack の [`crate::layerstack_weights`] と並ぶ、もう一方の
//! アーキ用フォーマット。
//!
//! ## Simple architecture (bucket 無し / PSQT 無し / skip 無し / L1f 無し)
//!
//! - FT: `ft_in → ft_out`、weight + bias 共有 stm/nstm (`ft_in` は feature set 依存、
//!   `ft_out` は CLI 設定値)
//! - per-perspective 活性化 (CReLU または SCReLU) → `concat(stm, nstm)` = `2*ft_out`
//! - L1 dense: `2*ft_out → l1_out` + bias → 活性化
//! - L2 dense: `l1_out → l2_out` + bias → 活性化
//! - L3 dense: `l2_out → 1` + bias → スカラ net_output
//!
//! ## file layout (top-level、little-endian)
//!
//! 1. header: `NNUE_VERSION` (u32) + `network_hash` (u32) + `arch_len` (u32) + `arch_str`
//! 2. `ft_hash` (u32)
//! 3. FT biases: `i16` × `ft_out` (raw、scale = QA)
//! 4. FT weights: `i16` × `ft_in * ft_out` (raw、scale = QA)
//! 5. `fc_hash` (u32) — dense 層群の topology hash
//! 6. L1: bias `i32` × `l1_out` (scale = `127*QB`)、weight `i8` × `l1_out * pad32(2*ft_out)` (scale = QB)
//! 7. L2: bias `i32` × `l2_out`、weight `i8` × `l2_out * pad32(l1_out)`
//! 8. L3: bias `i32` × 1、weight `i8` × `pad32(l2_out)`
//!
//! 推論エンジン互換 / bullet-shogi `.bin` との byte 互換は対象外。本 module の
//! `save_quantised` / `load` が round-trip する self-consistent な format で、
//! 量子化アルゴリズム (QA/QB scale、`pad32`、hash) の出典は `ATTRIBUTION.md`。
//!
//! ## 量子化 scale
//!
//! | layer | bias scale | weight scale |
//! |---|---|---|
//! | FT | QA (i16) | QA (i16) |
//! | L1 / L2 / L3 | `127*QB` (i32) | QB (i8) |
//!
//! QA は活性化で決まる: CReLU → 127、SCReLU → 255。dense 層の入力は活性化後で
//! 常に 127-scale (CReLU の出力、SCReLU の `x²>>9` いずれも 127-scale) のため、
//! dense bias scale は活性化に依らず `127*QB`。
//!
//! ## pad32
//!
//! dense 層の i8 weight は入力次元を 32 の倍数に pad して書く (SIMD load align)。
//! padding byte は 0。
//!
//! ## 重み layout
//!
//! 全 weight は **row-major** (`w[out * in_dim + in_idx]`)。FT は
//! `ft_w[feat * ft_out + out]`。

use std::io::{self, Read, Write};

use shogi_features::FeatureSetSpec;

// =============================================================================
// constants
// =============================================================================

/// rshogi NNUE quantised binary の format version magic。
pub const NNUE_VERSION: u32 = 0x7AF32F20;

/// dense 層 weight の量子化 multiplier。
pub const QB: i32 = 64;

/// pad to multiple of 32 (SIMD alignment)。
#[inline]
pub fn pad32(x: usize) -> usize {
    x.div_ceil(32) * 32
}

// =============================================================================
// SimpleActivation
// =============================================================================

/// Simple アーキの per-perspective 活性化関数。FT 出力の量子化 multiplier QA と
/// arch 文字列の活性化トークンを決める。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimpleActivation {
    /// Clipped ReLU (`clamp(x, 0, 1)`)。
    CReLU,
    /// Squared Clipped ReLU (`clamp(x, 0, 1)²`)。
    SCReLU,
}

impl SimpleActivation {
    /// CLI / arch identity が扱う canonical 名。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            SimpleActivation::CReLU => "crelu",
            SimpleActivation::SCReLU => "screlu",
        }
    }

    /// canonical 名から逆引きする。未知の名前は `None`。
    pub fn from_canonical_name(name: &str) -> Option<SimpleActivation> {
        match name {
            "crelu" => Some(SimpleActivation::CReLU),
            "screlu" => Some(SimpleActivation::SCReLU),
            _ => None,
        }
    }

    /// FT 出力の量子化 multiplier QA。CReLU は 127、SCReLU は 255。
    pub const fn qa(self) -> i32 {
        match self {
            SimpleActivation::CReLU => 127,
            SimpleActivation::SCReLU => 255,
        }
    }

    /// arch 文字列の活性化トークン (nnue-pytorch 系の層名)。
    const fn arch_token(self) -> &'static str {
        match self {
            SimpleActivation::CReLU => "ClippedReLU",
            SimpleActivation::SCReLU => "SqrClippedReLU",
        }
    }
}

// =============================================================================
// SimpleId — Simple ネットの構造化 identity
// =============================================================================

/// Simple ネットの identity を成す構造化フィールド。`load` の reject 契約は
/// この値を file の arch 文字列・hash と照合する。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimpleId {
    /// 入力 feature set (FT 入力次元 `ft_in` と feature hash の源)。
    pub feature_set: FeatureSetSpec,
    /// per-perspective 活性化関数。
    pub activation: SimpleActivation,
    /// FT 出力 (accumulator) 次元。
    pub ft_out: usize,
    /// L1 dense 出力 (隠れ層 1) 次元。
    pub l1_out: usize,
    /// L2 dense 出力 (隠れ層 2) 次元。
    pub l2_out: usize,
}

impl SimpleId {
    /// FT 入力次元 (`feature_set` 依存)。
    pub fn ft_in(&self) -> usize {
        self.feature_set.ft_in()
    }
}

// =============================================================================
// arch_str + hash
// =============================================================================

/// arch 文字列の identity 部 (feature / 層次元 / 活性化を含み、`fv_scale` を含まない)。
/// `load` の reject 契約はこの部分を照合する。
fn arch_identity(id: &SimpleId) -> String {
    let ft_in = id.ft_in();
    let ft_out = id.ft_out;
    let l1_out = id.l1_out;
    let l2_out = id.l2_out;
    let l1_input = ft_out * 2;
    let act = id.activation.arch_token();
    format!(
        "Features={}(Friend)[{}->{}x2],\
         Network=AffineTransform[1<-{}](\
         {}[{}](\
         AffineTransform[{}<-{}](\
         {}[{}](\
         AffineTransformSparseInput[{}<-{}](\
         InputSlice[{}(0:{})])))))",
        id.feature_set.arch_feature_name(),
        ft_in,
        ft_out,
        l2_out,   // Output layer input
        act,      // L2 出力の活性化
        l2_out,   // L2 output features
        l2_out,   // L3 input / L2 output
        l1_out,   // L2 input features
        act,      // L1 出力の活性化
        l1_out,   // L1 output features
        l1_out,   // L2 input / L1 output
        l1_input, // L1 input (dual perspective)
        l1_input, // InputSlice size
        l1_input, // InputSlice range
    )
}

/// arch 文字列全体 (`<identity>,fv_scale=<N>`)。`fv_scale` は推論時の評価値
/// スケールで identity には含めない (学習 `--scale` 由来で同一 topology でも変動)。
pub fn build_arch_str(id: &SimpleId, fv_scale: i32) -> String {
    format!("{},fv_scale={}", arch_identity(id), fv_scale)
}

/// dense 層群の topology hash (nnue-pytorch 系、出典は `ATTRIBUTION.md`)。
/// 層次元 (`ft_out` / `l1_out` / `l2_out`) のみに依存する。
pub const fn compute_fc_hash(ft_out: usize, l1_out: usize, l2_out: usize) -> u32 {
    // InputSlice hash (FT output × 2 dual perspective)。
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= (ft_out * 2) as u32;

    // FC 層の出力次元列: L1=l1_out / L2=l2_out / 出力=1。L1/L2 は活性化付き。
    // const fn なので index ベースの `while` で回す。
    let layer_sizes = [(l1_out, true), (l2_out, true), (1_usize, false)];
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

/// FT (FeatureTransformer) layer hash: `feature_hash ^ (ft_out * 2)`。
/// LayerStack と違い `ft_out` が runtime 値のため引数で受ける。
pub fn ft_hash(feature_hash: u32, ft_out: usize) -> u32 {
    feature_hash ^ (ft_out as u32 * 2)
}

/// network hash: `compute_fc_hash ^ ft_hash`。
pub fn network_hash(id: &SimpleId) -> u32 {
    compute_fc_hash(id.ft_out, id.l1_out, id.l2_out)
        ^ ft_hash(id.feature_set.feature_hash(), id.ft_out)
}

// =============================================================================
// SimpleWeights — トレーナー側 weight 表現 (f32、row-major)
// =============================================================================

/// Simple アーキの全 weight (f32、host 側保持)。
///
/// Layout (`ft_in` = `id.ft_in()`、`ft_out` / `l1_out` / `l2_out` = `id` の値):
/// - `ft_w`: `(ft_in, ft_out)` row-major、`ft_w[feat * ft_out + out]`
/// - `ft_b`: `(ft_out)` (stm/nstm 共有)
/// - `l1_w`: `(l1_out, 2*ft_out)` row-major
/// - `l1_b`: `(l1_out)`
/// - `l2_w`: `(l2_out, l1_out)` row-major
/// - `l2_b`: `(l2_out)`
/// - `l3_w`: `(l2_out)` (出力 1 次元なので out 軸省略)
/// - `l3_b`: `(1)`
#[derive(Debug, Clone)]
pub struct SimpleWeights {
    /// この weight の構造化 identity。
    pub id: SimpleId,
    /// 推論時の評価値スケール (`round(QA*QB / 学習 scale)`)。arch 文字列に記録する。
    pub fv_scale: i32,
    pub ft_w: Vec<f32>,
    pub ft_b: Vec<f32>,
    pub l1_w: Vec<f32>,
    pub l1_b: Vec<f32>,
    pub l2_w: Vec<f32>,
    pub l2_b: Vec<f32>,
    pub l3_w: Vec<f32>,
    pub l3_b: Vec<f32>,
}

impl SimpleWeights {
    /// 全 buffer を 0 で初期化した新規 instance。
    pub fn zeroed(id: SimpleId, fv_scale: i32) -> Self {
        Self {
            id,
            fv_scale,
            ft_w: vec![0.0; id.ft_in() * id.ft_out],
            ft_b: vec![0.0; id.ft_out],
            l1_w: vec![0.0; id.l1_out * 2 * id.ft_out],
            l1_b: vec![0.0; id.l1_out],
            l2_w: vec![0.0; id.l2_out * id.l1_out],
            l2_b: vec![0.0; id.l2_out],
            l3_w: vec![0.0; id.l2_out],
            l3_b: vec![0.0; 1],
        }
    }

    /// 各 weight buffer の要素数が `id` の層次元と整合するか検証する。
    /// 不整合は `save_quantised` 内の index 計算が panic / 末尾切り捨てに
    /// なるため、書き出し前に `InvalidInput` で弾く。
    fn check_buffer_lengths(&self) -> io::Result<()> {
        let id = &self.id;
        let groups = [
            ("ft_w", self.ft_w.len(), id.ft_in() * id.ft_out),
            ("ft_b", self.ft_b.len(), id.ft_out),
            ("l1_w", self.l1_w.len(), id.l1_out * 2 * id.ft_out),
            ("l1_b", self.l1_b.len(), id.l1_out),
            ("l2_w", self.l2_w.len(), id.l2_out * id.l1_out),
            ("l2_b", self.l2_b.len(), id.l2_out),
            ("l3_w", self.l3_w.len(), id.l2_out),
            ("l3_b", self.l3_b.len(), 1),
        ];
        for (name, got, want) in groups {
            if got != want {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("SimpleWeights.{name} has {got} elements, expected {want}"),
                ));
            }
        }
        Ok(())
    }

    /// Simple quantised binary を `writer` に書き出す。weight buffer の長さが
    /// `id` の層次元と不整合なら、書き出し前に `InvalidInput` で返す。
    pub fn save_quantised<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.check_buffer_lengths()?;
        let id = &self.id;

        // ---- header ----
        writer.write_all(&NNUE_VERSION.to_le_bytes())?;
        writer.write_all(&network_hash(id).to_le_bytes())?;
        let arch_bytes = build_arch_str(id, self.fv_scale).into_bytes();
        writer.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&arch_bytes)?;

        // ---- FT hash ----
        writer.write_all(&ft_hash(id.feature_set.feature_hash(), id.ft_out).to_le_bytes())?;

        // ---- FT biases / weights (i16, scale = QA) ----
        let qa = id.activation.qa() as f64;
        write_i16_quantised(writer, &self.ft_b, qa)?;
        write_i16_quantised(writer, &self.ft_w, qa)?;

        // ---- dense topology hash ----
        writer.write_all(&compute_fc_hash(id.ft_out, id.l1_out, id.l2_out).to_le_bytes())?;

        // ---- dense layers (bias i32 scale 127*QB, weight i8 scale QB) ----
        let bias_scale = (127 * QB) as f64;
        let weight_scale = QB as f64;
        // L1: (l1_out, 2*ft_out)
        write_i32_bias(writer, &self.l1_b, bias_scale)?;
        write_i8_weight(writer, &self.l1_w, id.l1_out, 2 * id.ft_out, weight_scale)?;
        // L2: (l2_out, l1_out)
        write_i32_bias(writer, &self.l2_b, bias_scale)?;
        write_i8_weight(writer, &self.l2_w, id.l2_out, id.l1_out, weight_scale)?;
        // L3: (1, l2_out)
        write_i32_bias(writer, &self.l3_b, bias_scale)?;
        write_i8_weight(writer, &self.l3_w, 1, id.l2_out, weight_scale)?;

        Ok(())
    }

    /// Simple quantised binary を parse する。`expected` は要求する identity
    /// (feature set / 活性化 / 層次元)。file の version・hash・arch identity が
    /// `expected` 由来の値と一致しなければ `InvalidData` で reject する。
    pub fn load<R: Read>(reader: &mut R, expected: SimpleId) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];

        // version
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != NNUE_VERSION {
            return Err(invalid(format!(
                "unknown NNUE version: {version:#x} (expected {NNUE_VERSION:#x})"
            )));
        }

        // network_hash
        reader.read_exact(&mut buf4)?;
        let file_network_hash = u32::from_le_bytes(buf4);
        let expected_network_hash = network_hash(&expected);
        if file_network_hash != expected_network_hash {
            return Err(invalid(format!(
                "network_hash mismatch: file {file_network_hash:#x}, \
                 expected {expected_network_hash:#x}"
            )));
        }

        // arch_str
        reader.read_exact(&mut buf4)?;
        let arch_len = u32::from_le_bytes(buf4) as usize;
        if arch_len == 0 || arch_len > 16384 {
            return Err(invalid(format!("invalid arch_len: {arch_len}")));
        }
        let mut arch_bytes = vec![0u8; arch_len];
        reader.read_exact(&mut arch_bytes)?;
        let arch_str = String::from_utf8(arch_bytes)
            .map_err(|e| invalid(format!("arch_str is not valid UTF-8: {e}")))?;
        // identity 部 (`,fv_scale=` の手前) を expected から組んだ値と照合する。
        let (file_identity, fv_scale_str) = arch_str
            .rsplit_once(",fv_scale=")
            .ok_or_else(|| invalid(format!("arch_str missing fv_scale field: `{arch_str}`")))?;
        let expected_identity = arch_identity(&expected);
        if file_identity != expected_identity {
            return Err(invalid(format!(
                "arch identity mismatch: file `{file_identity}`, expected `{expected_identity}`"
            )));
        }
        let fv_scale: i32 = fv_scale_str
            .parse()
            .map_err(|_| invalid(format!("invalid fv_scale in arch_str: `{fv_scale_str}`")))?;

        // ft_hash
        reader.read_exact(&mut buf4)?;
        let file_ft_hash = u32::from_le_bytes(buf4);
        let expected_ft_hash = ft_hash(expected.feature_set.feature_hash(), expected.ft_out);
        if file_ft_hash != expected_ft_hash {
            return Err(invalid(format!(
                "ft_hash mismatch: file {file_ft_hash:#x}, expected {expected_ft_hash:#x}"
            )));
        }

        // FT biases / weights (i16, scale = QA)
        let qa = expected.activation.qa() as f32;
        let ft_b = read_i16_quantised(reader, expected.ft_out, qa)?;
        let ft_w = read_i16_quantised(reader, expected.ft_in() * expected.ft_out, qa)?;

        // dense topology hash
        reader.read_exact(&mut buf4)?;
        let file_fc_hash = u32::from_le_bytes(buf4);
        let expected_fc_hash = compute_fc_hash(expected.ft_out, expected.l1_out, expected.l2_out);
        if file_fc_hash != expected_fc_hash {
            return Err(invalid(format!(
                "fc_hash mismatch: file {file_fc_hash:#x}, expected {expected_fc_hash:#x}"
            )));
        }

        // dense layers
        let bias_scale = (127 * QB) as f32;
        let weight_scale = QB as f32;
        let l1_b = read_i32_bias(reader, expected.l1_out, bias_scale)?;
        let l1_w = read_i8_weight(reader, expected.l1_out, 2 * expected.ft_out, weight_scale)?;
        let l2_b = read_i32_bias(reader, expected.l2_out, bias_scale)?;
        let l2_w = read_i8_weight(reader, expected.l2_out, expected.l1_out, weight_scale)?;
        let l3_b = read_i32_bias(reader, 1, bias_scale)?;
        let l3_w = read_i8_weight(reader, 1, expected.l2_out, weight_scale)?;

        Ok(Self {
            id: expected,
            fv_scale,
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
        })
    }
}

// =============================================================================
// quantise / dequantise helpers
// =============================================================================

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn clamp_round(v: f64, lo: f64, hi: f64) -> f64 {
    let r = v.round();
    if r < lo {
        lo
    } else if r > hi {
        hi
    } else {
        r
    }
}

/// f32 列を `scale` 倍して i16 量子化し、raw little-endian で書く。
fn write_i16_quantised<W: Write>(writer: &mut W, values: &[f32], scale: f64) -> io::Result<()> {
    for &v in values {
        let q = clamp_round(scale * v as f64, i16::MIN as f64, i16::MAX as f64) as i16;
        writer.write_all(&q.to_le_bytes())?;
    }
    Ok(())
}

/// `n` 個の i16 (raw LE) を読み、`scale` で割って f32 列に戻す。
fn read_i16_quantised<R: Read>(reader: &mut R, n: usize, scale: f32) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 2];
    for _ in 0..n {
        reader.read_exact(&mut buf)?;
        out.push(i16::from_le_bytes(buf) as f32 / scale);
    }
    Ok(out)
}

/// bias 列を `scale` 倍して i32 量子化し書く。
fn write_i32_bias<W: Write>(writer: &mut W, values: &[f32], scale: f64) -> io::Result<()> {
    for &v in values {
        let q = clamp_round(scale * v as f64, i32::MIN as f64, i32::MAX as f64) as i32;
        writer.write_all(&q.to_le_bytes())?;
    }
    Ok(())
}

/// `n` 個の i32 bias を読み、`scale` で割って戻す。
fn read_i32_bias<R: Read>(reader: &mut R, n: usize, scale: f32) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        reader.read_exact(&mut buf)?;
        out.push(i32::from_le_bytes(buf) as f32 / scale);
    }
    Ok(out)
}

/// row-major weight `(out_dim, in_dim)` を `pad32(in_dim)` 幅に pad しながら
/// `scale` 倍して i8 量子化し書く。padding byte は 0。
fn write_i8_weight<W: Write>(
    writer: &mut W,
    weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    scale: f64,
) -> io::Result<()> {
    let padded_in = pad32(in_dim);
    for out in 0..out_dim {
        for in_idx in 0..padded_in {
            let q: i8 = if in_idx < in_dim {
                clamp_round(scale * weights[out * in_dim + in_idx] as f64, -128.0, 127.0) as i8
            } else {
                0
            };
            writer.write_all(&[q as u8])?;
        }
    }
    Ok(())
}

/// `pad32(in_dim)` 幅で書かれた i8 weight を読み、padding を捨て row-major
/// `(out_dim, in_dim)` の f32 に戻す。
fn read_i8_weight<R: Read>(
    reader: &mut R,
    out_dim: usize,
    in_dim: usize,
    scale: f32,
) -> io::Result<Vec<f32>> {
    let padded_in = pad32(in_dim);
    let mut out = vec![0.0_f32; out_dim * in_dim];
    let mut buf = [0u8; 1];
    for o in 0..out_dim {
        for in_idx in 0..padded_in {
            reader.read_exact(&mut buf)?;
            if in_idx < in_dim {
                out[o * in_dim + in_idx] = buf[0] as i8 as f32 / scale;
            }
        }
    }
    Ok(out)
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::FeatureSet;

    fn test_id(activation: SimpleActivation) -> SimpleId {
        SimpleId {
            feature_set: FeatureSet::HalfKaHmMerged.spec(),
            activation,
            ft_out: 256,
            l1_out: 32,
            l2_out: 32,
        }
    }

    #[test]
    fn pad32_correct() {
        assert_eq!(pad32(0), 0);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(8), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(33), 64);
        assert_eq!(pad32(512), 512);
    }

    #[test]
    fn activation_name_round_trips() {
        for act in [SimpleActivation::CReLU, SimpleActivation::SCReLU] {
            assert_eq!(
                SimpleActivation::from_canonical_name(act.canonical_name()),
                Some(act)
            );
        }
        assert_eq!(SimpleActivation::from_canonical_name("bogus"), None);
    }

    #[test]
    fn zeroed_save_load_round_trips() {
        for act in [SimpleActivation::CReLU, SimpleActivation::SCReLU] {
            let id = test_id(act);
            let original = SimpleWeights::zeroed(id, 13);
            let mut buf = Vec::new();
            original.save_quantised(&mut buf).unwrap();
            let loaded =
                SimpleWeights::load(&mut std::io::Cursor::new(&buf), id).expect("load zeroed");

            assert_eq!(loaded.id, id);
            assert_eq!(loaded.fv_scale, 13);
            assert_eq!(loaded.ft_w.len(), id.ft_in() * id.ft_out);
            assert_eq!(loaded.l1_w.len(), id.l1_out * 2 * id.ft_out);
            assert_eq!(loaded.l2_w.len(), id.l2_out * id.l1_out);
            assert_eq!(loaded.l3_w.len(), id.l2_out);
            assert_eq!(loaded.l3_b.len(), 1);
            assert!(loaded.ft_w.iter().all(|&x| x == 0.0));
            assert!(loaded.l1_w.iter().all(|&x| x == 0.0));
            assert!(loaded.l3_b.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn quantisation_grid_values_round_trip_exactly() {
        // 量子化格子上の値 (k/QA, k/QB, k/(127*QB)) は厳密に round-trip する。
        let id = test_id(SimpleActivation::CReLU);
        let qa = id.activation.qa() as f32;
        let qb = QB as f32;
        let bias = (127 * QB) as f32;
        let mut w = SimpleWeights::zeroed(id, 16);
        for (i, v) in w.ft_w.iter_mut().enumerate() {
            *v = ((i % 7) as i32 - 3) as f32 / qa;
        }
        for (i, v) in w.ft_b.iter_mut().enumerate() {
            *v = ((i % 5) as i32 - 2) as f32 / qa;
        }
        for (i, v) in w.l1_w.iter_mut().enumerate() {
            *v = ((i % 9) as i32 - 4) as f32 / qb;
        }
        for (i, v) in w.l2_w.iter_mut().enumerate() {
            *v = ((i % 11) as i32 - 5) as f32 / qb;
        }
        for (i, v) in w.l3_w.iter_mut().enumerate() {
            *v = ((i % 13) as i32 - 6) as f32 / qb;
        }
        for (i, v) in w.l1_b.iter_mut().enumerate() {
            *v = ((i % 3) as i32 - 1) as f32 / bias;
        }
        w.l2_b[0] = 7.0 / bias;
        w.l3_b[0] = -9.0 / bias;

        let mut buf = Vec::new();
        w.save_quantised(&mut buf).unwrap();
        let loaded = SimpleWeights::load(&mut std::io::Cursor::new(&buf), id).unwrap();

        assert_eq!(loaded.ft_w, w.ft_w);
        assert_eq!(loaded.ft_b, w.ft_b);
        assert_eq!(loaded.l1_w, w.l1_w);
        assert_eq!(loaded.l1_b, w.l1_b);
        assert_eq!(loaded.l2_w, w.l2_w);
        assert_eq!(loaded.l2_b, w.l2_b);
        assert_eq!(loaded.l3_w, w.l3_w);
        assert_eq!(loaded.l3_b, w.l3_b);
    }

    #[test]
    fn load_rejects_feature_set_mismatch() {
        let merged = test_id(SimpleActivation::CReLU);
        let mut split = merged;
        split.feature_set = FeatureSet::HalfKaSplit.spec();
        let mut buf = Vec::new();
        SimpleWeights::zeroed(merged, 16)
            .save_quantised(&mut buf)
            .unwrap();
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), split)
            .expect_err("feature set mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_dimension_mismatch() {
        let id = test_id(SimpleActivation::CReLU);
        let mut wider = id;
        wider.l1_out = 64;
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), wider)
            .expect_err("dimension mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_activation_mismatch() {
        // feature set / 層次元が同一でも活性化が異なれば reject (arch identity)。
        let crelu = test_id(SimpleActivation::CReLU);
        let screlu = test_id(SimpleActivation::SCReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(crelu, 16)
            .save_quantised(&mut buf)
            .unwrap();
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), screlu)
            .expect_err("activation mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_truncated_input() {
        let id = test_id(SimpleActivation::CReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();
        buf.truncate(buf.len() / 2);
        assert!(SimpleWeights::load(&mut std::io::Cursor::new(&buf), id).is_err());
    }

    #[test]
    fn save_rejects_mismatched_buffer_length() {
        // weight buffer 長が id 次元と不整合なら save が InvalidInput で弾く。
        let id = test_id(SimpleActivation::CReLU);

        // 長すぎる buffer。
        let mut too_long = SimpleWeights::zeroed(id, 16);
        too_long.l1_b.push(0.0);
        assert_eq!(
            too_long
                .save_quantised(&mut Vec::new())
                .expect_err("too-long buffer must reject")
                .kind(),
            io::ErrorKind::InvalidInput
        );

        // 短すぎる buffer (検証が無ければ index panic / 末尾切り捨てになる経路)。
        let mut too_short = SimpleWeights::zeroed(id, 16);
        too_short.ft_w.pop();
        assert_eq!(
            too_short
                .save_quantised(&mut Vec::new())
                .expect_err("too-short buffer must reject")
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn load_rejects_invalid_utf8_arch_str() {
        // arch_str は version(4) + network_hash(4) + arch_len(4) の後ろから始まる。
        // 先頭バイトを不正 UTF-8 (0xFF) に潰すと load が InvalidData で reject する。
        let id = test_id(SimpleActivation::CReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();
        buf[12] = 0xFF;
        let err = SimpleWeights::load(&mut std::io::Cursor::new(&buf), id)
            .expect_err("invalid UTF-8 arch_str must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn file_size_matches_layout() {
        let id = test_id(SimpleActivation::CReLU);
        let mut buf = Vec::new();
        SimpleWeights::zeroed(id, 16)
            .save_quantised(&mut buf)
            .unwrap();

        let arch_len = build_arch_str(&id, 16).len();
        let header = 4 + 4 + 4 + arch_len; // version + network_hash + arch_len + arch_str
        let ft = 4 + (id.ft_out + id.ft_in() * id.ft_out) * 2; // ft_hash + i16 ft_b/ft_w
        let fc_hash = 4;
        let l1 = id.l1_out * 4 + id.l1_out * pad32(2 * id.ft_out);
        let l2 = id.l2_out * 4 + id.l2_out * pad32(id.l1_out);
        let l3 = 4 + pad32(id.l2_out);
        assert_eq!(buf.len(), header + ft + fc_hash + l1 + l2 + l3);
    }

    #[test]
    fn arch_str_contains_structured_fields() {
        let id = test_id(SimpleActivation::SCReLU);
        let s = build_arch_str(&id, 27);
        assert!(s.contains("HalfKA_hm(Friend)[73305->256x2]"));
        assert!(s.contains("SqrClippedReLU[32]"));
        assert!(s.contains("AffineTransformSparseInput[32<-512]"));
        assert!(s.contains("InputSlice[512(0:512)]"));
        assert!(s.ends_with(",fv_scale=27"));
    }

    #[test]
    fn arch_str_parentheses_balanced() {
        // nnue-pytorch 系の arch 文字列は層の入れ子を括弧で表すため、
        // `(` と `)` の数が一致していなければ malformed。
        for act in [SimpleActivation::CReLU, SimpleActivation::SCReLU] {
            let s = build_arch_str(&test_id(act), 16);
            let opens = s.matches('(').count();
            let closes = s.matches(')').count();
            assert_eq!(opens, closes, "arch_str parentheses unbalanced: `{s}`");
        }
    }

    #[test]
    fn fc_hash_depends_only_on_dimensions() {
        // topology hash は層次元のみに依存し、feature set / 活性化に依らない。
        let a = compute_fc_hash(256, 32, 32);
        assert_eq!(a, compute_fc_hash(256, 32, 32));
        assert_ne!(a, compute_fc_hash(512, 32, 32));
        assert_ne!(a, compute_fc_hash(256, 64, 32));
    }
}
