//! LayerStack NNUE quantised binary の save / load。
//!
//! `bins/nnue_train` が出力する binary 形式。推論エンジン rshogi が `EvalFile=`
//! で直接読み込める byte layout。
//!
//! ## LayerStack architecture (Threat 無し、HandCountDense 無し)
//!
//! - L0 (FT): `ft_in → ft_out`、weight + bias 共有 stm/nstm (ft_in は feature set
//!   依存、ft_out は `--ft-out`)
//! - per-perspective post: bias add → CReLU → pairwise_mul (ft_out → ft_out/2) → ×127/128
//! - combined = stm.concat(nstm) = ft_out
//! - L1 (per-bucket delta + shared l1f factorized): `num_buckets` × l1_out + (ft_out, l1_out)
//! - l1_total = L1.select(bucket) + L1f、main (l1_out - 1) + skip (1) に slice
//! - L2 (per-bucket): `num_buckets` × l2_out with input l2_in = (l1_out - 1) * 2
//! - L3 (per-bucket output): `num_buckets` × 1
//! - PSQT shortcut (任意): `(ft_in, num_buckets)` の per-feature × per-bucket スカラー。
//!   forward は `net_output += 0.5 * (psqt[stm,bucket] - psqt[nstm,bucket])` で
//!   dense path と並列に加算される (Stockfish SFNNv10 系)。
//!
//! ## file layout (top-level)
//!
//! 1. header: nnue_version (4 LE u32) + network_hash (4 LE u32) + arch_len (4 LE u32) + arch_str
//! 2. **num_buckets (4 LE u32)** — current [`NNUE_VERSION`] のみ。bump 前の
//!    [`LEGACY_NNUE_VERSION_BUCKETS9`] は本 field を持たず暗黙 `num_buckets = 9`
//! 3. ft_hash (4 LE u32)
//! 4. ft_biases LEB128 (magic `COMPRESSED_LEB128` + size + signed LEB128 i16 列)
//! 5. ft_weights LEB128 (同上、piece 部分 = `halfka_dim * ft_out`、threat 無し)
//! 6. **PSQT block (arch_str に `PSQT={num_buckets},` がある場合のみ)**: bias i32 LE
//!    × num_buckets, weights i32 LE × halfka_dim × num_buckets (feature-major、各
//!    feat 内 bucket 連番)。scale は `QA * QB = 8128` (bias は 0 固定)。
//! 7. layerstacks: `num_buckets` × {fc_hash (4 LE u32), L1 (bias + weight), L2 (同), L3 (同)}
//!
//! ## save 時の L1 / L1f coalesce
//!
//! per-bucket l1 と shared l1f を **save 時に merge** して per-bucket の単一
//! weight として書き出す:
//!
//! - `l1_bias_merged[bucket][out] = l1_b[bucket][out] + l1f_b[out]` (bias broadcast)
//! - `l1_weight_merged[bucket][out][in] = l1_w[bucket][out][in] + l1f_w[in][out]` (in/out 軸入替注意)
//!
//! 推論エンジン rshogi は `Factorizer` を含む arch を reject する (coalesced only を要求)
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
//! (SIMD load の align 要求のため)。ft_out は 128 の倍数なので pad32(ft_out) = ft_out、
//! pad32(30) = 32, pad32(32) = 32。
//! padding byte は 0 で埋める。
//!
//! ## 重み layout
//!
//! file は **row-major** per bucket (`for out in 0..out_dim: for in in
//! 0..padded_in: write byte`)。本 crate のトレーナー側 weight も row-major
//! (`l1_w[bucket * out_dim * in_dim + out_idx * in_dim + in_idx]`) なので、
//! そのまま file row-major に書ける (転置不要)。

use std::io::{self, Read, Write};

use shogi_features::{EffectBucketConfig, FeatureSetSpec, FtFactorizeMode};

// =============================================================================
// constants (LayerStack architecture)
// =============================================================================

/// 現行 `.bin` format version。`arch_str` の直後に `num_buckets: u32` を持つ
/// self-describing layout。
pub const NNUE_VERSION: u32 = 0x7AF32F21;
/// `num_buckets` field が無かった旧 layout の version。load 時は暗黙
/// `num_buckets = 9` として処理する (bump 前の配布 net 互換)。save には使わない。
pub const LEGACY_NNUE_VERSION_BUCKETS9: u32 = 0x7AF32F20;
pub const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

// FT 入力次元は feature set ごと、FT 出力次元は `--ft-out`、L1 出力次元は `--l1`、
// L2 出力次元は `--l2`、bucket 数は `--num-buckets` ごとに異なる runtime 値。
/// 既定 FT 出力次元 (1 perspective あたり)。`--ft-out` 未指定時の値。
pub const DEFAULT_FT_OUT: usize = 1536;
/// 既定 L1 出力次元。`--l1` 未指定時の値。L1 出力は skip 1 dim を除いた残り
/// (`l1_out - 1`) を 2 乗 + 連結して L2 入力次元 `(l1_out - 1) * 2` にする。
pub const DEFAULT_L1_OUT: usize = 16;
/// 既定 L2 出力次元。`--l2` 未指定時の値。L2 per-bucket dense 層の出力幅で、
/// CReLU を経て L3 出力層の入力になる。
pub const DEFAULT_L2_OUT: usize = 32;
/// 既定 bucket 数。`--num-buckets` 未指定時、および legacy `.bin`
/// ([`LEGACY_NNUE_VERSION_BUCKETS9`]) の暗黙 N。
pub const DEFAULT_NUM_BUCKETS: usize = 9;

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
/// 推論エンジン rshogi 側 (`nnue/leb128.rs::read_signed_leb128`) で
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
/// 推論エンジン rshogi 側 (`read_compressed_tensor_i16_all`) で読み戻せる。
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

/// LayerStack arch description string を生成。PSQT 無し / Threat 無し /
/// HandCountDense 無しの最小形。
///
/// 形式 (実際は改行無しの 1 行):
///
/// `Features=<feature_name>(Friend)[<input_size>-><ft_out>x2],
///  Network=AffineTransform[1<-<l2_out>](
///  ClippedReLU[<l2_out>](
///  AffineTransform[<l2_out>-<l2_in>](
///  SqrClippedReLU[<l2_in>](
///  AffineTransform[<l1_out>-<ft_out_x2>](
///  InputSlice[<ft_out_x2>(0:<ft_out_x2>)]))))),
///  fv_scale=<fv_scale>`
///
/// (実際は 1 行連結、ここでは可読性のため改行)
///
/// `feature_name` は `FeatureSetSpec::arch_feature_name` (`HalfKaHmMerged` 等)。
/// arch 文字列の `Features=...` トークンを生成する。
///
/// `load_quantised` の reject policy はこのトークンを構造化フィールドの権威として
/// `starts_with` 照合するため、生成は本関数 1 箇所に集約する。
pub fn features_token(feature_name: &str, input_size: usize, ft_out: usize) -> String {
    format!("Features={feature_name}(Friend)[{input_size}->{ft_out}x2]")
}

/// arch_str / stream に書く threat profile identity (rshogi 契約)。`dims` はその
/// profile の THREAT_DIMENSIONS、`profile_id` は profile の数値 id (full=0 /
/// same-class=1 / same-class-major-pawn=2 / step-attacker=3 / full-symdedup=4 /
/// cross-side=10)。full-symdedup は dims が full と同一 (216_720) なので profile
/// の判別は id token のみが担う。
#[derive(Clone, Copy, Debug)]
pub struct ThreatArch {
    pub dims: usize,
    pub profile_id: u32,
}

/// arch_str / stream に書く effect bucket identity。`nb` は bucket 数、`king_bucketed`
/// は玉 feature も bucket 化するかを表す。
#[derive(Clone, Copy, Debug)]
pub struct EffectBucketArch {
    pub nb: usize,
    pub king_bucketed: bool,
}

impl EffectBucketArch {
    fn token_value(self) -> String {
        let king = if self.king_bucketed {
            "bucketed"
        } else {
            "fixed"
        };
        let axes = match self.nb {
            4 => "2x2",
            9 => "3x3",
            _ => panic!("unsupported EffectBucket bucket count: {}", self.nb),
        };
        format!("{axes}{king}")
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_arch_str(
    feature_name: &str,
    input_size: usize,
    ft_out: usize,
    l1_out: usize,
    l2_in: usize,
    l2_out: usize,
    fv_scale: i32,
    psqt_buckets: Option<usize>,
    threat: Option<ThreatArch>,
    effect_bucket: Option<EffectBucketArch>,
) -> String {
    let psqt_part = match psqt_buckets {
        Some(n) => format!("PSQT={n},"),
        None => String::new(),
    };
    // Threat token は PSQT の直後 (Features と Network の間)。threat 無効 (None) は
    // 出さないので base / PSQT-only の arch_str は byte-identical のまま。rshogi 契約:
    // `Threat=<dims>` (その profile の次元数) は threat 有効時は必ず出す (rshogi は
    // threat block の有無を `Threat=` の有無で判定するため、profile 0 でも必須)。
    // `ThreatProfile=<id>` (と stream の u32 id) は profile id≠0 のときだけ付け、
    // id 0 = full では `ThreatProfile=`/u32 のみ省略する (`Threat=<dims>` は残す →
    // rshogi の「ThreatProfile= 無し = profile 0」経路で load される)。
    let threat_part = match threat {
        Some(t) if t.profile_id != 0 => {
            format!("Threat={},ThreatProfile={},", t.dims, t.profile_id)
        }
        Some(t) => format!("Threat={},", t.dims),
        None => String::new(),
    };
    let effect_bucket_part = match effect_bucket {
        Some(e) => format!("EffectBucket={},", e.token_value()),
        None => String::new(),
    };
    format!(
        "{},{}{}{}\
         Network=AffineTransform[1<-{}](\
         ClippedReLU[{}](\
         AffineTransform[{}<-{}](\
         SqrClippedReLU[{}](\
         AffineTransform[{}<-{}](\
         InputSlice[{}(0:{})]))))),\
         fv_scale={}",
        features_token(feature_name, input_size, ft_out),
        psqt_part,
        threat_part,
        effect_bucket_part,
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

/// fc hash の計算 (nnue-pytorch 系の hash アルゴリズム)。
///
/// 推論エンジン rshogi は本 hash を skip する (`network_layer_stacks.rs`) が、
/// format 仕様上の正しい値として computed value を書き出す。hash は隠れ層の
/// out_features 列のみに依存するため L2 入力次元は引数に取らない。
pub const fn compute_fc_hash(ft_out: usize, l2_out: usize) -> u32 {
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

/// FT hash: `feature_hash ^ (ft_out * 2)`。feature 定数 (feature set 由来) と
/// FT 出力次元の合成。`feature_hash` は `FeatureSetSpec::feature_hash`。
pub fn ft_hash(feature_hash: u32, ft_out: usize) -> u32 {
    feature_hash ^ (ft_out as u32 * 2)
}

/// network hash: `compute_fc_hash(ft_out, l2_out) ^ ft_hash(feature_hash, ft_out)`。
/// `l2_out` は `compute_fc_hash` の隠れ層列に織り込まれるため、`--l2` の異なる
/// `.bin` は network_hash が変わり `load_quantised` で reject される。
pub fn network_hash(feature_hash: u32, ft_out: usize, l2_out: usize) -> u32 {
    compute_fc_hash(ft_out, l2_out) ^ ft_hash(feature_hash, ft_out)
}

/// FT factorizer の仮想行を実行へ畳み込み、export 形状
/// (`feature_set.ft_in() × ft_out`) の FT weight を返す。
///
/// 学習時の FT weight は `[base real | threat real | piece-input 仮想行]` の
/// `train_ft_in × ft_out` (row-major、`ft_w[feat * ft_out + out]`)。piece-input 仮想行は
/// 玉位置に依らない駒価値を持つ。base factorizer は駒ごとに 1 仮想行を base の
/// king-bucket 数 (HalfKA_hm 系では 45) で共有する。effect bucket は各駒特徴を NB 個の被攻撃×被防御バケットに分割し、
/// `PoolEffectBuckets` では駒ごとに 1 仮想行を全バケットで共有し、
/// `PerEffectBucket` では (駒, バケット) ごとに仮想行を持つ。export では実行に
/// 仮想行を畳み込んで固定し、仮想行を捨てる。threat real 行
/// (`[base_ft_in, ft_in())`) は仮想行を持たないので **そのまま残す** (silent に
/// 切り落とさない)。戻り値は base 折り込み済み + threat 不可触の `ft_in() × ft_out`。
/// 量子化と飽和検査 (`warn_if_i16_saturates`) は畳み込み後の値に掛けること
/// (caller は本関数の戻り値で `LayerStackWeights::ft_w` を構築)。
///
/// factorizer 無効の spec では入力をそのまま返す。
pub fn coalesce_ft_factorized(
    feature_set: &FeatureSetSpec,
    ft_out: usize,
    ft_w_train: &[f32],
) -> Vec<f32> {
    if !feature_set.ft_factorize() {
        return ft_w_train.to_vec();
    }
    let base_ft_in = feature_set.base_ft_in();
    let ft_in = feature_set.ft_in(); // base + threat (= 仮想行の手前まで)
    let piece_inputs = feature_set.piece_inputs();
    let nb = feature_set.effect_bucket_config().map_or(1, |cfg| cfg.nb);
    let train_ft_in = feature_set.train_ft_in();
    assert_eq!(
        ft_w_train.len(),
        train_ft_in * ft_out,
        "ft_w length must be train_ft_in * ft_out"
    );
    // piece-input 仮想行は base+threat 実行の後ろ。export は実行部 (base + threat) を
    // まず複製し、base 実行に対応する仮想行を加算する。
    let virtual_base = ft_in * ft_out;
    let mut out = ft_w_train[..virtual_base].to_vec();
    let fold_ft_in = if feature_set.effect_bucket_config().is_some() {
        ft_in
    } else {
        base_ft_in
    };
    for feat in 0..fold_ft_in {
        let p = match feature_set.ft_factorize_mode() {
            FtFactorizeMode::Base => feat % piece_inputs,
            FtFactorizeMode::PoolEffectBuckets | FtFactorizeMode::PerEffectBucket => {
                (feat / nb) % piece_inputs
            }
        };
        let vrow = match feature_set.ft_factorize_mode() {
            FtFactorizeMode::PerEffectBucket => p * nb + feat % nb,
            FtFactorizeMode::Base | FtFactorizeMode::PoolEffectBuckets => p,
        };
        let dst = feat * ft_out;
        let src = virtual_base + vrow * ft_out;
        for o in 0..ft_out {
            out[dst + o] += ft_w_train[src + o];
        }
    }
    out
}

// =============================================================================
// LayerStackWeights — トレーナー側 weight 表現 (f32、kernel と同 layout)
// =============================================================================

/// LayerStack の全 weight (f32、host 側保持)。
///
/// Layout は本 crate trainer の kernel 内部 layout と一致 (`ft_in` =
/// `feature_set.ft_in()`、`num_buckets` は self field):
/// - `ft_w`: `(ft_in, ft_out)` row-major、`ft_w[feat * ft_out + out]`
/// - `ft_b`: `(ft_out)` (stm/nstm 共有)
/// - `l1_w`: `(num_buckets, l1_out, ft_out)` row-major、`l1_w[buc * l1_out * ft_out + out * ft_out + in]`
/// - `l1_b`: `(num_buckets, l1_out)` row-major
/// - `l1f_w`: `(ft_out, l1_out)` row-major、`l1f_w[in * l1_out + out]`
/// - `l1f_b`: `(l1_out)` — FT 出力同様、長さがそのまま L1 出力次元 `l1_out`
/// - `l2_w`: `(num_buckets, l2_out, l2_in)` row-major、`l2_in = (l1_out - 1) * 2`
/// - `l2_b`: `(num_buckets, l2_out)`
/// - `l3_w`: `(num_buckets, l2_out)` (out_dim=1 なので out 軸省略)
/// - `l3_b`: `(num_buckets)`
#[derive(Debug, Clone)]
pub struct LayerStackWeights {
    /// この weight が属する feature set。FT 入力次元 (`ft_in`) と
    /// artifact identity (arch 文字列 / hash) の単一の真実源。
    pub feature_set: FeatureSetSpec,
    /// LayerStack output bucket count (`--num-buckets`)。per-bucket weight
    /// buffer の bucket 軸長と、save / load 時の `.bin` `num_buckets` field を
    /// 駆動する。
    pub num_buckets: usize,
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
    /// PSQT shortcut weight (任意)、長さ `ft_in * num_buckets`、layout
    /// `psqt_w[feat * num_buckets + bucket]` (feature-major、各 feat 内 bucket 連番)。
    /// `Some` で save 時に PSQT block が出力され、arch 文字列に `PSQT={num_buckets},`
    /// token が入る。`None` (既定) は PSQT 無し layout。
    pub psqt_w: Option<Vec<f32>>,
}

impl LayerStackWeights {
    /// 全 buffer を 0 で初期化した新規 instance。FT 入力次元は
    /// `feature_set.ft_in()`、FT 出力次元は `ft_out` (`--ft-out`)、L1 出力次元は
    /// `l1_out` (`--l1`)、L2 出力次元は `l2_out` (`--l2`)、bucket 数は
    /// `num_buckets` (`--num-buckets`) で決まる。L2 入力次元 `l2_in` は `l1_out`
    /// から導出する。
    pub fn zeroed(
        feature_set: FeatureSetSpec,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
        num_buckets: usize,
    ) -> Self {
        assert!(num_buckets >= 1, "num_buckets must be >= 1");
        let l2_in = (l1_out - 1) * 2;
        Self {
            feature_set,
            num_buckets,
            ft_w: vec![0.0; feature_set.ft_in() * ft_out],
            ft_b: vec![0.0; ft_out],
            l1_w: vec![0.0; num_buckets * l1_out * ft_out],
            l1_b: vec![0.0; num_buckets * l1_out],
            l1f_w: vec![0.0; ft_out * l1_out],
            l1f_b: vec![0.0; l1_out],
            l2_w: vec![0.0; num_buckets * l2_out * l2_in],
            l2_b: vec![0.0; num_buckets * l2_out],
            l3_w: vec![0.0; num_buckets * l2_out],
            l3_b: vec![0.0; num_buckets],
            psqt_w: None,
        }
    }

    /// PSQT shortcut weight 領域を 0 で確保した版 (PSQT 有効時用)。
    /// `psqt_w` は長さ `feature_set.ft_in() * num_buckets` の `Some` で確保される。
    pub fn zeroed_with_psqt(
        feature_set: FeatureSetSpec,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
        num_buckets: usize,
    ) -> Self {
        let mut s = Self::zeroed(feature_set, ft_out, l1_out, l2_out, num_buckets);
        s.psqt_w = Some(vec![0.0; feature_set.ft_in() * num_buckets]);
        s
    }

    /// LayerStack quantised.bin を `writer` に書き出す。推論エンジン rshogi の
    /// `NetworkLayerStacks::read` で parse できる byte layout (`num_buckets` を
    /// header から読む対応が rshogi 側で要る)。
    pub fn save_quantised<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // ---- header ---- (arch 文字列・hash は feature set + 層次元から導出)
        // FT 出力次元は FT bias buffer の長さ、L1 出力次元は L1f bias buffer の長さ
        // (どちらも 1 perspective / 1 dim あたり 1 要素)。L2 出力次元は L2 bias buffer の
        // 長さを bucket 数で割った値 (`l2_b` は `(num_buckets, l2_out)`)。L2 入力次元は
        // L1 出力から導出。
        let num_buckets = self.num_buckets;
        if num_buckets == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "num_buckets must be >= 1",
            ));
        }
        // header の `num_buckets` field は u32。silent truncation を起こさず
        // overflow 時に InvalidInput で reject する。
        let num_buckets_u32 = u32::try_from(num_buckets).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("num_buckets {num_buckets} does not fit in u32 header field"),
            )
        })?;
        let ft_out = self.ft_b.len();
        let l1_out = self.l1f_b.len();
        let l2_out = self.l2_b.len() / num_buckets;
        let l2_in = (l1_out - 1) * 2;
        let feature_hash = self.feature_set.feature_hash();
        writer.write_all(&NNUE_VERSION.to_le_bytes())?;
        writer.write_all(&network_hash(feature_hash, ft_out, l2_out).to_le_bytes())?;
        let psqt_buckets = if self.psqt_w.is_some() {
            Some(num_buckets)
        } else {
            None
        };
        // arch_str の threat token は rshogi 契約 = `Threat=<dims>` (+ id≠0 のとき
        // `ThreatProfile=<id>`)。dims/id は spec の profile から導出する。
        let threat_arch = self.feature_set.threat_profile().map(|p| ThreatArch {
            dims: self.feature_set.threat_dims(),
            profile_id: p.profile_id(),
        });
        let effect_bucket_arch =
            self.feature_set
                .effect_bucket_config()
                .map(|c| EffectBucketArch {
                    nb: c.nb,
                    king_bucketed: c.king_bucketed,
                });
        let arch_str = build_arch_str(
            self.feature_set.arch_feature_name(),
            self.feature_set.ft_in(),
            ft_out,
            l1_out,
            l2_in,
            l2_out,
            FV_SCALE,
            psqt_buckets,
            threat_arch,
            effect_bucket_arch,
        );
        let arch_bytes = arch_str.as_bytes();
        writer.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(arch_bytes)?;

        // ---- num_buckets (NNUE_VERSION bump 後の新 field) ----
        writer.write_all(&num_buckets_u32.to_le_bytes())?;

        // ---- FT hash ----
        writer.write_all(&ft_hash(feature_hash, ft_out).to_le_bytes())?;

        // ---- FT biases LEB128 (i16, scale=QA) ----
        // FT weight/bias は training 中 clamp されない (i16 飽和域 ±i16::MAX/QA まで
        // 開放) ため、recipe によっては量子化で silent clip されうる。export 側で
        // 飽和件数を数え、発生していれば警告する (clamp 自体は数値破綻を防ぐが、
        // 無言の情報損失に気付けるようにする)。
        let qa_f = QA as f64;
        warn_if_i16_saturates("ft_b", &self.ft_b, qa_f);
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
        if self.ft_w.len() != self.feature_set.ft_in() * ft_out {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ft_w length {} != ft_in {} * ft_out {}",
                    self.ft_w.len(),
                    self.feature_set.ft_in(),
                    ft_out,
                ),
            ));
        }
        // threat は base FT block と raw i8 threat block に分ける。effect bucket は追加 block
        // を持たず、拡張済み row 全体を通常の i16 FT block に書く。
        let base_ft_w_n = if self.feature_set.threat_profile().is_some() {
            self.feature_set.base_ft_in() * ft_out
        } else {
            self.feature_set.ft_in() * ft_out
        };
        let ft_w_base = &self.ft_w[..base_ft_w_n];
        warn_if_i16_saturates("ft_w", ft_w_base, qa_f);
        let ft_w_i16: Vec<i16> = ft_w_base
            .iter()
            .map(|&v| {
                (qa_f * v as f64)
                    .round()
                    .clamp(i16::MIN as f64, i16::MAX as f64) as i16
            })
            .collect();
        write_leb128_tensor_i16(writer, &ft_w_i16)?;

        // ---- PSQT block (arch_str に PSQT={num_buckets}, が入っているときのみ) ----
        // bias は常に 0 固定 (forward 対称差で friend/enemy が打ち消し、勾配も常に 0)
        // を i32 LE で num_buckets 個書く。weights は `ft_in * num_buckets` を
        // feature-major (`psqt_w[feat * num_buckets + bucket]`) でそのまま i32 LE 列に
        // 書き出す。scale は `QA * QB = 8128`。
        if let Some(psqt) = self.psqt_w.as_ref() {
            let ft_in = self.feature_set.ft_in();
            let expected = ft_in * num_buckets;
            if psqt.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "psqt_w length {} != expected {} (= ft_in {} * num_buckets {})",
                        psqt.len(),
                        expected,
                        ft_in,
                        num_buckets,
                    ),
                ));
            }
            let psqt_scale = (QA * QB) as f64;
            for _ in 0..num_buckets {
                writer.write_all(&0_i32.to_le_bytes())?;
            }
            for &w in psqt {
                let val = (psqt_scale * w as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        // ---- Threat block (threat profile 有効時のみ) ----
        // rshogi 契約: profile id≠0 のとき **u32 LE id** を threat block 直前に書き、
        // 続けて threat weight (`ft_w[base_ft_in*ft_out ..]`) を **raw i8 feature-major**
        // で書く。id 0 (full) は u32 を省略 (rshogi の「id 無し = profile 0」経路)。
        // **量子化スケールは base FT と同じ `qa_f` を据え置く** (rshogi は threat i8 を
        // base FT と同一 i16 accumulator に sign-extend して直接加算するため、別係数を
        // 入れると eval が破綻する)。i16 範囲ではなく **±127 に clamp** して i8 化する。
        // PSQT と threat は CLI 排他なので両 block が同時に出ることはない。
        if let Some(profile) = self.feature_set.threat_profile() {
            let threat_w = &self.ft_w[base_ft_w_n..];
            warn_if_i8_saturates("threat_ft_w", threat_w, qa_f);
            if profile.profile_id() != 0 {
                writer.write_all(&profile.profile_id().to_le_bytes())?;
            }
            for &v in threat_w {
                let q = clamp_i8_symmetric((qa_f * v as f64).round());
                writer.write_all(&[q as u8])?;
            }
        }

        // ---- LayerStacks (num_buckets × {fc_hash, L1, L2, L3}) ----
        let qb_f = QB as f64;
        let l1_bias_scale = (QA * QB) as f64; // = 8128
        let l2_bias_scale = 127.0 * qb_f; // = 8128 (QA == 127 前提)
        let l3_bias_scale = 127.0 * qb_f; // = 8128

        let fc_hash = compute_fc_hash(ft_out, l2_out);
        for buc in 0..num_buckets {
            // fc_hash per bucket
            writer.write_all(&fc_hash.to_le_bytes())?;

            // --- L1 (merged delta + shared) ---
            // Biases: l1_out i32 scale = QA*QB = 8128
            for out in 0..l1_out {
                let merged = self.l1_b[buc * l1_out + out] + self.l1f_b[out];
                let val = (l1_bias_scale * merged as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
            // Weights: l1_out × pad32(ft_out) i8 scale = QB
            // For each (buc, out, in) in [0, l1_out) × [0, pad32(ft_out))
            // - in < ft_out: merged = l1_w[buc][out][in] + l1f_w[in][out]
            // - else: padding 0
            let l1_padded_in = pad32(ft_out);
            for out in 0..l1_out {
                for in_idx in 0..l1_padded_in {
                    let q: i8 = if in_idx < ft_out {
                        let buc_w = self.l1_w[buc * l1_out * ft_out + out * ft_out + in_idx];
                        let shared_w = self.l1f_w[in_idx * l1_out + out];
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
            for out in 0..l2_out {
                let val = (l2_bias_scale * self.l2_b[buc * l2_out + out] as f64).round() as i32;
                writer.write_all(&val.to_le_bytes())?;
            }
            // Weights: l2_out × pad32(l2_in) i8 scale = QB
            let l2_padded_in = pad32(l2_in);
            for out in 0..l2_out {
                for in_idx in 0..l2_padded_in {
                    let q: i8 = if in_idx < l2_in {
                        let w = self.l2_w[buc * l2_out * l2_in + out * l2_in + in_idx];
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
            let l3_padded_in = pad32(l2_out);
            for in_idx in 0..l3_padded_in {
                let q: i8 = if in_idx < l2_out {
                    let w = self.l3_w[buc * l2_out + in_idx];
                    clamp_i8((qb_f * w as f64).round())
                } else {
                    0_i8
                };
                writer.write_all(&[q as u8])?;
            }
        }

        Ok(())
    }

    /// LayerStack quantised.bin を parse し `LayerStackWeights` を返す。
    ///
    /// 注: save 時に per-bucket l1 と shared l1f は merge されて書き出されるため、
    /// load 時には分離不能。本実装は **l1_w に merged 値をそのまま入れ、l1f_w /
    /// l1f_b は 0 にする** 方針 (forward 計算は等価)。
    ///
    /// **継続学習時の注意**: forward は等価でも、l1f が「shared factorized 部」と
    /// しての意味は失われる (全て l1_w に畳まれた状態)。per-bucket l1 と shared
    /// l1f を別々に学習し続ける場合と勾配の流れ方が変わるため、本 method で得た
    /// `LayerStackWeights` から continue-training すると factorize を保ったまま
    /// 学習した場合の軌跡とは厳密一致しない。「pretrained 注入 → 1 step → save し、
    /// 出力 `.bin` が参照と byte 単位で一致するか」を確認する用途、あるいは l1f を
    /// 再び factorize し直す前提なら問題ない。
    /// `expected` は要求 feature set、`ft_out` は要求 FT 出力次元 (`--ft-out`)、
    /// `l1_out` は要求 L1 出力次元 (`--l1`)、`l2_out` は要求 L2 出力次元 (`--l2`)、
    /// `num_buckets` は要求 bucket 数 (`--num-buckets`)。file の arch 文字列・hash・
    /// `num_buckets` field・各 layer の byte 数がこれらと一致しなければ `InvalidData`
    /// で reject する (reject policy)。`l2_out` は `network_hash` に織り込まれるため、
    /// 不一致は hash mismatch で弾かれる。legacy `.bin`
    /// ([`LEGACY_NNUE_VERSION_BUCKETS9`]) は `num_buckets` field を持たず暗黙
    /// `num_buckets = 9` として扱われ、`num_buckets != 9` を要求した場合は
    /// `InvalidData` で reject する (古い配布 net のままだと N != 9 に切り替えられない、
    /// 想定挙動)。
    /// PSQT shortcut 層を含む `.bin` を load する場合は [`Self::load_quantised_with_psqt`]
    /// を使う。本 method は PSQT 無しを要求し、PSQT を含む `.bin` は `Unsupported`
    /// で reject する。
    pub fn load_quantised<R: Read>(
        reader: &mut R,
        expected: FeatureSetSpec,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
        num_buckets: usize,
    ) -> io::Result<Self> {
        Self::load_quantised_with_psqt(reader, expected, ft_out, l1_out, l2_out, num_buckets, false)
    }

    /// PSQT shortcut の有無を caller が指定する load。`with_psqt = true` で
    /// arch_str に `PSQT={num_buckets},` を要求 + PSQT block を読む、`false` で
    /// PSQT 無しを要求。不一致 (要求 true で `.bin` 側無し / 要求 false で `.bin`
    /// 側有り) は `InvalidData` で reject する。
    pub fn load_quantised_with_psqt<R: Read>(
        reader: &mut R,
        expected: FeatureSetSpec,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
        num_buckets: usize,
        with_psqt: bool,
    ) -> io::Result<Self> {
        assert!(num_buckets >= 1, "num_buckets must be >= 1");
        // L1 出力 (skip 1 dim を除く) を 2 乗 + 連結した L2 入力次元。
        let l2_in = (l1_out - 1) * 2;
        let mut buf4 = [0u8; 4];

        // version (current NNUE_VERSION か legacy LEGACY_NNUE_VERSION_BUCKETS9 を受理、
        // それ以外は reject。legacy は暗黙 num_buckets = 9 として処理する)。
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unknown NNUE version: {version:#x} (expected {NNUE_VERSION:#x} or legacy \
                     {LEGACY_NNUE_VERSION_BUCKETS9:#x})"
                ),
            ));
        }
        let file_is_legacy = version == LEGACY_NNUE_VERSION_BUCKETS9;

        // network_hash — 要求 feature set 由来の値と照合 (整合性チェック)。
        reader.read_exact(&mut buf4)?;
        let file_network_hash = u32::from_le_bytes(buf4);
        let expected_network_hash = network_hash(expected.feature_hash(), ft_out, l2_out);
        if file_network_hash != expected_network_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "network_hash mismatch: file {file_network_hash:#x}, \
                     expected {expected_network_hash:#x} for feature set {}",
                    expected.canonical_name()
                ),
            ));
        }

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

        // num_buckets field (current NNUE_VERSION のみ。legacy は暗黙 9 として扱う)。
        // legacy `.bin` を非 9 で load しようとした場合は明示 reject。
        let file_num_buckets = if file_is_legacy {
            if num_buckets != DEFAULT_NUM_BUCKETS {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "legacy NNUE version {LEGACY_NNUE_VERSION_BUCKETS9:#x} has implicit \
                         num_buckets = {DEFAULT_NUM_BUCKETS}, but {num_buckets} was requested \
                         (re-save the `.bin` with a current build to load it at non-default \
                         bucket counts)"
                    ),
                ));
            }
            DEFAULT_NUM_BUCKETS
        } else {
            reader.read_exact(&mut buf4)?;
            let v = u32::from_le_bytes(buf4) as usize;
            if v != num_buckets {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("num_buckets mismatch: file {v}, expected {num_buckets}"),
                ));
            }
            v
        };

        // HandCount は未対応 (reject のまま)。Threat は `expected` の profile と
        // arch_str の `Threat={profile},` token が一致するか検証する: token 有無 /
        // profile 名のどちらが食い違っても InvalidData で弾く (profile compaction で
        // row の意味が変わるため silent な row ずれを防ぐ)。
        if arch_str.contains("HandCount") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported arch (HandCount not implemented): {arch_str}"),
            ));
        }
        // rshogi 契約の threat token を構造的に検証する。`Threat=<dims>` (threat 有効
        // 時必須) + `ThreatProfile=<id>` (id≠0 のみ)。`parse_single_arch_token` で
        // comma 分割 + 先頭厳密一致 + 重複拒否。期待 profile の dims / id と完全一致を
        // 要求 (id 0 は ThreatProfile token 無しが正しい状態)。
        let file_threat_dims = parse_single_arch_token(&arch_str, "Threat=").map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("arch_str has multiple Threat= tokens: `{arch_str}`"),
            )
        })?;
        let file_threat_profile_id =
            parse_single_arch_token(&arch_str, "ThreatProfile=").map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("arch_str has multiple ThreatProfile= tokens: `{arch_str}`"),
                )
            })?;
        let file_effect_bucket =
            parse_single_arch_token(&arch_str, "EffectBucket=").map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("arch_str has multiple EffectBucket= tokens: `{arch_str}`"),
                )
            })?;
        let expected_threat = expected
            .threat_profile()
            .map(|p| (expected.threat_dims(), p.profile_id()));
        match expected_threat {
            Some((dims, id)) => {
                // dims token は数値で一致すること。
                let file_dims: Option<usize> = file_threat_dims.and_then(|s| s.parse().ok());
                if file_dims != Some(dims) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "threat dims mismatch: file Threat={}, expected Threat={dims} (arch_str = `{arch_str}`)",
                            file_threat_dims.unwrap_or("none"),
                        ),
                    ));
                }
                // id≠0 は ThreatProfile token が数値一致、id 0 は token 不在が正しい。
                let file_id: u32 = match file_threat_profile_id {
                    Some(s) => s.parse().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("invalid ThreatProfile= value `{s}` in arch_str"),
                        )
                    })?,
                    None => 0,
                };
                if file_id != id {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "threat profile id mismatch: file {file_id}, expected {id} (arch_str = `{arch_str}`)"
                        ),
                    ));
                }
            }
            None => {
                if file_threat_dims.is_some() || file_threat_profile_id.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "`.bin` has a Threat token but the requested feature set has no threat \
                             profile: `{arch_str}`"
                        ),
                    ));
                }
            }
        }
        let expected_effect_bucket = expected
            .effect_bucket_config()
            .map(effect_bucket_arch_token_value);
        match (expected_effect_bucket.as_deref(), file_effect_bucket) {
            (Some(expected_token), Some(file_token)) if expected_token == file_token => {}
            (Some(expected_token), Some(file_token)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "effect bucket config mismatch: file EffectBucket={file_token}, expected EffectBucket={expected_token} \
                         (arch_str = `{arch_str}`)"
                    ),
                ));
            }
            (Some(expected_token), None) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "effect bucket requested but not present in arch_str (expected EffectBucket={expected_token}): `{arch_str}`"
                    ),
                ));
            }
            (None, Some(file_token)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "`.bin` has EffectBucket={file_token} but the requested feature set has no effect bucket config: `{arch_str}`"
                    ),
                ));
            }
            (None, None) => {}
        }
        let psqt_token = format!("PSQT={file_num_buckets},");
        let file_has_psqt = arch_str.contains(&psqt_token);
        let file_has_any_psqt = arch_str.contains("PSQT=");
        // bucket 数不一致 PSQT (`PSQT=N` で N != file の num_buckets) は構造的に壊れた
        // file。num_buckets 検証で既に弾かれるはずだが defensive に reject する。
        if file_has_any_psqt && !file_has_psqt {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "PSQT bucket count in arch_str does not match num_buckets={file_num_buckets}: \
                     `{arch_str}`"
                ),
            ));
        }
        if with_psqt && !file_has_psqt {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSQT requested but not present in arch_str: `{arch_str}` \
                     (expected `{psqt_token}` token)"
                ),
            ));
        }
        if !with_psqt && file_has_psqt {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSQT not requested but `.bin` arch_str has PSQT: `{arch_str}` \
                     (use load_quantised_with_psqt with with_psqt=true to load PSQT models)"
                ),
            ));
        }
        // feature set の構造化フィールド (feature 名 + ft_in) を arch 文字列の
        // `Features=...` 前置部で直接照合する (reject policy の authority)。
        let expected_features_prefix =
            features_token(expected.arch_feature_name(), expected.ft_in(), ft_out);
        if !arch_str.starts_with(&expected_features_prefix) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "feature set mismatch: expected `{expected_features_prefix}` \
                     (feature set {}), file arch_str = `{arch_str}`",
                    expected.canonical_name()
                ),
            ));
        }
        // L1 出力次元の照合。network_hash / ft_hash は L1 出力次元に依存しないため、
        // `l1_out` 不一致は hash では弾けない。arch 文字列の L1 affine token
        // (`build_arch_str` が `AffineTransform[<l1_out>-<ft_out*2>]` で書く部分) を
        // 直接照合し、`--l1` と異なる `.bin` を InvalidData で reject する。
        let expected_l1_token = format!("AffineTransform[{l1_out}<-{}]", ft_out * 2);
        if !arch_str.contains(&expected_l1_token) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "L1 output dim mismatch: expected `{expected_l1_token}`, \
                     file arch_str = `{arch_str}`"
                ),
            ));
        }

        // ft_hash — 要求 feature set 由来の値と照合。
        reader.read_exact(&mut buf4)?;
        let file_ft_hash = u32::from_le_bytes(buf4);
        let expected_ft_hash = ft_hash(expected.feature_hash(), ft_out);
        if file_ft_hash != expected_ft_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ft_hash mismatch: file {file_ft_hash:#x}, \
                     expected {expected_ft_hash:#x} for feature set {}",
                    expected.canonical_name()
                ),
            ));
        }

        // FT biases (LEB128 i16, ft_out 個)
        let ft_b_i16 = read_leb128_tensor_i16(reader, Some(ft_out))?;
        let qa_f = QA as f32;
        let ft_b: Vec<f32> = ft_b_i16.iter().map(|&v| v as f32 / qa_f).collect();

        // FT weights (LEB128 i16)。threat row は PSQT block の後ろの Threat block
        // で読む。effect bucket は拡張済み row 全体をこの block に持つ。
        let base_ft_w_n = if expected.threat_profile().is_some() {
            expected.base_ft_in() * ft_out
        } else {
            expected.ft_in() * ft_out
        };
        let ft_w_i16 = read_leb128_tensor_i16(reader, Some(base_ft_w_n))?;
        let mut ft_w: Vec<f32> = ft_w_i16.iter().map(|&v| v as f32 / qa_f).collect();

        // PSQT block (with_psqt = true のときのみ): bias i32 × num_buckets + weights
        // i32 × (ft_in * num_buckets) を feature-major row-major で読む。scale は
        // `QA * QB = 8128` (save 側と対称)。
        //
        // PSQT bias は forward の対称差 (friend / enemy が同じ bias を打ち消す) と
        // backward の構造的勾配ゼロにより常に 0 で training される。本 trainer / kernel
        // は PSQT bias を内部状態として持たず、`.bin` 上も常に 0 を書く。**`.bin` 側に
        // 非ゼロ bias が入っていた場合は silent drop せず error で reject する** —
        // 別実装由来などで非ゼロ bias が混入した場合に inference round-trip が破綻する
        // のを防ぐため。
        let psqt_bias_scale = (QA * QB) as f32;
        let psqt_weight_scale = psqt_bias_scale;
        let psqt_w: Option<Vec<f32>> = if with_psqt {
            for bucket in 0..num_buckets {
                let mut bbuf = [0u8; 4];
                reader.read_exact(&mut bbuf)?;
                let raw = i32::from_le_bytes(bbuf);
                if raw != 0 {
                    let dequant = raw as f32 / psqt_bias_scale;
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "PSQT bias bucket {bucket} is non-zero (raw i32 {raw}, dequant {dequant}); \
                             tatara LayerStack PSQT is structurally bias-free (symmetric-diff design) \
                             and cannot represent non-zero PSQT bias"
                        ),
                    ));
                }
            }
            let n = expected.ft_in() * num_buckets;
            let mut w = Vec::with_capacity(n);
            let mut wbuf = [0u8; 4];
            for _ in 0..n {
                reader.read_exact(&mut wbuf)?;
                let q = i32::from_le_bytes(wbuf);
                w.push(q as f32 / psqt_weight_scale);
            }
            Some(w)
        } else {
            None
        };

        // Threat block (threat profile 有効時のみ): rshogi 契約と対称に read する。
        // id≠0 のとき u32 LE id を先に読んで照合し、続けて threat weight を **raw i8
        // feature-major** で読み、base FT と同じ `qa_f` で割って ft_w の base row の
        // 後ろに連結する (save 側と同一スケール、id 0 は u32 を読まない)。
        if let Some(profile) = expected.threat_profile() {
            if profile.profile_id() != 0 {
                reader.read_exact(&mut buf4)?;
                let file_id = u32::from_le_bytes(buf4);
                if file_id != profile.profile_id() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "threat block profile id mismatch: stream {file_id}, expected {}",
                            profile.profile_id()
                        ),
                    ));
                }
            }
            let threat_w_n = expected.threat_dims() * ft_out;
            let mut byte = [0u8; 1];
            ft_w.reserve(threat_w_n);
            for _ in 0..threat_w_n {
                reader.read_exact(&mut byte)?;
                ft_w.push(byte[0] as i8 as f32 / qa_f);
            }
        }

        // LayerStacks (num_buckets × {fc_hash, L1, L2, L3})
        let qb_f = QB as f32;
        let l1_bias_scale = (QA * QB) as f32;
        let l2_bias_scale = 127.0 * qb_f;
        let l3_bias_scale = 127.0 * qb_f;

        let mut l1_w = vec![0.0_f32; num_buckets * l1_out * ft_out];
        let mut l1_b = vec![0.0_f32; num_buckets * l1_out];
        let mut l2_w = vec![0.0_f32; num_buckets * l2_out * l2_in];
        let mut l2_b = vec![0.0_f32; num_buckets * l2_out];
        let mut l3_w = vec![0.0_f32; num_buckets * l2_out];
        let mut l3_b = vec![0.0_f32; num_buckets];

        let l1_padded_in = pad32(ft_out);
        let l2_padded_in = pad32(l2_in);
        let l3_padded_in = pad32(l2_out);

        for buc in 0..num_buckets {
            // fc_hash (skip)
            reader.read_exact(&mut buf4)?;

            // L1 biases (i32 × l1_out)
            for out in 0..l1_out {
                reader.read_exact(&mut buf4)?;
                let v = i32::from_le_bytes(buf4);
                l1_b[buc * l1_out + out] = v as f32 / l1_bias_scale;
            }
            // L1 weights (i8 × l1_out × l1_padded_in)、保存値は merged
            // → l1_w に直接書き込み、l1f_w は 0 のまま (forward 等価)
            for out in 0..l1_out {
                for in_idx in 0..l1_padded_in {
                    let mut buf1 = [0u8; 1];
                    reader.read_exact(&mut buf1)?;
                    if in_idx < ft_out {
                        let q = buf1[0] as i8;
                        l1_w[buc * l1_out * ft_out + out * ft_out + in_idx] = q as f32 / qb_f;
                    }
                    // padding 部分は破棄
                }
            }

            // L2 biases
            for out in 0..l2_out {
                reader.read_exact(&mut buf4)?;
                let v = i32::from_le_bytes(buf4);
                l2_b[buc * l2_out + out] = v as f32 / l2_bias_scale;
            }
            // L2 weights
            for out in 0..l2_out {
                for in_idx in 0..l2_padded_in {
                    let mut buf1 = [0u8; 1];
                    reader.read_exact(&mut buf1)?;
                    if in_idx < l2_in {
                        let q = buf1[0] as i8;
                        l2_w[buc * l2_out * l2_in + out * l2_in + in_idx] = q as f32 / qb_f;
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
                if in_idx < l2_out {
                    let q = buf1[0] as i8;
                    l3_w[buc * l2_out + in_idx] = q as f32 / qb_f;
                }
            }
        }

        Ok(Self {
            feature_set: expected,
            num_buckets,
            ft_w,
            ft_b,
            l1_w,
            l1_b,
            l1f_w: vec![0.0; ft_out * l1_out], // save 時に l1_w に merge 済 → load 側は 0
            l1f_b: vec![0.0; l1_out],
            l2_w,
            l2_b,
            l3_w,
            l3_b,
            psqt_w,
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

/// threat block 用の symmetric clamp (`[-127, 127]`)。base FT と同じ QA=127 スケールで
/// 量子化するため i8 の非対称下限 (-128) は使わず ±127 に揃える。飽和モニタ
/// (`i8_saturation_stats` の ±127 判定) と契約コメント (「±127 に clamp」) を一致させる。
fn clamp_i8_symmetric(v: f64) -> i8 {
    if v < -127.0 {
        -127
    } else if v > 127.0 {
        127
    } else {
        v as i8
    }
}

/// arch_str を comma 分割し `prefix` (`"Threat="` 等) で始まる token の値を返す。
/// token が無ければ `Ok(None)`、複数あれば `Err(())` (構造的に壊れた arch_str)。
/// substring 照合と違い、`prefix` が token 先頭に厳密一致する 1 区切りのみ拾う
/// (`Threat=fullX` のような余分 suffix も別 token として扱われ完全一致照合で弾く)。
fn parse_single_arch_token<'a>(arch_str: &'a str, prefix: &str) -> Result<Option<&'a str>, ()> {
    let mut found: Option<&'a str> = None;
    for tok in arch_str.split(',') {
        if let Some(value) = tok.strip_prefix(prefix) {
            if found.is_some() {
                return Err(());
            }
            found = Some(value);
        }
    }
    Ok(found)
}

fn effect_bucket_arch_token_value(config: EffectBucketConfig) -> String {
    EffectBucketArch {
        nb: config.nb,
        king_bucketed: config.king_bucketed,
    }
    .token_value()
}

/// `round(scale·v)` が i16 範囲 `[-32768, 32767]` を超える要素数を数える。`scale`
/// は FT 量子化スケール QA。値域に収まれば 0。FT i16 量子化が情報を落とす要素を
/// export 前に把握するための pure helper。
fn count_i16_saturations(values: &[f32], scale: f64) -> usize {
    values
        .iter()
        .filter(|&&v| {
            let q = (scale * v as f64).round();
            q < i16::MIN as f64 || q > i16::MAX as f64
        })
        .count()
}

/// FT テンソルの i16 量子化で飽和が起きていれば stderr に警告する。発生件数 0 なら
/// 無出力。飽和は export 時に i16 範囲へ clamp されるため数値破綻にはならないが、
/// weight が i16 表現域 (±i16::MAX/QA) を超えて育った場合に silent な情報損失と
/// なるので気付けるようにする。
fn warn_if_i16_saturates(name: &str, values: &[f32], scale: f64) {
    let n = count_i16_saturations(values, scale);
    if n > 0 {
        let bound = i16::MAX as f64 / scale;
        eprintln!(
            "[nnue-format] warning: {name} has {n}/{} elements saturating i16 quantisation \
             (|w| > {bound:.4}); values are clamped on export (silent precision loss). \
             Consider tightening the training-time weight clamp for this tensor.",
            values.len()
        );
    }
}

/// threat weight の i8 量子化 (`round(scale·v)` を `[-128, 127]` へ clamp) で飽和
/// する要素数と最大 |round(scale·v)| を返す。threat は base FT と同じ `scale` (QA)
/// で raw i8 にするため、i16 より狭い i8 表現域に収まるかを export 前に把握する。
fn i8_saturation_stats(values: &[f32], scale: f64) -> (usize, f64) {
    let mut n = 0;
    let mut max_abs = 0.0_f64;
    for &v in values {
        let q = (scale * v as f64).round();
        max_abs = max_abs.max(q.abs());
        // threat block は symmetric ±127 に clamp する (clamp_i8_symmetric) ので、
        // |q| > 127 を飽和としてカウントする (非対称下限 -128 も飽和扱い)。
        if q.abs() > i8::MAX as f64 {
            n += 1;
        }
    }
    (n, max_abs)
}

/// threat weight の i8 量子化で飽和が起きていれば stderr に warn (発生 0 なら無出力)。
/// clamp 発生率と max-abs を出して「i8 で表現力十分か」を実 run で裏取りできるように
/// する (threat は base FT と同一スケールのため別係数で逃げられない)。
fn warn_if_i8_saturates(name: &str, values: &[f32], scale: f64) {
    let (n, max_abs) = i8_saturation_stats(values, scale);
    if n > 0 {
        let bound = i8::MAX as f64 / scale;
        let ratio = n as f64 / values.len().max(1) as f64;
        eprintln!(
            "[nnue-format] warning: {name} has {n}/{} ({:.4}%) elements saturating i8 quantisation \
             (|w| > {bound:.4}, max |round(scale·w)| = {max_abs:.1}); values are clamped to ±127 on \
             export (silent precision loss). Threat shares the base-FT scale (no separate factor), \
             so confirm i8 has enough range for this profile.",
            values.len(),
            ratio * 100.0,
        );
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn coalesce_ft_factorized_folds_virtual_rows() {
        use shogi_features::FeatureSet;
        let base = FeatureSet::HalfKp.spec();
        let spec = FeatureSet::HalfKp.spec().with_ft_factorize();
        let ft_out = 2usize;
        let pi = base.piece_inputs();
        // 実行 (kb, p) に feat 番号、仮想行 p に 10_000 + p を入れて畳み込みを検証。
        let mut w = vec![0.0f32; spec.train_ft_in() * ft_out];
        for feat in 0..base.ft_in() {
            for o in 0..ft_out {
                w[feat * ft_out + o] = feat as f32 + o as f32 * 0.5;
            }
        }
        for p in 0..pi {
            for o in 0..ft_out {
                w[(base.ft_in() + p) * ft_out + o] = 10_000.0 + p as f32 + o as f32 * 0.25;
            }
        }
        let out = coalesce_ft_factorized(&spec, ft_out, &w);
        assert_eq!(out.len(), base.ft_in() * ft_out);
        for feat in [0usize, 1, pi - 1, pi, 2 * pi + 7, base.ft_in() - 1] {
            let p = feat % pi;
            for o in 0..ft_out {
                let want = (feat as f32 + o as f32 * 0.5) + (10_000.0 + p as f32 + o as f32 * 0.25);
                assert_eq!(out[feat * ft_out + o], want, "feat={feat} o={o}");
            }
        }
        // 無効 spec は素通し。
        let pass = coalesce_ft_factorized(&base, ft_out, &w[..base.ft_in() * ft_out]);
        assert_eq!(pass, &w[..base.ft_in() * ft_out]);
    }

    #[test]
    fn coalesce_keeps_threat_rows_and_folds_only_base() {
        use shogi_features::{FeatureSet, ThreatProfile};
        // factorizer × threat 同居: export 形状 = ft_in() (base+threat)、base 行は
        // piece 仮想行を畳み込み、threat 行は仮想行を持たないので素通しする。
        let spec = FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(ThreatProfile::CrossSide)
            .with_ft_factorize();
        let base_ft_in = spec.base_ft_in();
        let ft_in = spec.ft_in(); // base + threat
        let pi = spec.piece_inputs();
        let ft_out = 2usize;
        assert_eq!(spec.train_ft_in(), ft_in + pi);

        let mut w = vec![0.0f32; spec.train_ft_in() * ft_out];
        // base 実行 / threat 実行 / 仮想行を distinctive 値で埋める。
        for feat in 0..ft_in {
            for o in 0..ft_out {
                w[feat * ft_out + o] = feat as f32 + o as f32 * 0.5;
            }
        }
        for p in 0..pi {
            for o in 0..ft_out {
                w[(ft_in + p) * ft_out + o] = 10_000.0 + p as f32 + o as f32 * 0.25;
            }
        }
        let out = coalesce_ft_factorized(&spec, ft_out, &w);
        // export 形状は base + threat (仮想行を落とす)。
        assert_eq!(out.len(), ft_in * ft_out);
        // base 行は実行 + 同 p 仮想行。
        for feat in [0usize, 1, pi, base_ft_in - 1] {
            let p = feat % pi;
            for o in 0..ft_out {
                let want = (feat as f32 + o as f32 * 0.5) + (10_000.0 + p as f32 + o as f32 * 0.25);
                assert_eq!(out[feat * ft_out + o], want, "base feat={feat} o={o}");
            }
        }
        // threat 行は素通し。
        for feat in [base_ft_in, base_ft_in + 1, ft_in - 1] {
            for o in 0..ft_out {
                let want = feat as f32 + o as f32 * 0.5;
                assert_eq!(out[feat * ft_out + o], want, "threat feat={feat} o={o}");
            }
        }
    }

    #[test]
    fn coalesce_coexist_boundary_all_profiles_and_featuresets() {
        use shogi_features::{FeatureSet, ThreatProfile};
        // boundary 網羅: 全 base featureset × 全 profile で同居 coalesce が
        // (a) export 形状 = ft_in()*ft_out、(b) base 行 = 実行 + 同 p 仮想行、
        // (c) threat 行 = 実行のまま、を満たす。`base_ft_in % piece_inputs == 0` も確認。
        let ft_out = 2usize;
        for fs in FeatureSet::ALL {
            for profile in [
                ThreatProfile::Full,
                ThreatProfile::SameClass,
                ThreatProfile::SameClassMajorPawn,
                ThreatProfile::StepAttacker,
                ThreatProfile::FullSymDedup,
                ThreatProfile::CrossSide,
            ] {
                let spec = fs.spec().with_threat_profile(profile).with_ft_factorize();
                let base_ft_in = spec.base_ft_in();
                let ft_in = spec.ft_in();
                let pi = spec.piece_inputs();
                assert_eq!(
                    base_ft_in % pi,
                    0,
                    "{} {profile}: base_ft_in % pi",
                    fs.canonical_name()
                );

                // base+threat 実行は feat 番号、仮想行は plane ごと一定値。
                let mut w = vec![0.0f32; spec.train_ft_in() * ft_out];
                for feat in 0..ft_in {
                    for o in 0..ft_out {
                        w[feat * ft_out + o] = (feat % 97) as f32 + o as f32 * 0.5;
                    }
                }
                for p in 0..pi {
                    for o in 0..ft_out {
                        w[(ft_in + p) * ft_out + o] = 1000.0 + (p % 13) as f32;
                    }
                }
                let out = coalesce_ft_factorized(&spec, ft_out, &w);
                assert_eq!(
                    out.len(),
                    ft_in * ft_out,
                    "{} {profile}: export shape",
                    fs.canonical_name()
                );

                // base 行先頭/末尾と threat 行先頭/末尾を spot-check (全要素は重いので端点)。
                for &feat in &[0usize, base_ft_in - 1] {
                    let p = feat % pi;
                    for o in 0..ft_out {
                        let want =
                            ((feat % 97) as f32 + o as f32 * 0.5) + (1000.0 + (p % 13) as f32);
                        assert_eq!(
                            out[feat * ft_out + o],
                            want,
                            "{} {profile} base feat={feat}",
                            fs.canonical_name()
                        );
                    }
                }
                for &feat in &[base_ft_in, ft_in - 1] {
                    for o in 0..ft_out {
                        let want = (feat % 97) as f32 + o as f32 * 0.5;
                        assert_eq!(
                            out[feat * ft_out + o],
                            want,
                            "{} {profile} threat feat={feat}",
                            fs.canonical_name()
                        );
                    }
                }
            }
        }
    }

    use super::*;
    use shogi_features::FeatureSet;

    #[test]
    fn count_i16_saturations_counts_only_out_of_range() {
        // scale = QA = 127。|round(127·w)| > 32767 ⇔ |w| ≳ 258.0 のみ飽和。
        let qa = QA as f64;
        assert_eq!(
            count_i16_saturations(&[0.0, 1.98, -1.98, 100.0, -257.9], qa),
            0
        );
        assert_eq!(count_i16_saturations(&[300.0, -300.0, 1.0], qa), 2);
        assert_eq!(count_i16_saturations(&[], qa), 0);
    }

    #[test]
    fn count_i16_saturations_boundary_exact() {
        // round(scale·w) が i16 端点ちょうどに乗るケース。scale=1.0 で w を量子化値
        // そのものに使う。32767 / -32768 は範囲内 (0 件)、32768 / -32769 は飽和。
        assert_eq!(count_i16_saturations(&[32767.0, -32768.0], 1.0), 0);
        assert_eq!(count_i16_saturations(&[32768.0, -32769.0], 1.0), 2);
        // round 後判定: 32767.4→32767 (範囲内)、32767.5→32768 (飽和)。
        assert_eq!(count_i16_saturations(&[32767.4], 1.0), 0);
        assert_eq!(count_i16_saturations(&[32767.5], 1.0), 1);
    }

    /// テストで使う feature set spec (現 production の halfka-hm-merged)。
    fn test_spec() -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged.spec()
    }

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
        assert_eq!(pad32(DEFAULT_FT_OUT), DEFAULT_FT_OUT); // ft_out は 128 の倍数
        assert_eq!(pad32((DEFAULT_L1_OUT - 1) * 2), 32); // 既定 l2_in = 30
        assert_eq!(pad32(DEFAULT_L2_OUT), 32);
    }

    #[test]
    fn weights_zeroed_save_load_roundtrip() {
        // 0 weight で save → load → 同じ 0 weight が復元できる
        let original = LayerStackWeights::zeroed(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = LayerStackWeights::load_quantised(
            &mut cursor,
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();

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

    fn threat_spec(profile: shogi_features::ThreatProfile) -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(profile)
    }

    fn effect_bucket_spec(config: EffectBucketConfig) -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged
            .spec()
            .with_effect_bucket_config(config)
    }

    #[test]
    fn parse_single_arch_token_is_strict() {
        // 正常: comma 区切りの 1 token を厳密一致で拾う。
        let arch = "Features=X[1->2],PSQT=9,Threat=cross-side,Network=Y";
        assert_eq!(
            parse_single_arch_token(arch, "Threat="),
            Ok(Some("cross-side"))
        );
        assert_eq!(parse_single_arch_token(arch, "PSQT="), Ok(Some("9")));
        // token 無し。
        assert_eq!(
            parse_single_arch_token("Features=X,Network=Y", "Threat="),
            Ok(None)
        );
        // substring だが token 先頭でない (`...XThreat=...`) は拾わない。
        assert_eq!(
            parse_single_arch_token("Features=XThreat=full,Network=Y", "Threat="),
            Ok(None)
        );
        // 重複 token は Err。
        assert_eq!(
            parse_single_arch_token("Threat=full,Threat=cross-side", "Threat="),
            Err(())
        );
        // suffix 違い (`Threat=fullX`) は完全一致照合で別値として扱われる。
        assert_eq!(
            parse_single_arch_token("Threat=fullX,Network=Y", "Threat="),
            Ok(Some("fullX"))
        );
    }

    #[test]
    fn threat_save_load_roundtrip_preserves_base_and_threat_rows() {
        use shogi_features::ThreatProfile;
        // 最小 profile (cross-side) + 小 ft_out で base / threat 両 row を持つ FT を
        // roundtrip する。base row 先頭と threat row 先頭に distinctive 値を入れ、
        // FT block / Threat block が正しい順で保存・復元されることを確認する。
        let profile = ThreatProfile::CrossSide;
        let spec = threat_spec(profile);
        let ft_out = 128;
        let mut original = LayerStackWeights::zeroed(
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let base_ft_w_n = spec.base_ft_in() * ft_out;
        assert_eq!(original.ft_w.len(), spec.ft_in() * ft_out);
        // base row は i16 量子化 (QA=127) で復元できる値、threat row は **i8 範囲**
        // (|round(QA·v)| <= 127、すなわち |v| <= 1.0) の値を置く (threat は raw i8)。
        original.ft_w[0] = 1.0; // base: 127/127 (i16 path)
        original.ft_w[base_ft_w_n] = -1.0; // threat: -127/127 (i8 端値)
        original.ft_w[base_ft_w_n + 1] = 5.0 / 127.0; // threat: 5/127

        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();

        // arch_str は rshogi 契約: `Threat=<dims>` + cross-side (id 10≠0) は
        // `ThreatProfile=10` も付く。token は dims 値で、profile 名は使わない。
        let arch = String::from_utf8_lossy(&buf);
        let dims = spec.threat_dims();
        assert!(
            arch.contains(&format!("Threat={dims},")),
            "arch_str missing Threat=<dims> token: {arch}"
        );
        assert!(
            arch.contains("ThreatProfile=10,"),
            "arch_str missing ThreatProfile=<id> token: {arch}"
        );
        assert!(
            !arch.contains("Threat=cross-side"),
            "Threat token must be dims-based, not a profile name"
        );

        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = LayerStackWeights::load_quantised(
            &mut cursor,
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();

        assert_eq!(loaded.ft_w.len(), original.ft_w.len());
        assert!((loaded.ft_w[0] - 1.0).abs() < 1e-6);
        assert!((loaded.ft_w[base_ft_w_n] - (-1.0)).abs() < 1e-6);
        assert!((loaded.ft_w[base_ft_w_n + 1] - 5.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn threat_load_rejects_profile_mismatch() {
        use shogi_features::ThreatProfile;
        // cross-side で save した `.bin` を same-class で load → reject。
        let ft_out = 128;
        let saved = threat_spec(ThreatProfile::CrossSide);
        let original = LayerStackWeights::zeroed(
            saved,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let err = LayerStackWeights::load_quantised(
            &mut cursor,
            threat_spec(ThreatProfile::SameClass),
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap_err();
        // profile が変われば feature_hash も変わるので network_hash か threat token
        // のどちらかで弾かれる (どちらも InvalidData)。
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn threat_load_rejects_threat_off_request_on_threat_bin() {
        use shogi_features::ThreatProfile;
        // threat-on で save した `.bin` を threat-off (base) spec で load → reject。
        let ft_out = 128;
        let saved = threat_spec(ThreatProfile::CrossSide);
        let original = LayerStackWeights::zeroed(
            saved,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let err = LayerStackWeights::load_quantised(
            &mut cursor,
            FeatureSet::HalfKaHmMerged.spec(),
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn threat_full_profile_omits_threatprofile_token_and_u32_id() {
        use shogi_features::ThreatProfile;
        // full = profile id 0: arch_str に `Threat=<dims>` は出すが `ThreatProfile=`
        // token は省き、stream の u32 id も書かない (rshogi の「id 無し = profile 0」
        // 経路)。i8 threat block は u32 を挟まず arch_str (+ num_buckets/ft_hash/FT/
        // PSQT) の直後から始まる。round-trip で threat row が復元できることも確認。
        let spec = threat_spec(ThreatProfile::Full);
        assert_eq!(spec.threat_profile().unwrap().profile_id(), 0);
        let ft_out = 128;
        let mut original = LayerStackWeights::zeroed(
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let base_ft_w_n = spec.base_ft_in() * ft_out;
        original.ft_w[base_ft_w_n] = 1.0; // threat row 端値 (127/127)

        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let arch = String::from_utf8_lossy(&buf);
        assert!(
            arch.contains(&format!("Threat={},", spec.threat_dims())),
            "{arch}"
        );
        assert!(
            !arch.contains("ThreatProfile="),
            "id 0 は ThreatProfile token を省く: {arch}"
        );

        let loaded = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
        assert_eq!(loaded.ft_w.len(), spec.ft_in() * ft_out);
        assert!((loaded.ft_w[base_ft_w_n] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn threat_block_is_raw_i8_feature_major() {
        use shogi_features::ThreatProfile;
        // threat block が raw i8 (LEB128 でない) で書かれることを byte で固定する。
        // cross-side (id 10) の stream: ... → PSQT 無し → **u32 LE id (10)** →
        // threat_dims*ft_out 個の i8。weight をスケール (QA=127) で割り切れる i8 値に
        // 設定し、対応 byte を直接検証する。
        let spec = threat_spec(ThreatProfile::CrossSide);
        let ft_out = 128;
        let mut original = LayerStackWeights::zeroed(
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let base_ft_w_n = spec.base_ft_in() * ft_out;
        original.ft_w[base_ft_w_n] = 3.0 / 127.0; // → i8 3
        original.ft_w[base_ft_w_n + 1] = -5.0 / 127.0; // → i8 -5

        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();

        // threat block は末尾 (LayerStacks の前) ではなく、save 順では PSQT(無)→
        // Threat(u32 id + i8...)→LayerStacks。threat block の長さは 4 (u32) +
        // threat_dims*ft_out (i8) なので、その先頭を末尾から逆算して u32 id と
        // 最初の 2 byte を確認する。
        let threat_i8_n = spec.threat_dims() * ft_out;
        // LayerStacks 部の長さは threat に依らないので、threat block 開始位置を
        // 「u32 + i8 列」の構造で前方から特定するのは複雑。代わりに u32 id (10) が
        // little-endian で stream 内に存在し、その直後 2 byte が 3, -5(=251) である
        // 連続パターンを検索して raw i8 配置を確認する。
        let needle = [10u8, 0, 0, 0, 3u8, (-5i8) as u8];
        let found = buf.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "u32 id(10 LE) + raw i8 (3, -5) のパターンが見つからない (i8 raw でない?)"
        );
        // i8 block 長の健全性: buf は header+...+threat(4+i8_n)+layerstacks。
        assert!(
            buf.len() > 4 + threat_i8_n,
            "stream too short for i8 threat block"
        );
    }

    #[test]
    fn effect_bucket_save_load_roundtrip_preserves_expanded_ft_rows() {
        let spec = effect_bucket_spec(EffectBucketConfig::KINGFIXED_2X2);
        let ft_out = 128;
        let mut original = LayerStackWeights::zeroed(
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        assert_eq!(original.ft_w.len(), spec.ft_in() * ft_out);
        let last = original.ft_w.len() - 1;
        original.ft_w[0] = 1.0;
        original.ft_w[last] = -5.0 / 127.0;

        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let arch = String::from_utf8_lossy(&buf);
        assert!(arch.contains("EffectBucket=2x2fixed,"), "{arch}");
        assert!(!arch.contains("Threat="), "{arch}");

        let loaded = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            spec,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
        assert_eq!(loaded.ft_w.len(), original.ft_w.len());
        assert!((loaded.ft_w[0] - 1.0).abs() < 1e-6);
        assert!((loaded.ft_w[last] - (-5.0 / 127.0)).abs() < 1e-6);
    }

    #[test]
    fn effect_bucket_load_rejects_config_mismatch() {
        let ft_out = 128;
        let saved = effect_bucket_spec(EffectBucketConfig::KINGFIXED_2X2);
        let original = LayerStackWeights::zeroed(
            saved,
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();

        let err = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            effect_bucket_spec(EffectBucketConfig::KINGBUCKETED_2X2),
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn non_default_ft_out_save_load_roundtrip() {
        // 既定外の FT 出力次元で zeroed → save → load しても layout が対称で、
        // buffer 長が ft_out に追従する。
        let ft_out = 256;
        let original = LayerStackWeights::zeroed(
            test_spec(),
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        assert_eq!(original.ft_b.len(), ft_out);
        assert_eq!(original.ft_w.len(), test_spec().ft_in() * ft_out);
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let loaded = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            ft_out,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
        assert_eq!(loaded.ft_b.len(), ft_out);
        assert_eq!(loaded.ft_w.len(), test_spec().ft_in() * ft_out);
        assert_eq!(
            loaded.l1_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L1_OUT * ft_out
        );
        // 同じ .bin を別 ft_out で load すると arch / hash 不一致で reject される。
        let err = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .expect_err("ft_out 不一致は reject される");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn non_default_l1_out_save_load_roundtrip() {
        // 既定外の L1 出力次元で zeroed → save → load しても layout が対称で、
        // buffer 長が l1_out / l2_in に追従する。
        let l1_out = 24;
        let l2_in = (l1_out - 1) * 2;
        let original = LayerStackWeights::zeroed(
            test_spec(),
            DEFAULT_FT_OUT,
            l1_out,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        assert_eq!(original.l1f_b.len(), l1_out);
        assert_eq!(
            original.l1_w.len(),
            DEFAULT_NUM_BUCKETS * l1_out * DEFAULT_FT_OUT
        );
        assert_eq!(
            original.l2_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT * l2_in
        );
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let loaded = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            l1_out,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
        assert_eq!(loaded.l1_b.len(), DEFAULT_NUM_BUCKETS * l1_out);
        assert_eq!(
            loaded.l1_w.len(),
            DEFAULT_NUM_BUCKETS * l1_out * DEFAULT_FT_OUT
        );
        assert_eq!(
            loaded.l2_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT * l2_in
        );
        // 同じ .bin を別 l1_out で load すると arch token 不一致で reject される。
        let err = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .expect_err("l1_out 不一致は reject される");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn non_default_l2_out_save_load_roundtrip() {
        // 既定外の L2 出力次元で zeroed → save → load しても layout が対称で、
        // buffer 長が l2_out に追従する。
        let l2_out = 64;
        let l2_in = (DEFAULT_L1_OUT - 1) * 2;
        let original = LayerStackWeights::zeroed(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            l2_out,
            DEFAULT_NUM_BUCKETS,
        );
        assert_eq!(original.l2_b.len(), DEFAULT_NUM_BUCKETS * l2_out);
        assert_eq!(original.l2_w.len(), DEFAULT_NUM_BUCKETS * l2_out * l2_in);
        assert_eq!(original.l3_w.len(), DEFAULT_NUM_BUCKETS * l2_out);
        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let loaded = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            l2_out,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
        assert_eq!(loaded.l2_b.len(), DEFAULT_NUM_BUCKETS * l2_out);
        assert_eq!(loaded.l2_w.len(), DEFAULT_NUM_BUCKETS * l2_out * l2_in);
        assert_eq!(loaded.l3_w.len(), DEFAULT_NUM_BUCKETS * l2_out);
        // 同じ .bin を別 l2_out で load すると network_hash 不一致で reject される
        // (l2_out は compute_fc_hash の隠れ層列に織り込まれている)。
        let err = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .expect_err("l2_out 不一致は reject される");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_validates_feature_set_and_rejects_mismatch() {
        // halfka-split で save した .bin は halfka-split で load でき、
        // 異なる feature set (halfka-hm-merged) を要求すると reject される。
        let split = FeatureSet::HalfKaSplit.spec();
        let merged = FeatureSet::HalfKaHmMerged.spec();
        let mut buf = Vec::new();
        LayerStackWeights::zeroed(
            split,
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .save_quantised(&mut buf)
        .unwrap();

        let ok = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            split,
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        assert!(ok.is_ok(), "同一 feature set なら load できる");
        assert_eq!(ok.unwrap().ft_w.len(), split.ft_in() * DEFAULT_FT_OUT);

        let err = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            merged,
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .expect_err("feature set 不一致は reject される");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_external_reference_if_env_set() {
        // 外部 reference checkpoint を `RSHOGI_NNUE_LAYERSTACK_REF_BIN`
        // で指定すると load + sanity check を走らせる任意の regression check。
        // env var 未設定なら skip (CI 想定)。
        let Ok(path) = std::env::var("RSHOGI_NNUE_LAYERSTACK_REF_BIN") else {
            eprintln!("skipping load_external_reference (RSHOGI_NNUE_LAYERSTACK_REF_BIN not set)");
            return;
        };
        let mut file = std::fs::File::open(&path).expect("open reference bin");
        let weights = LayerStackWeights::load_quantised(
            &mut file,
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .expect("parse reference bin");

        // Sanity: 各 weight buffer の長さが期待値
        let l2_in = (DEFAULT_L1_OUT - 1) * 2;
        assert_eq!(weights.ft_w.len(), test_spec().ft_in() * DEFAULT_FT_OUT);
        assert_eq!(weights.ft_b.len(), DEFAULT_FT_OUT);
        assert_eq!(
            weights.l1_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L1_OUT * DEFAULT_FT_OUT
        );
        assert_eq!(weights.l1_b.len(), DEFAULT_NUM_BUCKETS * DEFAULT_L1_OUT);
        assert_eq!(
            weights.l2_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT * l2_in
        );
        assert_eq!(weights.l2_b.len(), DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT);
        assert_eq!(weights.l3_w.len(), DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT);
        assert_eq!(weights.l3_b.len(), DEFAULT_NUM_BUCKETS);

        // 非自明 weight (training 済みなので非 0 が多い)
        let nz_ft = weights.ft_w.iter().filter(|&&x| x != 0.0).count();
        let pct_ft = nz_ft as f64 / weights.ft_w.len() as f64 * 100.0;
        let max_ft = weights.ft_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let max_l1 = weights.l1_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let max_l2 = weights.l2_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let max_l3 = weights.l3_w.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        eprintln!(
            "[layerstack ref load] FT nonzero: {nz_ft}/{} ({pct_ft:.2}%)",
            weights.ft_w.len()
        );
        eprintln!("[layerstack ref load] FT weight max abs: {max_ft:.6}");
        eprintln!("[layerstack ref load] L1 weight max abs: {max_l1:.6}");
        eprintln!("[layerstack ref load] L2 weight max abs: {max_l2:.6}");
        eprintln!("[layerstack ref load] L3 weight max abs: {max_l3:.6}");

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
    fn resave_external_reference_if_env_set() {
        // 外部 reference checkpoint (`RSHOGI_NNUE_LAYERSTACK_REF_BIN`) を load → save
        // し直して size と byte 差を比較する regression check。env 未設定なら skip。
        //
        // save 側は常に current `NNUE_VERSION` で書き出す (legacy 形式に書き戻す
        // path は提供しない)。`LEGACY_NNUE_VERSION_BUCKETS9` の reference を load
        // して save し直すと、current format で追加された `num_buckets: u32` field
        // 4 bytes が増える (legacy → current への migration)。current format の
        // reference なら同 version 同 size で round-trip する。
        let Ok(in_path) = std::env::var("RSHOGI_NNUE_LAYERSTACK_REF_BIN") else {
            eprintln!(
                "skipping resave_external_reference (RSHOGI_NNUE_LAYERSTACK_REF_BIN not set)"
            );
            return;
        };
        // 入力 file の先頭 4 bytes (`NNUE_VERSION`) を読み、current / legacy の
        // どちらの format かを判定する。期待 size diff は legacy → current なら +4
        // (num_buckets: u32 が増える)、current → current なら 0。
        let in_bytes = std::fs::read(&in_path).unwrap();
        assert!(
            in_bytes.len() >= 4,
            "reference file too short to read version"
        );
        let in_version = u32::from_le_bytes(in_bytes[..4].try_into().unwrap());
        let expected_size_diff = match in_version {
            v if v == NNUE_VERSION => 0_i64,
            v if v == LEGACY_NNUE_VERSION_BUCKETS9 => 4_i64,
            v => panic!("unknown NNUE_VERSION in reference: {v:#x}"),
        };
        let out_path = std::env::temp_dir().join("layerstack_ref_resaved.bin");
        let mut reader = std::io::BufReader::new(std::fs::File::open(&in_path).unwrap());
        let weights = LayerStackWeights::load_quantised(
            &mut reader,
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
        weights.save_quantised(&mut writer).unwrap();
        drop(writer);

        let in_size = in_bytes.len() as i64;
        let out_size = std::fs::metadata(&out_path).unwrap().len() as i64;
        let diff = out_size - in_size;
        eprintln!(
            "[resave] in_version={in_version:#x} in_size={in_size}, out_size={out_size}, \
             diff={diff} (expected {expected_size_diff})"
        );
        // Byte size は format version に対応した想定 diff (current → current は 0、
        // legacy → current は +4) を期待 (layout regression detect)。
        assert_eq!(
            diff, expected_size_diff,
            "size diff {diff} != expected {expected_size_diff} for in_version {in_version:#x} — \
             layout regression?"
        );

        // 同 format round-trip (current → current) のみ byte-level 比較する。
        // legacy → current の migration では header + 全 layerstack section の offset
        // が 4 bytes ずれるため byte 単位比較は意味を持たない。
        if in_version == NNUE_VERSION {
            let out_bytes = std::fs::read(&out_path).unwrap();
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
    }

    #[test]
    fn fc_hash_depends_on_ft_out_and_l2_out() {
        // compute_fc_hash は placeholder 0 でない値を返し、ft_out / l2_out ごとに
        // 異なる (両出力次元が hash に織り込まれている = 異アーキを弁別できる)。
        let h_default = compute_fc_hash(DEFAULT_FT_OUT, DEFAULT_L2_OUT);
        assert_ne!(
            h_default, 0,
            "compute_fc_hash should be computed, not placeholder"
        );
        assert_ne!(
            h_default,
            compute_fc_hash(DEFAULT_FT_OUT * 2, DEFAULT_L2_OUT),
            "fc_hash は ft_out ごとに異なるべき"
        );
        assert_ne!(
            h_default,
            compute_fc_hash(DEFAULT_FT_OUT, DEFAULT_L2_OUT * 2),
            "fc_hash は l2_out ごとに異なるべき"
        );
    }

    #[test]
    fn arch_str_format() {
        let s = build_arch_str(
            "HalfKaHmMerged",
            test_spec().ft_in(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            (DEFAULT_L1_OUT - 1) * 2,
            DEFAULT_L2_OUT,
            FV_SCALE,
            None,
            None,
            None,
        );
        assert!(s.contains("HalfKaHmMerged"));
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

    #[test]
    fn build_arch_str_with_psqt_inserts_token() {
        // psqt_buckets = Some(9) で `PSQT=9,` token が Features と Network の間に入る。
        let s = build_arch_str(
            "HalfKaHmMerged",
            73_305,
            1536,
            16,
            30,
            32,
            28,
            Some(9),
            None,
            None,
        );
        assert!(s.contains("PSQT=9,"));
        // 順序: Features=...,PSQT=9,Network=...
        let psqt_pos = s.find("PSQT=9,").unwrap();
        let net_pos = s.find("Network=").unwrap();
        assert!(psqt_pos < net_pos);
        // 既存 token は維持。
        assert!(s.contains("Features=HalfKaHmMerged(Friend)[73305->1536x2]"));
        assert!(s.contains("fv_scale=28"));
    }

    #[test]
    fn build_arch_str_threat_token_uses_dims_and_id() {
        // rshogi 契約: profile id≠0 は `Threat=<dims>,ThreatProfile=<id>`、
        // id 0 (full) は `Threat=<dims>` のみ (ThreatProfile token は省略)。
        let with_id = build_arch_str(
            "HalfKaHmMerged",
            169_625,
            1536,
            16,
            30,
            32,
            28,
            None,
            Some(ThreatArch {
                dims: 96_320,
                profile_id: 10,
            }),
            None,
        );
        assert!(with_id.contains("Threat=96320,"), "{with_id}");
        assert!(with_id.contains("ThreatProfile=10,"), "{with_id}");
        // dims token は Network より前 (Features と Network の間)。
        assert!(with_id.find("Threat=96320,").unwrap() < with_id.find("Network=").unwrap());

        let id_zero = build_arch_str(
            "HalfKaHmMerged",
            290_025,
            1536,
            16,
            30,
            32,
            28,
            None,
            Some(ThreatArch {
                dims: 216_720,
                profile_id: 0,
            }),
            None,
        );
        assert!(id_zero.contains("Threat=216720,"), "{id_zero}");
        assert!(
            !id_zero.contains("ThreatProfile="),
            "id 0 は ThreatProfile token を省く: {id_zero}"
        );
    }

    #[test]
    fn build_arch_str_effect_bucket_token_uses_bucket_count_and_king_mode() {
        let s = build_arch_str(
            "HalfKaHmMerged",
            293_220,
            1536,
            16,
            30,
            32,
            28,
            None,
            None,
            Some(EffectBucketArch {
                nb: 4,
                king_bucketed: false,
            }),
        );
        assert!(s.contains("EffectBucket=2x2fixed,"), "{s}");
        assert!(s.find("EffectBucket=2x2fixed,").unwrap() < s.find("Network=").unwrap());
    }

    #[test]
    fn effect_bucket_arch_rejects_unsupported_bucket_count() {
        let err = std::panic::catch_unwind(|| {
            let _ = EffectBucketArch {
                nb: 5,
                king_bucketed: false,
            }
            .token_value();
        })
        .expect_err("unsupported EffectBucket bucket count must panic");
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            msg.contains("unsupported EffectBucket bucket count: 5"),
            "panic message should mention the unsupported bucket count: {msg}"
        );
    }

    #[test]
    fn psqt_zeroed_save_load_roundtrip() {
        let original = LayerStackWeights::zeroed_with_psqt(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        assert_eq!(
            original.psqt_w.as_ref().unwrap().len(),
            test_spec().ft_in() * DEFAULT_NUM_BUCKETS
        );

        let mut buf = Vec::new();
        original.save_quantised(&mut buf).unwrap();
        let loaded = LayerStackWeights::load_quantised_with_psqt(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            true,
        )
        .unwrap();
        let psqt = loaded.psqt_w.as_ref().expect("psqt_w should be Some");
        assert_eq!(psqt.len(), test_spec().ft_in() * DEFAULT_NUM_BUCKETS);
        assert!(psqt.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn psqt_nonzero_roundtrip_quantises_exactly() {
        // 1/QA/QB の境界内なら量子化 lossless。`1.0 / 8128 ≈ 1.23e-4` を 5 セルに置く。
        let mut net = LayerStackWeights::zeroed_with_psqt(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let psqt = net.psqt_w.as_mut().unwrap();
        let lsb = 1.0_f32 / (QA * QB) as f32;
        psqt[0] = lsb;
        psqt[DEFAULT_NUM_BUCKETS - 1] = -lsb;
        psqt[DEFAULT_NUM_BUCKETS] = 2.0 * lsb;
        psqt[100 * DEFAULT_NUM_BUCKETS + 3] = -3.0 * lsb;
        psqt[(test_spec().ft_in() - 1) * DEFAULT_NUM_BUCKETS + 5] = 7.0 * lsb;

        let mut buf = Vec::new();
        net.save_quantised(&mut buf).unwrap();
        let loaded = LayerStackWeights::load_quantised_with_psqt(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            true,
        )
        .unwrap();
        let loaded_psqt = loaded.psqt_w.as_ref().unwrap();
        for (i, (&orig, &got)) in net
            .psqt_w
            .as_ref()
            .unwrap()
            .iter()
            .zip(loaded_psqt.iter())
            .enumerate()
        {
            assert_eq!(orig, got, "psqt index {i}");
        }
    }

    #[test]
    fn load_rejects_psqt_when_not_requested() {
        // PSQT 含む .bin を with_psqt=false で読むと InvalidData で reject される。
        let net = LayerStackWeights::zeroed_with_psqt(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        net.save_quantised(&mut buf).unwrap();
        let err = LayerStackWeights::load_quantised(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .expect_err("PSQT 含む .bin を with_psqt=false で読むと reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_rejects_missing_psqt_when_requested() {
        // PSQT 無し .bin を with_psqt=true で読むと InvalidData で reject される。
        let net = LayerStackWeights::zeroed(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        net.save_quantised(&mut buf).unwrap();
        let err = LayerStackWeights::load_quantised_with_psqt(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            true,
        )
        .expect_err("PSQT 無し .bin を with_psqt=true で読むと reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// PSQT bias block の offset を「全 file 末尾から layerstacks 9 bucket 分 +
    /// PSQT weights 分」を差し引いて求める helper。LEB128 が可変長で前方からの計算は
    /// 困難だが、末尾の構造は size 固定なので逆算で確定する。
    fn psqt_bias_start(buf_len: usize, ft_in: usize) -> usize {
        let l1_padded_in = pad32(DEFAULT_FT_OUT);
        let l2_in = (DEFAULT_L1_OUT - 1) * 2;
        let l2_padded_in = pad32(l2_in);
        let l3_padded_in = pad32(DEFAULT_L2_OUT);
        let layerstack_per_bucket = 4 // fc_hash
            + DEFAULT_L1_OUT * 4 // L1 bias i32
            + DEFAULT_L1_OUT * l1_padded_in // L1 weights i8
            + DEFAULT_L2_OUT * 4 // L2 bias
            + DEFAULT_L2_OUT * l2_padded_in
            + 4 // L3 bias
            + l3_padded_in;
        let layerstacks_total = DEFAULT_NUM_BUCKETS * layerstack_per_bucket;
        let psqt_weights_bytes = ft_in * DEFAULT_NUM_BUCKETS * 4;
        let psqt_bias_bytes = DEFAULT_NUM_BUCKETS * 4;
        buf_len - layerstacks_total - psqt_weights_bytes - psqt_bias_bytes
    }

    #[test]
    fn load_rejects_nonzero_psqt_bias() {
        // PSQT は構造的に bias-free だが、外部由来 .bin で非ゼロ bias が混入した場合は
        // silent drop せず InvalidData で reject する。bucket 3 を patch することで、
        // bucket 0..=2 が zero-pass branch を通過したあとで bucket 3 で error 経路が
        // 起動するパスを exercise する (zero-branch coverage)。
        let net = LayerStackWeights::zeroed_with_psqt(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        net.save_quantised(&mut buf).unwrap();

        // bucket 3 の bias cell (i32 LE) を非ゼロに書き換える。
        // raw i32 = 16256 → dequant = 16256 / 8128 = 2.0 (整然と確認できる値)。
        let bias_start = psqt_bias_start(buf.len(), test_spec().ft_in());
        const TARGET_BUCKET: usize = 3;
        const NONZERO_RAW: i32 = 16_256; // = QA * QB * 2.0
        let cell_off = bias_start + TARGET_BUCKET * 4;
        buf[cell_off..cell_off + 4].copy_from_slice(&NONZERO_RAW.to_le_bytes());

        let err = LayerStackWeights::load_quantised_with_psqt(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            true,
        )
        .expect_err("non-zero PSQT bias must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = format!("{err}");
        // error メッセージのフィールドを 1 個ずつ assert (バグで一部が落ちる regression を捕捉)。
        assert!(msg.contains("PSQT bias"), "missing 'PSQT bias' in {msg}");
        assert!(msg.contains("non-zero"), "missing 'non-zero' in {msg}");
        // bucket index は patch した値 (TARGET_BUCKET=3、最初の zero-pass を抜けた後)
        // である必要がある。
        assert!(
            msg.contains(&format!("bucket {TARGET_BUCKET}")),
            "expected bucket index {TARGET_BUCKET} in {msg}"
        );
        // raw i32 値 (16256) と dequant 値 (2) が両方含まれる。
        assert!(
            msg.contains(&format!("raw i32 {NONZERO_RAW}")),
            "missing raw i32 value in {msg}"
        );
        assert!(msg.contains("dequant 2"), "missing dequant value in {msg}");
    }

    #[test]
    fn load_accepts_all_zero_psqt_bias() {
        // 全 DEFAULT_NUM_BUCKETS bucket の bias が 0 の通常 case が load を pass する
        // ことを明示的に確認する (zero-branch の正常 path coverage)。
        // psqt_zeroed_save_load_roundtrip も同じ path を通るが、本 test は「bias 領域
        // の全 9 cell が 0」という前提を破った版 (非ゼロ patch) と直接対比できる。
        let net = LayerStackWeights::zeroed_with_psqt(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf = Vec::new();
        net.save_quantised(&mut buf).unwrap();
        // bias 領域 9 個の i32 LE が全て 0 であることを直接確認する (zero-branch を
        // どの bucket でも通る前提の test)。
        let bias_start = psqt_bias_start(buf.len(), test_spec().ft_in());
        for bucket in 0..DEFAULT_NUM_BUCKETS {
            let off = bias_start + bucket * 4;
            let raw = i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            assert_eq!(raw, 0, "expected 0 PSQT bias at bucket {bucket}, got {raw}");
        }
        // 全 0 bias で load が成功する。
        LayerStackWeights::load_quantised_with_psqt(
            &mut std::io::Cursor::new(&buf),
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
            true,
        )
        .expect("all-zero PSQT bias must load successfully");
    }

    #[test]
    fn psqt_save_file_size_grows_by_expected_bytes() {
        // PSQT 有り .bin は PSQT 無し .bin より `(DEFAULT_NUM_BUCKETS + ft_in * DEFAULT_NUM_BUCKETS) * 4`
        // bytes 大きい (bias + weights の i32 LE 列、scale 共通 qa*qb)。
        let ft_in = test_spec().ft_in();
        let without = LayerStackWeights::zeroed(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let with = LayerStackWeights::zeroed_with_psqt(
            test_spec(),
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
            DEFAULT_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut buf_w = Vec::new();
        let mut buf_wo = Vec::new();
        with.save_quantised(&mut buf_w).unwrap();
        without.save_quantised(&mut buf_wo).unwrap();
        let arch_token_len = "PSQT=9,".len();
        let psqt_bytes = (DEFAULT_NUM_BUCKETS + ft_in * DEFAULT_NUM_BUCKETS) * 4;
        assert_eq!(
            buf_w.len() - buf_wo.len(),
            arch_token_len + psqt_bytes,
            "size diff = arch token ({arch_token_len}) + PSQT block ({psqt_bytes})"
        );
    }
}
