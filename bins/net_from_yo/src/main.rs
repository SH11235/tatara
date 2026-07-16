use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{QA, QB, pad32, read_leb128_tensor_i16};
use shogi_features::{FeatureSet, FeatureSetSpec};

/// YaneuraOu SFNNwoPSQT ビルドが評価ファイルに埋め込む 4 つのハッシュ。SFNNwoPSQT
/// では feature set / 次元に依らず固定定数。version 以外は YaneuraOu 側で warning
/// 扱いだが、変換ツールとしては保守的に hard reject する。
const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;
const YO_LEB128_PREFIX: u8 = b'_';

/// arch 由来次元の健全性ガード上限 (net_to_yo と対称)。実在アーキは十分収まり、
/// 壊れた arch 文字列 (巨大値) が overflow や過大 allocation を起こす前に弾く。
const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;
/// YaneuraOu SFNN KingRank9 の bucket 数。3x3 で常に 9。
const REQUIRED_BUCKETS: usize = 9;

/// tatara feature set と YaneuraOu SFNN feature の対応 (net_to_yo と対称)。
///
/// - `yo_name`: YaneuraOu `GetName()` (`Features=<yo_name>(Friend)[...]`)
/// - `gen_key`: `nnue_arch_gen.py` の feature キーワード
///   (非既定次元の `Network=SFNN_<gen_key>_<ft>_<h1>_<h2>_k3k3` に入る)
struct YoFeature {
    feature_set: FeatureSet,
    yo_name: &'static str,
    gen_key: &'static str,
}

const YO_FEATURES: [YoFeature; 5] = [
    YoFeature {
        feature_set: FeatureSet::HalfKp,
        yo_name: "HalfKP",
        gen_key: "halfkp",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaSplit,
        yo_name: "HalfKA1",
        gen_key: "halfka1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaMerged,
        yo_name: "HalfKA2",
        gen_key: "halfka2",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmSplit,
        yo_name: "HalfKA_hm1",
        gen_key: "halfkahm1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmMerged,
        yo_name: "HalfKA_hm2",
        gen_key: "halfkahm2",
    },
];

/// YaneuraOu arch 文字列から読み取った変換対象アーキ。
#[derive(Debug, Clone, Copy)]
struct DetectedArch {
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
}

#[derive(Parser)]
#[command(about = "Convert a YaneuraOu SFNN evaluation file to a tatara LayerStack net")]
struct Args {
    /// YaneuraOu nn.bin
    #[arg(long)]
    input: PathBuf,
    /// tatara LayerStack quantised .bin
    #[arg(long)]
    output: PathBuf,
    /// Assert that the net uses king-rank 9-bucket routing.
    /// Quantised `.bin` files do not record their bucket routing mode.
    #[arg(long)]
    assume_kingrank9: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output {
        return Err("input and output must be different paths".into());
    }
    require_kingrank9_assertion(args.assume_kingrank9)?;

    let input = File::open(&args.input)?;
    let mut reader = BufReader::new(input);
    let weights = read_yo(&mut reader)?;
    reject_trailing_data(&mut reader)?;

    let output = File::create(&args.output)?;
    let mut writer = BufWriter::new(output);
    weights.save_quantised(&mut writer, Some(nnue_format::layerstack_weights::FV_SCALE))?;
    writer.flush()?;
    Ok(())
}

fn require_kingrank9_assertion(assume_kingrank9: bool) -> io::Result<()> {
    if !assume_kingrank9 {
        return invalid_input(
            "YaneuraOu SFNN files do not identify the bucket routing rule; pass \
             --assume-kingrank9 only after confirming that the net uses king-rank routing",
        );
    }
    Ok(())
}

fn read_yo<R: BufRead>(reader: &mut R) -> io::Result<LayerStackWeights> {
    expect_u32(reader, YO_VERSION, "version")?;
    expect_u32(reader, YO_TOP_HASH, "top-level hash")?;

    let arch_len = read_u32(reader)? as usize;
    if arch_len == 0 || arch_len > 16_384 {
        return invalid_data(format!("invalid architecture string length: {arch_len}"));
    }
    let mut arch_bytes = vec![0_u8; arch_len];
    reader.read_exact(&mut arch_bytes)?;
    let arch = std::str::from_utf8(&arch_bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("architecture string is not UTF-8: {error}"),
        )
    })?;
    let detected = parse_yo_arch(arch)?;

    expect_u32(reader, YO_FT_HASH, "feature-transformer hash")?;
    consume_optional_leb128_prefix(reader)?;

    read_yo_tensors(
        reader,
        detected.feature_set.spec(),
        detected.ft_out,
        detected.l1_out,
        detected.l2_out,
        detected.num_buckets,
    )
}

/// YaneuraOu `GetArchitectureString()` (`ModelType=SFNNWithoutPsqt;Features=<name>(Friend)
/// [<in>-><ft>x2],Network=<struct>{LayerStack=<n>}`) から feature set と次元を検出する。
/// PSQT 付き (`SFNNWithoutPsqt` でない) や未知 feature は変換不可として reject。
fn parse_yo_arch(arch: &str) -> io::Result<DetectedArch> {
    if !arch.starts_with("ModelType=SFNNWithoutPsqt;") {
        return invalid_data(format!(
            "unsupported YaneuraOu architecture (expected ModelType=SFNNWithoutPsqt): `{arch}`"
        ));
    }

    // `Features=<name>(Friend)[<in>-><ft>x2]` を 1 トークンとして切り出し、その内部から
    // name / <in> / <ft> を取る。arch 全体を跨いだ `between` で最初の `(Friend)[` 等を
    // 拾うと、先行 decoy token (`X=(Friend)[<expected>->...]`) で cross-check を
    // すり抜けられるため、必ず選択した Features トークンに anchor する。
    let features = between(arch, "Features=", ",Network=").ok_or_else(|| {
        invalid_data_err(format!(
            "architecture has no `Features=...,Network=` section: `{arch}`"
        ))
    })?;
    let (feature_name, dims) = features.split_once("(Friend)[").ok_or_else(|| {
        invalid_data_err(format!("Features token has no `(Friend)[`: `{features}`"))
    })?;
    let feature = YO_FEATURES
        .iter()
        .find(|f| f.yo_name == feature_name)
        .ok_or_else(|| invalid_data_err(format!("unknown YaneuraOu feature `{feature_name}`")))?;

    let (input_str, ft_rest) = dims
        .split_once("->")
        .ok_or_else(|| invalid_data_err(format!("Features token has no `<in>->`: `{features}`")))?;
    let input_size = parse_usize(input_str)?;
    let expected_input = feature.feature_set.spec().ft_in();
    if input_size != expected_input {
        return invalid_data(format!(
            "Features input dimension {input_size} disagrees with feature {} (expected {expected_input})",
            feature.yo_name
        ));
    }
    let ft_out = ft_rest
        .split_once("x2")
        .map(|(ft, _)| ft)
        .ok_or_else(|| invalid_data_err(format!("Features token has no `<ft>x2`: `{features}`")))
        .and_then(parse_usize)?;

    // Network 節も Features より後ろに anchor して切り出す。
    let network_part = arch
        .split_once(",Network=")
        .map(|(_, rest)| rest)
        .ok_or_else(|| {
            invalid_data_err(format!("architecture has no `,Network=` token: `{arch}`"))
        })?;
    let network = network_part
        .split_once('{')
        .map(|(n, _)| n)
        .ok_or_else(|| invalid_data_err(format!("Network token has no `{{`: `{network_part}`")))?;
    let num_buckets = between(network_part, "LayerStack=", "}")
        .ok_or_else(|| {
            invalid_data_err(format!(
                "architecture has no `LayerStack=<n>}}` token: `{arch}`"
            ))
        })
        .and_then(parse_usize)?;
    let (l1_out, l2_out) = parse_network_dims(network, feature, ft_out)?;

    // 外部ファイル由来の arch 由来次元を、壊れた値 (0 次元 / 巨大値 → overflow・過大
    // allocation) の前に弾く健全性ガード (net_to_yo と対称)。受理する Network 名は
    // いずれも 3x3=9 bucket を意味し、`--assume-kingrank9` も 9 を主張するので
    // num_buckets != 9 は自己矛盾として reject する。
    if ft_out == 0 || ft_out > MAX_FT_OUT || ft_out % 32 != 0 {
        return invalid_data(format!(
            "unsupported FT output dimension {ft_out} (expected a positive multiple of 32 up to {MAX_FT_OUT})"
        ));
    }
    if l2_out == 0 || l2_out > MAX_HIDDEN_DIM {
        return invalid_data(format!(
            "unsupported L2 output dimension {l2_out} (expected 1..={MAX_HIDDEN_DIM})"
        ));
    }
    let l2_in = (l1_out - 1) * 2;
    if l2_in == 0 || l2_in > MAX_HIDDEN_DIM {
        return invalid_data(format!(
            "unsupported L1 output dimension {l1_out} (L2 input {l2_in} must be 1..={MAX_HIDDEN_DIM})"
        ));
    }
    if num_buckets != REQUIRED_BUCKETS {
        return invalid_data(format!(
            "unsupported LayerStack count {num_buckets} (YaneuraOu SFNN KingRank9 is always {REQUIRED_BUCKETS})"
        ));
    }

    Ok(DetectedArch {
        feature_set: feature.feature_set,
        ft_out,
        l1_out,
        l2_out,
        num_buckets,
    })
}

/// Network 構造名から L1/L2 出力次元を復元する。既定 `SFNN-1536`(-V2) は
/// `HalfKA_hm2`/1536 専用で 16/32、それ以外は生成器同形の
/// `SFNN_<GEN_KEY>_<ft>_<h1>_<h2>_K3K3` から `l1_out = h1 + 1`, `l2_out = h2` を取る。
/// `nnue_arch_gen.py` は生成名を大文字化する (`arch.upper()`) ため、実 YaneuraOu net
/// の構造名は大文字だが、大小どちらの綴りも受理する。
/// `feature`/`ft_out` は Features トークンとの整合を照合するために受け取る。
fn parse_network_dims(
    network: &str,
    feature: &YoFeature,
    ft_out: usize,
) -> io::Result<(usize, usize)> {
    // 生成名の casing は環境差があるため大文字に正規化して照合する
    // (数字・アンダースコアは大文字化の影響を受けない)。
    let net_upper = network.to_ascii_uppercase();
    if net_upper == "SFNN-1536" || net_upper == "SFNN-1536-V2" {
        if feature.feature_set != FeatureSet::HalfKaHmMerged || ft_out != 1536 {
            return invalid_data(format!(
                "Network `{network}` is HalfKA_hm2/1536 only, but Features declares {}(ft_out={ft_out})",
                feature.yo_name
            ));
        }
        return Ok((16, 32));
    }
    let body = net_upper
        .strip_prefix("SFNN_")
        .and_then(|s| s.strip_suffix("_K3K3"))
        .ok_or_else(|| {
            invalid_data_err(format!(
                "unsupported Network structure `{network}` (expected SFNN-1536 or SFNN_<key>_<ft>_<h1>_<h2>_k3k3)"
            ))
        })?;
    let parts: Vec<&str> = body.split('_').collect();
    if parts.len() != 4 || parts[0] != feature.gen_key.to_ascii_uppercase() {
        return invalid_data(format!(
            "Network structure `{network}` does not match feature `{}`",
            feature.gen_key
        ));
    }
    let name_ft = parse_usize(parts[1])?;
    if name_ft != ft_out {
        return invalid_data(format!(
            "Network `{network}` FT dimension {name_ft} disagrees with Features FT dimension {ft_out}"
        ));
    }
    let h1 = parse_usize(parts[2])?;
    let h2 = parse_usize(parts[3])?;
    // `l1_out = h1 + 1` / `l2_in = (l1_out-1)*2` を導出する前に h1/h2 を上限で弾く。
    // 後段のガードだけだと巨大 h1 で `h1 + 1` や `*2` が overflow/wrap して境界検査を
    // すり抜け得るため、算術の前に直接 bound する。
    if h1 > MAX_HIDDEN_DIM || h2 > MAX_HIDDEN_DIM {
        return invalid_data(format!(
            "Network `{network}` hidden dimensions out of range (h1={h1}, h2={h2}, max {MAX_HIDDEN_DIM})"
        ));
    }
    Ok((h1 + 1, h2))
}

fn between<'a>(haystack: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let after = haystack.split_once(start)?.1;
    Some(after.split_once(end)?.0)
}

fn parse_usize(value: &str) -> io::Result<usize> {
    value
        .parse::<usize>()
        .map_err(|error| invalid_data_err(format!("expected integer, got `{value}`: {error}")))
}

/// FT hash 直後の tensor 本体 (FT bias/weight LEB128 + 各 bucket の fc hash + L1/L2/L3
/// affine) を読み、逆量子化して [`LayerStackWeights`] に組む。次元は caller が arch
/// 検出結果から渡す。
fn read_yo_tensors<R: BufRead>(
    reader: &mut R,
    feature_set: FeatureSetSpec,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
) -> io::Result<LayerStackWeights> {
    let ft_b_i16 = read_leb128_tensor_i16(reader, Some(ft_out))?;
    let ft_w_i16 = read_leb128_tensor_i16(reader, Some(feature_set.ft_in() * ft_out))?;
    let qa = QA as f32;

    let mut weights = LayerStackWeights::zeroed(feature_set, ft_out, l1_out, l2_out, num_buckets);
    weights.ft_b = ft_b_i16
        .into_iter()
        .map(|value| value as f32 / qa)
        .collect();
    weights.ft_w = ft_w_i16
        .into_iter()
        .map(|value| value as f32 / qa)
        .collect();

    let l2_in = (l1_out - 1) * 2;
    for bucket in 0..num_buckets {
        expect_u32(reader, YO_NETWORK_HASH, "LayerStack hash")?;

        let (biases, dense_weights) = read_affine(reader, ft_out, l1_out)?;
        let l1_b_start = bucket * l1_out;
        weights.l1_b[l1_b_start..l1_b_start + l1_out].copy_from_slice(&biases);
        let l1_w_start = bucket * l1_out * ft_out;
        weights.l1_w[l1_w_start..l1_w_start + dense_weights.len()].copy_from_slice(&dense_weights);

        let (biases, dense_weights) = read_affine(reader, l2_in, l2_out)?;
        let l2_b_start = bucket * l2_out;
        weights.l2_b[l2_b_start..l2_b_start + l2_out].copy_from_slice(&biases);
        let l2_w_start = bucket * l2_out * l2_in;
        weights.l2_w[l2_w_start..l2_w_start + dense_weights.len()].copy_from_slice(&dense_weights);

        let (biases, dense_weights) = read_affine(reader, l2_out, 1)?;
        weights.l3_b[bucket] = biases[0];
        let l3_w_start = bucket * l2_out;
        weights.l3_w[l3_w_start..l3_w_start + l2_out].copy_from_slice(&dense_weights);
    }

    Ok(weights)
}

fn consume_optional_leb128_prefix<R: BufRead>(reader: &mut R) -> io::Result<()> {
    if reader.fill_buf()?.first() == Some(&YO_LEB128_PREFIX) {
        reader.consume(1);
    }
    Ok(())
}

fn read_affine<R: Read>(
    reader: &mut R,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let bias_scale = (QA * QB) as f32;
    let mut biases = Vec::with_capacity(output_dimensions);
    for _ in 0..output_dimensions {
        biases.push(read_i32(reader)? as f32 / bias_scale);
    }

    let weight_scale = QB as f32;
    let padded_input = pad32(input_dimensions);
    let mut weights = Vec::with_capacity(output_dimensions * input_dimensions);
    let mut row = vec![0_u8; padded_input];
    for _ in 0..output_dimensions {
        reader.read_exact(&mut row)?;
        weights.extend(
            row[..input_dimensions]
                .iter()
                .map(|&value| value as i8 as f32 / weight_scale),
        );
    }
    Ok((biases, weights))
}

fn expect_u32<R: Read>(reader: &mut R, expected: u32, field: &str) -> io::Result<()> {
    let actual = read_u32(reader)?;
    if actual != expected {
        return invalid_data(format!(
            "{field} mismatch: expected {expected:#010x}, got {actual:#010x}"
        ));
    }
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn reject_trailing_data<R: Read>(reader: &mut R) -> io::Result<()> {
    let mut byte = [0_u8; 1];
    if reader.read(&mut byte)? != 0 {
        return invalid_data("YaneuraOu input has trailing data after the last LayerStack");
    }
    Ok(())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn invalid_data<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_data_err(message.into()))
}

fn invalid_data_err(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use nnue_format::layerstack_weights::write_leb128_tensor_i16;

    /// YaneuraOu arch 文字列を net_to_yo と同形で組む (テスト用の期待入力)。
    fn yo_arch(feature_set: FeatureSet, ft_out: usize, l1_out: usize, l2_out: usize) -> String {
        let f = YO_FEATURES
            .iter()
            .find(|f| f.feature_set == feature_set)
            .unwrap();
        let input_size = feature_set.spec().ft_in();
        let network = if feature_set == FeatureSet::HalfKaHmMerged
            && ft_out == 1536
            && l1_out == 16
            && l2_out == 32
        {
            "SFNN-1536".to_string()
        } else {
            // 実 YaneuraOu (`nnue_arch_gen.py` の `arch.upper()`) と同じ大文字構造名。
            format!(
                "SFNN_{}_{}_{}_{}_k3k3",
                f.gen_key,
                ft_out,
                l1_out - 1,
                l2_out
            )
            .to_ascii_uppercase()
        };
        format!(
            "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{ft_out}x2],Network={network}{{LayerStack=9}}",
            f.yo_name
        )
    }

    #[test]
    fn parse_yo_arch_recovers_feature_and_dims() {
        let configs = [
            (FeatureSet::HalfKaHmMerged, 1536_usize, 16_usize, 32_usize),
            (FeatureSet::HalfKaHmMerged, 512, 16, 32),
            (FeatureSet::HalfKaHmMerged, 1024, 7, 16),
            (FeatureSet::HalfKp, 1536, 16, 32),
            (FeatureSet::HalfKaSplit, 768, 8, 16),
            (FeatureSet::HalfKaMerged, 1536, 16, 32),
            (FeatureSet::HalfKaHmSplit, 1536, 16, 32),
        ];
        for (fs, ft_out, l1_out, l2_out) in configs {
            let a = parse_yo_arch(&yo_arch(fs, ft_out, l1_out, l2_out)).expect("parses");
            assert_eq!(a.feature_set, fs);
            assert_eq!(a.ft_out, ft_out);
            assert_eq!(a.l1_out, l1_out);
            assert_eq!(a.l2_out, l2_out);
            assert_eq!(a.num_buckets, 9);
        }
        // baseline は Network 名 `SFNN-1536` と `SFNN-1536-V2` の両綴りを受理する。
        let v2 = parse_yo_arch(
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536-V2{LayerStack=9}",
        )
        .unwrap();
        assert_eq!(v2.feature_set, FeatureSet::HalfKaHmMerged);
        assert_eq!((v2.ft_out, v2.l1_out, v2.l2_out), (1536, 16, 32));
    }

    #[test]
    fn parse_yo_arch_accepts_generated_name_in_either_case() {
        // 実 YaneuraOu (`nnue_arch_gen.py` の `arch.upper()`) は大文字、旧 net_to_yo
        // 出力は小文字。どちらの綴りも同じ次元に解決する。
        let upper = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_HALFKAHM2_512_15_32_K3K3{LayerStack=9}";
        let lower = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_halfkahm2_512_15_32_k3k3{LayerStack=9}";
        for arch in [upper, lower] {
            let a = parse_yo_arch(arch).expect("parses");
            assert_eq!(a.feature_set, FeatureSet::HalfKaHmMerged);
            assert_eq!((a.ft_out, a.l1_out, a.l2_out), (512, 16, 32));
        }
    }

    #[test]
    fn parse_yo_arch_rejects_psqt_and_unknown_feature() {
        let psqt = "ModelType=SFNNWithPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}";
        assert_eq!(
            parse_yo_arch(psqt).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let unknown = "ModelType=SFNNWithoutPsqt;Features=HalfKZ9(Friend)[100->1536x2],Network=SFNN-1536{LayerStack=9}";
        assert!(
            parse_yo_arch(unknown)
                .unwrap_err()
                .to_string()
                .contains("unknown YaneuraOu feature")
        );
    }

    /// 完全な YaneuraOu ファイルバイト列を net_to_yo と同じ形式で組む。
    fn build_yo_bytes(
        w: &LayerStackWeights,
        arch: &str,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&YO_VERSION.to_le_bytes());
        out.extend_from_slice(&YO_TOP_HASH.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        out.extend_from_slice(arch.as_bytes());
        out.extend_from_slice(&YO_FT_HASH.to_le_bytes());

        let quant_i16 = |v: f32| (v as f64 * QA as f64).round() as i16;
        let ft_b: Vec<i16> = w.ft_b.iter().map(|&v| quant_i16(v)).collect();
        write_leb128_tensor_i16(&mut out, &ft_b).unwrap();
        let ft_w: Vec<i16> = w.ft_w.iter().map(|&v| quant_i16(v)).collect();
        write_leb128_tensor_i16(&mut out, &ft_w).unwrap();

        let l2_in = (l1_out - 1) * 2;
        let num_buckets = w.num_buckets;
        for b in 0..num_buckets {
            out.extend_from_slice(&YO_NETWORK_HASH.to_le_bytes());
            encode_affine(
                &mut out,
                &w.l1_b[b * l1_out..],
                &w.l1_w[b * l1_out * ft_out..],
                ft_out,
                l1_out,
            );
            encode_affine(
                &mut out,
                &w.l2_b[b * l2_out..],
                &w.l2_w[b * l2_out * l2_in..],
                l2_in,
                l2_out,
            );
            encode_affine(&mut out, &w.l3_b[b..], &w.l3_w[b * l2_out..], l2_out, 1);
        }
        out
    }

    fn encode_affine(
        out: &mut Vec<u8>,
        biases: &[f32],
        weights: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) {
        for &bias in &biases[..output_dim] {
            out.extend_from_slice(&((bias as f64 * (QA * QB) as f64).round() as i32).to_le_bytes());
        }
        let padded = pad32(input_dim);
        for o in 0..output_dim {
            for i in 0..padded {
                let byte = if i < input_dim {
                    (weights[o * input_dim + i] as f64 * QB as f64).round() as i8
                } else {
                    0
                };
                out.push(byte as u8);
            }
        }
    }

    #[test]
    fn read_yo_round_trips_weights_across_feature_sets_and_dims() {
        // 実 YaneuraOu SFNN は常に 9 bucket。FT 出力等を小さく取り全 5 feature set を
        // 高速に往復検証する。ft_out は健全性ガードの 32 の倍数制約を満たす最小値。
        let configs = [
            (FeatureSet::HalfKaHmMerged, 32_usize, 4_usize, 8_usize),
            (FeatureSet::HalfKaHmSplit, 32, 3, 8),
            (FeatureSet::HalfKp, 32, 5, 16),
            (FeatureSet::HalfKaSplit, 32, 4, 8),
            (FeatureSet::HalfKaMerged, 32, 6, 8),
        ];
        for (fs, ft_out, l1_out, l2_out) in configs {
            let spec = fs.spec();
            let mut w = LayerStackWeights::zeroed(spec, ft_out, l1_out, l2_out, REQUIRED_BUCKETS);
            // 量子化格子上の値 (k/QA, k/QB, k/(QA*QB)) を入れて厳密復元を確認する。
            for (i, v) in w.ft_b.iter_mut().enumerate() {
                *v = ((i % 7) as i16 - 3) as f32 / QA as f32;
            }
            for (i, v) in w.ft_w.iter_mut().enumerate() {
                *v = ((i % 5) as i16 - 2) as f32 / QA as f32;
            }
            for (i, v) in w.l1_w.iter_mut().enumerate() {
                *v = ((i % 9) as i8 - 4) as f32 / QB as f32;
            }
            for (i, v) in w.l2_w.iter_mut().enumerate() {
                *v = ((i % 11) as i8 - 5) as f32 / QB as f32;
            }
            for (i, v) in w.l3_w.iter_mut().enumerate() {
                *v = ((i % 3) as i8 - 1) as f32 / QB as f32;
            }
            for (i, v) in w.l1_b.iter_mut().enumerate() {
                *v = (i as i32 - 2) as f32 / (QA * QB) as f32;
            }
            for (i, v) in w.l2_b.iter_mut().enumerate() {
                *v = (i as i32 - 3) as f32 / (QA * QB) as f32;
            }
            for (i, v) in w.l3_b.iter_mut().enumerate() {
                *v = (i as i32 - 1) as f32 / (QA * QB) as f32;
            }

            let arch = yo_arch(fs, ft_out, l1_out, l2_out);
            let bytes = build_yo_bytes(&w, &arch, ft_out, l1_out, l2_out);

            let mut reader = Cursor::new(bytes);
            let parsed = read_yo(&mut reader).expect("read_yo");
            reject_trailing_data(&mut reader).expect("no trailing data");

            assert_eq!(parsed.ft_b, w.ft_b, "{fs:?} ft_b");
            assert_eq!(parsed.ft_w, w.ft_w, "{fs:?} ft_w");
            assert_eq!(parsed.l1_b, w.l1_b, "{fs:?} l1_b");
            assert_eq!(parsed.l1_w, w.l1_w, "{fs:?} l1_w");
            assert_eq!(parsed.l2_b, w.l2_b, "{fs:?} l2_b");
            assert_eq!(parsed.l2_w, w.l2_w, "{fs:?} l2_w");
            assert_eq!(parsed.l3_b, w.l3_b, "{fs:?} l3_b");
            assert_eq!(parsed.l3_w, w.l3_w, "{fs:?} l3_w");

            // 変換器の native 出力 (tatara .bin) が byte 一致することも確認する:
            // 元 weights と往復後 weights の save_quantised は同一バイト列になる。
            let mut direct = Vec::new();
            w.save_quantised(&mut direct, Some(nnue_format::layerstack_weights::FV_SCALE))
                .unwrap();
            let mut via = Vec::new();
            parsed
                .save_quantised(&mut via, Some(nnue_format::layerstack_weights::FV_SCALE))
                .unwrap();
            assert_eq!(direct, via, "{fs:?} save_quantised byte-identity");
        }
    }

    #[test]
    fn read_yo_rejects_bad_header_fields() {
        let arch = yo_arch(FeatureSet::HalfKaHmMerged, 1536, 16, 32);
        let header = |version: u32, top: u32, ft: u32, arch: &str| {
            let mut b = Vec::new();
            b.extend_from_slice(&version.to_le_bytes());
            b.extend_from_slice(&top.to_le_bytes());
            b.extend_from_slice(&(arch.len() as u32).to_le_bytes());
            b.extend_from_slice(arch.as_bytes());
            b.extend_from_slice(&ft.to_le_bytes());
            b
        };
        for (label, bytes) in [
            (
                "version",
                header(0xDEAD_BEEF, YO_TOP_HASH, YO_FT_HASH, &arch),
            ),
            (
                "top-level hash",
                header(YO_VERSION, 0xDEAD_BEEF, YO_FT_HASH, &arch),
            ),
            (
                "feature-transformer hash",
                header(YO_VERSION, YO_TOP_HASH, 0xDEAD_BEEF, &arch),
            ),
        ] {
            let error = read_yo(&mut Cursor::new(bytes)).unwrap_err();
            assert!(
                error.to_string().contains(label),
                "expected `{label}` in: {error}"
            );
        }
        // arch 不一致 (PSQT) も header 段で reject。
        let bad_arch = header(
            YO_VERSION,
            YO_TOP_HASH,
            YO_FT_HASH,
            "ModelType=SFNNWithPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}",
        );
        assert_eq!(
            read_yo(&mut Cursor::new(bad_arch)).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn read_yo_rejects_corrupt_layerstack_hash_and_trailing_garbage() {
        let (fs, ft_out, l1_out, l2_out) = (FeatureSet::HalfKaHmMerged, 32, 4, 8);
        // 全ゼロ weights: 0x6333718A の出現は per-bucket network hash だけになる。
        let w = LayerStackWeights::zeroed(fs.spec(), ft_out, l1_out, l2_out, REQUIRED_BUCKETS);
        let arch = yo_arch(fs, ft_out, l1_out, l2_out);
        let good = build_yo_bytes(&w, &arch, ft_out, l1_out, l2_out);

        // 正常入力 + 末尾ゴミ: read_yo は成功し reject_trailing_data が弾く。
        let mut trailing = good.clone();
        trailing.push(0xAB);
        let mut reader = Cursor::new(trailing);
        read_yo(&mut reader).unwrap();
        assert_eq!(
            reject_trailing_data(&mut reader).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        // per-LayerStack network hash を壊すと reject。
        let mut corrupt = good;
        let needle = YO_NETWORK_HASH.to_le_bytes();
        let pos = corrupt
            .windows(4)
            .position(|w| w == needle)
            .expect("network hash present");
        corrupt[pos] ^= 0xFF;
        let error = read_yo(&mut Cursor::new(corrupt)).unwrap_err();
        assert!(
            error.to_string().contains("LayerStack hash"),
            "got: {error}"
        );
    }

    #[test]
    fn parse_yo_arch_rejects_inconsistent_or_out_of_range_dims() {
        // Network 名の FT 次元が Features の FT 次元と食い違う。
        let mismatch = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_halfkahm2_1024_15_32_k3k3{LayerStack=9}";
        assert!(
            parse_yo_arch(mismatch)
                .unwrap_err()
                .to_string()
                .contains("disagrees")
        );
        // FT 次元が上限超過: 過大 allocation の前にガードで弾く。
        let huge = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->32768x2],Network=SFNN_halfkahm2_32768_15_32_k3k3{LayerStack=9}";
        assert!(
            parse_yo_arch(huge)
                .unwrap_err()
                .to_string()
                .contains("FT output")
        );
        // 隠れ層 h1 が上限超過: `h1 + 1` / `(l1_out-1)*2` の overflow 前に bound で弾く
        // (usize 幅に依らず 32-bit でもパース可能な値で境界を踏む)。
        let huge_h1 = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN_halfkahm2_1536_100000_32_k3k3{LayerStack=9}";
        assert!(
            parse_yo_arch(huge_h1)
                .unwrap_err()
                .to_string()
                .contains("hidden dimensions")
        );
        // Features の入力次元 <in> が feature の ft_in と食い違う。
        let bad_input = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[999->1536x2],Network=SFNN-1536{LayerStack=9}";
        assert!(
            parse_yo_arch(bad_input)
                .unwrap_err()
                .to_string()
                .contains("input dimension")
        );
        // 先行 decoy token が期待値を供給しても、Features トークンに anchor して弾く。
        let decoy = "ModelType=SFNNWithoutPsqt;X=(Friend)[73305->1536x2],Features=HalfKA_hm2(Friend)[999->1536x2],Network=SFNN-1536{LayerStack=9}";
        assert!(
            parse_yo_arch(decoy)
                .unwrap_err()
                .to_string()
                .contains("input dimension")
        );
        // 非 9 bucket (LayerStack=巨大値) は自己矛盾として reject。
        let buckets = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=1000000000000}";
        assert!(
            parse_yo_arch(buckets)
                .unwrap_err()
                .to_string()
                .contains("LayerStack count")
        );
        // baseline 名なのに feature が HalfKA_hm2 でない。
        let wrong_baseline = "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->1536x2],Network=SFNN-1536{LayerStack=9}";
        assert_eq!(
            parse_yo_arch(wrong_baseline).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn affine_reader_discards_padding_and_preserves_quantised_values() {
        let mut input = Vec::new();
        input.extend_from_slice(&(QA * QB).to_le_bytes());
        input.extend_from_slice(&(-(QA * QB)).to_le_bytes());
        input.extend_from_slice(&[1, 2, 255]);
        input.extend(std::iter::repeat_n(77, 29));
        input.extend_from_slice(&[254, 253, 252]);
        input.extend(std::iter::repeat_n(88, 29));

        let (biases, weights) = read_affine(&mut &input[..], 3, 2).unwrap();

        assert_eq!(biases, vec![1.0, -1.0]);
        assert_eq!(
            weights,
            vec![
                1.0 / 64.0,
                2.0 / 64.0,
                -1.0 / 64.0,
                -2.0 / 64.0,
                -3.0 / 64.0,
                -4.0 / 64.0
            ]
        );
    }

    #[test]
    fn optional_leb128_marker_prefix_is_consumed() {
        let mut prefixed = Cursor::new(b"_COMPRESSED_LEB128".to_vec());
        consume_optional_leb128_prefix(&mut prefixed).unwrap();
        assert_eq!(prefixed.position(), 1);

        let mut plain = Cursor::new(b"COMPRESSED_LEB128".to_vec());
        consume_optional_leb128_prefix(&mut plain).unwrap();
        assert_eq!(plain.position(), 0);
    }

    #[test]
    fn trailing_input_is_rejected() {
        let error = reject_trailing_data(&mut &b"x"[..]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        reject_trailing_data(&mut &b""[..]).unwrap();
    }

    #[test]
    fn kingrank9_requires_an_explicit_assertion() {
        let error = require_kingrank9_assertion(false).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--assume-kingrank9"));
        require_kingrank9_assertion(true).unwrap();
    }
}
