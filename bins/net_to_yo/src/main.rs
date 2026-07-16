use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    LEGACY_NNUE_VERSION_BUCKETS9, NNUE_VERSION, QA, QB, write_leb128_tensor_i16,
};
use shogi_features::FeatureSet;

/// YaneuraOu SFNNwoPSQT ビルドが評価ファイルに要求する 4 つのハッシュ。
/// SFNNwoPSQT では feature set / 次元に依らず固定定数で、YaneuraOu 側は
/// `evaluate_nnue.cpp` で version 以外の不一致を warning 扱いにするが、ここでは
/// 正しい定数を書き出して生成済み YaneuraOu ビルドと byte 一致させる。
const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;

/// LayerStack バケット数。YaneuraOu SFNN は KingRank9 (3x3) 固定で、変換対象も
/// これに揃える。
const YO_LAYER_STACKS: usize = 9;

/// 変換対象の SFNN 次元上限。実在アーキは十分収まり、壊れた arch 文字列 (0 次元 /
/// 巨大値) が overflow や過大 allocation を起こす前に弾くための健全性ガード。
const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;

/// tatara feature set と YaneuraOu SFNN feature の対応。
///
/// - `yo_name`: YaneuraOu `GetName()` (`Features=<yo_name>(Friend)[...]` に入る)
/// - `gen_key`: `nnue_arch_gen.py` の feature キーワード
///   (非既定次元の `Network=SFNN_<gen_key>_<ft>_<h1>_<h2>_k3k3` に入る)
///
/// tatara と YaneuraOu の feature index は同一 Apery-BonaPiece 規約で恒等一致する
/// ため、重みは並べ替えなしでそのまま移送できる (`DISTINGUISH_GOLDS` 無効の
/// YaneuraOu ビルドが前提)。
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

fn yo_feature(feature_set: FeatureSet) -> &'static YoFeature {
    YO_FEATURES
        .iter()
        .find(|f| f.feature_set == feature_set)
        .expect("every FeatureSet variant has a YaneuraOu mapping")
}

/// tatara `.bin` header から読み取った変換対象アーキ。
#[derive(Debug)]
struct DetectedArch {
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
}

#[derive(Parser)]
#[command(about = "Convert a tatara LayerStack net to a YaneuraOu SFNN evaluation file")]
struct Args {
    /// tatara LayerStack quantised .bin
    #[arg(long)]
    input: PathBuf,
    /// YaneuraOu nn.bin
    #[arg(long)]
    output: PathBuf,
    /// Assert that the input was trained with `--bucket-mode kingrank9`.
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

    let detect_input = File::open(&args.input)?;
    let arch = detect_arch(&mut BufReader::new(detect_input))?;

    let input = File::open(&args.input)?;
    let mut reader = BufReader::new(input);
    let weights = LayerStackWeights::load_quantised(
        &mut reader,
        arch.feature_set.spec(),
        arch.ft_out,
        arch.l1_out,
        arch.l2_out,
        YO_LAYER_STACKS,
    )?;
    reject_trailing_data(&mut reader)?;

    let output = File::create(&args.output)?;
    let mut writer = BufWriter::new(output);
    write_yo(&mut writer, &arch, &weights)?;
    writer.flush()?;
    Ok(())
}

fn require_kingrank9_assertion(assume_kingrank9: bool) -> io::Result<()> {
    if !assume_kingrank9 {
        return invalid_input(
            "tatara .bin files do not record bucket routing; pass --assume-kingrank9 only after confirming the net was trained with --bucket-mode kingrank9",
        );
    }
    Ok(())
}

/// `.bin` header (version + network_hash + arch_str + num_buckets) を読み、変換
/// 可能な SFNN アーキかを判定する。PSQT / threat / effect bucket / 非 9 bucket /
/// 未知 feature は YaneuraOu SFNN に受け皿が無いため明示的に reject する。
fn detect_arch<R: Read>(reader: &mut R) -> io::Result<DetectedArch> {
    let version = read_u32(reader)?;
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return invalid_input(format!(
            "unknown tatara NNUE version: {version:#x} (expected {NNUE_VERSION:#x} or legacy {LEGACY_NNUE_VERSION_BUCKETS9:#x})"
        ));
    }
    let _network_hash = read_u32(reader)?;

    let arch_len = read_u32(reader)? as usize;
    if arch_len == 0 || arch_len > 16_384 {
        return invalid_input(format!("invalid arch string length: {arch_len}"));
    }
    let mut arch_bytes = vec![0_u8; arch_len];
    reader.read_exact(&mut arch_bytes)?;
    let arch_str = std::str::from_utf8(&arch_bytes)
        .map_err(|error| invalid_input_err(format!("arch string is not UTF-8: {error}")))?;

    // num_buckets は現行 version のみ header に持ち、legacy は暗黙 9。
    let num_buckets = if version == LEGACY_NNUE_VERSION_BUCKETS9 {
        YO_LAYER_STACKS
    } else {
        read_u32(reader)? as usize
    };
    if num_buckets != YO_LAYER_STACKS {
        return invalid_input(format!(
            "YaneuraOu SFNN requires {YO_LAYER_STACKS} LayerStacks (KingRank9), but the input has {num_buckets} buckets"
        ));
    }

    parse_arch_str(arch_str)
}

/// tatara `build_arch_str` が生成する arch 文字列から feature set と隠れ層次元を
/// 取り出す。書式は
/// `Features=<name>(Friend)[<in>-><ft>x2],...,Network=AffineTransform[1<-<l2_out>](...
/// SqrClippedReLU[<l2_in>](AffineTransform[<l1_out>-<ft*2>]...`。
fn parse_arch_str(arch_str: &str) -> io::Result<DetectedArch> {
    for unsupported in ["PSQT=", "Threat=", "EffectBucket="] {
        if arch_str.contains(unsupported) {
            let token = unsupported.trim_end_matches('=');
            return invalid_input(format!(
                "{token} models are not representable in YaneuraOu SFNN and cannot be converted"
            ));
        }
    }

    let features = between(arch_str, "Features=", "(Friend)[").ok_or_else(|| {
        invalid_input_err("arch string has no `Features=<name>(Friend)[` token".to_string())
    })?;
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|fs| fs.spec().arch_feature_name() == features)
        .ok_or_else(|| invalid_input_err(format!("unknown feature set `{features}`")))?;

    let ft_out = between(arch_str, "->", "x2")
        .ok_or_else(|| invalid_input_err("arch string has no `-><ft>x2` token".to_string()))
        .and_then(parse_usize)?;
    // YaneuraOu は kTransformedFeatureDimensions % kMaxSimdWidth(32) == 0 を要求する。
    if ft_out == 0 || ft_out > MAX_FT_OUT || ft_out % 32 != 0 {
        return invalid_input(format!(
            "unsupported FT output dimension {ft_out} (expected a positive multiple of 32 up to {MAX_FT_OUT})"
        ));
    }

    let l2_out = between(arch_str, "AffineTransform[1<-", "]")
        .ok_or_else(|| invalid_input_err("arch string has no output affine token".to_string()))
        .and_then(parse_usize)?;
    if l2_out == 0 || l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {l2_out} (expected 1..={MAX_HIDDEN_DIM})"
        ));
    }

    let l2_in = between(arch_str, "SqrClippedReLU[", "]")
        .ok_or_else(|| invalid_input_err("arch string has no SqrClippedReLU token".to_string()))
        .and_then(parse_usize)?;
    if l2_in == 0 || l2_in % 2 != 0 || l2_in > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 input dimension {l2_in} (expected a positive even value up to {MAX_HIDDEN_DIM})"
        ));
    }
    // L1 出力のうち skip 1 dim を除いた `l1_out - 1` を 2 乗連結して L2 入力にする
    // ため、`l2_in = (l1_out - 1) * 2`。
    let l1_out = l2_in / 2 + 1;

    Ok(DetectedArch {
        feature_set,
        ft_out,
        l1_out,
        l2_out,
    })
}

fn between<'a>(haystack: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let after = haystack.split_once(start)?.1;
    Some(after.split_once(end)?.0)
}

fn parse_usize(value: &str) -> io::Result<usize> {
    value
        .parse::<usize>()
        .map_err(|error| invalid_input_err(format!("expected integer, got `{value}`: {error}")))
}

fn reject_trailing_data<R: Read>(reader: &mut R) -> io::Result<()> {
    let mut byte = [0_u8; 1];
    if reader.read(&mut byte)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tatara input has trailing data after the expected {YO_LAYER_STACKS} LayerStacks"
            ),
        ));
    }
    Ok(())
}

/// YaneuraOu `GetArchitectureString()` と同形の arch 文字列を生成する。既定次元
/// (halfkahm2 / 1536 / l1=16 / l2=32) は `sfnn-1536.h` の構造名 `SFNN-1536`、
/// それ以外は `nnue_arch_gen.py` が返す構造名に揃える。生成器はアーキ名を大文字化
/// する (`arch.upper()`) ため `SFNN_<GEN_KEY>_<ft>_<h1>_<h2>_K3K3` と大文字で出力する
/// (`h1 = l1_out - 1`, `h2 = l2_out`)。
fn yo_arch_string(arch: &DetectedArch) -> String {
    let feature = yo_feature(arch.feature_set);
    let input_size = arch.feature_set.spec().ft_in();
    let h1 = arch.l1_out - 1;
    let h2 = arch.l2_out;
    let network = if arch.feature_set == FeatureSet::HalfKaHmMerged
        && arch.ft_out == 1536
        && arch.l1_out == 16
        && arch.l2_out == 32
    {
        "SFNN-1536".to_string()
    } else {
        format!(
            "SFNN_{}_{}_{}_{}_k3k3",
            feature.gen_key, arch.ft_out, h1, h2
        )
        .to_ascii_uppercase()
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{}x2],Network={network}{{LayerStack={YO_LAYER_STACKS}}}",
        feature.yo_name, arch.ft_out
    )
}

fn write_yo<W: Write>(
    writer: &mut W,
    arch: &DetectedArch,
    weights: &LayerStackWeights,
) -> io::Result<()> {
    validate_weights(arch, weights)?;

    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    let arch_string = yo_arch_string(arch);
    write_u32(
        writer,
        u32::try_from(arch_string.len()).expect("arch string length fits in u32"),
    )?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    let ft_biases = quantize_i16(&weights.ft_b, QA as f64);
    write_leb128_tensor_i16(writer, &ft_biases)?;
    let ft_weights = quantize_i16(&weights.ft_w, QA as f64);
    write_leb128_tensor_i16(writer, &ft_weights)?;

    for bucket in 0..YO_LAYER_STACKS {
        write_u32(writer, YO_NETWORK_HASH)?;

        // l1f (factorizer 共有項) は save 時に l1 へ merge 済みで load 側は常に 0 を
        // 返す。加算は未 merge の入力が来ても正しい防御であって通常経路では no-op。
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1f_b[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1f_w[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out, l1_out)?;

        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;

        let l3_biases = std::iter::once(weights.l3_b[bucket]);
        let l3_weights = (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]);
        write_affine(writer, l3_biases, l3_weights, l2_out, 1)?;
    }
    Ok(())
}

fn validate_weights(arch: &DetectedArch, weights: &LayerStackWeights) -> io::Result<()> {
    let expected_feature_set = arch.feature_set.spec();
    if weights.feature_set != expected_feature_set {
        return invalid_input(format!(
            "feature set mismatch: detected {}, but weights carry a different spec",
            arch.feature_set.canonical_name()
        ));
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not supported");
    }
    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;
    let lengths = [
        ("ft_b", weights.ft_b.len(), ft_out),
        (
            "ft_w",
            weights.ft_w.len(),
            expected_feature_set.ft_in() * ft_out,
        ),
        ("l1_b", weights.l1_b.len(), YO_LAYER_STACKS * l1_out),
        (
            "l1_w",
            weights.l1_w.len(),
            YO_LAYER_STACKS * l1_out * ft_out,
        ),
        ("l1f_b", weights.l1f_b.len(), l1_out),
        ("l1f_w", weights.l1f_w.len(), ft_out * l1_out),
        ("l2_b", weights.l2_b.len(), YO_LAYER_STACKS * l2_out),
        ("l2_w", weights.l2_w.len(), YO_LAYER_STACKS * l2_out * l2_in),
        ("l3_b", weights.l3_b.len(), YO_LAYER_STACKS),
        ("l3_w", weights.l3_w.len(), YO_LAYER_STACKS * l2_out),
    ];
    for (name, actual, expected) in lengths {
        if actual != expected {
            return invalid_input(format!(
                "{name} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_input_err(message.into()))
}

fn invalid_input_err(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn write_affine<W, B, V>(
    writer: &mut W,
    biases: B,
    weights: V,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<()>
where
    W: Write,
    B: IntoIterator<Item = f32>,
    V: IntoIterator<Item = f32>,
{
    for bias in biases {
        writer.write_all(&quantize_i32(bias, (QA * QB) as f64).to_le_bytes())?;
    }

    let padded_input = input_dimensions.div_ceil(32) * 32;
    let mut weights = weights.into_iter();
    for _ in 0..output_dimensions {
        for input in 0..padded_input {
            let value = if input < input_dimensions {
                weights.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "affine weight iterator is short",
                    )
                })?
            } else {
                0.0
            };
            writer.write_all(&[quantize_i8(value, QB as f64) as u8])?;
        }
    }
    if weights.next().is_some() {
        return invalid_input("affine weight iterator has extra values");
    }
    Ok(())
}

fn quantize_i16(values: &[f32], scale: f64) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            (value as f64 * scale)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

fn quantize_i32(value: f32, scale: f64) -> i32 {
    (value as f64 * scale)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn quantize_i8(value: f32, scale: f64) -> i8 {
    (value as f64 * scale)
        .round()
        .clamp(i8::MIN as f64, i8::MAX as f64) as i8
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnue_format::layerstack_weights::build_arch_str;

    fn detected(
        feature_set: FeatureSet,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
    ) -> DetectedArch {
        DetectedArch {
            feature_set,
            ft_out,
            l1_out,
            l2_out,
        }
    }

    #[test]
    fn header_uses_fixed_sfnnwopsqt_constants() {
        assert_eq!(YO_VERSION, 0x7af3_2f16);
        assert_eq!(YO_TOP_HASH, 0x3c20_3b32);
        assert_eq!(YO_FT_HASH, 0x5f13_4ab8);
        assert_eq!(YO_NETWORK_HASH, 0x6333_718a);
    }

    #[test]
    fn arch_string_matches_yaneuraou_for_baseline() {
        let arch = detected(FeatureSet::HalfKaHmMerged, 1536, 16, 32);
        assert_eq!(
            yo_arch_string(&arch),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}"
        );
    }

    #[test]
    fn arch_string_uses_generated_network_name_off_baseline() {
        assert_eq!(
            yo_arch_string(&detected(FeatureSet::HalfKaHmMerged, 512, 16, 32)),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_HALFKAHM2_512_15_32_K3K3{LayerStack=9}"
        );
        assert_eq!(
            yo_arch_string(&detected(FeatureSet::HalfKp, 1536, 16, 32)),
            "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->1536x2],Network=SFNN_HALFKP_1536_15_32_K3K3{LayerStack=9}"
        );
    }

    #[test]
    fn every_feature_set_maps_to_a_yaneuraou_feature() {
        for fs in FeatureSet::ALL {
            let feature = yo_feature(fs);
            assert_eq!(feature.feature_set, fs);
            assert!(!feature.yo_name.is_empty());
            assert!(!feature.gen_key.is_empty());
        }
    }

    #[test]
    fn parse_arch_str_roundtrips_build_arch_str_over_dims_and_features() {
        let configs = [
            (FeatureSet::HalfKaHmMerged, 1536_usize, 16_usize, 32_usize),
            (FeatureSet::HalfKaHmMerged, 512, 16, 32),
            (FeatureSet::HalfKaHmMerged, 1024, 8, 16),
            (FeatureSet::HalfKp, 1536, 16, 32),
            (FeatureSet::HalfKaSplit, 768, 16, 32),
            (FeatureSet::HalfKaMerged, 1536, 16, 32),
            (FeatureSet::HalfKaHmSplit, 1536, 16, 32),
        ];
        for (fs, ft_out, l1_out, l2_out) in configs {
            let spec = fs.spec();
            let l2_in = (l1_out - 1) * 2;
            let arch_str = build_arch_str(
                spec.arch_feature_name(),
                spec.ft_in(),
                ft_out,
                l1_out,
                l2_in,
                l2_out,
                Some(28),
                None,
                None,
                None,
            );
            let parsed = parse_arch_str(&arch_str).expect("parses");
            assert_eq!(parsed.feature_set, fs);
            assert_eq!(parsed.ft_out, ft_out);
            assert_eq!(parsed.l1_out, l1_out);
            assert_eq!(parsed.l2_out, l2_out);
        }
    }

    #[test]
    fn parse_arch_str_rejects_psqt_threat_effect() {
        use nnue_format::layerstack_weights::{EffectBucketArch, ThreatArch};

        let name = FeatureSet::HalfKaHmMerged.spec().arch_feature_name();
        let cases = [
            ("PSQT", Some(9), None, None),
            (
                "Threat",
                None,
                Some(ThreatArch {
                    dims: 128,
                    profile_id: 0,
                }),
                None,
            ),
            (
                "EffectBucket",
                None,
                None,
                Some(EffectBucketArch {
                    nb: 4,
                    king_bucketed: false,
                }),
            ),
        ];
        for (token, psqt, threat, effect) in cases {
            let arch_str = build_arch_str(
                name,
                73305,
                1536,
                16,
                30,
                32,
                Some(28),
                psqt,
                threat,
                effect,
            );
            let error = parse_arch_str(&arch_str).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(
                error.to_string().contains(token),
                "expected {token} in error, got: {error}"
            );
        }
    }

    #[test]
    fn parse_arch_str_rejects_degenerate_dimensions() {
        let good = build_arch_str(
            FeatureSet::HalfKaHmMerged.spec().arch_feature_name(),
            73305,
            1536,
            16,
            30,
            32,
            Some(28),
            None,
            None,
            None,
        );
        // ft_out = 0 / ft_out が上限超過 (32 の倍数だが MAX_FT_OUT 超) / l2_out = 0。
        for (bad, needle) in [
            (good.replace("->1536x2", "->0x2"), "FT output"),
            (good.replace("->1536x2", "->32768x2"), "FT output"),
            (
                good.replace("AffineTransform[1<-32]", "AffineTransform[1<-0]"),
                "L2 output",
            ),
        ] {
            let error = parse_arch_str(&bad).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(needle), "got: {error}");
        }
    }

    fn header_bytes(version: u32, arch_str: &str, num_buckets: Option<u32>) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // network_hash (detect は無視)
        bytes.extend_from_slice(&(arch_str.len() as u32).to_le_bytes());
        bytes.extend_from_slice(arch_str.as_bytes());
        if let Some(n) = num_buckets {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn detect_arch_rejects_non9_current_buckets_and_accepts_legacy_implicit9() {
        let arch_str = build_arch_str(
            FeatureSet::HalfKaHmMerged.spec().arch_feature_name(),
            73305,
            1536,
            16,
            30,
            32,
            Some(28),
            None,
            None,
            None,
        );

        let non9 = header_bytes(NNUE_VERSION, &arch_str, Some(4));
        let error = detect_arch(&mut std::io::Cursor::new(non9)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("LayerStacks"), "got: {error}");

        let legacy = header_bytes(LEGACY_NNUE_VERSION_BUCKETS9, &arch_str, None);
        let arch = detect_arch(&mut std::io::Cursor::new(legacy)).expect("legacy implicit 9");
        assert_eq!(arch.feature_set, FeatureSet::HalfKaHmMerged);
        assert_eq!(arch.ft_out, 1536);
        assert_eq!(arch.l1_out, 16);
        assert_eq!(arch.l2_out, 32);
    }

    #[test]
    fn affine_file_weights_are_canonical_row_major_with_zero_padding() {
        let mut output = Vec::new();
        write_affine(
            &mut output,
            [1.0, -1.0],
            [
                1.0 / 64.0,
                2.0 / 64.0,
                3.0 / 64.0,
                -1.0 / 64.0,
                -2.0 / 64.0,
                -3.0 / 64.0,
            ],
            3,
            2,
        )
        .unwrap();

        assert_eq!(
            i32::from_le_bytes(output[0..4].try_into().unwrap()),
            QA * QB
        );
        assert_eq!(
            i32::from_le_bytes(output[4..8].try_into().unwrap()),
            -(QA * QB)
        );
        assert_eq!(&output[8..11], &[1, 2, 3]);
        assert!(output[11..40].iter().all(|&byte| byte == 0));
        assert_eq!(&output[40..43], &[255, 254, 253]);
        assert!(output[43..72].iter().all(|&byte| byte == 0));
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

    /// zeroed weights から合成した tatara `.bin` を返す。
    fn synthetic_bin(
        feature_set: FeatureSet,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
    ) -> Vec<u8> {
        let weights =
            LayerStackWeights::zeroed(feature_set.spec(), ft_out, l1_out, l2_out, YO_LAYER_STACKS);
        let mut bytes = Vec::new();
        weights
            .save_quantised(&mut bytes, Some(nnue_format::layerstack_weights::FV_SCALE))
            .expect("save synthetic .bin");
        bytes
    }

    /// 合成 `.bin` を `detect_arch` → `load_quantised` → `write_yo` のフル経路に
    /// 通し、YaneuraOu 出力バイト列を返す。detect が期待アーキと一致することも確認。
    fn convert(bytes: &[u8], expect: &DetectedArch) -> Vec<u8> {
        let arch = detect_arch(&mut std::io::Cursor::new(bytes)).expect("detect");
        assert_eq!(arch.feature_set, expect.feature_set);
        assert_eq!(arch.ft_out, expect.ft_out);
        assert_eq!(arch.l1_out, expect.l1_out);
        assert_eq!(arch.l2_out, expect.l2_out);

        let mut load_reader = std::io::Cursor::new(bytes);
        let weights = LayerStackWeights::load_quantised(
            &mut load_reader,
            arch.feature_set.spec(),
            arch.ft_out,
            arch.l1_out,
            arch.l2_out,
            YO_LAYER_STACKS,
        )
        .expect("load_quantised");
        reject_trailing_data(&mut load_reader).expect("no trailing data");

        let mut out = Vec::new();
        write_yo(&mut out, &arch, &weights).expect("write_yo");
        out
    }

    #[test]
    fn full_pipeline_produces_valid_yo_header_across_feature_sets_and_dims() {
        // 検証対象は header と affine のパディング済み次元追随なので、FT 出力は
        // 小さめ (128 の倍数) にして全 feature set を高速に網羅する。
        let configs = [
            (FeatureSet::HalfKaHmMerged, 256_usize, 16_usize, 32_usize),
            (FeatureSet::HalfKp, 128, 16, 32),
            (FeatureSet::HalfKaSplit, 128, 8, 16),
            (FeatureSet::HalfKaMerged, 128, 16, 32),
            (FeatureSet::HalfKaHmSplit, 256, 7, 16),
        ];
        for (fs, ft_out, l1_out, l2_out) in configs {
            let expect = detected(fs, ft_out, l1_out, l2_out);
            let bytes = synthetic_bin(fs, ft_out, l1_out, l2_out);
            let out = convert(&bytes, &expect);

            assert_eq!(
                u32::from_le_bytes(out[0..4].try_into().unwrap()),
                YO_VERSION
            );
            assert_eq!(
                u32::from_le_bytes(out[4..8].try_into().unwrap()),
                YO_TOP_HASH
            );
            let arch_len = u32::from_le_bytes(out[8..12].try_into().unwrap()) as usize;
            let arch_str = std::str::from_utf8(&out[12..12 + arch_len]).unwrap();
            assert_eq!(arch_str, yo_arch_string(&expect));
            let ft_hash_at = 12 + arch_len;
            assert_eq!(
                u32::from_le_bytes(out[ft_hash_at..ft_hash_at + 4].try_into().unwrap()),
                YO_FT_HASH
            );
        }
    }
}
