use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    DEFAULT_FT_OUT, DEFAULT_L1_OUT, DEFAULT_L2_OUT, DEFAULT_NUM_BUCKETS, QA, QB,
    write_leb128_tensor_i16,
};
use shogi_features::FeatureSet;

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;
const YO_ARCH: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536-V2{LayerStack=9}";

#[derive(Parser)]
#[command(about = "Convert a tatara LayerStack net for YaneuraOu SFNN-1536-V2")]
struct Args {
    /// tatara LayerStack quantised .bin
    #[arg(long)]
    input: PathBuf,
    /// YaneuraOu nn.bin
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output {
        return Err("input and output must be different paths".into());
    }

    let input = File::open(&args.input)?;
    let mut reader = BufReader::new(input);
    let weights = LayerStackWeights::load_quantised(
        &mut reader,
        FeatureSet::HalfKaHmMerged.spec(),
        DEFAULT_FT_OUT,
        DEFAULT_L1_OUT,
        DEFAULT_L2_OUT,
        DEFAULT_NUM_BUCKETS,
    )?;
    reject_trailing_data(&mut reader)?;

    let output = File::create(&args.output)?;
    let mut writer = BufWriter::new(output);
    write_yo(&mut writer, &weights)?;
    writer.flush()?;
    Ok(())
}

fn reject_trailing_data<R: Read>(reader: &mut R) -> io::Result<()> {
    let mut byte = [0_u8; 1];
    if reader.read(&mut byte)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tatara input has trailing data after the expected 9 LayerStacks",
        ));
    }
    Ok(())
}

fn write_yo<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()> {
    validate_weights(weights)?;

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    write_u32(
        writer,
        u32::try_from(YO_ARCH.len()).expect("YO architecture string length fits in u32"),
    )?;
    writer.write_all(YO_ARCH.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    let ft_biases = quantize_i16(&weights.ft_b, QA as f64);
    write_leb128_tensor_i16(writer, &ft_biases)?;
    let ft_weights = quantize_i16(&weights.ft_w, QA as f64);
    write_leb128_tensor_i16(writer, &ft_weights)?;

    let l2_in = (DEFAULT_L1_OUT - 1) * 2;
    for bucket in 0..DEFAULT_NUM_BUCKETS {
        write_u32(writer, YO_NETWORK_HASH)?;

        // l1f (factorizer 共有項) は save 時に l1 へ merge 済みで load 側は常に 0 を返す。
        // 加算は「未 merge の入力が来ても正しい」防御であって、通常経路では no-op。
        let l1_biases = (0..DEFAULT_L1_OUT)
            .map(|output| weights.l1_b[bucket * DEFAULT_L1_OUT + output] + weights.l1f_b[output]);
        let l1_weights = (0..DEFAULT_L1_OUT).flat_map(|output| {
            (0..DEFAULT_FT_OUT).map(move |input| {
                weights.l1_w
                    [bucket * DEFAULT_L1_OUT * DEFAULT_FT_OUT + output * DEFAULT_FT_OUT + input]
                    + weights.l1f_w[input * DEFAULT_L1_OUT + output]
            })
        });
        write_affine(
            writer,
            l1_biases,
            l1_weights,
            DEFAULT_FT_OUT,
            DEFAULT_L1_OUT,
        )?;

        let l2_biases =
            (0..DEFAULT_L2_OUT).map(|output| weights.l2_b[bucket * DEFAULT_L2_OUT + output]);
        let l2_weights = (0..DEFAULT_L2_OUT).flat_map(|output| {
            (0..l2_in).map(move |input| {
                weights.l2_w[bucket * DEFAULT_L2_OUT * l2_in + output * l2_in + input]
            })
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, DEFAULT_L2_OUT)?;

        let l3_biases = std::iter::once(weights.l3_b[bucket]);
        let l3_weights =
            (0..DEFAULT_L2_OUT).map(|input| weights.l3_w[bucket * DEFAULT_L2_OUT + input]);
        write_affine(writer, l3_biases, l3_weights, DEFAULT_L2_OUT, 1)?;
    }
    Ok(())
}

fn validate_weights(weights: &LayerStackWeights) -> io::Result<()> {
    let expected_feature_set = FeatureSet::HalfKaHmMerged.spec();
    if weights.feature_set != expected_feature_set {
        return invalid_input("feature set must be HalfKaHmMerged without extensions");
    }
    if weights.num_buckets != DEFAULT_NUM_BUCKETS {
        return invalid_input("LayerStack bucket count must be 9");
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not supported");
    }
    let l2_in = (DEFAULT_L1_OUT - 1) * 2;
    let lengths = [
        ("ft_b", weights.ft_b.len(), DEFAULT_FT_OUT),
        (
            "ft_w",
            weights.ft_w.len(),
            expected_feature_set.ft_in() * DEFAULT_FT_OUT,
        ),
        (
            "l1_b",
            weights.l1_b.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L1_OUT,
        ),
        (
            "l1_w",
            weights.l1_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L1_OUT * DEFAULT_FT_OUT,
        ),
        ("l1f_b", weights.l1f_b.len(), DEFAULT_L1_OUT),
        (
            "l1f_w",
            weights.l1f_w.len(),
            DEFAULT_FT_OUT * DEFAULT_L1_OUT,
        ),
        (
            "l2_b",
            weights.l2_b.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT,
        ),
        (
            "l2_w",
            weights.l2_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT * l2_in,
        ),
        ("l3_b", weights.l3_b.len(), DEFAULT_NUM_BUCKETS),
        (
            "l3_w",
            weights.l3_w.len(),
            DEFAULT_NUM_BUCKETS * DEFAULT_L2_OUT,
        ),
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
    Err(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
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

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_matches_yaneuraou_constants_and_architecture() {
        let mut output = Vec::new();
        write_u32(&mut output, YO_VERSION).unwrap();
        write_u32(&mut output, YO_TOP_HASH).unwrap();
        write_u32(&mut output, YO_ARCH.len() as u32).unwrap();
        output.extend_from_slice(YO_ARCH.as_bytes());
        write_u32(&mut output, YO_FT_HASH).unwrap();

        assert_eq!(&output[0..4], &0x7af3_2f16_u32.to_le_bytes());
        assert_eq!(&output[4..8], &1_008_745_266_u32.to_le_bytes());
        assert_eq!(&output[12..12 + YO_ARCH.len()], YO_ARCH.as_bytes());
        assert_eq!(
            &output[12 + YO_ARCH.len()..],
            &0x5f13_4ab8_u32.to_le_bytes()
        );
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
}
