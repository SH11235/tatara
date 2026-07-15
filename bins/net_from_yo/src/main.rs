use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    DEFAULT_FT_OUT, DEFAULT_L1_OUT, DEFAULT_L2_OUT, DEFAULT_NUM_BUCKETS, QA, QB, pad32,
    read_leb128_tensor_i16,
};
use shogi_features::FeatureSet;

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;
const YO_ARCH: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}";
const YO_ARCH_V2: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536-V2{LayerStack=9}";
const YO_LEB128_PREFIX: u8 = b'_';

#[derive(Parser)]
#[command(about = "Convert a YaneuraOu SFNN-1536 net to a tatara LayerStack net")]
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
    weights.save_quantised(&mut writer)?;
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
    if arch != YO_ARCH && arch != YO_ARCH_V2 {
        return invalid_data(format!(
            "unsupported YaneuraOu architecture: `{arch}`; expected SFNN-1536 or SFNN-1536-V2"
        ));
    }

    expect_u32(reader, YO_FT_HASH, "feature-transformer hash")?;
    consume_optional_leb128_prefix(reader)?;

    let feature_set = FeatureSet::HalfKaHmMerged.spec();
    let ft_b_i16 = read_leb128_tensor_i16(reader, Some(DEFAULT_FT_OUT))?;
    let ft_w_i16 = read_leb128_tensor_i16(reader, Some(feature_set.ft_in() * DEFAULT_FT_OUT))?;
    let qa = QA as f32;

    let mut weights = LayerStackWeights::zeroed(
        feature_set,
        DEFAULT_FT_OUT,
        DEFAULT_L1_OUT,
        DEFAULT_L2_OUT,
        DEFAULT_NUM_BUCKETS,
    );
    weights.ft_b = ft_b_i16
        .into_iter()
        .map(|value| value as f32 / qa)
        .collect();
    weights.ft_w = ft_w_i16
        .into_iter()
        .map(|value| value as f32 / qa)
        .collect();

    let l2_in = (DEFAULT_L1_OUT - 1) * 2;
    for bucket in 0..DEFAULT_NUM_BUCKETS {
        expect_u32(reader, YO_NETWORK_HASH, "LayerStack hash")?;

        let (biases, dense_weights) = read_affine(reader, DEFAULT_FT_OUT, DEFAULT_L1_OUT)?;
        let l1_b_start = bucket * DEFAULT_L1_OUT;
        weights.l1_b[l1_b_start..l1_b_start + DEFAULT_L1_OUT].copy_from_slice(&biases);
        let l1_w_start = bucket * DEFAULT_L1_OUT * DEFAULT_FT_OUT;
        weights.l1_w[l1_w_start..l1_w_start + dense_weights.len()].copy_from_slice(&dense_weights);

        let (biases, dense_weights) = read_affine(reader, l2_in, DEFAULT_L2_OUT)?;
        let l2_b_start = bucket * DEFAULT_L2_OUT;
        weights.l2_b[l2_b_start..l2_b_start + DEFAULT_L2_OUT].copy_from_slice(&biases);
        let l2_w_start = bucket * DEFAULT_L2_OUT * l2_in;
        weights.l2_w[l2_w_start..l2_w_start + dense_weights.len()].copy_from_slice(&dense_weights);

        let (biases, dense_weights) = read_affine(reader, DEFAULT_L2_OUT, 1)?;
        weights.l3_b[bucket] = biases[0];
        let l3_w_start = bucket * DEFAULT_L2_OUT;
        weights.l3_w[l3_w_start..l3_w_start + DEFAULT_L2_OUT].copy_from_slice(&dense_weights);
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
        return invalid_data("YaneuraOu input has trailing data after the expected 9 LayerStacks");
    }
    Ok(())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn invalid_data<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

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
                -4.0 / 64.0,
            ]
        );
    }

    #[test]
    fn optional_bulletou_leb128_prefix_is_consumed() {
        let mut prefixed = Cursor::new(b"_COMPRESSED_LEB128".to_vec());
        consume_optional_leb128_prefix(&mut prefixed).unwrap();
        assert_eq!(prefixed.position(), 1);

        let mut plain = Cursor::new(b"COMPRESSED_LEB128".to_vec());
        consume_optional_leb128_prefix(&mut plain).unwrap();
        assert_eq!(plain.position(), 0);
    }

    #[test]
    fn accepted_architectures_cover_both_network_names() {
        assert_ne!(YO_ARCH, YO_ARCH_V2);
        assert!(YO_ARCH.contains("Network=SFNN-1536{"));
        assert!(YO_ARCH_V2.contains("Network=SFNN-1536-V2{"));
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
