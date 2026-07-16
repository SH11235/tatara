use std::fs;
use std::path::PathBuf;

use clap::{ArgGroup, Parser};
use nnue_format::layerstack_weights::{LEB128_MAGIC, LEGACY_NNUE_VERSION_BUCKETS9, NNUE_VERSION};

const HEADER_LEN: usize = 12;
const MAX_ARCH_LEN: usize = 16_384;

#[derive(Parser)]
#[command(group(
    ArgGroup::new("operation")
        .required(true)
        .args(["fv_scale", "remove"])
))]
struct Args {
    /// LayerStack .bin to read.
    #[arg(long)]
    input: PathBuf,
    /// Path for the rewritten LayerStack .bin.
    #[arg(long)]
    output: PathBuf,
    /// Set fv_scale in the range 1..=128.
    #[arg(long, value_name = "N", value_parser = parse_fv_scale)]
    fv_scale: Option<u8>,
    /// Remove fv_scale from the architecture string.
    #[arg(long)]
    remove: bool,
}

#[derive(Clone, Copy)]
enum Rewrite {
    Set(u8),
    Remove,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let rewrite = match args.fv_scale {
        Some(value) => Rewrite::Set(value),
        None => Rewrite::Remove,
    };
    let input = fs::read(&args.input)?;
    let output = rewrite_fv_scale(&input, rewrite)?;
    fs::write(&args.output, output)?;
    Ok(())
}

fn parse_fv_scale(value: &str) -> Result<u8, String> {
    let value = value
        .parse::<i32>()
        .map_err(|_| "fv_scale must be an integer in the range 1..=128".to_string())?;
    u8::try_from(value)
        .ok()
        .filter(|value| (1..=128).contains(value))
        .ok_or_else(|| "fv_scale must be in the range 1..=128".to_string())
}

fn rewrite_fv_scale(input: &[u8], rewrite: Rewrite) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if input.len() < HEADER_LEN {
        return Err("input is too short for a LayerStack header".into());
    }

    let version = read_u32(input, 0)?;
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return Err(format!("unsupported LayerStack NNUE version: {version:#x}").into());
    }

    let arch_len = read_u32(input, 8)? as usize;
    let arch_end = HEADER_LEN
        .checked_add(arch_len)
        .ok_or("arch string length overflows usize")?;
    if arch_len == 0 || arch_len > MAX_ARCH_LEN || arch_end > input.len() {
        return Err(format!("invalid arch string length: {arch_len}").into());
    }
    let arch = std::str::from_utf8(&input[HEADER_LEN..arch_end])?;
    validate_layerstack(input, version, arch_end, arch)?;

    let new_arch = rewrite_arch(arch, rewrite)?;
    let new_arch_len = u32::try_from(new_arch.len())?;
    let mut output = Vec::with_capacity(
        input
            .len()
            .checked_add(new_arch.len())
            .and_then(|len| len.checked_sub(arch_len))
            .ok_or("output length overflows usize")?,
    );
    output.extend_from_slice(&input[..8]);
    output.extend_from_slice(&new_arch_len.to_le_bytes());
    output.extend_from_slice(new_arch.as_bytes());
    output.extend_from_slice(&input[arch_end..]);
    Ok(output)
}

fn validate_layerstack(
    input: &[u8],
    version: u32,
    arch_end: usize,
    arch: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !arch.starts_with("Features=")
        || !arch.contains(",Network=AffineTransform[")
        || !arch.contains("InputSlice[")
    {
        return Err("input architecture is not a LayerStack architecture string".into());
    }

    let fields_after_arch = if version == NNUE_VERSION { 8 } else { 4 };
    let leb128_offset = arch_end
        .checked_add(fields_after_arch)
        .ok_or("LayerStack payload offset overflows usize")?;
    let magic_end = leb128_offset
        .checked_add(LEB128_MAGIC.len())
        .ok_or("LayerStack payload offset overflows usize")?;
    if input.get(leb128_offset..magic_end) != Some(LEB128_MAGIC) {
        return Err("input does not contain a LayerStack feature-transformer payload".into());
    }
    Ok(())
}

fn rewrite_arch(arch: &str, rewrite: Rewrite) -> Result<String, Box<dyn std::error::Error>> {
    let fv_tokens = arch
        .split(',')
        .enumerate()
        .filter(|(_, token)| token.starts_with("fv_scale="))
        .collect::<Vec<_>>();
    if fv_tokens.len() > 1 {
        return Err("architecture string contains multiple fv_scale tokens".into());
    }
    if let Some((index, token)) = fv_tokens.first()
        && (*index + 1 != arch.split(',').count() || token["fv_scale=".len()..].is_empty())
    {
        return Err("fv_scale is not a canonical trailing architecture token".into());
    }

    let base = match fv_tokens.first() {
        Some((_, token)) => &arch[..arch.len() - token.len() - 1],
        None => arch,
    };
    match rewrite {
        Rewrite::Set(value) => Ok(format!("{base},fv_scale={value}")),
        Rewrite::Remove => Ok(base.to_string()),
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn std::error::Error>> {
    let end = offset.checked_add(4).ok_or("u32 offset overflows usize")?;
    let field = bytes
        .get(offset..end)
        .ok_or("input ended while reading a u32 field")?;
    Ok(u32::from_le_bytes(field.try_into()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use nnue_format::LayerStackWeights;
    use nnue_format::layerstack_weights::DEFAULT_NUM_BUCKETS;
    use shogi_features::FeatureSet;

    const ARCH_WITHOUT_SCALE: &str = "Features=HalfKaHmMerged(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-30](SqrClippedReLU[30](AffineTransform[16<-3072](InputSlice[3072(0:3072)])))))";
    const TEST_FT_OUT: usize = 8;
    const TEST_L1_OUT: usize = 4;
    const TEST_L2_OUT: usize = 3;

    #[test]
    fn real_layerstack_file_round_trips_set_and_remove() {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let net = LayerStackWeights::zeroed(
            spec,
            TEST_FT_OUT,
            TEST_L1_OUT,
            TEST_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );

        let mut without_scale = Vec::new();
        net.save_quantised(&mut without_scale, None).unwrap();
        let set = rewrite_fv_scale(&without_scale, Rewrite::Set(37)).unwrap();
        assert!(arch_str(&set).ends_with(",fv_scale=37"));
        load_real_file(&set, spec);
        assert_eq!(rewrite_fv_scale(&set, Rewrite::Set(37)).unwrap(), set);

        let removed = rewrite_fv_scale(&set, Rewrite::Remove).unwrap();
        assert_eq!(removed, without_scale);
        assert!(!arch_str(&removed).contains("fv_scale="));
        load_real_file(&removed, spec);
        assert_eq!(
            rewrite_fv_scale(&removed, Rewrite::Remove).unwrap(),
            removed
        );

        let mut with_scale = Vec::new();
        net.save_quantised(&mut with_scale, Some(28)).unwrap();
        let removed = rewrite_fv_scale(&with_scale, Rewrite::Remove).unwrap();
        assert!(!arch_str(&removed).contains("fv_scale="));
        load_real_file(&removed, spec);
        let restored = rewrite_fv_scale(&removed, Rewrite::Set(28)).unwrap();
        assert_eq!(restored, with_scale);
        load_real_file(&restored, spec);
    }

    #[test]
    fn sets_missing_fv_scale_and_preserves_all_other_fields() {
        let input = test_file(NNUE_VERSION, ARCH_WITHOUT_SCALE);
        let output = rewrite_fv_scale(&input, Rewrite::Set(37)).unwrap();

        assert_rewrite(
            &input,
            &output,
            &format!("{ARCH_WITHOUT_SCALE},fv_scale=37"),
        );
    }

    #[test]
    fn replaces_existing_fv_scale_and_preserves_all_other_fields() {
        let old_arch = format!("{ARCH_WITHOUT_SCALE},fv_scale=28");
        let input = test_file(NNUE_VERSION, &old_arch);
        let output = rewrite_fv_scale(&input, Rewrite::Set(128)).unwrap();

        assert_rewrite(
            &input,
            &output,
            &format!("{ARCH_WITHOUT_SCALE},fv_scale=128"),
        );
    }

    #[test]
    fn removes_fv_scale_and_preserves_all_other_fields() {
        let old_arch = format!("{ARCH_WITHOUT_SCALE},fv_scale=28");
        let input = test_file(LEGACY_NNUE_VERSION_BUCKETS9, &old_arch);
        let output = rewrite_fv_scale(&input, Rewrite::Remove).unwrap();

        assert_rewrite(&input, &output, ARCH_WITHOUT_SCALE);
    }

    #[test]
    fn setting_then_removing_restores_original_file() {
        let input = test_file(NNUE_VERSION, ARCH_WITHOUT_SCALE);
        let set = rewrite_fv_scale(&input, Rewrite::Set(41)).unwrap();
        let removed = rewrite_fv_scale(&set, Rewrite::Remove).unwrap();

        assert_eq!(removed, input);
    }

    #[test]
    fn cli_rejects_invalid_fv_scale_and_conflicting_operations() {
        for value in ["0", "-1", "129"] {
            assert!(
                Args::try_parse_from([
                    "net_fv_scale",
                    "--input",
                    "a.bin",
                    "--output",
                    "b.bin",
                    "--fv-scale",
                    value,
                ])
                .is_err(),
                "fv_scale={value} must be rejected"
            );
        }
        assert!(
            Args::try_parse_from([
                "net_fv_scale",
                "--input",
                "a.bin",
                "--output",
                "b.bin",
                "--fv-scale",
                "28",
                "--remove",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from(["net_fv_scale", "--input", "a.bin", "--output", "b.bin",])
                .is_err()
        );
    }

    #[test]
    fn rejects_non_layerstack_input() {
        let input = test_file(NNUE_VERSION, "Features=X,Network=Y");
        let error = rewrite_fv_scale(&input, Rewrite::Set(28)).unwrap_err();
        assert!(error.to_string().contains("not a LayerStack architecture"));
    }

    fn test_file(version: u32, arch: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0x1234_5678_u32.to_le_bytes());
        bytes.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        bytes.extend_from_slice(arch.as_bytes());
        if version == NNUE_VERSION {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
        }
        bytes.extend_from_slice(&0x9abc_def0_u32.to_le_bytes());
        bytes.extend_from_slice(LEB128_MAGIC);
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        bytes
    }

    fn assert_rewrite(input: &[u8], output: &[u8], expected_arch: &str) {
        let old_arch_len = read_u32(input, 8).unwrap() as usize;
        let new_arch_len = read_u32(output, 8).unwrap() as usize;
        assert_eq!(&output[..8], &input[..8]);
        assert_eq!(new_arch_len, expected_arch.len());
        assert_eq!(
            std::str::from_utf8(&output[HEADER_LEN..HEADER_LEN + new_arch_len]).unwrap(),
            expected_arch
        );
        assert_eq!(
            &output[HEADER_LEN + new_arch_len..],
            &input[HEADER_LEN + old_arch_len..]
        );
    }

    fn arch_str(bytes: &[u8]) -> &str {
        let arch_len = read_u32(bytes, 8).unwrap() as usize;
        std::str::from_utf8(&bytes[HEADER_LEN..HEADER_LEN + arch_len]).unwrap()
    }

    fn load_real_file(bytes: &[u8], spec: shogi_features::FeatureSetSpec) {
        let mut cursor = std::io::Cursor::new(bytes);
        LayerStackWeights::load_quantised(
            &mut cursor,
            spec,
            TEST_FT_OUT,
            TEST_L1_OUT,
            TEST_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        )
        .unwrap();
    }
}
