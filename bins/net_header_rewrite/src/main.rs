use std::fs;
use std::path::PathBuf;

use clap::Parser;
use nnue_format::layerstack_weights::{LEGACY_NNUE_VERSION_BUCKETS9, NNUE_VERSION};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy)]
struct TokenRewrite {
    old_token: &'static str,
    new_token: &'static str,
    old_hash: u32,
    new_hash: u32,
}

// 旧 wire 名 (arch token / feature hash 入力文字列) はこの移行ツールだけが保持する。
// 現行の emit / load 経路には旧名を書かないこと。
const TOKEN_REWRITES: [TokenRewrite; 4] = [
    TokenRewrite {
        old_token: "E4=4xfixed",
        new_token: "EffectBucket=2x2fixed",
        old_hash: fnv1a32("e4-4-kingfixed"),
        new_hash: fnv1a32("effect-bucket-2x2-kingfixed"),
    },
    TokenRewrite {
        old_token: "E4=4xbucketed",
        new_token: "EffectBucket=2x2bucketed",
        old_hash: fnv1a32("e4-4-kingbucketed"),
        new_hash: fnv1a32("effect-bucket-2x2-kingbucketed"),
    },
    TokenRewrite {
        old_token: "E4=9xfixed",
        new_token: "EffectBucket=3x3fixed",
        old_hash: fnv1a32("e4-9-kingfixed"),
        new_hash: fnv1a32("effect-bucket-3x3-kingfixed"),
    },
    TokenRewrite {
        old_token: "E4=9xbucketed",
        new_token: "EffectBucket=3x3bucketed",
        old_hash: fnv1a32("e4-9-kingbucketed"),
        new_hash: fnv1a32("effect-bucket-3x3-kingbucketed"),
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = fs::read(&args.input)?;
    let output = rewrite_header(&input)?;
    fs::write(&args.output, output)?;
    Ok(())
}

fn rewrite_header(input: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if input.len() < 16 {
        return Err("input is too short for a LayerStack header".into());
    }

    let version = read_u32(input, 0)?;
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return Err(format!("unsupported NNUE version: {version:#x}").into());
    }

    let old_network_hash = read_u32(input, 4)?;
    let arch_len = read_u32(input, 8)? as usize;
    let arch_start = 12usize;
    let arch_end = arch_start
        .checked_add(arch_len)
        .ok_or("arch string length overflows usize")?;
    if arch_len == 0 || arch_end > input.len() {
        return Err(format!("invalid arch string length: {arch_len}").into());
    }
    let arch = std::str::from_utf8(&input[arch_start..arch_end])?;

    let rewrite = TOKEN_REWRITES
        .iter()
        .copied()
        .find(|entry| arch.contains(entry.old_token))
        .ok_or("input arch string does not contain an old effect bucket token")?;
    let matching = TOKEN_REWRITES
        .iter()
        .filter(|entry| arch.contains(entry.old_token))
        .count();
    if matching != 1 {
        return Err("input arch string contains multiple old effect bucket tokens".into());
    }

    let ft_hash_offset = if version == NNUE_VERSION {
        arch_end
            .checked_add(4)
            .ok_or("num_buckets offset overflows usize")?
    } else {
        arch_end
    };
    if ft_hash_offset + 4 > input.len() {
        return Err("input is too short for the feature hash field".into());
    }
    let old_ft_hash = read_u32(input, ft_hash_offset)?;
    let new_ft_hash = old_ft_hash ^ rewrite.old_hash ^ rewrite.new_hash;
    let new_network_hash = old_network_hash ^ old_ft_hash ^ new_ft_hash;
    let new_arch = arch.replacen(rewrite.old_token, rewrite.new_token, 1);
    let new_arch_bytes = new_arch.as_bytes();
    let new_arch_len = u32::try_from(new_arch_bytes.len())?;

    let mut out = Vec::with_capacity(input.len() + new_arch_bytes.len().saturating_sub(arch_len));
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&new_network_hash.to_le_bytes());
    out.extend_from_slice(&new_arch_len.to_le_bytes());
    out.extend_from_slice(new_arch_bytes);
    if version == NNUE_VERSION {
        out.extend_from_slice(&input[arch_end..arch_end + 4]);
    }
    out.extend_from_slice(&new_ft_hash.to_le_bytes());
    out.extend_from_slice(&input[ft_hash_offset + 4..]);
    Ok(out)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn std::error::Error>> {
    let end = offset.checked_add(4).ok_or("u32 offset overflows usize")?;
    if end > bytes.len() {
        return Err("input ended while reading a u32 field".into());
    }
    Ok(u32::from_le_bytes(bytes[offset..end].try_into()?))
}

const fn fnv1a32(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnue_format::LayerStackWeights;
    use nnue_format::layerstack_weights::DEFAULT_NUM_BUCKETS;
    use shogi_features::{EffectBucketConfig, FeatureSet};

    const TEST_FT_OUT: usize = 8;
    const TEST_L1_OUT: usize = 4;
    const TEST_L2_OUT: usize = 3;

    #[test]
    fn rewrites_old_header_and_preserves_tensor_payload() {
        let spec = FeatureSet::HalfKaHmMerged
            .spec()
            .with_effect_bucket_config(EffectBucketConfig::KINGFIXED_2X2);
        let net = LayerStackWeights::zeroed(
            spec,
            TEST_FT_OUT,
            TEST_L1_OUT,
            TEST_L2_OUT,
            DEFAULT_NUM_BUCKETS,
        );
        let mut new_bytes = Vec::new();
        net.save_quantised(&mut new_bytes).unwrap();

        let old_bytes = synthesize_old_header(&new_bytes, &TOKEN_REWRITES[0]);
        let old_payload = payload_after_feature_hash(&old_bytes);
        let rewritten = rewrite_header(&old_bytes).unwrap();
        let rewritten_payload = payload_after_feature_hash(&rewritten);
        assert_eq!(rewritten_payload, old_payload);
        assert_eq!(rewritten, new_bytes);

        let mut cursor = std::io::Cursor::new(rewritten);
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

    #[test]
    fn rejects_inputs_without_old_token() {
        let input = minimal_current_header("Features=X,Network=Y,", 0, 0);
        let err = rewrite_header(&input).expect_err("missing token must be rejected");
        assert!(err.to_string().contains("old effect bucket token"));
    }

    fn synthesize_old_header(new_bytes: &[u8], rewrite: &TokenRewrite) -> Vec<u8> {
        let version = read_u32(new_bytes, 0).unwrap();
        let network_hash = read_u32(new_bytes, 4).unwrap();
        let arch_len = read_u32(new_bytes, 8).unwrap() as usize;
        let arch_start = 12usize;
        let arch_end = arch_start + arch_len;
        let arch = std::str::from_utf8(&new_bytes[arch_start..arch_end]).unwrap();
        let old_arch = arch.replacen(rewrite.new_token, rewrite.old_token, 1);
        let ft_hash_offset = arch_end + 4;
        let ft_hash = read_u32(new_bytes, ft_hash_offset).unwrap();
        let old_ft_hash = ft_hash ^ rewrite.new_hash ^ rewrite.old_hash;
        let old_network_hash = network_hash ^ ft_hash ^ old_ft_hash;

        let mut out = Vec::new();
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&old_network_hash.to_le_bytes());
        out.extend_from_slice(&(old_arch.len() as u32).to_le_bytes());
        out.extend_from_slice(old_arch.as_bytes());
        out.extend_from_slice(&new_bytes[arch_end..arch_end + 4]);
        out.extend_from_slice(&old_ft_hash.to_le_bytes());
        out.extend_from_slice(&new_bytes[ft_hash_offset + 4..]);
        out
    }

    fn payload_after_feature_hash(bytes: &[u8]) -> &[u8] {
        let version = read_u32(bytes, 0).unwrap();
        let arch_len = read_u32(bytes, 8).unwrap() as usize;
        let arch_end = 12 + arch_len;
        let ft_hash_offset = if version == NNUE_VERSION {
            arch_end + 4
        } else {
            arch_end
        };
        &bytes[ft_hash_offset + 4..]
    }

    fn minimal_current_header(arch: &str, network_hash: u32, ft_hash: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        out.extend_from_slice(&network_hash.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        out.extend_from_slice(arch.as_bytes());
        out.extend_from_slice(&(DEFAULT_NUM_BUCKETS as u32).to_le_bytes());
        out.extend_from_slice(&ft_hash.to_le_bytes());
        out
    }
}
