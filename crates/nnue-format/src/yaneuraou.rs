//! YaneuraOu SFNNWithoutPsqt evaluation-file serialization.

use std::io::{self, Write};

use shogi_features::FeatureSet;

use crate::LayerStackWeights;
use crate::layerstack_weights::{QA, QB, write_leb128_tensor_i16};

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;

/// YaneuraOu SFNN が要求する KingRank9 LayerStack 数。
pub const YANEURAOU_LAYER_STACKS: usize = 9;

const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;

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

/// LayerStack weights を YaneuraOu SFNNWithoutPsqt 形式で書き出す。
///
/// feature set と各層次元は weights の shape から決定する。YaneuraOu SFNN が
/// 表現できない拡張 feature、PSQT、KingRank9 以外の bucket 数は reject する。
/// bucket routing mode 自体は weights に含まれないため、caller は学習 config 等から
/// KingRank9 であることを確認してから呼ぶ必要がある。
pub fn save_yaneuraou<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()> {
    let arch = architecture(weights)?;
    validate_weights(&arch, weights)?;

    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    let arch_string = arch_string(&arch);
    write_u32(
        writer,
        u32::try_from(arch_string.len()).expect("architecture string length fits in u32"),
    )?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_b, QA as f64))?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_w, QA as f64))?;

    for bucket in 0..YANEURAOU_LAYER_STACKS {
        write_u32(writer, YO_NETWORK_HASH)?;

        // factorizer 共有項は通常 export 前に L1 へ fold 済み。未 fold の weights を
        // caller が渡した場合にも同じ推論 weight になるよう加算する。
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

        write_affine(
            writer,
            std::iter::once(weights.l3_b[bucket]),
            (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]),
            l2_out,
            1,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct Architecture {
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
}

fn architecture(weights: &LayerStackWeights) -> io::Result<Architecture> {
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|feature_set| feature_set.spec() == weights.feature_set)
        .ok_or_else(|| invalid_input_err("feature set is not representable in YaneuraOu SFNN"))?;
    let ft_out = weights.ft_b.len();
    let l1_out = weights.l1f_b.len();
    let num_buckets = weights.num_buckets;
    if num_buckets != YANEURAOU_LAYER_STACKS {
        return invalid_input(format!(
            "YaneuraOu SFNN requires {} LayerStacks (KingRank9), but weights have {num_buckets} buckets",
            YANEURAOU_LAYER_STACKS
        ));
    }
    let l2_out = weights.l2_b.len().checked_div(num_buckets).unwrap_or(0);
    Ok(Architecture {
        feature_set,
        ft_out,
        l1_out,
        l2_out,
    })
}

fn arch_string(arch: &Architecture) -> String {
    let feature = YO_FEATURES
        .iter()
        .find(|feature| feature.feature_set == arch.feature_set)
        .expect("every FeatureSet has a YaneuraOu mapping");
    let input_size = arch.feature_set.spec().ft_in();
    let h1 = arch.l1_out - 1;
    let network = if arch.feature_set == FeatureSet::HalfKaHmMerged
        && arch.ft_out == 1536
        && arch.l1_out == 16
        && arch.l2_out == 32
    {
        "SFNN-1536".to_string()
    } else {
        format!(
            "SFNN_{}_{}_{}_{}_k3k3",
            feature.gen_key, arch.ft_out, h1, arch.l2_out
        )
        .to_ascii_uppercase()
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{}x2],Network={network}{{LayerStack={YANEURAOU_LAYER_STACKS}}}",
        feature.yo_name, arch.ft_out
    )
}

fn validate_weights(arch: &Architecture, weights: &LayerStackWeights) -> io::Result<()> {
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not representable in YaneuraOu SFNN");
    }
    if arch.ft_out == 0 || arch.ft_out > MAX_FT_OUT || !arch.ft_out.is_multiple_of(32) {
        return invalid_input(format!(
            "unsupported FT output dimension {} (expected a positive multiple of 32 up to {MAX_FT_OUT})",
            arch.ft_out
        ));
    }
    if arch.l1_out < 2 || arch.l1_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L1 output dimension {} (expected 2..={MAX_HIDDEN_DIM})",
            arch.l1_out
        ));
    }
    if arch.l2_out == 0 || arch.l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {} (expected 1..={MAX_HIDDEN_DIM})",
            arch.l2_out
        ));
    }
    let l2_in = (arch.l1_out - 1) * 2;
    let spec = arch.feature_set.spec();
    let lengths = [
        ("ft_b", weights.ft_b.len(), arch.ft_out),
        ("ft_w", weights.ft_w.len(), spec.ft_in() * arch.ft_out),
        (
            "l1_b",
            weights.l1_b.len(),
            YANEURAOU_LAYER_STACKS * arch.l1_out,
        ),
        (
            "l1_w",
            weights.l1_w.len(),
            YANEURAOU_LAYER_STACKS * arch.l1_out * arch.ft_out,
        ),
        ("l1f_b", weights.l1f_b.len(), arch.l1_out),
        ("l1f_w", weights.l1f_w.len(), arch.ft_out * arch.l1_out),
        (
            "l2_b",
            weights.l2_b.len(),
            YANEURAOU_LAYER_STACKS * arch.l2_out,
        ),
        ("l3_b", weights.l3_b.len(), YANEURAOU_LAYER_STACKS),
        (
            "l2_w",
            weights.l2_w.len(),
            YANEURAOU_LAYER_STACKS * arch.l2_out * l2_in,
        ),
        (
            "l3_w",
            weights.l3_w.len(),
            YANEURAOU_LAYER_STACKS * arch.l2_out,
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
                weights
                    .next()
                    .ok_or_else(|| invalid_input_err("affine weight iterator is short"))?
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

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_input_err(message))
}

fn invalid_input_err(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_architecture_string_matches_yaneuraou() {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            1536,
            16,
            32,
            YANEURAOU_LAYER_STACKS,
        );
        assert_eq!(
            arch_string(&architecture(&weights).unwrap()),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}"
        );
    }

    #[test]
    fn generated_architecture_names_match_yaneuraou_loader_contract() {
        let cases = [
            (
                FeatureSet::HalfKaHmMerged,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaHmMerged,
                512,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_HALFKAHM2_512_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKp,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->1536x2],Network=SFNN_HALFKP_1536_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaSplit,
                768,
                8,
                16,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA1(Friend)[138510->768x2],Network=SFNN_HALFKA1_768_7_16_K3K3{LayerStack=9}",
            ),
        ];

        for (feature_set, ft_out, l1_out, l2_out, expected) in cases {
            let weights = LayerStackWeights::zeroed(
                feature_set.spec(),
                ft_out,
                l1_out,
                l2_out,
                YANEURAOU_LAYER_STACKS,
            );
            assert_eq!(arch_string(&architecture(&weights).unwrap()), expected);
        }
    }

    #[test]
    fn rejects_non_kingrank9_shape() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 8);
        let error = save_yaneuraou(&mut Vec::new(), &weights).unwrap_err();
        assert!(error.to_string().contains("KingRank9"), "{error}");
    }

    #[test]
    fn direct_export_matches_tatara_reload_export_byte_for_byte() {
        let mut weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        );
        weights.ft_b[0] = 0.25;
        weights.ft_w[17] = -0.5;
        weights.l1_b[2] = 0.75;
        weights.l1_w[31] = -0.25;
        weights.l1f_b[1] = 0.25;
        weights.l1f_w[7] = -0.5;
        weights.l2_b[4] = 0.5;
        weights.l2_w[9] = -0.75;
        weights.l3_b[3] = 0.125;
        weights.l3_w[5] = -0.125;

        let mut direct = Vec::new();
        save_yaneuraou(&mut direct, &weights).unwrap();

        let mut tatara = Vec::new();
        weights.save_quantised(&mut tatara, Some(28)).unwrap();
        let reloaded = LayerStackWeights::load_quantised(
            &mut tatara.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        )
        .unwrap();
        let mut post_hoc = Vec::new();
        save_yaneuraou(&mut post_hoc, &reloaded).unwrap();

        assert_eq!(direct, post_hoc);
    }

    #[test]
    fn affine_weights_are_row_major_and_padded() {
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
}
