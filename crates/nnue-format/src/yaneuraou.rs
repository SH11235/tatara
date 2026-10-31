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

/// progress kp abs routing で受理する LayerStack 数の上限 (tatara `progresskpabs` と同じ範囲)。
pub const MAX_PROGRESS_KPABS_BUCKETS: usize = 9;

/// YaneuraOu SFNN の LayerStack 選択規則。Network 構造名の末尾 token に対応し、
/// weights 自体には記録されないため caller が学習 config 等から指定する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YaneuraOuRouting {
    /// 両玉の段 3x3 (`_K3K3`)。LayerStack は常に 9。
    KingRank9,
    /// KP 絶対値の進行度を N 等分した bucket (`_PROGRESS<N>`、N=1 は suffix なし)。
    /// YaneuraOu は進行度係数を評価ファイルと同じ directory の `progress.bin` から読む。
    ProgressKpAbs,
}

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

/// LayerStack weights を KingRank9 routing の YaneuraOu SFNNWithoutPsqt 形式で書き出す。
///
/// [`save_yaneuraou_with_routing`] に [`YaneuraOuRouting::KingRank9`] を渡すのと同じ。
pub fn save_yaneuraou<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()> {
    save_yaneuraou_with_routing(writer, weights, YaneuraOuRouting::KingRank9)
}

/// LayerStack weights を YaneuraOu SFNNWithoutPsqt 形式で書き出す。
///
/// feature set と各層次元は weights の shape から決定する。YaneuraOu SFNN が
/// 表現できない拡張 feature、PSQT、routing と合わない bucket 数は reject する。
/// bucket routing mode 自体は weights に含まれないため、caller は学習 config 等から
/// routing を確認してから渡す必要がある。
pub fn save_yaneuraou_with_routing<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    routing: YaneuraOuRouting,
) -> io::Result<()> {
    let arch = architecture(weights, routing)?;
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

    for bucket in 0..arch.num_buckets {
        write_u32(writer, YO_NETWORK_HASH)?;

        // L1 shared term は通常 export 前に L1 へ fold 済み。未 fold の weights を
        // caller が渡した場合にも同じ推論 weight になるよう加算する。
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1_shared_bias[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1_shared_weight[input * l1_out + output]
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
    num_buckets: usize,
    routing: YaneuraOuRouting,
}

fn architecture(
    weights: &LayerStackWeights,
    routing: YaneuraOuRouting,
) -> io::Result<Architecture> {
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|feature_set| feature_set.spec() == weights.feature_set)
        .ok_or_else(|| invalid_input_err("feature set is not representable in YaneuraOu SFNN"))?;
    let ft_out = weights.ft_b.len();
    let l1_out = weights.l1_shared_bias.len();
    let num_buckets = weights.num_buckets;
    match routing {
        YaneuraOuRouting::KingRank9 if num_buckets != YANEURAOU_LAYER_STACKS => {
            return invalid_input(format!(
                "YaneuraOu SFNN requires {YANEURAOU_LAYER_STACKS} LayerStacks (KingRank9), but weights have {num_buckets} buckets"
            ));
        }
        YaneuraOuRouting::ProgressKpAbs
            if num_buckets == 0 || num_buckets > MAX_PROGRESS_KPABS_BUCKETS =>
        {
            return invalid_input(format!(
                "progress kp abs routing supports 1..={MAX_PROGRESS_KPABS_BUCKETS} LayerStacks, but weights have {num_buckets} buckets"
            ));
        }
        _ => {}
    }
    let l2_out = weights.l2_b.len().checked_div(num_buckets).unwrap_or(0);
    Ok(Architecture {
        feature_set,
        ft_out,
        l1_out,
        l2_out,
        num_buckets,
        routing,
    })
}

fn arch_string(arch: &Architecture) -> String {
    let feature = YO_FEATURES
        .iter()
        .find(|feature| feature.feature_set == arch.feature_set)
        .expect("every FeatureSet has a YaneuraOu mapping");
    let input_size = arch.feature_set.spec().ft_in();
    let h1 = arch.l1_out - 1;
    // 生成器 (`nnue_arch_gen.py`) は layer stack token を `_` で連結し、bucket が 1 つの
    // ときは token を付けない。`SFNN-1536` は KingRank9 の既定構造に限る。
    let layer_stack_token = match arch.routing {
        YaneuraOuRouting::KingRank9 => "_k3k3".to_string(),
        YaneuraOuRouting::ProgressKpAbs if arch.num_buckets == 1 => String::new(),
        YaneuraOuRouting::ProgressKpAbs => format!("_progress{}", arch.num_buckets),
    };
    let network = if arch.routing == YaneuraOuRouting::KingRank9
        && arch.feature_set == FeatureSet::HalfKaHmMerged
        && arch.ft_out == 1536
        && arch.l1_out == 16
        && arch.l2_out == 32
    {
        "SFNN-1536".to_string()
    } else {
        format!(
            "SFNN_{}_{}_{}_{}{layer_stack_token}",
            feature.gen_key, arch.ft_out, h1, arch.l2_out
        )
        .to_ascii_uppercase()
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{}x2],Network={network}{{LayerStack={}}}",
        feature.yo_name, arch.ft_out, arch.num_buckets
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
        ("l1_b", weights.l1_b.len(), arch.num_buckets * arch.l1_out),
        (
            "l1_w",
            weights.l1_w.len(),
            arch.num_buckets * arch.l1_out * arch.ft_out,
        ),
        ("l1_shared_bias", weights.l1_shared_bias.len(), arch.l1_out),
        (
            "l1_shared_weight",
            weights.l1_shared_weight.len(),
            arch.ft_out * arch.l1_out,
        ),
        ("l2_b", weights.l2_b.len(), arch.num_buckets * arch.l2_out),
        ("l3_b", weights.l3_b.len(), arch.num_buckets),
        (
            "l2_w",
            weights.l2_w.len(),
            arch.num_buckets * arch.l2_out * l2_in,
        ),
        ("l3_w", weights.l3_w.len(), arch.num_buckets * arch.l2_out),
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
            arch_string(&architecture(&weights, YaneuraOuRouting::KingRank9).unwrap()),
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
            assert_eq!(
                arch_string(&architecture(&weights, YaneuraOuRouting::KingRank9).unwrap()),
                expected
            );
        }
    }

    #[test]
    fn rejects_non_kingrank9_shape() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 8);
        let error = save_yaneuraou(&mut Vec::new(), &weights).unwrap_err();
        assert!(error.to_string().contains("KingRank9"), "{error}");
    }

    #[test]
    fn progress_kpabs_architecture_names_follow_generator_tokens() {
        let cases = [
            (
                FeatureSet::HalfKaHmMerged,
                2304,
                16,
                64,
                8,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->2304x2],Network=SFNN_HALFKAHM2_2304_15_64_PROGRESS8{LayerStack=8}",
            ),
            (
                FeatureSet::HalfKaHmMerged,
                1536,
                16,
                32,
                9,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN_HALFKAHM2_1536_15_32_PROGRESS9{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKp,
                512,
                16,
                32,
                1,
                "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->512x2],Network=SFNN_HALFKP_512_15_32{LayerStack=1}",
            ),
        ];
        for (feature_set, ft_out, l1_out, l2_out, buckets, expected) in cases {
            let weights =
                LayerStackWeights::zeroed(feature_set.spec(), ft_out, l1_out, l2_out, buckets);
            let arch = architecture(&weights, YaneuraOuRouting::ProgressKpAbs).unwrap();
            assert_eq!(arch_string(&arch), expected);
        }
    }

    #[test]
    fn progress_kpabs_rejects_bucket_counts_outside_supported_range() {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            16,
            32,
            MAX_PROGRESS_KPABS_BUCKETS + 1,
        );
        let error =
            save_yaneuraou_with_routing(&mut Vec::new(), &weights, YaneuraOuRouting::ProgressKpAbs)
                .unwrap_err();
        assert!(error.to_string().contains("progress kp abs"), "{error}");
    }

    #[test]
    fn progress_kpabs_export_writes_one_network_per_bucket() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 4, 3, 5);
        let mut output = Vec::new();
        save_yaneuraou_with_routing(&mut output, &weights, YaneuraOuRouting::ProgressKpAbs)
            .unwrap();
        let hash = YO_NETWORK_HASH.to_le_bytes();
        // 全ゼロ weights では network hash の byte 列は各 LayerStack の先頭にしか現れない。
        let count = output.windows(4).filter(|window| *window == hash).count();
        assert_eq!(count, 5);
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
        weights.l1_shared_bias[1] = 0.25;
        weights.l1_shared_weight[7] = -0.5;
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
