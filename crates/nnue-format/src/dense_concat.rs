use crate::layerstack_weights::QB;
use crate::{LayerStackHead, LayerStackWeights};

const SQUARE_SCALE: f32 = 127.0 / 128.0;

struct Affine<'a> {
    weights: &'a [f32],
    biases: &'a [f32],
}

impl Affine<'_> {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .chunks_exact(input.len())
            .zip(self.biases)
            .map(|(row, &bias)| {
                row.iter()
                    .zip(input)
                    .fold(bias, |sum, (&w, &x)| sum + w * x)
            })
            .collect()
    }

    fn forward_quantised(&self, input: &[u8], biases: &[i32]) -> Vec<i32> {
        self.weights
            .chunks_exact(input.len())
            .zip(biases)
            .map(|(row, &bias)| {
                row.iter().zip(input).fold(bias, |sum, (&w, &x)| {
                    let w = (w * QB as f32).round() as i8;
                    sum + i32::from(w) * i32::from(x)
                })
            })
            .collect()
    }
}

fn layers(weights: &LayerStackWeights, bucket: usize) -> [Affine<'_>; 3] {
    assert_eq!(weights.head, LayerStackHead::DenseConcat);
    let select = |tensor: &[f32]| {
        let len = tensor.len() / weights.num_buckets;
        bucket * len..(bucket + 1) * len
    };
    [
        (&weights.l1_w, &weights.l1_b),
        (&weights.l2_w, &weights.l2_b),
        (&weights.l3_w, &weights.l3_b),
    ]
    .map(|(w, b)| Affine {
        weights: &w[select(w)],
        biases: &b[select(b)],
    })
}

fn activations(input: &[f32]) -> impl Iterator<Item = f32> + '_ {
    input
        .iter()
        .map(|&x| (x * x * SQUARE_SCALE).clamp(0.0, 1.0))
        .chain(input.iter().map(|&x| x.clamp(0.0, 1.0)))
}

fn activations_quantised(input: &[i32]) -> impl Iterator<Item = u8> + '_ {
    input
        .iter()
        .map(|&x| {
            let wide = i64::from(x);
            ((wide * wide) >> 19).clamp(0, 127) as u8
        })
        .chain(input.iter().map(|&x| (x >> 6).clamp(0, 127) as u8))
}

/// Evaluate the head from the concatenated post-FT input.
pub fn forward(weights: &LayerStackWeights, bucket: usize, input: &[f32]) -> f32 {
    let [first, second, output] = layers(weights, bucket);
    let mut l1 = first.forward(input);
    let width = l1.len();
    for (col, value) in l1.iter_mut().enumerate() {
        *value += weights.l1_shared_bias[col]
            + input
                .iter()
                .zip(weights.l1_shared_weight[col..].iter().step_by(width))
                .map(|(&x, &w)| x * w)
                .sum::<f32>();
    }
    let mut concat: Vec<_> = activations(&l1).collect();
    let l2 = second.forward(&concat);
    concat.extend(activations(&l2));
    output.forward(&concat)[0] + (l1[l1.len() - 2] - l1[l1.len() - 1])
}

/// Affine biases in file order, kept as integers so evaluation preserves all i32 bits.
#[derive(Default)]
pub struct QuantisedBiases(pub(crate) Vec<i32>);

/// Evaluate the head of a loaded quantised net. L1 shared weights are folded into each bucket.
pub fn forward_quantised(
    weights: &LayerStackWeights,
    biases: &QuantisedBiases,
    bucket: usize,
    input: &[u8],
) -> i32 {
    let [first, second, output] = layers(weights, bucket);
    let l1_width = first.biases.len();
    let l2_width = second.biases.len();
    let row_width = l1_width + l2_width + 1;
    let row = &biases.0[bucket * row_width..][..row_width];
    let l1 = first.forward_quantised(input, &row[..l1_width]);
    let mut concat: Vec<_> = activations_quantised(&l1).collect();
    let l2 = second.forward_quantised(&concat, &row[l1_width..l1_width + l2_width]);
    concat.extend(activations_quantised(&l2));
    output.forward_quantised(&concat, &row[l1_width + l2_width..])[0]
        + (l1[l1.len() - 2] - l1[l1.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layerstack_weights::{QA, compute_fc_hash, network_hash, read_layerstack_arch};
    use shogi_features::FeatureSet;

    #[test]
    fn integer_activation_squares_the_full_i32_range() {
        let actual: Vec<_> = activations_quantised(&[i32::MIN, -4096, 4096, i32::MAX]).collect();
        assert_eq!(actual, [127, 32, 32, 127, 0, 0, 64, 127]);
    }

    #[test]
    fn dense_concat_export_layout_and_float_agree_with_integer_forward() -> std::io::Result<()> {
        let spec = FeatureSet::HalfKaHmMerged.spec();
        let (ft, l1, l2, buckets) = (34, 16, 32, 2);
        let mut weights = LayerStackWeights::zeroed_with_head(
            spec,
            ft,
            l1,
            l2,
            buckets,
            LayerStackHead::DenseConcat,
        );
        for (i, bias) in weights.l1_b.iter_mut().enumerate() {
            *bias = (i as i32 % 13 - 8) as f32 / 8.0;
        }
        weights.l1_shared_bias.fill(0.125);
        for (i, weight) in weights.l1_w.iter_mut().enumerate() {
            *weight = (i as i32 % 5 - 2) as f32 / QB as f32;
        }
        for (i, weight) in weights.l2_w.iter_mut().enumerate() {
            *weight = (i as i32 % 7 - 3) as f32 / QB as f32;
        }
        for (i, bias) in weights.l2_b.iter_mut().enumerate() {
            *bias = (i as i32 % 11 - 5) as f32 / 7.0;
        }
        for (i, weight) in weights.l3_w.iter_mut().enumerate() {
            *weight = (i as i32 % 9 - 4) as f32 / QB as f32;
        }
        let mut bytes = Vec::new();
        weights.save_quantised(&mut bytes, Some(28))?;
        let arch = read_layerstack_arch(&mut bytes.as_slice())?;
        assert!(arch.ends_with(concat!(
            ",Head=DenseConcat,Network=AffineTransform[1<-96](DenseConcat[96](",
            "AffineTransform[32<-32](SqrClippedReLU[32](",
            "AffineTransform[16<-68](InputSlice[68(0:68)]))))),fv_scale=28"
        )));
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            network_hash(spec.feature_hash(), ft, l2) ^ 0x44434E43,
        );
        let bucket_bytes = 4 + l1 * (4 + 64) + l2 * (4 + 32) + 4 + 96;
        let first_bucket = bytes.len() - buckets * bucket_bytes;
        for bucket in 0..buckets {
            let start = first_bucket + bucket * bucket_bytes;
            assert_eq!(
                u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()),
                compute_fc_hash(ft, l2) ^ 0x44434E43,
            );
            let first_fc0_row_padding = start + 4 + 4 * l1 + ft;
            assert_eq!(
                &bytes[first_fc0_row_padding..start + 4 + 4 * l1 + 64],
                &[0; 30]
            );
        }

        let (loaded, biases) = LayerStackWeights::load_quantised_for_integer_eval(
            &mut bytes.as_slice(),
            spec,
            ft,
            l1,
            l2,
            buckets,
            false,
        )?;
        assert_eq!(loaded.head, LayerStackHead::DenseConcat);
        let mut reexported = Vec::new();
        loaded.save_quantised(&mut reexported, Some(28))?;
        assert_eq!(bytes, reexported);
        assert_eq!(loaded.l3_w, weights.l3_w);
        let mut max_diff = 0.0_f32;
        for position in 0..8 {
            let input: Vec<_> = (0..ft)
                .map(|i| ((13 * i + 29 * position) % 127) as u8)
                .collect();
            let float_input: Vec<_> = input.iter().map(|&x| f32::from(x) / QA as f32).collect();
            for bucket in 0..buckets {
                let float = forward(&weights, bucket, &float_input);
                let integer = forward_quantised(&loaded, &biases, bucket, &input);
                let diff = (float - integer as f32 / (QA * QB) as f32).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < 0.01,
                    "position {position}, bucket {bucket}, error {diff}"
                );
            }
        }
        eprintln!("maximum float / quantised output difference: {max_diff:.8}");
        let output_bias = first_bucket + 4 + l1 * (4 + 64) + l2 * (4 + 32);
        let baseline = forward_quantised(&loaded, &biases, 0, &[0; 34]);
        let large_bias = (1_i32 << 30) + 7;
        bytes[output_bias..output_bias + 4].copy_from_slice(&large_bias.to_le_bytes());
        let (large, bias) = LayerStackWeights::load_quantised_for_integer_eval(
            &mut bytes.as_slice(),
            spec,
            ft,
            l1,
            l2,
            buckets,
            false,
        )?;
        assert_eq!(
            forward_quantised(&large, &bias, 0, &[0; 34]),
            baseline + large_bias,
        );
        let marker = arch.find("Head=DenseConcat").unwrap() + 12 + "Head=".len();
        bytes[marker] = b'X';
        let error =
            LayerStackWeights::load_quantised(&mut bytes.as_slice(), spec, ft, l1, l2, buckets)
                .unwrap_err();
        assert!(error.to_string().contains("unknown LayerStack head"));
        Ok(())
    }
}
