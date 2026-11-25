use std::fs::File;
use std::io::{BufWriter, Write};

use nnue_format::layerstack_weights::{DEFAULT_NUM_BUCKETS, QA, QB};
use nnue_format::{LayerStackHead, LayerStackWeights};
use shogi_features::FeatureSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: dense_concat_fixture <output.bin>")?;
    let mut weights = LayerStackWeights::zeroed_with_head(
        FeatureSet::HalfKaHmMerged.spec(),
        2304,
        16,
        64,
        DEFAULT_NUM_BUCKETS,
        LayerStackHead::DenseConcat,
    );
    weights.bucket_spec = Some("ownking:3+handmajor4".parse()?);

    let mut state = 0xD3C0_A7A5u32;
    let mut fill = |values: &mut [f32], center: i32, radius: i32, scale: i32| {
        for value in values {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let integer = center + (state % (2 * radius + 1) as u32) as i32 - radius;
            *value = integer as f32 / scale as f32;
        }
    };
    fill(&mut weights.ft_w, 0, 5, QA);
    fill(&mut weights.ft_b, 68, 17, QA);
    fill(&mut weights.l1_w, 0, 5, QB);
    fill(&mut weights.l1_b, 0, 2800, QA * QB);
    fill(&mut weights.l2_w, 0, 12, QB);
    fill(&mut weights.l2_b, 0, 3800, QA * QB);
    fill(&mut weights.l3_w, 0, 16, QB);
    fill(&mut weights.l3_b, 0, 1000, QA * QB);

    let mut output = BufWriter::new(File::create_new(path)?);
    weights.save_quantised(&mut output, None)?;
    output.flush()?;
    Ok(())
}
