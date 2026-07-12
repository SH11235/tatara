//! 量子化 `.bin` (LayerStack) の weight 張り付き率を計測する診断ツール。
//!
//! i8 dense weight (scale QB) は学習時 clamp ±127/64 に張り付くと量子化端点
//! ±127 で飽和する。その割合を層別に数え、i16 FT weight (scale QA=127) の
//! 飽和域 ±32767/127 への接近も併せて報告する。
//!
//! 使い方:
//!   cargo run --release -p nnue-format --example clamp_stats -- \
//!     <net.bin> <feature-set> <ft_out> <l1> <l2> <num_buckets>

use nnue_format::layerstack_weights::{LayerStackWeights, QA, QB};
use shogi_features::FeatureSet;

fn stats(name: &str, w: &[f32], boundary: f32) {
    if w.is_empty() {
        println!("{name:>8}: (empty)");
        return;
    }
    let mut abs: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = abs.len();
    let pct = |q: f64| abs[((n as f64 - 1.0) * q) as usize];
    let eps = boundary * 1e-5;
    let at_boundary = abs.iter().filter(|v| **v >= boundary - eps).count();
    let near = abs.iter().filter(|v| **v >= boundary * 0.9).count();
    println!(
        "{name:>8}: n={n:>9}  max={:>9.4}  p99={:>8.4}  p99.9={:>8.4}  \
         >=90%bnd={:>8.4}%  ==bnd={:>8.4}%   (bnd={boundary:.4})",
        abs[n - 1],
        pct(0.99),
        pct(0.999),
        near as f64 / n as f64 * 100.0,
        at_boundary as f64 / n as f64 * 100.0,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: clamp_stats <net.bin> <feature-set> <ft_out> <l1> <l2> <num_buckets>".into(),
        );
    }
    let path = &args[1];
    let fs = FeatureSet::from_canonical_name(&args[2])
        .ok_or_else(|| format!("unknown feature set: {}", args[2]))?;
    let ft_out: usize = args[3].parse()?;
    let l1: usize = args[4].parse()?;
    let l2: usize = args[5].parse()?;
    let num_buckets: usize = args[6].parse()?;

    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let w = LayerStackWeights::load_quantised(&mut reader, fs.spec(), ft_out, l1, l2, num_buckets)?;

    let dense_bnd = i8::MAX as f32 / QB as f32; // 127/64: 学習時 clamp = i8 量子化端点
    let ft_bnd = i16::MAX as f32 / QA as f32; // 32767/127: i16 飽和域 (clamp 無し)

    println!("net: {path}");
    println!("--- i8 dense weights (train-time clamp ±127/64) ---");
    stats("l1_w", &w.l1_w, dense_bnd);
    stats("l1f_w", &w.l1f_w, dense_bnd);
    stats("l2_w", &w.l2_w, dense_bnd);
    stats("l3_w", &w.l3_w, dense_bnd);
    println!("--- i16 FT (clamp 無し、飽和境界 ±32767/127) ---");
    stats("ft_w", &w.ft_w, ft_bnd);
    stats("ft_b", &w.ft_b, ft_bnd);
    Ok(())
}
