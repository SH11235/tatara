//! 2 つの LayerStack quantised NNUE checkpoint (`.bin`) を比較する診断ツール。
//!
//! 両 file を [`LayerStackWeights::load_quantised`] で読んで f32 に dequant し、tensor ごとに
//! 差分統計 — 非ゼロ差の要素割合 / max・mean 絶対差 / RMSE / 各 L2 ノルム / cosine
//! 類似度 — を表で出力する。精度モード変更 (TF32・FP16 量子化など) が学習後の重みを
//! どれだけ動かしたかを、対局を回さずに確認するのに使う。
//!
//! ```text
//! cargo run -p nnue-format --release --bin compare_quantised -- <a.bin> <b.bin>
//! ```
//!
//! exit code: 正常 0、引数不正 2、file I/O や parse 失敗 1。

use std::error::Error;

use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    feature_spec_from_arch, read_layerstack_arch, stored_num_buckets,
};
use shogi_features::FeatureSet;

fn load(path: &str) -> Result<LayerStackWeights, Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    load_bytes(&bytes).map_err(|e| format!("parse {path} as LayerStack quantised NNUE: {e}").into())
}

fn load_bytes(mut bytes: &[u8]) -> Result<LayerStackWeights, Box<dyn Error>> {
    let mut header = bytes;
    let arch = read_layerstack_arch(&mut header)?;
    let features = arch
        .strip_prefix("Features=")
        .and_then(|arch| arch.split_once("(Friend)["))
        .ok_or("invalid LayerStack features")?;
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|set| set.spec().arch_feature_name() == features.0)
        .ok_or("unsupported LayerStack features")?;
    let ft_out = features
        .1
        .split_once("->")
        .and_then(|(_, rest)| rest.split_once("x2]"))
        .ok_or("invalid LayerStack FT dimensions")?
        .0
        .parse()?;
    let affine_outputs = arch
        .split("AffineTransform[")
        .skip(1)
        .map(|layer| {
            layer
                .split_once("<-")
                .ok_or("invalid LayerStack affine dimensions")?
                .0
                .parse::<usize>()
                .map_err(|_| "invalid LayerStack affine dimensions")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let &[1, l2_out, l1_out] = affine_outputs.as_slice() else {
        return Err("invalid LayerStack affine chain".into());
    };
    let with_psqt = arch.split(',').any(|token| token.starts_with("PSQT="));
    let num_buckets = stored_num_buckets(bytes)?;
    Ok(LayerStackWeights::load_quantised_with_psqt(
        &mut bytes,
        feature_spec_from_arch(feature_set, &arch, ft_out)?,
        ft_out,
        l1_out,
        l2_out,
        num_buckets,
        with_psqt,
    )?)
}

fn print_stats(name: &str, a: &[f32], b: &[f32]) -> Result<(), Box<dyn Error>> {
    if a.len() != b.len() {
        return Err(format!(
            "tensor `{name}`: length mismatch ({} vs {})",
            a.len(),
            b.len()
        )
        .into());
    }
    let n = a.len();
    if n == 0 {
        println!("{name:32} | (empty)");
        return Ok(());
    }

    let mut max_abs_diff = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut diff_count = 0_usize;
    let mut sum_a_sq = 0.0_f64;
    let mut sum_b_sq = 0.0_f64;
    let mut sum_ab = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        // 2 つの f32 を先に f64 化してから引く。f32 同士の減算は値が近いとき
        // cancellation で有効桁を失う (このツールは似た checkpoint の微小差を測る
        // 用途なので致命的)。f64(f32) は lossless、その差は厳密に表せる。
        let xf = x as f64;
        let yf = y as f64;
        let d = xf - yf;
        let abs_d = d.abs();
        max_abs_diff = max_abs_diff.max(abs_d);
        // 1e-9 は dequant 後の同値判定 epsilon (厳密一致なら d == 0)。
        if abs_d > 1e-9 {
            diff_count += 1;
        }
        sum_abs_diff += abs_d;
        sum_sq_diff += d * d;
        sum_a_sq += xf * xf;
        sum_b_sq += yf * yf;
        sum_ab += xf * yf;
    }

    let mean_abs_diff = sum_abs_diff / n as f64;
    let rmse = (sum_sq_diff / n as f64).sqrt();
    let norm_a = sum_a_sq.sqrt();
    let norm_b = sum_b_sq.sqrt();
    // 全ゼロ tensor (LayerStack では l1_shared_weight / l1_shared_bias は load 後 0) は cosine 未定義。
    // 両方ゼロなら一致とみなし 1.0、片方だけゼロなら 0.0 を出す。
    let cos_sim = if norm_a > 0.0 && norm_b > 0.0 {
        sum_ab / (norm_a * norm_b)
    } else if norm_a == 0.0 && norm_b == 0.0 {
        1.0
    } else {
        0.0
    };
    let diff_pct = 100.0 * diff_count as f64 / n as f64;

    println!(
        "{name:32} | n={n:>10} | diff={diff_pct:>6.2}% | max_abs={max_abs_diff:.6e} | \
         mean_abs={mean_abs_diff:.6e} | rmse={rmse:.6e} | \
         ||a||={norm_a:.4e} ||b||={norm_b:.4e} | cos={cos_sim:.10}"
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: compare_quantised <a.bin> <b.bin>");
        std::process::exit(2);
    }
    let a = load(&args[1])?;
    let b = load(&args[2])?;
    if a.head != b.head {
        return Err("LayerStack heads differ".into());
    }
    if a.bucket_spec != b.bucket_spec
        || a.num_buckets != b.num_buckets
        || a.feature_set != b.feature_set
    {
        return Err("LayerStack feature sets or bucket routing differ".into());
    }
    match (&a.psqt_w, &b.psqt_w) {
        (Some(a), Some(b)) => print_stats("psqt_w", a, b)?,
        (None, None) => {}
        _ => return Err("LayerStack PSQT layouts differ".into()),
    }

    println!("a = {}", args[1]);
    println!("b = {}", args[2]);
    // 各行は自己記述的 (key=value)。tensor ごとに 1 行。
    // LayerStack の全 weight tensor。l1_shared_weight / l1_shared_bias は L1 shared term で、
    // load_quantised 後は 0 (l1_w 側に畳み込まれている)。
    print_stats("ft_w", &a.ft_w, &b.ft_w)?;
    print_stats("ft_b", &a.ft_b, &b.ft_b)?;
    print_stats("l1_w", &a.l1_w, &b.l1_w)?;
    print_stats("l1_b", &a.l1_b, &b.l1_b)?;
    print_stats("l1_shared_weight", &a.l1_shared_weight, &b.l1_shared_weight)?;
    print_stats("l1_shared_bias", &a.l1_shared_bias, &b.l1_shared_bias)?;
    print_stats("l2_w", &a.l2_w, &b.l2_w)?;
    print_stats("l2_b", &a.l2_b, &b.l2_b)?;
    print_stats("l3_w", &a.l3_w, &b.l3_w)?;
    print_stats("l3_b", &a.l3_b, &b.l3_b)?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("compare_quantised: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnue_format::LayerStackHead;
    use shogi_features::{EffectBucketConfig, ThreatProfile};

    #[test]
    fn loads_dense_concat_with_feature_modifiers() -> Result<(), Box<dyn Error>> {
        let base = FeatureSet::HalfKaHmMerged.spec();
        let ft_out = 8;
        for spec in [
            base.with_threat_profile(ThreatProfile::Full),
            base.with_threat_profile(ThreatProfile::CrossSide)
                .with_threat_width(4),
            base.with_effect_bucket_config(EffectBucketConfig::KINGBUCKETED_3X3),
        ] {
            let weights = LayerStackWeights::zeroed_with_head(
                spec,
                ft_out,
                16,
                32,
                1,
                LayerStackHead::DenseConcat,
            );
            let mut bytes = Vec::new();
            weights.save_quantised(&mut bytes, None)?;
            let loaded = load_bytes(&bytes)?;
            assert_eq!(loaded.head, weights.head);
            assert_eq!(loaded.feature_set, spec);
            assert_eq!(loaded.ft_w.len(), weights.ft_w.len());
        }
        Ok(())
    }
}
