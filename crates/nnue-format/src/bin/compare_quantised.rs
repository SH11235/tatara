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
use std::fs::File;
use std::io::BufReader;

use nnue_format::LayerStackWeights;
use shogi_features::FeatureSet;

/// `path` の LayerStack quantised checkpoint を読み込む。失敗時はどの file かを含むエラー。
///
/// 比較対象は production の `halfka-hm-merged` checkpoint を既定 FT 出力次元
/// (`DEFAULT_FT_OUT`) で想定し、その spec で load する (loader は arch / hash が
/// この feature set と FT 出力次元に一致するか検証する)。
fn load(path: &str) -> Result<LayerStackWeights, Box<dyn Error>> {
    let file = File::open(path).map_err(|e| format!("open `{path}`: {e}"))?;
    LayerStackWeights::load_quantised(
        &mut BufReader::new(file),
        FeatureSet::HalfKaHmMerged.spec(),
        nnue_format::layerstack_weights::DEFAULT_FT_OUT,
    )
    .map_err(|e| format!("parse `{path}` as LayerStack quantised NNUE: {e}").into())
}

/// tensor `a` / `b` の要素ごとの差分統計を 1 行で出力する。長さ不一致はエラー。
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
    // 全ゼロ tensor (LayerStack では l1f_w / l1f_b は load 後 0) は cosine 未定義。
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

    println!("a = {}", args[1]);
    println!("b = {}", args[2]);
    // 各行は自己記述的 (key=value)。tensor ごとに 1 行。
    // LayerStack の全 weight tensor。l1f_w / l1f_b は shared factorized 層で、
    // load_quantised 後は 0 (l1_w 側に畳み込まれている)。
    print_stats("ft_w", &a.ft_w, &b.ft_w)?;
    print_stats("ft_b", &a.ft_b, &b.ft_b)?;
    print_stats("l1_w", &a.l1_w, &b.l1_w)?;
    print_stats("l1_b", &a.l1_b, &b.l1_b)?;
    print_stats("l1f_w", &a.l1f_w, &b.l1f_w)?;
    print_stats("l1f_b", &a.l1f_b, &b.l1f_b)?;
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
