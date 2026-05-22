//! `progress-bucket-survey`: survey the progress-bucket distribution of a
//! `progress.bin` over PSV data.
//!
//! `progress-kpabs-train` produces `progress.bin` (KP-absolute progress
//! coefficients) that the LayerStack architecture uses to route each position
//! to an output bucket. This tool loads a `progress.bin`, assigns sampled PSV
//! positions to their progress8kpabs buckets, and prints the resulting
//! histogram — a quick way to check the buckets are not badly skewed.
//!
//! ```bash
//! cargo run --release -p progress-bucket-survey -- \
//!     --data <path/to/psv.bin> \
//!     --progress output/progress/<run-name>.e5.bin \
//!     --samples 200000
//! ```

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use shogi_features::ShogiProgressKPAbs;
use shogi_format::PackedSfenValue;

#[derive(Parser, Debug)]
#[command(name = "progress-bucket-survey")]
#[command(about = "Survey the progress8kpabs bucket distribution of a progress.bin over PSV data")]
struct Args {
    /// PSV data files (`.bin`). Pass several as a comma-separated list.
    #[arg(long)]
    data: String,

    /// progress.bin produced by progress-kpabs-train.
    #[arg(long)]
    progress: PathBuf,

    /// Number of positions to sample in total (across all --data files).
    #[arg(long, default_value_t = 50_000)]
    samples: usize,

    /// Read every N-th record (1 = dense scan).
    #[arg(long, default_value_t = 1)]
    stride: u64,

    /// Starting record offset applied to each --data file.
    #[arg(long, default_value_t = 0)]
    offset: u64,

    /// Also print a per-file histogram, not just the combined total.
    #[arg(long)]
    per_pack: bool,
}

/// 1 PSV ファイルから最大 `max` 局面をサンプリングする。`offset` レコード目から
/// 始め、`stride` レコードごとに 1 件読む。ファイル末尾 / レコード途中の EOF は
/// そこで打ち切る (末尾の半端バイトは無視)。EOF 以外の I/O エラーは伝播する。
fn read_samples(
    path: &PathBuf,
    offset: u64,
    stride: u64,
    max: usize,
) -> io::Result<Vec<PackedSfenValue>> {
    let record = size_of::<PackedSfenValue>() as u64;
    let mut file = File::open(path)?;
    let total_records = file.metadata()?.len() / record;
    let mut out = Vec::new();
    if offset >= total_records {
        return Ok(out);
    }

    file.seek(SeekFrom::Start(offset * record))?;
    while out.len() < max {
        let mut psv = PackedSfenValue::default();
        match file.read_exact(psv.as_bytes_mut()) {
            Ok(()) => {}
            // ファイル末尾 (レコード途中の EOF 含む) はそこで打ち切る。
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            // EOF 以外の I/O エラーは呼び出し元へ伝播する。
            Err(e) => return Err(e),
        }
        out.push(psv);
        if stride > 1 {
            // `(stride-1) * record` バイト先へ進める。極端な --stride でも u64
            // overflow しないよう saturating で計算し、SeekFrom::Current の i64
            // へ clamp する。EOF を越えた seek は成功し (次の read_exact が EOF を
            // 返す)、`seek` が Err を返すのは本物の I/O エラーなので伝播する。
            let skip = i64::try_from((stride - 1).saturating_mul(record)).unwrap_or(i64::MAX);
            file.seek(SeekFrom::Current(skip))?;
        }
    }
    Ok(out)
}

/// 最多 bucket の index と占有率 (%) を返す。空ヒストグラムでは `(0, 0.0)`。
fn top_bucket(hist: &[u64]) -> (usize, f64) {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return (0, 0.0);
    }
    let mut top = 0usize;
    for (i, &count) in hist.iter().enumerate() {
        if count > hist[top] {
            top = i;
        }
    }
    (top, 100.0 * hist[top] as f64 / total as f64)
}

fn print_hist(label: &str, hist: &[u64]) {
    let total: u64 = hist.iter().sum();
    println!("\n== {label} ==");
    if total == 0 {
        println!("(no positions)");
        return;
    }
    for (i, &count) in hist.iter().enumerate() {
        let pct = 100.0 * count as f64 / total as f64;
        println!("bucket {i}: {count:>10}  ({pct:>6.2}%)");
    }
    let (top, share) = top_bucket(hist);
    println!("total {total}, top bucket {top} ({share:.2}%)");
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.stride == 0 {
        return Err("--stride must be >= 1".into());
    }
    let data_paths: Vec<PathBuf> = args
        .data
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if data_paths.is_empty() {
        return Err("--data is required (comma-separated PSV files)".into());
    }

    // progress.bin をロード。`ShogiProgressKPAbs` は重みをプロセス global に
    // 持つため 1 プロセス 1 model (epoch 比較は本ツールを複数回実行する)。
    let kpabs = ShogiProgressKPAbs::load_from_bin(&args.progress)?;

    let n_buckets = ShogiProgressKPAbs::BUCKETS;
    let mut total_hist = vec![0u64; n_buckets];
    let mut grand_total = 0usize;
    let mut remaining = args.samples;

    for path in &data_paths {
        if remaining == 0 {
            break;
        }
        let samples = read_samples(path, args.offset, args.stride, remaining)?;
        remaining -= samples.len();
        grand_total += samples.len();

        let mut pack_hist = vec![0u64; n_buckets];
        for psv in &samples {
            pack_hist[kpabs.bucket(psv) as usize] += 1;
        }
        for (b, &count) in pack_hist.iter().enumerate() {
            total_hist[b] += count;
        }

        println!("loaded {} positions from {}", samples.len(), path.display());
        if args.per_pack {
            print_hist(&format!("per-pack: {}", path.display()), &pack_hist);
        }
    }

    if grand_total == 0 {
        return Err("no positions read from --data files".into());
    }
    print_hist("progress8kpabs bucket distribution (total)", &total_hist);
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::top_bucket;

    #[test]
    fn top_bucket_picks_the_largest_with_share() {
        let (idx, share) = top_bucket(&[10, 70, 20]);
        assert_eq!(idx, 1);
        assert!((share - 70.0).abs() < 1e-9);
    }

    #[test]
    fn top_bucket_empty_histogram_is_zero() {
        assert_eq!(top_bucket(&[0, 0, 0]), (0, 0.0));
    }

    #[test]
    fn top_bucket_ties_keep_the_first() {
        let (idx, _) = top_bucket(&[50, 50, 0]);
        assert_eq!(idx, 0);
    }
}
