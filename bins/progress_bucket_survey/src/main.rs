use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use shogi_features::{BucketSpec, MAX_LAYERSTACK_BUCKETS, progress_kpabs::ProgressKpAbsModel};
use shogi_format::{PackedSfenValue, ShogiBoard};

const RECORD_BYTES: u64 = size_of::<PackedSfenValue>() as u64;

#[derive(Parser, Debug)]
#[command(about = "Survey one or two bucket definitions over samples spread across PSV files")]
struct Args {
    /// PSV files. Repeat the option or pass a comma-separated list.
    #[arg(long, value_delimiter = ',', required = true)]
    data: Vec<PathBuf>,

    /// Coefficient file for the first bucket definition.
    #[arg(
        long,
        required_unless_present = "bucket_spec",
        conflicts_with = "bucket_spec"
    )]
    progress: Option<PathBuf>,

    /// Number of progress buckets for the first definition (default: 8).
    #[arg(long, requires = "progress")]
    num_buckets: Option<usize>,

    /// Bucket spec for the first definition, for example ownking:2/3/7+oppking:2/3/7.
    #[arg(long)]
    bucket_spec: Option<BucketSpec>,

    /// Cross-tabulate with a second progress coefficient file.
    #[arg(long, conflicts_with = "compare_bucket_spec")]
    compare_progress: Option<PathBuf>,

    /// Number of progress buckets for the second definition (default: 8).
    #[arg(long, requires = "compare_progress")]
    compare_num_buckets: Option<usize>,

    /// Cross-tabulate with a second bucket spec.
    #[arg(long)]
    compare_bucket_spec: Option<BucketSpec>,

    /// Total number of positions to sample, allocated by file size.
    #[arg(long, default_value_t = 50_000)]
    samples: u64,

    /// Number of evenly spaced sample windows per file (1 reads from the offset).
    #[arg(long, default_value_t = 32)]
    chunks: u64,

    /// Read every Nth record within a sample window.
    #[arg(long, default_value_t = 1)]
    stride: u64,

    /// Starting record offset applied to each file.
    #[arg(long, default_value_t = 0)]
    offset: u64,

    /// Include a separate table for each file.
    #[arg(long)]
    per_pack: bool,

    /// Write tab-separated rows with counts and fractions instead of a display table.
    #[arg(long)]
    tsv: bool,
}

enum Definition {
    Progress(ProgressKpAbsModel, usize),
    Spec(BucketSpec),
}

impl Definition {
    fn from_args(
        progress: Option<&std::path::Path>,
        count: Option<usize>,
        spec: Option<BucketSpec>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if let Some(path) = progress {
            let count = count.unwrap_or(8);
            if !(1..=MAX_LAYERSTACK_BUCKETS).contains(&count) {
                return Err(format!(
                    "progress bucket count must be in [1, {MAX_LAYERSTACK_BUCKETS}]"
                )
                .into());
            }
            let model = ProgressKpAbsModel::from_bytes(&std::fs::read(path)?)
                .map_err(|err| format!("{}: {err}", path.display()))?;
            return Ok(Some(Self::Progress(model, count)));
        }
        Ok(spec.map(Self::Spec))
    }

    fn count(&self) -> usize {
        match self {
            Self::Progress(_, count) => *count,
            Self::Spec(spec) => spec.num_buckets(),
        }
    }

    fn bucket(&self, board: &ShogiBoard) -> usize {
        usize::from(match self {
            Self::Progress(model, count) => model.bucket_board(board, *count),
            Self::Spec(spec) => spec.bucket_board(board),
        })
    }
}

fn scan(
    file: &mut BufReader<File>,
    args: &Args,
    available: u64,
    samples: u64,
    mut count: impl FnMut(&ShogiBoard),
) -> io::Result<()> {
    let chunks = args.chunks.min(samples);
    let split_at = |total: u64, index: u64| {
        (u128::from(total) * u128::from(index) / u128::from(chunks)) as u64
    };
    let mut record = PackedSfenValue::default();
    for chunk in 0..chunks {
        let len = split_at(samples, chunk + 1) - split_at(samples, chunk);
        let unused = available - samples;
        let gap_start = split_at(unused, chunk);
        let gap_end = split_at(unused, chunk + 1);
        let first = if args.chunks == 1 {
            0
        } else {
            split_at(samples, chunk) + gap_start + (gap_end - gap_start) / 2
        };
        let start = args.offset + first * args.stride;
        file.seek(SeekFrom::Start(start * RECORD_BYTES))?;
        for index in 0..len {
            if args.stride > 1 && index > 0 {
                file.seek(SeekFrom::Start(
                    (start + index * args.stride) * RECORD_BYTES,
                ))?;
            }
            file.read_exact(record.as_bytes_mut())?;
            count(&record.decode());
        }
    }
    Ok(())
}

struct Histogram {
    cells: Vec<Vec<u64>>,
    compare: bool,
}

impl Histogram {
    fn new(a: &Definition, b: Option<&Definition>) -> Self {
        Self {
            cells: vec![vec![0; b.map_or(1, Definition::count)]; a.count()],
            compare: b.is_some(),
        }
    }

    fn write(&self, out: &mut impl Write, label: &str, scope: &str, tsv: bool) -> io::Result<()> {
        let total: u64 = self.cells.iter().flatten().sum();
        let fraction = |count| {
            if total == 0 {
                0.0
            } else {
                count as f64 / total as f64
            }
        };
        let columns = self.cells[0].len();
        if tsv {
            let label = label
                .replace('\\', "\\\\")
                .replace('\t', "\\t")
                .replace('\r', "\\r")
                .replace('\n', "\\n");
            let mut row = |a: &str, b: &str, count| {
                writeln!(
                    out,
                    "{label}\t{a}\t{b}\t{count}\t{:.9}\t{scope}",
                    fraction(count)
                )
            };
            for (a, counts) in self.cells.iter().enumerate() {
                if self.compare {
                    for (b, &count) in counts.iter().enumerate() {
                        row(&a.to_string(), &b.to_string(), count)?;
                    }
                }
                row(&a.to_string(), "all", counts.iter().sum())?;
            }
            if self.compare {
                for b in 0..columns {
                    row(
                        "all",
                        &b.to_string(),
                        self.cells.iter().map(|counts| counts[b]).sum(),
                    )?;
                }
            }
            return row("all", "all", total);
        }

        writeln!(out, "\n{label} ({total} positions)")?;
        if self.compare {
            writeln!(out, "Share of all sampled positions (%)")?;
            write!(out, "{:>7}", "A \\ B")?;
            for b in 0..columns {
                write!(out, "{b:>8}")?;
            }
            writeln!(out, "{:>8}", "All")?;
            for (a, counts) in self.cells.iter().enumerate() {
                write!(out, "{a:>7}")?;
                for &count in counts {
                    write!(out, "{:>8.2}", 100.0 * fraction(count))?;
                }
                writeln!(out, "{:>8.2}", 100.0 * fraction(counts.iter().sum()))?;
            }
            write!(out, "{:>7}", "All")?;
            for b in 0..columns {
                let count = self.cells.iter().map(|counts| counts[b]).sum();
                write!(out, "{:>8.2}", 100.0 * fraction(count))?;
            }
            writeln!(out, "{:>8.2}", 100.0 * fraction(total))?;
        } else {
            writeln!(out, "{:>7} {:>12} {:>9}", "Bucket", "Count", "Share (%)")?;
            for (a, counts) in self.cells.iter().enumerate() {
                writeln!(
                    out,
                    "{a:>7} {:>12} {:>9.2}",
                    counts[0],
                    100.0 * fraction(counts[0])
                )?;
            }
        }
        Ok(())
    }
}

fn run(args: Args, out: &mut impl Write) -> Result<(), Box<dyn std::error::Error>> {
    if args.samples == 0 || args.stride == 0 || args.chunks == 0 {
        return Err("--samples, --stride, and --chunks must be at least 1".into());
    }
    let a = Definition::from_args(args.progress.as_deref(), args.num_buckets, args.bucket_spec)?
        .expect("first bucket definition is required");
    let b = Definition::from_args(
        args.compare_progress.as_deref(),
        args.compare_num_buckets,
        args.compare_bucket_spec,
    )?;

    let mut files = Vec::with_capacity(args.data.len());
    for path in &args.data {
        let path = path
            .to_str()
            .map_or(path.as_path(), |s| Path::new(s.trim()));
        if path.as_os_str().is_empty() {
            continue;
        }
        let bytes = std::fs::metadata(path)
            .map_err(|err| format!("{}: {err}", path.display()))?
            .len();
        if !bytes.is_multiple_of(RECORD_BYTES) {
            return Err(format!(
                "{}: file size does not match whole PSV records",
                path.display()
            )
            .into());
        }
        let available = (bytes / RECORD_BYTES)
            .saturating_sub(args.offset)
            .div_ceil(args.stride);
        files.push((path, available));
    }
    let available: u128 = files.iter().map(|(_, len)| u128::from(*len)).sum();
    if available == 0 {
        return Err("no positions available in --data files".into());
    }
    let samples = u128::from(args.samples).min(available);
    let mut total = Histogram::new(&a, b.as_ref());
    if args.tsv {
        writeln!(out, "file\tbucket_a\tbucket_b\tcount\tfraction\tscope")?;
    }
    let mut cumulative = 0u128;
    let mut allocated = 0u128;
    for (path, len) in files {
        cumulative += u128::from(len);
        let next = samples * cumulative / available;
        let count = (next - allocated) as u64;
        allocated = next;
        let mut hist = Histogram::new(&a, b.as_ref());
        if count > 0 {
            let file = File::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
            scan(&mut BufReader::new(file), &args, len, count, |board| {
                hist.cells[a.bucket(board)]
                    [b.as_ref().map_or(0, |definition| definition.bucket(board))] += 1;
            })
            .map_err(|err| format!("{}: {err}", path.display()))?;
        }
        for (target, source) in total
            .cells
            .iter_mut()
            .flatten()
            .zip(hist.cells.iter().flatten())
        {
            *target += source;
        }
        if args.per_pack {
            hist.write(out, &path.display().to_string(), "pack", args.tsv)?;
        }
    }
    total.write(out, "total", "aggregate", args.tsv)?;
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse(), &mut io::stdout().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survey_accepts_whitespace_in_comma_separated_data_paths() {
        let data = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../crates/shogi-format/tests/data/sample.psv"
        );
        let run_survey = |paths: &[&str]| {
            let args = Args::try_parse_from(
                [
                    "progress-bucket-survey",
                    "--bucket-spec",
                    "handmajor3",
                    "--samples",
                    "4",
                    "--per-pack",
                    "--tsv",
                ]
                .into_iter()
                .chain(paths.iter().copied()),
            )
            .unwrap();
            let mut output = Vec::new();
            run(args, &mut output).unwrap();
            output
        };
        let path_list = format!(" {data}, \t{data} , ");
        assert_eq!(
            run_survey(&["--data", &path_list]),
            run_survey(&["--data", data, "--data", data])
        );
    }

    #[test]
    fn survey_samples_both_ends_and_cross_tabulates_progress_and_spec() {
        let records = include_bytes!("../../../crates/shogi-format/tests/data/sample.psv");
        let spec: BucketSpec = "ownking:2/3/7+oppking:2/3/7".parse().unwrap();
        let head = &records[..40];
        let mut record = PackedSfenValue::default();
        let mut bucket = |bytes: &[u8]| {
            record.as_bytes_mut().copy_from_slice(bytes);
            spec.bucket_board(&record.decode())
        };
        let head_bucket = bucket(head);
        let tail = records
            .chunks_exact(40)
            .find(|bytes| bucket(bytes) != head_bucket)
            .unwrap();
        let tail_bucket = bucket(tail);

        let root =
            std::env::temp_dir().join(format!("progress-bucket-survey-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let data = root.join("teacher.psv");
        let second_data = root.join("second.psv");
        let coeff = root.join("progress.bin");
        std::fs::write(&data, [head.repeat(4), tail.repeat(6)].concat()).unwrap();
        std::fs::copy(&data, &second_data).unwrap();
        std::fs::write(
            &coeff,
            vec![0; 8 * shogi_features::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS],
        )
        .unwrap();
        let args = Args::try_parse_from([
            "progress-bucket-survey",
            "--data",
            data.to_str().unwrap(),
            "--progress",
            coeff.to_str().unwrap(),
            "--compare-bucket-spec",
            "ownking:2/3/7+oppking:2/3/7",
            "--samples",
            "2",
            "--chunks",
            "2",
            "--tsv",
        ])
        .unwrap();
        let mut output = Vec::new();
        run(args, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains(&format!("total\t4\t{head_bucket}\t1\t0.500000000")));
        assert!(output.contains(&format!("total\t4\t{tail_bucket}\t1\t0.500000000")));
        assert!(output.contains("total\tall\tall\t2\t1.000000000"));

        for (flags, expected_bucket) in [(&[][..], tail_bucket), (&["--chunks", "1"], head_bucket)]
        {
            let args = Args::try_parse_from(
                [
                    "progress-bucket-survey",
                    "--data",
                    data.to_str().unwrap(),
                    "--data",
                    second_data.to_str().unwrap(),
                    "--bucket-spec",
                    "ownking:2/3/7+oppking:2/3/7",
                    "--samples",
                    "2",
                    "--tsv",
                ]
                .into_iter()
                .chain(flags.iter().copied()),
            )
            .unwrap();
            let mut output = Vec::new();
            run(args, &mut output).unwrap();
            let output = String::from_utf8(output).unwrap();
            assert!(
                output.contains(&format!("total\t{expected_bucket}\tall\t2\t1.000000000")),
                "{flags:?}: {output}"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
