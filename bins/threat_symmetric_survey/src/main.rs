//! `threat-symmetric-survey`: measure the active-edge reduction achievable by
//! the symmetric-pair filter over PSV data.
//!
//! Each active threat edge `attacker → target` is classified with
//! [`shogi_features::is_canonical_dead`]. A "dead" edge can be dropped without
//! losing information because its reverse edge is active in every position where
//! it is active. The reduction rate `X = dead / total` bounds the NPS gain of
//! the filter (empirical anchor: NPS gain ≈ 0.2..0.3 · X%).
//!
//! It also verifies the filter's safety invariant on real data: every dead edge
//! must have its reverse edge active in the same position. A violation is a
//! classifier bug and aborts with a non-zero exit code.
//!
//! ```bash
//! cargo run --release -p threat-symmetric-survey -- \
//!     --data <path/to/psv.bin> --samples 50000
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use shogi_features::{for_each_active_threat_edge, is_canonical_dead};
use shogi_format::PackedSfenValue;

const NUM_CLASSES: usize = 9;
const PLY_BINS: usize = 4;

#[derive(Parser, Debug)]
#[command(name = "threat-symmetric-survey")]
#[command(
    about = "Measure active-edge reduction of the symmetric-pair threat filter over PSV data"
)]
struct Args {
    /// PSV data file (`.bin`).
    #[arg(long)]
    data: PathBuf,

    /// Number of positions to read from the head of the file.
    #[arg(long, default_value_t = 50_000)]
    samples: usize,

    /// Starting record offset.
    #[arg(long, default_value_t = 0)]
    offset: u64,

    /// Read every N-th record (1 = dense scan).
    #[arg(long, default_value_t = 1)]
    stride: u64,

    /// Number of class-pair rows to print (sorted by dead count).
    #[arg(long, default_value_t = 20)]
    top: usize,
}

/// class discriminant を短い可読名に。`ThreatClass` は index 空間の discriminant
/// 0-8 を持つ (Pawn..Dragon)。
fn class_name(c: usize) -> &'static str {
    match c {
        0 => "Pawn",
        1 => "Lance",
        2 => "Knight",
        3 => "Silver",
        4 => "GoldLike",
        5 => "Bishop",
        6 => "Rook",
        7 => "Horse",
        8 => "Dragon",
        _ => "?",
    }
}

/// game_ply を 4 bin に割り当てる (1-40 / 41-80 / 81-120 / 121+)。
fn ply_bin(ply: u16) -> usize {
    match ply {
        0..=40 => 0,
        41..=80 => 1,
        81..=120 => 2,
        _ => 3,
    }
}

fn ply_bin_label(bin: usize) -> &'static str {
    match bin {
        0 => "1-40",
        1 => "41-80",
        2 => "81-120",
        _ => "121+",
    }
}

/// 集計結果。
#[derive(Default)]
struct Stats {
    positions: u64,
    total_edges: u64,
    dead_edges: u64,
    /// `[attacker_class][attacked_class]` の total / dead。
    pair_total: [[u64; NUM_CLASSES]; NUM_CLASSES],
    pair_dead: [[u64; NUM_CLASSES]; NUM_CLASSES],
    ply_total: [u64; PLY_BINS],
    ply_dead: [u64; PLY_BINS],
    /// property 違反 (dead なのに逆向き edge が inactive) の件数。
    violations: u64,
    /// 最初に見つかった違反の説明 (最大数件)。
    first_violations: Vec<String>,
}

fn read_record(file: &mut File) -> io::Result<Option<PackedSfenValue>> {
    let mut psv = PackedSfenValue::default();
    match file.read_exact(psv.as_bytes_mut()) {
        Ok(()) => Ok(Some(psv)),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

fn accumulate(stats: &mut Stats, psv: &PackedSfenValue) {
    let board = psv.decode();
    let bin = ply_bin(board.ply);

    // 同一局面の active directed edge を集める (逆向き edge の存在確認に使う)。
    let mut active: HashSet<(u8, u8)> = HashSet::new();
    for_each_active_threat_edge(&board, |e| {
        active.insert((e.from_sq.0, e.to_sq.0));
    });

    stats.positions += 1;
    for_each_active_threat_edge(&board, |e| {
        let ac = e.attacker_class as usize;
        let dc = e.attacked_class as usize;
        stats.total_edges += 1;
        stats.pair_total[ac][dc] += 1;
        stats.ply_total[bin] += 1;
        if is_canonical_dead(e) {
            stats.dead_edges += 1;
            stats.pair_dead[ac][dc] += 1;
            stats.ply_dead[bin] += 1;
            if !active.contains(&(e.to_sq.0, e.from_sq.0)) {
                stats.violations += 1;
                if stats.first_violations.len() < 8 {
                    stats.first_violations.push(format!(
                        "{}@{}→{}@{} ply={} (reverse inactive)",
                        class_name(ac),
                        e.from_sq.0,
                        class_name(dc),
                        e.to_sq.0,
                        board.ply,
                    ));
                }
            }
        }
    });
}

fn pct(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

fn report(stats: &Stats, top: usize) {
    println!("== threat symmetric-pair filter: active-edge reduction ==");
    println!("positions           : {}", stats.positions);
    println!("total active edges  : {}", stats.total_edges);
    println!("canonical-dead edges: {}", stats.dead_edges);
    println!(
        "X (dead / total)    : {:.3}%",
        pct(stats.dead_edges, stats.total_edges)
    );
    let per_pos = if stats.positions == 0 {
        0.0
    } else {
        stats.total_edges as f64 / stats.positions as f64
    };
    println!("edges / position    : {per_pos:.2}");

    println!("\n== class-pair breakdown (attacker → attacked, top {top} by dead) ==");
    let mut rows: Vec<(usize, usize, u64, u64)> = Vec::new();
    for ac in 0..NUM_CLASSES {
        for dc in 0..NUM_CLASSES {
            let total = stats.pair_total[ac][dc];
            let dead = stats.pair_dead[ac][dc];
            if total > 0 {
                rows.push((ac, dc, total, dead));
            }
        }
    }
    rows.sort_by(|a, b| b.3.cmp(&a.3).then(b.2.cmp(&a.2)));
    println!(
        "{:>10} {:>10} {:>12} {:>12} {:>9}",
        "attacker", "attacked", "total", "dead", "X%"
    );
    for &(ac, dc, total, dead) in rows.iter().take(top) {
        println!(
            "{:>10} {:>10} {:>12} {:>12} {:>8.2}%",
            class_name(ac),
            class_name(dc),
            total,
            dead,
            pct(dead, total),
        );
    }

    println!("\n== ply-bin breakdown ==");
    println!("{:>10} {:>12} {:>12} {:>9}", "ply", "total", "dead", "X%");
    for bin in 0..PLY_BINS {
        println!(
            "{:>10} {:>12} {:>12} {:>8.2}%",
            ply_bin_label(bin),
            stats.ply_total[bin],
            stats.ply_dead[bin],
            pct(stats.ply_dead[bin], stats.ply_total[bin]),
        );
    }

    println!("\n== property invariant (dead ⇒ reverse active) ==");
    if stats.violations == 0 {
        println!(
            "PASS: all {} dead edges have an active reverse",
            stats.dead_edges
        );
    } else {
        println!("FAIL: {} violations", stats.violations);
        for v in &stats.first_violations {
            println!("  {v}");
        }
    }

    // NPS 試算 (実測アンカー: gain ≈ 0.2..0.3 · X%)。
    let x = pct(stats.dead_edges, stats.total_edges);
    println!("\n== NPS estimate ==");
    println!(
        "X = {x:.2}%  →  NPS gain ≈ {:.2}%..{:.2}%",
        0.2 * x,
        0.3 * x
    );
    let gate = if x >= 15.0 {
        "GO (>= 15%)"
    } else if x < 10.0 {
        "standalone NO-GO (< 10%)"
    } else {
        "GRAY (10-15%)"
    };
    println!("gate0 verdict       : {gate}");
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.stride == 0 {
        return Err("--stride must be >= 1".into());
    }
    let record = size_of::<PackedSfenValue>() as u64;
    let mut file = File::open(&args.data)?;
    let total_records = file.metadata()?.len() / record;
    if args.offset >= total_records {
        return Err(format!("--offset {} >= file records {}", args.offset, total_records).into());
    }
    file.seek(SeekFrom::Start(args.offset * record))?;

    let mut stats = Stats::default();
    while stats.positions < args.samples as u64 {
        let Some(psv) = read_record(&mut file)? else {
            break;
        };
        accumulate(&mut stats, &psv);
        if args.stride > 1 {
            let skip = i64::try_from((args.stride - 1).saturating_mul(record)).unwrap_or(i64::MAX);
            file.seek(SeekFrom::Current(skip))?;
        }
    }

    report(&stats, args.top);

    if stats.violations > 0 {
        return Err(format!(
            "property invariant violated on {} edges (classifier bug)",
            stats.violations
        )
        .into());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
