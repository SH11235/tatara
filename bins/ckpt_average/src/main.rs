//! LayerStack の学習 checkpoint (`*.ckpt`) 複数個の FP32 master weight を加重平均し
//! (一様平均 / snapshot 列の EMA)、trainer と同じ export 経路で量子化 `.bin` を書く。

mod raw_ckpt;

use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use nnue_format::LayerStackWeights;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Equal weight for every checkpoint.
    Uniform,
    /// Exponential moving average over the checkpoints in the given order, starting from the first.
    Ema,
}

/// Average the FP32 master weights of LayerStack training checkpoints and export a quantised .bin.
///
/// Only the master weights are read; optimizer state is ignored. Checkpoints must share the
/// feature set, topology, bucket mode and QAT mode.
#[derive(Parser)]
struct Args {
    /// Training checkpoints (*.ckpt), ordered from oldest to newest.
    #[arg(required = true)]
    checkpoints: Vec<PathBuf>,
    /// Path for the exported .bin. An existing file is never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Averaging rule.
    #[arg(long, value_enum, default_value_t = Mode::Uniform)]
    mode: Mode,
    /// Decay applied per checkpoint for --mode ema (0 <= decay < 1).
    #[arg(long, value_name = "DECAY")]
    ema_decay: Option<f64>,
    /// fv_scale written into the architecture string. Pass the training run's --fv-scale;
    /// omit it when that run exported without one.
    #[arg(long, value_name = "N")]
    fv_scale: Option<i32>,
    /// Quantised .bin with the same architecture to compare the output against.
    #[arg(long, value_name = "BIN")]
    reference: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run(&Args::parse())
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let coefficients = match (args.mode, args.ema_decay) {
        (Mode::Uniform, None) => raw_ckpt::uniform_coefficients(args.checkpoints.len()),
        (Mode::Uniform, Some(_)) => return Err("--ema-decay requires --mode ema".into()),
        (Mode::Ema, Some(d)) if (0.0..1.0).contains(&d) => {
            raw_ckpt::ema_coefficients(args.checkpoints.len(), d)
        }
        (Mode::Ema, Some(d)) => return Err(format!("--ema-decay {d} must be in [0, 1)").into()),
        (Mode::Ema, None) => return Err("--mode ema requires --ema-decay".into()),
    };
    if args.output.exists() {
        return Err(format!("{} already exists", args.output.display()).into());
    }

    let paths: Vec<&Path> = args.checkpoints.iter().map(PathBuf::as_path).collect();
    let infos = paths
        .iter()
        .map(|p| raw_ckpt::read_ckpt_info(p).map_err(|e| format!("{}: {e}", p.display())))
        .collect::<Result<Vec<_>, _>>()?;
    for (path, info) in paths.iter().zip(&infos).skip(1) {
        if info.identity != infos[0].identity {
            return Err(format!(
                "{} does not match the first checkpoint's architecture ({:?} vs {:?})",
                path.display(),
                info.identity,
                infos[0].identity
            )
            .into());
        }
    }
    if args.mode == Mode::Ema && infos.windows(2).any(|w| w[0].superbatch >= w[1].superbatch) {
        return Err("--mode ema needs checkpoints in strictly increasing superbatch order".into());
    }
    let layout = raw_ckpt::layerstack_layout(&infos[0].identity)?;

    println!(
        "architecture: {} ft_out={} l1={} l2={} buckets={} bucket_mode={} ft_factorize={} qat_dense={}",
        infos[0].identity.feature_set_name,
        layout.ft_out,
        layout.l1_out,
        layout.l2_out,
        layout.num_buckets,
        infos[0].identity.bucket_mode,
        infos[0].identity.ft_factorize,
        infos[0].identity.qat_dense,
    );
    for ((path, info), c) in paths.iter().zip(&infos).zip(&coefficients) {
        println!(
            "input: superbatch={} steps={} version={} coefficient={c:.6} {}",
            info.superbatch,
            info.step_count,
            info.version,
            path.display()
        );
    }

    let groups = raw_ckpt::weighted_average(&paths, &infos, &coefficients)?;
    let weights = raw_ckpt::to_layerstack_weights(&layout, groups);
    write_new(&args.output, |w| weights.save_quantised(w, args.fv_scale))?;
    println!("output: {}", args.output.display());

    if let Some(reference) = &args.reference {
        let load = |p: &Path| -> io::Result<LayerStackWeights> {
            LayerStackWeights::load_quantised(
                &mut io::BufReader::new(fs::File::open(p)?),
                layout.export_spec(),
                layout.ft_out,
                layout.l1_out,
                layout.l2_out,
                layout.num_buckets,
            )
        };
        let ours = load(&args.output)?;
        let theirs = load(reference)?;
        for line in compare_quantised(&ours, &theirs) {
            println!("vs reference: {line}");
        }
    }
    Ok(())
}

/// `path` を新規作成して書く。書き込みに失敗したら作りかけの file を消す。
fn write_new(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<fs::File>) -> io::Result<()>,
) -> io::Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let result = write(&mut writer).and_then(|()| writer.flush());
    let result = result.and_then(|()| writer.get_ref().sync_all());
    if result.is_err() {
        drop(writer);
        let _ = fs::remove_file(path);
    }
    result
}

/// 量子化値 (load 後の逆量子化値) が異なる要素数を tensor ごとに数える。
fn compare_quantised(a: &LayerStackWeights, b: &LayerStackWeights) -> Vec<String> {
    let tensors: [(&str, &[f32], &[f32]); 10] = [
        ("ft_w", &a.ft_w, &b.ft_w),
        ("ft_b", &a.ft_b, &b.ft_b),
        ("l1_w", &a.l1_w, &b.l1_w),
        ("l1_b", &a.l1_b, &b.l1_b),
        ("l1_shared_weight", &a.l1_shared_weight, &b.l1_shared_weight),
        ("l1_shared_bias", &a.l1_shared_bias, &b.l1_shared_bias),
        ("l2_w", &a.l2_w, &b.l2_w),
        ("l2_b", &a.l2_b, &b.l2_b),
        ("l3_w", &a.l3_w, &b.l3_w),
        ("l3_b", &a.l3_b, &b.l3_b),
    ];
    tensors
        .iter()
        .map(|(name, x, y)| {
            let changed = x.iter().zip(*y).filter(|(p, q)| p != q).count();
            let max_diff = x
                .iter()
                .zip(*y)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            format!(
                "{name}: changed {changed}/{} ({:.4}%) max_abs_diff={max_diff:e}",
                x.len(),
                100.0 * changed as f64 / x.len().max(1) as f64
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use raw_ckpt::tests::{FT_OUT, L1, L2, NB, layout, pattern, temp_path, write_ckpt};

    const TOPO: [u64; 4] = [FT_OUT as u64, L1 as u64, L2 as u64, NB as u64];

    fn args(checkpoints: Vec<PathBuf>, output: PathBuf) -> Args {
        Args {
            checkpoints,
            output,
            mode: Mode::Uniform,
            ema_decay: None,
            fv_scale: Some(14),
            reference: None,
        }
    }

    #[test]
    fn single_checkpoint_export_matches_direct_quantisation() {
        let ckpt = temp_path("single.ckpt");
        let out = temp_path("single.bin");
        let _ = fs::remove_file(&out);
        write_ckpt(&ckpt, 20, TOPO, pattern(1.0));
        run(&args(vec![ckpt.clone()], out.clone())).unwrap();

        let lay = layout();
        let groups = (0..10)
            .map(|g| {
                let info = raw_ckpt::read_ckpt_info(&ckpt).unwrap();
                let len = info.identity.group_lens[g] as usize;
                (0..len).map(|i| pattern(1.0)(g, i)).collect()
            })
            .collect();
        let mut direct = Vec::new();
        raw_ckpt::to_layerstack_weights(&lay, groups)
            .save_quantised(&mut direct, Some(14))
            .unwrap();
        assert_eq!(fs::read(&out).unwrap(), direct);
    }

    #[test]
    fn ema_requires_increasing_superbatches_and_decay() {
        let a = temp_path("ema-a.ckpt");
        let b = temp_path("ema-b.ckpt");
        write_ckpt(&a, 40, TOPO, pattern(1.0));
        write_ckpt(&b, 20, TOPO, pattern(2.0));
        let out = temp_path("ema.bin");
        let _ = fs::remove_file(&out);
        let mut ema = args(vec![a, b], out.clone());
        ema.mode = Mode::Ema;
        assert!(run(&ema).unwrap_err().to_string().contains("--ema-decay"));
        ema.ema_decay = Some(0.8);
        assert!(run(&ema).unwrap_err().to_string().contains("increasing"));
        assert!(!out.exists());
    }

    #[test]
    fn rejects_mismatched_topology_and_existing_output() {
        let a = temp_path("mix-a.ckpt");
        let b = temp_path("mix-b.ckpt");
        write_ckpt(&a, 20, TOPO, pattern(1.0));
        write_ckpt(
            &b,
            40,
            [FT_OUT as u64, L1 as u64, L2 as u64, 3],
            pattern(1.0),
        );
        let out = temp_path("mix.bin");
        let _ = fs::remove_file(&out);
        let err = run(&args(vec![a.clone(), b], out.clone())).unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");

        fs::write(&out, b"keep").unwrap();
        assert!(run(&args(vec![a], out.clone())).is_err());
        assert_eq!(fs::read(&out).unwrap(), b"keep");
    }

    #[test]
    fn reference_comparison_counts_changed_elements() {
        let a = temp_path("ref-a.ckpt");
        let b = temp_path("ref-b.ckpt");
        write_ckpt(&a, 20, TOPO, pattern(1.0));
        write_ckpt(&b, 40, TOPO, pattern(1.0));
        let first = temp_path("ref-first.bin");
        let avg = temp_path("ref-avg.bin");
        for p in [&first, &avg] {
            let _ = fs::remove_file(p);
        }
        run(&args(vec![a.clone()], first.clone())).unwrap();
        let mut both = args(vec![a, b], avg);
        both.reference = Some(first);
        run(&both).unwrap();

        let base = raw_ckpt::to_layerstack_weights(
            &layout(),
            (0..10).map(|g| vec![0.5; raw_ckpt_group_len(g)]).collect(),
        );
        let mut shifted = base.clone();
        shifted.l2_w[3] += 1.0;
        let lines = compare_quantised(&shifted, &base);
        assert!(lines[6].starts_with("l2_w: changed 1/"), "{}", lines[6]);
        assert!(
            lines
                .iter()
                .enumerate()
                .all(|(i, l)| i == 6 || l.contains("changed 0/"))
        );
    }

    fn raw_ckpt_group_len(g: usize) -> usize {
        let path = temp_path("len.ckpt");
        write_ckpt(&path, 20, TOPO, pattern(1.0));
        raw_ckpt::read_ckpt_info(&path).unwrap().identity.group_lens[g] as usize
    }
}
