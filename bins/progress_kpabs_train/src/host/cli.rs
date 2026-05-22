//! CLI 引数定義。
//!
//! bullet-shogi 上流 (`shogi_progress_kpabs_train_cuda.rs::Args`) のサブセット
//! を移植している。prefetch / multi-thread / checkpoint 等は本 binary の
//! スコープ外で未実装。

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "progress-kpabs-train")]
#[command(
    about = "KP-absolute progress trainer (cuda-oxide port of bullet-shogi shogi_progress_kpabs_train_cuda)"
)]
pub struct Args {
    /// Training data PSV file(s) (`.bin`). Separate multiple files with `,`.
    /// When omitted, `run_training` returns an error (effectively required).
    #[arg(long)]
    pub data: Option<String>,

    /// Output path for the trained progress.bin.
    #[arg(long)]
    pub output: PathBuf,

    /// Warm-start weights from an existing progress.bin.
    #[arg(long)]
    pub init_from: Option<PathBuf>,

    /// Mini-batch size per Adam step (in games).
    #[arg(long, default_value_t = 1024)]
    pub games_per_step: usize,

    /// Upper limit on the number of games to scan (0 = unlimited). When
    /// `--val-fraction` is enabled, this is the limit on the combined training +
    /// validation game count (both are split from the same scanned sequence).
    #[arg(long, default_value_t = 0)]
    pub max_games: usize,

    /// Fraction of games to set aside for held-out validation (0.0 = disabled,
    /// the default). For example, 0.05 reserves about 1/20 of all games for
    /// validation only and reports the loss on the games not used for training
    /// (val_loss) at the end of each epoch. The split is per game, so positions
    /// from the same game never mix between training and validation. Games are
    /// picked at a fixed interval, so if your data is grouped by tournament or
    /// time period, shuffle it beforehand. The valid range is 0.0..=0.5.
    #[arg(long, default_value_t = 0.0)]
    pub val_fraction: f32,

    /// Number of epochs.
    #[arg(long, default_value_t = 1)]
    pub epochs: usize,

    /// Learning rate (before lr_scale is applied).
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f32,

    /// LR scaling: `none` keeps it fixed, `sqrt` applies `lr *= sqrt(games_per_step)`.
    #[arg(long, default_value = "sqrt")]
    pub lr_scale: LrScaleMode,

    /// Step interval for log output. 0 suppresses sub-step logs (end of epoch only).
    #[arg(long, default_value_t = 100)]
    pub log_interval_steps: usize,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LrScaleMode {
    None,
    Sqrt,
}

impl Args {
    /// `lr_scale` を適用した実効 lr を返す。
    pub fn effective_lr(&self) -> f32 {
        match self.lr_scale {
            LrScaleMode::None => self.lr,
            LrScaleMode::Sqrt => self.lr * (self.games_per_step as f32).sqrt(),
        }
    }

    /// `--data` をカンマ分割して `Vec<PathBuf>` にする。`None` なら空 Vec。
    pub fn data_paths(&self) -> Vec<PathBuf> {
        match &self.data {
            None => Vec::new(),
            Some(s) => s
                .split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect(),
        }
    }

    /// `--val-fraction` から検証 game の抽出 stride を導出する。stride `N` は
    /// 「`N` game ごとに 1 個を検証へ回す」(= 全体の約 `1/N`)。`val_fraction`
    /// が 0 以下なら検証無効で `None`。
    ///
    /// 前提: `val_fraction` は呼び出し前に `0.0..=0.5` へ検証済みであること。
    /// この範囲では `1.0 / val_fraction >= 2.0` なので stride は必ず 2 以上に
    /// なる (範囲外の入力は `run_training` が reject する)。
    pub fn val_stride(&self) -> Option<u64> {
        if self.val_fraction <= 0.0 {
            None
        } else {
            Some((1.0 / self.val_fraction).round() as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Args;
    use clap::Parser;

    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["progress-kpabs-train", "--output", "/tmp/progress-test.bin"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("args should parse")
    }

    #[test]
    fn val_fraction_defaults_to_disabled() {
        assert_eq!(parse(&[]).val_fraction, 0.0);
        assert_eq!(parse(&[]).val_stride(), None);
    }

    #[test]
    fn val_stride_rounds_fraction_to_nearest_stride() {
        assert_eq!(parse(&["--val-fraction", "0.05"]).val_stride(), Some(20));
        assert_eq!(parse(&["--val-fraction", "0.1"]).val_stride(), Some(10));
        assert_eq!(parse(&["--val-fraction", "0.25"]).val_stride(), Some(4));
        assert_eq!(parse(&["--val-fraction", "0.5"]).val_stride(), Some(2));
    }
}
