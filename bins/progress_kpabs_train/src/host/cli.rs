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
    /// 学習データの PSV ファイル (`.bin`)。複数渡すには `,` 区切り。
    /// 引数省略時は `run_training` で error を返す (実質必須)。
    #[arg(long)]
    pub data: Option<String>,

    /// 学習結果の progress.bin 出力先。
    #[arg(long)]
    pub output: PathBuf,

    /// 既存 progress.bin から weight を warm-start する。
    #[arg(long)]
    pub init_from: Option<PathBuf>,

    /// 1 Adam step あたりの mini-batch サイズ (games 単位)。
    #[arg(long, default_value_t = 1024)]
    pub games_per_step: usize,

    /// 走査する game 数の上限 (0 = unlimited)。`--val-fraction` 有効時は訓練と
    /// 検証を合算した game 数に対する上限 (両者は同じ走査列から分割するため)。
    #[arg(long, default_value_t = 0)]
    pub max_games: usize,

    /// held-out 検証に回す game の割合 (0.0 = 無効、既定)。例: 0.05 で全 game の
    /// 約 1/20 を検証専用にし、各 epoch 末に訓練に使わなかった game 上の loss
    /// (val_loss) を出力する。game 単位で分割するため同一 game の局面が訓練と
    /// 検証に混ざらない。固定間隔で抜き出すので、データが棋戦・時期で
    /// グループ化されている場合は事前にシャッフルしておくこと。指定可能範囲は
    /// 0.0..=0.5。
    #[arg(long, default_value_t = 0.0)]
    pub val_fraction: f32,

    /// epoch 数。
    #[arg(long, default_value_t = 1)]
    pub epochs: usize,

    /// 学習率 (lr_scale 適用前)。
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f32,

    /// lr scaling: `none` で固定、`sqrt` で `lr *= sqrt(games_per_step)`。
    #[arg(long, default_value = "sqrt")]
    pub lr_scale: LrScaleMode,

    /// step ごとの log 出力間隔。0 で sub-step log を suppress (epoch 末のみ)。
    #[arg(long, default_value_t = 100)]
    pub log_interval_steps: usize,

    /// CUDA device ordinal。
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
