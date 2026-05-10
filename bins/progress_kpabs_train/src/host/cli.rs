//! CLI 引数定義。
//!
//! bullet-shogi 上流 (`shogi_progress_kpabs_train_cuda.rs::Args`) のサブセット
//! を移植。Stage 1-9 では「1 epoch 完走できる」「progress.bin が出力される」が
//! 受け入れ条件のため、prefetch / multi-thread / val split / checkpoint 等は
//! 範囲外として削っている (Stage 1-10 / Stage 2 で必要に応じて追加)。

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "progress-kpabs-train")]
#[command(
    about = "KP-absolute progress trainer (cuda-oxide port of bullet-shogi shogi_progress_kpabs_train_cuda)"
)]
pub struct Args {
    /// 学習データの PSV ファイル (`.bin`)。複数渡すには `,` 区切り。
    /// 引数省略時は `run_training` で error を返す (Stage 1-9 では必須)。
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

    /// 学習対象 game 数の上限 (0 = unlimited)。
    #[arg(long, default_value_t = 0)]
    pub max_games: usize,

    /// epoch 数。Stage 1-9 では 1 で 1 epoch 完走を確認する想定。
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
}
