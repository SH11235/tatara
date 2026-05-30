//! Learning rate / WDL lambda scheduling。
//!
//! superbatch loop が `superbatch` / `batch` index ごとに `LrScheduler::lr` /
//! `WdlScheduler::blend` を呼び、optimizer kernel (`adamw_step` / `radam_step`)
//! と loss kernel (`loss_wdl` / `loss_wrm`) の `lr` / `lambda` 引数として渡す
//! host-side state を提供する。
//!
//! ANSI 色付き terminal 出力は持たず、`std::fmt::Display` で plain string を
//! 返す形に統一する (色付けが必要なら呼び出し側で行う)。
//!
//! 両 trait に `+ 'static` を要求する: trainer state を `Arc` 共有 / thread
//! spawn する想定で、borrow を持つ scheduler は許さない設計。

use std::f32::consts::PI;
use std::fmt::{Debug, Display};

// =============================================================================
// LrScheduler trait + 実装
// =============================================================================

/// Learning rate scheduling。`superbatch` / `batch` index から f32 learning
/// rate を返す。
pub trait LrScheduler: Clone + Debug + Display + Send + Sync + 'static {
    /// 現在の batch / superbatch に対する learning rate を返す。
    /// 多くの scheduler は `batch` に依存しない (Warmup のみ参照)。
    fn lr(&self, batch: usize, superbatch: usize) -> f32;

    /// curve が終端値に到達する superbatch (horizon)。終端を持つ schedule
    /// (linear / cosine / exponential decay の `final_superbatch`、one-cycle の
    /// `total_superbatch`) は `Some` を、horizon を持たない schedule
    /// (constant / step / drop) は `None` を返す。caller は resume 用 checkpoint に
    /// 保存し、再開時に curve をその superbatch に固定するのに使う。
    fn horizon(&self) -> Option<usize> {
        None
    }
}

/// 一定の learning rate。
#[derive(Clone, Debug)]
pub struct ConstantLR {
    pub value: f32,
}

impl LrScheduler for ConstantLR {
    fn lr(&self, _batch: usize, _superbatch: usize) -> f32 {
        self.value
    }
}

impl Display for ConstantLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "constant {}", self.value)
    }
}

/// superbatch `drop` 経過後に `gamma` で 1 度だけ係数倍する。
#[derive(Clone, Debug)]
pub struct DropLR {
    pub start: f32,
    pub gamma: f32,
    pub drop: usize,
}

impl LrScheduler for DropLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch > self.drop {
            self.start * self.gamma
        } else {
            self.start
        }
    }
}

impl Display for DropLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start {} gamma {} drop at {} superbatches",
            self.start, self.gamma, self.drop
        )
    }
}

/// `step` superbatch ごとに `gamma` 係数倍する。
#[derive(Clone, Debug)]
pub struct StepLR {
    pub start: f32,
    pub gamma: f32,
    pub step: usize,
}

impl LrScheduler for StepLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        // saturating_sub(1) で superbatch = 0 でも安全に 0 step 扱い。
        let steps = superbatch.saturating_sub(1) / self.step;
        self.start * self.gamma.powi(steps as i32)
    }
}

impl Display for StepLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start {} gamma {} drop every {} superbatches",
            self.start, self.gamma, self.step
        )
    }
}

/// `final_superbatch` までに `initial_lr` から `final_lr` まで線形減衰。
#[derive(Clone, Debug)]
pub struct LinearDecayLR {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub final_superbatch: usize,
}

impl LrScheduler for LinearDecayLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }
        let lambda = superbatch as f32 / self.final_superbatch as f32;
        self.initial_lr + lambda * (self.final_lr - self.initial_lr)
    }

    fn horizon(&self) -> Option<usize> {
        Some(self.final_superbatch)
    }
}

impl Display for LinearDecayLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start at {} and linearly decay to {} at superbatch {}",
            self.initial_lr, self.final_lr, self.final_superbatch
        )
    }
}

/// cos taper で `final_superbatch` までに `initial_lr` → `final_lr`。
#[derive(Clone, Debug)]
pub struct CosineDecayLR {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub final_superbatch: usize,
}

impl LrScheduler for CosineDecayLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }
        let progress = superbatch as f32 / self.final_superbatch as f32;
        let lambda = 1.0 - 0.5 * (1.0 + (PI * progress).cos());
        self.initial_lr + lambda * (self.final_lr - self.initial_lr)
    }

    fn horizon(&self) -> Option<usize> {
        Some(self.final_superbatch)
    }
}

impl Display for CosineDecayLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start at {} and cosine decay to {} at superbatch {}",
            self.initial_lr, self.final_lr, self.final_superbatch
        )
    }
}

/// 指数減衰で `final_superbatch` までに `initial_lr` → `final_lr`。
#[derive(Clone, Debug)]
pub struct ExponentialDecayLR {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub final_superbatch: usize,
}

impl LrScheduler for ExponentialDecayLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }
        let lambda = superbatch as f32 / self.final_superbatch as f32;
        self.initial_lr * (self.final_lr / self.initial_lr).powf(lambda)
    }

    fn horizon(&self) -> Option<usize> {
        Some(self.final_superbatch)
    }
}

impl Display for ExponentialDecayLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "start at {} and exponentially decay to {} at superbatch {}",
            self.initial_lr, self.final_lr, self.final_superbatch
        )
    }
}

/// 1cycle policy (Smith, "super-convergence")。学習全体 `total_superbatch` を
/// warmup と anneal の 2 相に分け、`initial_lr` → `max_lr` → `final_lr` と山なりに
/// 動かす。両相とも half-cosine 補間 (`superbatch <= warmup_superbatch` で
/// `initial_lr` → `max_lr`、以降 `total_superbatch` までに `max_lr` → `final_lr`)。
/// 補間の形は PyTorch `OneCycleLR` の `anneal_strategy='cos'` と同じだが、進捗は
/// optimizer step ではなく **superbatch 単位**で測り、`progress = superbatch /
/// 境界 superbatch` とする (同 module の [`LinearDecayLR`] / [`CosineDecayLR`] と
/// 同じ刻み方で、`superbatch` は学習ループで 1..=total を渡す)。よって 1 番目の
/// superbatch は `initial_lr` ちょうどではなく 1/`warmup_superbatch` だけ進んだ点に
/// なる (warmup が十分長ければ無視できる)。
///
/// `LrScheduler::lr` は総 superbatch 数を受け取らないため、進捗の分母となる
/// `warmup_superbatch` / `total_superbatch` を field に保持する。
#[derive(Clone, Debug)]
pub struct OneCycleLR {
    pub initial_lr: f32,
    pub max_lr: f32,
    pub final_lr: f32,
    pub warmup_superbatch: usize,
    pub total_superbatch: usize,
}

impl OneCycleLR {
    /// PyTorch `OneCycleLR` と同じ導出で構築する。`initial_lr = max_lr /
    /// div_factor`、`final_lr = initial_lr / final_div_factor`、warmup 境界は
    /// `round(warmup_pct * total_superbatch)`。
    pub fn new(
        max_lr: f32,
        warmup_pct: f32,
        div_factor: f32,
        final_div_factor: f32,
        total_superbatch: usize,
    ) -> Self {
        let initial_lr = max_lr / div_factor;
        let final_lr = initial_lr / final_div_factor;
        let warmup_superbatch = (warmup_pct * total_superbatch as f32)
            .round()
            .clamp(0.0, total_superbatch as f32) as usize;
        Self {
            initial_lr,
            max_lr,
            final_lr,
            warmup_superbatch,
            total_superbatch,
        }
    }
}

/// half-cosine 補間。`p = 0` で `start`、`p = 1` で `end` (両端で傾き 0)。
fn cosine_interp(start: f32, end: f32, p: f32) -> f32 {
    end + 0.5 * (start - end) * (1.0 + (PI * p).cos())
}

impl LrScheduler for OneCycleLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch <= self.warmup_superbatch {
            // warmup: initial_lr → max_lr。warmup_superbatch=0 (warmup_pct=0)
            // のときはこの枝に superbatch=0 のみ入りうるので denom を 1 で守る。
            let denom = self.warmup_superbatch.max(1) as f32;
            cosine_interp(self.initial_lr, self.max_lr, superbatch as f32 / denom)
        } else if superbatch >= self.total_superbatch {
            self.final_lr
        } else {
            // anneal: max_lr → final_lr。
            let denom = (self.total_superbatch - self.warmup_superbatch).max(1) as f32;
            let p = (superbatch - self.warmup_superbatch) as f32 / denom;
            cosine_interp(self.max_lr, self.final_lr, p)
        }
    }

    fn horizon(&self) -> Option<usize> {
        Some(self.total_superbatch)
    }
}

impl Display for OneCycleLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "one-cycle {} → max {} (warmup to superbatch {}) → cosine anneal to {} at superbatch {}",
            self.initial_lr,
            self.max_lr,
            self.warmup_superbatch,
            self.final_lr,
            self.total_superbatch
        )
    }
}

/// `warmup_batches` 期間 (superbatch=1 内) で sub-scheduler の lr を warmup。
#[derive(Clone, Debug)]
pub struct WarmupLR<LR: LrScheduler> {
    pub inner: LR,
    pub warmup_batches: usize,
}

impl<LR: LrScheduler> LrScheduler for WarmupLR<LR> {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let base_lr = self.inner.lr(batch, superbatch);
        // 学習開始時 (superbatch=1) の batch < warmup_batches でのみ warmup
        // interp (`base_lr / (warmup_batches - batch)`)、それ以外は base_lr。
        if superbatch == 1 && batch < self.warmup_batches {
            base_lr / (self.warmup_batches - batch) as f32
        } else {
            base_lr
        }
    }

    fn horizon(&self) -> Option<usize> {
        // warmup wrapper は horizon を変えない (batch 単位の起動時 warmup のみ)。
        self.inner.horizon()
    }
}

impl<LR: LrScheduler> Display for WarmupLR<LR> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, warmup over {} batches",
            self.inner, self.warmup_batches
        )
    }
}

/// `first_scheduler_final_superbatch` で `first` → `second` に切り替え。
#[derive(Clone, Debug)]
pub struct SequenceLR<First: LrScheduler, Second: LrScheduler> {
    pub first: First,
    pub second: Second,
    pub first_scheduler_final_superbatch: usize,
}

impl<First: LrScheduler, Second: LrScheduler> LrScheduler for SequenceLR<First, Second> {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let midpoint = self.first_scheduler_final_superbatch;
        if superbatch <= midpoint {
            self.first.lr(batch, superbatch)
        } else {
            self.second.lr(batch, superbatch - midpoint)
        }
    }
}

impl<First: LrScheduler, Second: LrScheduler> Display for SequenceLR<First, Second> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, then after {} superbatches, {}",
            self.first, self.first_scheduler_final_superbatch, self.second
        )
    }
}

/// runtime selection 用 enum wrapper (CLI の `--lr-schedule` から生成する想定)。
/// `Warmup` variant は任意の非 Warmup scheduler を [`WarmupLR`] で包む (`Box` で
/// 再帰型のサイズを有限化する)。
#[derive(Clone, Debug)]
pub enum LrSchedulerEnum {
    Constant(ConstantLR),
    Step(StepLR),
    Drop(DropLR),
    LinearDecay(LinearDecayLR),
    CosineDecay(CosineDecayLR),
    ExponentialDecay(ExponentialDecayLR),
    OneCycle(OneCycleLR),
    Warmup(Box<WarmupLR<LrSchedulerEnum>>),
}

impl LrSchedulerEnum {
    /// `self` を `warmup_batches` の [`WarmupLR`] で包む。
    pub fn with_warmup(self, warmup_batches: usize) -> Self {
        Self::Warmup(Box::new(WarmupLR {
            inner: self,
            warmup_batches,
        }))
    }
}

impl LrScheduler for LrSchedulerEnum {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        match self {
            Self::Constant(s) => s.lr(batch, superbatch),
            Self::Step(s) => s.lr(batch, superbatch),
            Self::Drop(s) => s.lr(batch, superbatch),
            Self::LinearDecay(s) => s.lr(batch, superbatch),
            Self::CosineDecay(s) => s.lr(batch, superbatch),
            Self::ExponentialDecay(s) => s.lr(batch, superbatch),
            Self::OneCycle(s) => s.lr(batch, superbatch),
            Self::Warmup(s) => s.lr(batch, superbatch),
        }
    }

    fn horizon(&self) -> Option<usize> {
        match self {
            Self::Constant(s) => s.horizon(),
            Self::Step(s) => s.horizon(),
            Self::Drop(s) => s.horizon(),
            Self::LinearDecay(s) => s.horizon(),
            Self::CosineDecay(s) => s.horizon(),
            Self::ExponentialDecay(s) => s.horizon(),
            Self::OneCycle(s) => s.horizon(),
            Self::Warmup(s) => s.horizon(),
        }
    }
}

impl Display for LrSchedulerEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(s) => Display::fmt(s, f),
            Self::Step(s) => Display::fmt(s, f),
            Self::Drop(s) => Display::fmt(s, f),
            Self::LinearDecay(s) => Display::fmt(s, f),
            Self::CosineDecay(s) => Display::fmt(s, f),
            Self::ExponentialDecay(s) => Display::fmt(s, f),
            Self::OneCycle(s) => Display::fmt(s, f),
            Self::Warmup(s) => Display::fmt(s, f),
        }
    }
}

// =============================================================================
// WdlScheduler trait + 実装
// =============================================================================

/// WDL lambda scheduling。`superbatch` / `batch` / `max` から f32 lambda を返す。
/// loss kernel (`loss_wdl` / `loss_wrm`) の `lambda` 引数として渡される。
pub trait WdlScheduler: Clone + Debug + Display + Send + Sync + 'static {
    /// 現在の batch / superbatch (max = 総 superbatch 数) に対する WDL lambda。
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32;
}

/// 一定の WDL lambda。
#[derive(Clone, Debug)]
pub struct ConstantWDL {
    pub value: f32,
}

impl WdlScheduler for ConstantWDL {
    fn blend(&self, _batch: usize, _superbatch: usize, _max: usize) -> f32 {
        self.value
    }
}

impl Display for ConstantWDL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "constant {}", self.value)
    }
}

/// `start` から `end` に線形 taper。
#[derive(Clone, Debug)]
pub struct LinearWDL {
    pub start: f32,
    pub end: f32,
}

impl WdlScheduler for LinearWDL {
    fn blend(&self, _batch: usize, superbatch: usize, max: usize) -> f32 {
        // `(max - 1)` で正規化、`max == 1` のとき 0 除算回避のため `.max(1)`。
        let grad = (self.end - self.start) / (max - 1).max(1) as f32;
        self.start + grad * (superbatch - 1) as f32
    }
}

impl Display for LinearWDL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "linear taper start {} end {}", self.start, self.end)
    }
}

/// `warmup_batches` 期間 (superbatch=1 内) で sub-scheduler の lambda を warmup。
#[derive(Clone, Debug)]
pub struct WarmupWDL<W: WdlScheduler> {
    pub inner: W,
    pub warmup_batches: usize,
}

impl<W: WdlScheduler> WdlScheduler for WarmupWDL<W> {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        let base_wdl = self.inner.blend(batch, superbatch, max);
        if superbatch == 1 && batch < self.warmup_batches {
            base_wdl / (self.warmup_batches - batch) as f32
        } else {
            base_wdl
        }
    }
}

impl<W: WdlScheduler> Display for WarmupWDL<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, warmup over {} batches",
            self.inner, self.warmup_batches
        )
    }
}

/// `first_scheduler_final_superbatch` で `first` → `second` に切り替え。
#[derive(Clone, Debug)]
pub struct SequenceWDL<First: WdlScheduler, Second: WdlScheduler> {
    pub first: First,
    pub second: Second,
    pub first_scheduler_final_superbatch: usize,
}

impl<First: WdlScheduler, Second: WdlScheduler> WdlScheduler for SequenceWDL<First, Second> {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        let midpoint = self.first_scheduler_final_superbatch;
        if superbatch <= midpoint {
            self.first.blend(batch, superbatch, midpoint)
        } else {
            self.second
                .blend(batch, superbatch - midpoint, max - midpoint)
        }
    }
}

impl<First: WdlScheduler, Second: WdlScheduler> Display for SequenceWDL<First, Second> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, then after {} superbatches, {}",
            self.first, self.first_scheduler_final_superbatch, self.second
        )
    }
}

/// runtime selection 用 enum wrapper (CLI から `--wdl const 0.5` /
/// `--wdl linear 0.0 0.5` 等で生成する想定)。
#[derive(Clone, Debug)]
pub enum WdlSchedulerEnum {
    Constant(ConstantWDL),
    Linear(LinearWDL),
}

impl WdlSchedulerEnum {
    /// constant WDL scheduler を構築。
    pub fn constant(value: f32) -> Self {
        Self::Constant(ConstantWDL { value })
    }

    /// linear taper の WDL scheduler を構築。
    pub fn linear(start: f32, end: f32) -> Self {
        Self::Linear(LinearWDL { start, end })
    }
}

impl WdlScheduler for WdlSchedulerEnum {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        match self {
            Self::Constant(s) => s.blend(batch, superbatch, max),
            Self::Linear(s) => s.blend(batch, superbatch, max),
        }
    }
}

impl Display for WdlSchedulerEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(s) => Display::fmt(s, f),
            Self::Linear(s) => Display::fmt(s, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn constant_lr_is_invariant() {
        let lr = ConstantLR { value: 1e-3 };
        assert_eq!(lr.lr(0, 1), 1e-3);
        assert_eq!(lr.lr(100, 50), 1e-3);
        assert_eq!(format!("{lr}"), "constant 0.001");
    }

    #[test]
    fn drop_lr_steps_once_after_drop_superbatch() {
        let lr = DropLR {
            start: 1.0,
            gamma: 0.1,
            drop: 10,
        };
        assert_eq!(lr.lr(0, 5), 1.0);
        assert_eq!(lr.lr(0, 10), 1.0);
        assert!((lr.lr(0, 11) - 0.1).abs() < EPS);
        assert!((lr.lr(0, 100) - 0.1).abs() < EPS);
    }

    #[test]
    fn step_lr_multiplicative_every_step() {
        let lr = StepLR {
            start: 1.0,
            gamma: 0.5,
            step: 3,
        };
        // saturating_sub(1) / step → superbatch=0..3 で steps=0、
        // superbatch=4..6 で steps=1、superbatch=7..9 で steps=2。
        assert_eq!(lr.lr(0, 0), 1.0);
        assert_eq!(lr.lr(0, 3), 1.0);
        assert!((lr.lr(0, 4) - 0.5).abs() < EPS);
        assert!((lr.lr(0, 7) - 0.25).abs() < EPS);
    }

    #[test]
    fn linear_decay_lr_interpolates() {
        let lr = LinearDecayLR {
            initial_lr: 1.0,
            final_lr: 0.0,
            final_superbatch: 10,
        };
        assert!((lr.lr(0, 0) - 1.0).abs() < EPS);
        assert!((lr.lr(0, 5) - 0.5).abs() < EPS);
        assert!((lr.lr(0, 10) - 0.0).abs() < EPS);
        assert!((lr.lr(0, 100) - 0.0).abs() < EPS); // saturate at final
    }

    #[test]
    fn cosine_decay_lr_midpoint() {
        // 数式: lambda = 1 - 0.5 * (1 + cos(PI * progress))。
        // progress=0.5 で cos(PI/2)=0、lambda=0.5、midpoint で initial と final の中間。
        let lr = CosineDecayLR {
            initial_lr: 1.0,
            final_lr: 0.0,
            final_superbatch: 10,
        };
        // progress=0 で lambda=0
        assert!((lr.lr(0, 0) - 1.0).abs() < EPS);
        // progress=0.5 で lambda=0.5
        assert!((lr.lr(0, 5) - 0.5).abs() < EPS);
        // saturate
        assert!((lr.lr(0, 10) - 0.0).abs() < EPS);
    }

    #[test]
    fn exponential_decay_lr_factor() {
        // initial * (final/initial)^lambda、midpoint で sqrt(final/initial)。
        let lr = ExponentialDecayLR {
            initial_lr: 1.0,
            final_lr: 0.01,
            final_superbatch: 10,
        };
        let at_mid = lr.lr(0, 5);
        // sqrt(0.01) = 0.1
        assert!((at_mid - 0.1).abs() < 1e-5);
        assert!((lr.lr(0, 10) - 0.01).abs() < EPS);
    }

    #[test]
    fn warmup_lr_only_in_first_superbatch() {
        let inner = ConstantLR { value: 1.0 };
        let warmup = WarmupLR {
            inner,
            warmup_batches: 4,
        };
        // superbatch=1 + batch<warmup_batches で base/(warmup-batch)。
        // batch=0 → 1/4=0.25, batch=1 → 1/3, batch=2 → 0.5, batch=3 → 1.0, batch=4 → 1.0 (warmup 終了)
        assert!((warmup.lr(0, 1) - 0.25).abs() < EPS);
        assert!((warmup.lr(1, 1) - (1.0 / 3.0)).abs() < EPS);
        assert!((warmup.lr(2, 1) - 0.5).abs() < EPS);
        assert!((warmup.lr(3, 1) - 1.0).abs() < EPS);
        assert!((warmup.lr(4, 1) - 1.0).abs() < EPS);
        // superbatch != 1 では warmup は inactive。
        assert!((warmup.lr(0, 2) - 1.0).abs() < EPS);
    }

    #[test]
    fn sequence_lr_switches_at_midpoint() {
        let first = ConstantLR { value: 1.0 };
        let second = ConstantLR { value: 2.0 };
        let seq = SequenceLR {
            first,
            second,
            first_scheduler_final_superbatch: 5,
        };
        // superbatch <= 5 で first (1.0)、> 5 で second (2.0)。
        assert_eq!(seq.lr(0, 1), 1.0);
        assert_eq!(seq.lr(0, 5), 1.0);
        assert_eq!(seq.lr(0, 6), 2.0);
        assert_eq!(seq.lr(0, 100), 2.0);
    }

    #[test]
    fn one_cycle_lr_warmup_then_anneal() {
        // max_lr=1, div_factor=25 → initial=0.04、final_div_factor=100 →
        // final=0.04/100=0.0004。warmup_pct=0.2、total=10 → warmup_superbatch=2。
        let lr = OneCycleLR::new(1.0, 0.2, 25.0, 100.0, 10);
        assert_eq!(lr.initial_lr, 0.04);
        assert_eq!(lr.warmup_superbatch, 2);
        assert!((lr.final_lr - 0.0004).abs() < EPS);

        // superbatch=0: warmup p=0 → initial_lr。
        assert!((lr.lr(0, 0) - 0.04).abs() < EPS);
        // superbatch=2 (warmup 終端): p=1 → max_lr。
        assert!((lr.lr(0, 2) - 1.0).abs() < EPS);
        // warmup 中点 superbatch=1: p=0.5 → initial と max の中間。
        let mid_warmup = 0.04 + 0.5 * (1.0 - 0.04);
        assert!((lr.lr(0, 1) - mid_warmup).abs() < EPS);
        // anneal 中点 superbatch=6: (6-2)/(10-2)=0.5 → max と final の中間。
        let mid_anneal = 0.0004 + 0.5 * (1.0 - 0.0004);
        assert!((lr.lr(0, 6) - mid_anneal).abs() < EPS);
        // 終端以降は final_lr で saturate。
        assert!((lr.lr(0, 10) - 0.0004).abs() < EPS);
        assert!((lr.lr(0, 100) - 0.0004).abs() < EPS);
    }

    #[test]
    fn one_cycle_lr_zero_warmup_pct_has_no_division_by_zero() {
        // warmup_pct=0 → warmup_superbatch=0。superbatch=0 で denom.max(1) により
        // 0 除算を回避し initial_lr を返す。
        let lr = OneCycleLR::new(1.0, 0.0, 25.0, 100.0, 10);
        assert_eq!(lr.warmup_superbatch, 0);
        assert!(lr.lr(0, 0).is_finite());
        // superbatch>=1 は anneal 相 (max_lr から下る)。
        assert!(lr.lr(0, 1).is_finite());
    }

    #[test]
    fn lr_scheduler_enum_dispatches_and_wraps_warmup() {
        let step = LrSchedulerEnum::Step(StepLR {
            start: 1.0,
            gamma: 0.5,
            step: 3,
        });
        // enum dispatch は内側 StepLR と一致。
        assert!((step.lr(0, 4) - 0.5).abs() < EPS);

        // with_warmup は superbatch=1 内の batch warmup を加える (WarmupLR と同形)。
        let warmed = LrSchedulerEnum::Constant(ConstantLR { value: 1.0 }).with_warmup(4);
        assert!((warmed.lr(0, 1) - 0.25).abs() < EPS);
        assert!((warmed.lr(3, 1) - 1.0).abs() < EPS);
        // superbatch != 1 では warmup inactive。
        assert!((warmed.lr(0, 2) - 1.0).abs() < EPS);
    }

    #[test]
    fn horizon_reports_terminal_superbatch_for_horizon_schedules() {
        // 終端を持つ schedule は final/total superbatch を返す。
        assert_eq!(
            LinearDecayLR {
                initial_lr: 1.0,
                final_lr: 0.0,
                final_superbatch: 42,
            }
            .horizon(),
            Some(42)
        );
        assert_eq!(
            CosineDecayLR {
                initial_lr: 1.0,
                final_lr: 0.0,
                final_superbatch: 7,
            }
            .horizon(),
            Some(7)
        );
        assert_eq!(
            ExponentialDecayLR {
                initial_lr: 1.0,
                final_lr: 0.01,
                final_superbatch: 9,
            }
            .horizon(),
            Some(9)
        );
        assert_eq!(
            OneCycleLR::new(1.0, 0.2, 25.0, 100.0, 100).horizon(),
            Some(100)
        );

        // horizon を持たない schedule は None。
        assert_eq!(ConstantLR { value: 1e-3 }.horizon(), None);
        assert_eq!(
            StepLR {
                start: 1.0,
                gamma: 0.5,
                step: 3,
            }
            .horizon(),
            None
        );
        assert_eq!(
            DropLR {
                start: 1.0,
                gamma: 0.1,
                drop: 10,
            }
            .horizon(),
            None
        );

        // enum / warmup wrapper は内側の horizon を透過する。
        let oc = LrSchedulerEnum::OneCycle(OneCycleLR::new(1.0, 0.2, 25.0, 100.0, 50));
        assert_eq!(oc.horizon(), Some(50));
        assert_eq!(oc.with_warmup(4).horizon(), Some(50));
        assert_eq!(
            LrSchedulerEnum::Constant(ConstantLR { value: 1.0 })
                .with_warmup(4)
                .horizon(),
            None
        );
    }

    #[test]
    fn constant_wdl_is_invariant() {
        let w = ConstantWDL { value: 0.5 };
        assert_eq!(w.blend(0, 1, 10), 0.5);
        assert_eq!(w.blend(100, 5, 10), 0.5);
    }

    #[test]
    fn linear_wdl_interpolates_with_max() {
        let w = LinearWDL {
            start: 0.0,
            end: 1.0,
        };
        // grad = (1.0 - 0.0) / (max - 1) = 1/9 at max=10。
        // superbatch=1 → 0.0、superbatch=10 → 1.0、superbatch=5 → 4/9。
        assert!((w.blend(0, 1, 10) - 0.0).abs() < EPS);
        assert!((w.blend(0, 10, 10) - 1.0).abs() < EPS);
        assert!((w.blend(0, 5, 10) - (4.0 / 9.0)).abs() < EPS);
    }

    #[test]
    fn linear_wdl_handles_max_one_without_division_by_zero() {
        // `(max - 1).max(1)` で max=1 でも 0 除算を回避することの確認。
        let w = LinearWDL {
            start: 0.0,
            end: 1.0,
        };
        // max=1 のとき grad = 1/1、superbatch=1 で start + 1*0 = 0.0。
        let v = w.blend(0, 1, 1);
        assert!(v.is_finite());
    }

    #[test]
    fn warmup_wdl_only_in_first_superbatch() {
        let inner = ConstantWDL { value: 0.5 };
        let warmup = WarmupWDL {
            inner,
            warmup_batches: 2,
        };
        // batch=0 → 0.5/2=0.25, batch=1 → 0.5/1=0.5
        assert!((warmup.blend(0, 1, 10) - 0.25).abs() < EPS);
        assert!((warmup.blend(1, 1, 10) - 0.5).abs() < EPS);
        // batch=2 → 通常 (warmup 終了)
        assert!((warmup.blend(2, 1, 10) - 0.5).abs() < EPS);
        // superbatch != 1
        assert!((warmup.blend(0, 2, 10) - 0.5).abs() < EPS);
    }

    #[test]
    fn sequence_wdl_propagates_normalised_max() {
        // first scheduler は max=midpoint、second scheduler は max-midpoint で呼ぶ。
        let first = LinearWDL {
            start: 0.0,
            end: 1.0,
        };
        let second = LinearWDL {
            start: 1.0,
            end: 2.0,
        };
        let seq = SequenceWDL {
            first,
            second,
            first_scheduler_final_superbatch: 5,
        };
        // first phase: superbatch=1..5 で 0..1 線形 (max=5、grad=1/4)。
        // superbatch=1 → 0.0、superbatch=5 → 1.0。
        assert!((seq.blend(0, 1, 10) - 0.0).abs() < EPS);
        assert!((seq.blend(0, 5, 10) - 1.0).abs() < EPS);

        // second phase: superbatch=6..10 → first から見ると 1..5 (max=5、grad=1/4)。
        // superbatch=6 → second.blend(0, 1, 5) = 1.0、superbatch=10 → 2.0。
        assert!((seq.blend(0, 6, 10) - 1.0).abs() < EPS);
        assert!((seq.blend(0, 10, 10) - 2.0).abs() < EPS);
    }

    #[test]
    fn wdl_scheduler_enum_dispatches_to_constant_or_linear() {
        let c = WdlSchedulerEnum::constant(0.7);
        assert!((c.blend(0, 1, 10) - 0.7).abs() < EPS);

        let l = WdlSchedulerEnum::linear(0.0, 1.0);
        assert!((l.blend(0, 1, 10) - 0.0).abs() < EPS);
        assert!((l.blend(0, 10, 10) - 1.0).abs() < EPS);
    }
}
