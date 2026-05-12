//! Ranger (RAdam + Lookahead) host-side state + checkpoint。
//!
//! Stage 2-4 / 2-5 で landed した GPU kernel (`radam_step` /
//! `ranger_lookahead_lerp`) を Stage 3 trainer から launch するときの
//! **host-side state container** + checkpoint serialise / deserialise + bullet
//! 上流 `crates/trainer/src/optimiser/{radam,ranger}.rs::update` の host 制御 path
//! を提供する。
//!
//! ## 設計 (CPU-only crate の制約への対応)
//!
//! `crates/nnue-train` は CPU-only library 方針 (Stage 3-0 で確立、CI で test を
//! 通す)、本 module は `gpu-runtime` に depend せず **device buffer (`DeviceBuffer
//! <f32>`) は持たない**。代わりに `Vec<f32>` host buffer を保持し、Stage 3-7
//! `bins/nnue_train/src/main.rs::GpuTrainer` が:
//!
//! 1. 本 module の `RangerHostState` を初期化 → host `Vec<f32>` 0-init
//! 2. device buffer (`DeviceBuffer<f32>`) を確保 → host → device コピー
//! 3. 各 step で `radam_step` / `ranger_lookahead_lerp` kernel を `cuda_launch!`
//!    で起動 (kernel `#[kernel]` 本体は bin entry inline、Stage 1-9 / 2-5 と同 pattern)
//! 4. checkpoint 保存時は device → host コピー → 本 module の `save_to_writer`
//!
//! を行う。本 module は **state + step counter + host pre-compute helper + I/O**
//! のみを担当する責務分離設計 (Stage 1-9 の `GpuTrainer` と同流儀)。
//!
//! ## bullet 上流からの差分
//!
//! - bullet trait `OptimiserState<G: Gpu>` / `WrapOptimiser<Inner, Params>` は
//!   本リポでは **取り込まず**、`RAdamHostState` / `RangerHostState` の独立
//!   struct (Stage 1-1 / 3-1 / 3-4 / 3-5 と同じ bullet trait 削除ポリシー)
//! - bullet `Buffer<G>` (device buffer) は本リポでは `Vec<f32>` (host)、device
//!   側は bin 側で `DeviceBuffer<f32>` に host-to-device コピー
//! - bullet の `build_ranger_op` (`PointwiseIR` で kernel 構築) は本リポでは
//!   Stage 2-5 で landed した GPU `#[kernel] fn ranger_lookahead_lerp` (bin entry
//!   inline) を使うため不要
//! - bullet checkpoint は `slow.bin` (f32 LE) + `step_ranger.txt` (text、id, step
//!   形式) の 2 file 構成。本リポは **state を 1 binary file に集約** (Stage 1
//!   progress.bin 慣行と整合)、layout:
//!     - magic 4 bytes (`b"RNGR"`)
//!     - version u32 LE (本 PR は 1)
//!     - step u64 LE
//!     - n_params u64 LE
//!     - momentum f32 LE × n_params
//!     - velocity f32 LE × n_params
//!     - slow_params f32 LE × n_params
//! - bullet `step.is_multiple_of(self.k)` (1.87 stable) は MSRV 1.85 罠を踏まず
//!   `step % k == 0` 直書き (Stage 2-5 / 3-3 で確立済規約)

use std::io::{self, Read, Write};

pub use gpu_kernels::pointwise::radam_step::radam_compute_step_size_denom;

// =============================================================================
// パラメータ
// =============================================================================

/// Ranger optimizer のハイパパラメータ。
///
/// bullet 上流 `crates/trainer/src/optimiser/ranger.rs::RangerParams` と同
/// default (decay=0.01, beta1=0.99, beta2=0.999, min_weight=-1.98, max_weight=1.98,
/// alpha=0.5, k=6)、加えて `radam_step` kernel が要求する eps + n_sma_threshold を
/// field 化。
#[derive(Clone, Copy, Debug)]
pub struct RangerParams {
    /// weight decay 係数 (AdamW-style decoupled decay)。
    pub decay: f32,
    /// 1st moment EMA decay。
    pub beta1: f32,
    /// 2nd moment EMA decay。
    pub beta2: f32,
    /// 数値安定化用 epsilon (`1/sqrt(v)+eps`)。
    pub eps: f32,
    /// weight clip 下限。
    pub min_weight: f32,
    /// weight clip 上限。
    pub max_weight: f32,
    /// Lookahead lerp 係数 (`weights = alpha * weights + (1-alpha) * slow`)。
    pub alpha: f32,
    /// Lookahead lerp 周期 (`step % k == 0` で lerp 起動)。
    pub k: usize,
    /// RAdam variance 補正の閾値 (n_sma > threshold で `1/sqrt(v)` 経路を on)。
    pub n_sma_threshold: f32,
}

impl RangerParams {
    /// bullet 上流 `RangerParams::default()` と同値の `const` 定数。`Default` impl は
    /// これを返す。`bins/nnue_train::GpuTrainer` のように `const` 文脈で個々の field を
    /// 参照したい側 (kernel launch 引数) はこれを single source of truth として使う
    /// (Stage 3-quality #86: const 二重定義の解消)。
    pub const DEFAULT: Self = Self {
        decay: 0.01,
        beta1: 0.99,
        beta2: 0.999,
        eps: 1e-8,
        min_weight: -1.98,
        max_weight: 1.98,
        alpha: 0.5,
        k: 6,
        n_sma_threshold: 5.0,
    };
}

impl Default for RangerParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// =============================================================================
// RAdamHostState — RAdam 単体の host state (Ranger inner として使う)
// =============================================================================

/// RAdam optimizer の host-side state (1st/2nd moment + step counter)。
///
/// Stage 3-7 `GpuTrainer` が本 state を device buffer (`DeviceBuffer<f32>`) に
/// host-to-device コピーして `radam_step` kernel に渡す。
#[derive(Clone, Debug, PartialEq)]
pub struct RAdamHostState {
    /// 1st moment (`m[i] = beta1 * m[i] + (1-beta1) * g[i]`)、長さ `n_params`。
    pub momentum: Vec<f32>,
    /// 2nd moment (`v[i] = beta2 * v[i] + (1-beta2) * g[i]^2`)、長さ `n_params`。
    pub velocity: Vec<f32>,
    /// step counter (1-indexed、`radam_compute_step_size_denom` の引数として使う)。
    pub step: u64,
}

impl RAdamHostState {
    /// 全 zero の state を確保。
    pub fn new(n_params: usize) -> Self {
        Self {
            momentum: vec![0.0; n_params],
            velocity: vec![0.0; n_params],
            step: 0,
        }
    }

    /// step を 1 進めて新 step value (1-indexed) を返す。
    pub fn advance_step(&mut self) -> u64 {
        self.step += 1;
        self.step
    }

    /// `radam_compute_step_size_denom` を本 state の current step で呼び、
    /// `(step_size, denom)` を返す host pre-compute helper。`advance_step`
    /// 直後に呼ぶ前提 (step >= 1)。
    pub fn compute_step_size_denom(
        &self,
        beta1: f32,
        beta2: f32,
        n_sma_threshold: f32,
    ) -> (f32, i32) {
        debug_assert!(
            self.step >= 1,
            "advance_step を呼ぶ前に compute すると bc1=0 で +inf"
        );
        radam_compute_step_size_denom(self.step, beta1, beta2, n_sma_threshold)
    }

    /// state 全体を zero clear + step=0 (`reset`、bullet `OptimiserState::reset` 相当)。
    pub fn reset(&mut self) {
        self.momentum.fill(0.0);
        self.velocity.fill(0.0);
        self.step = 0;
    }
}

// =============================================================================
// RangerHostState — RAdam + Lookahead の host state
// =============================================================================

/// Ranger optimizer (RAdam + Lookahead) の host-side state。
///
/// `radam` field が RAdam 本体、`slow_params` が Lookahead の slow weight buffer。
/// `step % k == 0` で `ranger_lookahead_lerp` kernel を起動 (bullet 上流
/// `RangerLookahead::update` `:91-97` と同型)。
#[derive(Clone, Debug, PartialEq)]
pub struct RangerHostState {
    /// RAdam 本体 (`momentum` + `velocity` + step counter)。
    pub radam: RAdamHostState,
    /// Lookahead slow weights (`alpha * weights + (1-alpha) * slow` で lerp、
    /// kernel 内 + host orchestration、長さ `n_params`)。
    pub slow_params: Vec<f32>,
}

impl RangerHostState {
    /// `slow_params` を **初期 weight で初期化** した state を確保。
    ///
    /// bullet 上流 `RangerLookahead::new` は slow_params を 0 で初期化するが、
    /// それは `weights` が 0 init の場合に限り正しい挙動 (lerp 初回で
    /// `alpha * w + (1-alpha) * 0 = alpha * w` になり半減してしまう、初期 lerp で
    /// state が degenerate にならないようにする必要がある)。
    ///
    /// 本リポでは **明示的に初期 weight を渡し**、`slow_params = initial_weights.clone()`
    /// で初期化 (Stage 3-7 trainer が `init_weights(...)` を NNUE 初期化と同タイミングで
    /// 呼ぶ前提)。bullet は `weights` を 0 init して優先するか、または `update` 経由で
    /// slow が rolling average に追従するのを許容しており、本 PR の方が初期挙動として
    /// 明示的。
    pub fn new_with_initial_weights(initial_weights: &[f32]) -> Self {
        let n_params = initial_weights.len();
        Self {
            radam: RAdamHostState::new(n_params),
            slow_params: initial_weights.to_vec(),
        }
    }

    /// `slow_params` を 0 で初期化した state (bullet 上流 `new` と同型、init weight
    /// が 0 の場合や、初期 lerp degenerate を許容するときに使う)。
    pub fn new_zeroed(n_params: usize) -> Self {
        Self {
            radam: RAdamHostState::new(n_params),
            slow_params: vec![0.0; n_params],
        }
    }

    /// `step % k == 0` を満たすかを判定 (`step == 0` は除外)。
    ///
    /// **`k == 0` は panic せず `false` を返す** (defensive、bullet 上流は
    /// `is_multiple_of(0)` で常に true を返す挙動だがそれは意味的に間違いで、本リポは
    /// 「k=0 = Lookahead 無効」と解釈)。`step == 0` も学習開始前として false。
    /// MSRV 1.85 罠回避: `usize::is_multiple_of` (1.87 stable) は使わず `% != 0`
    /// 直書き (Stage 2-5 / 3-3 で確立済規約)。
    pub fn should_lookahead(&self, k: usize) -> bool {
        let step = self.radam.step;
        step > 0 && k > 0 && step % (k as u64) == 0
    }

    /// state 全体を zero clear (`radam` reset + **slow_params も 0 fill**)。
    ///
    /// **bullet 上流からの意図的 divergence** (Codex review #62 で明示化):
    /// bullet `RangerLookahead::reset` (`ranger.rs:102-105`) は `slow_params` を
    /// **変更しない** (inner.reset() のみ呼ぶ)。本リポは「reset = 完全 zero init」
    /// の semantics に揃え、`slow_params` も 0 fill する設計。
    ///
    /// `new_with_initial_weights` で確立した非零 slow を保持したい場合は本 method を
    /// 呼ばず、`self.radam.reset()` のみ手動で呼ぶこと。次 epoch / resume で
    /// slow を初期 weight に再 init したい場合は `*self =
    /// RangerHostState::new_with_initial_weights(&initial_weights)` で置き換える。
    pub fn reset(&mut self) {
        self.radam.reset();
        self.slow_params.fill(0.0);
    }
}

// =============================================================================
// Checkpoint serialise / deserialise
// =============================================================================

/// checkpoint format magic (`b"RNGR"`、本 PR で確定)。
pub const CHECKPOINT_MAGIC: [u8; 4] = *b"RNGR";

/// checkpoint format version (本 PR は 1、後続変更で increment 想定)。
pub const CHECKPOINT_VERSION: u32 = 1;

impl RangerHostState {
    /// checkpoint を `w` に書き出す。layout:
    ///
    /// ```text
    /// 0..4    magic (b"RNGR")
    /// 4..8    version u32 LE (1)
    /// 8..16   step u64 LE
    /// 16..24  n_params u64 LE
    /// 24..    momentum f32 LE × n_params
    /// ...     velocity f32 LE × n_params
    /// ...     slow_params f32 LE × n_params
    /// ```
    ///
    /// I/O 効率の注意 (Codex review #62 指摘): 本 method は f32 × n_params 個の
    /// `write_all` を sequential に呼ぶため、large `n_params` (e.g. NNUE 1536-16-32
    /// の 73_305 × 1536 = ~113M) では呼び出し側で `BufWriter` で wrap することを
    /// 強く推奨する (system call 数を削減)。
    pub fn save_to_writer<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let n = self.slow_params.len();
        if self.radam.momentum.len() != n || self.radam.velocity.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "RangerHostState dim mismatch: momentum={}, velocity={}, slow={}",
                    self.radam.momentum.len(),
                    self.radam.velocity.len(),
                    n
                ),
            ));
        }

        w.write_all(&CHECKPOINT_MAGIC)?;
        w.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;
        w.write_all(&self.radam.step.to_le_bytes())?;
        w.write_all(&(n as u64).to_le_bytes())?;

        for &v in &self.radam.momentum {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.radam.velocity {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.slow_params {
            w.write_all(&v.to_le_bytes())?;
        }

        Ok(())
    }

    /// checkpoint を `r` から読む。magic / version 不一致は `InvalidData`。
    ///
    /// `expected_n_params` を `Some(n)` で渡すと、checkpoint 内の `n_params` と
    /// 照合し不一致なら `InvalidData` で reject (Stage 3-7 trainer が model 次元と
    /// 整合性 check するときの安全策、Codex review #62 で追加)。`None` の場合は
    /// checkpoint 内 `n_params` をそのまま受け入れる (テスト / ダンプ用、production
    /// では `Some` 推奨)。
    pub fn load_from_reader<R: Read>(
        r: &mut R,
        expected_n_params: Option<usize>,
    ) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != CHECKPOINT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("RangerHostState magic mismatch: got {magic:?}, want {CHECKPOINT_MAGIC:?}"),
            ));
        }

        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != CHECKPOINT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "RangerHostState version mismatch: got {version}, want {CHECKPOINT_VERSION}"
                ),
            ));
        }

        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let step = u64::from_le_bytes(buf8);

        r.read_exact(&mut buf8)?;
        let n_u64 = u64::from_le_bytes(buf8);
        // u64 → usize の unchecked cast を避け、32-bit target / 破損ファイルでの
        // overflow を `InvalidData` で reject (Codex review #62 で追加)。
        let n: usize = n_u64.try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("RangerHostState n_params={n_u64} exceeds usize::MAX"),
            )
        })?;

        if let Some(expected) = expected_n_params {
            if n != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("RangerHostState n_params mismatch: got {n}, want {expected}"),
                ));
            }
        }

        let momentum = read_f32_vec(r, n)?;
        let velocity = read_f32_vec(r, n)?;
        let slow_params = read_f32_vec(r, n)?;

        Ok(Self {
            radam: RAdamHostState {
                momentum,
                velocity,
                step,
            },
            slow_params,
        })
    }
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn ranger_params_default_matches_bullet() {
        let p = RangerParams::default();
        // bullet 上流 `ranger.rs:176`: decay=0.01, beta1=0.99, beta2=0.999,
        // min_weight=-1.98, max_weight=1.98, alpha=0.5, k=6
        assert_eq!(p.decay, 0.01);
        assert_eq!(p.beta1, 0.99);
        assert_eq!(p.beta2, 0.999);
        assert_eq!(p.min_weight, -1.98);
        assert_eq!(p.max_weight, 1.98);
        assert_eq!(p.alpha, 0.5);
        assert_eq!(p.k, 6);
        // 本 PR 追加 (bullet と同 default):
        assert_eq!(p.eps, 1e-8);
        assert_eq!(p.n_sma_threshold, 5.0);
    }

    #[test]
    fn radam_host_state_advance_step_is_one_indexed() {
        let mut s = RAdamHostState::new(8);
        assert_eq!(s.step, 0);
        assert_eq!(s.advance_step(), 1);
        assert_eq!(s.advance_step(), 2);
        assert_eq!(s.step, 2);
    }

    #[test]
    fn radam_host_state_compute_step_size_denom_matches_kernel_helper() {
        // gpu_kernels::pointwise::radam_step::radam_compute_step_size_denom と同値を返す。
        let mut s = RAdamHostState::new(4);
        s.advance_step();
        let (ss1, dn1) = s.compute_step_size_denom(0.9, 0.999, 5.0);
        let (ss2, dn2) = radam_compute_step_size_denom(1, 0.9, 0.999, 5.0);
        assert_eq!(ss1, ss2);
        assert_eq!(dn1, dn2);
    }

    #[test]
    fn radam_host_state_reset_zeros_state_and_step() {
        let mut s = RAdamHostState::new(4);
        s.momentum[0] = 1.0;
        s.velocity[1] = 2.0;
        s.advance_step();
        s.reset();
        assert!(s.momentum.iter().all(|&v| v == 0.0));
        assert!(s.velocity.iter().all(|&v| v == 0.0));
        assert_eq!(s.step, 0);
    }

    #[test]
    fn ranger_new_with_initial_weights_copies_to_slow() {
        let initial = vec![0.1, -0.2, 0.3];
        let s = RangerHostState::new_with_initial_weights(&initial);
        assert_eq!(s.slow_params, initial);
        assert_eq!(s.radam.momentum.len(), 3);
        assert!(s.radam.momentum.iter().all(|&v| v == 0.0));
        assert_eq!(s.radam.step, 0);
    }

    #[test]
    fn ranger_new_zeroed_starts_with_zero_slow() {
        let s = RangerHostState::new_zeroed(3);
        assert_eq!(s.slow_params, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn ranger_should_lookahead_respects_k() {
        let mut s = RangerHostState::new_zeroed(2);
        // step=0 は除外 (まだ学習開始前)。
        assert!(!s.should_lookahead(6));

        // step=1..5: 起動せず。
        for _ in 0..5 {
            s.radam.advance_step();
        }
        assert!(!s.should_lookahead(6));

        // step=6: 起動。
        s.radam.advance_step();
        assert!(s.should_lookahead(6));

        // step=7: 再度待機。
        s.radam.advance_step();
        assert!(!s.should_lookahead(6));

        // k=1 では毎 step 起動。
        let mut s2 = RangerHostState::new_zeroed(2);
        s2.radam.advance_step();
        assert!(s2.should_lookahead(1));
        s2.radam.advance_step();
        assert!(s2.should_lookahead(1));

        // k=0 は安全に false (panic しない、defensive)。
        assert!(!s.should_lookahead(0));
    }

    #[test]
    fn ranger_reset_zeros_radam_and_slow() {
        let mut s = RangerHostState::new_with_initial_weights(&[0.5, -0.5]);
        s.radam.momentum[0] = 1.0;
        s.radam.advance_step();
        s.reset();
        assert!(s.radam.momentum.iter().all(|&v| v == 0.0));
        assert!(s.slow_params.iter().all(|&v| v == 0.0));
        assert_eq!(s.radam.step, 0);
    }

    #[test]
    fn checkpoint_round_trip_preserves_state() {
        let mut s = RangerHostState::new_with_initial_weights(&[0.1, 0.2, 0.3, 0.4]);
        s.radam.momentum.copy_from_slice(&[0.5, -0.5, 0.25, -0.25]);
        s.radam.velocity.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        s.radam.step = 42;

        let mut buf = Vec::new();
        s.save_to_writer(&mut buf).unwrap();

        // layout: 4 magic + 4 version + 8 step + 8 n + 4*3 (momentum/velocity/slow) * 4 bytes
        // = 24 + 4 * 4 * 3 = 24 + 48 = 72 bytes
        assert_eq!(buf.len(), 24 + 4 * 4 * 3);

        let r = RangerHostState::load_from_reader(&mut Cursor::new(&buf), None).unwrap();
        assert_eq!(r, s);
    }

    #[test]
    fn checkpoint_rejects_wrong_magic() {
        let mut buf = vec![b'X', b'X', b'X', b'X']; // wrong magic
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let err = RangerHostState::load_from_reader(&mut Cursor::new(&buf), None)
            .expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("magic mismatch"));
    }

    #[test]
    fn checkpoint_rejects_wrong_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&CHECKPOINT_MAGIC);
        buf.extend_from_slice(&999u32.to_le_bytes()); // wrong version
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let err = RangerHostState::load_from_reader(&mut Cursor::new(&buf), None)
            .expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("version mismatch"));
    }

    #[test]
    fn save_to_writer_rejects_dim_mismatch() {
        let mut s = RangerHostState::new_zeroed(4);
        s.radam.momentum.pop(); // 3 instead of 4
        let err = s.save_to_writer(&mut Vec::new()).expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(format!("{err}").contains("dim mismatch"));
    }

    #[test]
    fn checkpoint_round_trip_with_zero_size_state() {
        let s = RangerHostState::new_zeroed(0);
        let mut buf = Vec::new();
        s.save_to_writer(&mut buf).unwrap();
        // 24 bytes header only (no params).
        assert_eq!(buf.len(), 24);
        let r = RangerHostState::load_from_reader(&mut Cursor::new(&buf), None).unwrap();
        assert_eq!(r, s);
    }

    #[test]
    fn checkpoint_load_with_expected_n_validates_dim() {
        // `expected_n_params=Some(n)` で checkpoint 内次元と照合、不一致なら
        // InvalidData reject (Stage 3-7 trainer の安全策、Codex review #62 で追加)。
        let s = RangerHostState::new_with_initial_weights(&[0.1, 0.2, 0.3]);
        let mut buf = Vec::new();
        s.save_to_writer(&mut buf).unwrap();

        // 正しい次元: 受理。
        let r = RangerHostState::load_from_reader(&mut Cursor::new(&buf), Some(3)).unwrap();
        assert_eq!(r, s);

        // 期待 != 実際: reject。
        let err = RangerHostState::load_from_reader(&mut Cursor::new(&buf), Some(4))
            .expect_err("must reject n_params mismatch");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("n_params mismatch"));
    }

    #[test]
    fn reset_zeros_slow_params_diverging_from_bullet() {
        // bullet 上流 `RangerLookahead::reset` (`ranger.rs:102-105`) は slow を
        // 変更しないが、本リポは reset で slow_params も 0 fill する **意図的
        // divergence** (Codex review #62 で明示)。本 test は本リポ実装の挙動を pin
        // する (bullet 互換に戻すなら本 test を消すか、`reset_radam_only` 別 API を
        // 追加するか)。
        let mut s = RangerHostState::new_with_initial_weights(&[0.1, 0.2, 0.3]);
        assert_eq!(s.slow_params, vec![0.1, 0.2, 0.3]);
        s.reset();
        assert!(
            s.slow_params.iter().all(|&v| v == 0.0),
            "本リポの `reset` は bullet と divergence、slow も 0 fill する"
        );
    }
}
