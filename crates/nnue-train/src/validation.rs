//! held-out validation — 学習に使わない別 PSV / HCPE データで loss / accuracy を測る。
//!
//! training loss は学習が今まさにフィットしているデータ上の値でバイアスがある。
//! held-out (勾配更新に一度も使わない) データ上の loss / accuracy は汎化性能の
//! 不偏推定で、過学習や発散 (NaN) を SPRT 自己対局より早く・安く検出できる。
//!
//! ## 構成
//!
//! - [`HeldoutSet`] は test data の先頭から固定の検証 batch 集合を作る
//!   (毎 superbatch 同じ集合で測ることで loss/accuracy の軌跡が低分散になる)。
//! - 各 superbatch 末に [`HeldoutSet::evaluate`] が backend の
//!   [`TrainerBackend::validate_step`] (forward + loss のみ) を全 batch に回し、
//!   [`ValidationReport`] (平均 loss と sign-agreement accuracy) を返す。
//!
//! ## sign-agreement accuracy
//!
//! `test_value_accuracy` は「モデル出力の符号が実際の対局結果と一致した割合」。
//! 引き分け (`wdl == 0.5`) は符号が定義できないため分母から除外する。scale 不変
//! (出力の絶対値スケールに依らない) なので loss と違い run / 設定をまたいで
//! 比較できる。

use std::io;
use std::path::Path;

use shogi_features::FeatureSetSpec;
#[cfg(test)]
use shogi_features::progress_kpabs::ShogiProgressKPAbs;

use crate::dataloader::{Batch, BucketMode, HcpeFileLoader, PsvFileLoader};
use crate::trainer::{LossKind, TrainerBackend};

/// held-out validation 1 回分の集計結果。
#[derive(Debug, Clone, Copy)]
pub struct ValidationReport {
    /// held-out 全 position の平均 loss (`Σ err² / n_positions`)。training loss と
    /// 同じ式・同じ単位なので同 superbatch の training loss と直接比べられる。
    pub mean_loss: f64,
    /// sign-agreement accuracy (`[0, 1]`)。引き分けは分母から除外。分母が 0
    /// (全て引き分け) なら `NaN`。
    pub accuracy: f64,
    /// 検証に使った position 総数。
    pub n_positions: u64,
    /// accuracy の分母 (= `n_positions` − 引き分け数)。
    pub n_counted: u64,
}

/// 起動時に固定した held-out 検証集合。各 superbatch 末に同じ集合で評価する。
///
/// 検証 batch はすべて `batch_size` ちょうどの満タン batch。末尾の partial batch は
/// GPU の tiled kernel が `b % 16 == 0` を要求するため捨てる。
#[derive(Debug)]
pub struct HeldoutSet {
    /// `(Batch, per-position bucket)`。`bucket` は学習側 dataloader と同じく
    /// 学習側 dataloader と同じ [`BucketMode`] で計算する。
    batches: Vec<(Batch, Vec<i32>)>,
    /// 検証 position 総数 (`batches.len() * batch_size`)。
    n_positions: u64,
}

impl HeldoutSet {
    /// test PSV / HCPE file の **先頭から** 固定の検証集合を読み込む
    /// ([`HeldoutSet::load_from_range`] の `[0, file_size)` 特化)。
    ///
    /// `test_positions` を `batch_size` 単位に切り上げた数の満タン batch を作る。
    /// PSV を逐次読みし EOF で打ち切る (学習 loader のような epoch wrap はしない —
    /// 検証集合に同一局面が重複しないため)。`score_drop_abs` 指定時は学習と同じく
    /// `|score| >= t` の局面を除外し、`score_clamp_abs` 指定時は生き残った局面の
    /// score を `[-c, c]` に飽和させる。
    ///
    /// test file が満タン batch 1 個分にも満たない場合は error を返す。
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        path: &Path,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        test_positions: usize,
        bucket_mode: &(impl Copy + Into<BucketMode>),
        feature_set: FeatureSetSpec,
        num_buckets: usize,
    ) -> io::Result<Self> {
        if path.extension().is_some_and(|ext| {
            ext.to_str()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("hcpe"))
        }) {
            let loader = HcpeFileLoader::new(path)?;
            return Self::load_boards(
                loader,
                |loader| loader.next_board(),
                path,
                batch_size,
                score_drop_abs,
                score_clamp_abs,
                test_positions,
                bucket_mode,
                feature_set,
                num_buckets,
            );
        }

        let file_size = std::fs::metadata(path)?.len();
        Self::load_from_range(
            path,
            0,
            file_size,
            batch_size,
            score_drop_abs,
            score_clamp_abs,
            test_positions,
            bucket_mode,
            feature_set,
            num_buckets,
        )
    }

    /// test PSV file の `[start_offset, end_offset)` byte range から固定の検証
    /// 集合を読み込む。training PSV の末尾 N 局面を holdout 専用に分離する
    /// (`--test-tail-positions`) 経路で使う。挙動は [`HeldoutSet::load`] と同じく
    /// EOF (= range 末尾) で打ち切り、wrap はしない。range 検証 (alignment /
    /// `end <= file_size` / `start <= end`) は [`PsvFileLoader::new_range`] に
    /// 委譲する。
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_range(
        path: &Path,
        start_offset: u64,
        end_offset: u64,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        test_positions: usize,
        bucket_mode: &(impl Copy + Into<BucketMode>),
        feature_set: FeatureSetSpec,
        num_buckets: usize,
    ) -> io::Result<Self> {
        let loader = PsvFileLoader::new_range(path, start_offset, end_offset)?;
        Self::load_boards(
            loader,
            |loader| Ok(loader.next_psv()?.map(|psv| psv.decode())),
            path,
            batch_size,
            score_drop_abs,
            score_clamp_abs,
            test_positions,
            bucket_mode,
            feature_set,
            num_buckets,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_boards<L>(
        mut loader: L,
        mut next_board: impl FnMut(&mut L) -> io::Result<Option<shogi_format::ShogiBoard>>,
        path: &Path,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        test_positions: usize,
        bucket_mode: &(impl Copy + Into<BucketMode>),
        feature_set: FeatureSetSpec,
        num_buckets: usize,
    ) -> io::Result<Self> {
        assert!(batch_size >= 1, "batch_size must be >= 1");
        assert!(num_buckets >= 1, "num_buckets must be >= 1");
        let bucket_mode = (*bucket_mode).into();
        let n_batches = test_positions.div_ceil(batch_size).max(1);
        let mut batches: Vec<(Batch, Vec<i32>)> = Vec::with_capacity(n_batches);
        let mut cur = Batch::with_capacity(batch_size, feature_set);
        let mut cur_buckets: Vec<i32> = Vec::with_capacity(batch_size);

        while batches.len() < n_batches {
            let Some(mut board) = next_board(&mut loader)? else {
                break; // EOF — epoch wrap しない
            };
            // 学習側 `PsvEpochReader` と同じ score-drop 近似 (i64 cast で
            // `i16::MIN` の abs overflow を避ける)。
            if let Some(t) = score_drop_abs
                && i64::from(board.score).abs() >= i64::from(t)
            {
                continue;
            }
            // 学習側 `PsvEpochReader` と同じく drop 判定の後に score を飽和させる
            // (詰み stamp が clamp されて drop をすり抜けるのを防ぐ順序)。
            if let Some(c) = score_clamp_abs {
                board.score = board.score.clamp(-c, c);
            }
            let pushed = cur.push_decoded(&board)?;
            debug_assert!(pushed, "Batch::push_decoded refused below batch_size");
            cur_buckets.push(i32::from(bucket_mode.bucket_board(&board, num_buckets)));
            if cur.n_positions == batch_size {
                let full =
                    std::mem::replace(&mut cur, Batch::with_capacity(batch_size, feature_set));
                let full_buckets = std::mem::take(&mut cur_buckets);
                batches.push((full, full_buckets));
            }
        }

        if batches.is_empty() {
            return Err(io::Error::other(format!(
                "test data file {} has fewer than batch_size ({batch_size}) usable positions; \
                 held-out validation needs at least one full batch",
                path.display()
            )));
        }
        let n_positions = (batches.len() as u64) * (batch_size as u64);
        Ok(Self {
            batches,
            n_positions,
        })
    }

    /// 検証集合の position 総数。
    pub fn n_positions(&self) -> u64 {
        self.n_positions
    }

    /// 満タン検証 batch 数。
    pub fn n_batches(&self) -> usize {
        self.batches.len()
    }

    /// backend の forward + loss のみを全検証 batch に回し、平均 loss と
    /// sign-agreement accuracy を集計する。weight は一切更新しない。
    ///
    /// `wdl_lambda` / `loss` は training step と同じ値を渡す (test_loss を同
    /// superbatch の training loss と比較可能にするため)。
    pub fn evaluate<B: TrainerBackend>(
        &self,
        backend: &mut B,
        wdl_lambda: f32,
        loss: LossKind,
    ) -> io::Result<ValidationReport> {
        let mut sum_sq_err = 0.0_f64;
        let mut n_correct = 0_u64;
        let mut n_counted = 0_u64;
        for (batch, buckets) in &self.batches {
            let out = backend.validate_step(batch, buckets, wdl_lambda, loss)?;
            sum_sq_err += out.sum_sq_err;
            let (correct, counted) = sign_agreement(&out.net_output, batch);
            n_correct += correct;
            n_counted += counted;
        }
        // `HeldoutSet::load` が空集合を error にするため `n_positions >= batch_size > 0`。
        let mean_loss = sum_sq_err / self.n_positions as f64;
        let accuracy = if n_counted == 0 {
            f64::NAN
        } else {
            n_correct as f64 / n_counted as f64
        };
        Ok(ValidationReport {
            mean_loss,
            accuracy,
            n_positions: self.n_positions,
            n_counted,
        })
    }
}

/// model 出力の符号が実際の対局結果と一致した数を `(n_correct, n_counted)` で返す。
///
/// `net_output` と `batch.wdl` はともに **手番側 (side-to-move) 視点**で揃っている
/// (loss kernel が両者を blend する前提)。
///
/// - `net_output[i] > 0` を「手番側が有利と予測」、`batch.wdl[i] > 0.5` を「実際に
///   手番側が勝った」と解釈し、両者の bool が一致したら correct。
/// - 引き分け (`wdl == 0.5`) は符号が無いので分母 (`n_counted`) からも除外する。
///
/// `net_output` は `batch` の position 順に並ぶ per-position scalar。長さが
/// `batch.n_positions` 未満なら短い方までを見る (防御的)。
pub fn sign_agreement(net_output: &[f32], batch: &Batch) -> (u64, u64) {
    let n = batch.n_positions.min(net_output.len());
    let mut n_correct = 0_u64;
    let mut n_counted = 0_u64;
    for (&out, &wdl) in net_output[..n].iter().zip(&batch.wdl[..n]) {
        if wdl == 0.5 {
            continue; // 引き分けは符号が定義できないため除外
        }
        let predicted_win = out > 0.0;
        let actual_win = wdl > 0.5;
        if predicted_win == actual_win {
            n_correct += 1;
        }
        n_counted += 1;
    }
    (n_correct, n_counted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::FeatureSet;
    use std::path::PathBuf;

    fn test_spec() -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged.spec()
    }

    /// shogi-format crate test fixture (100 records × 40 bytes)。
    fn sample_psv_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/nnue-train has a parent dir")
            .join("shogi-format/tests/data/sample.psv")
    }

    #[test]
    #[ignore = "requires an external HCPE file"]
    fn heldout_set_loads_external_hcpe() {
        let path = std::env::var_os("TATARA_HCPE_CROSSCHECK")
            .map(PathBuf::from)
            .expect("set TATARA_HCPE_CROSSCHECK");
        let set = HeldoutSet::load(
            &path,
            16,
            None,
            None,
            128,
            &BucketMode::KingRank9,
            test_spec(),
            9,
        )
        .expect("load HCPE held-out data");
        assert_eq!(set.n_batches(), 8);
        assert_eq!(set.n_positions(), 128);
    }

    /// 任意の `wdl` ベクタで `n_positions` 件の `Batch` を作る (sign_agreement テスト用)。
    fn batch_with_wdl(wdl: &[f32]) -> Batch {
        let mut b = Batch::with_capacity(wdl.len().max(1), test_spec());
        b.n_positions = wdl.len();
        for (i, &w) in wdl.iter().enumerate() {
            b.wdl[i] = w;
        }
        b
    }

    #[test]
    fn sign_agreement_counts_matches_and_excludes_draws() {
        // wdl: Win, Loss, Draw, Win, Loss
        let batch = batch_with_wdl(&[1.0, 0.0, 0.5, 1.0, 0.0]);
        //          pred: +(win ✓) +(loss ✗) -(skip) -(win ✗) -(loss ✓)
        let net_output = [0.7_f32, 0.3, 0.9, -0.2, -0.5];
        let (correct, counted) = sign_agreement(&net_output, &batch);
        // 引き分け 1 件を除く 4 件が分母、うち一致は idx 0 と idx 4 の 2 件。
        assert_eq!(counted, 4);
        assert_eq!(correct, 2);
    }

    #[test]
    fn heldout_set_dispatches_kingrank9_without_progress_weights() {
        let path = sample_psv_path();
        let set = HeldoutSet::load(
            &path,
            8,
            None,
            None,
            8,
            &BucketMode::KingRank9,
            test_spec(),
            9,
        )
        .expect("load KingRank9 held-out set");

        let mut reader = PsvFileLoader::new(&path).expect("open sample PSV");
        let mut expected = Vec::new();
        for _ in 0..8 {
            let board = reader
                .next_psv()
                .expect("read sample PSV")
                .expect("sample record")
                .decode();
            expected.push(i32::from(shogi_features::kingrank9_bucket_board(&board)));
        }
        assert_eq!(set.batches[0].1, expected);
    }

    #[test]
    fn sign_agreement_all_draws_counts_nothing() {
        let batch = batch_with_wdl(&[0.5, 0.5, 0.5]);
        let net_output = [1.0_f32, -1.0, 0.0];
        assert_eq!(sign_agreement(&net_output, &batch), (0, 0));
    }

    #[test]
    fn sign_agreement_zero_output_predicts_loss() {
        // net_output == 0.0 は `> 0.0` が false なので「先手敗北」予測扱い。
        let batch = batch_with_wdl(&[1.0, 0.0]);
        let (correct, counted) = sign_agreement(&[0.0_f32, 0.0], &batch);
        assert_eq!(counted, 2);
        assert_eq!(correct, 1); // idx 1 (Loss) のみ一致
    }

    #[test]
    fn heldout_set_loads_full_batches_without_wrap() {
        // sample.psv は 100 records。batch_size 16 / test_positions 40 →
        // 切り上げ 3 batch (48 pos) を要求するが、100 records あるので wrap せず 3 batch。
        let progress = ShogiProgressKPAbs;
        let set = HeldoutSet::load(
            &sample_psv_path(),
            16,
            None,
            None,
            40,
            &progress,
            test_spec(),
            9,
        )
        .expect("load held-out set");
        assert_eq!(set.n_batches(), 3);
        assert_eq!(set.n_positions(), 48);
    }

    #[test]
    fn heldout_set_stops_at_eof_when_file_smaller_than_requested() {
        // test_positions を file 件数より大きく要求しても、EOF で打ち切られ
        // 満タン batch 分だけ (100 / 16 = 6 batch) になる (wrap で水増ししない)。
        let progress = ShogiProgressKPAbs;
        let set = HeldoutSet::load(
            &sample_psv_path(),
            16,
            None,
            None,
            100_000,
            &progress,
            test_spec(),
            9,
        )
        .expect("load held-out set");
        assert_eq!(set.n_batches(), 6);
        assert_eq!(set.n_positions(), 96);
    }

    #[test]
    fn heldout_set_score_clamp_saturates_batch_scores() {
        // sample.psv (実教師局面 = |score| > 10 を含む) を clamp 10 で読むと、
        // 全 batch の score が [-10, 10] に収まる。
        let progress = ShogiProgressKPAbs;
        let set = HeldoutSet::load(
            &sample_psv_path(),
            16,
            None,
            Some(10),
            96,
            &progress,
            test_spec(),
            9,
        )
        .expect("load held-out set");
        for (batch, _) in &set.batches {
            for bi in 0..batch.n_positions {
                assert!(
                    batch.score[bi].abs() <= 10.0,
                    "score {} exceeds clamp",
                    batch.score[bi]
                );
            }
        }
    }

    #[test]
    fn heldout_set_load_from_range_reads_tail_records() {
        // sample.psv は 100 records。末尾 30 records (offset 2800..4000) を
        // range 指定して読み、batch_size 16 で test_positions 30 → 切り上げ
        // 2 batch (= 32 pos 要求) を試す。range には 30 records しか無いので
        // EOF 打ち切りで満タン batch は 1 個 (16 pos)。
        let progress = ShogiProgressKPAbs;
        let set = HeldoutSet::load_from_range(
            &sample_psv_path(),
            2800,
            4000,
            16,
            None,
            None,
            30,
            &progress,
            test_spec(),
            9,
        )
        .expect("load tail range");
        assert_eq!(set.n_batches(), 1, "EOF で 2 batch 目は埋まらない");
        assert_eq!(set.n_positions(), 16);
    }

    #[test]
    fn heldout_set_load_from_range_errors_when_empty_range() {
        // 空 range (start == end) は 1 件も埋められず error。
        let progress = ShogiProgressKPAbs;
        let err = HeldoutSet::load_from_range(
            &sample_psv_path(),
            4000,
            4000,
            16,
            None,
            None,
            16,
            &progress,
            test_spec(),
            9,
        )
        .expect_err("empty range should error");
        assert!(
            err.to_string().contains("fewer than batch_size"),
            "got: {err}"
        );
    }

    #[test]
    fn heldout_set_errors_when_file_too_small_for_one_batch() {
        // batch_size 200 > sample.psv 100 records → 満タン batch を 1 個も作れず error。
        let progress = ShogiProgressKPAbs;
        let err = HeldoutSet::load(
            &sample_psv_path(),
            200,
            None,
            None,
            200,
            &progress,
            test_spec(),
            9,
        )
        .expect_err("too-small test file should error");
        assert!(
            err.to_string().contains("fewer than batch_size"),
            "got: {err}"
        );
    }
}
