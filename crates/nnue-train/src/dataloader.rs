//! PSV file → HalfKA_hm sparse batch dataloader (+ prefetch wrapper)。
//!
//! Stage 3 (EPIC #17) trainer の data 供給路。`PackedSfenValue` を `ShogiHalfKA_hm`
//! で sparse index 化し、`Batch` (`stm_indices` / `nstm_indices` / `score` /
//! `wdl` / `per_pos_norm`) にまとめる。Stage 3-8 trainer loop が GPU buffer
//! 転送前に本 dataloader から `Batch` を pull する。
//!
//! ## bullet 上流からの差分
//!
//! - bullet `crates/bullet_lib/src/value/dataloader.rs` の `ValueDataLoader<I,
//!   O, D, W>` (`bullet_compiler::TValue` / `OutputBuckets` / `LoadableDataType` /
//!   `WdlScheduler` を depend する trait 機構) は本リポでは使わず、**Stage 1
//!   `bins/progress_kpabs_train/src/host/batch.rs` 流儀の直接 struct
//!   実装** に簡素化 (Stage 1-1 / 3-1 / 3-4 と同じ bullet trait 削除ポリシー)
//! - bullet `value/loader.rs:301-315` で行う **data-layer WDL blend
//!   pre-compute** (`blend * result + (1-blend) * sigmoid(rscale * score)`) は
//!   本リポでは **行わない**: Stage 2-2 `fused_loss_wdl` kernel が GPU 側で
//!   blend を fuse するため、本 dataloader は `score` (raw cp) と `wdl`
//!   (game result {0, 0.5, 1}) を別 buffer に保持する
//! - bullet の `map_features_split` (asymmetric STM/NSTM Option emit) は
//!   ShogiHalfKA_hm では使用せず、`map_features` (symmetric (stm, nstm) emit) で
//!   STM/NSTM 同時 fill。-1 padding は bullet と同様 (Stage 2-6 `sparse_ft_forward`
//!   kernel の silent skip と整合)
//! - bullet の multi-thread prefetch (`bullet_trainer::run::dataloader` 経由) は
//!   本リポでは `std::thread::spawn` + `std::sync::mpsc::sync_channel` の minimal
//!   wrapper として `PrefetchedLoader` を提供 (prefetch depth は呼び出し側指定)

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use shogi_features::halfka_hm::ShogiHalfKA_hm;
use shogi_format::PackedSfenValue;

// =============================================================================
// Batch 構造体 (Stage 2-2 fused_loss_wdl + Stage 2-6 sparse_ft_forward 入力に整合)
// =============================================================================

/// 1 batch 分の HalfKA_hm sparse + score/wdl/norm。
///
/// - `stm_indices` / `nstm_indices`: shape `[batch_size, max_active]` を flatten
///   (row-major、`bi * max_active + j` で参照)。`-1` padding で未使用 slot を
///   埋める (Stage 2-6 `sparse_ft_forward` の silent-skip semantics と整合)
/// - `score`: raw cp (PackedSfenValue::score の i16 を f32 cast)
/// - `wdl`: game result を {0.0, 0.5, 1.0} に正規化 (`Loss=0 → 0.0`,
///   `Draw=1 → 0.5`, `Win=2 → 1.0`)
/// - `per_pos_norm`: batch averaging 用 weight (default 1.0、Stage 1
///   progress では `1/n_games` の game-relative norm、本 PR では trainer 側で
///   override 可能な値として保持)
/// - `n_positions`: 実際に詰めた数 (< batch_size の場合、末尾は uninitialised
///   ではなく zero/`-1` のまま、trainer 側で `n_positions` を見て効果範囲を制御)
#[derive(Clone, Debug)]
pub struct Batch {
    pub batch_size: usize,
    pub max_active: usize,
    pub stm_indices: Vec<i32>,
    pub nstm_indices: Vec<i32>,
    pub score: Vec<f32>,
    pub wdl: Vec<f32>,
    pub per_pos_norm: Vec<f32>,
    pub n_positions: usize,
}

impl Batch {
    /// `batch_size` × `max_active` の sparse 容量を持つ空 `Batch` を確保。
    /// 全 index は `-1` (padding)、score/wdl/norm は `0.0`。
    pub fn with_capacity(batch_size: usize, max_active: usize) -> Self {
        Self {
            batch_size,
            max_active,
            stm_indices: vec![-1; batch_size * max_active],
            nstm_indices: vec![-1; batch_size * max_active],
            score: vec![0.0; batch_size],
            wdl: vec![0.0; batch_size],
            per_pos_norm: vec![1.0; batch_size],
            n_positions: 0,
        }
    }

    /// 既存 `Batch` を再利用 (alloc 削減、`PsvFileLoader::fill_batch` 内部で
    /// 使われる)。全 slot を `-1` / `0.0` / `1.0` に reset する。
    ///
    /// 注: `PrefetchedLoader` の background loop では send 時に move されるため
    /// `reset()` 経由の reuse はできず、毎 iteration `Batch::with_capacity` を
    /// 新規 alloc している。alloc が hot path になった段階 (Stage 3-7/3-8 で
    /// 実 throughput 測定後) で `Clone` 経由 send / `Arc<Batch>` 化 / double-
    /// buffer 化等を検討する (本 PR では正しさ優先、性能 follow-up)。
    pub fn reset(&mut self) {
        for v in &mut self.stm_indices {
            *v = -1;
        }
        for v in &mut self.nstm_indices {
            *v = -1;
        }
        for v in &mut self.score {
            *v = 0.0;
        }
        for v in &mut self.wdl {
            *v = 0.0;
        }
        for v in &mut self.per_pos_norm {
            *v = 1.0;
        }
        self.n_positions = 0;
    }

    /// 1 position を batch に追加。`true` を返したら成功、`false` は batch 満杯。
    /// `ShogiHalfKA_hm::map_features` で sparse index を slot に fill (`-1` の
    /// 残りは padding として保持)。
    pub fn push(&mut self, pos: &PackedSfenValue) -> bool {
        if self.n_positions >= self.batch_size {
            return false;
        }

        let bi = self.n_positions;
        let row_off = bi * self.max_active;

        let mut j = 0usize;
        ShogiHalfKA_hm.map_features(pos, |stm_idx, nstm_idx| {
            if j < self.max_active {
                self.stm_indices[row_off + j] = stm_idx as i32;
                self.nstm_indices[row_off + j] = nstm_idx as i32;
                j += 1;
            }
            // overflow は silent skip。bullet 上流は `assert!(j_stm <=
            // max_active)` で panic するが、本リポは
            // `MAX_ACTIVE_FEATURES = 40` (合法局面固定) + `-1` padding の
            // defensive 設計で許容 (実 PSV では到達不能)。
        });

        // score / wdl / norm
        self.score[bi] = f32::from(pos.score());
        // bullet `loader::GameResult::{Loss=0, Draw=1, Win=2}` を {0.0, 0.5, 1.0}
        // に正規化 (bullet `loader.rs:312` と同型)。
        //
        // 注意: `PackedSfenValue::game_result()` は **raw i8** で `{-1, 0, +1}`
        // (Loss/Draw/Win、PSV wire 形式) を返すため、そのまま `/ 2.0` すると
        // Draw=0 と Win=1 が誤って 0.0 と 0.5 に圧縮される (Codex review #61
        // 修正、`as u8 / 2.0` 直訳の罠)。
        // 正しくは `pos.result()` (`packed_sfen.rs:473`、`GameResult` enum) を
        // 経由し、`as u8` で {0, 1, 2} に正規化してから `/ 2.0` で {0.0, 0.5, 1.0}
        // を得る。
        self.wdl[bi] = f32::from(pos.result() as u8) / 2.0;
        // per_pos_norm はデフォルト 1.0 (with_capacity 時に初期化済)。

        self.n_positions += 1;
        true
    }

    /// 詰めた position 数を返す (`n_positions` と同値)。
    pub fn len(&self) -> usize {
        self.n_positions
    }

    /// `n_positions == 0` 判定。
    pub fn is_empty(&self) -> bool {
        self.n_positions == 0
    }
}

// =============================================================================
// PsvFileLoader (single-threaded、Stage 1 progress と同流儀)
// =============================================================================

/// PSV file (PackedSfenValue × N、各 40 bytes 固定) を 1 record ずつ stream 読み。
pub struct PsvFileLoader {
    reader: BufReader<File>,
    eof: bool,
    path: PathBuf,
}

impl PsvFileLoader {
    /// `path` の PSV file を open。
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let reader = BufReader::new(File::open(&path_buf)?);
        Ok(Self {
            reader,
            eof: false,
            path: path_buf,
        })
    }

    /// 元 path への参照 (debug 用)。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 1 PSV record を読む。EOF なら `Ok(None)`、partial read は
    /// `UnexpectedEof` で panic 相当の io::Error を返す。
    pub fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.eof {
            return Ok(None);
        }
        let mut buf = [0u8; 40];
        match self.reader.read(&mut buf)? {
            0 => {
                self.eof = true;
                Ok(None)
            }
            40 => {
                let mut psv = PackedSfenValue::default();
                psv.as_bytes_mut().copy_from_slice(&buf);
                Ok(Some(psv))
            }
            n => {
                // partial read — 残りを fill するまで blocking read。
                let mut total = n;
                while total < 40 {
                    let got = self.reader.read(&mut buf[total..])?;
                    if got == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("partial PSV record: got {total} of 40 bytes"),
                        ));
                    }
                    total += got;
                }
                let mut psv = PackedSfenValue::default();
                psv.as_bytes_mut().copy_from_slice(&buf);
                Ok(Some(psv))
            }
        }
    }

    /// `batch` を batch_size まで PSV で埋める。詰めた件数を返す (EOF で
    /// 0 → end-of-stream)。
    pub fn fill_batch(&mut self, batch: &mut Batch) -> io::Result<usize> {
        batch.reset();
        loop {
            if batch.n_positions >= batch.batch_size {
                break;
            }
            match self.next_psv()? {
                Some(psv) => {
                    let ok = batch.push(&psv);
                    debug_assert!(ok, "batch.push should not refuse below batch_size");
                }
                None => break,
            }
        }
        Ok(batch.n_positions)
    }
}

// =============================================================================
// PrefetchedLoader (multi-thread prefetch、minimum wrapper)
// =============================================================================

/// `PsvFileLoader` を別 thread で先読み、main thread が `next_batch()` で
/// 取得する形の wrapper。`prefetch_depth` で channel 容量を制御。
///
/// 現状 background loop は毎 iteration `Batch::with_capacity` を新規 alloc する
/// 設計 (channel 経由で send=move、original を `reset()` reuse できないため)。
/// alloc が trainer ホットパスでボトルネックになった段階で `Clone` 経由 send /
/// `Arc<Batch>` 化 / double-buffer 化等を Stage 3-7/3-8 で検討する。
pub struct PrefetchedLoader {
    rx: mpsc::Receiver<io::Result<Batch>>,
    _handle: thread::JoinHandle<()>,
}

impl PrefetchedLoader {
    /// 指定 path から PSV を読み、`batch_size` × `max_active` の sparse batch
    /// として生成。`prefetch_depth` は背景 thread が main thread を先読みする
    /// 深さ (`sync_channel(prefetch_depth)` の bound)。
    pub fn spawn<P: AsRef<Path>>(
        path: P,
        batch_size: usize,
        max_active: usize,
        prefetch_depth: usize,
    ) -> io::Result<Self> {
        let loader = PsvFileLoader::new(path)?;
        let (tx, rx) = mpsc::sync_channel::<io::Result<Batch>>(prefetch_depth.max(1));

        let handle = thread::spawn(move || {
            let mut loader = loader;
            loop {
                // 毎ループ新規 alloc: mpsc::sync_channel が所有権を main thread に
                // 移すため、background 側で `Batch::reset()` 再利用は不可。
                // `prefetch_depth + 1` 個の batch を pool 化して return channel
                // で回す ring buffer 設計は Stage 3-7/3-8 trainer integration で
                // 必要に応じて追加 (本 PR scope では single-batch alloc 都度)。
                let mut batch = Batch::with_capacity(batch_size, max_active);
                match loader.fill_batch(&mut batch) {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        if tx.send(Ok(batch)).is_err() {
                            break; // receiver dropped
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
            // tx は drop で channel close → receiver 側 None。
        });

        Ok(Self {
            rx,
            _handle: handle,
        })
    }

    /// 次の `Batch` を取得。返り値:
    /// - `Ok(Some(batch))`: 正常 batch
    /// - `Ok(None)`: end-of-stream (EOF or thread 終了)
    /// - `Err(e)`: background thread が io::Error を伝搬
    pub fn next_batch(&mut self) -> io::Result<Option<Batch>> {
        match self.rx.recv() {
            Ok(Ok(batch)) => Ok(Some(batch)),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None), // channel closed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::halfka_hm::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES};
    use std::path::PathBuf;

    /// shogi-format crate test fixture (100 records × 40 bytes = 4000 bytes)。
    fn sample_psv_path() -> PathBuf {
        let dir = env!("CARGO_MANIFEST_DIR");
        // crates/nnue-train/Cargo.toml から相対で shogi-format/tests/data/sample.psv を参照。
        PathBuf::from(dir)
            .parent()
            .unwrap()
            .join("shogi-format/tests/data/sample.psv")
    }

    #[test]
    fn batch_with_capacity_initializes_padding_and_defaults() {
        let batch = Batch::with_capacity(4, 8);
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.max_active, 8);
        assert_eq!(batch.stm_indices.len(), 32);
        assert!(batch.stm_indices.iter().all(|&i| i == -1));
        assert!(batch.nstm_indices.iter().all(|&i| i == -1));
        assert!(batch.score.iter().all(|&s| s == 0.0));
        assert!(batch.wdl.iter().all(|&w| w == 0.0));
        assert!(batch.per_pos_norm.iter().all(|&n| n == 1.0));
        assert_eq!(batch.n_positions, 0);
        assert!(batch.is_empty());
    }

    #[test]
    fn psv_file_loader_reads_first_record() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).expect("open sample.psv");
        let psv = loader.next_psv().unwrap().expect("at least 1 record");
        assert_eq!(psv.as_bytes().len(), 40);
    }

    #[test]
    fn psv_file_loader_streams_until_eof() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut n = 0;
        while loader.next_psv().unwrap().is_some() {
            n += 1;
        }
        // sample.psv は 4000 bytes / 40 = 100 records。
        assert_eq!(n, 100);
    }

    #[test]
    fn fill_batch_indices_within_halfka_dim_or_padding() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(8, MAX_ACTIVE_FEATURES);
        let n = loader.fill_batch(&mut batch).unwrap();
        assert_eq!(n, 8);
        assert_eq!(batch.n_positions, 8);
        for (i, &idx) in batch.stm_indices.iter().enumerate() {
            assert!(
                idx == -1 || (0..HALFKA_HM_DIMENSIONS as i32).contains(&idx),
                "stm_indices[{i}] = {idx} は -1 padding か [0, HALFKA_HM_DIMENSIONS) の範囲"
            );
        }
        for (i, &idx) in batch.nstm_indices.iter().enumerate() {
            assert!(
                idx == -1 || (0..HALFKA_HM_DIMENSIONS as i32).contains(&idx),
                "nstm_indices[{i}] = {idx}"
            );
        }
        // 少なくとも 1 position は両玉ありで active features > 0 のはず。
        let total_active = batch.stm_indices.iter().filter(|&&i| i >= 0).count();
        assert!(total_active > 0, "全 padding は異常 (sample.psv は実局面)");
    }

    #[test]
    fn fill_batch_wdl_is_in_valid_range() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(4, MAX_ACTIVE_FEATURES);
        loader.fill_batch(&mut batch).unwrap();
        for (i, &w) in batch.wdl.iter().enumerate() {
            assert!(
                w == 0.0 || w == 0.5 || w == 1.0,
                "wdl[{i}] = {w} は {{0.0, 0.5, 1.0}} のいずれか"
            );
        }
    }

    #[test]
    fn fill_batch_wdl_covers_loss_and_win_with_correct_values() {
        // sample.psv は Loss=50 / Win=50 (Draw を含まない) という偏った fixture。
        // raw `game_result()` 直訳 (旧バグ: Win → 0.5) を回帰検出するため、
        // `wdl == 1.0` が少なくとも 1 件存在することを確認 (`pos.result()` 経由の
        // 正しい正規化なら Win → 1.0)。
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(100, MAX_ACTIVE_FEATURES);
        loader.fill_batch(&mut batch).unwrap();
        let win_count = batch.wdl.iter().filter(|&&w| w == 1.0).count();
        let loss_count = batch.wdl.iter().filter(|&&w| w == 0.0).count();
        assert!(
            win_count > 0,
            "sample.psv は Win 局面を含むはず (raw game_result 直訳の bug 回帰検出)"
        );
        assert!(loss_count > 0, "sample.psv は Loss 局面も含むはず");
        // Loss + Win + Draw = 100、合計 wdl sum = win_count * 1.0 + draw_count * 0.5
        assert_eq!(
            win_count + loss_count,
            100,
            "sample.psv 100 records は Draw なし"
        );
    }

    #[test]
    fn fill_batch_consumes_stream_partial_at_eof() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(150, MAX_ACTIVE_FEATURES);
        let n = loader.fill_batch(&mut batch).unwrap();
        // sample.psv の 100 records しかない → 100 で打ち切り。
        assert_eq!(n, 100);
        assert_eq!(batch.n_positions, 100);
        // 残り 150-100=50 slot は padding のまま (-1 / 0.0 / 1.0)。
        for j in 100 * MAX_ACTIVE_FEATURES..150 * MAX_ACTIVE_FEATURES {
            assert_eq!(batch.stm_indices[j], -1);
        }
        for j in 100..150 {
            assert_eq!(batch.score[j], 0.0);
            assert_eq!(batch.wdl[j], 0.0);
        }
    }

    #[test]
    fn batch_push_returns_false_when_full() {
        let mut batch = Batch::with_capacity(2, MAX_ACTIVE_FEATURES);
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let psv1 = loader.next_psv().unwrap().unwrap();
        let psv2 = loader.next_psv().unwrap().unwrap();
        let psv3 = loader.next_psv().unwrap().unwrap();
        assert!(batch.push(&psv1));
        assert!(batch.push(&psv2));
        assert!(!batch.push(&psv3), "3 件目は batch_size=2 で reject");
        assert_eq!(batch.n_positions, 2);
    }

    #[test]
    fn batch_reset_zeros_state() {
        let mut batch = Batch::with_capacity(4, MAX_ACTIVE_FEATURES);
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        loader.fill_batch(&mut batch).unwrap();
        assert_eq!(batch.n_positions, 4);
        batch.reset();
        assert_eq!(batch.n_positions, 0);
        assert!(batch.stm_indices.iter().all(|&i| i == -1));
        assert!(batch.score.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn prefetched_loader_streams_sample_psv() {
        let mut loader =
            PrefetchedLoader::spawn(sample_psv_path(), 8, MAX_ACTIVE_FEATURES, 2).unwrap();
        let mut total = 0;
        while let Some(batch) = loader.next_batch().unwrap() {
            total += batch.n_positions;
        }
        // sample.psv 100 records / batch_size=8 → 12 full batch + 1 partial (4)
        // = 13 batch、合計 100 positions。
        assert_eq!(total, 100);
    }

    #[test]
    fn prefetched_loader_handles_small_prefetch_depth() {
        // prefetch_depth=0 は内部で .max(1) で 1 に正規化。
        let mut loader =
            PrefetchedLoader::spawn(sample_psv_path(), 4, MAX_ACTIVE_FEATURES, 0).unwrap();
        let first = loader.next_batch().unwrap().expect("at least 1 batch");
        assert_eq!(first.n_positions, 4);
    }
}
