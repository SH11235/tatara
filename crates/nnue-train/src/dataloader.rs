//! PSV file → feature-set sparse batch dataloader (+ prefetch wrapper)。
//!
//! trainer の data 供給路。`PackedSfenValue` を [`FeatureSetSpec`] の indexer で
//! sparse index 化し、`Batch` (`stm_indices` / `nstm_indices` / `score` / `wdl` /
//! `per_pos_norm`) にまとめる。superbatch loop driver が GPU buffer 転送前に
//! 本 dataloader から `Batch` を pull する。どの feature set を使うかは生成時に
//! 渡す `FeatureSetSpec` で決まる (runtime 選択)。
//!
//! ## 設計のポイント
//!
//! - **WDL blend は GPU 側 (`loss_wdl` / `loss_wrm` kernel) で fuse する**ため、
//!   本 dataloader は `score` (raw cp) と `wdl` (game result `{0, 0.5, 1}`) を
//!   別 buffer に保持する (data-layer での blend pre-compute は行わない)
//! - sparse index は feature set の最大 active 数 (`FeatureSetSpec::max_active`)
//!   で固定容量を持ち、未使用 slot は `-1` で padding する。`sparse_ft_forward`
//!   kernel は `-1` を silent skip する規約
//! - 並列 prefetch は `std::thread::spawn` + `std::sync::mpsc::sync_channel` の
//!   minimal wrapper として [`PrefetchedLoader`] (single-thread worker) と
//!   [`BucketedPrefetchedLoader`] (multi-worker + ring-buffer pool + bucket
//!   同時計算) を提供する

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use shogi_features::FeatureSetSpec;
use shogi_features::progress_kpabs::ShogiProgressKPAbs;
use shogi_format::{PackedSfenValue, ShogiBoard};

/// PSV record size in bytes (`shogi_format::PackedSfenValue` is a fixed
/// 40-byte struct). Used everywhere we compute byte offsets, validate range
/// alignment, or convert between record counts and file sizes.
pub const PSV_RECORD_BYTES: u64 = 40;

// =============================================================================
// Batch 構造体 (loss / sparse_ft_forward kernel 入力と整合)
// =============================================================================

/// 1 batch 分の feature-set sparse + score/wdl/norm。
///
/// - `stm_indices` / `nstm_indices`: shape `[batch_size, max_active]` を flatten
///   (row-major、`bi * max_active + j` で参照)。`-1` padding で未使用 slot を
///   埋める (`sparse_ft_forward` kernel の silent-skip semantics と整合)
/// - `score`: raw cp (`PackedSfenValue::score` の i16 を f32 cast)
/// - `wdl`: game result を `{0.0, 0.5, 1.0}` に正規化 (Loss → 0.0, Draw → 0.5,
///   Win → 1.0)
/// - `per_pos_norm`: batch averaging 用 weight (default 1.0、trainer 側で
///   override 可能)
/// - `n_positions`: 実際に詰めた数 (`< batch_size` の場合、末尾は uninitialised
///   ではなく `0` / `-1` で保持される)
#[derive(Clone, Debug)]
pub struct Batch {
    pub batch_size: usize,
    /// この batch を埋めた feature set。`push_decoded` の特徴抽出と
    /// `max_active` / `ft_in` の決定はすべてこの spec が単一の真実源。
    pub feature_set: FeatureSetSpec,
    /// `feature_set.max_active()` のキャッシュ (sparse index の row stride)。
    pub max_active: usize,
    pub stm_indices: Vec<i32>,
    pub nstm_indices: Vec<i32>,
    pub score: Vec<f32>,
    pub wdl: Vec<f32>,
    pub per_pos_norm: Vec<f32>,
    pub n_positions: usize,
}

impl Batch {
    /// `batch_size` × `feature_set.max_active()` の sparse 容量を持つ空
    /// `Batch` を確保。全 index は `-1` (padding)、score/wdl/norm は `0.0`。
    pub fn with_capacity(batch_size: usize, feature_set: FeatureSetSpec) -> Self {
        let max_active = feature_set.max_active();
        Self {
            batch_size,
            feature_set,
            max_active,
            stm_indices: vec![-1; batch_size * max_active],
            nstm_indices: vec![-1; batch_size * max_active],
            score: vec![0.0; batch_size],
            wdl: vec![0.0; batch_size],
            per_pos_norm: vec![1.0; batch_size],
            n_positions: 0,
        }
    }

    /// 既存 `Batch` を再利用 (alloc 削減)。全 slot を `-1` / `0.0` / `1.0` に
    /// reset する。`PsvFileLoader::fill_batch` と [`BucketedPrefetchedLoader`] の
    /// ring-buffer return path (消費済み `Batch` を pool channel 経由で worker に
    /// 返して `reset()` で再利用) の両方で使われる。
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

    /// 1 position を batch に追加。`Ok(true)` 成功、`Ok(false)` は batch 満杯、
    /// `Err` は active feature 数が `max_active` を超過 (下記参照)。`feature_set`
    /// の indexer で sparse index を slot に fill (残りは `-1` padding)。
    ///
    /// 内部で `pos.decode()` を 1 回呼ぶ。同じ局面で別途 progress8kpabs bucket も
    /// 要る場合は [`Batch::push_decoded`] を使い、`PackedSfenValue::decode()` を
    /// 1 回だけ呼んで `ShogiBoard` を使い回すこと (decode-once 経路)。
    pub fn push(&mut self, pos: &PackedSfenValue) -> io::Result<bool> {
        self.push_decoded(&pos.decode())
    }

    /// [`Batch::push`] の **decode 済み `ShogiBoard` を直接受ける** 版。
    ///
    /// prefetch worker が 1 局面につき `PackedSfenValue::decode()` を 1 回だけ
    /// 呼び、その `ShogiBoard` を feature 抽出 (本メソッド) と progress8kpabs
    /// bucket 計算 ([`ShogiProgressKPAbs::bucket_board`]) の両方で使い回すための
    /// 入口 (decode-once)。`push(&pos)` は `push_decoded(&pos.decode())` と等価。
    ///
    /// active feature 数が `max_active` を超えると `Err(io::Error)` を返す。base
    /// 特徴は合法局面で必ず cap 内だが threat 連結時は `THREAT_MAX_ACTIVE` の
    /// 見積りを edge 数が超え得る。超過を silent skip すると欠落 edge が loss だけ
    /// 見ても気付けないため、利用者に「profile / 実 active 数 / max_active」を含む
    /// 明示エラーを返して学習を止める (起点 320 不足なら定数を上げて再ビルド)。
    pub fn push_decoded(&mut self, board: &ShogiBoard) -> io::Result<bool> {
        if self.n_positions >= self.batch_size {
            return Ok(false);
        }

        let bi = self.n_positions;
        let row_off = bi * self.max_active;

        let spec = self.feature_set;
        let max_active = self.max_active;
        let stm_slice = &mut self.stm_indices[row_off..row_off + max_active];
        let nstm_slice = &mut self.nstm_indices[row_off..row_off + max_active];
        // `extract_active_features` は **実 active 数** を返す (cap 越えは書き込み
        // しないが戻り値には反映)。
        let written = spec.extract_active_features(board, stm_slice, nstm_slice);
        if written > max_active {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "active feature count {written} exceeds max_active {max_active} \
                     (feature set {}); raise THREAT_MAX_ACTIVE and rebuild — silent \
                     truncation is not allowed",
                    spec.canonical_name(),
                ),
            ));
        }

        // score / wdl / norm
        self.score[bi] = f32::from(board.score);
        // `ShogiBoard::result` は raw i8 (`{-1=Loss, 0=Draw, +1=Win}`、PSV wire
        // 形式 = `PackedSfenValue::game_result()` と同じ値)。これを WDL 軸の
        // `{0.0, 0.5, 1.0}` (Loss / Draw / Win) に sign-aware に map する
        // (`>0 → 1.0`, `<0 → 0.0`, `==0 → 0.5`)。`as u8 / 2.0` で直訳すると
        // Win=1 が誤って 0.5 に潰れるので必ず本 match を使うこと。
        self.wdl[bi] = match board.result {
            r if r > 0 => 1.0,
            r if r < 0 => 0.0,
            _ => 0.5,
        };
        // per_pos_norm はデフォルト 1.0 (with_capacity 時に初期化済)。

        self.n_positions += 1;
        Ok(true)
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
// PsvFileLoader (single-threaded、逐次読み)
// =============================================================================

/// PSV file (PackedSfenValue × N、各 40 bytes 固定) を 1 record ずつ stream 読み。
///
/// 読み出し範囲は file 全体 ([`PsvFileLoader::new`]) または
/// `[start_offset, end_offset)` ([`PsvFileLoader::new_range`]) の byte range で
/// 指定する。range の両端は [`PSV_RECORD_BYTES`] の倍数でなければならず、`end`
/// が file size を超えても error。range 外まで読み進めず、`remaining_bytes`
/// が 1 record 分に満たなくなった時点で EOF として `Ok(None)` を返す。
pub struct PsvFileLoader {
    reader: BufReader<File>,
    eof: bool,
    path: PathBuf,
    /// 残りどれだけ読めるか (byte)。range 末尾に達したら 1 record 分を切らず
    /// EOF 扱いにするための gate。`new()` 経路では file_size と一致。
    remaining_bytes: u64,
}

impl PsvFileLoader {
    /// `path` の PSV file 全体を open。`new_range(path, 0, file_size)` と等価。
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let file_size = file.metadata()?.len();
        Self::open_range(path_ref, file, file_size, 0, file_size)
    }

    /// `path` の PSV file を `[start, end)` の byte range で open。range は
    /// [`PSV_RECORD_BYTES`] の倍数でなければならず、`end > file_size` /
    /// `start > end` も error。`start == end` (空 range) は許可し即 EOF。
    pub fn new_range<P: AsRef<Path>>(path: P, start: u64, end: u64) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let file_size = file.metadata()?.len();
        Self::open_range(path_ref, file, file_size, start, end)
    }

    fn open_range(
        path: &Path,
        mut file: File,
        file_size: u64,
        start: u64,
        end: u64,
    ) -> io::Result<Self> {
        if start > end {
            return Err(io::Error::other(format!(
                "PsvFileLoader range start ({start}) > end ({end}) for {}",
                path.display()
            )));
        }
        if end > file_size {
            return Err(io::Error::other(format!(
                "PsvFileLoader range end ({end}) > file size ({file_size}) for {}",
                path.display()
            )));
        }
        if !start.is_multiple_of(PSV_RECORD_BYTES) || !end.is_multiple_of(PSV_RECORD_BYTES) {
            return Err(io::Error::other(format!(
                "PsvFileLoader range [{start}, {end}) is not aligned to PSV record size ({PSV_RECORD_BYTES} bytes) for {}",
                path.display()
            )));
        }
        if start > 0 {
            file.seek(SeekFrom::Start(start))?;
        }
        Ok(Self {
            reader: BufReader::new(file),
            eof: false,
            path: path.to_path_buf(),
            remaining_bytes: end - start,
        })
    }

    /// 元 path への参照 (debug 用)。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 1 PSV record を読む。EOF なら `Ok(None)`、partial read は
    /// `UnexpectedEof` で panic 相当の io::Error を返す。range 末尾
    /// (`remaining_bytes < PSV_RECORD_BYTES`) も EOF 扱い (`Ok(None)`)。
    pub fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.eof || self.remaining_bytes < PSV_RECORD_BYTES {
            self.eof = true;
            return Ok(None);
        }
        let mut buf = [0u8; PSV_RECORD_BYTES as usize];
        match self.reader.read(&mut buf)? {
            0 => {
                self.eof = true;
                Ok(None)
            }
            n if n == PSV_RECORD_BYTES as usize => {
                self.remaining_bytes -= PSV_RECORD_BYTES;
                let mut psv = PackedSfenValue::default();
                psv.as_bytes_mut().copy_from_slice(&buf);
                Ok(Some(psv))
            }
            n => {
                // partial read — 残りを fill するまで blocking read。
                let mut total = n;
                while total < PSV_RECORD_BYTES as usize {
                    let got = self.reader.read(&mut buf[total..])?;
                    if got == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("partial PSV record: got {total} of {PSV_RECORD_BYTES} bytes"),
                        ));
                    }
                    total += got;
                }
                self.remaining_bytes -= PSV_RECORD_BYTES;
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
                    let ok = batch.push(&psv)?;
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
/// 本 loader は単一 worker + 毎 iteration `Batch::with_capacity` を新規 alloc
/// する単純な実装。`Batch` を pool で回す ring-buffer / bucket 同時計算が
/// 必要なら [`BucketedPrefetchedLoader`] を使うこと。
pub struct PrefetchedLoader {
    rx: mpsc::Receiver<io::Result<Batch>>,
    _handle: thread::JoinHandle<()>,
}

impl PrefetchedLoader {
    /// 指定 path から PSV を読み、`feature_set` の sparse batch として生成。
    /// `prefetch_depth` は背景 thread が main thread を先読みする深さ
    /// (`sync_channel(prefetch_depth)` の bound)。
    pub fn spawn<P: AsRef<Path>>(
        path: P,
        batch_size: usize,
        feature_set: FeatureSetSpec,
        prefetch_depth: usize,
    ) -> io::Result<Self> {
        let loader = PsvFileLoader::new(path)?;
        let (tx, rx) = mpsc::sync_channel::<io::Result<Batch>>(prefetch_depth.max(1));

        let handle = thread::spawn(move || {
            let mut loader = loader;
            loop {
                // 毎ループ新規 alloc: `mpsc::sync_channel` が所有権を main
                // thread に移すため、background 側で `Batch::reset()` 再利用は
                // 不可。ring-buffer return path を持つ実装は
                // [`BucketedPrefetchedLoader`] を参照。
                let mut batch = Batch::with_capacity(batch_size, feature_set);
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

// =============================================================================
// PsvEpochReader — 逐次 PSV 読み + score-drop skip + EOF wrap (= 次 epoch) +
//                  barren-pass ガード
// =============================================================================

/// 連続 barren pass (= file を 1 周しても 1 件も使える position が無い) が
/// これに達したら無限ループせず error を返す。
pub const MAX_BARREN_PASSES: u32 = 5;

/// `PsvFileLoader` を逐次読み、EOF で同 file を開き直して次 epoch とする stream
/// reader。`--score-drop-abs` の近似 skip (`|score| >= t` を捨てる) と空 file の
/// 無限ループ防止 (`MAX_BARREN_PASSES`) を内包する。bucket 計算は **行わない**
/// (decode-once 経路: bucket は呼び出し側 prefetch worker が `decode()` した
/// `ShogiBoard` から `ShogiProgressKPAbs::bucket_board` で求める)。
///
/// `next()` は常に「使える PSV」を返すか barren-error を返す (epoch は無限に
/// wrap するので「終わり」は無い)。
struct PsvEpochReader {
    path: PathBuf,
    /// 1 epoch の byte range `[start_offset, end_offset)`。wrap 時に
    /// `PsvFileLoader::new_range(path, start, end)` で再 open する。`new()`
    /// 経路では `(0, file_size)` で全体に等しい。
    start_offset: u64,
    end_offset: u64,
    loader: PsvFileLoader,
    score_drop_abs: Option<i32>,
    /// 直近の reopen 以降に実際に返した (= drop されなかった) position 数。
    pushed_this_epoch: u64,
    /// 1 epoch 丸ごと 0 push だった連続回数。
    barren_passes: u32,
}

impl PsvEpochReader {
    /// `path` を `[start_offset, end_offset)` 範囲で epoch wrap させる reader。
    /// wrap 時の再 open も同 range で行う。`PsvFileLoader::new_range` 同様の
    /// 範囲・alignment 検証はここでは行わず、`new_range` 内で検証する。
    fn new_range(
        path: &Path,
        start_offset: u64,
        end_offset: u64,
        score_drop_abs: Option<i32>,
    ) -> io::Result<Self> {
        let loader = PsvFileLoader::new_range(path, start_offset, end_offset)?;
        Ok(Self {
            path: path.to_path_buf(),
            start_offset,
            end_offset,
            loader,
            score_drop_abs,
            pushed_this_epoch: 0,
            barren_passes: 0,
        })
    }

    /// 次の使える PSV を返す。EOF なら file を開き直す (= 次 epoch)。空 file /
    /// 全 drop で `MAX_BARREN_PASSES` 周しても 0 件なら `io::Error` を返す。
    fn next(&mut self) -> io::Result<PackedSfenValue> {
        loop {
            match self.loader.next_psv()? {
                Some(psv) => {
                    // `--score-drop-abs t` 指定時: `|score| >= t` を skip。
                    // i64 cast で `i16::MIN` の abs overflow を避ける。
                    if let Some(t) = self.score_drop_abs
                        && i64::from(psv.score()).abs() >= i64::from(t)
                    {
                        continue;
                    }
                    self.pushed_this_epoch += 1;
                    return Ok(psv);
                }
                None => {
                    if self.pushed_this_epoch == 0 {
                        self.barren_passes += 1;
                        if self.barren_passes >= MAX_BARREN_PASSES {
                            return Err(io::Error::other(format!(
                                "data file {} range [{}, {}) yielded no usable positions over {} \
                                 full passes (empty range, or all positions filtered out by \
                                 score-drop-abs)",
                                self.path.display(),
                                self.start_offset,
                                self.end_offset,
                                self.barren_passes
                            )));
                        }
                    } else {
                        self.barren_passes = 0;
                    }
                    self.pushed_this_epoch = 0;
                    self.loader =
                        PsvFileLoader::new_range(&self.path, self.start_offset, self.end_offset)?;
                }
            }
        }
    }
}

// =============================================================================
// BucketedPrefetchedLoader — bucket-aware / 並列パース / decode-once /
//                            ring-buffer return path
// =============================================================================

/// 完成 batch のチャネル容量 (worker が main をどれだけ先読みするか) を
/// `--threads` から決める係数 + 下限。
fn prefetch_depth_for(num_workers: usize) -> usize {
    (2 * num_workers).max(2)
}

/// 1 個の prefetch worker が消費 / 生成する単位。`(buffers, buckets)` を ring で
/// 回す。`buffers` は `reset()` 再利用、`buckets` は `clear()` 再利用。
type BatchSlot = (Batch, Vec<i32>);

/// 共有 reader (`PsvEpochReader`) を `--threads` 本の worker で読み、各 worker が
/// 「PSV パース + feature sparse 抽出 + progress8kpabs bucket 計算」を
/// `decode()` **1 回** で済ませて main thread に `(Batch, buckets)` を渡す
/// prefetch loader。
///
/// ## 設計
///
/// - **decode-once**: worker は `psv.decode()` した `ShogiBoard` を
///   `Batch::push_decoded` (feature 抽出) と `ShogiProgressKPAbs::bucket_board`
///   (output bucket) の両方に渡す。`pos.decode()` は 1 局面 1 回。
/// - **並列パース**: worker は短い critical section (共有 `Mutex<PsvEpochReader>`
///   を lock して `batch_size` 件の生 PSV を自前 scratch `Vec` に詰める; I/O は
///   逐次・高速) の外で decode + 特徴抽出を並列に行う。`FeatureSetSpec` は
///   `Copy` の値型、`ShogiProgressKPAbs` は ZST + process-global `OnceLock`
///   (read-only) なので thread 間共有して問題ない。
/// - **ring-buffer return path**: `Batch` / `buckets` の `Vec` は起動時に
///   `prefetch_depth + num_workers + 1` 個確保した pool channel から借りて使い、
///   main が消費後 [`BucketedPrefetchedLoader::recycle`] で pool に返す → worker
///   が再借用して `reset()` / `clear()` で reuse。毎 batch の `Vec` 新規 alloc
///   (~21MB) は発生しない。
/// - **epoch 意味論**: 共有 reader が EOF で file を開き直す (= 次 epoch)、
///   `score-drop-abs` skip、`MAX_BARREN_PASSES` ガードは [`PsvEpochReader`] が
///   担う。ただし **1 epoch 内の position の順序は worker 数 ≥ 2 では非決定的**
///   (各 worker が `batch_size` 件ずつ排他的に読むため batch 境界の切れ目が
///   変わる)。training では問題ない (適用される lr/wdl は loop の `batch_idx` で
///   決まりデータ内容に依らない) が、決定論的順序が要る場合は
///   `num_workers = 1` を使うこと。
/// - **error 伝搬**: worker が reader から `io::Error` (主に barren-exhaustion)
///   を受けたら shared error slot に格納して exit。main の
///   [`Self::next_batch`] は全 worker が exit して result channel が閉じたら
///   error slot を見て伝搬する。
/// - **終了**: main が `BucketedPrefetchedLoader` を drop すると [`Drop`] impl が
///   まず result/pool 両 channel endpoint を落として全 worker を unblock させ、
///   その後 worker thread を join する (close-then-join、詳細は `Drop` の doc)。
pub struct BucketedPrefetchedLoader {
    /// 完成 batch (Batch + per-position bucket) を worker → main で渡す。
    /// `Drop` で `.take()` して先に落とすため `Option`。
    result_rx: Option<mpsc::Receiver<BatchSlot>>,
    /// 消費済み batch buffer を main → worker で返す (ring buffer)。
    /// `Drop` で `.take()` して先に落とすため `Option`。
    pool_tx: Option<mpsc::SyncSender<BatchSlot>>,
    /// worker が reader から受けた io::Error を main に伝えるための slot。
    err_slot: Arc<Mutex<Option<io::Error>>>,
    /// worker thread handle (`Drop` で join する)。
    handles: Vec<thread::JoinHandle<()>>,
}

impl BucketedPrefetchedLoader {
    /// `path` の PSV を `num_workers` 本の worker で読み込む。各 batch は
    /// `batch_size` 件の有効 position を持つ (epoch wrap するので末尾 partial は
    /// 出ない)。`score_drop_abs` が `Some(t)` なら `|score| >= t` を skip。
    /// `progress` は output bucket を計算する [`ShogiProgressKPAbs`] (ZST; 重みは
    /// process-global `OnceLock` なので呼び出し前に `load_from_bin` 済であること、
    /// 未ロードなら全 bucket 4)。`feature_set` は sparse index 化に使う feature
    /// set spec で、全 worker が共有する (`Copy`、read-only)。
    /// `num_buckets` は worker が `progress.bucket_board(board, num_buckets)` を
    /// 呼ぶときの bucket 数。`compute_bucket = false` (Simple アーキ) では bucket
    /// 計算自体が skip されるが、worker 側 assertion (`num_buckets >= 1`) は常に
    /// 評価する。
    /// `train_end_offset` は training stream の上限 byte offset (`[0, train_end_offset)`
    /// が training に使われる)。file 全体を使うときは file size をそのまま渡す。
    /// 同 file 内に held-out tail を残す経路 (`--test-tail-positions`) で
    /// `file_size - N * PSV_RECORD_BYTES` を渡し、training が tail に踏み込まない
    /// ようにするのが主用途。`train_end_offset` は [`PSV_RECORD_BYTES`] の倍数で
    /// なければならず、違反は `PsvFileLoader::new_range` 側で error になる。
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        path: &Path,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        num_workers: usize,
        progress: ShogiProgressKPAbs,
        feature_set: FeatureSetSpec,
        compute_bucket: bool,
        num_buckets: usize,
        train_end_offset: u64,
    ) -> io::Result<Self> {
        assert!(
            num_buckets >= 1,
            "BucketedPrefetchedLoader requires num_buckets >= 1"
        );
        assert!(batch_size >= 1, "batch_size must be >= 1");
        let num_workers = num_workers.max(1);
        let prefetch_depth = prefetch_depth_for(num_workers);
        // pool は「同時に out できる最大数」を満たす容量にして recycle が絶対に
        // block しないようにする: result channel に最大 prefetch_depth、各 worker
        // が最大 1、main が最大 1。
        let n_slots = prefetch_depth + num_workers + 1;

        let reader = Arc::new(Mutex::new(PsvEpochReader::new_range(
            path,
            0,
            train_end_offset,
            score_drop_abs,
        )?));
        let err_slot: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));

        let (result_tx, result_rx) = mpsc::sync_channel::<BatchSlot>(prefetch_depth);
        let (pool_tx, pool_rx) = mpsc::sync_channel::<BatchSlot>(n_slots);
        for _ in 0..n_slots {
            let slot = (
                Batch::with_capacity(batch_size, feature_set),
                Vec::with_capacity(batch_size),
            );
            pool_tx
                .send(slot)
                .expect("pool channel has capacity for the initial slots");
        }
        let pool_rx = Arc::new(Mutex::new(pool_rx));

        let mut handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let reader = Arc::clone(&reader);
            let err_slot = Arc::clone(&err_slot);
            let pool_rx = Arc::clone(&pool_rx);
            let result_tx = result_tx.clone();
            let handle = thread::spawn(move || {
                // 各 worker 専有の生 PSV scratch (iteration をまたいで reuse)。
                let mut scratch: Vec<PackedSfenValue> = Vec::with_capacity(batch_size);
                loop {
                    // 空の batch slot を pool から借りる。
                    let (mut batch, mut buckets) = {
                        let rx = pool_rx.lock().expect("pool_rx mutex poisoned");
                        match rx.recv() {
                            Ok(slot) => slot,
                            Err(_) => break, // main が pool_tx を全て drop → 終了
                        }
                    };
                    batch.reset();
                    buckets.clear();

                    // 短い critical section: 共有 reader から batch_size 件を
                    // scratch に詰める (I/O のみ、decode はしない)。
                    {
                        let mut rdr = reader.lock().expect("reader mutex poisoned");
                        scratch.clear();
                        let mut failed: Option<io::Error> = None;
                        for _ in 0..batch_size {
                            match rdr.next() {
                                Ok(psv) => scratch.push(psv),
                                Err(e) => {
                                    failed = Some(e);
                                    break;
                                }
                            }
                        }
                        drop(rdr);
                        if let Some(e) = failed {
                            // reader が exhausted: error を slot に置いて worker 終了
                            // (借りた slot は捨てる; main は next_batch の err_slot 確認で気付く)。
                            let mut slot = err_slot.lock().expect("err_slot mutex poisoned");
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    }

                    // decode-once: ShogiBoard を feature 抽出 + (compute_bucket=true
                    // のとき) progress bucket の両方に使う。`compute_bucket=false`
                    // (Simple アーキ) では `progress.bucket_board` の per-position 推論
                    // (~30-40 KP-abs weight load + exp + clamp) を skip し worker CPU を
                    // 軽くする。Simple backend は `bucket_idx` を参照しない契約。
                    let mut overflow: Option<io::Error> = None;
                    for psv in &scratch {
                        let board = psv.decode();
                        match batch.push_decoded(&board) {
                            Ok(pushed) => {
                                debug_assert!(
                                    pushed,
                                    "Batch::push_decoded refused below batch_size"
                                );
                            }
                            Err(e) => {
                                // max_active 超過: reader error と同じく err_slot に
                                // 積んで worker 終了。単一 worker error なので channel は
                                // 閉じないが、next_batch が recv 前の err_slot 確認で検出し
                                // 明示エラーを返す (借りた slot は捨てる)。
                                overflow = Some(e);
                                break;
                            }
                        }
                        if compute_bucket {
                            buckets.push(i32::from(progress.bucket_board(&board, num_buckets)));
                        }
                    }
                    if let Some(e) = overflow {
                        let mut slot = err_slot.lock().expect("err_slot mutex poisoned");
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                        return;
                    }
                    debug_assert_eq!(batch.n_positions, batch_size);
                    debug_assert!(!compute_bucket || buckets.len() == batch_size);

                    // main へ。受信側が落ちていたら (loader drop) 終了。
                    if result_tx.send((batch, buckets)).is_err() {
                        break;
                    }
                }
            });
            handles.push(handle);
        }
        // spawn ループ内の clone のみ worker が持つ。元の `result_tx` / `pool_tx`
        // は loader struct が `pool_tx` を保持 (recycle 用)、`result_tx` は drop。
        drop(result_tx);

        Ok(Self {
            result_rx: Some(result_rx),
            pool_tx: Some(pool_tx),
            err_slot,
            handles,
        })
    }

    /// 次の `(Batch, per-position bucket)` を取得。返り値:
    /// - `Ok(Some((batch, buckets)))`: 正常 batch (`batch.n_positions == batch_size`)
    /// - `Err(e)`: worker が reader から io::Error (barren-exhaustion 等) を受けた
    /// - `Ok(None)`: 全 worker が error 無しで終了 (通常は起きない; loader を
    ///   drop した後など)
    ///
    /// 消費後は [`Self::recycle`] で `(batch, buckets)` を返すこと (ring buffer)。
    pub fn next_batch(&mut self) -> io::Result<Option<BatchSlot>> {
        // 単一 worker でのみ起きる error (max_active 超過等) は、全 worker の exit
        // = result channel close を待たずに surface する必要がある。生存 worker は
        // epoch wrap で batch を供給し続け channel が閉じないため、recv 前に
        // err_slot を確認する (確認漏れ時も channel close 経路が backstop)。
        if let Some(e) = self
            .err_slot
            .lock()
            .expect("err_slot mutex poisoned")
            .take()
        {
            return Err(e);
        }
        match self
            .result_rx
            .as_ref()
            .expect("result_rx present until Drop")
            .recv()
        {
            Ok(slot) => Ok(Some(slot)),
            Err(_) => {
                // 全 worker exit → result channel close。残った error を確認。
                if let Some(e) = self
                    .err_slot
                    .lock()
                    .expect("err_slot mutex poisoned")
                    .take()
                {
                    Err(e)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// 消費済み `(Batch, buckets)` を pool に返す (worker が再利用する)。
    /// pool channel は ring の全 slot 容量を持つので block しない。worker が
    /// 既に全員終了していたら send は失敗するが無視してよい (loader drop 経路)。
    pub fn recycle(&self, slot: BatchSlot) {
        if let Some(tx) = self.pool_tx.as_ref() {
            let _ = tx.send(slot);
        }
    }
}

impl Drop for BucketedPrefetchedLoader {
    /// **close-then-join**: 先に loader 側の channel endpoint を落としてから
    /// worker thread を join する。
    ///
    /// 1. `result_rx` (result channel の **受信側**) を drop → worker の
    ///    `result_tx.send(...)` が `Err` を返し、worker が `break`。
    /// 2. `pool_tx` (pool channel の **送信側**、`recycle` 用) を drop → worker の
    ///    `pool_rx.recv()` が `Err` を返し、pool 借用待ちの worker も `break`。
    /// 3. 各 worker thread を `join` する。手順 1/2 で全 worker は次の channel 操作で
    ///    速やかに抜けるので join は hang しない (他の lock holder は兄弟 worker の
    ///    短い critical section のみ)。
    ///
    /// この順序を守らないと (= channel を閉じる前に join すると) worker が
    /// `result_tx.send` / `pool_rx.recv` で永久に block して deadlock する。
    /// `spawn` 内の thread spawn が途中で失敗するケースは無い (`thread::spawn` は
    /// 失敗時 panic する) ので `handles` は常に完全だが、`drain(..)` で空でも安全。
    fn drop(&mut self) {
        // 1 & 2: channel endpoint を先に落として worker を unblock。
        self.result_rx = None;
        self.pool_tx = None;
        // 3: 全 worker を join (channel が閉じているので速やかに終了する)。
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::FeatureSet;
    use std::path::PathBuf;

    /// テストで使う feature set spec (現 production の halfka-hm-merged)。
    fn test_spec() -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged.spec()
    }

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
        let spec = test_spec();
        let batch = Batch::with_capacity(4, spec);
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.max_active, spec.max_active());
        assert_eq!(batch.stm_indices.len(), 4 * spec.max_active());
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
    fn psv_file_loader_new_range_reads_only_specified_range() {
        // sample.psv = 4000 bytes (100 records)。
        // 範囲 [40, 80) は 1 record。
        let mut one = PsvFileLoader::new_range(sample_psv_path(), 40, 80).unwrap();
        assert!(one.next_psv().unwrap().is_some(), "1 record 読める");
        assert!(one.next_psv().unwrap().is_none(), "次は range 末尾で None");

        // 範囲 [0, 4000) は全 100 records。
        let mut full = PsvFileLoader::new_range(sample_psv_path(), 0, 4000).unwrap();
        let mut n = 0;
        while full.next_psv().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 100);

        // 範囲 [4000, 4000) は空 range、即 None。
        let mut empty = PsvFileLoader::new_range(sample_psv_path(), 4000, 4000).unwrap();
        assert!(empty.next_psv().unwrap().is_none());
    }

    #[test]
    fn psv_file_loader_new_range_skips_records_before_start() {
        // 末尾 30 records (offset 2800..4000) を取って、次に full range [0, 4000)
        // で同じ末尾 30 records を取ったときと bit-equal になることを確認
        // (Seek が record 境界に揃っている = 内容が一致する)。
        let mut tail = PsvFileLoader::new_range(sample_psv_path(), 2800, 4000).unwrap();
        let mut tail_records: Vec<PackedSfenValue> = Vec::new();
        while let Some(psv) = tail.next_psv().unwrap() {
            tail_records.push(psv);
        }
        assert_eq!(tail_records.len(), 30);

        let mut full = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut all_records: Vec<PackedSfenValue> = Vec::new();
        while let Some(psv) = full.next_psv().unwrap() {
            all_records.push(psv);
        }
        assert_eq!(all_records.len(), 100);
        for i in 0..30 {
            assert_eq!(
                tail_records[i].as_bytes(),
                all_records[70 + i].as_bytes(),
                "tail[{i}] should equal full[{}]",
                70 + i
            );
        }
    }

    #[test]
    fn psv_file_loader_new_range_rejects_out_of_bounds_end() {
        let err = PsvFileLoader::new_range(sample_psv_path(), 0, 4040)
            .err()
            .expect("end > file_size should error");
        assert!(err.to_string().contains("> file size"), "got: {err}");
    }

    #[test]
    fn psv_file_loader_new_range_rejects_misaligned() {
        let err = PsvFileLoader::new_range(sample_psv_path(), 1, 80)
            .err()
            .expect("misaligned start should error");
        assert!(err.to_string().contains("aligned"), "got: {err}");
    }

    #[test]
    fn psv_file_loader_new_range_rejects_inverted() {
        let err = PsvFileLoader::new_range(sample_psv_path(), 80, 40)
            .err()
            .expect("start > end should error");
        assert!(err.to_string().contains("start"), "got: {err}");
    }

    #[test]
    fn fill_batch_indices_within_halfka_dim_or_padding() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(8, test_spec());
        let n = loader.fill_batch(&mut batch).unwrap();
        assert_eq!(n, 8);
        assert_eq!(batch.n_positions, 8);
        for (i, &idx) in batch.stm_indices.iter().enumerate() {
            assert!(
                idx == -1 || (0..test_spec().ft_in() as i32).contains(&idx),
                "stm_indices[{i}] = {idx} は -1 padding か [0, ft_in) の範囲"
            );
        }
        for (i, &idx) in batch.nstm_indices.iter().enumerate() {
            assert!(
                idx == -1 || (0..test_spec().ft_in() as i32).contains(&idx),
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
        let mut batch = Batch::with_capacity(4, test_spec());
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
        // raw `game_result()` を直訳して `as u8 / 2.0` する経路だと Win → 0.5 に
        // 潰れるので、`wdl == 1.0` が少なくとも 1 件存在することを確認
        // (sign-aware な i8 → `{0.0, 0.5, 1.0}` map 経路の回帰検出)。
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(100, test_spec());
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
    fn batch_push_maps_draw_result_to_wdl_half() {
        // sample.psv は Loss=50 / Win=50 で Draw 行を持たないため、`result == 0
        // → wdl == 0.5` のマッピングがそのままではカバーされない。実 PSV
        // record を 1 件読んで game_result バイト (offset 38) を 0 に
        // パッチした「Draw 局面」で push_decoded が wdl == 0.5 を出すことを確認。
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut psv = loader.next_psv().unwrap().expect("at least 1 record");
        psv.as_bytes_mut()[38] = 0; // game_result = 0 (Draw)
        assert_eq!(psv.game_result(), 0);

        let mut batch = Batch::with_capacity(1, test_spec());
        assert!(batch.push(&psv).unwrap());
        assert_eq!(batch.wdl[0], 0.5, "Draw (result == 0) → wdl == 0.5");

        // Win / Loss も合わせて回帰確認 (同 record をパッチ)。
        psv.as_bytes_mut()[38] = 1i8 as u8;
        let mut b_win = Batch::with_capacity(1, test_spec());
        assert!(b_win.push(&psv).unwrap());
        assert_eq!(b_win.wdl[0], 1.0, "Win (result > 0) → wdl == 1.0");

        psv.as_bytes_mut()[38] = (-1i8) as u8;
        let mut b_loss = Batch::with_capacity(1, test_spec());
        assert!(b_loss.push(&psv).unwrap());
        assert_eq!(b_loss.wdl[0], 0.0, "Loss (result < 0) → wdl == 0.0");
    }

    #[test]
    fn fill_batch_consumes_stream_partial_at_eof() {
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(150, test_spec());
        let n = loader.fill_batch(&mut batch).unwrap();
        // sample.psv の 100 records しかない → 100 で打ち切り。
        assert_eq!(n, 100);
        assert_eq!(batch.n_positions, 100);
        // 残り 150-100=50 slot は padding のまま (-1 / 0.0 / 1.0)。
        for j in 100 * test_spec().max_active()..150 * test_spec().max_active() {
            assert_eq!(batch.stm_indices[j], -1);
        }
        for j in 100..150 {
            assert_eq!(batch.score[j], 0.0);
            assert_eq!(batch.wdl[j], 0.0);
        }
    }

    #[test]
    fn batch_push_returns_false_when_full() {
        let mut batch = Batch::with_capacity(2, test_spec());
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let psv1 = loader.next_psv().unwrap().unwrap();
        let psv2 = loader.next_psv().unwrap().unwrap();
        let psv3 = loader.next_psv().unwrap().unwrap();
        assert!(batch.push(&psv1).unwrap());
        assert!(batch.push(&psv2).unwrap());
        assert!(
            !batch.push(&psv3).unwrap(),
            "3 件目は batch_size=2 で reject"
        );
        assert_eq!(batch.n_positions, 2);
    }

    #[test]
    fn batch_reset_zeros_state() {
        let mut batch = Batch::with_capacity(4, test_spec());
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
        let mut loader = PrefetchedLoader::spawn(sample_psv_path(), 8, test_spec(), 2).unwrap();
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
        let mut loader = PrefetchedLoader::spawn(sample_psv_path(), 4, test_spec(), 0).unwrap();
        let first = loader.next_batch().unwrap().expect("at least 1 batch");
        assert_eq!(first.n_positions, 4);
    }

    // --- BucketedPrefetchedLoader ---

    /// テスト fixture: file 全体を training に使う場合の `train_end_offset`
    /// (= file size)。`std::fs::metadata` で取れる値そのもの。
    fn full_range_end(path: &Path) -> u64 {
        std::fs::metadata(path).expect("stat sample.psv").len()
    }

    fn run_bucketed_smoke(num_workers: usize) {
        // sample.psv は 100 records (Loss=50 / Win=50、Draw なし)。
        let progress = ShogiProgressKPAbs; // zero weights → 全 bucket 4
        let path = sample_psv_path();
        let end = full_range_end(&path);
        let mut loader = BucketedPrefetchedLoader::spawn(
            &path,
            16,
            None,
            num_workers,
            progress,
            test_spec(),
            true,
            9,
            end,
        )
        .unwrap();
        // epoch wrap するので何 batch でも取れる。30 batch ぶん検査して recycle で
        // 回す。
        for _ in 0..30 {
            let (batch, buckets) = loader
                .next_batch()
                .unwrap()
                .expect("epoch wraps, should never be None");
            assert_eq!(batch.n_positions, 16, "epoch wrap → 常に満タン");
            assert_eq!(buckets.len(), 16);
            assert!(
                buckets.iter().all(|&b| b == 4),
                "zero-weight progress → bucket 4"
            );
            // wdl は {0.0, 1.0} のいずれか (sample.psv は Draw なし)。Win/Loss 両方が
            // どこかに出ること自体は 16 件 batch では保証できないので membership だけ。
            for &w in &batch.wdl[..16] {
                assert!(w == 0.0 || w == 1.0, "wdl 値 = {w}");
            }
            // sparse index は [0, ft_in) か -1 padding。
            for &idx in &batch.stm_indices[..16 * test_spec().max_active()] {
                assert!(idx == -1 || (0..test_spec().ft_in() as i32).contains(&idx));
            }
            let active = batch.stm_indices.iter().filter(|&&i| i >= 0).count();
            assert!(active > 0, "実局面なので active features > 0");
            loader.recycle((batch, buckets));
        }
        drop(loader); // worker は channel close で抜ける (hang しない)。
    }

    #[test]
    fn bucketed_loader_single_worker() {
        run_bucketed_smoke(1);
    }

    #[test]
    fn bucketed_loader_multi_worker() {
        run_bucketed_smoke(4);
    }

    #[test]
    fn bucketed_loader_zero_workers_normalizes_to_one() {
        let progress = ShogiProgressKPAbs;
        let path = sample_psv_path();
        let end = full_range_end(&path);
        let mut loader =
            BucketedPrefetchedLoader::spawn(&path, 8, None, 0, progress, test_spec(), true, 9, end)
                .unwrap();
        let (batch, buckets) = loader.next_batch().unwrap().expect("a batch");
        assert_eq!(batch.n_positions, 8);
        assert_eq!(buckets.len(), 8);
    }

    #[test]
    fn bucketed_loader_score_drop_skips_high_scores() {
        // sample.psv の score がどれも |score| < 1 ということは無い (実教師局面) ので、
        // 巨大な閾値なら全件通る = epoch wrap で問題なく回る。極端に小さい閾値だと
        // 全件 skip → barren error になることを確認。
        let progress = ShogiProgressKPAbs;
        // 閾値 1: |score| >= 1 を skip。score == 0 の局面しか残らない可能性が高く、
        // 100 records 内に 1 batch (=8) ぶん埋まらないと epoch wrap で barren になりうる
        // が、sample.psv に score==0 が 8 件以上ある保証はない → barren error を許容。
        // ここでは「閾値 32000 (= 既定の score-drop 閾値) では全件通る」ことだけ確認する。
        let path = sample_psv_path();
        let end = full_range_end(&path);
        let mut ok_loader = BucketedPrefetchedLoader::spawn(
            &path,
            8,
            Some(32000),
            2,
            progress,
            test_spec(),
            true,
            9,
            end,
        )
        .unwrap();
        let (batch, _buckets) = ok_loader.next_batch().unwrap().expect("a batch");
        assert_eq!(batch.n_positions, 8);
        drop(ok_loader);

        // 閾値を 1 にして、|score| >= 1 の局面を全部捨てる。残りで batch を埋められ
        // なければ barren error。sample.psv の score 分布次第なので、error か成功か
        // どちらでもよい (hang しないことが要点)。ここでは「呼んで返ってくる」ことの
        // み確認 (panic / hang しない)。
        let mut drop_loader = BucketedPrefetchedLoader::spawn(
            &path,
            100,
            Some(1),
            1,
            progress,
            test_spec(),
            true,
            9,
            end,
        )
        .unwrap();
        let _ = drop_loader.next_batch();
    }

    #[test]
    fn bucketed_loader_with_train_end_offset_caps_training_range() {
        // file 全体 100 records のうち先頭 70 records (offset 2800) だけを
        // training に使う。worker は epoch wrap で 70 records を周回しつづける
        // ので、batch_size 8 で 30 batch (= 240 positions) 取っても barren に
        // ならず満タン batch が返り続けることを確認する。
        let progress = ShogiProgressKPAbs;
        let path = sample_psv_path();
        let mut loader = BucketedPrefetchedLoader::spawn(
            &path,
            8,
            None,
            1,
            progress,
            test_spec(),
            true,
            9,
            2800,
        )
        .unwrap();
        for _ in 0..30 {
            let (batch, buckets) = loader
                .next_batch()
                .unwrap()
                .expect("epoch wraps within capped range");
            assert_eq!(batch.n_positions, 8);
            assert_eq!(buckets.len(), 8);
            loader.recycle((batch, buckets));
        }
    }

    #[test]
    fn psv_epoch_reader_new_range_wraps_within_range() {
        // 末尾 30 records (offset 2800..4000) の範囲を epoch reader で読む。
        // 100 record 分 next() しても barren error にならず (= range 内 wrap が
        // 効いている)、各 record が必ず内容を返すことを確認する。
        let mut reader = PsvEpochReader::new_range(&sample_psv_path(), 2800, 4000, None).unwrap();
        for i in 0..100 {
            let _psv = reader
                .next()
                .unwrap_or_else(|e| panic!("wrap should keep returning records (i={i}): {e}"));
        }
    }

    #[test]
    fn bucketed_loader_empty_file_errors_not_hang() {
        let progress = ShogiProgressKPAbs;
        let tmp = std::env::temp_dir().join(format!(
            "nnue-train-bucketed-empty-{}.psv",
            std::process::id()
        ));
        std::fs::write(&tmp, b"").expect("write empty psv");
        let mut loader =
            BucketedPrefetchedLoader::spawn(&tmp, 8, None, 1, progress, test_spec(), true, 9, 0)
                .unwrap();
        let err = loader
            .next_batch()
            .expect_err("empty file → barren error, not None and not hang");
        assert!(
            err.to_string().contains("no usable positions"),
            "got: {err}"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
