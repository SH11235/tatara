//! PSV file → feature-set sparse batch dataloader (+ prefetch wrapper)。
//!
//! trainer の data 供給路。`PackedSfenValue` を [`FeatureSetSpec`] の indexer で
//! sparse index 化し、`Batch` (`stm_indices` / `nstm_indices` / `nnz` / `score` /
//! `wdl` / `per_pos_norm`) にまとめる。superbatch loop driver が GPU buffer 転送前に
//! 本 dataloader から `Batch` を pull する。どの feature set を使うかは生成時に
//! 渡す `FeatureSetSpec` で決まる (runtime 選択)。
//!
//! ## 設計のポイント
//!
//! - **WDL blend は GPU 側 (`loss_wdl` / `loss_wrm` kernel) で fuse する**ため、
//!   本 dataloader は `score` (raw cp) と `wdl` (game result `{0, 0.5, 1}`) を
//!   別 buffer に保持する (data-layer での blend pre-compute は行わない)
//! - sparse index は feature set の最大 active 数 (`FeatureSetSpec::max_active`)
//!   で固定容量を持つ。有効 slot は position ごとの `nnz` で決まり、下流 kernel は
//!   `nnz` までしか走査しない。実長超の slot の内容は未規定 (`Batch` reuse で前 batch
//!   の残骸が残りうる) — `-1` / 範囲外 index の防御 skip は残すが、正しさは `nnz`
//!   打ち切りが担保する
//! - 並列 prefetch は `std::thread::spawn` + `std::sync::mpsc::sync_channel` の
//!   minimal wrapper として [`PrefetchedLoader`] (single-thread worker) と
//!   [`BucketedPrefetchedLoader`] (multi-worker + ring-buffer pool + bucket
//!   同時計算) を提供する

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use shogi_features::progress_kpabs::ShogiProgressKPAbs;
use shogi_features::{FeatureSetSpec, kingrank9_bucket_board};
use shogi_format::{HCPE_RECORD_BYTES, HuffmanCodedPosAndEval, PackedSfenValue, ShogiBoard};

/// PSV record size in bytes (`shogi_format::PackedSfenValue` is a fixed
/// 40-byte struct). Used everywhere we compute byte offsets, validate range
/// alignment, or convert between record counts and file sizes.
pub const PSV_RECORD_BYTES: u64 = 40;

/// 40-byte PSV record 内の二つの score label から学習用 score を選ぶ方式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualLabelMode {
    /// 全 record で offset 34..36 の DL score を使う。
    All,
    /// padding bit 0 が立つ record では offset 32..34 の base score を温存する。
    Gated,
}

impl DualLabelMode {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Gated => "gated",
        }
    }

    /// dual-label PSV の予約 bit を検査し、選択した score を record に反映する。
    pub fn apply(
        self,
        psv: &mut PackedSfenValue,
        record_index: u64,
        path: &Path,
    ) -> io::Result<()> {
        let (dl_score, gate) = {
            let bytes = psv.as_bytes_mut();
            (i16::from_le_bytes([bytes[34], bytes[35]]), bytes[39])
        };
        if gate & !1 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "dual-label PSV record {record_index} in {} has non-zero reserved padding bits: 0x{gate:02x}",
                    path.display()
                ),
            ));
        }
        if self == Self::All || gate & 1 == 0 {
            psv.set_score(dl_score);
        }
        Ok(())
    }
}

/// Streaming reader for a score sidecar aligned to the records of one PSV file.
/// The optional bitmap uses LSB-first bits; a set bit preserves the PSV score.
#[derive(Debug)]
pub(crate) struct ScoreOverrideReader {
    score_path: PathBuf,
    mask_path: Option<PathBuf>,
    scores: BufReader<File>,
    mask: Option<BufReader<File>>,
    record_index: u64,
    mask_byte_index: u64,
    mask_byte: u8,
}

impl ScoreOverrideReader {
    pub(crate) fn new(
        data_path: &Path,
        score_path: &Path,
        mask_path: Option<&Path>,
        start_offset: u64,
    ) -> io::Result<Self> {
        let data_size = std::fs::metadata(data_path)?.len();
        if !data_size.is_multiple_of(PSV_RECORD_BYTES) {
            return Err(io::Error::other(format!(
                "data file {} size {data_size} is not a multiple of PSV record size ({PSV_RECORD_BYTES} bytes)",
                data_path.display()
            )));
        }
        if !start_offset.is_multiple_of(PSV_RECORD_BYTES) || start_offset > data_size {
            return Err(io::Error::other(format!(
                "score override start offset {start_offset} is not an aligned offset in data file {}",
                data_path.display()
            )));
        }
        let records = data_size / PSV_RECORD_BYTES;
        let expected_score_size = records
            .checked_mul(2)
            .ok_or_else(|| io::Error::other("score override size calculation overflowed u64"))?;
        let actual_score_size = std::fs::metadata(score_path)?.len();
        if actual_score_size != expected_score_size {
            return Err(io::Error::other(format!(
                "score override file {} size {actual_score_size} does not match data file {}: expected {expected_score_size} bytes for {records} records",
                score_path.display(),
                data_path.display()
            )));
        }
        if let Some(path) = mask_path {
            let expected_mask_size = records.div_ceil(8);
            let actual_mask_size = std::fs::metadata(path)?.len();
            if actual_mask_size != expected_mask_size {
                return Err(io::Error::other(format!(
                    "score override mask {} size {actual_mask_size} does not match data file {}: expected {expected_mask_size} bytes for {records} records",
                    path.display(),
                    data_path.display()
                )));
            }
            if !records.is_multiple_of(8) {
                let mut file = File::open(path)?;
                file.seek(SeekFrom::Start(expected_mask_size - 1))?;
                let mut final_byte = [0_u8; 1];
                file.read_exact(&mut final_byte)?;
                let used_mask = (1_u16 << (records % 8)) as u8 - 1;
                if final_byte[0] & !used_mask != 0 {
                    return Err(io::Error::other(format!(
                        "score override mask {} has non-zero unused bits in its final byte",
                        path.display()
                    )));
                }
            }
        }
        let mut reader = Self {
            score_path: score_path.to_path_buf(),
            mask_path: mask_path.map(Path::to_path_buf),
            scores: BufReader::with_capacity(1024 * 1024, File::open(score_path)?),
            mask: match mask_path {
                Some(path) => Some(BufReader::with_capacity(1024 * 1024, File::open(path)?)),
                None => None,
            },
            record_index: 0,
            mask_byte_index: u64::MAX,
            mask_byte: 0,
        };
        reader.seek_to(start_offset / PSV_RECORD_BYTES)?;
        Ok(reader)
    }

    pub(crate) fn seek_to(&mut self, record_index: u64) -> io::Result<()> {
        self.scores.seek(SeekFrom::Start(record_index * 2))?;
        if let Some(mask) = &mut self.mask {
            mask.seek(SeekFrom::Start(record_index / 8))?;
        }
        self.record_index = record_index;
        self.mask_byte_index = u64::MAX;
        Ok(())
    }

    pub(crate) fn apply(&mut self, psv: &mut PackedSfenValue) -> io::Result<()> {
        let mut score = [0_u8; 2];
        self.scores.read_exact(&mut score).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!(
                    "failed reading score override {}: {err}",
                    self.score_path.display()
                ),
            )
        })?;
        let preserve = if let Some(mask) = &mut self.mask {
            let byte_index = self.record_index / 8;
            if self.mask_byte_index != byte_index {
                mask.read_exact(std::slice::from_mut(&mut self.mask_byte))
                    .map_err(|err| {
                        io::Error::new(
                            err.kind(),
                            format!(
                                "failed reading score override mask {}: {err}",
                                self.mask_path
                                    .as_deref()
                                    .expect("mask path is present")
                                    .display()
                            ),
                        )
                    })?;
                self.mask_byte_index = byte_index;
            }
            self.mask_byte & (1 << (self.record_index % 8)) != 0
        } else {
            false
        };
        self.record_index += 1;
        if !preserve {
            psv.set_score(i16::from_le_bytes(score));
        }
        Ok(())
    }
}

/// Sequential reader for 38-byte Apery / dlshogi HCPE records.
pub struct HcpeFileLoader {
    reader: BufReader<File>,
    remaining_records: u64,
}

impl HcpeFileLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len();
        let record_bytes = HCPE_RECORD_BYTES as u64;
        if !file_size.is_multiple_of(record_bytes) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "HCPE file size {file_size} is not a multiple of {HCPE_RECORD_BYTES} bytes for {}",
                    path.as_ref().display()
                ),
            ));
        }
        Ok(Self {
            reader: BufReader::with_capacity(1024 * 1024, file),
            remaining_records: file_size / record_bytes,
        })
    }

    pub fn next_board(&mut self) -> io::Result<Option<ShogiBoard>> {
        if self.remaining_records == 0 {
            return Ok(None);
        }
        let mut record = HuffmanCodedPosAndEval::default();
        self.reader.read_exact(record.as_bytes_mut())?;
        self.remaining_records -= 1;
        record.decode().map(Some)
    }
}

/// LayerStack の position bucket 算出方式。
#[derive(Clone, Copy, Debug, Default)]
pub enum BucketMode {
    /// KP-absolute progress 推定値を `num_buckets` 等分する。
    #[default]
    Progress8KpAbs,
    /// 双方の玉段を手番視点に正規化した固定 9 bucket を使う。
    KingRank9,
}

impl BucketMode {
    /// checkpoint / experiment metadata に記録する canonical 名。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Progress8KpAbs => "progress8kpabs",
            Self::KingRank9 => "kingrank9",
        }
    }

    /// decode 済み局面の bucket index を返す。
    #[inline]
    pub fn bucket_board(self, board: &ShogiBoard, num_buckets: usize) -> u8 {
        match self {
            Self::Progress8KpAbs => ShogiProgressKPAbs.bucket_board(board, num_buckets),
            Self::KingRank9 => kingrank9_bucket_board(board),
        }
    }
}

impl From<ShogiProgressKPAbs> for BucketMode {
    fn from(_: ShogiProgressKPAbs) -> Self {
        Self::Progress8KpAbs
    }
}

// =============================================================================
// Batch 構造体 (loss / sparse_ft_forward kernel 入力と整合)
// =============================================================================

/// 1 batch 分の feature-set sparse + score/wdl/norm。
///
/// - `stm_indices` / `nstm_indices`: shape `[batch_size, max_active]` を flatten
///   (row-major、`bi * max_active + j` で参照)。有効 slot は各行の先頭 `nnz[bi]` 個で、
///   実長超の slot の内容は未規定 (`reset` は clear せず、下流 kernel は `nnz` 打ち切りで
///   読まない)。`with_capacity` は初期値として `-1` を入れるが依存してはならない
/// - `score`: raw cp (`PackedSfenValue::score` の i16 を f32 cast)
/// - `wdl`: game result を `{0.0, 0.5, 1.0}` に正規化 (Loss → 0.0, Draw → 0.5,
///   Win → 1.0)
/// - `per_pos_norm`: batch averaging 用 weight (default 1.0、trainer 側で
///   override 可能)
/// - `n_positions`: 実際に詰めた数。下流はどの buffer も `n_positions` (index は
///   `n_positions * max_active`) までしか読まないため、`[n_positions, batch_size)` の
///   末尾行の内容は未規定
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
    /// position ごとの実 active feature 数。stm / nstm は対称に emit されるため共通。
    pub nnz: Vec<i32>,
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
            nnz: vec![0; batch_size],
            score: vec![0.0; batch_size],
            wdl: vec![0.0; batch_size],
            per_pos_norm: vec![1.0; batch_size],
            n_positions: 0,
        }
    }

    /// 既存 `Batch` を再利用 (alloc 削減)。`n_positions` を 0 に戻すだけの O(1) 操作。
    /// `PsvFileLoader::fill_batch` と [`BucketedPrefetchedLoader`] の ring-buffer return
    /// path (消費済み `Batch` を pool channel 経由で worker に返して `reset()` で再利用)
    /// の両方で使われる。
    ///
    /// index / score / wdl / norm buffer は clear しない。次の fill で `push_decoded`
    /// が position `bi < n_positions` の slot `[0, nnz[bi])` と `score[bi]` / `wdl[bi]` /
    /// `nnz[bi]` を上書きし、下流はどの buffer も `n_positions` (index は `n_positions *
    /// max_active`) までしか読まない (`BatchData::from_batch_inner` の slice、kernel の
    /// `b = n_positions` launch、per-slot kernel の `nnz` early-out)。実長超の slot や
    /// `[n_positions, batch_size)` の行に前 batch の残骸が残るが、上流にも下流にも
    /// 観測されない。`per_pos_norm` Vec は下流が scalar `1/n_pos` を再計算するため未使用。
    pub fn reset(&mut self) {
        self.n_positions = 0;
    }

    /// 1 position を batch に追加。`Ok(true)` 成功、`Ok(false)` は batch 満杯、
    /// `Err` は active feature 数が `max_active` を超過 (下記参照)。`feature_set`
    /// の indexer が実 active index を行の先頭 `nnz[bi]` slot に書き、`nnz[bi]` を
    /// 記録する (実長超の slot は書かない — 下流は `nnz` までしか読まない契約)。
    ///
    /// 内部で `pos.decode()` を 1 回呼ぶ。同じ局面で別途 position bucket も
    /// 要る場合は [`Batch::push_decoded`] を使い、`PackedSfenValue::decode()` を
    /// 1 回だけ呼んで `ShogiBoard` を使い回すこと (decode-once 経路)。
    pub fn push(&mut self, pos: &PackedSfenValue) -> io::Result<bool> {
        self.push_decoded(&pos.decode())
    }

    /// [`Batch::push`] の **decode 済み `ShogiBoard` を直接受ける** 版。
    ///
    /// prefetch worker が 1 局面につき `PackedSfenValue::decode()` を 1 回だけ
    /// 呼び、その `ShogiBoard` を feature 抽出 (本メソッド) と bucket 計算の両方で
    /// 使い回すための
    /// 入口 (decode-once)。`push(&pos)` は `push_decoded(&pos.decode())` と等価。
    ///
    /// active feature 数が `max_active` を超えると `Err(io::Error)` を返す。base
    /// 特徴は合法局面で必ず cap 内だが threat 連結時は `THREAT_MAX_ACTIVE` の
    /// 見積りを edge 数が超え得る。超過を silent skip すると欠落 edge が loss だけ
    /// 見ても気付けないため、利用者に「profile / 実 active 数 / max_active」を含む
    /// 明示エラーを返して学習を止める (`THREAT_MAX_ACTIVE` 不足なら定数を上げて再ビルド)。
    pub fn push_decoded(&mut self, board: &ShogiBoard) -> io::Result<bool> {
        self.push_decoded_counting(board, None)
    }

    /// [`Batch::push_decoded`] と同一だが、成功 push 時に実 active feature 数
    /// `written` を `active_hist` (長さ `max_active + 1` の呼び出し側 histogram)
    /// の bin `written` に 1 加算する。`--monitor-active-features` の計装点。
    ///
    /// `active_hist` が `None` のときは histogram を一切触らない (計装 off 時の
    /// ホットパス no-op)。`Some` のときの `active_hist` は `feature_set.max_active()
    /// + 1` 以上の長さでなければならない (overflow は下の hard-error で弾かれるため
    /// `written <= max_active` が index の不変条件)。batch 満杯 (`Ok(false)`) や
    /// overflow (`Err`) では加算しない。
    pub fn push_decoded_counting(
        &mut self,
        board: &ShogiBoard,
        active_hist: Option<&mut [u64]>,
    ) -> io::Result<bool> {
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
        // overflow 済み後なので `written <= max_active`、bin index は必ず範囲内。
        if let Some(hist) = active_hist {
            hist[written] += 1;
        }
        self.nnz[bi] = written as i32;

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
    /// `Take` が raw read 自体を range 長で打ち切るため、BufReader の先読みが
    /// `end_offset` を越えて file を読むことはない。
    reader: BufReader<io::Take<File>>,
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
            reader: BufReader::with_capacity(1024 * 1024, file.take(end - start)),
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
        // record の byte 列を直接 `PackedSfenValue` のバッキングへ read する
        // (中間 stack buffer + copy を経由しない)。`as_bytes_mut` は丁度
        // `PSV_RECORD_BYTES` 長の `[u8; 40]`。
        let mut psv = PackedSfenValue::default();
        let buf = psv.as_bytes_mut();
        match self.reader.read(buf)? {
            0 => {
                self.eof = true;
                Ok(None)
            }
            n if n == PSV_RECORD_BYTES as usize => {
                self.remaining_bytes -= PSV_RECORD_BYTES;
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
/// `ShogiBoard` から選択された [`BucketMode`] で求める)。
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
    score_override: Option<ScoreOverrideReader>,
    dual_label_psv: Option<DualLabelMode>,
    record_index: u64,
    score_drop_abs: Option<i32>,
    score_clamp_abs: Option<i16>,
    /// 直近の reopen 以降に実際に返した (= drop されなかった) position 数。
    pushed_this_epoch: u64,
    /// 1 epoch 丸ごと 0 push だった連続回数。
    barren_passes: u32,
}

impl PsvEpochReader {
    /// `path` を `[start_offset, end_offset)` 範囲で epoch wrap させる reader。
    /// wrap 時の再 open も同 range で行う。`PsvFileLoader::new_range` 同様の
    /// 範囲・alignment 検証はここでは行わず、`new_range` 内で検証する。
    #[allow(clippy::too_many_arguments)]
    fn new_range(
        path: &Path,
        start_offset: u64,
        end_offset: u64,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        score_override: Option<&Path>,
        score_override_mask: Option<&Path>,
        dual_label_psv: Option<DualLabelMode>,
    ) -> io::Result<Self> {
        if dual_label_psv.is_some() && (score_override.is_some() || score_override_mask.is_some()) {
            return Err(io::Error::other(
                "dual_label_psv conflicts with score_override and score_override_mask",
            ));
        }
        let loader = PsvFileLoader::new_range(path, start_offset, end_offset)?;
        let score_override = score_override
            .map(|score_path| {
                ScoreOverrideReader::new(path, score_path, score_override_mask, start_offset)
            })
            .transpose()?;
        Ok(Self {
            path: path.to_path_buf(),
            start_offset,
            end_offset,
            loader,
            score_override,
            dual_label_psv,
            record_index: start_offset / PSV_RECORD_BYTES,
            score_drop_abs,
            score_clamp_abs,
            pushed_this_epoch: 0,
            barren_passes: 0,
        })
    }

    /// 現在の epoch にある次の使える PSV を返す。physical EOF では reader を次
    /// epoch の先頭へ戻して `None` を返す。これにより window shuffle 側は epoch
    /// 境界を跨がず、末尾の partial window も独立して shuffle できる。
    fn next_in_epoch(&mut self) -> io::Result<Option<PackedSfenValue>> {
        loop {
            match self.loader.next_psv()? {
                Some(mut psv) => {
                    let record_index = self.record_index;
                    self.record_index += 1;
                    // Sidecar indices are based on the complete PSV file. Applying the
                    // replacement before filtering matches a materialized PSV variant.
                    if let Some(score_override) = &mut self.score_override {
                        score_override.apply(&mut psv)?;
                    }
                    if let Some(mode) = self.dual_label_psv {
                        mode.apply(&mut psv, record_index, &self.path)?;
                    }
                    // `--score-drop-abs t` 指定時: `|score| >= t` を skip。
                    // i64 cast で `i16::MIN` の abs overflow を避ける。
                    if let Some(t) = self.score_drop_abs
                        && i64::from(psv.score()).abs() >= i64::from(t)
                    {
                        continue;
                    }
                    // `--score-clamp-abs c` 指定時: 生き残った position の score を
                    // `[-c, c]` に飽和させる。drop 判定の後に適用する (先に clamp
                    // すると `|score| >= drop` の詰み stamp が clamp されて drop を
                    // すり抜ける)。c >= 1 は TrainingConfig::validate が保証する
                    // ため `-c` は overflow しない。
                    if let Some(c) = self.score_clamp_abs {
                        psv.set_score(psv.score().clamp(-c, c));
                    }
                    self.pushed_this_epoch += 1;
                    return Ok(Some(psv));
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
                    self.record_index = self.start_offset / PSV_RECORD_BYTES;
                    if let Some(score_override) = &mut self.score_override {
                        score_override.seek_to(self.start_offset / PSV_RECORD_BYTES)?;
                    }
                    return Ok(None);
                }
            }
        }
    }

    /// 次の使える PSV を返す。EOF なら file を開き直す (= 次 epoch)。空 file /
    /// 全 drop で `MAX_BARREN_PASSES` 周しても 0 件なら `io::Error` を返す。
    fn next(&mut self) -> io::Result<PackedSfenValue> {
        loop {
            if let Some(psv) = self.next_in_epoch()? {
                return Ok(psv);
            }
        }
    }
}

const MIB_BYTES: usize = 1024 * 1024;

pub(crate) fn shuffle_window_records(
    buffer_mib: usize,
    batch_size: usize,
) -> io::Result<Option<usize>> {
    if buffer_mib == 0 {
        return Ok(None);
    }
    let bytes = buffer_mib
        .checked_mul(MIB_BYTES)
        .ok_or_else(|| io::Error::other("teacher shuffle buffer size overflow"))?;
    let records = bytes / PSV_RECORD_BYTES as usize;
    let aligned = records / batch_size * batch_size;
    if aligned == 0 {
        return Err(io::Error::other(format!(
            "teacher shuffle buffer ({buffer_mib} MiB) is smaller than one raw batch ({} bytes)",
            batch_size * PSV_RECORD_BYTES as usize
        )));
    }
    Ok(Some(aligned))
}

/// Small deterministic PRNG used only for Fisher-Yates indices. Keeping this local avoids
/// coupling training-data order to the version or algorithm of an external RNG crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn shuffle_window(values: &mut [PackedSfenValue], seed: u64, epoch: u64, window: u64) {
    let mut rng = SplitMix64(
        seed ^ epoch.wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ window.wrapping_mul(0xa5a3_56e4_e27f_886d),
    );
    for i in (1..values.len()).rev() {
        values.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
}

struct PsvWindow {
    records: Vec<PackedSfenValue>,
    next: usize,
}

/// Two-window producer/consumer reader. The producer fills the next raw PSV window while the
/// decode workers consume the current one. `window_records` is per window, so resident raw PSV
/// capacity is approximately `2 * window_records * 40` bytes.
struct WindowedPsvReader {
    ready_rx: mpsc::Receiver<io::Result<PsvWindow>>,
    empty_tx: Option<mpsc::Sender<Vec<PackedSfenValue>>>,
    current: Option<PsvWindow>,
    stop: Arc<AtomicBool>,
    producer: Option<thread::JoinHandle<()>>,
    /// producer の失敗 message。err_slot は first-write-wins なので、後続 caller にも
    /// 汎用 "stopped" でなく同じ詳細 message を返して race による握り潰しを防ぐ。
    failure: Option<String>,
}

impl WindowedPsvReader {
    fn spawn(mut source: PsvEpochReader, window_records: usize, shuffle: bool, seed: u64) -> Self {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (empty_tx, empty_rx) = mpsc::channel::<Vec<PackedSfenValue>>();
        let stop = Arc::new(AtomicBool::new(false));
        let producer_stop = Arc::clone(&stop);
        let producer = thread::spawn(move || {
            let mut allocated = 0usize;
            let mut spare = None;
            let mut epoch = 0u64;
            let mut window = 0u64;
            loop {
                if producer_stop.load(Ordering::Relaxed) {
                    break;
                }
                let mut records = if let Some(records) = spare.take() {
                    records
                } else if allocated < 2 {
                    allocated += 1;
                    let mut records = Vec::new();
                    if let Err(e) = records.try_reserve_exact(window_records) {
                        let _ = ready_tx.send(Err(io::Error::other(format!(
                            "failed to allocate teacher shuffle window for {window_records} records: {e}"
                        ))));
                        return;
                    }
                    records
                } else {
                    loop {
                        match empty_rx.recv_timeout(Duration::from_millis(100)) {
                            Ok(records) => break records,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if producer_stop.load(Ordering::Relaxed) {
                                    return;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                    }
                };
                records.clear();
                let mut reached_epoch_end = false;
                while records.len() < window_records {
                    if producer_stop.load(Ordering::Relaxed) {
                        return;
                    }
                    match source.next_in_epoch() {
                        Ok(Some(psv)) => records.push(psv),
                        Ok(None) => {
                            reached_epoch_end = true;
                            break;
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    }
                }
                if records.is_empty() {
                    if reached_epoch_end {
                        epoch = epoch.wrapping_add(1);
                        window = 0;
                    }
                    spare = Some(records);
                    continue;
                }
                if shuffle {
                    shuffle_window(&mut records, seed, epoch, window);
                }
                if ready_tx.send(Ok(PsvWindow { records, next: 0 })).is_err() {
                    return;
                }
                window = window.wrapping_add(1);
                if reached_epoch_end {
                    epoch = epoch.wrapping_add(1);
                    window = 0;
                }
            }
        });
        Self {
            ready_rx,
            empty_tx: Some(empty_tx),
            current: None,
            stop,
            producer: Some(producer),
            failure: None,
        }
    }

    fn next(&mut self) -> io::Result<PackedSfenValue> {
        if let Some(msg) = &self.failure {
            return Err(io::Error::other(msg.clone()));
        }
        loop {
            if let Some(window) = &mut self.current
                && window.next < window.records.len()
            {
                let value = window.records[window.next];
                window.next += 1;
                return Ok(value);
            }
            if let Some(old) = self.current.take()
                && self
                    .empty_tx
                    .as_ref()
                    .is_some_and(|tx| tx.send(old.records).is_err())
            {
                // producer 終了済み。queue 済みの詳細エラーを下の recv で表面化させる。
                self.empty_tx = None;
            }
            match self.ready_rx.recv() {
                Ok(Ok(window)) => self.current = Some(window),
                Ok(Err(e)) => {
                    self.failure = Some(e.to_string());
                    return Err(e);
                }
                Err(_) => {
                    let msg = "PSV window producer stopped".to_string();
                    self.failure = Some(msg.clone());
                    return Err(io::Error::other(msg));
                }
            }
        }
    }
}

impl Drop for WindowedPsvReader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.empty_tx.take();
        if let Some(producer) = self.producer.take() {
            let _ = producer.join();
        }
    }
}

enum TrainingPsvReader {
    Direct(Box<PsvEpochReader>),
    Windowed(WindowedPsvReader),
}

impl TrainingPsvReader {
    fn next(&mut self) -> io::Result<PackedSfenValue> {
        match self {
            Self::Direct(reader) => reader.next(),
            Self::Windowed(reader) => reader.next(),
        }
    }

    /// reader の Mutex を取らずに loader の Drop から producer を止めるための flag。
    fn producer_stop_flag(&self) -> Option<Arc<AtomicBool>> {
        match self {
            Self::Direct(_) => None,
            Self::Windowed(reader) => Some(Arc::clone(&reader.stop)),
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
/// 「PSV パース + feature sparse 抽出 + position bucket 計算」を
/// `decode()` **1 回** で済ませて main thread に `(Batch, buckets)` を渡す
/// prefetch loader。
///
/// ## 設計
///
/// - **decode-once**: worker は `psv.decode()` した `ShogiBoard` を
///   `Batch::push_decoded` (feature 抽出) と [`BucketMode::bucket_board`] の両方に
///   渡す。`pos.decode()` は 1 局面 1 回。
/// - **並列パース**: worker は短い critical section (共有 reader を lock して
///   `batch_size` 件の生 PSV を自前 scratch `Vec` に詰める; I/O は逐次・高速) の
///   外で decode + 特徴抽出を並列に行う。windowed reader では窓境界で lock 保持の
///   まま次窓完成を待ち得る (`Drop` は先に producer を止める; `producer_stop`)。`FeatureSetSpec` は
///   `Copy` の値型で、bucket mode も read-only なので thread 間共有できる。
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
    /// `--monitor-active-features` 時のみ `Some`。全 worker が共有する実 active
    /// feature 数の histogram (長さ `feature_set.max_active() + 1`、bin `k` =
    /// 実 active 数がちょうど `k` だった position 数の累積)。worker は自身の
    /// batch-local histogram を batch 単位でここに flush する (1 position ごとの
    /// lock を避ける)。`None` なら計装なし。
    active_hist: Option<Arc<Mutex<Vec<u64>>>>,
    /// worker thread handle (`Drop` で join する)。
    handles: Vec<thread::JoinHandle<()>>,
    /// windowed reader 使用時のみ `Some`。`Drop` の先頭で set しないと、reader lock を
    /// 持つ worker が次 window の完成待ちで block し、join が窓 1 枚分 stall する。
    producer_stop: Option<Arc<AtomicBool>>,
}

impl BucketedPrefetchedLoader {
    /// `path` の PSV を `num_workers` 本の worker で読み込む。各 batch は
    /// `batch_size` 件の有効 position を持つ (epoch wrap するので末尾 partial は
    /// 出ない)。`score_drop_abs` が `Some(t)` なら `|score| >= t` を skip。
    /// `score_clamp_abs` が `Some(c)` なら drop を生き残った position の score を
    /// `[-c, c]` に飽和させる (`--score-clamp-abs`)。
    /// `bucket_mode` は output bucket の算出方式。`Progress8KpAbs` の重みは
    /// process-global なので呼び出し前に `ShogiProgressKPAbs::load_from_bin` 済で
    /// あること、未ロードなら全 bucket 4。`KingRank9` は外部重みを参照しない。
    /// `feature_set` は sparse index 化に使う feature set spec で、全 worker が共有する。
    /// `num_buckets` は progress mode の bucket 数。`compute_bucket = false` (Simple アーキ) では bucket
    /// 計算自体が skip されるが、worker 側 assertion (`num_buckets >= 1`) は常に
    /// 評価する。
    /// `train_end_offset` は training stream の上限 byte offset (`[0, train_end_offset)`
    /// が training に使われる)。file 全体を使うときは file size をそのまま渡す。
    /// 同 file 内に held-out tail を残す経路 (`--test-tail-positions`) で
    /// `file_size - N * PSV_RECORD_BYTES` を渡し、training が tail に踏み込まない
    /// ようにするのが主用途。`train_end_offset` は [`PSV_RECORD_BYTES`] の倍数で
    /// なければならず、違反は `PsvFileLoader::new_range` 側で error になる。
    /// `monitor_active` が `true` のとき、各 position の実 active feature 数を
    /// histogram (`feature_set.max_active() + 1` bins) に集計し [`Self::active_histogram_snapshot`]
    /// で参照できるようにする (`--monitor-active-features`)。`false` では histogram
    /// を確保せず worker のホットパスに計装コードを一切通さない。
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        path: &Path,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        num_workers: usize,
        bucket_mode: impl Into<BucketMode>,
        feature_set: FeatureSetSpec,
        compute_bucket: bool,
        num_buckets: usize,
        train_end_offset: u64,
        monitor_active: bool,
    ) -> io::Result<Self> {
        Self::spawn_with_score_override(
            path,
            batch_size,
            score_drop_abs,
            score_clamp_abs,
            num_workers,
            bucket_mode,
            feature_set,
            compute_bucket,
            num_buckets,
            train_end_offset,
            monitor_active,
            None,
            None,
            0,
            false,
            0,
        )
    }

    /// Spawns a loader with an optional streaming score sidecar and preserve mask.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_score_override(
        path: &Path,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        num_workers: usize,
        bucket_mode: impl Into<BucketMode>,
        feature_set: FeatureSetSpec,
        compute_bucket: bool,
        num_buckets: usize,
        train_end_offset: u64,
        monitor_active: bool,
        score_override: Option<&Path>,
        score_override_mask: Option<&Path>,
        teacher_shuffle_buffer_mib: usize,
        teacher_shuffle: bool,
        teacher_shuffle_seed: u64,
    ) -> io::Result<Self> {
        Self::spawn_with_score_sources(
            path,
            batch_size,
            score_drop_abs,
            score_clamp_abs,
            num_workers,
            bucket_mode,
            feature_set,
            compute_bucket,
            num_buckets,
            train_end_offset,
            monitor_active,
            score_override,
            score_override_mask,
            teacher_shuffle_buffer_mib,
            teacher_shuffle,
            teacher_shuffle_seed,
            None,
        )
    }

    /// Spawns a loader with one explicit score source selection.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_score_sources(
        path: &Path,
        batch_size: usize,
        score_drop_abs: Option<i32>,
        score_clamp_abs: Option<i16>,
        num_workers: usize,
        bucket_mode: impl Into<BucketMode>,
        feature_set: FeatureSetSpec,
        compute_bucket: bool,
        num_buckets: usize,
        train_end_offset: u64,
        monitor_active: bool,
        score_override: Option<&Path>,
        score_override_mask: Option<&Path>,
        teacher_shuffle_buffer_mib: usize,
        teacher_shuffle: bool,
        teacher_shuffle_seed: u64,
        dual_label_psv: Option<DualLabelMode>,
    ) -> io::Result<Self> {
        assert!(
            num_buckets >= 1,
            "BucketedPrefetchedLoader requires num_buckets >= 1"
        );
        assert!(batch_size >= 1, "batch_size must be >= 1");
        let bucket_mode = bucket_mode.into();
        let num_workers = num_workers.max(1);
        let prefetch_depth = prefetch_depth_for(num_workers);
        // pool は「同時に out できる最大数」を満たす容量にして recycle が絶対に
        // block しないようにする: result channel に最大 prefetch_depth、各 worker
        // が最大 1、main が最大 1。
        let n_slots = prefetch_depth + num_workers + 1;

        let source = PsvEpochReader::new_range(
            path,
            0,
            train_end_offset,
            score_drop_abs,
            score_clamp_abs,
            score_override,
            score_override_mask,
            dual_label_psv,
        )?;
        let reader = match shuffle_window_records(teacher_shuffle_buffer_mib, batch_size)? {
            Some(window_records) => TrainingPsvReader::Windowed(WindowedPsvReader::spawn(
                source,
                window_records,
                teacher_shuffle,
                teacher_shuffle_seed,
            )),
            None => TrainingPsvReader::Direct(Box::new(source)),
        };
        let producer_stop = reader.producer_stop_flag();
        let reader = Arc::new(Mutex::new(reader));
        let err_slot: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));
        let active_hist: Option<Arc<Mutex<Vec<u64>>>> = if monitor_active {
            Some(Arc::new(Mutex::new(vec![
                0u64;
                feature_set.max_active() + 1
            ])))
        } else {
            None
        };

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
            let active_hist = active_hist.clone();
            let handle = thread::spawn(move || {
                // 各 worker 専有の生 PSV scratch (iteration をまたいで reuse)。
                let mut scratch: Vec<PackedSfenValue> = Vec::with_capacity(batch_size);
                // batch-local な active-feature histogram (計装 on のときだけ確保)。
                // batch 末に共有 `active_hist` へ一括加算 → 1 position ごとの lock を
                // 避ける。
                let mut local_hist: Option<Vec<u64>> = active_hist
                    .as_ref()
                    .map(|_| vec![0u64; feature_set.max_active() + 1]);
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
                    // のとき) position bucket の両方に使う。`compute_bucket=false`
                    // (Simple アーキ) では bucket mode ごとの per-position 計算を skip し worker CPU を
                    // 軽くする。Simple backend は `bucket_idx` を参照しない契約。
                    let mut overflow: Option<io::Error> = None;
                    for psv in &scratch {
                        let board = psv.decode();
                        match batch.push_decoded_counting(&board, local_hist.as_deref_mut()) {
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
                            buckets.push(i32::from(bucket_mode.bucket_board(&board, num_buckets)));
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

                    // batch-local histogram を共有 accumulator に flush して 0 に戻す
                    // (batch 単位の lock)。`active_hist` / `local_hist` は同時に
                    // `Some` / `None`。
                    if let (Some(shared), Some(local)) = (active_hist.as_ref(), local_hist.as_mut())
                    {
                        let mut g = shared.lock().expect("active_hist mutex poisoned");
                        for (dst, src) in g.iter_mut().zip(local.iter()) {
                            *dst += *src;
                        }
                        for v in local.iter_mut() {
                            *v = 0;
                        }
                    }

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
            active_hist,
            handles,
            producer_stop,
        })
    }

    /// `--monitor-active-features` の histogram の現時点 snapshot を返す
    /// (`spawn` で `monitor_active = false` なら `None`)。返す `Vec<u64>` は
    /// 長さ `feature_set.max_active() + 1` で、bin `k` = 実 active 数がちょうど
    /// `k` だった position 数の累積 (全 worker 合算)。lock 中に clone するので
    /// 呼び出しは superbatch 末など低頻度に留めること。
    pub fn active_histogram_snapshot(&self) -> Option<Vec<u64>> {
        self.active_hist
            .as_ref()
            .map(|h| h.lock().expect("active_hist mutex poisoned").clone())
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
    /// 1. windowed reader 使用時は producer の stop flag を set → producer が record
    ///    境界で止まり channel が閉じ、次 window 待ちの worker も unblock される。
    /// 2. `result_rx` (result channel の **受信側**) を drop → worker の
    ///    `result_tx.send(...)` が `Err` を返し、worker が `break`。
    /// 3. `pool_tx` (pool channel の **送信側**、`recycle` 用) を drop → worker の
    ///    `pool_rx.recv()` が `Err` を返し、pool 借用待ちの worker も `break`。
    /// 4. 各 worker thread を `join` する。手順 1..=3 で全 worker は次の channel
    ///    操作で速やかに抜けるので join は hang しない。
    ///
    /// この順序を守らないと (= channel を閉じる前に join すると) worker が
    /// `result_tx.send` / `pool_rx.recv` で永久に block して deadlock する。
    /// `spawn` 内の thread spawn が途中で失敗するケースは無い (`thread::spawn` は
    /// 失敗時 panic する) ので `handles` は常に完全だが、`drain(..)` で空でも安全。
    fn drop(&mut self) {
        // 1: window producer を止める (worker unblock の前提)。
        if let Some(stop) = &self.producer_stop {
            stop.store(true, Ordering::Relaxed);
        }
        // 2 & 3: channel endpoint を先に落として worker を unblock。
        self.result_rx = None;
        self.pool_tx = None;
        // 4: 全 worker を join (channel が閉じているので速やかに終了する)。
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

    fn synthetic_override_files(
        name: &str,
        base_scores: &[i16],
        override_scores: &[i16],
        mask: Option<&[u8]>,
    ) -> (PathBuf, PathBuf, Option<PathBuf>) {
        assert_eq!(base_scores.len(), override_scores.len());
        let prefix = std::env::temp_dir().join(format!(
            "nnue-train-score-override-{name}-{}",
            std::process::id()
        ));
        let data = prefix.with_extension("psv");
        let scores = prefix.with_extension("scores");
        let mask_path = mask.map(|_| prefix.with_extension("mask"));
        let mut data_bytes = Vec::with_capacity(base_scores.len() * PSV_RECORD_BYTES as usize);
        let mut score_bytes = Vec::with_capacity(override_scores.len() * 2);
        for (&base, &replacement) in base_scores.iter().zip(override_scores) {
            let mut record = [0_u8; PSV_RECORD_BYTES as usize];
            record[32..34].copy_from_slice(&base.to_le_bytes());
            data_bytes.extend_from_slice(&record);
            score_bytes.extend_from_slice(&replacement.to_le_bytes());
        }
        std::fs::write(&data, data_bytes).unwrap();
        std::fs::write(&scores, score_bytes).unwrap();
        if let (Some(path), Some(bytes)) = (&mask_path, mask) {
            std::fs::write(path, bytes).unwrap();
        }
        (data, scores, mask_path)
    }

    fn remove_override_files(data: &Path, scores: &Path, mask: Option<&Path>) {
        let _ = std::fs::remove_file(data);
        let _ = std::fs::remove_file(scores);
        if let Some(path) = mask {
            let _ = std::fs::remove_file(path);
        }
    }

    fn windowed_score_reader(
        data: &Path,
        records: usize,
        window_records: usize,
        shuffle: bool,
        seed: u64,
        score_override: Option<&Path>,
    ) -> WindowedPsvReader {
        let source = PsvEpochReader::new_range(
            data,
            0,
            records as u64 * PSV_RECORD_BYTES,
            None,
            None,
            score_override,
            None,
            None,
        )
        .unwrap();
        WindowedPsvReader::spawn(source, window_records, shuffle, seed)
    }

    #[test]
    fn shuffle_window_size_is_per_window_and_batch_aligned() {
        assert_eq!(shuffle_window_records(0, 65_536).unwrap(), None);
        let records = shuffle_window_records(4096, 65_536).unwrap().unwrap();
        assert_eq!(records % 65_536, 0);
        assert!(records * PSV_RECORD_BYTES as usize <= 4096 * MIB_BYTES);
        assert!((records + 65_536) * PSV_RECORD_BYTES as usize > 4096 * MIB_BYTES);
        assert!(shuffle_window_records(1, 65_536).is_err());
    }

    #[test]
    fn window_shuffle_is_reproducible_and_epoch_specific() {
        let base = [0, 1, 2, 3, 4, 5];
        let (data, scores, _) = synthetic_override_files("window-epochs", &base, &base, None);
        let read_two_epochs = || {
            let mut reader = windowed_score_reader(&data, base.len(), 3, true, 1234, None);
            (0..base.len() * 2)
                .map(|_| reader.next().unwrap().score())
                .collect::<Vec<_>>()
        };
        let first = read_two_epochs();
        let second = read_two_epochs();
        remove_override_files(&data, &scores, None);

        assert_eq!(first, second);
        for epoch in first.chunks_exact(base.len()) {
            let mut values = epoch.to_vec();
            values.sort_unstable();
            assert_eq!(values, base);
        }
        assert_ne!(&first[..base.len()], &first[base.len()..]);
        // A window never mixes records from either side of its physical boundary.
        for window in first.chunks_exact(3) {
            assert!(window.iter().all(|v| *v < 3) || window.iter().all(|v| *v >= 3));
        }
    }

    #[test]
    fn window_shuffle_applies_score_sidecar_before_reordering() {
        let base = [1, 2, 3, 4, 5];
        let replacement = [101, 102, 103, 104, 105];
        let (data, scores, _) =
            synthetic_override_files("window-sidecar", &base, &replacement, None);
        let mut reader =
            windowed_score_reader(&data, base.len(), base.len(), true, 9, Some(&scores));
        let mut got = (0..base.len())
            .map(|_| reader.next().unwrap().score())
            .collect::<Vec<_>>();
        drop(reader);
        remove_override_files(&data, &scores, None);
        got.sort_unstable();
        assert_eq!(got, replacement);
    }

    #[test]
    fn window_shuffle_applies_dual_label_before_reordering() {
        let base = [1, 2, 3, 4, 5];
        let dl = [101, 102, 103, 104, 105];
        let path = synthetic_dual_file("window", &base, &dl, &[0; 5]);
        let source = PsvEpochReader::new_range(
            &path,
            0,
            base.len() as u64 * PSV_RECORD_BYTES,
            None,
            None,
            None,
            None,
            Some(DualLabelMode::All),
        )
        .unwrap();
        let mut reader = WindowedPsvReader::spawn(source, base.len(), true, 9);
        let mut got = (0..base.len())
            .map(|_| reader.next().unwrap().score())
            .collect::<Vec<_>>();
        drop(reader);
        let _ = std::fs::remove_file(path);
        got.sort_unstable();
        assert_eq!(got, dl);
    }

    #[test]
    fn dropping_partially_consumed_window_stops_producer() {
        let base = [1, 2, 3, 4, 5];
        let (data, scores, _) = synthetic_override_files("window-drop", &base, &base, None);
        let mut reader = windowed_score_reader(&data, base.len(), 1_000_000, true, 0, None);
        let _ = reader.next().unwrap();
        drop(reader);
        remove_override_files(&data, &scores, None);
    }

    fn synthetic_dual_file(
        name: &str,
        base_scores: &[i16],
        dl_scores: &[i16],
        gates: &[u8],
    ) -> PathBuf {
        assert_eq!(base_scores.len(), dl_scores.len());
        assert_eq!(base_scores.len(), gates.len());
        let path = std::env::temp_dir().join(format!(
            "nnue-train-dual-label-{name}-{}.psv",
            std::process::id()
        ));
        let mut bytes = Vec::with_capacity(base_scores.len() * PSV_RECORD_BYTES as usize);
        for ((&base, &dl), &gate) in base_scores.iter().zip(dl_scores).zip(gates) {
            let mut record = [0_u8; PSV_RECORD_BYTES as usize];
            record[32..34].copy_from_slice(&base.to_le_bytes());
            record[34..36].copy_from_slice(&dl.to_le_bytes());
            record[39] = gate;
            bytes.extend_from_slice(&record);
        }
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn dual_label_all_replaces_every_score_and_wraps() {
        let base = [1, 2, 3];
        let dl = [101, -202, 303];
        let path = synthetic_dual_file("all-wrap", &base, &dl, &[1, 0, 1]);
        let mut reader = PsvEpochReader::new_range(
            &path,
            0,
            3 * PSV_RECORD_BYTES,
            None,
            None,
            None,
            None,
            Some(DualLabelMode::All),
        )
        .unwrap();
        let got: Vec<i16> = (0..6).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        let _ = std::fs::remove_file(&path);
        assert_eq!(got, [101, -202, 303, 101, -202, 303]);
    }

    #[test]
    fn dual_label_gated_preserves_boundary_records() {
        let base: Vec<i16> = (0..17).map(|i| i as i16).collect();
        let dl: Vec<i16> = (0..17).map(|i| 1000 + i as i16).collect();
        let gates: Vec<u8> = (0..17)
            .map(|i| u8::from([0, 7, 8, 16].contains(&i)))
            .collect();
        let path = synthetic_dual_file("gated-boundaries", &base, &dl, &gates);
        let mut reader = PsvEpochReader::new_range(
            &path,
            0,
            17 * PSV_RECORD_BYTES,
            None,
            None,
            None,
            None,
            Some(DualLabelMode::Gated),
        )
        .unwrap();
        let got: Vec<i16> = (0..17).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        let _ = std::fs::remove_file(&path);
        for i in 0..17 {
            let expected = if [0, 7, 8, 16].contains(&i) {
                base[i]
            } else {
                dl[i]
            };
            assert_eq!(got[i], expected, "record {i}");
        }
    }

    #[test]
    fn dual_label_rejects_reserved_padding_bits_with_index_and_path() {
        let path = synthetic_dual_file("reserved-bits", &[1, 2, 3], &[11, 12, 13], &[0, 2, 0]);
        let mut reader = PsvEpochReader::new_range(
            &path,
            0,
            3 * PSV_RECORD_BYTES,
            None,
            None,
            None,
            None,
            Some(DualLabelMode::All),
        )
        .unwrap();
        assert_eq!(reader.next().unwrap().score(), 11);
        let err = match reader.next() {
            Ok(_) => panic!("reserved padding bits must be rejected"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(message.contains("record 1"), "got: {message}");
        assert!(
            message.contains(&path.display().to_string()),
            "got: {message}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dual_label_range_uses_absolute_record_indices_and_wraps() {
        let base = [0, 1, 2, 3, 4];
        let dl = [100, 101, 102, 103, 104];
        let path = synthetic_dual_file("range", &base, &dl, &[0; 5]);
        let mut reader = PsvEpochReader::new_range(
            &path,
            2 * PSV_RECORD_BYTES,
            5 * PSV_RECORD_BYTES,
            None,
            None,
            None,
            None,
            Some(DualLabelMode::All),
        )
        .unwrap();
        let got: Vec<i16> = (0..4).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        let _ = std::fs::remove_file(&path);
        assert_eq!(got, [102, 103, 104, 102]);
    }

    #[test]
    fn dual_label_selection_precedes_score_drop_and_clamp() {
        let path =
            synthetic_dual_file("drop-clamp-order", &[10, 20, 30], &[500, 50, -250], &[0; 3]);
        let mut reader = PsvEpochReader::new_range(
            &path,
            0,
            3 * PSV_RECORD_BYTES,
            Some(400),
            Some(100),
            None,
            None,
            Some(DualLabelMode::All),
        )
        .unwrap();
        let got: Vec<i16> = (0..4).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        let _ = std::fs::remove_file(&path);
        assert_eq!(got, [50, -100, 50, -100]);
    }

    #[test]
    fn score_override_replaces_all_scores_and_wraps() {
        let base = [1, 2, 3];
        let replacement = [101, -202, 303];
        let (data, scores, _) = synthetic_override_files("all-wrap", &base, &replacement, None);
        let mut reader = PsvEpochReader::new_range(
            &data,
            0,
            3 * PSV_RECORD_BYTES,
            None,
            None,
            Some(&scores),
            None,
            None,
        )
        .unwrap();
        let got: Vec<i16> = (0..6).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        remove_override_files(&data, &scores, None);
        assert_eq!(got, [101, -202, 303, 101, -202, 303]);
    }

    #[test]
    fn score_override_mask_preserves_lsb_first_boundary_records() {
        let base: Vec<i16> = (0..17).map(|i| i as i16).collect();
        let replacement: Vec<i16> = (0..17).map(|i| 1000 + i as i16).collect();
        // Preserve indices 0, 7, 8, and 16, covering both sides of byte boundaries
        // and the final partial bitmap byte. Unused bits remain zero.
        let mask = [0b1000_0001, 0b0000_0001, 0b0000_0001];
        let (data, scores, mask_path) =
            synthetic_override_files("mask-boundary", &base, &replacement, Some(&mask));
        let mut reader = PsvEpochReader::new_range(
            &data,
            0,
            17 * PSV_RECORD_BYTES,
            None,
            None,
            Some(&scores),
            mask_path.as_deref(),
            None,
        )
        .unwrap();
        let got: Vec<i16> = (0..17).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        remove_override_files(&data, &scores, mask_path.as_deref());
        for i in 0..17 {
            let expected = if [0, 7, 8, 16].contains(&i) {
                base[i]
            } else {
                replacement[i]
            };
            assert_eq!(got[i], expected, "record {i}");
        }
    }

    #[test]
    fn score_override_range_uses_full_file_record_indices() {
        let base = [0, 1, 2, 3, 4];
        let replacement = [100, 101, 102, 103, 104];
        let (data, scores, _) = synthetic_override_files("range", &base, &replacement, None);
        let mut reader = PsvEpochReader::new_range(
            &data,
            2 * PSV_RECORD_BYTES,
            5 * PSV_RECORD_BYTES,
            None,
            None,
            Some(&scores),
            None,
            None,
        )
        .unwrap();
        let got: Vec<i16> = (0..4).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        remove_override_files(&data, &scores, None);
        assert_eq!(got, [102, 103, 104, 102]);
    }

    #[test]
    fn score_override_mask_range_wraps_from_non_byte_aligned_record() {
        let base: Vec<i16> = (0..12).map(|i| i as i16).collect();
        let replacement: Vec<i16> = (0..12).map(|i| 100 + i as i16).collect();
        let mask = [0b1000_1000, 0b0000_0101];
        let (data, scores, mask_path) =
            synthetic_override_files("mask-range-wrap", &base, &replacement, Some(&mask));
        let mut reader = PsvEpochReader::new_range(
            &data,
            3 * PSV_RECORD_BYTES,
            11 * PSV_RECORD_BYTES,
            None,
            None,
            Some(&scores),
            mask_path.as_deref(),
            None,
        )
        .unwrap();

        let got: Vec<i16> = (0..10).map(|_| reader.next().unwrap().score()).collect();
        drop(reader);
        remove_override_files(&data, &scores, mask_path.as_deref());
        assert_eq!(got, [3, 104, 105, 106, 7, 8, 109, 10, 3, 104]);
    }

    #[test]
    fn score_override_matches_materialized_psv_and_precedes_drop_clamp() {
        let base = [10, 20, 30];
        let replacement = [500, 50, -250];
        let (data, scores, _) = synthetic_override_files("equivalence", &base, &replacement, None);
        let materialized = data.with_extension("materialized.psv");
        let mut bytes = std::fs::read(&data).unwrap();
        for (record, score) in bytes
            .chunks_exact_mut(PSV_RECORD_BYTES as usize)
            .zip(replacement)
        {
            record[32..34].copy_from_slice(&score.to_le_bytes());
        }
        std::fs::write(&materialized, bytes).unwrap();
        let mut overridden = PsvEpochReader::new_range(
            &data,
            0,
            3 * PSV_RECORD_BYTES,
            Some(400),
            Some(100),
            Some(&scores),
            None,
            None,
        )
        .unwrap();
        let mut concrete = PsvEpochReader::new_range(
            &materialized,
            0,
            3 * PSV_RECORD_BYTES,
            Some(400),
            Some(100),
            None,
            None,
            None,
        )
        .unwrap();
        let override_scores: Vec<i16> =
            (0..4).map(|_| overridden.next().unwrap().score()).collect();
        let concrete_scores: Vec<i16> = (0..4).map(|_| concrete.next().unwrap().score()).collect();
        drop(overridden);
        drop(concrete);
        let _ = std::fs::remove_file(&materialized);
        remove_override_files(&data, &scores, None);
        assert_eq!(override_scores, concrete_scores);
        assert_eq!(override_scores, [50, -100, 50, -100]);
    }

    #[test]
    fn score_override_rejects_sidecar_and_mask_size_mismatches() {
        let base = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let replacement = [0; 9];
        let (data, scores, mask) =
            synthetic_override_files("bad-size", &base, &replacement, Some(&[0_u8; 2]));
        for delta in [-2_i64, 2] {
            let bad = scores.with_extension(format!("scores-{delta}"));
            let size = (replacement.len() as i64 * 2 + delta) as usize;
            std::fs::write(&bad, vec![0_u8; size]).unwrap();
            let err = ScoreOverrideReader::new(&data, &bad, None, 0).unwrap_err();
            let _ = std::fs::remove_file(&bad);
            assert!(err.to_string().contains("expected 18 bytes"));
        }
        for delta in [-1_i64, 1] {
            let bad = scores.with_extension(format!("mask-{delta}"));
            let size = (2_i64 + delta) as usize;
            std::fs::write(&bad, vec![0_u8; size]).unwrap();
            let err = ScoreOverrideReader::new(&data, &scores, Some(&bad), 0).unwrap_err();
            let _ = std::fs::remove_file(&bad);
            assert!(err.to_string().contains("expected 2 bytes"));
        }
        remove_override_files(&data, &scores, mask.as_deref());
        assert!(mask.is_some());
    }

    #[test]
    fn score_override_rejects_nonzero_unused_mask_bits() {
        let base = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let replacement = [0; 9];
        let mask = [0_u8, 0b0000_0010];
        let (data, scores, mask_path) =
            synthetic_override_files("unused-mask-bits", &base, &replacement, Some(&mask));

        let err = match PsvEpochReader::new_range(
            &data,
            0,
            9 * PSV_RECORD_BYTES,
            None,
            None,
            Some(&scores),
            mask_path.as_deref(),
            None,
        ) {
            Ok(_) => panic!("non-zero unused mask bits must be rejected"),
            Err(err) => err,
        };
        remove_override_files(&data, &scores, mask_path.as_deref());
        assert!(
            err.to_string().contains("non-zero unused bits"),
            "got: {err}"
        );
    }

    #[test]
    #[ignore = "requires matching external HCPE and PSV files"]
    fn hcpe_matches_psv_crosscheck() {
        let hcpe_path = std::env::var_os("TATARA_HCPE_CROSSCHECK")
            .map(PathBuf::from)
            .expect("set TATARA_HCPE_CROSSCHECK");
        let psv_path = std::env::var_os("TATARA_PSV_CROSSCHECK")
            .map(PathBuf::from)
            .expect("set TATARA_PSV_CROSSCHECK");
        let mut hcpe = HcpeFileLoader::new(hcpe_path).expect("open HCPE");
        let mut psv = PsvFileLoader::new(psv_path).expect("open PSV");
        let mut positions = 0_u64;

        loop {
            let hcpe_board = hcpe.next_board().expect("decode HCPE");
            let psv_board = psv.next_psv().expect("decode PSV").map(|p| p.decode());
            match (hcpe_board, psv_board) {
                (Some(h), Some(p)) => {
                    assert_eq!(h.board, p.board, "board mismatch at record {positions}");
                    assert_eq!(
                        h.black_hand.counts, p.black_hand.counts,
                        "black hand mismatch at record {positions}"
                    );
                    assert_eq!(
                        h.white_hand.counts, p.white_hand.counts,
                        "white hand mismatch at record {positions}"
                    );
                    assert_eq!(
                        h.side_to_move, p.side_to_move,
                        "side-to-move mismatch at record {positions}"
                    );
                    assert_eq!(
                        h.black_king_sq, p.black_king_sq,
                        "black king mismatch at record {positions}"
                    );
                    assert_eq!(
                        h.white_king_sq, p.white_king_sq,
                        "white king mismatch at record {positions}"
                    );
                    assert_eq!(h.score, p.score, "score mismatch at record {positions}");
                    assert_eq!(h.result, p.result, "result mismatch at record {positions}");
                    positions += 1;
                }
                (None, None) => break,
                _ => panic!("record count mismatch after {positions} positions"),
            }
        }
        assert!(positions > 0, "cross-check files are empty");
        eprintln!("cross-checked {positions} HCPE/PSV positions");
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
        // 末尾 50 行は fill が touch しない。fresh batch なので `with_capacity` の初期値
        // (-1 / 0.0) のまま (`reset` は buffer を clear しないが、初回 fill では
        // with_capacity 初期化がそのまま残る)。下流はこの領域を読まない。
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
    fn push_decoded_counting_aggregates_active_counts() {
        let spec = test_spec();
        let mut loader = PsvFileLoader::new(sample_psv_path()).unwrap();
        let mut batch = Batch::with_capacity(8, spec);
        let mut hist = vec![0u64; spec.max_active() + 1];

        let mut pushed = 0u64;
        for _ in 0..8 {
            let psv = loader.next_psv().unwrap().expect("record");
            let board = psv.decode();
            let bi = batch.n_positions;
            assert!(
                batch
                    .push_decoded_counting(&board, Some(&mut hist))
                    .unwrap()
            );
            let row = &batch.stm_indices[bi * spec.max_active()..(bi + 1) * spec.max_active()];
            let written = row.iter().take_while(|&&idx| idx >= 0).count();
            assert_eq!(batch.nnz[bi], written as i32);
            pushed += 1;
        }
        // histogram の総和は push した position 数と一致する。
        assert_eq!(hist.iter().sum::<u64>(), pushed);
        // すべての実 active 数は `max_active` の bin 域に収まる (padding index には
        // 入らない): non-zero の最大 bin が max_active 以下であることで確認。
        let max_bin = hist.iter().rposition(|&c| c > 0).expect("some active");
        assert!(max_bin <= spec.max_active());

        // batch 満杯後の push (`Ok(false)`) は histogram を増やさない。
        let extra = loader.next_psv().unwrap().expect("record").decode();
        assert!(
            !batch
                .push_decoded_counting(&extra, Some(&mut hist))
                .unwrap()
        );
        assert_eq!(
            hist.iter().sum::<u64>(),
            pushed,
            "batch 満杯時の push は histogram に加算しない"
        );
    }

    #[test]
    fn bucketed_loader_active_histogram_gated_by_flag() {
        let progress = ShogiProgressKPAbs;
        let path = sample_psv_path();
        let end = full_range_end(&path);

        // 計装 off: snapshot は None (histogram を確保しない = 集計しない)。
        let mut off = BucketedPrefetchedLoader::spawn(
            &path,
            8,
            None,
            None,
            1,
            progress,
            test_spec(),
            true,
            9,
            end,
            false,
        )
        .unwrap();
        let (batch, buckets) = off.next_batch().unwrap().expect("a batch");
        off.recycle((batch, buckets));
        assert!(
            off.active_histogram_snapshot().is_none(),
            "flag off では histogram を確保・集計しない"
        );
        drop(off);

        // 計装 on: snapshot は Some、長さ = max_active + 1、総和は消費した
        // position 数以上 (worker が先読みで余分に埋め得るため厳密一致は保証
        // しない)。全 active 数は max_active bin 域に収まる。
        let mut on = BucketedPrefetchedLoader::spawn(
            &path,
            8,
            None,
            None,
            1,
            progress,
            test_spec(),
            true,
            9,
            end,
            true,
        )
        .unwrap();
        let mut consumed = 0u64;
        for _ in 0..5 {
            let (batch, buckets) = on.next_batch().unwrap().expect("a batch");
            consumed += batch.n_positions as u64;
            on.recycle((batch, buckets));
        }
        let hist = on.active_histogram_snapshot().expect("histogram present");
        assert_eq!(hist.len(), test_spec().max_active() + 1);
        let total: u64 = hist.iter().sum();
        assert!(
            total >= consumed,
            "histogram total {total} は消費 position 数 {consumed} 以上"
        );
        assert!(total > 0, "on では position が集計される");
    }

    /// `reset` は `n_positions` を 0 に戻すだけの O(1) 操作で、index / score buffer を
    /// clear しない。再 fill 後、`nnz` 打ち切りで読む有効領域は前 batch の残骸に汚染
    /// されない (下流 kernel の per-slot early-out と同じ不変条件を host 側で検証する)。
    #[test]
    fn reset_is_o1_and_refill_ignores_stale_residue() {
        let spec = test_spec();
        let max_active = spec.max_active();
        // batch_size > sample.psv の record 数 (100) にして、`[n_positions, batch_size)`
        // の残骸 row が必ず存在する状態を作る (実長超 slot の有無に依存しない)。
        let cap = 150;
        let mut batch = Batch::with_capacity(cap, spec);

        // 1 回目の fill (100 record で打ち切り)。各 row の有効 index を記録しておく。
        PsvFileLoader::new(sample_psv_path())
            .unwrap()
            .fill_batch(&mut batch)
            .unwrap();
        let n_pos = batch.n_positions;
        assert_eq!(n_pos, 100);
        let valid_snapshot: Vec<Vec<i32>> = (0..n_pos)
            .map(|bi| {
                let base = bi * max_active;
                batch.stm_indices[base..base + batch.nnz[bi] as usize].to_vec()
            })
            .collect();

        // 「前 batch の残骸」を模した範囲内 index を実長超 slot (row 内 tail) と
        // `[n_pos, cap)` の未使用 row に書き込む (`idx >= 0` 防御 skip を素通りする値)。
        for bi in 0..n_pos {
            for ni in batch.nnz[bi] as usize..max_active {
                batch.stm_indices[bi * max_active + ni] = 7;
                batch.nstm_indices[bi * max_active + ni] = 7;
            }
        }
        for j in n_pos * max_active..cap * max_active {
            batch.stm_indices[j] = 7;
        }
        batch.score[cap - 1] = 12_345.0;

        // reset は index / score を clear しない (残骸が残ることで O(1) 化を確認)。
        batch.reset();
        assert_eq!(batch.n_positions, 0);
        assert_eq!(
            batch.stm_indices[n_pos * max_active],
            7,
            "reset は index buffer を clear しない (残骸 row が残る)"
        );
        assert_eq!(
            batch.score[cap - 1],
            12_345.0,
            "reset は score buffer を clear しない"
        );

        // 2 回目の fill。同一ファイル先頭からなので各 row は同じ局面で埋まる。
        PsvFileLoader::new(sample_psv_path())
            .unwrap()
            .fill_batch(&mut batch)
            .unwrap();
        assert_eq!(batch.n_positions, n_pos);

        // 下流が読む領域 (`nnz` 打ち切り / `n_pos` 行) は 1 回目と bit 一致し、
        // 残骸 (7) を含まない。tail / `[n_pos, cap)` の 7 は下流に観測されない。
        for (bi, expected) in valid_snapshot.iter().enumerate() {
            let base = bi * max_active;
            let n = batch.nnz[bi] as usize;
            assert_eq!(
                &batch.stm_indices[base..base + n],
                expected.as_slice(),
                "position {bi} の有効 slot は新データのみ (残骸は nnz 打ち切りで除外)"
            );
            assert!(
                batch.stm_indices[base..base + n].iter().all(|&i| i >= 0),
                "position {bi} の有効 slot に padding/残骸が混入していない"
            );
        }
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

    fn assert_inline_matches_sidecar(mode: DualLabelMode, preserve: &[usize]) {
        let base_path = sample_psv_path();
        let base_bytes = std::fs::read(&base_path).expect("read sample PSV");
        let records = base_bytes.len() / PSV_RECORD_BYTES as usize;
        let suffix = mode.canonical_name();
        let prefix = std::env::temp_dir().join(format!(
            "nnue-train-dual-label-stream-{suffix}-{}",
            std::process::id()
        ));
        let dual_path = prefix.with_extension("psv");
        let score_path = prefix.with_extension("i16");
        let mask_path = prefix.with_extension("bits");
        let mut dual_bytes = base_bytes.clone();
        let mut score_bytes = Vec::with_capacity(records * 2);
        let mut mask = vec![0_u8; records.div_ceil(8)];
        for (index, record) in dual_bytes
            .chunks_exact_mut(PSV_RECORD_BYTES as usize)
            .enumerate()
        {
            let score = (index as i16).wrapping_mul(37).wrapping_sub(1400);
            score_bytes.extend_from_slice(&score.to_le_bytes());
            record[34..36].copy_from_slice(&score.to_le_bytes());
            let is_preserved = preserve.contains(&index);
            record[39] = u8::from(is_preserved);
            if is_preserved {
                mask[index / 8] |= 1 << (index % 8);
            }
        }
        std::fs::write(&dual_path, dual_bytes).unwrap();
        std::fs::write(&score_path, score_bytes).unwrap();
        if mode == DualLabelMode::Gated {
            std::fs::write(&mask_path, &mask).unwrap();
        }

        let end = base_bytes.len() as u64;
        let mut sidecar = BucketedPrefetchedLoader::spawn_with_score_sources(
            &base_path,
            16,
            None,
            None,
            1,
            BucketMode::Progress8KpAbs,
            test_spec(),
            false,
            1,
            end,
            false,
            Some(&score_path),
            (mode == DualLabelMode::Gated).then_some(mask_path.as_path()),
            0,
            false,
            0,
            None,
        )
        .unwrap();
        let mut inline = BucketedPrefetchedLoader::spawn_with_score_sources(
            &dual_path,
            16,
            None,
            None,
            1,
            BucketMode::Progress8KpAbs,
            test_spec(),
            false,
            1,
            end,
            false,
            None,
            None,
            0,
            false,
            0,
            Some(mode),
        )
        .unwrap();

        // 13 × 16 = 208 positions, so the 100-record fixture wraps twice.
        for batch_index in 0..13 {
            let sidecar_slot = sidecar.next_batch().unwrap().expect("sidecar batch");
            let inline_slot = inline.next_batch().unwrap().expect("inline batch");
            let (sidecar_batch, sidecar_buckets) = &sidecar_slot;
            let (inline_batch, inline_buckets) = &inline_slot;
            assert_eq!(sidecar_batch.n_positions, inline_batch.n_positions);
            assert_eq!(sidecar_buckets, inline_buckets);
            let n = sidecar_batch.n_positions;
            assert_eq!(
                &sidecar_batch.score[..n],
                &inline_batch.score[..n],
                "score batch {batch_index}"
            );
            assert_eq!(&sidecar_batch.wdl[..n], &inline_batch.wdl[..n]);
            assert_eq!(&sidecar_batch.nnz[..n], &inline_batch.nnz[..n]);
            for position in 0..n {
                let nnz = sidecar_batch.nnz[position] as usize;
                let start = position * sidecar_batch.max_active;
                assert_eq!(
                    &sidecar_batch.stm_indices[start..start + nnz],
                    &inline_batch.stm_indices[start..start + nnz]
                );
                assert_eq!(
                    &sidecar_batch.nstm_indices[start..start + nnz],
                    &inline_batch.nstm_indices[start..start + nnz]
                );
            }
            sidecar.recycle(sidecar_slot);
            inline.recycle(inline_slot);
        }
        drop(sidecar);
        drop(inline);
        let _ = std::fs::remove_file(&dual_path);
        let _ = std::fs::remove_file(&score_path);
        let _ = std::fs::remove_file(&mask_path);
    }

    #[test]
    fn bucketed_stream_gated_dual_label_is_bit_identical_to_masked_sidecar() {
        assert_inline_matches_sidecar(DualLabelMode::Gated, &[0, 7, 8, 16, 63, 64, 99]);
    }

    #[test]
    fn bucketed_stream_all_dual_label_is_bit_identical_to_unmasked_sidecar() {
        assert_inline_matches_sidecar(DualLabelMode::All, &[]);
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
            None,
            num_workers,
            progress,
            test_spec(),
            true,
            9,
            end,
            false,
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
    fn bucketed_loader_dispatches_kingrank9_without_progress_weights() {
        let path = sample_psv_path();
        let end = full_range_end(&path);
        let mut expected_reader = PsvFileLoader::new(&path).expect("open sample PSV");
        let mut expected = Vec::new();
        for _ in 0..16 {
            let board = expected_reader
                .next_psv()
                .expect("read sample PSV")
                .expect("sample record")
                .decode();
            expected.push(i32::from(kingrank9_bucket_board(&board)));
        }

        let mut loader = BucketedPrefetchedLoader::spawn(
            &path,
            16,
            None,
            None,
            1,
            BucketMode::KingRank9,
            test_spec(),
            true,
            9,
            end,
            false,
        )
        .expect("spawn KingRank9 loader");
        let (batch, buckets) = loader
            .next_batch()
            .expect("load batch")
            .expect("full batch");
        assert_eq!(batch.n_positions, 16);
        assert_eq!(buckets, expected);
        assert!(buckets.iter().all(|&bucket| (0..9).contains(&bucket)));
        loader.recycle((batch, buckets));
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
    fn bucketed_multi_worker_surfaces_dual_label_reserved_bit_errors() {
        let path = std::env::temp_dir().join(format!(
            "nnue-train-dual-label-worker-error-{}.psv",
            std::process::id()
        ));
        let mut bytes = std::fs::read(sample_psv_path()).expect("read sample PSV");
        for (index, record) in bytes
            .chunks_exact_mut(PSV_RECORD_BYTES as usize)
            .enumerate()
        {
            let dl = 1000 + index as i16;
            record[34..36].copy_from_slice(&dl.to_le_bytes());
            record[39] = 0;
        }
        bytes[31 * PSV_RECORD_BYTES as usize + 39] = 2;
        std::fs::write(&path, bytes).expect("write invalid dual-label PSV");

        let end = full_range_end(&path);
        let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
            &path,
            8,
            None,
            None,
            4,
            BucketMode::KingRank9,
            test_spec(),
            false,
            1,
            end,
            false,
            None,
            None,
            0,
            false,
            0,
            Some(DualLabelMode::All),
        )
        .expect("spawn multi-worker dual-label loader");
        let mut error = None;
        for _ in 0..64 {
            match loader.next_batch() {
                Err(err) => {
                    error = Some(err);
                    break;
                }
                Ok(Some(slot)) => loader.recycle(slot),
                Ok(None) => break,
            }
        }
        let err = error.expect("reserved-bit worker error must reach next_batch");
        assert!(err.to_string().contains("record 31"), "got: {err}");
        drop(loader);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bucketed_loader_zero_workers_normalizes_to_one() {
        let progress = ShogiProgressKPAbs;
        let path = sample_psv_path();
        let end = full_range_end(&path);
        let mut loader = BucketedPrefetchedLoader::spawn(
            &path,
            8,
            None,
            None,
            0,
            progress,
            test_spec(),
            true,
            9,
            end,
            false,
        )
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
            None,
            2,
            progress,
            test_spec(),
            true,
            9,
            end,
            false,
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
            None,
            1,
            progress,
            test_spec(),
            true,
            9,
            end,
            false,
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
            None,
            1,
            progress,
            test_spec(),
            true,
            9,
            2800,
            false,
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
        let mut reader =
            PsvEpochReader::new_range(&sample_psv_path(), 2800, 4000, None, None, None, None, None)
                .unwrap();
        for i in 0..100 {
            let _psv = reader
                .next()
                .unwrap_or_else(|e| panic!("wrap should keep returning records (i={i}): {e}"));
        }
    }

    #[test]
    fn psv_epoch_reader_clamps_after_drop() {
        // score だけ既知の synthetic PSV (盤面 bytes は reader レベルでは decode
        // されないので zero 埋めで良い)。drop 32000 → 詰み stamp ±32000 は clamp
        // されずに drop され、生き残りは ±100 に飽和されることを確認する。
        let scores: [i16; 7] = [0, 50, -50, 200, -200, 32000, -32000];
        let mut bytes = Vec::with_capacity(scores.len() * 40);
        for s in scores {
            let mut rec = [0u8; 40];
            rec[32..34].copy_from_slice(&s.to_le_bytes());
            bytes.extend_from_slice(&rec);
        }
        let tmp = std::env::temp_dir().join(format!(
            "nnue-train-clamp-after-drop-{}.psv",
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes).expect("write synthetic psv");

        let mut reader = PsvEpochReader::new_range(
            &tmp,
            0,
            bytes.len() as u64,
            Some(32000),
            Some(100),
            None,
            None,
            None,
        )
        .unwrap();
        let got: Vec<i16> = (0..5).map(|_| reader.next().unwrap().score()).collect();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(got, vec![0, 50, -50, 100, -100]);
    }

    #[test]
    fn bucketed_loader_empty_file_errors_not_hang() {
        let progress = ShogiProgressKPAbs;
        let tmp = std::env::temp_dir().join(format!(
            "nnue-train-bucketed-empty-{}.psv",
            std::process::id()
        ));
        std::fs::write(&tmp, b"").expect("write empty psv");
        let mut loader = BucketedPrefetchedLoader::spawn(
            &tmp,
            8,
            None,
            None,
            1,
            progress,
            test_spec(),
            true,
            9,
            0,
            false,
        )
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
