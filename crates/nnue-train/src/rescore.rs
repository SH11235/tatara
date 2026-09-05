//! GPU リスコア (PSV → 1-node 静的評価 → i16 score sidecar) の host 側部品。
//!
//! 教師 pool の relabel は「sidecar 行 `i` = 入力 record `i`」の行対応が不変条件で、
//! **全行・原順序・無フィルタ** を要求する。学習用の [`BucketedPrefetchedLoader`]
//! は worker 数 ≥ 2 で順序非決定、score-drop skip と epoch wrap を持つため流用
//! できず、本 module が同じ channel ring / slot recycle パターンの上に順序保存の
//! 別実装を提供する。
//!
//! ## 構成
//!
//! ```text
//! [worker × N]  seq = 共有 counter を claim → 自分の record range を
//!    │          PsvFileLoader::new_range で読む → decode → Batch + bucket
//!    ▼          (chunk 内容は byte range だけで決まり、スケジューリング非依存)
//! [result channel] → next_chunk() が seq 順に再整列 (乱れた chunk は pending に保持)
//!    ▼
//! [消費側]  GPU forward → i16 変換 → ScoreSidecarWriter で追記
//! ```
//!
//! - back-pressure: slot pool (bounded channel)。各 in-flight chunk (worker 保有 /
//!   result channel 内 / 消費側 pending) が slot を 1 個ずつ占有し、slot 総数が
//!   有限なので再整列 buffer もメモリも有界。
//! - fail-closed: worker のエラーは result channel 経由で `Err` として届き
//!   (blocking 中の相手を必ず起床させる)、worker 内 panic もエラーに変換して同じ
//!   経路で伝搬する。さらに全 worker 終了時に受領 chunk 数 = 期待 chunk 数を
//!   検証するため、chunk の欠損が正常終了 (`Ok(None)`) に化けることはない。
//!   一度 `Err` を返した loader は毒化され、以降の [`OrderedPsvLoader::next_chunk`]
//!   も同じエラーを返し続ける。
//! - record 検証: `PackedSfenValue::decode` は checked ではなく、壊れた record も
//!   何らかの局面に化ける。安価な整合性検証 ([`validate_board`]) で検出可能な
//!   破損 (玉の欠落 / 重複、駒数超過等) は硬いエラーにする。完全な合法性検証は
//!   しない — 入力は教師生成パイプラインが書いた正当な PSV であることが契約で、
//!   本検証はその契約違反の検出網にすぎない。
//!
//! [`BucketedPrefetchedLoader`]: crate::dataloader::BucketedPrefetchedLoader

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use shogi_features::FeatureSetSpec;
use shogi_format::ShogiBoard;
use shogi_format::types::{Color, HAND_PIECE_TYPES, PieceType, Square};

use crate::dataloader::{Batch, BucketMode, PSV_RECORD_BYTES, PsvFileLoader};

/// 入力順 1 chunk 分の decode 結果。
///
/// `batch.n_positions` は `n_real` を `pad_multiple` の倍数へ切り上げた数で、
/// 末尾 `batch.n_positions - n_real` 行は最終 real 行の複製 (GPU tiled kernel の
/// `b % 16 == 0` 制約を満たすための padding)。消費側は forward 出力を `n_real`
/// で truncate してから書き出すこと。
#[derive(Debug)]
pub struct RescoreChunk {
    /// 0 始まりの chunk 番号。`next_chunk` は必ず 0, 1, 2, … の順に返す。
    pub seq: u64,
    /// decode 済み batch (`n_positions` = padding 込みの行数)。
    pub batch: Batch,
    /// per-position の output bucket index (`batch.n_positions` 長、padding 行含む)。
    pub buckets: Vec<i32>,
    /// この chunk が担当した実入力 record 数 (padding を除く)。
    pub n_real: usize,
}

/// pool を還流する空 slot (Batch + bucket buffer の再利用)。
type EmptySlot = (Batch, Vec<i32>);

/// panic payload から人間可読なメッセージを取り出す。
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// PSV file を record 順を保存したまま並列 decode する chunk loader。
///
/// [`BucketedPrefetchedLoader`] と違い:
///
/// - **順序決定的**: chunk `seq` は入力 record 範囲
///   `[start_record + seq * chunk_records, …)` に固定で対応し、`next_chunk` は
///   seq 昇順に返す。worker 数・スケジューリングは出力内容に影響しない。
/// - **無フィルタ・wrap なし**: score-drop / clamp を適用せず、EOF で終了する。
///   yield される実 record 数の合計は必ず `remaining_records()` に一致する
///   (一致し得ない状態は error)。
/// - **fail-closed**: worker のエラー / panic は `next_chunk` の `Err` として
///   伝搬し、以降の chunk を 1 個も yield しない (module doc 参照)。
///
/// progresskpabs mode の重みは process-global なので、呼び出し前に
/// `ShogiProgressKPAbs::load_from_bin` 済みであること (未ロードなら全 bucket 4 —
/// [`BucketedPrefetchedLoader`] と同じ契約)。
///
/// [`BucketedPrefetchedLoader`]: crate::dataloader::BucketedPrefetchedLoader
pub struct OrderedPsvLoader {
    /// worker → 消費側。完成 chunk (`Ok`) と worker のエラー / panic (`Err`) の
    /// 両方がこの channel を流れるため、消費側は blocking recv のままエラーに
    /// 気付ける (side channel だと recv 待ちのまま永久に起きない)。
    /// `Drop` / 毒化で先に落とすため `Option`。
    result_rx: Option<mpsc::Receiver<io::Result<RescoreChunk>>>,
    /// 消費済み slot を worker へ返す ring。`Drop` / 毒化で先に落とすため `Option`。
    pool_tx: Option<mpsc::SyncSender<EmptySlot>>,
    /// seq 順再整列: 期待 seq より先に届いた chunk の待機所。in-flight chunk が
    /// slot を占有するため、この map も slot 総数で有界。
    pending: BTreeMap<u64, RescoreChunk>,
    /// 次に返すべき chunk 番号。
    next_seq: u64,
    /// これまでに result channel から受領した chunk 数 (pending 行き含む)。
    /// 全 worker 終了時に `expected_chunks` と照合し、欠損を正常終了に見せない。
    chunks_received: u64,
    /// yield されるべき chunk の総数 (= `remaining_records / chunk_records` 切上げ)。
    expected_chunks: u64,
    /// 一度 `Err` を返したらそのメッセージを保持し、以降の `next_chunk` も同じ
    /// エラーを返す (エラー後の呼び出しが `Ok(None)` = 完了に化けるのを防ぐ)。
    poisoned: Option<String>,
    /// file 全体の record 数。
    total_records: u64,
    /// 読み出し開始 record (resume 用)。`[start_record, total_records)` を yield する。
    start_record: u64,
    handles: Vec<thread::JoinHandle<()>>,
}

/// 既定の decode worker 数。実測 ~430k pos/s/worker に対し GPU forward は
/// ~1.0–1.4M pos/s (RTX 3080 Ti / 3072 net) なので、8 worker で GPU 律速に
/// 十分な余裕がある。`available_parallelism` (論理コア数) を 8 で cap する —
/// 対象マシンは物理コア ≥ 8 のため、cap 後は物理/論理の区別が消える。
pub fn default_decode_workers() -> usize {
    thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8)
}

impl OrderedPsvLoader {
    /// `path` の PSV を `[start_record, file 末尾)` について並列 decode する。
    ///
    /// - `chunk_records`: 1 chunk あたりの実 record 数。`pad_multiple` の倍数で
    ///   あること (末尾 chunk 以外が padding 不要になり、batch 容量 =
    ///   `chunk_records` で足りる)。
    /// - `pad_multiple`: `batch.n_positions` をこの倍数へ切り上げる (GPU tiled
    ///   kernel の `b % 16 == 0` 制約には 16 を渡す。1 で padding 無効)。
    /// - `num_workers`: decode worker thread 数 (最低 1 に切り上げ)。
    /// - `start_record`: 読み出し開始 record 番号 (sidecar resume 用)。chunk の
    ///   record 範囲はこの値からの相対で決まる。
    ///
    /// file size が [`PSV_RECORD_BYTES`] の倍数でない、または
    /// `start_record > 総 record 数` は error。
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        path: &Path,
        chunk_records: usize,
        pad_multiple: usize,
        num_workers: usize,
        bucket_mode: BucketMode,
        num_buckets: usize,
        feature_set: FeatureSetSpec,
        start_record: u64,
    ) -> io::Result<Self> {
        assert!(chunk_records >= 1, "chunk_records must be >= 1");
        assert!(pad_multiple >= 1, "pad_multiple must be >= 1");
        assert!(
            chunk_records.is_multiple_of(pad_multiple),
            "chunk_records ({chunk_records}) must be a multiple of pad_multiple ({pad_multiple})"
        );
        assert!(num_buckets >= 1, "num_buckets must be >= 1");
        let num_workers = num_workers.max(1);

        let file_size = std::fs::metadata(path)?.len();
        if !file_size.is_multiple_of(PSV_RECORD_BYTES) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSV file {} size {file_size} is not a multiple of the record size \
                     ({PSV_RECORD_BYTES} bytes); refusing to rescore a torn file",
                    path.display()
                ),
            ));
        }
        let total_records = file_size / PSV_RECORD_BYTES;
        if start_record > total_records {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "start record {start_record} exceeds the {total_records} records in {}",
                    path.display()
                ),
            ));
        }
        let chunk_records_u64 = chunk_records as u64;
        let expected_chunks = (total_records - start_record).div_ceil(chunk_records_u64);

        // slot pool: worker が同時に持つ分 + 消費側 (pending + 手元) が持つ分。
        // 2 倍 + 2 で「遅い chunk を待つ間に他 worker が先の chunk を進める」余裕を
        // 確保する。result channel は「全 slot が結果として滞留 + 各 worker の
        // エラー 1 件」まで格納できる容量にし、エラー送信が block しないようにする。
        let n_slots = num_workers * 2 + 2;
        let (result_tx, result_rx) =
            mpsc::sync_channel::<io::Result<RescoreChunk>>(n_slots + num_workers);
        let (pool_tx, pool_rx) = mpsc::sync_channel::<EmptySlot>(n_slots);
        for _ in 0..n_slots {
            let slot = (
                Batch::with_capacity(chunk_records, feature_set),
                Vec::with_capacity(chunk_records),
            );
            pool_tx
                .send(slot)
                .expect("pool channel has capacity for the initial slots");
        }
        let pool_rx = Arc::new(Mutex::new(pool_rx));
        let next_chunk_counter = Arc::new(AtomicU64::new(0));
        let path = path.to_path_buf();

        let mut handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let path = path.clone();
            let pool_rx = Arc::clone(&pool_rx);
            let result_tx = result_tx.clone();
            let counter = Arc::clone(&next_chunk_counter);
            let handle = thread::spawn(move || {
                loop {
                    // 空 slot を借りてから seq を claim する。claim 済み chunk は
                    // 必ず slot を伴うため、in-flight chunk 数 (= pending の上限)
                    // が slot 総数で抑えられる。
                    let (mut batch, mut buckets) = {
                        let rx = pool_rx.lock().expect("pool_rx mutex poisoned");
                        match rx.recv() {
                            Ok(slot) => slot,
                            Err(_) => return, // 消費側が pool_tx を drop → 終了
                        }
                    };
                    let seq = counter.fetch_add(1, Ordering::Relaxed);
                    let first = start_record + seq * chunk_records_u64;
                    if first >= total_records {
                        return; // 全 chunk 分配済み (借りた slot は捨てる)
                    }
                    let last = (first + chunk_records_u64).min(total_records);

                    // decode 中の panic (index 演算・下層の assert 等) も握って
                    // エラーとして送る。放置すると thread だけが死に、結果 channel
                    // 経由では何も伝わらず deadlock か silent EOF になる。
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        decode_chunk_into(
                            &path,
                            first,
                            last,
                            pad_multiple,
                            bucket_mode,
                            num_buckets,
                            &mut batch,
                            &mut buckets,
                        )
                    }));
                    let error = match outcome {
                        Ok(Ok(n_real)) => {
                            let chunk = RescoreChunk {
                                seq,
                                batch,
                                buckets,
                                n_real,
                            };
                            if result_tx.send(Ok(chunk)).is_err() {
                                return; // 消費側が drop 済み
                            }
                            continue;
                        }
                        Ok(Err(e)) => e,
                        Err(payload) => io::Error::other(format!(
                            "decode worker panicked on records [{first}, {last}): {}",
                            panic_message(payload.as_ref())
                        )),
                    };
                    let _ = result_tx.send(Err(error));
                    return;
                }
            });
            handles.push(handle);
        }
        drop(result_tx);

        Ok(Self {
            result_rx: Some(result_rx),
            pool_tx: Some(pool_tx),
            pending: BTreeMap::new(),
            next_seq: 0,
            chunks_received: 0,
            expected_chunks,
            poisoned: None,
            total_records,
            start_record,
            handles,
        })
    }

    /// file 全体の record 数 (`start_record` を含む)。
    pub fn total_records(&self) -> u64 {
        self.total_records
    }

    /// yield される実 record 数の合計 (= `total_records - start_record`)。
    pub fn remaining_records(&self) -> u64 {
        self.total_records - self.start_record
    }

    /// 次の chunk を **seq 昇順で** 返す。
    ///
    /// - `Ok(Some(chunk))`: `chunk.seq` は前回 +1 (初回は 0)。
    /// - `Ok(None)`: 全 chunk (`remaining_records()` 分) を返し終えた。受領
    ///   chunk 数の照合を通った場合のみ返る。
    /// - `Err(e)`: worker のエラー / panic、または chunk 欠損。loader は毒化され、
    ///   以降の呼び出しも同じエラーを返す。
    ///
    /// 消費後は [`Self::recycle`] で slot を返すこと (返さないと pool が枯れて
    /// 以降の decode が止まる)。
    pub fn next_chunk(&mut self) -> io::Result<Option<RescoreChunk>> {
        if let Some(message) = &self.poisoned {
            return Err(io::Error::other(message.clone()));
        }
        loop {
            if let Some(chunk) = self.pending.remove(&self.next_seq) {
                self.next_seq += 1;
                return Ok(Some(chunk));
            }
            let Some(result_rx) = self.result_rx.as_ref() else {
                return Ok(None); // 完了済み
            };
            match result_rx.recv() {
                Ok(Ok(chunk)) => {
                    self.chunks_received += 1;
                    if chunk.seq == self.next_seq {
                        self.next_seq += 1;
                        return Ok(Some(chunk));
                    }
                    let evicted = self.pending.insert(chunk.seq, chunk);
                    debug_assert!(evicted.is_none(), "duplicate chunk seq");
                }
                Ok(Err(e)) => return Err(self.poison(e)),
                Err(_) => {
                    // 全 worker が result_tx を落とした = 終了済み。catch_unwind の
                    // 外 (channel 送信等) で panic した worker はエラーを送れない
                    // ため、join でも panic を確認する (backstop)。
                    if let Some(e) = self.join_worker_panics() {
                        return Err(self.poison(e));
                    }
                    if let Some(chunk) = self.pending.remove(&self.next_seq) {
                        self.next_seq += 1;
                        return Ok(Some(chunk));
                    }
                    if self.chunks_received != self.expected_chunks {
                        let e = io::Error::other(format!(
                            "decode workers exited after delivering {} of {} chunks; \
                             refusing to treat the missing tail as a clean EOF",
                            self.chunks_received, self.expected_chunks
                        ));
                        return Err(self.poison(e));
                    }
                    if !self.pending.is_empty() {
                        // 受領数が合うのに next_seq が欠けている = seq 重複等の
                        // 内部契約違反。欠損のまま完了扱いにしない。
                        let e = io::Error::other(format!(
                            "chunk {} is missing while later chunks exist; refusing to \
                             continue with a gap in the record order",
                            self.next_seq
                        ));
                        return Err(self.poison(e));
                    }
                    self.result_rx = None;
                    self.pool_tx = None;
                    return Ok(None);
                }
            }
        }
    }

    /// 消費済み chunk の buffer を worker pool へ返す (ring buffer)。
    pub fn recycle(&mut self, chunk: RescoreChunk) {
        if let Some(pool_tx) = self.pool_tx.as_ref() {
            // worker が全て終了した後は返せなくてよい (エラーは無視して捨てる)。
            let _ = pool_tx.send((chunk.batch, chunk.buckets));
        }
    }

    /// エラーで停止する: channel を閉じて全 worker を起こし、join してから
    /// 毒化する。以降の `next_chunk` は同じメッセージのエラーを返す。
    fn poison(&mut self, error: io::Error) -> io::Error {
        self.pool_tx.take();
        self.result_rx.take();
        self.pending.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.poisoned = Some(error.to_string());
        error
    }

    /// 全 worker (終了済み) を join し、panic していれば最初の 1 件をエラーに
    /// して返す。result channel が閉じた後にのみ呼ぶこと。
    fn join_worker_panics(&mut self) -> Option<io::Error> {
        let mut first_panic = None;
        for handle in self.handles.drain(..) {
            if let Err(payload) = handle.join() {
                first_panic.get_or_insert_with(|| panic_message(payload.as_ref()));
            }
        }
        first_panic.map(|message| io::Error::other(format!("decode worker panicked: {message}")))
    }
}

impl Drop for OrderedPsvLoader {
    fn drop(&mut self) {
        // channel の両端を先に落として worker の recv / send を unblock してから
        // join する (close-then-join。BucketedPrefetchedLoader と同じ手順)。
        self.result_rx.take();
        self.pool_tx.take();
        self.pending.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// record 範囲 `[first, last)` を decode して `batch` / `buckets` に詰め、実
/// record 数を返す。`pad_multiple` の倍数まで最終行を複製して padding する。
#[allow(clippy::too_many_arguments)]
fn decode_chunk_into(
    path: &Path,
    first: u64,
    last: u64,
    pad_multiple: usize,
    bucket_mode: BucketMode,
    num_buckets: usize,
    batch: &mut Batch,
    buckets: &mut Vec<i32>,
) -> io::Result<usize> {
    batch.reset();
    buckets.clear();
    let mut loader =
        PsvFileLoader::new_range(path, first * PSV_RECORD_BYTES, last * PSV_RECORD_BYTES)?;
    let mut last_decoded = None;
    while let Some(psv) = loader.next_psv()? {
        let board = psv.decode();
        if let Err(reason) = validate_board(&board) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "record {} in {} decodes to a corrupt position ({reason}); \
                     refusing to write a score for it",
                    first + batch.n_positions as u64,
                    path.display()
                ),
            ));
        }
        let pushed = batch.push_decoded(&board)?;
        debug_assert!(pushed, "Batch::push_decoded refused below chunk capacity");
        buckets.push(i32::from(bucket_mode.bucket_board(&board, num_buckets)));
        last_decoded = Some(board);
    }
    let n_real = batch.n_positions;
    if n_real as u64 != last - first {
        // spawn 時の file size に対し short read = 実行中に file が縮んだ等。
        // 欠損行を黙って詰めると行対応が壊れるため fail-closed。
        return Err(io::Error::other(format!(
            "chunk [{first}, {last}) of {} yielded {n_real} records (expected {}); \
             the input changed while rescoring",
            path.display(),
            last - first
        )));
    }
    if let Some(board) = &last_decoded {
        let pad_bucket = *buckets.last().expect("n_real >= 1 implies a bucket");
        while !batch.n_positions.is_multiple_of(pad_multiple) {
            let pushed = batch.push_decoded(board)?;
            debug_assert!(pushed, "padding must fit in the chunk capacity");
            buckets.push(pad_bucket);
        }
    }
    Ok(n_real)
}

/// decode 済み局面の安価な整合性検証。
///
/// `PackedSfenValue::decode` は checked ではなく、壊れた record も何らかの
/// `ShogiBoard` に化けるため、検出可能な破損だけでも硬いエラーにする:
///
/// - 玉が両陣営に 1 枚ずつ盤上にあり、`black_king_sq` / `white_king_sq` の
///   マスと一致する
/// - 盤上 + 持ち駒の総数が 40 枚以下 (将棋の全駒数)
///
/// 完全な合法性検証はしない (入力は教師生成パイプラインが書いた正当な PSV で
/// あることが契約)。
fn validate_board(board: &ShogiBoard) -> Result<(), String> {
    let mut on_board = 0_u32;
    let mut black_kings = 0_u32;
    let mut white_kings = 0_u32;
    for piece in &board.board {
        if piece.piece_type == PieceType::None {
            continue;
        }
        on_board += 1;
        if piece.piece_type == PieceType::King {
            match piece.color {
                Color::Black => black_kings += 1,
                Color::White => white_kings += 1,
            }
        }
    }
    if black_kings != 1 || white_kings != 1 {
        return Err(format!(
            "kings on board: black {black_kings} / white {white_kings} (expected exactly 1 each)"
        ));
    }
    let king_matches = |sq: Square, color: Color| {
        sq.index() < 81 && {
            let piece = board.board[sq.index()];
            piece.piece_type == PieceType::King && piece.color == color
        }
    };
    if !king_matches(board.black_king_sq, Color::Black)
        || !king_matches(board.white_king_sq, Color::White)
    {
        return Err("king square fields do not match the board".to_string());
    }
    let mut in_hand = 0_u32;
    for pt in HAND_PIECE_TYPES {
        in_hand += u32::from(board.black_hand.count(pt)) + u32::from(board.white_hand.count(pt));
    }
    let total = on_board + in_hand;
    if total > 40 {
        return Err(format!(
            "{total} pieces on board + in hand (shogi has at most 40)"
        ));
    }
    Ok(())
}

/// i16 score sidecar の record サイズ (little-endian i16)。
pub const SCORE_RECORD_BYTES: u64 = 2;

/// [`ScoreSidecarWriter::open`] の結果。
pub enum SidecarOpen {
    /// `.done` marker と sidecar が現在の fingerprint に一致 — 再生成不要。
    Complete,
    /// 書き込み続行。`resume_records` は既に書かれている record 数で、入力の
    /// 読み出しをこの record から始めること。
    Writer {
        writer: ScoreSidecarWriter,
        resume_records: u64,
    },
}

/// sidecar の書き込み先。本番は [`File`] で、write エラー注入テストが差し替える。
trait SidecarSink: Write + Send {
    /// OS buffer まで含めて永続化する ([`File::sync_all`] 相当)。
    fn sync_all(&self) -> io::Result<()>;
}

impl SidecarSink for File {
    fn sync_all(&self) -> io::Result<()> {
        File::sync_all(self)
    }
}

/// little-endian i16 score sidecar の追記 writer + marker 管理。
///
/// rshogi `rescore_psv --out-scores` と同じ規約:
///
/// - 出力と並べて `<出力名>.in-progress` marker に fingerprint text を置き、
///   fingerprint が一致するときだけ既存 sidecar へ件数ベースで追記再開する。
/// - 正常完了時に全件数を検証して `<出力名>.done` (同じ fingerprint) へ昇格し、
///   in-progress marker を削除する。
/// - 途中終了 (drop) は marker と書き込み済み prefix を残す — prefix は常に
///   入力と行対応しているので、次回そのまま resume できる。
/// - **write エラー後は毒化する**: 部分書き込みされた slice を同一 writer で
///   retry すると prefix の二重 append や record 途中への継ぎ足しが起きるため、
///   以降の `write_scores` / `finish` は同じエラーを返し続ける。復旧は次回
///   起動時の `open` (件数ベース resume。2 byte 境界で切れていない末尾は硬い
///   エラー) が担う。
///
/// fingerprint は呼び出し側 (rescore driver) が組み立てる不透明な text で、
/// 入力・net・routing・スケール等「出力を変える全条件」を書き込む契約。writer は
/// byte 等値でのみ比較する。
pub struct ScoreSidecarWriter {
    out: BufWriter<Box<dyn SidecarSink>>,
    path: PathBuf,
    fingerprint: String,
    /// 書き込み済み record 数 (resume 分を含む)。
    written: u64,
    /// 完了時に一致すべき総 record 数。
    expected: u64,
    /// write エラー後の毒化メッセージ (module doc の fail-closed 方針と同じ思想)。
    poisoned: Option<String>,
}

/// `<sidecar>.in-progress` の path。
pub fn in_progress_marker_path(sidecar: &Path) -> PathBuf {
    append_extension(sidecar, ".in-progress")
}

/// `<sidecar>.done` の path。
pub fn done_marker_path(sidecar: &Path) -> PathBuf {
    append_extension(sidecar, ".done")
}

fn append_extension(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// marker file を読む。無ければ `None`。
fn read_marker(path: &Path) -> io::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

impl ScoreSidecarWriter {
    /// sidecar を resume 判定込みで開く。
    ///
    /// - `.done` marker が `fingerprint` と一致し sidecar size も
    ///   `expected_records * 2` に一致 → [`SidecarOpen::Complete`] (何も変更しない)。
    /// - `.in-progress` marker が `fingerprint` と一致 → sidecar の
    ///   `size / 2` record から追記再開。size が 2 の倍数でない場合は error
    ///   (書きかけ末尾の自己修復は事後検出不能なため fail-closed)。
    /// - それ以外 (marker 無し / 不一致 / 壊れた `.done`) → sidecar を truncate
    ///   し、marker を書き直して record 0 から。
    pub fn open(
        sidecar: &Path,
        expected_records: u64,
        fingerprint: &str,
    ) -> io::Result<SidecarOpen> {
        let in_progress = in_progress_marker_path(sidecar);
        let done = done_marker_path(sidecar);

        if let Some(done_text) = read_marker(&done)? {
            let size_matches = std::fs::metadata(sidecar)
                .map(|m| m.len() == expected_records * SCORE_RECORD_BYTES)
                .unwrap_or(false);
            if done_text == fingerprint && size_matches {
                // 完了済み。中断残骸の in-progress marker だけ掃除する。
                remove_if_exists(&in_progress)?;
                return Ok(SidecarOpen::Complete);
            }
            // 設定が変わった / sidecar が欠けた → 作り直し。sidecar を先に空に
            // してから marker を消す (逆順だと truncate 失敗時に古い marker +
            // 別条件の sidecar が残り、誤って完了扱いになり得る)。
            truncate_if_exists(sidecar)?;
            remove_if_exists(&done)?;
        }

        let resume_records = match read_marker(&in_progress)? {
            Some(text) if text == fingerprint && sidecar.exists() => {
                let size = std::fs::metadata(sidecar)?.len();
                if !size.is_multiple_of(SCORE_RECORD_BYTES) {
                    return Err(io::Error::other(format!(
                        "sidecar {} size {size} is not a multiple of {SCORE_RECORD_BYTES}; \
                         delete the marker {} so the next run regenerates the sidecar",
                        sidecar.display(),
                        in_progress.display()
                    )));
                }
                let resume = size / SCORE_RECORD_BYTES;
                if resume > expected_records {
                    return Err(io::Error::other(format!(
                        "sidecar {} already has {resume} records but only \
                         {expected_records} are expected; delete the sidecar and the \
                         marker {} to regenerate",
                        sidecar.display(),
                        in_progress.display()
                    )));
                }
                resume
            }
            _ => {
                // marker 無し / fingerprint 不一致 / sidecar 不在 → 最初から。
                truncate_if_exists(sidecar)?;
                std::fs::write(&in_progress, fingerprint)?;
                0
            }
        };

        let file = File::options().create(true).append(true).open(sidecar)?;
        Ok(SidecarOpen::Writer {
            writer: Self {
                out: BufWriter::with_capacity(1 << 20, Box::new(file)),
                path: sidecar.to_path_buf(),
                fingerprint: fingerprint.to_string(),
                written: resume_records,
                expected: expected_records,
                poisoned: None,
            },
            resume_records,
        })
    }

    /// write エラー注入テスト用: marker 判定を通さず任意 sink の writer を作る。
    /// buffer 容量 0 で write が即座に sink へ到達する。
    #[cfg(test)]
    fn with_sink_for_test(
        sink: Box<dyn SidecarSink>,
        sidecar: &Path,
        expected_records: u64,
        fingerprint: &str,
    ) -> Self {
        Self {
            out: BufWriter::with_capacity(0, sink),
            path: sidecar.to_path_buf(),
            fingerprint: fingerprint.to_string(),
            written: 0,
            expected: expected_records,
            poisoned: None,
        }
    }

    /// 1 chunk 分の score を追記する。呼び出し側は **エラーの無い完結した chunk**
    /// だけを渡す契約 (途中行までの書き込みは行対応を壊すため、chunk 単位で
    /// all-or-nothing)。write エラーは slice の一部だけが出力へ残った可能性が
    /// あるため writer を毒化する (超過検出は 1 byte も書く前に返すので毒化
    /// しない)。
    pub fn write_scores(&mut self, scores: &[i16]) -> io::Result<()> {
        if let Some(message) = &self.poisoned {
            return Err(io::Error::other(message.clone()));
        }
        let new_total = self.written + scores.len() as u64;
        if new_total > self.expected {
            return Err(io::Error::other(format!(
                "writing {} scores would exceed the expected {} records \
                 (already written {})",
                scores.len(),
                self.expected,
                self.written
            )));
        }
        for &score in scores {
            if let Err(e) = self.out.write_all(&score.to_le_bytes()) {
                let message = format!(
                    "sidecar write to {} failed and may have left a partial record \
                     ({e}); this writer is poisoned — restart to resume from the \
                     in-progress marker",
                    self.path.display()
                );
                self.poisoned = Some(message.clone());
                return Err(io::Error::new(e.kind(), message));
            }
        }
        self.written = new_total;
        Ok(())
    }

    /// 書き込み済み record 数 (resume 分を含む)。
    pub fn written(&self) -> u64 {
        self.written
    }

    /// 全件書き込みを検証して `.done` へ昇格する。
    ///
    /// 件数不足・毒化済みは error で、その場合 in-progress marker は残る (次回
    /// resume 可能)。昇格前に**物理ファイルサイズ = 論理 record 数 × 2** を検証
    /// する (write エラーの取りこぼしや外部からの混入で崩れた sidecar に `.done`
    /// を付けない)。`.done` は一時 file + rename で atomic に書き、その後
    /// in-progress marker を削除する (この順序なら間で停止しても次回は `.done`
    /// を正として回収できる)。
    pub fn finish(mut self) -> io::Result<()> {
        if let Some(message) = &self.poisoned {
            return Err(io::Error::other(message.clone()));
        }
        if self.written != self.expected {
            return Err(io::Error::other(format!(
                "sidecar {} has {} of {} expected records; keeping the in-progress \
                 marker so the next run resumes",
                self.path.display(),
                self.written,
                self.expected
            )));
        }
        self.out.flush()?;
        self.out.get_ref().sync_all()?;
        let physical = std::fs::metadata(&self.path)?.len();
        if physical != self.written * SCORE_RECORD_BYTES {
            return Err(io::Error::other(format!(
                "sidecar {} physical size {physical} does not match the {} written \
                 records ({} bytes); the file was modified or a write was torn — \
                 keeping the in-progress marker instead of promoting",
                self.path.display(),
                self.written,
                self.written * SCORE_RECORD_BYTES
            )));
        }

        let done = done_marker_path(&self.path);
        let tmp = append_extension(&self.path, ".done.tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(self.fingerprint.as_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &done)?;
        remove_if_exists(&in_progress_marker_path(&self.path))?;
        Ok(())
    }
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn truncate_if_exists(path: &Path) -> io::Result<()> {
    match File::options().write(true).open(path) {
        Ok(f) => f.set_len(0),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_features::FeatureSet;
    use std::time::Duration;

    fn test_spec() -> FeatureSetSpec {
        FeatureSet::HalfKaHmMerged.spec()
    }

    /// shogi-format crate test fixture (100 records × 40 bytes)。
    fn sample_psv_bytes() -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/nnue-train has a parent dir")
            .join("shogi-format/tests/data/sample.psv");
        std::fs::read(path).expect("read sample.psv fixture")
    }

    /// sample.psv を `repeat` 回連結した一時 PSV を作る (100 * repeat records)。
    fn temp_psv(tag: &str, repeat: usize) -> PathBuf {
        let bytes = sample_psv_bytes();
        let mut all = Vec::with_capacity(bytes.len() * repeat);
        for _ in 0..repeat {
            all.extend_from_slice(&bytes);
        }
        let path = std::env::temp_dir().join(format!(
            "tatara-rescore-{tag}-{}-{repeat}.psv",
            std::process::id()
        ));
        std::fs::write(&path, all).expect("write temp PSV");
        path
    }

    /// 単一スレッドの逐次 decode を参照実装として `(score, bucket)` 列を作る。
    fn sequential_reference(path: &Path, start_record: u64) -> (Vec<f32>, Vec<i32>) {
        let size = std::fs::metadata(path).unwrap().len();
        let mut loader =
            PsvFileLoader::new_range(path, start_record * PSV_RECORD_BYTES, size).unwrap();
        let mut scores = Vec::new();
        let mut buckets = Vec::new();
        while let Some(psv) = loader.next_psv().unwrap() {
            let board = psv.decode();
            scores.push(f32::from(board.score));
            buckets.push(i32::from(BucketMode::ProgressKpAbs.bucket_board(&board, 9)));
        }
        (scores, buckets)
    }

    /// loader を最後まで消費し、実 record 分の `(score, bucket)` 列と chunk ごとの
    /// `(n_real, n_positions)` を集める。seq の昇順連番も検証する。
    fn drain(loader: &mut OrderedPsvLoader) -> (Vec<f32>, Vec<i32>, Vec<(usize, usize)>) {
        let mut scores = Vec::new();
        let mut buckets = Vec::new();
        let mut shapes = Vec::new();
        let mut expect_seq = 0;
        while let Some(chunk) = loader.next_chunk().expect("next_chunk") {
            assert_eq!(chunk.seq, expect_seq, "chunks must arrive in order");
            expect_seq += 1;
            assert!(chunk.n_real >= 1);
            assert!(chunk.n_real <= chunk.batch.n_positions);
            scores.extend_from_slice(&chunk.batch.score[..chunk.n_real]);
            buckets.extend_from_slice(&chunk.buckets[..chunk.n_real]);
            shapes.push((chunk.n_real, chunk.batch.n_positions));
            loader.recycle(chunk);
        }
        (scores, buckets, shapes)
    }

    /// loader をエラーが出るまで消費して、そのエラーを返す (timeout 付き実行用)。
    fn drain_until_error(mut loader: OrderedPsvLoader) -> io::Result<()> {
        loop {
            match loader.next_chunk() {
                Ok(Some(chunk)) => loader.recycle(chunk),
                Ok(None) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    }

    #[test]
    fn ordered_loader_preserves_input_order_and_yields_every_record() {
        // 500 records / chunk 64 / worker 4。score-drop 相当の大 |score| 行を含む
        // fixture が 1 行も落ちず、逐次参照と同一順序・同一内容で出てくる。
        let path = temp_psv("order", 5);
        let (ref_scores, ref_buckets) = sequential_reference(&path, 0);
        assert_eq!(ref_scores.len(), 500);

        let mut loader = OrderedPsvLoader::spawn(
            &path,
            64,
            16,
            4,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        assert_eq!(loader.total_records(), 500);
        assert_eq!(loader.remaining_records(), 500);
        let (scores, buckets, shapes) = drain(&mut loader);
        assert_eq!(scores, ref_scores);
        assert_eq!(buckets, ref_buckets);
        // 500 = 64 × 7 + 52 → 8 chunks。
        assert_eq!(shapes.len(), 8);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ordered_loader_pads_final_partial_chunk_with_last_row() {
        // 500 % 64 = 52、52 は 16 の倍数でないため 64 へ padding される。padding 行は
        // 最終 real 行の複製 (score / bucket が一致)。他の chunk は padding 無し。
        let path = temp_psv("pad", 5);
        let mut loader = OrderedPsvLoader::spawn(
            &path,
            64,
            16,
            3,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        let mut last_chunk = None;
        while let Some(chunk) = loader.next_chunk().expect("next_chunk") {
            if let Some(prev) = last_chunk.replace(chunk) {
                assert_eq!(prev.n_real, 64);
                assert_eq!(prev.batch.n_positions, 64);
                loader.recycle(prev);
            }
        }
        let last = last_chunk.expect("at least one chunk");
        assert_eq!(last.n_real, 52);
        assert_eq!(
            last.batch.n_positions, 64,
            "52 rounds up to the next multiple of 16"
        );
        let pad_score = last.batch.score[51];
        let pad_bucket = last.buckets[51];
        for i in 52..64 {
            assert_eq!(
                last.batch.score[i], pad_score,
                "padding row {i} duplicates the last row"
            );
            assert_eq!(last.buckets[i], pad_bucket, "padding bucket {i}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ordered_loader_exact_multiple_needs_no_padding() {
        // 400 records / chunk 80 (16 の倍数) → 全 chunk が満杯で padding 無し。
        let path = temp_psv("exact", 4);
        let mut loader = OrderedPsvLoader::spawn(
            &path,
            80,
            16,
            2,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        let (scores, _, shapes) = drain(&mut loader);
        assert_eq!(scores.len(), 400);
        assert_eq!(shapes, vec![(80, 80); 5]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ordered_loader_resumes_from_start_record() {
        // start_record = 130 → 逐次参照の 130 行目以降と一致する。
        let path = temp_psv("resume", 3);
        let (ref_scores, ref_buckets) = sequential_reference(&path, 130);
        assert_eq!(ref_scores.len(), 170);

        let mut loader = OrderedPsvLoader::spawn(
            &path,
            48,
            16,
            4,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            130,
        )
        .expect("spawn loader");
        assert_eq!(loader.remaining_records(), 170);
        let (scores, buckets, _) = drain(&mut loader);
        assert_eq!(scores, ref_scores);
        assert_eq!(buckets, ref_buckets);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ordered_loader_is_deterministic_across_runs() {
        let path = temp_psv("determinism", 4);
        let run = || {
            let mut loader = OrderedPsvLoader::spawn(
                &path,
                32,
                16,
                4,
                BucketMode::ProgressKpAbs,
                9,
                test_spec(),
                0,
            )
            .expect("spawn loader");
            drain(&mut loader)
        };
        let (s1, b1, shapes1) = run();
        let (s2, b2, shapes2) = run();
        assert_eq!(s1, s2);
        assert_eq!(b1, b2);
        assert_eq!(shapes1, shapes2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ordered_loader_rejects_torn_file_and_bad_start() {
        // record 境界で終わらない file は spawn で拒否 (fail-closed)。
        let path =
            std::env::temp_dir().join(format!("tatara-rescore-torn-{}.psv", std::process::id()));
        std::fs::write(&path, vec![0_u8; 40 * 3 + 7]).unwrap();
        let err = OrderedPsvLoader::spawn(
            &path,
            16,
            16,
            1,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .map(|_| ())
        .expect_err("torn file must be rejected");
        assert!(err.to_string().contains("not a multiple"), "{err}");
        let _ = std::fs::remove_file(&path);

        // start_record > 総 record 数も拒否。
        let path = temp_psv("badstart", 1);
        let err = OrderedPsvLoader::spawn(
            &path,
            16,
            16,
            1,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            101,
        )
        .map(|_| ())
        .expect_err("start beyond EOF must be rejected");
        assert!(err.to_string().contains("exceeds"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ordered_loader_fails_closed_when_input_shrinks() {
        // spawn 後に file が縮むと、該当 chunk の short read が error として
        // 伝搬し、以降の chunk は yield されない。
        let path = temp_psv("shrink", 500); // 50k records
        let loader = OrderedPsvLoader::spawn(
            &path,
            16,
            16,
            1,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(40 * 100)
            .unwrap();
        let err = drain_until_error(loader).expect_err("shrunk input must not complete cleanly");
        let msg = err.to_string();
        assert!(
            msg.contains("the input changed") || msg.contains("range"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// worker ≥ 2 でのエラー: 消費側が deadlock せず、正常完了にも化けず、
    /// timeout 内に `Err` で停止する。
    #[test]
    fn ordered_loader_error_with_multiple_workers_fails_fast_without_deadlock() {
        let path = temp_psv("mwerr", 500); // 50k records
        let loader = OrderedPsvLoader::spawn(
            &path,
            16,
            16,
            4,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(40 * 100)
            .unwrap();

        let (tx, rx) = mpsc::channel();
        let consumer = thread::spawn(move || {
            let _ = tx.send(drain_until_error(loader));
        });
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(result) => {
                let err = result.expect_err("shrunk input must not complete cleanly");
                let msg = err.to_string();
                assert!(
                    msg.contains("the input changed") || msg.contains("range"),
                    "unexpected error: {msg}"
                );
            }
            Err(_) => panic!("consumer deadlocked after a worker error"),
        }
        let _ = consumer.join();
        let _ = std::fs::remove_file(&path);
    }

    /// worker 内 panic はエラーに変換されて伝搬し、silent EOF (`Ok(None)`) に
    /// 化けない。エラー後の loader は毒化され、以降の呼び出しも同じエラー。
    #[test]
    fn ordered_loader_propagates_worker_panic_and_stays_poisoned() {
        let path = temp_psv("panic", 2);
        // ProgressKpAbs の bucket_board は num_buckets > 256 を worker 内 assert で
        // 拒否する。spawn 自体は通るため、panic → エラー変換の検証に使える。
        let mut loader = OrderedPsvLoader::spawn(
            &path,
            16,
            16,
            2,
            BucketMode::ProgressKpAbs,
            257,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        let err = loop {
            match loader.next_chunk() {
                Ok(Some(chunk)) => loader.recycle(chunk),
                Ok(None) => panic!("panicking workers must not look like a clean EOF"),
                Err(e) => break e,
            }
        };
        assert!(err.to_string().contains("panicked"), "{err}");

        let err2 = loader
            .next_chunk()
            .map(|_| ())
            .expect_err("a poisoned loader must keep failing");
        assert!(err2.to_string().contains("panicked"), "{err2}");
        let _ = std::fs::remove_file(&path);
    }

    /// 破損 record (decode が矛盾した局面に化けるもの) は fail-closed の
    /// エラーになり、score が書かれる側に流れない。
    #[test]
    fn ordered_loader_rejects_corrupt_record() {
        // 正常 3 records + 全 0 の 40 bytes (玉が両陣営に置けない) + 正常 1 record。
        let sample = sample_psv_bytes();
        let mut bytes = sample[..40 * 3].to_vec();
        bytes.extend_from_slice(&[0_u8; 40]);
        bytes.extend_from_slice(&sample[..40]);
        let path =
            std::env::temp_dir().join(format!("tatara-rescore-corrupt-{}.psv", std::process::id()));
        std::fs::write(&path, bytes).unwrap();

        let loader = OrderedPsvLoader::spawn(
            &path,
            16,
            16,
            1,
            BucketMode::ProgressKpAbs,
            9,
            test_spec(),
            0,
        )
        .expect("spawn loader");
        let err = drain_until_error(loader).expect_err("corrupt record must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("corrupt position") || msg.contains("panicked"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    // ---- ScoreSidecarWriter ----

    fn temp_sidecar(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tatara-rescore-sidecar-{tag}-{}.scores.i16",
            std::process::id()
        ))
    }

    fn cleanup_sidecar(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(in_progress_marker_path(path));
        let _ = std::fs::remove_file(done_marker_path(path));
    }

    fn open_writer(path: &Path, expected: u64, fingerprint: &str) -> (ScoreSidecarWriter, u64) {
        match ScoreSidecarWriter::open(path, expected, fingerprint).expect("open sidecar") {
            SidecarOpen::Writer {
                writer,
                resume_records,
            } => (writer, resume_records),
            SidecarOpen::Complete => panic!("expected a writer, got Complete"),
        }
    }

    #[test]
    fn sidecar_writer_writes_promotes_and_skips_when_complete() {
        let path = temp_sidecar("roundtrip");
        cleanup_sidecar(&path);
        let scores: Vec<i16> = (0..10).map(|i| i * 100 - 450).collect();

        let (mut writer, resume) = open_writer(&path, 10, "fp-v1");
        assert_eq!(resume, 0);
        assert!(in_progress_marker_path(&path).exists());
        writer.write_scores(&scores[..4]).unwrap();
        writer.write_scores(&scores[4..]).unwrap();
        assert_eq!(writer.written(), 10);
        writer.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let expected_bytes: Vec<u8> = scores.iter().flat_map(|s| s.to_le_bytes()).collect();
        assert_eq!(bytes, expected_bytes);
        assert!(!in_progress_marker_path(&path).exists());
        assert_eq!(
            std::fs::read_to_string(done_marker_path(&path)).unwrap(),
            "fp-v1"
        );

        // 完了済み + fingerprint 一致 → Complete (sidecar は不変)。
        match ScoreSidecarWriter::open(&path, 10, "fp-v1").unwrap() {
            SidecarOpen::Complete => {}
            SidecarOpen::Writer { .. } => panic!("completed sidecar must be skipped"),
        }
        assert_eq!(std::fs::read(&path).unwrap(), expected_bytes);
        cleanup_sidecar(&path);
    }

    #[test]
    fn sidecar_writer_resumes_by_count_bit_identical() {
        let path = temp_sidecar("resume");
        cleanup_sidecar(&path);
        let scores: Vec<i16> = (0..20).map(|i| i * 37 - 300).collect();
        let one_shot: Vec<u8> = scores.iter().flat_map(|s| s.to_le_bytes()).collect();

        // 前半だけ書いて finish せずに drop → marker と prefix が残る。
        let (mut writer, _) = open_writer(&path, 20, "fp-resume");
        writer.write_scores(&scores[..7]).unwrap();
        drop(writer);
        assert!(in_progress_marker_path(&path).exists());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 14);

        // 同じ fingerprint で再開 → record 7 から追記し、一気通貫と bit 一致。
        let (mut writer, resume) = open_writer(&path, 20, "fp-resume");
        assert_eq!(resume, 7);
        writer.write_scores(&scores[7..]).unwrap();
        writer.finish().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), one_shot);
        cleanup_sidecar(&path);
    }

    #[test]
    fn sidecar_writer_truncates_on_fingerprint_mismatch() {
        let path = temp_sidecar("mismatch");
        cleanup_sidecar(&path);
        let (mut writer, _) = open_writer(&path, 5, "fp-old");
        writer.write_scores(&[1, 2, 3]).unwrap();
        drop(writer);

        // 条件が変わったら prefix を捨てて最初から (別条件の行が混ざらない)。
        let (writer, resume) = open_writer(&path, 5, "fp-new");
        assert_eq!(resume, 0);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        assert_eq!(
            std::fs::read_to_string(in_progress_marker_path(&path)).unwrap(),
            "fp-new"
        );
        drop(writer);
        cleanup_sidecar(&path);
    }

    #[test]
    fn sidecar_writer_rejects_odd_size_and_overflow() {
        let path = temp_sidecar("oddsize");
        cleanup_sidecar(&path);
        std::fs::write(&path, [0_u8; 5]).unwrap();
        std::fs::write(in_progress_marker_path(&path), "fp-odd").unwrap();
        let err = ScoreSidecarWriter::open(&path, 10, "fp-odd")
            .map(|_| ())
            .expect_err("odd size must fail");
        assert!(err.to_string().contains("not a multiple"), "{err}");
        // 事後検出不能なため自己修復せず、file はそのまま残る。
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 5);
        cleanup_sidecar(&path);

        // expected 超過の書き込みも拒否。
        let path = temp_sidecar("overflow");
        cleanup_sidecar(&path);
        let (mut writer, _) = open_writer(&path, 3, "fp-ov");
        let err = writer
            .write_scores(&[1, 2, 3, 4])
            .expect_err("overflow must fail");
        assert!(err.to_string().contains("exceed"), "{err}");
        drop(writer);
        cleanup_sidecar(&path);
    }

    /// `fail_after` byte 書いた後の write を失敗させる sink (部分書き込み注入)。
    struct FailingSink {
        accepted: usize,
        fail_after: usize,
    }

    impl Write for FailingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.accepted >= self.fail_after {
                return Err(io::Error::other("injected write failure"));
            }
            let n = buf.len().min(self.fail_after - self.accepted);
            self.accepted += n;
            Ok(n)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl SidecarSink for FailingSink {
        fn sync_all(&self) -> io::Result<()> {
            Ok(())
        }
    }

    /// 部分書き込み後の write エラーで writer が毒化され、同一 writer での retry
    /// (prefix の二重 append / record 途中への継ぎ足し) と `.done` 昇格が拒否される。
    #[test]
    fn sidecar_writer_poisons_after_partial_write_failure() {
        let path = temp_sidecar("poison");
        cleanup_sidecar(&path);
        // 5 byte (= record 2.5 個分) 受理後に失敗 → slice 途中の部分書き込み。
        let sink = Box::new(FailingSink {
            accepted: 0,
            fail_after: 5,
        });
        let mut writer = ScoreSidecarWriter::with_sink_for_test(sink, &path, 10, "fp-poison");
        let err = writer
            .write_scores(&[1, 2, 3, 4])
            .expect_err("partial write must fail");
        assert!(err.to_string().contains("poisoned"), "{err}");
        assert_eq!(
            writer.written(),
            0,
            "failed chunk must not advance the counter"
        );

        let retry_err = writer
            .write_scores(&[1])
            .expect_err("a poisoned writer must reject retries");
        assert!(retry_err.to_string().contains("poisoned"), "{retry_err}");

        let finish_err = writer
            .finish()
            .expect_err("a poisoned writer must not promote");
        assert!(finish_err.to_string().contains("poisoned"), "{finish_err}");
        cleanup_sidecar(&path);
    }

    /// 論理カウンタが揃っていても物理サイズが一致しない sidecar は `.done` へ
    /// 昇格しない (外部からの混入 / torn write の検出網)。
    #[test]
    fn sidecar_writer_finish_rejects_physical_size_mismatch() {
        let path = temp_sidecar("physmismatch");
        cleanup_sidecar(&path);
        let (mut writer, _) = open_writer(&path, 3, "fp-phys");
        writer.write_scores(&[1, 2, 3]).unwrap();
        // writer の buffer が flush される前に、外部プロセス相当の 1 byte 混入を
        // 模す (finish の flush で正常 6 byte が後続し、物理 7 byte になる)。
        {
            let mut external = File::options().append(true).open(&path).unwrap();
            external.write_all(&[0xAA]).unwrap();
        }
        let err = writer
            .finish()
            .expect_err("physical size mismatch must not promote");
        assert!(err.to_string().contains("physical size"), "{err}");
        assert!(in_progress_marker_path(&path).exists());
        assert!(!done_marker_path(&path).exists());
        cleanup_sidecar(&path);
    }

    #[test]
    fn sidecar_writer_finish_requires_full_count_and_keeps_marker() {
        let path = temp_sidecar("short");
        cleanup_sidecar(&path);
        let (mut writer, _) = open_writer(&path, 8, "fp-short");
        writer.write_scores(&[1, 2, 3]).unwrap();
        let err = writer.finish().expect_err("short sidecar must not promote");
        assert!(err.to_string().contains("3 of 8"), "{err}");
        // marker は残る → 次回 resume できる。
        assert!(in_progress_marker_path(&path).exists());
        assert!(!done_marker_path(&path).exists());
        cleanup_sidecar(&path);
    }

    #[test]
    fn sidecar_writer_regenerates_when_done_marker_is_stale() {
        let path = temp_sidecar("staledone");
        cleanup_sidecar(&path);
        let (mut writer, _) = open_writer(&path, 2, "fp-a");
        writer.write_scores(&[10, 20]).unwrap();
        writer.finish().unwrap();

        // fingerprint が変わった → done を捨てて最初から。
        let (writer, resume) = open_writer(&path, 2, "fp-b");
        assert_eq!(resume, 0);
        assert!(!done_marker_path(&path).exists());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        drop(writer);
        cleanup_sidecar(&path);
    }

    /// loader → writer を通した resume の end-to-end: 中断 → 再開の sidecar が
    /// 一気通貫実行と bit 一致する。
    #[test]
    fn loader_and_writer_resume_is_bit_identical_to_one_shot() {
        let psv = temp_psv("e2e", 3); // 300 records
        let one_shot_path = temp_sidecar("e2e-oneshot");
        let resumed_path = temp_sidecar("e2e-resumed");
        cleanup_sidecar(&one_shot_path);
        cleanup_sidecar(&resumed_path);
        let fingerprint = "fp-e2e";

        // 「forward」の代役: score をそのまま i16 化する (行対応の検証には十分)。
        let chunk_scores = |chunk: &RescoreChunk| -> Vec<i16> {
            chunk.batch.score[..chunk.n_real]
                .iter()
                .map(|&s| s as i16)
                .collect()
        };
        let run = |sidecar: &Path, start: u64, stop_after_chunks: Option<usize>| {
            let mut loader = OrderedPsvLoader::spawn(
                &psv,
                64,
                16,
                4,
                BucketMode::ProgressKpAbs,
                9,
                test_spec(),
                start,
            )
            .expect("spawn loader");
            let (mut writer, resume) = open_writer(sidecar, loader.total_records(), fingerprint);
            assert_eq!(resume, start);
            let mut chunks_done = 0;
            while let Some(chunk) = loader.next_chunk().expect("next_chunk") {
                writer.write_scores(&chunk_scores(&chunk)).unwrap();
                loader.recycle(chunk);
                chunks_done += 1;
                if Some(chunks_done) == stop_after_chunks {
                    return; // 中断: writer drop で marker と prefix が残る
                }
            }
            writer.finish().expect("finish sidecar");
        };

        run(&one_shot_path, 0, None);

        // 2 chunk (128 records) で中断 → writer の書き込み済み件数から再開。
        run(&resumed_path, 0, Some(2));
        let written = std::fs::metadata(&resumed_path).unwrap().len() / SCORE_RECORD_BYTES;
        assert_eq!(written, 128);
        run(&resumed_path, written, None);

        assert_eq!(
            std::fs::read(&one_shot_path).unwrap(),
            std::fs::read(&resumed_path).unwrap()
        );
        assert!(done_marker_path(&resumed_path).exists());
        cleanup_sidecar(&one_shot_path);
        cleanup_sidecar(&resumed_path);
        let _ = std::fs::remove_file(&psv);
    }
}
