//! PSV reader & game splitter。
//!
//! bullet-shogi 上流 (`shogi_progress_kpabs_train_cuda.rs::PackCursor` /
//! `GameIterator`) を移植。1 ファイル分の PSV record を順次読み、`game_ply`
//! の減少をゲーム境界として `Vec<PackedSfenValue>` を返す iterator。
//!
//! ## ファイル形式
//!
//! `.bin` (PSV) は `PackedSfenValue` (40 bytes) の連続列。
//! 1 ファイル = 複数ゲーム連結、ゲーム境界は明示マーカーなく `game_ply`
//! が 1 → ... → max → 1 → ... と reset することで判定する (bullet 上流の
//! 慣例)。

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::mem::size_of;
use std::path::Path;

use shogi_format::PackedSfenValue;

const PACK_RECORD_BYTES: usize = size_of::<PackedSfenValue>();

/// 1 PSV ファイルを sequential に読み出す cursor。
pub struct PackCursor {
    reader: BufReader<File>,
    remaining_records: u64,
}

impl PackCursor {
    /// ファイルを開いて、record 数を size から推定する。
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len % PACK_RECORD_BYTES as u64 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{}: file size {} is not a multiple of PSV record size {}",
                    path.display(),
                    len,
                    PACK_RECORD_BYTES
                ),
            ));
        }
        Ok(Self {
            reader: BufReader::new(file),
            remaining_records: len / PACK_RECORD_BYTES as u64,
        })
    }

    /// 残り record 数を返す (debug / progress 表示用)。
    pub fn remaining(&self) -> u64 {
        self.remaining_records
    }

    /// 次の 1 record を読む。EOF で `Ok(None)`。
    pub fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.remaining_records == 0 {
            return Ok(None);
        }
        let mut psv = PackedSfenValue::default();
        match self.reader.read_exact(psv.as_bytes_mut()) {
            Ok(()) => {
                self.remaining_records -= 1;
                Ok(Some(psv))
            }
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
                self.remaining_records = 0;
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }
}

/// PSV を ゲーム単位 (`Vec<PackedSfenValue>`) に切り分ける iterator。
///
/// `game_ply` が前 record より小さくなった時点で「新しいゲームが始まった」
/// と判定し、それまで積んだ records を 1 game として返す。
pub struct GameIterator {
    cursor: PackCursor,
    /// 現在組み立て中のゲームの records。
    buffer: Vec<PackedSfenValue>,
    /// 直前 record の `game_ply`。`None` なら最初の read 前。
    prev_ply: Option<u16>,
    done: bool,
}

impl GameIterator {
    pub fn new(cursor: PackCursor) -> Self {
        Self {
            cursor,
            buffer: Vec::new(),
            prev_ply: None,
            done: false,
        }
    }

    /// 次の 1 ゲーム分を返す。EOF + 残バッファなしで `Ok(None)`。
    pub fn next_game(&mut self) -> io::Result<Option<Vec<PackedSfenValue>>> {
        if self.done {
            return Ok(None);
        }
        loop {
            match self.cursor.next_psv()? {
                Some(psv) => {
                    let cur_ply = psv.game_ply();
                    // bullet-shogi 上流の境界判定は `cur_ply <= prev` (= は前ゲーム末尾と
                    // 同じ ply で新ゲームが始まる場合)。`<` だけだと等値境界が誤って
                    // 前ゲームに merge されるため `<=` を踏襲する。
                    if let Some(prev) = self.prev_ply
                        && cur_ply <= prev
                    {
                        let game = std::mem::take(&mut self.buffer);
                        self.buffer.push(psv);
                        self.prev_ply = Some(cur_ply);
                        return Ok(Some(game));
                    }
                    self.buffer.push(psv);
                    self.prev_ply = Some(cur_ply);
                }
                None => {
                    // EOF: 残 buffer があれば最後のゲームとして返す。
                    self.done = true;
                    if self.buffer.is_empty() {
                        return Ok(None);
                    }
                    let game = std::mem::take(&mut self.buffer);
                    return Ok(Some(game));
                }
            }
        }
    }
}
